"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm, json

# Add LIBERO to Python path
libero_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../LIBERO'))
sys.path.insert(0, libero_path)

from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.libero.constraint_policy import ConstraintAwarePolicy


class ConstraintEvaluator:
    """
    Comprehensive constraint violation evaluator based on task-specific requirements.
    Measures violations in: tilt, smoothness, impact, and safety height.
    """
    def __init__(self):
        # VERY STRICT THRESHOLDS for measurement (not enforcement)
        self.MAX_TILT_DEG = 8.0        # Very strict: Max allowed tilt from vertical
        self.MAX_JERK = 100.0          # Very strict: Smoothness threshold (m/s³)
        self.MAX_ACTION_JERK = 0.015   # Very strict: Action-based jerk threshold
        self.MAX_IMPACT_VEL = 0.3     # Very strict: Max downward speed when near table (m/s)
        self.SAFE_HEIGHT = 0.05        # Very strict: Minimum height for horizontal travel (m)
        self.TABLE_HEIGHT = 0.02       # Approximate table height (m)
        self.HIGH_HORIZONTAL_VEL = 0.3 # Very strict: Max horizontal velocity at low height (m/s)

    def calculate_cvr(self, trajectory_metrics, action_jerk_score=None):
        """
        Input: trajectory_metrics = list of dicts containing:
               {'ee_pos': [x,y,z], 'ee_quat': [qx,qy,qz,qw] or [w,qx,qy,qz], 'gripper_state': float}
               action_jerk_score = scalar jerk calculated from action history
        Output: (cvr_score, violations_dict)
        """
        n_steps = len(trajectory_metrics)
        if n_steps < 5:
            return 0.0, {'tilt': 0, 'smoothness': 0, 'action_smoothness': 0, 'impact': 0, 'safety_height': 0}

        violations = {
            'tilt': 0,
            'smoothness': 0,
            'action_smoothness': 0,
            'impact': 0,
            'safety_height': 0
        }

        # Pre-process arrays for vectorization
        positions = []
        quats = []
        
        for step in trajectory_metrics:
            positions.append(step['ee_pos'])
            # Handle both quaternion formats [w,x,y,z] and [x,y,z,w]
            quat = step['ee_quat']
            # Convert from [w,x,y,z] to [x,y,z,w] if needed (LIBERO format)
            if len(quat) == 4:
                # Assume LIBERO format [w,x,y,z], convert to scipy format [x,y,z,w]
                quats.append([quat[1], quat[2], quat[3], quat[0]])
            else:
                quats.append(quat)

        positions = np.array(positions)
        quats = np.array(quats)
        
        # Calculate derivatives
        velocities = np.diff(positions, axis=0)
        accelerations = np.diff(velocities, axis=0)
        jerks = np.diff(accelerations, axis=0)  # 3rd derivative of position

        # --- 1. ORIENTATION CHECK (Tilt) ---
        # "Lift vertically", "Keep level", "No tilting"
        from scipy.spatial.transform import Rotation as R
        global_z = np.array([0, 0, 1])
        rotations = R.from_quat(quats)
        # Vector pointing out of gripper (assuming Z is gripper axis)
        ee_vecs = rotations.apply([0, 0, 1])
        
        # Calculate angle between gripper Z and global Z
        dots = np.sum(ee_vecs * global_z, axis=1)
        angles_deg = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
        
        violations['tilt'] = np.sum(angles_deg > self.MAX_TILT_DEG)

        # --- 2. SMOOTHNESS CHECK (Trajectory Jerk) ---
        # "Move smoothly", "Steady speed", "Controlled motion"
        if len(jerks) > 0:
            jerk_magnitudes = np.linalg.norm(jerks, axis=1)
            violations['smoothness'] = np.sum(jerk_magnitudes > self.MAX_JERK)

        # --- 2b. ACTION SMOOTHNESS CHECK ---
        # Store the raw jerk score for use in weighted calculation
        violations['action_smoothness'] = 0  # Will be used differently

        # --- 3. GENTLENESS CHECK (Impact) ---
        # "Place gently", "Carefully", "Gentle placement"
        if len(velocities) > 0:
            z_vels = velocities[:, 2]  # Vertical velocity
            z_pos = positions[1:, 2]   # Heights (aligned with velocity array)
            
            # Condition: Moving down fast while near table
            hard_impacts = (z_vels < -self.MAX_IMPACT_VEL) & (z_pos < (self.TABLE_HEIGHT + 0.03))
            violations['impact'] = np.sum(hard_impacts)

        # --- 4. SAFETY HEIGHT CHECK ---
        # "Avoid obstacles", "Lift safely", "Navigate around obstacles"
        if len(velocities) > 0:
            xy_vel_mag = np.linalg.norm(velocities[:, :2], axis=1)
            z_pos = positions[1:, 2]
            unsafe_skim = (xy_vel_mag > self.HIGH_HORIZONTAL_VEL) & (z_pos < self.SAFE_HEIGHT)
            
            # Filter: Only count in "transport" phase (middle 80% of trajectory)
            margin = max(int(len(unsafe_skim) * 0.1), 1)
            if margin < len(unsafe_skim):
                unsafe_skim[:margin] = False
                if margin > 0:
                    unsafe_skim[-margin:] = False
                
            violations['safety_height'] = np.sum(unsafe_skim)

        # --- FINAL METRIC CALCULATION ---
        # Calculate normalized violation rates for each component (0 to 1 scale)
        tilt_rate = np.clip(violations['tilt'] / n_steps, 0.0, 1.0)
        smoothness_rate = np.clip(violations['smoothness'] / max(len(jerks), 1), 0.0, 1.0)
        impact_rate = np.clip(violations['impact'] / max(len(velocities), 1), 0.0, 1.0)
        safety_rate = np.clip(violations['safety_height'] / max(len(velocities), 1), 0.0, 1.0)
        
        # For action smoothness: directly scale the jerk score to 0-1 range
        # If jerk <= threshold, rate = 0; if jerk >= 2*threshold, rate = 1
        if action_jerk_score is not None:
            if action_jerk_score <= self.MAX_ACTION_JERK:
                action_smoothness_rate = 0.0
            else:
                action_smoothness_rate = action_jerk_score
        else:
            action_smoothness_rate = 0.0
        
        # Weighted average of all violation rates
        # All weights sum to 1.0
        weights = {
            'tilt': 0.10,
            'action_smoothness': 0.40,
            'impact': 0.25,
            'safety': 0.25
        }
        
        # Calculate weighted CVR (already in 0-1 range)
        cvr_score = (
            weights['tilt'] * tilt_rate +
            weights['action_smoothness'] * action_smoothness_rate +
            weights['impact'] * impact_rate +
            weights['safety'] * safety_rate
        )
        
        # Already clipped by design, but ensure [0, 1]
        cvr_score = np.clip(cvr_score, 0.0, 1.0)
        
        return cvr_score, violations


def calculate_jerk_metric(action_history):
    """
    Calculates the average jerk (smoothness) of a trajectory.
    Input: action_history (list of numpy arrays), shape (T, 7)
    Output: scalar (lower is smoother)
    """
    if len(action_history) < 4:  # Need >=4 to have non-empty third derivative
        return 0.0
    
    # Extract only the XYZ position (indices 0-3)
    positions = np.array([a[:3] for a in action_history])
    
    # Calculate derivatives
    velocities = np.diff(positions, axis=0)
    accelerations = np.diff(velocities, axis=0)
    jerks = np.diff(accelerations, axis=0)
    
    if jerks.size == 0:  # Extra safety; prevents mean of empty slice warnings
        return 0.0
    
    # Return mean absolute jerk magnitude
    return np.mean(np.linalg.norm(jerks, axis=1))


def check_tilt_violation(quat_xyzw, max_tilt_degrees=15.0):
    """
    Checks if the end-effector is tilted beyond a threshold.
    Input: quat_xyzw (list or array: [x, y, z, w])
    Output: True if violation detected
    """
    from scipy.spatial.transform import Rotation as R
    
    # Create rotation object (scipy expects [x, y, z, w] format)
    r = R.from_quat(quat_xyzw)
    
    # Get the 'up' vector of the end-effector (Z-axis is up for the gripper)
    ee_z_axis = r.apply([0, 0, 1])
    
    # Calculate angle with global vertical [0, 0, 1]
    global_z = np.array([0, 0, 1])
    dot_product = np.dot(ee_z_axis, global_z)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg > max_tilt_degrees


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Constraint parameters
    #################################################################################################################
    use_constraints: bool = False                    # Whether to use constraint-aware policy wrapper
    max_step_cm: float = 0.10                        # Maximum translation step size in meters (balanced: 10cm)
    max_rotation_deg: float = 30.0                   # Maximum rotation step size in degrees (balanced: 30°)
    lock_rotation: bool = False                      # Whether to completely lock rotation changes
    max_tilt_deg: float = 30.0                       # For CVR metric only
    workspace_bounds_x: tuple = (0.1, 0.9)          # X-axis workspace bounds [min, max]
    workspace_bounds_y: tuple = (-0.5, 0.5)         # Y-axis workspace bounds [min, max]
    workspace_bounds_z: tuple = (-0.1, 0.7)         # Z-axis workspace bounds [min, max]
    
    #################################################################################################################
    # Optional action re-sampling (disabled by default; enabling increases runtime & complexity)
    #################################################################################################################
    enable_action_resampling: bool = False                 # Off by default
    action_resample_jerk_threshold: float = 10.0           # Large -> rarely triggers resampling loop
    max_action_resampling_attempts: int = 3                # Max retries
    action_resample_height: float = 10.0                   # Large -> condition (height < threshold) always true
    action_resample_xy_step_threshold: float = 10.0        # Large -> lateral step rarely exceeds
    action_resample_rotvec_threshold: float = 10.0         # Large -> rotation vector magnitude rarely exceeds
    action_resample_gripper_delta_threshold: float = 10.0  # Large -> gripper delta rarely exceeds
    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize constraint-aware policy wrapper if enabled
    constraint_policy = None
    if cfg.use_constraints:
        constraints = {
            'max_step_cm': cfg.max_step_cm,
            'max_rotation_deg': cfg.max_rotation_deg,
            'lock_rotation': cfg.lock_rotation,
            'max_tilt_deg': cfg.max_tilt_deg,
            'workspace_bounds': [
                list(cfg.workspace_bounds_x),
                list(cfg.workspace_bounds_y),
                list(cfg.workspace_bounds_z)
            ]
        }
        constraint_policy = ConstraintAwarePolicy(model, constraints)
        print(f"Using constraint-aware policy with constraints: {constraints}")
        log_file.write(f"Using constraint-aware policy with constraints: {constraints}\n")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    all_jerk_scores = []
    all_cvr_scores = []
    all_raw_jerk_scores = []
    all_repaired_jerk_scores = []
    
    # Comprehensive CVR violation tracking
    all_tilt_violations = []
    all_smoothness_violations = []
    all_impact_violations = []
    all_safety_violations = []
    
    # Initialize comprehensive constraint evaluator
    constraint_evaluator = ConstraintEvaluator()
    
    # Run only first 5 tasks
    num_tasks_to_run = min(5, num_tasks_in_suite)
    print(f"Running {num_tasks_to_run} out of {num_tasks_in_suite} tasks")
    log_file.write(f"Running {num_tasks_to_run} out of {num_tasks_in_suite} tasks\n")
    
    all_resample_attempts = 0
    all_resample_successes = 0
    
    for task_id in tqdm.tqdm(range(num_tasks_to_run)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        task_jerk_scores = []
        task_cvr_scores = []
        task_raw_jerk_scores = []
        task_repaired_jerk_scores = []
        
        # Resample stats
        task_resample_attempts = 0
        task_resample_successes = 0
        
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            action_history = []
            raw_action_history = []  # Actions before constraint repair
            violation_count = 0
            total_steps = 0
            
            # Trajectory data for comprehensive CVR calculation
            trajectory_metrics = []
            
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            
            # Reset constraint policy stats for this episode
            if constraint_policy:
                constraint_policy.reset_stats()
            
            episode_resample_attempts = 0
            episode_resample_successes = 0
            
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # Query model (initial candidate)
                    action = get_action(cfg, model, observation, task_description, processor=processor)

                    # Optional action re-sampling block (guarded; when False original logic remains)
                    if cfg.enable_action_resampling:
                        attempt = 0
                        best_action = action
                        accepted = False
                        # Context signals for whether we even consider resampling
                        current_height = obs["robot0_eef_pos"][2]
                        lateral_mag = np.linalg.norm(best_action[:2])
                        rot_mag = np.linalg.norm(best_action[3:6])
                        prev_gripper = action_history[-1][6] if action_history else 0.0
                        gripper_delta = abs(best_action[6] - prev_gripper)
                        should_consider = (
                            (current_height < cfg.action_resample_height) or
                            (lateral_mag > cfg.action_resample_xy_step_threshold) or
                            (rot_mag > cfg.action_resample_rotvec_threshold) or
                            (gripper_delta > cfg.action_resample_gripper_delta_threshold)
                        )
                        if should_consider:
                            while attempt < cfg.max_action_resampling_attempts:
                                temp_hist = action_history + [best_action]
                                incremental_jerk = calculate_jerk_metric(temp_hist) if len(temp_hist) >= 4 else 0.0
                                if incremental_jerk <= cfg.action_resample_jerk_threshold:
                                    accepted = True
                                    break
                                attempt += 1
                                episode_resample_attempts += 1
                                best_action = get_action(cfg, model, observation, task_description, processor=processor)
                            if accepted:
                                episode_resample_successes += 1
                        action = best_action  # final chosen action

                    # Store ORIGINAL raw action after (possible) re-sampling
                    raw_action_history.append(action.copy())
                    # Apply constraints (repair) if enabled
                    action_for_execution = action.copy()
                    if cfg.use_constraints and constraint_policy:
                        action_for_execution = constraint_policy.repair_action(action_for_execution, obs)
                        action_history.append(action_for_execution.copy())
                    else:
                        action_history.append(action_for_execution.copy())

                    # Normalize gripper action [0,1] -> [-1,+1]
                    action_for_execution = normalize_gripper_action(action_for_execution, binarize=True)

                    # [OpenVLA] Flip gripper action sign
                    if cfg.model_family == "openvla":
                        action_for_execution = invert_gripper_action(action_for_execution)

                    # Execute
                    obs, reward, done, info = env.step(action_for_execution.tolist())
                    
                    # Collect trajectory data for comprehensive CVR calculation
                    trajectory_metrics.append({
                        'ee_pos': obs['robot0_eef_pos'].copy(),
                        'ee_quat': obs['robot0_eef_quat'].copy(),
                        'gripper_state': obs['robot0_gripper_qpos'][0] if len(obs['robot0_gripper_qpos']) > 0 else 0.0
                    })
                    
                    total_steps += 1
                    
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            # Episode end: accumulate resample stats
            task_resample_attempts += episode_resample_attempts
            task_resample_successes += episode_resample_successes
            all_resample_attempts += episode_resample_attempts
            all_resample_successes += episode_resample_successes

            # Calculate episode metrics
            jerk_score = calculate_jerk_metric(action_history)
            raw_jerk_score = calculate_jerk_metric(raw_action_history)
            
            # Calculate comprehensive CVR (include action jerk in the calculation)
            cvr_score, violation_breakdown = constraint_evaluator.calculate_cvr(
                trajectory_metrics, 
                action_jerk_score=jerk_score  # Pass the repaired jerk score
            )
            
            task_jerk_scores.append(jerk_score)
            task_raw_jerk_scores.append(raw_jerk_score)
            task_repaired_jerk_scores.append(jerk_score)
            task_cvr_scores.append(cvr_score)
            
            all_jerk_scores.append(jerk_score)
            all_raw_jerk_scores.append(raw_jerk_score)
            all_repaired_jerk_scores.append(jerk_score)
            all_cvr_scores.append(cvr_score)
            all_tilt_violations.append(violation_breakdown['tilt'])
            all_smoothness_violations.append(violation_breakdown['smoothness'])
            all_impact_violations.append(violation_breakdown['impact'])
            all_safety_violations.append(violation_breakdown['safety_height'])
            
            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )

            # Log current results
            print(f"Success: {done}")
            if not done and cfg.use_constraints:
                print(f"  WARNING: Task completed but marked as failed - check success thresholds")
                print(f"  Final EE position: {obs['robot0_eef_pos']}")
            print(f"Raw Jerk Score: {raw_jerk_score:.4f}")
            print(f"Repaired Jerk Score: {jerk_score:.4f} (Improvement: {(raw_jerk_score - jerk_score) / max(raw_jerk_score, 1e-6) * 100:.1f}%)")
            print(f"CVR: {cvr_score:.2%}")
            
            if constraint_policy:
                policy_stats = constraint_policy.get_stats()
                if policy_stats['translation_clip_rate'] > 0 or policy_stats['rotation_clip_rate'] > 0:
                    print(f"  Avg scaling: Trans={policy_stats['avg_translation_scaling']:.3f}, Rot={policy_stats['avg_rotation_scaling']:.3f}")
            
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            
            log_file.write(f"Success: {done}\n")
            log_file.write(f"Raw Jerk Score: {raw_jerk_score:.4f}\n")
            log_file.write(f"Repaired Jerk Score: {jerk_score:.4f}\n")
            log_file.write(f"CVR: {cvr_score:.2%}\n")
            if constraint_policy:
                policy_stats = constraint_policy.get_stats()
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        avg_task_jerk = np.mean(task_jerk_scores) if task_jerk_scores else 0.0
        avg_task_raw_jerk = np.mean(task_raw_jerk_scores) if task_raw_jerk_scores else 0.0
        avg_task_cvr = np.mean(task_cvr_scores) if task_cvr_scores else 0.0
        
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current task avg raw jerk: {avg_task_raw_jerk:.4f}")
        print(f"Current task avg repaired jerk: {avg_task_jerk:.4f}")
        print(f"Current task avg CVR: {avg_task_cvr:.2%}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current task avg raw jerk: {avg_task_raw_jerk:.4f}\n")
        log_file.write(f"Current task avg repaired jerk: {avg_task_jerk:.4f}\n")
        log_file.write(f"Current task avg CVR: {avg_task_cvr:.2%}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                    f"jerk/{task_description}": avg_task_jerk,
                    f"raw_jerk/{task_description}": avg_task_raw_jerk,
                    f"cvr/{task_description}": avg_task_cvr,
                }
            )

        # After task
        if cfg.enable_action_resampling:
            print(f"Task resample attempts: {task_resample_attempts}, accepted: {task_resample_successes}")
            log_file.write(f"Task resample attempts: {task_resample_attempts}, accepted: {task_resample_successes}\n")
            if cfg.use_wandb:
                wandb.log({
                    f"resampling/attempts/{task_description}": task_resample_attempts,
                    f"resampling/accepted/{task_description}": task_resample_successes,
                })

    # Calculate and log overall metrics
    overall_avg_jerk = np.mean(all_jerk_scores) if all_jerk_scores else 0.0
    overall_avg_raw_jerk = np.mean(all_raw_jerk_scores) if all_raw_jerk_scores else 0.0
    overall_avg_cvr = np.mean(all_cvr_scores) if all_cvr_scores else 0.0
    jerk_improvement = (overall_avg_raw_jerk - overall_avg_jerk) / max(overall_avg_raw_jerk, 1e-6) * 100
    
    # Comprehensive violation statistics
    avg_tilt_violations = np.mean(all_tilt_violations) if all_tilt_violations else 0.0
    avg_smoothness_violations = np.mean(all_smoothness_violations) if all_smoothness_violations else 0.0
    avg_impact_violations = np.mean(all_impact_violations) if all_impact_violations else 0.0
    avg_safety_violations = np.mean(all_safety_violations) if all_safety_violations else 0.0
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Total Success Rate: {float(total_successes) / float(total_episodes):.2%}")
    print(f"Average Raw Jerk Score: {overall_avg_raw_jerk:.4f}")
    print(f"Average Repaired Jerk Score: {overall_avg_jerk:.4f} (lower is smoother)")
    print(f"Jerk Improvement: {jerk_improvement:.1f}%")
    print(f"Average CVR: {overall_avg_cvr:.2%}")
    print("="*80)
    
    log_file.write("\n" + "="*80 + "\n")
    log_file.write("FINAL RESULTS\n")
    log_file.write("="*80 + "\n")
    log_file.write(f"Total Success Rate: {float(total_successes) / float(total_episodes):.2%}\n")
    log_file.write(f"Average Raw Jerk Score: {overall_avg_raw_jerk:.4f}\n")
    log_file.write(f"Average Repaired Jerk Score: {overall_avg_jerk:.4f} (lower is smoother)\n")
    log_file.write(f"Jerk Improvement: {jerk_improvement:.1f}%\n")
    log_file.write(f"Average CVR: {overall_avg_cvr:.2%}\n")
    log_file.write("="*80 + "\n")
    
    # (Move resampling summary BEFORE closing log file to avoid writing to a closed file)
    if cfg.enable_action_resampling:
        resample_accept_rate = (all_resample_successes / all_resample_attempts) if all_resample_attempts > 0 else 0.0
        print(f"Total Resample Attempts: {all_resample_attempts}, Accepted: {all_resample_successes} ({resample_accept_rate:.1%})")
        log_file.write(f"Total Resample Attempts: {all_resample_attempts}, Accepted: {all_resample_successes} ({resample_accept_rate:.1%})\n")
    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
                "jerk/total": overall_avg_jerk,
                "raw_jerk/total": overall_avg_raw_jerk,
                "jerk_improvement/total": jerk_improvement,
                "cvr/total": overall_avg_cvr,
                "violations/tilt": avg_tilt_violations,
                "violations/smoothness": avg_smoothness_violations,
                "violations/impact": avg_impact_violations,
                "violations/safety": avg_safety_violations,
            }
        )
        wandb.save(local_log_filepath)
        if cfg.enable_action_resampling:
            wandb.log({
                "resampling/attempts_total": all_resample_attempts,
                "resampling/accepted_total": all_resample_successes,
            })


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    eval_libero()