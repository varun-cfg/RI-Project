# OpenVLA LIBERO evaluation: Optional action re-sampling

This repository extends the OpenVLA LIBERO evaluator to optionally re-sample actions that would introduce excessive jerk or abrupt changes. The feature is disabled by default to keep the original behavior and avoid added complexity and runtime.

What changed
- New optional action re-sampling step after querying the VLA policy.
- If enabled, the candidate action is tentatively appended to the current (post-repair) action history and a jerk metric is computed; if it exceeds a threshold, the action is re-sampled up to N attempts.
- Re-sampling can be further gated by contextual thresholds (height, lateral step, rotation change, gripper delta).
- Adds a comprehensive CVR evaluator and reporting (tilt, trajectory smoothness, impact, safety height).
- Full metrics are logged locally and to W&B (if enabled).

Why it’s off by default
- Re-sampling introduces extra model queries per control step and increases runtime and implementation complexity. Keeping it off preserves the original evaluation path.

Configuration flags (in experiments/robot/libero/run_libero_eval.py)
- enable_action_resampling: bool = False
  - Master switch. When False, the original action path is used exactly (no extra model calls).
- max_action_resampling_attempts: int = 3
  - Max number of additional model queries per step while searching for a smoother action.
- action_resample_jerk_threshold: float = 10.0
  - If the incremental jerk (computed over the temporary history including the candidate) exceeds this value, the candidate is rejected and we re-sample (subject to gating below).
- Contextual gating (re-sampling considered only if at least one is triggered):
  - action_resample_height: float = 10.0
  - action_resample_xy_step_threshold: float = 10.0
  - action_resample_rotvec_threshold: float = 10.0
  - action_resample_gripper_delta_threshold: float = 10.0

Notes on thresholds
- Large thresholds reduce the chance of triggering re-sampling. With the provided large defaults, re-sampling will rarely run even if enabled. For tighter control, reduce thresholds (and consider lowering action_resample_height to gate re-sampling to near-surface motions only).

Jerk metric
- We use a simple third-derivative magnitude on the translational XYZ component of the action stream. It requires at least 4 actions to be meaningful; otherwise jerk is treated as zero.

W&B metrics (if use_wandb=True)
- resampling/attempts/{task}, resampling/accepted/{task}
- resampling/attempts_total, resampling/accepted_total
- Existing success, jerk, and CVR metrics remain unchanged.

CLI examples
- Original behavior (default; no re-sampling):
  - python experiments/robot/libero/run_libero_eval.py --model_family openvla --pretrained_checkpoint <CKPT> --task_suite_name libero_spatial
- Enable re-sampling (runtime increases):
  - python experiments/robot/libero/run_libero_eval.py --model_family openvla --pretrained_checkpoint <CKPT> --task_suite_name libero_spatial --enable_action_resampling True

Caveats
- Enabling re-sampling increases runtime and the number of model forward passes per control step.
- Thresholds are dataset- and model-dependent; tune cautiously to balance smoothness vs. performance.

<!-- ## Jerk calculation details
- We compute jerk over the XYZ translation part of the action stream:
  - Let p_t be the 3D translation from action t. v_t = p_{t} - p_{t-1}, a_t = v_{t} - v_{t-1}, j_t = a_{t} - a_{t-1}.
  - Episode jerk score = mean_t ||j_t||_2.
- Guard against small histories:
  - Requires at least 4 actions to have a non-empty third derivative. If fewer, jerk is 0.0 (prevents NumPy mean-of-empty-slice warnings).
- Two usages:
  - Episode jerk: computed after the episode from the repaired action history.
  - Incremental jerk (for re-sampling): computed on a temporary history = existing repaired history + candidate action. If this exceeds action_resample_jerk_threshold, we can re-sample (subject to contextual gating).

## CVR calculation details
- Inputs per step: end-effector position, quaternion, and gripper state recorded during rollout.
- Components (all normalized to 0–1 rates, then weighted):
  - Tilt: angle between gripper Z and world Z; counts steps with angle > MAX_TILT_DEG.
  - Smoothness (trajectory): counts steps with ||jerk_pos|| > MAX_JERK (jerk from end-effector position over time).
  - Action smoothness: uses the repaired episode jerk score; mapped directly and weighted (no step-wise counting).
  - Impact: counts fast downward motions near the table (z-vel < -MAX_IMPACT_VEL and height < TABLE_HEIGHT + 0.03).
  - Safety height: counts high lateral velocity while at low height; only in the middle 80% of the trajectory.
- Quaternion handling:
  - Converts [w, x, y, z] to scipy format [x, y, z, w] when needed before computing tilt.
- Weights (sum to 1.0):
  - tilt: 0.10, action_smoothness: 0.40, impact: 0.25, safety: 0.25.
- Strict thresholds are used for evaluation (not enforcement). See ConstraintEvaluator in experiments/robot/libero/run_libero_eval.py for values and logic.

## Differences vs upstream OpenVLA
- Upstream OpenVLA does not include jerk-based action correction/resampling or the CVR metric/evaluator; these are evaluator-side extensions in this repository.
- No changes are made to OpenVLA model weights or training; only the inference-time evaluation loop is extended.
- Safety guard in jerk computation (≥4 actions) to avoid NumPy warnings and support incremental jerk checks used by resampling.

## Configuration pointers
- All flags live in experiments/robot/libero/run_libero_eval.py (GenerateConfig):
  - enable_action_resampling (default False), max_action_resampling_attempts.
  - action_resample_jerk_threshold, action_resample_height,
    action_resample_xy_step_threshold, action_resample_rotvec_threshold,
    action_resample_gripper_delta_threshold.
- With defaults (large thresholds), enabling re-sampling will rarely trigger and will increase runtime due to extra model calls. -->

## Setup and quick start

Prereqs
- Python 3.10, CUDA-capable GPU recommended.
- OpenVLA installed (see openvla/README.md for full setup).
- LIBERO installed in editable mode:
  - git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
  - cd LIBERO && pip install -e .
- Extra deps for LIBERO evaluator:
  - From repo root: pip install -r experiments/robot/libero/libero_requirements.txt

You can also set up the conda envs from the yaml files we provide. Run the below for creating the env for running vlms like qwen and llama, and for running the constraint generation, summarisation, and evaluation scripts.
```bash
conda env create -f ri_env.yaml
conda activate ri
```

For the openvla env, use the below:
```bash
conda env create -f openvla_env.yaml
conda activate openvla
```

This env is used to run the following 2 commands:

Run (constraints ON, re-sampling OFF by default)
```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 10 \
  --use_constraints True \
  --center_crop True
```

Optional: enable re-sampling (increases runtime; thresholds set large by default to rarely trigger)
```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 10 \
  --use_constraints True \
  --center_crop True \
  --enable_action_resampling True
```

Notes
- Keep center_crop True for fine-tuned LIBERO checkpoints.
- Resampling thresholds (large defaults to suppress triggering): action_resample_jerk_threshold, action_resample_height, action_resample_xy_step_threshold, action_resample_rotvec_threshold, action_resample_gripper_delta_threshold.
- Jerk and CVR are evaluator-side metrics; upstream OpenVLA does not include these corrections.

Please note: Use the ri env for the below commands.
Running the offline constraint generation script:
```bash
export TASK_SUITE="libero_spatial"
export CHECKPOINT="openvla/openvla-7b-finetuned-libero-spatial"
python experiments/robot/libero/offline.py     --model_family openvla     --pretrained_checkpoint "$CHECKPOINT"     --task_suite_name "$TASK_SUITE"     --center_crop True     --num_trials_per_task 10     --run_id_note "test_run"
```

The constraints will be saved to cons.json. To summarise them, run the following:
```bash
python summarise_constraints.py
```

The summarised constraints will be saved to sums.json. To score the subtasks and constraints using LLama 3.2 vision instruct, run the following:

```bash
python experiments/robot/libero/eval.py     --model_family openvla     --pretrained_checkpoint "$CHECKPOINT"     --task_suite_name "$TASK_SUITE"     --center_crop True     --num_trials_per_task 10     --run_id_note "test_run"
```

Scores will be saved to scores_llama.json.
