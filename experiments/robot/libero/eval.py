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
import json, re
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
# Add LIBERO to path
libero_path = os.path.expanduser("~/ri_project/LIBERO")
if libero_path not in sys.path:
    sys.path.insert(0, libero_path)

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
# Import LIBERO
try:
    from libero.libero import benchmark
except ImportError:
    from libero import benchmark

import wandb
import re

def extract_score(text): # to parse vlm output
    match = re.search(r"\b([1-5])\b", text)
    if match:
        return int(match.group(1))
    else:
        return None

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


from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

model = model.to("cuda")
def make_prompt_llama(images=None, text="", sys_prompt=""):
    messages = []

    if sys_prompt:
        messages.append({
            "role": "system",
            "content": [
                {"type": "text", "text": sys_prompt}
            ]
        })

    user_content = []

    # Add image placeholders
    if images is not None:
        if not isinstance(images, (list, tuple)):
            images = [images]
        for _ in images:
            user_content.append({"type": "image"})

    if text:
        user_content.append({"type": "text", "text": text})

    messages.append({"role": "user", "content": user_content})
    return messages

from PIL import Image

def prepare_llama_inputs(processor, messages, images):
    # 1. Build chat template text
    chat_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    # 2. Load image pixel data
    if isinstance(images, (list, tuple)):
        img_data = [Image.open(img).convert("RGB") for img in images]
    else:
        img_data = Image.open(images).convert("RGB")

    # 3. Build model inputs
    inputs = processor(
        img_data,
        chat_text,
        add_special_tokens=False,
        return_tensors="pt"
    )

    return inputs

def prompt_llama(model, processor, messages, images, max_new_tokens=256, temperature=0.3):
    inputs = prepare_llama_inputs(processor, messages, images)
    inputs = inputs.to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9
    )

    return processor.decode(output_ids[0], skip_special_tokens=True).strip()



def generate_depth_map(input_path, output_path="scene_depth.png", model_type="DPT_Large"):

    if "DPT" in model_type:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    img = Image.open(input_path).convert("RGB")
    img = np.array(img)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_img = (depth_map * 255).astype(np.uint8)

    cv2.imwrite(output_path, depth_img)
    return output_path


def make_prompt(images=None, video=None, text="", sys_prompt=""):

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": []},
    ]

    if images is not None:
        if isinstance(images, (list, tuple)):
            for img in images:
                messages[1]["content"].append({"type": "image", "image": img})
        else:
            messages[1]["content"].append({"type": "image", "image": images})

    if video is not None:
        messages[1]["content"].append({"type": "video", "video": video})
    if text:
        messages[1]["content"].append({"type": "text", "text": text})

    return messages

def prompt_qwen(model, processor, messages, max_new_tokens=4, temperature=0.3):

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def extract_scene_understanding(images=None, video=None):
    sys = """
    You are a visual reasoning model.
    Your job is to analyze the provided RGB and depth images and describe:
    1. The count of distinct object types.
    2. The spatial or semantic relations among them.
    3. Any notable depth-based relations (e.g., closer, behind, on top).

    Your answers MUST be grounded in what is visible — do not hallucinate unseen objects.
    Your answer must be in valid JSON format.
    """
    prompt = """
    Analyze the given image(s) and output ONLY a valid JSON object in this format:

    {
        "objects": {
            "<object1>": <count>,
            "<object2>": <count>
        },
        "relations": [
            {"relation": "<objectA> on <objectB>"},
            {"relation": "<objectA> near <objectB>"},
            {"relation": "<objectA> closer than <objectB>"}
        ]
    }
    """
    messages = make_prompt(images=images, video=video, text=prompt, sys_prompt=sys)
    return prompt_qwen(model, processor, messages, max_new_tokens=200)

def decompose_into_subtasks(images=None, video=None, goal="", scene_info=""):
    sys = """
    You are a robotic task planner for a 6-DOF arm.
    Decompose the given task into physically feasible subtasks,
    grounded in the RGB + depth input and the textual scene information.
    Your answer must be in valid JSON format.
    """
    prompt = f"""
    Task: {goal}

    Scene Information:
    {scene_info}

    Output a JSON object in this format:
    {{
        'subtasks': [
            'subtask 1 description',
            'subtask 2 description',
            ...
        ]
    }}
    """
    messages = make_prompt(images=images, video=video, text=prompt, sys_prompt=sys)
    return prompt_qwen(model, processor, messages)


def verify_subtasks(images=None, video=None, goal="", subtasks="", scene_info=""):
    sys = """
    You are a verification model that ensures subtasks are valid and grounded in both the RGB and depth inputs, as well as the scene information. Your answer must be in valid JSON format.
    """
    prompt = f"""
    Task: {goal}
    Scene Information: {scene_info}
    Proposed Subtasks: {subtasks}

    Verify and correct the list if necessary. If no corrections are needed for a subtask, just copy it directly.
    Output JSON:
    {{
        'verified_subtasks': [
            'corrected (if needed) subtask 1',
            'corrected (if needed) subtask 2',
            ...
        ]
    }}
    """
    messages = make_prompt(images=images, video=video, text=prompt, sys_prompt=sys)
    return prompt_qwen(model, processor, messages)


def generate_constraints(images=None, video=None, subtasks="", scene_info=""):
    sys = """
    You are a robotic safety and control expert.
    For each subtask, generate movement-related constraints based on physical reality and visual grounding in both RGB and depth, along with the scene information in text form.

    """
    prompt = f"""
    Scene Information:
    {scene_info}

    Subtasks:
    {subtasks}

    Output JSON:
    {{
        'constraints': [
            {{
                'subtask': '<subtask text>',
                'constraint': '<constraint text>'
            }},
            ...
        ]
    }}
    """
    messages = make_prompt(images=images, video=video, text=prompt, sys_prompt=sys)
    return prompt_qwen(model, processor, messages)


def integrate_constraints_with_subtasks(subtasks, constraints):
    return f"Integrated Grounded Plan:\n{subtasks}\n\nConstraints:\n{constraints}"
def safe_extract_json(text):
    """Extract the first valid JSON object from an LLM response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except:
        return None

def do_stuff(img, task, episode, goal):

    # ---- Save RGB image ----
    rgb_image = Image.fromarray(img)
    rgb_path = f"/home/vijval22569/ri_project/openvla/rgbs/{task}_{episode}.png"
    rgb_image.save(rgb_path)
    # print(f"Saved RGB image to {rgb_path}")


    # print("Generating depth map...")
    depth_path = generate_depth_map(rgb_path, f"/home/vijval22569/ri_project/openvla/depths/{task}_{episode}.png")
    # print(f"Saved depth map to {depth_path}")

    # ---- Step 1: Scene Understanding ----
    # print("\nStep 1: Extracting grounded scene understanding...")
    scene_info = extract_scene_understanding(images=[rgb_path, depth_path])
    # print("Scene Understanding:\n", scene_info)

    # ---- Step 2: Subtask Decomposition ----
    # print("\nStep 2: Generating grounded subtasks...")
    subtasks = decompose_into_subtasks(
        images=[rgb_path, depth_path],
        goal=goal,
        scene_info=scene_info
    )
    # print("Generated Subtasks:\n", subtasks)

    # ---- Step 3: Verification ----
    # print("\nStep 3: Verifying grounded subtasks...")
    verified_subtasks = verify_subtasks(
        images=[rgb_path, depth_path],
        goal=goal,
        subtasks=subtasks,
        scene_info=scene_info
    )
    # print("Verified Subtasks:\n", verified_subtasks)

    # ---- Step 4: Constraints ----
    # print("\nStep 4: Generating grounded constraints...")
    constraints = generate_constraints(
        images=[rgb_path, depth_path],
        subtasks=verified_subtasks,
        scene_info=scene_info
    )
    # print("Generated Constraints:\n", constraints)

    # ---- Step 5: Final Integration ----
    final_plan = integrate_constraints_with_subtasks(
        verified_subtasks,
        constraints
    )

    constraints_refined = safe_extract_json(constraints)
    if(constraints_refined==None):
        return constraints
    else:
        return constraints_refined

    # print("\nFinal Integrated Grounded Plan:\n", final_plan)

    return final_plan


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

    with open("/home/vijval22569/ri_project/openvla/cons.json") as f:
        storage = json.load(f)

    scores = {}

    # map_dict = {'0':'9','1':'5','2':'0','3':'0','4':'9'}
    map_dict = {'0':'2','1':'3','2':'4','3':'7','4':'4'}
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

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
    print("num tasks in suite: ",num_tasks_in_suite)
    overall_sum = 0.0
    num_tasks = 0
    # storage = {}
    for task_id in tqdm.tqdm(range(num_tasks_in_suite-5)):
        # Get task
        task = task_suite.get_task(task_id)
        scores[task_id] = {}
        # storage[task_id] = {}

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        task_sum = 0.0
        num_eps = 0

        # Start episodes
        task_episodes, task_successes = 0, 0
        scores[task_id] = []
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")

            print(f"Starting episode {task_episodes+1}...")
            # log_file.write(f"Starting episode {task_episodes+1}...\n")

            sys_prompt = """You are an expert judge who is adept at evaluating how well a task has been decomposed into subtasks, and how well the given constraints fit
                            the subtask and the overall task. You will be given an RGB image, the overall task, the subtasks and their 
                            corresponding constraints, and you need to provide a score on an integer scale of 1 to 5, where 1 means gvery poor subtask decomposition 
                            and constraint fidelity, and 5 indicates excellent, coherent decomposition and high fidelity and relevance of the constraints.
                            IMPORTANT: ONLY OUTPUT A SINGLE INTEGER FROM {1,2,3,4,5}. DO NOT OUTPUT ANYTHING ELSE"""
            
            text = "Overall task: " + task_description
            text += "\nSubtasks and Constraints: \n"
            text += str(storage[str(task_id)][str(episode_idx)])

            rgb = f"/home/vijval22569/ri_project/openvla/rgbs/{task_id}_{episode_idx}.png"
            depth = f"/home/vijval22569/ri_project/openvla/depths/{task_id}_{episode_idx}.png"
            ep_sum = 0.0
            num_trials = 0

            messages = make_prompt_llama(images = [rgb], text = text, sys_prompt = sys_prompt)

            for j in range(5):
                outs = prompt_llama(
                        model,
                        processor,
                        messages,
                        images=[rgb],
                        max_new_tokens=10
                    )
                # print("OUT: ",outs)
                # print("---------------------------")
                # print(outs)
                # print("---------------------------")
                score = extract_score(outs.split("Subtasks and Constraints")[-1])
                if(score!=None):
                    ep_sum+=score
                    num_trials+=1
            ep_avg = (ep_sum/num_trials if num_trials!=0 else 0.0)
            scores[task_id].append(ep_avg)
            task_sum+=ep_avg
            num_eps+= (1 if num_trials!=0 else 0)
            
        task_avg = task_sum/num_eps
        print(f"Task {task_id} average score: ",task_avg)
        overall_sum+=task_avg
        num_tasks+=1

    overall_avg = overall_sum/num_tasks
    print("Overall average score: ",overall_avg)      

    with open("scores_llama.json","w") as f:
        json.dump(scores,f,indent = 4)

if __name__ == "__main__":
    eval_libero()
