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

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
model = model.to("cuda")
model_type="DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

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

# code taken from
def prompt_qwen(model, processor, messages, max_new_tokens=256, temperature=0.01):

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



with open("cons.json") as f:
    cons = json.load(f)

store = {}

from tqdm import tqdm
for tid in tqdm(cons):
    store[tid] = {}
    for ep in cons[tid]:
        sys_prompt = """You are an expert summarizer. You will be given some constraints related to a robotics task. you need to create a concise, general one-line summary of these. 
                        do NOT give specific constraints.
                        IMPORTANT: DO NOT give specific constraints like lift the bowl this much, etc. Be general, like move the bowl smoothly. 
                        Give GENERAL motion-related constraints, not task-specific. do NOT refer to specific objects, be as general as possible.
                        Keep summaries SHORT."""
        text = "Constraints:\n"
        counter = 1
        if (not isinstance(cons[tid][ep],dict)):
            res = "move smoothly and gently"
        else:
            for k in cons[tid][ep]['constraints']:
                text+=f"{counter+1}. {k['constraint']}"
            prompt = make_prompt(text = text, sys_prompt = sys_prompt)
            res = prompt_qwen(model,processor, prompt)
        
        # print("res: ",res)
        store[tid][ep] = res

with open("sums.json","w") as f:
    json.dump(store,f,indent = 4)
