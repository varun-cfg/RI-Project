import re
import glob
import numpy as np

# Regex patterns for each metric
patterns = {
    "final_total_success": r"Total Success Rate:\s*([\d.]+)%",
    "final_raw_jerk": r"Average Raw Jerk Score:\s*([\d.]+)",
    "final_repaired_jerk": r"Average Repaired Jerk Score:\s*([\d.]+)",
    "final_jerk_improvement": r"Jerk Improvement:\s*([\d.]+)%",
    "final_cvr": r"Average CVR:\s*([\d.]+)%"
}

# Store extracted values
results = {key: [] for key in patterns}

# Process all text files in the current folder
file_list = ["/home/vijval22569/ri_project/openvla/experiments/logs/EVAL-libero_spatial-openvla-2025_11_28-11_34_56.txt",
"/home/vijval22569/ri_project/openvla/experiments/logs/EVAL-libero_spatial-openvla-2025_11_28-12_15_46.txt",
"/home/vijval22569/ri_project/openvla/experiments/logs/EVAL-libero_spatial-openvla-2025_11_28-12_57_10.txt",
"/home/vijval22569/ri_project/openvla/experiments/logs/EVAL-libero_spatial-openvla-2025_11_28-13_37_48.txt",
"/home/vijval22569/ri_project/openvla/experiments/logs/EVAL-libero_spatial-openvla-2025_11_28-14_19_09.txt",
"/home/vijval22569/ri_project/openvla/experiments/logs/EVAL-libero_spatial-openvla-2025_11_28-14_59_56.txt",
"/home/vijval22569/ri_project/openvla/experiments/logs/EVAL-libero_spatial-openvla-2025_11_28-15_41_48.txt"]

for filename in file_list:
    with open(filename, "r") as f:
        text = f.read()

    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            results[key].append(float(m.group(1)))
        else:
            print(f"Warning: {key} not found in {filename}")

# Compute mean and std for each metric
print("\n================ METRICS SUMMARY ================")
for key, values in results.items():
    if len(values) == 0:
        print(f"{key}: NO DATA")
        continue

    mean = np.mean(values)
    std = np.std(values)

    print(f"{key:25s}  mean = {mean:.4f},   std = {std:.4f}")


print("results successes: ",results['final_cvr'])
