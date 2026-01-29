#!/usr/bin/env python3
import json

# -----------------------------
# Paths
# -----------------------------
json_file = "/scratch/elena/WCTE_recovery/PMTs_calib_root_files/run2055_good_parts.json"
tasks_file = "/scratch/elena/WCTE_recovery/scripts/tasks_run2055.txt"

# -----------------------------
# Load JSON
# -----------------------------
with open(json_file, "r") as f:
    data = json.load(f)

# -----------------------------
# Collect good parts
# -----------------------------
tasks = []

for part_str, info in data.items():
    status = info.get("status", "").strip().lower()  # robust check
    if status == "good":
        # remove "P" prefix, strip whitespace, convert to int
        part_number = int(part_str.strip()[1:])
        tasks.append(part_number)

# Sort parts numerically
tasks.sort()

# Convert to task strings (part_number, chunk_id=0)
task_lines = [f"{p} 0" for p in tasks]

# -----------------------------
# Write tasks file
# -----------------------------
with open(tasks_file, "w") as f:
    f.write("\n".join(task_lines))

print(f"[INFO] {len(task_lines)} tasks written to {tasks_file}")
print(f"[INFO] Parts: {task_lines}")
