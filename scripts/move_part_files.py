#!/usr/bin/env python3
import os
import json
import shutil
import glob

# -----------------------------
# Configuration
# -----------------------------
base_path = "/scratch/elena/WCTE_recovery/PMTs_calib_root_files"
run_number = 2055
json_file = os.path.join(base_path, f"run{run_number}_good_parts.json")

# -----------------------------
# Load JSON
# -----------------------------
with open(json_file, "r") as f:
    part_status = json.load(f)

# -----------------------------
# List ROOT files
# -----------------------------
pattern = os.path.join(base_path, f"WCTE_offline_R{run_number}S0P*.root")
files = glob.glob(pattern)

if not files:
    raise FileNotFoundError(f"No ROOT files found with pattern: {pattern}")

# -----------------------------
# Move only GOOD parts
# -----------------------------
good_dir = os.path.join(base_path, "good")
os.makedirs(good_dir, exist_ok=True)

n_good = 0

for f in files:
    basename = os.path.basename(f)
    part_str = basename.split("S0P")[-1].split(".root")[0]
    part_key = f"P{part_str}"

    if part_key not in part_status:
        print(f"[WARN] {part_key} not found in JSON, skipping.")
        continue

    status = part_status[part_key]["status"]

    if status != "good":
        continue

    dest_path = os.path.join(good_dir, basename)
    shutil.move(f, dest_path)
    print(f"[INFO] Moved {basename} → {good_dir}")
    n_good += 1

print(f"[SUMMARY] Moved {n_good} GOOD parts to {good_dir}")
