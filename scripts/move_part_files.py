#!/usr/bin/env python3
import os
import json
import shutil
import glob

# -----------------------------
# Configuration
# -----------------------------
base_path = "/scratch/elena/WCTE_recovery/PMTs_calib_root_files"
run_number = 2306
json_file = os.path.join(base_path, f"run{run_number}_part_pulse_widths.json")

# -----------------------------
# Load JSON
# -----------------------------
with open(json_file, "r") as f:
    part_pulse_width = json.load(f)

# -----------------------------
# Move files according to pulse width
# -----------------------------
# List all part files
pattern = os.path.join(base_path, f"WCTE_offline_R{run_number}S0P*.root")
files = glob.glob(pattern)

if not files:
    raise FileNotFoundError(f"No ROOT files found with pattern: {pattern}")

for f in files:
    basename = os.path.basename(f)
    part_str = basename.split("S0P")[-1].split(".root")[0]
    part_key = f"P{part_str}"

    if part_key not in part_pulse_width:
        print(f"[WARN] Part {part_key} not in JSON, skipping.")
        continue

    pw = part_pulse_width[part_key]["pulse_width"]

    if pw == "discard":
        print(f"[INFO] Part {part_key} is 'discard', leaving in place.")
        continue

    # Destination folder
    dest_dir = os.path.join(base_path, pw)
    os.makedirs(dest_dir, exist_ok=True)

    # Move file
    dest_path = os.path.join(dest_dir, basename)
    shutil.move(f, dest_path)
    print(f"[INFO] Moved {basename} -> {dest_dir}")

print("[INFO] All parts moved according to pulse width.")
