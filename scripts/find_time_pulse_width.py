#!/usr/bin/env python3
import os
import glob
import json
import uproot
from datetime import datetime, timedelta

# -----------------------------
# Configuration
# -----------------------------
base_path = "/scratch/elena/WCTE_recovery/PMTs_calib_root_files"
run_number = 2055
tree_name = "WCTEReadoutWindows"

run_date = "2025-05-08"     # <-- fill in
RUN_START_TIME = "17:03:00"

# Laser ON window (position 3)
GOOD_WINDOW = ("17:41:00", "17:51:00")

# -----------------------------
# Helpers
# -----------------------------
def parse_time(hms):
    return datetime.strptime(f"{run_date} {hms}", "%Y-%m-%d %H:%M:%S")

run_start = parse_time(RUN_START_TIME)
good_start, good_end = map(parse_time, GOOD_WINDOW)

def part_number_from_filename(f):
    return int(os.path.basename(f).split("S0P")[-1].split(".root")[0])

# -----------------------------
# Locate ROOT files
# -----------------------------
pattern = os.path.join(base_path, f"WCTE_offline_R{run_number}S0P*.root")
files = sorted(glob.glob(pattern), key=part_number_from_filename)

if not files:
    raise RuntimeError("No ROOT files found")

# -----------------------------
# Classify parts
# -----------------------------
part_status = {}

for f in files:
    part = f"P{part_number_from_filename(f)}"

    with uproot.open(f"{f}:{tree_name}") as tree:
        t_ns = tree["window_time"].array(entry_start=0, entry_stop=1)[0]
        t_s = t_ns * 1e-9
        part_start_time = run_start + timedelta(seconds=t_s)

    status = "good" if good_start <= part_start_time <= good_end else "discard"

    part_status[part] = {
        "start_time": part_start_time.strftime("%H:%M:%S"),
        "status": status
    }

# -----------------------------
# Save JSON
# -----------------------------
out_json = os.path.join(base_path, f"run{run_number}_good_parts.json")
with open(out_json, "w") as f:
    json.dump(part_status, f, indent=4)

print(f"[INFO] Saved {out_json}")

# Summary
good = sum(v["status"] == "good" for v in part_status.values())
print(f"[SUMMARY] Good parts: {good} / {len(part_status)}")
