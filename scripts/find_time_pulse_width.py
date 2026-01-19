#!/usr/bin/env python3
import os
import glob
import json
import uproot
import numpy as np
from datetime import datetime, timedelta  # <-- make sure timedelta is imported

# -----------------------------
# Configuration
# -----------------------------
base_path = "/scratch/elena/WCTE_recovery/PMTs_calib_root_files"
run_number = 2306
tree_name = "WCTEReadoutWindows"

# IMPORTANT: adjust if needed
run_date = "2025-05-20"

# Authoritative pulse-width time windows (from shift log)
PULSE_WINDOWS = {
    "770ps": ("16:25:00", "16:42:00"),
    "780ps": ("16:45:00", "17:01:00"),
    "790ps": ("17:03:00", "17:13:00"),
}

# -----------------------------
# Helper functions
# -----------------------------
def parse_time(hms):
    return datetime.strptime(f"{run_date} {hms}", "%Y-%m-%d %H:%M:%S")

parsed_windows = {
    pw: (parse_time(t0), parse_time(t1))
    for pw, (t0, t1) in PULSE_WINDOWS.items()
}

def part_number_from_filename(f):
    basename = os.path.basename(f)
    part_str = basename.split("S0P")[-1].split(".root")[0]
    return int(part_str)

def assign_pulse_width(start_time):
    for pw, (t0, t1) in parsed_windows.items():
        if t0 <= start_time <= t1:
            return pw
    return "discard"

# -----------------------------
# Find all part files
# -----------------------------
pattern = os.path.join(base_path, f"WCTE_offline_R{run_number}S0P*.root")
files = glob.glob(pattern)
if not files:
    raise FileNotFoundError(f"No ROOT files found with pattern: {pattern}")

files_sorted = sorted(files, key=part_number_from_filename)

# -----------------------------
# Loop over parts and assign pulse width
# -----------------------------
part_pulse_width = {}

for f in files_sorted:
    part = f"P{part_number_from_filename(f)}"

    with uproot.open(f"{f}:{tree_name}") as tree:
        # Read first event timestamp from 'window_time' (in ns)
        times_ns = tree["window_time"].array(entry_start=0, entry_stop=1)
        start_time_ns = times_ns[0]
        start_time_s = start_time_ns * 1e-9  # convert ns -> s

        # Convert to wall-clock datetime
        # Run start is 16:22:00 according to your log
        start_time = parse_time("16:22:00") + timedelta(seconds=start_time_s)

    pw = assign_pulse_width(start_time)
    part_pulse_width[part] = {
        "start_time": start_time.strftime("%H:%M:%S"),
        "pulse_width": pw
    }

# -----------------------------
# Save JSON
# -----------------------------
out_json = os.path.join(base_path, f"run{run_number}_part_pulse_widths.json")
with open(out_json, "w") as f:
    json.dump(part_pulse_width, f, indent=4)

print(f"[INFO] Pulse width dictionary saved to {out_json}")

# Summary
summary = {}
for v in part_pulse_width.values():
    summary[v["pulse_width"]] = summary.get(v["pulse_width"], 0) + 1

print("[SUMMARY]")
for k, v in summary.items():
    print(f"  {k}: {v} parts")
