#!/usr/bin/env python3
import uproot
import glob
import math
import json

# ---------- USER PARAMETERS ----------
RUN = 2307
CHUNK_SIZE = 250
TREE_NAME = "WCTEReadoutWindows"
ROOT_DIR = "/dipc/elena/WCTE_2025_commissioning/root_files/PMTs_calib"
OUTPUT_JSON = "chunks_per_part.json"
# -------------------------------------

parts = {}

# Find all ROOT files for this run
root_files = sorted(glob.glob(f"{ROOT_DIR}/WCTE_offline_R{RUN}S0P*.root"))
print(f"Found {len(root_files)} ROOT files for run {RUN}:")
for f in root_files:
    print("  ", f)

# Process each part
for fname in root_files:
    part = int(fname.split("P")[-1].split(".")[0])

    with uproot.open(f"{fname}:{TREE_NAME}") as tree:
        n_events = tree.num_entries

    n_chunks = math.ceil(n_events / CHUNK_SIZE)
    parts[part] = n_chunks
    print(f"Part {part}: {n_events} events → {n_chunks} chunks")

# Save JSON with number of chunks per part
with open(OUTPUT_JSON, "w") as f:
    json.dump(parts, f, indent=2)

print(f"\nPre-scan complete. Results saved in {OUTPUT_JSON}")
