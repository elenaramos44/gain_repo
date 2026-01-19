import uproot
import glob
import json
import os
import math

RUN = 2308
CHUNK_SIZE = 250
BASE_PATH = "/scratch/elena/WCTE_recovery/PMTs_calib_root_files"

parts = {}

files = sorted(glob.glob(f"{BASE_PATH}/WCTE_offline_R{RUN}S0P*.root"))

for f in files:
    part = int(f.split("P")[-1].replace(".root", ""))
    with uproot.open(f) as file:
        tree = file["WCTEReadoutWindows"]
        n_events = tree.num_entries
        n_chunks = math.ceil(n_events / CHUNK_SIZE)
        parts[part] = n_chunks
        print(f"Part {part}: {n_events} events → {n_chunks} chunks")

with open("chunks_per_part.json", "w") as out:
    json.dump(parts, out, indent=2)

print("Saved chunks_per_part.json")
