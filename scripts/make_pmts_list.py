#!/usr/bin/env python3

import os
import json
import glob
import re

FOLDER = "/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run2307"
OUTFILE = os.path.join(FOLDER, "pmts_list.json")

pattern = re.compile(
    r"card(\d+)_slot(\d+)_ch(\d+)_pos(\d+)_part\d+\.npz"
)

pmts = set()

for f in glob.glob(os.path.join(FOLDER, "card*_part*.npz")):
    name = os.path.basename(f)
    m = pattern.match(name)
    if m:
        pmts.add(tuple(map(int, m.groups())))

pmts = sorted(pmts)

with open(OUTFILE, "w") as f:
    json.dump(pmts, f, indent=2)

print(f"[OK] Saved {len(pmts)} PMTs to {OUTFILE}")
