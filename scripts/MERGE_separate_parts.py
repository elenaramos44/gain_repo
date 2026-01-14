#!/usr/bin/env python3
import os
import numpy as np
import json
import argparse
import glob

def merge_raw_waveforms(folder, pmts_json, start_idx, end_idx):
    with open(pmts_json, "r") as f:
        pmts = json.load(f)

    pmts_to_process = pmts[start_idx : min(end_idx + 1, len(pmts))]
    if not pmts_to_process:
        print(f"[INFO] No PMTs to process in range {start_idx}-{end_idx}")
        return

    for card_id, slot_id, ch_id, pos_id in pmts_to_process:
        pattern = os.path.join(
            folder,
            f"card{card_id}_slot{slot_id}_ch{ch_id}_pos{pos_id}_part*.npz"
        )
        part_files = sorted(glob.glob(pattern))
        if not part_files:
            print(f"[WARN] No part files found for PMT {card_id}_{slot_id}_{ch_id}_{pos_id}")
            continue

        waveforms_all = []

        for fpath in part_files:
            try:
                with np.load(fpath, allow_pickle=True) as data:
                    if "waveforms" not in data:
                        print(f"  [WARN] Missing key 'waveforms' in {fpath}")
                        continue
                    w = data["waveforms"]
                    if w.size > 0:
                        waveforms_all.append(w)
            except Exception as e:
                print(f"  [ERROR] Failed to read {fpath}: {e}")

        if not waveforms_all:
            print(f"[ERROR] No valid waveforms for PMT {card_id}_{slot_id}_{ch_id}_{pos_id}")
            continue

        merged_waveforms = np.concatenate(waveforms_all, axis=0)
        outname = os.path.join(
            folder,
            f"card{card_id}_slot{slot_id}_ch{ch_id}_pos{pos_id}.npz"
        )
        np.savez_compressed(outname, waveforms=merged_waveforms)
        print(f"[OK] PMT {card_id}_{slot_id}_{ch_id}_{pos_id} → {merged_waveforms.shape[0]} waveforms")

    print("[DONE] Merge completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge raw waveform NPZ files per PMT")
    parser.add_argument("--folder", required=True)
    parser.add_argument("--pmt-json", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()

    merge_raw_waveforms(
        folder=args.folder,
        pmts_json=args.pmt_json,
        start_idx=args.start,
        end_idx=args.end
    )
