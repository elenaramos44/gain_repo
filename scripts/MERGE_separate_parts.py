#!/usr/bin/env python3

import os
import numpy as np
import json
import argparse
import glob


def merge_pmts_all_parts(folder, pmts_json, start_idx, end_idx, delete_parts=True):
    """
    Merge all part*.npz files for each PMT into a single NPZ.
    Optionally delete per-part files after successful merge.
    """

    with open(pmts_json, "r") as f:
        pmts = json.load(f)  # list of [card, slot, ch, pos]

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
        total_wfs = 0

        for fpath in part_files:
            try:
                with np.load(fpath, allow_pickle=True) as data:
                    w = data.get("waveforms", None)
                    if w is not None and w.size > 0:
                        waveforms_all.append(w)
                        total_wfs += w.shape[0]
                    else:
                        print(f"  [WARN] Empty waveforms in {os.path.basename(fpath)}")
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

        print(
            f"[OK] PMT {card_id}_{slot_id}_{ch_id}_{pos_id} "
            f"→ {merged_waveforms.shape[0]} waveforms"
        )

        # Remove part files to free space
        if delete_parts:
            for fpath in part_files:
                os.remove(fpath)

    print("[DONE] Merge completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge all part NPZ files into one NPZ per PMT"
    )
    parser.add_argument("--folder", required=True)
    parser.add_argument("--pmt-json", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--keep-parts", action="store_true",
                        help="Do NOT delete per-part files")

    args = parser.parse_args()

    merge_pmts_all_parts(
        folder=args.folder,
        pmts_json=args.pmt_json,
        start_idx=args.start,
        end_idx=args.end,
        delete_parts=not args.keep_parts
    )
