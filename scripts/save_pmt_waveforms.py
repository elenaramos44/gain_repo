import os
import glob
import json
import numpy as np
import awkward as ak
import uproot
from datetime import datetime
from collections import defaultdict


def load_root_part(run_number, part=None, chunk_id=None, chunk_size=10,
                   tree_name="WCTEReadoutWindows", max_events=None,
                   base_path=None, verbose=False, quiet=False):
    """
    Load a single part (or all parts if part=None) of a run from ROOT files.
    Supports optional chunking by chunk_id and chunk_size.
    """
    if base_path is None:
        base_path = "/scratch/elena/WCTE_recovery/PMTs_calib_root_files"

    if part is not None:
        pattern = os.path.join(base_path, f"WCTE_offline_R{run_number}S0P{part}.root")
        files = [pattern] if os.path.exists(pattern) else []
    else:
        pattern = os.path.join(base_path, f"WCTE_offline_R{run_number}S0P*.root")
        files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No ROOT files found for pattern: {pattern}")

    if verbose:
        print(f"[INFO] Found {len(files)} ROOT files for run {run_number}.")

    arrays = []
    for f in files:
        if verbose:
            print(f"[INFO] Opening {os.path.basename(f)}")
        with uproot.open(f"{f}:{tree_name}") as tree:
            n_entries = tree.num_entries
            n_to_read = min(n_entries, max_events) if max_events is not None else n_entries

            if chunk_id is not None:
                start = chunk_id * chunk_size
                stop = min(start + chunk_size, n_to_read)
                if start >= n_to_read:
                    if not quiet:
                        print(f"[WARN] chunk_id {chunk_id} out of range for part {part}")
                    return ak.Array([])
                arr = tree.arrays(
                    [
                        "pmt_waveforms",
                        "pmt_waveform_mpmt_card_ids",
                        "pmt_waveform_mpmt_slot_ids",
                        "pmt_waveform_pmt_channel_ids",
                        "pmt_waveform_pmt_position_ids"
                    ],
                    entry_start=start,
                    entry_stop=stop,
                    library="ak",
                )
                arrays.append(arr)
            else:
                for start in range(0, n_to_read, chunk_size):
                    stop = min(start + chunk_size, n_to_read)
                    arr = tree.arrays(
                        [
                            "pmt_waveforms",
                            "pmt_waveform_mpmt_card_ids",
                            "pmt_waveform_mpmt_slot_ids",
                            "pmt_waveform_pmt_channel_ids",
                            "pmt_waveform_pmt_position_ids"
                        ],
                        entry_start=start,
                        entry_stop=stop,
                        library="ak",
                    )
                    arrays.append(arr)

    if not arrays:
        return ak.Array([])

    return ak.concatenate(arrays, axis=0)


def process_and_save(run_number, outdir, part=None, chunk_id=None, chunk_size=10,
                     max_events=None, verbose=False, quiet=False):
    """
    Load ROOT part(s), subtract baseline, group waveforms per PMT, save compressed NPZs.
    """
    if verbose:
        print(f"[INFO] Processing run {run_number}, part={part}, chunk={chunk_id}")

    data = load_root_part(run_number, part=part, chunk_id=chunk_id, chunk_size=chunk_size,
                          max_events=max_events, verbose=verbose, quiet=quiet)

    if len(data) == 0:
        if not quiet:
            print(f"[WARN] No events found for part={part}, chunk={chunk_id}")
        return

    waveforms = data["pmt_waveforms"]
    card_ids = data["pmt_waveform_mpmt_card_ids"]
    slot_ids = data["pmt_waveform_mpmt_slot_ids"]
    channel_ids = data["pmt_waveform_pmt_channel_ids"]
    pos_ids = data["pmt_waveform_pmt_position_ids"]

    waveforms_per_pmt = defaultdict(list)

    # Process events
    for evt_wfs, evt_cids, evt_sids, evt_chids, evt_posids in zip(
        waveforms, card_ids, slot_ids, channel_ids, pos_ids
    ):
        for wf, cid, sid, chid, posid in zip(evt_wfs, evt_cids, evt_sids, evt_chids, evt_posids):
            if sid == -1 or cid > 120:
                continue
            pmt_key = (int(cid), int(sid), int(chid), int(posid))
            wf_np = np.array(wf, dtype=np.float32)
            baseline = np.mean(wf_np[:10])
            wf_corrected = wf_np - baseline
            waveforms_per_pmt[pmt_key].append(wf_corrected)

    os.makedirs(outdir, exist_ok=True)

    # Save NPZ per PMT
    for pmt_key, wfs in waveforms_per_pmt.items():
        cid, sid, chid, posid = pmt_key
        wfs_array = np.array(wfs, dtype=np.float32)
        file_name = f"card{cid}_slot{sid}_ch{chid}_pos{posid}_part{part}.npz"
        file_path = os.path.join(outdir, file_name)
        np.savez_compressed(file_path, waveforms=wfs_array)
        if verbose:
            print(f"[INFO] Saved {len(wfs)} waveforms → {file_path}")

    # Save metadata
    metadata = {
        "run_number": run_number,
        "part": part,
        "chunk_id": chunk_id,
        "output_directory": outdir,
        "total_events_processed": len(data),
        "total_pmts": len(waveforms_per_pmt),
        "timestamp": datetime.now().isoformat()
    }
    meta_path = os.path.join(outdir, f"metadata_run{run_number}_part{part}_chunk{chunk_id}.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    if verbose:
        print(f"[INFO] Metadata saved → {meta_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process and save PMT waveforms from ROOT files")
    parser.add_argument("--base-path", type=str, default=None, help="Path where ROOT files are located")
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--part", type=int, default=None, help="Part number (P0, P1, ...)")
    parser.add_argument("--chunk-id", type=int, default=None, help="Chunk ID within the part")
    parser.add_argument("--chunk-size", type=int, default=10, help="Events per chunk")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    process_and_save(
        run_number=args.run,
        outdir=args.outdir,
        part=args.part,
        chunk_id=args.chunk_id,
        chunk_size=args.chunk_size,
        max_events=args.max_events,
        verbose=args.verbose,
        quiet=args.quiet
    )
