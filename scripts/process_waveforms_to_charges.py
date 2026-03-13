#!/usr/bin/env python3
"""
STEP 1: Process waveform npz files and extract charges and useful parameters.
Output is npz per PMT, used later for fitting in STEP 2.
"""

import os
import numpy as np
import argparse
import fnmatch

parser = argparse.ArgumentParser(description="Process waveforms into charges (STEP 1)")
parser.add_argument("--pattern", type=str, default="card*_slot*_ch*_pos*.npz")
parser.add_argument("--chunk-id", type=int, default=0, help="Index of the PMT chunk to process (0,1,2,...)")
parser.add_argument("--chunk-size", type=int, default=100, help="Number of PMTs per job")
parser.add_argument("--signal-dir", type=str, required=True, help="Directory with waveform NPZs")
parser.add_argument("--out-dir", type=str, required=True, help="Directory to save charge NPZs")
args = parser.parse_args()
chunk_id = args.chunk_id
chunk_size = args.chunk_size
signal_dir = args.signal_dir
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

# ----------------- LIST PMTs -----------------
signal_files = [
    f for f in os.listdir(signal_dir)
    if fnmatch.fnmatch(f, args.pattern) and "_part" not in f
]
pmts_all = sorted([f.replace(".npz","") for f in signal_files])
start_idx = chunk_id * chunk_size
end_idx = min(start_idx + chunk_size, len(pmts_all))

# ----------------- WCTE FUNCTIONS -----------------

def do_pulse_finding(waveform):
    threshold = 20
    fIntegralPreceding = 4
    fIntegralFollowing = 2

    above_threshold = np.where(waveform[3:-2] > threshold)[0] + 3
    pulses_found = []
    last_index = 0

    for index in above_threshold:
        if (waveform[index] <= waveform[index-1]): continue
        if (waveform[index] < waveform[index+1]): continue
        if (waveform[index] <= waveform[index+2]): continue
        if (waveform[index] <= waveform[index-2]): continue

        start = max(0, index - fIntegralPreceding)
        end = min(len(waveform), index + fIntegralFollowing + 1)
        integral = np.sum(waveform[start:end])
        if integral < threshold * 2: continue
        if (last_index > 0) and (index - last_index) <= 20: continue

        pulses_found.append(index)
        last_index = index

    return pulses_found

def charge_calculation_mPMT_method(wf, peak_sample):
    n = len(wf)
    start = max(0, peak_sample - 5)
    end = min(n, peak_sample + 2)
    charge = np.sum(wf[start:end])
    if peak_sample + 2 < n and wf[peak_sample + 2] > 0:
        charge += wf[peak_sample + 2]
    return charge

# ----------------- PROCESS PMTs -----------------
for idx, pmt_label in enumerate(pmts_all[start_idx:end_idx], start=start_idx):
    try:
        parts = pmt_label.split("_")
        card_id = int(parts[0].replace("card",""))
        slot_id = int(parts[1].replace("slot",""))
        channel_id = int(parts[2].replace("ch",""))
        pos_id = int(parts[3].replace("pos",""))

        signal_npz = os.path.join(signal_dir, pmt_label + ".npz")
        data = np.load(signal_npz)
        signal_waveforms = data["waveforms"]

        # --- pulse finding ---
        all_peaks = [do_pulse_finding(wf) for wf in signal_waveforms]
        pulse_mask = np.array([len(p) > 0 for p in all_peaks])

        #---------------- charge calculation ------------------------------
        
        charges = np.array([charge_calculation_mPMT_method(wf, (p[0] if len(p) > 0 else int(np.argmax(wf))))
                            for wf,p in zip(signal_waveforms, all_peaks)])

        
        pulse_count = np.sum(pulse_mask)
        total_waveforms = len(signal_waveforms)
        pulse_ratio = pulse_count / total_waveforms if total_waveforms > 0 else np.nan
        mu_pe = -np.log(1 - pulse_ratio) if pulse_ratio < 1 else np.nan

        # --- save results ---
        outname = os.path.join(out_dir, f"{pmt_label}_charges.npz")
        np.savez_compressed(outname,
                            charges=charges,
                            pulse_ratio=pulse_ratio,
                            mu_pe=mu_pe,
                            n_waveforms=total_waveforms)
        print(f"[OK] PMT {pmt_label} → {len(charges)} charges saved")

    except Exception as e:
        print(f"[ERROR] PMT {pmt_label}: {e}")
