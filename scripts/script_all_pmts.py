#!/usr/bin/env python3
import os
import numpy as np
import argparse
import fnmatch
from scipy.stats import norm, gaussian_kde
from scipy.optimize import minimize, curve_fit
from scipy.signal import argrelextrema
import json

# ----------------- ARGPARSE -----------------
parser = argparse.ArgumentParser(description="Two-step Gaussian fit for PMTs (batch mode)")
parser.add_argument("--pattern", type=str, default="card*_slot*_ch*_pos*.npz")
parser.add_argument("--chunk-id", type=int, default=0, help="Index of the PMT chunk to process (0,1,2,...)")
parser.add_argument("--chunk-size", type=int, default=100, help="Number of PMTs per job")
args = parser.parse_args()
chunk_id = args.chunk_id
chunk_size = args.chunk_size

# ----------------- DIRECTORIES -----------------
signal_dir = "/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run2307"
out_dir    = "/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run2307/results"
os.makedirs(out_dir, exist_ok=True)

signal_files = [
    f for f in os.listdir(signal_dir)
    if fnmatch.fnmatch(f, args.pattern) and "_part" not in f
]

pmts_all = sorted([f.replace(".npz", "") for f in signal_files])
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

# ----------------- GAUSSIAN FIT -----------------
def fit_gaussian_with_bounds(data, mu0, sigma0, sigma_bounds):
    N = len(data)
    A0 = N

    def nll(params):
        mu, sigma, A = params
        if sigma <= 0 or A <= 0:
            return np.inf
        pdf = A * norm.pdf(data, mu, sigma)
        return -np.sum(np.log(pdf.clip(min=1e-12)))

    bounds = [(-np.inf, np.inf), sigma_bounds, (1, np.inf)]
    try:
        res = minimize(nll, [mu0, sigma0, A0], method="L-BFGS-B", bounds=bounds)
        if not res.success:
            raise RuntimeError("Gaussian fit failed")
        mu_fit, sigma_fit, A_fit = res.x
        err_mu = sigma_fit / np.sqrt(N)
        err_sigma = sigma_fit / np.sqrt(2*N)
        fit_failed = False
    except Exception:
        mu_fit, sigma_fit, A_fit = mu0, sigma0, N
        err_mu, err_sigma = np.inf, np.inf
        fit_failed = True

    return dict(mu=mu_fit, sigma=sigma_fit, A=A_fit, err_mu=err_mu, err_sigma=err_sigma, n=N, fit_failed=fit_failed)

# ----------------- KDE + PEAK SEED -----------------
def fit_pedestal_and_spe(charges, label="PMT"):
    x = np.asarray(charges)
    ped_center, spe_center = 0.0, 0.0
    try:
        xs = np.linspace(np.min(x), np.max(x), 2500)
        kde = gaussian_kde(x)
        ys = kde(xs)
        peak_idx = argrelextrema(ys, np.greater)[0]

        if len(peak_idx) >= 2:
            expected_spe_min, expected_spe_max = 100.0, 200.0
            peak_xs, peak_ys = xs[peak_idx], ys[peak_idx]
            spe_candidates_mask = (peak_xs >= expected_spe_min) & (peak_xs <= expected_spe_max)
            spe_candidates_idx = np.where(spe_candidates_mask)[0]

            if len(spe_candidates_idx) >= 1:
                chosen_spe_rel = spe_candidates_idx[np.argmax(peak_ys[spe_candidates_idx])]
                spe_center = peak_xs[chosen_spe_rel]
                other_idx = [i for i in range(len(peak_xs)) if i != chosen_spe_rel]
                ped_center = peak_xs[other_idx[np.argmin(peak_xs[other_idx])]] if len(other_idx) > 0 else 0.0
            else:
                idx_sorted = peak_idx[np.argsort(ys[peak_idx])][-2:]
                idx_sorted = np.sort(idx_sorted)
                ped_center, spe_center = xs[idx_sorted[0]], xs[idx_sorted[1]]
        else:
            ped_center = np.median(x[x < 50]) if np.any(x < 50) else 0.0
            spe_center = np.median(x[(x >= 100) & (x <= 200)]) if np.any((x >= 100) & (x <= 200)) else 100.0

    except Exception:
        ped_center = 0.0
        spe_center = np.median(x[(x >= 100) & (x <= 200)]) if np.any((x >= 100) & (x <= 200)) else 100.0

    # ---- PED + SPE WINDOW ----
    ped_mask = np.abs(x - ped_center) < 6
    if ped_mask.sum() < 10: ped_mask = np.abs(x - ped_center) < 10
    ped_data = x[ped_mask] if np.any(ped_mask) else x

    sep = abs(spe_center - ped_center)
    spe_hw = max(40, 0.35*sep)
    spe_mask = np.abs(x - spe_center) < spe_hw
    if spe_mask.sum() < 20: spe_mask = np.abs(x - spe_center) < (1.6*spe_hw)
    spe_data = x[spe_mask] if np.any(spe_mask) else x

    pedestal_fit = fit_gaussian_with_bounds(ped_data, np.median(ped_data), np.std(ped_data), (0.1, 10.0))
    spe_fit = fit_gaussian_with_bounds(spe_data, np.median(spe_data), max(20, np.std(spe_data)), (10, 150))

    gain = spe_fit['mu'] - pedestal_fit['mu']
    err_gain = np.sqrt(pedestal_fit['err_mu']**2 + spe_fit['err_mu']**2)
    fit_failed = pedestal_fit['fit_failed'] or spe_fit['fit_failed']

    return dict(pedestal=pedestal_fit, spe=spe_fit, gain=gain, err_gain=err_gain,
                n_ped=len(ped_data), n_spe=len(spe_data), fit_failed=fit_failed)

# ----------------- GAUSSIAN FUNCTION -----------------
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# ----------------- PROCESS PMTs -----------------
results_list, failed_pmts = [], []

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

        all_peaks = [do_pulse_finding(wf) for wf in signal_waveforms]
        pulse_mask = np.array([len(p) > 0 for p in all_peaks])

        charges = np.array([charge_calculation_mPMT_method(wf, (p[0] if len(p) > 0 else int(np.argmax(wf)))) 
                            for wf,p in zip(signal_waveforms, all_peaks)])

        pulse_count = np.sum(pulse_mask)
        total_waveforms = len(signal_waveforms)
        pulse_ratio = pulse_count / total_waveforms if total_waveforms > 0 else np.nan
        mu_pe = -np.log(1 - pulse_ratio) if pulse_ratio < 1 else np.nan

        seeds = fit_pedestal_and_spe(charges, label=f"PMT {pmt_label}")

        # ------------------ χ²/ndof SPE como antes ------------------
        spe_mu = seeds['spe']['mu']
        spe_sigma = seeds['spe']['sigma']

        # histogram solo de SPE window
        spe_mask_hist = (charges >= spe_mu - 2*spe_sigma) & (charges <= spe_mu + 2*spe_sigma)
        spe_charges_hist = charges[spe_mask_hist]
        counts, bin_edges = np.histogram(spe_charges_hist, bins=50)
        bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
        p0 = [len(spe_charges_hist), spe_mu, spe_sigma]

        # fit Gaussian to SPE histogram
        bounds = ([0, spe_mu - 3*spe_sigma, 0.1], [np.inf, spe_mu + 3*spe_sigma, 80.0])
        try:
            params_spe, _ = curve_fit(gaussian, bin_centers, counts, p0=p0, bounds=bounds, maxfev=20000)
            A_spe, mu_spe, sigma_spe = params_spe
            chi2_spe = np.sum((counts - gaussian(bin_centers,*params_spe))**2 / (counts+1))
            ndof_spe = len(counts)-3
            chi2ndof_spe = chi2_spe / ndof_spe if ndof_spe>0 else np.nan
        except Exception:
            chi2ndof_spe = np.nan

        results_list.append((
            card_id, slot_id, channel_id, pos_id,
            seeds['pedestal']['mu'], seeds['pedestal']['sigma'], seeds['n_ped'],
            seeds['spe']['mu'], seeds['spe']['sigma'], seeds['n_spe'],
            seeds['gain'], seeds['err_gain'],
            pulse_ratio, mu_pe,
            chi2ndof_spe,
            seeds['fit_failed']
        ))

    except Exception as e:
        failed_pmts.append((pmt_label, str(e)))
        results_list.append((
            card_id, slot_id, channel_id, pos_id,
            np.nan, np.nan, 0,
            np.nan, np.nan, 0,
            np.nan, np.nan,
            np.nan, np.nan,
            np.nan,
            True
        ))

# ----------------- SAVE RESULTS -----------------
dtype = np.dtype([
    ('card_id','i4'),('slot_id','i4'),('channel_id','i4'),('pos_id','i4'),
    ('pedestal_mean','f8'),('pedestal_sigma','f8'),('N_pedestal','i4'),
    ('spe_mean','f8'),('spe_sigma','f8'),('N_spe','i4'),
    ('gain','f8'),('gain_error','f8'),
    ('pulse_ratio','f8'),('mu_pe','f8'),
    ('chi2ndof_spe','f8'),
    ('fit_failed','?')
])
results_array = np.array(results_list, dtype=dtype)
np.savez(os.path.join(out_dir, f"Final_run2307_chunk{chunk_id}.npz"), results=results_array)

failed_file = os.path.join(out_dir, f"failed_pmts_run2307_chunk{chunk_id}.json")
with open(failed_file,"w") as f: json.dump(failed_pmts,f,indent=2)

print(f"Done. Processed PMTs {start_idx}..{end_idx-1}. Failed PMTs: {len(failed_pmts)}")
