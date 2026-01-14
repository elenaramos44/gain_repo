# FUNCTIONS

# Find peaks and integrate charges with WCTE given integration window for consistency!

def do_pulse_finding(waveform, debug=False):
    threshold = 20                                # distinguish between pulse and noise waveforms
    fIntegralPreceding = 4
    fIntegralFollowing = 2
    
    above_threshold = np.where(waveform[3:-2] > threshold)[0] + 3
    pulses_found = []
    last_index = 0
    
    for index in above_threshold:
        # local maximum conditions
        if (waveform[index] <= waveform[index-1]): continue
        if (waveform[index] < waveform[index+1]): continue
        if (waveform[index] <= waveform[index+2]): continue
        if (waveform[index] <= waveform[index-2]): continue
        
        # integral condition
        start = max(0, index - fIntegralPreceding)
        end = min(len(waveform), index + fIntegralFollowing + 1)
        integral = np.sum(waveform[start:end])
        if integral < threshold * 2:
            continue
        
        # minimum spacing conditions between pulses
        if (last_index > 0) and (index - last_index) <= 20:
            continue
        
        pulses_found.append(index)
        last_index = index
    
    return pulses_found


def charge_calculation_mPMT_method(wf, peak_sample):
    n = len(wf)
    if peak_sample < 5 or peak_sample + 2 >= n:
        pass
    start = max(0, peak_sample - 5)
    end = min(n, peak_sample + 2)
    charge = np.sum(wf[start:end])
    if peak_sample + 2 < n and wf[peak_sample + 2] > 0:
        charge += wf[peak_sample + 2]
    return charge


#  Unbinned Gaussian negative log-likelihood
def nll_gauss(params, data):
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    pdf = norm.pdf(data, mu, sigma)
    pdf = np.clip(pdf, 1e-12, None)
    return -np.sum(np.log(pdf))


def fit_gaussian_with_bounds(data, mu0, sigma0, sigma_bounds):

    N = len(data)
    A0 = N
    bounds = [(-np.inf, np.inf), sigma_bounds, (1, np.inf)]  # μ free, σ limited, A > 0

    def nll(params):
        mu, sigma, A = params
        if sigma <= 0 or A <= 0:
            return np.inf
        pdf = A * norm.pdf(data, mu, sigma)
        return -np.sum(np.log(pdf.clip(min=1e-12)))

    res = minimize(nll, [mu0, sigma0, A0], method="L-BFGS-B", bounds=bounds)
    if not res.success:
        raise RuntimeError("Gaussian fit failed")

    mu_fit, sigma_fit, A_fit = res.x
    err_mu = sigma_fit / np.sqrt(N)
    err_sigma = sigma_fit / np.sqrt(2*N)

    return dict(mu=mu_fit, 
                sigma=sigma_fit, 
                A=A_fit,
                err_mu=err_mu, 
                err_sigma=err_sigma, n=N
                )

def fit_pedestal_and_spe(charges, label="PMT", plot=True):
    x = np.asarray(charges)
    if len(x) < 80:
        raise RuntimeError("Too few points to identify pedestal + SPE peaks reliably")

    # KDE and peak finding
    xs = np.linspace(np.min(x), np.max(x), 2500)
    kde = gaussian_kde(x)
    ys = kde(xs)
    peak_idx = argrelextrema(ys, np.greater)[0]

    spe_center = None
    ped_center = None

    try:
        if len(peak_idx) < 2:
            raise RuntimeError("Only one peak found - cannot separate pedestal and SPE")

        expected_spe_min = 100.0
        expected_spe_max = 200.0

        peak_xs = xs[peak_idx]
        peak_ys = ys[peak_idx]

        spe_candidates_mask = (peak_xs >= expected_spe_min) & (peak_xs <= expected_spe_max)
        spe_candidates_idx = np.where(spe_candidates_mask)[0]

        if len(spe_candidates_idx) >= 1:
            # strongest SPE inside expected range
            chosen_spe_rel = spe_candidates_idx[np.argmax(peak_ys[spe_candidates_idx])]
            spe_center = peak_xs[chosen_spe_rel]

            # pedestal = lowest remaining peak
            other_idx = [i for i in range(len(peak_xs)) if i != chosen_spe_rel]
            ped_center = peak_xs[other_idx[np.argmin(peak_xs[other_idx])]] if len(other_idx) > 0 else 0.0
        else:
            # fallback: two strongest peaks
            idx_sorted = peak_idx[np.argsort(ys[peak_idx])][-2:]
            idx_sorted = np.sort(idx_sorted)
            ped_center = xs[idx_sorted[0]]
            spe_center = xs[idx_sorted[1]]

    except Exception:
        # If something fails, set pedestal mean to 0 and continue
        ped_center = 0.0
        # choose strongest peak in expected SPE range
        spe_candidates = x[(x >= 100) & (x <= 200)]
        if len(spe_candidates) == 0:
            raise RuntimeError("Cannot find SPE candidates")
        spe_center = np.median(spe_candidates)

    # Define pedestal and SPE regions
    ped_mask = np.abs(x - ped_center) < 6
    if ped_mask.sum() < 20:
        ped_mask = np.abs(x - ped_center) < 10
    ped_data = x[ped_mask]

    sep = abs(spe_center - ped_center)
    spe_hw = max(40, 0.35 * sep)
    spe_mask = np.abs(x - spe_center) < spe_hw
    if spe_mask.sum() < 30:
        spe_mask = np.abs(x - spe_center) < (1.6 * spe_hw)
    spe_data = x[spe_mask]

    if len(spe_data) < 20:
        raise RuntimeError("Not enough points near the SPE peak for a stable fit.")

    # Gaussian fits
    pedestal_fit = fit_gaussian_with_bounds(
        ped_data,
        np.median(ped_data) if len(ped_data) > 0 else 0.0,
        np.std(ped_data) if len(ped_data) > 0 and ped_data.std() > 0 else 1.0,
        (0.1, 2.0)
    )
    
    
    spe_fit = fit_gaussian_with_bounds(
        spe_data,
        np.median(spe_data),
        max(20, np.std(spe_data)),
        (80, 200)
    )

    # gain
    gain = spe_fit["mu"] - ped_center
    err_gain = np.sqrt(pedestal_fit["err_mu"]**2 + spe_fit["err_mu"]**2)

    result = dict(
        pedestal=pedestal_fit,
        spe=spe_fit,
        gain=gain,
        err_gain=err_gain,
        n_ped=len(ped_data),
        n_spe=len(spe_data)
    )

    # Plotting (same as before)
    if plot:
        bins = np.linspace(np.min(x), np.max(x), 500)
        bc = 0.5*(bins[1:] + bins[:-1])
        bw = bins[1] - bins[0]
        fig, ax = plt.subplots(figsize=(9,6), dpi=150)
        ax.hist(x, bins=bins, histtype="step", color="black", lw=1, label="Data")
        ped_pdf = pedestal_fit["n"] * norm.pdf(bc, pedestal_fit["mu"], pedestal_fit["sigma"]) * bw
        spe_pdf  = spe_fit["n"] * norm.pdf(bc,  spe_fit["mu"],  spe_fit["sigma"]) * bw
        ax.plot(bc, ped_pdf, "b--", lw=1.5, label=f"Pedestal μ={pedestal_fit['mu']:.2f}, σ={pedestal_fit['sigma']:.2f}, N={len(ped_data)}")
        ax.plot(bc, spe_pdf,  "g--", lw=1.5, label=f"SPE μ={spe_fit['mu']:.2f}, σ={spe_fit['sigma']:.2f}, N={len(spe_data)}")
        ax.plot(bc, ped_pdf + spe_pdf, "r", lw=1, alpha=0.7, label=f"Sum (Gain = {gain:.2f} ± {err_gain:.2f})")
        ax.set_xlim(-20, 500)
        ax.set_ylim(0, 300)
        ax.set_xlabel("Integrated charge [ADC·ns]")
        ax.set_ylabel(f"counts / {bw:.2f} ADC·ns")
        ax.set_title(f"Run {run_number} - PMT {pmt_label}: Unbinned Gaussian fit for seeds")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=rcParams['legend.fontsize'])
        plt.show()

    return result



# STEP 2: Binned two-Gaussian fits using seeds and limited regions

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))



def fit_and_plot_gaussians(charges, seeds, bins=500):
    x = np.asarray(charges)

    # histogram --> binned
    fig, ax = plt.subplots(figsize=(9,6), dpi=150)
    counts, bin_edges, _ = ax.hist(x, bins=bins, histtype='step', color='black', linewidth=1.2, label='Data')
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    bw = bin_centers[1] - bin_centers[0]

    # Pedestal seeds
    mu_seed = seeds['pedestal']['mu']
    sigma_seed = seeds['pedestal']['sigma']
    n_seed  = seeds['pedestal']['n']

    # fallback if pedestal unreliable
    if mu_seed is None or np.isnan(mu_seed):
        mu_seed = 0.0
        sigma_seed = 1.0
        n_seed = 1

    ped_mask = (bin_centers >= mu_seed - 3*sigma_seed) & (bin_centers <= mu_seed + 3*sigma_seed)
    x_fit_ped = bin_centers[ped_mask]
    y_fit_ped = counts[ped_mask]

    try:
        p0_ped = [n_seed, mu_seed, sigma_seed]
        bounds_ped = ([0, mu_seed - 5*sigma_seed, 0.05], [np.inf, mu_seed + 5*sigma_seed, 8.0])
        params_ped, _ = curve_fit(gaussian, x_fit_ped, y_fit_ped, p0=p0_ped, bounds=bounds_ped, maxfev=20000)
        A_ped, mu_ped, sigma_ped = params_ped
    except Exception:
        # fallback: use seed values
        mu_ped = mu_seed
        sigma_ped = sigma_seed
        A_ped = n_seed
        params_ped = [A_ped, mu_ped, sigma_ped]  # <<< important for plotting

    # SPE seeds
    mu_spe_seed = seeds['spe']['mu']
    sigma_spe_seed = seeds['spe']['sigma']
    n_spe_seed = seeds['spe']['n']

    spe_mask = (
        (bin_centers >= mu_spe_seed - 1.2*sigma_spe_seed) & 
        (bin_centers <= mu_spe_seed + 1.2*sigma_spe_seed) &
        (bin_centers <= mu_spe_seed + 80)
    )
    
    x_fit_spe = bin_centers[spe_mask]
    y_fit_spe = counts[spe_mask]

    p0_spe = [n_spe_seed, mu_spe_seed, sigma_spe_seed]
    bounds_spe = ([0, mu_spe_seed - 3*sigma_spe_seed, 0.1], [np.inf, mu_spe_seed + 3*sigma_spe_seed, 80.0])
    params_spe, _ = curve_fit(gaussian, x_fit_spe, y_fit_spe, p0=p0_spe, bounds=bounds_spe, maxfev=20000)
    A_spe, mu_spe, sigma_spe = params_spe

    # Gain and error
    gain = mu_spe - mu_ped
    N_ped = max(1, len(x[(x >= mu_ped - 3*sigma_ped) & (x <= mu_ped + 3*sigma_ped)]))
    N_spe = len(x[(x >= mu_spe_seed - 2*sigma_spe_seed) & (x <= mu_spe_seed + 2*sigma_spe_seed)])
    err_gain = np.sqrt(sigma_ped**2 / N_ped + sigma_spe**2 / N_spe)

    # χ²/ndof SPE
    chi2_spe = np.sum((y_fit_spe - gaussian(x_fit_spe, *params_spe))**2 / (y_fit_spe + 1))
    ndof_spe = len(y_fit_spe) - 3
    chi2ndof_spe = chi2_spe / ndof_spe if ndof_spe > 0 else np.nan

    # Plot
    x_full = np.linspace(-20, 500, 2000)
    ax.plot(x_fit_ped, gaussian(x_fit_ped, *params_ped), 'b--', lw=2, label=f"Pedestal μ={mu_ped:.2f}, σ={sigma_ped:.2f}, N={N_ped}")
    ax.plot(x_fit_spe, gaussian(x_fit_spe, *params_spe), 'g--', lw=2, label=f"SPE μ={mu_spe:.2f}, σ={sigma_spe:.2f}, N={N_spe}")
    ax.plot(x_full, gaussian(x_full, *params_ped) + gaussian(x_full, *params_spe), 'r-', lw=0.7, label="Sum")

    # Only draw regions if non-empty
    if len(x_fit_ped) > 0:
        ax.axvspan(x_fit_ped[0], x_fit_ped[-1], color='blue', alpha=0.08)
    if len(x_fit_spe) > 0:
        ax.axvspan(x_fit_spe[0], x_fit_spe[-1], color='green', alpha=0.08)

    ypos = max(gaussian(x_fit_spe, *params_spe)) * 1.1 if len(x_fit_spe) > 0 else 0
    xpos = 0.5*(x_fit_spe[0] + x_fit_spe[-1]) if len(x_fit_spe) > 0 else 0
    ax.text(xpos, ypos, f"χ²/ndof SPE = {chi2ndof_spe:.2f}", color='green', fontsize=rcParams['legend.fontsize'], ha='center', va='center')

    ax.set_xlim(-50, 500)
    ax.set_ylim(0, 500)
    ax.set_xlabel("Integrated charge [ADC·ns]")
    ax.set_ylabel(f"counts / {bw:.2f} ADC·ns")
    ax.set_title(f"Run {run_number} - PMT {pmt_label}: Binned two separate Gaussian fit")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.show()

    return dict(
        pedestal=(mu_ped, sigma_ped, A_ped),
        spe=(mu_spe, sigma_spe, A_spe, chi2ndof_spe),
        gain=(gain, err_gain),
        N_ped=N_ped,
        N_spe=N_spe,
        ped_region = (x_fit_ped[0], x_fit_ped[-1]) if len(x_fit_ped) > 0 else (0.0, 0.0),
        spe_region = (x_fit_spe[0], x_fit_spe[-1]) if len(x_fit_spe) > 0 else (0.0, 0.0)

    )

