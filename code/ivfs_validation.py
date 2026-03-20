from __future__ import annotations

"""main script -- run this one and it does the whole calibration + scenarios + figures"""

import gzip
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

from common import ASSETS_DIR, HIGGS_GZ, HIGGS_TXT, HIGGS_URL, ensure_layout

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


FIGURE_PATH = ASSETS_DIR / 'apm348_results.png'
DIAG_FIGURE_PATH = ASSETS_DIR / 'apm348_calibration_diagnostics.png'
PHI_SENS_FIGURE_PATH = ASSETS_DIR / 'phi_sensitivity.png'

# --- model params (shared across files) ---
KAPPA = 0.8
ETA = 0.3
PHI = 0.056   # external reference ratio; not calibrated directly from Higgs
PSI = 0.1
RHO = 0.06
LAMBDA_U = 0.02
NU = 1.0
MU_C = 0.01
DELTA = 0.05
W = 10.0

# --- calibration / plotting constants ---
FIT_WINDOW_HOURS = 100
DISPLAY_WINDOW_HOURS = 50
SPIKE_WINDOW_HOURS = 35
TAIL_START_HOURS = 40
SCENARIO_DISPLAY_HOURS = 200
SMOOTH_WINDOW = 3

IVF_PARAM_BOUNDS = (
    (0.2, 4.0),     # beta0
    (0.05, 1.0),    # gamma0
    (0.0, 0.2),     # lambda0
    (0.001, 0.2),   # lambda decay
    (1e-5, 0.1),    # V0
)


def ensure_dataset() -> None:
    """download + unzip the higgs data if we don't have it yet"""
    ensure_layout()

    # if the extracted activity file is already here, that's enough for all current scripts
    if HIGGS_TXT.exists():
        return

    if not HIGGS_GZ.exists():
        print('[data] dont have the higgs gz, downloading from SNAP...')
        urllib.request.urlretrieve(HIGGS_URL, HIGGS_GZ)
        print(f'[data] saved to {HIGGS_GZ}')

    if not HIGGS_TXT.exists() or (HIGGS_GZ.exists() and HIGGS_TXT.stat().st_mtime < HIGGS_GZ.stat().st_mtime):
        print('[data] unzipping higgs activity file...')
        with gzip.open(HIGGS_GZ, 'rt') as src, HIGGS_TXT.open('w') as dst:
            shutil.copyfileobj(src, dst)
        print(f'[data] Saved {HIGGS_TXT}')


def parse_activity_file(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """read the activity file and grab RT + RE timestamps"""
    rt_timestamps: list[int] = []
    re_timestamps: list[int] = []
    total_rows = 0

    with path.open('r') as handle:
        for line in handle:
            total_rows += 1
            parts = line.split()
            if len(parts) >= 4:
                if parts[3] == 'RT':
                    rt_timestamps.append(int(parts[2]))
                elif parts[3] == 'RE':
                    re_timestamps.append(int(parts[2]))

    return (np.array(rt_timestamps, dtype=np.int64),
            np.array(re_timestamps, dtype=np.int64),
            total_rows)


def build_hourly_curve(rt_timestamps: np.ndarray,
                       re_timestamps: np.ndarray | None = None) -> dict:
    """bin RTs (and optionally REs) into hours, return the 100h calibration window"""
    t0 = int(rt_timestamps.min())
    hours = (rt_timestamps - t0) / 3600.0
    max_hour = int(np.ceil(hours.max()))
    hourly_counts, _ = np.histogram(hours, bins=np.arange(0, max_hour + 1, 1))

    peak_hour = int(np.argmax(hourly_counts))

    # put the peak near the start so i catch both the rise and the tail
    start = max(0, peak_hour - 10)
    end = min(len(hourly_counts), start + 100)
    start = max(0, end - 100)

    window_counts = hourly_counts[start:end].astype(float)

    result = {
        'hourly_counts': hourly_counts,
        'peak_hour': peak_hour,
        'window_start': start,
        'rt_window': window_counts,
    }

    # if we got RE timestamps, bin those too and build a reply-pressure proxy
    if re_timestamps is not None and len(re_timestamps) > 0:
        hours_re = (re_timestamps - t0) / 3600.0
        re_hourly, _ = np.histogram(hours_re, bins=np.arange(0, max_hour + 1, 1))
        re_window = re_hourly[start:end].astype(float)
        raw_proxy = re_window / (window_counts + 1.0)
        kernel = np.ones(3) / 3.0
        smoothed = np.convolve(raw_proxy, kernel, mode='same')
        smoothed = np.clip(smoothed, 0.0, 1.0)
        result['re_window'] = re_window
        result['reply_proxy'] = smoothed

    return result


def moving_average(series: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """simple centered moving average for plotting trend vs raw hourly noise"""
    if window <= 1:
        return series.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(series, kernel, mode='same')


def _clip_to_bounds(params: np.ndarray, bounds=IVF_PARAM_BOUNDS) -> np.ndarray:
    clipped = np.array(params, dtype=float, copy=True)
    for i, (lo, hi) in enumerate(bounds):
        clipped[i] = float(np.clip(clipped[i], lo, hi))
    return clipped


def _powell_bounded(loss_fn,
                    x0: np.ndarray,
                    bounds,
                    maxiter: int = 3000) -> tuple[float, np.ndarray]:
    x0 = np.array(x0, dtype=float, copy=True)
    for i, (lo, hi) in enumerate(bounds):
        x0[i] = float(np.clip(x0[i], lo, hi))
    res = minimize(
        loss_fn,
        x0=x0,
        method='Powell',
        bounds=bounds,
        options={'maxiter': maxiter, 'xtol': 1e-5, 'ftol': 1e-5},
    )
    x_star = np.array(res.x, dtype=float, copy=True)
    for i, (lo, hi) in enumerate(bounds):
        x_star[i] = float(np.clip(x_star[i], lo, hi))
    return float(loss_fn(x_star)), x_star


def simulate_basic_ivf(beta: float,
                       gamma: float,
                       lambda0: float,
                       lambda_decay: float,
                       v0: float,
                       n_steps: int,
                       mu: float = 0.01) -> np.ndarray:
    """simulate the baseline IVF fit with a decaying exogenous seed term"""
    def ivf_ode(y, t, beta_val, gamma_val, mu_val, lam0_val, lam_decay_val):
        i_val, v_val, f_val = y
        lam_t = lam0_val * np.exp(-lam_decay_val * t)
        di = -beta_val * i_val * v_val - mu_val * i_val + lam_t
        dv = beta_val * i_val * v_val - gamma_val * v_val - mu_val * v_val
        df = gamma_val * v_val - mu_val * f_val
        return [di, dv, df]

    t_data = np.arange(n_steps, dtype=float)
    v0 = float(np.clip(v0, 1e-5, 0.1))
    fit_sol = odeint(ivf_ode, [1.0 - v0, v0, 0.0], t_data,
                     args=(beta, gamma, mu, lambda0, lambda_decay), mxstep=10000)
    v_fit = fit_sol[:, 1]
    return v_fit / (np.max(v_fit) + 1e-12)


def fit_basic_ivf(v_empirical_norm: np.ndarray,
                  window_counts: np.ndarray | None = None,
                  n_steps: int | None = None) -> tuple[float, float, float, float, float, float, np.ndarray]:
    """fit IVF to the full Higgs window with a decaying exogenous tail term"""
    t_data = np.arange(len(v_empirical_norm), dtype=float)
    if window_counts is not None:
        weights = 1.0 / (np.sqrt(window_counts) + 1.0)
    else:
        weights = np.ones_like(v_empirical_norm)

    def loss(params):
        beta, gamma, lambda0, lambda_decay, v0 = _clip_to_bounds(np.abs(params))
        v_sim_norm = simulate_basic_ivf(
            float(beta),
            float(gamma),
            float(lambda0),
            float(lambda_decay),
            float(v0),
            n_steps=len(v_empirical_norm),
        )
        vmax = float(np.max(v_sim_norm))
        if vmax <= 1e-12:
            return 1e9
        return float(np.sum(weights * (v_sim_norm - v_empirical_norm) ** 2))

    # a few broad starts are enough once the seed term decays instead of flattening the tail
    seeds = [
        np.array([1.0, 0.30, 0.10, 0.04, 0.0001]),
        np.array([1.3, 0.20, 0.05, 0.03, 0.0005]),
        np.array([0.8, 0.40, 0.15, 0.05, 0.0001]),
        np.array([1.6, 0.15, 0.02, 0.02, 0.0010]),
    ]
    best_sse = 1e9
    best_x = _clip_to_bounds(seeds[0])
    for seed in seeds:
        sse, x_star = _powell_bounded(loss, seed, IVF_PARAM_BOUNDS, maxiter=3000)
        if sse < best_sse:
            best_sse = sse
            best_x = x_star

    beta0, gamma0, lambda0, lambda_decay, v0_fit = best_x
    if n_steps is None:
        n_steps = len(v_empirical_norm)
    v_fit_norm = simulate_basic_ivf(
        float(beta0),
        float(gamma0),
        float(lambda0),
        float(lambda_decay),
        float(v0_fit),
        n_steps=n_steps,
    )
    return (
        float(beta0),
        float(gamma0),
        float(lambda0),
        float(lambda_decay),
        float(v0_fit),
        float(best_sse),
        v_fit_norm,
    )


def _ivfs_rhs(y, t, alpha, beta0, gamma0,
              kappa, eta, phi, psi, rho, lambda_u, nu, mu_c, delta, w):
    """the full 6-state ODE rhs (I, V, F, S, tau, U)"""
    i_val, v_val, f_val, s_val, tau, u_val = y

    i_pos = max(i_val, 0.0)
    v_pos = max(v_val, 0.0)
    f_pos = max(f_val, 0.0)
    s_pos = max(s_val, 0.0)
    tau_pos = max(tau, 0.0)
    u_pos = max(u_val, 0.0)

    beta_eff = alpha * beta0 * (1.0 + kappa * tau_pos)
    gamma_eff = gamma0 * (1.0 + eta * tau_pos)
    n_inject = rho * u_pos / (1.0 + u_pos)

    di = n_inject - beta_eff * i_pos * v_pos - delta * i_pos - mu_c * i_pos
    dv = beta_eff * i_pos * v_pos - gamma_eff * v_pos - mu_c * v_pos
    df = gamma_eff * v_pos - mu_c * f_pos
    ds = delta * i_pos - mu_c * s_pos
    dtau = phi * v_pos - psi * tau_pos
    du = -lambda_u * (1.0 + w * tau_pos) * u_pos + nu
    return [di, dv, df, ds, dtau, du]


def full_ivfs_ode(y, t, alpha, beta0, gamma0):
    """wrapper that plugs in the default params"""
    return _ivfs_rhs(y, t, alpha, beta0, gamma0,
                     KAPPA, ETA, PHI, PSI, RHO, LAMBDA_U, NU, MU_C, DELTA, W)


def run_scenarios(beta0: float, gamma0: float):
    t = np.linspace(0, 500, 5001)
    # start from DFE and give V a small push
    a_loss = DELTA + MU_C
    u_dfe = NU / LAMBDA_U
    i_dfe = RHO * u_dfe / ((1.0 + u_dfe) * a_loss)
    y0 = [i_dfe, 0.01, 0.0, 0.0, 0.0, u_dfe]
    scenarios = {
        'Engagement-First (alpha=0.9)': 0.9,
        'Moderate (alpha=0.5)': 0.5,
        'Health-First (alpha=0.2)': 0.2,
    }

    out = {}
    for label, alpha in scenarios.items():
        sol = odeint(full_ivfs_ode, y0, t, args=(alpha, beta0, gamma0), mxstep=20000)
        v_eq = float(sol[-1, 1])
        tau_eq = float(sol[-1, 4])
        u_eq = float(sol[-1, 5])
        out[label] = {
            'alpha': alpha,
            'solution': sol,
            'V_star': v_eq,
            'tau_star': tau_eq,
            'U_star': u_eq,
            'engagement': float(alpha * v_eq * u_eq),
        }
    return t, out


def run_continuation(beta0: float, gamma0: float):
    a_loss = DELTA + MU_C
    u_dfe = NU / LAMBDA_U
    i_dfe = RHO * u_dfe / ((1.0 + u_dfe) * a_loss)
    y0 = [i_dfe, 0.01, 0.0, 0.0, 0.0, u_dfe]
    t_grid = np.linspace(0, 2000, 10001)
    alphas = np.linspace(0.05, 1.0, 60)

    v_star_list = []
    tau_star = []
    u_star_list = []
    for alpha in alphas:
        sol = odeint(full_ivfs_ode, y0, t_grid, args=(float(alpha), beta0, gamma0), mxstep=20000)
        v_star_list.append(max(0.0, float(sol[-1, 1])))
        tau_star.append(max(0.0, float(sol[-1, 4])))
        u_star_list.append(float(sol[-1, 5]))

    n_star = RHO * u_dfe / (1.0 + u_dfe)
    alpha_r0 = float((a_loss * (gamma0 + MU_C)) / (beta0 * n_star))

    return alphas, np.array(tau_star), np.array(v_star_list), np.array(u_star_list), alpha_r0


def make_calibration_figure(empirical_norm, fitted_norm, window_counts,
                            re_window, reply_proxy,
                            fit_window_hours,
                            weighted_loss, r2_fit, r2_full, r2_spike, tail_bias):
    """2x2 calibration diagnostics -- compare what was fit vs what is only extrapolated"""
    plt.style.use('ggplot')
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    t_data = np.arange(len(empirical_norm))
    fit_end = min(int(fit_window_hours), len(empirical_norm))
    zoom_end = min(DISPLAY_WINDOW_HOURS, len(empirical_norm))
    spike_end = min(SPIKE_WINDOW_HOURS, len(empirical_norm))
    tail_start = min(TAIL_START_HOURS, len(empirical_norm))
    empirical_smooth = moving_average(empirical_norm, SMOOTH_WINDOW)
    sigma_band = 2.0 * np.sqrt(window_counts) / (np.max(window_counts) + 1e-12)
    lower_band = np.clip(empirical_norm - sigma_band, 0.0, None)
    upper_band = empirical_norm + sigma_band

    zoom_empirical = empirical_norm[:zoom_end]
    zoom_fitted = fitted_norm[:zoom_end]
    zoom_smooth = empirical_smooth[:zoom_end]
    zoom_ss_tot = float(np.sum((zoom_empirical - np.mean(zoom_empirical)) ** 2))
    r2_zoom = 1.0 - float(np.sum((zoom_fitted - zoom_empirical) ** 2)) / zoom_ss_tot if zoom_ss_tot > 0 else 0.0
    smooth_ss_tot = float(np.sum((zoom_smooth - np.mean(zoom_smooth)) ** 2))
    if smooth_ss_tot > 0:
        r2_vs_smooth = 1.0 - float(np.sum((zoom_fitted - zoom_smooth) ** 2)) / smooth_ss_tot
    else:
        r2_vs_smooth = 0.0

    # (a) zoom on the spike and shoulder region where alignment matters most visually
    axs[0, 0].fill_between(t_data, lower_band, upper_band, color='#B0BEC5', alpha=0.25,
                           label='Approx. ±2σ count noise')
    axs[0, 0].scatter(t_data, empirical_norm, s=14, alpha=0.35, color='#616161',
                      label='Raw hourly RT')
    axs[0, 0].plot(t_data, empirical_smooth, color='#37474F', linewidth=2.0,
                   label=f'Empirical {SMOOTH_WINDOW}h mean')
    axs[0, 0].plot(t_data, fitted_norm, color='#C62828', linewidth=2.3, label='Fitted IVF')
    if fit_end < len(empirical_norm):
        axs[0, 0].axvline(fit_end, color='gray', linestyle='--', linewidth=1.0, alpha=0.8,
                          label=f'Fit cutoff (h {fit_end})')
    axs[0, 0].set_xlim(0, zoom_end)
    axs[0, 0].set_ylim(bottom=0)
    axs[0, 0].set_title(f'(a) Spike Region Zoom (0\u2013{zoom_end - 1} h)')
    axs[0, 0].set_xlabel('Hour index')
    axs[0, 0].set_ylabel('Normalized viral volume')
    axs[0, 0].annotate(
        f'Weighted loss = {weighted_loss:.4f}\n'
        f'R\u00B2 (zoom 0\u2013{zoom_end - 1}h) = {r2_zoom:.4f}\n'
        f'R\u00B2 vs {SMOOTH_WINDOW}h mean (0\u2013{zoom_end - 1}h) = {r2_vs_smooth:.4f}',
        xy=(0.41, 0.72), xycoords='axes fraction', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    axs[0, 0].legend(fontsize=8)

    # (b) full window -- now a proper calibration fit rather than a tail extrapolation
    axs[0, 1].scatter(t_data, empirical_norm, s=14, alpha=0.35, color='#616161',
                      label='Raw hourly RT')
    axs[0, 1].plot(t_data, empirical_smooth, color='#37474F', linewidth=2.0,
                   label=f'Empirical {SMOOTH_WINDOW}h mean')
    axs[0, 1].plot(t_data, fitted_norm, color='#C62828', linewidth=2.1, label='Fitted IVF')
    axs[0, 1].axvline(spike_end, color='gray', linestyle=':', linewidth=1.0, alpha=0.7,
                      label=f'Spike cutoff (h {spike_end})')
    axs[0, 1].axvline(tail_start, color='gray', linestyle='--', linewidth=1.0, alpha=0.6,
                      label=f'Tail window start (h {tail_start})')
    axs[0, 1].set_title('(b) Full 100-Hour Window Calibration')
    axs[0, 1].set_xlabel('Hour index')
    axs[0, 1].set_ylabel('Normalized viral volume')
    axs[0, 1].annotate(
        f'R\u00B2 (spike 0\u2013{spike_end - 1}h) = {r2_spike:.4f}\n'
        f'R\u00B2 (full 0\u2013{len(empirical_norm) - 1}h) = {r2_full:.4f}\n'
        f'Tail mean bias (h {tail_start}\u2013{len(empirical_norm) - 1}) = {tail_bias:+.4f}',
        xy=(0.42, 0.70), xycoords='axes fraction', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    axs[0, 1].legend(fontsize=8)

    # (c) RE reply-pressure proxy from the same dataset -- useful context, not a toxicity calibration
    if re_window is not None and reply_proxy is not None:
        t_re = np.arange(len(re_window))
        re_norm = re_window / (np.max(re_window) + 1e-12)
        proxy_norm = reply_proxy / (np.max(reply_proxy) + 1e-12)
        rt_reference = empirical_smooth / (np.max(empirical_smooth) + 1e-12)
        re_corr = float(np.corrcoef(re_window[:fit_end], window_counts[:fit_end])[0, 1])

        axs[1, 0].bar(t_re, re_norm, alpha=0.35, color='#7B1FA2',
                      label='RE events (normalized)')
        axs[1, 0].plot(t_re, proxy_norm, color='#4A148C', linewidth=2.0,
                       label='Reply proxy (smoothed RE/RT)')
        axs[1, 0].plot(t_data, rt_reference, color='#455A64', linewidth=1.8, linestyle='--',
                       label='RT reference (smoothed)')
        axs[1, 0].set_xlim(0, zoom_end)
        axs[1, 0].set_ylim(bottom=0)
        axs[1, 0].set_title('(c) Reply-Pressure Proxy (Higgs RE Events)')
        axs[1, 0].set_xlabel('Hour index')
        axs[1, 0].set_ylabel('Normalized reply activity')
        re_total = int(np.sum(re_window))
        axs[1, 0].annotate(
            f'RE events in window: {re_total:,}\n'
            f'corr(RE, RT) over fit window = {re_corr:.4f}',
            xy=(0.04, 0.84), xycoords='axes fraction', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        axs[1, 0].legend(fontsize=8, loc='lower right')
    else:
        axs[1, 0].text(0.5, 0.5, 'No RE data available', transform=axs[1, 0].transAxes,
                       ha='center', va='center', fontsize=12, color='gray')
        axs[1, 0].set_title('(c) Reply-Pressure Proxy')

    # (d) residuals by hour -- where the simple one-spike model systematically misses
    residuals = fitted_norm - empirical_norm
    bar_colors = ['#C62828' if r > 0 else '#1565C0' for r in residuals]
    axs[1, 1].bar(t_data, residuals, color=bar_colors, alpha=0.6)
    axs[1, 1].plot(t_data, moving_average(residuals, SMOOTH_WINDOW), color='#263238',
                   linewidth=1.8, label=f'{SMOOTH_WINDOW}h residual mean')
    axs[1, 1].axhline(0, color='black', linewidth=0.8)
    axs[1, 1].axvline(spike_end, color='gray', linestyle=':', linewidth=1.0, alpha=0.7,
                      label=f'Spike cutoff (h {spike_end})')
    axs[1, 1].axvline(tail_start, color='gray', linestyle='--', linewidth=1.0, alpha=0.6,
                      label=f'Tail window start (h {tail_start})')
    axs[1, 1].set_title('(d) Residuals (Model \u2212 Raw Data)')
    axs[1, 1].set_xlabel('Hour index')
    axs[1, 1].set_ylabel('Residual')
    axs[1, 1].legend(fontsize=8)

    fig.suptitle('APM348 IVFS Calibration Diagnostics', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout(pad=2)
    fig.savefig(DIAG_FIGURE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)


def make_results_figure(t_scenario, scenario_results,
                        alphas, tau_star, v_star_cont, u_star_cont, alpha_r0):
    """2x3 policy results -- no calibration panel, purely scenario output"""
    plt.style.use('ggplot')
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    colors = {
        'Engagement-First (alpha=0.9)': '#C62828',
        'Moderate (alpha=0.5)': '#1565C0',
        'Health-First (alpha=0.2)': '#2E7D32',
    }

    # (a) V(t) scenarios
    for label, data in scenario_results.items():
        axs[0, 0].plot(t_scenario, data['solution'][:, 1], linewidth=2.0,
                        color=colors[label], label=label)
    axs[0, 0].set_title('(a) Viral Volume V(t) by Policy')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Viral volume V')
    axs[0, 0].set_xlim(0, SCENARIO_DISPLAY_HOURS)
    axs[0, 0].set_ylim(bottom=0)
    axs[0, 0].legend(fontsize=8)

    # (b) tau(t) scenarios
    for label, data in scenario_results.items():
        axs[0, 1].plot(t_scenario, data['solution'][:, 4], linewidth=2.0,
                        color=colors[label], label=label)
    axs[0, 1].set_title('(b) Toxicity \u03C4(t) by Policy')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Aggregate toxicity \u03C4')
    axs[0, 1].set_xlim(0, SCENARIO_DISPLAY_HOURS)
    axs[0, 1].set_ylim(bottom=0)
    axs[0, 1].legend(fontsize=8)

    # (c) U(t) scenarios
    for label, data in scenario_results.items():
        axs[0, 2].plot(t_scenario, data['solution'][:, 5], linewidth=2.0,
                        color=colors[label], label=label)
    axs[0, 2].set_title('(c) Active Users U(t) by Policy')
    axs[0, 2].set_xlabel('Time')
    axs[0, 2].set_ylabel('Active users U')
    axs[0, 2].set_xlim(0, SCENARIO_DISPLAY_HOURS)
    axs[0, 2].legend(fontsize=8)

    # (d) tau*(alpha) continuation
    axs[1, 0].plot(alphas, tau_star, color='#6A1B9A', linewidth=2.2)
    axs[1, 0].axvline(alpha_r0, color='black', linestyle='--', linewidth=1.5,
                       label=f'R\u2080=1 at \u03B1\u2248{alpha_r0:.2f}')
    axs[1, 0].set_title('(d) Equilibrium Toxicity \u03C4* vs \u03B1')
    axs[1, 0].set_xlabel('Amplification \u03B1')
    axs[1, 0].set_ylabel('\u03C4*')
    axs[1, 0].legend(fontsize=8)

    # (e) U*(alpha) continuation
    axs[1, 1].plot(alphas, u_star_cont, color='#00695C', linewidth=2.2)
    axs[1, 1].axvline(alpha_r0, color='black', linestyle='--', linewidth=1.5,
                       label=f'R\u2080=1 at \u03B1\u2248{alpha_r0:.2f}')
    axs[1, 1].set_title('(e) Equilibrium Users U* vs \u03B1')
    axs[1, 1].set_xlabel('Amplification \u03B1')
    axs[1, 1].set_ylabel('U*')
    axs[1, 1].legend(fontsize=8)

    # (f) E* = alpha * V* * U*
    e_star = alphas * v_star_cont * u_star_cont
    axs[1, 2].plot(alphas, e_star, color='#E65100', linewidth=2.2)
    axs[1, 2].axvline(alpha_r0, color='black', linestyle='--', linewidth=1.5,
                       label=f'R\u2080=1 at \u03B1\u2248{alpha_r0:.2f}')
    peak_idx = int(np.argmax(e_star))
    axs[1, 2].axvline(alphas[peak_idx], color='#E65100', linestyle=':', linewidth=1.5,
                       label=f'Peak E* at \u03B1\u2248{alphas[peak_idx]:.2f}')
    axs[1, 2].set_title('(f) Engagement E* = \u03B1\u00B7V*\u00B7U* vs \u03B1')
    axs[1, 2].set_xlabel('Amplification \u03B1')
    axs[1, 2].set_ylabel('E*')
    axs[1, 2].legend(fontsize=8)

    fig.suptitle('APM348 IVFS Policy Scenarios', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout(pad=2)
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)


def run_sensitivity(beta0: float, gamma0: float, alpha: float = 0.5):
    """bump each param +/-50% and see how tau* and U* move"""
    a_loss = DELTA + MU_C
    u_dfe = NU / LAMBDA_U
    i_dfe = RHO * u_dfe / ((1.0 + u_dfe) * a_loss)
    y0 = [i_dfe, 0.01, 0.0, 0.0, 0.0, u_dfe]
    t_grid = np.linspace(0, 2000, 10001)

    def get_eq(ode_func):
        sol = odeint(ode_func, y0, t_grid, args=(alpha, beta0, gamma0), mxstep=20000)
        return float(sol[-1, 4]), float(sol[-1, 5])  # tau*, U*

    tau_base, u_base = get_eq(full_ivfs_ode)
    params_to_vary = {'kappa': KAPPA, 'eta': ETA, 'phi': PHI, 'psi': PSI, 'w': W}
    rows = []
    for name, base_val in params_to_vary.items():
        for factor, label in [(0.5, '-50%'), (1.5, '+50%')]:
            ov = {name: base_val * factor}
            def ode_mod(y, t, a, b, g, _ov=ov):
                return _ivfs_rhs(y, t, a, b, g,
                                 _ov.get('kappa', KAPPA), _ov.get('eta', ETA),
                                 _ov.get('phi', PHI), _ov.get('psi', PSI),
                                 RHO, LAMBDA_U, NU, MU_C, DELTA,
                                 _ov.get('w', W))
            tau_mod, u_mod = get_eq(ode_mod)
            dtau = tau_mod - tau_base
            du = u_mod - u_base
            frac_change = factor - 1.0  # -0.5 or +0.5
            tau_si = (dtau / tau_base) / frac_change if tau_base > 1e-12 else 0.0
            u_si = (du / u_base) / frac_change if abs(u_base) > 1e-12 else 0.0
            rows.append((name, label, base_val * factor, dtau, du, tau_si, u_si))
    return tau_base, u_base, rows


def run_phi_sensitivity(beta0: float,
                        gamma0: float,
                        phi_grid: np.ndarray,
                        scenario_alphas: dict[str, float] | None = None):
    """sweep PHI while holding PSI fixed to show how robust the policy ordering is"""
    if scenario_alphas is None:
        scenario_alphas = {
            'Engagement-First (alpha=0.9)': 0.9,
            'Moderate (alpha=0.5)': 0.5,
            'Health-First (alpha=0.2)': 0.2,
        }

    a_loss = DELTA + MU_C
    u_dfe = NU / LAMBDA_U
    i_dfe = RHO * u_dfe / ((1.0 + u_dfe) * a_loss)
    y0 = [i_dfe, 0.01, 0.0, 0.0, 0.0, u_dfe]
    t_grid = np.linspace(0, 2000, 10001)

    out: dict[str, dict[str, np.ndarray | float]] = {}
    for label, alpha in scenario_alphas.items():
        tau_vals: list[float] = []
        u_vals: list[float] = []
        v_vals: list[float] = []
        e_vals: list[float] = []
        for phi_val in phi_grid:
            def ode_phi(y, t, a, b, g, _phi=float(phi_val)):
                return _ivfs_rhs(y, t, a, b, g,
                                 KAPPA, ETA, _phi, PSI, RHO, LAMBDA_U, NU, MU_C, DELTA, W)

            sol = odeint(ode_phi, y0, t_grid, args=(alpha, beta0, gamma0), mxstep=20000)
            v_eq = max(0.0, float(sol[-1, 1]))
            tau_eq = max(0.0, float(sol[-1, 4]))
            u_eq = float(sol[-1, 5])
            tau_vals.append(tau_eq)
            u_vals.append(u_eq)
            v_vals.append(v_eq)
            e_vals.append(float(alpha * v_eq * u_eq))
        out[label] = {
            'alpha': alpha,
            'tau_star': np.array(tau_vals),
            'U_star': np.array(u_vals),
            'V_star': np.array(v_vals),
            'E_star': np.array(e_vals),
        }
    return out


def make_phi_sensitivity_figure(phi_grid: np.ndarray,
                                phi_results: dict[str, dict[str, np.ndarray | float]],
                                current_phi: float,
                                external_phi: float | None = None,
                                external_source: str | None = None) -> None:
    """save a small robustness figure for the toxicity coupling PHI"""
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))

    colors = {
        'Engagement-First (alpha=0.9)': '#C62828',
        'Moderate (alpha=0.5)': '#1565C0',
        'Health-First (alpha=0.2)': '#2E7D32',
    }

    for label, data in phi_results.items():
        axs[0].plot(phi_grid, data['tau_star'], linewidth=2.1, color=colors[label], label=label)
        axs[1].plot(phi_grid, data['U_star'], linewidth=2.1, color=colors[label], label=label)

    for ax in axs:
        ax.axvline(current_phi, color='black', linestyle='--', linewidth=1.2,
                   label=f'Current PHI={current_phi:.3f}')
        if external_phi is not None:
            ext_label = f'External PHI≈{external_phi:.3f}'
            if external_source:
                ext_label = f'{external_source} PHI≈{external_phi:.3f}'
            ax.axvline(external_phi, color='#6A1B9A', linestyle=':', linewidth=1.4, label=ext_label)
        ax.set_xlabel('Toxicity coupling PHI')

    axs[0].set_title('(a) Equilibrium Toxicity $\\tau^*$ vs PHI')
    axs[0].set_ylabel('$\\tau^*$')
    axs[0].set_ylim(bottom=0)
    axs[0].legend(fontsize=8)

    axs[1].set_title('(b) Equilibrium Users $U^*$ vs PHI')
    axs[1].set_ylabel('$U^*$')
    axs[1].legend(fontsize=8)

    fig.suptitle('APM348 PHI Sensitivity (PSI fixed)', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout(pad=2)
    fig.savefig(PHI_SENS_FIGURE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)


def find_W_for_interior_Emax(beta0: float, gamma0: float,
                              w_grid: np.ndarray | None = None):
    """scan W values to see if E* ever peaks at an interior alpha"""
    if w_grid is None:
        w_grid = np.linspace(5, 80, 50)
    a_loss = DELTA + MU_C
    u_dfe_base = NU / LAMBDA_U
    i_dfe = RHO * u_dfe_base / ((1.0 + u_dfe_base) * a_loss)
    y0 = [i_dfe, 0.01, 0.0, 0.0, 0.0, u_dfe_base]
    t_grid = np.linspace(0, 3000, 15001)
    alphas = np.linspace(0.1, 1.0, 50)

    for w_test in w_grid:
        e_vals = []
        for alpha in alphas:
            def ode_w(y, t, a, b, g, _w=w_test):
                return _ivfs_rhs(y, t, a, b, g, KAPPA, ETA, PHI, PSI,
                                 RHO, LAMBDA_U, NU, MU_C, DELTA, _w)
            sol = odeint(ode_w, y0, t_grid, args=(float(alpha), beta0, gamma0), mxstep=20000)
            v_eq = max(0.0, float(sol[-1, 1]))
            u_eq = float(sol[-1, 5])
            e_vals.append(float(alpha) * v_eq * u_eq)
        peak_idx = int(np.argmax(e_vals))
        if peak_idx < len(alphas) - 2:
            return float(w_test), float(alphas[peak_idx])
    return None, None


def bootstrap_calibration(empirical_norm: np.ndarray,
                           window_counts: np.ndarray,
                           B: int = 500) -> dict:
    """poisson bootstrap -- resample counts and refit B times to get CIs"""
    from scipy.stats import poisson

    beta0, gamma0, lambda0, lambda_decay, v0_fit, _loss, fitted_norm = fit_basic_ivf(empirical_norm, window_counts)
    fitted_counts = fitted_norm * np.max(window_counts)

    # hold the tail-shape nuisance terms fixed and only refit beta/gamma in the bootstrap
    def fast_refit(syn_norm, syn_counts):
        w = 1.0 / (np.sqrt(syn_counts) + 1.0)
        def loss_bg(params):
            beta = float(np.clip(abs(params[0]), IVF_PARAM_BOUNDS[0][0], IVF_PARAM_BOUNDS[0][1]))
            gamma = float(np.clip(abs(params[1]), IVF_PARAM_BOUNDS[1][0], IVF_PARAM_BOUNDS[1][1]))
            v_sim = simulate_basic_ivf(beta, gamma, lambda0, lambda_decay, v0_fit, n_steps=len(syn_norm))
            return float(np.sum(w * (v_sim - syn_norm) ** 2))
        beta_gamma_bounds = [IVF_PARAM_BOUNDS[0], IVF_PARAM_BOUNDS[1]]
        _fun, x_star = _powell_bounded(loss_bg, np.array([beta0, gamma0]), beta_gamma_bounds, maxiter=2000)
        return float(x_star[0]), float(x_star[1])

    estimates: dict[str, list[float]] = {'beta0': [], 'gamma0': [], 'alpha_r0': []}
    a_loss = DELTA + MU_C
    u_dfe = NU / LAMBDA_U
    n_star = RHO * u_dfe / (1.0 + u_dfe)

    for i in range(B):
        if (i + 1) % 50 == 0 or i == 0:
            print(f'  bootstrap {i + 1}/{B}...', end='\r')
        synthetic_counts = poisson.rvs(np.maximum(fitted_counts, 0.5)).astype(float)
        if np.max(synthetic_counts) == 0:
            continue
        syn_norm = synthetic_counts / np.max(synthetic_counts)
        b, g = fast_refit(syn_norm, synthetic_counts)
        estimates['beta0'].append(float(b))
        estimates['gamma0'].append(float(g))
        estimates['alpha_r0'].append(float((a_loss * (g + MU_C)) / (b * n_star)))
    print()  # clear \r

    ci: dict[str, tuple[float, float]] = {}
    for key, vals in estimates.items():
        arr = np.array(vals)
        ci[key] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return ci


def profile_likelihood(empirical_norm: np.ndarray,
                        window_counts: np.ndarray,
                        best_params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """sweep beta0 and gamma0 one at a time with warm starts to avoid optimizer jumps"""
    beta0_best, gamma0_best, _lambda0_best, _lambda_decay_best, _v0_best = best_params
    beta_grid = np.linspace(max(0.55, 0.70 * beta0_best), min(1.45, 1.35 * beta0_best), 35)
    gamma_grid = np.linspace(max(0.10, 0.60 * gamma0_best), min(0.50, 1.40 * gamma0_best), 35)
    weights = 1.0 / (np.sqrt(window_counts) + 1.0)
    all_bounds = list(IVF_PARAM_BOUNDS)

    def total_loss(params_full: np.ndarray) -> float:
        params_clipped = _clip_to_bounds(np.abs(params_full))
        v_sim = simulate_basic_ivf(
            params_clipped[0],
            params_clipped[1],
            params_clipped[2],
            params_clipped[3],
            params_clipped[4],
            n_steps=len(empirical_norm),
        )
        return float(np.sum(weights * (v_sim - empirical_norm) ** 2))

    def sweep_profile(grid: np.ndarray, fixed_idx: int) -> np.ndarray:
        profile = np.zeros(len(grid), dtype=float)
        free_idx = [i for i in range(len(best_params)) if i != fixed_idx]
        free_bounds = [all_bounds[i] for i in free_idx]
        center = int(np.argmin(np.abs(grid - best_params[fixed_idx])))

        def optimize_at(fixed_value: float, free_start: np.ndarray) -> tuple[np.ndarray, float]:
            def loss_free(free_params):
                params_full = best_params.copy()
                params_full[fixed_idx] = fixed_value
                params_full[free_idx] = free_params
                return total_loss(params_full)
            fun, free_star = _powell_bounded(loss_free, free_start, free_bounds, maxiter=2500)
            params_star = best_params.copy()
            params_star[fixed_idx] = fixed_value
            params_star[free_idx] = free_star
            return params_star, fun

        params_center, fun_center = optimize_at(float(grid[center]), best_params[free_idx])
        profile[center] = fun_center

        params_prev = params_center.copy()
        for idx in range(center + 1, len(grid)):
            params_prev, fun = optimize_at(float(grid[idx]), params_prev[free_idx])
            profile[idx] = fun

        params_prev = params_center.copy()
        for idx in range(center - 1, -1, -1):
            params_prev, fun = optimize_at(float(grid[idx]), params_prev[free_idx])
            profile[idx] = fun
        return profile

    print('  profiling beta0...', end='\r')
    profile_beta = sweep_profile(beta_grid, fixed_idx=0)
    print('  profiling gamma0...', end='\r')
    profile_gamma = sweep_profile(gamma_grid, fixed_idx=1)
    print()

    return beta_grid, profile_beta, gamma_grid, profile_gamma


def plot_profile(beta_grid, profile_beta, gamma_grid, profile_gamma,
                 beta0_best, gamma0_best) -> None:
    """save the profile likelihood plots"""
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    beta_excess = profile_beta - float(np.min(profile_beta))
    gamma_excess = profile_gamma - float(np.min(profile_gamma))

    ax1.plot(beta_grid, beta_excess, 'o-', markersize=3, color='#C62828')
    ax1.axvline(beta0_best, color='black', linestyle='--', linewidth=1.2,
                label=f'\u03B2\u2080={beta0_best:.3f}')
    ax1.set_xlabel('\u03B2\u2080 (fixed)')
    ax1.set_ylabel('Excess profile weighted loss')
    ax1.set_title('Local Profile Likelihood: \u03B2\u2080')
    ax1.legend(fontsize=9)

    ax2.plot(gamma_grid, gamma_excess, 'o-', markersize=3, color='#1565C0')
    ax2.axvline(gamma0_best, color='black', linestyle='--', linewidth=1.2,
                label=f'\u03B3\u2080={gamma0_best:.3f}')
    ax2.set_xlabel('\u03B3\u2080 (fixed)')
    ax2.set_ylabel('Excess profile weighted loss')
    ax2.set_title('Local Profile Likelihood: \u03B3\u2080')
    ax2.legend(fontsize=9)

    fig.tight_layout()
    out_path = ASSETS_DIR / 'profile_likelihood.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved profile likelihood figure: {out_path}')


def main() -> None:
    print('=' * 68)
    print('APM348 IVFS -- running calibration + scenarios + figures')
    print('=' * 68)

    ensure_dataset()
    rt_timestamps, re_timestamps, total_rows = parse_activity_file(HIGGS_TXT)
    cal = build_hourly_curve(rt_timestamps, re_timestamps)
    window_counts = cal['rt_window']
    re_window = cal.get('re_window')
    reply_proxy = cal.get('reply_proxy')
    peak_hour = cal['peak_hour']
    window_start = cal['window_start']
    hourly_counts = cal['hourly_counts']

    empirical_norm = window_counts / np.max(window_counts)
    fit_window_hours = min(FIT_WINDOW_HOURS, len(empirical_norm))
    fit_target = empirical_norm[:fit_window_hours]
    fit_counts = window_counts[:fit_window_hours]
    beta0, gamma0, lambda0, lambda_decay, v0_fit, weighted_loss, fitted_norm = fit_basic_ivf(
        fit_target,
        fit_counts,
        n_steps=len(empirical_norm),
    )

    # compute diagnostic metrics -- fit-window R², spike R², full-window R², tail mean bias
    fit_sse = float(np.sum((fitted_norm[:fit_window_hours] - empirical_norm[:fit_window_hours]) ** 2))
    fit_ss_tot = float(np.sum((empirical_norm[:fit_window_hours] - np.mean(empirical_norm[:fit_window_hours])) ** 2))
    r2_fit = 1.0 - fit_sse / fit_ss_tot if fit_ss_tot > 0 else 0.0

    spike_end = min(SPIKE_WINDOW_HOURS, len(empirical_norm))
    spike_sse = float(np.sum((fitted_norm[:spike_end] - empirical_norm[:spike_end]) ** 2))
    spike_ss_tot = float(np.sum((empirical_norm[:spike_end] - np.mean(empirical_norm[:spike_end])) ** 2))
    r2_spike = 1.0 - spike_sse / spike_ss_tot if spike_ss_tot > 0 else 0.0

    full_sse = float(np.sum((fitted_norm - empirical_norm) ** 2))
    full_ss_tot = float(np.sum((empirical_norm - np.mean(empirical_norm)) ** 2))
    r2_full = 1.0 - full_sse / full_ss_tot if full_ss_tot > 0 else 0.0

    tail_start = min(TAIL_START_HOURS, len(empirical_norm))
    tail_bias = float(np.mean(empirical_norm[tail_start:] - fitted_norm[tail_start:])) if tail_start < len(empirical_norm) else 0.0

    t_scenario, scenario_results = run_scenarios(beta0, gamma0)
    alphas, tau_star, v_star_cont, u_star_cont, alpha_r0 = run_continuation(beta0, gamma0)

    external_tau_ref = None
    external_source = None
    try:
        from toxicity_calibration import get_external_tau_reference
        ext_ref = get_external_tau_reference()
        if ext_ref is not None:
            external_tau_ref = float(ext_ref['tau_reference'])
            external_source = str(ext_ref['dataset'])
    except Exception:
        ext_ref = None

    moderate_v_star = max(0.0, float(scenario_results['Moderate (alpha=0.5)']['V_star']))
    external_phi = None
    if external_tau_ref is not None and moderate_v_star > 1e-12:
        external_phi = float(external_tau_ref * PSI / moderate_v_star)
    phi_candidates = [PHI]
    if external_phi is not None and np.isfinite(external_phi):
        phi_candidates.append(external_phi)
    phi_min = max(0.02, 0.5 * min(phi_candidates))
    phi_max = min(0.25, 1.2 * max(phi_candidates))
    phi_grid = np.linspace(phi_min, phi_max, 30)
    phi_results = run_phi_sensitivity(beta0, gamma0, phi_grid)
    make_phi_sensitivity_figure(phi_grid, phi_results, PHI, external_phi, external_source)

    # calibration diagnostics figure (2x2)
    make_calibration_figure(empirical_norm, fitted_norm, window_counts,
                            re_window, reply_proxy,
                            fit_window_hours,
                            weighted_loss, r2_fit, r2_full, r2_spike, tail_bias)

    # policy results figure (2x3, no calibration panel)
    make_results_figure(t_scenario, scenario_results,
                        alphas, tau_star, v_star_cont, u_star_cont, alpha_r0)

    # need DFE values for R0
    a_loss = DELTA + MU_C
    u_dfe = NU / LAMBDA_U
    i_dfe = RHO * u_dfe / ((1.0 + u_dfe) * a_loss)

    print(f'Total activity rows: {total_rows:,}')
    print(f'RT events: {len(rt_timestamps):,}')
    print(f'RE events: {len(re_timestamps):,}')
    print(f'Peak RT hour: {peak_hour} (count={int(hourly_counts[peak_hour]):,})')
    print(f'Calibration window: hours {window_start}..{window_start + len(window_counts) - 1}')
    print(f'Calibration fit window used: 0..{fit_window_hours - 1}')
    print()
    print(f'Fitted beta0: {beta0:.6f} per hour')
    print(f'Fitted gamma0: {gamma0:.6f} per hour')
    print(f'Fitted lambda0: {lambda0:.6f}')
    print(f'Fitted lambda decay: {lambda_decay:.6f}')
    print(f'Fitted V0: {v0_fit:.6f}')
    print(f'Weighted loss (Poisson weights): {weighted_loss:.6f}')
    print(f'R\u00B2 fit window (0\u2013{fit_window_hours - 1}h): {r2_fit:.4f}')
    print(f'R\u00B2 full window (0\u2013{len(empirical_norm)-1}h): {r2_full:.4f}')
    print(f'R\u00B2 spike window (0\u2013{spike_end - 1}h): {r2_spike:.4f}')
    print(f'Tail mean bias (h {tail_start}\u2013{len(empirical_norm)-1}): {tail_bias:+.4f}')
    if re_window is not None:
        re_rt_corr = float(np.corrcoef(cal['re_window'], window_counts)[0, 1])
        print(f'RE-RT correlation in window: {re_rt_corr:.4f}')
    print()
    print('Scenario long-run values (integrated to t=500):')
    for label, data in scenario_results.items():
        r0_val = data['alpha'] * beta0 * i_dfe / (gamma0 + MU_C)
        print(
            f"  {label}: R0={r0_val:.2f}, V*={max(0, data['V_star']):.6f}, "
            f"tau*={max(0, data['tau_star']):.6f}, U*={data['U_star']:.4f}, "
            f"E={max(0, data['engagement']):.6f}"
        )
    print()
    print(f'R0=1 threshold alpha: {alpha_r0:.4f}')
    print(f'Saved calibration diagnostics: {DIAG_FIGURE_PATH}')
    print(f'Saved policy results: {FIGURE_PATH}')
    print(f'Saved PHI sensitivity figure: {PHI_SENS_FIGURE_PATH}')

    print()
    print(f'PHI robustness check (PSI fixed at {PSI:.3f}):')
    print(f'  Current model PHI: {PHI:.4f}')
    if external_phi is not None and np.isfinite(external_phi):
        source_label = external_source if external_source is not None else 'external reference'
        print(f'  {source_label} implies PHI≈{external_phi:.4f}')
    print(f"  Health-First tau* range over PHI grid: [{np.min(phi_results['Health-First (alpha=0.2)']['tau_star']):.4f}, {np.max(phi_results['Health-First (alpha=0.2)']['tau_star']):.4f}]")
    print(f"  Moderate tau* range over PHI grid: [{np.min(phi_results['Moderate (alpha=0.5)']['tau_star']):.4f}, {np.max(phi_results['Moderate (alpha=0.5)']['tau_star']):.4f}]")
    print(f"  Engagement-First tau* range over PHI grid: [{np.min(phi_results['Engagement-First (alpha=0.9)']['tau_star']):.4f}, {np.max(phi_results['Engagement-First (alpha=0.9)']['tau_star']):.4f}]")
    tau_order_ok = bool(np.all(phi_results['Engagement-First (alpha=0.9)']['tau_star'] >= phi_results['Moderate (alpha=0.5)']['tau_star'] - 1e-10)
                        and np.all(phi_results['Moderate (alpha=0.5)']['tau_star'] >= phi_results['Health-First (alpha=0.2)']['tau_star'] - 1e-10))
    u_order_ok = bool(np.all(phi_results['Engagement-First (alpha=0.9)']['U_star'] <= phi_results['Moderate (alpha=0.5)']['U_star'] + 1e-10)
                      and np.all(phi_results['Moderate (alpha=0.5)']['U_star'] <= phi_results['Health-First (alpha=0.2)']['U_star'] + 1e-10))
    print(f'  tau* ordering preserved across PHI grid: {tau_order_ok}')
    print(f'  U* ordering preserved across PHI grid: {u_order_ok}')

    # sensitivity analysis with normalized indices
    print()
    print('Sensitivity (alpha=0.5, +/-50% each param):')
    print(f'{"param":<8} {"change":<8} {"value":<8} {"d_tau*":<10} {"d_U*":<10} {"S_tau":<8} {"S_U":<8}')
    tau_base, u_base, sens_rows = run_sensitivity(beta0, gamma0)
    print(f'{"(base)":<8} {"":<8} {"":<8} {tau_base:<10.4f} {u_base:<10.4f}')
    for name, label, val, dtau, du, s_tau, s_u in sens_rows:
        print(f'{name:<8} {label:<8} {val:<8.3f} {dtau:<+10.4f} {du:<+10.4f} {s_tau:<+8.3f} {s_u:<+8.3f}')

    # check if there's a W where E* peaks at some interior alpha
    print()
    w_opt, alpha_peak = find_W_for_interior_Emax(beta0, gamma0)
    if w_opt is not None:
        print(f'Interior E* peak found: W={w_opt:.1f} gives peak at alpha={alpha_peak:.2f}')
    else:
        print('No interior E* peak found for W <= 80; E* peaks at alpha=1.0.')
        print('This is expected -- the platform that maximizes engagement also gets the worst toxicity/retention.')

    # profile likelihood to check identifiability
    print()
    print('Computing profile likelihood...')
    bg, pb, gg, pg = profile_likelihood(
        fit_target,
        fit_counts,
        np.array([beta0, gamma0, lambda0, lambda_decay, v0_fit], dtype=float),
    )
    plot_profile(bg, pb, gg, pg, beta0, gamma0)

    # bootstrap for confidence intervals
    print()
    print('Running parametric bootstrap (B=500)...')
    ci = bootstrap_calibration(fit_target, fit_counts, B=500)
    print(f"beta0 95% CI: [{ci['beta0'][0]:.4f}, {ci['beta0'][1]:.4f}]")
    print(f"gamma0 95% CI: [{ci['gamma0'][0]:.4f}, {ci['gamma0'][1]:.4f}]")
    print(f"R0=1 threshold alpha 95% CI: [{ci['alpha_r0'][0]:.4f}, {ci['alpha_r0'][1]:.4f}]")


if __name__ == '__main__':
    main()
