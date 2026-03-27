from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

from ivfs_config import (FIT_WINDOW_HOURS, HIGGS_GZ, HIGGS_TXT, HIGGS_URL, IVF_PARAM_BOUNDS, PHI, PSI,
                         SMOOTH_WINDOW, TAU_DECAY_MIN_POINTS, TAU_PARAM_BOUNDS, TAU_PROXY_WEIGHTS,
                         ensure_layout)


def ensure_dataset() -> None:
    """Download and unzip the Higgs data if we don't have it yet"""
    ensure_layout()

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


def parse_activity_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Read the activity file and grab RT + RE + MT timestamps"""
    rt_timestamps: list[int] = []
    re_timestamps: list[int] = []
    mt_timestamps: list[int] = []
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
                elif parts[3] == 'MT':
                    mt_timestamps.append(int(parts[2]))

    return (
        np.array(rt_timestamps, dtype=np.int64),
        np.array(re_timestamps, dtype=np.int64),
        np.array(mt_timestamps, dtype=np.int64),
        total_rows,
    )


def build_hourly_curve(rt_timestamps: np.ndarray,
                       re_timestamps: np.ndarray | None = None,
                       mt_timestamps: np.ndarray | None = None) -> dict:
    """Bin RT/RE/MT into hours and build the calibration + pressure-proxy windows"""
    t0 = int(rt_timestamps.min())
    hours = (rt_timestamps - t0) / 3600.0
    max_hour = int(np.ceil(hours.max()))
    hourly_counts, _ = np.histogram(hours, bins=np.arange(0, max_hour + 1, 1))

    peak_hour = int(np.argmax(hourly_counts))
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

    re_window = None
    mt_window = None
    if re_timestamps is not None and len(re_timestamps) > 0:
        hours_re = (re_timestamps - t0) / 3600.0
        re_hourly, _ = np.histogram(hours_re, bins=np.arange(0, max_hour + 1, 1))
        re_window = re_hourly[start:end].astype(float)

    if mt_timestamps is not None and len(mt_timestamps) > 0:
        hours_mt = (mt_timestamps - t0) / 3600.0
        mt_hourly, _ = np.histogram(hours_mt, bins=np.arange(0, max_hour + 1, 1))
        mt_window = mt_hourly[start:end].astype(float)

    if re_window is not None or mt_window is not None:
        kernel = np.ones(3) / 3.0
        re_norm = (re_window / (np.max(re_window) + 1e-12)) if re_window is not None else np.zeros_like(window_counts)
        mt_norm = (mt_window / (np.max(mt_window) + 1e-12)) if mt_window is not None else np.zeros_like(window_counts)
        re_proxy = np.convolve(re_norm, kernel, mode='same')
        mt_proxy = np.convolve(mt_norm, kernel, mode='same')
        raw_ratio = (re_window if re_window is not None else np.zeros_like(window_counts)) / (window_counts + 1.0)
        ratio_smoothed = np.convolve(raw_ratio, kernel, mode='same')
        ratio_norm = ratio_smoothed / (np.max(ratio_smoothed) + 1e-12)

        weights = TAU_PROXY_WEIGHTS.copy()
        if mt_window is None:
            weights['re'] += weights['mt']
            weights['mt'] = 0.0
        weight_total = weights['re'] + weights['ratio'] + weights['mt']
        tau_proxy = (
            weights['re'] * re_proxy
            + weights['ratio'] * ratio_norm
            + weights['mt'] * mt_proxy
        ) / max(weight_total, 1e-12)

        result['re_window'] = re_window
        result['mt_window'] = mt_window
        result['re_proxy'] = np.clip(re_proxy, 0.0, 1.0)
        result['mt_proxy'] = np.clip(mt_proxy, 0.0, 1.0)
        result['reply_ratio'] = np.clip(ratio_norm, 0.0, 1.0)
        result['tau_proxy'] = np.clip(tau_proxy, 0.0, 1.0)

    return result


def moving_average(series: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Simple centered moving average for plotting the trend vs raw hourly noise"""
    if window <= 1:
        return series.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(series, kernel, mode='same')


def _clip_to_bounds(params: np.ndarray, bounds=IVF_PARAM_BOUNDS) -> np.ndarray:
    clipped = np.array(params, dtype=float, copy=True)
    for i, (lo, hi) in enumerate(bounds):
        clipped[i] = float(np.clip(clipped[i], lo, hi))
    return clipped


def _clip_tau_params(params: np.ndarray, bounds=TAU_PARAM_BOUNDS) -> np.ndarray:
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
    """Simulate the baseline IVF fit with a decaying exogenous seed term"""
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
    """Fit IVF to the full Higgs window with a decaying exogenous tail term"""
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


def simulate_tau_from_v(v_series: np.ndarray,
                        phi: float,
                        psi: float,
                        tau0: float = 0.0) -> np.ndarray:
    """Propagate tau using the fitted V(t) curve as an exogenous input"""
    v_arr = np.asarray(v_series, dtype=float)
    tau = np.zeros_like(v_arr, dtype=float)
    if tau.size == 0:
        return tau

    tau[0] = max(float(tau0), 0.0)
    psi_safe = max(float(psi), 1e-8)
    decay = float(np.exp(-psi_safe))
    gain = float((1.0 - decay) * float(phi) / psi_safe)
    for idx in range(1, tau.size):
        tau[idx] = decay * tau[idx - 1] + gain * max(float(v_arr[idx - 1]), 0.0)
    return np.clip(tau, 0.0, None)


def tau_fit_metrics(tau_target: np.ndarray,
                    tau_model: np.ndarray,
                    weights: np.ndarray | None = None) -> tuple[float, float]:
    residual = np.asarray(tau_model, dtype=float) - np.asarray(tau_target, dtype=float)
    if weights is None:
        weights = np.ones_like(residual)
    weighted_loss = float(np.sum(weights * residual ** 2))
    ss_tot = float(np.sum((tau_target - np.mean(tau_target)) ** 2))
    r2 = 1.0 - float(np.sum(residual ** 2)) / ss_tot if ss_tot > 0 else 0.0
    return weighted_loss, r2


def estimate_psi_from_decay(tau_target: np.ndarray,
                            min_points: int = TAU_DECAY_MIN_POINTS) -> tuple[float, dict[str, float]]:
    """Estimate psi from the post-peak decay timescale of the proxy"""
    tau_arr = np.asarray(tau_target, dtype=float)
    if tau_arr.size == 0:
        return PSI, {'peak_index': 0.0, 'n_decay_points': 0.0, 'decay_r2': 0.0}

    peak_idx = int(np.argmax(tau_arr))
    peak_val = float(np.max(tau_arr))
    if peak_val <= 1e-12:
        return PSI, {'peak_index': float(peak_idx), 'n_decay_points': 0.0, 'decay_r2': 0.0}

    tail_idx = np.arange(peak_idx, tau_arr.size, dtype=float)
    tail_vals = tau_arr[peak_idx:]
    mask = tail_vals > max(0.05 * peak_val, 1e-6)
    if int(np.sum(mask)) < min_points:
        return PSI, {'peak_index': float(peak_idx), 'n_decay_points': float(np.sum(mask)), 'decay_r2': 0.0}

    x = tail_idx[mask] - tail_idx[mask][0]
    y = np.log(tail_vals[mask])
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - fitted) ** 2))
    decay_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    psi_est = float(np.clip(-slope, TAU_PARAM_BOUNDS[1][0], TAU_PARAM_BOUNDS[1][1]))
    return psi_est, {
        'peak_index': float(peak_idx),
        'n_decay_points': float(np.sum(mask)),
        'decay_r2': float(decay_r2),
    }


def fit_phi_for_fixed_psi(v_series: np.ndarray,
                          tau_target: np.ndarray,
                          psi: float,
                          weights: np.ndarray | None = None) -> tuple[float, np.ndarray, float, float]:
    """With psi fixed, fit phi by weighted least squares"""
    v_arr = np.asarray(v_series, dtype=float)
    tau_target_arr = np.asarray(tau_target, dtype=float)
    tau0 = float(max(tau_target_arr[0], 0.0)) if tau_target_arr.size else 0.0
    if weights is None:
        weights = np.ones_like(tau_target_arr)
    weight_arr = np.asarray(weights, dtype=float)

    tau_phi0 = simulate_tau_from_v(v_arr, 0.0, float(psi), tau0=tau0)
    tau_unit = simulate_tau_from_v(v_arr, 1.0, float(psi), tau0=0.0)
    rhs = tau_target_arr - tau_phi0
    denom = float(np.sum(weight_arr * tau_unit ** 2))
    if denom <= 1e-12:
        phi_fit = PHI
    else:
        phi_fit = float(np.sum(weight_arr * tau_unit * rhs) / denom)
    phi_fit = float(np.clip(phi_fit, TAU_PARAM_BOUNDS[0][0], TAU_PARAM_BOUNDS[0][1]))
    tau_fit = tau_phi0 + phi_fit * tau_unit
    weighted_loss, r2 = tau_fit_metrics(tau_target_arr, tau_fit, weight_arr)
    return phi_fit, tau_fit, weighted_loss, r2


def fit_tau_proxy_unconstrained(v_series: np.ndarray,
                                tau_target: np.ndarray,
                                weights: np.ndarray) -> tuple[float, float, float, float, np.ndarray]:
    """Fit phi and psi directly to the chosen same-dataset proxy"""
    v_arr = np.asarray(v_series, dtype=float)
    tau_target_arr = np.asarray(tau_target, dtype=float)
    tau0 = float(max(tau_target_arr[0], 0.0)) if tau_target_arr.size else 0.0

    def loss(params: np.ndarray) -> float:
        phi, psi = _clip_tau_params(np.abs(params))
        tau_model = simulate_tau_from_v(v_arr, float(phi), float(psi), tau0=tau0)
        weighted_loss, _r2 = tau_fit_metrics(tau_target_arr, tau_model, weights)
        return weighted_loss

    seeds = [
        np.array([PHI, PSI], dtype=float),
        np.array([0.10, 0.08], dtype=float),
        np.array([0.15, 0.12], dtype=float),
        np.array([0.08, 0.18], dtype=float),
        np.array([0.50, 0.50], dtype=float),
        np.array([0.90, 0.90], dtype=float),
    ]
    best_loss = 1e9
    best_x = _clip_tau_params(seeds[0])
    for seed in seeds:
        loss_val, x_star = _powell_bounded(loss, seed, TAU_PARAM_BOUNDS, maxiter=2500)
        if loss_val < best_loss:
            best_loss = loss_val
            best_x = _clip_tau_params(x_star)

    phi_fit, psi_fit = best_x
    tau_fit = simulate_tau_from_v(v_arr, float(phi_fit), float(psi_fit), tau0=tau0)
    weighted_loss, r2 = tau_fit_metrics(tau_target_arr, tau_fit, weights)
    return float(phi_fit), float(psi_fit), float(weighted_loss), float(r2), tau_fit


def fit_tau_proxy_reference_constrained(
    v_series: np.ndarray,
    tau_target: np.ndarray,
    r_gain: float,
    re_counts: np.ndarray | None = None,
    mt_counts: np.ndarray | None = None,
) -> tuple[float, float, float, float, np.ndarray]:
    """Find the optimal (phi, psi) on the constraint line phi = r_gain * psi.

    The external toxicity reference fixes r_gain = tau_ref / V*_moderate = phi/psi.
    Psi is the single free parameter -- I'm optimising it by 1-D grid-then-refine
    for the best weighted proxy-fit quality.
    """
    v_arr = np.asarray(v_series, dtype=float)
    tau_target_arr = np.asarray(tau_target, dtype=float)
    tau0 = float(max(tau_target_arr[0], 0.0)) if tau_target_arr.size else 0.0

    if re_counts is not None:
        base_counts = np.asarray(re_counts, dtype=float)
        if mt_counts is not None:
            base_counts = base_counts + 0.5 * np.asarray(mt_counts, dtype=float)
        weights = np.sqrt(base_counts + 1.0)
        weights /= float(np.mean(weights) + 1e-12)
    else:
        weights = np.ones_like(tau_target_arr)

    psi_lo, psi_hi = TAU_PARAM_BOUNDS[1]
    phi_lo, phi_hi = TAU_PARAM_BOUNDS[0]
    r = float(r_gain)

    # Restricting psi so that phi = r*psi stays within bounds
    psi_max = float(np.clip(phi_hi / r, psi_lo, psi_hi)) if r > 0.0 else psi_hi

    def loss_at_psi(psi_val: float) -> float:
        psi_c = float(np.clip(psi_val, psi_lo, psi_max))
        phi_c = float(np.clip(r * psi_c, phi_lo, phi_hi))
        tau_model = simulate_tau_from_v(v_arr, phi_c, psi_c, tau0=tau0)
        w_loss, _ = tau_fit_metrics(tau_target_arr, tau_model, weights)
        return w_loss

    psi_grid = np.linspace(psi_lo, psi_max, 60)
    losses = np.array([loss_at_psi(p) for p in psi_grid])
    best_psi_init = float(psi_grid[int(np.argmin(losses))])

    result = minimize(
        lambda x: loss_at_psi(float(x[0])),
        x0=np.array([best_psi_init]),
        method='Powell',
        bounds=[(psi_lo, psi_max)],
        options={'maxiter': 1000, 'xtol': 1e-6, 'ftol': 1e-6},
    )
    opt_psi = float(np.clip(result.x[0], psi_lo, psi_max))
    opt_phi = float(np.clip(r * opt_psi, phi_lo, phi_hi))
    tau_fit = simulate_tau_from_v(v_arr, opt_phi, opt_psi, tau0=tau0)
    best_loss, best_r2 = tau_fit_metrics(tau_target_arr, tau_fit, weights)
    return opt_phi, opt_psi, float(best_loss), float(best_r2), tau_fit


def fit_tau_proxy(v_series: np.ndarray,
                  tau_target: np.ndarray,
                  re_counts: np.ndarray | None = None,
                  mt_counts: np.ndarray | None = None) -> tuple[float, float, float, float, np.ndarray, dict[str, float]]:
    """Fit the chosen proxy directly, also recording a decay-based psi diagnostic"""
    v_arr = np.asarray(v_series, dtype=float)
    tau_target = np.asarray(tau_target, dtype=float)
    if re_counts is not None:
        base_counts = np.asarray(re_counts, dtype=float)
        if mt_counts is not None:
            base_counts = base_counts + 0.5 * np.asarray(mt_counts, dtype=float)
        weights = np.sqrt(base_counts + 1.0)
        weights /= float(np.mean(weights) + 1e-12)
    else:
        weights = np.ones_like(tau_target)

    psi_anchor, decay_info = estimate_psi_from_decay(tau_target)
    phi_decay, tau_fit_decay, loss_decay, r2_decay = fit_phi_for_fixed_psi(v_arr, tau_target, float(psi_anchor), weights)
    phi_fit, psi_fit, weighted_loss, r2, tau_fit = fit_tau_proxy_unconstrained(
        v_arr,
        tau_target,
        weights,
    )
    diagnostics = {
        'psi_decay': float(psi_anchor),
        'psi_decay_r2': float(decay_info['decay_r2']),
        'decay_points': float(decay_info['n_decay_points']),
        'phi_decay_fit': float(phi_decay),
        'loss_decay_fit': float(loss_decay),
        'r2_decay_fit': float(r2_decay),
        'phi_unconstrained': float(phi_fit),
        'psi_unconstrained': float(psi_fit),
        'loss_unconstrained': float(weighted_loss),
        'r2_unconstrained': float(r2),
        'fit_mode': 1.0,
    }
    return phi_fit, psi_fit, weighted_loss, r2, tau_fit, diagnostics


def bootstrap_calibration(empirical_norm: np.ndarray,
                          window_counts: np.ndarray,
                          B: int = 500,
                          seed: int = 348) -> dict:
    """Poisson bootstrap -- resample counts and refit B times to get CIs"""
    beta0, gamma0, lambda0, lambda_decay, v0_fit, _loss, fitted_norm = fit_basic_ivf(empirical_norm, window_counts)
    fitted_counts = fitted_norm * np.max(window_counts)
    rng = np.random.default_rng(seed)

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
    from ivfs_config import DELTA, LAMBDA_U, MU_C, NU, RHO

    a_loss = DELTA + MU_C
    u_dfe = NU / LAMBDA_U
    n_star = RHO * u_dfe / (1.0 + u_dfe)

    for i in range(B):
        if (i + 1) % 50 == 0 or i == 0:
            print(f'  bootstrap {i + 1}/{B}...', end='\r')
        synthetic_counts = rng.poisson(np.maximum(fitted_counts, 0.5)).astype(float)
        if np.max(synthetic_counts) == 0:
            continue
        syn_norm = synthetic_counts / np.max(synthetic_counts)
        b, g = fast_refit(syn_norm, synthetic_counts)
        estimates['beta0'].append(float(b))
        estimates['gamma0'].append(float(g))
        estimates['alpha_r0'].append(float((a_loss * (g + MU_C)) / (b * n_star)))
    print()

    ci: dict[str, tuple[float, float]] = {}
    for key, vals in estimates.items():
        arr = np.array(vals)
        ci[key] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return ci


def profile_likelihood(empirical_norm: np.ndarray,
                       window_counts: np.ndarray,
                       best_params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sweep beta0 and gamma0 one at a time with guarded warm starts"""
    beta0_best, gamma0_best, _lambda0_best, _lambda_decay_best, _v0_best = best_params
    beta_grid = np.unique(np.sort(np.append(
        np.linspace(max(0.55, 0.70 * beta0_best), min(1.45, 1.35 * beta0_best), 35),
        beta0_best,
    )))
    gamma_grid = np.unique(np.sort(np.append(
        np.linspace(max(0.10, 0.60 * gamma0_best), min(0.50, 1.40 * gamma0_best), 35),
        gamma0_best,
    )))
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

        def optimize_at(fixed_value: float, seed_candidates: list[np.ndarray]) -> tuple[np.ndarray, float]:
            def loss_free(free_params):
                params_full = best_params.copy()
                params_full[fixed_idx] = fixed_value
                params_full[free_idx] = free_params
                return total_loss(params_full)

            best_fun = float('inf')
            best_params_local = best_params.copy()
            best_params_local[fixed_idx] = fixed_value
            for seed in seed_candidates:
                seed_arr = np.array(seed, dtype=float, copy=True)
                for j, (lo, hi) in enumerate(free_bounds):
                    seed_arr[j] = float(np.clip(seed_arr[j], lo, hi))

                seed_fun = float(loss_free(seed_arr))
                if seed_fun < best_fun:
                    best_fun = seed_fun
                    best_params_local[free_idx] = seed_arr

                fun, free_star = _powell_bounded(loss_free, seed_arr, free_bounds, maxiter=2500)
                if fun < best_fun:
                    best_fun = float(fun)
                    best_params_local[free_idx] = free_star
            return best_params_local.copy(), best_fun

        params_center, fun_center = optimize_at(float(grid[center]), [best_params[free_idx]])
        profile[center] = fun_center

        params_prev = params_center.copy()
        for idx in range(center + 1, len(grid)):
            params_prev, fun = optimize_at(float(grid[idx]), [params_prev[free_idx], best_params[free_idx]])
            profile[idx] = fun

        params_prev = params_center.copy()
        for idx in range(center - 1, -1, -1):
            params_prev, fun = optimize_at(float(grid[idx]), [params_prev[free_idx], best_params[free_idx]])
            profile[idx] = fun
        return profile

    print('  profiling beta0...', end='\r')
    profile_beta = sweep_profile(beta_grid, fixed_idx=0)
    print('  profiling gamma0...', end='\r')
    profile_gamma = sweep_profile(gamma_grid, fixed_idx=1)
    print()

    return beta_grid, profile_beta, gamma_grid, profile_gamma
