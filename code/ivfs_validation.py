from __future__ import annotations

"""
This is the main script for the repo.
If someone clones the folder and only runs one file, this is the one I would point them to.
It rebuilds the Higgs calibration, runs the full IVFS scenarios, and saves the results figure.
"""

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

# ------ shared model parameters ------
KAPPA = 0.8
ETA = 0.3
PHI = 0.056   # calibrated from Jigsaw: tau_empirical * PSI / V*_moderate = 0.1017 * 0.1 / 0.18
PSI = 0.1
RHO = 0.06
LAMBDA_U = 0.02
NU = 1.0
MU_C = 0.01
DELTA = 0.05
W = 10.0


def ensure_dataset() -> None:
    """Keep the Higgs dataset in data/ and unpack it if the plain-text file is missing."""
    ensure_layout()

    if not HIGGS_GZ.exists():
        print('[data] Higgs gzip file is missing, so I am downloading it from SNAP.')
        urllib.request.urlretrieve(HIGGS_URL, HIGGS_GZ)
        print(f'[data] Saved {HIGGS_GZ}')

    if not HIGGS_TXT.exists() or HIGGS_TXT.stat().st_mtime < HIGGS_GZ.stat().st_mtime:
        print('[data] Unzipping the Higgs activity file into data/.')
        with gzip.open(HIGGS_GZ, 'rt') as src, HIGGS_TXT.open('w') as dst:
            shutil.copyfileobj(src, dst)
        print(f'[data] Saved {HIGGS_TXT}')


def parse_activity_file(path: Path) -> tuple[np.ndarray, int]:
    """Stream the big activity file once and keep only retweet timestamps."""
    rt_timestamps: list[int] = []
    total_rows = 0

    with path.open('r') as handle:
        for line in handle:
            total_rows += 1
            parts = line.split()
            if len(parts) >= 4 and parts[3] == 'RT':
                rt_timestamps.append(int(parts[2]))

    return np.array(rt_timestamps, dtype=np.int64), total_rows


def build_hourly_curve(rt_timestamps: np.ndarray) -> tuple[np.ndarray, int, int, np.ndarray]:
    """Convert raw retweet timestamps into hourly counts and extract the 100-hour fit window."""
    t0 = int(rt_timestamps.min())
    hours = (rt_timestamps - t0) / 3600.0
    max_hour = int(np.ceil(hours.max()))
    hourly_counts, _ = np.histogram(hours, bins=np.arange(0, max_hour + 1, 1))

    peak_hour = int(np.argmax(hourly_counts))

    # I keep the peak near the front of the window on purpose.
    # That gives the fit room to capture both the climb and the long tail.
    start = max(0, peak_hour - 10)
    end = min(len(hourly_counts), start + 100)
    start = max(0, end - 100)

    window_counts = hourly_counts[start:end].astype(float)
    return hourly_counts, peak_hour, start, window_counts


def fit_basic_ivf(v_empirical_norm: np.ndarray,
                  window_counts: np.ndarray | None = None) -> tuple[float, float, float, float, np.ndarray]:
    """Fit the IVF curve to the normalized Higgs spike.

    We add a small background influx (lambda_bg) so the model doesn't
    just drop to zero in the tail — the real data has persistent low-level
    activity that a pure epidemic curve misses.
    Uses Poisson-weighted SSE when raw counts are provided.
    Returns (beta0, gamma0, lambda_bg, sse, fitted_norm).
    """
    t_data = np.arange(len(v_empirical_norm), dtype=float)
    if window_counts is not None:
        weights = 1.0 / (np.sqrt(window_counts) + 1.0)
    else:
        weights = np.ones_like(v_empirical_norm)

    def ivf_ode(y, t, beta, gamma, mu, lam):
        i_val, v_val, f_val = y
        di = -beta * i_val * v_val - mu * i_val + lam
        dv = beta * i_val * v_val - gamma * v_val - mu * v_val
        df = gamma * v_val - mu * f_val
        return [di, dv, df]

    def loss(params):
        beta, gamma, lam, v0 = np.abs(params)
        v0 = np.clip(v0, 1e-5, 0.1)
        if beta < 1e-4 or gamma < 1e-4:
            return 1e9

        sol = odeint(ivf_ode, [1.0 - v0, v0, 0.0], t_data, args=(beta, gamma, 0.01, lam), mxstep=10000)
        v_sim = sol[:, 1]
        vmax = float(np.max(v_sim))
        if vmax <= 1e-12:
            return 1e9

        v_sim_norm = v_sim / vmax
        return float(np.sum(weights * (v_sim_norm - v_empirical_norm) ** 2))

    # Multi-start to avoid bad local minima.
    best_x = np.array([0.8, 0.2, 0.005, 0.001])
    best_sse = 1e9
    for b0 in (0.5, 0.8, 1.2):
        for g0 in (0.1, 0.2, 0.3):
            for l0 in (0.002, 0.005, 0.01):
                for v0_init in (0.0005, 0.001, 0.003, 0.005):
                    res = minimize(
                        loss,
                        x0=np.array([b0, g0, l0, v0_init]),
                        method='Nelder-Mead',
                        options={'maxiter': 15000, 'xatol': 1e-9, 'fatol': 1e-9},
                    )
                    if res.fun < best_sse:
                        best_sse = res.fun
                        best_x = np.abs(res.x)

    beta0, gamma0, lambda_bg, v0_fit = best_x
    v0_fit = np.clip(v0_fit, 1e-5, 0.1)
    fit_sol = odeint(ivf_ode, [1.0 - v0_fit, v0_fit, 0.0], t_data, args=(beta0, gamma0, 0.01, lambda_bg), mxstep=10000)
    v_fit = fit_sol[:, 1]
    v_fit_norm = v_fit / (np.max(v_fit) + 1e-12)
    return float(beta0), float(gamma0), float(lambda_bg), float(best_sse), v_fit_norm


def _ivfs_rhs(y, t, alpha, beta0, gamma0,
              kappa, eta, phi, psi, rho, lambda_u, nu, mu_c, delta, w):
    """Parametric IVFS right-hand side (6 states: I, V, F, S, tau, U)."""
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
    """Six-state IVFS ODE using default module-level parameters."""
    return _ivfs_rhs(y, t, alpha, beta0, gamma0,
                     KAPPA, ETA, PHI, PSI, RHO, LAMBDA_U, NU, MU_C, DELTA, W)


def run_scenarios(beta0: float, gamma0: float):
    t = np.linspace(0, 500, 5001)
    # Start at the DFE, nudge V to 0.01 to kick things off.
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


def make_figure(empirical_norm, fitted_norm, sse, r2,
                t_scenario, scenario_results,
                alphas, tau_star, v_star_cont, u_star_cont, alpha_r0):
    plt.style.use('ggplot')
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    colors = {
        'Engagement-First (alpha=0.9)': '#C62828',
        'Moderate (alpha=0.5)': '#1565C0',
        'Health-First (alpha=0.2)': '#2E7D32',
    }

    # (a) Calibration
    t_data = np.arange(len(empirical_norm))
    axs[0, 0].scatter(t_data, empirical_norm, s=14, alpha=0.55, color='#666666',
                       label='Higgs RT (normalized)')
    axs[0, 0].plot(t_data, fitted_norm, color='#C62828', linewidth=2.2, label='Fitted IVF')
    axs[0, 0].axvline(35, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    axs[0, 0].set_title('(a) Model Calibration')
    axs[0, 0].set_xlabel('Hour index')
    axs[0, 0].set_ylabel('Normalized viral volume')
    axs[0, 0].annotate(f'SSE = {sse:.2f}\nR\u00B2 = {r2:.2f} (spike, 0\u201335 h)',
                        xy=(0.42, 0.70), xycoords='axes fraction', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    axs[0, 0].legend(fontsize=8)

    # (b) tau(t) scenarios
    for label, data in scenario_results.items():
        axs[0, 1].plot(t_scenario, data['solution'][:, 4], linewidth=2.0,
                        color=colors[label], label=label)
    axs[0, 1].set_title('(b) Toxicity \u03C4(t) by Policy')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Aggregate toxicity \u03C4')
    axs[0, 1].legend(fontsize=8)

    # (c) U(t) scenarios
    for label, data in scenario_results.items():
        axs[0, 2].plot(t_scenario, data['solution'][:, 5], linewidth=2.0,
                        color=colors[label], label=label)
    axs[0, 2].set_title('(c) Active Users U(t) by Policy')
    axs[0, 2].set_xlabel('Time')
    axs[0, 2].set_ylabel('Active users U')
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

    # (f) Engagement E* = alpha * V* * U*
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

    fig.suptitle('APM348 IVFS Model Results', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout(pad=2)
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)


def run_sensitivity(beta0: float, gamma0: float, alpha: float = 0.5):
    """Vary each secondary param by +/-50% and report how tau* and U* shift."""
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


def find_W_for_interior_Emax(beta0: float, gamma0: float,
                              w_grid: np.ndarray | None = None):
    """Find smallest W such that E*=alpha*V**U* peaks at an interior alpha."""
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
    """Parametric bootstrap: resample from Poisson(fitted_counts), refit B times.

    Uses a fast single-start refit seeded from the best-fit point to keep
    runtime reasonable (~1-2 min for B=500 instead of hours).
    """
    from scipy.stats import poisson

    beta0, gamma0, lbg, sse, fitted_norm = fit_basic_ivf(empirical_norm, window_counts)
    fitted_counts = fitted_norm * np.max(window_counts)

    # Build a fast single-start refitter seeded from the MLE.
    t_data = np.arange(len(empirical_norm), dtype=float)

    def ivf_ode(y, t, beta, gamma, mu, lam):
        i_val, v_val, f_val = y
        di = -beta * i_val * v_val - mu * i_val + lam
        dv = beta * i_val * v_val - gamma * v_val - mu * v_val
        df = gamma * v_val - mu * f_val
        return [di, dv, df]

    def fast_refit(syn_norm, syn_counts):
        w = 1.0 / (np.sqrt(syn_counts) + 1.0)
        def loss(params):
            beta, gamma, lam, v0 = np.abs(params)
            v0 = np.clip(v0, 1e-5, 0.1)
            if beta < 1e-4 or gamma < 1e-4:
                return 1e9
            sol = odeint(ivf_ode, [1.0 - v0, v0, 0.0], t_data,
                         args=(beta, gamma, 0.01, lam), mxstep=10000)
            v_sim = sol[:, 1]
            vmax = float(np.max(v_sim))
            if vmax <= 1e-12:
                return 1e9
            return float(np.sum(w * (v_sim / vmax - syn_norm) ** 2))
        res = minimize(loss, x0=np.array([beta0, gamma0, lbg, 0.001]),
                       method='Nelder-Mead',
                       options={'maxiter': 8000, 'xatol': 1e-8, 'fatol': 1e-8})
        return np.abs(res.x[:2])  # beta, gamma

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
                        beta0_best: float,
                        gamma0_best: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute profile likelihood for beta0 and gamma0."""
    beta_grid = np.linspace(0.4, 3.0, 50)
    gamma_grid = np.linspace(0.05, 0.5, 50)
    profile_beta: list[float] = []
    profile_gamma: list[float] = []
    weights = 1.0 / (np.sqrt(window_counts) + 1.0)
    t_data = np.arange(len(empirical_norm), dtype=float)

    def ivf_ode(y, t, beta, gamma, mu, lam):
        i_val, v_val, f_val = y
        di = -beta * i_val * v_val - mu * i_val + lam
        dv = beta * i_val * v_val - gamma * v_val - mu * v_val
        df = gamma * v_val - mu * f_val
        return [di, dv, df]

    print('  profiling beta0...', end='\r')
    for b_fixed in beta_grid:
        def loss_b(params, _bf=b_fixed):
            gamma, lam, v0 = np.abs(params)
            v0 = np.clip(v0, 1e-5, 0.1)
            if gamma < 1e-4:
                return 1e9
            sol = odeint(ivf_ode, [1.0 - v0, v0, 0.0], t_data,
                         args=(_bf, gamma, 0.01, lam), mxstep=10000)
            v = sol[:, 1]
            vm = float(np.max(v))
            if vm < 1e-12:
                return 1e9
            return float(np.sum(weights * (v / vm - empirical_norm) ** 2))
        res = minimize(loss_b, [gamma0_best, 0.005, 0.001], method='Nelder-Mead',
                       options={'maxiter': 8000})
        profile_beta.append(float(res.fun))

    print('  profiling gamma0...', end='\r')
    for g_fixed in gamma_grid:
        def loss_g(params, _gf=g_fixed):
            beta, lam, v0 = np.abs(params)
            v0 = np.clip(v0, 1e-5, 0.1)
            if beta < 1e-4:
                return 1e9
            sol = odeint(ivf_ode, [1.0 - v0, v0, 0.0], t_data,
                         args=(beta, _gf, 0.01, lam), mxstep=10000)
            v = sol[:, 1]
            vm = float(np.max(v))
            if vm < 1e-12:
                return 1e9
            return float(np.sum(weights * (v / vm - empirical_norm) ** 2))
        res = minimize(loss_g, [beta0_best, 0.005, 0.001], method='Nelder-Mead',
                       options={'maxiter': 8000})
        profile_gamma.append(float(res.fun))
    print()  # clear \r

    return beta_grid, np.array(profile_beta), gamma_grid, np.array(profile_gamma)


def plot_profile(beta_grid, profile_beta, gamma_grid, profile_gamma,
                 beta0_best, gamma0_best) -> None:
    """Save profile likelihood figure."""
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(beta_grid, profile_beta, 'o-', markersize=3, color='#C62828')
    ax1.axvline(beta0_best, color='black', linestyle='--', linewidth=1.2,
                label=f'\u03B2\u2080={beta0_best:.3f}')
    ax1.set_xlabel('\u03B2\u2080 (fixed)')
    ax1.set_ylabel('Profile SSE')
    ax1.set_title('Profile Likelihood: \u03B2\u2080')
    ax1.legend(fontsize=9)

    ax2.plot(gamma_grid, profile_gamma, 'o-', markersize=3, color='#1565C0')
    ax2.axvline(gamma0_best, color='black', linestyle='--', linewidth=1.2,
                label=f'\u03B3\u2080={gamma0_best:.3f}')
    ax2.set_xlabel('\u03B3\u2080 (fixed)')
    ax2.set_ylabel('Profile SSE')
    ax2.set_title('Profile Likelihood: \u03B3\u2080')
    ax2.legend(fontsize=9)

    fig.tight_layout()
    out_path = ASSETS_DIR / 'profile_likelihood.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved profile likelihood figure: {out_path}')


def main() -> None:
    print('=' * 68)
    print('APM348 IVFS validation')
    print('=' * 68)

    ensure_dataset()
    rt_timestamps, total_rows = parse_activity_file(HIGGS_TXT)
    hourly_counts, peak_hour, window_start, window_counts = build_hourly_curve(rt_timestamps)

    empirical_norm = window_counts / np.max(window_counts)
    beta0, gamma0, lambda_bg, sse, fitted_norm = fit_basic_ivf(empirical_norm, window_counts)

    # R-squared over the spike region only (first 35 hours).
    # The tail has secondary bursts that no deterministic ODE should match.
    spike_end = 35
    spike_sse = float(np.sum((fitted_norm[:spike_end] - empirical_norm[:spike_end]) ** 2))
    spike_ss_tot = float(np.sum((empirical_norm[:spike_end] - np.mean(empirical_norm[:spike_end])) ** 2))
    r2_spike = 1.0 - spike_sse / spike_ss_tot if spike_ss_tot > 0 else 0.0

    t_scenario, scenario_results = run_scenarios(beta0, gamma0)
    alphas, tau_star, v_star_cont, u_star_cont, alpha_r0 = run_continuation(beta0, gamma0)
    make_figure(empirical_norm, fitted_norm, sse, r2_spike,
                t_scenario, scenario_results,
                alphas, tau_star, v_star_cont, u_star_cont, alpha_r0)

    # DFE values for R0 display.
    a_loss = DELTA + MU_C
    u_dfe = NU / LAMBDA_U
    i_dfe = RHO * u_dfe / ((1.0 + u_dfe) * a_loss)

    print(f'Total activity rows: {total_rows:,}')
    print(f'Total RT events: {len(rt_timestamps):,}')
    print(f'Peak RT hour: {peak_hour} (count={int(hourly_counts[peak_hour]):,})')
    print(f'Calibration window: hours {window_start}..{window_start + len(window_counts) - 1}')
    print()
    print(f'Fitted beta0: {beta0:.6f} per hour')
    print(f'Fitted gamma0: {gamma0:.6f} per hour')
    print(f'Fitted lambda_bg: {lambda_bg:.6f}')
    print(f'Normalized fit SSE (full window): {sse:.6f}')
    print(f'R\u00B2 (spike window, 0-{spike_end}h): {r2_spike:.4f}')
    print()
    print('Scenario long-run values (integrated to t=500):')
    for label, data in scenario_results.items():
        r0_val = data['alpha'] * beta0 * i_dfe / (gamma0 + MU_C)
        print(
            f"- {label}: R0={r0_val:.2f}, V*={max(0, data['V_star']):.6f}, tau*={max(0, data['tau_star']):.6f}, "
            f"U*={data['U_star']:.4f}, E={max(0, data['engagement']):.6f}"
        )
    print()
    print(f'R0=1 threshold alpha: {alpha_r0:.4f}')
    print(f'Saved figure: {FIGURE_PATH}')

    # Sensitivity analysis with normalized indices.
    print()
    print('Sensitivity (alpha=0.5, +/-50% each param):')
    print(f'{"param":<8} {"change":<8} {"value":<8} {"d_tau*":<10} {"d_U*":<10} {"S_tau":<8} {"S_U":<8}')
    tau_base, u_base, sens_rows = run_sensitivity(beta0, gamma0)
    print(f'{"(base)":<8} {"":<8} {"":<8} {tau_base:<10.4f} {u_base:<10.4f}')
    for name, label, val, dtau, du, s_tau, s_u in sens_rows:
        print(f'{name:<8} {label:<8} {val:<8.3f} {dtau:<+10.4f} {du:<+10.4f} {s_tau:<+8.3f} {s_u:<+8.3f}')

    # Gap 6: Search for W that gives an interior E* peak.
    print()
    w_opt, alpha_peak = find_W_for_interior_Emax(beta0, gamma0)
    if w_opt is not None:
        print(f'Interior E* peak found: W={w_opt:.1f} gives peak at alpha={alpha_peak:.2f}')
    else:
        print('No interior E* peak found for W <= 80; E* peaks at alpha=1.0.')
        print('Policy insight: reducing alpha below R0=1 maximizes user retention U*(alpha).')

    # Gap 4: Profile likelihood.
    print()
    print('Computing profile likelihood...')
    bg, pb, gg, pg = profile_likelihood(empirical_norm, window_counts, beta0, gamma0)
    plot_profile(bg, pb, gg, pg, beta0, gamma0)

    # Gap 3: Bootstrap confidence intervals.
    print()
    print('Running parametric bootstrap (B=500)...')
    ci = bootstrap_calibration(empirical_norm, window_counts, B=500)
    print(f"beta0 95% CI: [{ci['beta0'][0]:.4f}, {ci['beta0'][1]:.4f}]")
    print(f"gamma0 95% CI: [{ci['gamma0'][0]:.4f}, {ci['gamma0'][1]:.4f}]")
    print(f"R0=1 threshold alpha 95% CI: [{ci['alpha_r0'][0]:.4f}, {ci['alpha_r0'][1]:.4f}]")


if __name__ == '__main__':
    main()
