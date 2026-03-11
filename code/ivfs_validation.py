from __future__ import annotations

"""
This is the main script for the repo.
If someone clones the folder and only runs one file, this is the one I would point them to.
It rebuilds the Higgs calibration, runs the full IVFS scenarios, and saves the three-panel figure.
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


def fit_basic_ivf(v_empirical_norm: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Fit the baseline IVF curve with Nelder-Mead."""
    t_data = np.arange(len(v_empirical_norm), dtype=float)

    def ivf_ode(y, t, beta, gamma, mu):
        i_val, v_val, f_val = y
        di = -beta * i_val * v_val - mu * i_val
        dv = beta * i_val * v_val - gamma * v_val - mu * v_val
        df = gamma * v_val - mu * f_val
        return [di, dv, df]

    def loss(params):
        beta, gamma = np.abs(params)
        if beta < 1e-4 or gamma < 1e-4:
            return 1e9

        sol = odeint(ivf_ode, [0.995, 0.005, 0.0], t_data, args=(beta, gamma, 0.01), mxstep=10000)
        v_sim = sol[:, 1]
        vmax = float(np.max(v_sim))
        if vmax <= 1e-12:
            return 1e9

        v_sim_norm = v_sim / vmax
        return float(np.sum((v_sim_norm - v_empirical_norm) ** 2))

    result = minimize(
        loss,
        x0=np.array([0.8, 0.1]),
        method='Nelder-Mead',
        options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8},
    )

    beta0, gamma0 = np.abs(result.x)
    fit_sol = odeint(ivf_ode, [0.995, 0.005, 0.0], t_data, args=(beta0, gamma0, 0.01), mxstep=10000)
    v_fit = fit_sol[:, 1]
    v_fit_norm = v_fit / (np.max(v_fit) + 1e-12)
    return float(beta0), float(gamma0), float(result.fun), v_fit_norm


def full_ivfs_ode(y, t, alpha, beta0, gamma0):
    """Five-state content model with toxicity and active-user feedback."""
    i_val, v_val, f_val, tau, u_val = y

    kappa = 2.0
    eta = 3.0
    phi = 0.5
    psi = 0.1
    rho = 0.1
    lambda_u = 0.02
    nu = 1.0
    mu_c = 0.01
    delta = 0.01
    w = 5.0

    i_pos = max(i_val, 0.0)
    v_pos = max(v_val, 0.0)
    f_pos = max(f_val, 0.0)
    tau_pos = max(tau, 0.0)
    u_pos = max(u_val, 0.0)

    beta_eff = alpha * beta0 * (1.0 + kappa * tau_pos)
    gamma_eff = gamma0 * (1.0 + eta * tau_pos)

    # Without the saturation here, content inflow overwhelms everything and alpha stops mattering.
    n_inject = rho * u_pos / (1.0 + u_pos)

    di = n_inject - beta_eff * i_pos * v_pos - delta * i_pos - mu_c * i_pos
    dv = beta_eff * i_pos * v_pos - gamma_eff * v_pos - mu_c * v_pos
    df = gamma_eff * v_pos - mu_c * f_pos
    dtau = phi * v_pos - psi * tau_pos
    du = -lambda_u * (1.0 + w * tau_pos) * u_pos + nu
    return [di, dv, df, dtau, du]


def run_scenarios(beta0: float, gamma0: float):
    t = np.linspace(0, 1000, 10001)
    y0 = [1.0, 0.01, 0.0, 0.0, 50.0]
    scenarios = {
        'Engagement-First (alpha=0.9)': 0.9,
        'Moderate (alpha=0.5)': 0.5,
        'Health-First (alpha=0.2)': 0.2,
    }

    out = {}
    for label, alpha in scenarios.items():
        sol = odeint(full_ivfs_ode, y0, t, args=(alpha, beta0, gamma0), mxstep=20000)
        v_eq = float(sol[-1, 1])
        tau_eq = float(sol[-1, 3])
        u_eq = float(sol[-1, 4])
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
    y0 = [1.0, 0.01, 0.0, 0.0, 50.0]
    t_grid = np.linspace(0, 1000, 8001)
    alphas = np.linspace(0.05, 1.0, 50)

    tau_star = []
    for alpha in alphas:
        sol = odeint(full_ivfs_ode, y0, t_grid, args=(float(alpha), beta0, gamma0), mxstep=20000)
        tau_star.append(float(sol[-1, 3]))

    rho = 0.1
    lambda_u = 0.02
    nu = 1.0
    delta = 0.01
    mu_c = 0.01
    u_star = nu / lambda_u
    n_star = rho * u_star / (1.0 + u_star)
    alpha_r0 = float(((delta + mu_c) * (gamma0 + mu_c)) / (beta0 * n_star))

    return alphas, np.array(tau_star), alpha_r0


def make_figure(empirical_norm, fitted_norm, t_scenario, scenario_results, alphas, tau_star, alpha_r0):
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    t_data = np.arange(len(empirical_norm))
    axs[0].scatter(t_data, empirical_norm, s=14, alpha=0.6, color='#666666', label='Higgs RT (normalized)')
    axs[0].plot(t_data, fitted_norm, color='#C62828', linewidth=2.2, label='Fitted IVF')
    axs[0].set_title('Model Calibration (Higgs RT)')
    axs[0].set_xlabel('Hour index in 100-hour window')
    axs[0].set_ylabel('Normalized viral volume')
    axs[0].legend(fontsize=8)

    colors = {
        'Engagement-First (alpha=0.9)': '#C62828',
        'Moderate (alpha=0.5)': '#1565C0',
        'Health-First (alpha=0.2)': '#2E7D32',
    }
    for label, data in scenario_results.items():
        axs[1].plot(t_scenario, data['solution'][:, 3], linewidth=2.0, color=colors[label], label=label)
    axs[1].set_title('Scenario Time Series: Toxicity tau(t)')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Aggregate toxicity tau')
    axs[1].legend(fontsize=8)

    axs[2].plot(alphas, tau_star, color='#6A1B9A', linewidth=2.2)
    axs[2].axvline(alpha_r0, color='black', linestyle='--', linewidth=1.5, label=f'R0=1 at alpha~{alpha_r0:.4f}')
    axs[2].set_title('Long-Run Continuation: tau* vs alpha')
    axs[2].set_xlabel('Amplification alpha')
    axs[2].set_ylabel('Equilibrium toxicity tau*')
    axs[2].legend(fontsize=8)

    fig.tight_layout(pad=2)
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    print('=' * 68)
    print('APM348 IVFS validation')
    print('=' * 68)

    ensure_dataset()
    rt_timestamps, total_rows = parse_activity_file(HIGGS_TXT)
    hourly_counts, peak_hour, window_start, window_counts = build_hourly_curve(rt_timestamps)

    empirical_norm = window_counts / np.max(window_counts)
    beta0, gamma0, sse, fitted_norm = fit_basic_ivf(empirical_norm)

    t_scenario, scenario_results = run_scenarios(beta0, gamma0)
    alphas, tau_star, alpha_r0 = run_continuation(beta0, gamma0)
    make_figure(empirical_norm, fitted_norm, t_scenario, scenario_results, alphas, tau_star, alpha_r0)

    print(f'Total activity rows: {total_rows:,}')
    print(f'Total RT events: {len(rt_timestamps):,}')
    print(f'Peak RT hour: {peak_hour} (count={int(hourly_counts[peak_hour]):,})')
    print(f'Calibration window: hours {window_start}..{window_start + len(window_counts) - 1}')
    print()
    print(f'Fitted beta0: {beta0:.6f} per hour')
    print(f'Fitted gamma0: {gamma0:.6f} per hour')
    print(f'Normalized fit SSE: {sse:.6f}')
    print()
    print('Scenario long-run values (integrated to t=1000):')
    for label, data in scenario_results.items():
        print(
            f"- {label}: V*={data['V_star']:.6f}, tau*={data['tau_star']:.6f}, "
            f"U*={data['U_star']:.6f}, E={data['engagement']:.6f}"
        )
    print()
    print(f'R0=1 threshold alpha (disease-free approximation): {alpha_r0:.6f}')
    print(f'Saved figure: {FIGURE_PATH}')


if __name__ == '__main__':
    main()
