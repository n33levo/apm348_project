from __future__ import annotations

import numpy as np
from scipy.integrate import odeint

from ivfs_calibration import fit_tau_proxy, simulate_tau_from_v, tau_fit_metrics
from ivfs_config import (DELTA, ETA, KAPPA, LAMBDA_U, MU_C, NU, PHI, PSI, RHO, SCENARIO_ALPHAS, W)


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


def run_scenarios(beta0: float,
                  gamma0: float,
                  phi: float = PHI,
                  psi: float = PSI,
                  scenario_alphas: dict[str, float] | None = None):
    if scenario_alphas is None:
        scenario_alphas = SCENARIO_ALPHAS
    t = np.linspace(0, 500, 5001)
    a_loss = DELTA + MU_C
    u_dfe = NU / LAMBDA_U
    i_dfe = RHO * u_dfe / ((1.0 + u_dfe) * a_loss)
    y0 = [i_dfe, 0.01, 0.0, 0.0, 0.0, u_dfe]

    out = {}
    for label, alpha in scenario_alphas.items():
        def ode_custom(y, t_now, a, b, g, _phi=float(phi), _psi=float(psi)):
            return _ivfs_rhs(y, t_now, a, b, g,
                             KAPPA, ETA, _phi, _psi, RHO, LAMBDA_U, NU, MU_C, DELTA, W)

        sol = odeint(ode_custom, y0, t, args=(alpha, beta0, gamma0), mxstep=20000)
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


def run_continuation(beta0: float,
                     gamma0: float,
                     phi: float = PHI,
                     psi: float = PSI):
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
        def ode_custom(y, t_now, a, b, g, _phi=float(phi), _psi=float(psi)):
            return _ivfs_rhs(y, t_now, a, b, g,
                             KAPPA, ETA, _phi, _psi, RHO, LAMBDA_U, NU, MU_C, DELTA, W)

        sol = odeint(ode_custom, y0, t_grid, args=(float(alpha), beta0, gamma0), mxstep=20000)
        v_star_list.append(max(0.0, float(sol[-1, 1])))
        tau_star.append(max(0.0, float(sol[-1, 4])))
        u_star_list.append(float(sol[-1, 5]))

    n_star = RHO * u_dfe / (1.0 + u_dfe)
    alpha_r0 = float((a_loss * (gamma0 + MU_C)) / (beta0 * n_star))
    return alphas, np.array(tau_star), np.array(v_star_list), np.array(u_star_list), alpha_r0


def run_sensitivity(beta0: float, gamma0: float, alpha: float = 0.5):
    """bump each param +/-50% and see how tau* and U* move"""
    a_loss = DELTA + MU_C
    u_dfe = NU / LAMBDA_U
    i_dfe = RHO * u_dfe / ((1.0 + u_dfe) * a_loss)
    y0 = [i_dfe, 0.01, 0.0, 0.0, 0.0, u_dfe]
    t_grid = np.linspace(0, 2000, 10001)

    def get_eq(ode_func):
        sol = odeint(ode_func, y0, t_grid, args=(alpha, beta0, gamma0), mxstep=20000)
        return float(sol[-1, 4]), float(sol[-1, 5])

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
            frac_change = factor - 1.0
            tau_si = (dtau / tau_base) / frac_change if tau_base > 1e-12 else 0.0
            u_si = (du / u_base) / frac_change if abs(u_base) > 1e-12 else 0.0
            rows.append((name, label, base_val * factor, dtau, du, tau_si, u_si))
    return tau_base, u_base, rows


def build_tau_configurations(beta0: float,
                             gamma0: float,
                             fitted_v: np.ndarray,
                             tau_proxy_candidates: dict[str, np.ndarray] | None,
                             re_window: np.ndarray | None,
                             mt_window: np.ndarray | None,
                             external_phi: float | None = None,
                             external_source: str | None = None) -> dict[str, dict]:
    """assemble the baseline, Higgs-fit, and external tau-side configurations"""
    if not tau_proxy_candidates:
        return {}

    tau_configs: dict[str, dict] = {}

    best_proxy_name = None
    best_proxy_target = None
    best_proxy_fit = None
    for proxy_name, proxy_series in tau_proxy_candidates.items():
        tau_target_try = np.asarray(proxy_series, dtype=float)
        phi_fit_try, psi_fit_try, fit_loss_try, fit_r2_try, tau_fit_try, fit_diag_try = fit_tau_proxy(
            fitted_v,
            tau_target_try,
            re_window,
            mt_window,
        )
        if (
            best_proxy_fit is None
            or fit_r2_try > best_proxy_fit['proxy_r2'] + 1e-12
            or (abs(fit_r2_try - best_proxy_fit['proxy_r2']) <= 1e-12 and fit_loss_try < best_proxy_fit['proxy_loss'])
        ):
            best_proxy_name = proxy_name
            best_proxy_target = tau_target_try
            best_proxy_fit = {
                'phi': phi_fit_try,
                'psi': psi_fit_try,
                'proxy_loss': fit_loss_try,
                'proxy_r2': fit_r2_try,
                'tau_curve': tau_fit_try,
                'fit_diag': fit_diag_try,
            }

    assert best_proxy_name is not None and best_proxy_target is not None and best_proxy_fit is not None
    tau_target = best_proxy_target
    tau0 = float(max(tau_target[0], 0.0))

    def add_config(key: str, label: str, phi_val: float, psi_val: float, color: str) -> None:
        tau_curve = simulate_tau_from_v(fitted_v, phi_val, psi_val, tau0=tau0)
        proxy_loss, proxy_r2 = tau_fit_metrics(
            tau_target,
            tau_curve,
            np.sqrt(np.asarray(re_window, dtype=float) + 1.0) / float(np.mean(np.sqrt(np.asarray(re_window, dtype=float) + 1.0)) + 1e-12)
            if re_window is not None else None,
        )
        t_scenario, scenario_results = run_scenarios(beta0, gamma0, phi=phi_val, psi=psi_val)
        tau_configs[key] = {
            'label': label,
            'phi': float(phi_val),
            'psi': float(psi_val),
            'color': color,
            'tau_curve': tau_curve,
            'proxy_loss': proxy_loss,
            'proxy_r2': proxy_r2,
            'proxy_name': best_proxy_name,
            't_scenario': t_scenario,
            'scenario_results': scenario_results,
        }

    add_config('current', 'Current fixed τ setup', PHI, PSI, '#1565C0')

    phi_fit = best_proxy_fit['phi']
    psi_fit = best_proxy_fit['psi']
    fit_loss = best_proxy_fit['proxy_loss']
    fit_r2 = best_proxy_fit['proxy_r2']
    tau_fit = best_proxy_fit['tau_curve']
    fit_diag = best_proxy_fit['fit_diag']
    t_scenario_fit, scenario_results_fit = run_scenarios(beta0, gamma0, phi=phi_fit, psi=psi_fit)
    tau_configs['higgs_fit'] = {
        'label': f'Decay-regularized Higgs fit ({best_proxy_name})',
        'phi': phi_fit,
        'psi': psi_fit,
        'color': '#C62828',
        'tau_curve': tau_fit,
        'proxy_loss': fit_loss,
        'proxy_r2': fit_r2,
        'fit_diag': fit_diag,
        'proxy_name': best_proxy_name,
        't_scenario': t_scenario_fit,
        'scenario_results': scenario_results_fit,
    }

    if external_phi is not None and np.isfinite(external_phi):
        ext_label = 'External reference scale'
        if external_source:
            ext_label = f'External scale ({external_source})'
        add_config('external', ext_label, float(external_phi), PSI, '#6A1B9A')

    return tau_configs


def run_phi_sensitivity(beta0: float,
                        gamma0: float,
                        phi_grid: np.ndarray,
                        scenario_alphas: dict[str, float] | None = None):
    """sweep PHI while holding PSI fixed to show how robust the policy ordering is"""
    if scenario_alphas is None:
        scenario_alphas = SCENARIO_ALPHAS

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
