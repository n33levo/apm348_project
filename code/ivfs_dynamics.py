from __future__ import annotations

import numpy as np

from .common import solve_trajectory
from .ivfs_calibration import fit_tau_proxy, fit_tau_proxy_reference_constrained, simulate_tau_from_v, tau_fit_metrics
from .ivfs_config import (DELTA, ETA, KAPPA, LAMBDA_U, MU_C, NU, PHI, PSI, RHO,
                          SCENARIO_ALPHAS, SOLVER_ATOL, SOLVER_MAX_STEP,
                          SOLVER_METHOD, SOLVER_RTOL, W)


def _ivfs_rhs(t, y, alpha, beta0, gamma0,
              kappa, eta, phi, psi, rho, lambda_u, nu, mu_c, delta, w):
    """The full 6-state ODE right-hand side (I, V, F, S, tau, U).

    The implementation clamps negative state values to zero as a numerical safeguard.
    This modifies the vector field only in a small neighbourhood of the boundary,
    where adaptive solvers can otherwise step slightly negative.
    """
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


def full_ivfs_ode(t, y, alpha, beta0, gamma0):
    """Wrap _ivfs_rhs with the default params plugged in"""
    return _ivfs_rhs(t, y, alpha, beta0, gamma0,
                     KAPPA, ETA, PHI, PSI, RHO, LAMBDA_U, NU, MU_C, DELTA, W)


def run_scenarios(beta0: float,
                  gamma0: float,
                  phi: float = PHI,
                  psi: float = PSI,
                  scenario_alphas: dict[str, float] | None = None):
    if scenario_alphas is None:
        scenario_alphas = SCENARIO_ALPHAS
    t = np.linspace(0, 2000, 10001)
    a_loss = DELTA + MU_C
    u_dfe = NU / LAMBDA_U
    i_dfe = RHO * u_dfe / ((1.0 + u_dfe) * a_loss)
    y0 = [i_dfe, 0.01, 0.0, 0.0, 0.0, u_dfe]

    out = {}
    for label, alpha in scenario_alphas.items():
        def ode_custom(t_now, y, a, b, g, _phi=float(phi), _psi=float(psi)):
            return _ivfs_rhs(t_now, y, a, b, g,
                             KAPPA, ETA, _phi, _psi, RHO, LAMBDA_U, NU, MU_C, DELTA, W)

        sol = solve_trajectory(
            ode_custom, y0, t, args=(alpha, beta0, gamma0),
            method=SOLVER_METHOD, rtol=SOLVER_RTOL, atol=SOLVER_ATOL, max_step=SOLVER_MAX_STEP,
        )
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
        def ode_custom(t_now, y, a, b, g, _phi=float(phi), _psi=float(psi)):
            return _ivfs_rhs(t_now, y, a, b, g,
                             KAPPA, ETA, _phi, _psi, RHO, LAMBDA_U, NU, MU_C, DELTA, W)

        sol = solve_trajectory(
            ode_custom, y0, t_grid, args=(float(alpha), beta0, gamma0),
            method=SOLVER_METHOD, rtol=SOLVER_RTOL, atol=SOLVER_ATOL, max_step=SOLVER_MAX_STEP,
        )
        v_star_list.append(max(0.0, float(sol[-1, 1])))
        tau_star.append(max(0.0, float(sol[-1, 4])))
        u_star_list.append(float(sol[-1, 5]))

    n_star = RHO * u_dfe / (1.0 + u_dfe)
    alpha_r0 = float((a_loss * (gamma0 + MU_C)) / (beta0 * n_star))
    return alphas, np.array(tau_star), np.array(v_star_list), np.array(u_star_list), alpha_r0


def run_sensitivity(beta0: float, gamma0: float, alpha: float = 0.5):
    """Compute central finite-difference sensitivity indices (averaging +/-1% perturbations)"""
    a_loss = DELTA + MU_C
    u_dfe = NU / LAMBDA_U
    i_dfe = RHO * u_dfe / ((1.0 + u_dfe) * a_loss)
    y0 = [i_dfe, 0.01, 0.0, 0.0, 0.0, u_dfe]
    t_grid = np.linspace(0, 2000, 10001)

    def get_eq(ode_func, alpha_ov=alpha, beta0_ov=beta0, gamma0_ov=gamma0):
        sol = solve_trajectory(
            ode_func, y0, t_grid, args=(alpha_ov, beta0_ov, gamma0_ov),
            method=SOLVER_METHOD, rtol=SOLVER_RTOL, atol=SOLVER_ATOL, max_step=SOLVER_MAX_STEP,
        )
        return float(sol[-1, 1]), float(sol[-1, 4]), float(sol[-1, 5])

    v_base, tau_base, u_base = get_eq(full_ivfs_ode)

    def _elasticity(v_lo, v_hi, base, dp_frac=0.02):
        """Central-difference elasticity: S = (x_hi - x_lo) / (2*dp) * (p/x_base)"""
        if abs(base) < 1e-12:
            return 0.0
        return ((v_hi - v_lo) / base) / dp_frac

    # Parameters passed inside the ODE (kappa, eta, phi, psi, w)
    internal_params = {'kappa': KAPPA, 'eta': ETA, 'phi': PHI, 'psi': PSI, 'w': W}
    rows = []
    for name, base_val in internal_params.items():
        results = {}
        for factor in (0.99, 1.01):
            ov = {name: base_val * factor}

            def ode_mod(t_now, y, a, b, g, _ov=ov):
                return _ivfs_rhs(t_now, y, a, b, g,
                                 _ov.get('kappa', KAPPA), _ov.get('eta', ETA),
                                 _ov.get('phi', PHI), _ov.get('psi', PSI),
                                 RHO, LAMBDA_U, NU, MU_C, DELTA,
                                 _ov.get('w', W))

            results[factor] = get_eq(ode_mod)

        v_lo, tau_lo, u_lo = results[0.99]
        v_hi, tau_hi, u_hi = results[1.01]
        v_si = _elasticity(v_lo, v_hi, v_base)
        tau_si = _elasticity(tau_lo, tau_hi, tau_base)
        u_si = _elasticity(u_lo, u_hi, u_base)
        rows.append((name, v_si, tau_si, u_si))

    # Parameters passed as ODE args (alpha, beta0, gamma0)
    arg_params = {'alpha': alpha, 'beta0': beta0, 'gamma0': gamma0}
    for name, base_val in arg_params.items():
        results = {}
        for factor in (0.99, 1.01):
            new_val = base_val * factor
            a_ov = new_val if name == 'alpha' else alpha
            b_ov = new_val if name == 'beta0' else beta0
            g_ov = new_val if name == 'gamma0' else gamma0
            results[factor] = get_eq(full_ivfs_ode, alpha_ov=a_ov, beta0_ov=b_ov, gamma0_ov=g_ov)

        v_lo, tau_lo, u_lo = results[0.99]
        v_hi, tau_hi, u_hi = results[1.01]
        v_si = _elasticity(v_lo, v_hi, v_base)
        tau_si = _elasticity(tau_lo, tau_hi, tau_base)
        u_si = _elasticity(u_lo, u_hi, u_base)
        rows.append((name, v_si, tau_si, u_si))

    return v_base, tau_base, u_base, rows


def build_tau_configurations(beta0: float,
                             gamma0: float,
                             fitted_v: np.ndarray,
                             tau_proxy_candidates: dict[str, np.ndarray] | None,
                             re_window: np.ndarray | None,
                             mt_window: np.ndarray | None,
                             external_phi: float | None = None,
                             external_source: str | None = None) -> dict[str, dict]:
    """Assemble baseline, same-dataset, and external tau-side configurations"""
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

    fit_diag = best_proxy_fit['fit_diag']
    add_config(
        'higgs_decay',
        f'Decay-anchored Higgs fit ({best_proxy_name})',
        float(fit_diag['phi_decay_fit']),
        float(fit_diag['psi_decay']),
        '#EF6C00',
    )

    t_scenario_fit, scenario_results_fit = run_scenarios(
        beta0,
        gamma0,
        phi=float(best_proxy_fit['phi']),
        psi=float(best_proxy_fit['psi']),
    )
    tau_configs['higgs_unconstrained'] = {
        'label': f'Unconstrained Higgs fit ({best_proxy_name})',
        'phi': float(best_proxy_fit['phi']),
        'psi': float(best_proxy_fit['psi']),
        'color': '#C62828',
        'tau_curve': best_proxy_fit['tau_curve'],
        'proxy_loss': float(best_proxy_fit['proxy_loss']),
        'proxy_r2': float(best_proxy_fit['proxy_r2']),
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

    # Reference-constrained fit
    # The external reference fixes r = phi/psi, so any point on that line
    # gives the same long-run tau*. I'm optimising psi along the constraint
    # for the best proxy fit.
    if external_phi is not None and np.isfinite(external_phi):
        r_gain = float(external_phi) / PSI
        re_w_arr = np.asarray(re_window, dtype=float) if re_window is not None else None
        mt_w_arr = np.asarray(mt_window, dtype=float) if mt_window is not None else None
        phi_rc, psi_rc, _, _, tau_curve_rc = fit_tau_proxy_reference_constrained(
            fitted_v, tau_target, r_gain, re_w_arr, mt_w_arr,
        )
        rc_label = 'Reference-constrained'
        if external_source:
            rc_label = f'Ref.-constrained ({external_source})'
        weights_rc = (
            np.sqrt(re_w_arr + 1.0) / float(np.mean(np.sqrt(re_w_arr + 1.0)) + 1e-12)
            if re_w_arr is not None else None
        )
        proxy_loss_rc, proxy_r2_rc = tau_fit_metrics(tau_target, tau_curve_rc, weights_rc)
        t_scenario_rc, scenario_results_rc = run_scenarios(beta0, gamma0, phi=phi_rc, psi=psi_rc)
        tau_configs['reference_constrained'] = {
            'label': rc_label,
            'phi': phi_rc,
            'psi': float(psi_rc),
            'color': '#2E7D32',
            'tau_curve': tau_curve_rc,
            'proxy_loss': proxy_loss_rc,
            'proxy_r2': proxy_r2_rc,
            'proxy_name': best_proxy_name,
            't_scenario': t_scenario_rc,
            'scenario_results': scenario_results_rc,
        }

    return tau_configs


def run_phi_sensitivity(beta0: float,
                        gamma0: float,
                        phi_grid: np.ndarray,
                        scenario_alphas: dict[str, float] | None = None):
    """Sweep PHI while holding PSI fixed to check how robust the policy ordering is"""
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
            def ode_phi(t_now, y, a, b, g, _phi=float(phi_val)):
                return _ivfs_rhs(t_now, y, a, b, g,
                                 KAPPA, ETA, _phi, PSI, RHO, LAMBDA_U, NU, MU_C, DELTA, W)

            sol = solve_trajectory(
                ode_phi, y0, t_grid, args=(alpha, beta0, gamma0),
                method=SOLVER_METHOD, rtol=SOLVER_RTOL, atol=SOLVER_ATOL, max_step=SOLVER_MAX_STEP,
            )
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
    """Scan W values to see if E* ever peaks at an interior alpha"""
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
            def ode_w(t_now, y, a, b, g, _w=w_test):
                return _ivfs_rhs(t_now, y, a, b, g, KAPPA, ETA, PHI, PSI,
                                 RHO, LAMBDA_U, NU, MU_C, DELTA, _w)

            sol = solve_trajectory(
                ode_w, y0, t_grid, args=(float(alpha), beta0, gamma0),
                method=SOLVER_METHOD, rtol=SOLVER_RTOL, atol=SOLVER_ATOL, max_step=SOLVER_MAX_STEP,
            )
            v_eq = max(0.0, float(sol[-1, 1]))
            u_eq = float(sol[-1, 5])
            e_vals.append(float(alpha) * v_eq * u_eq)
        peak_idx = int(np.argmax(e_vals))
        if peak_idx < len(alphas) - 2:
            return float(w_test), float(alphas[peak_idx])
    return None, None
