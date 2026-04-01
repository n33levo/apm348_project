from __future__ import annotations

"""Equilibrium and stability analysis for the IVFS model"""

import numpy as np
from numpy.linalg import eigvals
from scipy.optimize import root_scalar

from .ivfs_calibration import build_hourly_curve, ensure_dataset, fit_basic_ivf, parse_activity_file
from .ivfs_config import DELTA, ETA, FIT_WINDOW_HOURS, KAPPA, LAMBDA_U, MU_C, NU, PHI, PSI, RHO, W

A_LOSS = DELTA + MU_C


def inflow(u_val: float) -> float:
    return RHO * u_val / (1.0 + u_val)


def dfe(alpha: float, beta0: float, gamma0: float) -> tuple[float, float, float]:
    u_star = NU / LAMBDA_U
    i_star = inflow(u_star) / A_LOSS
    r0 = alpha * beta0 * i_star / (gamma0 + MU_C)
    return i_star, u_star, r0


def scalar_balance(tau: float, alpha: float, beta0: float, gamma0: float) -> float:
    u_star = NU / (LAMBDA_U * (1.0 + W * tau))
    spread_term = inflow(u_star) / (gamma0 * (1.0 + ETA * tau) + MU_C)
    removal_term = A_LOSS / (alpha * beta0 * (1.0 + KAPPA * tau))
    toxicity_term = PSI * tau / PHI
    return spread_term - removal_term - toxicity_term


def positive_equilibrium(alpha: float, beta0: float, gamma0: float) -> np.ndarray | None:
    if scalar_balance(0.0, alpha, beta0, gamma0) <= 0.0:
        return None

    right = 1.0
    while scalar_balance(right, alpha, beta0, gamma0) > 0.0 and right < 1e6:
        right *= 2.0
    if scalar_balance(right, alpha, beta0, gamma0) > 0.0:
        return None

    root = root_scalar(lambda x: scalar_balance(x, alpha, beta0, gamma0), bracket=[0.0, right], method='brentq')
    tau_star = float(root.root)
    v_star = PSI * tau_star / PHI
    u_star = NU / (LAMBDA_U * (1.0 + W * tau_star))
    beta_eff = alpha * beta0 * (1.0 + KAPPA * tau_star)
    gamma_eff = gamma0 * (1.0 + ETA * tau_star)
    i_star = (gamma_eff + MU_C) / beta_eff
    f_star = gamma_eff * v_star / MU_C
    s_star = DELTA * i_star / MU_C
    return np.array([i_star, v_star, f_star, s_star, tau_star, u_star], dtype=float)


def jacobian(state: np.ndarray, alpha: float, beta0: float, gamma0: float) -> np.ndarray:
    i_star, v_star, f_star, s_star, tau_star, u_star = state
    beta_eff = alpha * beta0 * (1.0 + KAPPA * tau_star)
    gamma_eff = gamma0 * (1.0 + ETA * tau_star)
    dbeta_dtau = alpha * beta0 * KAPPA
    dgamma_dtau = gamma0 * ETA
    dinflow_du = RHO / (1.0 + u_star) ** 2

    jac = np.zeros((6, 6))
    jac[0, 0] = -beta_eff * v_star - A_LOSS
    jac[0, 1] = -beta_eff * i_star
    jac[0, 4] = -dbeta_dtau * i_star * v_star
    jac[0, 5] = dinflow_du

    jac[1, 0] = beta_eff * v_star
    jac[1, 1] = beta_eff * i_star - gamma_eff - MU_C
    jac[1, 4] = dbeta_dtau * i_star * v_star - dgamma_dtau * v_star

    jac[2, 1] = gamma_eff
    jac[2, 2] = -MU_C
    jac[2, 4] = dgamma_dtau * v_star

    # S row
    jac[3, 0] = DELTA
    jac[3, 3] = -MU_C

    jac[4, 1] = PHI
    jac[4, 4] = -PSI

    jac[5, 4] = -LAMBDA_U * W * u_star
    jac[5, 5] = -LAMBDA_U * (1.0 + W * tau_star)
    return jac


def main(dataset_path=None) -> None:
    resolved_dataset = ensure_dataset(dataset_path=dataset_path)
    rt_timestamps, *_ = parse_activity_file(resolved_dataset)
    cal = build_hourly_curve(rt_timestamps)
    window_counts = cal['rt_window']
    empirical_norm = window_counts / np.max(window_counts)
    fit_window_hours = min(FIT_WINDOW_HOURS, len(empirical_norm))
    beta0, gamma0, _lam0, _lam_decay, _v0, _sse, _fit = fit_basic_ivf(
        empirical_norm[:fit_window_hours],
        window_counts[:fit_window_hours],
    )

    print('Equilibrium checks for the IVFS model (Higgs-calibrated beta0/gamma0, scenario-driven tau block)')
    print(f'Calibrated beta0={beta0:.6f}, gamma0={gamma0:.6f}')
    print()

    summary_rows: list[dict[str, float | bool]] = []
    for alpha in (0.2, 0.5, 0.9):
        i_dfe, u_dfe, r0 = dfe(alpha, beta0, gamma0)
        dfe_state = np.array([i_dfe, 0.0, 0.0, 0.0, 0.0, u_dfe], dtype=float)
        dfe_max = float(np.max(eigvals(jacobian(dfe_state, alpha, beta0, gamma0)).real))
        eq_state = positive_equilibrium(alpha, beta0, gamma0)

        print(f'alpha={alpha:.1f}')
        print(f'  R0={r0:.6f}')
        print(f'  max Re(lambda) at DFE = {dfe_max:.6f}')
        if eq_state is None:
            print('  no positive equilibrium found')
            summary_rows.append({'alpha': alpha, 'r0': r0, 'dfe_max_real_part': dfe_max, 'has_positive_equilibrium': False})
        else:
            eq_max = float(np.max(eigvals(jacobian(eq_state, alpha, beta0, gamma0)).real))
            print(f'  tau*={eq_state[4]:.6f}, V*={eq_state[1]:.6f}, U*={eq_state[5]:.6f}')
            print(f'  max Re(lambda) at positive equilibrium = {eq_max:.6f}')
            summary_rows.append({
                'alpha': alpha,
                'r0': r0,
                'dfe_max_real_part': dfe_max,
                'has_positive_equilibrium': True,
                'tau_star': float(eq_state[4]),
                'V_star': float(eq_state[1]),
                'U_star': float(eq_state[5]),
                'positive_eq_max_real_part': eq_max,
            })
        print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run equilibrium and stability checks for the IVFS model.')
    parser.add_argument('--dataset-path', type=str, default=None)
    args = parser.parse_args()
    main(dataset_path=args.dataset_path)
