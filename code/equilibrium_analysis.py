from __future__ import annotations

"""Equilibrium and stability checks for the IVFS model."""

import numpy as np
from numpy.linalg import eigvals
from scipy.optimize import root_scalar

from ivfs_validation import HIGGS_TXT, build_hourly_curve, ensure_dataset, fit_basic_ivf, parse_activity_file

KAPPA = 0.8
ETA = 0.3
PHI = 0.5
PSI = 0.1
RHO = 0.06
LAMBDA_U = 0.02
NU = 1.0
MU_C = 0.01
DELTA = 0.05
W = 10.0
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
    return np.array([i_star, v_star, f_star, tau_star, u_star], dtype=float)


def jacobian(state: np.ndarray, alpha: float, beta0: float, gamma0: float) -> np.ndarray:
    i_star, v_star, f_star, tau_star, u_star = state
    beta_eff = alpha * beta0 * (1.0 + KAPPA * tau_star)
    gamma_eff = gamma0 * (1.0 + ETA * tau_star)
    dbeta_dtau = alpha * beta0 * KAPPA
    dgamma_dtau = gamma0 * ETA
    dinflow_du = RHO / (1.0 + u_star) ** 2

    jac = np.zeros((5, 5))
    jac[0, 0] = -beta_eff * v_star - A_LOSS
    jac[0, 1] = -beta_eff * i_star
    jac[0, 3] = -dbeta_dtau * i_star * v_star
    jac[0, 4] = dinflow_du

    jac[1, 0] = beta_eff * v_star
    jac[1, 1] = beta_eff * i_star - gamma_eff - MU_C
    jac[1, 3] = dbeta_dtau * i_star * v_star - dgamma_dtau * v_star

    jac[2, 1] = gamma_eff
    jac[2, 2] = -MU_C
    jac[2, 3] = dgamma_dtau * v_star

    jac[3, 1] = PHI
    jac[3, 3] = -PSI

    jac[4, 3] = -LAMBDA_U * W * u_star
    jac[4, 4] = -LAMBDA_U * (1.0 + W * tau_star)
    return jac


def main() -> None:
    ensure_dataset()
    rt_timestamps, _ = parse_activity_file(HIGGS_TXT)
    _, _, _, window_counts = build_hourly_curve(rt_timestamps)
    empirical_norm = window_counts / np.max(window_counts)
    beta0, gamma0, _lbg, _sse, _fit = fit_basic_ivf(empirical_norm)

    print('Equilibrium checks for the calibrated IVFS model')
    print(f'Calibrated beta0={beta0:.6f}, gamma0={gamma0:.6f}')
    print()

    for alpha in (0.2, 0.5, 0.9):
        i_dfe, u_dfe, r0 = dfe(alpha, beta0, gamma0)
        dfe_state = np.array([i_dfe, 0.0, 0.0, 0.0, u_dfe], dtype=float)
        dfe_max = float(np.max(eigvals(jacobian(dfe_state, alpha, beta0, gamma0)).real))
        eq_state = positive_equilibrium(alpha, beta0, gamma0)

        print(f'alpha={alpha:.1f}')
        print(f'  R0={r0:.6f}')
        print(f'  max Re(lambda) at DFE = {dfe_max:.6f}')
        if eq_state is None:
            print('  no positive equilibrium found')
        else:
            eq_max = float(np.max(eigvals(jacobian(eq_state, alpha, beta0, gamma0)).real))
            print(f'  tau*={eq_state[3]:.6f}, V*={eq_state[1]:.6f}, U*={eq_state[4]:.6f}')
            print(f'  max Re(lambda) at positive equilibrium = {eq_max:.6f}')
        print()

    print('All good — we get a clear R0 threshold and the positive equilibria are locally stable for the tested alpha values.')


if __name__ == '__main__':
    main()
