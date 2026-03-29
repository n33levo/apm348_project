"""Generate 2-D parameter-space heatmaps for the IVFS model.

Producing:
  assets/alpha_kappa_heatmap.png   -- long-run tau* over (alpha, kappa) grid
  assets/phase_portrait.png        -- V-tau phase portrait for the three scenarios
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import ASSETS_DIR, build_run_metadata, ensure_layout, solve_trajectory, write_json
from .ivfs_config import (
    DELTA, ETA, FITTED_BETA0, FITTED_GAMMA0, KAPPA, LAMBDA_U, MU_C, NU,
    PHI, PSI, RHO, SOLVER_ATOL, SOLVER_MAX_STEP, SOLVER_METHOD, SOLVER_RTOL, W,
)

BETA0 = FITTED_BETA0
GAMMA0 = FITTED_GAMMA0
METADATA_PATH = ASSETS_DIR / 'heatmap_metadata.json'


# Helpers

def ivfs_rhs(t, y, alpha, kappa_val, eta_val=ETA, phi_val=PHI, psi_val=PSI):
    I, V, F, S, tau, U = y
    beta_eff = alpha * BETA0 * (1 + kappa_val * tau)
    gamma_eff = GAMMA0 * (1 + eta_val * tau)
    dI = RHO * U / (1 + U) - beta_eff * I * V - (DELTA + MU_C) * I
    dV = beta_eff * I * V - gamma_eff * V - MU_C * V
    dF = gamma_eff * V - MU_C * F
    dS = DELTA * I - MU_C * S
    dtau = phi_val * V - psi_val * tau
    dU = NU - LAMBDA_U * (1 + W * tau) * U
    return [dI, dV, dF, dS, dtau, dU]


def steady_state(alpha, kappa_val, T=2000):
    U0 = NU / LAMBDA_U
    I0 = RHO * U0 / ((1 + U0) * (DELTA + MU_C))
    y0 = [I0, 0.01, 0.0, 0.0, 0.0, U0]
    t_eval = np.linspace(0.0, float(T), 4001)
    sol = solve_trajectory(
        ivfs_rhs,
        y0,
        t_eval,
        args=(alpha, kappa_val),
        method=SOLVER_METHOD,
        rtol=SOLVER_RTOL,
        atol=SOLVER_ATOL,
        max_step=SOLVER_MAX_STEP,
    )
    return sol[-1]  # I, V, F, S, tau, U at t=T


# Alpha-kappa heatmap of tau*

def make_heatmap():
    ensure_layout()
    alphas = np.linspace(0.05, 1.0, 60)
    kappas = np.linspace(0.0, 2.0, 60)
    tau_grid = np.zeros((len(kappas), len(alphas)))
    R0_grid = np.zeros_like(tau_grid)

    U_dfe = NU / LAMBDA_U
    I_dfe = RHO * U_dfe / ((1 + U_dfe) * (DELTA + MU_C))

    for i, kap in enumerate(kappas):
        for j, alp in enumerate(alphas):
            ss = steady_state(alp, kap)
            tau_grid[i, j] = ss[4]
            R0_grid[i, j] = alp * BETA0 * I_dfe / (GAMMA0 + MU_C)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("white")

    # tau* heatmap
    ax = axes[0]
    im = ax.pcolormesh(alphas, kappas, tau_grid, shading="auto",
                       cmap="YlOrRd")
    cb = fig.colorbar(im, ax=ax, label=r"Long-run $\tau^*$")
    # R0=1 contour
    ax.contour(alphas, kappas, R0_grid, levels=[1.0],
               colors="white", linewidths=2, linestyles="--")
    ax.set_xlabel(r"Amplification $\alpha$", fontsize=12)
    ax.set_ylabel(r"Pressure coupling $\kappa$", fontsize=12)
    ax.set_title(r"(a) Equilibrium pressure $\tau^*(\alpha,\kappa)$",
                 fontsize=13, fontweight="bold")
    # mark the three scenarios
    for alp, lab in [(0.2, "H"), (0.5, "M"), (0.9, "E")]:
        ax.axvline(alp, color="white", lw=0.8, ls=":")
        ax.text(alp, kappas[-1]*0.97, lab, color="white", fontsize=9,
                ha="center", va="top", fontweight="bold")
    ax.axhline(KAPPA, color="cyan", lw=0.8, ls=":")
    ax.text(alphas[-1]*0.98, KAPPA+0.04, f"baseline $\\kappa$={KAPPA}",
            color="cyan", fontsize=8, ha="right")

    # V* heatmap
    V_grid = np.zeros_like(tau_grid)
    for i, kap in enumerate(kappas):
        for j, alp in enumerate(alphas):
            ss = steady_state(alp, kap)
            V_grid[i, j] = ss[1]

    ax2 = axes[1]
    im2 = ax2.pcolormesh(alphas, kappas, V_grid, shading="auto",
                         cmap="YlGnBu")
    fig.colorbar(im2, ax=ax2, label=r"Long-run $V^*$")
    ax2.contour(alphas, kappas, R0_grid, levels=[1.0],
                colors="white", linewidths=2, linestyles="--")
    ax2.set_xlabel(r"Amplification $\alpha$", fontsize=12)
    ax2.set_ylabel(r"Pressure coupling $\kappa$", fontsize=12)
    ax2.set_title(r"(b) Equilibrium viral volume $V^*(\alpha,\kappa)$",
                  fontsize=13, fontweight="bold")
    for alp, lab in [(0.2, "H"), (0.5, "M"), (0.9, "E")]:
        ax2.axvline(alp, color="white", lw=0.8, ls=":")
        ax2.text(alp, kappas[-1]*0.97, lab, color="white", fontsize=9,
                 ha="center", va="top", fontweight="bold")

    fig.tight_layout(pad=1.2)
    out = ASSETS_DIR / "alpha_kappa_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


# Phase portrait V vs tau

def make_phase_portrait():
    ensure_layout()
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")

    colours = {"Engagement-first ($\\alpha=0.9$)": "#E11D48",
               "Moderate ($\\alpha=0.5$)": "#C2410C",
               "Health-first ($\\alpha=0.2$)": "#2563EB"}
    alphas = [0.9, 0.5, 0.2]

    U0 = NU / LAMBDA_U
    I0 = RHO * U0 / ((1 + U0) * (DELTA + MU_C))
    y0 = [I0, 0.01, 0.0, 0.0, 0.0, U0]

    for (lbl, clr), alp in zip(colours.items(), alphas):
        t_eval = np.linspace(0.0, 2000.0, 4001)
        sol = solve_trajectory(
            ivfs_rhs,
            y0,
            t_eval,
            args=(alp, KAPPA),
            method=SOLVER_METHOD,
            rtol=SOLVER_RTOL,
            atol=SOLVER_ATOL,
            max_step=0.5,
        )
        V = sol[:, 1]
        tau = sol[:, 4]
        ax.plot(V, tau, color=clr, lw=1.8, label=lbl, zorder=3)
        ax.plot(V[0], tau[0], "o", color=clr, ms=6, zorder=4)
        ax.plot(V[-1], tau[-1], "s", color=clr, ms=7, zorder=4)

    ax.set_xlabel(r"Viral volume $V$", fontsize=12)
    ax.set_ylabel(r"Discussion pressure $\tau$", fontsize=12)
    ax.set_title(r"Phase portrait in the $(V,\tau)$ plane", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    out = ASSETS_DIR / "phase_portrait.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    print("computing alpha-kappa heatmap (this may take a minute)...")
    make_heatmap()
    print("computing phase portrait...")
    make_phase_portrait()
    write_json(
        METADATA_PATH,
        build_run_metadata(
            script_name='heatmap_figures',
            dataset_path=ASSETS_DIR,
            parameters={
                'beta0': float(BETA0),
                'gamma0': float(GAMMA0),
                'kappa_baseline': float(KAPPA),
            },
            solver={
                'method': SOLVER_METHOD,
                'rtol': SOLVER_RTOL,
                'atol': SOLVER_ATOL,
                'max_step': SOLVER_MAX_STEP,
            },
            outputs={
                'alpha_kappa_heatmap': str(ASSETS_DIR / 'alpha_kappa_heatmap.png'),
                'phase_portrait': str(ASSETS_DIR / 'phase_portrait.png'),
            },
            notes={'dataset': 'No external data are required for these deterministic phase diagrams.'},
        ),
    )
    print("done.")
