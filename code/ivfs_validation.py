from __future__ import annotations

"""Main script -- run this and it does the whole calibration + scenarios + figures"""

import argparse

import numpy as np

from .common import add_dataset_cli_arguments
from .ivfs_calibration import (FIT_WINDOW_HOURS, bootstrap_calibration, bootstrap_curve_band,
                               build_hourly_curve, ensure_dataset, fit_basic_ivf,
                               parse_activity_file, profile_likelihood)
from .ivfs_config import (DELTA, DIAG_FIGURE_PATH, FIGURE_PATH, LAMBDA_U, MU_C,
                          NU, PHI, PHI_SENS_FIGURE_PATH, PSI, RHO,
                          ROBUST_BIN_HOURS, ROBUST_WINDOW_HOURS, SPIKE_WINDOW_HOURS,
                          TAIL_START_HOURS, TAU_COMPARE_FIGURE_PATH)
from .ivfs_dynamics import (build_tau_configurations, find_W_for_interior_Emax, run_continuation,
                            run_phi_sensitivity, run_scenarios, run_sensitivity)
from .ivfs_figures import (make_calibration_figure, make_phi_sensitivity_figure, make_results_figure,
                           make_tau_comparison_figure, plot_profile)


def main(dataset_path: str | None = None,
        ) -> None:
    print('Running calibration + scenarios + figures')

    resolved_dataset = ensure_dataset(dataset_path=dataset_path)
    rt_timestamps, re_timestamps, mt_timestamps, total_rows = parse_activity_file(resolved_dataset)
    cal = build_hourly_curve(rt_timestamps, re_timestamps, mt_timestamps)
    window_counts = cal['rt_window']
    re_window = cal.get('re_window')
    mt_window = cal.get('mt_window')
    tau_proxy = cal.get('tau_proxy')
    re_proxy = cal.get('re_proxy')
    ratio_proxy = cal.get('reply_ratio')
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
    fit_band_low, fit_band_high = bootstrap_curve_band(
        fit_target,
        fit_counts,
        n_steps=len(empirical_norm),
        B=120,
    )

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
        from .toxicity_calibration import get_external_tau_reference
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

    tau_proxy_candidates = {}
    if re_proxy is not None:
        tau_proxy_candidates['RE proxy'] = np.asarray(re_proxy, dtype=float)
    if ratio_proxy is not None:
        tau_proxy_candidates['RE/(RT+1) proxy'] = np.asarray(ratio_proxy, dtype=float)
    if tau_proxy is not None:
        tau_proxy_candidates['composite proxy'] = np.asarray(tau_proxy, dtype=float)

    tau_configs = build_tau_configurations(
        beta0,
        gamma0,
        fitted_norm,
        tau_proxy_candidates,
        re_window,
        mt_window,
        external_phi=external_phi,
        external_source=external_source,
    )

    selected_proxy_name = 'selected same-dataset proxy'
    selected_proxy = tau_proxy
    chosen_proxy_name = next(
        (
            str(cfg.get('proxy_name'))
            for cfg in tau_configs.values()
            if cfg.get('proxy_name') in tau_proxy_candidates
        ),
        None,
    )
    if chosen_proxy_name is not None:
        selected_proxy_name = chosen_proxy_name
        selected_proxy = tau_proxy_candidates[selected_proxy_name]

    phi_candidates = [PHI]
    if external_phi is not None and np.isfinite(external_phi):
        phi_candidates.append(external_phi)
    phi_min = max(0.02, 0.5 * min(phi_candidates))
    phi_max = min(0.25, 1.2 * max(phi_candidates))
    phi_grid = np.linspace(phi_min, phi_max, 30)
    phi_results = run_phi_sensitivity(beta0, gamma0, phi_grid)
    make_phi_sensitivity_figure(phi_grid, phi_results, PHI, external_phi, external_source)

    make_calibration_figure(empirical_norm, fitted_norm, window_counts,
                            re_window, mt_window, tau_proxy,
                            re_proxy, ratio_proxy,
                            selected_proxy, selected_proxy_name,
                            fit_window_hours,
                            weighted_loss, r2_fit, r2_full, r2_spike, tail_bias,
                            fit_band=(fit_band_low, fit_band_high))
    if tau_configs and selected_proxy is not None:
        make_tau_comparison_figure(selected_proxy, selected_proxy_name, re_window, tau_configs)
    make_results_figure(t_scenario, scenario_results,
                        alphas, tau_star, v_star_cont, u_star_cont, alpha_r0)

    a_loss = DELTA + MU_C
    u_dfe = NU / LAMBDA_U
    i_dfe = RHO * u_dfe / ((1.0 + u_dfe) * a_loss)

    print(f'Total activity rows: {total_rows:,}')
    print(f'RT events: {len(rt_timestamps):,}')
    print(f'RE events: {len(re_timestamps):,}')
    print(f'MT events: {len(mt_timestamps):,}')
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
    print(f'R² fit window (0–{fit_window_hours - 1}h): {r2_fit:.4f}')
    print(f'R² full window (0–{len(empirical_norm)-1}h): {r2_full:.4f}')
    print(f'R² spike window (0–{spike_end - 1}h): {r2_spike:.4f}')
    print(f'Tail mean bias (h {tail_start}–{len(empirical_norm)-1}): {tail_bias:+.4f}')
    if re_window is not None:
        re_rt_corr = float(np.corrcoef(cal['re_window'], window_counts)[0, 1])
        print(f'RE-RT correlation in window: {re_rt_corr:.4f}')
    if mt_window is not None:
        mt_rt_corr = float(np.corrcoef(mt_window, window_counts)[0, 1])
        print(f'MT-RT correlation in window: {mt_rt_corr:.4f}')

    print()
    print('Baseline scenario long-run values (integrated to t=2000; current fixed tau setup):')
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
    if tau_configs:
        print(f'Saved tau comparison figure: {TAU_COMPARE_FIGURE_PATH}')

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

    if tau_configs:
        print()
        chosen_proxy = next((cfg.get('proxy_name') for cfg in tau_configs.values() if cfg.get('proxy_name')), 'same-dataset proxy')
        print(f'Latent-pressure proxy comparison (fit target: {chosen_proxy}):')
        print(f'{"key":<10} {"phi":<8} {"psi":<8} {"proxy_loss":<12} {"proxy_R2":<10} {"tau*_0.2":<10} {"tau*_0.5":<10} {"tau*_0.9":<10} {"ordered":<8} {"subcrit":<8}')
        for key, cfg in tau_configs.items():
            cfg_scen = cfg['scenario_results']
            tau_hf = max(0.0, float(cfg_scen['Health-First (alpha=0.2)']['tau_star']))
            tau_mod = max(0.0, float(cfg_scen['Moderate (alpha=0.5)']['tau_star']))
            tau_eng = max(0.0, float(cfg_scen['Engagement-First (alpha=0.9)']['tau_star']))
            ordered = (tau_eng >= tau_mod - 1e-10) and (tau_mod >= tau_hf - 1e-10)
            subcrit = max(0.0, float(cfg_scen['Health-First (alpha=0.2)']['V_star'])) < 1e-6
            print(
                f"{key:<10} {cfg['phi']:<8.4f} {cfg['psi']:<8.4f} "
                f"{cfg['proxy_loss']:<12.4f} {cfg['proxy_r2']:<10.4f} "
                f"{tau_hf:<10.4f} {tau_mod:<10.4f} {tau_eng:<10.4f} "
                f"{str(ordered):<8} {str(subcrit):<8}"
            )
            print(f"{'':<10} label={cfg['label']}")
            if 'fit_diag' in cfg:
                print(
                    f"{'':<10} decay-anchored psi={cfg['fit_diag']['psi_decay']:.4f}, "
                    f"decay R2={cfg['fit_diag']['psi_decay_r2']:.4f}, "
                    f"points={int(cfg['fit_diag']['decay_points'])}, "
                    f"anchored-fit R2={cfg['fit_diag']['r2_decay_fit']:.4f}"
                )
                print(
                    f"{'':<10} unconstrained phi={cfg['fit_diag']['phi_unconstrained']:.4f}, "
                    f"psi={cfg['fit_diag']['psi_unconstrained']:.4f}, "
                    f"unconstrained R2={cfg['fit_diag']['r2_unconstrained']:.4f}"
                )

    print()
    print('Sensitivity (alpha=0.5, central finite differences +/-1%):')
    print(f'{"param":<8} {"S_V":<10} {"S_tau":<10} {"S_U":<10}')
    v_base, tau_base, u_base, sens_rows = run_sensitivity(beta0, gamma0)
    print(f'{"(base)":<8} {v_base:<10.4f} {tau_base:<10.4f} {u_base:<10.4f}')
    for name, s_v, s_tau, s_u in sens_rows:
        print(f'{name:<8} {s_v:<+10.3f} {s_tau:<+10.3f} {s_u:<+10.3f}')

    print()
    w_opt, alpha_peak = find_W_for_interior_Emax(beta0, gamma0)
    if w_opt is not None:
        print(f'interior E* peak found: W={w_opt:.1f} gives peak at alpha={alpha_peak:.2f}')
    else:
        print('no interior E* peak found for W <= 80; E* peaks at alpha=1.0')

    print()
    print('Computing profile likelihood...')
    bg, pb, gg, pg = profile_likelihood(
        fit_target,
        fit_counts,
        np.array([beta0, gamma0, lambda0, lambda_decay, v0_fit], dtype=float),
    )
    plot_profile(bg, pb, gg, pg, beta0, gamma0)

    print()
    print('Running parametric bootstrap (B=500)...')
    ci = bootstrap_calibration(fit_target, fit_counts, B=500)
    print(f"beta0 95% CI: [{ci['beta0'][0]:.4f}, {ci['beta0'][1]:.4f}]")
    print(f"gamma0 95% CI: [{ci['gamma0'][0]:.4f}, {ci['gamma0'][1]:.4f}]")
    print(f"R0=1 threshold alpha 95% CI: [{ci['alpha_r0'][0]:.4f}, {ci['alpha_r0'][1]:.4f}]")

    robustness_rows = []
    print()
    print('Robustness checks across alternate windows and bin sizes:')
    for window_hours in ROBUST_WINDOW_HOURS:
        for bin_hours in ROBUST_BIN_HOURS:
            robust_cal = build_hourly_curve(
                rt_timestamps,
                re_timestamps,
                mt_timestamps,
                window_hours=window_hours,
                bin_hours=bin_hours,
            )
            robust_counts = np.asarray(robust_cal['rt_window'], dtype=float)
            robust_norm = robust_counts / np.max(robust_counts)
            robust_fit_hours = min(FIT_WINDOW_HOURS // bin_hours, len(robust_norm))
            robust_beta, robust_gamma, _l0, _ld, _v0, robust_loss, robust_fit = fit_basic_ivf(
                robust_norm[:robust_fit_hours],
                robust_counts[:robust_fit_hours],
                n_steps=len(robust_norm),
            )
            robust_sse = float(np.sum((robust_fit[:robust_fit_hours] - robust_norm[:robust_fit_hours]) ** 2))
            robust_ss_tot = float(np.sum((robust_norm[:robust_fit_hours] - np.mean(robust_norm[:robust_fit_hours])) ** 2))
            robust_r2 = 1.0 - robust_sse / robust_ss_tot if robust_ss_tot > 0 else 0.0
            robustness_rows.append({
                'window_hours': int(window_hours),
                'bin_hours': int(bin_hours),
                'beta0': float(robust_beta),
                'gamma0': float(robust_gamma),
                'loss': float(robust_loss),
                'r2': float(robust_r2),
            })
            print(
                f"  window={window_hours:>3}h, bin={bin_hours}h: "
                f"beta0={robust_beta:.4f}, gamma0={robust_gamma:.4f}, R2={robust_r2:.4f}"
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the full IVFS calibration, robustness, and figure pipeline.')
    add_dataset_cli_arguments(parser)
    args = parser.parse_args()
    main(dataset_path=args.dataset_path)
