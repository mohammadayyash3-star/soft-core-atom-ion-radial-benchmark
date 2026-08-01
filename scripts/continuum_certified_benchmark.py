from __future__ import annotations

from pathlib import Path
import csv
import json
import numpy as np
from scipy.optimize import curve_fit

from unified_model_params import PARAMS
from fdm_fourth_order_core import (
    GridSpec,
    build_kinetic_banded,
    solve_banded_levels_MHz,
    solve_sparse_levels_MHz,
)

OUTDIR = Path(__file__).resolve().parent
GRID_N = (1800, 2556, 3630, 5155)
N_LEVELS = 3
RMAX_M = 650.0e-9


def fit_level(h_nm: np.ndarray, energies: np.ndarray) -> dict[str, float]:
    def model(h, e_inf, amp, power):
        return e_inf + amp * h**power

    best = None
    for p0 in (3.5, 4.0, 4.5):
        popt, pcov = curve_fit(
            model,
            h_nm,
            energies,
            p0=(float(energies[-1]), float(energies[0] - energies[-1]), p0),
            bounds=([-np.inf, -np.inf, 2.0], [np.inf, np.inf, 6.0]),
            maxfev=200000,
        )
        fitted = model(h_nm, *popt)
        residuals = energies - fitted
        sse = float(np.sum(residuals**2))
        if best is None or sse < best[0]:
            best = (sse, popt, pcov, fitted, residuals)

    assert best is not None
    sse, popt, pcov, fitted, residuals = best
    sst = float(np.sum((energies - np.mean(energies)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0.0 else 1.0
    return {
        "E_infinity_MHz": float(popt[0]),
        "amplitude": float(popt[1]),
        "observed_order": float(popt[2]),
        "R2": float(r2),
        "max_abs_fit_residual_kHz": float(np.max(np.abs(residuals)) * 1.0e3),
        "fit_standard_error_kHz": float(np.sqrt(max(0.0, pcov[0, 0])) * 1.0e3),
    }


def main() -> None:
    rows = []
    for N in GRID_N:
        op = build_kinetic_banded(GridSpec(N=N, r_max_m=RMAX_M))
        energies = solve_banded_levels_MHz(
            op,
            rc_m=PARAMS.r_c,
            n_levels=N_LEVELS,
            ell=PARAMS.l,
        )
        row = {
            "N": int(N),
            "delta_r_nm": float(op.dr_m * 1.0e9),
        }
        for n, energy in enumerate(energies):
            row[f"E{n}_MHz"] = float(energy)
        rows.append(row)

    h = np.asarray([row["delta_r_nm"] for row in rows], dtype=float)
    fits = []
    for n in range(N_LEVELS):
        y = np.asarray([row[f"E{n}_MHz"] for row in rows], dtype=float)
        fit = fit_level(h, y)
        fit["state"] = n
        fits.append(fit)

    # Backend agreement at the finest grid.
    op_finest = build_kinetic_banded(GridSpec(N=GRID_N[-1], r_max_m=RMAX_M))
    banded = solve_banded_levels_MHz(
        op_finest, rc_m=PARAMS.r_c, n_levels=N_LEVELS, ell=PARAMS.l
    )
    sparse = solve_sparse_levels_MHz(
        op_finest, rc_m=PARAMS.r_c, n_levels=N_LEVELS, ell=PARAMS.l
    )
    backend_delta_kHz = (sparse - banded) * 1.0e3

    manifest = {
        "scheme": "fourth-order five-point with explicit Dirichlet boundaries and odd-reflection ghosts",
        "rc_nm": float(PARAMS.r_c * 1.0e9),
        "omega_over_2pi_MHz": float(PARAMS.omega_ion / (2.0 * np.pi * 1.0e6)),
        "ell": int(PARAMS.l),
        "r_min_nm": 0.0,
        "r_max_nm": RMAX_M * 1.0e9,
        "grid_N": list(GRID_N),
        "grid_rows": rows,
        "fits": fits,
        "continuum_levels_MHz": [fit["E_infinity_MHz"] for fit in fits],
        "sparse_minus_banded_kHz": backend_delta_kHz.tolist(),
    }

    (OUTDIR / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    with (OUTDIR / "continuum_grid_sequence.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (OUTDIR / "continuum_fit_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fits[0].keys()))
        writer.writeheader()
        writer.writerows(fits)

    print("=== Continuum-certified benchmark ===")
    print(f"r_c = {PARAMS.r_c * 1e9:.9f} nm")
    for row in rows:
        print(
            f"N={row['N']:4d}, dr={row['delta_r_nm']:.9f} nm, "
            + ", ".join(f"E{n}={row[f'E{n}_MHz']:+.12f} MHz" for n in range(N_LEVELS))
        )
    print("Continuum fits:")
    for fit in fits:
        print(
            f"n={fit['state']}: E_inf={fit['E_infinity_MHz']:+.12f} MHz, "
            f"p={fit['observed_order']:.6f}, R2={fit['R2']:.10f}, "
            f"max residual={fit['max_abs_fit_residual_kHz']:.6e} kHz"
        )
    print("Sparse-banded differences (kHz):", backend_delta_kHz)
    print("Saved benchmark_manifest.json and CSV summaries.")


if __name__ == "__main__":
    main()
