from __future__ import annotations


from pathlib import Path
from typing import Any, Iterable
import csv
import json
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from unified_model_params import PARAMS, PI
from fdm_fourth_order_core import (
    GridSpec,
    build_kinetic_banded,
    solve_banded_levels_MHz,
)


# ============================================================
# Final shared numerical convention
# ============================================================
OUTDIR = Path(__file__).resolve().parent
GRID_N = 5155
R_MAX_M = 650.0e-9
N_REQUESTED_LEVELS = 8
N_REPORTED_LEVELS = 3

TARGET_MHZ = PARAMS.target_Hz / 1.0e6
EXPERIMENTAL_BAND_MHZ = (13.0, 17.0)
RC_REFERENCE_NM = PARAMS.r_c * 1.0e9
OMEGA_OVER_2PI_MHZ = PARAMS.omega_ion / (2.0 * PI * 1.0e6)

# Representative values printed in the supplement table.
TABLE_RC_NM = (
    24.000000,
    24.500000,
    25.000000,
    25.166667,
    25.500000,
    RC_REFERENCE_NM,
    26.166667,
    26.500000,
    26.666667,
    26.833333,
    27.000000,
    28.000000,
)

PLOT_RC_NM = np.linspace(24.0, 28.0, 41)


# ============================================================
# Utilities
# ============================================================
def unique_sorted(values: Iterable[float], decimals: int = 9) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    arr = np.unique(np.round(arr, decimals=decimals))
    arr.sort()
    return arr


def solve_at_rc(operator, rc_nm: float) -> dict[str, Any]:
    levels = np.asarray(
        solve_banded_levels_MHz(
            operator,
            rc_m=float(rc_nm) * 1.0e-9,
            n_levels=N_REQUESTED_LEVELS,
            omega_rad_s=PARAMS.omega_ion,
            ell=PARAMS.l,
            regulator="V1",
        ),
        dtype=float,
    )

    n_negative = int(np.sum(levels < 0.0))
    if n_negative == N_REQUESTED_LEVELS:
        raise RuntimeError(
            "All requested levels are negative. Increase N_REQUESTED_LEVELS "
            "before using this sweep over a deeper parameter range."
        )
    if levels.size < N_REPORTED_LEVELS:
        raise RuntimeError("Too few eigenvalues were returned.")

    e0_abs = abs(float(levels[0]))
    return {
        "rc_nm": float(rc_nm),
        "E0_MHz": float(levels[0]),
        "E1_MHz": float(levels[1]),
        "E2_MHz": float(levels[2]),
        "E0_abs_MHz": e0_abs,
        "n_negative": n_negative,
        "delta15_MHz": e0_abs - TARGET_MHZ,
        "delta15_abs_MHz": abs(e0_abs - TARGET_MHZ),
        "delta15_pct": 100.0 * abs(e0_abs - TARGET_MHZ) / TARGET_MHZ,
        "inside_13_17_MHz_band": bool(
            EXPERIMENTAL_BAND_MHZ[0]
            <= e0_abs
            <= EXPERIMENTAL_BAND_MHZ[1]
        ),
        "is_reference_point": bool(
            math.isclose(float(rc_nm), RC_REFERENCE_NM, rel_tol=0.0, abs_tol=5.0e-10)
        ),
    }


def run_sweep() -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    operator = build_kinetic_banded(GridSpec(N=GRID_N, r_max_m=R_MAX_M))

    rc_plot = unique_sorted(
        list(PLOT_RC_NM) + list(TABLE_RC_NM) + [RC_REFERENCE_NM]
    )
    all_rows = [solve_at_rc(operator, float(rc_nm)) for rc_nm in rc_plot]

    row_by_key = {round(float(row["rc_nm"]), 9): row for row in all_rows}
    table_rows = [
        row_by_key[round(float(rc_nm), 9)]
        for rc_nm in unique_sorted(TABLE_RC_NM)
    ]

    return all_rows, table_rows, float(operator.dr_m * 1.0e9)


# ============================================================
# Validation
# ============================================================
def validate_reference_point(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reference = min(
        rows,
        key=lambda row: abs(float(row["rc_nm"]) - RC_REFERENCE_NM),
    )
    if abs(float(reference["rc_nm"]) - RC_REFERENCE_NM) > 5.0e-9:
        raise RuntimeError("The exact calibrated reference point is missing.")

    # Fixed-grid values expected from the authoritative N=5155 sequence.
    expected = np.array(
        [-14.999999970076715, -9.211774917564070, -3.530419912528486],
        dtype=float,
    )
    obtained = np.array(
        [reference["E0_MHz"], reference["E1_MHz"], reference["E2_MHz"]],
        dtype=float,
    )
    difference_Hz = (obtained - expected) * 1.0e6
    tolerance_Hz = 0.02
    if np.max(np.abs(difference_Hz)) > tolerance_Hz:
        raise RuntimeError(
            "Reference-point self-check failed: fixed-grid values differ from "
            f"the certified sequence by {difference_Hz} Hz."
        )

    return {
        "reference_row": reference,
        "expected_fixed_grid_levels_MHz": expected.tolist(),
        "obtained_minus_expected_Hz": difference_Hz.tolist(),
        "self_check_tolerance_Hz": tolerance_Hz,
        "self_check_pass": True,
    }


# ============================================================
# Output writers
# ============================================================
def write_csv(rows: list[dict[str, Any]]) -> Path:
    path = OUTDIR / "rc_sensitivity_fdm_table.csv"
    fields = [
        "rc_nm",
        "E0_MHz",
        "E1_MHz",
        "E2_MHz",
        "E0_abs_MHz",
        "n_negative",
        "delta15_MHz",
        "delta15_abs_MHz",
        "delta15_pct",
        "inside_13_17_MHz_band",
        "is_reference_point",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    return path


def write_detailed_latex(rows: list[dict[str, Any]]) -> Path:
    path = OUTDIR / "rc_sensitivity_fdm_table.tex"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            r"""\begin{table}[t]
\centering
\caption{Local sensitivity of the low-lying finite-grid spectrum to the soft-core radius. Only $r_c$ is varied. The common numerical representation uses $N=5155$ on $0\le r\le650~\mathrm{nm}$ with explicit Dirichlet boundaries and odd-reflection closure. This sweep is a diagnostic and not a recalibration procedure.}
\label{tab:rc_sensitivity_detailed}
\begin{tabular}{c c c c c c}
\hline
$r_c$ (nm) & $E_0/h$ & $E_1/h$ & $E_2/h$ & $N_-$ & $\delta_{15}$ \\
& \multicolumn{3}{c}{(MHz)} & & (\%) \\
\hline
"""
        )
        for row in rows:
            handle.write(
                f"{float(row['rc_nm']):.6f} & "
                f"{float(row['E0_MHz']):.9f} & "
                f"{float(row['E1_MHz']):.9f} & "
                f"{float(row['E2_MHz']):.9f} & "
                f"{int(row['n_negative'])} & "
                f"{float(row['delta15_pct']):.4f} \\\\\n"
            )
        handle.write(
            r"""\hline
\end{tabular}
\end{table}
"""
        )
    return path


def write_compact_latex(rows: list[dict[str, Any]]) -> Path:
    path = OUTDIR / "rc_sensitivity_fdm_slice.tex"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            r"""\begin{table}[t]
\centering
\caption{Representative fixed-frequency slice at $\omega/2\pi=1.2~\mathrm{MHz}$, calculated on the common $N=5155$ grid. The sweep is a sensitivity diagnostic and not a recalibration procedure.}
\label{tab:rc_sensitivity_slice}
\begin{tabular}{c c c c}
\hline
$r_c$ (nm) & $|E_0|/h$ (MHz) & $N_-$ & Inside $13$--$17~\mathrm{MHz}$? \\
\hline
"""
        )
        for row in rows:
            inside = "yes" if bool(row["inside_13_17_MHz_band"]) else "no"
            handle.write(
                f"{float(row['rc_nm']):.6f} & "
                f"{float(row['E0_abs_MHz']):.6f} & "
                f"{int(row['n_negative'])} & {inside} \\\\\n"
            )
        handle.write(
            r"""\hline
\end{tabular}
\end{table}
"""
        )
    return path


def write_summary_json(
    all_rows: list[dict[str, Any]],
    table_rows: list[dict[str, Any]],
    dr_nm: float,
    validation: dict[str, Any],
) -> Path:
    path = OUTDIR / "rc_sensitivity_fdm_summary.json"
    payload = {
        "purpose": "fixed-grid local rc sensitivity; no recalibration",
        "numerical_scheme": (
            "fourth-order five-point FDM with explicit Dirichlet boundaries "
            "and odd-reflection ghost closure"
        ),
        "grid_N": GRID_N,
        "delta_r_nm": dr_nm,
        "r_min_nm": 0.0,
        "r_max_nm": R_MAX_M * 1.0e9,
        "omega_over_2pi_MHz": OMEGA_OVER_2PI_MHZ,
        "ell": int(PARAMS.l),
        "reference_rc_nm": RC_REFERENCE_NM,
        "target_MHz": TARGET_MHZ,
        "experimental_band_MHz": list(EXPERIMENTAL_BAND_MHZ),
        "n_plot_points": len(all_rows),
        "table_rows": table_rows,
        "validation": validation,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ============================================================
# Figure
# ============================================================
def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 170,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 0.85,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def format_axis(ax) -> None:
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", direction="in", top=True, right=True)


def write_figure(rows: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    set_publication_style()

    rc = np.asarray([float(row["rc_nm"]) for row in rows], dtype=float)
    e0 = np.asarray([float(row["E0_MHz"]) for row in rows], dtype=float)
    e1 = np.asarray([float(row["E1_MHz"]) for row in rows], dtype=float)
    e2 = np.asarray([float(row["E2_MHz"]) for row in rows], dtype=float)
    e0_abs = np.asarray([float(row["E0_abs_MHz"]) for row in rows], dtype=float)

    ref_index = int(np.argmin(np.abs(rc - RC_REFERENCE_NM)))

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.25), constrained_layout=True)

    ax = axes[0]
    ax.plot(rc, e0, marker="o", markersize=2.8, markevery=4, linewidth=1.35, label=r"$E_0/h$")
    ax.plot(rc, e1, marker="s", markersize=2.8, markevery=4, linewidth=1.35, label=r"$E_1/h$")
    ax.plot(rc, e2, marker="^", markersize=2.8, markevery=4, linewidth=1.35, label=r"$E_2/h$")
    ax.axhline(0.0, linewidth=0.9, linestyle="--")
    ax.axvline(RC_REFERENCE_NM, linewidth=1.0, linestyle=":")
    ax.scatter(
        [RC_REFERENCE_NM] * 3,
        [e0[ref_index], e1[ref_index], e2[ref_index]],
        marker="*",
        s=58,
        zorder=5,
    )
    ax.set_xlabel(r"Soft-core radius $r_c$ (nm)")
    ax.set_ylabel(r"Energy $E_n/h$ (MHz)")
    ax.set_title(r"(a) Low-lying fixed-grid spectrum")
    ax.legend(frameon=False)
    format_axis(ax)

    ax = axes[1]
    ax.axhspan(
        EXPERIMENTAL_BAND_MHZ[0],
        EXPERIMENTAL_BAND_MHZ[1],
        alpha=0.18,
        linewidth=0.0,
        label=r"Experimental $13$--$17$ MHz band",
    )
    ax.plot(rc, e0_abs, marker="o", markersize=2.8, markevery=4, linewidth=1.35, label=r"$|E_0|/h$")
    ax.axhline(TARGET_MHZ, linewidth=1.0, linestyle="--", label=r"$15$ MHz target")
    ax.axvline(RC_REFERENCE_NM, linewidth=1.0, linestyle=":")
    ax.scatter(
        [RC_REFERENCE_NM],
        [e0_abs[ref_index]],
        marker="*",
        s=72,
        zorder=5,
        label=r"Calibrated point",
    )
    ax.set_xlabel(r"Soft-core radius $r_c$ (nm)")
    ax.set_ylabel(r"Ground-state binding $|E_0|/h$ (MHz)")
    ax.set_title(r"(b) Experimental-anchor sensitivity")
    ax.legend(frameon=False, fontsize=7.4, loc="lower left")
    format_axis(ax)

    pdf_path = OUTDIR / "rc_sensitivity_fdm_figure.pdf"
    png_path = OUTDIR / "rc_sensitivity_fdm_figure.png"
    svg_path = OUTDIR / "rc_sensitivity_fdm_figure.svg"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    fig.savefig(svg_path)
    plt.close(fig)
    return pdf_path, png_path, svg_path


# ============================================================
# Console report
# ============================================================
def print_table(rows: list[dict[str, Any]], dr_nm: float) -> None:
    print("=== Final FDM soft-core-radius sensitivity sweep ===")
    print(f"r_c reference            : {RC_REFERENCE_NM:.9f} nm")
    print(f"omega/2pi                : {OMEGA_OVER_2PI_MHZ:.9f} MHz")
    print(f"ell                      : {PARAMS.l}")
    print(f"radial domain            : [0, {R_MAX_M * 1e9:.1f}] nm")
    print(f"grid                     : N={GRID_N}, dr={dr_nm:.9f} nm")
    print("boundary closure         : explicit Dirichlet + odd reflection")
    print("interpretation           : fixed-grid sensitivity; no recalibration\n")

    header = (
        f"{'r_c [nm]':>12s}  {'E0/h [MHz]':>14s}  {'E1/h [MHz]':>14s}  "
        f"{'E2/h [MHz]':>14s}  {'N_-':>4s}  {'inside 13-17':>12s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{float(row['rc_nm']):12.6f}  "
            f"{float(row['E0_MHz']):14.9f}  "
            f"{float(row['E1_MHz']):14.9f}  "
            f"{float(row['E2_MHz']):14.9f}  "
            f"{int(row['n_negative']):4d}  "
            f"{str(bool(row['inside_13_17_MHz_band'])):>12s}"
        )


def main() -> None:
    all_rows, table_rows, dr_nm = run_sweep()
    validation = validate_reference_point(all_rows)

    print_table(table_rows, dr_nm)
    csv_path = write_csv(all_rows)
    detailed_tex_path = write_detailed_latex(table_rows)
    compact_tex_path = write_compact_latex(table_rows)
    summary_path = write_summary_json(
        all_rows, table_rows, dr_nm, validation
    )
    pdf_path, png_path, svg_path = write_figure(all_rows)

    print("\nReference-point self-check: PASS")
    print("Files written:")
    for path in (
        csv_path,
        detailed_tex_path,
        compact_tex_path,
        summary_path,
        pdf_path,
        png_path,
        svg_path,
    ):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
