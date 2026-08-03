# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Generate the final Numerov verification figure by reading the numerical
results from numerov_crosscheck_final.csv.

Run order
---------
1) python numerov_crosscheck_final.py
2) python numerov_verification_negative_spectrum_from_csv.py

Required CSV columns
--------------------
n
E_FDM_continuum_MHz
E_Numerov_MHz
delta_Numerov_minus_FDM_Hz
nodes
match_r_nm
mismatch_m_inverse
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import patheffects as pe


# ============================================================
# 1. Input and output locations
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "numerov_crosscheck_final.csv"

OUTDIR = BASE_DIR / "numerov_figure_final"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Journal style
# ============================================================

def set_journal_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 170,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,

        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",

        "axes.linewidth": 0.95,
        "axes.labelsize": 10.7,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 8.3,

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4.2,
        "ytick.major.size": 4.2,
        "xtick.minor.size": 2.2,
        "ytick.minor.size": 2.2,

        # Embed TrueType fonts instead of Type 3 fonts.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


set_journal_style()


# ============================================================
# 3. Read and validate the CSV
# ============================================================

REQUIRED_COLUMNS = {
    "n",
    "E_FDM_continuum_MHz",
    "E_Numerov_MHz",
    "delta_Numerov_minus_FDM_Hz",
    "nodes",
    "match_r_nm",
    "mismatch_m_inverse",
}


def load_numerov_results(csv_path: Path) -> list[dict[str, float | int]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file was not found:\n{csv_path}\n\n"
            "Run numerov_crosscheck_final.py first."
        )

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise RuntimeError(f"The CSV file has no header: {csv_path}")

        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise RuntimeError(
                "The CSV file is missing required columns: "
                + ", ".join(sorted(missing))
            )

        rows: list[dict[str, float | int]] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                rows.append({
                    "n": int(row["n"]),
                    "E_FDM_continuum_MHz": float(row["E_FDM_continuum_MHz"]),
                    "E_Numerov_MHz": float(row["E_Numerov_MHz"]),
                    "delta_Numerov_minus_FDM_Hz": float(
                        row["delta_Numerov_minus_FDM_Hz"]
                    ),
                    "nodes": int(row["nodes"]),
                    "match_r_nm": float(row["match_r_nm"]),
                    "mismatch_m_inverse": float(row["mismatch_m_inverse"]),
                })
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Invalid numerical value in CSV line {line_number}."
                ) from exc

    if not rows:
        raise RuntimeError(f"The CSV file contains no data rows: {csv_path}")

    rows.sort(key=lambda item: int(item["n"]))

    expected_states = list(range(len(rows)))
    actual_states = [int(item["n"]) for item in rows]
    if actual_states != expected_states:
        raise RuntimeError(
            f"Unexpected state indices. Expected {expected_states}, "
            f"but found {actual_states}."
        )

    for item in rows:
        n_value = int(item["n"])
        nodes_value = int(item["nodes"])
        if nodes_value != n_value:
            raise RuntimeError(
                f"Node-count check failed for state n={n_value}: "
                f"nodes={nodes_value}."
            )

        # Independent consistency check:
        # recompute Numerov - FDM from the two energies.
        recomputed_delta = 1.0e6 * (
            float(item["E_Numerov_MHz"])
            - float(item["E_FDM_continuum_MHz"])
        )
        csv_delta = float(item["delta_Numerov_minus_FDM_Hz"])

        if not np.isclose(
            recomputed_delta,
            csv_delta,
            rtol=0.0,
            atol=5.0e-7,
        ):
            raise RuntimeError(
                f"Delta consistency check failed for n={n_value}: "
                f"CSV delta={csv_delta:.9f} Hz, "
                f"recomputed delta={recomputed_delta:.9f} Hz."
            )

    return rows


rows = load_numerov_results(CSV_PATH)

n = np.array([int(row["n"]) for row in rows], dtype=int)
E_fdm_MHz = np.array(
    [float(row["E_FDM_continuum_MHz"]) for row in rows],
    dtype=float,
)
E_num_MHz = np.array(
    [float(row["E_Numerov_MHz"]) for row in rows],
    dtype=float,
)
delta_Hz = np.array(
    [float(row["delta_Numerov_minus_FDM_Hz"]) for row in rows],
    dtype=float,
)
nodes = np.array([int(row["nodes"]) for row in rows], dtype=int)
match_r_nm = np.array(
    [float(row["match_r_nm"]) for row in rows],
    dtype=float,
)

mean_abs_delta_Hz = float(np.mean(np.abs(delta_Hz)))
max_abs_delta_Hz = float(np.max(np.abs(delta_Hz)))


# ============================================================
# 4. Shared benchmark metadata
# ============================================================

r_c_nm = 25.876730807
omega_over_2pi_MHz = 1.2
ell = 0

fdm_grid_sequence = (1800, 2556, 3630, 5155)
numerov_N = 24000


# ============================================================
# 5. Save a clean plotting-data CSV
# ============================================================

plot_csv_path = OUTDIR / "numerov_verification_negative_spectrum.csv"

with plot_csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow([
        "n",
        "E_FDM_continuum_MHz",
        "E_Numerov_MHz",
        "Numerov_minus_FDM_Hz",
        "nodes",
        "match_r_nm",
    ])
    for i in range(len(n)):
        writer.writerow([
            int(n[i]),
            f"{E_fdm_MHz[i]:.12f}",
            f"{E_num_MHz[i]:.12f}",
            f"{delta_Hz[i]:.9f}",
            int(nodes[i]),
            f"{match_r_nm[i]:.6f}",
        ])


# ============================================================
# 6. Publication-quality figure
# ============================================================

COL_BAR = "#6c757d"
COL_POINT = "#b23a48"
COL_MEAN = "#9c6b1f"
COL_ZERO = "0.18"
COL_SHADE = "#f4f4f4"

fig, ax = plt.subplots(figsize=(6.9, 5.15))
fig.subplots_adjust(left=0.13, right=0.98, top=0.86, bottom=0.40)

margin = max(0.008, 0.28 * max_abs_delta_Hz)
ymin = float(delta_Hz.min() - margin)
ymax = float(max(0.008, margin))

ax.axhspan(ymin, 0.0, color=COL_SHADE, alpha=0.75, lw=0, zorder=0)
ax.axhline(0.0, color=COL_ZERO, lw=1.0, zorder=1)

ax.bar(
    n,
    delta_Hz,
    width=0.48,
    color=COL_BAR,
    edgecolor="0.28",
    linewidth=0.7,
    zorder=3,
)

ax.scatter(
    n,
    delta_Hz,
    s=34,
    facecolor="white",
    edgecolor=COL_POINT,
    linewidth=1.1,
    zorder=4,
)

# All current deviations are negative, so the signed mean is -mean(|Delta|).
mean_line_value = (
    -mean_abs_delta_Hz
    if np.all(delta_Hz <= 0.0)
    else float(np.mean(delta_Hz))
)

ax.axhline(
    mean_line_value,
    color=COL_MEAN,
    lw=1.1,
    ls=(0, (4.0, 2.3)),
    zorder=2,
    label=rf"Mean absolute difference: {mean_abs_delta_Hz:.4f} Hz",
)

for ni, d_hz in zip(n, delta_Hz):
    vertical_offset = -0.0022 if d_hz <= 0.0 else 0.0022
    vertical_alignment = "top" if d_hz <= 0.0 else "bottom"

    txt = ax.text(
        ni,
        d_hz + vertical_offset,
        rf"${d_hz:.4f}$",
        color="0.20",
        fontsize=8.0,
        ha="center",
        va=vertical_alignment,
    )
    txt.set_path_effects([
        pe.withStroke(linewidth=2.4, foreground="white")
    ])

ax.set_xlim(float(n.min()) - 0.60, float(n.max()) + 0.60)
ax.set_ylim(ymin, ymax)
ax.set_xticks(n)
ax.set_xticklabels([rf"${ni}$" for ni in n])
ax.set_xlabel(r"Radial-state index, $n$")
ax.set_ylabel(r"$E_{\mathrm{Numerov}}-E_{\mathrm{FDM}}$ (Hz)")
ax.set_title(
    "Numerov validation of the continuum-certified spectrum",
    pad=9,
)

ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.tick_params(direction="in", which="both", top=True, right=True)
ax.legend(frameon=False, loc="lower right")

parameter_text = (
    rf"$r_c={r_c_nm:.9f}\,\mathrm{{nm}},\ "
    rf"\omega/2\pi={omega_over_2pi_MHz:.1f}\,\mathrm{{MHz}},\ "
    rf"\ell={ell}$"
    "\n"
    rf"FDM continuum grids: $N={fdm_grid_sequence}$; "
    rf"Numerov grid: $N={numerov_N}$"
)

fdm_values_text = ",\\,".join(f"{value:.9f}" for value in E_fdm_MHz)
num_values_text = ",\\,".join(f"{value:.9f}" for value in E_num_MHz)

energy_text = (
    rf"$E_{{\rm FDM}}/h=({fdm_values_text})\,\mathrm{{MHz}}$"
    "\n"
    rf"$E_{{\rm Num}}/h=({num_values_text})\,\mathrm{{MHz}}$"
    "\n"
    rf"$\max_n|\Delta_n|={max_abs_delta_Hz:.4f}\,\mathrm{{Hz}}$; "
    rf"$\langle|\Delta|\rangle={mean_abs_delta_Hz:.4f}\,\mathrm{{Hz}}$"
)

box_style = dict(
    boxstyle="round,pad=0.34",
    facecolor="white",
    edgecolor="0.70",
    linewidth=0.65,
    alpha=0.99,
)

fig.text(
    0.5,
    0.235,
    parameter_text,
    ha="center",
    va="center",
    fontsize=8.0,
    color="0.14",
    bbox=box_style,
)

fig.text(
    0.5,
    0.090,
    energy_text,
    ha="center",
    va="center",
    fontsize=7.8,
    color="0.14",
)

pdf_path = OUTDIR / "numerov_verification_negative_spectrum.pdf"
png_path = OUTDIR / "numerov_verification_negative_spectrum.png"
svg_path = OUTDIR / "numerov_verification_negative_spectrum.svg"

fig.savefig(pdf_path)
fig.savefig(png_path, dpi=600)
fig.savefig(svg_path)
plt.close(fig)


# ============================================================
# 7. Console report
# ============================================================

print("=== Numerov verification figure from CSV ===")
print(f"Input CSV: {CSV_PATH}")
print("Plot script:", Path(__file__).resolve())
print("CSV being read:", CSV_PATH.resolve())
print("State indices:                 ", n)
print("Node counts:                   ", nodes)
print("FDM continuum energies (MHz):  ", E_fdm_MHz)
print("Numerov energies (MHz):        ", E_num_MHz)
print("Numerov - FDM (Hz):            ", delta_Hz)
print("Match radii (nm):              ", match_r_nm)
print(f"Mean absolute difference:       {mean_abs_delta_Hz:.9f} Hz")
print(f"Maximum absolute difference:    {max_abs_delta_Hz:.9f} Hz")
print("Files written:")
for path in (plot_csv_path, pdf_path, png_path, svg_path):
    print(f"  - {path}")
