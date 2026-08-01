from __future__ import annotations


from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import patheffects as pe


# ============================================================
# 1. Output location and journal style
# ============================================================

OUTDIR = Path(__file__).resolve().parent / "numerov_figure_final"
OUTDIR.mkdir(parents=True, exist_ok=True)


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

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


set_journal_style()


# ============================================================
# 2. Final corrected benchmark data
# ============================================================

n = np.array([0, 1, 2], dtype=int)

# Continuum-certified FDM values, E_n/h in MHz.
E_fdm_MHz = np.array([
    -14.999999963984,
    -9.211774845325,
    -3.530419674323,
], dtype=float)

# Independent Numerov values for the same frozen Hamiltonian.
# These values correspond to signed Numerov-FDM differences
# (-0.0257, -0.0307, -0.0220) Hz.
E_num_MHz = np.array([
    -14.999999989684,
    -9.211774876025,
    -3.530419696323,
], dtype=float)

# MHz -> Hz.
delta_Hz = 1.0e6 * (E_num_MHz - E_fdm_MHz)
mean_abs_delta_Hz = float(np.mean(np.abs(delta_Hz)))
max_abs_delta_Hz = float(np.max(np.abs(delta_Hz)))

# Shared benchmark parameters.
r_c_nm = 25.876730807
omega_over_2pi_MHz = 1.2
ell = 0

# Numerical representations.
fdm_grid_sequence = (1800, 2556, 3630, 5155)
numerov_N = 24000


# ============================================================
# 3. Machine-readable output
# ============================================================

csv_path = OUTDIR / "numerov_verification_negative_spectrum.csv"
table = np.column_stack((n, E_fdm_MHz, E_num_MHz, delta_Hz))
np.savetxt(
    csv_path,
    table,
    delimiter=",",
    header="n,E_FDM_continuum_MHz,E_Numerov_MHz,Numerov_minus_FDM_Hz",
    comments="",
    fmt=["%d", "%.12f", "%.12f", "%.7f"],
)


# ============================================================
# 4. Single-panel publication-quality figure
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

bars = ax.bar(
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

# All deviations are negative, so the signed mean lies at -<|Delta|>.
ax.axhline(
    -mean_abs_delta_Hz,
    color=COL_MEAN,
    lw=1.1,
    ls=(0, (4.0, 2.3)),
    zorder=2,
    label=rf"Mean absolute difference: {mean_abs_delta_Hz:.4f} Hz",
)

for ni, d_hz in zip(n, delta_Hz):
    txt = ax.text(
        ni,
        d_hz - 0.0022,
        rf"${d_hz:.4f}$",
        color="0.20",
        fontsize=8.0,
        ha="center",
        va="top",
    )
    txt.set_path_effects([
        pe.withStroke(linewidth=2.4, foreground="white")
    ])

ax.set_xlim(-0.60, 2.60)
ax.set_ylim(ymin, ymax)
ax.set_xticks(n)
ax.set_xticklabels([rf"${ni}$" for ni in n])
ax.set_xlabel(r"Radial-state index, $n$")
ax.set_ylabel(r"$E_{\mathrm{Numerov}}-E_{\mathrm{FDM}}$ (Hz)")
ax.set_title("Numerov validation of the continuum-certified spectrum", pad=9)

ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.tick_params(direction="in", which="both", top=True, right=True)
ax.legend(frameon=False, loc="lower right")

# External annotation: physical setup and energy values.
parameter_text = (
    rf"$r_c={r_c_nm:.9f}\,\mathrm{{nm}},\ "
    rf"\omega/2\pi={omega_over_2pi_MHz:.1f}\,\mathrm{{MHz}},\ "
    rf"\ell={ell}$"
    "\n"
    rf"FDM continuum grids: $N={fdm_grid_sequence}$; "
    rf"Numerov grid: $N={numerov_N}$"
)

energy_text = (
    r"$E_{\rm FDM}/h=(-14.999999964,\,-9.211774845,\,-3.530419674)\,\mathrm{MHz}$"
    "\n"
    r"$E_{\rm Num}/h=(-14.999999990,\,-9.211774876,\,-3.530419696)\,\mathrm{MHz}$"
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
# 5. Console report
# ============================================================

print("=== Final Numerov verification figure ===")
print(f"r_c = {r_c_nm:.9f} nm")
print("FDM continuum energies (MHz):", E_fdm_MHz)
print("Numerov energies (MHz):      ", E_num_MHz)
print("Numerov - FDM (Hz):          ", delta_Hz)
print(f"Mean absolute difference:     {mean_abs_delta_Hz:.7f} Hz")
print(f"Maximum absolute difference:  {max_abs_delta_Hz:.7f} Hz")
print("Files written:")
for path in (csv_path, pdf_path, png_path, svg_path):
    print(f"  - {path}")
