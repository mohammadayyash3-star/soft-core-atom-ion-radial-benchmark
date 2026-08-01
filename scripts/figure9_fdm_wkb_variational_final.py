

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import patheffects as pe
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ============================================================
# 1. Journal-quality plotting style
# ============================================================

def set_journal_style():
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
        "axes.titlesize": 9.8,
        "xtick.labelsize": 8.9,
        "ytick.labelsize": 8.9,
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
# 2. Data
# ============================================================

# Negative-energy states only.
# Energies are reported as E/h in MHz.
n = np.array([0, 1, 2], dtype=int)

E_fdm_MHz = np.array([
    -14.999999964,
    -9.211774845,
    -3.530419674,
], dtype=float)

E_wkb_MHz = np.array([
    -15.024342909,
    -9.180825654,
    -3.500548970,
], dtype=float)

# Variational result is ground-state only.
E_var0_MHz = -14.992134017

# Differences relative to FDM, in kHz.
delta_wkb_kHz = 1e3 * (E_wkb_MHz - E_fdm_MHz)
delta_var0_kHz = 1e3 * (E_var0_MHz - E_fdm_MHz[0])

mean_abs_wkb_delta = np.mean(np.abs(delta_wkb_kHz))

# Shared benchmark parameters
r_c_nm = 25.876730807
omega_over_2pi_MHz = 1.200000
ell = 0
target_MHz = 15.000000

# ============================================================
# 3. Colors
# ============================================================

COL_FDM = "#1f4e79"          # deep academic blue
COL_WKB = "#b23a48"          # muted crimson
COL_VAR = "#287c49"          # green
COL_DELTA = "#6c757d"        # neutral grey
COL_ZERO = "0.18"
COL_REF = "#9c6b1f"          # muted brown
COL_SHADE = "#eaf3f8"
COL_DEV_SHADE = "#f4f4f4"
COL_GRID = "0.88"

# ============================================================
# 4. Figure layout
# ============================================================

fig = plt.figure(figsize=(7.65, 5.25))

gs = GridSpec(
    2, 2,
    figure=fig,
    height_ratios=[1.0, 0.20],
    width_ratios=[1.35, 1.00],
    hspace=0.08,
    wspace=0.28
)

ax1 = fig.add_subplot(gs[0, 0])   # Panel (a)
ax2 = fig.add_subplot(gs[0, 1])   # Panel (b)
bx1 = fig.add_subplot(gs[1, 0])   # text box below panel (a)
bx2 = fig.add_subplot(gs[1, 1])   # text box below panel (b)

fig.subplots_adjust(
    left=0.085,
    right=0.985,
    top=0.900,
    bottom=0.155
)

for bx in (bx1, bx2):
    bx.axis("off")

# ============================================================
# 5. Panel (a): Spectrum comparison
# ============================================================

ax1.axhspan(
    min(E_wkb_MHz) - 0.90,
    0.0,
    color=COL_SHADE,
    alpha=0.82,
    lw=0,
    zorder=0
)

ax1.axhline(
    0.0,
    color=COL_ZERO,
    lw=1.05,
    zorder=1
)

for ni in n:
    ax1.axvline(
        ni,
        color=COL_GRID,
        lw=0.55,
        zorder=0
    )

level_half_width = 0.145
offset_fdm = -0.16
offset_wkb = +0.03
offset_var = +0.22

for ni, Ef, Ew in zip(n, E_fdm_MHz, E_wkb_MHz):

    # FDM level
    ax1.hlines(
        y=Ef,
        xmin=ni + offset_fdm - level_half_width,
        xmax=ni + offset_fdm + level_half_width,
        color=COL_FDM,
        lw=2.75,
        zorder=4
    )

    ax1.scatter(
        ni + offset_fdm,
        Ef,
        s=25,
        facecolor="white",
        edgecolor=COL_FDM,
        lw=1.05,
        zorder=5
    )

    # WKB level
    ax1.hlines(
        y=Ew,
        xmin=ni + offset_wkb - level_half_width,
        xmax=ni + offset_wkb + level_half_width,
        color=COL_WKB,
        lw=2.45,
        ls=(0, (5.0, 2.2)),
        zorder=4
    )

    ax1.scatter(
        ni + offset_wkb,
        Ew,
        s=25,
        facecolor="white",
        edgecolor=COL_WKB,
        lw=1.05,
        zorder=5
    )

    # Connector between FDM and WKB
    ax1.plot(
        [ni + offset_fdm, ni + offset_wkb],
        [Ef, Ew],
        color="0.55",
        lw=0.75,
        alpha=0.70,
        zorder=3
    )

    # Labels only for FDM and WKB
    txt = ax1.text(
        ni + offset_fdm - 0.05,
        Ef + 0.30,
        rf"${Ef:.3f}$",
        color=COL_FDM,
        fontsize=7.35,
        ha="right",
        va="bottom"
    )
    txt.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])

    txt = ax1.text(
        ni + offset_wkb + 0.05,
        Ew - 0.30,
        rf"${Ew:.3f}$",
        color=COL_WKB,
        fontsize=7.35,
        ha="left",
        va="top"
    )
    txt.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])

# Variational ground-state point only
ax1.hlines(
    y=E_var0_MHz,
    xmin=0 + offset_var - level_half_width,
    xmax=0 + offset_var + level_half_width,
    color=COL_VAR,
    lw=2.35,
    ls=(0, (2.0, 1.6)),
    zorder=4
)

ax1.scatter(
    0 + offset_var,
    E_var0_MHz,
    s=34,
    marker="D",
    facecolor="white",
    edgecolor=COL_VAR,
    lw=1.05,
    zorder=5
)

ax1.plot(
    [0 + offset_fdm, 0 + offset_var],
    [E_fdm_MHz[0], E_var0_MHz],
    color="0.55",
    lw=0.75,
    alpha=0.70,
    zorder=3
)

txt = ax1.text(
    0 + offset_var + 0.05,
    E_var0_MHz + 0.32,
    rf"${E_var0_MHz:.3f}$",
    color=COL_VAR,
    fontsize=7.35,
    ha="left",
    va="bottom"
)
txt.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])

# Panel label
ax1.text(
    0.025,
    0.965,
    "a",
    transform=ax1.transAxes,
    fontsize=11.5,
    fontweight="bold",
    va="top"
)

ax1.set_title("Approximate spectra against the FDM benchmark", pad=8)

ax1.set_xlim(-0.55, 2.55)
ax1.set_ylim(min(E_wkb_MHz) - 0.90, 0.95)

ax1.set_xlabel(r"State index, $n$")
ax1.set_ylabel(r"Energy, $E/h$ (MHz)")

ax1.set_xticks(n)
ax1.set_xticklabels([rf"${ni}$" for ni in n])

ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
ax1.xaxis.set_minor_locator(AutoMinorLocator(2))

# ============================================================
# 6. Panel (b): Deviations from FDM
# ============================================================

ax2.axhline(
    0.0,
    color=COL_ZERO,
    lw=1.05,
    zorder=1
)

# Data-driven signed-deviation range for the corrected results.
all_deviations_kHz = np.concatenate(
    [delta_wkb_kHz, np.array([delta_var0_kHz], dtype=float)]
)
deviation_margin_kHz = max(
    12.0,
    0.25 * np.max(np.abs(all_deviations_kHz))
)
ymin = min(all_deviations_kHz.min() - deviation_margin_kHz, -5.0)
ymax = max(all_deviations_kHz.max() + deviation_margin_kHz, +5.0)

ax2.axhspan(
    ymin,
    0.0,
    color=COL_DEV_SHADE,
    alpha=0.72,
    lw=0,
    zorder=0
)

ax2.axhspan(
    0.0,
    ymax,
    color="#f7fbf7",
    alpha=0.62,
    lw=0,
    zorder=0
)

# Bar positions
x_wkb = n - 0.12
x_var = np.array([0.24])

bar_width = 0.38

ax2.bar(
    x_wkb,
    delta_wkb_kHz,
    width=bar_width,
    color=COL_DELTA,
    edgecolor="0.30",
    linewidth=0.65,
    zorder=3,
    label="WKB deviation"
)

ax2.scatter(
    x_wkb,
    delta_wkb_kHz,
    s=26,
    facecolor="white",
    edgecolor=COL_DELTA,
    lw=1.0,
    zorder=4
)

# Variational ground-state deviation
ax2.bar(
    x_var,
    [delta_var0_kHz],
    width=bar_width,
    color=COL_VAR,
    alpha=0.92,
    edgecolor="0.25",
    linewidth=0.65,
    zorder=3,
    label="Variational deviation"
)

ax2.scatter(
    x_var,
    [delta_var0_kHz],
    s=34,
    marker="D",
    facecolor="white",
    edgecolor=COL_VAR,
    lw=1.05,
    zorder=4
)

# Mean absolute WKB deviation shown as symmetric magnitude guides.
for sign in (-1.0, +1.0):
    ax2.axhline(
        sign * mean_abs_wkb_delta,
        color=COL_REF,
        lw=1.0,
        ls=(0, (4.0, 2.3)),
        zorder=2
    )

# Deviation labels
for xi, dlt in zip(x_wkb, delta_wkb_kHz):
    txt = ax2.text(
        xi,
        dlt - 7.0,
        rf"${dlt:.1f}$",
        color="0.25",
        fontsize=7.45,
        ha="center",
        va="top"
    )
    txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])

txt = ax2.text(
    x_var[0],
    delta_var0_kHz + 13.0,
    rf"${delta_var0_kHz:.1f}$",
    color=COL_VAR,
    fontsize=7.45,
    ha="center",
    va="bottom"
)
txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])

# Panel label
ax2.text(
    0.035,
    0.965,
    "b",
    transform=ax2.transAxes,
    fontsize=11.5,
    fontweight="bold",
    va="top"
)

ax2.set_title("Deviation from FDM", pad=8)

ax2.set_xlim(-0.55, 2.55)
ax2.set_ylim(ymin, ymax)

ax2.set_xlabel(r"State index, $n$")
ax2.set_ylabel(r"$\Delta$ (kHz)")

ax2.set_xticks(n)
ax2.set_xticklabels([rf"${ni}$" for ni in n])

ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
ax2.xaxis.set_minor_locator(AutoMinorLocator(2))

# ============================================================
# 7. External text boxes below the panels
# ============================================================

box_style = dict(
    boxstyle="round,pad=0.28",
    fc="white",
    ec="0.72",
    lw=0.60,
    alpha=0.98
)

text_a = (
    rf"$r_c={r_c_nm:.6f}\,\mathrm{{nm}},\ "
    rf"\omega/2\pi={omega_over_2pi_MHz:.1f}\,\mathrm{{MHz}},\ "
    rf"\ell={ell}$" "\n"
    r"WKB is shown for $n=0,1,2$; variational result is ground-state only."
)

bx1.text(
    0.5,
    0.13,
    text_a,
    ha="center",
    va="center",
    fontsize=7.55,
    color="0.15",
    bbox=box_style
)

# Under panel (b)
text_b = (
    r"$\Delta = E_{\mathrm{method}} - E_{\mathrm{FDM}}$"
    r"  (reported in kHz)" "\n"
    rf"$\langle|\Delta_{{\mathrm{{WKB}}}}|\rangle = "
    rf"{mean_abs_wkb_delta:.1f}\,\mathrm{{kHz}},\quad "
    rf"\Delta_{{\mathrm{{var}},0}} = "
    rf"{delta_var0_kHz:.1f}\,\mathrm{{kHz}}$"
)

bx2.text(
    0.5,
    0.13,
    text_b,
    ha="center",
    va="center",
    fontsize=7.55,
    color="0.15",
    bbox=box_style
)

# ============================================================
# 8. Common external legend
# ============================================================

fdm_handle = ax1.plot(
    [],
    [],
    color=COL_FDM,
    lw=2.75,
    label=r"FDM benchmark"
)[0]

wkb_handle = ax1.plot(
    [],
    [],
    color=COL_WKB,
    lw=2.45,
    label=r"WKB"
)[0]

var_handle = ax1.plot(
    [],
    [],
    color=COL_VAR,
    lw=2.35,
    ls=(0, (2.0, 1.6)),
    marker="D",
    markerfacecolor="white",
    markeredgecolor=COL_VAR,
    label=r"Variational ground state"
)[0]

dev_handle = ax2.plot(
    [],
    [],
    color=COL_DELTA,
    lw=5.0,
    label=r"WKB--FDM deviation"
)[0]

mean_handle = ax2.plot(
    [],
    [],
    color=COL_REF,
    lw=1.0,
    ls=(0, (4.0, 2.3)),
    label=r"Mean $|\Delta_{\rm WKB}|$"
)[0]

fig.legend(
    handles=[fdm_handle, wkb_handle, var_handle, dev_handle, mean_handle],
    loc="lower center",
    bbox_to_anchor=(0.5, 0.025),
    ncol=5,
    frameon=False,
    handlelength=2.55,
    columnspacing=1.10,
    labelspacing=0.35,
    borderpad=0.2
)

# ============================================================
# 9. Export
# ============================================================

output_dir = Path("journal_figures")
output_dir.mkdir(exist_ok=True)

fig.savefig(output_dir / "figure9_fdm_wkb_variational_comparison.pdf")
fig.savefig(output_dir / "figure9_fdm_wkb_variational_comparison.svg")
fig.savefig(output_dir / "figure9_fdm_wkb_variational_comparison.png", dpi=600)

plt.close(fig)