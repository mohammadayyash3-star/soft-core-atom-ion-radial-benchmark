

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
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
        "legend.fontsize": 8.2,

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
# 2. Perturbation-theory data
# ============================================================

# Low-lying states
n = np.array([0, 1, 2], dtype=int)

# Energies are E/h in MHz.
E0_MHz = np.array([
    1.800000,
    4.200000,
    6.600000,
], dtype=float)

E1_MHz = np.array([
    -15.216254,
    -10.370959,
    -8.108601,
], dtype=float)

E2_MHz = np.array([
    -4.478331,
    -3.630400,
    -0.885831,
], dtype=float)

E_pt_MHz = np.array([
    -17.894585,
    -9.801359,
    -2.394432,
], dtype=float)

E_fdm_MHz = np.array([
    -14.999999965215,
    -9.211774846321,
    -3.530419675022,
], dtype=float)

# Authoritative full-precision PT2-minus-FDM deviations from the final
# main-text/supplementary table. The displayed energies above are rounded
# to six decimals, so the deviations are stored explicitly rather than
# reconstructed from the rounded labels.
delta_pt_kHz = np.array([
    -2894.585,
    -589.585,
    +1135.988,
], dtype=float)

# Relative errors, computed from the authoritative deviations.
rel_err_pct = 100.0 * np.abs(delta_pt_kHz / 1e3) / np.abs(E_fdm_MHz)

# Perturbative smallness diagnostics.
r10 = np.array([
    8.4535,
    2.4693,
    1.2286,
], dtype=float)

r21 = np.array([
    0.2943,
    0.3501,
    0.1092,
], dtype=float)

eta_max = np.array([
    1.3604,
    1.7978,
    1.8273,
], dtype=float)

# Shared benchmark parameters.
r_c_nm = 25.876730807
omega_over_2pi_MHz = 1.200000
ell = 0
N_basis_PT = 80

# ============================================================
# 3. Colors
# ============================================================

COL_FDM = "#1f4e79"       # deep academic blue
COL_PT = "#b23a48"        # muted crimson
COL_R10 = "#7b3294"       # purple
COL_R21 = "#6c757d"       # neutral grey
COL_ETA = "#287c49"       # green
COL_LIMIT = "#9c6b1f"     # muted brown
COL_ZERO = "0.18"
COL_SHADE = "#eaf3f8"
COL_DIAG_SHADE = "#f4f4f4"
COL_GRID = "0.88"

# ============================================================
# 4. Figure layout
# ============================================================

fig = plt.figure(figsize=(7.65, 5.25))

gs = GridSpec(
    2, 2,
    figure=fig,
    height_ratios=[1.0, 0.22],
    width_ratios=[1.30, 1.05],
    hspace=0.08,
    wspace=0.30
)

ax1 = fig.add_subplot(gs[0, 0])   # Panel (a)
ax2 = fig.add_subplot(gs[0, 1])   # Panel (b)
bx1 = fig.add_subplot(gs[1, 0])   # Text box below panel (a)
bx2 = fig.add_subplot(gs[1, 1])   # Text box below panel (b)

fig.subplots_adjust(
    left=0.085,
    right=0.985,
    top=0.900,
    bottom=0.155
)

for bx in (bx1, bx2):
    bx.axis("off")

# ============================================================
# 5. Panel (a): PT energy comparison against FDM
# ============================================================

ymin_a = min(E_pt_MHz.min(), E_fdm_MHz.min()) - 0.95
ymax_a = 0.95

ax1.axhspan(
    ymin_a,
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

level_half_width = 0.155
offset_fdm = -0.12
offset_pt = +0.12

for ni, Ef, Ep in zip(n, E_fdm_MHz, E_pt_MHz):

    # FDM benchmark level
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

    # Second-order PT level
    ax1.hlines(
        y=Ep,
        xmin=ni + offset_pt - level_half_width,
        xmax=ni + offset_pt + level_half_width,
        color=COL_PT,
        lw=2.45,
        ls=(0, (5.0, 2.2)),
        zorder=4
    )

    ax1.scatter(
        ni + offset_pt,
        Ep,
        s=25,
        facecolor="white",
        edgecolor=COL_PT,
        lw=1.05,
        zorder=5
    )

    # Connector between FDM and PT
    ax1.plot(
        [ni + offset_fdm, ni + offset_pt],
        [Ef, Ep],
        color="0.55",
        lw=0.75,
        alpha=0.70,
        zorder=3
    )

    # FDM energy label
    if ni == 0:
        fdm_offset = 0.32
        fdm_va = "bottom"
    else:
        fdm_offset = 0.30
        fdm_va = "bottom"

    txt = ax1.text(
        ni + offset_fdm - 0.05,
        Ef + fdm_offset,
        rf"${Ef:.3f}$",
        color=COL_FDM,
        fontsize=7.35,
        ha="right",
        va=fdm_va
    )
    txt.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])

    # PT energy label
    if ni == 0:
        pt_offset = -0.34
        pt_va = "top"
    elif ni == 2:
        pt_offset = 0.34
        pt_va = "bottom"
    else:
        pt_offset = -0.34
        pt_va = "top"

    txt = ax1.text(
        ni + offset_pt + 0.05,
        Ep + pt_offset,
        rf"${Ep:.3f}$",
        color=COL_PT,
        fontsize=7.35,
        ha="left",
        va=pt_va
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

ax1.set_title("Second-order perturbation theory against FDM", pad=8)

ax1.set_xlim(-0.55, 2.55)
ax1.set_ylim(ymin_a-1, ymax_a)

ax1.set_xlabel(r"State index, $n$")
ax1.set_ylabel(r"Energy, $E/h$ (MHz)")

ax1.set_xticks(n)
ax1.set_xticklabels([rf"${ni}$" for ni in n])

ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
ax1.xaxis.set_minor_locator(AutoMinorLocator(2))

# ============================================================
# 6. Panel (b): Perturbative smallness diagnostics
# ============================================================

ax2.set_yscale("log")

ax2.axhspan(
    0.07,
    1.0,
    color=COL_DIAG_SHADE,
    alpha=0.62,
    lw=0,
    zorder=0
)

ax2.axhspan(
    1.0,
    12.0,
    color="#fff7ec",
    alpha=0.52,
    lw=0,
    zorder=0
)

ax2.axhline(
    1.0,
    color=COL_LIMIT,
    lw=1.15,
    ls=(0, (4.0, 2.3)),
    zorder=2
)

bar_width = 0.22
x_r10 = n - bar_width
x_r21 = n
x_eta = n + bar_width

bars_r10 = ax2.bar(
    x_r10,
    r10,
    width=bar_width,
    color=COL_R10,
    edgecolor="0.25",
    linewidth=0.60,
    zorder=3
)

bars_r21 = ax2.bar(
    x_r21,
    r21,
    width=bar_width,
    color=COL_R21,
    edgecolor="0.25",
    linewidth=0.60,
    zorder=3
)

bars_eta = ax2.bar(
    x_eta,
    eta_max,
    width=bar_width,
    color=COL_ETA,
    edgecolor="0.25",
    linewidth=0.60,
    zorder=3
)

# Bar-end markers
ax2.scatter(
    x_r10, r10,
    s=22,
    facecolor="white",
    edgecolor=COL_R10,
    lw=0.9,
    zorder=4
)

ax2.scatter(
    x_r21, r21,
    s=22,
    facecolor="white",
    edgecolor=COL_R21,
    lw=0.9,
    zorder=4
)

ax2.scatter(
    x_eta, eta_max,
    s=22,
    facecolor="white",
    edgecolor=COL_ETA,
    lw=0.9,
    zorder=4
)

# Numeric labels, placed multiplicatively for log scale.
def label_log_bars(ax, xs, ys, color, above=True):
    for xval, yval in zip(xs, ys):
        factor = 1.16 if above else 0.86
        va = "bottom" if above else "top"
        txt = ax.text(
            xval,
            yval * factor,
            rf"${yval:.2f}$",
            color=color,
            fontsize=7.25,
            ha="center",
            va=va
        )
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])

label_log_bars(ax2, x_r10, r10, COL_R10, above=True)
label_log_bars(ax2, x_r21, r21, COL_R21, above=False)
label_log_bars(ax2, x_eta, eta_max, COL_ETA, above=True)

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

ax2.set_title("Perturbative smallness diagnostics", pad=8)

ax2.set_xlim(-0.55, 2.55)
ax2.set_ylim(0.07, 15.0)

ax2.set_xlabel(r"State index, $n$")
ax2.set_ylabel(r"Diagnostic ratio")

ax2.set_xticks(n)
ax2.set_xticklabels([rf"${ni}$" for ni in n])

ax2.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
ax2.yaxis.set_minor_formatter(NullFormatter())
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
    rf"PT sums: $N_{{\rm basis}}={N_basis_PT}$ oscillator states" "\n"
    rf"$\Delta_{{\rm PT2-FDM}}=({delta_pt_kHz[0]:+.3f},\,{delta_pt_kHz[1]:+.3f},\,{delta_pt_kHz[2]:+.3f})\,\mathrm{{kHz}}$"
)

bx1.text(
    0.5,
    0.13,
    text_a,
    ha="center",
    va="center",
    fontsize=7.45,
    color="0.15",
    bbox=box_style
)

text_b = (
    r"$r_{10}=|E^{(1)}/E^{(0)}|,\quad "
    r"r_{21}=|E^{(2)}/E^{(1)}|,\quad "
    r"\eta_{\max}=\max |V_{n'n}/\Delta E^{(0)}|$" "\n"
    r"Controlled perturbation requires all ratios to remain well below unity."
)

bx2.text(
    0.5,
    0.13,
    text_b,
    ha="center",
    va="center",
    fontsize=7.35,
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

pt_handle = ax1.plot(
    [],
    [],
    color=COL_PT,
    lw=2.45,
    label=r"Second-order PT"
)[0]

r10_handle = ax2.plot(
    [],
    [],
    color=COL_R10,
    lw=5.0,
    label=r"$r_{10}$"
)[0]

r21_handle = ax2.plot(
    [],
    [],
    color=COL_R21,
    lw=5.0,
    label=r"$r_{21}$"
)[0]

eta_handle = ax2.plot(
    [],
    [],
    color=COL_ETA,
    lw=5.0,
    label=r"$\eta_{\max}$"
)[0]

limit_handle = ax2.plot(
    [],
    [],
    color=COL_LIMIT,
    lw=1.15,
    ls=(0, (4.0, 2.3)),
    label=r"Unity threshold"
)[0]

fig.legend(
    handles=[
        fdm_handle,
        pt_handle,
        r10_handle,
        r21_handle,
        eta_handle,
        limit_handle
    ],
    loc="lower center",
    bbox_to_anchor=(0.5, 0.025),
    ncol=6,
    frameon=False,
    handlelength=2.45,
    columnspacing=1.00,
    labelspacing=0.35,
    borderpad=0.2
)

# ============================================================
# 9. Final numerical consistency checks
# ============================================================

EXPECTED_DELTA_KHZ = np.array([-2894.585, -589.585, 1135.988])
if not np.allclose(delta_pt_kHz, EXPECTED_DELTA_KHZ, rtol=0.0, atol=5e-7):
    raise RuntimeError(
        "Figure 10 PT deviations do not match the final Table 11/S8.1 values."
    )

if N_basis_PT != 80:
    raise RuntimeError("Final PT2 data require 80 oscillator basis states.")

# ============================================================
# 10. Export
# ============================================================

output_dir = Path("journal_figures")
output_dir.mkdir(exist_ok=True)

fig.savefig(output_dir / "figure10_perturbation_breakdown_diagnostics_final.pdf")
fig.savefig(output_dir / "figure10_perturbation_breakdown_diagnostics_final.svg")
fig.savefig(output_dir / "figure10_perturbation_breakdown_diagnostics_final.png", dpi=600)

plt.show()