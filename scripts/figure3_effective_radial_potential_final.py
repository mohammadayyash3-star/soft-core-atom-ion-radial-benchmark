

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


# ============================================================
# 1. Journal-quality style
# ============================================================

def set_journal_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,

        "font.family": "serif",
        "font.serif": [
            "STIXGeneral",
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "text.usetex": False,

        "axes.linewidth": 0.85,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.6,

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.1,
        "ytick.minor.size": 2.1,

        "lines.linewidth": 1.6,

        # Scalable embedded fonts for journal-quality vector output
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


set_journal_style()


# ============================================================
# 2. Physical constants and final benchmark parameters
# ============================================================

HBAR = 1.054_571_817e-34       # J s
HPLANCK = 6.626_070_15e-34     # J s
PI = np.pi

# Benchmark-fixed physical parameters
MU = 7.260_802_511e-26         # kg, reduced mass of 87Rb-88Sr+
OMEGA = 2.0 * PI * 1.2e6       # rad/s
C4 = 1.09e-56                  # J m^4
ALPHA = -0.5 * C4              # J m^4
ELL = 0                        # benchmark angular-momentum sector

# Continuum-calibrated soft-core radius used in the final manuscript
R_C = 25.876_730_807e-9        # m

# Derived harmonic oscillator length
A_HO = np.sqrt(HBAR / (MU * OMEGA))

# Continuum-certified FDM negative-energy levels, E/h in MHz
CONTINUUM_FDM_LEVELS_MHZ = np.array([
    -14.999_999_965,
    -9.211_774_846,
    -3.530_419_675,
], dtype=float)


# ============================================================
# 3. Unit conversion and potential definitions
# ============================================================

def joule_to_mhz(energy_joule: np.ndarray | float) -> np.ndarray | float:
    """Convert energy in joules to E/h in MHz."""
    return energy_joule / HPLANCK / 1e6


def v_harmonic(r_m: np.ndarray) -> np.ndarray:
    """Harmonic confinement contribution."""
    return 0.5 * MU * OMEGA**2 * r_m**2


def v_soft_core(r_m: np.ndarray) -> np.ndarray:
    """Regularized atom-ion polarization contribution."""
    return ALPHA / (r_m**4 + R_C**4)


def v_centrifugal(r_m: np.ndarray) -> np.ndarray:
    """
    Standard radial centrifugal term used in the numerical benchmark.

    The benchmark Hamiltonian uses ell(ell+1), not the Langer replacement.
    For ELL = 0 this term is exactly zero.
    """
    if ELL == 0:
        return np.zeros_like(r_m)

    return (HBAR**2 * ELL * (ELL + 1)) / (2.0 * MU * r_m**2)


def v_effective(r_m: np.ndarray) -> np.ndarray:
    """Total benchmark effective radial potential."""
    return v_harmonic(r_m) + v_centrifugal(r_m) + v_soft_core(r_m)


# ============================================================
# 4. Dense plotting grid and potential values
# ============================================================

# The small positive lower limit keeps the code safe if ELL is later changed.
R_NM = np.linspace(0.01, 170.0, 6000)
R_M = R_NM * 1e-9

VH_MHZ = joule_to_mhz(v_harmonic(R_M))
VS_MHZ = joule_to_mhz(v_soft_core(R_M))
VC_MHZ = joule_to_mhz(v_centrifugal(R_M))
VEFF_MHZ = joule_to_mhz(v_effective(R_M))

RC_NM = R_C * 1e9
AHO_NM = A_HO * 1e9


# ============================================================
# 5. Outer turning points of the negative-energy levels
# ============================================================

def find_outer_turning_point(
    r_values_nm: np.ndarray,
    potential_values_mhz: np.ndarray,
    energy_mhz: float,
) -> float | None:
    """
    Return the outer turning point satisfying V_eff(r_t) = E.

    For the final ell=0 benchmark potential, V_eff(r) rises from its
    finite short-range value and each displayed negative-energy level
    has one outer turning point. The origin is the radial boundary and
    is not treated as a second classical turning point.
    """
    difference = potential_values_mhz - energy_mhz

    crossing_indices = np.where(
        difference[:-1] * difference[1:] <= 0.0
    )[0]

    if crossing_indices.size == 0:
        return None

    # Use the outermost crossing. This remains robust if the potential
    # is later generalized and more than one crossing is present.
    idx = int(crossing_indices[-1])

    x1 = r_values_nm[idx]
    x2 = r_values_nm[idx + 1]
    y1 = difference[idx]
    y2 = difference[idx + 1]

    if np.isclose(y2, y1):
        return 0.5 * (x1 + x2)

    return x1 - y1 * (x2 - x1) / (y2 - y1)


OUTER_TURNING_POINTS_NM = {
    state_index: find_outer_turning_point(R_NM, VEFF_MHZ, energy_mhz)
    for state_index, energy_mhz in enumerate(CONTINUUM_FDM_LEVELS_MHZ)
}


# ============================================================
# 6. Color palette
# ============================================================

COL_TOTAL = "#111111"
COL_HARM = "#1f78b4"
COL_SOFT = "#d95f02"
COL_CENT = "#7570b3"
COL_LEVELS = [
    "#244f73",
    "#5b6f80",
    "#8a6d3b",
]
COL_RC = "#8c510a"
COL_AHO = "#01665e"
COL_SHADE = "0.90"
COL_TURNING = "#ffffff"


# ============================================================
# 7. Plot helpers
# ============================================================

def add_panel_label(ax: plt.Axes, label: str) -> None:
    """Add a journal-style panel label."""
    ax.text(
        0.03,
        0.95,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.12",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
        },
        zorder=20,
    )


# ============================================================
# 8. Main figure
# ============================================================

def make_figure() -> plt.Figure:
    fig = plt.figure(figsize=(7.20, 4.55))

    grid = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[1.12, 0.92],
        height_ratios=[1.0, 0.11, 0.12],
        wspace=0.24,
        hspace=0.16,
    )

    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax_empty = fig.add_subplot(grid[1, 0])
    ax2_note = fig.add_subplot(grid[1, 1])
    ax_legend = fig.add_subplot(grid[2, :])

    ax_empty.axis("off")
    ax2_note.axis("off")
    ax_legend.axis("off")

    # --------------------------------------------------------
    # Panel (a): Potential decomposition
    # --------------------------------------------------------
    ax1.plot(
        R_NM,
        VEFF_MHZ,
        color=COL_TOTAL,
        lw=2.0,
        label=rf"$V_{{\rm eff}}^{{(\ell={ELL})}}(r)$",
        zorder=5,
    )

    ax1.plot(
        R_NM,
        VH_MHZ,
        color=COL_HARM,
        lw=1.5,
        ls="--",
        label=r"$\frac{1}{2}\mu\omega^2r^2$",
        zorder=3,
    )

    ax1.plot(
        R_NM,
        VS_MHZ,
        color=COL_SOFT,
        lw=1.5,
        ls=":",
        label=r"$\alpha/(r^4+r_c^4)$",
        zorder=3,
    )

    if ELL != 0:
        ax1.plot(
            R_NM,
            VC_MHZ,
            color=COL_CENT,
            lw=1.3,
            ls="-.",
            label=r"$\hbar^2\ell(\ell+1)/(2\mu r^2)$",
            zorder=3,
        )

    ax1.axhline(0.0, color="0.25", lw=0.8, zorder=1)

    ax1.axvline(RC_NM, color=COL_RC, lw=1.1, ls="--", zorder=2)
    ax1.axvline(AHO_NM, color=COL_AHO, lw=1.1, ls="-.", zorder=2)

    ax1.text(
        RC_NM + 1.6,
        47.0,
        r"$r_c$",
        color=COL_RC,
        fontsize=8.2,
        rotation=90,
        va="top",
        ha="left",
    )

    ax1.text(
        AHO_NM + 1.6,
        47.0,
        r"$a_{\rm ho}$",
        color=COL_AHO,
        fontsize=8.2,
        rotation=90,
        va="top",
        ha="left",
    )

    ax1.set_xlim(0.0, 160.0)
    ax1.set_ylim(-22.0, 50.0)
    ax1.set_xlabel(r"$r$ (nm)")
    ax1.set_ylabel(r"Energy divided by $h$ (MHz)")

    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_major_locator(MaxNLocator(6))

    add_panel_label(ax1, "a")

    # --------------------------------------------------------
    # Panel (b): Continuum-certified negative-energy levels
    # --------------------------------------------------------
    ax2.plot(
        R_NM,
        VEFF_MHZ,
        color=COL_TOTAL,
        lw=2.0,
        label=rf"$V_{{\rm eff}}^{{(\ell={ELL})}}(r)$",
        zorder=5,
    )

    ax2.axhline(0.0, color="0.25", lw=0.8, zorder=2)

    # Shade the negative-energy sector. Positive-energy eigenstates
    # remain trap confined because harmonic confinement is retained.
    ax2.fill_between(
        R_NM,
        -20.0,
        0.0,
        color=COL_SHADE,
        alpha=0.55,
        linewidth=0,
        zorder=0,
    )

    ax2.text(
        0.965,
        0.085,
        "negative-energy sector",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        color="0.35",
        zorder=10,
    )

    ax2.axvline(RC_NM, color=COL_RC, lw=1.1, ls="--", zorder=2)
    ax2.axvline(AHO_NM, color=COL_AHO, lw=1.1, ls="-.", zorder=2)

    for state_index, energy_mhz in enumerate(CONTINUUM_FDM_LEVELS_MHZ):
        color = COL_LEVELS[state_index]
        outer_turning_nm = OUTER_TURNING_POINTS_NM[state_index]

        ax2.axhline(
            energy_mhz,
            color=color,
            lw=0.75,
            ls=(0, (3.5, 2.5)),
            alpha=0.65,
            zorder=2,
        )

        if outer_turning_nm is not None:
            # The origin is the radial boundary; the displayed interval
            # ends at the single outer turning point.
            ax2.plot(
                [R_NM[0], outer_turning_nm],
                [energy_mhz, energy_mhz],
                color=color,
                lw=2.4,
                solid_capstyle="round",
                zorder=6,
            )

            ax2.plot(
                outer_turning_nm,
                energy_mhz,
                marker="o",
                markersize=4.8,
                markerfacecolor=COL_TURNING,
                markeredgecolor=color,
                markeredgewidth=1.0,
                zorder=8,
            )

            ax2.text(
                outer_turning_nm + 3.0,
                energy_mhz + 0.25,
                rf"$n={state_index}$,  $E_n/h={energy_mhz:.3f}\,\mathrm{{MHz}}$",
                fontsize=7.1,
                color=color,
                ha="left",
                va="bottom",
                zorder=9,
            )

    ax2.text(
        RC_NM + 1.0,
        -19.2,
        r"$r_c$",
        color=COL_RC,
        fontsize=8.0,
        rotation=90,
        va="bottom",
        ha="left",
    )

    ax2.text(
        AHO_NM + 1.0,
        -19.2,
        r"$a_{\rm ho}$",
        color=COL_AHO,
        fontsize=8.0,
        rotation=90,
        va="bottom",
        ha="left",
    )

    ax2.set_xlim(0.0, 72.0)
    ax2.set_ylim(-20.0, 8.0)
    ax2.set_xlabel(r"$r$ (nm)")
    ax2.set_ylabel(r"Energy divided by $h$ (MHz)")

    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_major_locator(MaxNLocator(6))

    add_panel_label(ax2, "b")

    # --------------------------------------------------------
    # Parameter note below panel (b)
    # --------------------------------------------------------
    ax2_note.text(
        0.5,
        0.10,
        (
            r"$r_c=25.876731\,\mathrm{nm}$, "
            r"$a_{\mathrm{ho}}=13.879225\,\mathrm{nm}$, "
            r"$\ell=0$, "
            r"$\omega/2\pi=1.2\,\mathrm{MHz}$"
        ),
        transform=ax2_note.transAxes,
        fontsize=7.2,
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.30",
            "facecolor": "white",
            "edgecolor": "0.55",
            "linewidth": 0.75,
        },
    )

    # --------------------------------------------------------
    # Full-width legend
    # --------------------------------------------------------
    handles, labels = ax1.get_legend_handles_labels()
    ax_legend.legend(
        handles,
        labels,
        loc="center",
        ncol=3 if ELL == 0 else 4,
        frameon=False,
        handlelength=3.2,
        columnspacing=1.5,
        handletextpad=0.65,
        fontsize=8.0,
    )

    return fig


# ============================================================
# 9. Final export
# ============================================================

def main() -> None:
    fig = make_figure()

    output_dir = Path("journal_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "figure3_effective_radial_potential.pdf"
    svg_path = output_dir / "figure3_effective_radial_potential.svg"
    png_path = output_dir / "figure3_effective_radial_potential.png"

    fig.savefig(
        pdf_path,
        facecolor="white",
        metadata={
            "Title": "Continuum-certified effective radial potential",
            "Subject": "Calibrated soft-core atom-ion radial Hamiltonian",
            "Creator": "Matplotlib",
        },
    )

    fig.savefig(
        svg_path,
        facecolor="white",
    )

    fig.savefig(
        png_path,
        dpi=600,
        facecolor="white",
    )

    print("Saved:")
    print(pdf_path)
    print(svg_path)
    print(png_path)
    print()
    print("Derived outer turning points:")

    for state_index, radius_nm in OUTER_TURNING_POINTS_NM.items():
        if radius_nm is None:
            print(f"  n={state_index}: not found")
        else:
            print(f"  n={state_index}: {radius_nm:.6f} nm")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
