


from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patheffects as pe


# ============================================================
# 1. Output configuration
# ============================================================

OUTPUT_DIR = Path("figures")
OUTPUT_BASENAME = "Figure5"
SHOW_FIGURE = True


# ============================================================
# 2. Final manuscript data
# ============================================================

# Continuum-certified FDM levels, E_n/h in MHz.
STATE_INDEX = np.array([0, 1, 2], dtype=int)

E_CONTINUUM_MHZ = np.array(
    [
        -14.999999965,
        -9.211774846,
        -3.530419675,
    ],
    dtype=float,
)

NODE_COUNT = np.array([0, 1, 2], dtype=int)

# Observed grid-convergence order for each certified level.
OBSERVED_ORDER = np.array(
    [
        3.9996,
        3.9991,
        3.9986,
    ],
    dtype=float,
)

# Largest level-resolved validation difference among the continuum fit,
# independent matrix backend, Numerov result, and HO-basis result.
VALIDATION_SCALE_HZ = np.array(
    [
        0.0654,
        0.3470,
        0.6485,
    ],
    dtype=float,
)

# Final calibrated model parameters.
RC_NM = 25.876730807
OMEGA_OVER_2PI_MHZ = 1.2
ELL = 0

# Experimental anchor used only for the calibration of the ground-state scale.
EXPERIMENTAL_BINDING_CENTRAL_MHZ = 15.0
EXPERIMENTAL_BINDING_SIGMA_MHZ = 2.0

# The signed-energy band corresponding to E_0/h = -15(2) MHz.
ANCHOR_BAND_LOW_MHZ = -(
    EXPERIMENTAL_BINDING_CENTRAL_MHZ + EXPERIMENTAL_BINDING_SIGMA_MHZ
)
ANCHOR_BAND_HIGH_MHZ = -(
    EXPERIMENTAL_BINDING_CENTRAL_MHZ - EXPERIMENTAL_BINDING_SIGMA_MHZ
)


# ============================================================
# 3. Visual style
# ============================================================

COL = {
    "ink": "#20252b",
    "muted": "#626b75",
    "grid": "#d8dde3",
    "negative_bg": "#edf4f8",
    "positive_bg": "#f4f4f4",
    "anchor": "#9a6a1d",
    "anchor_fill": "#f4e5bf",
    "prediction": "#24557a",
    "prediction_light": "#dceaf4",
    "zero": "#30343a",
    "white": "#ffffff",
}


def set_journal_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 170,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,

            "font.family": "serif",
            "font.serif": [
                "STIXGeneral",
                "Times New Roman",
                "Times",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "text.usetex": False,

            "axes.linewidth": 0.90,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "legend.fontsize": 7.6,

            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,

            # Embed scalable fonts in vector output.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


# ============================================================
# 4. Validation helpers
# ============================================================

def validate_input_data() -> None:
    arrays = (
        STATE_INDEX,
        E_CONTINUUM_MHZ,
        NODE_COUNT,
        OBSERVED_ORDER,
        VALIDATION_SCALE_HZ,
    )
    lengths = {len(array) for array in arrays}

    if lengths != {3}:
        raise ValueError("All state-resolved arrays must contain exactly three entries.")

    if not np.array_equal(STATE_INDEX, np.array([0, 1, 2])):
        raise ValueError("STATE_INDEX must be [0, 1, 2].")

    if not np.array_equal(NODE_COUNT, STATE_INDEX):
        raise ValueError("The certified states must preserve node ordering 0, 1, 2.")

    if not np.all(np.isfinite(E_CONTINUUM_MHZ)):
        raise ValueError("All continuum energies must be finite.")

    if not np.all(E_CONTINUUM_MHZ < 0.0):
        raise ValueError("This figure must contain only certified negative-energy levels.")

    if not np.all(np.diff(E_CONTINUUM_MHZ) > 0.0):
        raise ValueError("The energies must be ordered from the ground state upward.")

    if not np.all((OBSERVED_ORDER > 3.9) & (OBSERVED_ORDER < 4.1)):
        raise ValueError("Observed convergence orders are inconsistent with fourth order.")

    if not np.all(VALIDATION_SCALE_HZ > 0.0):
        raise ValueError("Validation scales must be positive.")


def stroked_text(text_artist, linewidth: float = 2.4) -> None:
    text_artist.set_path_effects(
        [pe.withStroke(linewidth=linewidth, foreground=COL["white"])]
    )


# ============================================================
# 5. Figure construction
# ============================================================

def make_figure() -> plt.Figure:
    set_journal_style()
    validate_input_data()

    fig, ax = plt.subplots(figsize=(7.20, 4.20))

    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        top=0.930,
        bottom=0.225,
    )

    y_min = -18.2
    y_max = 2.4

    # Negative- and positive-energy sectors.
    ax.axhspan(
        y_min,
        0.0,
        color=COL["negative_bg"],
        alpha=1.0,
        linewidth=0,
        zorder=0,
    )
    ax.axhspan(
        0.0,
        y_max,
        color=COL["positive_bg"],
        alpha=1.0,
        linewidth=0,
        zorder=0,
    )

    # Experimental 15(2) MHz anchor band.
    ax.axhspan(
        ANCHOR_BAND_LOW_MHZ,
        ANCHOR_BAND_HIGH_MHZ,
        color=COL["anchor_fill"],
        alpha=0.62,
        linewidth=0,
        zorder=1,
    )

    # Central calibration target and E = 0 boundary.
    ax.axhline(
        -EXPERIMENTAL_BINDING_CENTRAL_MHZ,
        color=COL["anchor"],
        linewidth=1.05,
        linestyle=(0, (4.0, 2.3)),
        zorder=2,
    )
    ax.axhline(
        0.0,
        color=COL["zero"],
        linewidth=1.05,
        zorder=3,
    )

    # Light vertical state guides.
    for state_index in STATE_INDEX:
        ax.axvline(
            state_index,
            color=COL["grid"],
            linewidth=0.55,
            zorder=0,
        )

    level_half_width = 0.31

    for state_index, energy_mhz, node_count, validation_hz in zip(
        STATE_INDEX,
        E_CONTINUUM_MHZ,
        NODE_COUNT,
        VALIDATION_SCALE_HZ,
    ):
        is_ground_state = state_index == 0
        color = COL["anchor"] if is_ground_state else COL["prediction"]
        linewidth = 3.0 if is_ground_state else 2.7
        marker = "o" if is_ground_state else "s"

        # Horizontal certified level.
        ax.hlines(
            y=energy_mhz,
            xmin=state_index - level_half_width,
            xmax=state_index + level_half_width,
            color=color,
            linewidth=linewidth,
            zorder=5,
        )

        # Central state marker.
        ax.scatter(
            state_index,
            energy_mhz,
            s=34 if is_ground_state else 30,
            marker=marker,
            facecolor=COL["white"],
            edgecolor=color,
            linewidth=1.15,
            zorder=6,
        )

        # Energy label above the level.
        energy_text = ax.text(
            state_index,
            energy_mhz + 0.54,
            rf"$E_{{{state_index}}}/h={energy_mhz:.6f}\,\mathrm{{MHz}}$",
            color=color,
            fontsize=8.1,
            ha="center",
            va="bottom",
            zorder=7,
        )
        stroked_text(energy_text)

        # Node and validation annotation below the level.
        node_word = "node" if node_count == 1 else "nodes"
        detail_text = ax.text(
            state_index,
            energy_mhz - 0.58,
            rf"${node_count}$ {node_word}; "
            rf"$\delta_{{\rm val}}\leq {validation_hz:.4g}\,\mathrm{{Hz}}$",
            color=COL["muted"],
            fontsize=7.15,
            ha="center",
            va="top",
            zorder=7,
        )
        stroked_text(detail_text, linewidth=2.2)

    # Sector labels.
    negative_label = ax.text(
        0.025,
        0.50,
        "continuum-certified\nnegative-energy sector",
        transform=ax.transAxes,
        color=COL["prediction"],
        fontsize=8.0,
        ha="left",
        va="center",
        zorder=8,
    )
    stroked_text(negative_label)

    positive_label = ax.text(
        0.975,
        0.985,
        "positive trap-confined sector\n(no free-particle continuum)",
        transform=ax.transAxes,
        color=COL["muted"],
        fontsize=7.65,
        ha="right",
        va="top",
        zorder=8,
    )
    stroked_text(positive_label)

    # E = 0 label.
    zero_text = ax.text(
        2.42,
        -0.28,
        r"$E=0$",
        color=COL["zero"],
        fontsize=8.0,
        ha="right",
        va="top",
        zorder=8,
    )
    stroked_text(zero_text)

    # Experimental anchor-band label.
    band_text = ax.text(
        2.34,
        ANCHOR_BAND_LOW_MHZ + 0.28,
        r"experimental anchor: $E_{\rm bind}/h=15(2)\,\mathrm{MHz}$",
        color=COL["anchor"],
        fontsize=7.45,
        ha="right",
        va="bottom",
        zorder=8,
    )
    stroked_text(band_text)

    # Model and certification information.
    parameter_text = (
        rf"$r_c={RC_NM:.6f}\,\mathrm{{nm}}$, "
        rf"$\omega/2\pi={OMEGA_OVER_2PI_MHZ:.1f}\,\mathrm{{MHz}}$, "
        rf"$\ell={ELL}$"
        "\n"
        rf"$p_n\simeq4$; "
        rf"$\max(\delta_{{\rm val}})={np.max(VALIDATION_SCALE_HZ):.4g}\,\mathrm{{Hz}}$"
    )

    ax.text(
        0.025,
        0.965,
        parameter_text,
        transform=ax.transAxes,
        fontsize=7.45,
        ha="left",
        va="top",
        color=COL["ink"],
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": COL["white"],
            "edgecolor": "#aeb6bf",
            "linewidth": 0.65,
            "alpha": 0.97,
        },
        zorder=9,
    )

    # Distinguish the fitted anchor from predicted excited levels.
    ax.text(
        0.50,
        0.965,
        "Certified negative-energy ladder",
        transform=ax.transAxes,
        fontsize=9.3,
        ha="center",
        va="top",
        color=COL["ink"],
        zorder=9,
    )

    # Axes.
    ax.set_xlim(-0.52, 2.52)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"Radial-state index, $n$")
    ax.set_ylabel(r"Energy divided by $h$ (MHz)")

    ax.set_xticks(STATE_INDEX)
    ax.set_xticklabels([rf"${state_index}$" for state_index in STATE_INDEX])

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # External legend.
    legend_handles = [
        Line2D(
            [],
            [],
            color=COL["anchor"],
            linewidth=3.0,
            marker="o",
            markerfacecolor=COL["white"],
            markeredgecolor=COL["anchor"],
            label=r"Ground-state scale used in calibration",
        ),
        Line2D(
            [],
            [],
            color=COL["prediction"],
            linewidth=2.7,
            marker="s",
            markerfacecolor=COL["white"],
            markeredgecolor=COL["prediction"],
            label=r"Predicted excited negative levels",
        ),
        Patch(
            facecolor=COL["anchor_fill"],
            edgecolor="none",
            alpha=0.75,
            label=r"Experimental anchor band, $15(2)$ MHz",
        ),
        Line2D(
            [],
            [],
            color=COL["zero"],
            linewidth=1.05,
            label=r"$E=0$ sector boundary",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.145),
        ncol=2,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.6,
        handletextpad=0.65,
        labelspacing=0.55,
        borderpad=0.15,
    )

    return fig


# ============================================================
# 6. Export
# ============================================================

def save_figure(fig: plt.Figure) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.pdf"
    svg_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.svg"
    png_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.png"

    metadata = {
        "Title": "Continuum-certified negative-energy ladder",
        "Subject": "Calibrated soft-core atom-ion radial Hamiltonian",
        "Creator": "Matplotlib",
    }

    fig.savefig(
        pdf_path,
        facecolor=COL["white"],
        metadata=metadata,
    )
    fig.savefig(
        svg_path,
        facecolor=COL["white"],
    )
    fig.savefig(
        png_path,
        dpi=600,
        facecolor=COL["white"],
    )

    print("Saved:")
    print(pdf_path)
    print(svg_path)
    print(png_path)


def main() -> None:
    fig = make_figure()
    save_figure(fig)

    if SHOW_FIGURE:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
