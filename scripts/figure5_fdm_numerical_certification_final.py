from __future__ import annotations



from pathlib import Path
import csv
import json
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
from matplotlib import patheffects as pe


# ============================================================
# 1. Paths and style
# ============================================================

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "figure5_final_outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

MANIFEST_CANDIDATES = (
    HERE / "benchmark_manifest_final.json",
    HERE / "softcore_benchmark_outputs" / "benchmark_manifest_final.json",
    HERE / "softcore_benchmark_final_package"
         / "softcore_benchmark_outputs"
         / "benchmark_manifest_final.json",
)


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
        "axes.labelsize": 9.9,
        "axes.titlesize": 9.6,
        "xtick.labelsize": 8.4,
        "ytick.labelsize": 8.4,
        "legend.fontsize": 7.8,

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4.2,
        "ytick.major.size": 4.2,
        "xtick.minor.size": 2.2,
        "ytick.minor.size": 2.2,

        "lines.linewidth": 1.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


set_journal_style()


# ============================================================
# 2. Frozen fallback dataset
# ============================================================

FALLBACK_MANIFEST = {
    "scheme": (
        "fourth-order five-point FDM with explicit Dirichlet boundaries "
        "and odd-reflection ghost closure"
    ),
    "rc_nm": 25.876730807,
    "omega_over_2pi_MHz": 1.2,
    "ell": 0,
    "r_min_nm": 0.0,
    "r_max_nm": 650.0,
    "continuum_grid_N": [1800, 2556, 3630, 5155],
    "continuum": {
        "rows": [
            {
                "N": 1800,
                "delta_r_nm": 0.3613118399110617,
                "E0_MHz": -15.000000375417246,
                "E1_MHz": -9.211779725677731,
                "E2_MHz": -3.530435794490245,
            },
            {
                "N": 2556,
                "delta_r_nm": 0.25440313111545987,
                "E0_MHz": -15.000000065128232,
                "E1_MHz": -9.211776045265877,
                "E2_MHz": -3.530423638331058,
            },
            {
                "N": 3630,
                "delta_r_nm": 0.17911270322402867,
                "E0_MHz": -14.999999988830954,
                "E1_MHz": -9.211775140034204,
                "E2_MHz": -3.5304206475998394,
            },
            {
                "N": 5155,
                "delta_r_nm": 0.12611563833915407,
                "E0_MHz": -14.999999970076715,
                "E1_MHz": -9.211774917564070,
                "E2_MHz": -3.5304199125284863,
            },
        ],
        "fits": [
            {
                "state": 0,
                "E_infinity_MHz": -14.999999963984289,
                "amplitude": -2.4142180454252093e-05,
                "observed_order": 4.000021137769207,
                "R2": 0.9999999941116077,
                "max_abs_fit_residual_kHz": 2.006039778734703e-08,
            },
            {
                "state": 1,
                "E_infinity_MHz": -9.211774845325102,
                "amplitude": -2.8637209477937984e-04,
                "observed_order": 3.999985371318147,
                "R2": 0.9999999864854895,
                "max_abs_fit_residual_kHz": 3.608153775758183e-07,
            },
            {
                "state": 2,
                "E_infinity_MHz": -3.5304196743229888,
                "amplitude": -9.459141753199784e-04,
                "observed_order": 3.9999811691324467,
                "R2": 0.9999999720815406,
                "max_abs_fit_residual_kHz": 1.6589609685979667e-06,
            },
        ],
        "continuum_levels_MHz": [
            -14.999999963984289,
            -9.211774845325102,
            -3.5304196743229888,
        ],
        "sparse_minus_banded_kHz": [
            -1.801225835151854e-09,
            -5.066169705969514e-09,
            2.4589219549397967e-09,
        ],
    },
    "numerov": [
        {
            "n": 0,
            "E_FDM_continuum_MHz": -14.999999963984289,
            "E_Numerov_MHz": -14.999999990955239,
            "delta_Numerov_minus_FDM_Hz": -0.025770949917881626,
        },
        {
            "n": 1,
            "E_FDM_continuum_MHz": -9.211774845325102,
            "E_Numerov_MHz": -9.211774877056060,
            "delta_Numerov_minus_FDM_Hz": -0.030730957417153150,
        },
        {
            "n": 2,
            "E_FDM_continuum_MHz": -3.5304196743229888,
            "E_Numerov_MHz": -3.5304196970678725,
            "delta_Numerov_minus_FDM_Hz": -0.022044883754199918,
        },
    ],
}


def load_manifest() -> tuple[dict, str]:
    for candidate in MANIFEST_CANDIDATES:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                return json.load(handle), str(candidate)
    return FALLBACK_MANIFEST, "embedded frozen fallback"


manifest, manifest_source = load_manifest()


# ============================================================
# 3. Validation and data extraction
# ============================================================

expected_grid = (1800, 2556, 3630, 5155)
grid_from_manifest = tuple(int(x) for x in manifest["continuum_grid_N"])

if grid_from_manifest != expected_grid:
    raise RuntimeError(
        "Figure 5 requires the certified grid sequence "
        f"{expected_grid}, but found {grid_from_manifest}."
    )

if not math.isclose(float(manifest["r_min_nm"]), 0.0, abs_tol=1.0e-15):
    raise RuntimeError("The corrected Figure 5 requires r_min = 0.")

if not math.isclose(float(manifest["r_max_nm"]), 650.0, abs_tol=1.0e-12):
    raise RuntimeError("The corrected Figure 5 requires r_max = 650 nm.")

rows = manifest["continuum"]["rows"]
fits = sorted(manifest["continuum"]["fits"], key=lambda item: int(item["state"]))

N_values = np.asarray([int(row["N"]) for row in rows], dtype=int)
dr_nm = np.asarray([float(row["delta_r_nm"]) for row in rows], dtype=float)

energy_grid_MHz = np.asarray([
    [float(row["E0_MHz"]), float(row["E1_MHz"]), float(row["E2_MHz"])]
    for row in rows
], dtype=float)

E_inf_MHz = np.asarray(
    manifest["continuum"]["continuum_levels_MHz"],
    dtype=float,
)

# Shape: (number of grids, number of states).
grid_error_Hz = 1.0e6 * (energy_grid_MHz - E_inf_MHz[None, :])
abs_grid_error_Hz = np.abs(grid_error_Hz)

observed_orders = np.asarray(
    [float(item["observed_order"]) for item in fits],
    dtype=float,
)
fit_R2 = np.asarray([float(item["R2"]) for item in fits], dtype=float)
fit_amplitude = np.asarray(
    [float(item["amplitude"]) for item in fits],
    dtype=float,
)

backend_diff_microHz = (
    1.0e9
    * np.asarray(
        manifest["continuum"]["sparse_minus_banded_kHz"],
        dtype=float,
    )
)

numerov_rows = sorted(manifest["numerov"], key=lambda item: int(item["n"]))
numerov_diff_Hz = np.asarray(
    [float(item["delta_Numerov_minus_FDM_Hz"]) for item in numerov_rows],
    dtype=float,
)

states = np.arange(3, dtype=int)
state_labels = [r"$n=0$", r"$n=1$", r"$n=2$"]


# ============================================================
# 4. Machine-readable outputs
# ============================================================

data_csv = OUTDIR / "Figure_5_data.csv"
with data_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow([
        "N",
        "delta_r_nm",
        "E0_grid_MHz",
        "E1_grid_MHz",
        "E2_grid_MHz",
        "dE0_grid_minus_continuum_Hz",
        "dE1_grid_minus_continuum_Hz",
        "dE2_grid_minus_continuum_Hz",
    ])
    for index, N in enumerate(N_values):
        writer.writerow([
            int(N),
            f"{dr_nm[index]:.15g}",
            f"{energy_grid_MHz[index, 0]:.15g}",
            f"{energy_grid_MHz[index, 1]:.15g}",
            f"{energy_grid_MHz[index, 2]:.15g}",
            f"{grid_error_Hz[index, 0]:.15g}",
            f"{grid_error_Hz[index, 1]:.15g}",
            f"{grid_error_Hz[index, 2]:.15g}",
        ])

summary = {
    "manifest_source": manifest_source,
    "scheme": manifest["scheme"],
    "r_c_nm": float(manifest["rc_nm"]),
    "omega_over_2pi_MHz": float(manifest["omega_over_2pi_MHz"]),
    "ell": int(manifest["ell"]),
    "radial_domain_nm": [
        float(manifest["r_min_nm"]),
        float(manifest["r_max_nm"]),
    ],
    "continuum_grid_N": [int(x) for x in N_values],
    "continuum_levels_MHz": [float(x) for x in E_inf_MHz],
    "observed_orders": [float(x) for x in observed_orders],
    "fit_R2": [float(x) for x in fit_R2],
    "sparse_minus_banded_microHz": [
        float(x) for x in backend_diff_microHz
    ],
    "numerov_minus_FDM_Hz": [float(x) for x in numerov_diff_Hz],
    "max_abs_backend_difference_microHz": float(
        np.max(np.abs(backend_diff_microHz))
    ),
    "max_abs_Numerov_difference_Hz": float(
        np.max(np.abs(numerov_diff_Hz))
    ),
}

summary_json = OUTDIR / "Figure_5_summary.json"
summary_json.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)


# ============================================================
# 5. Figure helpers
# ============================================================
# ============================================================
# 5. Figure helpers
# ============================================================

STATE_COLORS = ("#1f77b4", "#d95f02", "#2ca02c")


def apply_panel_label(ax, label: str) -> None:
    """Place the panel label outside the plotting region."""
    ax.text(
        -0.105,
        1.055,
        label,
        transform=ax.transAxes,
        fontsize=11.0,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def apply_axes_style(ax) -> None:
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
    )
    ax.grid(True, which="major", linewidth=0.42, alpha=0.24)


def annotate_value(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    dy_points: float,
    va: str,
) -> None:
    annotation = ax.annotate(
        text,
        xy=(x, y),
        xytext=(0.0, dy_points),
        textcoords="offset points",
        ha="center",
        va=va,
        fontsize=7.6,
    )
    annotation.set_path_effects([
        pe.withStroke(linewidth=2.3, foreground="white")
    ])


# ============================================================
# 6. Figure layout
# ============================================================

# The larger canvas, dedicated bottom note strip, and increased inter-panel
# spacing prevent titles, labels, legends, and annotations from colliding.
fig = plt.figure(figsize=(9.20, 6.75))

grid = fig.add_gridspec(
    nrows=2,
    ncols=2,
    left=0.095,
    right=0.985,
    top=0.925,
    bottom=0.105,
    hspace=0.53,
    wspace=0.34,
)

ax1 = fig.add_subplot(grid[0, 0])
ax2 = fig.add_subplot(grid[0, 1])
ax3 = fig.add_subplot(grid[1, 0])
ax4 = fig.add_subplot(grid[1, 1])


# ============================================================
# 7. Panel (a): signed continuum-grid drift
# ============================================================

N_scaled = N_values / 1000.0

for state in states:
    ax1.plot(
        N_scaled,
        grid_error_Hz[:, state],
        color=STATE_COLORS[state],
        marker="o",
        markersize=5.0,
        markerfacecolor="white",
        markeredgewidth=1.05,
        label=state_labels[state],
    )

ax1.axhline(0.0, color="0.18", linewidth=0.95)
ax1.set_title("Continuum-grid drift", pad=9)
ax1.set_xlabel(r"Grid points, $N$ $(10^3)$", labelpad=5)
ax1.set_ylabel(r"$E_n(N)-E_n^{(\infty)}$ (Hz)")
ax1.set_xticks(N_scaled)
ax1.set_xticklabels(["1.80", "2.56", "3.63", "5.16"])
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
ax1.legend(
    frameon=False,
    loc="lower right",
    ncol=1,
    handlelength=2.0,
    borderaxespad=0.35,
)
apply_panel_label(ax1, "a")
apply_axes_style(ax1)


# ============================================================
# 8. Panel (b): fourth-order scaling
# ============================================================

h_dense = np.geomspace(float(np.min(dr_nm)), float(np.max(dr_nm)), 300)

for state in states:
    ax2.loglog(
        dr_nm,
        abs_grid_error_Hz[:, state],
        color=STATE_COLORS[state],
        marker="o",
        markersize=5.0,
        markerfacecolor="white",
        markeredgewidth=1.05,
        linestyle="none",
    )

    fitted_error_Hz = (
        np.abs(fit_amplitude[state])
        * h_dense ** observed_orders[state]
        * 1.0e6
    )
    ax2.loglog(
        h_dense,
        fitted_error_Hz,
        color=STATE_COLORS[state],
        linewidth=1.35,
    )

# Compact two-line summary to avoid covering the data.
order_text = (
    r"$p=(4.000021,\ 3.999985,\ 3.999981)$"
    "\n"
    rf"$\min R^2={np.min(fit_R2):.9f}$"
)

ax2.text(
    0.040,
    0.955,
    order_text,
    transform=ax2.transAxes,
    ha="left",
    va="top",
    fontsize=7.5,
    bbox={
        "boxstyle": "round,pad=0.25",
        "facecolor": "white",
        "edgecolor": "0.72",
        "linewidth": 0.55,
        "alpha": 0.96,
    },
)

ax2.set_title("Fourth-order continuum scaling", pad=9)
ax2.set_xlabel(r"Grid spacing, $\Delta r$ (nm)", labelpad=5)
ax2.set_ylabel(r"$|E_n(N)-E_n^{(\infty)}|$ (Hz)")
ax2.xaxis.set_major_locator(LogLocator(base=10.0))
ax2.xaxis.set_minor_locator(
    LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
)
ax2.xaxis.set_minor_formatter(NullFormatter())
ax2.yaxis.set_major_locator(LogLocator(base=10.0))
ax2.yaxis.set_minor_locator(
    LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
)
ax2.yaxis.set_minor_formatter(NullFormatter())
apply_panel_label(ax2, "b")
apply_axes_style(ax2)


# ============================================================
# 9. Panel (c): sparse-versus-banded backend agreement
# ============================================================

ax3.bar(
    states,
    backend_diff_microHz,
    width=0.50,
    color=STATE_COLORS,
    edgecolor="0.25",
    linewidth=0.70,
)

ax3.axhline(0.0, color="0.18", linewidth=0.95)

for index, value in enumerate(backend_diff_microHz):
    annotate_value(
        ax3,
        float(index),
        float(value),
        rf"${value:+.3f}$",
        dy_points=6.0 if value >= 0.0 else -6.0,
        va="bottom" if value >= 0.0 else "top",
    )

backend_margin = max(
    1.7,
    0.40 * float(np.max(np.abs(backend_diff_microHz))),
)
ax3.set_ylim(
    float(np.min(backend_diff_microHz) - backend_margin),
    float(np.max(backend_diff_microHz) + backend_margin),
)
ax3.set_title("Independent FDM backends", pad=9)
ax3.set_xlabel(r"State index, $n$", labelpad=5)
ax3.set_ylabel(r"$E_{\rm sparse}-E_{\rm banded}$ ($\mu$Hz)")
ax3.set_xticks(states)
ax3.set_xticklabels(state_labels)
ax3.yaxis.set_minor_locator(AutoMinorLocator(2))
apply_panel_label(ax3, "c")
apply_axes_style(ax3)


# ============================================================
# 10. Panel (d): Numerov cross-check
# ============================================================

ax4.bar(
    states,
    numerov_diff_Hz,
    width=0.50,
    color=STATE_COLORS,
    edgecolor="0.25",
    linewidth=0.70,
)

ax4.axhline(0.0, color="0.18", linewidth=0.95)

for index, value in enumerate(numerov_diff_Hz):
    annotate_value(
        ax4,
        float(index),
        float(value),
        rf"${value:+.4f}$",
        dy_points=-6.0 if value < 0.0 else 6.0,
        va="top" if value < 0.0 else "bottom",
    )

numerov_margin = max(
    0.009,
    0.34 * float(np.max(np.abs(numerov_diff_Hz))),
)
ax4.set_ylim(
    float(np.min(numerov_diff_Hz) - numerov_margin),
    float(max(0.011, numerov_margin)),
)
ax4.set_title("Independent Numerov cross-check", pad=9)
ax4.set_xlabel(r"State index, $n$", labelpad=5)
ax4.set_ylabel(r"$E_{\rm Numerov}-E_{\rm FDM}$ (Hz)")
ax4.set_xticks(states)
ax4.set_xticklabels(state_labels)
ax4.yaxis.set_minor_locator(AutoMinorLocator(2))
apply_panel_label(ax4, "d")
apply_axes_style(ax4)


# ============================================================
# 11. Minimal footer
# ============================================================

# Keep the figure itself uncluttered. Full numerical conventions are stated
# in the manuscript caption; only the certified grid sequence is retained here.
fig.text(
    0.5,
    0.030,
    r"Certified continuum sequence: "
    r"$N=(1800,2556,3630,5155)$; "
    r"$0\leq r\leq650\,\mathrm{nm}$; "
    r"odd-reflection boundary closure.",
    ha="center",
    va="center",
    fontsize=7.6,
    color="0.18",
)


# ============================================================
# 12. Export
# ============================================================

pdf_path = OUTDIR / "Figure_5.pdf"
png_path = OUTDIR / "Figure_5.png"
svg_path = OUTDIR / "Figure_5.svg"

fig.savefig(pdf_path)
fig.savefig(png_path, dpi=600)
fig.savefig(svg_path)
plt.close(fig)


# ============================================================
# 13. Final self-check
# ============================================================

checks = {
    "grid_sequence": tuple(N_values.tolist()) == expected_grid,
    "origin_is_zero": math.isclose(
        float(manifest["r_min_nm"]),
        0.0,
        abs_tol=1.0e-15,
    ),
    "fourth_order": bool(np.all(np.abs(observed_orders - 4.0) < 5.0e-4)),
    "fit_quality": bool(np.all(fit_R2 > 0.9999999)),
    "backend_agreement": bool(
        np.max(np.abs(backend_diff_microHz)) < 10.0
    ),
    "numerov_agreement": bool(
        np.max(np.abs(numerov_diff_Hz)) < 0.05
    ),
}

if not all(checks.values()):
    raise RuntimeError(f"Figure 5 self-check failed: {checks}")

print("=== Figure 5 numerical certification — clean publication layout ===")
print(f"Manifest source: {manifest_source}")
print("Continuum levels (MHz):")
for state, energy in enumerate(E_inf_MHz):
    print(f"  n={state}: {energy:+.12f}")
print("Observed orders:", observed_orders)
print("Sparse-banded differences (microHz):", backend_diff_microHz)
print("Numerov-FDM differences (Hz):", numerov_diff_Hz)
print("SELF-CHECK: PASS")
print("Files written:")
for path in (pdf_path, png_path, svg_path, data_csv, summary_json):
    print(f"  - {path}")
