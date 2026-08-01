# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Two-parameter FDM map of the benchmark spectrum in (r_c, omega).
================================================================

Purpose
-------
This script extends the one-dimensional soft-core-radius sensitivity test to a
controlled two-parameter map in the soft-core radius r_c and the effective trap
frequency omega. For every pair (r_c, omega), the same finite-difference radial
Hamiltonian is solved and the following quantities are recorded:

    1) number of negative-energy states, N_-
    2) low-lying energies E_n/h in MHz
    3) ground-state binding magnitude |E_0|/h
    4) deviation from the central 15 MHz benchmark
    5) whether |E_0|/h lies inside the experimental band 15(2) MHz

Scientific convention
---------------------
- FDM is used as the primary numerical benchmark.
- This script does NOT recalibrate r_c.
- This script does NOT modify unified_model_params.py.
- This script changes only r_c and omega inside the local sweep.
- All other physical and numerical settings remain fixed unless changed below.
- WKB, variational, and perturbation-theory scripts are not touched here.

Outputs
-------
1) rc_omega_parameter_map_full.csv
2) rc_omega_E0abs_matrix.csv
3) rc_omega_Nbound_matrix.csv
4) rc_omega_parameter_map_summary.tex
5) rc_omega_parameter_map_figure.png
6) rc_omega_parameter_map_figure.pdf

Recommended workflow
--------------------
1) First run with QUICK_TEST = True.
2) After confirming that the script works, set QUICK_TEST = False for the final figure.

The QUICK_TEST mode uses fewer grid points and a smaller FDM matrix so you can verify
that the code runs quickly. The production mode uses the benchmark FDM grid settings.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import csv
import math
import time
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter

from unified_model_params import PARAMS, HBAR, HPLANCK, PI


# ============================================================
# 1. User-facing controls
# ============================================================

# Set True for a fast test. Set False for the final paper-quality run.
QUICK_TEST = False

# Central experimental benchmark: E_bind/h = 15(2) MHz.
TARGET_MHZ = 15.0
EXPERIMENTAL_BAND_MHZ = (13.0, 17.0)

# Central benchmark point already used in the paper.
RC_REF_NM = PARAMS.r_c * 1e9
OMEGA_REF_MHZ = PARAMS.omega_ion / (2.0 * PI * 1e6)

# Output directory: same folder as this script.
OUTDIR = Path(__file__).resolve().parent

# Parameter-map domain.
# The exact reference point is inserted automatically even if it is not exactly on the grid.
if QUICK_TEST:
    RC_GRID_NM = np.linspace(24.5, 27.0, 6)
    OMEGA_GRID_MHZ = np.linspace(0.8, 1.6, 5)
else:
    RC_GRID_NM = np.linspace(24.0, 28.0, 25)
    OMEGA_GRID_MHZ = np.linspace(0.8, 1.6, 25)

# Number of negative states to report explicitly in the CSV.
MAX_REPORTED_LEVELS = 5


# ============================================================
# 2. FDM settings
# ============================================================

@dataclass(frozen=True)
class FDMConfig:
    """Numerical FDM controls."""

    # Use the same physical radial box as the benchmark script.
    r_min: float = 1.0e-10
    r_max: float = 650.0e-9

    # QUICK_TEST is intended only for code checking, not final paper data.
    N: int = 5000 if QUICK_TEST else 12000

    # Shift-invert center. This is close to the 15 MHz binding scale.
    sigma: float = -1.0e-26

    # Number of eigenpairs requested near sigma.
    # Increase if a wider parameter map produces more negative states.
    k: int = 10

    eig_tol: float = 1.0e-11
    maxiter: int = 30000


@dataclass
class ModelParams:
    """Local model parameters for one FDM solve."""

    # swept quantities
    r_c: float
    omega_ion: float

    # solver/grid-specific quantities
    r_min: float
    r_max: float
    N: int
    sigma: float = -1.0e-26

    # shared physical parameters imported from unified_model_params.py
    m_atom: float = PARAMS.m_atom
    m_ion: float = PARAMS.m_ion
    C4: float = PARAMS.C4
    l: int = PARAMS.l
    use_langer: bool = PARAMS.use_langer_numerical


# ============================================================
# 3. Physics helpers
# ============================================================

def reduced_mass(p: ModelParams) -> float:
    return p.m_atom * p.m_ion / (p.m_atom + p.m_ion)


def l_eff(p: ModelParams) -> float:
    return (p.l + 0.5) ** 2 if p.use_langer else p.l * (p.l + 1.0)


def alpha_pol(p: ModelParams) -> float:
    return -0.5 * p.C4


def harmonic_length(p: ModelParams) -> float:
    mu = reduced_mass(p)
    return math.sqrt(HBAR / (mu * p.omega_ion))


def rc_dimensionless(p: ModelParams) -> float:
    return p.r_c / harmonic_length(p)


def alpha_dimensionless(p: ModelParams) -> float:
    a_ho = harmonic_length(p)
    return alpha_pol(p) / (HBAR * p.omega_ion * a_ho**4)


def effective_potential(r: np.ndarray, p: ModelParams) -> np.ndarray:
    """Effective radial potential used in the FDM benchmark Hamiltonian."""
    mu = reduced_mass(p)
    alpha = alpha_pol(p)
    leff = l_eff(p)

    v_harm = 0.5 * mu * (p.omega_ion ** 2) * r ** 2
    v_core = alpha / (r ** 4 + p.r_c ** 4)

    if leff != 0.0:
        v_cent = (HBAR ** 2 * leff) / (2.0 * mu * r ** 2)
    else:
        v_cent = np.zeros_like(r)

    return v_harm + v_core + v_cent


# ============================================================
# 4. FDM operator and solver
# ============================================================

@dataclass
class FDMGridCache:
    """Reusable grid and kinetic-energy matrix for the same FDM box."""

    cfg: FDMConfig
    r: np.ndarray
    r_int: np.ndarray
    interior_slice: Tuple[int, int]
    tmat: sp.csr_matrix


def build_grid_cache(cfg: FDMConfig, p_template: ModelParams) -> FDMGridCache:
    """Build grid and kinetic-energy operator once, then reuse for all map points."""
    if cfg.N < 7:
        raise ValueError("N must be at least 7 for a five-point stencil.")

    r = np.linspace(cfg.r_min, cfg.r_max, cfg.N, dtype=float)
    dr = r[1] - r[0]
    mu = reduced_mass(p_template)

    # Five-point second derivative requires excluding two points at each edge.
    i0, i1 = 2, cfg.N - 3
    idx = np.arange(i0, i1 + 1)
    r_int = r[idx]
    m = r_int.size

    inv_h2 = 1.0 / (dr * dr)
    c0 = -30.0 / 12.0 * inv_h2
    c1 = +16.0 / 12.0 * inv_h2
    c2 = -1.0 / 12.0 * inv_h2

    lap = sp.diags(
        diagonals=[
            c2 * np.ones(m - 2),
            c1 * np.ones(m - 1),
            c0 * np.ones(m),
            c1 * np.ones(m - 1),
            c2 * np.ones(m - 2),
        ],
        offsets=[-2, -1, 0, 1, 2],
        format="csr",
    )

    tmat = (-(HBAR ** 2) / (2.0 * mu) * lap).tocsr()
    return FDMGridCache(cfg=cfg, r=r, r_int=r_int, interior_slice=(i0, i1), tmat=tmat)


def normalize_u(u: np.ndarray, r: np.ndarray) -> np.ndarray:
    norm = math.sqrt(np.trapezoid(np.abs(u) ** 2, r))
    if norm <= 0.0 or not np.isfinite(norm):
        raise FloatingPointError("Normalization failed.")
    return u / norm


def count_nodes(v: np.ndarray) -> int:
    amp = np.max(np.abs(v))
    eps = max(1e-14, 1e-8 * amp)
    s = np.sign(np.where(np.abs(v) < eps, 0.0, v))
    return int(np.sum(s[1:] * s[:-1] < 0))


def fdm_eigen_residual(hmat: sp.csr_matrix, vec_int: np.ndarray, E_J: float) -> float:
    res = hmat @ vec_int - E_J * vec_int
    denom = max(1e-30, abs(E_J) * np.linalg.norm(vec_int))
    return float(np.linalg.norm(res) / denom)


def solve_fdm_states(p: ModelParams, cache: FDMGridCache, cfg: FDMConfig) -> Dict[str, object]:
    """Solve the low-lying FDM spectrum near cfg.sigma for one parameter pair."""
    v_int = effective_potential(cache.r_int, p)
    hmat = cache.tmat + sp.diags(v_int, format="csr")

    dim = hmat.shape[0]
    k_eff = min(cfg.k, dim - 2)
    ncv = min(dim, max(2 * k_eff + 8, 20))

    try:
        evals, evecs = spla.eigsh(
            hmat,
            k=k_eff,
            sigma=p.sigma,
            which="LM",
            tol=cfg.eig_tol,
            maxiter=cfg.maxiter,
            ncv=ncv,
        )
    except Exception as exc:
        raise RuntimeError(
            f"eigsh failed for r_c={p.r_c * 1e9:.6f} nm, "
            f"omega/2pi={p.omega_ion / (2 * PI * 1e6):.6f} MHz"
        ) from exc

    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    i0, i1 = cache.interior_slice
    states: List[Dict[str, object]] = []

    for j, E_J in enumerate(evals):
        vec_int = evecs[:, j].copy()

        u = np.zeros_like(cache.r)
        u[i0 : i1 + 1] = vec_int
        u = normalize_u(u, cache.r)

        states.append(
            {
                "index": j,
                "E_J": float(E_J),
                "E_over_h_MHz": float(E_J / HPLANCK / 1.0e6),
                "absE_over_h_MHz": float(abs(E_J) / HPLANCK / 1.0e6),
                "nodes": count_nodes(u[i0 : i1 + 1]),
                "eig_residual_rel": fdm_eigen_residual(hmat, vec_int, float(E_J)),
            }
        )

    states.sort(key=lambda s: float(s["E_J"]))
    negative_states = [s for s in states if float(s["E_J"]) < 0.0]

    return {
        "params": asdict(p),
        "states": states,
        "negative_states": negative_states,
        "n_negative": len(negative_states),
    }


# ============================================================
# 5. Parameter-map utilities
# ============================================================

def include_reference_point(values: Sequence[float], reference: float, decimals: int = 9) -> np.ndarray:
    arr = np.array(list(values) + [reference], dtype=float)
    arr = np.unique(np.round(arr, decimals=decimals))
    arr.sort()
    return arr


def make_model_params(rc_nm: float, omega_mhz: float, cfg: FDMConfig) -> ModelParams:
    return ModelParams(
        r_c=float(rc_nm) * 1.0e-9,
        omega_ion=2.0 * PI * float(omega_mhz) * 1.0e6,
        r_min=cfg.r_min,
        r_max=cfg.r_max,
        N=cfg.N,
        sigma=cfg.sigma,
    )


def row_for_pair(rc_nm: float, omega_mhz: float, cfg: FDMConfig, cache: FDMGridCache) -> Dict[str, float | int | bool]:
    p = make_model_params(rc_nm, omega_mhz, cfg)
    result = solve_fdm_states(p, cache=cache, cfg=cfg)
    neg = result["negative_states"]

    row: Dict[str, float | int | bool] = {
        "rc_nm": float(rc_nm),
        "omega_over_2pi_MHz": float(omega_mhz),
        "omega_rad_s": float(p.omega_ion),
        "a_ho_nm": float(harmonic_length(p) * 1e9),
        "x_c": float(rc_dimensionless(p)),
        "alpha_prime": float(alpha_dimensionless(p)),
        "n_negative": int(result["n_negative"]),
    }

    for i in range(MAX_REPORTED_LEVELS):
        row[f"E{i}_MHz"] = float(neg[i]["E_over_h_MHz"]) if len(neg) > i else float("nan")
        row[f"E{i}_abs_MHz"] = float(neg[i]["absE_over_h_MHz"]) if len(neg) > i else float("nan")
        row[f"nodes{i}"] = int(neg[i]["nodes"]) if len(neg) > i else -1

    e0_abs = float(row["E0_abs_MHz"])
    row["delta15_MHz"] = abs(e0_abs - TARGET_MHZ) if np.isfinite(e0_abs) else float("nan")
    row["delta15_pct"] = 100.0 * abs(e0_abs - TARGET_MHZ) / TARGET_MHZ if np.isfinite(e0_abs) else float("nan")
    row["inside_13_17_MHz_band"] = bool(
        np.isfinite(e0_abs) and EXPERIMENTAL_BAND_MHZ[0] <= e0_abs <= EXPERIMENTAL_BAND_MHZ[1]
    )
    row["ground_eig_residual_rel"] = float(neg[0]["eig_residual_rel"]) if neg else float("nan")

    return row


def run_parameter_map() -> Tuple[List[Dict[str, float | int | bool]], np.ndarray, np.ndarray]:
    cfg = FDMConfig()

    rc_values_nm = include_reference_point(RC_GRID_NM, RC_REF_NM)
    omega_values_mhz = include_reference_point(OMEGA_GRID_MHZ, OMEGA_REF_MHZ)

    p_template = make_model_params(RC_REF_NM, OMEGA_REF_MHZ, cfg)
    cache = build_grid_cache(cfg, p_template)

    n_total = len(rc_values_nm) * len(omega_values_mhz)
    rows: List[Dict[str, float | int | bool]] = []
    t0 = time.perf_counter()

    print("=== Two-parameter FDM map in (r_c, omega) ===")
    print(f"QUICK_TEST                 : {QUICK_TEST}")
    print(f"FDM grid N                 : {cfg.N}")
    print(f"FDM radial box             : r_min={cfg.r_min * 1e9:.6f} nm, r_max={cfg.r_max * 1e9:.6f} nm")
    print(f"Requested eigenpairs k      : {cfg.k}")
    print(f"r_c grid points             : {len(rc_values_nm)}")
    print(f"omega grid points           : {len(omega_values_mhz)}")
    print(f"total FDM solves            : {n_total}")
    print(f"reference r_c               : {RC_REF_NM:.6f} nm")
    print(f"reference omega/2pi         : {OMEGA_REF_MHZ:.6f} MHz")
    print("No recalibration is performed. Only r_c and omega are swept.\n")

    counter = 0
    for omega_mhz in omega_values_mhz:
        for rc_nm in rc_values_nm:
            counter += 1
            print(
                f"[{counter:4d}/{n_total:4d}] "
                f"r_c={rc_nm:10.6f} nm, omega/2pi={omega_mhz:7.4f} MHz ...",
                flush=True,
            )
            row = row_for_pair(rc_nm, omega_mhz, cfg=cfg, cache=cache)
            rows.append(row)

    elapsed = time.perf_counter() - t0
    print(f"\nCompleted map in {elapsed / 60.0:.2f} min.")
    return rows, rc_values_nm, omega_values_mhz


# ============================================================
# 6. Output writers
# ============================================================

def write_full_csv(rows: List[Dict[str, float | int | bool]]) -> Path:
    path = OUTDIR / "rc_omega_parameter_map_full.csv"

    base_cols = [
        "rc_nm",
        "omega_over_2pi_MHz",
        "omega_rad_s",
        "a_ho_nm",
        "x_c",
        "alpha_prime",
        "n_negative",
    ]
    level_cols: List[str] = []
    for i in range(MAX_REPORTED_LEVELS):
        level_cols += [f"E{i}_MHz", f"E{i}_abs_MHz", f"nodes{i}"]
    tail_cols = ["delta15_MHz", "delta15_pct", "inside_13_17_MHz_band", "ground_eig_residual_rel"]
    cols = base_cols + level_cols + tail_cols

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in cols})
    return path


def matrix_from_rows(
    rows: List[Dict[str, float | int | bool]],
    rc_values_nm: np.ndarray,
    omega_values_mhz: np.ndarray,
    key: str,
) -> np.ndarray:
    matrix = np.full((len(omega_values_mhz), len(rc_values_nm)), np.nan, dtype=float)

    rc_to_j = {round(float(v), 9): j for j, v in enumerate(rc_values_nm)}
    om_to_i = {round(float(v), 9): i for i, v in enumerate(omega_values_mhz)}

    for r in rows:
        i = om_to_i[round(float(r["omega_over_2pi_MHz"]), 9)]
        j = rc_to_j[round(float(r["rc_nm"]), 9)]
        matrix[i, j] = float(r[key])

    return matrix


def write_matrix_csv(matrix: np.ndarray, rc_values_nm: np.ndarray, omega_values_mhz: np.ndarray, filename: str) -> Path:
    path = OUTDIR / filename
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["omega_over_2pi_MHz \\ rc_nm"] + [f"{x:.9f}" for x in rc_values_nm])
        for omega, row in zip(omega_values_mhz, matrix):
            writer.writerow([f"{omega:.9f}"] + [f"{val:.12g}" for val in row])
    return path


def write_summary_latex(rows: List[Dict[str, float | int | bool]]) -> Path:
    """Write a compact table around the official reference omega only."""
    path = OUTDIR / "rc_omega_parameter_map_summary.tex"

    # Select the row nearest the official omega for each r_c.
    target_omega = OMEGA_REF_MHZ
    candidates = sorted(
        rows,
        key=lambda r: (abs(float(r["omega_over_2pi_MHz"]) - target_omega), float(r["rc_nm"])),
    )
    omega_selected = float(candidates[0]["omega_over_2pi_MHz"])
    slice_rows = [r for r in rows if abs(float(r["omega_over_2pi_MHz"]) - omega_selected) < 5e-10]
    slice_rows = sorted(slice_rows, key=lambda r: float(r["rc_nm"]))

    with path.open("w", encoding="utf-8") as f:
        f.write(r"""\begin{table}[t]
\centering
\caption{Representative slice of the two-parameter FDM map at fixed $\omega/2\pi \simeq 1.2~\mathrm{MHz}$. The sweep is not a recalibration procedure; only $r_c$ and $\omega$ are varied locally to test the spectral stability of the benchmark-calibrated soft-core Hamiltonian.}
\label{tab:rc_omega_map_slice}
\begin{tabular}{c c c c c}
\hline
$r_c$ (nm) & $\omega/2\pi$ (MHz) & $|E_0|/h$ (MHz) & $N_-$ & Inside $15(2)$ MHz? \\
\hline
""")
        for r in slice_rows:
            inside = "yes" if bool(r["inside_13_17_MHz_band"]) else "no"
            f.write(
                f"{float(r['rc_nm']):.6f} & "
                f"{float(r['omega_over_2pi_MHz']):.6f} & "
                f"{float(r['E0_abs_MHz']):.6f} & "
                f"{int(r['n_negative'])} & "
                f"{inside} \\\\n"
            )
        f.write(r"""\hline
\end{tabular}
\end{table}
""")
    return path




# ============================================================
# 7. Figure generation -- stable PRA/Q1 version
# ============================================================

# ============================================================
# 7. Figure generation -- stable PRA/Q1 version
# ============================================================

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
from matplotlib.lines import Line2D


def set_publication_style():
    mpl.rcParams.update({
        "figure.dpi": 170,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,

        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "text.usetex": False,

        "font.size": 9.0,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 7.4,

        "axes.linewidth": 0.75,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.major.width": 0.70,
        "ytick.major.width": 0.70,
        "xtick.minor.width": 0.55,
        "ytick.minor.width": 0.55,

        "axes.grid": False,
        "legend.frameon": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def add_panel_label(ax, label):
    ax.text(
        0.035, 0.955, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9.2,
        fontweight="bold",
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.16",
            facecolor="white",
            edgecolor="0.72",
            linewidth=0.45,
            alpha=0.92,
        ),
        zorder=50,
    )


def add_reference_guides(ax):
    ax.axvline(RC_REF_NM, color="0.40", lw=0.75, ls="--", zorder=12)
    ax.axhline(OMEGA_REF_MHZ, color="0.40", lw=0.75, ls="--", zorder=12)


def add_reference_star(ax):
    ax.scatter(
        RC_REF_NM,
        OMEGA_REF_MHZ,
        marker="*",
        s=185,
        facecolor="#FFD43B",
        edgecolor="black",
        linewidth=0.95,
        zorder=60,
    )
    ax.scatter(
        RC_REF_NM,
        OMEGA_REF_MHZ,
        marker="*",
        s=105,
        facecolor="#FFD43B",
        edgecolor="white",
        linewidth=0.55,
        zorder=61,
    )


def safe_contour(ax, X, Y, Z, levels, **kwargs):
    finite = np.asarray(Z)[np.isfinite(Z)]
    if finite.size == 0:
        return None

    zmin = float(np.nanmin(finite))
    zmax = float(np.nanmax(finite))

    valid_levels = []
    for lvl in levels:
        lvl = float(lvl)
        if zmin < lvl < zmax:
            valid_levels.append(lvl)

    if not valid_levels:
        return None

    try:
        return ax.contour(X, Y, Z, levels=valid_levels, **kwargs)
    except Exception:
        return None


def add_binding_contours(ax, RC, OM, Z):
    cs_band = safe_contour(
        ax, RC, OM, Z,
        [EXPERIMENTAL_BAND_MHZ[0], EXPERIMENTAL_BAND_MHZ[1]],
        colors="0.10",
        linewidths=0.85,
        linestyles="dashed",
        zorder=20,
    )

    if cs_band is not None:
        try:
            ax.clabel(
                cs_band,
                fmt={
                    float(EXPERIMENTAL_BAND_MHZ[0]): r"$13$",
                    float(EXPERIMENTAL_BAND_MHZ[1]): r"$17$",
                },
                inline=True,
                fontsize=7.2,
                colors="0.10",
            )
        except Exception:
            pass

    safe_contour(
        ax, RC, OM, Z,
        [TARGET_MHZ],
        colors="black",
        linewidths=1.35,
        linestyles="solid",
        zorder=21,
    )


def format_common_axis(ax):
    ax.set_xlim(24.0, 28.0)
    ax.set_ylim(0.8, 1.6)

    ax.set_xticks(np.arange(24.0, 28.1, 1.0))
    ax.set_yticks(np.arange(0.8, 1.61, 0.2))

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", top=True, right=True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.75)


def make_discrete_cmap(n_categories):
    base_colors = [
        "#2F5D8C",
        "#2F9C95",
        "#C24A73",
        "#E4BC42",
        "#7A5AA6",
        "#82A85B",
        "#9A6A45",
        "#607D8B",
    ]

    if n_categories <= len(base_colors):
        return ListedColormap(base_colors[:n_categories])

    sampled = plt.get_cmap("tab20")(np.linspace(0.05, 0.95, n_categories))
    return ListedColormap(sampled)


def write_parameter_map_figure(
    E0abs_matrix,
    Nbound_matrix,
    rc_values_nm,
    omega_values_mhz,
):
    set_publication_style()

    RC, OM = np.meshgrid(rc_values_nm, omega_values_mhz)

    # أعرض شوي حتى نخلق مسافة بين الرسمتين
    fig = plt.figure(figsize=(7.85, 3.85))

    # ax1 | cbar1 | spacer | ax2 | cbar2
    gs = fig.add_gridspec(
        nrows=2,
        ncols=5,
        height_ratios=[1.0, 0.14],
        width_ratios=[1.0, 0.042, 0.24, 1.0, 0.042],
        left=0.065,
        right=0.985,
        bottom=0.095,
        top=0.900,
        wspace=0.045,
        hspace=0.28,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    cax1 = fig.add_subplot(gs[0, 1])

    spacer = fig.add_subplot(gs[0, 2])
    spacer.axis("off")

    ax2 = fig.add_subplot(gs[0, 3], sharey=ax1)
    cax2 = fig.add_subplot(gs[0, 4])

    legax = fig.add_subplot(gs[1, :])
    legax.axis("off")

    # --------------------------------------------------------
    # Panel (a): |E0|/h
    # --------------------------------------------------------
    finite_e = np.asarray(E0abs_matrix)[np.isfinite(E0abs_matrix)]
    if finite_e.size == 0:
        raise RuntimeError("No finite E0 values are available for plotting.")

    e_min = float(np.nanmin(finite_e))
    e_max = float(np.nanmax(finite_e))

    if abs(e_max - e_min) < 1e-12:
        e_levels = np.linspace(e_min - 0.5, e_max + 0.5, 50)
    else:
        e_levels = np.linspace(
            np.floor(2.0 * e_min) / 2.0,
            np.ceil(2.0 * e_max) / 2.0,
            80,
        )

    cf = ax1.contourf(
        RC,
        OM,
        E0abs_matrix,
        levels=e_levels,
        cmap="viridis",
        extend="both",
    )

    add_binding_contours(ax1, RC, OM, E0abs_matrix)
    add_reference_guides(ax1)
    add_reference_star(ax1)

    cb1 = fig.colorbar(cf, cax=cax1)
    cb1.set_ticks([10, 12, 14, 16, 18, 20])
    cb1.ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    cb1.set_label(r"$|E_0|/h$ (MHz)", labelpad=4)
    cb1.ax.tick_params(length=2.4, width=0.65)

    ax1.set_title(r"Ground-state binding $|E_0|/h$", pad=5)
    ax1.set_xlabel(r"Soft-core radius $r_c$ (nm)", labelpad=2)
    ax1.set_ylabel(r"Trap frequency $\omega/2\pi$ (MHz)", labelpad=4)
    add_panel_label(ax1, r"(a)")
    format_common_axis(ax1)

    # --------------------------------------------------------
    # Panel (b): N_-
    # --------------------------------------------------------
    finite_n = np.asarray(Nbound_matrix)[np.isfinite(Nbound_matrix)]
    if finite_n.size == 0:
        raise RuntimeError("No finite N_- values are available for plotting.")

    nmin = int(np.nanmin(finite_n))
    nmax = int(np.nanmax(finite_n))
    n_categories = max(1, nmax - nmin + 1)

    cmap_n = make_discrete_cmap(n_categories)
    boundaries = np.arange(nmin - 0.5, nmax + 1.5, 1.0)
    norm_n = BoundaryNorm(boundaries, cmap_n.N)

    pm = ax2.pcolormesh(
        RC,
        OM,
        Nbound_matrix,
        shading="auto",
        cmap=cmap_n,
        norm=norm_n,
        rasterized=True,
    )

    boundary_levels = np.arange(nmin + 0.5, nmax + 0.5, 1.0)
    if boundary_levels.size > 0:
        safe_contour(
            ax2, RC, OM, Nbound_matrix,
            boundary_levels,
            colors="white",
            linewidths=1.15,
            zorder=18,
        )
        safe_contour(
            ax2, RC, OM, Nbound_matrix,
            boundary_levels,
            colors="0.25",
            linewidths=0.35,
            zorder=19,
        )

    add_reference_guides(ax2)
    add_reference_star(ax2)

    cb2 = fig.colorbar(pm, cax=cax2, ticks=np.arange(nmin, nmax + 1))
    cb2.set_label(r"$N_{-}$", rotation=0, labelpad=11)
    cb2.ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    cb2.ax.tick_params(length=2.4, width=0.65)

    ax2.set_title(r"Negative-energy state count $N_{-}$", pad=5)
    ax2.set_xlabel(r"Soft-core radius $r_c$ (nm)", labelpad=2)

    # مهم جداً: نحذف تكرار y-label و y tick labels من الرسم الثاني
    # حتى لا يتداخل مع colorbar تبع الرسم الأول.
    ax2.set_ylabel("")
    ax2.tick_params(labelleft=False)

    add_panel_label(ax2, r"(b)")
    format_common_axis(ax2)

    # --------------------------------------------------------
    # Shared legend
    # --------------------------------------------------------
    legend_handles = [
        Line2D(
            [0], [0],
            marker="*",
            markersize=10.5,
            markerfacecolor="#FFD43B",
            markeredgecolor="black",
            markeredgewidth=0.75,
            linestyle="None",
            label=(
                r"Benchmark: "
                rf"$r_c={RC_REF_NM:.6f}\,\mathrm{{nm}}$, "
                rf"$\omega/2\pi={OMEGA_REF_MHZ:.1f}\,\mathrm{{MHz}}$"
            ),
        ),
        Line2D(
            [0], [0],
            color="black",
            lw=1.35,
            label=rf"{TARGET_MHZ:.0f} MHz target contour",
        ),
        Line2D(
            [0], [0],
            color="0.10",
            lw=0.95,
            linestyle="--",
            label=(
                rf"{EXPERIMENTAL_BAND_MHZ[0]:.0f}--"
                rf"{EXPERIMENTAL_BAND_MHZ[1]:.0f} MHz binding band"
            ),
        ),
        Line2D(
            [0], [0],
            color="0.25",
            lw=1.15,
            label=r"$N_{-}$ region boundary",
        ),
    ]

    leg = legax.legend(
        handles=legend_handles,
        loc="center",
        ncol=4,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="0.35",
        borderpad=0.55,
        handlelength=2.55,
        columnspacing=1.05,
        handletextpad=0.60,
    )
    leg.get_frame().set_linewidth(0.55)
    leg.get_frame().set_facecolor("white")

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    pdf_path = OUTDIR / "rc_omega_parameter_map_figure_PRA.pdf"
    png_path = OUTDIR / "rc_omega_parameter_map_figure_PRA.png"
    svg_path = OUTDIR / "rc_omega_parameter_map_figure_PRA.svg"

    fig.savefig(pdf_path)
    fig.savefig(png_path)
    fig.savefig(svg_path)

    plt.close(fig)

    return pdf_path, png_path


# ============================================================
# 8. Reporting
# ============================================================

def print_reference_row(rows: List[Dict[str, float | int | bool]]) -> None:
    best = min(
        rows,
        key=lambda r: (
            abs(float(r["rc_nm"]) - RC_REF_NM),
            abs(float(r["omega_over_2pi_MHz"]) - OMEGA_REF_MHZ),
        ),
    )
    print("\nClosest row to the official benchmark point:")
    print(f"  r_c [nm]            : {float(best['rc_nm']):.9f}")
    print(f"  omega/2pi [MHz]    : {float(best['omega_over_2pi_MHz']):.9f}")
    print(f"  a_ho [nm]           : {float(best['a_ho_nm']):.9f}")
    print(f"  x_c                 : {float(best['x_c']):.9f}")
    print(f"  alpha_prime         : {float(best['alpha_prime']):.9f}")
    print(f"  N_negative          : {int(best['n_negative'])}")
    print(f"  E0/h [MHz]          : {float(best['E0_MHz']):.9f}")
    print(f"  |E0|/h [MHz]        : {float(best['E0_abs_MHz']):.9f}")
    print(f"  delta15 [%]         : {float(best['delta15_pct']):.6e}")
    print(f"  inside 13-17 MHz    : {bool(best['inside_13_17_MHz_band'])}")
    print(f"  eig residual        : {float(best['ground_eig_residual_rel']):.3e}")


def warn_if_too_few_eigenpairs(rows: List[Dict[str, float | int | bool]], cfg: FDMConfig) -> None:
    max_neg = max(int(r["n_negative"]) for r in rows)
    if max_neg >= cfg.k:
        warnings.warn(
            "The maximum number of negative states is equal to the number of requested eigenpairs. "
            "Increase FDMConfig.k to make sure no negative levels are missed.",
            RuntimeWarning,
        )


# ============================================================
# 9. Main entry point
# ============================================================

def main() -> None:
    cfg = FDMConfig()

    rows, rc_values_nm, omega_values_mhz = run_parameter_map()
    warn_if_too_few_eigenpairs(rows, cfg)
    print_reference_row(rows)

    E0abs_matrix = matrix_from_rows(rows, rc_values_nm, omega_values_mhz, "E0_abs_MHz")
    Nbound_matrix = matrix_from_rows(rows, rc_values_nm, omega_values_mhz, "n_negative")

    full_csv = write_full_csv(rows)
    e0_csv = write_matrix_csv(E0abs_matrix, rc_values_nm, omega_values_mhz, "rc_omega_E0abs_matrix.csv")
    nb_csv = write_matrix_csv(Nbound_matrix, rc_values_nm, omega_values_mhz, "rc_omega_Nbound_matrix.csv")
    tex_path = write_summary_latex(rows)
    pdf_path, png_path = write_parameter_map_figure(E0abs_matrix, Nbound_matrix, rc_values_nm, omega_values_mhz)

    print("\nFiles written:")
    print(f"  Full CSV      : {full_csv}")
    print(f"  E0 matrix CSV : {e0_csv}")
    print(f"  N- matrix CSV : {nb_csv}")
    print(f"  LaTeX table   : {tex_path}")
    print(f"  Figure PDF    : {pdf_path}")
    print(f"  Figure PNG    : {png_path}")

    if QUICK_TEST:
        print("\nNote: QUICK_TEST=True. For final paper-quality output, set QUICK_TEST=False and rerun.")


if __name__ == "__main__":
    main()
