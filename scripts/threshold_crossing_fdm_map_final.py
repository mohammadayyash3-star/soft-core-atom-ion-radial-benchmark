from __future__ import annotations



from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import csv
import math
import time
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter

# The project parameter file should be in the same working directory or Python path.
try:
    from unified_model_params import PARAMS, HBAR, HPLANCK, PI
except Exception as exc:  # pragma: no cover - user-facing error message
    raise ImportError(
        "This script requires unified_model_params.py in the same folder or PYTHONPATH. "
        "Place this file beside your existing research codes and rerun."
    ) from exc


# ============================================================
# 1. User-facing controls
# ============================================================

# Choose one of: "quick", "draft", "paper", "production".
# Start with "quick" to verify that everything works.
RUN_MODE = "production"

# Central experimental benchmark: E_bind/h = 15(2) MHz.
TARGET_MHZ = 15.0
EXPERIMENTAL_BAND_MHZ = (13.0, 17.0)

# Central benchmark point already used in the paper.
RC_REF_NM = PARAMS.r_c * 1e9
OMEGA_REF_MHZ = PARAMS.omega_ion / (2.0 * PI * 1e6)

# Scale the sparse eigenproblem from joules to MHz before calling ARPACK.
# This avoids premature convergence caused by O(1e-26 J) matrix eigenvalues
# and makes both smallest-algebraic and shift-invert modes numerically reliable.
FDM_EIGEN_ENERGY_SCALE_J = HPLANCK * 1.0e6

# Output directory: same folder as this script.
OUTDIR = Path(__file__).resolve().parent

# Parameter-map domain for the zero-energy spectral analysis.
# This wider domain is intentional: it helps reveal the boundaries where N_- changes.
RC_DOMAIN_NM = (22.0, 32.0)
OMEGA_DOMAIN_MHZ = (0.50, 2.00)

# Number of signed levels L_n stored and plotted.
# Increase if the selected domain supports more negative states.
MAX_LEVELS = 6

# The map may use shift-invert acceleration; direct zero-energy crossings are
# independently refined with a smallest-algebraic solve on the scaled operator.
# Options: "lowest" or "shift_invert".
EIGEN_SOLVER_MODE = "shift_invert"
# Critical convention:
# The benchmark FDM Hamiltonian must use the ordinary radial centrifugal factor l(l+1).
# The Langer replacement (l+1/2)^2 is a WKB-only correction and must NOT enter this FDM map.
FORCE_STANDARD_FDM_CENTRIFUGAL = True

# Prevent accidentally publishing a map that does not reproduce the official benchmark point.
# The check is intentionally loose at draft resolution and should become tighter for final runs.
ENFORCE_BENCHMARK_VALIDATION = True
BENCHMARK_EXPECTED_NNEG = 3
BENCHMARK_E0_MHZ = -15.0
BENCHMARK_E0_TOL_MHZ = 0.50

# Fixed-frequency slice.  The lower limit is chosen to retain the full E0 curve
# over the displayed r_c interval, while the upper limit keeps the zero-energy
# structure readable.  Positive levels above this window remain trap confined.
SLICE_YLIM_MHZ = (-32.0, 12.0)
MAX_SLICE_LEVELS_TO_PLOT = 5

# Robust node counting and Sturm-order diagnostics.
NODE_RELATIVE_CUTOFFS = (1.0e-6, 3.0e-7, 1.0e-7, 3.0e-8, 1.0e-8)
NODE_ABSOLUTE_CUTOFF = 1.0e-14
NODE_TAIL_GUARD_POINTS = 4
ENFORCE_REFERENCE_NODE_ORDER = True

# Direct zero-energy crossing refinement on the fixed omega slice.
REFINE_ZERO_ENERGY_CROSSINGS = True
CROSSING_SOLVER_MODE = "shift_invert"  # fast direct FDM refinement with node-order checks
CROSSING_ROOT_XTOL_NM = 2.0e-5
CROSSING_ROOT_RTOL = 1.0e-10
CROSSING_ENERGY_RESIDUAL_TOL_MHZ = 1.0e-5
CROSSING_BRACKET_PAD_NM = 0.30
CROSSING_REPORT_DECIMALS = 4

# Dedicated crossing-convergence tests.  These are intentionally separate from
# ground-state convergence because E_n≈0 levels are the relevant observables here.
RUN_ZERO_ENERGY_CONVERGENCE = RUN_MODE in {"paper", "production"}
CROSSING_N_SWEEP = (9000, 12000, 15000)
CROSSING_RMAX_SWEEP_NM = (500.0, 650.0, 800.0)
CROSSING_SIGMA_SWEEP_J = (-2.0e-26, -1.0e-26, -5.0e-27)
CROSSING_CONVERGENCE_TOL_NM = 2.0e-2


# ============================================================
# 2. Presets and FDM settings
# ============================================================

@dataclass(frozen=True)
class Preset:
    n_rc: int
    n_omega: int
    N: int
    k: int
    eig_tol: float
    maxiter: int


PRESETS: Dict[str, Preset] = {
    # Fast sanity check. Not for final numerical values.
    "quick": Preset(n_rc=9, n_omega=7, N=1800, k=MAX_LEVELS + 4, eig_tol=1.0e-9, maxiter=12000),
    # Useful for testing the layout and approximate zero-energy locations.
    "draft": Preset(n_rc=25, n_omega=21, N=4000, k=MAX_LEVELS + 4, eig_tol=3.0e-10, maxiter=20000),
    # Reasonable paper-quality map on a workstation.
    "paper": Preset(n_rc=41, n_omega=35, N=7000, k=MAX_LEVELS + 6, eig_tol=1.0e-10, maxiter=30000),
    # Expensive high-resolution map. Consider running overnight.
    "production": Preset(n_rc=61, n_omega=51, N=12000, k=MAX_LEVELS + 8, eig_tol=1.0e-11, maxiter=50000),
}


@dataclass(frozen=True)
class FDMConfig:
    """Numerical FDM controls."""

    r_min: float = 1.0e-10
    r_max: float = 650.0e-9
    N: int = PRESETS[RUN_MODE].N

    # Used only if EIGEN_SOLVER_MODE == "shift_invert".
    sigma: float = -1.0e-26

    k: int = PRESETS[RUN_MODE].k
    eig_tol: float = PRESETS[RUN_MODE].eig_tol
    maxiter: int = PRESETS[RUN_MODE].maxiter
    solver_mode: str = EIGEN_SOLVER_MODE


@dataclass
class ModelParams:
    """Local model parameters for one FDM solve."""

    # Swept quantities.
    r_c: float
    omega_ion: float

    # Solver/grid-specific quantities.
    r_min: float
    r_max: float
    N: int
    sigma: float = -1.0e-26

    # Shared physical parameters imported from unified_model_params.py.
    m_atom: float = PARAMS.m_atom
    m_ion: float = PARAMS.m_ion
    C4: float = PARAMS.C4
    l: int = PARAMS.l
    use_langer: bool = False if FORCE_STANDARD_FDM_CENTRIFUGAL else PARAMS.use_langer_numerical


# ============================================================
# 3. Physics helpers
# ============================================================

def reduced_mass(p: ModelParams) -> float:
    return p.m_atom * p.m_ion / (p.m_atom + p.m_ion)


def l_eff(p: ModelParams) -> float:
    """Centrifugal factor used in the numerical FDM Hamiltonian.

    For the fixed-Hamiltonian benchmark, the FDM reference must use l(l+1).
    The Langer replacement is reserved for WKB only.
    """
    if FORCE_STANDARD_FDM_CENTRIFUGAL:
        return p.l * (p.l + 1.0)
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


def _count_nodes_at_cutoff(v: np.ndarray, relative_cutoff: float) -> int:
    """Count sign changes after removing numerically insignificant tails.

    Values close to a true node are removed and the remaining signs are
    compressed, so a physical sign change across the node is retained.  The
    first/last significant-amplitude points define the trusted spatial window;
    this prevents isolated tail noise from creating spurious nodes.
    """
    w = np.asarray(v, dtype=float)
    if w.ndim != 1 or w.size < 3:
        return 0

    amp = np.abs(w)
    peak = float(np.max(amp))
    if not np.isfinite(peak) or peak <= 0.0:
        return 0

    cutoff = max(NODE_ABSOLUTE_CUTOFF, relative_cutoff * peak)
    significant = np.flatnonzero(amp >= cutoff)
    if significant.size < 2:
        return 0

    lo = max(0, int(significant[0]) - NODE_TAIL_GUARD_POINTS)
    hi = min(w.size - 1, int(significant[-1]) + NODE_TAIL_GUARD_POINTS)
    core = w[lo : hi + 1]

    signs = np.sign(core[np.abs(core) >= cutoff])
    signs = signs[signs != 0.0]
    if signs.size < 2:
        return 0

    # Remove repeated signs; only transitions between sign domains matter.
    compressed = signs[np.r_[True, signs[1:] != signs[:-1]]]
    return int(np.count_nonzero(compressed[1:] * compressed[:-1] < 0.0))


def count_nodes_robust(v: np.ndarray) -> Tuple[int, bool, Tuple[int, ...]]:
    """Return a cutoff-stable node count and the counts used to obtain it."""
    counts = tuple(_count_nodes_at_cutoff(v, c) for c in NODE_RELATIVE_CUTOFFS)
    values, multiplicities = np.unique(np.asarray(counts, dtype=int), return_counts=True)
    best = int(values[int(np.argmax(multiplicities))])
    stable = bool(np.count_nonzero(np.asarray(counts) == best) >= max(3, len(counts) - 1))
    return best, stable, counts


def fdm_eigen_residual(hmat: sp.csr_matrix, vec_int: np.ndarray, E_J: float) -> float:
    res = hmat @ vec_int - E_J * vec_int
    denom = max(1e-30, abs(E_J) * np.linalg.norm(vec_int))
    return float(np.linalg.norm(res) / denom)


def solve_fdm_states(p: ModelParams, cache: FDMGridCache, cfg: FDMConfig) -> Dict[str, object]:
    """Solve the low-lying FDM spectrum for one parameter pair."""
    v_int = effective_potential(cache.r_int, p)
    hmat = cache.tmat + sp.diags(v_int, format="csr")

    # ARPACK behaves much more reliably when the operator is O(1) rather than
    # O(1e-26).  Solve in MHz units and convert the eigenvalues back to joules.
    hsolve = (hmat * (1.0 / FDM_EIGEN_ENERGY_SCALE_J)).tocsr()

    dim = hsolve.shape[0]
    k_eff = min(cfg.k, dim - 2)
    ncv = min(dim, max(2 * k_eff + 12, 24))

    try:
        if cfg.solver_mode == "lowest":
            evals, evecs = spla.eigsh(
                hsolve,
                k=k_eff,
                which="SA",
                tol=cfg.eig_tol,
                maxiter=cfg.maxiter,
                ncv=ncv,
            )
        elif cfg.solver_mode == "shift_invert":
            evals, evecs = spla.eigsh(
                hsolve,
                k=k_eff,
                sigma=p.sigma / FDM_EIGEN_ENERGY_SCALE_J,
                which="LM",
                tol=cfg.eig_tol,
                maxiter=cfg.maxiter,
                ncv=ncv,
            )
        else:
            raise ValueError("cfg.solver_mode must be 'lowest' or 'shift_invert'.")
    except Exception as exc:
        raise RuntimeError(
            f"eigsh failed for r_c={p.r_c * 1e9:.6f} nm, "
            f"omega/2pi={p.omega_ion / (2 * PI * 1e6):.6f} MHz"
        ) from exc

    evals = np.asarray(evals, dtype=float) * FDM_EIGEN_ENERGY_SCALE_J
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

        node_count, node_stable, node_counts_by_cutoff = count_nodes_robust(u[i0 : i1 + 1])
        states.append(
            {
                "index": j,
                "E_J": float(E_J),
                "E_over_h_MHz": float(E_J / HPLANCK / 1.0e6),
                "absE_over_h_MHz": float(abs(E_J) / HPLANCK / 1.0e6),
                "nodes": node_count,
                "expected_nodes": j,
                "node_order_ok": bool(node_count == j),
                "node_count_stable": node_stable,
                "node_counts_by_cutoff": node_counts_by_cutoff,
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


def make_parameter_grids() -> Tuple[np.ndarray, np.ndarray]:
    if RUN_MODE not in PRESETS:
        raise ValueError(f"Unknown RUN_MODE={RUN_MODE!r}. Valid modes: {list(PRESETS)}")

    preset = PRESETS[RUN_MODE]
    rc_values_nm = np.linspace(RC_DOMAIN_NM[0], RC_DOMAIN_NM[1], preset.n_rc)
    omega_values_mhz = np.linspace(OMEGA_DOMAIN_MHZ[0], OMEGA_DOMAIN_MHZ[1], preset.n_omega)

    # Insert the official benchmark point exactly.
    rc_values_nm = include_reference_point(rc_values_nm, RC_REF_NM)
    omega_values_mhz = include_reference_point(omega_values_mhz, OMEGA_REF_MHZ)
    return rc_values_nm, omega_values_mhz


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
    states = result["states"]
    neg = result["negative_states"]

    row: Dict[str, float | int | bool] = {
        "run_mode": RUN_MODE,
        "solver_mode": cfg.solver_mode,
        "rc_nm": float(rc_nm),
        "omega_over_2pi_MHz": float(omega_mhz),
        "omega_rad_s": float(p.omega_ion),
        "a_ho_nm": float(harmonic_length(p) * 1e9),
        "x_c": float(rc_dimensionless(p)),
        "alpha_prime": float(alpha_dimensionless(p)),
        "n_negative": int(result["n_negative"]),
    }

    # Signed low-lying levels L_n = E_n/h in MHz.  These are the levels used for E_n=0 contours.
    for i in range(MAX_LEVELS):
        if len(states) > i:
            row[f"L{i}_MHz"] = float(states[i]["E_over_h_MHz"])
            row[f"L{i}_abs_MHz"] = float(states[i]["absE_over_h_MHz"])
            row[f"L{i}_nodes"] = int(states[i]["nodes"])
            row[f"L{i}_expected_nodes"] = int(states[i]["expected_nodes"])
            row[f"L{i}_node_order_ok"] = bool(states[i]["node_order_ok"])
            row[f"L{i}_node_count_stable"] = bool(states[i]["node_count_stable"])
            row[f"L{i}_eig_residual_rel"] = float(states[i]["eig_residual_rel"])
        else:
            row[f"L{i}_MHz"] = float("nan")
            row[f"L{i}_abs_MHz"] = float("nan")
            row[f"L{i}_nodes"] = -1
            row[f"L{i}_expected_nodes"] = i
            row[f"L{i}_node_order_ok"] = False
            row[f"L{i}_node_count_stable"] = False
            row[f"L{i}_eig_residual_rel"] = float("nan")

    # Backward-compatible negative-state columns from the old script.
    for i in range(MAX_LEVELS):
        row[f"Eneg{i}_MHz"] = float(neg[i]["E_over_h_MHz"]) if len(neg) > i else float("nan")
        row[f"Eneg{i}_abs_MHz"] = float(neg[i]["absE_over_h_MHz"]) if len(neg) > i else float("nan")
        row[f"nodes_neg{i}"] = int(neg[i]["nodes"]) if len(neg) > i else -1

    ground_E = float(row["L0_MHz"])
    ground_binding = -ground_E if np.isfinite(ground_E) and ground_E < 0.0 else float("nan")
    row["E0_abs_MHz"] = ground_binding
    row["delta15_MHz"] = abs(ground_binding - TARGET_MHZ) if np.isfinite(ground_binding) else float("nan")
    row["delta15_pct"] = 100.0 * abs(ground_binding - TARGET_MHZ) / TARGET_MHZ if np.isfinite(ground_binding) else float("nan")
    row["inside_13_17_MHz_band"] = bool(
        np.isfinite(ground_binding)
        and EXPERIMENTAL_BAND_MHZ[0] <= ground_binding <= EXPERIMENTAL_BAND_MHZ[1]
    )
    row["ground_eig_residual_rel"] = float(row["L0_eig_residual_rel"])

    return row


def run_parameter_map() -> Tuple[List[Dict[str, float | int | bool]], np.ndarray, np.ndarray]:
    cfg = FDMConfig()
    rc_values_nm, omega_values_mhz = make_parameter_grids()

    p_template = make_model_params(RC_REF_NM, OMEGA_REF_MHZ, cfg)
    cache = build_grid_cache(cfg, p_template)

    n_total = len(rc_values_nm) * len(omega_values_mhz)
    rows: List[Dict[str, float | int | bool]] = []
    t0 = time.perf_counter()

    print("=== Zero-energy spectral-structure FDM map ===")
    print(f"RUN_MODE                   : {RUN_MODE}")
    print(f"MAP EIGEN_SOLVER_MODE      : {cfg.solver_mode}")
    print(f"FDM centrifugal convention  : {'standard l(l+1)' if FORCE_STANDARD_FDM_CENTRIFUGAL else 'from PARAMS.use_langer_numerical'}")
    print(f"FDM grid N                 : {cfg.N}")
    print(f"FDM radial box             : r_min={cfg.r_min * 1e9:.6f} nm, r_max={cfg.r_max * 1e9:.6f} nm")
    print(f"Requested eigenpairs k      : {cfg.k}")
    print(f"Stored signed levels        : L0...L{MAX_LEVELS - 1}")
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
    path = OUTDIR / "threshold_map_full.csv"

    base_cols = [
        "run_mode",
        "solver_mode",
        "rc_nm",
        "omega_over_2pi_MHz",
        "omega_rad_s",
        "a_ho_nm",
        "x_c",
        "alpha_prime",
        "n_negative",
    ]

    signed_cols: List[str] = []
    for i in range(MAX_LEVELS):
        signed_cols += [
            f"L{i}_MHz", f"L{i}_abs_MHz", f"L{i}_nodes", f"L{i}_expected_nodes",
            f"L{i}_node_order_ok", f"L{i}_node_count_stable", f"L{i}_eig_residual_rel",
        ]

    neg_cols: List[str] = []
    for i in range(MAX_LEVELS):
        neg_cols += [f"Eneg{i}_MHz", f"Eneg{i}_abs_MHz", f"nodes_neg{i}"]

    tail_cols = ["E0_abs_MHz", "delta15_MHz", "delta15_pct", "inside_13_17_MHz_band", "ground_eig_residual_rel"]
    cols = base_cols + signed_cols + neg_cols + tail_cols

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


def find_nearest_omega_slice(rows: List[Dict[str, float | int | bool]]) -> Tuple[float, List[Dict[str, float | int | bool]]]:
    omega_values = sorted({float(r["omega_over_2pi_MHz"]) for r in rows})
    omega_selected = min(omega_values, key=lambda x: abs(x - OMEGA_REF_MHZ))
    slice_rows = [r for r in rows if abs(float(r["omega_over_2pi_MHz"]) - omega_selected) < 5e-10]
    slice_rows = sorted(slice_rows, key=lambda r: float(r["rc_nm"]))
    return omega_selected, slice_rows


def find_zero_crossings_1d(x: np.ndarray, y: np.ndarray) -> List[float]:
    """Linear interpolation of x values where y crosses zero."""
    crossings: List[float] = []
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        return crossings

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    for i in range(len(x) - 1):
        y0, y1 = y[i], y[i + 1]
        x0, x1 = x[i], x[i + 1]

        if y0 == 0.0:
            crossings.append(float(x0))
        elif y0 * y1 < 0.0:
            t = -y0 / (y1 - y0)
            crossings.append(float(x0 + t * (x1 - x0)))

    if y[-1] == 0.0:
        crossings.append(float(x[-1]))

    # Remove duplicates caused by exact-zero grid points.
    unique: List[float] = []
    for c in crossings:
        if not unique or abs(c - unique[-1]) > 1e-8:
            unique.append(c)
    return unique


def find_zero_crossing_brackets(
    x: np.ndarray,
    y: np.ndarray,
) -> List[Tuple[float, float, float]]:
    """Return sign-changing brackets and their linear-interpolation estimates."""
    brackets: List[Tuple[float, float, float]] = []
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[finite], dtype=float)
    y = np.asarray(y[finite], dtype=float)
    if x.size < 2:
        return brackets

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    for i in range(x.size - 1):
        x0, x1 = float(x[i]), float(x[i + 1])
        y0, y1 = float(y[i]), float(y[i + 1])
        if y0 == 0.0:
            brackets.append((x0, x0, x0))
        elif y0 * y1 < 0.0:
            interp = x0 - y0 * (x1 - x0) / (y1 - y0)
            brackets.append((x0, x1, float(interp)))

    if y[-1] == 0.0:
        xlast = float(x[-1])
        brackets.append((xlast, xlast, xlast))

    unique: List[Tuple[float, float, float]] = []
    for item in brackets:
        if not unique or abs(item[2] - unique[-1][2]) > 1.0e-8:
            unique.append(item)
    return unique


def make_crossing_config(
    *,
    N: int,
    r_max_nm: float,
    solver_mode: str = CROSSING_SOLVER_MODE,
    sigma: Optional[float] = None,
) -> FDMConfig:
    """Construct a high-accuracy FDM configuration for E_n=0 root finding."""
    base = FDMConfig()
    return FDMConfig(
        r_min=base.r_min,
        r_max=float(r_max_nm) * 1.0e-9,
        N=int(N),
        sigma=base.sigma if sigma is None else float(sigma),
        k=max(8, MAX_LEVELS + 2),
        eig_tol=1.0e-10,
        maxiter=max(base.maxiter, 30000),
        solver_mode=solver_mode,
    )


def _level_energy_mhz(
    rc_nm: float,
    omega_mhz: float,
    level: int,
    cfg: FDMConfig,
    grid_cache: FDMGridCache,
    value_cache: Dict[Tuple[int, float], float],
) -> float:
    """Evaluate one signed FDM level, with memoization inside a root solve."""
    key = (int(level), round(float(rc_nm), 12))
    if key not in value_cache:
        p = make_model_params(rc_nm, omega_mhz, cfg)
        result = solve_fdm_states(p, cache=grid_cache, cfg=cfg)
        states = result["states"]
        if len(states) <= level:
            raise RuntimeError(
                f"Requested E_{level}, but only {len(states)} eigenpairs were returned "
                f"for r_c={rc_nm:.6f} nm."
            )
        for state_index in range(level + 1):
            if not bool(states[state_index]["node_order_ok"]):
                raise RuntimeError(
                    f"Eigensolver completeness check failed at r_c={rc_nm:.6f} nm: "
                    f"sorted state {state_index} has {states[state_index]['nodes']} nodes."
                )
        value_cache[key] = float(states[level]["E_over_h_MHz"])
    return value_cache[key]


def refine_zero_crossing_for_config(
    *,
    level: int,
    omega_mhz: float,
    bracket_nm: Tuple[float, float],
    interpolation_nm: float,
    cfg: FDMConfig,
    grid_cache: FDMGridCache,
    value_cache: Dict[Tuple[int, float], float],
) -> Dict[str, float | int | bool]:
    """Refine one E_n=0 location by direct FDM evaluation and Brent's method."""
    domain_lo, domain_hi = RC_DOMAIN_NM
    raw_lo, raw_hi = sorted((float(bracket_nm[0]), float(bracket_nm[1])))
    lo = max(domain_lo, raw_lo - CROSSING_BRACKET_PAD_NM)
    hi = min(domain_hi, raw_hi + CROSSING_BRACKET_PAD_NM)

    def f(rc_nm: float) -> float:
        return _level_energy_mhz(
            rc_nm,
            omega_mhz,
            level,
            cfg,
            grid_cache,
            value_cache,
        )

    if raw_lo == raw_hi:
        probe = max(4.0 * CROSSING_ROOT_XTOL_NM, 0.05)
        lo = max(domain_lo, raw_lo - probe)
        hi = min(domain_hi, raw_hi + probe)

    flo = f(lo)
    fhi = f(hi)

    # If a convergence variation shifts the root beyond the original map cell,
    # scan the padded interval and select the sign-changing pair nearest the
    # original interpolation estimate.
    if flo != 0.0 and fhi != 0.0 and flo * fhi > 0.0:
        scan_x = np.linspace(lo, hi, 17)
        scan_y = np.array([f(float(xi)) for xi in scan_x], dtype=float)
        candidates = find_zero_crossing_brackets(scan_x, scan_y)
        if not candidates:
            raise RuntimeError(
                f"No sign-changing bracket for E_{level}=0 in [{lo:.6f}, {hi:.6f}] nm "
                f"at omega/2pi={omega_mhz:.6f} MHz."
            )
        chosen = min(candidates, key=lambda b: abs(float(b[2]) - interpolation_nm))
        lo, hi = float(chosen[0]), float(chosen[1])
        flo, fhi = f(lo), f(hi)

    if flo == 0.0:
        root = lo
        iterations = 0
        function_calls = 1
        converged = True
    elif fhi == 0.0:
        root = hi
        iterations = 0
        function_calls = 1
        converged = True
    else:
        root, info = brentq(
            f,
            lo,
            hi,
            xtol=CROSSING_ROOT_XTOL_NM,
            rtol=CROSSING_ROOT_RTOL,
            full_output=True,
            disp=True,
        )
        iterations = int(info.iterations)
        function_calls = int(info.function_calls)
        converged = bool(info.converged)

    residual_mhz = abs(f(float(root)))
    return {
        "rc_crossing_nm": float(root),
        "energy_residual_MHz": float(residual_mhz),
        "bracket_low_nm": float(lo),
        "bracket_high_nm": float(hi),
        "iterations": iterations,
        "function_calls": function_calls,
        "converged": converged,
    }


def _crossing_validation_configs(base_cfg: FDMConfig) -> List[Tuple[str, FDMConfig]]:
    """Build N and r_max sweeps without duplicating identical configurations."""
    configs: List[Tuple[str, FDMConfig]] = []
    seen: set[Tuple[int, int, str, int]] = {
        (
            base_cfg.N,
            int(round(base_cfg.r_max * 1.0e12)),
            base_cfg.solver_mode,
            int(round(base_cfg.sigma * 1.0e30)),
        )
    }

    def add(label: str, cfg: FDMConfig) -> None:
        key = (
            cfg.N,
            int(round(cfg.r_max * 1.0e12)),
            cfg.solver_mode,
            int(round(cfg.sigma * 1.0e30)),
        )
        if key not in seen:
            seen.add(key)
            configs.append((label, cfg))

    for N in CROSSING_N_SWEEP:
        add(
            "N_sweep",
            make_crossing_config(N=N, r_max_nm=base_cfg.r_max * 1.0e9),
        )

    # Preserve approximately the baseline grid spacing during the r_max sweep,
    # so box-size sensitivity is not confused with a large change in Delta r.
    dr_ref = (base_cfg.r_max - base_cfg.r_min) / (base_cfg.N - 1)
    for rmax_nm in CROSSING_RMAX_SWEEP_NM:
        rmax_m = rmax_nm * 1.0e-9
        N_scaled = max(7, int(round((rmax_m - base_cfg.r_min) / dr_ref)) + 1)
        add(
            "rmax_sweep",
            make_crossing_config(N=N_scaled, r_max_nm=rmax_nm),
        )

    if base_cfg.solver_mode == "shift_invert":
        for sigma in CROSSING_SIGMA_SWEEP_J:
            add(
                "sigma_sweep",
                make_crossing_config(
                    N=base_cfg.N,
                    r_max_nm=base_cfg.r_max * 1.0e9,
                    solver_mode=base_cfg.solver_mode,
                    sigma=sigma,
                ),
            )

    return configs


def refine_fixed_slice_zero_crossings(
    rows: List[Dict[str, float | int | bool]],
) -> Tuple[float, List[Dict[str, object]], List[Dict[str, object]]]:
    """Locate fixed-slice E_n=0 boundaries and optionally validate convergence."""
    omega_selected, slice_rows = find_nearest_omega_slice(rows)
    rc = np.array([float(r["rc_nm"]) for r in slice_rows], dtype=float)

    crossing_specs: List[Dict[str, object]] = []
    for level in range(MAX_LEVELS):
        y = np.array([float(r[f"L{level}_MHz"]) for r in slice_rows], dtype=float)
        for ordinal, (lo, hi, interp) in enumerate(find_zero_crossing_brackets(rc, y)):
            crossing_specs.append(
                {
                    "crossing_id": f"E{level}_{ordinal}",
                    "level": level,
                    "bracket_low_map_nm": lo,
                    "bracket_high_map_nm": hi,
                    "rc_interpolated_nm": interp,
                }
            )

    if not crossing_specs:
        return omega_selected, [], []

    map_cfg = FDMConfig()
    base_cfg = make_crossing_config(
        N=map_cfg.N,
        r_max_nm=map_cfg.r_max * 1.0e9,
        solver_mode=CROSSING_SOLVER_MODE,
    )
    base_template = make_model_params(RC_REF_NM, omega_selected, base_cfg)
    base_grid = build_grid_cache(base_cfg, base_template)
    base_values: Dict[Tuple[int, float], float] = {}

    crossing_rows: List[Dict[str, object]] = []
    for spec in crossing_specs:
        level = int(spec["level"])
        if REFINE_ZERO_ENERGY_CROSSINGS:
            refined = refine_zero_crossing_for_config(
                level=level,
                omega_mhz=omega_selected,
                bracket_nm=(
                    float(spec["bracket_low_map_nm"]),
                    float(spec["bracket_high_map_nm"]),
                ),
                interpolation_nm=float(spec["rc_interpolated_nm"]),
                cfg=base_cfg,
                grid_cache=base_grid,
                value_cache=base_values,
            )
        else:
            refined = {
                "rc_crossing_nm": float(spec["rc_interpolated_nm"]),
                "energy_residual_MHz": float("nan"),
                "bracket_low_nm": float(spec["bracket_low_map_nm"]),
                "bracket_high_nm": float(spec["bracket_high_map_nm"]),
                "iterations": 0,
                "function_calls": 0,
                "converged": False,
            }

        crossing_rows.append(
            {
                **spec,
                **refined,
                "omega_over_2pi_MHz": omega_selected,
                "solver_mode": base_cfg.solver_mode,
                "sigma_J": base_cfg.sigma,
                "N": base_cfg.N,
                "r_max_nm": base_cfg.r_max * 1.0e9,
                "root_residual_pass": bool(
                    float(refined["energy_residual_MHz"]) <= CROSSING_ENERGY_RESIDUAL_TOL_MHZ
                    if np.isfinite(float(refined["energy_residual_MHz"]))
                    else False
                ),
                "convergence_spread_nm": float("nan"),
                "convergence_pass": True,
                "interpretation": f"E_{level}=0 boundary of the negative-energy sector",
            }
        )

    crossing_rows.sort(key=lambda row: float(row["rc_crossing_nm"]))
    for crossing in crossing_rows:
        if not bool(crossing["root_residual_pass"]):
            warnings.warn(
                f"{crossing['crossing_id']} root residual is "
                f"{float(crossing['energy_residual_MHz']):.3e} MHz, above the target "
                f"{CROSSING_ENERGY_RESIDUAL_TOL_MHZ:.3e} MHz.",
                RuntimeWarning,
            )

    # Verify that each zero crossing coincides with a one-state change in N_- on
    # the sampled map.  The sign of the transition depends on sweep direction.
    for crossing in crossing_rows:
        root = float(crossing["rc_crossing_nm"])
        left_candidates = [r for r in slice_rows if float(r["rc_nm"]) < root]
        right_candidates = [r for r in slice_rows if float(r["rc_nm"]) > root]
        if left_candidates and right_candidates:
            left = max(left_candidates, key=lambda r: float(r["rc_nm"]))
            right = min(right_candidates, key=lambda r: float(r["rc_nm"]))
            n_left = int(left["n_negative"])
            n_right = int(right["n_negative"])
            crossing["Nminus_left"] = n_left
            crossing["Nminus_right"] = n_right
            crossing["Nminus_change"] = n_right - n_left
            crossing["topology_check_pass"] = bool(abs(n_right - n_left) == 1)
            if abs(n_right - n_left) != 1:
                warnings.warn(
                    f"{crossing['crossing_id']} does not show a one-state N_- change "
                    f"on the sampled slice ({n_left} -> {n_right}).",
                    RuntimeWarning,
                )
        else:
            crossing["Nminus_left"] = -1
            crossing["Nminus_right"] = -1
            crossing["Nminus_change"] = 0
            crossing["topology_check_pass"] = False

    validation_rows: List[Dict[str, object]] = []
    if RUN_ZERO_ENERGY_CONVERGENCE and REFINE_ZERO_ENERGY_CROSSINGS:
        roots_by_id: Dict[str, List[float]] = {
            str(r["crossing_id"]): [float(r["rc_crossing_nm"])] for r in crossing_rows
        }

        for sweep_label, cfg in _crossing_validation_configs(base_cfg):
            template = make_model_params(RC_REF_NM, omega_selected, cfg)
            grid = build_grid_cache(cfg, template)
            value_cache: Dict[Tuple[int, float], float] = {}

            for spec in crossing_specs:
                crossing_id = str(spec["crossing_id"])
                level = int(spec["level"])
                try:
                    result = refine_zero_crossing_for_config(
                        level=level,
                        omega_mhz=omega_selected,
                        bracket_nm=(
                            float(spec["bracket_low_map_nm"]),
                            float(spec["bracket_high_map_nm"]),
                        ),
                        interpolation_nm=float(spec["rc_interpolated_nm"]),
                        cfg=cfg,
                        grid_cache=grid,
                        value_cache=value_cache,
                    )
                    root_nm = float(result["rc_crossing_nm"])
                    roots_by_id[crossing_id].append(root_nm)
                    status = "ok"
                    error_message = ""
                except Exception as exc:
                    result = {
                        "rc_crossing_nm": float("nan"),
                        "energy_residual_MHz": float("nan"),
                        "iterations": -1,
                        "function_calls": -1,
                        "converged": False,
                    }
                    root_nm = float("nan")
                    status = "failed"
                    error_message = str(exc)
                    warnings.warn(
                        f"Crossing validation failed for {crossing_id}, N={cfg.N}, "
                        f"r_max={cfg.r_max * 1e9:.1f} nm: {exc}",
                        RuntimeWarning,
                    )

                validation_rows.append(
                    {
                        "crossing_id": crossing_id,
                        "level": level,
                        "omega_over_2pi_MHz": omega_selected,
                        "sweep": sweep_label,
                        "N": cfg.N,
                        "r_max_nm": cfg.r_max * 1.0e9,
                        "solver_mode": cfg.solver_mode,
                        "sigma_J": cfg.sigma,
                        "rc_crossing_nm": root_nm,
                        "energy_residual_MHz": float(result["energy_residual_MHz"]),
                        "iterations": int(result["iterations"]),
                        "function_calls": int(result["function_calls"]),
                        "converged": bool(result["converged"]),
                        "status": status,
                        "error_message": error_message,
                    }
                )

        for row in crossing_rows:
            roots = np.asarray(roots_by_id[str(row["crossing_id"])], dtype=float)
            roots = roots[np.isfinite(roots)]
            spread = float(np.max(roots) - np.min(roots)) if roots.size >= 2 else float("nan")
            row["convergence_spread_nm"] = spread
            row["convergence_pass"] = bool(
                np.isfinite(spread) and spread <= CROSSING_CONVERGENCE_TOL_NM
            )
            if np.isfinite(spread) and spread > CROSSING_CONVERGENCE_TOL_NM:
                warnings.warn(
                    f"{row['crossing_id']} crossing spread is {spread:.6f} nm, larger than "
                    f"the target {CROSSING_CONVERGENCE_TOL_NM:.6f} nm.",
                    RuntimeWarning,
                )

    return omega_selected, crossing_rows, validation_rows


def write_zero_energy_crossing_outputs(
    crossing_rows: List[Dict[str, object]],
    validation_rows: List[Dict[str, object]],
) -> Tuple[Path, Path, Path, Path]:
    """Write refined crossings and the dedicated convergence report."""
    csv_path = OUTDIR / "threshold_crossings_fixed_omega.csv"
    crossing_fields = [
        "crossing_id",
        "level",
        "omega_over_2pi_MHz",
        "bracket_low_map_nm",
        "bracket_high_map_nm",
        "rc_interpolated_nm",
        "rc_crossing_nm",
        "energy_residual_MHz",
        "solver_mode",
        "sigma_J",
        "N",
        "r_max_nm",
        "iterations",
        "function_calls",
        "converged",
        "root_residual_pass",
        "convergence_spread_nm",
        "convergence_pass",
        "Nminus_left",
        "Nminus_right",
        "Nminus_change",
        "topology_check_pass",
        "interpretation",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=crossing_fields)
        writer.writeheader()
        for row in crossing_rows:
            writer.writerow({key: row.get(key, "") for key in crossing_fields})

    tex_path = OUTDIR / "threshold_crossings_fixed_omega.tex"
    with tex_path.open("w", encoding="utf-8") as f:
        f.write(r"""\begin{table}[t]
\centering
\caption{Directly refined zero-energy crossings along the fixed-frequency slice nearest the benchmark value. Each location satisfies $E_n(r_c,\omega)=0$ and marks a boundary of the negative-energy sector. Positive-energy eigenstates remain trap confined. The reported spread is the full range obtained from the dedicated $N$, $r_{\max}$, and shift-parameter validation sweeps.}
\label{tab:threshold_crossings_fixed_omega}
\begin{tabular}{c c c c c}
\hline
Level & $\omega/2\pi$ (MHz) & $r_c$ at $E_n=0$ (nm) & $N_-$ change & Spread (nm) \\
\hline
""")
        if crossing_rows:
            for row in crossing_rows:
                spread = float(row["convergence_spread_nm"])
                spread_text = f"{spread:.3e}" if np.isfinite(spread) else "--"
                f.write(
                    f"$E_{int(row['level'])}$ & "
                    f"{float(row['omega_over_2pi_MHz']):.6f} & "
                    f"{float(row['rc_crossing_nm']):.{CROSSING_REPORT_DECIMALS}f} & "
                    f"{int(row['Nminus_left'])}\\to{int(row['Nminus_right'])} & "
                    f"{spread_text} " + r"\\" + "\n"
                )
        else:
            f.write("No crossing in displayed slice & -- & -- & -- & -- " + r"\\" + "\n")
        f.write(r"""\hline
\end{tabular}
\end{table}
""")

    validation_csv = OUTDIR / "zero_energy_crossing_validation.csv"
    validation_fields = [
        "crossing_id",
        "level",
        "omega_over_2pi_MHz",
        "sweep",
        "N",
        "r_max_nm",
        "solver_mode",
        "sigma_J",
        "rc_crossing_nm",
        "energy_residual_MHz",
        "iterations",
        "function_calls",
        "converged",
        "status",
        "error_message",
    ]
    with validation_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=validation_fields)
        writer.writeheader()
        for row in validation_rows:
            writer.writerow({key: row.get(key, "") for key in validation_fields})

    validation_tex = OUTDIR / "zero_energy_crossing_validation.tex"
    with validation_tex.open("w", encoding="utf-8") as f:
        f.write(r"""\begin{table}[t]
\centering
\caption{Numerical validation configurations used for the direct zero-energy crossing refinement. During the $r_{\max}$ sweep, the number of grid points is scaled to preserve approximately the reference grid spacing; the $\sigma$ sweep checks shift-invert independence.}
\label{tab:zero_energy_crossing_validation}
\begin{tabular}{c c c c c c}
\hline
Crossing & Sweep & $N$ & $r_{\max}$ (nm) & $\sigma/10^{-26}$ & $r_c$ (nm) \\
\hline
""")
        if validation_rows:
            for row in validation_rows:
                root = float(row["rc_crossing_nm"])
                root_text = f"{root:.{CROSSING_REPORT_DECIMALS}f}" if np.isfinite(root) else "--"
                f.write(
                    f"$E_{int(row['level'])}=0$ & "
                    f"{str(row['sweep']).replace('_', r'\_')} & "
                    f"{int(row['N'])} & "
                    f"{float(row['r_max_nm']):.1f} & "
                    f"{float(row['sigma_J']) / 1.0e-26:.2f} & "
                    f"{root_text} " + r"\\" + "\n"
                )
        else:
            f.write("-- & convergence disabled & -- & -- & -- & -- " + r"\\" + "\n")
        f.write(r"""\hline
\end{tabular}
\end{table}
""")

    return csv_path, tex_path, validation_csv, validation_tex


def write_summary_latex(rows: List[Dict[str, float | int | bool]]) -> Path:
    """Write a compact table around the official reference omega only."""
    path = OUTDIR / "threshold_map_summary_slice.tex"
    omega_selected, slice_rows = find_nearest_omega_slice(rows)

    with path.open("w", encoding="utf-8") as f:
        f.write(r"""\begin{table}[t]
\centering
\caption{Representative fixed-frequency slice of the zero-energy spectral map.  The signed levels $L_n=E_n/h$ show where individual eigenvalues cross $E=0$ and change the negative-energy-state count.  Positive-energy levels remain trap confined; the sweep is not a recalibration procedure.}
\label{tab:threshold_map_slice}
\begin{tabular}{c c c c c c}
\hline
$r_c$ (nm) & $\omega/2\pi$ (MHz) & $L_0$ (MHz) & $L_1$ (MHz) & $L_2$ (MHz) & $N_-$ \\
\hline
""")
        for r in slice_rows:
            f.write(
                f"{float(r['rc_nm']):.6f} & "
                f"{float(r['omega_over_2pi_MHz']):.6f} & "
                f"{float(r['L0_MHz']):.6f} & "
                f"{float(r['L1_MHz']):.6f} & "
                f"{float(r['L2_MHz']):.6f} & "
                f"{int(r['n_negative'])} " + r"\\" + "\n"
            )
        f.write(r"""\hline
\end{tabular}
\end{table}
""")
    return path


# ============================================================
# 7. Figure generation
# ============================================================

def set_publication_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 170,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "text.usetex": False,
        "font.size": 8.8,
        "axes.labelsize": 9.2,
        "axes.titlesize": 9.6,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 7.2,
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


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.035,
        0.955,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.1,
        fontweight="bold",
        color="black",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="0.72", linewidth=0.45, alpha=0.92),
        zorder=80,
    )


def add_reference_guides(ax) -> None:
    ax.axvline(RC_REF_NM, color="0.35", lw=0.75, ls="--", zorder=30)
    ax.axhline(OMEGA_REF_MHZ, color="0.35", lw=0.75, ls="--", zorder=30)


def add_reference_star(ax) -> None:
    ax.scatter(
        RC_REF_NM,
        OMEGA_REF_MHZ,
        marker="*",
        s=165,
        facecolor="#FFD43B",
        edgecolor="black",
        linewidth=0.9,
        zorder=70,
    )
    ax.scatter(
        RC_REF_NM,
        OMEGA_REF_MHZ,
        marker="*",
        s=95,
        facecolor="#FFD43B",
        edgecolor="white",
        linewidth=0.5,
        zorder=71,
    )


def safe_contour(ax, X, Y, Z, levels, **kwargs):
    finite = np.asarray(Z)[np.isfinite(Z)]
    if finite.size == 0:
        return None
    zmin = float(np.nanmin(finite))
    zmax = float(np.nanmax(finite))
    valid_levels = [float(lvl) for lvl in levels if zmin < float(lvl) < zmax]
    if not valid_levels:
        return None
    try:
        return ax.contour(X, Y, Z, levels=valid_levels, **kwargs)
    except Exception:
        return None


def format_map_axis(ax, rc_values_nm: np.ndarray, omega_values_mhz: np.ndarray) -> None:
    ax.set_xlim(float(np.min(rc_values_nm)), float(np.max(rc_values_nm)))
    ax.set_ylim(float(np.min(omega_values_mhz)), float(np.max(omega_values_mhz)))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)


def make_discrete_cmap(n_categories: int) -> ListedColormap:
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


def zero_energy_line_style(n: int) -> Tuple[str, str, float]:
    colors = ["black", "0.18", "0.32", "0.46", "0.58", "0.68"]
    linestyles = ["solid", "dashed", "dashdot", "dotted", (0, (5, 2, 1, 2)), (0, (2, 2))]
    linewidths = [1.45, 1.20, 1.15, 1.20, 1.10, 1.05]
    return colors[n % len(colors)], linestyles[n % len(linestyles)], linewidths[n % len(linewidths)]


def add_zero_energy_contours(ax, RC, OM, level_matrices: List[np.ndarray]) -> List[int]:
    existing: List[int] = []
    for n, Z in enumerate(level_matrices):
        color, linestyle, linewidth = zero_energy_line_style(n)
        cs = safe_contour(
            ax,
            RC,
            OM,
            Z,
            levels=[0.0],
            colors=color,
            linestyles=linestyle,
            linewidths=linewidth,
            zorder=40 + n,
        )
        if cs is not None:
            existing.append(n)
    return existing


def add_binding_contours(ax, RC, OM, ground_binding_matrix) -> None:
    cs_band = safe_contour(
        ax,
        RC,
        OM,
        ground_binding_matrix,
        [EXPERIMENTAL_BAND_MHZ[0], EXPERIMENTAL_BAND_MHZ[1]],
        colors="0.10",
        linewidths=0.85,
        linestyles="dashed",
        zorder=35,
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
                fontsize=7.0,
                colors="0.10",
            )
        except Exception:
            pass

    safe_contour(
        ax,
        RC,
        OM,
        ground_binding_matrix,
        [TARGET_MHZ],
        colors="black",
        linewidths=1.25,
        linestyles="solid",
        zorder=36,
    )

def write_threshold_structure_figure(
    ground_binding_matrix: np.ndarray,
    Nminus_matrix: np.ndarray,
    level_matrices: List[np.ndarray],
    rc_values_nm: np.ndarray,
    omega_values_mhz: np.ndarray,
    rows: List[Dict[str, float | int | bool]],
    crossing_rows: Optional[List[Dict[str, object]]] = None,
) -> Tuple[Path, Path, Path]:
    """Create the publication figure with a dedicated, unclipped legend row."""
    set_publication_style()
    crossing_rows = crossing_rows or []

    RC, OM = np.meshgrid(rc_values_nm, omega_values_mhz)
    fig = plt.figure(figsize=(7.7, 6.15))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[1.0, 0.13, 0.92],
        left=0.075,
        right=0.965,
        bottom=0.080,
        top=0.965,
        wspace=0.28,
        hspace=0.22,
    )

    axN = fig.add_subplot(gs[0, 0])
    axE = fig.add_subplot(gs[0, 1], sharey=axN)
    axL = fig.add_subplot(gs[1, :])
    axS = fig.add_subplot(gs[2, :])
    axL.axis("off")

    # --------------------------------------------------------
    # Panel (a): N_- regions and E_n=0 boundaries.
    # --------------------------------------------------------
    finite_n = np.asarray(Nminus_matrix)[np.isfinite(Nminus_matrix)]
    if finite_n.size == 0:
        raise RuntimeError("No finite N_- values are available for plotting.")
    nmin = int(np.nanmin(finite_n))
    nmax = int(np.nanmax(finite_n))
    n_categories = max(1, nmax - nmin + 1)
    cmap_n = make_discrete_cmap(n_categories)
    boundaries = np.arange(nmin - 0.5, nmax + 1.5, 1.0)
    norm_n = BoundaryNorm(boundaries, cmap_n.N)

    pm = axN.pcolormesh(
        RC,
        OM,
        Nminus_matrix,
        shading="auto",
        cmap=cmap_n,
        norm=norm_n,
        rasterized=True,
    )
    boundary_levels = np.arange(nmin + 0.5, nmax + 0.5, 1.0)
    if boundary_levels.size > 0:
        safe_contour(
            axN,
            RC,
            OM,
            Nminus_matrix,
            boundary_levels,
            colors="white",
            linewidths=1.10,
            zorder=25,
        )
        safe_contour(
            axN,
            RC,
            OM,
            Nminus_matrix,
            boundary_levels,
            colors="0.25",
            linewidths=0.35,
            zorder=26,
        )

    add_zero_energy_contours(axN, RC, OM, level_matrices)
    add_reference_guides(axN)
    add_reference_star(axN)

    cbN = fig.colorbar(
        pm,
        ax=axN,
        pad=0.025,
        fraction=0.050,
        ticks=np.arange(nmin, nmax + 1),
    )
    cbN.set_label(r"$N_{-}$", rotation=0, labelpad=10)
    cbN.ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    cbN.ax.tick_params(length=2.4, width=0.65)

    axN.set_title(r"Negative-energy sectors: $N_{-}(r_c,\omega)$", pad=5)
    axN.set_xlabel(r"Soft-core radius $r_c$ (nm)")
    axN.set_ylabel(r"Trap frequency $\omega/2\pi$ (MHz)")
    add_panel_label(axN, "(a)")
    format_map_axis(axN, rc_values_nm, omega_values_mhz)

    # --------------------------------------------------------
    # Panel (b): ground-state binding and experimental band.
    # --------------------------------------------------------
    finite_e = np.asarray(ground_binding_matrix)[np.isfinite(ground_binding_matrix)]
    if finite_e.size == 0:
        raise RuntimeError("No finite ground-state binding values are available for plotting.")
    e_min = float(np.nanmin(finite_e))
    e_max = float(np.nanmax(finite_e))
    e_levels = np.linspace(
        np.floor(2.0 * e_min) / 2.0,
        np.ceil(2.0 * e_max) / 2.0,
        80,
    )

    cf = axE.contourf(
        RC,
        OM,
        ground_binding_matrix,
        levels=e_levels,
        cmap="viridis",
        extend="both",
    )
    add_binding_contours(axE, RC, OM, ground_binding_matrix)
    add_zero_energy_contours(axE, RC, OM, level_matrices)
    add_reference_guides(axE)
    add_reference_star(axE)

    cbE = fig.colorbar(cf, ax=axE, pad=0.025, fraction=0.050)
    cbE.set_label(r"$|E_0|/h$ (MHz)", labelpad=4)
    cbE.ax.tick_params(length=2.4, width=0.65)

    axE.set_title(r"Calibration band and zero-energy curves", pad=5)
    axE.set_xlabel(r"Soft-core radius $r_c$ (nm)")
    axE.set_ylabel("")
    axE.tick_params(labelleft=False)
    add_panel_label(axE, "(b)")
    format_map_axis(axE, rc_values_nm, omega_values_mhz)

    # --------------------------------------------------------
    # Dedicated figure legend.  This prevents E4=0 (or any later available
    # boundary) from being clipped or lost between the two plot rows.
    # --------------------------------------------------------
    zero_levels_for_legend: List[int] = []
    for level, matrix in enumerate(level_matrices):
        finite = np.asarray(matrix)[np.isfinite(matrix)]
        if finite.size and float(np.min(finite)) < 0.0 < float(np.max(finite)):
            zero_levels_for_legend.append(level)

    handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            markersize=10.0,
            markerfacecolor="#FFD43B",
            markeredgecolor="black",
            linestyle="None",
            label="benchmark point",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            lw=1.25,
            label=rf"{TARGET_MHZ:.0f} MHz target",
        ),
        Line2D(
            [0],
            [0],
            color="0.10",
            lw=0.85,
            linestyle="--",
            label=rf"{EXPERIMENTAL_BAND_MHZ[0]:.0f}--{EXPERIMENTAL_BAND_MHZ[1]:.0f} MHz band",
        ),
    ]
    for level in zero_levels_for_legend:
        color, linestyle, linewidth = zero_energy_line_style(level)
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=linewidth,
                linestyle=linestyle,
                label=rf"$E_{level}=0$",
            )
        )

    legend = axL.legend(
        handles=handles,
        loc="center",
        ncol=min(5, max(1, len(handles))),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="0.35",
        borderpad=0.35,
        handlelength=2.1,
        columnspacing=0.9,
        handletextpad=0.45,
    )
    legend.get_frame().set_linewidth(0.50)
    legend.get_frame().set_facecolor("white")

    # --------------------------------------------------------
    # Panel (c): fixed-omega signed-level slice.
    # --------------------------------------------------------
    omega_selected, slice_rows = find_nearest_omega_slice(rows)
    rc_slice = np.array([float(r["rc_nm"]) for r in slice_rows], dtype=float)

    refined_by_level: Dict[int, List[float]] = {}
    for row in crossing_rows:
        level = int(row["level"])
        omega = float(row["omega_over_2pi_MHz"])
        root = float(row["rc_crossing_nm"])
        if abs(omega - omega_selected) < 5.0e-10 and np.isfinite(root):
            refined_by_level.setdefault(level, []).append(root)

    plotted_levels: List[int] = []
    ylo, yhi = SLICE_YLIM_MHZ
    for level in range(min(MAX_LEVELS, MAX_SLICE_LEVELS_TO_PLOT)):
        y = np.array([float(r[f"L{level}_MHz"]) for r in slice_rows], dtype=float)
        finite_y = y[np.isfinite(y)]
        if finite_y.size == 0:
            continue
        if np.nanmin(finite_y) > yhi and np.nanmax(finite_y) > yhi:
            continue

        color, linestyle, linewidth = zero_energy_line_style(level)
        axS.plot(
            rc_slice,
            y,
            color=color,
            linestyle=linestyle,
            lw=linewidth,
            label=rf"$E_{level}/h$",
        )
        plotted_levels.append(level)

        roots = refined_by_level.get(level)
        if roots is None:
            roots = find_zero_crossings_1d(rc_slice, y)
        for root in roots:
            axS.plot(
                root,
                0.0,
                marker="o",
                ms=4.4,
                color=color,
                markerfacecolor="white",
                markeredgewidth=0.90,
                zorder=50,
            )

    axS.axhline(0.0, color="black", lw=0.9, ls="--", zorder=10)
    axS.axvline(RC_REF_NM, color="0.35", lw=0.75, ls="--", zorder=10)
    axS.text(
        0.14,
        0.92,
        rf"fixed slice: $\omega/2\pi={omega_selected:.3f}\,\mathrm{{MHz}}$",
        transform=axS.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="0.75",
            linewidth=0.45,
        ),
    )
    axS.text(
        0.985,
        0.04,
        r"$E>0$: trap-confined levels",
        transform=axS.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="0.30",
    )
    axS.set_xlim(float(np.min(rc_values_nm)), float(np.max(rc_values_nm)))
    axS.set_xlabel(r"Soft-core radius $r_c$ (nm)")
    axS.set_ylabel(r"Signed levels $E_n/h$ (MHz)")
    axS.set_ylim(*SLICE_YLIM_MHZ)
    axS.xaxis.set_minor_locator(AutoMinorLocator(2))
    axS.yaxis.set_minor_locator(AutoMinorLocator(2))
    axS.tick_params(which="both", top=True, right=True)
    if plotted_levels:
        axS.legend(
            loc="upper right",
            ncol=min(3, len(plotted_levels)),
            frameon=True,
            edgecolor="0.35",
        )
    add_panel_label(axS, "(c)")

    pdf_path = OUTDIR / "threshold_structure_figure_PRA.pdf"
    png_path = OUTDIR / "threshold_structure_figure_PRA.png"
    svg_path = OUTDIR / "threshold_structure_figure_PRA.svg"

    fig.savefig(pdf_path)
    fig.savefig(png_path)
    fig.savefig(svg_path)
    plt.close(fig)

    return pdf_path, png_path, svg_path


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
    for n in range(min(MAX_LEVELS, 4)):
        print(f"  L{n}/h [MHz]         : {float(best[f'L{n}_MHz']):.9f}")
    print(f"  |E0|/h [MHz]        : {float(best['E0_abs_MHz']):.9f}")
    print(f"  delta15 [%]         : {float(best['delta15_pct']):.6e}")
    print(f"  inside 13-17 MHz    : {bool(best['inside_13_17_MHz_band'])}")
    print(f"  eig residual        : {float(best['ground_eig_residual_rel']):.3e}")


def validate_benchmark_reference(rows: List[Dict[str, float | int | bool]]) -> None:
    """Check that the map reproduces the official benchmark point before accepting the figure."""
    best = min(
        rows,
        key=lambda r: (
            abs(float(r["rc_nm"]) - RC_REF_NM),
            abs(float(r["omega_over_2pi_MHz"]) - OMEGA_REF_MHZ),
        ),
    )
    nneg = int(best["n_negative"])
    e0 = float(best["L0_MHz"])

    ok_n = (nneg == BENCHMARK_EXPECTED_NNEG)
    ok_e = abs(e0 - BENCHMARK_E0_MHZ) <= BENCHMARK_E0_TOL_MHZ

    if not (ok_n and ok_e):
        msg = (
            "\nBenchmark validation failed. The zero-energy map does not reproduce the central "
            "fixed-Hamiltonian benchmark point.\n"
            f"  Expected: N_-={BENCHMARK_EXPECTED_NNEG}, E0/h≈{BENCHMARK_E0_MHZ:.3f} MHz\n"
            f"  Obtained: N_-={nneg}, E0/h={e0:.6f} MHz at "
            f"r_c={float(best['rc_nm']):.6f} nm, "
            f"omega/2pi={float(best['omega_over_2pi_MHz']):.6f} MHz\n"
            "Do not use the generated figure until this is fixed. Check that the FDM map uses "
            "standard l(l+1), not the Langer correction, and use draft/paper/production settings."
        )
        if ENFORCE_BENCHMARK_VALIDATION:
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning)


def warn_if_too_few_eigenpairs(rows: List[Dict[str, float | int | bool]], cfg: FDMConfig) -> None:
    max_neg = max(int(r["n_negative"]) for r in rows)
    if max_neg >= cfg.k:
        warnings.warn(
            "The maximum number of negative states is equal to the number of requested eigenpairs. "
            "Increase PRESETS[RUN_MODE].k or MAX_LEVELS to make sure no negative levels are missed.",
            RuntimeWarning,
        )
    if max_neg >= MAX_LEVELS:
        warnings.warn(
            "The map contains at least as many negative levels as MAX_LEVELS. "
            "Increase MAX_LEVELS if you want all zero-energy curves to be stored and plotted.",
            RuntimeWarning,
        )


def write_node_diagnostics(rows: List[Dict[str, float | int | bool]]) -> Path:
    """Write only unstable or Sturm-order-inconsistent node-count entries."""
    path = OUTDIR / "zero_energy_node_diagnostics.csv"
    fields = [
        "rc_nm",
        "omega_over_2pi_MHz",
        "level",
        "energy_MHz",
        "nodes",
        "expected_nodes",
        "node_order_ok",
        "node_count_stable",
        "eig_residual_rel",
        "issue",
    ]
    diagnostics: List[Dict[str, object]] = []
    for row in rows:
        for level in range(MAX_LEVELS):
            energy = float(row[f"L{level}_MHz"])
            if not np.isfinite(energy):
                continue
            order_ok = bool(row[f"L{level}_node_order_ok"])
            stable = bool(row[f"L{level}_node_count_stable"])
            if order_ok and stable:
                continue
            issues: List[str] = []
            if not order_ok:
                issues.append("Sturm-order mismatch")
            if not stable:
                issues.append("cutoff-sensitive count")
            diagnostics.append(
                {
                    "rc_nm": float(row["rc_nm"]),
                    "omega_over_2pi_MHz": float(row["omega_over_2pi_MHz"]),
                    "level": level,
                    "energy_MHz": energy,
                    "nodes": int(row[f"L{level}_nodes"]),
                    "expected_nodes": int(row[f"L{level}_expected_nodes"]),
                    "node_order_ok": order_ok,
                    "node_count_stable": stable,
                    "eig_residual_rel": float(row[f"L{level}_eig_residual_rel"]),
                    "issue": "; ".join(issues),
                }
            )

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(diagnostics)

    if diagnostics:
        mismatch_count = sum(not bool(d["node_order_ok"]) for d in diagnostics)
        unstable_count = sum(not bool(d["node_count_stable"]) for d in diagnostics)
        warnings.warn(
            f"Node diagnostics found {mismatch_count} Sturm-order mismatches and "
            f"{unstable_count} cutoff-sensitive counts. Inspect {path.name} before publication.",
            RuntimeWarning,
        )
    return path


def validate_reference_node_order(rows: List[Dict[str, float | int | bool]]) -> None:
    """Enforce the expected 0,1,2,... node ordering at the benchmark point."""
    best = min(
        rows,
        key=lambda row: (
            abs(float(row["rc_nm"]) - RC_REF_NM),
            abs(float(row["omega_over_2pi_MHz"]) - OMEGA_REF_MHZ),
        ),
    )
    failures: List[str] = []
    for level in range(MAX_LEVELS):
        energy = float(best[f"L{level}_MHz"])
        if not np.isfinite(energy):
            continue
        nodes = int(best[f"L{level}_nodes"])
        if nodes != level:
            failures.append(f"E{level}: nodes={nodes}, expected={level}")

    if failures:
        message = (
            "Reference-point node ordering failed: " + "; ".join(failures) + ". "
            "Do not use the map until the eigensolver completeness and node-counting "
            "diagnostics are resolved."
        )
        if ENFORCE_REFERENCE_NODE_ORDER:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning)


def print_refined_crossings(
    omega_selected: float,
    crossing_rows: List[Dict[str, object]],
) -> None:
    print(f"\nZero-energy crossings along omega/2pi={omega_selected:.9f} MHz:")
    if not crossing_rows:
        print("  No E_n=0 crossing found in this fixed-frequency slice/domain.")
        return

    for row in crossing_rows:
        spread = float(row["convergence_spread_nm"])
        spread_text = f", spread={spread:.3e} nm" if np.isfinite(spread) else ""
        transition = f"{int(row['Nminus_left'])}->{int(row['Nminus_right'])}"
        print(
            f"  E{int(row['level'])}=0 at r_c="
            f"{float(row['rc_crossing_nm']):.{CROSSING_REPORT_DECIMALS}f} nm "
            f"(N_-: {transition}, |E| residual="
            f"{float(row['energy_residual_MHz']):.3e} MHz{spread_text})"
        )


# ============================================================
# 9. Main entry point
# ============================================================

def main() -> None:
    cfg = FDMConfig()

    rows, rc_values_nm, omega_values_mhz = run_parameter_map()
    warn_if_too_few_eigenpairs(rows, cfg)
    print_reference_row(rows)
    validate_benchmark_reference(rows)
    validate_reference_node_order(rows)
    node_diagnostics_csv = write_node_diagnostics(rows)

    omega_selected, crossing_rows, crossing_validation_rows = refine_fixed_slice_zero_crossings(rows)
    print_refined_crossings(omega_selected, crossing_rows)

    ground_binding_matrix = matrix_from_rows(
        rows,
        rc_values_nm,
        omega_values_mhz,
        "E0_abs_MHz",
    )
    Nminus_matrix = matrix_from_rows(
        rows,
        rc_values_nm,
        omega_values_mhz,
        "n_negative",
    )
    level_matrices = [
        matrix_from_rows(rows, rc_values_nm, omega_values_mhz, f"L{i}_MHz")
        for i in range(MAX_LEVELS)
    ]

    full_csv = write_full_csv(rows)
    nb_csv = write_matrix_csv(
        Nminus_matrix,
        rc_values_nm,
        omega_values_mhz,
        "threshold_map_Nminus_matrix.csv",
    )
    ground_csv = write_matrix_csv(
        ground_binding_matrix,
        rc_values_nm,
        omega_values_mhz,
        "threshold_map_ground_binding_matrix.csv",
    )
    level_csvs = [
        write_matrix_csv(
            matrix,
            rc_values_nm,
            omega_values_mhz,
            f"threshold_map_L{i}_matrix.csv",
        )
        for i, matrix in enumerate(level_matrices)
    ]
    crossing_csv, crossing_tex, crossing_validation_csv, crossing_validation_tex = (
        write_zero_energy_crossing_outputs(crossing_rows, crossing_validation_rows)
    )
    summary_tex = write_summary_latex(rows)
    pdf_path, png_path, svg_path = write_threshold_structure_figure(
        ground_binding_matrix,
        Nminus_matrix,
        level_matrices,
        rc_values_nm,
        omega_values_mhz,
        rows,
        crossing_rows=crossing_rows,
    )

    print("\nFiles written:")
    print(f"  Full CSV                    : {full_csv}")
    print(f"  N- matrix CSV               : {nb_csv}")
    print(f"  Ground binding matrix CSV   : {ground_csv}")
    for path in level_csvs:
        print(f"  Signed level matrix CSV     : {path}")
    print(f"  Refined crossing CSV        : {crossing_csv}")
    print(f"  Refined crossing LaTeX      : {crossing_tex}")
    print(f"  Crossing validation CSV     : {crossing_validation_csv}")
    print(f"  Crossing validation LaTeX   : {crossing_validation_tex}")
    print(f"  Node diagnostics CSV        : {node_diagnostics_csv}")
    print(f"  Summary LaTeX table         : {summary_tex}")
    print(f"  Figure PDF                  : {pdf_path}")
    print(f"  Figure PNG                  : {png_path}")
    print(f"  Figure SVG                  : {svg_path}")

    if RUN_MODE == "quick":
        print(
            "\nNote: RUN_MODE='quick'. Use RUN_MODE='draft', 'paper', or "
            "'production' for final research output."
        )


if __name__ == "__main__":
    main()
