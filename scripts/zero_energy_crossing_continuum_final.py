from __future__ import annotations


from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import csv
import importlib.util
import math
import sys
import warnings

import numpy as np
from scipy.linalg import eig_banded
from scipy.optimize import brentq, curve_fit
import matplotlib as mpl
import matplotlib.pyplot as plt


# ============================================================
# 1. User controls
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CONTINUUM_N_VALUES: Tuple[int, ...] = (
    6000,
    7500,
    9000,
    10500,
    12000,
    14000,
    16000,
)

CONTINUUM_RMIN_NM = 0.10
CONTINUUM_RMAX_NM = 650.0

PRIMARY_FINEST_POINTS = 5
ROBUSTNESS_FINEST_POINTS: Tuple[int, ...] = (4, 5, 6)
CANDIDATE_CONVERGENCE_ORDERS: Tuple[float, ...] = (1.0, 2.0, 4.0)
ORDER_MATCH_TOLERANCE = 0.15

ROOT_XTOL_NM = 2.0e-5
ROOT_RTOL = 1.0e-10
ROOT_RESIDUAL_TOL_MHZ = 1.0e-5
BRACKET_PAD_NM = 0.30

# Agreement required between the banded solver and the original sparse
# production-grid crossing at N=12000.
PRODUCTION_SOLVER_AGREEMENT_TOL_NM = 1.0e-5

# Fit-quality criteria.
MIN_FIT_R2 = 0.995
MAX_FIT_RESIDUAL_NM = 5.0e-3
MAX_FIT_WINDOW_SPREAD_NM = 5.0e-3
MIN_REPORTED_UNCERTAINTY_NM = 1.0e-5

WRITE_DIAGNOSTIC_FIGURES = True

MAP_MODULE_CANDIDATES = (
    "threshold_crossing_fdm_map_PRA_final.py",
    "threshold_crossing_fdm_map_PRA_fixed.py",
)
CROSSING_CSV_PATTERN = "threshold_crossings_fixed_omega*.csv"


# ============================================================
# 2. Data containers
# ============================================================

@dataclass(frozen=True)
class CrossingSpec:
    crossing_id: str
    level: int
    omega_mhz: float
    bracket_low_nm: float
    bracket_high_nm: float
    interpolation_nm: float
    production_root_nm: float
    production_N: int
    production_rmax_nm: float
    nminus_left: int
    nminus_right: int


@dataclass(frozen=True)
class BandedGrid:
    N: int
    r_min_m: float
    r_max_m: float
    delta_r_nm: float
    r_int_m: np.ndarray
    kinetic_main_MHz: float
    kinetic_off1_MHz: float
    kinetic_off2_MHz: float


@dataclass(frozen=True)
class LinearFitResult:
    n_points: int
    convergence_order: float
    intercept_nm: float
    slope_nm_per_nm_power: float
    r2: float
    max_abs_residual_nm: float
    intercept_stderr_nm: float
    used_indices: np.ndarray
    fitted_nm: np.ndarray
    residuals_nm: np.ndarray


@dataclass(frozen=True)
class FreeOrderFitResult:
    n_points: int
    intercept_nm: float
    amplitude: float
    observed_order: float
    r2: float
    max_abs_residual_nm: float


# ============================================================
# 3. File discovery and module loading
# ============================================================

def load_map_module():
    module_path: Optional[Path] = None
    for name in MAP_MODULE_CANDIDATES:
        candidate = BASE_DIR / name
        if candidate.exists():
            module_path = candidate
            break

    if module_path is None:
        raise FileNotFoundError(
            "Place threshold_crossing_fdm_map_PRA_final.py beside this script."
        )

    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))

    spec = importlib.util.spec_from_file_location("threshold_map_solver", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    required = (
        "PARAMS",
        "HBAR",
        "HPLANCK",
        "PI",
        "RC_REF_NM",
        "ModelParams",
        "effective_potential",
        "reduced_mass",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"{module_path.name} is missing: {missing}")

    return module, module_path


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _as_float(row: Mapping[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def _as_int(row: Mapping[str, str], key: str, default: int = -1) -> int:
    try:
        return int(float(row.get(key, "")))
    except (TypeError, ValueError):
        return default


def parse_crossing_specs(path: Path) -> List[CrossingSpec]:
    rows = read_csv_rows(path)
    if not rows:
        raise RuntimeError(f"{path.name} contains no crossing rows.")

    required = {
        "crossing_id",
        "level",
        "omega_over_2pi_MHz",
        "bracket_low_map_nm",
        "bracket_high_map_nm",
        "rc_interpolated_nm",
        "rc_crossing_nm",
    }
    missing = required.difference(rows[0].keys())
    if missing:
        raise KeyError(f"Missing refined-crossing columns: {sorted(missing)}")

    specs: List[CrossingSpec] = []
    for row in rows:
        root = _as_float(row, "rc_crossing_nm")
        if not np.isfinite(root):
            continue
        specs.append(
            CrossingSpec(
                crossing_id=str(row["crossing_id"]),
                level=_as_int(row, "level"),
                omega_mhz=_as_float(row, "omega_over_2pi_MHz"),
                bracket_low_nm=_as_float(row, "bracket_low_map_nm"),
                bracket_high_nm=_as_float(row, "bracket_high_map_nm"),
                interpolation_nm=_as_float(row, "rc_interpolated_nm", root),
                production_root_nm=root,
                production_N=_as_int(row, "N", 12000),
                production_rmax_nm=_as_float(row, "r_max_nm", CONTINUUM_RMAX_NM),
                nminus_left=_as_int(row, "Nminus_left"),
                nminus_right=_as_int(row, "Nminus_right"),
            )
        )

    if not specs:
        raise RuntimeError(f"No finite refined crossings found in {path.name}.")
    return specs


def find_compatible_crossing_csv() -> Tuple[Path, List[CrossingSpec]]:
    candidates = [p for p in BASE_DIR.glob(CROSSING_CSV_PATTERN) if p.is_file()]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    errors: List[str] = []
    for path in candidates:
        try:
            return path, parse_crossing_specs(path)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    detail = "\n  ".join(errors) if errors else "no matching files"
    raise RuntimeError(
        "No compatible refined crossing CSV was found. Tried:\n  " + detail
    )


def select_adjacent_boundaries(
    specs: Sequence[CrossingSpec],
    benchmark_rc_nm: float,
) -> List[CrossingSpec]:
    lower = [s for s in specs if s.production_root_nm < benchmark_rc_nm]
    upper = [s for s in specs if s.production_root_nm > benchmark_rc_nm]
    if not lower or not upper:
        raise RuntimeError("The benchmark is not bracketed by two refined crossings.")

    selected = [
        max(lower, key=lambda s: s.production_root_nm),
        min(upper, key=lambda s: s.production_root_nm),
    ]
    if not math.isclose(selected[0].omega_mhz, selected[1].omega_mhz, abs_tol=1e-12):
        raise RuntimeError("Adjacent crossings are not on the same omega slice.")
    return selected


# ============================================================
# 4. Exact banded representation of the five-point FDM matrix
# ============================================================

def build_banded_grid(solver, N: int) -> BandedGrid:
    if N < 7:
        raise ValueError("N must be at least 7 for the five-point stencil.")

    r_min_m = CONTINUUM_RMIN_NM * 1.0e-9
    r_max_m = CONTINUUM_RMAX_NM * 1.0e-9
    r = np.linspace(r_min_m, r_max_m, N, dtype=float)
    dr = float(r[1] - r[0])
    r_int = r[2 : N - 2]

    p_template = solver.ModelParams(
        r_c=float(solver.RC_REF_NM) * 1.0e-9,
        omega_ion=solver.PARAMS.omega_ion,
        r_min=r_min_m,
        r_max=r_max_m,
        N=N,
        sigma=-1.0e-26,
    )
    mu = float(solver.reduced_mass(p_template))

    inv_h2 = 1.0 / (dr * dr)
    c0 = -30.0 / 12.0 * inv_h2
    c1 = +16.0 / 12.0 * inv_h2
    c2 = -1.0 / 12.0 * inv_h2

    kinetic_factor = -(float(solver.HBAR) ** 2) / (2.0 * mu)
    energy_scale_J = float(solver.HPLANCK) * 1.0e6

    return BandedGrid(
        N=N,
        r_min_m=r_min_m,
        r_max_m=r_max_m,
        delta_r_nm=dr * 1.0e9,
        r_int_m=r_int,
        kinetic_main_MHz=kinetic_factor * c0 / energy_scale_J,
        kinetic_off1_MHz=kinetic_factor * c1 / energy_scale_J,
        kinetic_off2_MHz=kinetic_factor * c2 / energy_scale_J,
    )


def lowest_levels_banded_MHz(
    solver,
    grid: BandedGrid,
    rc_nm: float,
    omega_mhz: float,
    highest_level: int,
) -> np.ndarray:
    """Return E_0,...,E_highest_level for the same discretized FDM operator."""
    p = solver.ModelParams(
        r_c=float(rc_nm) * 1.0e-9,
        omega_ion=float(omega_mhz) * 2.0 * float(solver.PI) * 1.0e6,
        r_min=grid.r_min_m,
        r_max=grid.r_max_m,
        N=grid.N,
        sigma=-1.0e-26,
    )

    potential_J = solver.effective_potential(grid.r_int_m, p)
    potential_MHz = np.asarray(potential_J, dtype=float) / (
        float(solver.HPLANCK) * 1.0e6
    )

    m = grid.r_int_m.size
    # Upper-band storage for scipy.linalg.eig_banded with two superdiagonals:
    # ab[2,j] = A[j,j], ab[1,j] = A[j-1,j], ab[0,j] = A[j-2,j].
    ab = np.zeros((3, m), dtype=float)
    ab[2, :] = grid.kinetic_main_MHz + potential_MHz
    ab[1, 1:] = grid.kinetic_off1_MHz
    ab[0, 2:] = grid.kinetic_off2_MHz

    evals = eig_banded(
        ab,
        lower=False,
        eigvals_only=True,
        overwrite_a_band=True,
        select="i",
        select_range=(0, int(highest_level)),
        check_finite=False,
    )
    return np.asarray(evals, dtype=float)


def refine_banded_crossing(
    solver,
    grid: BandedGrid,
    spec: CrossingSpec,
) -> Dict[str, Any]:
    domain_lo, domain_hi = 22.0, 32.0
    raw_lo, raw_hi = sorted((spec.bracket_low_nm, spec.bracket_high_nm))
    lo = max(domain_lo, raw_lo - BRACKET_PAD_NM)
    hi = min(domain_hi, raw_hi + BRACKET_PAD_NM)

    energy_cache: Dict[float, float] = {}

    def f(rc_nm: float) -> float:
        key = round(float(rc_nm), 12)
        if key not in energy_cache:
            levels = lowest_levels_banded_MHz(
                solver,
                grid,
                rc_nm=float(rc_nm),
                omega_mhz=spec.omega_mhz,
                highest_level=spec.level,
            )
            energy_cache[key] = float(levels[spec.level])
        return energy_cache[key]

    flo, fhi = f(lo), f(hi)
    if flo != 0.0 and fhi != 0.0 and flo * fhi > 0.0:
        scan_x = np.linspace(lo, hi, 17)
        scan_y = np.asarray([f(float(x)) for x in scan_x], dtype=float)
        candidates: List[Tuple[float, float, float]] = []
        for i in range(scan_x.size - 1):
            if scan_y[i] == 0.0:
                candidates.append((scan_x[i], scan_x[i], scan_x[i]))
            elif scan_y[i] * scan_y[i + 1] < 0.0:
                interp = scan_x[i] - scan_y[i] * (scan_x[i + 1] - scan_x[i]) / (
                    scan_y[i + 1] - scan_y[i]
                )
                candidates.append((scan_x[i], scan_x[i + 1], interp))
        if not candidates:
            raise RuntimeError(
                f"No sign-changing bracket for {spec.crossing_id} at N={grid.N}."
            )
        chosen = min(candidates, key=lambda b: abs(b[2] - spec.interpolation_nm))
        lo, hi = float(chosen[0]), float(chosen[1])

    root, info = brentq(
        f,
        lo,
        hi,
        xtol=ROOT_XTOL_NM,
        rtol=ROOT_RTOL,
        full_output=True,
        disp=True,
    )
    residual = abs(f(float(root)))

    return {
        "rc_crossing_nm": float(root),
        "energy_residual_MHz": float(residual),
        "iterations": int(info.iterations),
        "function_calls": int(info.function_calls),
        "converged": bool(info.converged),
        "bracket_low_nm": float(lo),
        "bracket_high_nm": float(hi),
    }


# ============================================================
# 5. Continuum sweep
# ============================================================

def compute_raw_rows(
    solver,
    specs: Sequence[CrossingSpec],
) -> List[Dict[str, Any]]:
    raw_rows: List[Dict[str, Any]] = []

    for N in CONTINUUM_N_VALUES:
        grid = build_banded_grid(solver, N)
        for spec in specs:
            print(f"Computing {spec.crossing_id} at N={N} ...", flush=True)
            status = "ok"
            error_message = ""
            try:
                result = refine_banded_crossing(solver, grid, spec)
                print(
                    f"  -> r_c={float(result['rc_crossing_nm']):.6f} nm, "
                    f"|E|={float(result['energy_residual_MHz']):.3e} MHz",
                    flush=True,
                )
            except Exception as exc:
                result = {
                    "rc_crossing_nm": float("nan"),
                    "energy_residual_MHz": float("nan"),
                    "iterations": -1,
                    "function_calls": -1,
                    "converged": False,
                }
                status = "failed"
                error_message = str(exc)
                warnings.warn(
                    f"Root solve failed for {spec.crossing_id}, N={N}: {exc}",
                    RuntimeWarning,
                )

            residual = float(result["energy_residual_MHz"])
            residual_pass = bool(
                np.isfinite(residual) and residual <= ROOT_RESIDUAL_TOL_MHZ
            )
            if status == "ok" and not residual_pass:
                status = "failed_residual"
                error_message = (
                    f"Residual {residual:.3e} MHz exceeds "
                    f"{ROOT_RESIDUAL_TOL_MHZ:.3e} MHz."
                )

            production_difference = float("nan")
            production_agreement_pass = True
            if N == spec.production_N:
                production_difference = (
                    float(result["rc_crossing_nm"]) - spec.production_root_nm
                )
                production_agreement_pass = bool(
                    np.isfinite(production_difference)
                    and abs(production_difference) <= PRODUCTION_SOLVER_AGREEMENT_TOL_NM
                )
                if not production_agreement_pass:
                    status = "failed_production_crosscheck"
                    error_message = (
                        f"Banded/sparse production crossing mismatch is "
                        f"{production_difference:.3e} nm."
                    )

            raw_rows.append(
                {
                    "crossing_id": spec.crossing_id,
                    "level": spec.level,
                    "omega_over_2pi_MHz": spec.omega_mhz,
                    "N": N,
                    "r_min_nm": CONTINUUM_RMIN_NM,
                    "r_max_nm": CONTINUUM_RMAX_NM,
                    "delta_r_nm": grid.delta_r_nm,
                    "delta_r2_nm2": grid.delta_r_nm**2,
                    "matrix_solver": "scipy.linalg.eig_banded",
                    "rc_crossing_nm": float(result["rc_crossing_nm"]),
                    "energy_residual_MHz": residual,
                    "iterations": int(result["iterations"]),
                    "function_calls": int(result["function_calls"]),
                    "converged": bool(result["converged"]),
                    "root_residual_pass": residual_pass,
                    "production_sparse_rc_nm": (
                        spec.production_root_nm if N == spec.production_N else float("nan")
                    ),
                    "banded_minus_sparse_nm": production_difference,
                    "production_solver_agreement_pass": production_agreement_pass,
                    "status": status,
                    "error_message": error_message,
                }
            )

    return raw_rows


# ============================================================
# 6. Convergence-order diagnosis and continuum fits
# ============================================================

def fixed_order_fit_finest(
    delta_r_nm: np.ndarray,
    y_nm: np.ndarray,
    n_finest: int,
    convergence_order: float,
) -> LinearFitResult:
    if delta_r_nm.ndim != 1 or y_nm.ndim != 1 or delta_r_nm.size != y_nm.size:
        raise ValueError("delta_r and y must be one-dimensional and equally sized.")
    if n_finest < 3 or delta_r_nm.size < n_finest:
        raise ValueError(f"Need at least {n_finest} valid points.")
    if convergence_order <= 0.0:
        raise ValueError("The convergence order must be positive.")

    order = np.argsort(delta_r_nm)  # finest grids first
    used = order[:n_finest]
    h = np.asarray(delta_r_nm[used], dtype=float)
    y = np.asarray(y_nm[used], dtype=float)
    x = h ** float(convergence_order)

    # Scale x for a well-conditioned covariance calculation. The continuum
    # intercept at x=0 is invariant under this rescaling.
    x_scale = float(np.max(np.abs(x)))
    if x_scale <= 0.0:
        raise ValueError("Invalid grid-spacing scale.")
    z = x / x_scale

    design = np.column_stack([np.ones_like(z), z])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    intercept = float(beta[0])
    slope = float(beta[1] / x_scale)
    fitted = design @ beta
    residuals = y - fitted

    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0.0 else 1.0
    max_abs_residual = float(np.max(np.abs(residuals)))

    dof = n_finest - 2
    sigma2 = sse / dof
    covariance = sigma2 * np.linalg.inv(design.T @ design)
    intercept_stderr = float(math.sqrt(max(0.0, covariance[0, 0])))

    return LinearFitResult(
        n_points=n_finest,
        convergence_order=float(convergence_order),
        intercept_nm=intercept,
        slope_nm_per_nm_power=slope,
        r2=r2,
        max_abs_residual_nm=max_abs_residual,
        intercept_stderr_nm=intercept_stderr,
        used_indices=used,
        fitted_nm=np.asarray(fitted, dtype=float),
        residuals_nm=np.asarray(residuals, dtype=float),
    )


def free_order_fit_finest(
    delta_r_nm: np.ndarray,
    y_nm: np.ndarray,
    n_finest: int,
) -> FreeOrderFitResult:
    if delta_r_nm.size < n_finest or n_finest < 4:
        raise ValueError("At least four points are required for a free-order fit.")

    order = np.argsort(delta_r_nm)
    used = order[:n_finest]
    h = np.asarray(delta_r_nm[used], dtype=float)
    y = np.asarray(y_nm[used], dtype=float)

    def model(hh, intercept, amplitude, power):
        return intercept + amplitude * hh**power

    # The data are monotone and the observed order is expected to be positive.
    # Several initial guesses are tried to avoid dependence on one nonlinear seed.
    best = None
    for p0 in CANDIDATE_CONVERGENCE_ORDERS:
        try:
            popt, _ = curve_fit(
                model,
                h,
                y,
                p0=(float(y[-1]), -0.5, p0),
                bounds=([-np.inf, -np.inf, 0.25], [np.inf, np.inf, 6.0]),
                maxfev=100000,
            )
            fitted = model(h, *popt)
            residuals = y - fitted
            sse = float(np.sum(residuals**2))
            if best is None or sse < best[0]:
                best = (sse, popt, fitted, residuals)
        except Exception:
            continue

    if best is None:
        raise RuntimeError("The free-order convergence fit did not converge.")

    sse, popt, fitted, residuals = best
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0.0 else 1.0
    return FreeOrderFitResult(
        n_points=n_finest,
        intercept_nm=float(popt[0]),
        amplitude=float(popt[1]),
        observed_order=float(popt[2]),
        r2=float(r2),
        max_abs_residual_nm=float(np.max(np.abs(residuals))),
    )


def choose_supported_order(observed_order: float) -> Tuple[float, float]:
    distances = [abs(observed_order - p) for p in CANDIDATE_CONVERGENCE_ORDERS]
    index = int(np.argmin(distances))
    return float(CANDIDATE_CONVERGENCE_ORDERS[index]), float(distances[index])


def build_summary(
    raw_rows: Sequence[Mapping[str, Any]],
    specs: Sequence[CrossingSpec],
    benchmark_rc_nm: float,
) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []

    for spec in specs:
        rows = [
            row for row in raw_rows
            if row["crossing_id"] == spec.crossing_id
            and row["status"] == "ok"
            and np.isfinite(float(row["rc_crossing_nm"]))
        ]
        rows.sort(key=lambda row: int(row["N"]))
        if len(rows) < max(ROBUSTNESS_FINEST_POINTS):
            raise RuntimeError(
                f"Only {len(rows)} valid points remain for {spec.crossing_id}."
            )

        h = np.asarray([float(row["delta_r_nm"]) for row in rows], dtype=float)
        y = np.asarray([float(row["rc_crossing_nm"]) for row in rows], dtype=float)

        free_order = free_order_fit_finest(h, y, PRIMARY_FINEST_POINTS)
        selected_order, order_distance = choose_supported_order(
            free_order.observed_order
        )

        fits = {
            n: fixed_order_fit_finest(h, y, n, selected_order)
            for n in ROBUSTNESS_FINEST_POINTS
        }
        primary = fits[PRIMARY_FINEST_POINTS]
        candidate_fits = {
            p: fixed_order_fit_finest(h, y, PRIMARY_FINEST_POINTS, p)
            for p in CANDIDATE_CONVERGENCE_ORDERS
        }

        intercepts = np.asarray(
            [fits[n].intercept_nm for n in ROBUSTNESS_FINEST_POINTS],
            dtype=float,
        )
        fit_window_spread = float(
            np.max(np.abs(intercepts - primary.intercept_nm))
        )
        model_selection_difference = abs(
            primary.intercept_nm - free_order.intercept_nm
        )
        uncertainty = float(
            max(
                primary.max_abs_residual_nm,
                primary.intercept_stderr_nm,
                fit_window_spread,
                model_selection_difference,
                MIN_REPORTED_UNCERTAINTY_NM,
            )
        )

        order_supported = bool(order_distance <= ORDER_MATCH_TOLERANCE)
        fit_pass = bool(
            order_supported
            and primary.r2 >= MIN_FIT_R2
            and primary.max_abs_residual_nm <= MAX_FIT_RESIDUAL_NM
            and fit_window_spread <= MAX_FIT_WINDOW_SPREAD_NM
        )
        if not fit_pass:
            warnings.warn(
                f"Fit check failed for {spec.crossing_id}: "
                f"observed p={free_order.observed_order:.4f}, selected p={selected_order:.1f}, "
                f"R2={primary.r2:.6f}, residual="
                f"{primary.max_abs_residual_nm:.3e} nm, window spread="
                f"{fit_window_spread:.3e} nm.",
                RuntimeWarning,
            )

        production_row = next(
            row for row in rows if int(row["N"]) == spec.production_N
        )

        summary.append(
            {
                "crossing_id": spec.crossing_id,
                "level": spec.level,
                "omega_over_2pi_MHz": spec.omega_mhz,
                "nminus_left": spec.nminus_left,
                "nminus_right": spec.nminus_right,
                "production_N": spec.production_N,
                "production_sparse_rc_nm": spec.production_root_nm,
                "production_banded_rc_nm": float(production_row["rc_crossing_nm"]),
                "banded_minus_sparse_nm": float(
                    production_row["banded_minus_sparse_nm"]
                ),
                "production_solver_agreement_pass": bool(
                    production_row["production_solver_agreement_pass"]
                ),
                "continuum_rc_crossing_nm": primary.intercept_nm,
                "fit_uncertainty_nm": uncertainty,
                # Backward-compatible alias. This is not a total numerical uncertainty.
                "numerical_uncertainty_nm": uncertainty,
                "production_banded_minus_continuum_nm": (
                    float(production_row["rc_crossing_nm"]) - primary.intercept_nm
                ),
                "observed_convergence_order": free_order.observed_order,
                "free_order_fit_R2": free_order.r2,
                "free_order_intercept_nm": free_order.intercept_nm,
                "model_selection_difference_nm": model_selection_difference,
                "selected_convergence_order": selected_order,
                "order_distance_to_selected": order_distance,
                "order_supported": order_supported,
                "candidate_p1_R2": candidate_fits[1.0].r2,
                "candidate_p2_R2": candidate_fits[2.0].r2,
                "candidate_p4_R2": candidate_fits[4.0].r2,
                "slope_nm_per_nm_power": primary.slope_nm_per_nm_power,
                "primary_fit_finest_points": PRIMARY_FINEST_POINTS,
                "primary_fit_R2": primary.r2,
                "primary_fit_max_abs_residual_nm": primary.max_abs_residual_nm,
                "primary_fit_intercept_stderr_nm": primary.intercept_stderr_nm,
                "fit4_intercept_nm": fits[4].intercept_nm,
                "fit5_intercept_nm": fits[5].intercept_nm,
                "fit6_intercept_nm": fits[6].intercept_nm,
                "fit_window_spread_nm": fit_window_spread,
                "fit_pass": fit_pass,
                "benchmark_rc_nm": benchmark_rc_nm,
                "fit_model": (
                    "r_c(Delta r)=r_c(infinity)+A(Delta r)^"
                    f"{selected_order:g}"
                ),
            }
        )

    summary.sort(key=lambda row: float(row["continuum_rc_crossing_nm"]))
    if len(summary) != 2:
        raise RuntimeError("Exactly two adjacent boundaries were expected.")

    benchmark_inside = bool(
        float(summary[0]["continuum_rc_crossing_nm"])
        < benchmark_rc_nm
        < float(summary[1]["continuum_rc_crossing_nm"])
    )
    lower_boundary = float(summary[0]["continuum_rc_crossing_nm"])
    upper_boundary = float(summary[1]["continuum_rc_crossing_nm"])
    for row in summary:
        row["benchmark_inside_continuum_sector"] = benchmark_inside
        row["benchmark_margin_to_lower_nm"] = benchmark_rc_nm - lower_boundary
        row["benchmark_margin_to_upper_nm"] = upper_boundary - benchmark_rc_nm

    if not benchmark_inside:
        raise RuntimeError("Continuum boundaries do not bracket the benchmark point.")

    return summary


# ============================================================
# 7. Output files
# ============================================================

def write_dict_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_raw_csv(raw_rows: Sequence[Mapping[str, Any]]) -> Path:
    path = BASE_DIR / "zero_energy_crossing_continuum_raw.csv"
    fields = (
        "crossing_id",
        "level",
        "omega_over_2pi_MHz",
        "N",
        "r_min_nm",
        "r_max_nm",
        "delta_r_nm",
        "delta_r2_nm2",
        "matrix_solver",
        "rc_crossing_nm",
        "energy_residual_MHz",
        "iterations",
        "function_calls",
        "converged",
        "root_residual_pass",
        "production_sparse_rc_nm",
        "banded_minus_sparse_nm",
        "production_solver_agreement_pass",
        "status",
        "error_message",
    )
    write_dict_csv(path, raw_rows, fields)
    return path


def write_summary_csv(summary_rows: Sequence[Mapping[str, Any]]) -> Path:
    path = BASE_DIR / "zero_energy_crossing_continuum_summary.csv"
    fields = (
        "crossing_id",
        "level",
        "omega_over_2pi_MHz",
        "nminus_left",
        "nminus_right",
        "production_N",
        "production_sparse_rc_nm",
        "production_banded_rc_nm",
        "banded_minus_sparse_nm",
        "production_solver_agreement_pass",
        "continuum_rc_crossing_nm",
        "fit_uncertainty_nm",
        "numerical_uncertainty_nm",
        "production_banded_minus_continuum_nm",
        "observed_convergence_order",
        "free_order_fit_R2",
        "free_order_intercept_nm",
        "model_selection_difference_nm",
        "selected_convergence_order",
        "order_distance_to_selected",
        "order_supported",
        "candidate_p1_R2",
        "candidate_p2_R2",
        "candidate_p4_R2",
        "slope_nm_per_nm_power",
        "primary_fit_finest_points",
        "primary_fit_R2",
        "primary_fit_max_abs_residual_nm",
        "primary_fit_intercept_stderr_nm",
        "fit4_intercept_nm",
        "fit5_intercept_nm",
        "fit6_intercept_nm",
        "fit_window_spread_nm",
        "fit_pass",
        "benchmark_rc_nm",
        "benchmark_inside_continuum_sector",
        "benchmark_margin_to_lower_nm",
        "benchmark_margin_to_upper_nm",
        "fit_model",
    )
    write_dict_csv(path, summary_rows, fields)
    return path


def write_raw_latex(raw_rows: Sequence[Mapping[str, Any]]) -> Path:
    path = BASE_DIR / "zero_energy_crossing_continuum_table.tex"
    rows_sorted = sorted(
        raw_rows,
        key=lambda row: (int(row["level"]), int(row["N"])),
    )

    with path.open("w", encoding="utf-8") as f:
        f.write(r"""\begin{table}[t]
\centering
\caption{Fixed-cutoff grid sequence used to extrapolate the two zero-energy boundaries adjacent to the benchmark sector. Only $N$ and therefore $\Delta r$ are varied; $r_{\min}$, $r_{\max}$, $\omega$, and all physical Hamiltonian parameters remain fixed. The final column is the root residual, not a physical level energy.}
\label{tab:zero_energy_crossing_continuum_raw}
\begin{tabular}{c c c c c}
\hline
Level & $N$ & $\Delta r$ (nm) & $r_c(E_n=0)$ (nm) & Root residual $|E_n(r_c^\star)|/h$ (MHz) \\
\hline
""")
        for row in rows_sorted:
            f.write(
                f"$E_{int(row['level'])}=0$ & "
                f"{int(row['N'])} & "
                f"{float(row['delta_r_nm']):.8f} & "
                f"{float(row['rc_crossing_nm']):.6f} & "
                f"{float(row['energy_residual_MHz']):.2e} "
                + r"\\" + "\n"
            )
        f.write(r"""\hline
\end{tabular}
\end{table}
""")
    return path


def write_summary_latex(summary_rows: Sequence[Mapping[str, Any]]) -> Path:
    path = BASE_DIR / "zero_energy_crossing_continuum_summary.tex"
    with path.open("w", encoding="utf-8") as f:
        f.write(r"""\begin{table}[t]
\centering
\caption{Continuum-grid extrapolation of the zero-energy boundaries adjacent to the benchmark three-negative-state sector. A free-power diagnostic $r_{c,n}(\Delta r)=r_{c,n}^{(\infty)}+A_n(\Delta r)^p$ identifies the observed leading order, after which the nearest supported order is used for the five-finest-grid intercept. The quoted uncertainty is a fit/extrapolation uncertainty at fixed cutoffs, defined as a conservative envelope of fit-window variation, the largest primary-fit residual, the intercept standard error, and the free-order versus selected-order intercept difference. It is not a total physical-model uncertainty.}
\label{tab:zero_energy_crossing_continuum_summary}
\begin{tabular}{c c c c c c}
\hline
Boundary & $N_-$ change & $p_{\rm obs}$ & $r_c^{(12000)}$ (nm) & $r_c^{(\infty)}$ (nm) & $R^2$ \\
\hline
""")
        for row in summary_rows:
            continuum = float(row["continuum_rc_crossing_nm"])
            uncertainty = float(row["fit_uncertainty_nm"])
            f.write(
                f"$E_{int(row['level'])}=0$ & "
                f"${int(row['nminus_left'])}\\to{int(row['nminus_right'])}$ & "
                f"{float(row['observed_convergence_order']):.4f} & "
                f"{float(row['production_banded_rc_nm']):.6f} & "
                f"${continuum:.6f}\\pm{uncertainty:.6f}$ & "
                f"{float(row['primary_fit_R2']):.6f} "
                + r"\\" + "\n"
            )
        f.write(r"""\hline
\end{tabular}
\end{table}
""")
    return path


def set_plot_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })


def write_diagnostic_figures(
    raw_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
) -> List[Path]:
    if not WRITE_DIAGNOSTIC_FIGURES:
        return []

    set_plot_style()
    paths: List[Path] = []

    for summary in summary_rows:
        crossing_id = str(summary["crossing_id"])
        level = int(summary["level"])
        rows = [
            row for row in raw_rows
            if row["crossing_id"] == crossing_id and row["status"] == "ok"
        ]
        h = np.asarray([float(row["delta_r_nm"]) for row in rows], dtype=float)
        y = np.asarray([float(row["rc_crossing_nm"]) for row in rows], dtype=float)
        selected_order = float(summary["selected_convergence_order"])
        fit = fixed_order_fit_finest(
            h, y, PRIMARY_FINEST_POINTS, selected_order
        )
        x = h**selected_order

        x_line = np.linspace(0.0, float(np.max(x)) * 1.04, 300)
        y_line = fit.intercept_nm + fit.slope_nm_per_nm_power * x_line

        fig, ax = plt.subplots(figsize=(5.3, 3.8))
        ax.plot(x, y, "o", ms=5.0, label="fixed-cutoff five-point FDM crossings")
        ax.plot(x_line, y_line, "-", lw=1.4, label="data-supported five-finest-grid fit")
        ax.plot([0.0], [fit.intercept_nm], "s", ms=5.5, label=r"$r_c^{(\infty)}$")
        if math.isclose(selected_order, 1.0):
            ax.set_xlabel(r"$\Delta r\;(\mathrm{nm})$")
        else:
            ax.set_xlabel(
                rf"$(\Delta r)^{{{selected_order:g}}}\;"
                rf"(\mathrm{{nm}}^{{{selected_order:g}}})$"
            )
        ax.set_ylabel(rf"$r_c$ at $E_{{{level}}}=0$ (nm)")
        ax.set_title(rf"Continuum extrapolation: $E_{{{level}}}=0$")
        ax.legend(frameon=True, fontsize=8)
        fig.tight_layout()

        pdf = BASE_DIR / f"zero_energy_crossing_continuum_fit_E{level}.pdf"
        png = BASE_DIR / f"zero_energy_crossing_continuum_fit_E{level}.png"
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
        fig.savefig(png, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        paths.extend([pdf, png])

    return paths


# ============================================================
# 8. Console report and main
# ============================================================

def print_summary(
    summary_rows: Sequence[Mapping[str, Any]],
    benchmark_rc_nm: float,
) -> None:
    print("\nContinuum-grid extrapolation results")
    print("------------------------------------")
    for row in summary_rows:
        print(
            f"E{int(row['level'])}=0: r_c(infinity)="
            f"{float(row['continuum_rc_crossing_nm']):.6f} +/- "
            f"{float(row['fit_uncertainty_nm']):.6f} nm (fit-only)"
        )
        print(
            f"  production sparse/banded difference = "
            f"{float(row['banded_minus_sparse_nm']):+.3e} nm"
        )
        print(
            f"  observed order p={float(row['observed_convergence_order']):.6f}; "
            f"selected p={float(row['selected_convergence_order']):.1f}"
        )
        print(
            f"  R^2={float(row['primary_fit_R2']):.8f}, "
            f"max residual={float(row['primary_fit_max_abs_residual_nm']):.3e} nm, "
            f"fit-window spread={float(row['fit_window_spread_nm']):.3e} nm, "
            f"model spread={float(row['model_selection_difference_nm']):.3e} nm, "
            f"pass={bool(row['fit_pass'])}"
        )

    lower = float(summary_rows[0]["continuum_rc_crossing_nm"])
    upper = float(summary_rows[1]["continuum_rc_crossing_nm"])
    print(
        f"\nBenchmark r_c={benchmark_rc_nm:.6f} nm remains inside "
        f"({lower:.6f}, {upper:.6f}) nm."
    )
    print(f"Margin to lower boundary : {benchmark_rc_nm - lower:.6f} nm")
    print(f"Margin to upper boundary : {upper - benchmark_rc_nm:.6f} nm")
    print("The quoted boundary uncertainties are fit/extrapolation uncertainties at fixed cutoffs.")


def main() -> None:
    solver, solver_path = load_map_module()
    crossing_path, all_specs = find_compatible_crossing_csv()
    selected_specs = select_adjacent_boundaries(
        all_specs,
        float(solver.RC_REF_NM),
    )

    print(f"Hamiltonian module : {solver_path.name}")
    print(f"Crossing input     : {crossing_path.name}")
    print("Selected adjacent boundaries:")
    for spec in selected_specs:
        print(
            f"  {spec.crossing_id}: E{spec.level}=0, "
            f"production r_c={spec.production_root_nm:.6f} nm, "
            f"N_- {spec.nminus_left}->{spec.nminus_right}"
        )

    raw_rows = compute_raw_rows(solver, selected_specs)
    failed = [row for row in raw_rows if row["status"] != "ok"]
    if failed:
        details = "; ".join(
            f"{row['crossing_id']} N={row['N']}: {row['error_message']}"
            for row in failed
        )
        raise RuntimeError("Continuum sweep contains failed rows: " + details)

    summary_rows = build_summary(
        raw_rows,
        selected_specs,
        float(solver.RC_REF_NM),
    )

    raw_csv = write_raw_csv(raw_rows)
    summary_csv = write_summary_csv(summary_rows)
    raw_tex = write_raw_latex(raw_rows)
    summary_tex = write_summary_latex(summary_rows)
    figure_paths = write_diagnostic_figures(raw_rows, summary_rows)

    print_summary(summary_rows, float(solver.RC_REF_NM))

    print("\nFiles written")
    print("-------------")
    print(f"Raw continuum CSV     : {raw_csv}")
    print(f"Summary continuum CSV : {summary_csv}")
    print(f"Raw LaTeX table       : {raw_tex}")
    print(f"Summary LaTeX table   : {summary_tex}")
    for path in figure_paths:
        print(f"Diagnostic figure     : {path}")

    checks_pass = all(
        bool(row["fit_pass"])
        and bool(row["production_solver_agreement_pass"])
        and bool(row["benchmark_inside_continuum_sector"])
        for row in summary_rows
    )
    if not checks_pass:
        raise RuntimeError(
            "At least one final continuum validation check failed. Inspect the outputs."
        )


if __name__ == "__main__":
    main()
