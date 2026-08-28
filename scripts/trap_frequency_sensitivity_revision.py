from __future__ import annotations

"""
Trap-frequency sensitivity study for the calibrated soft-core radial Hamiltonian.
==============================================================================

Purpose
-------
This script addresses the trap-frequency sensitivity requested during manuscript
revision while preserving the numerical conventions of the continuum-certified
benchmark.

It performs two logically distinct tests at the experimentally relevant radial
secular frequencies 1.10, 1.20, and 1.30 MHz:

A) Fixed-r_c sensitivity
The reference continuum-calibrated soft-core radius is kept fixed and only
   omega is changed.  This measures the direct sensitivity of the frozen reduced
   operator to the effective confinement scale.

B) Recalibrated sensitivity
   For each omega, r_c(omega) is recalibrated at the CONTINUUM level so that

       E_0^(infinity) / h = -15 MHz.

   The resulting E_1, E_2, and N_- then measure what remains frequency-sensitive
   after the same single experimental binding-energy anchor is imposed.

Authoritative numerical conventions
-----------------------------------
- fourth-order five-point FDM core from ``fdm_fourth_order_core.py``;
- radial domain 0 <= r <= 650 nm;
- explicit Dirichlet boundaries with odd-reflection stencil closure;
- continuum sequence N = (1800, 2556, 3630, 5155);
- ordinary centrifugal factor l(l+1) in FDM;
- V1 soft-core regulator;
- no method-dependent refitting beyond the explicitly labelled r_c(omega)
  recalibration in Part B.

The script is deliberately independent of the older r_min > 0 parameter-map
implementations.  It imports the certified common FDM core directly.

Required local files
--------------------
- fdm_fourth_order_core.py
- unified_model_params.py
- benchmark_reference.py

For convenience, if ``unified_model_params.py`` is absent but exactly one file
matching ``unified_model_params*.py`` is found beside this script, that file is
loaded under the canonical module name.  This makes uploaded copies such as
``unified_model_params(4).py`` usable without editing the certified FDM core.

Outputs
-------
A subdirectory ``trap_frequency_revision_results`` is created containing:

- trap_frequency_sensitivity_results.csv
- trap_frequency_sensitivity_comparison.csv
- trap_frequency_sensitivity_table.tex
- trap_frequency_sensitivity_text.tex
- trap_frequency_sensitivity_summary.json
- trap_frequency_sensitivity_figure.pdf
- trap_frequency_sensitivity_figure.png

The calculation aborts if the 1.20 MHz benchmark cannot be reproduced within
strict numerical tolerances.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import csv
import hashlib
import importlib.util
import json
import math
import sys

import numpy as np
from scipy.optimize import brentq, minimize_scalar
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


# =============================================================================
# 0. Robust local imports
# =============================================================================

HERE = Path(__file__).resolve().parent


def _bootstrap_unified_model_params() -> None:
    """Make the canonical ``unified_model_params`` module importable.

    The research repository normally contains ``unified_model_params.py``.
    Chat/upload systems sometimes rename duplicate files by appending ``(n)``;
    in that case we accept exactly one matching fallback file.
    """
    if "unified_model_params" in sys.modules:
        return

    canonical = HERE / "unified_model_params.py"
    if canonical.exists():
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        return

    candidates = sorted(
        p for p in HERE.glob("unified_model_params*.py")
        if p.name != Path(__file__).name
    )
    if len(candidates) != 1:
        names = ", ".join(p.name for p in candidates) or "none"
        raise ImportError(
            "Could not resolve unified_model_params.py. Expected the canonical "
            "file or exactly one fallback matching unified_model_params*.py; "
            f"found: {names}."
        )

    spec = importlib.util.spec_from_file_location("unified_model_params", candidates[0])
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {candidates[0].name}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["unified_model_params"] = module
    spec.loader.exec_module(module)


_bootstrap_unified_model_params()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from unified_model_params import PARAMS, PI  # noqa: E402
from fdm_fourth_order_core import (  # noqa: E402
    GridSpec,
    build_kinetic_banded,
    solve_banded_levels_MHz,
)
from benchmark_reference import (  # noqa: E402
    RC_NM as BENCHMARK_RC_NM,
    OMEGA_OVER_2PI_MHZ as BENCHMARK_OMEGA_MHZ,
    ELL as BENCHMARK_ELL,
    CONTINUUM_FDM_LEVELS_MHZ as BENCHMARK_LEVELS_MHZ,
)


# =============================================================================
# 1. Scientific and numerical configuration
# =============================================================================

FREQUENCIES_MHZ: Tuple[float, ...] = (1.10, 1.20, 1.30)
TARGET_E0_MHZ = -15.0
EXPERIMENTAL_BINDING_BAND_MHZ = (13.0, 17.0)

GRID_N: Tuple[int, ...] = (1800, 2556, 3630, 5155)
R_MAX_M = 650.0e-9
N_CONTINUUM_LEVELS = 3
N_LEVELS_FOR_COUNT = 8
REGULATOR = "V1"
ELL = int(PARAMS.l)

# Continuum calibration bracket.  It is expanded automatically if necessary.
RC_BRACKET_NM = (20.0, 35.0)
RC_MIN_ALLOWED_NM = 2.0
RC_MAX_ALLOWED_NM = 200.0
MAX_BRACKET_EXPANSIONS = 10
RC_ROOT_XTOL_NM = 1.0e-9
RC_ROOT_RTOL = 1.0e-12

# Validation gates.
BENCHMARK_LEVEL_TOL_HZ = 0.50
BENCHMARK_RC_TOL_NM = 1.0e-4
TARGET_RECALIBRATION_TOL_HZ = 1.0
ORDER_ACCEPTANCE_RANGE = (3.5, 4.5)
REQUIRE_STABLE_NEGATIVE_COUNT = True

OUTDIR = HERE / "trap_frequency_revision_results"


# =============================================================================
# 2. Data containers
# =============================================================================

@dataclass(frozen=True)
class ContinuumFit:
    E_infinity_MHz: float
    amplitude: float
    observed_order: float
    R2: float
    max_abs_fit_residual_Hz: float
    fit_standard_error_Hz: float


@dataclass(frozen=True)
class SpectrumResult:
    mode: str
    omega_MHz: float
    rc_nm: float
    grid_rows: Tuple[Dict[str, Any], ...]
    fits: Tuple[ContinuumFit, ...]
    negative_counts: Tuple[int, ...]

    @property
    def continuum_levels_MHz(self) -> Tuple[float, ...]:
        return tuple(f.E_infinity_MHz for f in self.fits)

    @property
    def n_negative(self) -> int:
        return int(self.negative_counts[-1])

    @property
    def negative_count_stable(self) -> bool:
        return len(set(self.negative_counts)) == 1


# =============================================================================
# 3. Shared continuum machinery
# =============================================================================


def build_operator_sequence() -> Tuple[Any, ...]:
    return tuple(
        build_kinetic_banded(GridSpec(N=N, r_max_m=R_MAX_M))
        for N in GRID_N
    )


OPERATORS = build_operator_sequence()
GRID_SPACINGS_NM = np.asarray([op.dr_m * 1.0e9 for op in OPERATORS], dtype=float)


def fit_level(h_nm: np.ndarray, energies_MHz: np.ndarray) -> ContinuumFit:
    """Fit E(h)=E_inf+A h^p with the observed order determined from the data.

    For a fixed p, E_inf and A enter linearly.  We therefore minimize the
    linear-least-squares residual over p in [2, 6].  This is considerably more
    stable than a simultaneous three-parameter nonlinear fit when the grid
    drift is only sub-Hz.
    """
    h_nm = np.asarray(h_nm, dtype=float)
    energies_MHz = np.asarray(energies_MHz, dtype=float)
    if h_nm.shape != energies_MHz.shape or h_nm.size < 4:
        raise ValueError("Continuum fit requires at least four matched grid points.")

    def linear_fit_for_power(power: float) -> Tuple[np.ndarray, np.ndarray, float]:
        x = h_nm ** float(power)
        design = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(design, energies_MHz, rcond=None)
        residuals = energies_MHz - design @ beta
        sse = float(np.sum(residuals**2))
        return beta, residuals, sse

    optimum = minimize_scalar(
        lambda power: linear_fit_for_power(float(power))[2],
        bounds=(2.0, 6.0),
        method="bounded",
        options={"xatol": 1.0e-13, "maxiter": 500},
    )
    if not optimum.success:
        raise RuntimeError(f"Continuum-order minimization failed: {optimum.message}")

    power = float(optimum.x)
    beta, residuals, sse = linear_fit_for_power(power)
    x = h_nm**power
    design = np.column_stack([np.ones_like(x), x])

    sst = float(np.sum((energies_MHz - np.mean(energies_MHz)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0.0 else 1.0

    dof = max(1, h_nm.size - 2)
    covariance = (sse / dof) * np.linalg.inv(design.T @ design)
    intercept_stderr_MHz = math.sqrt(max(0.0, float(covariance[0, 0])))

    return ContinuumFit(
        E_infinity_MHz=float(beta[0]),
        amplitude=float(beta[1]),
        observed_order=power,
        R2=float(r2),
        max_abs_fit_residual_Hz=float(np.max(np.abs(residuals)) * 1.0e6),
        fit_standard_error_Hz=float(intercept_stderr_MHz * 1.0e6),
    )


def omega_rad_s(omega_MHz: float) -> float:
    return 2.0 * PI * float(omega_MHz) * 1.0e6


def solve_grid_sequence(
    *,
    rc_nm: float,
    omega_MHz: float,
    mode: str,
    n_levels: int = N_LEVELS_FOR_COUNT,
) -> SpectrumResult:
    """Solve all certified grids and continuum-extrapolate E0, E1, E2."""
    if n_levels < N_CONTINUUM_LEVELS:
        raise ValueError("n_levels is smaller than the number of continuum levels.")

    rows: List[Dict[str, Any]] = []
    energy_matrix: List[np.ndarray] = []
    negative_counts: List[int] = []

    for N, op in zip(GRID_N, OPERATORS):
        levels = np.asarray(
            solve_banded_levels_MHz(
                op,
                rc_m=float(rc_nm) * 1.0e-9,
                n_levels=n_levels,
                omega_rad_s=omega_rad_s(omega_MHz),
                ell=ELL,
                regulator=REGULATOR,
            ),
            dtype=float,
        )
        if levels.size != n_levels:
            raise RuntimeError(
                f"Expected {n_levels} eigenvalues at N={N}, got {levels.size}."
            )

        n_negative = int(np.sum(levels < 0.0))
        if n_negative == n_levels:
            raise RuntimeError(
                f"All {n_levels} requested levels are negative at omega/2pi={omega_MHz:.3f} MHz, "
                f"r_c={rc_nm:.9f} nm, N={N}. Increase N_LEVELS_FOR_COUNT."
            )

        row: Dict[str, Any] = {
            "N": int(N),
            "delta_r_nm": float(op.dr_m * 1.0e9),
            "n_negative": n_negative,
        }
        for i, level in enumerate(levels):
            row[f"E{i}_MHz"] = float(level)
        rows.append(row)
        energy_matrix.append(levels)
        negative_counts.append(n_negative)

    matrix = np.vstack(energy_matrix)
    fits = tuple(
        fit_level(GRID_SPACINGS_NM, matrix[:, n])
        for n in range(N_CONTINUUM_LEVELS)
    )

    result = SpectrumResult(
        mode=mode,
        omega_MHz=float(omega_MHz),
        rc_nm=float(rc_nm),
        grid_rows=tuple(rows),
        fits=fits,
        negative_counts=tuple(negative_counts),
    )
    validate_continuum_result(result)
    return result


def validate_continuum_result(result: SpectrumResult) -> None:
    pmin, pmax = ORDER_ACCEPTANCE_RANGE
    for n, fit in enumerate(result.fits):
        if not (pmin <= fit.observed_order <= pmax):
            raise RuntimeError(
                f"Observed continuum order for state n={n} is p={fit.observed_order:.6f}, "
                f"outside the accepted range [{pmin}, {pmax}] at "
                f"omega/2pi={result.omega_MHz:.3f} MHz, r_c={result.rc_nm:.9f} nm."
            )
        if not np.isfinite(fit.R2) or fit.R2 < 0.999:
            raise RuntimeError(
                f"Continuum fit R^2={fit.R2:.9f} is unexpectedly low for state n={n}."
            )

    if REQUIRE_STABLE_NEGATIVE_COUNT and not result.negative_count_stable:
        raise RuntimeError(
            "Negative-state count changed across the certified grid sequence at "
            f"omega/2pi={result.omega_MHz:.3f} MHz, r_c={result.rc_nm:.9f} nm: "
            f"{result.negative_counts}."
        )


# =============================================================================
# 4. Continuum-level recalibration of r_c(omega)
# =============================================================================


def continuum_ground_energy_MHz(
    rc_nm: float,
    omega_MHz: float,
    cache: Dict[float, float],
) -> float:
    """Continuum E0/h in MHz for one (r_c, omega), memoized during root finding."""
    key = round(float(rc_nm), 12)
    if key in cache:
        return cache[key]

    values = np.asarray(
        [
            solve_banded_levels_MHz(
                op,
                rc_m=float(rc_nm) * 1.0e-9,
                n_levels=1,
                omega_rad_s=omega_rad_s(omega_MHz),
                ell=ELL,
                regulator=REGULATOR,
            )[0]
            for op in OPERATORS
        ],
        dtype=float,
    )
    fit = fit_level(GRID_SPACINGS_NM, values)
    cache[key] = float(fit.E_infinity_MHz)
    return cache[key]


def calibrate_rc_continuum(omega_MHz: float) -> Tuple[float, Dict[str, Any]]:
    """Find r_c such that the continuum-extrapolated ground state is -15 MHz."""
    cache: Dict[float, float] = {}

    def objective(rc_nm: float) -> float:
        return continuum_ground_energy_MHz(rc_nm, omega_MHz, cache) - TARGET_E0_MHZ

    lo, hi = map(float, RC_BRACKET_NM)
    flo, fhi = objective(lo), objective(hi)
    expansions = 0

    while flo * fhi > 0.0 and expansions < MAX_BRACKET_EXPANSIONS:
        span = hi - lo
        lo = max(RC_MIN_ALLOWED_NM, lo - 0.5 * span)
        hi = min(RC_MAX_ALLOWED_NM, hi + 0.5 * span)
        flo, fhi = objective(lo), objective(hi)
        expansions += 1
        if lo <= RC_MIN_ALLOWED_NM and hi >= RC_MAX_ALLOWED_NM:
            break

    if flo * fhi > 0.0:
        raise RuntimeError(
            f"Could not bracket continuum r_c calibration at omega/2pi={omega_MHz:.3f} MHz. "
            f"Last bracket [{lo:.6f}, {hi:.6f}] nm gave objective values "
            f"{flo:+.6f}, {fhi:+.6f} MHz."
        )

    root, info = brentq(
        objective,
        lo,
        hi,
        xtol=RC_ROOT_XTOL_NM,
        rtol=RC_ROOT_RTOL,
        maxiter=100,
        full_output=True,
        disp=True,
    )
    root_energy = continuum_ground_energy_MHz(float(root), omega_MHz, cache)

    diagnostic = {
        "omega_over_2pi_MHz": float(omega_MHz),
        "initial_bracket_nm": list(RC_BRACKET_NM),
        "final_bracket_nm": [float(lo), float(hi)],
        "bracket_expansions": int(expansions),
        "root_iterations": int(info.iterations),
        "root_function_calls": int(info.function_calls),
        "converged": bool(info.converged),
        "rc_nm": float(root),
        "continuum_E0_MHz": float(root_energy),
        "target_offset_Hz": float((root_energy - TARGET_E0_MHZ) * 1.0e6),
        "objective_evaluations": int(len(cache)),
    }
    return float(root), diagnostic


# =============================================================================
# 5. Validation gates
# =============================================================================


def find_result(results: Sequence[SpectrumResult], mode: str, omega_MHz: float) -> SpectrumResult:
    matches = [
        r for r in results
        if r.mode == mode and abs(r.omega_MHz - omega_MHz) < 5.0e-12
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {mode} result at {omega_MHz:.3f} MHz.")
    return matches[0]


def validate_benchmark(results: Sequence[SpectrumResult]) -> Dict[str, Any]:
    central = find_result(results, "fixed_rc", float(BENCHMARK_OMEGA_MHZ))
    obtained = np.asarray(central.continuum_levels_MHz, dtype=float)
    expected = np.asarray(BENCHMARK_LEVELS_MHZ[:N_CONTINUUM_LEVELS], dtype=float)
    delta_Hz = (obtained - expected) * 1.0e6

    if float(np.max(np.abs(delta_Hz))) > BENCHMARK_LEVEL_TOL_HZ:
        raise RuntimeError(
            "Central benchmark self-check FAILED. Continuum levels differ from "
            f"benchmark_reference.py by {delta_Hz.tolist()} Hz; tolerance is "
            f"{BENCHMARK_LEVEL_TOL_HZ:.3f} Hz."
        )

    if abs(float(BENCHMARK_RC_NM) - float(PARAMS.r_c * 1.0e9)) > 5.0e-10:
        raise RuntimeError(
            "PARAMS.r_c and benchmark_reference.RC_NM are inconsistent."
        )
    if ELL != int(BENCHMARK_ELL):
        raise RuntimeError("Angular-momentum sector differs from benchmark_reference.py.")

    return {
        "status": "PASS",
        "reference_rc_nm": float(BENCHMARK_RC_NM),
        "reference_omega_MHz": float(BENCHMARK_OMEGA_MHZ),
        "expected_levels_MHz": expected.tolist(),
        "obtained_levels_MHz": obtained.tolist(),
        "obtained_minus_expected_Hz": delta_Hz.tolist(),
        "tolerance_Hz": float(BENCHMARK_LEVEL_TOL_HZ),
    }


def validate_recalibration(
    results: Sequence[SpectrumResult],
    calibration_diagnostics: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    offsets = []
    for result in results:
        if result.mode != "recalibrated_rc":
            continue
        offset_Hz = (result.continuum_levels_MHz[0] - TARGET_E0_MHZ) * 1.0e6
        offsets.append(float(offset_Hz))
        if abs(offset_Hz) > TARGET_RECALIBRATION_TOL_HZ:
            raise RuntimeError(
                f"Recalibration target check FAILED at {result.omega_MHz:.3f} MHz: "
                f"E0 target offset = {offset_Hz:+.6f} Hz."
            )

    central = find_result(results, "recalibrated_rc", float(BENCHMARK_OMEGA_MHZ))
    rc_delta_nm = central.rc_nm - float(BENCHMARK_RC_NM)
    if abs(rc_delta_nm) > BENCHMARK_RC_TOL_NM:
        raise RuntimeError(
    "Central recalibrated r_c does not reproduce the reference continuum-calibrated "
    f"radius: delta r_c={rc_delta_nm:+.9e} nm; tolerance=")

    return {
        "status": "PASS",
        "target_MHz": float(TARGET_E0_MHZ),
        "target_offsets_Hz": offsets,
        "target_tolerance_Hz": float(TARGET_RECALIBRATION_TOL_HZ),
        "central_recalibrated_rc_nm": float(central.rc_nm),
        "reference_rc_nm": float(BENCHMARK_RC_NM),
        "central_rc_difference_nm": float(rc_delta_nm),
        "rc_tolerance_nm": float(BENCHMARK_RC_TOL_NM),
        "root_diagnostics": list(calibration_diagnostics),
    }


# =============================================================================
# 6. Reporting helpers
# =============================================================================


def result_to_flat_row(result: SpectrumResult) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "mode": result.mode,
        "omega_over_2pi_MHz": result.omega_MHz,
        "rc_nm": result.rc_nm,
        "rc_minus_reference_nm": result.rc_nm - float(BENCHMARK_RC_NM),
        "n_negative": result.n_negative,
        "negative_count_grid_sequence": ";".join(str(x) for x in result.negative_counts),
        "negative_count_stable": result.negative_count_stable,
    }
    for n, fit in enumerate(result.fits):
        row[f"E{n}_continuum_MHz"] = fit.E_infinity_MHz
        row[f"p{n}"] = fit.observed_order
        row[f"R2_{n}"] = fit.R2
        row[f"fit_residual{n}_Hz"] = fit.max_abs_fit_residual_Hz
        row[f"fit_stderr{n}_Hz"] = fit.fit_standard_error_Hz
    row["E0_target_offset_kHz"] = 1.0e3 * (
        result.continuum_levels_MHz[0] - TARGET_E0_MHZ
    )
    return row


def comparison_rows(results: Sequence[SpectrumResult]) -> List[Dict[str, Any]]:
    fixed_central = find_result(results, "fixed_rc", float(BENCHMARK_OMEGA_MHZ))
    recal_central = find_result(results, "recalibrated_rc", float(BENCHMARK_OMEGA_MHZ))

    rows: List[Dict[str, Any]] = []
    for omega in FREQUENCIES_MHZ:
        fixed = find_result(results, "fixed_rc", omega)
        recal = find_result(results, "recalibrated_rc", omega)
        rows.append({
            "omega_over_2pi_MHz": omega,
            "fixed_rc_nm": fixed.rc_nm,
            "fixed_E0_MHz": fixed.continuum_levels_MHz[0],
            "fixed_E1_MHz": fixed.continuum_levels_MHz[1],
            "fixed_E2_MHz": fixed.continuum_levels_MHz[2],
            "fixed_N_negative": fixed.n_negative,
            "fixed_delta_E0_vs_1p2_kHz": 1.0e3 * (
                fixed.continuum_levels_MHz[0] - fixed_central.continuum_levels_MHz[0]
            ),
            "fixed_delta_E1_vs_1p2_kHz": 1.0e3 * (
                fixed.continuum_levels_MHz[1] - fixed_central.continuum_levels_MHz[1]
            ),
            "fixed_delta_E2_vs_1p2_kHz": 1.0e3 * (
                fixed.continuum_levels_MHz[2] - fixed_central.continuum_levels_MHz[2]
            ),
            "recalibrated_rc_nm": recal.rc_nm,
            "recalibrated_rc_shift_vs_1p2_nm": recal.rc_nm - recal_central.rc_nm,
            "recalibrated_E0_MHz": recal.continuum_levels_MHz[0],
            "recalibrated_E1_MHz": recal.continuum_levels_MHz[1],
            "recalibrated_E2_MHz": recal.continuum_levels_MHz[2],
            "recalibrated_N_negative": recal.n_negative,
            "recalibrated_delta_E1_vs_1p2_kHz": 1.0e3 * (
                recal.continuum_levels_MHz[1] - recal_central.continuum_levels_MHz[1]
            ),
            "recalibrated_delta_E2_vs_1p2_kHz": 1.0e3 * (
                recal.continuum_levels_MHz[2] - recal_central.continuum_levels_MHz[2]
            ),
        })
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest() -> Dict[str, Any]:
    files = [
        Path(__file__).resolve(),
        HERE / "fdm_fourth_order_core.py",
        HERE / "benchmark_reference.py",
    ]

    canonical = HERE / "unified_model_params.py"
    if canonical.exists():
        files.append(canonical)
    else:
        candidates = sorted(HERE.glob("unified_model_params*.py"))
        if candidates:
            files.append(candidates[0])

    repo_root = HERE.parent.resolve()

    repo_root = HERE.parent.resolve()

    payload: Dict[str, Any] = {}
    for path in files:
        if path.exists():
            resolved_path = path.resolve()

            try:
                relative_path = resolved_path.relative_to(repo_root).as_posix()
            except ValueError:
                relative_path = resolved_path.name

            payload[resolved_path.name] = {
                "path": relative_path,
                "sha256": sha256_file(resolved_path),
            }

    return payload


# =============================================================================
# 7. Output writers
# =============================================================================


def write_results_csv(results: Sequence[SpectrumResult]) -> Path:
    path = OUTDIR / "trap_frequency_sensitivity_results.csv"
    rows = [result_to_flat_row(r) for r in results]
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_comparison_csv(rows: Sequence[Dict[str, Any]]) -> Path:
    path = OUTDIR / "trap_frequency_sensitivity_comparison.csv"
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_latex_table(rows: Sequence[Dict[str, Any]]) -> Path:
    path = OUTDIR / "trap_frequency_sensitivity_table.tex"
    with path.open("w", encoding="utf-8") as f:
        f.write(r"""\begin{table*}[t]
\centering
\caption{Continuum-certified sensitivity to the effective radial trap frequency. In the fixed-$r_c$ test the reference calibrated radius is retained while only $\omega$ is varied. In the recalibrated test, $r_c(\omega)$ is adjusted at the continuum level so that $E_0/h=-15~\mathrm{MHz}$ at each frequency. All entries use the same fourth-order FDM sequence, radial domain, boundary closure, and $\ell=0$ Hamiltonian.}
\label{tab:trap_frequency_sensitivity}
\begin{tabular}{c c c c c c c c c}
\hline
$\omega/2\pi$ & \multicolumn{4}{c}{Fixed $r_c$} & \multicolumn{4}{c}{Continuum-recalibrated $r_c(\omega)$} \\
(MHz) & $E_0/h$ & $E_1/h$ & $E_2/h$ & $N_-$ & $r_c$ (nm) & $E_1/h$ & $E_2/h$ & $N_-$ \\
& \multicolumn{3}{c}{(MHz)} & & & \multicolumn{2}{c}{(MHz)} & \\
\hline
""")
        for row in rows:
            f.write(
                f"{float(row['omega_over_2pi_MHz']):.2f} & "
                f"{float(row['fixed_E0_MHz']):+.6f} & "
                f"{float(row['fixed_E1_MHz']):+.6f} & "
                f"{float(row['fixed_E2_MHz']):+.6f} & "
                f"{int(row['fixed_N_negative'])} & "
                f"{float(row['recalibrated_rc_nm']):.6f} & "
                f"{float(row['recalibrated_E1_MHz']):+.6f} & "
                f"{float(row['recalibrated_E2_MHz']):+.6f} & "
                f"{int(row['recalibrated_N_negative'])} " + r"\\" + "\n"
            )
        f.write(r"""\hline
\end{tabular}
\end{table*}
""")
    return path


def write_paper_text(rows: Sequence[Dict[str, Any]]) -> Path:
    """Write a compact manuscript-ready paragraph using the computed values."""
    path = OUTDIR / "trap_frequency_sensitivity_text.tex"
    by_omega = {round(float(r["omega_over_2pi_MHz"]), 2): r for r in rows}
    r11 = by_omega[1.10]
    r12 = by_omega[1.20]
    r13 = by_omega[1.30]

    fixed_e0_span = max(r["fixed_E0_MHz"] for r in rows) - min(r["fixed_E0_MHz"] for r in rows)
    fixed_e1_span = max(r["fixed_E1_MHz"] for r in rows) - min(r["fixed_E1_MHz"] for r in rows)
    fixed_e2_span = max(r["fixed_E2_MHz"] for r in rows) - min(r["fixed_E2_MHz"] for r in rows)
    recal_e1_span = max(r["recalibrated_E1_MHz"] for r in rows) - min(r["recalibrated_E1_MHz"] for r in rows)
    recal_e2_span = max(r["recalibrated_E2_MHz"] for r in rows) - min(r["recalibrated_E2_MHz"] for r in rows)
    rc_span = max(r["recalibrated_rc_nm"] for r in rows) - min(r["recalibrated_rc_nm"] for r in rows)

    text = rf"""As a direct sensitivity test of the isotropic confinement scale, we repeated the continuum-certified calculation at $\omega/2\pi=1.10$, $1.20$, and $1.30~\mathrm{{MHz}}$, spanning the two experimental radial secular frequencies.  With the calibrated radius fixed at $r_c={BENCHMARK_RC_NM:.6f}~\mathrm{{nm}}$, the three negative-energy levels remain present throughout the interval.  The continuum ground-state energies are {r11['fixed_E0_MHz']:+.6f}, {r12['fixed_E0_MHz']:+.6f}, and {r13['fixed_E0_MHz']:+.6f} MHz, respectively, corresponding to a total endpoint span of {fixed_e0_span:.6f} MHz.  The analogous spans of the first and second excited levels are {fixed_e1_span:.6f} and {fixed_e2_span:.6f} MHz.

To separate direct confinement sensitivity from the use of the single binding-energy calibration anchor, we also recalibrated $r_c$ independently at each frequency using the continuum condition $E_0/h=-15~\mathrm{{MHz}}$.  The required radii are {r11['recalibrated_rc_nm']:.6f}, {r12['recalibrated_rc_nm']:.6f}, and {r13['recalibrated_rc_nm']:.6f} nm, a total variation of {rc_span:.6f} nm across the tested interval.  After recalibration, the negative-state count remains $N_-=3$, while the first excited level changes from {r11['recalibrated_E1_MHz']:+.6f} to {r13['recalibrated_E1_MHz']:+.6f} MHz and the second from {r11['recalibrated_E2_MHz']:+.6f} to {r13['recalibrated_E2_MHz']:+.6f} MHz, with endpoint spans of {recal_e1_span:.6f} and {recal_e2_span:.6f} MHz.  These shifts quantify sensitivity within the reduced stationary model and should not be interpreted as an uncertainty estimate for the full anisotropic Paul-trap dynamics.
"""
    path.write_text(text, encoding="utf-8")
    return path


def write_summary_json(
    results: Sequence[SpectrumResult],
    comparisons: Sequence[Dict[str, Any]],
    benchmark_check: Dict[str, Any],
    recalibration_check: Dict[str, Any],
) -> Path:
    path = OUTDIR / "trap_frequency_sensitivity_summary.json"
    payload = {
        "purpose": "continuum-certified trap-frequency sensitivity for manuscript revision",
        "scientific_tests": {
            "fixed_rc": (
                "vary omega at the reference continuum-calibrated r_c; no recalibration"
            ),
            "recalibrated_rc": (
                "for each omega, recalibrate r_c at the continuum level to E0/h=-15 MHz"
            ),
        },
        "frequencies_MHz": list(FREQUENCIES_MHZ),
        "target_E0_MHz": TARGET_E0_MHZ,
        "experimental_binding_band_MHz": list(EXPERIMENTAL_BINDING_BAND_MHZ),
        "reference_rc_nm": float(BENCHMARK_RC_NM),
        "ell": ELL,
        "regulator": REGULATOR,
        "r_min_nm": 0.0,
        "r_max_nm": R_MAX_M * 1.0e9,
        "grid_N": list(GRID_N),
        "boundary_convention": "explicit Dirichlet values with odd-reflection fourth-order closure",
        "continuum_fit": "E_n(h)=E_n(infinity)+A_n h^p_n with p_n determined from the grid sequence",
        "results": [result_to_flat_row(r) for r in results],
        "comparison": list(comparisons),
        "validation": {
            "benchmark": benchmark_check,
            "recalibration": recalibration_check,
        },
        "source_manifest": source_manifest(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# =============================================================================
# 8. Figure
# =============================================================================


def set_publication_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.0,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.7,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 7.6,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def format_axis(ax: Any) -> None:
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", direction="in", top=True, right=True)


def write_figure(rows: Sequence[Dict[str, Any]]) -> Tuple[Path, Path]:
    set_publication_style()
    omega = np.asarray([r["omega_over_2pi_MHz"] for r in rows], dtype=float)

    fixed = {
        n: np.asarray([r[f"fixed_E{n}_MHz"] for r in rows], dtype=float)
        for n in range(3)
    }
    recal = {
        n: np.asarray([r[f"recalibrated_E{n}_MHz"] for r in rows], dtype=float)
        for n in range(3)
    }
    rc = np.asarray([r["recalibrated_rc_nm"] for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.15), constrained_layout=True)

    ax = axes[0]
    ax.axhspan(-EXPERIMENTAL_BINDING_BAND_MHZ[1], -EXPERIMENTAL_BINDING_BAND_MHZ[0], alpha=0.12)
    for n, marker in zip(range(3), ("o", "s", "^")):
        ax.plot(omega, fixed[n], marker=marker, linewidth=1.25, markersize=4.0, label=rf"$E_{n}/h$")
    ax.axhline(TARGET_E0_MHZ, linestyle="--", linewidth=0.9)
    ax.set_xlabel(r"Trap frequency $\omega/2\pi$ (MHz)")
    ax.set_ylabel(r"Energy $E_n/h$ (MHz)")
    ax.set_title(r"(a) Fixed $r_c$")
    ax.legend(frameon=False)
    format_axis(ax)

    ax = axes[1]
    ax.plot(omega, recal[1], marker="s", linewidth=1.25, markersize=4.0, label=r"$E_1/h$")
    ax.plot(omega, recal[2], marker="^", linewidth=1.25, markersize=4.0, label=r"$E_2/h$")
    ax.set_xlabel(r"Trap frequency $\omega/2\pi$ (MHz)")
    ax.set_ylabel(r"Conditional excited energy (MHz)")
    ax.set_title(r"(b) After $E_0$ recalibration")
    ax.legend(frameon=False)
    format_axis(ax)

    ax = axes[2]
    ax.plot(omega, rc, marker="o", linewidth=1.25, markersize=4.0)
    ax.axhline(float(BENCHMARK_RC_NM), linestyle="--", linewidth=0.9)
    ax.set_xlabel(r"Trap frequency $\omega/2\pi$ (MHz)")
    ax.set_ylabel(r"Calibrated $r_c$ (nm)")
    ax.set_title(r"(c) Continuum-calibrated $r_c(\omega)$")
    format_axis(ax)

    pdf_path = OUTDIR / "trap_frequency_sensitivity_figure.pdf"
    png_path = OUTDIR / "trap_frequency_sensitivity_figure.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    return pdf_path, png_path


# =============================================================================
# 9. Console report
# =============================================================================


def print_result_block(result: SpectrumResult) -> None:
    e = result.continuum_levels_MHz
    p = tuple(f.observed_order for f in result.fits)
    print(
        f"{result.mode:>16s} | omega/2pi={result.omega_MHz:.2f} MHz | "
        f"r_c={result.rc_nm:.9f} nm | "
        f"E=({e[0]:+.9f}, {e[1]:+.9f}, {e[2]:+.9f}) MHz | "
        f"p=({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}) | "
        f"N_-={result.n_negative}"
    )


def print_comparison_table(rows: Sequence[Dict[str, Any]]) -> None:
    print("\n=== Trap-frequency sensitivity summary ===")
    header = (
        f"{'omega':>6s} | {'fixed E0':>11s} {'fixed E1':>11s} {'fixed E2':>11s} {'N-':>3s} | "
        f"{'recal rc':>11s} {'recal E1':>11s} {'recal E2':>11s} {'N-':>3s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{float(r['omega_over_2pi_MHz']):6.2f} | "
            f"{float(r['fixed_E0_MHz']):+11.6f} "
            f"{float(r['fixed_E1_MHz']):+11.6f} "
            f"{float(r['fixed_E2_MHz']):+11.6f} "
            f"{int(r['fixed_N_negative']):3d} | "
            f"{float(r['recalibrated_rc_nm']):11.6f} "
            f"{float(r['recalibrated_E1_MHz']):+11.6f} "
            f"{float(r['recalibrated_E2_MHz']):+11.6f} "
            f"{int(r['recalibrated_N_negative']):3d}"
        )


# =============================================================================
# 10. Main workflow
# =============================================================================


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=== CONTINUUM-CERTIFIED TRAP-FREQUENCY SENSITIVITY ===")
    print(f"Frequencies omega/2pi [MHz] : {FREQUENCIES_MHZ}")
    print(f"Reference r_c [nm]          : {BENCHMARK_RC_NM:.9f}")
    print(f"Target E0/h [MHz]           : {TARGET_E0_MHZ:+.6f}")
    print(f"Grid sequence               : {GRID_N}")
    print(f"Radial domain [nm]          : [0, {R_MAX_M * 1e9:.1f}]")
    print("Boundary closure             : explicit Dirichlet + odd reflection")
    print("Recalibration definition     : continuum E0/h = -15 MHz")
    print()

    results: List[SpectrumResult] = []
    calibration_diagnostics: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Part A: fixed-r_c sensitivity.
    # -------------------------------------------------------------------------
    print("--- Part A: fixed-r_c frequency sensitivity ---")
    for omega in FREQUENCIES_MHZ:
        result = solve_grid_sequence(
            rc_nm=float(BENCHMARK_RC_NM),
            omega_MHz=float(omega),
            mode="fixed_rc",
        )
        results.append(result)
        print_result_block(result)

    benchmark_check = validate_benchmark(results)
    print("Benchmark self-check: PASS")

    # -------------------------------------------------------------------------
    # Part B: continuum-recalibrated r_c(omega).
    # -------------------------------------------------------------------------
    print("\n--- Part B: continuum-recalibrated frequency sensitivity ---")
    for omega in FREQUENCIES_MHZ:
        rc_star_nm, diagnostic = calibrate_rc_continuum(float(omega))
        calibration_diagnostics.append(diagnostic)
        result = solve_grid_sequence(
            rc_nm=rc_star_nm,
            omega_MHz=float(omega),
            mode="recalibrated_rc",
        )
        results.append(result)
        print_result_block(result)
        print(
            f"                 calibration target offset = "
            f"{(result.continuum_levels_MHz[0] - TARGET_E0_MHZ) * 1e6:+.6f} Hz"
        )

    recalibration_check = validate_recalibration(results, calibration_diagnostics)
    print("Recalibration gates: PASS")

    # Keep a stable output order: all fixed rows, then all recalibrated rows.
    results.sort(key=lambda r: (0 if r.mode == "fixed_rc" else 1, r.omega_MHz))
    comparisons = comparison_rows(results)
    print_comparison_table(comparisons)

    # -------------------------------------------------------------------------
    # Write outputs.
    # -------------------------------------------------------------------------
    results_csv = write_results_csv(results)
    comparison_csv = write_comparison_csv(comparisons)
    table_tex = write_latex_table(comparisons)
    text_tex = write_paper_text(comparisons)
    summary_json = write_summary_json(
        results,
        comparisons,
        benchmark_check,
        recalibration_check,
    )
    figure_pdf, figure_png = write_figure(comparisons)

    print("\n=== FINAL STATUS: PASS ===")
    print("All benchmark, convergence-order, state-count, and recalibration gates passed.")
    print("Files written:")
    for path in (
        results_csv,
        comparison_csv,
        table_tex,
        text_tex,
        summary_json,
        figure_pdf,
        figure_png,
    ):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
