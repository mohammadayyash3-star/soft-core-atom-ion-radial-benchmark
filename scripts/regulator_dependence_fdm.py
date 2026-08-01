from __future__ import annotations


from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import math
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from unified_model_params import PARAMS, HBAR, HPLANCK


# ============================================================
# User controls
# ============================================================
TARGET_MHZ = 15.0
TARGET_E_J = HPLANCK * TARGET_MHZ * 1.0e6
OUTDIR = Path(__file__).resolve().parent

# Calibration bracket for r_c. The bracket is expanded automatically if needed.
RC_BRACKET_NM = (10.0, 80.0)
RC_MIN_ALLOWED_NM = 2.0
RC_MAX_ALLOWED_NM = 250.0

# Number of negative levels to display/compare.
N_COMPARE_LEVELS = 3

# Plotting range for potentials and radial densities.
PLOT_R_MIN_NM = 0.5
PLOT_R_MAX_NM = 90.0
DENSITY_R_MAX_NM = 40.0

# ============================================================
# Post-calibration validation controls
# ============================================================
RUN_VALIDATION_SWEEPS = True

VALIDATION_N_SWEEP = (9000, 12000, 16000)
VALIDATION_RMAX_NM_SWEEP = (650.0, 800.0, 1000.0)


@dataclass(frozen=True)
class FDMConfig:
    r_min: float = 1.0e-10       # m = 0.1 nm
    r_max: float = 650.0e-9      # m
    N: int = 12000
    sigma: float = -1.0e-26
    k: int = 10
    eig_tol: float = 1.0e-11
    eig_maxiter: int = 30000


@dataclass
class ModelParams:
    r_c: float
    r_min: float
    r_max: float
    N: int
    sigma: float

    m_atom: float = PARAMS.m_atom
    m_ion: float = PARAMS.m_ion
    C4: float = PARAMS.C4
    omega_ion: float = PARAMS.omega_ion
    l: int = PARAMS.l
    use_langer: bool = PARAMS.use_langer_numerical


@dataclass(frozen=True)
class RegulatorSpec:
    key: str
    label: str
    latex: str
    potential: Callable[[np.ndarray, float, float], np.ndarray]


# ============================================================
# Shared physics helpers
# ============================================================
def reduced_mass(p: ModelParams) -> float:
    return p.m_atom * p.m_ion / (p.m_atom + p.m_ion)


def alpha_pol(p: ModelParams) -> float:
    return -0.5 * p.C4


def l_eff_numerical(p: ModelParams) -> float:
    return (p.l + 0.5) ** 2 if p.use_langer else p.l * (p.l + 1.0)


def energy_J_to_MHz(E_J: float) -> float:
    return E_J / HPLANCK / 1.0e6


def MHz_to_J(E_MHz: float) -> float:
    return HPLANCK * E_MHz * 1.0e6


# ============================================================
# Regulator potential definitions
# ============================================================
def V_reg_1(r: np.ndarray, alpha: float, rc: float) -> np.ndarray:
    """Production soft-core regulator: alpha/(r^4 + rc^4)."""
    return alpha / (r**4 + rc**4)


def V_reg_2(r: np.ndarray, alpha: float, rc: float) -> np.ndarray:
    """Squared quadratic regulator: alpha/(r^2 + rc^2)^2."""
    return alpha / (r**2 + rc**2) ** 2


def V_reg_3(r: np.ndarray, alpha: float, rc: float) -> np.ndarray:
    """
    Exponential cutoff regulator:
        alpha/r^4 * [1 - exp(-r^4/rc^4)].

    Implemented in the numerically stable equivalent form
        alpha/rc^4 * [1 - exp(-z)]/z,  z = r^4/rc^4,
    with the z -> 0 limit equal to alpha/rc^4.
    """
    z = (r / rc) ** 4
    ratio = np.empty_like(z, dtype=float)
    small = z < 1.0e-10
    ratio[small] = 1.0 - 0.5 * z[small] + (z[small] ** 2) / 6.0
    ratio[~small] = -np.expm1(-z[~small]) / z[~small]
    return alpha / rc**4 * ratio


REGULATORS: Tuple[RegulatorSpec, ...] = (
    RegulatorSpec(
        key="V1",
        label="V1: alpha/(r^4 + rc^4)",
        latex=r"$V_1(r)=\alpha/(r^4+r_c^4)$",
        potential=V_reg_1,
    ),
    RegulatorSpec(
        key="V2",
        label="V2: alpha/(r^2 + rc^2)^2",
        latex=r"$V_2(r)=\alpha/(r^2+r_c^2)^2$",
        potential=V_reg_2,
    ),
    RegulatorSpec(
        key="V3",
        label="V3: exponential cutoff",
        latex=r"$V_3(r)=\alpha r^{-4}\left[1-e^{-r^4/r_c^4}\right]$",
        potential=V_reg_3,
    ),
)


# ============================================================
# Effective potential and FDM operator
# ============================================================
def effective_potential(r: np.ndarray, p: ModelParams, reg: RegulatorSpec) -> np.ndarray:
    mu = reduced_mass(p)
    alpha = alpha_pol(p)
    leff = l_eff_numerical(p)

    v_harm = 0.5 * mu * (p.omega_ion**2) * r**2
    v_core = reg.potential(r, alpha, p.r_c)

    if leff != 0.0:
        v_cent = (HBAR**2 * leff) / (2.0 * mu * r**2)
    else:
        v_cent = np.zeros_like(r)

    return v_harm + v_core + v_cent


def build_fdm_operator(p: ModelParams, reg: RegulatorSpec) -> Tuple[sp.csr_matrix, np.ndarray, Tuple[int, int], float]:
    """
    Build the radial Hamiltonian matrix using a fourth-order, five-point stencil
    for the second derivative. Boundary rows near the edges are omitted, matching
    the established benchmark convention.
    """
    if p.N < 7:
        raise ValueError("N must be at least 7 for a five-point finite-difference stencil.")

    r = np.linspace(p.r_min, p.r_max, p.N, dtype=float)
    dr = float(r[1] - r[0])
    mu = reduced_mass(p)

    i0, i1 = 2, p.N - 3
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

    tmat = -(HBAR**2) / (2.0 * mu) * lap
    hmat = tmat + sp.diags(effective_potential(r_int, p, reg), format="csr")
    return hmat.tocsr(), r, (i0, i1), dr


# ============================================================
# State diagnostics
# ============================================================
def normalize_u(u: np.ndarray, r: np.ndarray) -> np.ndarray:
    norm = math.sqrt(float(np.trapezoid(np.abs(u) ** 2, r)))
    if not np.isfinite(norm) or norm <= 0.0:
        raise FloatingPointError("Radial wavefunction normalization failed.")
    return u / norm


def count_nodes(u_int: np.ndarray) -> int:
    amp = float(np.max(np.abs(u_int)))
    eps = max(1.0e-14, 1.0e-8 * amp)
    s = np.sign(np.where(np.abs(u_int) < eps, 0.0, u_int))
    return int(np.sum(s[1:] * s[:-1] < 0.0))


def eigen_residual(hmat: sp.csr_matrix, vec_int: np.ndarray, E_J: float) -> float:
    res = hmat @ vec_int - E_J * vec_int
    denom = max(1.0e-30, abs(E_J) * np.linalg.norm(vec_int))
    return float(np.linalg.norm(res) / denom)


def radial_observables(u: np.ndarray, r: np.ndarray, rc: float) -> Dict[str, float]:
    dens = np.abs(u) ** 2
    r_mean = float(np.trapezoid(r * dens, r))
    r2_mean = float(np.trapezoid((r**2) * dens, r))
    rms = math.sqrt(max(0.0, r2_mean))

    mask_core = r < rc
    if np.any(mask_core):
        p_inside = float(np.trapezoid(dens[mask_core], r[mask_core]))
    else:
        p_inside = 0.0

    peak_index = int(np.argmax(dens))
    r_peak = float(r[peak_index])

    return {
        "r_mean_nm": r_mean * 1.0e9,
        "r_rms_nm": rms * 1.0e9,
        "r_peak_nm": r_peak * 1.0e9,
        "P_r_lt_rc": p_inside,
    }


# ============================================================
# FDM solver
# ============================================================
def solve_fdm_states(
    p: ModelParams,
    reg: RegulatorSpec,
    cfg: FDMConfig,
    return_wavefunctions: bool = False,
) -> Dict[str, object]:
    hmat, r, (i0, i1), dr = build_fdm_operator(p, reg)
    dim = hmat.shape[0]
    k_eff = min(cfg.k, dim - 2)
    ncv = min(dim, max(2 * k_eff + 8, 24))

    evals, evecs = spla.eigsh(
        hmat,
        k=k_eff,
        sigma=p.sigma,
        which="LM",
        tol=cfg.eig_tol,
        maxiter=cfg.eig_maxiter,
        ncv=ncv,
    )

    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    states: List[Dict[str, object]] = []
    for j, E_J in enumerate(evals):
        vec_int = evecs[:, j].copy()
        u = np.zeros_like(r)
        u[i0 : i1 + 1] = vec_int
        u = normalize_u(u, r)

        obs = radial_observables(u, r, p.r_c)
        entry: Dict[str, object] = {
            "index": j,
            "E_J": float(E_J),
            "E_over_h_MHz": float(energy_J_to_MHz(float(E_J))),
            "absE_over_h_MHz": float(abs(energy_J_to_MHz(float(E_J)))),
            "nodes": count_nodes(u[i0 : i1 + 1]),
            "classification": "bound" if E_J < 0.0 else "trap-confined",
            "eig_residual_rel": eigen_residual(hmat, vec_int, float(E_J)),
            **obs,
        }
        if return_wavefunctions:
            entry["u"] = u
        states.append(entry)

    states.sort(key=lambda s: float(s["E_J"]))
    negative_states = [s for s in states if float(s["E_J"]) < 0.0]
    ground = min(states, key=lambda s: float(s["E_J"]))

    return {
        "params": asdict(p),
        "regulator_key": reg.key,
        "regulator_label": reg.label,
        "r": r,
        "dr": dr,
        "states": states,
        "negative_states": negative_states,
        "n_negative": len(negative_states),
        "ground": ground,
    }


def make_params(rc_nm: float, cfg: FDMConfig) -> ModelParams:
    return ModelParams(
        r_c=rc_nm * 1.0e-9,
        r_min=cfg.r_min,
        r_max=cfg.r_max,
        N=cfg.N,
        sigma=cfg.sigma,
    )


def ground_energy_for_rc(rc_nm: float, reg: RegulatorSpec, cfg: FDMConfig) -> float:
    p = make_params(rc_nm, cfg)
    result = solve_fdm_states(p, reg, cfg, return_wavefunctions=False)
    return float(result["ground"]["E_J"])


# ============================================================
# Calibration of r_c for each regulator
# ============================================================
def calibration_objective(rc_nm: float, reg: RegulatorSpec, cfg: FDMConfig) -> float:
    E0 = ground_energy_for_rc(rc_nm, reg, cfg)
    return abs(E0) - TARGET_E_J


def find_calibration_bracket(reg: RegulatorSpec, cfg: FDMConfig) -> Tuple[float, float]:
    lo, hi = RC_BRACKET_NM
    f_lo = calibration_objective(lo, reg, cfg)
    f_hi = calibration_objective(hi, reg, cfg)

    # The expected behavior is f_lo > 0 and f_hi < 0. If not, expand safely.
    for _ in range(10):
        if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi <= 0.0:
            return lo, hi

        if f_lo < 0.0:
            lo = max(RC_MIN_ALLOWED_NM, lo * 0.70)
            f_lo = calibration_objective(lo, reg, cfg)

        if f_hi > 0.0:
            hi = min(RC_MAX_ALLOWED_NM, hi * 1.35)
            f_hi = calibration_objective(hi, reg, cfg)

        if lo <= RC_MIN_ALLOWED_NM and hi >= RC_MAX_ALLOWED_NM:
            break

    raise RuntimeError(
        f"Could not bracket calibration root for {reg.key}. "
        f"Last bracket: [{lo:.6f}, {hi:.6f}] nm, f_lo={f_lo:.6e}, f_hi={f_hi:.6e}."
    )


def calibrate_rc(reg: RegulatorSpec, cfg: FDMConfig) -> Dict[str, float]:
    print(f"\n--- Calibrating {reg.key}: {reg.label} ---")
    lo, hi = find_calibration_bracket(reg, cfg)
    print(f"Bracket: rc = [{lo:.6f}, {hi:.6f}] nm")

    def obj(x: float) -> float:
        return calibration_objective(x, reg, cfg)

    rc_star = brentq(obj, lo, hi, xtol=1.0e-6, rtol=1.0e-8, maxiter=80)
    E0_star = ground_energy_for_rc(rc_star, reg, cfg)
    print(f"Calibrated rc = {rc_star:.9f} nm")
    print(f"E0/h = {energy_J_to_MHz(E0_star):+.9f} MHz")

    return {
        "rc_cal_nm": float(rc_star),
        "E0_cal_MHz": float(energy_J_to_MHz(E0_star)),
        "target_error_Hz": float((abs(E0_star) - TARGET_E_J) / HPLANCK),
    }


# ============================================================
# Post-calibration comparison
# ============================================================
def state_overlap(u_a: np.ndarray, u_b: np.ndarray, r: np.ndarray) -> float:
    ov = float(np.trapezoid(u_a * u_b, r))
    return ov * ov


def add_overlaps(results: Dict[str, Dict[str, object]]) -> None:
    """Add same-index fidelity against V1, using V1 as the reference regulator."""
    ref_key = "V1"
    if ref_key not in results:
        return

    r_ref = results[ref_key]["r"]
    neg_ref = results[ref_key]["negative_states"]

    for key, res in results.items():
        r = res["r"]
        if len(r) != len(r_ref) or not np.allclose(r, r_ref, rtol=0.0, atol=0.0):
            raise RuntimeError("Overlap comparison requires a common radial grid.")

        neg = res["negative_states"]
        for n in range(min(len(neg_ref), len(neg), N_COMPARE_LEVELS)):
            u_ref = np.asarray(neg_ref[n]["u"], dtype=float)
            u_cur = np.asarray(neg[n]["u"], dtype=float)
            neg[n]["F_vs_V1"] = state_overlap(u_ref, u_cur, r)


def run_regulator_dependence(cfg: FDMConfig) -> Dict[str, Dict[str, object]]:
    calibration_rows: Dict[str, Dict[str, float]] = {}
    results: Dict[str, Dict[str, object]] = {}

    for reg in REGULATORS:
        cal = calibrate_rc(reg, cfg)
        calibration_rows[reg.key] = cal

        p = make_params(cal["rc_cal_nm"], cfg)
        res = solve_fdm_states(p, reg, cfg, return_wavefunctions=True)
        res["calibration"] = cal
        res["regulator_latex"] = reg.latex
        results[reg.key] = res

    add_overlaps(results)
    return results

def validation_rows_for_calibrated_rc(
    results: Dict[str, Dict[str, object]]
) -> List[Dict[str, object]]:
    """
    Validate the calibrated regulator spectra without recalibrating rc.
    This tests whether the post-calibration negative-state count and shallow
    excited states remain stable under changes of N and r_max.
    """
    rows: List[Dict[str, object]] = []

    for reg in REGULATORS:
        rc_nm = float(results[reg.key]["calibration"]["rc_cal_nm"])

        # N sweep at fixed r_max
        for N in VALIDATION_N_SWEEP:
            cfg = FDMConfig(N=N, r_max=650.0e-9)
            p = make_params(rc_nm, cfg)
            res = solve_fdm_states(p, reg, cfg, return_wavefunctions=False)
            neg = res["negative_states"]

            rows.append({
                "regulator": reg.key,
                "test_type": "N_sweep",
                "N": int(N),
                "r_max_nm": float(cfg.r_max * 1.0e9),
                "rc_cal_nm": rc_nm,
                "N_negative": int(res["n_negative"]),
                "E0_MHz": float(neg[0]["E_over_h_MHz"]) if len(neg) > 0 else float("nan"),
                "E1_MHz": float(neg[1]["E_over_h_MHz"]) if len(neg) > 1 else float("nan"),
                "E2_MHz": float(neg[2]["E_over_h_MHz"]) if len(neg) > 2 else float("nan"),
                "E2_exists": int(len(neg) > 2),
            })

        # r_max sweep at fixed N
        for rmax_nm in VALIDATION_RMAX_NM_SWEEP:
            cfg = FDMConfig(N=12000, r_max=rmax_nm * 1.0e-9)
            p = make_params(rc_nm, cfg)
            res = solve_fdm_states(p, reg, cfg, return_wavefunctions=False)
            neg = res["negative_states"]

            rows.append({
                "regulator": reg.key,
                "test_type": "rmax_sweep",
                "N": int(cfg.N),
                "r_max_nm": float(rmax_nm),
                "rc_cal_nm": rc_nm,
                "N_negative": int(res["n_negative"]),
                "E0_MHz": float(neg[0]["E_over_h_MHz"]) if len(neg) > 0 else float("nan"),
                "E1_MHz": float(neg[1]["E_over_h_MHz"]) if len(neg) > 1 else float("nan"),
                "E2_MHz": float(neg[2]["E_over_h_MHz"]) if len(neg) > 2 else float("nan"),
                "E2_exists": int(len(neg) > 2),
            })

    return rows
# ============================================================
# Export helpers
# ============================================================
def fmt_float(x: float, digits: int = 9) -> str:
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{x:.{digits}f}"


def collect_summary_rows(results: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for reg in REGULATORS:
        res = results[reg.key]
        cal = res["calibration"]
        neg = res["negative_states"]
        ground = neg[0] if neg else res["ground"]

        row: Dict[str, object] = {
            "regulator": reg.key,
            "regulator_label": reg.label,
            "rc_cal_nm": float(cal["rc_cal_nm"]),
            "E0_MHz": float(neg[0]["E_over_h_MHz"]) if len(neg) > 0 else float("nan"),
            "E1_MHz": float(neg[1]["E_over_h_MHz"]) if len(neg) > 1 else float("nan"),
            "E2_MHz": float(neg[2]["E_over_h_MHz"]) if len(neg) > 2 else float("nan"),
            "N_negative": int(res["n_negative"]),
            "r_mean_0_nm": float(ground["r_mean_nm"]),
            "r_rms_0_nm": float(ground["r_rms_nm"]),
            "r_peak_0_nm": float(ground["r_peak_nm"]),
            "P0_r_lt_rc": float(ground["P_r_lt_rc"]),
            "F0_vs_V1": float(ground.get("F_vs_V1", 1.0 if reg.key == "V1" else float("nan"))),
            "target_error_Hz": float(cal["target_error_Hz"]),
        }
        rows.append(row)
    return rows


def collect_level_rows(results: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for reg in REGULATORS:
        res = results[reg.key]
        cal = res["calibration"]
        for state in res["negative_states"]:
            rows.append(
                {
                    "regulator": reg.key,
                    "rc_cal_nm": float(cal["rc_cal_nm"]),
                    "n": int(state["index"]),
                    "E_MHz": float(state["E_over_h_MHz"]),
                    "absE_MHz": float(state["absE_over_h_MHz"]),
                    "nodes": int(state["nodes"]),
                    "r_mean_nm": float(state["r_mean_nm"]),
                    "r_rms_nm": float(state["r_rms_nm"]),
                    "r_peak_nm": float(state["r_peak_nm"]),
                    "P_r_lt_rc": float(state["P_r_lt_rc"]),
                    "F_vs_V1": float(state.get("F_vs_V1", 1.0 if reg.key == "V1" else float("nan"))),
                    "eig_residual_rel": float(state["eig_residual_rel"]),
                }
            )
    return rows
def add_relative_energy_differences(rows: List[Dict[str, object]]) -> None:
    ref = next(row for row in rows if row["regulator"] == "V1")
    E1_ref = float(ref["E1_MHz"])
    E2_ref = float(ref["E2_MHz"])

    for row in rows:
        row["Delta_E1_vs_V1_MHz"] = float(row["E1_MHz"]) - E1_ref
        row["Delta_E2_vs_V1_MHz"] = float(row["E2_MHz"]) - E2_ref

def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            vals = []
            for k in keys:
                v = row[k]
                if isinstance(v, float):
                    vals.append(f"{v:.12g}")
                else:
                    vals.append(str(v).replace(",", ";"))
            f.write(",".join(vals) + "\n")


def write_latex_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(r"\begin{table}[t]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Regulator-form dependence of the FDM benchmark spectrum after independent calibration of each regulator to $E_0/h\simeq -15~\mathrm{MHz}$. The ground state is a calibration condition; the nontrivial comparison is the post-calibration behavior of $E_1$, $E_2$, $N_-$, and the ground-state spatial diagnostics.}" + "\n")
        f.write(r"\label{tab:regulator_dependence_summary}" + "\n")
        f.write(r"\begin{tabular}{lrrrrrrrr}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"Reg. & $r_c^{\rm cal}$ (nm) & $E_0/h$ & $E_1/h$ & $E_2/h$ & $N_-$ & $\langle r\rangle_0$ & $r_{{\rm rms},0}$ & $P_0(r<r_c)$ \\" + "\n")
        f.write(r" & & \multicolumn{3}{c}{(MHz)} & & \multicolumn{2}{c}{(nm)} & \\" + "\n")
        f.write(r"\hline" + "\n")
        for row in rows:
            f.write(
                f"{row['regulator']} & "
                f"{float(row['rc_cal_nm']):.6f} & "
                f"{float(row['E0_MHz']):.6f} & "
                f"{float(row['E1_MHz']):.6f} & "
                f"{float(row['E2_MHz']):.6f} & "
                f"{int(row['N_negative'])} & "
                f"{float(row['r_mean_0_nm']):.3f} & "
                f"{float(row['r_rms_0_nm']):.3f} & "
                f"{float(row['P0_r_lt_rc']):.6f} \\\n"
            )
        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")


def write_latex_levels(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(r"\begin{table}[t]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Negative-energy FDM levels for the calibrated regulator-form test. Each regulator is first calibrated to the same ground-state binding scale, after which the excited negative levels and spatial diagnostics are compared.}" + "\n")
        f.write(r"\label{tab:regulator_dependence_levels}" + "\n")
        f.write(r"\begin{tabular}{llrrrrrr}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"Reg. & $n$ & $E_n/h$ (MHz) & nodes & $\langle r\rangle$ (nm) & $r_{\rm rms}$ (nm) & $P(r<r_c)$ & $\mathcal{F}_n$ vs. $V_1$ \\" + "\n")
        f.write(r"\hline" + "\n")
        for row in rows:
            if int(row["n"]) >= N_COMPARE_LEVELS:
                continue
            f.write(
                f"{row['regulator']} & "
                f"{int(row['n'])} & "
                f"{float(row['E_MHz']):.6f} & "
                f"{int(row['nodes'])} & "
                f"{float(row['r_mean_nm']):.3f} & "
                f"{float(row['r_rms_nm']):.3f} & "
                f"{float(row['P_r_lt_rc']):.6f} & "
                f"{float(row['F_vs_V1']):.6f} \\\n"
            )
        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")


# ============================================================
# Figures
# ============================================================
def apply_publication_axes(ax) -> None:
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="major", linewidth=0.4, alpha=0.35)


def save_current_figure(stem: str) -> None:
    pdf = OUTDIR / f"{stem}.pdf"
    png = OUTDIR / f"{stem}.png"
    plt.tight_layout()
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved: {pdf.name}, {png.name}")

from matplotlib.lines import Line2D

def plot_energy_ladder(results: Dict[str, Dict[str, object]]) -> None:
    # Wider figure
    fig, ax = plt.subplots(figsize=(9.2, 4.6))

    width = 0.26
    x_positions = np.arange(len(REGULATORS), dtype=float)

    # Plot negative bound states
    for i, reg in enumerate(REGULATORS):
        neg = results[reg.key]["negative_states"]

        for state in neg:
            E = float(state["E_over_h_MHz"])
            n_idx = int(state["index"])

            ax.hlines(
                E,
                i - width,
                i + width,
                linewidth=2.4,
                color="C0",
                zorder=3,
            )

            if n_idx < N_COMPARE_LEVELS and n_idx in (0, 1, 2):
                ax.text(
                    i + width + 0.04,
                    E + 0.12,
                    rf"$n={n_idx}$",
                    va="center",
                    ha="left",
                    fontsize=8.5,
                    clip_on=False,
                )

    # Reference lines
    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.1,
        color="C0",
        alpha=0.85,
        zorder=1,
    )

    ax.axhline(
        -TARGET_MHZ,
        linestyle=":",
        linewidth=1.4,
        color="C0",
        alpha=0.95,
        zorder=1,
    )

    # Axes labels and limits
    ax.set_xticks(x_positions)
    ax.set_xticklabels([reg.key for reg in REGULATORS])

    ax.set_ylabel(r"$E_n/h$ (MHz)")
    ax.set_xlabel("Regulator form")

    ax.set_ylim(-16.2, 0.8)
    ax.set_xlim(-0.45, len(REGULATORS) - 0.35)

    # Publication-style axes
    apply_publication_axes(ax)

    # Custom legend under the figure
    legend_handles = [
        Line2D(
            [0], [0],
            color="C0",
            linewidth=2.6,
            label=r"Bound states, $E<0$",
        ),
        Line2D(
            [0], [0],
            color="C0",
            linestyle="--",
            linewidth=1.2,
            label=r"$E=0$",
        ),
        Line2D(
            [0], [0],
            color="C0",
            linestyle=":",
            linewidth=1.5,
            label=rf"Calibration target, $-15~\mathrm{{MHz}}$",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=8.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        handlelength=3.4,
        columnspacing=2.0,
        handletextpad=0.7,
    )

    # Leave enough bottom space for the legend
    fig.subplots_adjust(bottom=0.26)

    save_current_figure("regulator_dependence_energy_ladder")


def plot_potentials(results: Dict[str, Dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    r_nm = np.linspace(PLOT_R_MIN_NM, PLOT_R_MAX_NM, 1200)
    r = r_nm * 1.0e-9

    for reg in REGULATORS:
        rc_nm = float(results[reg.key]["calibration"]["rc_cal_nm"])
        p = make_params(rc_nm, FDMConfig())
        V = reg.potential(r, alpha_pol(p), p.r_c)
        ax.plot(r_nm, V / HPLANCK / 1.0e6, label=f"{reg.key}, rc={rc_nm:.3f} nm", linewidth=1.6)

    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlim(PLOT_R_MIN_NM, PLOT_R_MAX_NM)
    ax.set_xlabel(r"$r$ (nm)")
    ax.set_ylabel(r"$V_{\rm reg}(r)/h$ (MHz)")
    ax.set_title("Regularized polarization potentials after calibration")
    ax.legend(frameon=False, fontsize=8)
    apply_publication_axes(ax)
    save_current_figure("regulator_dependence_potentials")


def plot_ground_density(results: Dict[str, Dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    for reg in REGULATORS:
        res = results[reg.key]
        r_nm = np.asarray(res["r"], dtype=float) * 1.0e9
        u0 = np.asarray(res["negative_states"][0]["u"], dtype=float)
        dens = u0**2
        mask = r_nm <= DENSITY_R_MAX_NM
        ax.plot(r_nm[mask], dens[mask] * 1.0e-9, label=reg.key, linewidth=1.6)
        rc_nm = float(res["calibration"]["rc_cal_nm"])
        ax.axvline(rc_nm, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlim(0.0, DENSITY_R_MAX_NM)
    ax.set_xlabel(r"$r$ (nm)")
    ax.set_ylabel(r"$|u_0(r)|^2$ (nm$^{-1}$)")
    #ax.set_title("Ground-state radial probability density")
    ax.legend(frameon=False, fontsize=8)
    apply_publication_axes(ax)
    save_current_figure("regulator_dependence_ground_density")


# ============================================================
# Console report
# ============================================================
def print_summary(rows: List[Dict[str, object]]) -> None:
    print("\n" + "=" * 110)
    print("REGULATOR-FORM DEPENDENCE SUMMARY")
    print("=" * 110)
    print(
        f"{'Reg.':<4} {'rc_cal(nm)':>12} {'E0/h(MHz)':>14} {'E1/h(MHz)':>14} "
        f"{'E2/h(MHz)':>14} {'N-':>4} {'<r>0(nm)':>10} {'rms0(nm)':>10} "
        f"{'P0(r<rc)':>11} {'F0 vs V1':>10}"
    )
    print("-" * 110)
    for row in rows:
        print(
            f"{row['regulator']:<4} "
            f"{float(row['rc_cal_nm']):12.6f} "
            f"{float(row['E0_MHz']):14.9f} "
            f"{float(row['E1_MHz']):14.9f} "
            f"{float(row['E2_MHz']):14.9f} "
            f"{int(row['N_negative']):4d} "
            f"{float(row['r_mean_0_nm']):10.3f} "
            f"{float(row['r_rms_0_nm']):10.3f} "
            f"{float(row['P0_r_lt_rc']):11.6f} "
            f"{float(row['F0_vs_V1']):10.6f}"
        )
    print("=" * 110)


def print_level_table(rows: List[Dict[str, object]]) -> None:
    print("\nNEGATIVE-LEVEL DETAILS")
    print("-" * 120)
    print(
        f"{'Reg.':<4} {'n':>3} {'E/h(MHz)':>14} {'|E|/h(MHz)':>14} {'nodes':>6} "
        f"{'<r>(nm)':>10} {'rms(nm)':>10} {'r_peak(nm)':>11} {'P(r<rc)':>10} {'F vs V1':>10} {'resid':>10}"
    )
    print("-" * 120)
    for row in rows:
        print(
            f"{row['regulator']:<4} "
            f"{int(row['n']):3d} "
            f"{float(row['E_MHz']):14.9f} "
            f"{float(row['absE_MHz']):14.9f} "
            f"{int(row['nodes']):6d} "
            f"{float(row['r_mean_nm']):10.3f} "
            f"{float(row['r_rms_nm']):10.3f} "
            f"{float(row['r_peak_nm']):11.3f} "
            f"{float(row['P_r_lt_rc']):10.6f} "
            f"{float(row['F_vs_V1']):10.6f} "
            f"{float(row['eig_residual_rel']):10.2e}"
        )
    print("-" * 120)
def write_latex_validation(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(r"\begin{table}[t]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Post-calibration numerical validation of the regulator-form test. "
                r"The calibrated value of $r_c$ is held fixed for each regulator; only the "
                r"FDM grid size $N$ or the outer boundary $r_{\max}$ is varied.}" + "\n")
        f.write(r"\label{tab:regulator_dependence_validation}" + "\n")
        f.write(r"\begin{tabular}{llrrrrrr}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"Reg. & Test & $N$ & $r_{\max}$ (nm) & $E_0/h$ & $E_1/h$ & $E_2/h$ & $N_-$ \\" + "\n")
        f.write(r" & & & & \multicolumn{3}{c}{(MHz)} & \\" + "\n")
        f.write(r"\hline" + "\n")

        for row in rows:
            f.write(
                f"{row['regulator']} & "
                f"{row['test_type']} & "
                f"{int(row['N'])} & "
                f"{float(row['r_max_nm']):.1f} & "
                f"{float(row['E0_MHz']):.6f} & "
                f"{float(row['E1_MHz']):.6f} & "
                f"{float(row['E2_MHz']):.6f} & "
                f"{int(row['N_negative'])} \\\\\n"
            )

        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

# ============================================================
# Main
# ============================================================
def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    cfg = FDMConfig()

    print("=== Regulator-form dependence FDM test ===")
    print(f"Target ground-state binding: |E0|/h = {TARGET_MHZ:.6f} MHz")
    print(f"FDM grid: N={cfg.N}, r_min={cfg.r_min*1e9:.6f} nm, r_max={cfg.r_max*1e9:.6f} nm")
    print(f"Shared physical model: omega/2pi={PARAMS.omega_ion/(2.0*math.pi*1e6):.6f} MHz, l={PARAMS.l}")
    print("This script does not modify unified_model_params.py.\n")

    results = run_regulator_dependence(cfg)

    summary_rows = collect_summary_rows(results)
    add_relative_energy_differences(summary_rows)
    level_rows = collect_level_rows(results)
    if RUN_VALIDATION_SWEEPS:
        validation_rows = validation_rows_for_calibrated_rc(results)
        write_csv(OUTDIR / "regulator_dependence_validation.csv", validation_rows)
        write_latex_validation(OUTDIR / "regulator_dependence_validation.tex", validation_rows)
        print("  regulator_dependence_validation.csv")
        print("  regulator_dependence_validation.tex")
    print_summary(summary_rows)
    print_level_table(level_rows)

    write_csv(OUTDIR / "regulator_dependence_summary.csv", summary_rows)
    write_csv(OUTDIR / "regulator_dependence_levels.csv", level_rows)
    write_latex_summary(OUTDIR / "regulator_dependence_summary.tex", summary_rows)
    write_latex_levels(OUTDIR / "regulator_dependence_levels.tex", level_rows)

    print("\nSaved tables:")
    print("  regulator_dependence_summary.csv")
    print("  regulator_dependence_levels.csv")
    print("  regulator_dependence_summary.tex")
    print("  regulator_dependence_levels.tex")

    plot_energy_ladder(results)
    plot_potentials(results)
    plot_ground_density(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
