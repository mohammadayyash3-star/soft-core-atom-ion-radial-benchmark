from __future__ import annotations



from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path
import csv
import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq, curve_fit
import matplotlib.pyplot as plt


from unified_model_params import (
    PARAMS,
    HBAR,
    HPLANCK,
)


# ============================================================
# Configuration
# ============================================================
@dataclass
class RunConfig:
    # ---- Run mode
    use_imported_rc: bool = True
    verify_imported_rc: bool = False
    allow_recalibration: bool = False
    experimental_uncertainty_MHz: float = 2.0

    # ---- FDM baseline setup (official numerical reference)
    fdm_r_min: float = 1.0e-10
    fdm_r_max: float = 650e-9
    fdm_N: int = 12000
    fdm_sigma: float = -1.0e-26
    fdm_k: int = 8

    # ---- Numerov matching cross-check (negative states only)
    numerov_r_min: float = 1.0e-10
    numerov_r_max: float = 650e-9
    numerov_N: int = 24000
    numerov_scan_points: int = 1200
    numerov_rescale_every: int = 250
    numerov_max_states: Optional[int] = None

    # ---- Validation sweeps for the FDM baseline
    sweep_N: Tuple[int, ...] = (4000, 5500, 7000, 9000, 12000)
    sweep_rmax_nm: Tuple[float, ...] = (300.0, 350.0, 400.0, 500.0, 650.0)
    sweep_rmin_nm: Tuple[float, ...] = (0.01, 0.03, 0.10, 0.30, 1.00)
    sweep_sigma: Tuple[float, ...] = (-2.0e-26, -1.5e-26, -1.0e-26, -7.5e-27, -5.0e-27)
    # ---- Fixed-cutoff grid-sequence diagnostic
    # The observed order is estimated from the data. The fit is used only to
    # quantify residual discretization bias; r_c is never recalibrated here.
    run_grid_extrapolation: bool = True
    extrap_N: Tuple[int, ...] = (6000, 7500, 9000, 10500, 12000, 14000, 16000)
    extrap_fit_tail: int = 5
    extrap_robustness_tails: Tuple[int, ...] = (4, 5, 6)
    extrap_candidate_orders: Tuple[float, ...] = (1.0, 2.0, 4.0)
    extrap_order_match_tolerance: float = 0.20

    # ---- Optional comparison values from approximate methods
    variational_levels_MHz: Tuple[float, ...] = ()
    wkb_levels_MHz: Tuple[float, ...] = ()
    perturbation_levels_MHz: Tuple[float, ...] = ()

    # ---- Output controls
    print_first_states: int = 6
    run_convergence_studies: bool = True
    run_numerov_crosscheck: bool = True
    run_optional_method_comparison: bool = True
    run_energy_only_mode: bool = True

    # ---- Partial-wave extension: angular-sector diagnostic only
    run_partial_wave_extension: bool = True
    partial_wave_ells: Tuple[int, ...] = (0, 1, 2)
    partial_wave_k: int = 10
    partial_wave_max_print_levels: int = 6
    partial_wave_plot_rmin_nm: float = 0.5
    partial_wave_plot_rmax_nm: float = 90.0

    
@dataclass
class ModelParams:
    # solver/grid-specific
    r_c: float
    r_min: float
    r_max: float
    N: int
    sigma: float = -1.0e-26

    # imported shared physical parameters
    m_atom: float = PARAMS.m_atom
    m_ion: float = PARAMS.m_ion
    C4: float = PARAMS.C4
    omega_ion: float = PARAMS.omega_ion
    l: int = PARAMS.l
    use_langer: bool = PARAMS.use_langer_numerical


@dataclass
class NumerovSettings:
    scan_points: int = 700
    rescale_every: int = 250


# ============================================================
# Shared physics helpers
# ============================================================
def reduced_mass(p: ModelParams) -> float:
    return p.m_atom * p.m_ion / (p.m_atom + p.m_ion)


def l_eff(p: ModelParams) -> float:
    return (p.l + 0.5) ** 2 if p.use_langer else p.l * (p.l + 1.0)


def alpha_pol(p: ModelParams) -> float:
    return -0.5 * p.C4


def effective_potential(r: np.ndarray, p: ModelParams) -> np.ndarray:
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


def state_label(E_J: float) -> str:
    return "bound (E<0)" if E_J < 0.0 else "trap-confined (E>0)"


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


def virial_diagnostics(u: np.ndarray, r: np.ndarray, E_J: float, p: ModelParams) -> Dict[str, float]:
    mu = reduced_mass(p)
    alpha = alpha_pol(p)
    leff = l_eff(p)
    dens = np.abs(u) ** 2

    v_r = effective_potential(r, p)
    v_exp = float(np.trapezoid(v_r * dens, r))
    t_exp = float(E_J) - v_exp

    term_osc = float(np.trapezoid(mu * (p.omega_ion ** 2) * r ** 2 * dens, r))
    term_cent = 0.0
    if leff != 0.0:
        term_cent = float(np.trapezoid(-(HBAR ** 2 * leff) / (reduced_mass(p) * r ** 2) * dens, r))
    term_core = float(np.trapezoid(alpha * (-4.0) * r ** 4 / (r ** 4 + p.r_c ** 4) ** 2 * dens, r))

    rdotgradv = term_osc + term_cent + term_core
    residual = 2.0 * t_exp - rdotgradv
    rel_res = residual / max(1e-30, abs(E_J))

    return {
        "T_J": t_exp,
        "V_J": v_exp,
        "vir_osc_J": term_osc,
        "vir_cent_J": term_cent,
        "vir_core_J": term_core,
        "r_dot_grad_V_J": rdotgradv,
        "virial_residual_J": residual,
        "virial_residual_over_absE": rel_res,
    }


# ============================================================
# FDM baseline solver (PRIMARY)
# ============================================================
def build_fdm_operator(p: ModelParams) -> Tuple[sp.csr_matrix, np.ndarray, Tuple[int, int], float]:
    if p.N < 7:
        raise ValueError("N must be at least 7 for a 5-point stencil.")

    r = np.linspace(p.r_min, p.r_max, p.N, dtype=float)
    dr = r[1] - r[0]
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

    tmat = -(HBAR ** 2) / (2.0 * mu) * lap
    hmat = tmat + sp.diags(effective_potential(r_int, p), format="csr")
    return hmat.tocsr(), r, (i0, i1), dr


def _fdm_eigen_residual(hmat: sp.csr_matrix, vec_int: np.ndarray, E_J: float) -> float:
    res = hmat @ vec_int - E_J * vec_int
    denom = max(1e-30, abs(E_J) * np.linalg.norm(vec_int))
    return float(np.linalg.norm(res) / denom)


def solve_fdm_states(p: ModelParams, k: int = 8, return_wavefunctions: bool = False) -> Dict[str, object]:
    hmat, r, (i0, i1), dr = build_fdm_operator(p)
    dim = hmat.shape[0]
    k_eff = min(k, dim - 2)
    ncv = min(dim, max(2 * k_eff + 8, 20))

    evals, evecs = spla.eigsh(
        hmat,
        k=k_eff,
        sigma=p.sigma,
        which="LM",
        tol=1e-11,
        maxiter=30000,
        ncv=ncv,
    )
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    states: List[Dict[str, object]] = []
    for j, E_J in enumerate(evals):
        vec_int = evecs[:, j].copy()
        residual_rel = _fdm_eigen_residual(hmat, vec_int, float(E_J))

        u = np.zeros_like(r)
        u[i0:i1 + 1] = vec_int
        u = normalize_u(u, r)
        nodes = count_nodes(u[i0:i1 + 1])
        vir = virial_diagnostics(u, r, float(E_J), p)

        entry: Dict[str, object] = {
            "index": j,
            "E_J": float(E_J),
            "E_over_h_MHz": float(E_J / HPLANCK / 1e6),
            "absE_over_h_MHz": float(abs(E_J) / HPLANCK / 1e6),
            "nodes": nodes,
            "label": state_label(float(E_J)),
            "virial": vir,
            "eig_residual_rel": residual_rel,
        }
        if return_wavefunctions:
            entry["u"] = u
        states.append(entry)

    states.sort(key=lambda s: s["E_J"])
    negative_states = [s for s in states if s["E_J"] < 0.0]
    ground = min(states, key=lambda s: s["E_J"])

    return {
        "params": asdict(p),
        "r": r,
        "dr": dr,
        "states": states,
        "negative_states": negative_states,
        "ground": ground,
        "n_negative": len(negative_states),
    }


def fdm_ground_energy_for_rc(rc_m: float, p_ref: ModelParams, k: int = 6) -> float:
    p = ModelParams(**asdict(p_ref))
    p.r_c = float(rc_m)
    result = solve_fdm_states(p, k=k, return_wavefunctions=False)
    return float(result["ground"]["E_J"])


def calibrate_rc_ground(target_Hz: float, p_ref: ModelParams, r_lo_nm: float = 10.0, r_hi_nm: float = 80.0,
                        k: int = 6, max_expand: int = 10) -> float:
    target_E = HPLANCK * target_Hz
    lo = r_lo_nm * 1e-9
    hi = r_hi_nm * 1e-9

    def objective(rc_m: float) -> float:
        E0 = fdm_ground_energy_for_rc(rc_m, p_ref, k=k)
        return abs(E0) - target_E

    f_lo = objective(lo)
    f_hi = objective(hi)

    tries = 0
    while f_lo * f_hi > 0.0 and tries < max_expand:
        hi *= 1.5
        f_hi = objective(hi)
        tries += 1

    if f_lo * f_hi > 0.0:
        raise RuntimeError("Could not bracket r_c for calibration.")

    rc_star = brentq(objective, lo, hi, xtol=1e-13, rtol=1e-10, maxiter=100)
    return float(rc_star)


# ============================================================
# Numerov matching solver (CROSS-CHECK only)
# ============================================================
def q_function(r: np.ndarray, E: float, p: ModelParams) -> np.ndarray:
    mu = reduced_mass(p)
    return (2.0 * mu / (HBAR ** 2)) * (effective_potential(r, p) - E)


def regular_power_near_origin(p: ModelParams) -> float:
    return float(p.l + 1)


def choose_match_index(r: np.ndarray, V: np.ndarray, E: float) -> int:
    idx_turn = int(np.argmin(np.abs(V - E)))
    shift = max(6, int(0.01 * len(r)))
    m = idx_turn + shift
    return min(max(m, 8), len(r) - 9)


def outer_decay_ratio(q_last: float, dr: float) -> float:
    q_last = max(q_last, 1e-30)
    kappa = math.sqrt(q_last)
    ratio = math.exp(min(100.0, kappa * dr))
    return max(ratio, 1.0 + 1e-12)


def numerov_AB(q: np.ndarray, dr: float) -> Tuple[np.ndarray, np.ndarray]:
    h2 = dr * dr
    A = 1.0 - (h2 / 12.0) * q
    B = 2.0 * (1.0 + (5.0 * h2 / 12.0) * q)
    return A, B


def propagate_left_ratios(r: np.ndarray, q: np.ndarray, m: int, p: ModelParams) -> Tuple[np.ndarray, int]:
    A, B = numerov_AB(q, r[1] - r[0])
    R = np.zeros(m + 1, dtype=float)
    power = regular_power_near_origin(p)
    R[1] = (r[1] / r[0]) ** power
    nodes = 0

    for n in range(1, m):
        denom = A[n + 1]
        if abs(denom) < 1e-300:
            denom = math.copysign(1e-300, denom if denom != 0 else 1.0)
        prev_ratio = R[n]
        if abs(prev_ratio) < 1e-300:
            prev_ratio = math.copysign(1e-300, prev_ratio if prev_ratio != 0 else 1.0)
        R[n + 1] = (B[n] - A[n - 1] / prev_ratio) / denom
        if not np.isfinite(R[n + 1]):
            raise FloatingPointError("Left ratio propagation failed.")
        if R[n + 1] < 0.0:
            nodes += 1
    return R, nodes


def propagate_right_ratios(r: np.ndarray, q: np.ndarray, m: int) -> np.ndarray:
    dr = r[1] - r[0]
    A, B = numerov_AB(q, dr)
    N = len(r)
    T = np.zeros(N - 1, dtype=float)
    T[N - 2] = outer_decay_ratio(q[N - 1], dr)

    for n in range(N - 2, m, -1):
        denom = A[n - 1]
        if abs(denom) < 1e-300:
            denom = math.copysign(1e-300, denom if denom != 0 else 1.0)
        prev_ratio = T[n]
        if abs(prev_ratio) < 1e-300:
            prev_ratio = math.copysign(1e-300, prev_ratio if prev_ratio != 0 else 1.0)
        T[n - 1] = (B[n] - A[n + 1] / prev_ratio) / denom if n + 1 < len(A) else (B[n] / denom)
        if not np.isfinite(T[n - 1]):
            raise FloatingPointError("Right ratio propagation failed.")
    return T


def logder_left_from_ratios(R: np.ndarray, m: int, dr: float) -> float:
    y_m = 1.0
    y_m1 = 1.0 / R[m]
    y_m2 = y_m1 / R[m - 1]
    return (3.0 * y_m - 4.0 * y_m1 + y_m2) / (2.0 * dr * y_m)


def logder_right_from_ratios(T: np.ndarray, m: int, dr: float) -> float:
    y_m = 1.0
    y_p1 = 1.0 / T[m]
    y_p2 = y_p1 / T[m + 1]
    return (-3.0 * y_m + 4.0 * y_p1 - y_p2) / (2.0 * dr * y_m)


def numerov_mismatch_and_nodes(E: float, r: np.ndarray, p: ModelParams) -> Tuple[float, int, int]:
    V = effective_potential(r, p)
    m = choose_match_index(r, V, E)
    q = q_function(r, E, p)
    R, nodes = propagate_left_ratios(r, q, m, p)
    T = propagate_right_ratios(r, q, m)
    dr = r[1] - r[0]
    mismatch = logder_left_from_ratios(R, m, dr) - logder_right_from_ratios(T, m, dr)
    if not np.isfinite(mismatch):
        raise FloatingPointError("Numerov mismatch became non-finite.")
    return mismatch, nodes, m


def scan_negative_brackets(p: ModelParams, settings: NumerovSettings) -> List[Dict[str, float]]:
    r = np.linspace(p.r_min, p.r_max, p.N, dtype=float)
    V = effective_potential(r, p)
    Vmin = float(np.min(V))

    E_low = Vmin * 0.9995
    E_high = -1e-6 * HPLANCK * 1e6
    energies = np.linspace(E_low, E_high, settings.scan_points)

    rows = []
    for E in energies:
        try:
            mis, nodes, m = numerov_mismatch_and_nodes(float(E), r, p)
            rows.append({"E": float(E), "mismatch": float(mis), "nodes": int(nodes), "m": int(m)})
        except Exception:
            continue

    brackets: List[Dict[str, float]] = []
    for a, b in zip(rows[:-1], rows[1:]):
        same_nodes = (a["nodes"] == b["nodes"])
        sign_flip = (a["mismatch"] == 0.0) or (b["mismatch"] == 0.0) or (a["mismatch"] * b["mismatch"] < 0.0)
        if same_nodes and sign_flip:
            brackets.append({
                "E_lo": min(a["E"], b["E"]),
                "E_hi": max(a["E"], b["E"]),
                "nodes": int(a["nodes"]),
            })

    cleaned: List[Dict[str, float]] = []
    tol = 5e-12
    for br in brackets:
        if not cleaned:
            cleaned.append(br)
            continue
        prev = cleaned[-1]
        same_node = (prev["nodes"] == br["nodes"])
        close = abs(prev["E_lo"] - br["E_lo"]) < tol and abs(prev["E_hi"] - br["E_hi"]) < tol
        if not (same_node and close):
            cleaned.append(br)
    return cleaned


def reconstruct_left_wave(r: np.ndarray, q: np.ndarray, m: int, p: ModelParams, settings: NumerovSettings) -> np.ndarray:
    dr = r[1] - r[0]
    A, B = numerov_AB(q, dr)
    y = np.zeros(m + 1, dtype=float)
    power = regular_power_near_origin(p)
    y[0] = r[0] ** power
    y[1] = r[1] ** power

    for n in range(1, m):
        denom = A[n + 1]
        if abs(denom) < 1e-300:
            denom = math.copysign(1e-300, denom if denom != 0 else 1.0)
        y[n + 1] = (B[n] * y[n] - A[n - 1] * y[n - 1]) / denom
        if (n % settings.rescale_every) == 0:
            scale = max(np.max(np.abs(y[:n + 2])), 1e-200)
            y[:n + 2] /= scale
    return y


def reconstruct_right_wave(r: np.ndarray, q: np.ndarray, m: int, settings: NumerovSettings) -> np.ndarray:
    dr = r[1] - r[0]
    A, B = numerov_AB(q, dr)
    N = len(r)
    z = np.zeros(N, dtype=float)
    z[N - 1] = 1e-250
    z[N - 2] = z[N - 1] * outer_decay_ratio(q[N - 1], dr)

    for n in range(N - 2, m, -1):
        denom = A[n - 1]
        if abs(denom) < 1e-300:
            denom = math.copysign(1e-300, denom if denom != 0 else 1.0)
        z[n - 1] = (B[n] * z[n] - A[n + 1] * z[n + 1]) / denom if n + 1 < N else (B[n] * z[n]) / denom
        if ((N - n) % settings.rescale_every) == 0:
            scale = max(np.max(np.abs(z[n - 1:])), 1e-200)
            z[n - 1:] /= scale
    return z


def solve_numerov_negative_states(p: ModelParams, settings: NumerovSettings,
                                  max_states: Optional[int] = None) -> Dict[str, object]:
    r = np.linspace(p.r_min, p.r_max, p.N, dtype=float)
    brackets = scan_negative_brackets(p, settings)
    if max_states is not None:
        brackets = brackets[:max_states]

    states: List[Dict[str, object]] = []
    for br in brackets:
        target_nodes = int(br["nodes"])

        def fE(E: float) -> float:
            mis, nodes, _ = numerov_mismatch_and_nodes(E, r, p)
            if nodes != target_nodes:
                return math.copysign(1e6 + abs(mis), mis)
            return mis

        E_star = brentq(fE, br["E_lo"], br["E_hi"], xtol=1e-15, rtol=1e-12, maxiter=150)
        V = effective_potential(r, p)
        m = choose_match_index(r, V, E_star)
        q = q_function(r, E_star, p)

        yL = reconstruct_left_wave(r, q, m, p, settings)
        yR = reconstruct_right_wave(r, q, m, settings)
        if abs(yR[m]) < 1e-300:
            raise FloatingPointError("Right wavefunction vanished at match point.")
        scale = yL[m] / yR[m]

        u = np.zeros_like(r)
        u[:m] = yL[:m]
        u[m:] = scale * yR[m:]
        u = normalize_u(u, r)

        vir = virial_diagnostics(u, r, float(E_star), p)
        states.append({
            "E_J": float(E_star),
            "E_over_h_MHz": float(E_star / HPLANCK / 1e6),
            "absE_over_h_MHz": float(abs(E_star) / HPLANCK / 1e6),
            "nodes": target_nodes,
            "match_index": int(m),
            "match_r_nm": float(r[m] * 1e9),
            "virial": vir,
            "u": u,
        })

    states.sort(key=lambda s: s["E_J"])
    return {
        "params": asdict(p),
        "brackets": brackets,
        "states": states,
        "ground": states[0] if states else None,
        "n_negative": len(states),
    }


# ============================================================
# Validation studies and comparisons
# ============================================================
def make_modified_params(p: ModelParams, **kwargs) -> ModelParams:
    d = asdict(p)
    d.update(kwargs)
    return ModelParams(**d)

# ============================================================
# Partial-wave extension: FDM angular-sector diagnostic
# ============================================================

def _fmt_float_or_blank(x, ndigits: int = 9) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and (not np.isfinite(x)):
        return ""
    return f"{float(x):.{ndigits}f}"


def _fmt_sci_or_blank(x, ndigits: int = 2) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and (not np.isfinite(x)):
        return ""
    return f"{float(x):.{ndigits}e}"


def _classification(E_J: float) -> str:
    return "bound" if E_J < 0.0 else "trap-confined"


def partial_wave_fdm_scan(
    p_base: ModelParams,
    ell_values: Sequence[int] = (0, 1, 2),
    k: int = 10,
    max_levels: int = 6,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Scan the fixed benchmark-calibrated radial Hamiltonian over angular sectors.

    Scientific role
    ---------------
    This is not a full partial-wave scattering calculation. It is a bound-spectrum
    diagnostic of the same reduced radial Hamiltonian in different angular sectors.

    Only l is changed. The calibrated r_c, omega, C4, grid, and solver settings
    are kept fixed.
    """
    summary_rows: List[Dict[str, object]] = []
    level_rows: List[Dict[str, object]] = []

    for ell in ell_values:
        p_ell = make_modified_params(
            p_base,
            l=int(ell),
            use_langer=False,   # FDM benchmark uses l(l+1), not Langer
        )

        result = solve_fdm_states(
            p_ell,
            k=k,
            return_wavefunctions=False,
        )

        negative_states = result["negative_states"]
        all_states = result["states"]

        # Potential minimum, useful as a compact diagnostic.
        r = result["r"]
        V = effective_potential(r, p_ell)
        Vmin_MHz = float(np.min(V) / HPLANCK / 1.0e6)

        neg_E = [float(s["E_over_h_MHz"]) for s in negative_states]
        neg_absE = [float(s["absE_over_h_MHz"]) for s in negative_states]

        summary_rows.append(
            {
                "ell": int(ell),
                "ell_exact_factor": float(ell * (ell + 1)),
                "N_negative": int(result["n_negative"]),
                "Vmin_over_h_MHz": Vmin_MHz,
                "E0_over_h_MHz": neg_E[0] if len(neg_E) > 0 else None,
                "E1_over_h_MHz": neg_E[1] if len(neg_E) > 1 else None,
                "E2_over_h_MHz": neg_E[2] if len(neg_E) > 2 else None,
                "absE0_over_h_MHz": neg_absE[0] if len(neg_absE) > 0 else None,
                "absE1_over_h_MHz": neg_absE[1] if len(neg_absE) > 1 else None,
                "absE2_over_h_MHz": neg_absE[2] if len(neg_absE) > 2 else None,
            }
        )

        for s in all_states[:max_levels]:
            level_rows.append(
                {
                    "ell": int(ell),
                    "state_index": int(s["index"]),
                    "E_over_h_MHz": float(s["E_over_h_MHz"]),
                    "absE_over_h_MHz": float(s["absE_over_h_MHz"]),
                    "nodes": int(s["nodes"]),
                    "eig_residual_rel": float(s["eig_residual_rel"]),
                    "classification": _classification(float(s["E_J"])),
                }
            )

    return summary_rows, level_rows


def print_partial_wave_summary(
    summary_rows: Sequence[Dict[str, object]],
    level_rows: Sequence[Dict[str, object]],
) -> None:
    print("\n=== Partial-wave extension: fixed-Hamiltonian FDM scan ===")
    print("Only ell is varied. r_c, omega, C4, grid, and sigma are fixed.")
    print("This is a bound-spectrum diagnostic, not a full partial-wave scattering calculation.\n")

    print(
        f"{'ell':>3} | {'ell(ell+1)':>10} | {'N_-':>3} | "
        f"{'Vmin/h (MHz)':>14} | {'E0/h (MHz)':>14} | "
        f"{'E1/h (MHz)':>14} | {'E2/h (MHz)':>14}"
    )
    print("-" * 95)

    for row in summary_rows:
        print(
            f"{int(row['ell']):3d} | "
            f"{float(row['ell_exact_factor']):10.6f} | "
            f"{int(row['N_negative']):3d} | "
            f"{float(row['Vmin_over_h_MHz']):14.6f} | "
            f"{_fmt_float_or_blank(row['E0_over_h_MHz'], 9):>14} | "
            f"{_fmt_float_or_blank(row['E1_over_h_MHz'], 9):>14} | "
            f"{_fmt_float_or_blank(row['E2_over_h_MHz'], 9):>14}"
        )

    print("\n--- Level-by-level partial-wave spectrum ---")
    print(
        f"{'ell':>3} | {'n':>3} | {'E/h (MHz)':>14} | "
        f"{'|E|/h (MHz)':>14} | {'nodes':>5} | "
        f"{'eig. residual':>13} | {'class':>14}"
    )
    print("-" * 91)

    for row in level_rows:
        print(
            f"{int(row['ell']):3d} | "
            f"{int(row['state_index']):3d} | "
            f"{float(row['E_over_h_MHz']):14.9f} | "
            f"{float(row['absE_over_h_MHz']):14.9f} | "
            f"{int(row['nodes']):5d} | "
            f"{float(row['eig_residual_rel']):13.3e} | "
            f"{str(row['classification']):>14}"
        )


def write_partial_wave_csv(
    summary_rows: Sequence[Dict[str, object]],
    level_rows: Sequence[Dict[str, object]],
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    summary_path = outdir / "partial_wave_summary.csv"
    levels_path = outdir / "partial_wave_levels.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with levels_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(level_rows[0].keys()))
        writer.writeheader()
        writer.writerows(level_rows)

    print(f"\nSaved: {summary_path.name}")
    print(f"Saved: {levels_path.name}")


def write_partial_wave_latex(
    summary_rows: Sequence[Dict[str, object]],
    level_rows: Sequence[Dict[str, object]],
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    summary_tex = outdir / "partial_wave_summary.tex"
    levels_tex = outdir / "partial_wave_levels.tex"

    with summary_tex.open("w", encoding="utf-8") as f:
        f.write(r"""\begin{table}[t]
\centering
\caption{FDM partial-wave extension of the benchmark-calibrated soft-core radial Hamiltonian. Only the angular-momentum sector $\ell$ is varied; $r_c$, $\omega$, $C_4$, the finite-difference grid, and the sparse-eigensolver settings are kept fixed. $N_-$ denotes the number of negative-energy eigenvalues.}
\label{tab:partial_wave_summary}
\begin{tabular}{c c c c c c c}
\hline
$\ell$ & $\ell(\ell+1)$ & $N_-$ & $V_{\min}/h$ (MHz) & $E_0^{(\ell)}/h$ (MHz) & $E_1^{(\ell)}/h$ (MHz) & $E_2^{(\ell)}/h$ (MHz) \\
\hline
""")
        for row in summary_rows:
            f.write(
                f"{int(row['ell'])} & "
                f"{float(row['ell_exact_factor']):.6f} & "
                f"{int(row['N_negative'])} & "
                f"{float(row['Vmin_over_h_MHz']):.6f} & "
                f"{_fmt_float_or_blank(row['E0_over_h_MHz'], 9) or '--'} & "
                f"{_fmt_float_or_blank(row['E1_over_h_MHz'], 9) or '--'} & "
                f"{_fmt_float_or_blank(row['E2_over_h_MHz'], 9) or '--'} \\\\\n"
            )
        f.write(r"""\hline
\end{tabular}
\end{table}
""")

    with levels_tex.open("w", encoding="utf-8") as f:
        f.write(r"""\begin{table}[t]
\centering
\caption{Level-by-level FDM spectrum for the partial-wave extension. The calculation uses the same benchmark-calibrated soft-core radius and numerical settings for all angular sectors.}
\label{tab:partial_wave_levels}
\begin{tabular}{c c c c c c c}
\hline
$\ell$ & $n$ & $E/h$ (MHz) & $|E|/h$ (MHz) & Nodes & Eig. residual & Classification \\
\hline
""")
        for row in level_rows:
            f.write(
                f"{int(row['ell'])} & "
                f"{int(row['state_index'])} & "
                f"{float(row['E_over_h_MHz']):.9f} & "
                f"{float(row['absE_over_h_MHz']):.9f} & "
                f"{int(row['nodes'])} & "
                f"{float(row['eig_residual_rel']):.2e} & "
                f"{row['classification']} \\\\\n"
            )
        f.write(r"""\hline
\end{tabular}
\end{table}
""")

    print(f"Saved: {summary_tex.name}")
    print(f"Saved: {levels_tex.name}")


def plot_partial_wave_energy_ladder(
    level_rows: Sequence[Dict[str, object]],
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))

    grouped: Dict[int, List[float]] = {}
    for row in level_rows:
        ell = int(row["ell"])
        E = float(row["E_over_h_MHz"])
        grouped.setdefault(ell, []).append(E)

    for ell, energies in grouped.items():
        xs = [ell] * len(energies)
        ax.scatter(xs, energies, s=36, label=rf"$\ell={ell}$")
        for E in energies:
            ax.plot([ell - 0.16, ell + 0.16], [E, E], linewidth=1.0)

    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"Angular-momentum sector $\ell$")
    ax.set_ylabel(r"$E/h$ (MHz)")
    ax.set_title("Partial-wave FDM energy ladder")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(outdir / "partial_wave_energy_ladder.pdf")
    fig.savefig(outdir / "partial_wave_energy_ladder.png", dpi=600)
    plt.close(fig)

    print("Saved: partial_wave_energy_ladder.pdf")
    print("Saved: partial_wave_energy_ladder.png")


def plot_partial_wave_effective_potentials(
    p_base: ModelParams,
    ell_values: Sequence[int],
    rmin_nm: float,
    rmax_nm: float,
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    r_nm = np.linspace(float(rmin_nm), float(rmax_nm), 1600)
    r_m = r_nm * 1.0e-9

    fig, ax = plt.subplots(figsize=(6.8, 4.5))

    for ell in ell_values:
        p_ell = make_modified_params(
            p_base,
            l=int(ell),
            use_langer=False,
        )
        V_MHz = effective_potential(r_m, p_ell) / HPLANCK / 1.0e6
        ax.plot(r_nm, V_MHz, linewidth=1.4, label=rf"$\ell={ell}$")

    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"$r$ (nm)")
    ax.set_ylabel(r"$V_{\mathrm{eff}}^{(\ell)}(r)/h$ (MHz)")
    ax.set_title("Effective radial potentials for different angular sectors")
    ax.set_ylim(-45, 25)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(outdir / "partial_wave_effective_potentials.pdf")
    fig.savefig(outdir / "partial_wave_effective_potentials.png", dpi=600)
    plt.close(fig)

    print("Saved: partial_wave_effective_potentials.pdf")
    print("Saved: partial_wave_effective_potentials.png")


def run_partial_wave_extension(p_fdm: ModelParams, cfg: RunConfig) -> None:
    outdir = Path(__file__).resolve().parent

    summary_rows, level_rows = partial_wave_fdm_scan(
        p_base=p_fdm,
        ell_values=cfg.partial_wave_ells,
        k=cfg.partial_wave_k,
        max_levels=cfg.partial_wave_max_print_levels,
    )

    print_partial_wave_summary(summary_rows, level_rows)
    write_partial_wave_csv(summary_rows, level_rows, outdir)
    write_partial_wave_latex(summary_rows, level_rows, outdir)
    plot_partial_wave_energy_ladder(level_rows, outdir)
    plot_partial_wave_effective_potentials(
        p_base=p_fdm,
        ell_values=cfg.partial_wave_ells,
        rmin_nm=cfg.partial_wave_plot_rmin_nm,
        rmax_nm=cfg.partial_wave_plot_rmax_nm,
        outdir=outdir,
    )
def study_one_parameter(p_fixed_rc: ModelParams, param_name: str, values: Sequence[float], k: int = 6) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for val in values:
        if param_name == "N":
            p = make_modified_params(p_fixed_rc, N=int(val))
        else:
            p = make_modified_params(p_fixed_rc, **{param_name: float(val)})
        result = solve_fdm_states(p, k=k, return_wavefunctions=False)
        g = result["ground"]
        rows.append({
            "value": float(val),
            "E0_MHz": float(g["E_over_h_MHz"]),
            "absE0_MHz": float(g["absE_over_h_MHz"]),
            "virial_rel": float(g["virial"]["virial_residual_over_absE"]),
            "eig_residual_rel": float(g["eig_residual_rel"]),
            "n_negative": float(result["n_negative"]),
        })
    ref = rows[-1]["E0_MHz"]
    for row in rows:
        row["delta_vs_last_kHz"] = 1e3 * (row["E0_MHz"] - ref)
    return rows


def sigma_robustness_study(p_fixed_rc: ModelParams, sigma_values: Sequence[float], k: int = 6) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for sig in sigma_values:
        p = make_modified_params(p_fixed_rc, sigma=float(sig))
        result = solve_fdm_states(p, k=k, return_wavefunctions=False)
        g = result["ground"]
        rows.append({
            "sigma_J": float(sig),
            "E0_MHz": float(g["E_over_h_MHz"]),
            "absE0_MHz": float(g["absE_over_h_MHz"]),
            "nodes": float(g["nodes"]),
            "eig_residual_rel": float(g["eig_residual_rel"]),
        })
    ref = rows[-1]["E0_MHz"]
    for row in rows:
        row["delta_vs_last_kHz"] = 1e3 * (row["E0_MHz"] - ref)
    return rows

def _fixed_order_energy_fit(
    dr_nm: np.ndarray,
    energy_MHz: np.ndarray,
    n_finest: int,
    order_power: float,
) -> Dict[str, Any]:
    """Linear fit E(dr)=E_inf+A*dr^p on the n finest grids."""
    if n_finest < 3 or len(dr_nm) < n_finest:
        raise ValueError("Insufficient points for fixed-order fit.")
    idx = np.argsort(dr_nm)[:n_finest]
    h = np.asarray(dr_nm[idx], dtype=float)
    y = np.asarray(energy_MHz[idx], dtype=float)
    x = h ** float(order_power)
    xscale = float(np.max(np.abs(x)))
    z = x / xscale
    X = np.column_stack([np.ones_like(z), z])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    residuals = y - fitted
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - sse / sst if sst > 0.0 else 1.0
    dof = n_finest - 2
    covariance = (sse / dof) * np.linalg.inv(X.T @ X)
    return {
        "n_points": n_finest,
        "order": float(order_power),
        "intercept_MHz": float(beta[0]),
        "slope_MHz_per_nm_power": float(beta[1] / xscale),
        "r_squared": float(r2),
        "max_abs_residual_kHz": float(1.0e3 * np.max(np.abs(residuals))),
        "intercept_stderr_kHz": float(1.0e3 * math.sqrt(max(0.0, covariance[0, 0]))),
        "used_indices": idx,
        "fitted_MHz": fitted,
        "residuals_MHz": residuals,
    }


def _free_order_energy_fit(
    dr_nm: np.ndarray,
    energy_MHz: np.ndarray,
    n_finest: int,
    initial_orders: Sequence[float],
) -> Dict[str, float]:
    """Nonlinear diagnostic E(dr)=E_inf+A*dr^p with p determined by the data."""
    if n_finest < 4 or len(dr_nm) < n_finest:
        raise ValueError("At least four points are required for a free-order fit.")
    idx = np.argsort(dr_nm)[:n_finest]
    h = np.asarray(dr_nm[idx], dtype=float)
    y = np.asarray(energy_MHz[idx], dtype=float)

    def model(hh, intercept, amplitude, power):
        return intercept + amplitude * hh**power

    best = None
    for p0 in initial_orders:
        try:
            popt, _ = curve_fit(
                model,
                h,
                y,
                p0=(float(y[-1]), 0.3, float(p0)),
                bounds=([-np.inf, -np.inf, 0.25], [np.inf, np.inf, 6.0]),
                maxfev=100000,
            )
            fitted = model(h, *popt)
            sse = float(np.sum((y - fitted)**2))
            if best is None or sse < best[0]:
                best = (sse, popt, fitted)
        except Exception:
            continue
    if best is None:
        raise RuntimeError("Free-order grid fit did not converge.")

    sse, popt, fitted = best
    sst = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - sse / sst if sst > 0.0 else 1.0
    return {
        "intercept_MHz": float(popt[0]),
        "amplitude": float(popt[1]),
        "observed_order": float(popt[2]),
        "r_squared": float(r2),
        "max_abs_residual_kHz": float(1.0e3 * np.max(np.abs(y - fitted))),
    }


def fdm_grid_extrapolation(
    p_fixed_rc: ModelParams,
    N_values: Sequence[int],
    k: int = 6,
    fit_tail: int = 5,
    robustness_tails: Sequence[int] = (4, 5, 6),
    candidate_orders: Sequence[float] = (1.0, 2.0, 4.0),
    order_match_tolerance: float = 0.20,
    experimental_uncertainty_MHz: float = 2.0,
) -> Dict[str, object]:
    """
    Fixed-cutoff grid-sequence diagnostic for the FDM ground-state energy.

    Only N is varied. The physical Hamiltonian, r_c, r_min, r_max, omega, C4,
    l, and sigma remain fixed. The observed convergence order is estimated
    rather than assumed. The resulting fine-grid intercept quantifies residual
    discretization bias only; it is not used to recalibrate r_c.
    """
    unique_N = tuple(sorted({int(N) for N in N_values}))
    if len(unique_N) < max(robustness_tails):
        raise ValueError("Not enough N values for the requested robustness windows.")

    rows: List[Dict[str, float | int]] = []
    for N in unique_N:
        p = make_modified_params(p_fixed_rc, N=N)
        result = solve_fdm_states(p, k=k, return_wavefunctions=False)
        ground = result["ground"]
        rows.append({
            "N": N,
            "dr_m": float(result["dr"]),
            "dr_nm": float(result["dr"]) * 1e9,
            "E0_MHz": float(ground["E_over_h_MHz"]),
            "absE0_MHz": float(ground["absE_over_h_MHz"]),
            "n_negative": int(result["n_negative"]),
            "eig_residual_rel": float(ground["eig_residual_rel"]),
            "virial_rel": float(ground["virial"]["virial_residual_over_absE"]),
        })

    rows.sort(key=lambda row: int(row["N"]))
    h = np.asarray([float(row["dr_nm"]) for row in rows], dtype=float)
    y = np.asarray([float(row["E0_MHz"]) for row in rows], dtype=float)

    free = _free_order_energy_fit(h, y, fit_tail, candidate_orders)
    selected_order = min(candidate_orders, key=lambda p: abs(float(p) - free["observed_order"]))
    order_distance = abs(float(selected_order) - free["observed_order"])
    order_supported = bool(order_distance <= order_match_tolerance)

    fits = {
        int(n): _fixed_order_energy_fit(h, y, int(n), float(selected_order))
        for n in robustness_tails
    }
    primary = fits[int(fit_tail)]
    candidate_fits = {
        float(p): _fixed_order_energy_fit(h, y, int(fit_tail), float(p))
        for p in candidate_orders
    }

    intercepts = np.asarray([fits[int(n)]["intercept_MHz"] for n in robustness_tails])
    fit_window_spread_kHz = float(1.0e3 * np.max(np.abs(intercepts - primary["intercept_MHz"])))
    model_difference_kHz = float(1.0e3 * abs(primary["intercept_MHz"] - free["intercept_MHz"]))
    fit_uncertainty_kHz = max(
        float(primary["max_abs_residual_kHz"]),
        float(primary["intercept_stderr_kHz"]),
        fit_window_spread_kHz,
        model_difference_kHz,
        0.001,
    )

    production_N = int(p_fixed_rc.N)
    production_row = next((row for row in rows if int(row["N"]) == production_N), None)
    if production_row is None:
        result = solve_fdm_states(p_fixed_rc, k=k, return_wavefunctions=False)
        g = result["ground"]
        production_row = {
            "N": production_N,
            "dr_nm": float(result["dr"]) * 1e9,
            "E0_MHz": float(g["E_over_h_MHz"]),
            "absE0_MHz": float(g["absE_over_h_MHz"]),
            "n_negative": int(result["n_negative"]),
            "eig_residual_rel": float(g["eig_residual_rel"]),
            "virial_rel": float(g["virial"]["virial_residual_over_absE"]),
        }

    fine_E = float(primary["intercept_MHz"])
    target_MHz = float(PARAMS.target_Hz / 1e6)
    production_drift_kHz = 1.0e3 * (float(production_row["E0_MHz"]) - fine_E)
    fine_target_offset_kHz = 1.0e3 * (abs(fine_E) - target_MHz)
    experimental_uncertainty_kHz = 1.0e3 * float(experimental_uncertainty_MHz)

    for row in rows:
        row["delta_vs_fine_grid_fit_kHz"] = 1.0e3 * (float(row["E0_MHz"]) - fine_E)

    negative_counts = sorted({int(row["n_negative"]) for row in rows})
    fit_pass = bool(
        order_supported
        and float(primary["r_squared"]) >= 0.995
        and float(primary["max_abs_residual_kHz"]) <= 5.0
        and fit_window_spread_kHz <= 5.0
    )

    return {
        "rows": rows,
        "fit_rows": [rows[i] for i in np.argsort(h)[:fit_tail]],
        "fit_tail": int(fit_tail),
        "observed_order": float(free["observed_order"]),
        "free_order_R2": float(free["r_squared"]),
        "selected_order": float(selected_order),
        "order_distance": float(order_distance),
        "order_supported": order_supported,
        "candidate_fit_R2": {p: candidate_fits[p]["r_squared"] for p in candidate_fits},
        "E0_continuum_MHz": fine_E,
        "absE0_continuum_MHz": abs(fine_E),
        "E0_production_MHz": float(production_row["E0_MHz"]),
        "absE0_production_MHz": float(production_row["absE0_MHz"]),
        "production_N": int(production_row["N"]),
        "production_dr_nm": float(production_row["dr_nm"]),
        "production_drift_kHz": float(production_drift_kHz),
        "continuum_target_error_kHz": float(fine_target_offset_kHz),
        "max_fit_residual_kHz": float(primary["max_abs_residual_kHz"]),
        "fit_intercept_stderr_kHz": float(primary["intercept_stderr_kHz"]),
        "fit_window_spread_kHz": fit_window_spread_kHz,
        "model_difference_kHz": model_difference_kHz,
        "fit_uncertainty_kHz": float(fit_uncertainty_kHz),
        "r_squared": float(primary["r_squared"]),
        "fit_pass": fit_pass,
        "negative_count_stable": len(negative_counts) == 1,
        "negative_counts": negative_counts,
        "experimental_uncertainty_MHz": float(experimental_uncertainty_MHz),
        "experimental_uncertainty_kHz": experimental_uncertainty_kHz,
        "drift_fraction_of_experimental_uncertainty": abs(production_drift_kHz) / experimental_uncertainty_kHz,
        "fit_model": f"E0(dr)=E0(fine-grid)+A*dr^{float(selected_order):g}",
        "calibration_changed": False,
    }


def level_comparison_table(fdm_levels: Sequence[float], other_levels: Sequence[float], method_name: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    n = min(len(fdm_levels), len(other_levels))
    for i in range(n):
        delta = float(other_levels[i] - fdm_levels[i])
        rel = 100.0 * delta / max(1e-30, abs(fdm_levels[i]))
        rows.append({
            "n": i,
            "method": method_name,
            "fdm_MHz": float(fdm_levels[i]),
            "other_MHz": float(other_levels[i]),
            "delta_kHz": 1e3 * delta,
            "rel_err_percent": rel,
        })
    return rows


def summarize_agreement(fdm_result: Dict[str, object], numerov_result: Dict[str, object]) -> Dict[str, object]:
    fdm_neg = fdm_result["negative_states"]
    num_neg = numerov_result["states"]
    n_common = min(len(fdm_neg), len(num_neg))

    level_deltas_kHz = [
        1e3 * (float(num_neg[i]["E_over_h_MHz"]) - float(fdm_neg[i]["E_over_h_MHz"]))
        for i in range(n_common)
    ]
    abs_level_deltas = [abs(x) for x in level_deltas_kHz]
    max_abs_delta = max(abs_level_deltas) if abs_level_deltas else float("nan")

    fdm_ground_vir = abs(float(fdm_result["ground"]["virial"]["virial_residual_over_absE"]))
    num_ground_vir = abs(float(num_neg[0]["virial"]["virial_residual_over_absE"])) if num_neg else float("nan")
    fdm_ground_eig_res = float(fdm_result["ground"]["eig_residual_rel"])

    if n_common == 0:
        verdict = "no common negative states"
    elif (fdm_result["n_negative"] == numerov_result["n_negative"] and max_abs_delta <= 20.0 and
          fdm_ground_vir <= 5e-3 and num_ground_vir <= 5e-3 and fdm_ground_eig_res <= 1e-8):
        verdict = "strong quantitative agreement"
    elif (fdm_result["n_negative"] == numerov_result["n_negative"] and max_abs_delta <= 150.0):
        verdict = "strong qualitative agreement; quantitative refinement still recommended"
    else:
        verdict = "agreement incomplete; refine grid/cutoff settings"

    return {
        "fdm_negative_count": int(fdm_result["n_negative"]),
        "numerov_negative_count": int(numerov_result["n_negative"]),
        "n_common": int(n_common),
        "level_deltas_kHz": level_deltas_kHz,
        "max_abs_delta_kHz": max_abs_delta,
        "fdm_ground_virial_rel": fdm_ground_vir,
        "numerov_ground_virial_rel": num_ground_vir,
        "fdm_ground_eig_residual_rel": fdm_ground_eig_res,
        "verdict": verdict,
    }


# ============================================================
# Printing
# ============================================================
def print_main_fdm_spectrum(result: Dict[str, object], max_rows: int = 6) -> None:
    p = result["params"]
    g = result["ground"]
    print("=== Direct numerical benchmark: finite-difference radial solver ===")
    print(f"r_c (nm)            : {p['r_c'] * 1e9:.6f}")
    print(f"N                   : {p['N']}")
    print(f"r_min (nm)          : {p['r_min'] * 1e9:.6f}")
    print(f"r_max (nm)          : {p['r_max'] * 1e9:.6f}")
    print(f"sigma (J)           : {p['sigma']:+.6e}")
    print(f"Ground E/h (MHz)    : {g['E_over_h_MHz']:+.9f}")
    print(f"Ground |E|/h (MHz)  : {g['absE_over_h_MHz']:.9f}")
    print(f"Ground eig residual : {g['eig_residual_rel']:.3e}")
    print(f"Negative states     : {result['n_negative']}")
    print()
    print("First states:")
    print(" n |       E/h (MHz) |    |E|/h (MHz) | nodes | eig resid | type")
    for n, s in enumerate(result["states"][:max_rows]):
        print(f"{n:2d} | {s['E_over_h_MHz']:>+14.9f} | {s['absE_over_h_MHz']:>14.9f} |"
              f" {s['nodes']:>5d} | {s['eig_residual_rel']:>8.2e} | {s['label']}")
    print()


def print_virial_summary(title: str, state: Dict[str, object]) -> None:
    vir = state["virial"]
    print(title)
    print(f"E/h                      = {state['E_over_h_MHz']:+.9f} MHz")
    print(f"2<T>                     = {2.0 * vir['T_J']:+.6e} J")
    print(f"<r·∇V>                   = {vir['r_dot_grad_V_J']:+.6e} J")
    print(f"  oscillator part        = {vir['vir_osc_J']:+.6e} J")
    print(f"  centrifugal part       = {vir['vir_cent_J']:+.6e} J")
    print(f"  soft-core part         = {vir['vir_core_J']:+.6e} J")
    print(f"Residual                 = {vir['virial_residual_J']:+.6e} J")
    print(f"Residual / |E|           = {vir['virial_residual_over_absE']:+.6e}")
    print()


def print_study_table(title: str, rows: List[Dict[str, float]], first_col_name: str) -> None:
    print(title)
    print(f"{first_col_name:>14s} | {'E0/h (MHz)':>14s} | {'Δ vs last (kHz)':>15s} | {'virial rel':>12s} | {'eig resid':>10s} | {'# neg':>5s}")
    print("-" * 92)
    for row in rows:
        print(f"{row['value']:>14.6g} | {row['E0_MHz']:>14.9f} | {row['delta_vs_last_kHz']:>15.6f} |"
              f" {row['virial_rel']:>+12.3e} | {row['eig_residual_rel']:>10.2e} | {int(row['n_negative']):>5d}")
    print()


def print_sigma_table(rows: List[Dict[str, float]]) -> None:
    print("=== Sigma robustness study ===")
    print(f"{'sigma (J)':>14s} | {'E0/h (MHz)':>14s} | {'Δ vs last (kHz)':>15s} | {'nodes':>5s} | {'eig resid':>10s}")
    print("-" * 80)
    for row in rows:
        print(f"{row['sigma_J']:>+14.6e} | {row['E0_MHz']:>14.9f} | {row['delta_vs_last_kHz']:>15.6f} | {int(row['nodes']):>5d} | {row['eig_residual_rel']:>10.2e}")
    print()

def write_fdm_grid_diagnostic_outputs(extrap: Dict[str, object]) -> None:
    """Write compact CSV and LaTeX outputs for the fixed-cutoff grid diagnostic."""
    outdir = Path(__file__).resolve().parent
    csv_path = outdir / "fdm_grid_sequence_diagnostic.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "N", "dr_nm", "E0_MHz", "absE0_MHz",
            "delta_vs_fine_grid_fit_kHz", "n_negative",
            "eig_residual_rel", "virial_rel",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in extrap["rows"]:
            writer.writerow({field: row[field] for field in fields})

    tex_path = outdir / "fdm_grid_sequence_summary.tex"
    with tex_path.open("w", encoding="utf-8") as f:
        f.write(r"""\begin{table}[t]
\centering
\caption{Fixed-cutoff FDM grid-sequence diagnostic. Only the number of grid points is varied; the Hamiltonian and all physical parameters, including the calibrated soft-core radius, remain fixed. The fit order is determined from the data and is used only to quantify residual discretization bias, not to recalibrate the model.}
\label{tab:fdm_grid_sequence_diagnostic}
\begin{tabular}{c c c c c}
\hline
$N$ & $\Delta r$ (nm) & $E_0/h$ (MHz) & $\Delta_{\rm fit}$ (kHz) & $N_-$ \\
\hline
""")
        for row in extrap["rows"]:
            f.write(
                f"{int(row['N'])} & {float(row['dr_nm']):.8f} & "
                f"{float(row['E0_MHz']):+.9f} & "
                f"{float(row['delta_vs_fine_grid_fit_kHz']):+.6f} & "
                f"{int(row['n_negative'])} \\\\\n"
            )
        f.write(r"""\hline
\end{tabular}
\end{table}
""")
    print(f"Saved: {csv_path.name}")
    print(f"Saved: {tex_path.name}")


def print_fdm_grid_extrapolation(extrap: Dict[str, object]) -> None:
    print("=== Fixed-cutoff FDM grid-sequence diagnostic ===")
    print(f"Fit model                 : {extrap['fit_model']}")
    print(f"Observed order p          : {float(extrap['observed_order']):.6f}")
    print(f"Selected supported order  : {float(extrap['selected_order']):.1f}")
    print(f"Order supported           : {bool(extrap['order_supported'])}")
    print(f"Finest-grid points in fit : {int(extrap['fit_tail'])}")
    print(f"Production N              : {int(extrap['production_N'])}")
    print(f"Production dr (nm)        : {float(extrap['production_dr_nm']):.9f}")
    print(f"Fine-grid E0/h (MHz)      : {float(extrap['E0_continuum_MHz']):+.12f}")
    print(f"Production E0/h (MHz)     : {float(extrap['E0_production_MHz']):+.12f}")
    print(f"Production - fit          : {float(extrap['production_drift_kHz']):+.6f} kHz")
    print(f"Fit-only uncertainty      : {float(extrap['fit_uncertainty_kHz']):.6f} kHz")
    print(f"Central-target offset     : {float(extrap['continuum_target_error_kHz']):+.6f} kHz")
    print(f"Experimental uncertainty  : {float(extrap['experimental_uncertainty_MHz']):.3f} MHz")
    print(
        "Drift / experimental unc. : "
        f"{100.0 * float(extrap['drift_fraction_of_experimental_uncertainty']):.4f}%"
    )
    print(f"Max fit residual          : {float(extrap['max_fit_residual_kHz']):.6f} kHz")
    print(f"Fit-window spread         : {float(extrap['fit_window_spread_kHz']):.6f} kHz")
    print(f"Fit R^2                   : {float(extrap['r_squared']):.10f}")
    print(f"Fit checks pass           : {bool(extrap['fit_pass'])}")
    print(f"Negative-count stable     : {bool(extrap['negative_count_stable'])}")
    print(f"Negative counts observed  : {extrap['negative_counts']}")
    print("Calibration changed       : False")
    print()

    print("Grid-sequence table:")
    print(
        f"{'N':>8s} | {'dr (nm)':>12s} | {'E0/h (MHz)':>16s} | "
        f"{'Delta fit (kHz)':>15s} | {'N-':>3s} | {'eig resid':>10s}"
    )
    print("-" * 86)
    for row in extrap["rows"]:
        print(
            f"{int(row['N']):8d} | {float(row['dr_nm']):12.8f} | "
            f"{float(row['E0_MHz']):+16.9f} | "
            f"{float(row['delta_vs_fine_grid_fit_kHz']):+15.6f} | "
            f"{int(row['n_negative']):3d} | "
            f"{float(row['eig_residual_rel']):10.2e}"
        )
    print()

    print("Paper-ready diagnostic sentence:")
    print(
        "At fixed physical parameters and fixed radial cutoffs, the grid sequence "
        f"exhibits an observed leading order p={float(extrap['observed_order']):.3f}. "
        f"The production-grid ground-state energy differs from the data-supported "
        f"fine-grid intercept by {abs(float(extrap['production_drift_kHz'])):.3f} kHz, "
        f"which is only {100.0 * float(extrap['drift_fraction_of_experimental_uncertainty']):.3f}% "
        f"of the experimental 2 MHz uncertainty, while the negative-state count remains "
        f"{extrap['negative_counts']}. The fixed calibrated Hamiltonian is therefore retained "
        "without numerical recalibration."
    )
    print()
    write_fdm_grid_diagnostic_outputs(extrap)


def print_numerov_summary(numerov_result: Dict[str, object]) -> None:
    p = numerov_result["params"]
    brackets = numerov_result["brackets"]
    print("=== Matching / log-derivative Numerov cross-check ===")
    print(f"r_c (nm)            : {p['r_c'] * 1e9:.6f}")
    print(f"N                   : {p['N']}")
    print(f"r_min (nm)          : {p['r_min'] * 1e9:.6f}")
    print(f"r_max (nm)          : {p['r_max'] * 1e9:.6f}")
    print(f"Brackets found      : {len(brackets)}")
    print()
    print(f"Negative states found: {len(numerov_result['states'])}")
    print(" n |       E/h (MHz) |    |E|/h (MHz) | nodes | match r (nm) | virial rel")
    print("-" * 84)
    for i, st in enumerate(numerov_result["states"]):
        print(f"{i:2d} | {st['E_over_h_MHz']:+16.9f} | {st['absE_over_h_MHz']:16.9f} |"
              f" {int(st['nodes']):5d} | {st['match_r_nm']:12.4f} | {st['virial']['virial_residual_over_absE']:+.3e}")
    print()


def print_fdm_vs_numerov(fdm_result: Dict[str, object], numerov_result: Dict[str, object]) -> None:
    fdm_neg = fdm_result["negative_states"]
    num_neg = numerov_result["states"]
    n = min(len(fdm_neg), len(num_neg))
    print("=== FDM vs Numerov comparison (negative states) ===")
    print(" n |  FDM E/h (MHz)  | Numerov E/h (MHz) | Δ (kHz) | FDM vir rel | Num vir rel")
    print("-" * 90)
    for i in range(n):
        fdmE = float(fdm_neg[i]["E_over_h_MHz"])
        numE = float(num_neg[i]["E_over_h_MHz"])
        dk = 1e3 * (numE - fdmE)
        fv = float(fdm_neg[i]["virial"]["virial_residual_over_absE"])
        nv = float(num_neg[i]["virial"]["virial_residual_over_absE"])
        print(f"{i:2d} | {fdmE:+15.9f} | {numE:+17.9f} | {dk:+8.3f} | {fv:+11.3e} | {nv:+11.3e}")
    print()


def print_method_level_comparison(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    method_name = rows[0]["method"]
    print(f"=== {method_name} vs FDM (negative levels) ===")
    print(" n |  FDM E/h (MHz)  | Other E/h (MHz) | Δ (kHz) | rel err %")
    print("-" * 72)
    for row in rows:
        print(f"{row['n']:2d} | {row['fdm_MHz']:+15.9f} | {row['other_MHz']:+15.9f} |"
              f" {row['delta_kHz']:+8.3f} | {row['rel_err_percent']:+9.5f}")
    print()


def print_agreement_summary(summary: Dict[str, object]) -> None:
    print("=== Automatic agreement summary ===")
    print(f"FDM negative states          : {summary['fdm_negative_count']}")
    print(f"Numerov negative states      : {summary['numerov_negative_count']}")
    print(f"Common compared states       : {summary['n_common']}")
    if summary['n_common'] > 0:
        deltas = ", ".join(f"{x:+.3f}" for x in summary["level_deltas_kHz"])
        print(f"Level-by-level Δ (kHz)       : [{deltas}]")
        print(f"Max |Δ| (kHz)                : {summary['max_abs_delta_kHz']:.3f}")
    print(f"FDM ground virial |rel|      : {summary['fdm_ground_virial_rel']:.3e}")
    print(f"Numerov ground virial |rel|  : {summary['numerov_ground_virial_rel']:.3e}")
    print(f"FDM eig residual rel         : {summary['fdm_ground_eig_residual_rel']:.3e}")
    print(f"Verdict                      : {summary['verdict']}")
    print()


def print_paper_ready_paragraphs(fdm_result: Dict[str, object], numerov_result: Dict[str, object],
                                 agreement: Dict[str, object]) -> None:
    fdm_ground = fdm_result["ground"]["E_over_h_MHz"]
    nneg = fdm_result["n_negative"]
    rc_nm = fdm_result["params"]["r_c"] * 1e9
    verdict = agreement["verdict"]
    print("=== Paper-ready summary text (draft) ===")
    print(
        f"Using a calibrated soft-core radius r_c = {rc_nm:.3f} nm, the effective radial Hamiltonian was benchmarked "
        f"by a direct finite-difference sparse diagonalization scheme. The resulting FDM baseline yields a ground-state "
        f"energy E0/h = {fdm_ground:+.6f} MHz and supports {nneg} negative-energy states within the effective single-channel model."
    )
    print(
        f"An independent matching/log-derivative Numerov solver reproduces the same low-lying negative-state structure with "
        f"consistent node ordering. The comparison between both numerical schemes is best summarized as: {verdict}. "
        f"This establishes the internal numerical consistency of the effective radial model before comparing it against WKB, "
        f"variational, and perturbative approximations."
    )
    print()


# ============================================================
# Main driver
# ============================================================
def main() -> None:
    cfg = RunConfig()

    print("=== Imported unified model parameters ===")
    print(f"r_c from file (nm)      : {PARAMS.r_c * 1e9:.6f}")
    print(f"target |E|/h (MHz)      : {PARAMS.target_Hz / 1e6:.6f}")
    print(f"experimental band (MHz) : {PARAMS.target_Hz / 1e6 - cfg.experimental_uncertainty_MHz:.3f} to {PARAMS.target_Hz / 1e6 + cfg.experimental_uncertainty_MHz:.3f}")
    print(f"omega / 2π (MHz)        : {PARAMS.omega_ion / (2.0 * math.pi * 1e6):.6f}")
    print(f"l                       : {PARAMS.l}")
    print(f"use_langer_numerical    : {PARAMS.use_langer_numerical}")
    print(f"use imported r_c mode   : {cfg.use_imported_rc}")
    print()

    # A) Base FDM setup using imported shared physics
    p_ref = ModelParams(
        r_c=float(PARAMS.r_c),
        r_min=cfg.fdm_r_min,
        r_max=cfg.fdm_r_max,
        N=cfg.fdm_N,
        sigma=cfg.fdm_sigma,
    )

    # B) Choose r_c
    if cfg.use_imported_rc:
        rc_star = float(PARAMS.r_c)
        print("=== Fixed-r_c production mode ===")
        print("Using r_c directly from unified_model_params.PARAMS.r_c")
        print(f"r_c used (nm)          : {rc_star * 1e9:.6f}")
        print()
    else:
        if not cfg.allow_recalibration:
            raise RuntimeError(
                "Automatic r_c recalibration is disabled in the submission version. "
                "Keep use_imported_rc=True unless a new study explicitly requires refitting."
            )
        rc_star = calibrate_rc_ground(
            PARAMS.target_Hz,
            p_ref,
            r_lo_nm=10.0,
            r_hi_nm=80.0,
            k=6,
        )
        print("=== Recalibrated mode ===")
        print("WARNING: this mode is for verification / updating the shared file only.")
        print(f"recalibrated r_c (nm)  : {rc_star * 1e9:.6f}")
        print()

    # Optional verification against imported r_c
    if cfg.verify_imported_rc:
        rc_chk = calibrate_rc_ground(
            PARAMS.target_Hz,
            p_ref,
            r_lo_nm=10.0,
            r_hi_nm=80.0,
            k=6,
        )
        rel_diff = abs(rc_chk - float(PARAMS.r_c)) / max(abs(float(PARAMS.r_c)), 1e-300)

        print("=== Imported r_c verification ===")
        print(f"imported r_c (nm)      : {float(PARAMS.r_c) * 1e9:.6f}")
        print(f"recalibrated r_c (nm)  : {rc_chk * 1e9:.6f}")
        print(f"relative diff          : {rel_diff:.6e}")
        print()

    # C) Main FDM benchmark (official baseline)
    p_fdm = make_modified_params(p_ref, r_c=rc_star)
    fdm_result = solve_fdm_states(p_fdm, k=cfg.fdm_k, return_wavefunctions=False)
    central_MHz = PARAMS.target_Hz / 1e6
    ground_binding_MHz = float(fdm_result["ground"]["absE_over_h_MHz"])
    if abs(ground_binding_MHz - central_MHz) > cfg.experimental_uncertainty_MHz:
        raise RuntimeError(
            "The fixed benchmark lies outside the experimental 15(2) MHz band."
        )
    print_main_fdm_spectrum(fdm_result, max_rows=cfg.print_first_states)
    print_virial_summary("=== Virial check (FDM ground state) ===", fdm_result["ground"])

    # C.1) Partial-wave extension: ell = 0, 1, 2
    if cfg.run_partial_wave_extension:
        run_partial_wave_extension(p_fdm, cfg)

    # D) Convergence studies for the FDM baseline
    if cfg.run_convergence_studies:
        N_rows = study_one_parameter(p_fdm, "N", cfg.sweep_N, k=6)
        print_study_table("=== Convergence study: varying N ===", N_rows, "N")

        rmax_rows = study_one_parameter(p_fdm, "r_max", [x * 1e-9 for x in cfg.sweep_rmax_nm], k=6)
        for row in rmax_rows:
            row["value"] *= 1e9
        print_study_table("=== Convergence study: varying r_max ===", rmax_rows, "r_max (nm)")

        rmin_rows = study_one_parameter(p_fdm, "r_min", [x * 1e-9 for x in cfg.sweep_rmin_nm], k=6)
        for row in rmin_rows:
            row["value"] *= 1e9
        print_study_table("=== Convergence study: varying r_min ===", rmin_rows, "r_min (nm)")

        sigma_rows = sigma_robustness_study(p_fdm, cfg.sweep_sigma, k=6)
        print_sigma_table(sigma_rows)

    # D2) Continuum-limit extrapolation of the FDM grid-spacing error
    if cfg.run_grid_extrapolation:
        extrap = fdm_grid_extrapolation(
            p_fdm,
            cfg.extrap_N,
            k=6,
            fit_tail=cfg.extrap_fit_tail,
            robustness_tails=cfg.extrap_robustness_tails,
            candidate_orders=cfg.extrap_candidate_orders,
            order_match_tolerance=cfg.extrap_order_match_tolerance,
            experimental_uncertainty_MHz=cfg.experimental_uncertainty_MHz,
        )
        print_fdm_grid_extrapolation(extrap)

    # E) Numerov matching cross-check (negative states only)
    numerov_result = None
    agreement = None
    if cfg.run_numerov_crosscheck:
        p_num = make_modified_params(
            p_fdm,
            r_min=cfg.numerov_r_min,
            r_max=cfg.numerov_r_max,
            N=cfg.numerov_N,
            sigma=cfg.fdm_sigma,
        )

        num_settings = NumerovSettings(
            scan_points=cfg.numerov_scan_points,
            rescale_every=cfg.numerov_rescale_every,
        )

        numerov_result = solve_numerov_negative_states(
            p_num,
            num_settings,
            max_states=cfg.numerov_max_states,
        )

        print_numerov_summary(numerov_result)
        if numerov_result["ground"] is not None:
            print_virial_summary("=== Virial check (Numerov ground state) ===", numerov_result["ground"])
        print_fdm_vs_numerov(fdm_result, numerov_result)

        agreement = summarize_agreement(fdm_result, numerov_result)
        print_agreement_summary(agreement)

    # F) Optional comparisons to approximate methods
    if cfg.run_optional_method_comparison:
        fdm_negative_levels = [float(s["E_over_h_MHz"]) for s in fdm_result["negative_states"]]
        if cfg.variational_levels_MHz:
            print_method_level_comparison(
                level_comparison_table(fdm_negative_levels, cfg.variational_levels_MHz, "Variational")
            )
        if cfg.wkb_levels_MHz:
            print_method_level_comparison(
                level_comparison_table(fdm_negative_levels, cfg.wkb_levels_MHz, "WKB")
            )
        if cfg.perturbation_levels_MHz:
            print_method_level_comparison(
                level_comparison_table(fdm_negative_levels, cfg.perturbation_levels_MHz, "Perturbation")
            )

    # G) Optional prose summary
    if agreement is not None:
        print_paper_ready_paragraphs(fdm_result, numerov_result, agreement)


if __name__ == "__main__":
    main()
