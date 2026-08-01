from __future__ import annotations


from pathlib import Path
import csv
import math

import numpy as np
from numba import njit
from scipy.optimize import brentq

from unified_model_params import PARAMS, HBAR, HPLANCK, reduced_mass, alpha_pol
from benchmark_reference import CONTINUUM_FDM_LEVELS_MHZ

OUTDIR = Path(__file__).resolve().parent
N_GRID = 24000
R_MAX_M = 650.0e-9
SEARCH_HALF_WIDTH_MHZ = 0.02
SCAN_POINTS = 401


@njit(cache=True)
def _mismatch_and_nodes(E_J, r, potential_J, mu, hbar):
    n_grid = r.size
    dr = r[1] - r[0]
    h2 = dr * dr
    q = (2.0 * mu / (hbar * hbar)) * (potential_J - E_J)

    # Match slightly outside the outer turning point.
    turning_index = 1
    best = 1.0e300
    for i in range(1, n_grid - 1):
        value = abs(potential_J[i] - E_J)
        if value < best:
            best = value
            turning_index = i
    match = turning_index + max(20, n_grid // 200)
    match = min(max(match, 3), n_grid - 4)

    # Left ratios R_n = y_n/y_{n-1}; regular l=0 seed y(0)=0, y(dr)=dr.
    ratios_left = np.empty(match + 1)
    ratios_left[0] = 0.0
    ratios_left[1] = 1.0e300

    a0 = 1.0 - h2 * q[0] / 12.0
    a2 = 1.0 - h2 * q[2] / 12.0
    b1 = 2.0 * (1.0 + 5.0 * h2 * q[1] / 12.0)
    y0 = 0.0
    y1 = dr
    y2 = (b1 * y1 - a0 * y0) / a2
    ratios_left[2] = y2 / y1
    nodes = 1 if y1 * y2 < 0.0 else 0

    for n in range(2, match):
        a_prev = 1.0 - h2 * q[n - 1] / 12.0
        a_next = 1.0 - h2 * q[n + 1] / 12.0
        b_cur = 2.0 * (1.0 + 5.0 * h2 * q[n] / 12.0)
        ratio = ratios_left[n]
        if abs(ratio) < 1.0e-300:
            ratio = 1.0e-300 if ratio >= 0.0 else -1.0e-300
        ratios_left[n + 1] = (b_cur - a_prev / ratio) / a_next
        if ratios_left[n + 1] < 0.0:
            nodes += 1

    # Right ratios T_n = y_n/y_{n+1}; decaying asymptotic seed.
    ratios_right = np.empty(n_grid - 1)
    kappa = math.sqrt(max(q[n_grid - 1], 1.0e-300))
    ratios_right[n_grid - 2] = math.exp(min(100.0, kappa * dr))

    for n in range(n_grid - 2, match, -1):
        a_prev = 1.0 - h2 * q[n - 1] / 12.0
        a_next = 1.0 - h2 * q[n + 1] / 12.0
        b_cur = 2.0 * (1.0 + 5.0 * h2 * q[n] / 12.0)
        ratio = ratios_right[n]
        if abs(ratio) < 1.0e-300:
            ratio = 1.0e-300 if ratio >= 0.0 else -1.0e-300
        ratios_right[n - 1] = (b_cur - a_next / ratio) / a_prev

    ym1 = 1.0 / ratios_left[match]
    ym2 = ym1 / ratios_left[match - 1]
    log_left = (3.0 - 4.0 * ym1 + ym2) / (2.0 * dr)

    yp1 = 1.0 / ratios_right[match]
    yp2 = yp1 / ratios_right[match + 1]
    log_right = (-3.0 + 4.0 * yp1 - yp2) / (2.0 * dr)
    return log_left - log_right, nodes, match


def solve_state(reference_MHz: float, expected_nodes: int, r: np.ndarray, potential_J: np.ndarray) -> dict[str, float | int]:
    mu = reduced_mass(PARAMS)
    scan_MHz = np.linspace(
        reference_MHz - SEARCH_HALF_WIDTH_MHZ,
        reference_MHz + SEARCH_HALF_WIDTH_MHZ,
        SCAN_POINTS,
    )
    scan_J = scan_MHz * HPLANCK * 1.0e6
    values = [
        _mismatch_and_nodes(E, r, potential_J, mu, HBAR) for E in scan_J
    ]

    brackets = []
    for i in range(len(scan_J) - 1):
        f1, n1, _ = values[i]
        f2, n2, _ = values[i + 1]
        if n1 == expected_nodes and n2 == expected_nodes and np.isfinite(f1) and np.isfinite(f2) and f1 * f2 < 0.0:
            brackets.append((scan_J[i], scan_J[i + 1]))
    if not brackets:
        raise RuntimeError(f"No Numerov bracket found for state n={expected_nodes}.")

    lo, hi = min(
        brackets,
        key=lambda pair: abs((pair[0] + pair[1]) / (2.0 * HPLANCK * 1.0e6) - reference_MHz),
    )
    root = brentq(
        lambda E: _mismatch_and_nodes(E, r, potential_J, mu, HBAR)[0],
        lo,
        hi,
        xtol=1.0e-40,
        rtol=1.0e-14,
        maxiter=200,
    )
    mismatch, nodes, match = _mismatch_and_nodes(root, r, potential_J, mu, HBAR)
    energy_MHz = root / (HPLANCK * 1.0e6)
    return {
        "n": expected_nodes,
        "E_FDM_continuum_MHz": reference_MHz,
        "E_Numerov_MHz": energy_MHz,
        "delta_Numerov_minus_FDM_Hz": (energy_MHz - reference_MHz) * 1.0e6,
        "nodes": int(nodes),
        "match_r_nm": float(r[match] * 1.0e9),
        "mismatch_m_inverse": float(mismatch),
    }


def main() -> None:
    if PARAMS.l != 0:
        raise NotImplementedError("This verification script is configured for the ell=0 benchmark.")
    r = np.linspace(0.0, R_MAX_M, N_GRID, dtype=float)
    mu = reduced_mass(PARAMS)
    alpha = alpha_pol(PARAMS)
    potential_J = 0.5 * mu * PARAMS.omega_ion**2 * r**2 + alpha / (r**4 + PARAMS.r_c**4)

    rows = [
        solve_state(float(reference), n, r, potential_J)
        for n, reference in enumerate(CONTINUUM_FDM_LEVELS_MHZ)
    ]

    path = OUTDIR / "numerov_crosscheck_final.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("=== Two-sided Numerov cross-check ===")
    print(f"N={N_GRID}, r in [0,{R_MAX_M*1e9:.1f}] nm, r_c={PARAMS.r_c*1e9:.9f} nm")
    for row in rows:
        print(
            f"n={row['n']}: E_Numerov/h={row['E_Numerov_MHz']:+.12f} MHz, "
            f"Delta={row['delta_Numerov_minus_FDM_Hz']:+.6f} Hz, "
            f"nodes={row['nodes']}, match={row['match_r_nm']:.3f} nm"
        )
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
