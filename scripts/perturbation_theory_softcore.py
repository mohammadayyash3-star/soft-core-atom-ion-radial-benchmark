from __future__ import annotations


from pathlib import Path
import csv
import math

import numpy as np
from scipy.special import eval_genlaguerre, gammaln, roots_genlaguerre

from unified_model_params import PARAMS, harmonic_length, alpha_dimensionless
from benchmark_reference import CONTINUUM_FDM_LEVELS_MHZ

OUTDIR = Path(__file__).resolve().parent
N_BASIS = 80
GL_ORDER = 340
N_LEVELS = 3


def h0_dimless(n: int, ell: int) -> float:
    return 2.0 * n + ell + 1.5


def normalization_constants(n_basis: int, ell: int) -> np.ndarray:
    n = np.arange(n_basis, dtype=float)
    return np.exp(
        0.5 * (math.log(2.0) + gammaln(n + 1.0) - gammaln(n + ell + 1.5))
    )


def potential_matrix_dimless(n_basis: int = N_BASIS, gl_order: int = GL_ORDER) -> np.ndarray:
    ell = int(PARAMS.l)
    alpha_lag = ell + 0.5
    y, w = roots_genlaguerre(gl_order, alpha_lag)
    L = np.empty((n_basis, y.size), dtype=float)
    for n in range(n_basis):
        L[n] = eval_genlaguerre(n, alpha_lag, y)
    N = normalization_constants(n_basis, ell)

    x_c = PARAMS.r_c / harmonic_length(PARAMS)
    alpha_p = alpha_dimensionless(PARAMS)
    kernel = np.sqrt(w / (y**2 + x_c**4))
    basis = (N[:, None] * L) * kernel[None, :]
    V = 0.5 * alpha_p * (basis @ basis.T)
    return 0.5 * (V + V.T)


def compute_rows() -> list[dict[str, float | int | str]]:
    V = potential_matrix_dimless()
    ell = int(PARAMS.l)
    energy_unit_MHz = PARAMS.omega_ion / (2.0 * math.pi) / 1.0e6
    rows = []

    for n in range(N_LEVELS):
        e0 = h0_dimless(n, ell)
        e1 = float(V[n, n])
        e2 = 0.0
        eta_max = 0.0
        dominant_m = -1
        for m in range(N_BASIS):
            if m == n:
                continue
            gap = e0 - h0_dimless(m, ell)
            matrix_element = float(V[n, m])
            e2 += matrix_element**2 / gap
            eta = abs(matrix_element / gap)
            if eta > eta_max:
                eta_max = eta
                dominant_m = m

        ept = (e0 + e1 + e2) * energy_unit_MHz
        efdm = float(CONTINUUM_FDM_LEVELS_MHZ[n])
        r10 = abs(e1 / e0)
        r21 = abs(e2 / e1)
        rows.append(
            {
                "n": n,
                "N_basis_sum": N_BASIS,
                "E0_MHz": e0 * energy_unit_MHz,
                "E1_MHz": e1 * energy_unit_MHz,
                "E2_MHz": e2 * energy_unit_MHz,
                "EPT_MHz": ept,
                "EFDM_continuum_MHz": efdm,
                "delta_PT_minus_FDM_kHz": 1.0e3 * (ept - efdm),
                "r10": r10,
                "r21": r21,
                "eta_max": eta_max,
                "dominant_basis_index": dominant_m,
                "verdict": "breakdown" if max(r10, r21, eta_max) >= 0.3 else "controlled",
            }
        )
    return rows


def main() -> None:
    rows = compute_rows()
    path = OUTDIR / "perturbation_breakdown_final.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("=== Second-order perturbation diagnostic ===")
    print(f"r_c = {PARAMS.r_c * 1e9:.9f} nm, N_basis sum = {N_BASIS}, GL order = {GL_ORDER}")
    for row in rows:
        print(
            f"n={row['n']}: E_PT/h={row['EPT_MHz']:+.12f} MHz, "
            f"Delta={row['delta_PT_minus_FDM_kHz']:+.3f} kHz, "
            f"r10={row['r10']:.4f}, r21={row['r21']:.4f}, "
            f"eta_max={row['eta_max']:.4f}, {row['verdict']}"
        )
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
