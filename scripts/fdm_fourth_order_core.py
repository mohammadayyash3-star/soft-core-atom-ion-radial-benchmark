from __future__ import annotations



from dataclasses import dataclass
from typing import Callable, Optional
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eig_banded

from unified_model_params import PARAMS, HBAR, HPLANCK, reduced_mass, alpha_pol

PotentialFunction = Callable[[np.ndarray, float, float], np.ndarray]


def regulator_v1(r: np.ndarray, alpha: float, rc: float) -> np.ndarray:
    return alpha / (r**4 + rc**4)


def regulator_v2(r: np.ndarray, alpha: float, rc: float) -> np.ndarray:
    return alpha / (r**2 + rc**2) ** 2


def regulator_v3(r: np.ndarray, alpha: float, rc: float) -> np.ndarray:
    z = (r / rc) ** 4
    ratio = np.empty_like(z, dtype=float)
    small = z < 1.0e-10
    ratio[small] = 1.0 - 0.5 * z[small] + z[small] ** 2 / 6.0
    ratio[~small] = -np.expm1(-z[~small]) / z[~small]
    return alpha / rc**4 * ratio

REGULATORS: dict[str, PotentialFunction] = {
    "V1": regulator_v1,
    "V2": regulator_v2,
    "V3": regulator_v3,
}


@dataclass(frozen=True)
class GridSpec:
    N: int
    r_max_m: float = 650.0e-9
    r_min_m: float = 0.0

    def validate(self) -> None:
        if self.N < 5:
            raise ValueError("N must be at least 5 for the fourth-order stencil.")
        if abs(self.r_min_m) > 1.0e-30:
            raise ValueError(
                "The certified odd-reflection closure requires r_min_m=0. "
                "Use an explicitly labelled cutoff study for r_min>0."
            )
        if self.r_max_m <= 0.0:
            raise ValueError("r_max_m must be positive.")


@dataclass(frozen=True)
class BandedOperator:
    grid: GridSpec
    r_full_m: np.ndarray
    r_int_m: np.ndarray
    dr_m: float
    kinetic_main_MHz: np.ndarray
    kinetic_off1_MHz: float
    kinetic_off2_MHz: float


def build_kinetic_banded(grid: GridSpec, mu: Optional[float] = None) -> BandedOperator:
    grid.validate()
    if mu is None:
        mu = reduced_mass(PARAMS)

    r = np.linspace(0.0, grid.r_max_m, grid.N, dtype=float)
    dr = float(r[1] - r[0])
    r_int = r[1:-1]
    m = r_int.size

    inv_h2 = 1.0 / dr**2
    c0 = -30.0 / 12.0 * inv_h2
    c1 = +16.0 / 12.0 * inv_h2
    c2 = -1.0 / 12.0 * inv_h2

    # Odd reflection gives u_{-1}=-u_1 and u_N=-u_{N-2}.
    # Hence the first/last diagonal coefficient is c0-c2 = -29/(12 h^2).
    diag_d2 = np.full(m, c0, dtype=float)
    diag_d2[0] = c0 - c2
    diag_d2[-1] = c0 - c2

    kinetic_factor = -(HBAR**2) / (2.0 * mu) / (HPLANCK * 1.0e6)
    return BandedOperator(
        grid=grid,
        r_full_m=r,
        r_int_m=r_int,
        dr_m=dr,
        kinetic_main_MHz=kinetic_factor * diag_d2,
        kinetic_off1_MHz=kinetic_factor * c1,
        kinetic_off2_MHz=kinetic_factor * c2,
    )


def effective_potential_J(
    r_m: np.ndarray,
    *,
    rc_m: float,
    omega_rad_s: float = PARAMS.omega_ion,
    ell: int = PARAMS.l,
    regulator: str = "V1",
) -> np.ndarray:
    if regulator not in REGULATORS:
        raise KeyError(f"Unknown regulator {regulator!r}; choose from {tuple(REGULATORS)}")
    mu = reduced_mass(PARAMS)
    alpha = alpha_pol(PARAMS)
    v = 0.5 * mu * omega_rad_s**2 * r_m**2
    v = v + REGULATORS[regulator](r_m, alpha, rc_m)
    if ell != 0:
        v = v + HBAR**2 * ell * (ell + 1.0) / (2.0 * mu * r_m**2)
    return v


def banded_matrix_MHz(
    operator: BandedOperator,
    *,
    rc_m: float,
    omega_rad_s: float = PARAMS.omega_ion,
    ell: int = PARAMS.l,
    regulator: str = "V1",
) -> np.ndarray:
    m = operator.r_int_m.size
    ab = np.zeros((3, m), dtype=float)
    potential_MHz = effective_potential_J(
        operator.r_int_m,
        rc_m=rc_m,
        omega_rad_s=omega_rad_s,
        ell=ell,
        regulator=regulator,
    ) / (HPLANCK * 1.0e6)
    ab[2, :] = operator.kinetic_main_MHz + potential_MHz
    ab[1, 1:] = operator.kinetic_off1_MHz
    ab[0, 2:] = operator.kinetic_off2_MHz
    return ab


def solve_banded_levels_MHz(
    operator: BandedOperator,
    *,
    rc_m: float,
    n_levels: int = 6,
    omega_rad_s: float = PARAMS.omega_ion,
    ell: int = PARAMS.l,
    regulator: str = "V1",
    eigenvectors: bool = False,
):
    if n_levels < 1:
        raise ValueError("n_levels must be positive.")
    ab = banded_matrix_MHz(
        operator,
        rc_m=rc_m,
        omega_rad_s=omega_rad_s,
        ell=ell,
        regulator=regulator,
    )
    result = eig_banded(
        ab,
        lower=False,
        eigvals_only=not eigenvectors,
        select="i",
        select_range=(0, n_levels - 1),
        overwrite_a_band=True,
        check_finite=False,
    )
    return result


def sparse_matrix_J(
    operator: BandedOperator,
    *,
    rc_m: float,
    omega_rad_s: float = PARAMS.omega_ion,
    ell: int = PARAMS.l,
    regulator: str = "V1",
) -> sp.csr_matrix:
    m = operator.r_int_m.size
    main_J = operator.kinetic_main_MHz * HPLANCK * 1.0e6
    off1_J = operator.kinetic_off1_MHz * HPLANCK * 1.0e6
    off2_J = operator.kinetic_off2_MHz * HPLANCK * 1.0e6
    potential_J = effective_potential_J(
        operator.r_int_m,
        rc_m=rc_m,
        omega_rad_s=omega_rad_s,
        ell=ell,
        regulator=regulator,
    )
    return sp.diags(
        [
            off2_J * np.ones(m - 2),
            off1_J * np.ones(m - 1),
            main_J + potential_J,
            off1_J * np.ones(m - 1),
            off2_J * np.ones(m - 2),
        ],
        [-2, -1, 0, 1, 2],
        format="csr",
    )


def solve_sparse_levels_MHz(
    operator: BandedOperator,
    *,
    rc_m: float,
    n_levels: int = 6,
    omega_rad_s: float = PARAMS.omega_ion,
    ell: int = PARAMS.l,
    regulator: str = "V1",
    sigma_J: float = -1.0e-26,
    return_eigenvectors: bool = False,
):
    hmat = sparse_matrix_J(
        operator,
        rc_m=rc_m,
        omega_rad_s=omega_rad_s,
        ell=ell,
        regulator=regulator,
    )
    vals, vecs = spla.eigsh(
        hmat,
        k=n_levels,
        sigma=sigma_J,
        which="LM",
        tol=1.0e-12,
        maxiter=50000,
        ncv=min(hmat.shape[0], max(2 * n_levels + 12, 24)),
    )
    order = np.argsort(vals)
    vals = vals[order] / (HPLANCK * 1.0e6)
    vecs = vecs[:, order]
    return (vals, vecs, hmat) if return_eigenvectors else vals


def normalize_reduced_radial(u_int: np.ndarray, operator: BandedOperator) -> np.ndarray:
    u = np.zeros_like(operator.r_full_m)
    u[1:-1] = np.asarray(u_int, dtype=float)
    norm = math.sqrt(float(np.trapezoid(np.abs(u) ** 2, operator.r_full_m)))
    if not np.isfinite(norm) or norm <= 0.0:
        raise FloatingPointError("Wavefunction normalization failed.")
    return u / norm


def count_nodes(u: np.ndarray) -> int:
    v = np.asarray(u, dtype=float)
    amp = float(np.max(np.abs(v)))
    eps = max(1.0e-14, 1.0e-9 * amp)
    signs = np.sign(np.where(np.abs(v) < eps, 0.0, v))
    nz = signs[signs != 0.0]
    return int(np.sum(nz[1:] * nz[:-1] < 0.0)) if nz.size > 1 else 0
