from __future__ import annotations



from typing import Any

import mpmath as mp
import numpy as np
from scipy.optimize import minimize

from unified_model_params import (
    PARAMS,
    HBAR,
    HPLANCK,
    reduced_mass,
    alpha_pol,
    harmonic_length,
)

# ============================================================
# Numerical controls
# ============================================================
mp.mp.dps = 50

RMAX_FACTOR = 18.0
MU_PEN = 1e-21

LBFGSB_WARM_OPTIONS = {"maxiter": 300, "ftol": 1e-12}
LBFGSB_FINAL_OPTIONS = {"maxiter": 600, "ftol": 1e-13}

SEED_PAIRS = (
    (3.0, 0.12),
    (1.5, 0.05),
    (6.0, 0.25),
)

# Frozen FDM baseline at the same Hamiltonian and the same r_c.
FDM_GROUND_E_MHZ = -15.000001606

SAFE_EPS = mp.mpf("1e-30")
MPF = Any

# ============================================================
# Shared physical parameters
# ============================================================
P = PARAMS


def _mp(x: float | str) -> MPF:
    return mp.mpf(str(x))


hbar = _mp(HBAR)
h = _mp(HPLANCK)
mu = _mp(reduced_mass(P))
omega = _mp(P.omega_ion)
alpha = _mp(alpha_pol(P))
l = int(P.l)
a_ho = _mp(harmonic_length(P))
r_c_fixed = _mp(P.r_c)
r_max_default = _mp(RMAX_FACTOR) * a_ho


# ============================================================
# Trial wavefunction
# ============================================================
def u_unnorm(r, beta, gamma):
    return (r ** (l + 1)) * mp.e ** (-mp.mpf("0.5") * beta * r * r - gamma / r)


def du_unnorm_dr(r, beta, gamma):
    u = u_unnorm(r, beta, gamma)
    return u * ((l + 1) / r - beta * r + gamma / r**2)


def norm_const(beta, gamma, r_c, r_max):
    integrand = lambda r: u_unnorm(r, beta, gamma) ** 2
    I = mp.quad(integrand, [0, r_c, r_max])
    return 1 / mp.sqrt(I)


def make_u(beta, gamma, r_c, r_max):
    A = norm_const(beta, gamma, r_c, r_max)
    u = lambda r: A * u_unnorm(r, beta, gamma)
    du = lambda r: A * du_unnorm_dr(r, beta, gamma)
    return u, du


# ============================================================
# Energy functional
# ============================================================
def total_energy(beta, gamma, r_c, r_max):
    u, du = make_u(beta, gamma, r_c, r_max)

    T_rad = mp.quad(lambda r: (hbar**2 / (2 * mu)) * (du(r) ** 2), [0, r_c, r_max])
    T_cent = mp.quad(
        lambda r: (hbar**2 / (2 * mu)) * (l * (l + 1)) / r**2 * (u(r) ** 2),
        [0, r_c, r_max],
    )
    V_osc = mp.quad(
        lambda r: mp.mpf("0.5") * mu * omega**2 * r**2 * (u(r) ** 2),
        [0, r_c, r_max],
    )
    V_core = mp.quad(
        lambda r: alpha / (r**4 + r_c**4) * (u(r) ** 2),
        [0, r_c, r_max],
    )
    return T_rad + T_cent + V_osc + V_core


def virial_residual(beta, gamma, r_c, r_max):
    """Internal stabilization diagnostic used only inside J."""
    u, du = make_u(beta, gamma, r_c, r_max)

    T_rad = mp.quad(lambda r: (hbar**2 / (2 * mu)) * (du(r) ** 2), [0, r_c, r_max])
    T_cent = mp.quad(
        lambda r: (hbar**2 / (2 * mu)) * (l * (l + 1)) / r**2 * (u(r) ** 2),
        [0, r_c, r_max],
    )
    Ttot = T_rad + T_cent

    term_osc = mp.quad(lambda r: (mu * omega**2) * r**2 * (u(r) ** 2), [0, r_c, r_max])
    term_core = mp.quad(
        lambda r: alpha * (-4) * (r**4) / (r**4 + r_c**4) ** 2 * (u(r) ** 2),
        [0, r_c, r_max],
    )
    return 2 * Ttot - (term_osc + term_core)


# ============================================================
# Log-parameter objectives
# ============================================================
def z_to_params(z):
    z0 = mp.mpf(str(z[0]))
    z1 = mp.mpf(str(z[1]))
    return mp.e ** z0, mp.e ** z1


def E_of_z(z, r_c, r_max) -> float:
    beta, gamma = z_to_params(z)
    return float(total_energy(beta, gamma, r_c, r_max))


def J_of_z(z, r_c, r_max, mu_pen: float = MU_PEN) -> float:
    beta, gamma = z_to_params(z)
    E = total_energy(beta, gamma, r_c, r_max)
    res = virial_residual(beta, gamma, r_c, r_max)
    scale = max(SAFE_EPS, abs(E))
    return float(E + mp.mpf(str(mu_pen)) * (res / scale) ** 2)


# ============================================================
# Optimization helpers
# ============================================================
def make_z_bounds():
    beta_floor = mp.mpf("1e-6") / (a_ho**2)
    gamma_floor = mp.mpf("1e-3") * a_ho
    return [
        (float(mp.log(beta_floor)), None),
        (float(mp.log(gamma_floor)), None),
    ]


def default_initial_guesses():
    for beta_factor, gamma_factor in SEED_PAIRS:
        beta0 = beta_factor / (float(a_ho) ** 2)
        gamma0 = gamma_factor * float(a_ho)
        yield np.array([np.log(beta0), np.log(gamma0)], dtype=float)


def optimize_energy_at_fixed_rc(r_c=r_c_fixed, r_max=r_max_default):
    bounds = make_z_bounds()

    warm_candidates = []
    for idx, z0 in enumerate(default_initial_guesses()):
        resJ = minimize(
            lambda z: J_of_z(z, r_c, r_max),
            np.asarray(z0, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
            options=LBFGSB_WARM_OPTIONS,
        )
        zJ = np.asarray(resJ.x, dtype=float)
        JJ = float(J_of_z(zJ, r_c, r_max))
        EJ = float(E_of_z(zJ, r_c, r_max))
        warm_candidates.append((JJ, EJ, idx, zJ))

    warm_candidates.sort(key=lambda item: (item[0], item[1]))
    _, _, best_seed_index, z_start = warm_candidates[0]

    resE = minimize(
        lambda z: E_of_z(z, r_c, r_max),
        z_start,
        method="L-BFGS-B",
        bounds=bounds,
        options=LBFGSB_FINAL_OPTIONS,
    )
    zE = np.asarray(resE.x, dtype=float)
    beta_mp, gamma_mp = z_to_params(zE)
    E_var = total_energy(beta_mp, gamma_mp, r_c, r_max)

    return {
        "best_seed_index": int(best_seed_index),
        "E_var_J": float(E_var),
        "E_var_MHz_signed": float(E_var / h / mp.mpf("1e6")),
        "E_var_MHz_abs": float(abs(E_var) / h / mp.mpf("1e6")),
        "FDM_ground_MHz": FDM_GROUND_E_MHZ,
        "delta_vs_FDM_kHz": 1e3 * (float(E_var / h / mp.mpf("1e6")) - FDM_GROUND_E_MHZ),
    }


# ============================================================
# Reporting
# ============================================================
def print_result(result):
    print("=== Variational ground-state solver (energy-only) ===")
    print(f"best warm-start seed index : {result['best_seed_index']}")
    print()
    print(f"E_var                      : {result['E_var_J']:.12e} J")
    print(f"E_var / h                  : {result['E_var_MHz_signed']:.9f} MHz")
    print(f"|E_var| / h                : {result['E_var_MHz_abs']:.9f} MHz")
    print()
    print(f"FDM ground E / h           : {result['FDM_ground_MHz']:.9f} MHz")
    print(f"Δ(E_var - E_FDM)           : {result['delta_vs_FDM_kHz']:.3f} kHz")


def main():
    result = optimize_energy_at_fixed_rc()
    print_result(result)


if __name__ == "__main__":
    main()
