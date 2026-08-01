from __future__ import annotations


import math
from dataclasses import dataclass

import numpy as np

from unified_model_params import (
    PARAMS,
    HBAR,
    HPLANCK,
    reduced_mass,
    alpha_pol,
    harmonic_length,
    l_eff_exact,
    l_eff_langer,
)

# ============================================================
# Shared physical / calibrated model parameters
# ============================================================
P = PARAMS

h = HPLANCK
hbar = HBAR
MU = reduced_mass(P)
OMEGA = P.omega_ion
ALPHA = alpha_pol(P)
RC = P.r_c
L = int(P.l)

# WKB uses the radial Langer prescription when enabled in shared params.
LEFF_WKB = l_eff_langer(P) if P.use_langer_wkb else l_eff_exact(P)

# ============================================================
# WKB-specific numerical controls
# ============================================================
GL_ORDER = 96
N_LEVELS = 3

# Frozen FDM baseline at the same Hamiltonian / same calibrated r_c.
FDM_BASELINE_MHZ = (-15.000001606, -9.163901439, -3.460486631)

# ============================================================
# Dimensionless scaling
# ============================================================
a_ho = harmonic_length(P)
rc_p = RC / a_ho
alpha_p = ALPHA / (hbar * OMEGA * a_ho**4)


# ============================================================
# Dimensionless effective potential V'(x) in x = r/a_ho
#   V'(x) = 1/2 x^2 + alpha'/(x^4 + rc'^4) + l_eff/(2 x^2)
# ============================================================
def Vprime_soft(
    x: float,
    leff: float = LEFF_WKB,
    ap: float = alpha_p,
    rcp: float = rc_p,
) -> float:
    if x <= 0.0:
        return float("inf")
    return 0.5 * x * x + ap / (x**4 + rcp**4) + 0.5 * leff / (x * x)


def dVdx_soft(
    x: float,
    leff: float = LEFF_WKB,
    ap: float = alpha_p,
    rcp: float = rc_p,
) -> float:
    if x <= 0.0:
        return -float("inf")
    lfac = 0.5 * leff
    denom = x**4 + rcp**4
    return x - (4.0 * ap * x**3) / (denom * denom) - 2.0 * lfac / (x**3)


def d2Vdx2_soft(
    x: float,
    leff: float = LEFF_WKB,
    ap: float = alpha_p,
    rcp: float = rc_p,
) -> float:
    if x <= 0.0:
        return float("inf")
    lfac = 0.5 * leff
    denom = x**4 + rcp**4
    term_core = (-12.0 * ap * x * x) / (denom * denom) + 32.0 * ap * x**6 / (denom**3)
    term_cent = 6.0 * lfac / (x**4)
    return 1.0 + term_core + term_cent


# ============================================================
# Safeguarded Newton helpers
# ============================================================
def newton_safeguarded(func, dfunc, x0: float, lo: float, hi: float, tol: float = 1e-12, itmax: int = 60) -> float:
    x = float(np.clip(x0, lo, hi))
    for _ in range(itmax):
        fx = func(x)
        dfx = dfunc(x)

        if not (np.isfinite(fx) and np.isfinite(dfx)) or dfx == 0.0:
            x = 0.5 * (lo + hi)
        else:
            xn = x - fx / dfx
            x = xn if (lo < xn < hi) else 0.5 * (lo + hi)

        fx = func(x)
        flo = func(lo)
        if flo * fx <= 0:
            hi = x
        else:
            lo = x

        if abs(fx) < tol or abs(hi - lo) < tol * max(1.0, abs(x)):
            break
    return x


def find_minimum_x_soft(leff: float = LEFF_WKB) -> float:
    x0 = max(1e-3, math.sqrt(max(leff, 0.25)))
    lo, hi = 1e-5, 60.0
    return newton_safeguarded(
        lambda xx: dVdx_soft(xx, leff=leff),
        lambda xx: d2Vdx2_soft(xx, leff=leff),
        x0,
        lo,
        hi,
    )


# ============================================================
# Classical turning points x_inner < x_outer for given E'
# ============================================================
def turning_points_soft(Ep: float, leff: float = LEFF_WKB):
    xm = find_minimum_x_soft(leff=leff)
    Vmin = Vprime_soft(xm, leff=leff)
    if (not np.isfinite(Vmin)) or Ep <= Vmin:
        return None

    def bracket_left():
        xhi = xm
        xlo = max(1e-6, xm / 2.0)
        g = lambda xx: Ep - Vprime_soft(xx, leff=leff)
        fhi = g(xhi)
        for _ in range(60):
            flo = g(xlo)
            if np.isfinite(flo) and np.isfinite(fhi) and flo * fhi <= 0:
                return xlo, xhi
            xhi = xlo
            xlo = max(1e-6, xlo / 2.0)
        return None

    def bracket_right():
        xlo = xm
        xhi = xm * 2.0 if xm > 0 else 2.0
        g = lambda xx: Ep - Vprime_soft(xx, leff=leff)
        flo = g(xlo)
        for _ in range(60):
            fhi = g(xhi)
            if np.isfinite(flo) and np.isfinite(fhi) and flo * fhi <= 0:
                return xlo, xhi
            xlo = xhi
            xhi = min(60.0, xhi * 2.0)
        return None

    left = bracket_left()
    right = bracket_right()
    if left is None or right is None:
        return None

    def newton_brent(a: float, b: float) -> float:
        g = lambda xx: Ep - Vprime_soft(xx, leff=leff)
        dg = lambda xx: -dVdx_soft(xx, leff=leff)
        fa = g(a)
        x = 0.5 * (a + b)
        for _ in range(40):
            fx = g(x)
            dfx = dg(x)
            if np.isfinite(fx) and np.isfinite(dfx) and dfx != 0.0:
                xn = x - fx / dfx
                x = xn if (a < xn < b) else 0.5 * (a + b)
            else:
                x = 0.5 * (a + b)
            fx = g(x)
            if abs(fx) < 1e-12:
                break
            if fa * fx <= 0:
                b = x
            else:
                a = x
                fa = fx
            if abs(b - a) < 1e-12 * max(1.0, abs(x)):
                break
        return x

    x1 = newton_brent(*left)
    x2 = newton_brent(*right)
    if not (np.isfinite(x1) and np.isfinite(x2) and x1 < x2):
        return None
    return x1, x2


# ============================================================
# Gauss–Legendre nodes for the theta-form action integral
# ============================================================
def gl_nodes_weights_on_theta(n: int = GL_ORDER):
    z, w = np.polynomial.legendre.leggauss(n)
    u = 0.5 * (z + 1.0)
    wu = 0.5 * w
    theta = (math.pi / 2.0) * u
    wtheta = wu * (math.pi / 2.0)
    return theta, wtheta


THETA, WTH = gl_nodes_weights_on_theta(GL_ORDER)


# ============================================================
# Action S'(E') and derivative dS'/dE'
#   Bohr–Sommerfeld: S'(E') = pi (n_r + 1/2)
# ============================================================
def action_and_dS_soft(Ep: float, leff: float = LEFF_WKB):
    tp = turning_points_soft(Ep, leff=leff)
    if tp is None:
        return float("nan"), float("nan"), None

    x1, x2 = tp
    s = np.sin(THETA)
    c = np.cos(THETA)
    x_vals = x1 + (x2 - x1) * (s**2)

    Vx = np.array([Vprime_soft(xx, leff=leff) for xx in x_vals])
    diff = Ep - Vx
    diff[diff < 0.0] = 0.0
    root = np.sqrt(2.0 * diff)

    dx_dtheta = 2.0 * (x2 - x1) * s * c
    Sprime = float(np.sum(root * dx_dtheta * WTH))

    denom = np.where(root > 0.0, root, np.inf)
    dSdE = float(np.sum((dx_dtheta / denom) * WTH))
    return Sprime, dSdE, (x1, x2)


# ============================================================
# Solve Bohr–Sommerfeld condition
# ============================================================
def solve_level_soft(n_r: int, leff: float = LEFF_WKB):
    xm = find_minimum_x_soft(leff=leff)
    Vmin = Vprime_soft(xm, leff=leff)
    Emin = Vmin + 1e-8

    Ehi = max(Emin + 1e-3, 2 * n_r + 1.5)
    target = math.pi * (n_r + 0.5)

    Spr, _, _ = action_and_dS_soft(Ehi, leff=leff)
    tries = 0
    while (not np.isfinite(Spr)) or Spr <= target:
        Ehi *= 1.5
        Spr, _, _ = action_and_dS_soft(Ehi, leff=leff)
        tries += 1
        if tries > 40:
            raise RuntimeError(f"Failed to bracket WKB level for n_r={n_r}")

    a, b = Emin, Ehi
    Sa, _, _ = action_and_dS_soft(a, leff=leff)
    if (not np.isfinite(Sa)) or Sa >= target:
        for _ in range(20):
            a = 0.5 * (a + b)
            Sa, _, _ = action_and_dS_soft(a, leff=leff)
            if np.isfinite(Sa) and Sa < target:
                break

    E = 0.5 * (a + b)
    for _ in range(30):
        Spr, dSdE, _ = action_and_dS_soft(E, leff=leff)
        if not (np.isfinite(Spr) and np.isfinite(dSdE) and dSdE > 0.0):
            E = 0.5 * (a + b)
            continue
        res = Spr - target
        if abs(res) < 1e-10:
            break
        En = E - res / dSdE
        if not (a < En < b):
            En = 0.5 * (a + b)
        if res > 0.0:
            b = E
        else:
            a = E
        E = En
        if abs(b - a) < 1e-12 * max(1.0, abs(E)):
            break

    Ep = float(E)
    E_J = Ep * hbar * OMEGA
    E_MHz_signed = E_J / h / 1e6
    E_MHz_abs = abs(E_MHz_signed)
    return Ep, E_J, E_MHz_signed, E_MHz_abs


# ============================================================
# Data structure
# ============================================================
@dataclass
class LevelRow:
    n_r: int
    Ep: float
    E_J: float
    E_MHz_signed: float
    E_MHz_abs: float
    FDM_MHz_signed: float | None
    delta_vs_FDM_kHz: float | None


# ============================================================
# Driver
# ============================================================
def solve_and_collect(n_levels: int = N_LEVELS, leff: float = LEFF_WKB):
    rows: list[LevelRow] = []
    for nr in range(n_levels):
        Ep, E_J, E_MHz_signed, E_MHz_abs = solve_level_soft(nr, leff=leff)
        fdm = FDM_BASELINE_MHZ[nr] if nr < len(FDM_BASELINE_MHZ) else None
        delta_kHz = None if fdm is None else 1e3 * (E_MHz_signed - fdm)
        rows.append(
            LevelRow(
                n_r=nr,
                Ep=Ep,
                E_J=E_J,
                E_MHz_signed=E_MHz_signed,
                E_MHz_abs=E_MHz_abs,
                FDM_MHz_signed=fdm,
                delta_vs_FDM_kHz=delta_kHz,
            )
        )
    return rows


# ============================================================
# Reporting
# ============================================================
def print_report(rows: list[LevelRow]):
    print("=== WKB radial solver (energy-only) ===")
    print(f"r_c                      : {RC * 1e9:.6f} nm")
    print(f"l                        : {L}")
    print(f"use_langer_wkb           : {P.use_langer_wkb}")
    print(f"l_eff (WKB)              : {LEFF_WKB:.6f}")
    print()
    print(" n_r      E_WKB [J]           E_WKB/h [MHz]      |E_WKB|/h [MHz]      E_FDM/h [MHz]      Δ(WKB-FDM) [kHz]")
    for row in rows:
        fdm_str = f"{row.FDM_MHz_signed: .9f}" if row.FDM_MHz_signed is not None else "     ---    "
        delta_str = f"{row.delta_vs_FDM_kHz: .3f}" if row.delta_vs_FDM_kHz is not None else "   ---   "
        print(
            f" {row.n_r:2d}   {row.E_J: .12e}    {row.E_MHz_signed: .9f}        "
            f"{row.E_MHz_abs: .9f}        {fdm_str}        {delta_str}"
        )


# ============================================================
# Main
# ============================================================
def main():
    rows = solve_and_collect(N_LEVELS, LEFF_WKB)
    print_report(rows)


if __name__ == "__main__":
    main()
