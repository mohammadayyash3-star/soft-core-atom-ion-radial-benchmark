from __future__ import annotations



from dataclasses import dataclass
import math


# ============================================================
# Fundamental constants (SI)
# ============================================================
HBAR = 1.054_571_817e-34       # J*s
HPLANCK = 6.626_070_15e-34     # J*s
KBOLTZ = 1.380_649e-23         # J/K
PI = math.pi


# ============================================================
# Core calibrated effective-model parameters
# ============================================================
@dataclass(frozen=True)
class EffectiveModelParams:
    # Species
    m_atom: float = 1.443_160_60e-25        # 87Rb (kg)
    m_ion: float = 1.461_273_36e-25         # 88Sr+ (kg)

    # Long-range polarization interaction
    C4: float = 1.09e-56                    # J*m^4

    # Trap frequency used in the effective radial model
    omega_ion: float = 2.0 * PI * 1.2e6     # rad/s

    # Angular momentum sector
    l: int = 0

    # FINAL calibrated soft-core radius from the numerical benchmark
    r_c: float = 25.876730807e-9               # m

    # Benchmark target used in calibration
    target_Hz: float = 15.0e6               # Hz

    # Numerical baseline conventions
    use_langer_numerical: bool = False      # numerical benchmark uses l(l+1)
    use_langer_wkb: bool = True             # WKB uses (l+1/2)^2

    # Experimental/simple-model phenomenology (keep here only as shared references)
    p0_se: float = 0.122
    R_low_over_high: float = 1.0 / 1.46


# Single default instance for import everywhere
PARAMS = EffectiveModelParams()


# ============================================================
# Derived physical quantities
# ============================================================
def reduced_mass(p: EffectiveModelParams = PARAMS) -> float:
    return p.m_atom * p.m_ion / (p.m_atom + p.m_ion)


def alpha_pol(p: EffectiveModelParams = PARAMS) -> float:
    """alpha = -C4/2 in the soft-core polarization potential."""
    return -0.5 * p.C4


def harmonic_length(p: EffectiveModelParams = PARAMS) -> float:
    mu = reduced_mass(p)
    return math.sqrt(HBAR / (mu * p.omega_ion))


def target_energy_J(p: EffectiveModelParams = PARAMS) -> float:
    return HPLANCK * p.target_Hz


def target_energy_MHz(p: EffectiveModelParams = PARAMS) -> float:
    return p.target_Hz / 1e6


def l_eff_exact(p: EffectiveModelParams = PARAMS) -> float:
    return p.l * (p.l + 1.0)


def l_eff_langer(p: EffectiveModelParams = PARAMS) -> float:
    return (p.l + 0.5) ** 2


def alpha_dimensionless(p: EffectiveModelParams = PARAMS) -> float:
    """alpha' = alpha / (ħ ω a_ho^4)"""
    a_ho = harmonic_length(p)
    return alpha_pol(p) / (HBAR * p.omega_ion * a_ho**4)


def rc_dimensionless(p: EffectiveModelParams = PARAMS) -> float:
    """r_c' = r_c / a_ho"""
    return p.r_c / harmonic_length(p)


# ============================================================
# Energy conversion helpers
# ============================================================
def J_to_Hz(E_J: float) -> float:
    return E_J / HPLANCK


def J_to_MHz(E_J: float) -> float:
    return E_J / HPLANCK / 1e6


def Hz_to_J(f_Hz: float) -> float:
    return HPLANCK * f_Hz


def MHz_to_J(f_MHz: float) -> float:
    return HPLANCK * f_MHz * 1e6


def J_to_mK(E_J: float) -> float:
    """Convert energy to mK using k_B."""
    return 1e3 * E_J / KBOLTZ


# ============================================================
# Shared reporting helper
# ============================================================
def summary_dict(p: EffectiveModelParams = PARAMS) -> dict:
    mu = reduced_mass(p)
    a_ho = harmonic_length(p)
    return {
        "m_atom_kg": p.m_atom,
        "m_ion_kg": p.m_ion,
        "mu_kg": mu,
        "C4_J_m4": p.C4,
        "alpha_J_m4": alpha_pol(p),
        "omega_rad_s": p.omega_ion,
        "omega_MHz_over_2pi": p.omega_ion / (2.0 * PI * 1e6),
        "l": p.l,
        "r_c_m": p.r_c,
        "r_c_nm": p.r_c * 1e9,
        "a_ho_m": a_ho,
        "a_ho_nm": a_ho * 1e9,
        "r_c_over_a_ho": rc_dimensionless(p),
        "alpha_dimensionless": alpha_dimensionless(p),
        "target_Hz": p.target_Hz,
        "target_MHz": target_energy_MHz(p),
        "target_E_J": target_energy_J(p),
        "p0_se": p.p0_se,
        "R_low_over_high": p.R_low_over_high,
    }


if __name__ == "__main__":
    s = summary_dict()
    print("=== Unified effective-model parameters ===")
    for k, v in s.items():
        print(f"{k:>24s} : {v}")
