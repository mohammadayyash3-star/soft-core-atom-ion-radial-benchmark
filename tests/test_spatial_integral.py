"""Regression checks for the physical radial-cutoff integral and FDM boundary."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from regulator_dependence_fdm import (  # noqa: E402
    FDMConfig, REGULATORS, build_fdm_operator, make_params, radial_observables
)

@pytest.mark.parametrize("cutoff", [0.31, 0.35, 0.8, 1.0])
def test_normalized_constant_density(cutoff):
    r = np.linspace(0.0, 1.0, 11)
    u = np.ones_like(r)
    assert abs(radial_observables(u, r, cutoff)["P_r_lt_rc"] - cutoff) < 1e-12

def test_linear_density_fractional_cell():
    r = np.linspace(0.0, 1.0, 11)
    u = np.sqrt(2.0 * r)
    assert abs(radial_observables(u, r, 0.45)["P_r_lt_rc"] - 0.45**2) < 1e-12

def test_cutoff_outside_grid():
    r = np.linspace(0.0, 1.0, 11)
    u = np.ones_like(r)
    assert radial_observables(u, r, -0.1)["P_r_lt_rc"] == 0.0
    assert abs(radial_observables(u, r, 2.0)["P_r_lt_rc"] - 1.0) < 1e-12

def test_regulator_boundary_matches_certified_domain():
    cfg = FDMConfig(N=101)
    p = make_params(25.876730807, cfg)
    h, r, (i0, i1), _ = build_fdm_operator(p, REGULATORS[0])
    assert r[0] == 0.0
    assert (i0, i1) == (1, 99)
    assert h.shape == (99, 99)
