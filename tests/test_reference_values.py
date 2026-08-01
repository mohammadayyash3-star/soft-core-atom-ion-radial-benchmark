from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from unified_model_params import PARAMS, PI  # noqa: E402
from benchmark_reference import (  # noqa: E402
    CONTINUUM_FDM_LEVELS_MHZ,
    NUMEROV_LEVELS_MHZ,
    PT2_LEVELS_MHZ,
    VARIATIONAL_GROUND_MHZ,
)
from run_all_core import SCRIPTS as CORE_SCRIPTS  # noqa: E402


def test_shared_parameters() -> None:
    assert math.isclose(PARAMS.r_c * 1.0e9, 25.876730807, abs_tol=1.0e-12)
    assert math.isclose(
        PARAMS.omega_ion / (2.0 * PI * 1.0e6),
        1.2,
        abs_tol=1.0e-12,
    )
    assert PARAMS.l == 0


def test_certified_spectrum() -> None:
    expected = (
        -14.999999963984,
        -9.211774845325,
        -3.530419674323,
    )
    for actual, target in zip(CONTINUUM_FDM_LEVELS_MHZ, expected):
        assert math.isclose(actual, target, rel_tol=0.0, abs_tol=5.0e-12)


def test_cross_method_references() -> None:
    assert VARIATIONAL_GROUND_MHZ > CONTINUUM_FDM_LEVELS_MHZ[0]
    assert len(NUMEROV_LEVELS_MHZ) == 3
    assert len(PT2_LEVELS_MHZ) == 3


def test_full_workflow_layout() -> None:
    assert len(CORE_SCRIPTS) == len(set(CORE_SCRIPTS))
    assert "angular_sector_diagnostic_final.py" in CORE_SCRIPTS
    assert "rc_sensitivity_fdm_final.py" in CORE_SCRIPTS
    assert "rc_omega_parameter_map_fdm.py" not in CORE_SCRIPTS
    for script in CORE_SCRIPTS:
        assert (SCRIPTS_DIR / script).is_file(), script
