from __future__ import annotations

"""Certified numerical reference values used for cross-method comparisons.

These are results of the continuum-certified fourth-order FDM workflow, not
independent model parameters.  The authoritative generator is
``continuum_certified_benchmark.py``; this compact module provides a stable
import target for the approximation scripts and prevents stale hard-coded
baselines from being duplicated across files.
"""

RC_NM = 25.876730807
OMEGA_OVER_2PI_MHZ = 1.2
ELL = 0
CONTINUUM_FDM_LEVELS_MHZ = (
    -14.999999963984,
    -9.211774845325,
    -3.530419674323,
)
NUMEROV_LEVELS_MHZ = (
    -14.999999990955,
    -9.211774877056,
    -3.530419697068,
)
PT2_LEVELS_MHZ = (
    -17.894585389,
    -9.801359428,
    -2.394431501,
)
VARIATIONAL_GROUND_MHZ = -14.992134017480
