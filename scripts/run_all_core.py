from __future__ import annotations

"""Run the reviewed full analysis workflow in a reproducible order.

The full two-parameter ``(r_c, omega)`` map and publication-figure scripts are
intentionally excluded. Run them separately when needed.
"""

from pathlib import Path
import os
import subprocess
import sys

BASE = Path(__file__).resolve().parent
SCRIPTS = (
    "continuum_certified_benchmark.py",
    "numerov_crosscheck_final.py",
    "angular_sector_diagnostic_final.py",
    "rc_sensitivity_fdm_final.py",
    "wkb_energy_only_final.py",
    "variational_energy_only_fixed.py",
    "perturbation_theory_softcore.py",
    "ho_basis_diagonalization_no_jinja.py",
    "regulator_dependence_fdm.py",
    "zero_energy_crossing_continuum_final.py",
)


def main() -> None:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("NUMBA_CACHE_DIR", str(BASE / ".numba_cache"))

    for index, script in enumerate(SCRIPTS, start=1):
        path = BASE / script
        if not path.is_file():
            raise FileNotFoundError(f"Required workflow script is missing: {path}")
        print(f"\n=== [{index}/{len(SCRIPTS)}] {script} ===", flush=True)
        subprocess.run(
            [sys.executable, str(path)],
            cwd=BASE,
            env=env,
            check=True,
        )

    print("\nAll reviewed core calculations completed successfully.")


if __name__ == "__main__":
    main()
