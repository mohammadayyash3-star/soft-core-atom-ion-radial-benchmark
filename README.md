# Continuum-certified atom-ion spectral benchmark

Reproducibility source code for the manuscript:

**Continuum-certified spectral benchmark for a calibrated trap-assisted atom-ion model: approximation hierarchy and short-range dependence**

This repository contains the Python programs, tests, environment files, citation metadata, and GitHub Actions configuration needed to regenerate the numerical analyses and publication figures. Generated numerical tables, plots, PDFs, and local caches are intentionally not version-controlled.

## Final model convention

- System: $^{87}$Rb--$^{88}$Sr$^+$ effective radial model
- Soft-core radius: `r_c = 25.876730807 nm`
- Effective confinement: `omega / (2 pi) = 1.2 MHz`
- Benchmark sector: `ell = 0`
- Radial domain: `0 <= r <= 650 nm`
- Boundary conditions: `u(0) = u(r_max) = 0`
- FDM: fourth-order five-point stencil with odd-reflection ghost closure
- Continuum grids: `N = (1800, 2556, 3630, 5155)`
- Langer replacement: used only by the WKB calculation

The calibration is motivated by the experimental binding scale reported by Pinkas *et al.*, *Nature Physics* **19**, 1573--1578 (2023), DOI `10.1038/s41567-023-02158-5`.

## Repository contents

```text
scripts/                    Numerical solvers and figure-generation programs
tests/                      Lightweight consistency and layout tests
docs/                       Numerical and repository audit notes
.github/workflows/tests.yml  Continuous-integration checks
README.md                    Installation and execution instructions
requirements.txt             Python dependencies
environment.yml              Conda environment
CITATION.cff                 Software citation metadata
.zenodo.json                 Zenodo deposit metadata
LICENSE                      MIT License
```

No manuscript PDF, generated figure, CSV table, JSON result file, or local virtual environment is tracked.

## Tested environment

The reviewed workflow was tested with **Python 3.12.10** on Windows. Run all commands from the repository root and use one Python interpreter consistently. In VS Code, select `.venv\Scripts\python.exe` after creating the environment.

## Installation

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux or macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Verification before production runs

```bash
python -m pytest -q
python -m compileall -q scripts tests
python scripts/unified_model_params.py
```

The expected test result is:

```text
4 passed
```

The parameter script must report `r_c_nm = 25.876730807`, `omega_MHz_over_2pi = 1.2`, and `l = 0`.

## Fast benchmark workflow

```bash
python scripts/softcore_benchmark_submission_final.py
```

This focused workflow executes:

1. the continuum-certified fourth-order FDM benchmark;
2. the independent two-sided Numerov cross-check;
3. the fixed-input angular-sector diagnostic for `ell = 0, 1, 2`.

## Full analysis workflow

```bash
python scripts/run_all_core.py
```

The full driver executes, without method-dependent refitting:

1. continuum-certified FDM extrapolation;
2. independent Numerov validation;
3. angular-sector diagnostic;
4. local soft-core-radius sensitivity;
5. radial WKB energies;
6. variational ground-state upper bound;
7. second-order perturbative-breakdown diagnostics;
8. non-perturbative harmonic-oscillator basis diagonalization;
9. regulator-family calibration and validation;
10. continuum-extrapolated zero-energy boundaries.

The regulator and sensitivity calculations are longer than the basic FDM/Numerov verification. The full two-parameter `(r_c, omega)` map is intentionally excluded from the driver and must be run separately:

```bash
python scripts/rc_omega_parameter_map_fdm.py
```

## Main analysis scripts

| Script | Purpose |
|---|---|
| `unified_model_params.py` | Single source of physical and calibrated model parameters |
| `benchmark_reference.py` | Certified continuum and cross-method reference values |
| `fdm_fourth_order_core.py` | Shared fourth-order FDM operator and eigensolvers |
| `continuum_certified_benchmark.py` | Grid sequence, continuum extrapolation, and backend comparison |
| `numerov_crosscheck_final.py` | Independent state-resolved two-sided Numerov validation |
| `angular_sector_diagnostic_final.py` | Fixed-input `ell = 0, 1, 2` diagnostic |
| `rc_sensitivity_fdm_final.py` | Local sensitivity to the calibrated soft-core radius |
| `wkb_energy_only_final.py` | Langer-corrected radial WKB energies |
| `variational_energy_only_fixed.py` | Fixed-Hamiltonian variational ground-state upper bound |
| `perturbation_theory_softcore.py` | Converged PT2 energies and breakdown diagnostics |
| `ho_basis_diagonalization_no_jinja.py` | Non-perturbative HO-basis recovery and mixing diagnostics |
| `regulator_dependence_fdm.py` | Calibrated regulator-family comparison |
| `zero_energy_crossing_continuum_final.py` | Continuum-extrapolated spectral boundaries |
| `rc_omega_parameter_map_fdm.py` | Optional long two-parameter map |

## Figure-generation scripts

```bash
python scripts/figure3_effective_radial_potential_final.py
python scripts/figure4_continuum_negative_ladder_final.py
python scripts/figure5_fdm_numerical_certification_final.py
python scripts/figure9_fdm_wkb_variational_final.py
python scripts/figure10_perturbation_breakdown_final.py
python scripts/supplement_figure_numerov_final.py
```

The scripts write generated files beside the source programs or into ignored subdirectories under `scripts/`. These outputs remain local and are excluded by `.gitignore`.

## Certified negative-energy spectrum

The continuum-extrapolated reference values are:

```text
E0/h = -14.999999963984 MHz
E1/h =  -9.211774845325 MHz
E2/h =  -3.530419674323 MHz
```

The independent Numerov values are:

```text
E0/h = -14.999999990955 MHz
E1/h =  -9.211774877056 MHz
E2/h =  -3.530419697068 MHz
```

The corresponding Numerov-minus-FDM differences are approximately `-0.026971`, `-0.031731`, and `-0.022745 Hz`.

## Reproducibility rules

- Run programs from the repository root.
- Do not use the obsolete workflow based on `r_min = 0.10 nm` or the old excited-state spectrum.
- Do not recalibrate `r_c` separately for WKB, variational, perturbative, or HO-basis calculations.
- Keep `unified_model_params.py` as the sole source of physical and calibrated model parameters.
- Treat `benchmark_reference.py` as a compact record of certified numerical outputs, not as an independent solver.
- Do not commit generated plots, tables, caches, local environments, or temporary output directories.

## Citation and archival release

The repository can be published first as a public GitHub repository. After the final manuscript-consistent commit:

1. create GitHub release `v1.0.0`;
2. archive that release with Zenodo;
3. add the Zenodo DOI and final GitHub URL to `CITATION.cff`, `.zenodo.json`, and the manuscript Code Availability statement;
4. create a small follow-up commit containing only the final metadata updates.

## License

The source code is released under the MIT License. See `LICENSE`.
