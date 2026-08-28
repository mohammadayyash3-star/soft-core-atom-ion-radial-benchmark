# Numerical certification and approximation benchmarking of a soft-core atom-ion radial Hamiltonian

Reproducibility repository for the paper:

**Numerical Certification and Approximation Benchmarking of an Experimentally Anchored Soft-Core Atom-Ion Radial Hamiltonian**

The repository contains the final Python implementation used for the continuum-certified finite-difference benchmark, independent Numerov validation, WKB quantization, variational estimate, perturbative-breakdown analysis, harmonic-oscillator-basis diagonalization, regulator diagnostics, angular-sector checks, zero-energy boundaries, continuum-certified trap-frequency sensitivity, and publication figures and supporting numerical outputs.

## Final model convention

- System: $^{87}$Rb--$^{88}$Sr$^+$ effective radial model
- Soft-core radius: `r_c = 25.876730807 nm`
- Effective confinement: `omega / 2pi = 1.2 MHz`
- Benchmark sector: `ell = 0`
- Radial domain: `0 <= r <= 650 nm`
- Boundary conditions: `u(0) = u(r_max) = 0`
- FDM: fourth-order five-point stencil with odd-reflection ghost closure
- Continuum grids: `N = (1800, 2556, 3630, 5155)`
- Langer replacement is used only in WKB

The experimental binding scale motivating the calibration is reported by Pinkas *et al.*, Nature Physics 19, 1573--1578 (2023), DOI: 10.1038/s41567-023-02158-5.

## Repository structure

```text
scripts/             Final analysis and figure-generation programs
reference_outputs/   Audited CSV/JSON outputs used for manuscript checks
tests/               Lightweight consistency tests
figures/             Destination for regenerated publication figures
docs/                Code-audit notes
```

## Installation

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick verification

```bash
python -m pytest -q
```

## Run the core workflow

```bash
python scripts/run_all_core.py
```

The core driver executes:

1. continuum-certified FDM benchmark and Numerov cross-check,
2. WKB energies,
3. variational ground-state estimate,
4. perturbative-breakdown analysis,
5. non-perturbative HO-basis diagonalization,
6. regulator-family diagnostics,
7. continuum zero-energy boundaries.

The two-parameter $(r_c,\omega)$ map is intentionally excluded from the quick workflow because it is computationally longer:

```bash
python scripts/rc_omega_parameter_map_fdm.py
```

## Main scripts

| Script | Purpose |
|---|---|
| `unified_model_params.py` | Single source of physical and calibrated parameters |
| `fdm_fourth_order_core.py` | Shared fourth-order FDM operator and eigensolvers |
| `continuum_certified_benchmark.py` | Grid sequence and continuum extrapolation |
| `numerov_crosscheck_final.py` | Independent state-resolved Numerov validation |
| `wkb_energy_only_final.py` | Langer-corrected radial WKB energies |
| `variational_energy_only_fixed.py` | Fixed-Hamiltonian variational ground-state estimate |
| `perturbation_theory_softcore.py` | PT2 energies and breakdown diagnostics |
| `ho_basis_diagonalization_no_jinja.py` | Non-perturbative HO-basis recovery and mixing |
| `regulator_dependence_fdm.py` | Calibrated regulator-family comparison |
| `angular_sector_diagnostic_final.py` | Fixed-input $\ell=0,1,2$ diagnostic |
| `zero_energy_crossing_continuum_final.py` | Continuum-extrapolated spectral boundaries |
| `rc_sensitivity_fdm_final.py` | Local soft-core-radius sensitivity |
| `trap_frequency_sensitivity_revision.py` | Continuum-certified 1.10-1.30 MHz trap-frequency sensitivity using fixed-\(r_c\) and continuum-recalibrated-\(r_c\) protocols |

### Trap-frequency sensitivity revision

The manuscript trap-frequency sensitivity analysis is reproduced with:

```bash
python scripts/trap_frequency_sensitivity_revision.py
```

The calculation evaluates effective radial frequencies

```text
omega/(2*pi) = 1.10, 1.20, 1.30 MHz
```

using two complementary protocols:

1. fixed `r_c = 25.876730807 nm`, with only the effective confinement frequency varied;
2. continuum-level recalibration of `r_c(omega)` so that `E0/h = -15 MHz` at each frequency.

The calculation preserves three negative-energy states throughout both protocols.

The generated outputs are written to:

```text
scripts/trap_frequency_revision_results/
```

and include machine-readable CSV/JSON results, manuscript-ready LaTeX output, and PDF/PNG figures.

## Figure scripts

## Figure scripts

- `figure3_effective_radial_potential_final.py`
- `figure4_continuum_negative_ladder_final.py`
- `figure5_fdm_numerical_certification_final.py`
- `figure9_fdm_wkb_variational_final.py`
- `figure10_perturbation_breakdown_final.py`
- `supplement_figure_numerov_final.py`

Several analysis scripts also generate their own supplementary figures, especially the regulator, angular-sector, sensitivity, and HO-basis programs.

## Certified negative-energy spectrum

The final continuum-extrapolated values are

```text
E0/h = -14.999999963984 MHz
E1/h =  -9.211774845325 MHz
E2/h =  -3.530419674323 MHz
```

Small differences in the last displayed decimal can arise if a manuscript table uses fewer digits. The values above are the repository reference values.

## Reproducibility rules

Do not mix outputs from this repository with the earlier workflow based on `r_min = 0.10 nm`, `N = 12000`, or deleted endpoint rows. Those files were intentionally excluded.

Generated caches, local environments, temporary plots, and obsolete output variants are excluded through `.gitignore`.

## Citation

Use the metadata in `CITATION.cff`. After archiving a GitHub release on Zenodo, add the Zenodo DOI to `CITATION.cff`, `.zenodo.json`, the manuscript data-availability statement, and this README.

## License

The code is released under the MIT License. See `LICENSE`.
