# Code and repository audit report

## Audit scope

This report records the numerical corrections already incorporated into the reviewed source package and the repository-consistency review completed on 2026-08-01. The repository is intentionally source-code-only: generated numerical and graphical outputs are recreated locally and excluded from Git.

## Repository consistency result

The original documentation was scientifically close to the final workflow but was not fully aligned with the intended GitHub upload policy. The reviewed package corrects the following issues:

1. `README.md` no longer claims that audited CSV/JSON outputs are included.
2. Python 3.12 is identified as the tested and recommended environment.
3. `run_all_core.py` no longer executes the angular-sector diagnostic twice.
4. The full driver now includes the local soft-core-radius sensitivity calculation.
5. `CITATION.cff` uses valid release metadata and contains no placeholder GitHub URL.
6. `.zenodo.json` describes a source-code release rather than a data archive.
7. `mpmath` was removed because no tracked source file imports it.
8. GitHub Actions now performs compilation, unit tests, and a focused numerical smoke test.
9. `.gitignore` consistently excludes generated figures, tables, caches, archives, and obsolete local files.
10. `SHA256SUMS` is regenerated from the final tracked source and support files.

## Numerical convention

The final shared model convention is:

- `r_c = 25.876730807 nm`;
- `omega/(2 pi) = 1.2 MHz`;
- benchmark sector `ell = 0`;
- radial domain `0 <= r <= 650 nm`;
- explicit Dirichlet values at both endpoints;
- fourth-order five-point FDM with odd-reflection ghost closure;
- continuum grids `N = (1800, 2556, 3630, 5155)`;
- no Langer replacement in numerical diagonalization;
- Langer replacement only in WKB.

## Critical numerical findings already corrected

### Shared parameter file

`unified_model_params.py` contains the final calibrated radius and remains the sole source of physical and calibrated model parameters.

### Finite-difference boundary convention

The obsolete workflow started at `r_min = 0.10 nm` and applied an interior stencil after deleting endpoint rows. The corrected implementation starts at `r = 0`, imposes explicit Dirichlet endpoint values, and uses odd-reflection ghost relations. This restores the observed fourth-order continuum behavior.

### Certified reference spectrum

All cross-method comparisons use:

```text
E0/h = -14.999999963984 MHz
E1/h =  -9.211774845325 MHz
E2/h =  -3.530419674323 MHz
```

The earlier excited-state spectrum is retained only in historical explanations and must not be used in tables or figures.

### Variational calculation

The variational solver treats the allowed boundary optimum `gamma = 0` explicitly and gives:

```text
E_var/h = -14.992134017480 MHz
Delta(var-FDM) = +7.865947 kHz
```

### Perturbation theory

The PT2 calculation uses generalized Gauss-Laguerre matrix elements and an 80-state intermediate sum. Its large energy deviations and diagnostic ratios establish perturbative breakdown for the calibrated fixed Hamiltonian.

### Harmonic-oscillator basis diagonalization

The non-perturbative basis calculation uses the final shared parameters and certified FDM references. It performs no method-dependent recalibration and no Langer replacement.

### Regulator and zero-energy studies

The regulator-family and zero-energy scripts use the corrected fourth-order boundary convention. The long regulator validation and full two-parameter map are not part of the lightweight CI test.

## Verification performed for this repository review

The following checks were completed successfully on the reviewed package:

- Python compilation of all tracked source and test files;
- unit tests of shared parameters, certified values, cross-method ordering, and repository layout;
- continuum-certified FDM grid sequence and fits;
- independent two-sided Numerov cross-check;
- angular-sector diagnostic;
- WKB, variational, PT2, and HO-basis programs through their normal execution path.

The long regulator-family calculation was not independently rerun to completion during this documentation-only review; its prior numerical audit is preserved in the source package and it remains available through `run_all_core.py`.

## Release rule

Before creating GitHub release `v1.0.0`, rerun:

```bash
python -m pytest -q
python -m compileall -q scripts tests
python scripts/softcore_benchmark_submission_final.py
```

Then confirm that `git status` is clean and that no generated image, PDF, CSV, JSON result, local environment, or cache is staged.
