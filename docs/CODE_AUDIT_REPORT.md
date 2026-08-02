# Code and repository audit report

## Audit scope

This report records the numerical corrections incorporated into the
reviewed source package and the repository-consistency review completed
on 2026-08-02.

The repository is intentionally source-code-only. Generated numerical
tables, graphical outputs, and local caches are recreated locally and
excluded from version control.

The audit distinguishes between:

1. numerical verification of the selected frozen operator;
2. calibration uncertainty;
3. within-model short-range regulator sensitivity;
4. unquantified model-form uncertainty associated with omitted trap
   dynamics and degrees of freedom.

## Repository consistency result

The reviewed repository has been synchronized with the revised scientific
scope of the associated manuscript.

The following issues were corrected:

1. `README.md` now describes the work as numerical certification and
   approximation benchmarking of an effective radial operator.
2. The experimental binding scale is identified as a single calibration
   anchor rather than an independently predicted ground-state result.
3. Excited levels are classified as regulator- and model-conditional
   eigenvalues.
4. Sub-hertz cross-method differences are identified as numerical
   diagnostics rather than physical uncertainties.
5. Regulator variation is identified as a within-model sensitivity and
   not as a complete uncertainty band.
6. Python 3.12 is identified as the tested and recommended environment.
7. `run_all_core.py` does not execute the angular-sector diagnostic twice.
8. The full driver includes the local soft-core-radius sensitivity
   calculation.
9. `CITATION.cff` and `.zenodo.json` use manuscript-consistent metadata
   for release `v1.0.2`.
10. `.zenodo.json` describes a software source-code release rather than a
    data archive.
11. `mpmath` was removed because no tracked source file imports it.
12. GitHub Actions performs source compilation, unit tests, and a focused
    numerical smoke test.
13. `.gitignore` excludes generated figures, tables, caches, archives,
    environments, and temporary output files.
14. `SHA256SUMS` is regenerated from the final tracked source and support
    files.
15. Legacy filenames containing terms such as
    `continuum_certified` are retained only where renaming could break
    imports, tests, workflows, or established execution paths.

## Reference-operator convention

The selected shared numerical convention is:

- `r_c = 25.876730807 nm`;
- `omega/(2 pi) = 1.2 MHz`;
- reference sector `ell = 0`;
- radial domain `0 <= r <= 650 nm`;
- explicit Dirichlet values at both endpoints;
- fourth-order five-point FDM with odd-reflection ghost closure;
- continuum grids `N = (1800, 2556, 3630, 5155)`;
- no Langer replacement in numerical diagonalization;
- Langer replacement only in WKB.

The full value of `r_c` is retained to identify the reproducible numerical
operator. It is a fitted effective parameter and must not be interpreted
as an experimentally measured microscopic length.

## Numerical corrections incorporated

### Shared parameter file

`unified_model_params.py` contains the final calibrated radius and remains
the sole source of shared physical and reference-operator parameters.

No approximation method independently recalibrates `r_c`.

### Finite-difference boundary convention

The obsolete workflow started at `r_min = 0.10 nm` and applied an interior
stencil after deleting endpoint rows.

The corrected implementation:

- starts at `r = 0`;
- imposes explicit Dirichlet endpoint values;
- uses odd-reflection ghost relations;
- restores the observed fourth-order grid-convergence behavior.

### Numerically converged reference spectrum

All fixed-operator cross-method comparisons use:

```text
E0/h = -14.999999963984 MHz
E1/h =  -9.211774845325 MHz
E2/h =  -3.530419674323 MHz