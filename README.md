# Numerical Certification and Approximation Benchmark for an Experimentally Anchored Soft-Core Atom–Ion Radial Hamiltonian

Reproducibility source code for the manuscript:

**Numerical certification and approximation hierarchy for an experimentally anchored soft-core atom–ion radial Hamiltonian**

This repository contains the Python programs, tests, environment files,
citation metadata, and GitHub Actions configuration required to reproduce
the numerical analyses and publication figures for an experimentally
anchored effective soft-core atom–ion radial Hamiltonian.

The calculations verify the numerical solution of a selected frozen
operator and compare finite-difference, Numerov, WKB, variational,
perturbative, and harmonic-oscillator-basis methods. They do not constitute
a precision prediction of the complete Paul-trap spectrum, the dynamical
formation process, or the full center-of-mass--relative-coordinate problem.

Generated numerical tables, plots, PDFs, and local caches are intentionally
not version-controlled.

## Reference-operator convention

- System: $^{87}$Rb--$^{88}$Sr$^{+}$ effective radial model
- Soft-core radius: `r_c = 25.876730807 nm`
- Effective confinement: `omega / (2 pi) = 1.2 MHz`
- Reference sector: `ell = 0`
- Radial domain: `0 <= r <= 650 nm`
- Boundary conditions: `u(0) = u(r_max) = 0`
- FDM: fourth-order five-point stencil with odd-reflection ghost closure
- Continuum grids: `N = (1800, 2556, 3630, 5155)`
- Langer replacement: used only by the WKB calculation

The full value of `r_c` is retained solely to identify the reproducible
numerical operator. It is a fitted effective parameter and must not be
interpreted as an experimentally measured microscopic length.

The experimental binding scale reported by Pinkas *et al.* is used as a
single ground-state calibration anchor. The calibrated ground-state energy
is therefore not an independent prediction, while the additional
eigenvalues remain conditional on the selected reduced Hamiltonian and
short-range regulator.

Reference: Pinkas *et al.*, *Nature Physics* **19**, 1573--1578 (2023),
DOI `10.1038/s41567-023-02158-5`.

## Repository contents

```text
scripts/                     Numerical solvers and figure-generation programs
tests/                       Lightweight consistency and layout tests
docs/                        Numerical and repository audit notes
.github/workflows/tests.yml  Continuous-integration checks
README.md                    Installation and execution instructions
requirements.txt             Python dependencies
environment.yml              Conda environment
CITATION.cff                 Software citation metadata
.zenodo.json                 Zenodo deposit metadata
LICENSE                      MIT License