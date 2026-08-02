# Repository manifest

## Included source code

- unified physical and calibrated reference-operator parameters;
- numerically converged benchmark reference values;
- fourth-order finite-difference core;
- fourth-order grid sequence and continuum extrapolation;
- independent two-sided Numerov verification;
- angular-sector diagnostics;
- local soft-core-radius sensitivity;
- radial WKB solver;
- variational ground-state solver;
- second-order perturbative-breakdown solver;
- non-perturbative harmonic-oscillator-basis diagonalization;
- calibrated regulator-family comparison and validation;
- continuum-extrapolated zero-energy boundaries;
- optional two-parameter `(r_c, omega)` map;
- focused benchmark and full-workflow drivers;
- publication-figure generation programs.

The retained numerical digits identify and verify the selected frozen
operator. They do not represent experimental precision or a complete
physical uncertainty estimate for the trapped atom-ion system.

## Included repository support files

- installation and execution instructions;
- Python and Conda dependency specifications;
- consistency and repository-layout tests;
- GitHub Actions workflow;
- citation and Zenodo metadata;
- MIT License;
- numerical and repository audit notes;
- final-file SHA-256 manifest.

## Scientific interpretation

The experimental binding scale is used as a single ground-state
calibration anchor. The calibrated ground-state energy is therefore not
an independent prediction.

Additional negative-energy eigenvalues, angular-sector results, spatial
observables, and energy gaps are conditional outputs of the selected
reduced Hamiltonian and short-range regulator.

The repository does not calculate:

- the complete time-dependent Paul-trap dynamics;
- center-of-mass--relative-coordinate coupling;
- trap anisotropy;
- micromotion-driven energy exchange;
- internal-channel dynamics;
- spin-exchange rates;
- molecular lifetime distributions;
- dynamical complex-formation probabilities.

Sub-hertz differences between numerical methods diagnose consistency of
the frozen operator only. Regulator variation is a within-model
sensitivity and must not be interpreted as a complete physical
uncertainty band.

## Deliberately excluded

- generated CSV, JSON, TeX, PDF, PNG, SVG, and other result files;
- manuscript and Supplementary Material PDFs;
- obsolete `r_min = 0.10 nm` scripts and stale reference spectra;
- rejected graphical-abstract variants;
- duplicate or intermediate code drafts;
- `__pycache__`, compiled Python files, and Numba caches;
- local virtual environments and editor settings;
- ZIP archives and temporary files.