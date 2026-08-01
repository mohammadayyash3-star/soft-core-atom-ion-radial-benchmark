from __future__ import annotations



from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.linalg import eigh
from scipy.special import eval_genlaguerre, gammaln, roots_genlaguerre

from unified_model_params import (
    PARAMS,
    HBAR,
    HPLANCK,
    alpha_pol,
    harmonic_length,
    reduced_mass,
)



OUTDIR = Path(__file__).resolve().parent

# Main convergence sweep. Keep 120 as the production default because it is already
# converged for the displayed low-lying levels with the present quadrature settings.
BASIS_SIZES: Tuple[int, ...] = (10, 20, 40, 80, 120)

# Number of physical levels reported against the FDM benchmark.
N_COMPARE_LEVELS = 3

# Generalized Gauss--Laguerre quadrature order used for the soft-core matrix elements.
# This is intentionally higher than 2*N_basis because the denominator is not a polynomial.
GL_ORDER = 340

# Frozen FDM production baseline for the same benchmark Hamiltonian.
FDM_LEVELS_MHZ: Dict[int, float] = {
    0: -15.000001606,
    1: -9.163901439,
    2: -3.460486631,
}

# Low-order PT values retained only as an optional reference in the printed report.
PT2_LEVELS_MHZ: Dict[int, float] = {
    0: -17.967248,
    1: -9.835178,
    2: -2.403982,
}


# ============================================================
# Data structures
# ============================================================
@dataclass(frozen=True)
class SharedModel:
    mu: float
    omega: float
    alpha: float
    r_c: float
    l: int
    a_ho: float
    x_c: float
    alpha_prime: float
    energy_unit_MHz: float


@dataclass(frozen=True)
class DiagonalizationResult:
    n_basis: int
    eigenvalues_dimless: np.ndarray
    eigenvalues_MHz: np.ndarray
    eigenvectors: np.ndarray
    h_matrix_dimless: np.ndarray
    v_matrix_dimless: np.ndarray


# ============================================================
# Shared physics
# ============================================================
def load_shared_model() -> SharedModel:
    p = PARAMS
    mu = reduced_mass(p)
    omega = p.omega_ion
    alpha = alpha_pol(p)
    a_ho = harmonic_length(p)
    x_c = p.r_c / a_ho
    alpha_prime = alpha / (HBAR * omega * a_ho**4)
    energy_unit_MHz = (HBAR * omega) / HPLANCK / 1.0e6

    return SharedModel(
        mu=mu,
        omega=omega,
        alpha=alpha,
        r_c=p.r_c,
        l=int(p.l),
        a_ho=a_ho,
        x_c=x_c,
        alpha_prime=alpha_prime,
        energy_unit_MHz=energy_unit_MHz,
    )


def h0_energy_dimless(n: int, ell: int) -> float:
    """Dimensionless radial harmonic-oscillator energy E/(hbar omega)."""
    return 2.0 * n + ell + 1.5


def normalization_constants(n_basis: int, ell: int) -> np.ndarray:
    """
    Normalization constants for the dimensionless radial HO functions in dx measure:

        phi_{n ell}(x) = N_{n ell} x^{ell+1} exp(-x^2/2)
                         L_n^{ell+1/2}(x^2),

    with integral_0^infty |phi_{n ell}(x)|^2 dx = 1.
    """
    n = np.arange(n_basis, dtype=float)
    log_norm = 0.5 * (math.log(2.0) + gammaln(n + 1.0) - gammaln(n + ell + 1.5))
    return np.exp(log_norm)


# ============================================================
# Matrix construction
# ============================================================
def laguerre_table(n_basis: int, ell: int, y_nodes: np.ndarray) -> np.ndarray:
    """Return L[n, i] = L_n^{ell+1/2}(y_i)."""
    alpha_lag = ell + 0.5
    table = np.empty((n_basis, y_nodes.size), dtype=float)
    for n in range(n_basis):
        table[n, :] = eval_genlaguerre(n, alpha_lag, y_nodes)

    if not np.all(np.isfinite(table)):
        raise FloatingPointError(
            "Non-finite generalized Laguerre values encountered. "
            "Reduce BASIS_SIZES or GL_ORDER."
        )
    return table


def softcore_matrix_dimless(model: SharedModel, n_basis: int, gl_order: int = GL_ORDER) -> np.ndarray:
    """
    Build V_mn/(hbar omega) in the dimensionless HO basis.

    With y = x^2 and alpha_L = ell + 1/2,

        V_mn' = alpha' * 1/2 * N_m N_n
                int_0^infty dy y^{alpha_L} exp(-y)
                L_m^{alpha_L}(y) L_n^{alpha_L}(y) / (y^2 + x_c^4).

    The integral is evaluated using generalized Gauss--Laguerre quadrature with
    the y^{alpha_L} exp(-y) weight built into the quadrature weights.
    """
    ell = model.l
    alpha_lag = ell + 0.5
    y, w = roots_genlaguerre(gl_order, alpha_lag)

    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(w))):
        raise FloatingPointError("Non-finite Gauss--Laguerre nodes or weights encountered.")

    L = laguerre_table(n_basis, ell, y)
    N = normalization_constants(n_basis, ell)

    # Use a square-root weighted Gram form for numerical stability:
    # G_mn = sum_i [N_m L_m(y_i)] [N_n L_n(y_i)] w_i/(y_i^2 + x_c^4)
    denom = y**2 + model.x_c**4
    sqrt_weighted_kernel = np.sqrt(w / denom)
    basis_values = (N[:, None] * L) * sqrt_weighted_kernel[None, :]
    gram = basis_values @ basis_values.T

    V = 0.5 * model.alpha_prime * gram
    V = 0.5 * (V + V.T)  # enforce exact symmetry against roundoff

    if not np.all(np.isfinite(V)):
        raise FloatingPointError("Non-finite potential matrix elements encountered.")
    return V


def build_hamiltonian_dimless(model: SharedModel, n_basis: int, gl_order: int = GL_ORDER) -> Tuple[np.ndarray, np.ndarray]:
    V = softcore_matrix_dimless(model, n_basis, gl_order=gl_order)
    H0 = np.diag([h0_energy_dimless(n, model.l) for n in range(n_basis)])
    H = H0 + V
    H = 0.5 * (H + H.T)
    return H, V


def diagonalize_ho_basis(model: SharedModel, n_basis: int, gl_order: int = GL_ORDER) -> DiagonalizationResult:
    H, V = build_hamiltonian_dimless(model, n_basis, gl_order=gl_order)
    evals, evecs = eigh(H)
    return DiagonalizationResult(
        n_basis=n_basis,
        eigenvalues_dimless=evals,
        eigenvalues_MHz=evals * model.energy_unit_MHz,
        eigenvectors=evecs,
        h_matrix_dimless=H,
        v_matrix_dimless=V,
    )


# ============================================================
# Diagnostics and reporting helpers
# ============================================================
def count_negative(values_MHz: np.ndarray) -> int:
    return int(np.sum(values_MHz < 0.0))


def relative_error_percent(test: float, reference: float) -> float:
    return 100.0 * abs(test - reference) / max(1.0e-30, abs(reference))


def convergence_table(results: Iterable[DiagonalizationResult]) -> pd.DataFrame:
    rows: List[Dict[str, float | int]] = []

    for res in results:
        row: Dict[str, float | int] = {
            "N_basis": res.n_basis,
            "N_negative": count_negative(res.eigenvalues_MHz),
        }

        max_abs_err = 0.0
        for n in range(N_COMPARE_LEVELS):
            E = float(res.eigenvalues_MHz[n])
            Efdm = FDM_LEVELS_MHZ[n]
            d_kHz = 1.0e3 * (E - Efdm)
            max_abs_err = max(max_abs_err, abs(d_kHz))
            row[f"E{n}_MHz"] = E
            row[f"Delta{n}_vs_FDM_kHz"] = d_kHz

        row["max_abs_delta_first3_kHz"] = max_abs_err
        rows.append(row)

    return pd.DataFrame(rows)


def final_levels_table(res: DiagonalizationResult) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    for n in range(N_COMPARE_LEVELS):
        E_ho = float(res.eigenvalues_MHz[n])
        E_fdm = FDM_LEVELS_MHZ[n]
        d_kHz = 1.0e3 * (E_ho - E_fdm)
        rel = relative_error_percent(E_ho, E_fdm)
        E_pt = PT2_LEVELS_MHZ.get(n, float("nan"))
        pt_delta_kHz = 1.0e3 * (E_pt - E_fdm)
        improvement = abs(pt_delta_kHz) / max(1.0e-30, abs(d_kHz))

        rows.append(
            {
                "n": n,
                "E_FDM_MHz": E_fdm,
                "E_HOdiag_MHz": E_ho,
                "Delta_HOdiag_FDM_kHz": d_kHz,
                "Rel_error_percent": rel,
                "E_PT2_MHz": E_pt,
                "Delta_PT2_FDM_kHz": pt_delta_kHz,
                "PT_error_over_HOdiag_error": improvement,
            }
        )
    return pd.DataFrame(rows)


def mixing_table(res: DiagonalizationResult, top_k: int = 6) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    coeffs = res.eigenvectors

    for state in range(N_COMPARE_LEVELS):
        weights = np.abs(coeffs[:, state]) ** 2
        order = np.argsort(weights)[::-1]
        participation_ratio = 1.0 / np.sum(weights**2)
        cumulative = np.cumsum(weights[order])
        n90 = int(np.searchsorted(cumulative, 0.90) + 1)
        n99 = int(np.searchsorted(cumulative, 0.99) + 1)

        dominant = "; ".join(
            f"n={int(idx)}: {weights[idx]:.4f}" for idx in order[:top_k]
        )

        rows.append(
            {
                "state": state,
                "E_HOdiag_MHz": float(res.eigenvalues_MHz[state]),
                "participation_ratio": participation_ratio,
                "components_for_90_percent": n90,
                "components_for_99_percent": n99,
                "dominant_components_weight": dominant,
                "max_single_weight": float(weights[order[0]]),
            }
        )

    return pd.DataFrame(rows)


def _latex_escape(value: object) -> str:
 
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _format_latex_cell(value: object, float_format: str = "%.6f") -> str:
    if isinstance(value, (float, np.floating)):
        return float_format % float(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return _latex_escape(value)


def write_latex_table(
    df: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
    float_format: str = "%.6f",
) -> None:
   
    ncols = len(df.columns)
    col_spec = "c" * ncols
    header = " & ".join(_latex_escape(col) for col in df.columns) + r" \\"

    body_lines = []
    for _, row in df.iterrows():
        body_lines.append(
            " & ".join(_format_latex_cell(row[col], float_format) for col in df.columns)
            + r" \\"
        )

    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + _latex_escape(caption) + r"}",
        r"\label{" + _latex_escape(label) + r"}",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\hline",
        header,
        r"\hline",
        *body_lines,
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(latex_lines), encoding="utf-8")


# ============================================================
# Plotting
# ============================================================
def plot_energy_comparison(final_df: pd.DataFrame, outdir: Path) -> None:
    states = final_df["n"].to_numpy(dtype=int)
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.axhline(0.0, linewidth=0.9, linestyle="--")
    ax.bar(states - width, final_df["E_FDM_MHz"], width=width, label="FDM benchmark")
    ax.bar(states, final_df["E_HOdiag_MHz"], width=width, label="HO-basis diagonalization")
    ax.bar(states + width, final_df["E_PT2_MHz"], width=width, label="Second-order PT")

    ax.set_xlabel("State index n")
    ax.set_ylabel("E/h (MHz)")
    ax.set_title("Non-perturbative HO-basis diagonalization vs FDM")
    ax.set_xticks(states)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "ho_basis_diag_energy_comparison.pdf")
    fig.savefig(outdir / "ho_basis_diag_energy_comparison.png", dpi=300)
    plt.close(fig)


def plot_convergence(conv_df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for n in range(N_COMPARE_LEVELS):
        ax.plot(
            conv_df["N_basis"],
            conv_df[f"Delta{n}_vs_FDM_kHz"],
            marker="o",
            label=fr"$n={n}$",
        )
    ax.axhline(0.0, linewidth=0.9, linestyle="--")
    ax.set_xlabel(r"$N_{\mathrm{basis}}$")
    ax.set_ylabel(r"$E_{\mathrm{HOdiag}}-E_{\mathrm{FDM}}$ (kHz)")
    ax.set_title("Convergence of HO-basis diagonalization")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "ho_basis_diag_convergence.pdf")
    fig.savefig(outdir / "ho_basis_diag_convergence.png", dpi=300)
    plt.close(fig)


def plot_mixing_weights(res: DiagonalizationResult, outdir: Path, max_component: int = 20) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    n_grid = np.arange(min(max_component, res.n_basis))
    for state in range(N_COMPARE_LEVELS):
        weights = np.abs(res.eigenvectors[: n_grid.size, state]) ** 2
        ax.plot(n_grid, weights, marker="o", label=fr"state $n={state}$")

    ax.set_xlabel("Unperturbed HO basis index")
    ax.set_ylabel(r"Expansion weight $|c_n|^2$")
    ax.set_title("Low-state mixing in the radial HO basis")
    ax.set_yscale("log")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "ho_basis_diag_mixing_weights.pdf")
    fig.savefig(outdir / "ho_basis_diag_mixing_weights.png", dpi=300)
    plt.close(fig)


# ============================================================
# Console report
# ============================================================
def print_model_summary(model: SharedModel) -> None:
    print("\n=== Non-perturbative HO-basis diagonalization ===")
    print(f"mu (kg)                  : {model.mu:.12e}")
    print(f"omega/2pi (MHz)          : {model.energy_unit_MHz:.9f}")
    print(f"ell                      : {model.l}")
    print(f"r_c (nm)                 : {model.r_c * 1.0e9:.9f}")
    print(f"a_ho (nm)                : {model.a_ho * 1.0e9:.9f}")
    print(f"x_c = r_c/a_ho           : {model.x_c:.9f}")
    print(f"alpha'                   : {model.alpha_prime:.9f}")
    print(f"Gauss-Laguerre order     : {GL_ORDER}")
    print("No r_c recalibration is performed.")
    print("No Langer replacement is used in this basis diagonalization.\n")


def print_results(conv_df: pd.DataFrame, final_df: pd.DataFrame, mix_df: pd.DataFrame) -> None:
    print("--- Convergence against production FDM benchmark ---")
    display_cols = ["N_basis", "N_negative"]
    for n in range(N_COMPARE_LEVELS):
        display_cols += [f"E{n}_MHz", f"Delta{n}_vs_FDM_kHz"]
    display_cols += ["max_abs_delta_first3_kHz"]
    print(conv_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n--- Final comparison at largest N_basis ---")
    print(final_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n--- Mixing diagnostics ---")
    print(mix_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


# ============================================================
# Main
# ============================================================
def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    model = load_shared_model()
    print_model_summary(model)

    if max(BASIS_SIZES) > GL_ORDER:
        raise ValueError("For this script, keep GL_ORDER >= max(BASIS_SIZES).")

    results: List[DiagonalizationResult] = []
    for n_basis in BASIS_SIZES:
        print(f"Building and diagonalizing N_basis = {n_basis} ...")
        results.append(diagonalize_ho_basis(model, n_basis, gl_order=GL_ORDER))

    conv_df = convergence_table(results)
    final_res = results[-1]
    final_df = final_levels_table(final_res)
    mix_df = mixing_table(final_res)

    # Write machine-readable outputs.
    conv_df.to_csv(OUTDIR / "ho_basis_diag_convergence.csv", index=False)
    final_df.to_csv(OUTDIR / "ho_basis_diag_final_levels.csv", index=False)
    mix_df.to_csv(OUTDIR / "ho_basis_diag_mixing.csv", index=False)

    # Write LaTeX outputs.
    write_latex_table(
        conv_df,
        OUTDIR / "ho_basis_diag_convergence.tex",
        caption=(
            "Convergence of the non-perturbative harmonic-oscillator basis "
            "diagonalization for the benchmark-fixed soft-core radial Hamiltonian."
        ),
        label="tab:ho_basis_diag_convergence",
    )
    write_latex_table(
        final_df,
        OUTDIR / "ho_basis_diag_final_levels.tex",
        caption=(
            "Final harmonic-oscillator basis diagonalization compared with the FDM "
            "benchmark and the second-order perturbative result."
        ),
        label="tab:ho_basis_diag_final_levels",
    )
    write_latex_table(
        mix_df,
        OUTDIR / "ho_basis_diag_mixing.tex",
        caption=(
            "Mixing diagnostics for the low-lying eigenstates obtained by "
            "non-perturbative diagonalization in the harmonic-oscillator basis."
        ),
        label="tab:ho_basis_diag_mixing",
    )

    # Write figures.
    plot_energy_comparison(final_df, OUTDIR)
    plot_convergence(conv_df, OUTDIR)
    plot_mixing_weights(final_res, OUTDIR)

    print_results(conv_df, final_df, mix_df)

    print("\nFiles written:")
    for name in [
        "ho_basis_diag_convergence.csv",
        "ho_basis_diag_final_levels.csv",
        "ho_basis_diag_mixing.csv",
        "ho_basis_diag_convergence.tex",
        "ho_basis_diag_final_levels.tex",
        "ho_basis_diag_mixing.tex",
        "ho_basis_diag_energy_comparison.pdf",
        "ho_basis_diag_convergence.pdf",
        "ho_basis_diag_mixing_weights.pdf",
    ]:
        print(f"  - {OUTDIR / name}")


if __name__ == "__main__":
    main()
