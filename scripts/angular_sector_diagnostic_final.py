from __future__ import annotations

"""Fixed-Hamiltonian angular-sector diagnostic for ell=0,1,2."""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

from unified_model_params import PARAMS, HPLANCK, reduced_mass, alpha_pol
from fdm_fourth_order_core import (
    GridSpec,
    build_kinetic_banded,
    solve_banded_levels_MHz,
    effective_potential_J,
)

OUTDIR = Path(__file__).resolve().parent
N_GRID = 5155
RMAX_M = 650.0e-9
ELLS = (0, 1, 2)
N_LEVELS = 6


def main() -> None:
    operator = build_kinetic_banded(GridSpec(N=N_GRID, r_max_m=RMAX_M))
    summary = []
    levels_rows = []

    for ell in ELLS:
        levels = solve_banded_levels_MHz(
            operator,
            rc_m=PARAMS.r_c,
            n_levels=N_LEVELS,
            ell=ell,
        )
        potential = effective_potential_J(
            operator.r_int_m,
            rc_m=PARAMS.r_c,
            ell=ell,
        ) / (HPLANCK * 1.0e6)
        negative = levels[levels < 0.0]
        row = {
            "ell": ell,
            "ell_factor": ell * (ell + 1),
            "N_negative": int(np.sum(levels < 0.0)),
            "Vmin_MHz": float(np.min(potential)),
            "E0_MHz": float(negative[0]) if len(negative) > 0 else np.nan,
            "E1_MHz": float(negative[1]) if len(negative) > 1 else np.nan,
            "E2_MHz": float(negative[2]) if len(negative) > 2 else np.nan,
        }
        summary.append(row)
        for n, energy in enumerate(levels):
            levels_rows.append(
                {
                    "ell": ell,
                    "state_index": n,
                    "E_MHz": float(energy),
                    "classification": "negative" if energy < 0.0 else "positive trap-confined",
                }
            )

    with (OUTDIR / "angular_sector_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader(); writer.writerows(summary)
    with (OUTDIR / "angular_sector_levels.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(levels_rows[0].keys()))
        writer.writeheader(); writer.writerows(levels_rows)

    # Compact energy-ladder figure.
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for ell in ELLS:
        vals = [row["E_MHz"] for row in levels_rows if row["ell"] == ell]
        for n, energy in enumerate(vals):
            color = "C0" if energy < 0.0 else "0.55"
            ax.hlines(energy, ell - 0.25, ell + 0.25, color=color, lw=2.2)
            if n < 3:
                ax.text(ell + 0.28, energy, rf"$n={n}$", va="center", fontsize=7.5)
    ax.axhline(0.0, color="0.15", ls="--", lw=0.9)
    ax.set_xticks(ELLS)
    ax.set_xlabel(r"Angular sector, $\ell$")
    ax.set_ylabel(r"Energy divided by $h$ (MHz)")
    ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    fig.savefig(OUTDIR / "partial_wave_energy_ladder.pdf")
    fig.savefig(OUTDIR / "partial_wave_energy_ladder.png", dpi=600)
    plt.close(fig)

    print("=== Angular-sector diagnostic ===")
    print(f"N={N_GRID}, dr={operator.dr_m*1e9:.9f} nm, r_c={PARAMS.r_c*1e9:.9f} nm")
    for row in summary:
        print(
            f"ell={row['ell']}: N_-={row['N_negative']}, Vmin/h={row['Vmin_MHz']:+.6f} MHz, "
            f"E0={row['E0_MHz']:+.9f}, E1={row['E1_MHz']:+.9f}, "
            f"E2={row['E2_MHz'] if np.isfinite(row['E2_MHz']) else float('nan'):+.9f}"
        )


if __name__ == "__main__":
    main()
