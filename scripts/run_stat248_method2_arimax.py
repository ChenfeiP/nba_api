#!/usr/bin/env python3
"""
Fast batch ARIMAX (parallel) + a few slide-ready figures.

Uses per team-season ``SARIMAX(1,0,0)`` with fatigue exogenous vs home-only baseline.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from panel_structure import sort_panel  # noqa: E402
from stat248_arimax import aggregate_bic_summary  # noqa: E402
from stat248_arimax import fit_panel_arimax  # noqa: E402
from stat248_arimax import fit_panel_arimax_parallel  # noqa: E402
from stat248_arimax import prepare_point_diff_series  # noqa: E402
from stat248_figure_utils import coef_digest_wide  # noqa: E402
from stat248_figure_utils import plot_method2_bic_means  # noqa: E402
from stat248_figure_utils import plot_method2_bic_win_frac  # noqa: E402
from stat248_figure_utils import plot_method2_median_exog_coefs  # noqa: E402
from stat248_figure_utils import plot_method2_sarimax_diagnostics  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, min((os.cpu_count() or 4) - 1, 12)),
        help="Process pool size (default: CPU−1 capped at 12)",
    )
    ap.add_argument(
        "--maxiter",
        type=int,
        default=80,
        help="L-BFGS cap per fit (speed knob; default 80)",
    )
    ap.add_argument("--serial", action="store_true", help="Disable parallel pool")
    args = ap.parse_args()

    csv_in = _REPO / "data" / "nba_team_game_panel_stat248.csv"
    out_dir = _REPO / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = prepare_point_diff_series(
        sort_panel(pd.read_csv(csv_in, parse_dates=["GAME_DATE"]))
    )

    if args.serial:
        meta, coefs, diag = fit_panel_arimax(
            panel,
            sarimax_orders=((1, 0, 0),),
            fit_maxiter=args.maxiter,
        )
    else:
        meta, coefs, diag = fit_panel_arimax_parallel(
            panel,
            sarimax_orders=((1, 0, 0),),
            fit_maxiter=args.maxiter,
            max_workers=args.workers,
        )

    meta.to_csv(out_dir / "method2_arimax_meta.csv", index=False)
    coefs.to_csv(out_dir / "method2_arimax_coefs_wide.csv", index=False)
    bic_s = aggregate_bic_summary(meta)
    bic_s.to_frame("value").to_csv(out_dir / "method2_arimax_bic_summary.csv")

    digest = coef_digest_wide(coefs)
    digest.to_frame("median").to_csv(out_dir / "method2_arimax_coef_medians.csv")

    plot_method2_bic_means(bic_s, out_dir / "slides_method2_bic_means.png")
    plot_method2_bic_win_frac(bic_s, out_dir / "slides_method2_bic_full_wins_frac.png")
    plot_method2_median_exog_coefs(digest, out_dir / "slides_method2_median_exog_coefs.png")

    if diag.get("result") is not None:
        pd.Series(
            {
                "diagnostic_TEAM_ID": diag["TEAM_ID"],
                "diagnostic_SEASON_ID": diag["SEASON_ID"],
                "diagnostic_BIC_full": diag["bic"],
            }
        ).to_frame("value").to_csv(out_dir / "method2_arimax_diagnostic_pick.csv")

        plot_method2_sarimax_diagnostics(
            diag["result"],
            diag,
            out_dir / "slides_method2_arimax_residual_diagnostics.png",
        )

        legacy = out_dir / "method2_arimax_residual_diagnostics.png"
        if not legacy.exists():
            import shutil

            shutil.copy(
                out_dir / "slides_method2_arimax_residual_diagnostics.png", legacy
            )

    print("Wrote under", out_dir)
    print(" Tables: method2_arimax_*.csv")
    print(
        " Figures: slides_method2_bic_means.png · slides_method2_bic_full_wins_frac.png "
        "· slides_method2_median_exog_coefs.png · slides_method2_arimax_residual_diagnostics.png"
    )
    print("\nBIC summary:\n", bic_s.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
