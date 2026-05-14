#!/usr/bin/env python3
"""Forward expanding season CV + slide-ready figures."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from panel_structure import sort_panel  # noqa: E402
from stat248_figure_utils import plot_method3_cv  # noqa: E402
from stat248_rolling_cv import prepare_forward_cv_panel  # noqa: E402
from stat248_rolling_cv import run_forward_season_cv  # noqa: E402


def main() -> int:
    out_dir = _REPO / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_in = _REPO / "data" / "nba_team_game_panel_stat248.csv"

    raw = pd.read_csv(csv_in, parse_dates=["GAME_DATE"])
    panel = sort_panel(raw)
    playable = prepare_forward_cv_panel(panel)

    per_fold, pooled = run_forward_season_cv(playable)
    per_fold.to_csv(out_dir / "method3_cv_per_fold.csv", index=False)
    pooled.to_frame("value").to_csv(out_dir / "method3_cv_pooled.csv")

    if len(per_fold):
        plot_method3_cv(per_fold, out_dir)

    print("Wrote", out_dir / "method3_cv_per_fold.csv")
    print("Wrote", out_dir / "method3_cv_pooled.csv")
    if len(per_fold):
        print(
            "Figures: slides_method3_cv_rmse_mae_by_fold.png · "
            "slides_method3_cv_rmse_delta.png"
        )
    print("\nPer-fold:\n", per_fold.to_string(index=False))
    print("\nPooled test rows:\n", pooled.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
