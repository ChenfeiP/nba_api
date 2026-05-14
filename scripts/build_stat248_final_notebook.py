#!/usr/bin/env python3
"""Regenerate notebooks/stat248_final_report.ipynb (Stat 248 unified report scaffold)."""

from __future__ import annotations

import json
from pathlib import Path


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def _lines(text: str) -> list[str]:
    text = text.lstrip("\n")
    if not text.endswith("\n"):
        text += "\n"
    return [text]


_REPO = Path(__file__).resolve().parents[1]
_NB_PATH = _REPO / "notebooks" / "stat248_final_report.ipynb"


def build() -> None:
    cells: list[dict] = [
        _md(
            """# STAT 248 Final — NBA schedule congestion & team performance

**Chenfei Peng**

NBA schedules deliberately expose teams to unequal rest: **back-to-backs** and **short-turnaround** games stack on top of travel and time-zone change, similar to athlete fatigue studies elsewhere in sport science. At the unit of a **single team’s game**, we want to know whether crowded rest is still associated with weaker results **after** we account for momentum (a lag in performance), opponent schedule, home court, and season-wide shocks—not just a raw comparison of “tired” and “fresh” nights.

This notebook is the thread that connects that question to code. **§1** loads and audits the team-game table so every downstream model respects chronological order within each franchise-year. **Method 1** estimates pooled regressions with clustered standard errors by `TEAM_ID`; **Method 2** fits compact ARIMAX(1,0,0) models **inside each team-season streak** so the AR dynamics are not contaminated by other clubs; **Method 3** runs forward-expanding seasonal holdouts so test rows are always strictly future seasons. Each section saves CSVs and figures under `results/` for slides or a written report.

## Research question

Do **short rest** and **true back-to-backs** predict worse team-level **margin, shooting efficiency, and turnovers** once dynamics, opponents, and season are modeled—and do fatigue indicators help **out-of-sample prediction** beyond a parsimonious baseline?

The analysis compares **lagged OLS (cluster-robust)**, **ARIMAX by team-season**, and **forward-expanding season CV**.
"""
        ),
        _md(
            """## Reproducibility

**Environment.** Python ≥ 3.10. Install dependencies:

```
pip install pandas numpy matplotlib statsmodels
```

(Add `pip install nba-api` if you rebuild the panel from NBA Stats; see the rebuild subsection below.)

**Working directory.** The first code cell resolves the project root automatically: you may launch Jupyter from the repository root **or** from the `notebooks/` folder (it walks up one level when needed).

**Data.** The submission includes `data/nba_team_game_panel_stat248.csv` (regular season, 2022–23 through 2024–25). To rebuild that file from the API, set `REBUILD_PANEL = True` in the setup cell (network access; rate limits apply).

Re-executing later sections rewrites matching filenames in **`results/`**.

To regenerate this notebook file from the template script: ``python scripts/build_stat248_final_notebook.py``.
"""
        ),
        _code(
            '''
%matplotlib inline

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

warnings.filterwarnings("ignore", category=UserWarning)

_cwd = Path.cwd().resolve()
if (_cwd / "scripts" / "panel_structure.py").is_file():
    REPO_ROOT = _cwd
elif _cwd.name == "notebooks" and (_cwd.parent / "scripts" / "panel_structure.py").is_file():
    REPO_ROOT = _cwd.parent.resolve()
else:
    raise RuntimeError(
        "Cannot find scripts/panel_structure.py. Open the notebook from the repo root "
        f"(folder containing scripts/) or from notebooks/ inside it. Current cwd: {_cwd}"
    )

SCRIPTS = REPO_ROOT / "scripts"
DATA = REPO_ROOT / "data"
RESULTS = REPO_ROOT / "results"

RESULTS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SCRIPTS))

# --- toggles ---
REBUILD_PANEL = False  # True: download panel from NBA Stats (needs network; slower)
METHOD2_PARALLEL = True
METHOD2_MAXITER = 80
METHOD2_MAX_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 8))

RAW_CSV = DATA / "nba_team_game_panel_stat248.csv"
SEASONS = ["2022-23", "2023-24", "2024-25"]

os.environ.setdefault("PYTHONPATH", str(REPO_ROOT / "src"))
print("REPO_ROOT =", REPO_ROOT)
'''
        ),
        _md(
            """### Rebuild league panel from `nba_api`

Uses `scripts/build_nba_team_game_dataset.py` when `REBUILD_PANEL` is `True` in the setup cell."""
        ),
        _code(
            '''if REBUILD_PANEL:
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_nba_team_game_dataset.py"),
        "--seasons",
        *SEASONS,
        "--rolling-window",
        "5",
        "-o",
        str(RAW_CSV),
        "--season-delay",
        "0.65",
        "--timeout",
        "90",
    ]
    subprocess.run(
        cmd,
        check=True,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
else:
    print("Using existing CSV:", RAW_CSV)
'''
        ),
        _md(
            """---

## 1. Data & chronological index

Observation = team-side regular-season row. Rows sorted by `(TEAM_ID, SEASON_ID, GAME_DATE, GAME_ID)`. Season openers omit `days_rest` by construction."""
        ),
        _code(
            '''from panel_structure import PANEL_SORT_KEYS, sort_panel, validate_team_game_panel

assert RAW_CSV.exists(), f"Missing {RAW_CSV} — rebuild or restore file"

panel_sorted = sort_panel(pd.read_csv(RAW_CSV, parse_dates=["GAME_DATE"]))
print("canonical keys:", PANEL_SORT_KEYS)

summary = validate_team_game_panel(panel_sorted, sample_series=True, series_sample_n=2)
display(pd.Series({k: summary[k] for k in summary if k != "series_sample_heads"}, dtype=object))

print("\\nPeek at first team-season:")
display(summary["series_sample_heads"][0]["head"])

n_open = panel_sorted["days_rest"].isna().sum()
print(f"Rows with NA days_rest (season openers): {n_open}")

display(panel_sorted["SEASON_ID"].value_counts().sort_index())
'''
        ),
        _md(
            """---

## Method 1 — clustered lagged regression (three outcomes)

Spec matches proposal: lagged DV + fatigue + opponent fatigue + home + season FE; clustered SEs by `TEAM_ID`."""
        ),
        _code(
            '''from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

from stat248_lagged_regression import (
    fit_lagged_team_models,
    prepare_method1_sample,
    stack_regression_tables,
)

method_df = prepare_method1_sample(panel_sorted, outcomes=("point_diff", "EFG_PCT", "TOV"))
fits_m1 = fit_lagged_team_models(method_df)

for name, res in fits_m1.items():
    display(res.summary().tables[1])

stack_tbl = stack_regression_tables(fits_m1)
stack_tbl.sort_index().to_csv(RESULTS / "method1_lagged_coef_table.csv")
display(stack_tbl.round(5))

diag = fits_m1["point_diff"]
resid_frame = pd.DataFrame({"fitted": diag.fittedvalues, "residual": diag.resid})

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(resid_frame["fitted"], resid_frame["residual"], alpha=0.06, s=18)
axes[0].axhline(0, color="gray", linestyle="--")
axes[0].set_xlabel("Fitted margin")
axes[0].set_ylabel("Residual")

rr = resid_frame["residual"].values
axes[1].hist(rr, bins=40, density=True, color="#4f6fa9", alpha=0.85)
axes[1].set_xlabel("Residual")
axes[1].set_title("Pooled residuals (conditional on lag DV)")
plt.tight_layout()
fig.savefig(RESULTS / "report_method1_point_diff_residuals.png", dpi=200)
plt.show()

print("Durbin-Watson:", durbin_watson(rr))
lm, lm_pvalue, *_ = het_breuschpagan(rr, diag.model.exog)
print(f"Breusch-Pagan LM: {lm:.3f}; p-value: {lm_pvalue:.4g}")
'''
        ),
        _md(
            """---

## Method 2 — compact ARIMAX(1,0,0) with exogenous regressors

Per **team-season** trajectory (~82 games) we estimate `point_diff ~ AR(1) + regressors`; compare **fatigue-augmented** vs **home-only** dynamics for BIC, then summarise medians."""
        ),
        _code(
            '''from stat248_arimax import (
    aggregate_bic_summary,
    fit_panel_arimax,
    fit_panel_arimax_parallel,
    prepare_point_diff_series,
)

play_m2 = prepare_point_diff_series(panel_sorted)

if METHOD2_PARALLEL:
    try:
        meta2, coef2, diag2 = fit_panel_arimax_parallel(
            play_m2,
            sarimax_orders=((1, 0, 0),),
            fit_maxiter=METHOD2_MAXITER,
            max_workers=METHOD2_MAX_WORKERS,
        )
    except Exception as exc:
        print("Parallel ARIMAX failed, falling back to serial:", exc)
        meta2, coef2, diag2 = fit_panel_arimax(
            play_m2,
            sarimax_orders=((1, 0, 0),),
            fit_maxiter=METHOD2_MAXITER,
        )
else:
    meta2, coef2, diag2 = fit_panel_arimax(
        play_m2,
        sarimax_orders=((1, 0, 0),),
        fit_maxiter=METHOD2_MAXITER,
    )

bic2 = aggregate_bic_summary(meta2)
meta2.to_csv(RESULTS / "method2_arimax_meta.csv", index=False)
coef2.to_csv(RESULTS / "method2_arimax_coefs_wide.csv", index=False)
bic2.to_frame("value").to_csv(RESULTS / "method2_arimax_bic_summary.csv")
display(pd.concat({"bic_summary": bic2}, axis=1))

from stat248_figure_utils import (
    coef_digest_wide,
    plot_method2_bic_means,
    plot_method2_bic_win_frac,
    plot_method2_median_exog_coefs,
    plot_method2_sarimax_diagnostics,
)

_digest = coef_digest_wide(coef2)
_digest.to_frame("median").to_csv(RESULTS / "method2_arimax_coef_medians.csv")

plot_method2_bic_means(bic2, RESULTS / "slides_method2_bic_means.png")
plot_method2_bic_win_frac(bic2, RESULTS / "slides_method2_bic_full_wins_frac.png")
plot_method2_median_exog_coefs(_digest, RESULTS / "slides_method2_median_exog_coefs.png")

if diag2.get("result") is not None:
    pd.Series(
        {
            "diagnostic_TEAM_ID": diag2["TEAM_ID"],
            "diagnostic_SEASON_ID": diag2["SEASON_ID"],
            "diagnostic_BIC_full": diag2["bic"],
        }
    ).to_frame("value").to_csv(RESULTS / "method2_arimax_diagnostic_pick.csv")
    plot_method2_sarimax_diagnostics(
        diag2["result"], diag2, RESULTS / "slides_method2_arimax_residual_diagnostics.png"
    )

for p in sorted(RESULTS.glob("slides_method2_*.png")):
    display(Image(filename=str(p.resolve())))
'''
        ),
        _md(
            """---

## Method 3 — forward-expanding seasonal CV

Train only on earlier `SEASON_ID`s, hold out the chronologically **next season**, compare RMSE / MAE for models with vs without fatigue covariates. `pd_lag1` is computed within `TEAM_ID` so streaks spill across seasons."""
        ),
        _code(
            '''from stat248_rolling_cv import prepare_forward_cv_panel, run_forward_season_cv

from stat248_figure_utils import plot_method3_cv

play_cv = prepare_forward_cv_panel(panel_sorted)

fold_tbl, pooled_tbl = run_forward_season_cv(play_cv)
fold_tbl.to_csv(RESULTS / "method3_cv_per_fold.csv", index=False)
pooled_tbl.to_frame("value").to_csv(RESULTS / "method3_cv_pooled.csv")

display(fold_tbl)
display(pd.concat({"pooled_test_rows": pooled_tbl}, axis=1))

plot_method3_cv(fold_tbl, RESULTS)

for name in ["slides_method3_cv_rmse_mae_by_fold.png", "slides_method3_cv_rmse_delta.png"]:
    display(Image(filename=str((RESULTS / name).resolve())))
'''
        ),
        _md(
            """---

## Conclusion — linking results to the question

**Method 1 (lagged OLS, cluster by team).** Own **back-to-back** nights line up with **lower scoring margins** and slightly **lower eFG%** in the pooled specification after controls; the turnover channel is noisier. **Opponent** back-to-backs move margins in the intuitive direction (you benefit when the other side is short-rested), which is useful context but not a causal “fatigue” claim by itself.

**Method 2 (ARIMAX(1,0,0) per team-season).** Information criteria **often favour** the smaller model with only intercept + home relative to the fatigue-augmented specification, because BIC punishes extra parameters on short (~82-game) streaks. Pooled coefficient summaries still echo Method 1’s sign patterns for key schedule indicators.

**Method 3 (forward seasonal CV).** Holding out entire future seasons, models **with fatigue predictors achieve slightly better** RMSE/MAE than “lag + home only,” so there is **modest out-of-sample predictive value** even when in-sample ICs are stingy.

**Takeaway across instruments.** Schedule congestion leaves **economically interpretable** traces on margins and efficient shooting more clearly than on turnovers in these specs; predictive gains are **real but small** relative to baseline basketball noise.

**Limitations omitted by design:** transmeridian travel distance, minutes load, injuries, coaching experimentation; each would thicken the state vector. **Natural extensions** include hierarchical partial pooling across franchises or richer residual dynamics once compute allows.
"""
        ),
        _code(
            '''artifacts = sorted(RESULTS.glob("*.csv")) + sorted(RESULTS.glob("*.png"))
for p in artifacts:
    rel = p.relative_to(REPO_ROOT)
    print(rel)
'''
        ),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }

    _NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _NB_PATH.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", _NB_PATH)


if __name__ == "__main__":
    build()
