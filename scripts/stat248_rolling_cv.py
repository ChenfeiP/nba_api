"""
STAT 248 — Method 3: forward-expanding prediction CV on team-game panel.

Trains on earlier NBA seasons only, tests on the next season (`SEASON_ID` sort order).
Lag of ``point_diff`` shifts within ``TEAM_ID`` so it spans season boundaries.

Compares ``full`` (fatigue + lag + home) vs ``no_fatigue`` (lag + home) OLS specs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
except ImportError as exc:  # pragma: no cover
    raise ImportError("statsmodels required (pip install statsmodels)") from exc


FATIGUE_PREDICTORS = (
    "is_back_to_back",
    "is_short_rest",
    "opp_is_back_to_back",
    "opp_is_short_rest",
)


def prepare_forward_cv_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Omit season openers (``days_rest`` NaN); require nonzero ``pd_lag1`` chronology
    for the team's **previous league game**, which may be in another season string.
    """
    need = (
        "TEAM_ID",
        "SEASON_ID",
        "GAME_DATE",
        "GAME_ID",
        "point_diff",
        "days_rest",
        "IS_HOME",
        *FATIGUE_PREDICTORS,
    )
    miss = sorted(set(need) - set(df.columns))
    if miss:
        raise ValueError(f"prepare_forward_cv_panel: missing {miss}")

    d = df.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).copy()
    d["pd_lag1"] = d.groupby("TEAM_ID", sort=False)["point_diff"].shift(1)
    d["IS_HOME_int"] = d["IS_HOME"].astype(int)
    playable = d[d["days_rest"].notna() & d["pd_lag1"].notna()].copy()
    playable["GAME_DATE"] = pd.to_datetime(playable["GAME_DATE"])
    return playable


def season_forward_folds(playable: pd.DataFrame) -> list[tuple[list[Any], Any]]:
    """``(sorted_train_SEASON_ID_list, holdout_SEASON_ID)`` chronological."""
    seasons = sorted(playable["SEASON_ID"].dropna().unique().tolist())
    return [(seasons[: i + 1], seasons[i + 1]) for i in range(len(seasons) - 1)]


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    e = y - pred
    return float(np.sqrt(np.mean(e**2)))


def _mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


FULL_FORMULA = (
    "point_diff ~ pd_lag1 + is_back_to_back + is_short_rest + "
    "opp_is_back_to_back + opp_is_short_rest + IS_HOME_int"
)
REDUCED_FORMULA = "point_diff ~ pd_lag1 + IS_HOME_int"


def run_forward_season_cv(playable: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    fold_rows: list[dict[str, Any]] = []
    y_blocks: list[np.ndarray] = []
    pred_full_blocks: list[np.ndarray] = []
    pred_reduced_blocks: list[np.ndarray] = []

    folds = season_forward_folds(playable)

    for train_seasons, test_sid in folds:
        train = playable[playable["SEASON_ID"].isin(train_seasons)]
        test = playable[playable["SEASON_ID"] == test_sid]

        mf = smf.ols(FULL_FORMULA, data=train).fit()
        mr = smf.ols(REDUCED_FORMULA, data=train).fit()

        pf = mf.predict(test)
        pr = mr.predict(test)
        y_t = np.asarray(test["point_diff"].values, dtype=float)

        fold_rows.append(
            {
                "train_seasons": ",".join(str(s) for s in train_seasons),
                "test_season": test_sid,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "rmse_full": _rmse(y_t, np.asarray(pf, dtype=float)),
                "rmse_reduced": _rmse(y_t, np.asarray(pr, dtype=float)),
                "mae_full": _mae(y_t, np.asarray(pf, dtype=float)),
                "mae_reduced": _mae(y_t, np.asarray(pr, dtype=float)),
            }
        )
        y_blocks.append(y_t)
        pred_full_blocks.append(np.asarray(pf, dtype=float))
        pred_reduced_blocks.append(np.asarray(pr, dtype=float))

    per_fold = pd.DataFrame(fold_rows)

    if y_blocks:
        y_concat = np.concatenate(y_blocks)
        pooled = pd.Series(
            {
                "folds_concatenated_rmse_full": _rmse(
                    y_concat, np.concatenate(pred_full_blocks)
                ),
                "folds_concatenated_rmse_reduced": _rmse(
                    y_concat, np.concatenate(pred_reduced_blocks)
                ),
                "folds_concatenated_mae_full": _mae(
                    y_concat, np.concatenate(pred_full_blocks)
                ),
                "folds_concatenated_mae_reduced": _mae(
                    y_concat, np.concatenate(pred_reduced_blocks)
                ),
                "n_test_rows_total": len(y_concat),
            }
        )
    else:
        pooled = pd.Series(dtype=float)

    return per_fold, pooled
