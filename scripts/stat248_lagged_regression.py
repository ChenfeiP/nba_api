"""
STAT 248 — Method 1: team-game lagged OLS with rest covariates.

Each outcome is regressed on one within-season lag of itself, fatigue indicators,
opponent fatigue, home court, season effects, with cluster–robust SEs by team.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper


METHOD1_BASE_TERMS = (
    "is_back_to_back + is_short_rest + "
    "opp_is_back_to_back + opp_is_short_rest + IS_HOME_int + C(season)"
)


def prepare_method1_sample(
    df: pd.DataFrame,
    *,
    outcomes: tuple[str, ...] = ("point_diff", "EFG_PCT", "TOV"),
) -> pd.DataFrame:
    """
    Canonical sort implied: caller should pass df already sorted (e.g. `sort_panel`).
    Drops season openers (no lag) and any row missing rest / fatigue columns.
    """
    need = (
        "TEAM_ID",
        "SEASON_ID",
        "GAME_DATE",
        "GAME_ID",
        "days_rest",
        "is_back_to_back",
        "is_short_rest",
        "opp_is_back_to_back",
        "opp_is_short_rest",
        "IS_HOME",
        *outcomes,
    )
    missing = sorted(set(need) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df.copy()
    for col in outcomes:
        out[f"{col}_lag1"] = (
            out.groupby(["TEAM_ID", "SEASON_ID"], sort=False)[col].shift(1)
        )

    lag_cols = [f"{c}_lag1" for c in outcomes]
    subset = out.dropna(
        subset=list(lag_cols)
        + [
            "days_rest",
            "is_back_to_back",
            "is_short_rest",
            "opp_is_back_to_back",
            "opp_is_short_rest",
            "IS_HOME",
        ]
    ).copy()
    subset["IS_HOME_int"] = subset["IS_HOME"].astype(int)
    subset["season"] = subset["SEASON_ID"].astype("category")
    return subset


def fit_lagged_team_models(
    subset: pd.DataFrame,
    *,
    outcomes: tuple[str, ...] = ("point_diff", "EFG_PCT", "TOV"),
    cluster_column: str = "TEAM_ID",
) -> dict[str, RegressionResultsWrapper]:
    """OLS with HC1-style cluster covariance by ``cluster_column`` (default team)."""
    results: dict[str, RegressionResultsWrapper] = {}
    groups = subset[cluster_column].astype(int)
    for y in outcomes:
        lag_term = f"{y}_lag1"
        formula = f"{y} ~ {lag_term} + " + METHOD1_BASE_TERMS
        model = smf.ols(formula, data=subset)
        fitted = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
        results[y] = fitted
    return results


def stack_regression_tables(
    models: dict[str, RegressionResultsWrapper],
) -> pd.DataFrame:
    layers = {}
    for y, res in models.items():
        layers[y] = pd.DataFrame(
            {"coef": res.params, "cluster_se": res.bse, "pvalue": res.pvalues},
        )
    return pd.concat(layers, axis=1)
