"""
Team-game panel: ordering, duplication, and time-series index checks.

Use after building CSV with scripts/build_nba_team_game_dataset.py or when
loading saved output.
"""

from __future__ import annotations

import pandas as pd

PANEL_SORT_KEYS = ("TEAM_ID", "SEASON_ID", "GAME_DATE", "GAME_ID")


def sort_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy canonically sorted for team-game rows."""
    out = df.copy()
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"])
    out.sort_values(list(PANEL_SORT_KEYS), kind="mergesort", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def validate_team_game_panel(
    df: pd.DataFrame,
    *,
    require_keys: bool = True,
    sample_series: bool = False,
    series_sample_n: int = 3,
) -> dict[str, object]:
    """
    Run assertions and return a small summary dict.

    Raises
    ------
    ValueError
        If required columns are missing, duplicate keys exist, or dates within
        a team-season are not non-decreasing.
    """
    if require_keys:
        need = {"TEAM_ID", "SEASON_ID", "GAME_ID", "GAME_DATE"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    dup = df.duplicated(subset=("GAME_ID", "TEAM_ID"), keep=False)
    if dup.any():
        n = dup.sum()
        raise ValueError(f"Duplicate (GAME_ID, TEAM_ID) rows: {int(n)} rows flagged")

    d = sort_panel(df)

    bad_groups = []
    for (tid, sid), g in d.groupby(["TEAM_ID", "SEASON_ID"], sort=False):
        gd = g["GAME_DATE"]
        if not gd.is_monotonic_increasing:
            bad_groups.append((tid, sid))
            if len(bad_groups) >= 5:
                break
    if bad_groups:
        raise ValueError(
            "GAME_DATE is not monotone within TEAM_ID × SEASON_ID for groups: "
            f"{bad_groups}"
        )

    n_rows = len(d)
    n_series = d.groupby(["TEAM_ID", "SEASON_ID"], sort=False).ngroups
    n_games_unique = d["GAME_ID"].nunique()
    seasons = sorted(d["SEASON_ID"].dropna().unique().tolist())

    rows_na_rest = (
        int(d["days_rest"].isna().sum()) if "days_rest" in d.columns else None
    )
    b2b_rate = (
        float(d["is_back_to_back"].mean())
        if "is_back_to_back" in d.columns
        else None
    )

    summary: dict[str, object] = {
        "n_rows": int(n_rows),
        "n_distinct_team_season": int(n_series),
        "n_distinct_GAME_ID": int(n_games_unique),
        "SEASON_ID_values": seasons,
        "rows_with_na_days_rest": rows_na_rest,
        "mean_is_back_to_back": b2b_rate,
    }

    if sample_series and series_sample_n > 0:
        keys_df = d[["TEAM_ID", "SEASON_ID"]].drop_duplicates().head(series_sample_n)
        sample = []
        for _, row in keys_df.iterrows():
            tid, sid = int(row["TEAM_ID"]), row["SEASON_ID"]
            cols = [
                c
                for c in (
                    "GAME_DATE",
                    "GAME_ID",
                    "days_rest",
                    "is_back_to_back",
                    "is_short_rest",
                    "point_diff",
                )
                if c in d.columns
            ]
            sub = d[(d["TEAM_ID"] == tid) & (d["SEASON_ID"] == sid)].iloc[:5][cols]
            sample.append({"TEAM_ID": tid, "SEASON_ID": sid, "head": sub})
        summary["series_sample_heads"] = sample

    return summary
