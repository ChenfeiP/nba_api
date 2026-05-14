#!/usr/bin/env python3
"""
Build a team-game panel from NBA.com via nba_api: outcomes, merge keys, rest,
rolling team stats, and home-minus-away rolling differences.

Rolling features use only *prior* games (no leakage) within each season.

Requires Python 3.10+ (same as nba_api). Install the package from this repo, then run
(do not paste the comment into the shell on the same line as pip):

    pip install pandas
    pip install -e .

Example:

    python scripts/build_nba_team_game_dataset.py --seasons 2023-24 -o nba_team_games.csv
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections.abc import Iterable

import pandas as pd

from nba_api.stats.endpoints import leaguegamefinder as lgf


def _normalize_abbr(token: str) -> str:
    """Uppercase abbreviation; strip trailing 'vs.' noise."""
    token = token.strip().upper().replace(".", "")
    return token.split()[0] if token else token


def parse_matchup(matchup: str, team_abbr: str) -> tuple[bool, str]:
    """
    Return (is_home, opponent_abbr) for a LeagueGameFinder MATCHUP string.

    Examples: 'LAL @ BOS' (away LAL), 'LAL vs. BOS' / 'LAL vs BOS' (home LAL).
    """
    m = matchup.strip()
    team_abbr = team_abbr.upper()
    if " @ " in m:
        left, right = m.split(" @ ", 1)
        away = _normalize_abbr(left)
        home = _normalize_abbr(right)
        is_home = team_abbr == home
        opp = home if not is_home else away
        return is_home, opp
    parts = re.split(r"\s+vs\.?\s+", m, flags=re.IGNORECASE)
    if len(parts) == 2:
        home = _normalize_abbr(parts[0])
        away = _normalize_abbr(parts[1])
        is_home = team_abbr == home
        opp = away if is_home else home
        return is_home, opp
    return False, ""


def fetch_season_games(season: str, timeout: float) -> pd.DataFrame:
    """Fetch all regular-season team games for one league season string."""
    resp = lgf.LeagueGameFinder(
        player_or_team_abbreviation="T",
        season_nullable=season,
        season_type_nullable="Regular Season",
        timeout=timeout,
    )
    return resp.league_game_finder_results.get_data_frame()


def fetch_league_games(
    seasons: Iterable[str], *, season_delay_s: float, timeout: float
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in seasons:
        frames.append(fetch_season_games(season, timeout=timeout))
        time.sleep(season_delay_s)
    df = pd.concat(frames, ignore_index=True)
    df.drop_duplicates(subset=["GAME_ID", "TEAM_ID"], inplace=True)
    return df


def attach_opponent(df: pd.DataFrame) -> pd.DataFrame:
    """Add OPPONENT_ID, OPP_PTS, pts_allowed (same as OPP_PTS)."""
    opp = df[["GAME_ID", "TEAM_ID", "PTS"]].rename(
        columns={"TEAM_ID": "OPPONENT_ID", "PTS": "OPP_PTS"}
    )
    out = df.merge(opp, on="GAME_ID", how="left")
    out = out[out["TEAM_ID"] != out["OPPONENT_ID"]]
    out["PTS_ALLOWED"] = out["OPP_PTS"]
    return out


def prior_mean_last_n_by_mask(
    subset_mask: list[bool] | tuple[bool, ...],
    values: list[float],
    window: int,
) -> list[float]:
    """For each row i, mean of last `window` values from rows j < i with subset_mask[j]."""
    n = len(values)
    out: list[float] = [float("nan")] * n
    for i in range(n):
        prior_vals = [values[j] for j in range(i) if subset_mask[j]]
        if not prior_vals:
            continue
        chunk = prior_vals[-window:]
        out[i] = sum(chunk) / len(chunk)
    return out


def add_efg_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Effective FG% per team-game: (FGM + 0.5 * FG3M) / FGA."""
    df = df.copy()
    req = {"FGM", "FGA", "FG3M"}
    missing = req - set(df.columns)
    if missing:
        return df
    fga = df["FGA"].astype(float)
    df["EFG_PCT"] = (df["FGM"].astype(float) + 0.5 * df["FG3M"].astype(float)) / fga.where(
        fga > 0
    )
    return df


def add_rest_and_b2b(df: pd.DataFrame) -> pd.DataFrame:
    """Add days_rest, back-to-back, short-rest (0–1 days), and opponent counterparts."""
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df.sort_values(["TEAM_ID", "SEASON_ID", "GAME_DATE", "GAME_ID"], inplace=True)
    gaps = df.groupby(["TEAM_ID", "SEASON_ID"], sort=False)["GAME_DATE"].diff().dt.days
    # days between games: gap - 1; first game of season -> NaN
    df["days_rest"] = gaps - 1
    df["is_back_to_back"] = ((df["days_rest"] == 0) & df["days_rest"].notna()).astype(int)
    df["is_short_rest"] = ((df["days_rest"] <= 1) & df["days_rest"].notna()).astype(int)

    opp_rest = df[["GAME_ID", "TEAM_ID", "days_rest"]].rename(
        columns={"TEAM_ID": "OPPONENT_ID", "days_rest": "opp_days_rest"}
    )
    df = df.merge(opp_rest, on=["GAME_ID", "OPPONENT_ID"], how="left")
    df["opp_is_back_to_back"] = (
        ((df["opp_days_rest"] == 0) & df["opp_days_rest"].notna()).astype(int)
    )
    df["opp_is_short_rest"] = (
        ((df["opp_days_rest"] <= 1) & df["opp_days_rest"].notna()).astype(int)
    )
    return df


def add_overall_rolling(
    team_df: pd.DataFrame,
    window: int,
    col: str,
    out_name: str,
) -> pd.Series:
    """Rolling mean of `col` over prior `window` games (any location)."""
    return team_df[col].shift(1).rolling(window, min_periods=1).mean().rename(out_name)


def build_team_features(
    df: pd.DataFrame,
    *,
    rolling_window: int,
) -> pd.DataFrame:
    """Per-team chronological features: overall + home/away prior rollings."""
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df.sort_values(["TEAM_ID", "SEASON_ID", "GAME_DATE", "GAME_ID"], inplace=True)

    parsed = df.apply(
        lambda r: parse_matchup(r["MATCHUP"], r["TEAM_ABBREVIATION"]), axis=1
    )
    df["IS_HOME"] = [p[0] for p in parsed]
    df["OPP_ABBREVIATION"] = [p[1] for p in parsed]
    df["IS_WIN"] = (df["WL"] == "W").astype(int)
    df["outcome_win"] = df["IS_WIN"]

    df["point_diff"] = df["PTS"] - df["PTS_ALLOWED"]

    metric_cols = [
        "IS_WIN",
        "PTS",
        "PTS_ALLOWED",
        "point_diff",
        "FG_PCT",
        "FG3_PCT",
        "EFG_PCT",
        "FT_PCT",
        "REB",
        "AST",
        "TOV",
        "STL",
        "BLK",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    out_frames: list[pd.DataFrame] = []
    for _, g in df.groupby(["TEAM_ID", "SEASON_ID"], sort=False):
        g = g.sort_values(["GAME_DATE", "GAME_ID"])
        idx = g.index
        home_mask = tuple(g["IS_HOME"].tolist())
        away_mask = tuple(not x for x in home_mask)

        roll = {}
        for c in metric_cols:
            if c == "IS_WIN":
                name = f"roll_last{rolling_window}_win_pct"
            elif c == "PTS_ALLOWED":
                name = f"roll_last{rolling_window}_pts_allowed"
            elif c == "point_diff":
                name = f"roll_last{rolling_window}_point_diff"
            else:
                name = f"roll_last{rolling_window}_{c.lower()}"

            roll[name] = add_overall_rolling(g, rolling_window, c, name)

        for loc_name, mask in (("home", home_mask), ("away", away_mask)):
            for c in metric_cols:
                v = g[c].astype(float).tolist()
                if c == "IS_WIN":
                    suffix = "win_pct"
                elif c == "PTS_ALLOWED":
                    suffix = "pts_allowed"
                elif c == "point_diff":
                    suffix = "point_diff"
                else:
                    suffix = c.lower()
                col_name = f"roll_{loc_name}_last{rolling_window}_{suffix}"
                roll[col_name] = pd.Series(
                    prior_mean_last_n_by_mask(mask, v, rolling_window),
                    index=idx,
                )

        for c in metric_cols:
            if c == "IS_WIN":
                short = "win_pct"
            elif c == "PTS_ALLOWED":
                short = "pts_allowed"
            elif c == "point_diff":
                short = "point_diff"
            else:
                short = c.lower()
            h = f"roll_home_last{rolling_window}_{short}"
            a = f"roll_away_last{rolling_window}_{short}"
            roll[f"diff_last{rolling_window}_{short}_home_minus_away"] = (
                roll[h] - roll[a]
            )

        team_out = pd.DataFrame(roll, index=idx)
        out_frames.append(team_out)

    rolled = pd.concat(out_frames).sort_index()
    result = df.join(rolled)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download NBA team-game logs and engineer rolling features.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["2023-24"],
        help="One or more NBA season strings, e.g. 2023-24",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=5,
        help="Number of prior games for rolling stats (default: 5)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="nba_team_game_features.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--season-delay",
        type=float,
        default=0.6,
        help="Seconds to sleep between season API calls (rate limiting)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout seconds per request",
    )
    args = parser.parse_args(argv)

    seasons = args.seasons
    print(f"Fetching seasons: {seasons}", file=sys.stderr)
    raw = fetch_league_games(
        seasons, season_delay_s=args.season_delay, timeout=args.timeout
    )
    print(f"Rows: {len(raw)}", file=sys.stderr)

    raw = attach_opponent(raw)
    raw = add_efg_pct(raw)
    raw = add_rest_and_b2b(raw)
    panel = build_team_features(raw, rolling_window=args.rolling_window)

    # Stable column order: ids + outcome + context + rollings
    id_cols = [
        "SEASON_ID",
        "GAME_ID",
        "GAME_DATE",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "OPPONENT_ID",
        "OPP_ABBREVIATION",
        "IS_HOME",
        "MATCHUP",
        "WL",
        "outcome_win",
    ]
    ctx_cols = [
        "days_rest",
        "is_back_to_back",
        "is_short_rest",
        "opp_days_rest",
        "opp_is_back_to_back",
        "opp_is_short_rest",
    ]
    rest_cols = [c for c in panel.columns if c not in id_cols + ctx_cols]
    rest_cols.sort()
    ordered = [c for c in id_cols + ctx_cols if c in panel.columns] + rest_cols
    panel = panel[ordered]

    panel.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(panel)} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
