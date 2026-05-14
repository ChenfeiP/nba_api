"""
STAT 248 — Method 2: ARIMAX on point differential per TEAM_ID × SEASON_ID.

Pooling all teams into a single chronological series would splice unrelated AR
structures; fit compact ``SARIMAX`` models within each ~82-game team-season,
summarise BIC / coefficients across strata, then diagnose one best-BIC streak.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import os
from typing import Any

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "`statsmodels.tsa.statespace.sarimax` is required (pip install statsmodels)"
    ) from exc


FATIGUE_COLS = (
    "is_back_to_back",
    "is_short_rest",
    "opp_is_back_to_back",
    "opp_is_short_rest",
)


@dataclass
class SubgroupFitMeta:
    team_id: int
    season_id: int | str
    n_obs: int
    order_full: tuple[int, int, int]
    order_reduced: tuple[int, int, int]
    bic_full: float | None
    bic_reduced: float | None
    bic_delta: float | None  # full - restricted (negative if full better)


def prepare_point_diff_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Omit season openers (`days_rest` NaN) so fatigue regressors behave like Methods 1.
    Ensures ``IS_HOME_int`` exists if only ``IS_HOME`` is stored.
    """
    need_common = (
        "TEAM_ID",
        "SEASON_ID",
        "GAME_DATE",
        "GAME_ID",
        "days_rest",
        *FATIGUE_COLS,
        "IS_HOME",
        "point_diff",
    )
    miss = sorted(set(need_common) - set(df.columns))
    if miss:
        raise ValueError(f"prepare_point_diff_series: missing columns {miss}")
    subset = df[df["days_rest"].notna()].copy()
    if "IS_HOME_int" not in subset.columns:
        subset["IS_HOME_int"] = subset["IS_HOME"].astype(int)
    return subset


def _safe_fit(
    y: np.ndarray,
    exog: pd.DataFrame,
    order: tuple[int, int, int],
    *,
    maxiter: int = 150,
) -> Any:
    mod = SARIMAX(
        y.astype(float),
        exog=exog.astype(float),
        order=order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return mod.fit(
        disp=False, maxiter=maxiter, warn_convergence=False, method="lbfgs"
    )


def _design_from_group(g: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Chronologically ordered ``point_diff`` and exogenous matrices for one team-season."""
    g = g.sort_values(["GAME_DATE", "GAME_ID"]).copy()
    y_arr = np.asarray(g["point_diff"].values, dtype=float)
    Xf_vals = pd.DataFrame(
        {c: pd.to_numeric(g[c], errors="coerce") for c in FATIGUE_COLS}
        | {"IS_HOME_int": pd.to_numeric(g["IS_HOME_int"], errors="coerce")}
    ).fillna(0.0)
    X_full = pd.concat(
        [
            pd.Series(1.0, index=g.index, name="const"),
            Xf_vals,
        ],
        axis=1,
    )
    Xr = pd.concat(
        [
            pd.Series(1.0, index=g.index, name="const"),
            Xf_vals["IS_HOME_int"].rename("IS_HOME_int"),
        ],
        axis=1,
    )
    return y_arr, X_full, Xr


def _arimax_subgroup_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Picklable one-stratum fit (for ``ProcessPoolExecutor``).
    Returns meta dict, flattened coef row, and BIC values only (no Results object).
    """
    y_arr = np.asarray(payload["y"], dtype=float)
    X_full = pd.DataFrame(
        payload["X_full_vals"], columns=list(payload["X_full_cols"]), dtype=float
    )
    Xr = pd.DataFrame(
        payload["X_red_vals"], columns=list(payload["X_red_cols"]), dtype=float
    )
    orders = tuple(tuple(o) for o in payload["orders"])
    fit_maxiter = int(payload["fit_maxiter"])
    tid = int(payload["team_id"])
    sid = payload["season_id"]

    res_full, order_f = fit_sarimax_group(
        y_arr, X_full, orders=orders, fit_maxiter=fit_maxiter
    )
    res_red, order_r = fit_sarimax_group(
        y_arr, Xr, orders=orders, fit_maxiter=fit_maxiter
    )

    bic_full = float(res_full.bic) if res_full is not None else None
    bic_red = float(res_red.bic) if res_red is not None else None
    bic_delta = (bic_full - bic_red) if bic_full is not None and bic_red is not None else None

    meta = asdict(
        SubgroupFitMeta(
            team_id=tid,
            season_id=sid,
            n_obs=int(len(y_arr)),
            order_full=order_f,
            order_reduced=order_r,
            bic_full=bic_full,
            bic_reduced=bic_red,
            bic_delta=bic_delta,
        )
    )

    row: dict[str, Any] = {
        "TEAM_ID": tid,
        "SEASON_ID": sid,
        "order_full": repr(order_f),
        "bic_full": bic_full,
        "bic_reduced": bic_red,
        "bic_delta_full_minus_reduced": bic_delta,
    }
    if res_full is not None:
        for lab, val in res_full.params.items():
            row[f"coef__{str(lab)}"] = float(val)

    return {"meta": meta, "row": row, "bic_full": bic_full}


def fit_sarimax_group(
    y: np.ndarray,
    exog: pd.DataFrame,
    *,
    orders: tuple[tuple[int, int, int], ...] = ((1, 0, 0),),
    fit_maxiter: int = 150,
) -> tuple[Any | None, tuple[int, int, int]]:
    for order in orders:
        try:
            res = _safe_fit(y, exog, order, maxiter=fit_maxiter)
            if res is None:
                continue
            if hasattr(res, "llf") and np.isfinite(res.llf):
                return res, order
        except Exception:
            continue
    return None, orders[-1]


def fit_panel_arimax_parallel(
    panel: pd.DataFrame,
    *,
    min_obs: int = 30,
    sarimax_orders: tuple[tuple[int, int, int], ...] = ((1, 0, 0),),
    fit_maxiter: int = 150,
    max_workers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Same outputs as ``fit_panel_arimax``, but strata are fitted in parallel.
    Diagnostics refit uses one extra MLE solve on the best-BIC team-season.
    """
    if max_workers is None:
        cpus = os.cpu_count() or 4
        max_workers = max(1, min(cpus - 1, 12))

    orders_serializable = [list(o) for o in sarimax_orders]
    jobs: list[dict[str, Any]] = []

    for (tid, sid), g0 in panel.groupby(["TEAM_ID", "SEASON_ID"], sort=False):
        g = g0
        if len(g) < min_obs:
            continue
        y_arr, X_full, Xr = _design_from_group(g)
        jobs.append(
            {
                "team_id": int(tid),
                "season_id": int(sid) if isinstance(sid, np.integer) else sid,
                "y": y_arr,
                "X_full_vals": X_full.to_numpy(dtype=float, copy=True),
                "X_full_cols": list(X_full.columns),
                "X_red_vals": Xr.to_numpy(dtype=float, copy=True),
                "X_red_cols": list(Xr.columns),
                "orders": orders_serializable,
                "fit_maxiter": fit_maxiter,
            }
        )

    metas: list[dict[str, Any]] = []
    coeff_rows: list[dict[str, Any]] = []
    bic_best: tuple[float, int, Any, Any] | None = None  # bic, tid, sid, _

    # macOS/Python: `fork` pitfalls; rely on spawn + top-level worker
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_arimax_subgroup_worker, j): j for j in jobs}
        for fu in as_completed(futs):
            out = fu.result()
            metas.append(out["meta"])
            coeff_rows.append(out["row"])
            bf = out["bic_full"]
            jid = futs[fu]
            if bf is not None and np.isfinite(bf):
                if bic_best is None or bf < bic_best[0]:
                    bic_best = (bf, jid["team_id"], jid["season_id"], None)

    meta_df = (
        pd.DataFrame(metas)
        .rename(columns={"team_id": "TEAM_ID", "season_id": "SEASON_ID"})
        .sort_values(["TEAM_ID", "SEASON_ID"], ignore_index=True)
    )
    coef_df = pd.DataFrame(coeff_rows).sort_values(
        ["TEAM_ID", "SEASON_ID"], ignore_index=True
    )

    diag: dict[str, Any] = {}
    if bic_best is not None:
        bf, bt, bs, _ = bic_best
        refit = pick_group_refit(
            panel,
            team_id=int(bt),
            season_id=bs,
            sarimax_orders=sarimax_orders,
            fit_maxiter=fit_maxiter,
        )
        full_res = refit.get("full")
        if full_res is not None:
            _, Xfull, _ = _design_from_group(
                panel[(panel["TEAM_ID"] == bt) & (panel["SEASON_ID"] == bs)]
            )
            diag = {
                "TEAM_ID": bt,
                "SEASON_ID": bs,
                "bic": bf,
                "result": full_res,
                "X_full": Xfull,
                "y": np.asarray(
                    panel[(panel["TEAM_ID"] == bt) & (panel["SEASON_ID"] == bs)]
                    .sort_values(["GAME_DATE", "GAME_ID"])["point_diff"],
                    dtype=float,
                ),
            }

    return meta_df, coef_df, diag


def fit_panel_arimax(
    panel: pd.DataFrame,
    *,
    min_obs: int = 30,
    sarimax_orders: tuple[tuple[int, int, int], ...] = ((1, 0, 0),),
    fit_maxiter: int = 150,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return `(meta summaries, coef table, diagnostics_pick)`."""
    metas: list[dict[str, Any]] = []
    coeff_rows: list[dict[str, Any]] = []

    bic_best: tuple[float, tuple[int, Any], Any] | None = None

    for (tid, sid), g0 in panel.groupby(["TEAM_ID", "SEASON_ID"], sort=False):
        g = g0
        if len(g) < min_obs:
            continue
        y_arr, X_full, Xr = _design_from_group(g)

        res_full, order_f = fit_sarimax_group(
            y_arr, X_full, orders=sarimax_orders, fit_maxiter=fit_maxiter
        )
        res_red, order_r = fit_sarimax_group(
            y_arr, Xr, orders=sarimax_orders, fit_maxiter=fit_maxiter
        )

        bic_full = float(res_full.bic) if res_full is not None else None
        bic_red = float(res_red.bic) if res_red is not None else None
        bic_delta = (bic_full - bic_red) if bic_full is not None and bic_red is not None else None

        metas.append(
            asdict(
                SubgroupFitMeta(
                    team_id=int(tid),
                    season_id=sid,
                    n_obs=len(y_arr),
                    order_full=order_f,
                    order_reduced=order_r,
                    bic_full=bic_full,
                    bic_reduced=bic_red,
                    bic_delta=bic_delta,
                )
            )
        )

        row: dict[str, Any] = {
            "TEAM_ID": int(tid),
            "SEASON_ID": sid,
            "order_full": repr(order_f),
            "bic_full": bic_full,
            "bic_reduced": bic_red,
            "bic_delta_full_minus_reduced": bic_delta,
        }

        if res_full is not None:
            for lab, val in res_full.params.items():
                row[f"coef__{str(lab)}"] = float(val)
            bic_val = bic_full if bic_full is not None else float("inf")
            if bic_best is None or bic_val < bic_best[0]:
                bic_best = (
                    bic_val,
                    (int(tid), sid),
                    {
                        "result": res_full,
                        "X_full": X_full,
                        "y": y_arr,
                    },
                )
        coeff_rows.append(row)

    meta_df = (
        pd.DataFrame(metas)
        .rename(columns={"team_id": "TEAM_ID", "season_id": "SEASON_ID"})
        .sort_values(["TEAM_ID", "SEASON_ID"], ignore_index=True)
    )
    coef_df = pd.DataFrame(coeff_rows).sort_values(
        ["TEAM_ID", "SEASON_ID"], ignore_index=True
    )

    diag: dict[str, Any]
    if bic_best is None:
        diag = {}
    else:
        _, (bt, bs), payload = bic_best
        diag = {"TEAM_ID": bt, "SEASON_ID": bs, "bic": bic_best[0], **payload}

    return meta_df, coef_df, diag


def pick_group_refit(
    panel: pd.DataFrame,
    *,
    team_id: int,
    season_id: int | str,
    sarimax_orders: tuple[tuple[int, int, int], ...] = ((1, 0, 0),),
    fit_maxiter: int = 150,
) -> dict[str, Any]:
    """Convenience: refit a single team-season (for pedagogical reproducibility)."""
    g = (
        panel[(panel["TEAM_ID"] == team_id) & (panel["SEASON_ID"] == season_id)]
        .sort_values(["GAME_DATE", "GAME_ID"])
        .copy()
    )
    if len(g) == 0:
        raise ValueError("empty subgroup")
    y_arr = np.asarray(g["point_diff"].values, dtype=float)
    Xf = pd.DataFrame(
        {c: pd.to_numeric(g[c], errors="coerce") for c in FATIGUE_COLS}
        | {"IS_HOME_int": pd.to_numeric(g["IS_HOME_int"], errors="coerce")}
    ).fillna(0.0)
    X_full = pd.concat([pd.Series(1.0, index=g.index, name="const"), Xf], axis=1)
    Xr = pd.concat(
        [
            pd.Series(1.0, index=g.index, name="const"),
            Xf["IS_HOME_int"].rename("IS_HOME_int"),
        ],
        axis=1,
    )
    res_full, ord_f = fit_sarimax_group(
        y_arr, X_full, orders=sarimax_orders, fit_maxiter=fit_maxiter
    )
    res_red, ord_r = fit_sarimax_group(
        y_arr, Xr, orders=sarimax_orders, fit_maxiter=fit_maxiter
    )
    return {
        "team_id": team_id,
        "season_id": season_id,
        "n": len(y_arr),
        "order_full": ord_f,
        "order_reduced": ord_r,
        "full": res_full,
        "reduced": res_red,
        "X_full": X_full,
        "X_reduced": Xr,
        "y": y_arr,
    }


def aggregate_bic_summary(meta_df: pd.DataFrame) -> pd.Series:
    """Means over converged strata (finite BIC pairs)."""
    both = meta_df.dropna(subset=["bic_full", "bic_reduced"]).copy()
    return pd.Series(
        {
            "n_strata": len(both),
            "bic_full_mean": both["bic_full"].mean(),
            "bic_reduced_mean": both["bic_reduced"].mean(),
            "bic_delta_mean": both["bic_delta"].mean(),
            "bic_full_better_frac": float((both["bic_delta"] < 0).mean()),
        },
        dtype=object,
    )
