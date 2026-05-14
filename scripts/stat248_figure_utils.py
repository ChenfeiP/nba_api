"""Shared plotting + table helpers for STAT 248 runners and the final Jupyter report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def slide_style_apply() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )


def coef_digest_wide(coef_df: pd.DataFrame) -> pd.Series:
    prefix = "coef__"
    cols = [c for c in coef_df.columns if c.startswith(prefix)]
    if not cols:
        return pd.Series(dtype=float)
    stacked = coef_df[cols].melt(var_name="term", value_name="value").dropna(
        subset=["value"]
    )
    stacked["term"] = stacked["term"].str[len(prefix) :]
    return stacked.groupby("term", sort=False).value.median()


def plot_method2_bic_means(bic_s: pd.Series, out: Path) -> None:
    slide_style_apply()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    cats = ["Full\n(fatigue+home)", "Reduced\n(home only)"]
    vals = [float(bic_s["bic_full_mean"]), float(bic_s["bic_reduced_mean"])]
    colors = ["#2c5282", "#a8b8c8"]
    ax.bar(cats, vals, color=colors, width=0.55, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Mean BIC (lower is better)")
    ax.set_title("Method 2: ARIMAX — average BIC across team-seasons")
    ymax = max(vals) * 1.05
    ax.set_ylim(0, ymax)
    for i, v in enumerate(vals):
        ax.text(i, v + ymax * 0.02, f"{v:.1f}", ha="center", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_method2_bic_win_frac(bic_s: pd.Series, out: Path) -> None:
    slide_style_apply()
    frac = float(bic_s["bic_full_better_frac"])
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.barh([0], [frac], height=0.35, color="#276749", edgecolor="black")
    ax.set_xlim(0, 1)
    ax.set_yticks([0])
    ax.set_yticklabels(["Share with lower BIC\n(full vs home-only)"])
    ax.set_xlabel("Fraction of team-seasons")
    ax.set_title("How often does adding fatigue predictors improve BIC?")
    ax.text(
        min(frac + 0.02, 0.92),
        0,
        f"{frac:.0%}",
        va="center",
        fontsize=18,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_method2_median_exog_coefs(digest: pd.Series, out: Path) -> None:
    slide_style_apply()
    order = [
        ("is_back_to_back", "Own B2B"),
        ("is_short_rest", "Own short rest (≤1 d)"),
        ("opp_is_back_to_back", "Opponent B2B"),
        ("opp_is_short_rest", "Opponent short rest"),
        ("IS_HOME_int", "Home court"),
    ]
    vals: list[float] = []
    labels: list[str] = []
    idx = digest.index.astype(str)
    for key, lab in order:
        sel = digest[idx == key]
        if sel.empty:
            sel = digest[idx.str.endswith(key)]
        if sel.empty:
            continue
        v = float(sel.median())
        if np.isfinite(v):
            vals.append(v)
            labels.append(lab)
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(vals))
    colors = ["#c53030" if v < 0 else "#2b6cb0" for v in vals]
    ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel(
        "Median coef. across strata (ARIMAX(1,0,0); outcome: point differential)"
    )
    ax.set_title("Pooled median exogenous coefficients")
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_method2_sarimax_diagnostics(result, diag: dict, out: Path) -> None:
    slide_style_apply()
    fig = result.plot_diagnostics(figsize=(12.5, 8))
    fig.suptitle(
        (
            "ARIMAX residuals (best-BIC team-season: TEAM "
            f"{diag['TEAM_ID']} · SEASON {diag['SEASON_ID']})"
        ),
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_method3_cv(per_fold: pd.DataFrame, out_dir: Path) -> None:
    slide_style_apply()
    x = np.arange(len(per_fold))
    w = 0.34
    labs = [f"H.out\n{s}" for s in per_fold["test_season"].astype(str)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].bar(
        x - w / 2,
        per_fold["rmse_full"],
        width=w,
        label="With fatigue",
        color="#2c5282",
        edgecolor="black",
        linewidth=0.5,
    )
    axes[0].bar(
        x + w / 2,
        per_fold["rmse_reduced"],
        width=w,
        label="Lag + home only",
        color="#cbd5e1",
        edgecolor="black",
        linewidth=0.5,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labs, fontsize=12)
    axes[0].set_ylabel("RMSE (points)")
    axes[0].set_title("Holdout RMSE by fold")
    axes[0].legend(loc="upper right")

    axes[1].bar(
        x - w / 2,
        per_fold["mae_full"],
        width=w,
        label="With fatigue",
        color="#276749",
        edgecolor="black",
        linewidth=0.5,
    )
    axes[1].bar(
        x + w / 2,
        per_fold["mae_reduced"],
        width=w,
        label="Lag + home only",
        color="#c6f6d5",
        edgecolor="black",
        linewidth=0.5,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labs, fontsize=12)
    axes[1].set_ylabel("MAE (points)")
    axes[1].set_title("Holdout MAE by fold")
    axes[1].legend(loc="upper right")
    plt.suptitle(
        "Method 3 · expanding-window CV (train earlier seasons → test next)",
        y=1.05,
        fontsize=13,
    )
    plt.tight_layout()
    fig.savefig(out_dir / "slides_method3_cv_rmse_mae_by_fold.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    slide_style_apply()
    delta = per_fold["rmse_reduced"] - per_fold["rmse_full"]
    fig2, ax = plt.subplots(figsize=(6.5, 4))
    cols = ["#276749" if d > 0 else "#9b2c2c" for d in delta]
    ax.bar(np.arange(len(delta)), delta, color=cols, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(np.arange(len(delta)))
    ax.set_xticklabels([str(s) for s in per_fold["test_season"].astype(str)])
    ax.set_xlabel("Holdout season")
    ax.set_ylabel("RMSE(no fatigue) − RMSE(full)")
    ax.set_title("Positive ⇒ fatigue regressors reduce out-of-sample RMSE")
    plt.tight_layout()
    fig2.savefig(out_dir / "slides_method3_cv_rmse_delta.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)
