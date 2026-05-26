"""Prompts 8-9 — Domain sub-scores (USD/Oro/MBS) and routing engine."""

import numpy as np
import pandas as pd
from src.data_loader import PROJECT_ROOT
from src.features import (
    ROUTING_PATH, SPREADS_CLEAN_PATH,
    ROUTING_DOMAINS, RULE_BASED_LEVELS,
)

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

# Sub-score sign weights (Prompt 8)
SUBSCORE_USD = {
    "libor_3m_spread_chg4w": +1,
    "dxy_chg4w": +1,
    "vrp": +1,
    "us_10y_diff_chg4w": -1,
    "usa_world_relative": +1,
}

SUBSCORE_ORO = {
    "real_yield_proxy_chg4w": -1,
    "dxy_chg4w": -1,
    "jpy_strength": +1,
    "equity_bond_corr_13w": +1,
    "gold_oil_ratio_chg4w": +1,
}


def compute_subscore_zscore(triggers, dev_triggers, weights):
    """Signed z-score sub-score: mean of (sign * z-score) per trigger.
    mu, sigma estimated on dev_triggers only."""
    zscores = pd.DataFrame(index=triggers.index)
    for feat, sign in weights.items():
        if feat not in triggers.columns:
            continue
        mu = dev_triggers[feat].mean()
        sigma = dev_triggers[feat].std()
        if sigma == 0:
            sigma = 1.0
        zscores[feat] = sign * (triggers[feat] - mu) / sigma
    return zscores.mean(axis=1)


def compute_mbs_subscore(triggers, dev_triggers, spreads=None):
    """MBS sub-score: binary rule-based."""
    active = (
        (triggers["vix_level"].between(20, 28)) &
        (triggers["us_term_10y_2y_level"] > 0) &
        (triggers["mxus_drawdown_52w"].between(-0.12, -0.05))
    ).astype(int)

    # Blocking conditions
    p90_libor = dev_triggers["libor_3m_spread_level"].quantile(0.90)

    blocked = (
        (triggers["vix_level"] > 30) |
        (triggers["libor_3m_spread_level"] > p90_libor)
    )

    # hy_ig_spread_chg4w from spreads_clean
    if spreads is not None and "hy_ig_spread_chg4w" in spreads.columns:
        chg = spreads["hy_ig_spread_chg4w"].reindex(triggers.index)
        blocked = blocked | (chg > 0.005)

    mbs = active.copy()
    mbs[blocked] = 0
    return mbs


def compute_all_subscores(wf_routing, wf_models):
    """Compute USD, Oro, MBS sub-scores for all walk-forward data."""
    triggers_dev = wf_routing["development"]
    triggers_test = wf_routing["test_holdout"]

    spreads = pd.read_parquet(SPREADS_CLEAN_PATH)

    # Compute on test holdout
    sub_usd = compute_subscore_zscore(triggers_test, triggers_dev, SUBSCORE_USD)
    sub_oro = compute_subscore_zscore(triggers_test, triggers_dev, SUBSCORE_ORO)
    sub_mbs = compute_mbs_subscore(triggers_test, triggers_dev, spreads)

    subscores = pd.DataFrame({
        "sub_usd": sub_usd,
        "sub_oro": sub_oro,
        "sub_mbs": sub_mbs,
        "dxy_chg4w": triggers_test["dxy_chg4w"],
    }, index=triggers_test.index)

    # Also compute per fold for CV
    fold_subscores = []
    for fold in wf_routing["cv_folds"]:
        fold_triggers = fold["val"]
        fold_train = fold["train"]
        fs = pd.DataFrame({
            "sub_usd": compute_subscore_zscore(fold_triggers, fold_train, SUBSCORE_USD),
            "sub_oro": compute_subscore_zscore(fold_triggers, fold_train, SUBSCORE_ORO),
            "sub_mbs": compute_mbs_subscore(fold_triggers, fold_train, spreads),
            "dxy_chg4w": fold_triggers["dxy_chg4w"],
        }, index=fold_triggers.index)
        fold_subscores.append({"fold_id": fold["fold_id"], "subscores": fs})

    return subscores, fold_subscores


# --- Prompt 9: Routing Engine ---

def route_allocation(ensemble_signal, sub_usd, sub_oro, sub_mbs, dxy_chg4w, thresholds):
    """Single-week routing decision."""
    if ensemble_signal == 0:
        return "LEVERED_EQUITY"
    if sub_usd > thresholds["usd"] and dxy_chg4w > 0:
        return "CASH_USD"
    elif sub_oro > thresholds["oro"]:
        return "GOLD"
    elif sub_mbs == 1:
        return "MBS"
    else:
        return "CASH_USD"


def route_series(ensemble_signals, subscores, thresholds):
    """Apply routing to a full time series."""
    common_idx = ensemble_signals.index.intersection(subscores.index)
    allocations = pd.Series(index=common_idx, dtype=str)
    for dt in common_idx:
        sig = int(ensemble_signals.loc[dt])
        allocations.loc[dt] = route_allocation(
            sig,
            subscores.loc[dt, "sub_usd"],
            subscores.loc[dt, "sub_oro"],
            subscores.loc[dt, "sub_mbs"],
            subscores.loc[dt, "dxy_chg4w"],
            thresholds,
        )
    return allocations


DEFAULT_THRESHOLDS = {"usd": 1.0, "oro": 1.0}
