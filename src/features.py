"""Prompts 2, 3, 3.1 — Stationarity transforms, spreads, collinearity cleanup, routing triggers."""

from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import (
    load_dataset, PROCESSED_DIR, TARGET_COL, PROJECT_ROOT
)

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FEATURES_PATH = PROCESSED_DIR / "features_stationary.parquet"
FEATURES_CLEAN_PATH = PROCESSED_DIR / "features_stationary_clean.parquet"
SPREADS_PATH = PROCESSED_DIR / "spreads.parquet"
SPREADS_CLEAN_PATH = PROCESSED_DIR / "spreads_clean.parquet"
ROUTING_PATH = PROCESSED_DIR / "routing_triggers.parquet"

HORIZON_4W = 4
REALIZED_VOL_WEEKS = 4
WEEKS_PER_YEAR = 52

# --- Prompt 2: Stationarity ---

TRANSFORM_MAP = {
    # log_return: price-like series
    "MXUS": "log_return", "MXEU": "log_return", "MXJP": "log_return",
    "MXBR": "log_return", "MXRU": "log_return", "MXIN": "log_return",
    "MXCN": "log_return",
    "DXY": "log_return", "GBP": "log_return", "JPY": "log_return",
    "XAUBGNL": "log_return", "Cl1": "log_return", "CRY": "log_return",
    "BDIY": "log_return",
    "EMUSTRUU": "log_return", "LF94TRUU": "log_return",
    "LF98TRUU": "log_return", "LG30TRUU": "log_return",
    "LMBITR": "log_return", "LP01TREU": "log_return",
    "LUACTRUU": "log_return", "LUMSTRUU": "log_return",
    # diff: yields/rates
    "GT10": "diff", "USGG2YR": "diff", "USGG30YR": "diff",
    "USGG3M": "diff", "US0001M": "diff", "EONIA": "diff",
    "GTDEM2Y": "diff", "GTDEM10Y": "diff", "GTDEM30Y": "diff",
    "GTGBP2Y": "diff", "GTGBP20Y": "diff", "GTGBP30Y": "diff",
    "GTITL2YR": "diff", "GTITL10YR": "diff", "GTITL30YR": "diff",
    "GTJPY2YR": "diff", "GTJPY10YR": "diff", "GTJPY30YR": "diff",
    # level: already stationary
    "VIX": "level", "ECSURPUS": "level", "Y": "level",
}


def make_stationary(df, transform_map=None, dropna=True):
    transform_map = transform_map or TRANSFORM_MAP
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        t = transform_map.get(col, "level")
        if t == "log_return":
            out[col] = np.log(df[col]).diff()
        elif t == "diff":
            out[col] = df[col].diff()
        else:
            out[col] = df[col]
    if dropna:
        out = out.iloc[1:]
    return out


def adf_table(df, signif=0.05, transform_map=None):
    transform_map = transform_map or TRANSFORM_MAP
    rows = []
    for col in df.columns:
        if col == TARGET_COL:
            continue
        result = adfuller(df[col].dropna(), autolag="AIC")
        rows.append({
            "feature": col,
            "adf_stat": result[0],
            "p_value": result[1],
            "used_lag": result[2],
            "n_obs": result[3],
            "crit_1pct": result[4]["1%"],
            "crit_5pct": result[4]["5%"],
            "crit_10pct": result[4]["10%"],
            "transform": transform_map.get(col, "level"),
            "stationary": result[1] < signif,
        })
    return pd.DataFrame(rows).set_index("feature")


# --- Prompt 3: Spreads ---

def build_spreads(df_raw, horizon=HORIZON_4W):
    """Build cross-asset spread features from raw-level data."""
    s = pd.DataFrame(index=df_raw.index)

    # Term / sovereign spreads (true yield-point)
    spread_pairs = {
        "us_term_10y_3m": ("GT10", "USGG3M"),
        "us_term_10y_2y": ("GT10", "USGG2YR"),
        "de_term_10y_2y": ("GTDEM10Y", "GTDEM2Y"),
        "it_de_10y": ("GTITL10YR", "GTDEM10Y"),
        "us_de_10y": ("GT10", "GTDEM10Y"),
    }
    for name, (a, b) in spread_pairs.items():
        s[name] = df_raw[a] - df_raw[b]
        s[f"{name}_chg4w"] = s[name].diff(horizon)

    # Credit / EM spreads (log price-index ratios, safe - risky)
    log_pairs = {
        "hy_spread": ("LUMSTRUU", "LF98TRUU"),      # safe leg - risky leg (inverted economics)
        "hy_ig_spread": ("LUACTRUU", "LF98TRUU"),
        "em_spread": ("LUACTRUU", "EMUSTRUU"),
    }
    for name, (safe, risky) in log_pairs.items():
        s[name] = np.log(df_raw[safe]) - np.log(df_raw[risky])
        s[f"{name}_chg4w"] = s[name].diff(horizon)

    # Standalone features (single series, no level+chg pair)
    r4w_mxus = np.log(df_raw["MXUS"]).diff(horizon)
    r4w_bond = np.log(df_raw["LUACTRUU"]).diff(horizon)
    s["equity_bond_rot"] = r4w_mxus - r4w_bond

    s["gold_oil_ratio"] = df_raw["XAUBGNL"] / df_raw["Cl1"]

    real_vol = np.log(df_raw["MXUS"]).diff().rolling(REALIZED_VOL_WEEKS).std() * np.sqrt(WEEKS_PER_YEAR) * 100
    s["vrp"] = df_raw["VIX"] - real_vol

    s["jpy_strength"] = -np.log(df_raw["JPY"]).diff(horizon)

    s = s.dropna()
    return s


def correlation_heatmap(originals, spreads, fig_path=None, threshold=0.9):
    combined = originals.join(spreads, how="inner")
    corr = combined.corr()
    if fig_path:
        fig, ax = plt.subplots(figsize=(20, 16))
        sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
        ax.set_title("Feature-Spread Correlation")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    high = corr.where(mask).stack()
    redundant = high[high.abs() >= threshold]
    return redundant


# --- Prompt 3.1: Collinearity cleanup + routing triggers ---

COLLINEAR_DROP = [
    "hy_spread", "hy_spread_chg4w",
    "us_term_10y_3m", "us_term_10y_3m_chg4w",
    "GTGBP20Y", "GTDEM30Y", "USGG30YR",
]


def remove_collinear_features(df, drop_list=None):
    drop_list = drop_list or COLLINEAR_DROP
    to_drop = [c for c in drop_list if c in df.columns]
    return df.drop(columns=to_drop)


ROUTING_DOMAINS = {
    "USD": ["libor_3m_spread_chg4w", "dxy_chg4w", "vrp",
            "us_10y_diff_chg4w", "usa_world_relative"],
    "Oro": ["real_yield_proxy_chg4w", "dxy_chg4w", "jpy_strength",
            "equity_bond_corr_13w", "gold_oil_ratio_chg4w"],
    "MBS": ["vix_level", "us_10y_vol_4w", "us_term_10y_2y_level",
            "libor_3m_spread_level", "mxus_drawdown_52w"],
}

TRIGGER_TYPE = {
    "libor_3m_spread_chg4w": "continuous",
    "dxy_chg4w": "continuous",
    "vrp": "continuous",
    "us_10y_diff_chg4w": "continuous",
    "usa_world_relative": "continuous",
    "real_yield_proxy_chg4w": "continuous",
    "jpy_strength": "continuous",
    "equity_bond_corr_13w": "continuous",
    "gold_oil_ratio_chg4w": "continuous",
    "vix_level": "rule_based",
    "us_10y_vol_4w": "continuous",
    "us_term_10y_2y_level": "rule_based",
    "libor_3m_spread_level": "rule_based",
    "mxus_drawdown_52w": "rule_based",
}

RULE_BASED_LEVELS = {
    "vix_level", "us_term_10y_2y_level",
    "libor_3m_spread_level", "mxus_drawdown_52w",
}


def build_routing_triggers(df_raw, df_stationary, df_spreads, horizon=HORIZON_4W):
    t = pd.DataFrame(index=df_raw.index)

    # USD domain
    t["libor_3m_spread_chg4w"] = (df_raw["US0001M"] - df_raw["USGG3M"]).diff(horizon)
    t["dxy_chg4w"] = np.log(df_raw["DXY"]).diff(horizon)
    t["vrp"] = df_spreads["vrp"].reindex(df_raw.index)
    t["us_10y_diff_chg4w"] = df_raw["GT10"].diff().diff(horizon)

    dm_composite = np.log(df_raw[["MXUS", "MXEU", "MXJP"]]).diff(horizon).mean(axis=1)
    t["usa_world_relative"] = np.log(df_raw["MXUS"]).diff(horizon) - dm_composite

    # Oro domain
    t["real_yield_proxy_chg4w"] = (
        df_raw["GT10"] - np.log(df_raw["LF94TRUU"]).diff(horizon)
    ).diff(horizon)
    t["jpy_strength"] = df_spreads["jpy_strength"].reindex(df_raw.index)
    t["equity_bond_corr_13w"] = (
        df_stationary["MXUS"].rolling(13).corr(df_stationary["LUACTRUU"])
    ).reindex(df_raw.index)
    t["gold_oil_ratio_chg4w"] = df_spreads["gold_oil_ratio"].reindex(df_raw.index).diff(horizon)

    # MBS domain
    t["vix_level"] = df_raw["VIX"]
    t["us_10y_vol_4w"] = df_raw["GT10"].diff().rolling(REALIZED_VOL_WEEKS).std()
    t["us_term_10y_2y_level"] = df_raw["GT10"] - df_raw["USGG2YR"]
    t["libor_3m_spread_level"] = df_raw["US0001M"] - df_raw["USGG3M"]
    t["mxus_drawdown_52w"] = df_raw["MXUS"] / df_raw["MXUS"].rolling(WEEKS_PER_YEAR).max() - 1

    t = t.dropna()
    return t


def routing_correlation(triggers, fig_path=None):
    domain_order = []
    for domain in ["USD", "Oro", "MBS"]:
        for feat in ROUTING_DOMAINS[domain]:
            if feat not in domain_order:
                domain_order.append(feat)
    cols = [c for c in domain_order if c in triggers.columns]
    corr = triggers[cols].corr()

    if fig_path:
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                    vmin=-1, vmax=1, ax=ax)
        ax.set_title("Routing Triggers Correlation")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    high = corr.where(mask).stack()
    pairs = high[high.abs() > 0.7]
    if len(pairs):
        print("|corr| > 0.7 pairs:")
        for (a, b), v in pairs.items():
            print(f"  {a} ~ {b} = {v:.3f}")
    return corr


def routing_trigger_summary(triggers):
    to_test = [c for c in triggers.columns if c not in RULE_BASED_LEVELS]
    rows = []
    for col in to_test:
        result = adfuller(triggers[col].dropna(), autolag="AIC")
        rows.append({
            "trigger": col,
            "adf_stat": result[0],
            "p_value": result[1],
            "stationary_5pct": result[1] < 0.05,
        })
    tab = pd.DataFrame(rows).set_index("trigger")
    n_stat = tab["stationary_5pct"].sum()
    print(f"ADF: {n_stat}/{len(tab)} triggers stationary at 5%")
    return tab


# --- Build functions ---

def build(verbose=True):
    """Prompt 2-3: stationarity + spreads."""
    df_raw = load_dataset(verbose=verbose)

    # Stationarity
    df_stat = make_stationary(df_raw)
    adf = adf_table(df_stat)
    if verbose:
        n_stat = adf["stationary"].sum()
        print(f"\nADF: {n_stat}/{len(adf)} features stationary at 5%")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_stat.to_parquet(FEATURES_PATH)
    if verbose:
        print(f"Saved {FEATURES_PATH} ({df_stat.shape})")

    # Spreads
    spreads = build_spreads(df_raw)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / "feature_spread_correlation.png"
    redundant = correlation_heatmap(
        df_stat.drop(columns=[TARGET_COL]),
        spreads, fig_path=fig_path
    )
    if verbose and len(redundant):
        print(f"\nRedundant pairs (|corr| >= 0.9):")
        for (a, b), v in redundant.items():
            print(f"  {a} ~ {b} = {v:.3f}")

    spreads.to_parquet(SPREADS_PATH)
    if verbose:
        print(f"Saved {SPREADS_PATH} ({spreads.shape})")

    return df_raw, df_stat, spreads


def build_routing(verbose=True):
    """Prompt 3.1: collinearity cleanup + routing triggers."""
    df_raw = load_dataset(verbose=False)
    df_stat = pd.read_parquet(FEATURES_PATH)
    spreads = pd.read_parquet(SPREADS_PATH)

    # Collinearity cleanup
    df_stat_clean = remove_collinear_features(df_stat)
    df_stat_clean.to_parquet(FEATURES_CLEAN_PATH)
    if verbose:
        print(f"features_stationary_clean: {df_stat_clean.shape}")

    spreads_clean = remove_collinear_features(spreads)
    spreads_clean.to_parquet(SPREADS_CLEAN_PATH)
    if verbose:
        print(f"spreads_clean: {spreads_clean.shape}")

    # Routing triggers
    triggers = build_routing_triggers(df_raw, df_stat, spreads)
    triggers.to_parquet(ROUTING_PATH)
    if verbose:
        print(f"routing_triggers: {triggers.shape}")

    fig_path = FIG_DIR / "routing_triggers_correlation.png"
    routing_correlation(triggers, fig_path=fig_path)
    routing_trigger_summary(triggers)

    return triggers


if __name__ == "__main__":
    build()
    build_routing()
