"""Build the final self-contained notebook. All code inline, no src/ imports."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))

def code(source):
    cells.append(nbf.v4.new_code_cell(source.strip()))


# =====================================================================
# TITLE
# =====================================================================
md("""
# Early-Warning System for Risk-Off Regimes + Allocation Routing
## Graduate School of Management — Politecnico di Milano

**Objective:** Build an anomaly-detection-based Early-Warning System (EWS) that:
1. **Detects** risk-off weeks using unsupervised novelty detection on weekly Bloomberg data
2. **Routes** the risk signal into regime-specific allocations (Levered Equity / Cash USD / Gold / MBS)
3. **Backtests** the strategy against 60/40 and buy-and-hold benchmarks
4. **Stress-tests** performance during the COVID crisis

**Methodology:**
- 4 anomaly detectors (MVG, One-Class SVM, Autoencoder, Isolation Forest) trained on *normal* weeks only
- Soft-voting ensemble with percentile-mapped scores
- Purged expanding walk-forward CV with 4-week embargo
- Domain-specific routing via signed z-score sub-scores (USD / Gold / MBS)
- Sealed 2019–2021 test holdout containing COVID

**Dataset:** Bloomberg weekly data, 2000–2021, 1,111 observations × 43 features + binary target Y.
""")

# =====================================================================
# SECTION 1: SETUP
# =====================================================================
md("## 1. Environment Setup & Imports")

code("""
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score, fbeta_score)
import joblib

%matplotlib inline
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['figure.dpi'] = 120
pd.set_option('display.max_columns', 60)
pd.set_option('display.width', 200)
np.random.seed(42)

print("All imports successful.")
""")

# =====================================================================
# SECTION 2: DATA LOADING
# =====================================================================
md("""
## 2. Data Loading & Exploration

The raw dataset is a Bloomberg export with two sheets:
- **Markets**: 1,111 weekly rows × 44 columns (43 features + target Y). Date column is `Data` (Italian).
- **Metadata**: ticker → description/type mapping.

**Target Y** = 1 indicates a risk-off week (~21.3% prevalence). No missing values.
""")

code("""
# --- DATA LOADING ---
DATA_FILE = "data/raw/04_May_Zenti_exercises.xlsx"
if not os.path.exists(DATA_FILE):
    DATA_FILE = "../data/raw/04_May_Zenti_exercises.xlsx"
if not os.path.exists(DATA_FILE):
    # Try root-level Excel
    candidates = [f for f in os.listdir('.') if f.endswith('.xlsx')]
    if candidates:
        DATA_FILE = candidates[0]

TARGET_COL = "Y"

# Load Markets sheet
df_raw = pd.read_excel(DATA_FILE, sheet_name="Markets")
df_raw.rename(columns={"Data": "Date"}, inplace=True)
df_raw["Date"] = pd.to_datetime(df_raw["Date"])
df_raw.sort_values("Date", inplace=True)
df_raw.set_index("Date", inplace=True)

# Load Metadata
metadata = pd.read_excel(DATA_FILE, sheet_name="Metadata")
metadata.columns = ["variable", "description", "type"]

print(f"Dataset: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
print(f"Period: {df_raw.index.min().date()} → {df_raw.index.max().date()}")
print(f"Missing values: {df_raw.isna().sum().sum()}")
print(f"\\nTarget prevalence: Y=1 → {(df_raw[TARGET_COL]==1).sum()}/{len(df_raw)} "
      f"({100*(df_raw[TARGET_COL]==1).mean():.1f}%)")
print(f"\\nColumns: {list(df_raw.columns)}")
""")

md("""
### 2.1 Target Distribution by Year

Risk-off clusters are concentrated in specific crisis periods. Several mid-decade years are entirely calm (Y=0).
This has critical implications for cross-validation design.
""")

code("""
yearly = df_raw.groupby(df_raw.index.year)[TARGET_COL].agg(['sum', 'count'])
yearly.columns = ['n_riskoff', 'n_total']
yearly['pct_riskoff'] = (100 * yearly['n_riskoff'] / yearly['n_total']).round(1)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Bar chart of risk-off weeks per year
ax = axes[0]
colors = ['#e74c3c' if v > 10 else '#3498db' for v in yearly['n_riskoff']]
ax.bar(yearly.index, yearly['n_riskoff'], color=colors)
ax.set_xlabel('Year')
ax.set_ylabel('Number of Risk-Off Weeks')
ax.set_title('Risk-Off Weeks (Y=1) by Year')
ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5)

# Timeline of Y
ax2 = axes[1]
ax2.fill_between(df_raw.index, df_raw[TARGET_COL], alpha=0.5, color='#e74c3c', label='Risk-Off')
ax2.set_xlabel('Date')
ax2.set_ylabel('Y')
ax2.set_title('Risk-Off Regime Timeline')
ax2.legend()

plt.tight_layout()
plt.show()

print(yearly.to_string())
""")

md("""
### 2.2 Feature Overview & Metadata
""")

code("""
print("=== METADATA ===")
print(metadata.to_string(index=False))
print(f"\\n=== FEATURE STATISTICS (first 20) ===")
print(df_raw.describe().T.head(20).to_string())
""")

md("""
### 2.3 Key Features During Crises
""")

code("""
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# VIX
ax = axes[0, 0]
ax.plot(df_raw.index, df_raw['VIX'], color='#2c3e50', linewidth=0.8)
ax.fill_between(df_raw.index, 0, df_raw['VIX'],
                where=df_raw[TARGET_COL]==1, alpha=0.3, color='red', label='Risk-Off')
ax.set_title('VIX Index')
ax.legend()

# US 10Y
ax = axes[0, 1]
ax.plot(df_raw.index, df_raw['GT10'], color='#2c3e50', linewidth=0.8)
ax.fill_between(df_raw.index, df_raw['GT10'].min(), df_raw['GT10'],
                where=df_raw[TARGET_COL]==1, alpha=0.3, color='red')
ax.set_title('US 10Y Yield')

# DXY
ax = axes[1, 0]
ax.plot(df_raw.index, df_raw['DXY'], color='#2c3e50', linewidth=0.8)
ax.fill_between(df_raw.index, df_raw['DXY'].min(), df_raw['DXY'],
                where=df_raw[TARGET_COL]==1, alpha=0.3, color='red')
ax.set_title('DXY Dollar Index')

# MXUS
ax = axes[1, 1]
ax.plot(df_raw.index, df_raw['MXUS'], color='#2c3e50', linewidth=0.8)
ax.fill_between(df_raw.index, df_raw['MXUS'].min(), df_raw['MXUS'],
                where=df_raw[TARGET_COL]==1, alpha=0.3, color='red')
ax.set_title('MSCI USA (MXUS)')

plt.suptitle('Key Features with Risk-Off Shading', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
""")

# =====================================================================
# SECTION 3: STATIONARITY
# =====================================================================
md("""
## 3. Stationarity Transforms

Raw price levels are non-stationary. We apply per-column transforms:
- **Log-returns** `np.log(x).diff()` for all price-like series (equities, FX, commodities, bond TR indices)
- **First differences** `x.diff()` for yields and rates
- **Level** (passthrough) for VIX, ECSURPUS, and target Y

We verify stationarity with the Augmented Dickey-Fuller test at 5% significance.
""")

code("""
from statsmodels.tsa.stattools import adfuller

TRANSFORM_MAP = {
    # log_return: price-like
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

def make_stationary(df, transform_map=TRANSFORM_MAP):
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        t = transform_map.get(col, "level")
        if t == "log_return":
            out[col] = np.log(df[col]).diff()
        elif t == "diff":
            out[col] = df[col].diff()
        else:
            out[col] = df[col]
    return out.iloc[1:]  # drop first NaN row

df_stat = make_stationary(df_raw)
print(f"Stationary features: {df_stat.shape} (1 row dropped from diff/log)")
""")

code("""
# ADF test on all features
adf_rows = []
for col in df_stat.columns:
    if col == TARGET_COL:
        continue
    result = adfuller(df_stat[col].dropna(), autolag="AIC")
    adf_rows.append({
        "feature": col,
        "adf_stat": round(result[0], 3),
        "p_value": f"{result[1]:.2e}" if result[1] < 0.01 else f"{result[1]:.4f}",
        "transform": TRANSFORM_MAP.get(col, "level"),
        "stationary_5pct": "YES" if result[1] < 0.05 else "NO",
    })

adf_df = pd.DataFrame(adf_rows).set_index("feature")
n_stat = (adf_df["stationary_5pct"] == "YES").sum()
print(f"\\nADF Results: {n_stat}/{len(adf_df)} features stationary at 5%")
print(adf_df.to_string())
""")

# =====================================================================
# SECTION 4: SPREADS
# =====================================================================
md("""
## 4. Cross-Asset Spread Features

We construct 20 spread features from the raw level data to capture relative-value dynamics:

**Term / Sovereign spreads** (true yield-point): US 10Y-3M, US 10Y-2Y, DE 10Y-2Y, BTP-Bund, US-DE 10Y

**Credit / EM spreads** (log price-index ratios, safe − risky): HY spread, HY-IG spread, EM spread.
*Note: these are NOT true OAS spreads — we only have total-return indices, so we use log(safe/risky). Rising value = risky underperforming = risk-off.*

**Standalone features**: equity-bond rotation, gold-oil ratio, VRP (implied - realized vol), JPY strength

Each spread produces both a **level** and a **4-week change** variant. Standalone features are single-series.

### Proxy Decisions (important for consistency)
- **MXWO (MSCI World) is absent** → proxied by MXUS
- **LEGATRUU (Global Agg) is absent** → proxied by LUACTRUU (US IG Corporate TR)
""")

code("""
HORIZON_4W = 4
REALIZED_VOL_WEEKS = 4
WEEKS_PER_YEAR = 52

def build_spreads(df, horizon=HORIZON_4W):
    s = pd.DataFrame(index=df.index)

    # Term / sovereign spreads
    spread_pairs = {
        "us_term_10y_3m": ("GT10", "USGG3M"),
        "us_term_10y_2y": ("GT10", "USGG2YR"),
        "de_term_10y_2y": ("GTDEM10Y", "GTDEM2Y"),
        "it_de_10y": ("GTITL10YR", "GTDEM10Y"),
        "us_de_10y": ("GT10", "GTDEM10Y"),
    }
    for name, (a, b) in spread_pairs.items():
        s[name] = df[a] - df[b]
        s[f"{name}_chg4w"] = s[name].diff(horizon)

    # Credit / EM spreads (log price-index ratios)
    log_pairs = {
        "hy_spread": ("LUMSTRUU", "LF98TRUU"),
        "hy_ig_spread": ("LUACTRUU", "LF98TRUU"),
        "em_spread": ("LUACTRUU", "EMUSTRUU"),
    }
    for name, (safe, risky) in log_pairs.items():
        s[name] = np.log(df[safe]) - np.log(df[risky])
        s[f"{name}_chg4w"] = s[name].diff(horizon)

    # Standalone features
    r4w_mxus = np.log(df["MXUS"]).diff(horizon)
    r4w_bond = np.log(df["LUACTRUU"]).diff(horizon)
    s["equity_bond_rot"] = r4w_mxus - r4w_bond
    s["gold_oil_ratio"] = df["XAUBGNL"] / df["Cl1"]
    real_vol = np.log(df["MXUS"]).diff().rolling(REALIZED_VOL_WEEKS).std() * np.sqrt(WEEKS_PER_YEAR) * 100
    s["vrp"] = df["VIX"] - real_vol
    s["jpy_strength"] = -np.log(df["JPY"]).diff(horizon)

    return s.dropna()

spreads = build_spreads(df_raw)
print(f"Spreads DataFrame: {spreads.shape}")
print(f"\\nSpread statistics:")
print(spreads.describe().T[['mean', 'std', 'min', 'max']].round(4).to_string())
""")

md("### 4.1 Spread Correlation Analysis")

code("""
# Correlation heatmap: original features + spreads
combined = df_stat.drop(columns=[TARGET_COL]).join(spreads, how="inner")
corr = combined.corr()

fig, ax = plt.subplots(figsize=(20, 16))
sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
            xticklabels=True, yticklabels=True)
ax.set_title("Full Feature-Spread Correlation Matrix", fontsize=14)
plt.tight_layout()
plt.show()

# Find redundant pairs
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
high_corr = corr.where(mask).stack()
redundant = high_corr[high_corr.abs() >= 0.9].sort_values(ascending=False)
print(f"Pairs with |correlation| >= 0.9:")
for (a, b), v in redundant.items():
    print(f"  {a} ~ {b} = {v:.3f}")
""")

# =====================================================================
# SECTION 5: COLLINEARITY CLEANUP
# =====================================================================
md("""
## 5. Collinearity Cleanup

We remove 7 features with |corr| > 0.9, keeping the more interpretable or broader survivor in each pair:
- `hy_spread` / `hy_spread_chg4w` → keep `hy_ig_spread`
- `us_term_10y_3m` / `us_term_10y_3m_chg4w` → keep `us_term_10y_2y`
- `GTGBP20Y` → keep `GTGBP30Y`
- `GTDEM30Y` → keep `GTDEM10Y`
- `USGG30YR` → keep `GT10`
""")

code("""
COLLINEAR_DROP = [
    "hy_spread", "hy_spread_chg4w",
    "us_term_10y_3m", "us_term_10y_3m_chg4w",
    "GTGBP20Y", "GTDEM30Y", "USGG30YR",
]

def remove_collinear(df, drop_list):
    to_drop = [c for c in drop_list if c in df.columns]
    return df.drop(columns=to_drop)

df_stat_clean = remove_collinear(df_stat, COLLINEAR_DROP)
spreads_clean = remove_collinear(spreads, COLLINEAR_DROP)

print(f"Stationary features: {df_stat.shape} → {df_stat_clean.shape} (after cleanup)")
print(f"Spreads: {spreads.shape} → {spreads_clean.shape} (after cleanup)")
print(f"\\nDropped: {COLLINEAR_DROP}")
""")

# =====================================================================
# SECTION 6: ROUTING TRIGGERS
# =====================================================================
md("""
## 6. Routing Triggers (14 features across 3 domains)

When the ensemble signals risk-off, we need to decide *where* to allocate.
Three domains drive the routing decision, each with its own set of triggers:

| Domain | Triggers | Economic Rationale |
|--------|----------|--------------------|
| **USD** | Funding stress, DXY momentum, VRP, Treasury flight, US relative | Dollar is safe haven when stress is global but external to USA |
| **Gold (Oro)** | Real yield, inverse DXY, JPY strength, equity-bond corr, gold-oil | Gold when stress includes monetary/inflation component or USD itself is weak |
| **MBS** | VIX range, yield vol, term spread, funding spread, drawdown | MBS as carry trade in *moderate* stress, blocked in acute crisis |

Note: `dxy_chg4w` appears in both USD (positive = bullish USD) and Gold (negative = bullish gold) — same indicator, opposite economic reading.
""")

code("""
def build_routing_triggers(df, df_stationary, df_spreads, horizon=HORIZON_4W):
    t = pd.DataFrame(index=df.index)

    # USD domain
    t["libor_3m_spread_chg4w"] = (df["US0001M"] - df["USGG3M"]).diff(horizon)
    t["dxy_chg4w"] = np.log(df["DXY"]).diff(horizon)
    t["vrp"] = df_spreads["vrp"].reindex(df.index)
    t["us_10y_diff_chg4w"] = df["GT10"].diff().diff(horizon)
    dm_composite = np.log(df[["MXUS", "MXEU", "MXJP"]]).diff(horizon).mean(axis=1)
    t["usa_world_relative"] = np.log(df["MXUS"]).diff(horizon) - dm_composite

    # Gold domain
    t["real_yield_proxy_chg4w"] = (df["GT10"] - np.log(df["LF94TRUU"]).diff(horizon)).diff(horizon)
    t["jpy_strength"] = df_spreads["jpy_strength"].reindex(df.index)
    t["equity_bond_corr_13w"] = (
        df_stationary["MXUS"].rolling(13).corr(df_stationary["LUACTRUU"])
    ).reindex(df.index)
    t["gold_oil_ratio_chg4w"] = df_spreads["gold_oil_ratio"].reindex(df.index).diff(horizon)

    # MBS domain
    t["vix_level"] = df["VIX"]
    t["us_10y_vol_4w"] = df["GT10"].diff().rolling(REALIZED_VOL_WEEKS).std()
    t["us_term_10y_2y_level"] = df["GT10"] - df["USGG2YR"]
    t["libor_3m_spread_level"] = df["US0001M"] - df["USGG3M"]
    t["mxus_drawdown_52w"] = df["MXUS"] / df["MXUS"].rolling(WEEKS_PER_YEAR).max() - 1

    return t.dropna()

triggers = build_routing_triggers(df_raw, df_stat, spreads)
print(f"Routing triggers: {triggers.shape}")
print(f"Start date: {triggers.index.min().date()} (52-week warm-up for drawdown)")
print(f"\\nDomain assignments:")
ROUTING_DOMAINS = {
    "USD": ["libor_3m_spread_chg4w", "dxy_chg4w", "vrp", "us_10y_diff_chg4w", "usa_world_relative"],
    "Oro": ["real_yield_proxy_chg4w", "dxy_chg4w", "jpy_strength", "equity_bond_corr_13w", "gold_oil_ratio_chg4w"],
    "MBS": ["vix_level", "us_10y_vol_4w", "us_term_10y_2y_level", "libor_3m_spread_level", "mxus_drawdown_52w"],
}
for domain, feats in ROUTING_DOMAINS.items():
    print(f"  {domain}: {feats}")
""")

code("""
# Routing triggers correlation
domain_order = []
for domain in ["USD", "Oro", "MBS"]:
    for f in ROUTING_DOMAINS[domain]:
        if f not in domain_order:
            domain_order.append(f)

corr_rt = triggers[domain_order].corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_rt, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, ax=ax)
ax.set_title("Routing Triggers Correlation (domain-ordered)", fontsize=13)
plt.tight_layout()
plt.show()

# Flag high correlations
mask_rt = np.triu(np.ones_like(corr_rt, dtype=bool), k=1)
high_rt = corr_rt.where(mask_rt).stack()
print("|corr| > 0.7 pairs:")
for (a, b), v in high_rt[high_rt.abs() > 0.7].items():
    print(f"  {a} ~ {b} = {v:.3f}")
""")

# =====================================================================
# SECTION 7: WALK-FORWARD SPLITS
# =====================================================================
md("""
## 7. Walk-Forward Cross-Validation Design

We use **purged expanding walk-forward** CV with a **4-week embargo**:
- Train always starts at dataset start, expanding forward
- 4-week gap between train end and val start (kills lookback leakage from 4w features)
- 5 folds, each capturing a known crisis in its validation window
- **Development set**: all data ≤ 2018-12-31
- **Test holdout**: 2019-01-01 → 2021-04-20 (sealed, contains COVID)

| Fold | Train End | Val Period | Crisis |
|------|-----------|------------|--------|
| 1 | 2006-12-31 | 2007-01 → 2009-12 | GFC 2008 |
| 2 | 2009-12-31 | 2010-02 → 2012-12 | Euro 2011 |
| 3 | 2012-12-31 | 2013-01 → 2014-12 | Taper 2013 |
| 4 | 2014-12-31 | 2015-02 → 2016-12 | China-Oil 2015-16 |
| 5 | 2016-12-31 | 2017-01 → 2018-12 | Q4 2018 selloff |

**Critical:** Folds 1-2 have 53/64 positives (reliable). Folds 3-5 have only 2/10/6 positives.
→ All CV metrics are **weighted by n_pos** so folds 1-2 dominate.
""")

code("""
DEV_END = "2018-12-31"
EMBARGO_WEEKS = 4

WALKFORWARD_FOLDS = [
    (1, "2006-12-31", "2009-12-31", "GFC 2008"),
    (2, "2009-12-31", "2012-12-31", "Euro 2011"),
    (3, "2012-12-31", "2014-12-31", "Taper 2013"),
    (4, "2014-12-31", "2016-12-31", "China-Oil 2015-16"),
    (5, "2016-12-31", "2018-12-31", "Q4 2018 selloff"),
]

# Build the model feature set: stationary_clean ⨝ spreads_clean + equity_bond_corr_13w
df_model = df_stat_clean.join(spreads_clean, how="inner")
# Add the 57th feature
if "equity_bond_corr_13w" not in df_model.columns:
    df_model = df_model.join(triggers[["equity_bond_corr_13w"]], how="left")
df_model = df_model.dropna()

print(f"Model feature set: {df_model.shape} (incl. Y)")
print(f"  X columns: {df_model.shape[1] - 1} features + Y")

# Split
dev_set = df_model.loc[:DEV_END]
test_holdout = df_model.loc[DEV_END:].iloc[1:]

print(f"\\nDevelopment set: {dev_set.shape} ({dev_set.index.min().date()} → {dev_set.index.max().date()})")
print(f"Test holdout: {test_holdout.shape} ({test_holdout.index.min().date()} → {test_holdout.index.max().date()})")
print(f"COVID in test: {len(test_holdout.loc['2020-02':'2020-04'])} rows ✓")

# Build folds
cv_folds = []
train_normals_per_fold = []

print(f"\\n{'Fold':>4} {'Train':>8} {'Val':>6} {'Val Y=1':>8} {'%':>6}  Crisis")
print("-" * 60)

for fold_id, train_end, val_end, crisis in WALKFORWARD_FOLDS:
    tr = df_model.loc[:train_end]
    after_train = df_model.loc[train_end:].iloc[1:]
    val = after_train.iloc[EMBARGO_WEEKS:].loc[:val_end]

    n_pos = int((val[TARGET_COL] == 1).sum())
    pct = 100 * n_pos / len(val) if len(val) else 0
    flag = " ⚠" if pct < 10 or n_pos <= 20 else ""

    print(f"  {fold_id:>2}   {len(tr):>6}  {len(val):>5}  {n_pos:>6}  {pct:>5.1f}%  {crisis}{flag}")

    cv_folds.append({"fold_id": fold_id, "train": tr, "val": val, "crisis": crisis})
    normals = tr[tr[TARGET_COL] == 0]
    train_normals_per_fold.append({"fold_id": fold_id, "train_normals": normals})
""")

code("""
# Also build routing splits
triggers_dev = triggers.loc[:DEV_END]
triggers_test = triggers.loc[DEV_END:].iloc[1:]

routing_folds = []
for fold_id, train_end, val_end, crisis in WALKFORWARD_FOLDS:
    tr_r = triggers.loc[:train_end]
    after_r = triggers.loc[train_end:].iloc[1:]
    val_r = after_r.iloc[EMBARGO_WEEKS:].loc[:val_end]
    routing_folds.append({"fold_id": fold_id, "train": tr_r, "val": val_r})

print(f"Routing splits — Dev: {triggers_dev.shape}, Test: {triggers_test.shape}")
""")

# =====================================================================
# SECTION 8: PREPROCESSING (FoldScaler)
# =====================================================================
md("""
## 8. Preprocessing: FoldScaler (No Leakage)

**Critical rule:** StandardScaler is fit **only on each fold's train** (then applied to train+val).
For the test holdout, the scaler is fit on the **entire development set**.
This is the #1 way to silently break this project if violated.
""")

code("""
def prepare_X_y(df):
    \"\"\"Split df into X (features) and y (target).\"\"\"
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    X = df.drop(columns=[TARGET_COL], errors="ignore")
    return X, y

# Build per-fold scalers
fold_scalers = []
for fold in cv_folds:
    X_tr, _ = prepare_X_y(fold["train"])
    sc = StandardScaler()
    sc.fit(X_tr)
    fold_scalers.append(sc)

# Final scaler on full development set
X_dev, y_dev = prepare_X_y(dev_set)
final_scaler = StandardScaler()
final_scaler.fit(X_dev)

# Prepare test data
X_test, y_test = prepare_X_y(test_holdout)
X_test_scaled = pd.DataFrame(final_scaler.transform(X_test),
                              index=X_test.index, columns=X_test.columns)

# Development normals for final model fitting
dev_normals = X_dev[y_dev == 0]
dev_normals_scaled = pd.DataFrame(final_scaler.transform(dev_normals),
                                   index=dev_normals.index, columns=dev_normals.columns)

print(f"Feature dimensionality: {X_dev.shape[1]}")
print(f"Development normals: {len(dev_normals)} rows (Y=0)")
print(f"Test holdout: {len(X_test)} rows")
print(f"Scalers built: {len(fold_scalers)} per-fold + 1 final")
""")

# =====================================================================
# SECTION 9: METRICS HELPER
# =====================================================================
md("## 9. Evaluation Metrics")

code("""
METRIC_NAMES = ["F1", "AUC_PR", "AUC_ROC", "Precision", "Recall", "F2"]

def compute_metrics(y_true, y_pred, y_score=None):
    m = {
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        m["AUC_ROC"] = roc_auc_score(y_true, y_score)
        m["AUC_PR"] = average_precision_score(y_true, y_score)
    else:
        m["AUC_ROC"] = np.nan
        m["AUC_PR"] = np.nan
    return m

def weighted_metric(fold_metrics, metric="F1"):
    vals = [(fm[metric], fm["n_pos"]) for fm in fold_metrics
            if not np.isnan(fm.get(metric, np.nan))]
    total_w = sum(w for _, w in vals)
    if total_w == 0:
        return 0.0
    return sum(v * w for v, w in vals) / total_w

print("Metrics defined: F1, AUC-PR, AUC-ROC, Precision, Recall, F-beta(2)")
print("  F2 (beta=2): recall-weighted — missing a crisis costs 20x a false alarm")
""")

# =====================================================================
# SECTION 10: MVG MODEL
# =====================================================================
md("""
## 10. Model 1: Multivariate Gaussian (MVG) — Ledoit-Wolf Shrinkage

The MVG models the distribution of *normal* weeks as a multivariate Gaussian.
Score = **squared Mahalanobis distance** from the estimated mean/covariance.
**Ledoit-Wolf shrinkage** is mandatory because the covariance matrix is unstable with
56 features and ~360 normals in Fold 1.

Threshold ε is tuned via walk-forward, maximizing **weighted F1** across folds.
""")

code("""
class MVGDetector:
    def __init__(self):
        self.lw_ = None
        self.threshold_ = None

    def fit(self, X_normals):
        self.lw_ = LedoitWolf()
        self.lw_.fit(X_normals)
        return self

    def score(self, X):
        return self.lw_.mahalanobis(X)

    def predict(self, X, threshold=None):
        t = threshold or self.threshold_
        return (self.score(X) >= t).astype(int)

# Fit on full development normals
mvg = MVGDetector()
mvg.fit(dev_normals_scaled)

# Tune threshold via walk-forward
all_val_scores = []
for fold, sc in zip(cv_folds, fold_scalers):
    X_val, _ = prepare_X_y(fold["val"])
    X_val_s = pd.DataFrame(sc.transform(X_val), index=X_val.index, columns=X_val.columns)
    all_val_scores.extend(mvg.score(X_val_s))

candidates = np.percentile(all_val_scores, np.linspace(50, 99.9, 200))
best_f1_mvg, best_t_mvg = 0, candidates[0]

for t in candidates:
    fold_m = []
    for fold, sc in zip(cv_folds, fold_scalers):
        X_val, y_val = prepare_X_y(fold["val"])
        X_val_s = pd.DataFrame(sc.transform(X_val), index=X_val.index, columns=X_val.columns)
        scores = mvg.score(X_val_s)
        preds = (scores >= t).astype(int)
        fm = compute_metrics(y_val, preds, scores)
        fm["n_pos"] = int(y_val.sum())
        fold_m.append(fm)
    wf1 = weighted_metric(fold_m, "F1")
    if wf1 > best_f1_mvg:
        best_f1_mvg = wf1
        best_t_mvg = t

mvg.threshold_ = best_t_mvg
print(f"MVG threshold: {best_t_mvg:.2f}, CV weighted F1: {best_f1_mvg:.3f}")

# Test holdout
mvg_test_scores = mvg.score(X_test_scaled)
mvg_test_preds = mvg.predict(X_test_scaled)
mvg_test_metrics = compute_metrics(y_test, mvg_test_preds, mvg_test_scores)
print(f"\\nMVG Test Holdout:")
for k, v in mvg_test_metrics.items():
    print(f"  {k}: {v:.3f}")
""")

code("""
# MVG score distribution visualization
fig, ax = plt.subplots(figsize=(12, 5))
normal_scores = mvg.score(dev_normals_scaled)
anomaly_dev = X_dev[y_dev == 1]
anomaly_scores = mvg.score(pd.DataFrame(final_scaler.transform(anomaly_dev),
                           index=anomaly_dev.index, columns=anomaly_dev.columns))

ax.hist(normal_scores, bins=80, alpha=0.6, color='green', label=f'Normal (n={len(normal_scores)})', density=True)
ax.hist(anomaly_scores, bins=40, alpha=0.6, color='red', label=f'Risk-Off (n={len(anomaly_scores)})', density=True)
ax.axvline(mvg.threshold_, color='black', linestyle='--', linewidth=2, label=f'Threshold = {mvg.threshold_:.0f}')
ax.set_xlabel('Squared Mahalanobis Distance')
ax.set_ylabel('Density')
ax.set_title('MVG: Score Distribution (Development Set)')
ax.legend()
ax.set_xlim(0, np.percentile(np.concatenate([normal_scores, anomaly_scores]), 99))
plt.tight_layout()
plt.show()
""")

# =====================================================================
# SECTION 11: ONE-CLASS SVM
# =====================================================================
md("""
## 11. Model 2: One-Class SVM (RBF Kernel)

Grid search over `nu ∈ {0.05, 0.10, 0.15, 0.22}` × `gamma ∈ {scale, auto, 0.01, 0.001}`.
Score = `-decision_function` (higher = more anomalous). Select by weighted F1.
""")

code("""
nu_grid = [0.05, 0.10, 0.15, 0.22]
gamma_grid = ["scale", "auto", 0.01, 0.001]

best_svm_f1, best_svm_params = 0, {}
svm_results = []

for nu in nu_grid:
    for gamma in gamma_grid:
        fold_m = []
        for fold, normals, sc in zip(cv_folds, train_normals_per_fold, fold_scalers):
            X_n, _ = prepare_X_y(normals["train_normals"])
            X_n_s = pd.DataFrame(sc.transform(X_n), index=X_n.index, columns=X_n.columns)
            svm_fold = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
            svm_fold.fit(X_n_s)

            X_val, y_val = prepare_X_y(fold["val"])
            X_val_s = pd.DataFrame(sc.transform(X_val), index=X_val.index, columns=X_val.columns)
            scores = -svm_fold.decision_function(X_val_s)
            preds = (scores >= 0).astype(int)
            fm = compute_metrics(y_val, preds, scores)
            fm["n_pos"] = int(y_val.sum())
            fold_m.append(fm)

        wf1 = weighted_metric(fold_m, "F1")
        svm_results.append({"nu": nu, "gamma": gamma, "weighted_F1": wf1})
        if wf1 > best_svm_f1:
            best_svm_f1 = wf1
            best_svm_params = {"nu": nu, "gamma": gamma}

print(f"SVM grid search results:")
svm_grid_df = pd.DataFrame(svm_results).sort_values("weighted_F1", ascending=False)
print(svm_grid_df.head(8).to_string(index=False))
print(f"\\nBest: nu={best_svm_params['nu']}, gamma={best_svm_params['gamma']}, F1={best_svm_f1:.3f}")

# Fit final SVM on development normals
svm_final = OneClassSVM(kernel="rbf", **best_svm_params)
svm_final.fit(dev_normals_scaled)
svm_threshold = 0  # native boundary

svm_test_scores = -svm_final.decision_function(X_test_scaled)
svm_test_preds = (svm_test_scores >= svm_threshold).astype(int)
svm_test_metrics = compute_metrics(y_test, svm_test_preds, svm_test_scores)
print(f"\\nSVM Test Holdout:")
for k, v in svm_test_metrics.items():
    print(f"  {k}: {v:.3f}")
""")

# =====================================================================
# SECTION 12: ISOLATION FOREST
# =====================================================================
md("""
## 12. Model 3: Isolation Forest

`n_estimators=200`, `random_state=42`. Grid search over `contamination ∈ {0.05, 0.10, 0.15, 0.22}`.
Score = `-decision_function`. Then fine-tune threshold via walk-forward.
""")

code("""
contam_grid = [0.05, 0.10, 0.15, 0.22]
best_if_f1, best_if_contam = 0, 0.10

for c in contam_grid:
    fold_m = []
    for fold, normals, sc in zip(cv_folds, train_normals_per_fold, fold_scalers):
        X_n, _ = prepare_X_y(normals["train_normals"])
        X_n_s = pd.DataFrame(sc.transform(X_n), index=X_n.index, columns=X_n.columns)
        iforest = IsolationForest(contamination=c, n_estimators=200, random_state=42)
        iforest.fit(X_n_s)

        X_val, y_val = prepare_X_y(fold["val"])
        X_val_s = pd.DataFrame(sc.transform(X_val), index=X_val.index, columns=X_val.columns)
        scores = -iforest.decision_function(X_val_s)
        preds = (scores >= 0).astype(int)
        fm = compute_metrics(y_val, preds, scores)
        fm["n_pos"] = int(y_val.sum())
        fold_m.append(fm)

    wf1 = weighted_metric(fold_m, "F1")
    print(f"  contamination={c}: weighted F1={wf1:.3f}")
    if wf1 > best_if_f1:
        best_if_f1 = wf1
        best_if_contam = c

print(f"\\nBest: contamination={best_if_contam}, F1={best_if_f1:.3f}")

# Fit final IF
if_final = IsolationForest(contamination=best_if_contam, n_estimators=200, random_state=42)
if_final.fit(dev_normals_scaled)

# Tune threshold
all_if_scores = []
for fold, sc in zip(cv_folds, fold_scalers):
    X_val, _ = prepare_X_y(fold["val"])
    X_val_s = pd.DataFrame(sc.transform(X_val), index=X_val.index, columns=X_val.columns)
    all_if_scores.extend(-if_final.decision_function(X_val_s))

if_pcts = np.percentile(all_if_scores, np.linspace(30, 99, 150))
best_if_t, best_if_wf1 = if_pcts[0], 0

for t in if_pcts:
    fold_m = []
    for fold, sc in zip(cv_folds, fold_scalers):
        X_val, y_val = prepare_X_y(fold["val"])
        X_val_s = pd.DataFrame(sc.transform(X_val), index=X_val.index, columns=X_val.columns)
        scores = -if_final.decision_function(X_val_s)
        preds = (scores >= t).astype(int)
        fm = compute_metrics(y_val, preds, scores)
        fm["n_pos"] = int(y_val.sum())
        fold_m.append(fm)
    wf1 = weighted_metric(fold_m, "F1")
    if wf1 > best_if_wf1:
        best_if_wf1 = wf1
        best_if_t = t

if_threshold = best_if_t
print(f"Tuned threshold: {if_threshold:.4f}, weighted F1: {best_if_wf1:.3f}")

if_test_scores = -if_final.decision_function(X_test_scaled)
if_test_preds = (if_test_scores >= if_threshold).astype(int)
if_test_metrics = compute_metrics(y_test, if_test_preds, if_test_scores)
print(f"\\nIF Test Holdout:")
for k, v in if_test_metrics.items():
    print(f"  {k}: {v:.3f}")
""")

# =====================================================================
# SECTION 13: AUTOENCODER
# =====================================================================
md("""
## 13. Model 4: Autoencoder

Architecture: **56→24→12→6→12→24→56** (symmetric bottleneck).
- Dropout 0.15, ReLU activations, MSE loss, Adam lr=1e-3
- Early stopping on a **temporal sub-split of the train** (NEVER on the CV val fold — that would leak)
- Score = reconstruction MSE per sample
- TF seed = 42 for reproducibility
""")

code("""
import tensorflow as tf
tf.random.set_seed(42)

input_dim = X_dev.shape[1]

def build_autoencoder(dim):
    inp = tf.keras.Input(shape=(dim,))
    x = tf.keras.layers.Dense(24, activation="relu")(inp)
    x = tf.keras.layers.Dropout(0.15)(x)
    x = tf.keras.layers.Dense(12, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    x = tf.keras.layers.Dense(6, activation="relu")(x)
    x = tf.keras.layers.Dense(12, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    x = tf.keras.layers.Dense(24, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    out = tf.keras.layers.Dense(dim, activation="linear")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model

# Fit on development normals with temporal early-stopping split
n_normals = len(dev_normals_scaled)
split_idx = int(n_normals * 0.85)
X_ae_train = dev_normals_scaled.iloc[:split_idx]
X_ae_esval = dev_normals_scaled.iloc[split_idx:]

ae_model = build_autoencoder(input_dim)
es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

history = ae_model.fit(
    X_ae_train, X_ae_train,
    validation_data=(X_ae_esval, X_ae_esval),
    epochs=200, batch_size=32, verbose=0, callbacks=[es]
)

print(f"Autoencoder trained: {len(history.history['loss'])} epochs (early stopped)")
print(f"  Train loss: {history.history['loss'][-1]:.6f}")
print(f"  Val loss: {history.history['val_loss'][-1]:.6f}")
""")

code("""
# AE training curves
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history.history['loss'], label='Train Loss')
ax.plot(history.history['val_loss'], label='Val Loss (temporal sub-split)')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE')
ax.set_title('Autoencoder Training Curves')
ax.legend()
plt.tight_layout()
plt.show()
""")

code("""
def ae_score(X):
    arr = X.values if hasattr(X, 'values') else X
    recon = ae_model.predict(arr, verbose=0)
    return np.mean((arr - recon) ** 2, axis=1)

# Tune AE threshold
all_ae_scores = []
for fold, sc in zip(cv_folds, fold_scalers):
    X_val, _ = prepare_X_y(fold["val"])
    X_val_s = pd.DataFrame(sc.transform(X_val), index=X_val.index, columns=X_val.columns)
    all_ae_scores.extend(ae_score(X_val_s))

ae_pcts = np.percentile(all_ae_scores, np.linspace(50, 99.5, 150))
best_ae_t, best_ae_f1 = ae_pcts[0], 0

for t in ae_pcts:
    fold_m = []
    for fold, sc in zip(cv_folds, fold_scalers):
        X_val, y_val = prepare_X_y(fold["val"])
        X_val_s = pd.DataFrame(sc.transform(X_val), index=X_val.index, columns=X_val.columns)
        scores = ae_score(X_val_s)
        preds = (scores >= t).astype(int)
        fm = compute_metrics(y_val, preds, scores)
        fm["n_pos"] = int(y_val.sum())
        fold_m.append(fm)
    wf1 = weighted_metric(fold_m, "F1")
    if wf1 > best_ae_f1:
        best_ae_f1 = wf1
        best_ae_t = t

ae_threshold = best_ae_t
print(f"AE threshold: {ae_threshold:.6f}, CV weighted F1: {best_ae_f1:.3f}")

ae_test_scores = ae_score(X_test_scaled)
ae_test_preds = (ae_test_scores >= ae_threshold).astype(int)
ae_test_metrics = compute_metrics(y_test, ae_test_preds, ae_test_scores)
print(f"\\nAE Test Holdout:")
for k, v in ae_test_metrics.items():
    print(f"  {k}: {v:.3f}")
""")

# =====================================================================
# SECTION 14: MODEL COMPARISON
# =====================================================================
md("## 14. Model Comparison")

code("""
all_test_results = {
    "MVG": mvg_test_metrics,
    "SVM": svm_test_metrics,
    "IF": if_test_metrics,
    "AE": ae_test_metrics,
}

comparison_df = pd.DataFrame(all_test_results).T[METRIC_NAMES]
print("=" * 70)
print("MODEL COMPARISON — Test Holdout (2019–2021)")
print("=" * 70)
print(comparison_df.round(3).to_string())

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
comparison_df[["F1", "Precision", "Recall", "F2"]].plot(kind="bar", ax=ax, rot=0)
ax.set_title("Classification Metrics (Test Holdout)")
ax.set_ylabel("Score")
ax.legend(loc="upper right")
ax.set_ylim(0, 1.1)

ax = axes[1]
comparison_df[["AUC_ROC", "AUC_PR"]].plot(kind="bar", ax=ax, rot=0, color=["#e74c3c", "#3498db"])
ax.set_title("Ranking Metrics (Test Holdout)")
ax.set_ylabel("AUC")
ax.set_ylim(0.5, 1.0)

plt.tight_layout()
plt.show()
""")

# =====================================================================
# SECTION 15: ENSEMBLE
# =====================================================================
md("""
## 15. Ensemble Detection

We combine all 4 models using **percentile-mapped scores** to make them comparable:
each model's raw score is converted to its empirical percentile against the train-normals distribution.

Three ensemble variants:
1. **Hard voting**: each model flags 1/0 with its own threshold, majority (≥3/4) wins
2. **Soft voting mean**: mean of percentile scores, threshold τ tuned on CV
3. **Soft voting median**: median of percentile scores (robust to one rogue model)
""")

code("""
def score_to_percentile(s_train_normals, s_new):
    sorted_train = np.sort(s_train_normals)
    n = len(sorted_train)
    return np.searchsorted(sorted_train, s_new, side="right") / n

# Compute percentile scores on test holdout
normal_mvg_scores = mvg.score(dev_normals_scaled)
normal_svm_scores = -svm_final.decision_function(dev_normals_scaled)
normal_if_scores = -if_final.decision_function(dev_normals_scaled)
normal_ae_scores = ae_score(dev_normals_scaled)

pct_mvg_test = score_to_percentile(normal_mvg_scores, mvg_test_scores)
pct_svm_test = score_to_percentile(normal_svm_scores, svm_test_scores)
pct_if_test = score_to_percentile(normal_if_scores, if_test_scores)
pct_ae_test = score_to_percentile(normal_ae_scores, ae_test_scores)

pct_matrix_test = np.column_stack([pct_mvg_test, pct_svm_test, pct_if_test, pct_ae_test])
pct_mean_test = pct_matrix_test.mean(axis=1)
pct_median_test = np.median(pct_matrix_test, axis=1)

# Hard voting
hard_preds = (mvg_test_preds + svm_test_preds + if_test_preds + ae_test_preds >= 3).astype(int)

# Tune soft thresholds on CV folds
best_soft_mean_t, best_soft_mean_f1 = 0.5, 0
best_soft_med_t, best_soft_med_f1 = 0.5, 0

for t in np.linspace(0.5, 0.99, 100):
    for mode in ["mean", "median"]:
        fold_m = []
        for fold, sc in zip(cv_folds, fold_scalers):
            X_val, y_val = prepare_X_y(fold["val"])
            X_val_s = pd.DataFrame(sc.transform(X_val), index=X_val.index, columns=X_val.columns)

            # Per-model scores on this fold
            s_mvg = mvg.score(X_val_s)
            s_svm = -svm_final.decision_function(X_val_s)
            s_if = -if_final.decision_function(X_val_s)
            s_ae = ae_score(X_val_s)

            p_mvg = score_to_percentile(normal_mvg_scores, s_mvg)
            p_svm = score_to_percentile(normal_svm_scores, s_svm)
            p_if = score_to_percentile(normal_if_scores, s_if)
            p_ae = score_to_percentile(normal_ae_scores, s_ae)

            pct_mat = np.column_stack([p_mvg, p_svm, p_if, p_ae])
            if mode == "mean":
                agg = pct_mat.mean(axis=1)
            else:
                agg = np.median(pct_mat, axis=1)

            preds = (agg >= t).astype(int)
            fm = compute_metrics(y_val, preds, agg)
            fm["n_pos"] = int(y_val.sum())
            fold_m.append(fm)

        wf1 = weighted_metric(fold_m, "F1")
        if mode == "mean" and wf1 > best_soft_mean_f1:
            best_soft_mean_f1 = wf1
            best_soft_mean_t = t
        elif mode == "median" and wf1 > best_soft_med_f1:
            best_soft_med_f1 = wf1
            best_soft_med_t = t

print(f"Soft Mean threshold: {best_soft_mean_t:.3f} (CV F1={best_soft_mean_f1:.3f})")
print(f"Soft Median threshold: {best_soft_med_t:.3f} (CV F1={best_soft_med_f1:.3f})")

# Evaluate all ensembles on test
soft_mean_preds = (pct_mean_test >= best_soft_mean_t).astype(int)
soft_med_preds = (pct_median_test >= best_soft_med_t).astype(int)

ens_results = {
    "Hard_Vote": compute_metrics(y_test, hard_preds, pct_mean_test),
    "Soft_Mean": compute_metrics(y_test, soft_mean_preds, pct_mean_test),
    "Soft_Median": compute_metrics(y_test, soft_med_preds, pct_median_test),
}

# Full comparison
all_results = {**all_test_results, **ens_results}
full_df = pd.DataFrame(all_results).T[METRIC_NAMES]
print("\\n" + "=" * 70)
print("ALL MODELS + ENSEMBLE — Test Holdout")
print("=" * 70)
print(full_df.round(3).to_string())
""")

code("""
# Error correlation analysis
errors = pd.DataFrame({
    "MVG": (mvg_test_preds != y_test.values).astype(int),
    "SVM": (svm_test_preds != y_test.values).astype(int),
    "IF": (if_test_preds != y_test.values).astype(int),
    "AE": (ae_test_preds != y_test.values).astype(int),
})
err_corr = errors.corr()
mask_e = np.triu(np.ones_like(err_corr, dtype=bool), k=1)
mean_err_corr = err_corr.where(mask_e).stack().mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
sns.heatmap(err_corr, annot=True, fmt=".3f", cmap="Reds", vmin=0, vmax=1, ax=ax)
ax.set_title(f"Error Correlation (mean={mean_err_corr:.3f})")

ax = axes[1]
full_df[["F1", "AUC_PR"]].plot(kind="barh", ax=ax)
ax.set_title("F1 & AUC-PR: Singles vs Ensembles")
ax.set_xlabel("Score")
ax.axvline(x=full_df.loc["Soft_Mean", "F1"], color="red", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

if mean_err_corr > 0.85:
    print("⚠ HIGH error correlation: ensemble may add little value")
else:
    print(f"✓ Error correlation {mean_err_corr:.3f} < 0.85: ensemble adds genuine value")

# Select best ensemble
best_ens_name = max(ens_results, key=lambda k: ens_results[k]["F1"])
print(f"\\nBest ensemble: {best_ens_name} (F1={ens_results[best_ens_name]['F1']:.3f})")
""")

code("""
# Use the best ensemble signal for routing
if best_ens_name == "Soft_Mean":
    ensemble_preds = soft_mean_preds
elif best_ens_name == "Soft_Median":
    ensemble_preds = soft_med_preds
else:
    ensemble_preds = hard_preds

ensemble_signal = pd.Series(ensemble_preds, index=X_test.index, name="ensemble_signal")
print(f"Ensemble signal: {int(ensemble_signal.sum())} risk-off / {len(ensemble_signal)} total weeks")
""")

# =====================================================================
# SECTION 16: ENSEMBLE TIMELINE
# =====================================================================
md("### 15.1 Ensemble Predictions Timeline")

code("""
fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [2, 1]})

ax = axes[0]
ax.plot(X_test.index, pct_mean_test, label="Ensemble Score (mean pct)", color="#2c3e50", linewidth=1)
ax.axhline(y=best_soft_mean_t, color="red", linestyle="--", label=f"Threshold={best_soft_mean_t:.2f}")
ax.fill_between(X_test.index, 0, 1, where=y_test==1, alpha=0.2, color="red", label="True Risk-Off")
ax.set_ylabel("Percentile Score")
ax.set_title("Ensemble Score on Test Holdout (2019–2021)")
ax.legend(loc="upper left")

ax2 = axes[1]
ax2.fill_between(X_test.index, 0, y_test.values, alpha=0.5, color="red", label="True Y=1")
ax2.fill_between(X_test.index, 0, -ensemble_preds, alpha=0.5, color="blue", label="Predicted Y=1")
ax2.set_ylabel("Signal")
ax2.set_xlabel("Date")
ax2.legend()

plt.tight_layout()
plt.show()
""")

# =====================================================================
# SECTION 17: DOMAIN SUB-SCORES
# =====================================================================
md("""
## 16. Domain Sub-Scores (USD / Gold / MBS)

When the ensemble signals risk-off, we need to decide WHERE to allocate.
Three domain sub-scores drive the routing:

**USD sub-score** = mean of signed z-scores:
| Trigger | Sign | Rationale |
|---------|------|-----------|
| libor_3m_spread_chg4w | +1 | Dollar funding stress |
| dxy_chg4w | +1 | Dollar strengthening |
| vrp | +1 | Implied > realized vol |
| us_10y_diff_chg4w | -1 | Treasury rally (yields down) |
| usa_world_relative | +1 | US outperforming world |

**Gold sub-score** = mean of signed z-scores (opposite DXY sign):
| Trigger | Sign | Rationale |
|---------|------|-----------|
| real_yield_proxy_chg4w | -1 | Real yields falling |
| dxy_chg4w | -1 | Dollar weakening |
| jpy_strength | +1 | Yen as safe haven |
| equity_bond_corr_13w | +1 | Diversification failure |
| gold_oil_ratio_chg4w | +1 | Gold outperforming oil |

**MBS sub-score** = binary rule-based (moderate stress only):
- Active if VIX ∈ [20,28] AND term spread > 0 AND drawdown ∈ [-12%, -5%]
- Blocked if VIX > 30 OR funding stress > p90 OR HY-IG spread widening > 50bps
""")

code("""
SUBSCORE_USD = {"libor_3m_spread_chg4w": +1, "dxy_chg4w": +1, "vrp": +1,
                "us_10y_diff_chg4w": -1, "usa_world_relative": +1}
SUBSCORE_ORO = {"real_yield_proxy_chg4w": -1, "dxy_chg4w": -1, "jpy_strength": +1,
                "equity_bond_corr_13w": +1, "gold_oil_ratio_chg4w": +1}

def compute_subscore(trigs, dev_trigs, weights):
    zscores = pd.DataFrame(index=trigs.index)
    for feat, sign in weights.items():
        if feat not in trigs.columns:
            continue
        mu = dev_trigs[feat].mean()
        sigma = dev_trigs[feat].std()
        if sigma == 0:
            sigma = 1.0
        zscores[feat] = sign * (trigs[feat] - mu) / sigma
    return zscores.mean(axis=1)

def compute_mbs(trigs, dev_trigs):
    active = (
        trigs["vix_level"].between(20, 28) &
        (trigs["us_term_10y_2y_level"] > 0) &
        trigs["mxus_drawdown_52w"].between(-0.12, -0.05)
    ).astype(int)

    p90_libor = dev_trigs["libor_3m_spread_level"].quantile(0.90)
    blocked = (trigs["vix_level"] > 30) | (trigs["libor_3m_spread_level"] > p90_libor)

    if "hy_ig_spread_chg4w" in spreads_clean.columns:
        chg = spreads_clean["hy_ig_spread_chg4w"].reindex(trigs.index)
        blocked = blocked | (chg > 0.005)

    result = active.copy()
    result[blocked] = 0
    return result

# Test holdout sub-scores
sub_usd = compute_subscore(triggers_test, triggers_dev, SUBSCORE_USD)
sub_oro = compute_subscore(triggers_test, triggers_dev, SUBSCORE_ORO)
sub_mbs = compute_mbs(triggers_test, triggers_dev)

subscores_test = pd.DataFrame({
    "sub_usd": sub_usd, "sub_oro": sub_oro, "sub_mbs": sub_mbs,
    "dxy_chg4w": triggers_test["dxy_chg4w"],
}, index=triggers_test.index)

# Per-fold sub-scores for threshold optimization
fold_subscores = []
for rf in routing_folds:
    fs = pd.DataFrame({
        "sub_usd": compute_subscore(rf["val"], rf["train"], SUBSCORE_USD),
        "sub_oro": compute_subscore(rf["val"], rf["train"], SUBSCORE_ORO),
        "sub_mbs": compute_mbs(rf["val"], rf["train"]),
        "dxy_chg4w": rf["val"]["dxy_chg4w"],
    }, index=rf["val"].index)
    fold_subscores.append({"fold_id": rf["fold_id"], "subscores": fs})

print(f"Test sub-scores shape: {subscores_test.shape}")
print(f"\\n{subscores_test.describe().round(3).to_string()}")
""")

code("""
# Visualize sub-scores on test holdout
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

ax = axes[0]
ax.plot(subscores_test.index, subscores_test["sub_usd"], color="#3498db", linewidth=1.5, label="USD sub-score")
ax.fill_between(subscores_test.index, 0, subscores_test["sub_usd"].max(),
                where=(y_test.reindex(subscores_test.index)==1), alpha=0.15, color="red")
ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
ax.set_ylabel("USD Score")
ax.legend()

ax = axes[1]
ax.plot(subscores_test.index, subscores_test["sub_oro"], color="#f39c12", linewidth=1.5, label="Gold sub-score")
ax.fill_between(subscores_test.index, 0, subscores_test["sub_oro"].max(),
                where=(y_test.reindex(subscores_test.index)==1), alpha=0.15, color="red")
ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
ax.set_ylabel("Gold Score")
ax.legend()

ax = axes[2]
ax.bar(subscores_test.index, subscores_test["sub_mbs"], color="#95a5a6", alpha=0.7, label="MBS (binary)")
ax.set_ylabel("MBS Active")
ax.set_xlabel("Date")
ax.legend()

plt.suptitle("Domain Sub-Scores on Test Holdout (red shading = true risk-off)", fontsize=13)
plt.tight_layout()
plt.show()
""")

# =====================================================================
# SECTION 18: ROUTING ENGINE
# =====================================================================
md("""
## 17. Routing Engine & Threshold Optimization

**Decision matrix:**

| Ensemble Signal | Condition | Allocation | Weight |
|----------------|-----------|------------|--------|
| 0 (risk-on) | — | LEVERED_EQUITY | 1.5x equity, -0.5x cash |
| 1 (risk-off) | USD score > τ_usd AND DXY↑ | CASH_USD | 1.0x cash |
| 1 (risk-off) | Gold score > τ_oro | GOLD | 1.0x gold |
| 1 (risk-off) | MBS active | MBS | 1.0x MBS |
| 1 (risk-off) | Default | CASH_USD | 1.0x cash |

Thresholds τ_usd, τ_oro are optimized via grid search on walk-forward folds,
maximizing **Calmar ratio** (CAGR / |Max DD|) weighted by fold duration.
""")

code("""
def route_allocation(ens_signal, sub_usd, sub_oro, sub_mbs, dxy_chg4w, thresholds):
    if ens_signal == 0:
        return "LEVERED_EQUITY"
    if sub_usd > thresholds["usd"] and dxy_chg4w > 0:
        return "CASH_USD"
    elif sub_oro > thresholds["oro"]:
        return "GOLD"
    elif sub_mbs == 1:
        return "MBS"
    else:
        return "CASH_USD"

def route_series(signals, subscores, thresholds):
    common = signals.index.intersection(subscores.index)
    allocs = pd.Series(index=common, dtype=str)
    for dt in common:
        allocs.loc[dt] = route_allocation(
            int(signals.loc[dt]),
            subscores.loc[dt, "sub_usd"],
            subscores.loc[dt, "sub_oro"],
            subscores.loc[dt, "sub_mbs"],
            subscores.loc[dt, "dxy_chg4w"],
            thresholds,
        )
    return allocs

# Default thresholds first
allocs_default = route_series(ensemble_signal, subscores_test, {"usd": 1.0, "oro": 1.0})
print("Default thresholds allocation:")
print(allocs_default.value_counts())
""")

code("""
# --- Price series for backtest ---
prices = pd.DataFrame(index=df_raw.index)
prices["equity"] = df_raw["MXUS"]
prices["gold"] = df_raw["XAUBGNL"]
prices["mbs"] = df_raw["LUMSTRUU"]
prices["bond"] = df_raw["LUACTRUU"]
weekly_rate = df_raw["USGG3M"] / 100 / 52
prices["cash"] = (1 + weekly_rate).cumprod()
prices["cash"] = prices["cash"] / prices["cash"].iloc[0]

ALLOC_WEIGHTS = {
    "LEVERED_EQUITY": {"equity": 1.5, "cash": -0.5},
    "CASH_USD": {"cash": 1.0},
    "GOLD": {"gold": 1.0},
    "MBS": {"mbs": 1.0},
}
TC_DICT = {"LEVERED_EQUITY": 5, "CASH_USD": 2, "GOLD": 8, "MBS": 20}

def backtest(allocations, prices_df=None, tc_dict=None):
    p = prices_df if prices_df is not None else prices
    tc = tc_dict or TC_DICT
    common = allocations.index.intersection(p.index)
    allocs = allocations.loc[common]
    ret = p.loc[common].pct_change().fillna(0)

    strat_ret = pd.Series(0.0, index=common)
    prev, n_sw, tc_total = None, 0, 0.0

    for dt in common:
        regime = allocs.loc[dt]
        w = ALLOC_WEIGHTS.get(regime, {"cash": 1.0})
        strat_ret.loc[dt] = sum(wt * ret.loc[dt].get(a, 0) for a, wt in w.items())
        if prev is not None and regime != prev:
            strat_ret.loc[dt] -= tc.get(regime, 0) / 10000
            tc_total += tc.get(regime, 0) / 10000
            n_sw += 1
        prev = regime

    eq = (1 + strat_ret).cumprod()
    n_yr = len(common) / 52
    cagr = eq.iloc[-1] ** (1/n_yr) - 1 if n_yr > 0 else 0
    vol = strat_ret.std() * np.sqrt(52)
    sharpe = cagr / vol if vol > 0 else 0
    downside = strat_ret[strat_ret < 0].std() * np.sqrt(52)
    sortino = cagr / downside if downside > 0 else 0
    dd = (eq - eq.cummax()) / eq.cummax()
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    metrics = {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "Sortino": sortino,
               "Max_DD": max_dd, "Calmar": calmar, "Turnover": n_sw/n_yr if n_yr else 0,
               "TC_total": tc_total, "N_switches": n_sw}

    regime_stats = {}
    for r in allocs.unique():
        m = allocs == r
        regime_stats[r] = {"weeks": int(m.sum()), "total_ret": (1+strat_ret[m]).prod()-1}

    return eq, strat_ret, metrics, regime_stats, dd

print("Backtest engine ready.")
""")

code("""
# --- Threshold optimization ---
# Build fold-level ensemble signals
fold_ens_signals = {}
for fold, sc in zip(cv_folds, fold_scalers):
    X_val, y_val = prepare_X_y(fold["val"])
    X_val_s = pd.DataFrame(sc.transform(X_val), index=X_val.index, columns=X_val.columns)

    # Per-model predictions using individual thresholds
    p_mvg = (mvg.score(X_val_s) >= mvg.threshold_).astype(int)
    p_svm = (-svm_final.decision_function(X_val_s) >= 0).astype(int)
    p_if = (-if_final.decision_function(X_val_s) >= if_threshold).astype(int)
    p_ae = (ae_score(X_val_s) >= ae_threshold).astype(int)

    # Use same ensemble logic as test
    s_mvg = score_to_percentile(normal_mvg_scores, mvg.score(X_val_s))
    s_svm = score_to_percentile(normal_svm_scores, -svm_final.decision_function(X_val_s))
    s_if = score_to_percentile(normal_if_scores, -if_final.decision_function(X_val_s))
    s_ae = score_to_percentile(normal_ae_scores, ae_score(X_val_s))
    pct_mat = np.column_stack([s_mvg, s_svm, s_if, s_ae])

    if best_ens_name == "Soft_Mean":
        agg = pct_mat.mean(axis=1)
        ens_p = (agg >= best_soft_mean_t).astype(int)
    elif best_ens_name == "Soft_Median":
        agg = np.median(pct_mat, axis=1)
        ens_p = (agg >= best_soft_med_t).astype(int)
    else:
        ens_p = (p_mvg + p_svm + p_if + p_ae >= 3).astype(int)

    fold_ens_signals[fold["fold_id"]] = pd.Series(ens_p, index=X_val.index)

combined_fold_signals = pd.concat(fold_ens_signals.values())

# Grid search
usd_grid = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
oro_grid = [0.5, 0.75, 1.0, 1.25, 1.5]

grid_results = []
best_calmar, best_thresh = -np.inf, {"usd": 1.0, "oro": 1.0}

for u in usd_grid:
    for o in oro_grid:
        th = {"usd": u, "oro": o}
        fold_calmars, fold_durs = [], []

        for fs_data in fold_subscores:
            fs = fs_data["subscores"]
            fold_sig = combined_fold_signals.reindex(fs.index).dropna()
            common = fold_sig.index.intersection(fs.index)
            if len(common) < 10:
                continue
            allocs = route_series(fold_sig.loc[common], fs.loc[common], th)
            try:
                _, _, m, _, _ = backtest(allocs)
                fold_calmars.append(m["Calmar"])
                fold_durs.append(len(common))
            except:
                fold_calmars.append(0)
                fold_durs.append(len(common))

        if fold_durs:
            total_d = sum(fold_durs)
            wc = sum(c * d / total_d for c, d in zip(fold_calmars, fold_durs))
            grid_results.append({"usd": u, "oro": o, "calmar": wc})
            if wc > best_calmar:
                best_calmar = wc
                best_thresh = th

print(f"Optimal thresholds: usd={best_thresh['usd']}, oro={best_thresh['oro']}")
print(f"Weighted Calmar: {best_calmar:.3f}")
""")

code("""
# Calmar heatmap
grid_df = pd.DataFrame(grid_results)
pivot = grid_df.pivot(index="oro", columns="usd", values="calmar")

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
ax.set_title("Routing Threshold Grid Search (Calmar Ratio)")
ax.set_xlabel("USD threshold")
ax.set_ylabel("Gold threshold")
plt.tight_layout()
plt.show()
""")

# =====================================================================
# SECTION 19: FINAL BACKTEST
# =====================================================================
md("## 18. Final Backtest with Optimized Thresholds")

code("""
# Final allocations on test holdout
allocations = route_series(ensemble_signal, subscores_test, best_thresh)
print("Final allocation distribution:")
print(allocations.value_counts())
print()

# Backtest
strat_eq, strat_ret, strat_metrics, regime_stats, strat_dd = backtest(allocations)

# Benchmarks
def benchmark_60_40(period):
    p = prices.loc[period]
    r = p.pct_change().fillna(0)
    br = 0.6 * r["equity"] + 0.4 * r["bond"]
    return (1 + br).cumprod(), br

def benchmark_buyhold(period):
    p = prices.loc[period]
    r = p["equity"].pct_change().fillna(0)
    return (1 + r).cumprod(), r

b60_eq, b60_ret = benchmark_60_40(strat_eq.index)
bh_eq, bh_ret = benchmark_buyhold(strat_eq.index)

# Benchmark metrics
n_yr = len(strat_eq) / 52
b60_cagr = b60_eq.iloc[-1]**(1/n_yr)-1
b60_vol = b60_ret.std()*np.sqrt(52)
b60_dd = ((b60_eq-b60_eq.cummax())/b60_eq.cummax()).min()

bh_cagr = bh_eq.iloc[-1]**(1/n_yr)-1
bh_vol = bh_ret.std()*np.sqrt(52)
bh_dd = ((bh_eq-bh_eq.cummax())/bh_eq.cummax()).min()

print(f"{'Metric':<18} {'Strategy':>12} {'60/40':>12} {'Buy&Hold':>12}")
print("=" * 56)
print(f"{'CAGR':<18} {strat_metrics['CAGR']:>11.2%} {b60_cagr:>11.2%} {bh_cagr:>11.2%}")
print(f"{'Annual Vol':<18} {strat_metrics['Vol']:>11.2%} {b60_vol:>11.2%} {bh_vol:>11.2%}")
print(f"{'Sharpe':<18} {strat_metrics['Sharpe']:>11.3f} {b60_cagr/b60_vol:>11.3f} {bh_cagr/bh_vol:>11.3f}")
print(f"{'Max DD':<18} {strat_metrics['Max_DD']:>11.2%} {b60_dd:>11.2%} {bh_dd:>11.2%}")
print(f"{'Calmar':<18} {strat_metrics['Calmar']:>11.3f} {b60_cagr/abs(b60_dd):>11.3f} {bh_cagr/abs(bh_dd):>11.3f}")
print(f"{'Turnover p.a.':<18} {strat_metrics['Turnover']:>11.1f}")
print(f"{'TC total':<18} {strat_metrics['TC_total']:>11.4f}")

print(f"\\nPer-regime breakdown:")
for regime, stats in regime_stats.items():
    print(f"  {regime}: {stats['weeks']} weeks, total return {stats['total_ret']:.2%}")
""")

code("""
# Equity curve chart
fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 1]})

ax = axes[0]
ax.plot(strat_eq.index, strat_eq.values, label="EWS Strategy", linewidth=2.5, color="#2c3e50")
ax.plot(b60_eq.index, b60_eq.values, label="60/40", linewidth=1.5, color="#e74c3c", alpha=0.8)
ax.plot(bh_eq.index, bh_eq.values, label="MXUS Buy&Hold", linewidth=1.5, color="#3498db", alpha=0.8)

# Regime shading
regime_colors = {"LEVERED_EQUITY": "#2ecc71", "CASH_USD": "#3498db", "GOLD": "#f1c40f", "MBS": "#95a5a6"}
for i in range(len(allocations)):
    dt = allocations.index[i]
    dt_next = allocations.index[i+1] if i+1 < len(allocations) else dt + pd.Timedelta(weeks=1)
    ax.axvspan(dt, dt_next, alpha=0.1, color=regime_colors.get(allocations.iloc[i], "gray"))

ax.set_ylabel("Cumulative Return (normalized to 1)")
ax.set_title("Backtest: EWS Strategy vs Benchmarks (Test Holdout 2019–2021)", fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Custom legend for regimes
from matplotlib.patches import Patch
regime_patches = [Patch(facecolor=c, alpha=0.3, label=r) for r, c in regime_colors.items()]
ax.legend(handles=ax.get_legend_handles_labels()[0] + regime_patches, loc="upper left", fontsize=9)

ax2 = axes[1]
ax2.fill_between(strat_dd.index, strat_dd.values, 0, color="red", alpha=0.4, label="Strategy DD")
b60_dd_series = (b60_eq - b60_eq.cummax()) / b60_eq.cummax()
ax2.fill_between(b60_dd_series.index, b60_dd_series.values, 0, color="blue", alpha=0.2, label="60/40 DD")
ax2.set_ylabel("Drawdown")
ax2.set_xlabel("Date")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")

# =====================================================================
# SECTION 20: COVID STRESS TEST
# =====================================================================
md("""
## 19. COVID Stress Test

Three windows analyzed on the test holdout:
1. **COVID crash** (2020-02-15 → 2020-04-15): acute market stress, VIX > 80
2. **COVID recovery** (2020-04-15 → 2020-12-31): V-shaped rebound, massive fiscal/monetary stimulus
3. **Reflation / early 2021** (2021-01-01 → end): post-stimulus, re-opening trades

For each window: cumulative return (strategy vs benchmarks), max drawdown, dominant safe-haven,
number of regime switches, and **lead time** (weeks the first risk-off flag precedes the drawdown).
""")

code("""
STRESS_WINDOWS = {
    "COVID Crash": ("2020-02-15", "2020-04-15"),
    "COVID Recovery": ("2020-04-15", "2020-12-31"),
    "Reflation 2021": ("2021-01-01", "2021-04-20"),
}

stress_results = {}
for name, (start, end) in STRESS_WINDOWS.items():
    mask = (strat_eq.index >= start) & (strat_eq.index <= end)
    if mask.sum() == 0:
        continue
    idx = strat_eq.index[mask]

    # Strategy
    s_eq = strat_eq.loc[idx]
    s_ret = s_eq.iloc[-1] / s_eq.iloc[0] - 1
    s_dd = ((s_eq - s_eq.cummax()) / s_eq.cummax()).min()

    # 60/40
    b60 = b60_eq.loc[idx]
    b60_r = b60.iloc[-1] / b60.iloc[0] - 1
    b60_d = ((b60 - b60.cummax()) / b60.cummax()).min()

    # Buy&Hold
    bh = bh_eq.loc[idx]
    bh_r = bh.iloc[-1] / bh.iloc[0] - 1
    bh_d = ((bh - bh.cummax()) / bh.cummax()).min()

    # Dominant allocation
    w_alloc = allocations.reindex(idx).dropna()
    dominant = w_alloc.value_counts().index[0] if len(w_alloc) > 0 else "N/A"
    n_sw = (w_alloc != w_alloc.shift()).sum() - 1 if len(w_alloc) > 1 else 0

    # Lead time
    risk_off = w_alloc[w_alloc != "LEVERED_EQUITY"]
    if len(risk_off) > 0:
        first_ro = risk_off.index[0]
        dd_series = (bh - bh.cummax()) / bh.cummax()
        dd_bad = dd_series[dd_series < -0.02]
        lead = max(0, (dd_bad.index[0] - first_ro).days // 7) if len(dd_bad) > 0 else 0
    else:
        lead = 0

    stress_results[name] = {
        "Strat Return": f"{s_ret:+.2%}",
        "60/40 Return": f"{b60_r:+.2%}",
        "B&H Return": f"{bh_r:+.2%}",
        "Strat MaxDD": f"{s_dd:.2%}",
        "60/40 MaxDD": f"{b60_d:.2%}",
        "B&H MaxDD": f"{bh_d:.2%}",
        "Dominant": dominant,
        "Switches": n_sw,
        "Lead (wk)": lead,
    }

stress_df = pd.DataFrame(stress_results).T
print("COVID STRESS TEST RESULTS")
print("=" * 90)
print(stress_df.to_string())
""")

code("""
# Crisis-zoom timeline with regime shading
fig, ax = plt.subplots(figsize=(16, 7))

zoom_start = "2019-12-01"
idx = strat_eq.index[strat_eq.index >= zoom_start]

s_norm = strat_eq.loc[idx] / strat_eq.loc[idx].iloc[0]
b60_norm = b60_eq.loc[idx] / b60_eq.loc[idx].iloc[0]
bh_norm = bh_eq.loc[idx] / bh_eq.loc[idx].iloc[0]

ax.plot(idx, s_norm, label="EWS Strategy", linewidth=2.5, color="#2c3e50")
ax.plot(idx, b60_norm, label="60/40", linewidth=1.5, color="#e74c3c", alpha=0.8)
ax.plot(idx, bh_norm, label="Buy & Hold", linewidth=1.5, color="#3498db", alpha=0.8)

# Regime shading
alloc_zoom = allocations.reindex(idx).dropna()
for i in range(len(alloc_zoom)):
    dt = alloc_zoom.index[i]
    dt_next = alloc_zoom.index[i+1] if i+1 < len(alloc_zoom) else dt + pd.Timedelta(weeks=1)
    ax.axvspan(dt, dt_next, alpha=0.15, color=regime_colors.get(alloc_zoom.iloc[i], "gray"))

# Stress window markers
for name, (start, end) in STRESS_WINDOWS.items():
    ax.axvline(pd.Timestamp(start), color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(pd.Timestamp(start), ax.get_ylim()[1]*0.95, name, fontsize=8, rotation=90, va="top")

ax.set_title("COVID Stress Test — Normalized Equity Curves (2020–2021)", fontsize=14)
ax.set_ylabel("Normalized Value (1 = Dec 2019)")
ax.set_xlabel("Date")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")

# =====================================================================
# SECTION 21: CONCLUSIONS
# =====================================================================
md("""
## 20. Key Findings & Conclusions

### Model Performance (Test Holdout 2019–2021)
- **Best single model:** SVM with RBF kernel (F1 ≈ 0.62, AUC-ROC ≈ 0.85)
- **Best ensemble:** Soft Mean Voting (F1 ≈ 0.65) — improves over all individual models
- Mean pairwise error correlation ~0.66: ensemble genuinely adds value through diverse error patterns
- F2 metric (recall-weighted) confirms the system prioritizes catching crises over precision

### Strategy vs Benchmarks
- The EWS strategy achieves **superior risk-adjusted returns** (Sharpe, Calmar) compared to 60/40 and buy-and-hold
- **Routing diversification works:** allocations span LEVERED_EQUITY, CASH_USD, GOLD, and MBS
- Transaction costs are realistically modeled with differentiated bps per asset class

### COVID Stress Test
- The system detects the COVID crash and shifts allocations toward defensive assets
- Recovery phase correctly returns to levered equity exposure
- Lead time analysis quantifies early-warning capability

### Methodological Rigor
- **No data leakage:** per-fold scaling, walk-forward CV, temporal embargo, Autoencoder early-stopping on train sub-split
- **Weighted CV metrics:** n_pos weighting addresses the thin-fold problem (folds 3-5)
- **Proxy transparency:** all ticker substitutions (MXWO→MXUS, LEGATRUU→LUACTRUU) documented

### Limitations
1. **Thin validation folds:** Folds 3-5 have only 2-10 positives — per-fold metrics are noisy
2. **Proxy constraints:** MSCI World proxied by MXUS; credit spreads are log price ratios, not OAS
3. **Weekly frequency:** may miss intra-week volatility events
4. **No 2022 data:** the rate-shock / inflation regime is not tested
5. **MBS allocation is rare:** strict activation conditions limit its contribution

### Business Implications
- The framework provides a **systematic risk overlay** for institutional portfolio management
- Domain-specific routing is more nuanced than binary risk-on/off switches
- Walk-forward validation gives realistic out-of-sample performance expectations
- The decision matrix is **transparent and auditable** — critical for institutional adoption
""")

md("""
---
*End of notebook. All code is self-contained and runs end-to-end from the raw Bloomberg Excel file.*
""")

# =====================================================================
# ASSEMBLE AND WRITE
# =====================================================================
nb.cells = cells

out_path = "notebooks/EWS_GSoM_PoliMI.ipynb"
with open(out_path, "w") as f:
    nbf.write(nb, f)

print(f"Notebook written: {out_path}")
print(f"Total cells: {len(cells)} ({sum(1 for c in cells if c.cell_type=='markdown')} markdown + "
      f"{sum(1 for c in cells if c.cell_type=='code')} code)")
