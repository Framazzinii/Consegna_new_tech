"""Generate all figures for the Beamer presentation (1.5x leverage version)."""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score, fbeta_score)
from statsmodels.tsa.stattools import adfuller

np.random.seed(42)

FIGDIR = "presentation/figures"
DPI = 250

# ============================================================
# STYLE
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
})
COLORS = {
    'primary': '#1a3a5c',
    'secondary': '#c0392b',
    'accent1': '#2980b9',
    'accent2': '#27ae60',
    'accent3': '#f39c12',
    'accent4': '#8e44ad',
    'gray': '#7f8c8d',
    'equity': '#27ae60',
    'cash': '#2980b9',
    'gold': '#f1c40f',
    'mbs': '#95a5a6',
}

# ============================================================
# DATA LOADING
# ============================================================
print("Loading data...")
DATA_FILE = "data/raw/04_May_Zenti_exercises.xlsx"
TARGET_COL = "Y"
HORIZON_4W = 4
REALIZED_VOL_WEEKS = 4
WEEKS_PER_YEAR = 52

df_raw = pd.read_excel(DATA_FILE, sheet_name="Markets")
df_raw.rename(columns={"Data": "Date"}, inplace=True)
df_raw["Date"] = pd.to_datetime(df_raw["Date"])
df_raw.sort_values("Date", inplace=True)
df_raw.set_index("Date", inplace=True)

# ============================================================
# STATIONARITY
# ============================================================
TRANSFORM_MAP = {
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
    "GT10": "diff", "USGG2YR": "diff", "USGG30YR": "diff",
    "USGG3M": "diff", "US0001M": "diff", "EONIA": "diff",
    "GTDEM2Y": "diff", "GTDEM10Y": "diff", "GTDEM30Y": "diff",
    "GTGBP2Y": "diff", "GTGBP20Y": "diff", "GTGBP30Y": "diff",
    "GTITL2YR": "diff", "GTITL10YR": "diff", "GTITL30YR": "diff",
    "GTJPY2YR": "diff", "GTJPY10YR": "diff", "GTJPY30YR": "diff",
    "VIX": "level", "ECSURPUS": "level", "Y": "level",
}

def make_stationary(df):
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        t = TRANSFORM_MAP.get(col, "level")
        if t == "log_return":
            out[col] = np.log(df[col]).diff()
        elif t == "diff":
            out[col] = df[col].diff()
        else:
            out[col] = df[col]
    return out.iloc[1:]

df_stat = make_stationary(df_raw)

# ============================================================
# SPREADS
# ============================================================
def build_spreads(df, horizon=HORIZON_4W):
    s = pd.DataFrame(index=df.index)
    spread_pairs = {
        "us_term_10y_3m": ("GT10", "USGG3M"), "us_term_10y_2y": ("GT10", "USGG2YR"),
        "de_term_10y_2y": ("GTDEM10Y", "GTDEM2Y"), "it_de_10y": ("GTITL10YR", "GTDEM10Y"),
        "us_de_10y": ("GT10", "GTDEM10Y"),
    }
    for name, (a, b) in spread_pairs.items():
        s[name] = df[a] - df[b]
        s[f"{name}_chg4w"] = s[name].diff(horizon)
    log_pairs = {
        "hy_spread": ("LUMSTRUU", "LF98TRUU"), "hy_ig_spread": ("LUACTRUU", "LF98TRUU"),
        "em_spread": ("LUACTRUU", "EMUSTRUU"),
    }
    for name, (safe, risky) in log_pairs.items():
        s[name] = np.log(df[safe]) - np.log(df[risky])
        s[f"{name}_chg4w"] = s[name].diff(horizon)
    r4w_mxus = np.log(df["MXUS"]).diff(horizon)
    r4w_bond = np.log(df["LUACTRUU"]).diff(horizon)
    s["equity_bond_rot"] = r4w_mxus - r4w_bond
    s["gold_oil_ratio"] = df["XAUBGNL"] / df["Cl1"]
    real_vol = np.log(df["MXUS"]).diff().rolling(REALIZED_VOL_WEEKS).std() * np.sqrt(WEEKS_PER_YEAR) * 100
    s["vrp"] = df["VIX"] - real_vol
    s["jpy_strength"] = -np.log(df["JPY"]).diff(horizon)
    return s.dropna()

spreads = build_spreads(df_raw)

COLLINEAR_DROP = ["hy_spread","hy_spread_chg4w","us_term_10y_3m","us_term_10y_3m_chg4w",
                  "GTGBP20Y","GTDEM30Y","USGG30YR"]
def remove_collinear(df, drop_list):
    return df.drop(columns=[c for c in drop_list if c in df.columns])

df_stat_clean = remove_collinear(df_stat, COLLINEAR_DROP)
spreads_clean = remove_collinear(spreads, COLLINEAR_DROP)

# ============================================================
# ROUTING TRIGGERS
# ============================================================
def build_routing_triggers(df, df_stationary, df_spreads, horizon=HORIZON_4W):
    t = pd.DataFrame(index=df.index)
    t["libor_3m_spread_chg4w"] = (df["US0001M"] - df["USGG3M"]).diff(horizon)
    t["dxy_chg4w"] = np.log(df["DXY"]).diff(horizon)
    t["vrp"] = df_spreads["vrp"].reindex(df.index)
    t["us_10y_diff_chg4w"] = df["GT10"].diff().diff(horizon)
    dm_composite = np.log(df[["MXUS","MXEU","MXJP"]]).diff(horizon).mean(axis=1)
    t["usa_world_relative"] = np.log(df["MXUS"]).diff(horizon) - dm_composite
    t["real_yield_proxy_chg4w"] = (df["GT10"] - np.log(df["LF94TRUU"]).diff(horizon)).diff(horizon)
    t["jpy_strength"] = df_spreads["jpy_strength"].reindex(df.index)
    t["equity_bond_corr_13w"] = df_stationary["MXUS"].rolling(13).corr(df_stationary["LUACTRUU"]).reindex(df.index)
    t["gold_oil_ratio_chg4w"] = df_spreads["gold_oil_ratio"].reindex(df.index).diff(horizon)
    t["vix_level"] = df["VIX"]
    t["us_10y_vol_4w"] = df["GT10"].diff().rolling(REALIZED_VOL_WEEKS).std()
    t["us_term_10y_2y_level"] = df["GT10"] - df["USGG2YR"]
    t["libor_3m_spread_level"] = df["US0001M"] - df["USGG3M"]
    t["mxus_drawdown_52w"] = df["MXUS"] / df["MXUS"].rolling(WEEKS_PER_YEAR).max() - 1
    return t.dropna()

triggers = build_routing_triggers(df_raw, df_stat, spreads)

# ============================================================
# SPLITS
# ============================================================
DEV_END = "2018-12-31"
EMBARGO_WEEKS = 4
WALKFORWARD_FOLDS = [
    (1,"2006-12-31","2009-12-31","GFC 2008"),
    (2,"2009-12-31","2012-12-31","Euro 2011"),
    (3,"2012-12-31","2014-12-31","Taper 2013"),
    (4,"2014-12-31","2016-12-31","China-Oil 2015-16"),
    (5,"2016-12-31","2018-12-31","Q4 2018 selloff"),
]

df_model = df_stat_clean.join(spreads_clean, how="inner")
if "equity_bond_corr_13w" not in df_model.columns:
    df_model = df_model.join(triggers[["equity_bond_corr_13w"]], how="left")
df_model = df_model.dropna()

dev_set = df_model.loc[:DEV_END]
test_holdout = df_model.loc[DEV_END:].iloc[1:]

cv_folds = []
train_normals_per_fold = []
for fold_id, train_end, val_end, crisis in WALKFORWARD_FOLDS:
    tr = df_model.loc[:train_end]
    after_train = df_model.loc[train_end:].iloc[1:]
    val = after_train.iloc[EMBARGO_WEEKS:].loc[:val_end]
    cv_folds.append({"fold_id": fold_id, "train": tr, "val": val, "crisis": crisis})
    train_normals_per_fold.append({"fold_id": fold_id, "train_normals": tr[tr[TARGET_COL]==0]})

triggers_dev = triggers.loc[:DEV_END]
triggers_test = triggers.loc[DEV_END:].iloc[1:]
routing_folds = []
for fold_id, train_end, val_end, crisis in WALKFORWARD_FOLDS:
    tr_r = triggers.loc[:train_end]
    after_r = triggers.loc[train_end:].iloc[1:]
    val_r = after_r.iloc[EMBARGO_WEEKS:].loc[:val_end]
    routing_folds.append({"fold_id": fold_id, "train": tr_r, "val": val_r})

def prepare_X_y(df):
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    X = df.drop(columns=[TARGET_COL], errors="ignore")
    return X, y

fold_scalers = []
for fold in cv_folds:
    X_tr, _ = prepare_X_y(fold["train"])
    sc = StandardScaler(); sc.fit(X_tr)
    fold_scalers.append(sc)

X_dev, y_dev = prepare_X_y(dev_set)
final_scaler = StandardScaler(); final_scaler.fit(X_dev)
X_test, y_test = prepare_X_y(test_holdout)
X_test_scaled = pd.DataFrame(final_scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
dev_normals = X_dev[y_dev==0]
dev_normals_scaled = pd.DataFrame(final_scaler.transform(dev_normals), index=dev_normals.index, columns=dev_normals.columns)

METRIC_NAMES = ["F1","AUC_PR","AUC_ROC","Precision","Recall","F2"]
def compute_metrics(y_true, y_pred, y_score=None):
    m = {"F1": f1_score(y_true, y_pred, zero_division=0),
         "Precision": precision_score(y_true, y_pred, zero_division=0),
         "Recall": recall_score(y_true, y_pred, zero_division=0),
         "F2": fbeta_score(y_true, y_pred, beta=2, zero_division=0)}
    if y_score is not None and len(np.unique(y_true))>1:
        m["AUC_ROC"] = roc_auc_score(y_true, y_score)
        m["AUC_PR"] = average_precision_score(y_true, y_score)
    else:
        m["AUC_ROC"] = np.nan; m["AUC_PR"] = np.nan
    return m

def weighted_metric(fold_metrics, metric="F1"):
    vals = [(fm[metric], fm["n_pos"]) for fm in fold_metrics if not np.isnan(fm.get(metric, np.nan))]
    total_w = sum(w for _,w in vals)
    return sum(v*w for v,w in vals)/total_w if total_w else 0

def score_to_percentile(s_train, s_new):
    sorted_t = np.sort(s_train)
    return np.searchsorted(sorted_t, s_new, side="right") / len(sorted_t)

# ============================================================
# MODELS
# ============================================================
print("Training models...")

# MVG
mvg_lw = LedoitWolf(); mvg_lw.fit(dev_normals_scaled)
all_val_scores_mvg = []
for fold, sc in zip(cv_folds, fold_scalers):
    X_v, _ = prepare_X_y(fold["val"])
    X_v_s = pd.DataFrame(sc.transform(X_v), index=X_v.index, columns=X_v.columns)
    all_val_scores_mvg.extend(mvg_lw.mahalanobis(X_v_s))
cands = np.percentile(all_val_scores_mvg, np.linspace(50,99.9,200))
best_mvg_f1, best_mvg_t = 0, cands[0]
for t in cands:
    fm_list = []
    for fold, sc in zip(cv_folds, fold_scalers):
        X_v, y_v = prepare_X_y(fold["val"])
        X_v_s = pd.DataFrame(sc.transform(X_v), index=X_v.index, columns=X_v.columns)
        sc_arr = mvg_lw.mahalanobis(X_v_s)
        fm = compute_metrics(y_v, (sc_arr>=t).astype(int), sc_arr)
        fm["n_pos"] = int(y_v.sum()); fm_list.append(fm)
    wf1 = weighted_metric(fm_list)
    if wf1 > best_mvg_f1: best_mvg_f1=wf1; best_mvg_t=t

mvg_test_scores = mvg_lw.mahalanobis(X_test_scaled)
mvg_test_preds = (mvg_test_scores >= best_mvg_t).astype(int)
mvg_test_m = compute_metrics(y_test, mvg_test_preds, mvg_test_scores)

# SVM
nu_grid = [0.05,0.10,0.15,0.22]; gamma_grid = ["scale","auto",0.01,0.001]
best_svm_f1, best_svm_p = 0, {}
for nu in nu_grid:
    for gamma in gamma_grid:
        fm_list = []
        for fold, normals, sc in zip(cv_folds, train_normals_per_fold, fold_scalers):
            X_n, _ = prepare_X_y(normals["train_normals"])
            X_n_s = pd.DataFrame(sc.transform(X_n), index=X_n.index, columns=X_n.columns)
            m = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma); m.fit(X_n_s)
            X_v, y_v = prepare_X_y(fold["val"])
            X_v_s = pd.DataFrame(sc.transform(X_v), index=X_v.index, columns=X_v.columns)
            scores = -m.decision_function(X_v_s)
            fm = compute_metrics(y_v, (scores>=0).astype(int), scores)
            fm["n_pos"]=int(y_v.sum()); fm_list.append(fm)
        wf1 = weighted_metric(fm_list)
        if wf1>best_svm_f1: best_svm_f1=wf1; best_svm_p={"nu":nu,"gamma":gamma}

svm_final = OneClassSVM(kernel="rbf", **best_svm_p); svm_final.fit(dev_normals_scaled)
svm_test_scores = -svm_final.decision_function(X_test_scaled)
svm_test_preds = (svm_test_scores>=0).astype(int)
svm_test_m = compute_metrics(y_test, svm_test_preds, svm_test_scores)

# IF
best_if_f1, best_if_c = 0, 0.10
for c in [0.05,0.10,0.15,0.22]:
    fm_list = []
    for fold, normals, sc in zip(cv_folds, train_normals_per_fold, fold_scalers):
        X_n, _ = prepare_X_y(normals["train_normals"])
        X_n_s = pd.DataFrame(sc.transform(X_n), index=X_n.index, columns=X_n.columns)
        m = IsolationForest(contamination=c, n_estimators=200, random_state=42); m.fit(X_n_s)
        X_v, y_v = prepare_X_y(fold["val"])
        X_v_s = pd.DataFrame(sc.transform(X_v), index=X_v.index, columns=X_v.columns)
        scores = -m.decision_function(X_v_s)
        fm = compute_metrics(y_v, (scores>=0).astype(int), scores)
        fm["n_pos"]=int(y_v.sum()); fm_list.append(fm)
    wf1 = weighted_metric(fm_list)
    if wf1>best_if_f1: best_if_f1=wf1; best_if_c=c

if_final = IsolationForest(contamination=best_if_c, n_estimators=200, random_state=42)
if_final.fit(dev_normals_scaled)
# tune IF threshold
all_if_s = []
for fold, sc in zip(cv_folds, fold_scalers):
    X_v, _ = prepare_X_y(fold["val"])
    X_v_s = pd.DataFrame(sc.transform(X_v), index=X_v.index, columns=X_v.columns)
    all_if_s.extend(-if_final.decision_function(X_v_s))
if_pcts = np.percentile(all_if_s, np.linspace(30,99,150))
best_if_t, best_if_wf1 = if_pcts[0], 0
for t in if_pcts:
    fm_list = []
    for fold, sc in zip(cv_folds, fold_scalers):
        X_v, y_v = prepare_X_y(fold["val"])
        X_v_s = pd.DataFrame(sc.transform(X_v), index=X_v.index, columns=X_v.columns)
        scores = -if_final.decision_function(X_v_s)
        fm = compute_metrics(y_v, (scores>=t).astype(int), scores)
        fm["n_pos"]=int(y_v.sum()); fm_list.append(fm)
    wf1 = weighted_metric(fm_list)
    if wf1>best_if_wf1: best_if_wf1=wf1; best_if_t=t

if_test_scores = -if_final.decision_function(X_test_scaled)
if_test_preds = (if_test_scores >= best_if_t).astype(int)
if_test_m = compute_metrics(y_test, if_test_preds, if_test_scores)

# AE
print("Training autoencoder...")
import tensorflow as tf
tf.random.set_seed(42)
input_dim = X_dev.shape[1]
def build_ae(dim):
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
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return m

n_n = len(dev_normals_scaled); split = int(n_n*0.85)
ae_model = build_ae(input_dim)
es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
history = ae_model.fit(dev_normals_scaled.iloc[:split], dev_normals_scaled.iloc[:split],
                       validation_data=(dev_normals_scaled.iloc[split:], dev_normals_scaled.iloc[split:]),
                       epochs=200, batch_size=32, verbose=0, callbacks=[es])

def ae_score(X):
    arr = X.values if hasattr(X,'values') else X
    recon = ae_model.predict(arr, verbose=0)
    return np.mean((arr-recon)**2, axis=1)

all_ae_s = []
for fold, sc in zip(cv_folds, fold_scalers):
    X_v, _ = prepare_X_y(fold["val"])
    X_v_s = pd.DataFrame(sc.transform(X_v), index=X_v.index, columns=X_v.columns)
    all_ae_s.extend(ae_score(X_v_s))
ae_pcts = np.percentile(all_ae_s, np.linspace(50,99.5,150))
best_ae_t, best_ae_f1 = ae_pcts[0], 0
for t in ae_pcts:
    fm_list = []
    for fold, sc in zip(cv_folds, fold_scalers):
        X_v, y_v = prepare_X_y(fold["val"])
        X_v_s = pd.DataFrame(sc.transform(X_v), index=X_v.index, columns=X_v.columns)
        scores = ae_score(X_v_s)
        fm = compute_metrics(y_v, (scores>=t).astype(int), scores)
        fm["n_pos"]=int(y_v.sum()); fm_list.append(fm)
    wf1 = weighted_metric(fm_list)
    if wf1>best_ae_f1: best_ae_f1=wf1; best_ae_t=t

ae_test_scores = ae_score(X_test_scaled)
ae_test_preds = (ae_test_scores >= best_ae_t).astype(int)
ae_test_m = compute_metrics(y_test, ae_test_preds, ae_test_scores)

print(f"MVG test F1={mvg_test_m['F1']:.3f}, SVM={svm_test_m['F1']:.3f}, IF={if_test_m['F1']:.3f}, AE={ae_test_m['F1']:.3f}")

# ============================================================
# ENSEMBLE
# ============================================================
print("Building ensemble...")
n_mvg_s = mvg_lw.mahalanobis(dev_normals_scaled)
n_svm_s = -svm_final.decision_function(dev_normals_scaled)
n_if_s = -if_final.decision_function(dev_normals_scaled)
n_ae_s = ae_score(dev_normals_scaled)

pct_mvg = score_to_percentile(n_mvg_s, mvg_test_scores)
pct_svm = score_to_percentile(n_svm_s, svm_test_scores)
pct_if = score_to_percentile(n_if_s, if_test_scores)
pct_ae = score_to_percentile(n_ae_s, ae_test_scores)
pct_matrix = np.column_stack([pct_mvg, pct_svm, pct_if, pct_ae])
pct_mean = pct_matrix.mean(axis=1)
pct_median = np.median(pct_matrix, axis=1)
hard_preds = (mvg_test_preds + svm_test_preds + if_test_preds + ae_test_preds >= 3).astype(int)

# Tune soft thresholds
best_sm_t, best_sm_f1 = 0.5, 0
best_smed_t, best_smed_f1 = 0.5, 0
for t in np.linspace(0.5, 0.99, 100):
    for mode in ["mean","median"]:
        fm_list = []
        for fold, sc in zip(cv_folds, fold_scalers):
            X_v, y_v = prepare_X_y(fold["val"])
            X_v_s = pd.DataFrame(sc.transform(X_v), index=X_v.index, columns=X_v.columns)
            p_m = score_to_percentile(n_mvg_s, mvg_lw.mahalanobis(X_v_s))
            p_s = score_to_percentile(n_svm_s, -svm_final.decision_function(X_v_s))
            p_i = score_to_percentile(n_if_s, -if_final.decision_function(X_v_s))
            p_a = score_to_percentile(n_ae_s, ae_score(X_v_s))
            mat = np.column_stack([p_m,p_s,p_i,p_a])
            agg = mat.mean(axis=1) if mode=="mean" else np.median(mat, axis=1)
            fm = compute_metrics(y_v, (agg>=t).astype(int), agg)
            fm["n_pos"]=int(y_v.sum()); fm_list.append(fm)
        wf1 = weighted_metric(fm_list)
        if mode=="mean" and wf1>best_sm_f1: best_sm_f1=wf1; best_sm_t=t
        elif mode=="median" and wf1>best_smed_f1: best_smed_f1=wf1; best_smed_t=t

soft_mean_preds = (pct_mean >= best_sm_t).astype(int)
soft_med_preds = (pct_median >= best_smed_t).astype(int)

all_test_results = {"MVG":mvg_test_m, "SVM":svm_test_m, "IF":if_test_m, "AE":ae_test_m}
ens_results = {
    "Hard Vote": compute_metrics(y_test, hard_preds, pct_mean),
    "Soft Mean": compute_metrics(y_test, soft_mean_preds, pct_mean),
    "Soft Median": compute_metrics(y_test, soft_med_preds, pct_median),
}
best_ens = max(ens_results, key=lambda k: ens_results[k]["F1"])
if best_ens == "Soft Mean": ensemble_preds = soft_mean_preds
elif best_ens == "Soft Median": ensemble_preds = soft_med_preds
else: ensemble_preds = hard_preds
ensemble_signal = pd.Series(ensemble_preds, index=X_test.index)

print(f"Best ensemble: {best_ens} F1={ens_results[best_ens]['F1']:.3f}")

# ============================================================
# ROUTING + BACKTEST
# ============================================================
print("Routing & backtest...")
SUBSCORE_USD = {"libor_3m_spread_chg4w":+1,"dxy_chg4w":+1,"vrp":+1,"us_10y_diff_chg4w":-1,"usa_world_relative":+1}
SUBSCORE_ORO = {"real_yield_proxy_chg4w":-1,"dxy_chg4w":-1,"jpy_strength":+1,"equity_bond_corr_13w":+1,"gold_oil_ratio_chg4w":+1}

def compute_subscore(trigs, dev_trigs, weights):
    zs = pd.DataFrame(index=trigs.index)
    for feat, sign in weights.items():
        if feat not in trigs.columns: continue
        mu = dev_trigs[feat].mean(); sigma = dev_trigs[feat].std()
        if sigma==0: sigma=1
        zs[feat] = sign * (trigs[feat]-mu)/sigma
    return zs.mean(axis=1)

def compute_mbs(trigs, dev_trigs):
    active = (trigs["vix_level"].between(20,28) & (trigs["us_term_10y_2y_level"]>0) & trigs["mxus_drawdown_52w"].between(-0.12,-0.05)).astype(int)
    p90 = dev_trigs["libor_3m_spread_level"].quantile(0.90)
    blocked = (trigs["vix_level"]>30) | (trigs["libor_3m_spread_level"]>p90)
    if "hy_ig_spread_chg4w" in spreads_clean.columns:
        chg = spreads_clean["hy_ig_spread_chg4w"].reindex(trigs.index)
        blocked = blocked | (chg>0.005)
    r = active.copy(); r[blocked]=0; return r

sub_usd = compute_subscore(triggers_test, triggers_dev, SUBSCORE_USD)
sub_oro = compute_subscore(triggers_test, triggers_dev, SUBSCORE_ORO)
sub_mbs = compute_mbs(triggers_test, triggers_dev)
subscores_test = pd.DataFrame({"sub_usd":sub_usd,"sub_oro":sub_oro,"sub_mbs":sub_mbs,"dxy_chg4w":triggers_test["dxy_chg4w"]}, index=triggers_test.index)

fold_subscores = []
for rf in routing_folds:
    fs = pd.DataFrame({"sub_usd":compute_subscore(rf["val"],rf["train"],SUBSCORE_USD),
                        "sub_oro":compute_subscore(rf["val"],rf["train"],SUBSCORE_ORO),
                        "sub_mbs":compute_mbs(rf["val"],rf["train"]),
                        "dxy_chg4w":rf["val"]["dxy_chg4w"]}, index=rf["val"].index)
    fold_subscores.append({"fold_id":rf["fold_id"],"subscores":fs})

def route(sig, su, so, sm, dxy, th):
    if sig==0: return "LEVERED_EQUITY"
    if su>th["usd"] and dxy>0: return "CASH_USD"
    elif so>th["oro"]: return "GOLD"
    elif sm==1: return "MBS"
    else: return "CASH_USD"

def route_series(signals, subscores, th):
    common = signals.index.intersection(subscores.index)
    allocs = pd.Series(index=common, dtype=str)
    for dt in common:
        allocs.loc[dt] = route(int(signals.loc[dt]), subscores.loc[dt,"sub_usd"],
                               subscores.loc[dt,"sub_oro"], subscores.loc[dt,"sub_mbs"],
                               subscores.loc[dt,"dxy_chg4w"], th)
    return allocs

# Prices
prices = pd.DataFrame(index=df_raw.index)
prices["equity"] = df_raw["MXUS"]; prices["gold"] = df_raw["XAUBGNL"]
prices["mbs"] = df_raw["LUMSTRUU"]; prices["bond"] = df_raw["LUACTRUU"]
weekly_rate = df_raw["USGG3M"]/100/52
prices["cash"] = (1+weekly_rate).cumprod(); prices["cash"] = prices["cash"]/prices["cash"].iloc[0]

ALLOC_W = {"LEVERED_EQUITY":{"equity":1.5,"cash":-0.5},"CASH_USD":{"cash":1.0},"GOLD":{"gold":1.0},"MBS":{"mbs":1.0}}
TC = {"LEVERED_EQUITY":5,"CASH_USD":2,"GOLD":8,"MBS":20}

def backtest(allocs):
    common = allocs.index.intersection(prices.index)
    al = allocs.loc[common]; ret = prices.loc[common].pct_change().fillna(0)
    sr = pd.Series(0.0, index=common); prev=None; nsw=0; tc_tot=0
    for dt in common:
        r = al.loc[dt]; w = ALLOC_W.get(r, {"cash":1.0})
        sr.loc[dt] = sum(wt*ret.loc[dt].get(a,0) for a,wt in w.items())
        if prev and r!=prev: sr.loc[dt]-=TC.get(r,0)/10000; tc_tot+=TC.get(r,0)/10000; nsw+=1
        prev=r
    eq = (1+sr).cumprod(); ny = len(common)/52
    cagr = eq.iloc[-1]**(1/ny)-1 if ny>0 else 0
    vol = sr.std()*np.sqrt(52); sharpe = cagr/vol if vol else 0
    ds = sr[sr<0].std()*np.sqrt(52); sortino = cagr/ds if ds else 0
    dd = (eq-eq.cummax())/eq.cummax(); mdd = dd.min()
    calmar = cagr/abs(mdd) if mdd!=0 else 0
    m = {"CAGR":cagr,"Vol":vol,"Sharpe":sharpe,"Sortino":sortino,"Max_DD":mdd,"Calmar":calmar,"Turnover":nsw/ny if ny else 0,"TC":tc_tot,"N_sw":nsw}
    rs = {}
    for reg in al.unique():
        mask = al==reg; rs[reg] = {"weeks":int(mask.sum()), "ret":(1+sr[mask]).prod()-1}
    return eq, sr, m, rs, dd

# Fold-level ensemble signals for optimization
fold_ens = {}
for fold, sc in zip(cv_folds, fold_scalers):
    X_v, _ = prepare_X_y(fold["val"])
    X_v_s = pd.DataFrame(sc.transform(X_v), index=X_v.index, columns=X_v.columns)
    p_m = score_to_percentile(n_mvg_s, mvg_lw.mahalanobis(X_v_s))
    p_s = score_to_percentile(n_svm_s, -svm_final.decision_function(X_v_s))
    p_i = score_to_percentile(n_if_s, -if_final.decision_function(X_v_s))
    p_a = score_to_percentile(n_ae_s, ae_score(X_v_s))
    mat = np.column_stack([p_m,p_s,p_i,p_a])
    if best_ens=="Soft Mean": agg=mat.mean(axis=1); ep=(agg>=best_sm_t).astype(int)
    elif best_ens=="Soft Median": agg=np.median(mat,axis=1); ep=(agg>=best_smed_t).astype(int)
    else: ep=(mvg_test_preds+svm_test_preds+if_test_preds+ae_test_preds>=3).astype(int)[:len(X_v)]
    fold_ens[fold["fold_id"]] = pd.Series(ep, index=X_v.index)
combined_fold_sig = pd.concat(fold_ens.values())

# Grid search
usd_grid = [0.5,0.75,1.0,1.25,1.5,2.0]; oro_grid = [0.5,0.75,1.0,1.25,1.5]
grid_res = []; best_cal=-np.inf; best_th={"usd":1.0,"oro":1.0}
for u in usd_grid:
    for o in oro_grid:
        th = {"usd":u,"oro":o}; fc=[]; fd=[]
        for fsd in fold_subscores:
            fs = fsd["subscores"]; fsig = combined_fold_sig.reindex(fs.index).dropna()
            common = fsig.index.intersection(fs.index)
            if len(common)<10: continue
            al = route_series(fsig.loc[common], fs.loc[common], th)
            try:
                _,_,m,_,_ = backtest(al); fc.append(m["Calmar"]); fd.append(len(common))
            except: fc.append(0); fd.append(len(common))
        if fd:
            td = sum(fd); wc = sum(c*d/td for c,d in zip(fc,fd))
            grid_res.append({"usd":u,"oro":o,"calmar":wc})
            if wc>best_cal: best_cal=wc; best_th=th

print(f"Best thresholds: {best_th}, Calmar: {best_cal:.3f}")

allocations = route_series(ensemble_signal, subscores_test, best_th)
strat_eq, strat_ret, strat_m, regime_stats, strat_dd = backtest(allocations)

# Benchmarks
def bench60(period):
    p=prices.loc[period]; r=p.pct_change().fillna(0)
    br=0.6*r["equity"]+0.4*r["bond"]; return (1+br).cumprod(), br
def bench_bh(period):
    p=prices.loc[period]; r=p["equity"].pct_change().fillna(0); return (1+r).cumprod(), r

b60_eq, b60_ret = bench60(strat_eq.index)
bh_eq, bh_ret = bench_bh(strat_eq.index)

print(f"Strategy: CAGR={strat_m['CAGR']:.2%}, Sharpe={strat_m['Sharpe']:.3f}, MaxDD={strat_m['Max_DD']:.2%}")
print(f"Allocations: {allocations.value_counts().to_dict()}")

# ============================================================
# FIGURE 1: Y distribution by year
# ============================================================
print("Generating figures...")

yearly = df_raw.groupby(df_raw.index.year)[TARGET_COL].agg(['sum','count'])
yearly.columns = ['n_riskoff','n_total']
fig, ax = plt.subplots(figsize=(10, 3.5))
colors_bar = [COLORS['secondary'] if v>10 else COLORS['accent1'] for v in yearly['n_riskoff']]
ax.bar(yearly.index, yearly['n_riskoff'], color=colors_bar, edgecolor='white', linewidth=0.5)
ax.set_xlabel('Year'); ax.set_ylabel('Risk-Off Weeks (Y=1)')
ax.set_title('Distribution of Risk-Off Weeks by Year')
ax.set_xticks(yearly.index); ax.set_xticklabels(yearly.index, rotation=45, fontsize=8)
plt.tight_layout(); fig.savefig(f"{FIGDIR}/fig_y_distribution.png", dpi=DPI, bbox_inches='tight'); plt.close()

# ============================================================
# FIGURE 2: Key features during crises
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for ax, col, title in zip(axes.flat, ['VIX','GT10','DXY','MXUS'],
                           ['VIX Index','US 10Y Yield','DXY Dollar Index','MSCI USA (MXUS)']):
    ax.plot(df_raw.index, df_raw[col], color=COLORS['primary'], linewidth=0.7)
    ax.fill_between(df_raw.index, df_raw[col].min(), df_raw[col],
                    where=df_raw[TARGET_COL]==1, alpha=0.2, color=COLORS['secondary'])
    ax.set_title(title, fontsize=10)
plt.suptitle('Key Features with Risk-Off Shading (red)', fontsize=12, y=1.01)
plt.tight_layout(); fig.savefig(f"{FIGDIR}/fig_key_features.png", dpi=DPI, bbox_inches='tight'); plt.close()

# ============================================================
# FIGURE 3: Correlation heatmap
# ============================================================
combined = df_stat.drop(columns=[TARGET_COL]).join(spreads, how="inner")
corr = combined.corr()
fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
            xticklabels=True, yticklabels=True)
ax.tick_params(labelsize=5)
ax.set_title("Feature-Spread Correlation Matrix", fontsize=12)
plt.tight_layout(); fig.savefig(f"{FIGDIR}/fig_correlation.png", dpi=DPI, bbox_inches='tight'); plt.close()

# ============================================================
# FIGURE 4: Routing triggers correlation
# ============================================================
ROUTING_DOMAINS = {
    "USD": ["libor_3m_spread_chg4w","dxy_chg4w","vrp","us_10y_diff_chg4w","usa_world_relative"],
    "Oro": ["real_yield_proxy_chg4w","dxy_chg4w","jpy_strength","equity_bond_corr_13w","gold_oil_ratio_chg4w"],
    "MBS": ["vix_level","us_10y_vol_4w","us_term_10y_2y_level","libor_3m_spread_level","mxus_drawdown_52w"],
}
domain_order = []
for d in ["USD","Oro","MBS"]:
    for f in ROUTING_DOMAINS[d]:
        if f not in domain_order: domain_order.append(f)
corr_rt = triggers[domain_order].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_rt, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, annot_kws={"size":7})
ax.set_title("Routing Triggers Correlation (domain-ordered)", fontsize=11)
plt.tight_layout(); fig.savefig(f"{FIGDIR}/fig_routing_corr.png", dpi=DPI, bbox_inches='tight'); plt.close()

# ============================================================
# FIGURE 5: MVG score distribution
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4))
anom_dev = X_dev[y_dev==1]
anom_s = pd.DataFrame(final_scaler.transform(anom_dev), index=anom_dev.index, columns=anom_dev.columns)
ns = mvg_lw.mahalanobis(dev_normals_scaled)
als = mvg_lw.mahalanobis(anom_s)
ax.hist(ns, bins=80, alpha=0.6, color=COLORS['accent2'], label=f'Normal (n={len(ns)})', density=True)
ax.hist(als, bins=40, alpha=0.6, color=COLORS['secondary'], label=f'Risk-Off (n={len(als)})', density=True)
ax.axvline(best_mvg_t, color='black', linestyle='--', linewidth=2, label=f'Threshold = {best_mvg_t:.0f}')
ax.set_xlabel('Squared Mahalanobis Distance'); ax.set_ylabel('Density')
ax.set_title('MVG: Score Distribution (Development Set)'); ax.legend()
ax.set_xlim(0, np.percentile(np.concatenate([ns,als]),99))
plt.tight_layout(); fig.savefig(f"{FIGDIR}/fig_mvg_scores.png", dpi=DPI, bbox_inches='tight'); plt.close()

# ============================================================
# FIGURE 6: AE training curves
# ============================================================
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(history.history['loss'], label='Train Loss', color=COLORS['accent1'])
ax.plot(history.history['val_loss'], label='Val Loss (temporal sub-split)', color=COLORS['secondary'])
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE'); ax.set_title('Autoencoder Training Curves'); ax.legend()
plt.tight_layout(); fig.savefig(f"{FIGDIR}/fig_ae_training.png", dpi=DPI, bbox_inches='tight'); plt.close()

# ============================================================
# FIGURE 7: Model comparison
# ============================================================
all_res = {**all_test_results, **ens_results}
full_df = pd.DataFrame(all_res).T[METRIC_NAMES]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
full_df[["F1","Precision","Recall","F2"]].plot(kind="bar", ax=ax, rot=0,
    color=[COLORS['primary'],COLORS['accent2'],COLORS['secondary'],COLORS['accent3']])
ax.set_title("Classification Metrics"); ax.set_ylabel("Score"); ax.set_ylim(0,1.15)
ax.legend(fontsize=8)
ax = axes[1]
full_df[["AUC_ROC","AUC_PR"]].plot(kind="bar", ax=ax, rot=0, color=[COLORS['secondary'],COLORS['accent1']])
ax.set_title("Ranking Metrics"); ax.set_ylabel("AUC"); ax.set_ylim(0.5,1.05)
ax.legend(fontsize=8)
plt.tight_layout(); fig.savefig(f"{FIGDIR}/fig_model_comparison.png", dpi=DPI, bbox_inches='tight'); plt.close()

# ============================================================
# FIGURE 8: Error correlation
# ============================================================
errors = pd.DataFrame({
    "MVG":(mvg_test_preds!=y_test.values).astype(int),
    "SVM":(svm_test_preds!=y_test.values).astype(int),
    "IF":(if_test_preds!=y_test.values).astype(int),
    "AE":(ae_test_preds!=y_test.values).astype(int),
})
err_corr = errors.corr()
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(err_corr, annot=True, fmt=".3f", cmap="Reds", vmin=0, vmax=1, ax=ax)
ax.set_title(f"Error Correlation Matrix\n(mean pairwise = {err_corr.where(np.triu(np.ones_like(err_corr,dtype=bool),k=1)).stack().mean():.3f})")
plt.tight_layout(); fig.savefig(f"{FIGDIR}/fig_error_corr.png", dpi=DPI, bbox_inches='tight'); plt.close()

# ============================================================
# FIGURE 9: Ensemble timeline
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios":[2,1]})
ax = axes[0]
ax.plot(X_test.index, pct_mean, color=COLORS['primary'], linewidth=1, label="Ensemble Score")
ax.axhline(best_sm_t, color=COLORS['secondary'], linestyle='--', label=f"Threshold={best_sm_t:.2f}")
ax.fill_between(X_test.index, 0, 1, where=y_test==1, alpha=0.15, color=COLORS['secondary'], label="True Risk-Off")
ax.set_ylabel("Percentile Score"); ax.set_title("Ensemble Score on Test Holdout (2019-2021)")
ax.legend(loc="upper left", fontsize=8)
ax2 = axes[1]
ax2.fill_between(X_test.index, 0, y_test.values, alpha=0.4, color=COLORS['secondary'], label="True Y=1")
ax2.fill_between(X_test.index, 0, -ensemble_preds, alpha=0.4, color=COLORS['accent1'], label="Predicted Y=1")
ax2.set_ylabel("Signal"); ax2.legend(fontsize=8)
plt.tight_layout(); fig.savefig(f"{FIGDIR}/fig_ensemble_timeline.png", dpi=DPI, bbox_inches='tight'); plt.close()

# ============================================================
# FIGURE 10: Sub-scores timeline
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
y_reind = y_test.reindex(subscores_test.index)
ax = axes[0]
ax.plot(subscores_test.index, subscores_test["sub_usd"], color=COLORS['accent1'], linewidth=1.5, label="USD sub-score")
ax.fill_between(subscores_test.index, subscores_test["sub_usd"].min(), subscores_test["sub_usd"].max(),
                where=(y_reind==1), alpha=0.1, color=COLORS['secondary'])
ax.axhline(0, color='gray', alpha=0.5); ax.set_ylabel("USD Score"); ax.legend(fontsize=8)
ax = axes[1]
ax.plot(subscores_test.index, subscores_test["sub_oro"], color=COLORS['accent3'], linewidth=1.5, label="Gold sub-score")
ax.fill_between(subscores_test.index, subscores_test["sub_oro"].min(), subscores_test["sub_oro"].max(),
                where=(y_reind==1), alpha=0.1, color=COLORS['secondary'])
ax.axhline(0, color='gray', alpha=0.5); ax.set_ylabel("Gold Score"); ax.legend(fontsize=8)
ax = axes[2]
ax.bar(subscores_test.index, subscores_test["sub_mbs"], color=COLORS['mbs'], alpha=0.7, label="MBS (binary)")
ax.set_ylabel("MBS Active"); ax.legend(fontsize=8)
plt.suptitle("Domain Sub-Scores on Test Holdout (red shading = true risk-off)", fontsize=11)
plt.tight_layout(); fig.savefig(f"{FIGDIR}/fig_subscores.png", dpi=DPI, bbox_inches='tight'); plt.close()

# ============================================================
# FIGURE 11: Calmar heatmap
# ============================================================
grid_df = pd.DataFrame(grid_res)
if len(grid_df) > 0:
    pivot = grid_df.pivot(index="oro", columns="usd", values="calmar")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
    ax.set_title("Routing Threshold Optimization (Calmar Ratio)"); ax.set_xlabel("USD threshold"); ax.set_ylabel("Gold threshold")
    plt.tight_layout(); fig.savefig(f"{FIGDIR}/fig_calmar_heatmap.png", dpi=DPI, bbox_inches='tight'); plt.close()

# ============================================================
# FIGURE 12: Equity curves
# ============================================================
regime_colors = {"LEVERED_EQUITY":COLORS['equity'],"CASH_USD":COLORS['cash'],"GOLD":COLORS['gold'],"MBS":COLORS['mbs']}
fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios":[3,1]})
ax = axes[0]
ax.plot(strat_eq.index, strat_eq.values, label="EWS Strategy (1.5x)", linewidth=2.5, color=COLORS['primary'])
ax.plot(b60_eq.index, b60_eq.values, label="60/40", linewidth=1.5, color=COLORS['secondary'], alpha=0.8)
ax.plot(bh_eq.index, bh_eq.values, label="MXUS Buy&Hold", linewidth=1.5, color=COLORS['accent1'], alpha=0.8)
for i in range(len(allocations)):
    dt = allocations.index[i]
    dt_n = allocations.index[i+1] if i+1<len(allocations) else dt+pd.Timedelta(weeks=1)
    ax.axvspan(dt, dt_n, alpha=0.12, color=regime_colors.get(allocations.iloc[i],'gray'))
ax.set_ylabel("Cumulative Return"); ax.set_title("Backtest: EWS Strategy vs Benchmarks (Test Holdout 2019-2021)", fontsize=12)
from matplotlib.patches import Patch
handles = ax.get_legend_handles_labels()[0]
rp = [Patch(facecolor=c, alpha=0.3, label=r.replace("LEVERED_","")) for r,c in regime_colors.items()]
ax.legend(handles=handles+rp, loc="upper left", fontsize=8)
ax2 = axes[1]
ax2.fill_between(strat_dd.index, strat_dd.values, 0, color=COLORS['secondary'], alpha=0.4, label="Strategy DD")
b60_dd = (b60_eq-b60_eq.cummax())/b60_eq.cummax()
ax2.fill_between(b60_dd.index, b60_dd.values, 0, color=COLORS['accent1'], alpha=0.2, label="60/40 DD")
ax2.set_ylabel("Drawdown"); ax2.legend(fontsize=8)
plt.tight_layout(); fig.savefig(f"{FIGDIR}/fig_equity_curves.png", dpi=DPI, bbox_inches='tight'); plt.close()

# ============================================================
# FIGURE 13: COVID stress test zoom
# ============================================================
fig, ax = plt.subplots(figsize=(13, 6))
zoom_start = "2019-12-01"
idx = strat_eq.index[strat_eq.index>=zoom_start]
s_n = strat_eq.loc[idx]/strat_eq.loc[idx].iloc[0]
b60_n = b60_eq.loc[idx]/b60_eq.loc[idx].iloc[0]
bh_n = bh_eq.loc[idx]/bh_eq.loc[idx].iloc[0]
ax.plot(idx, s_n, label="EWS Strategy (1.5x)", linewidth=2.5, color=COLORS['primary'])
ax.plot(idx, b60_n, label="60/40", linewidth=1.5, color=COLORS['secondary'], alpha=0.8)
ax.plot(idx, bh_n, label="Buy & Hold", linewidth=1.5, color=COLORS['accent1'], alpha=0.8)
az = allocations.reindex(idx).dropna()
for i in range(len(az)):
    dt = az.index[i]; dt_n = az.index[i+1] if i+1<len(az) else dt+pd.Timedelta(weeks=1)
    ax.axvspan(dt, dt_n, alpha=0.15, color=regime_colors.get(az.iloc[i],'gray'))
STRESS_W = {"COVID Crash":("2020-02-15","2020-04-15"),"COVID Recovery":("2020-04-15","2020-12-31"),"Reflation 2021":("2021-01-01","2021-04-20")}
for name,(start,end) in STRESS_W.items():
    ax.axvline(pd.Timestamp(start), color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(pd.Timestamp(start), ax.get_ylim()[1]*0.97 if 'Crash' in name else ax.get_ylim()[1]*0.93,
            name, fontsize=7, rotation=90, va='top', color=COLORS['secondary'])
ax.set_title("COVID Stress Test: Normalized Equity (Dec 2019 = 1.0)", fontsize=12)
ax.set_ylabel("Normalized Value"); ax.legend(loc="upper left", fontsize=8)
plt.tight_layout(); fig.savefig(f"{FIGDIR}/fig_stress_timeline.png", dpi=DPI, bbox_inches='tight'); plt.close()

# ============================================================
# FIGURE 14: COVID trigger analysis
# ============================================================
covid_crash = subscores_test.loc["2020-02-01":"2020-04-15"]
covid_recovery = subscores_test.loc["2020-04-15":"2020-12-31"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
crash_triggers = triggers_test.loc["2020-02-01":"2020-04-15"]
crash_zscores_usd = pd.DataFrame()
crash_zscores_oro = pd.DataFrame()
for feat, sign in SUBSCORE_USD.items():
    if feat in crash_triggers.columns:
        mu = triggers_dev[feat].mean(); sigma = triggers_dev[feat].std()
        crash_zscores_usd[feat] = sign * (crash_triggers[feat]-mu)/sigma
for feat, sign in SUBSCORE_ORO.items():
    if feat in crash_triggers.columns:
        crash_zscores_oro[feat] = sign * (crash_triggers[feat]-mu)/sigma

mean_usd = crash_zscores_usd.mean()
mean_oro = crash_zscores_oro.mean()
combined_means = pd.concat([mean_usd.rename(lambda x: f"USD:{x}"), mean_oro.rename(lambda x: f"ORO:{x}")])
colors_trig = [COLORS['accent1']]*len(mean_usd) + [COLORS['accent3']]*len(mean_oro)
combined_means.plot(kind="barh", ax=ax, color=colors_trig)
ax.set_title("COVID Crash: Mean Signed Z-Scores", fontsize=10)
ax.set_xlabel("Z-Score (positive = risk-off signal)")
ax.axvline(0, color='gray', alpha=0.5)

ax = axes[1]
# Allocation breakdown during COVID
crash_alloc = allocations.loc["2020-02-01":"2020-04-15"]
recov_alloc = allocations.loc["2020-04-15":"2020-12-31"]

alloc_data = {"COVID Crash": crash_alloc.value_counts(), "COVID Recovery": recov_alloc.value_counts()}
alloc_combined = pd.DataFrame(alloc_data).fillna(0).T
alloc_combined = alloc_combined[[c for c in ["LEVERED_EQUITY","CASH_USD","GOLD","MBS"] if c in alloc_combined.columns]]
alloc_combined.plot(kind="bar", stacked=True, ax=ax,
                    color=[regime_colors.get(c,'gray') for c in alloc_combined.columns])
ax.set_title("Allocation Breakdown (COVID Windows)", fontsize=10)
ax.set_ylabel("Weeks"); ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(fontsize=7)

plt.tight_layout(); fig.savefig(f"{FIGDIR}/fig_covid_triggers.png", dpi=DPI, bbox_inches='tight'); plt.close()

print(f"\nAll figures saved to {FIGDIR}/")
print(f"Files: {os.listdir(FIGDIR)}")

# Save key metrics for the TeX
with open(f"{FIGDIR}/../metrics.txt", "w") as f:
    f.write(f"CAGR={strat_m['CAGR']:.2%}\n")
    f.write(f"Sharpe={strat_m['Sharpe']:.3f}\n")
    f.write(f"MaxDD={strat_m['Max_DD']:.2%}\n")
    f.write(f"Calmar={strat_m['Calmar']:.3f}\n")
    f.write(f"Sortino={strat_m['Sortino']:.3f}\n")
    f.write(f"Turnover={strat_m['Turnover']:.1f}\n")
    f.write(f"best_ens={best_ens}\n")
    f.write(f"best_ens_f1={ens_results[best_ens]['F1']:.3f}\n")
    f.write(f"best_thresh={best_th}\n")
    f.write(f"alloc={allocations.value_counts().to_dict()}\n")
    ny = len(strat_eq)/52
    b60_cagr = b60_eq.iloc[-1]**(1/ny)-1
    b60_dd = ((b60_eq-b60_eq.cummax())/b60_eq.cummax()).min()
    bh_cagr = bh_eq.iloc[-1]**(1/ny)-1
    bh_dd = ((bh_eq-bh_eq.cummax())/bh_eq.cummax()).min()
    f.write(f"b60_cagr={b60_cagr:.2%}\nb60_dd={b60_dd:.2%}\nbh_cagr={bh_cagr:.2%}\nbh_dd={bh_dd:.2%}\n")
