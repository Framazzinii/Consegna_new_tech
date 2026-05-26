"""Prompts 5-6 — Anomaly detection models: MVG, One-Class SVM, Autoencoder, Isolation Forest."""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.covariance import LedoitWolf
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, fbeta_score,
)

from src.data_loader import PROJECT_ROOT, TARGET_COL
from src.preprocessing import FoldScaler, prepare_X_y
from src.splits import walkforward_split

MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGS_DIR = PROJECT_ROOT / "outputs" / "figures"

METRICS_NAMES = ["F1", "AUC_PR", "AUC_ROC", "Precision", "Recall", "F2"]


def compute_metrics(y_true, y_pred, y_score=None):
    """Compute the 6 standard metrics."""
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


def weighted_metric(fold_metrics, metric_name="F1"):
    """Weighted mean of a metric across folds, weighted by n_pos."""
    vals = [(fm[metric_name], fm["n_pos"]) for fm in fold_metrics
            if not np.isnan(fm.get(metric_name, np.nan))]
    if not vals:
        return 0.0
    total_w = sum(w for _, w in vals)
    if total_w == 0:
        return 0.0
    return sum(v * w for v, w in vals) / total_w


# ──────────────────────────────────────────────
# MVG Anomaly Detector (Prompt 5)
# ──────────────────────────────────────────────

class MVGAnomalyDetector:
    """Multivariate Gaussian with Ledoit-Wolf shrinkage. Score = squared Mahalanobis distance."""

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
        scores = self.score(X)
        return (scores >= t).astype(int)

    def tune_threshold(self, fold_data, scaler_per_fold):
        """Walk-forward threshold tuning weighted by n_pos."""
        candidates = np.linspace(0, 1, 200)
        # First pass: collect score percentiles to define sensible range
        all_scores = []
        for fold, sc in zip(fold_data["cv_folds"],
                            scaler_per_fold):
            X_val, y_val = prepare_X_y(fold["val"])
            X_val_s = sc.transform(X_val)
            all_scores.extend(self.score(X_val_s))
        pcts = np.percentile(all_scores, np.linspace(50, 99.9, 200))

        best_f1, best_t = 0, pcts[0]
        for t in pcts:
            fold_metrics = []
            for fold, normals, sc in zip(
                fold_data["cv_folds"],
                fold_data["train_normals_per_fold"],
                scaler_per_fold,
            ):
                X_val, y_val = prepare_X_y(fold["val"])
                X_val_s = sc.transform(X_val)
                scores = self.score(X_val_s)
                preds = (scores >= t).astype(int)
                n_pos = int(y_val.sum())
                fm = compute_metrics(y_val, preds, scores)
                fm["n_pos"] = n_pos
                fold_metrics.append(fm)
            wf1 = weighted_metric(fold_metrics, "F1")
            if wf1 > best_f1:
                best_f1 = wf1
                best_t = t
        self.threshold_ = best_t
        return best_t, best_f1


# ──────────────────────────────────────────────
# One-Class SVM Detector (Prompt 6)
# ──────────────────────────────────────────────

class OneClassSVMDetector:
    """One-Class SVM with RBF kernel. Score = -decision_function (higher = more anomalous)."""

    def __init__(self, nu=0.10, gamma="scale"):
        self.nu = nu
        self.gamma = gamma
        self.model_ = None
        self.threshold_ = None

    def fit(self, X_normals):
        self.model_ = OneClassSVM(kernel="rbf", nu=self.nu, gamma=self.gamma)
        self.model_.fit(X_normals)
        return self

    def score(self, X):
        return -self.model_.decision_function(X)

    def predict(self, X, threshold=None):
        t = threshold or self.threshold_
        return (self.score(X) >= t).astype(int)

    @staticmethod
    def grid_search(fold_data, scaler_per_fold):
        """Grid search over nu x gamma, selecting by weighted F1."""
        nu_grid = [0.05, 0.10, 0.15, 0.22]
        gamma_grid = ["scale", "auto", 0.01, 0.001]
        best_score, best_params, best_model = 0, None, None

        for nu in nu_grid:
            for gamma in gamma_grid:
                fold_metrics = []
                det = OneClassSVMDetector(nu=nu, gamma=gamma)

                for fold, normals, sc in zip(
                    fold_data["cv_folds"],
                    fold_data["train_normals_per_fold"],
                    scaler_per_fold,
                ):
                    X_tr_n, _ = prepare_X_y(normals["train_normals"])
                    X_tr_n_s = sc.transform(X_tr_n)
                    det.fit(X_tr_n_s)

                    X_val, y_val = prepare_X_y(fold["val"])
                    X_val_s = sc.transform(X_val)
                    scores = det.score(X_val_s)
                    # Use 0 as threshold (SVM native boundary)
                    preds = (scores >= 0).astype(int)
                    fm = compute_metrics(y_val, preds, scores)
                    fm["n_pos"] = int(y_val.sum())
                    fold_metrics.append(fm)

                wf1 = weighted_metric(fold_metrics, "F1")
                if wf1 > best_score:
                    best_score = wf1
                    best_params = {"nu": nu, "gamma": gamma}
                    best_model = det

        return best_model, best_params, best_score


# ──────────────────────────────────────────────
# Autoencoder Detector (Prompt 6)
# ──────────────────────────────────────────────

class AutoencoderDetector:
    """Symmetric autoencoder (57→24→12→6→12→24→57). Score = reconstruction MSE."""

    def __init__(self, input_dim=56, lr=1e-3, dropout=0.15, epochs=200, seed=42):
        self.input_dim = input_dim
        self.lr = lr
        self.dropout = dropout
        self.epochs = epochs
        self.seed = seed
        self.model_ = None
        self.threshold_ = None

    def _build_model(self):
        import tensorflow as tf
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        inp = tf.keras.Input(shape=(self.input_dim,))
        x = tf.keras.layers.Dense(24, activation="relu")(inp)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = tf.keras.layers.Dense(12, activation="relu")(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = tf.keras.layers.Dense(6, activation="relu")(x)
        x = tf.keras.layers.Dense(12, activation="relu")(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = tf.keras.layers.Dense(24, activation="relu")(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        out = tf.keras.layers.Dense(self.input_dim, activation="linear")(x)

        model = tf.keras.Model(inp, out)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                      loss="mse")
        return model

    def fit(self, X_normals, val_split=0.15):
        """Fit on normals with early stopping on a temporal sub-split of train (NOT the CV val)."""
        import tensorflow as tf
        tf.random.set_seed(self.seed)

        self.input_dim = X_normals.shape[1]
        self.model_ = self._build_model()

        n = len(X_normals)
        split_idx = int(n * (1 - val_split))
        X_train = X_normals[:split_idx] if hasattr(X_normals, "iloc") else X_normals[:split_idx]
        X_es_val = X_normals[split_idx:] if hasattr(X_normals, "iloc") else X_normals[split_idx:]

        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        )
        self.model_.fit(
            X_train, X_train,
            validation_data=(X_es_val, X_es_val),
            epochs=self.epochs, batch_size=32, verbose=0,
            callbacks=[es],
        )
        return self

    def score(self, X):
        arr = X.values if hasattr(X, "values") else X
        recon = self.model_.predict(arr, verbose=0)
        return np.mean((arr - recon) ** 2, axis=1)

    def predict(self, X, threshold=None):
        t = threshold or self.threshold_
        return (self.score(X) >= t).astype(int)

    def tune_threshold(self, fold_data, scaler_per_fold):
        all_scores = []
        for fold, sc in zip(fold_data["cv_folds"], scaler_per_fold):
            X_val, _ = prepare_X_y(fold["val"])
            X_val_s = sc.transform(X_val)
            all_scores.extend(self.score(X_val_s))
        pcts = np.percentile(all_scores, np.linspace(50, 99.5, 150))

        best_f1, best_t = 0, pcts[0]
        for t in pcts:
            fold_metrics = []
            for fold, sc in zip(fold_data["cv_folds"], scaler_per_fold):
                X_val, y_val = prepare_X_y(fold["val"])
                X_val_s = sc.transform(X_val)
                scores = self.score(X_val_s)
                preds = (scores >= t).astype(int)
                fm = compute_metrics(y_val, preds, scores)
                fm["n_pos"] = int(y_val.sum())
                fold_metrics.append(fm)
            wf1 = weighted_metric(fold_metrics, "F1")
            if wf1 > best_f1:
                best_f1 = wf1
                best_t = t
        self.threshold_ = best_t
        return best_t, best_f1


# ──────────────────────────────────────────────
# Isolation Forest Detector (Prompt 6)
# ──────────────────────────────────────────────

class IsolationForestDetector:
    """Isolation Forest anomaly detector. Score = -decision_function."""

    def __init__(self, contamination=0.10, n_estimators=200, random_state=42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model_ = None
        self.threshold_ = None

    def fit(self, X_normals):
        self.model_ = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        self.model_.fit(X_normals)
        return self

    def score(self, X):
        return -self.model_.decision_function(X)

    def predict(self, X, threshold=None):
        t = threshold or self.threshold_
        return (self.score(X) >= t).astype(int)

    @staticmethod
    def grid_search(fold_data, scaler_per_fold):
        contam_grid = [0.05, 0.10, 0.15, 0.22]
        best_score, best_params, best_model = 0, None, None

        for c in contam_grid:
            fold_metrics = []
            det = IsolationForestDetector(contamination=c)

            for fold, normals, sc in zip(
                fold_data["cv_folds"],
                fold_data["train_normals_per_fold"],
                scaler_per_fold,
            ):
                X_tr_n, _ = prepare_X_y(normals["train_normals"])
                X_tr_n_s = sc.transform(X_tr_n)
                det.fit(X_tr_n_s)

                X_val, y_val = prepare_X_y(fold["val"])
                X_val_s = sc.transform(X_val)
                scores = det.score(X_val_s)
                preds = (scores >= 0).astype(int)
                fm = compute_metrics(y_val, preds, scores)
                fm["n_pos"] = int(y_val.sum())
                fold_metrics.append(fm)

            wf1 = weighted_metric(fold_metrics, "F1")
            if wf1 > best_score:
                best_score = wf1
                best_params = {"contamination": c}
                best_model = det

        return best_model, best_params, best_score

    def tune_threshold(self, fold_data, scaler_per_fold):
        all_scores = []
        for fold, sc in zip(fold_data["cv_folds"], scaler_per_fold):
            X_val, _ = prepare_X_y(fold["val"])
            X_val_s = sc.transform(X_val)
            all_scores.extend(self.score(X_val_s))
        pcts = np.percentile(all_scores, np.linspace(30, 99, 150))

        best_f1, best_t = 0, pcts[0]
        for t in pcts:
            fold_metrics = []
            for fold, sc in zip(fold_data["cv_folds"], scaler_per_fold):
                X_val, y_val = prepare_X_y(fold["val"])
                X_val_s = sc.transform(X_val)
                scores = self.score(X_val_s)
                preds = (scores >= t).astype(int)
                fm = compute_metrics(y_val, preds, scores)
                fm["n_pos"] = int(y_val.sum())
                fold_metrics.append(fm)
            wf1 = weighted_metric(fold_metrics, "F1")
            if wf1 > best_f1:
                best_f1 = wf1
                best_t = t
        self.threshold_ = best_t
        return best_t, best_f1


# ──────────────────────────────────────────────
# Pipeline runner
# ──────────────────────────────────────────────

def run_all_models(verbose=True):
    """Train and evaluate all 4 models with walk-forward CV."""
    wf = walkforward_split(embargo_weeks=4, save=False, verbose=False)
    fold_data = wf["models"]

    # Build per-fold scalers
    scalers = []
    for fold, normals in zip(fold_data["cv_folds"],
                             fold_data["train_normals_per_fold"]):
        X_tr, _ = prepare_X_y(fold["train"])
        sc = FoldScaler()
        sc.fit(X_tr)
        scalers.append(sc)

    # Final scaler on full development set
    X_dev, y_dev = prepare_X_y(fold_data["development"])
    final_scaler = FoldScaler()
    final_scaler.fit(X_dev)

    X_test, y_test = prepare_X_y(fold_data["test_holdout"])
    X_test_s = final_scaler.transform(X_test)

    # Development normals for final model fitting
    dev_normals = X_dev[y_dev == 0]
    dev_normals_s = final_scaler.transform(dev_normals)

    results = {}
    all_models = {}

    # --- MVG ---
    if verbose:
        print("=" * 60)
        print("MVG (Ledoit-Wolf)")
    mvg = MVGAnomalyDetector()
    # Train per-fold MVGs to tune threshold
    mvg_fold_models = []
    for fold, normals, sc in zip(fold_data["cv_folds"],
                                 fold_data["train_normals_per_fold"],
                                 scalers):
        X_n, _ = prepare_X_y(normals["train_normals"])
        X_n_s = sc.transform(X_n)
        fold_mvg = MVGAnomalyDetector()
        fold_mvg.fit(X_n_s)
        mvg_fold_models.append(fold_mvg)

    # Tune threshold using per-fold models (need a unified approach)
    # For threshold tuning, we fit one MVG on all dev normals and use fold scalers
    mvg.fit(dev_normals_s)
    t, wf1 = mvg.tune_threshold(fold_data, scalers)
    if verbose:
        print(f"  Threshold: {t:.2f}, weighted F1: {wf1:.3f}")

    # Test holdout
    test_scores = mvg.score(X_test_s)
    test_preds = mvg.predict(X_test_s)
    mvg_test = compute_metrics(y_test, test_preds, test_scores)
    results["MVG"] = mvg_test
    all_models["MVG"] = mvg
    if verbose:
        print(f"  Test: F1={mvg_test['F1']:.3f}, AUC_PR={mvg_test['AUC_PR']:.3f}, "
              f"AUC_ROC={mvg_test['AUC_ROC']:.3f}, Prec={mvg_test['Precision']:.3f}, "
              f"Rec={mvg_test['Recall']:.3f}, F2={mvg_test['F2']:.3f}")

    # --- One-Class SVM ---
    if verbose:
        print("=" * 60)
        print("One-Class SVM (grid search)")
    svm_best, svm_params, svm_cv_f1 = OneClassSVMDetector.grid_search(fold_data, scalers)
    if verbose:
        print(f"  Best params: {svm_params}, CV weighted F1: {svm_cv_f1:.3f}")

    # Refit on full dev normals with best params
    svm_final = OneClassSVMDetector(nu=svm_params["nu"], gamma=svm_params["gamma"])
    svm_final.fit(dev_normals_s)
    # Threshold = 0 (SVM native boundary)
    svm_final.threshold_ = 0

    test_scores_svm = svm_final.score(X_test_s)
    test_preds_svm = svm_final.predict(X_test_s)
    svm_test = compute_metrics(y_test, test_preds_svm, test_scores_svm)
    results["SVM"] = svm_test
    all_models["SVM"] = svm_final
    if verbose:
        print(f"  Test: F1={svm_test['F1']:.3f}, AUC_PR={svm_test['AUC_PR']:.3f}, "
              f"AUC_ROC={svm_test['AUC_ROC']:.3f}, Prec={svm_test['Precision']:.3f}, "
              f"Rec={svm_test['Recall']:.3f}, F2={svm_test['F2']:.3f}")

    # --- Isolation Forest ---
    if verbose:
        print("=" * 60)
        print("Isolation Forest (grid search)")
    if_best, if_params, if_cv_f1 = IsolationForestDetector.grid_search(fold_data, scalers)
    if verbose:
        print(f"  Best params: {if_params}, CV weighted F1: {if_cv_f1:.3f}")

    if_final = IsolationForestDetector(contamination=if_params["contamination"])
    if_final.fit(dev_normals_s)
    if_final.threshold_ = 0
    # Also tune threshold
    t_if, wf1_if = if_final.tune_threshold(fold_data, scalers)
    if verbose:
        print(f"  Tuned threshold: {t_if:.4f}, weighted F1: {wf1_if:.3f}")

    test_scores_if = if_final.score(X_test_s)
    test_preds_if = if_final.predict(X_test_s)
    if_test = compute_metrics(y_test, test_preds_if, test_scores_if)
    results["IF"] = if_test
    all_models["IF"] = if_final
    if verbose:
        print(f"  Test: F1={if_test['F1']:.3f}, AUC_PR={if_test['AUC_PR']:.3f}, "
              f"AUC_ROC={if_test['AUC_ROC']:.3f}, Prec={if_test['Precision']:.3f}, "
              f"Rec={if_test['Recall']:.3f}, F2={if_test['F2']:.3f}")

    # --- Autoencoder ---
    if verbose:
        print("=" * 60)
        print("Autoencoder")
    try:
        ae = AutoencoderDetector(input_dim=dev_normals_s.shape[1])
        ae.fit(dev_normals_s)
        t_ae, wf1_ae = ae.tune_threshold(fold_data, scalers)
        if verbose:
            print(f"  Threshold: {t_ae:.6f}, weighted F1: {wf1_ae:.3f}")

        test_scores_ae = ae.score(X_test_s)
        test_preds_ae = ae.predict(X_test_s)
        ae_test = compute_metrics(y_test, test_preds_ae, test_scores_ae)
        results["AE"] = ae_test
        all_models["AE"] = ae
        if verbose:
            print(f"  Test: F1={ae_test['F1']:.3f}, AUC_PR={ae_test['AUC_PR']:.3f}, "
                  f"AUC_ROC={ae_test['AUC_ROC']:.3f}, Prec={ae_test['Precision']:.3f}, "
                  f"Rec={ae_test['Recall']:.3f}, F2={ae_test['F2']:.3f}")
    except ImportError:
        if verbose:
            print("  ⚠️ tensorflow not installed, skipping Autoencoder")
        results["AE"] = {m: np.nan for m in METRICS_NAMES}
        all_models["AE"] = None

    # Save results table
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df_results = pd.DataFrame(results).T
    df_results.to_csv(TABLES_DIR / "model_comparison.csv")
    if verbose:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON (Test Holdout)")
        print(df_results.to_string())

    # Save models
    for name, model in all_models.items():
        if model is not None:
            if name == "AE":
                model.model_.save(MODELS_DIR / "ae_final.keras")
            else:
                joblib.dump(model, MODELS_DIR / f"{name.lower()}_final.pkl")

    # Also save final scaler
    joblib.dump(final_scaler, MODELS_DIR / "final_scaler.pkl")

    return all_models, results, final_scaler, fold_data, scalers


if __name__ == "__main__":
    run_all_models()
