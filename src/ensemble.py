"""Prompt 7 — Ensemble detector: hard voting, soft voting (mean/median)."""

import numpy as np
import pandas as pd
from src.models import compute_metrics, weighted_metric, METRICS_NAMES
from src.preprocessing import prepare_X_y, FoldScaler
from src.data_loader import PROJECT_ROOT

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGS_DIR = PROJECT_ROOT / "outputs" / "figures"


def score_to_percentile(s_train_normals, s_new):
    """Empirical percentile of new scores vs the train-normals score distribution."""
    sorted_train = np.sort(s_train_normals)
    n = len(sorted_train)
    return np.searchsorted(sorted_train, s_new, side="right") / n


class EnsembleDetector:

    def __init__(self, models, model_names=None):
        self.models = models
        self.model_names = model_names or [f"model_{i}" for i in range(len(models))]
        self.threshold_soft_mean_ = None
        self.threshold_soft_median_ = None

    def _get_percentiles(self, X_scaled, X_normals_scaled):
        """Compute percentile-mapped scores for each model."""
        pcts = {}
        for name, model in zip(self.model_names, self.models):
            if model is None:
                continue
            s_normals = model.score(X_normals_scaled)
            s_new = model.score(X_scaled)
            pcts[name] = score_to_percentile(s_normals, s_new)
        return pcts

    def predict_hard(self, X_scaled, majority=3):
        """Hard voting: each model uses its own threshold, majority wins."""
        votes = np.zeros(len(X_scaled))
        n_models = 0
        for model in self.models:
            if model is None or model.threshold_ is None:
                continue
            preds = model.predict(X_scaled)
            votes += preds
            n_models += 1
        return (votes >= majority).astype(int)

    def predict_soft_mean(self, pcts, threshold=None):
        t = threshold or self.threshold_soft_mean_
        arr = np.column_stack(list(pcts.values()))
        mean_pct = arr.mean(axis=1)
        return (mean_pct >= t).astype(int), mean_pct

    def predict_soft_median(self, pcts, threshold=None):
        t = threshold or self.threshold_soft_median_
        arr = np.column_stack(list(pcts.values()))
        med_pct = np.median(arr, axis=1)
        return (med_pct >= t).astype(int), med_pct

    def tune_soft_thresholds(self, fold_data, scalers, dev_normals_scaled):
        """Tune soft voting thresholds via walk-forward weighted by n_pos."""
        candidates = np.linspace(0.5, 0.99, 100)

        for mode in ["mean", "median"]:
            best_f1, best_t = 0, 0.5
            for t in candidates:
                fold_metrics = []
                for fold, sc in zip(fold_data["cv_folds"], scalers):
                    X_val, y_val = prepare_X_y(fold["val"])
                    X_val_s = sc.transform(X_val)
                    pcts = self._get_percentiles(X_val_s, dev_normals_scaled)

                    if mode == "mean":
                        preds, scores = self.predict_soft_mean(pcts, threshold=t)
                    else:
                        preds, scores = self.predict_soft_median(pcts, threshold=t)

                    fm = compute_metrics(y_val, preds, scores)
                    fm["n_pos"] = int(y_val.sum())
                    fold_metrics.append(fm)

                wf1 = weighted_metric(fold_metrics, "F1")
                if wf1 > best_f1:
                    best_f1 = wf1
                    best_t = t

            if mode == "mean":
                self.threshold_soft_mean_ = best_t
            else:
                self.threshold_soft_median_ = best_t

    def error_correlation(self, X_scaled, y_true):
        """4x4 error correlation matrix of binary predictions."""
        errors = {}
        for name, model in zip(self.model_names, self.models):
            if model is None:
                continue
            preds = model.predict(X_scaled)
            errors[name] = (preds != y_true).astype(int)
        err_df = pd.DataFrame(errors)
        corr = err_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        mean_corr = corr.where(mask).stack().mean()
        if mean_corr > 0.85:
            print(f"WARNING: mean pairwise error correlation = {mean_corr:.3f} > 0.85 — "
                  "ensemble adds little value")
        return corr, mean_corr


def run_ensemble(all_models, results, final_scaler, fold_data, scalers, verbose=True):
    """Build and evaluate ensemble variants."""
    model_names = ["MVG", "SVM", "IF", "AE"]
    models = [all_models.get(n) for n in model_names]
    active_models = [(n, m) for n, m in zip(model_names, models) if m is not None]

    ensemble = EnsembleDetector(
        [m for _, m in active_models],
        [n for n, _ in active_models],
    )

    # Prepare test data
    X_test, y_test = prepare_X_y(fold_data["test_holdout"])
    X_test_s = final_scaler.transform(X_test)

    # Dev normals for percentile calibration
    X_dev, y_dev = prepare_X_y(fold_data["development"])
    dev_normals = X_dev[y_dev == 0]
    dev_normals_s = final_scaler.transform(dev_normals)

    # Tune soft thresholds
    ensemble.tune_soft_thresholds(fold_data, scalers, dev_normals_s)
    if verbose:
        print(f"Soft mean threshold: {ensemble.threshold_soft_mean_:.3f}")
        print(f"Soft median threshold: {ensemble.threshold_soft_median_:.3f}")

    # Evaluate all ensemble variants
    pcts_test = ensemble._get_percentiles(X_test_s, dev_normals_s)
    ens_results = {}

    # Hard voting
    hard_preds = ensemble.predict_hard(X_test_s, majority=len(active_models) - 1)
    # For score: use mean percentile
    arr = np.column_stack(list(pcts_test.values()))
    hard_scores = arr.mean(axis=1)
    ens_results["Hard_Vote"] = compute_metrics(y_test, hard_preds, hard_scores)

    # Soft mean
    soft_mean_preds, soft_mean_scores = ensemble.predict_soft_mean(pcts_test)
    ens_results["Soft_Mean"] = compute_metrics(y_test, soft_mean_preds, soft_mean_scores)

    # Soft median
    soft_med_preds, soft_med_scores = ensemble.predict_soft_median(pcts_test)
    ens_results["Soft_Median"] = compute_metrics(y_test, soft_med_preds, soft_med_scores)

    # Error correlation
    if verbose:
        print("\nError correlation analysis:")
    corr, mean_corr = ensemble.error_correlation(X_test_s, y_test)
    if verbose:
        print(f"  Mean pairwise error correlation: {mean_corr:.3f}")
        print(corr.to_string())

    # Combine with single model results
    all_results = {**results, **ens_results}
    df_all = pd.DataFrame(all_results).T
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(TABLES_DIR / "all_models_ensemble.csv")

    if verbose:
        print("\n" + "=" * 60)
        print("ALL MODELS + ENSEMBLE (Test Holdout)")
        print(df_all.to_string())

    # Return the best ensemble predictions for downstream use
    best_ens_name = max(ens_results, key=lambda k: ens_results[k]["F1"])
    if best_ens_name == "Hard_Vote":
        best_preds = hard_preds
        best_scores = hard_scores
    elif best_ens_name == "Soft_Mean":
        best_preds = soft_mean_preds
        best_scores = soft_mean_scores
    else:
        best_preds = soft_med_preds
        best_scores = soft_med_scores

    if verbose:
        print(f"\nBest ensemble: {best_ens_name} (F1={ens_results[best_ens_name]['F1']:.3f})")

    # Build prediction series for downstream
    pred_series = pd.Series(best_preds, index=X_test.index, name="ensemble_signal")
    score_series = pd.Series(best_scores, index=X_test.index, name="ensemble_score")

    return ensemble, ens_results, pred_series, score_series
