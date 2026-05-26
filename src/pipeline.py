"""End-to-end pipeline: models → ensemble → routing → backtest."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import PROJECT_ROOT
from src.splits import walkforward_split
from src.models import run_all_models
from src.ensemble import run_ensemble
from src.routing import (
    compute_all_subscores, route_series, DEFAULT_THRESHOLDS,
)
from src.backtest import (
    backtest_strategy, benchmark_60_40, benchmark_buyhold,
    optimize_routing_thresholds, plot_equity_curves, _get_price_series,
)

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGS_DIR = PROJECT_ROOT / "outputs" / "figures"


def run_pipeline(verbose=True):
    """Run full pipeline: Prompts 5-10."""

    # ── Step 1: Models (Prompt 5-6) ──
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 1: Training all 4 anomaly detection models")
        print("=" * 70)
    all_models, results, final_scaler, fold_data, scalers = run_all_models(verbose=verbose)

    # ── Step 2: Ensemble (Prompt 7) ──
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 2: Building ensemble")
        print("=" * 70)
    ensemble, ens_results, pred_series, score_series = run_ensemble(
        all_models, results, final_scaler, fold_data, scalers, verbose=verbose
    )

    # ── Step 3: Sub-scores (Prompt 8) ──
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 3: Computing domain sub-scores")
        print("=" * 70)
    wf = walkforward_split(embargo_weeks=4, save=False, verbose=False)
    subscores_test, subscores_folds = compute_all_subscores(
        wf["routing"], wf["models"]
    )

    if verbose:
        print(f"Test holdout sub-scores shape: {subscores_test.shape}")
        print(f"Sub-score stats:\n{subscores_test.describe()}")

    # ── Step 4: Routing threshold optimization (Prompt 9-10) ──
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 4: Optimizing routing thresholds")
        print("=" * 70)

    # For fold-level optimization, we need ensemble signals per fold
    # Use the test holdout ensemble signals for now, optimize thresholds on folds
    prices = _get_price_series()

    # First try with defaults
    allocations_default = route_series(pred_series, subscores_test, DEFAULT_THRESHOLDS)
    if verbose:
        print(f"\nDefault thresholds allocation counts:")
        print(allocations_default.value_counts())

    # Optimize thresholds (use fold subscores)
    # For optimization, we need ensemble predictions on fold vals too
    # Re-run ensemble per fold
    from src.preprocessing import prepare_X_y, FoldScaler
    fold_ensemble_signals = {}
    for fold, sc in zip(fold_data["cv_folds"], scalers):
        X_val, y_val = prepare_X_y(fold["val"])
        X_val_s = sc.transform(X_val)
        preds = ensemble.predict_hard(X_val_s)
        fold_ensemble_signals[fold["fold_id"]] = pd.Series(
            preds, index=X_val.index, name="ensemble_signal"
        )

    # Build combined fold signals for optimization
    combined_fold_signals = pd.concat(fold_ensemble_signals.values())

    best_thresh, best_calmar = optimize_routing_thresholds(
        combined_fold_signals, subscores_folds, wf["routing"], prices, verbose=verbose
    )

    # ── Step 5: Final backtest (Prompt 10) ──
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 5: Final backtest with optimized thresholds")
        print("=" * 70)

    allocations = route_series(pred_series, subscores_test, best_thresh)
    if verbose:
        print(f"\nOptimized allocation counts:")
        print(allocations.value_counts())

    strat_eq, strat_ret, metrics, regime_stats, dd = backtest_strategy(allocations, prices)

    # Benchmarks
    bench60_eq, bench60_ret = benchmark_60_40(prices, strat_eq.index)
    buyhold_eq, buyhold_ret = benchmark_buyhold(prices, strat_eq.index)

    if verbose:
        print(f"\n{'Metric':<20} {'Strategy':>12} {'60/40':>12} {'Buy&Hold':>12}")
        print("-" * 56)

        # Benchmark metrics
        n_years = len(strat_eq) / 52
        for name, eq, ret in [("60/40", bench60_eq, bench60_ret),
                               ("Buy&Hold", buyhold_eq, buyhold_ret)]:
            cagr_b = eq.iloc[-1] ** (1/n_years) - 1
            vol_b = ret.std() * np.sqrt(52)
            sharpe_b = cagr_b / vol_b if vol_b > 0 else 0
            dd_b = ((eq - eq.cummax()) / eq.cummax()).min()
            calmar_b = cagr_b / abs(dd_b) if dd_b != 0 else 0

        print(f"{'CAGR':<20} {metrics['CAGR']:>11.2%}")
        print(f"{'Annual Vol':<20} {metrics['Annual_Vol']:>11.2%}")
        print(f"{'Sharpe':<20} {metrics['Sharpe']:>11.3f}")
        print(f"{'Sortino':<20} {metrics['Sortino']:>11.3f}")
        print(f"{'Max DD':<20} {metrics['Max_DD']:>11.2%}")
        print(f"{'Calmar':<20} {metrics['Calmar']:>11.3f}")
        print(f"{'Turnover p.a.':<20} {metrics['Turnover_pa']:>11.1f}")
        print(f"{'TC total':<20} {metrics['TC_total']:>11.4f}")

        print(f"\nPer-regime stats:")
        for regime, stats in regime_stats.items():
            print(f"  {regime}: {stats['n_weeks']} weeks, "
                  f"total return {stats['total_ret']:.2%}")

    # Save results
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame({"Strategy": metrics}, index=metrics.keys())
    metrics_df.to_csv(TABLES_DIR / "backtest_metrics.csv")

    alloc_df = allocations.to_frame("allocation")
    alloc_df.to_csv(TABLES_DIR / "test_allocations.csv")

    # Plot
    plot_equity_curves(
        strat_eq, bench60_eq, buyhold_eq, dd,
        allocations=allocations,
        save_path=FIGS_DIR / "equity_curves.png",
    )

    return {
        "all_models": all_models,
        "ensemble": ensemble,
        "pred_series": pred_series,
        "subscores_test": subscores_test,
        "allocations": allocations,
        "metrics": metrics,
        "strat_eq": strat_eq,
        "best_thresh": best_thresh,
        "fold_data": fold_data,
        "final_scaler": final_scaler,
        "scalers": scalers,
        "results": results,
        "ens_results": ens_results,
        "wf_routing": wf["routing"],
    }


if __name__ == "__main__":
    run_pipeline()
