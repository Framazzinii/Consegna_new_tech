"""Prompt 10 — Backtest engine with transaction costs and threshold optimization."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import load_dataset, PROJECT_ROOT

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGS_DIR = PROJECT_ROOT / "outputs" / "figures"

# Transaction costs in bps
TC_DICT = {
    "LEVERED_EQUITY": 5,
    "CASH_USD": 2,
    "GOLD": 8,
    "MBS": 20,
}

# Allocation weights: asset -> weight on each raw price series
ALLOC_WEIGHTS = {
    "LEVERED_EQUITY": {"equity": 1.5, "cash": -0.5},
    "CASH_USD": {"cash": 1.0},
    "GOLD": {"gold": 1.0},
    "MBS": {"mbs": 1.0},
}


def _get_price_series():
    """Load raw price series for backtest."""
    df = load_dataset(verbose=False)
    prices = pd.DataFrame(index=df.index)
    prices["equity"] = df["MXUS"]
    prices["gold"] = df["XAUBGNL"]
    prices["mbs"] = df["LUMSTRUU"]
    prices["bond"] = df["LUACTRUU"]
    # Cash accrual from 3M rate
    weekly_rate = df["USGG3M"] / 100 / 52
    cash = (1 + weekly_rate).cumprod()
    cash = cash / cash.iloc[0]
    prices["cash"] = cash
    return prices


def backtest_strategy(allocations, prices=None, tc_dict=None):
    """Backtest a strategy given allocations and price series.
    Returns equity curve, metrics, per-regime stats."""
    if prices is None:
        prices = _get_price_series()
    tc_dict = tc_dict or TC_DICT

    common_idx = allocations.index.intersection(prices.index)
    allocs = allocations.loc[common_idx]
    p = prices.loc[common_idx]

    # Weekly returns per asset
    ret = p.pct_change().fillna(0)

    # Strategy returns
    strat_ret = pd.Series(0.0, index=common_idx)
    tc_total = 0.0
    prev_alloc = None
    n_switches = 0

    for i, dt in enumerate(common_idx):
        regime = allocs.loc[dt]
        weights = ALLOC_WEIGHTS.get(regime, {"cash": 1.0})

        # Portfolio return
        port_ret = sum(w * ret.loc[dt].get(asset, 0) for asset, w in weights.items())
        strat_ret.loc[dt] = port_ret

        # Transaction cost on switch
        if prev_alloc is not None and regime != prev_alloc:
            tc_bps = tc_dict.get(regime, 0)
            strat_ret.loc[dt] -= tc_bps / 10000
            tc_total += tc_bps / 10000
            n_switches += 1
        prev_alloc = regime

    # Equity curve
    equity = (1 + strat_ret).cumprod()

    # Metrics
    n_years = len(common_idx) / 52
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1]) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = strat_ret.std() * np.sqrt(52)
    sharpe = cagr / ann_vol if ann_vol > 0 else 0
    downside = strat_ret[strat_ret < 0].std() * np.sqrt(52)
    sortino = cagr / downside if downside > 0 else 0

    # Max drawdown
    cum_max = equity.cummax()
    dd = (equity - cum_max) / cum_max
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    turnover = n_switches / n_years if n_years > 0 else 0

    metrics = {
        "CAGR": cagr,
        "Annual_Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max_DD": max_dd,
        "Calmar": calmar,
        "Turnover_pa": turnover,
        "TC_total": tc_total,
        "N_switches": n_switches,
    }

    # Per-regime performance
    regime_stats = {}
    for regime in allocs.unique():
        mask = allocs == regime
        r = strat_ret[mask]
        regime_stats[regime] = {
            "n_weeks": int(mask.sum()),
            "mean_ret_weekly": r.mean(),
            "total_ret": (1 + r).prod() - 1,
        }

    return equity, strat_ret, metrics, regime_stats, dd


def benchmark_60_40(prices=None, period_index=None):
    """Static 60/40 benchmark: 60% MXUS + 40% LUACTRUU."""
    if prices is None:
        prices = _get_price_series()
    if period_index is not None:
        prices = prices.loc[period_index]
    ret = prices.pct_change().fillna(0)
    bench_ret = 0.6 * ret["equity"] + 0.4 * ret["bond"]
    equity = (1 + bench_ret).cumprod()
    return equity, bench_ret


def benchmark_buyhold(prices=None, period_index=None):
    """MXUS buy-and-hold."""
    if prices is None:
        prices = _get_price_series()
    if period_index is not None:
        prices = prices.loc[period_index]
    ret = prices["equity"].pct_change().fillna(0)
    equity = (1 + ret).cumprod()
    return equity, ret


def optimize_routing_thresholds(ensemble_signals, subscores_folds, wf_routing,
                                prices=None, verbose=True):
    """Grid search routing thresholds on CV folds, maximize median Calmar weighted by duration."""
    from src.routing import route_series

    if prices is None:
        prices = _get_price_series()

    usd_grid = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    oro_grid = [0.5, 0.75, 1.0, 1.25, 1.5]

    best_calmar, best_thresh = -np.inf, {"usd": 1.0, "oro": 1.0}

    for u in usd_grid:
        for o in oro_grid:
            th = {"usd": u, "oro": o}
            fold_calmars = []
            fold_durations = []

            for fs_data in subscores_folds:
                fs = fs_data["subscores"]
                # Need ensemble signals for this fold's period
                fold_idx = fs.index
                fold_signals = ensemble_signals.reindex(fold_idx).dropna()
                common = fold_signals.index.intersection(fs.index)
                if len(common) < 10:
                    continue

                allocs = route_series(fold_signals.loc[common], fs.loc[common], th)
                try:
                    eq, _, metrics, _, _ = backtest_strategy(allocs, prices)
                    fold_calmars.append(metrics["Calmar"])
                    fold_durations.append(len(common))
                except Exception:
                    fold_calmars.append(0)
                    fold_durations.append(len(common))

            if not fold_calmars:
                continue

            # Weighted median by duration
            total_dur = sum(fold_durations)
            if total_dur == 0:
                continue
            weights = [d / total_dur for d in fold_durations]
            weighted_calmar = sum(c * w for c, w in zip(fold_calmars, weights))

            if weighted_calmar > best_calmar:
                best_calmar = weighted_calmar
                best_thresh = th

    if verbose:
        print(f"Optimal thresholds: usd={best_thresh['usd']}, oro={best_thresh['oro']}")
        print(f"Weighted Calmar: {best_calmar:.3f}")

    return best_thresh, best_calmar


def plot_equity_curves(strategy_eq, bench60_eq, buyhold_eq, dd,
                       allocations=None, save_path=None):
    """Plot equity curves and drawdown."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(strategy_eq.index, strategy_eq.values, label="EWS Strategy", linewidth=2)
    ax.plot(bench60_eq.index, bench60_eq.values, label="60/40", linewidth=1.5, alpha=0.7)
    ax.plot(buyhold_eq.index, buyhold_eq.values, label="MXUS Buy&Hold", linewidth=1.5, alpha=0.7)
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.set_title("Backtest: EWS Strategy vs Benchmarks")
    ax.grid(True, alpha=0.3)

    # Regime shading
    if allocations is not None:
        colors = {
            "LEVERED_EQUITY": "#2ecc71",
            "CASH_USD": "#3498db",
            "GOLD": "#f1c40f",
            "MBS": "#95a5a6",
        }
        prev = None
        for dt, regime in allocations.items():
            if regime != prev:
                ax.axvline(dt, color=colors.get(regime, "gray"), alpha=0.1, linewidth=0.5)
            prev = regime

    ax2 = axes[1]
    ax2.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return fig
