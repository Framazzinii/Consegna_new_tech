"""Prompt 11 — COVID stress test on test holdout."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import PROJECT_ROOT
from src.backtest import _get_price_series, backtest_strategy, benchmark_60_40, benchmark_buyhold

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGS_DIR = PROJECT_ROOT / "outputs" / "figures"

STRESS_WINDOWS = {
    "COVID_crash": ("2020-02-15", "2020-04-15"),
    "COVID_recovery": ("2020-04-15", "2020-12-31"),
    "Reflation_2021": ("2021-01-01", "2021-04-20"),
}


def stress_test(allocations, prices=None, verbose=True):
    """Run stress test on predefined crisis windows."""
    if prices is None:
        prices = _get_price_series()

    full_eq, full_ret, _, _, _ = backtest_strategy(allocations, prices)
    b60_eq, b60_ret = benchmark_60_40(prices, full_eq.index)
    bh_eq, bh_ret = benchmark_buyhold(prices, full_eq.index)

    results = {}
    for name, (start, end) in STRESS_WINDOWS.items():
        # Slice to window
        mask = (full_eq.index >= start) & (full_eq.index <= end)
        if mask.sum() == 0:
            continue

        idx = full_eq.index[mask]

        # Strategy
        s_eq = full_eq.loc[idx]
        s_ret = s_eq.iloc[-1] / s_eq.iloc[0] - 1
        s_dd = ((s_eq - s_eq.cummax()) / s_eq.cummax()).min()

        # 60/40
        b60 = b60_eq.loc[idx]
        b60_r = b60.iloc[-1] / b60.iloc[0] - 1
        b60_dd = ((b60 - b60.cummax()) / b60.cummax()).min()

        # Buy & Hold
        bh = bh_eq.loc[idx]
        bh_r = bh.iloc[-1] / bh.iloc[0] - 1
        bh_dd = ((bh - bh.cummax()) / bh.cummax()).min()

        # Dominant allocation
        window_alloc = allocations.loc[idx]
        dominant = window_alloc.value_counts().index[0] if len(window_alloc) > 0 else "N/A"
        n_switches = (window_alloc != window_alloc.shift()).sum() - 1

        # Lead time: weeks the first risk-off flag precedes the real drawdown
        risk_off_weeks = window_alloc[window_alloc != "LEVERED_EQUITY"]
        if len(risk_off_weeks) > 0 and len(bh) > 1:
            first_riskoff = risk_off_weeks.index[0]
            bh_peak = bh.cummax()
            dd_start_mask = bh < bh_peak * 0.98  # 2% threshold
            if dd_start_mask.any():
                dd_start = bh.index[dd_start_mask][0]
                lead_weeks = max(0, (dd_start - first_riskoff).days // 7)
            else:
                lead_weeks = 0
        else:
            lead_weeks = 0

        results[name] = {
            "Strategy_Return": s_ret,
            "60_40_Return": b60_r,
            "BuyHold_Return": bh_r,
            "Strategy_MaxDD": s_dd,
            "60_40_MaxDD": b60_dd,
            "BuyHold_MaxDD": bh_dd,
            "Dominant_Asset": dominant,
            "N_Switches": n_switches,
            "Lead_Weeks": lead_weeks,
            "N_Weeks": len(idx),
        }

    df = pd.DataFrame(results).T
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / "stress_test.csv")

    if verbose:
        print("\nSTRESS TEST RESULTS")
        print("=" * 80)
        for name, r in results.items():
            print(f"\n{name} ({STRESS_WINDOWS[name][0]} → {STRESS_WINDOWS[name][1]})")
            print(f"  Strategy: {r['Strategy_Return']:+.2%}  (MaxDD {r['Strategy_MaxDD']:.2%})")
            print(f"  60/40:    {r['60_40_Return']:+.2%}  (MaxDD {r['60_40_MaxDD']:.2%})")
            print(f"  BuyHold:  {r['BuyHold_Return']:+.2%}  (MaxDD {r['BuyHold_MaxDD']:.2%})")
            print(f"  Dominant: {r['Dominant_Asset']}, Switches: {r['N_Switches']}, "
                  f"Lead: {r['Lead_Weeks']}w")

    return df


def plot_stress_timeline(allocations, prices=None, save_path=None):
    """Plot crisis-zoom timeline with regime shading."""
    if prices is None:
        prices = _get_price_series()

    full_eq, _, _, _, _ = backtest_strategy(allocations, prices)
    b60_eq, _ = benchmark_60_40(prices, full_eq.index)
    bh_eq, _ = benchmark_buyhold(prices, full_eq.index)

    # Zoom to 2020-2021
    mask = full_eq.index >= "2019-12-01"
    idx = full_eq.index[mask]

    fig, ax = plt.subplots(figsize=(14, 7))

    # Normalize to 1 at start of zoom
    s = full_eq.loc[idx] / full_eq.loc[idx].iloc[0]
    b60 = b60_eq.loc[idx] / b60_eq.loc[idx].iloc[0]
    bh = bh_eq.loc[idx] / bh_eq.loc[idx].iloc[0]

    ax.plot(idx, s, label="EWS Strategy", linewidth=2.5, color="#2c3e50")
    ax.plot(idx, b60, label="60/40", linewidth=1.5, alpha=0.8, color="#e74c3c")
    ax.plot(idx, bh, label="MXUS Buy&Hold", linewidth=1.5, alpha=0.8, color="#3498db")

    # Regime shading
    colors = {
        "LEVERED_EQUITY": "#2ecc71",
        "CASH_USD": "#3498db",
        "GOLD": "#f1c40f",
        "MBS": "#95a5a6",
    }
    alloc_zoom = allocations.reindex(idx).dropna()
    for i in range(len(alloc_zoom)):
        dt = alloc_zoom.index[i]
        regime = alloc_zoom.iloc[i]
        if i + 1 < len(alloc_zoom):
            dt_next = alloc_zoom.index[i + 1]
        else:
            dt_next = dt + pd.Timedelta(weeks=1)
        ax.axvspan(dt, dt_next, alpha=0.15, color=colors.get(regime, "gray"))

    # Mark stress windows
    for name, (start, end) in STRESS_WINDOWS.items():
        ax.axvline(pd.Timestamp(start), color="red", linestyle="--", alpha=0.5)

    ax.set_title("COVID Stress Test: Strategy vs Benchmarks (normalized)")
    ax.set_ylabel("Normalized Equity")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return fig
