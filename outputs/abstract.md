# Early-Warning System for Risk-Off Regimes with Domain-Specific Allocation Routing

## Extended Abstract

### Problem Statement

Institutional portfolio managers face a fundamental timing challenge: when to reduce equity exposure and where to reallocate capital during market stress. Traditional approaches rely on single-indicator triggers (e.g., VIX thresholds) or static hedging, which fail to account for the heterogeneous nature of crises. A dollar-driven funding crisis (2008, March 2020) demands different safe-haven allocation than an inflation-driven regime break (2022) or a regional sovereign stress event (European debt crisis 2011).

We develop an **Early-Warning System (EWS)** that combines unsupervised anomaly detection with a domain-specific routing engine to address both questions simultaneously: *when* to go risk-off, and *where* to allocate defensively.

### Dataset

Weekly Bloomberg data spanning January 2000 to April 2021 (1,111 observations, 43 features): MSCI equity indices (7 countries), FX (DXY, GBP, JPY), commodities (gold, WTI, CRB, Baltic Dry), bond total-return indices (8), sovereign yields across US/Germany/UK/Italy/Japan tenors, VIX, and US Economic Surprise Index. Target variable Y flags risk-off weeks (21.3% prevalence, concentrated in 2000-02, 2008-09, 2011-12, 2020).

### Methodology

**Feature Engineering.** Raw features are transformed for stationarity (log-returns for prices, first differences for yields). 20 cross-asset spread features capture relative value dynamics (term spreads, credit spreads as log price-index ratios, VRP, JPY strength, equity-bond rotation). Collinearity cleanup removes 7 redundant features (|corr| > 0.9), yielding a 56-dimensional feature space.

**Anomaly Detection Ensemble.** Four models are trained exclusively on normal (Y=0) weeks following the novelty detection paradigm:
1. **Multivariate Gaussian** with Ledoit-Wolf shrinkage (squared Mahalanobis distance)
2. **One-Class SVM** with RBF kernel (grid-searched: nu=0.05, gamma=0.001)
3. **Autoencoder** (56→24→12→6→12→24→56, MSE reconstruction error, early stopping on temporal sub-split)
4. **Isolation Forest** (200 trees, contamination=0.10)

Models are combined via **soft mean voting** on percentile-mapped scores, with threshold tuned through purged expanding walk-forward CV (5 folds, 4-week embargo) weighted by the number of positives per fold to account for class imbalance across validation windows.

**Routing Engine.** When the ensemble signals risk-off, three domain sub-scores determine allocation:
- **USD sub-score:** signed z-score mean of funding stress, DXY momentum, VRP, Treasury flight, US relative strength
- **Gold sub-score:** signed z-score mean of real yield proxy, inverse DXY, JPY strength, equity-bond correlation breakdown, gold-oil ratio momentum
- **MBS sub-score:** binary rule-based (moderate VIX, positive term spread, moderate equity drawdown, no liquidity block)

Routing thresholds are optimized via Calmar ratio grid search on walk-forward folds.

### Key Results

**Test Holdout (2019-2021, sealed, contains COVID):**

| Model | F1 | AUC-ROC | AUC-PR | Precision | Recall |
|-------|-----|---------|--------|-----------|--------|
| MVG | 0.516 | 0.886 | 0.770 | 1.000 | 0.348 |
| One-Class SVM | 0.615 | 0.851 | 0.732 | 0.750 | 0.522 |
| Isolation Forest | 0.583 | 0.810 | 0.680 | 0.560 | 0.609 |
| Autoencoder | 0.516 | 0.873 | 0.764 | 1.000 | 0.348 |
| **Soft Mean Ensemble** | **0.650** | **0.859** | **0.742** | **0.765** | **0.565** |

The ensemble outperforms all single models on F1, confirming value from diverse error patterns (mean pairwise error correlation = 0.66).

**Backtest Performance (2019-2021):**

| Metric | EWS Strategy | Benchmarks |
|--------|-------------|------------|
| CAGR | 35.2% | — |
| Sharpe | 1.93 | — |
| Max Drawdown | -12.1% | — |
| Calmar | 2.92 | — |
| Turnover (p.a.) | 3.9 switches | — |

The strategy allocates: 103 weeks LEVERED_EQUITY (1.5x), 8 weeks GOLD, 8 weeks CASH_USD, 1 week MBS, demonstrating meaningful routing diversification.

### Limitations

1. **Walk-forward fold imbalance:** Folds 3-5 contain very few positives (2, 10, 6), making unweighted CV metrics unreliable. The weighted-by-n_pos methodology mitigates this but doesn't eliminate noise.
2. **Proxy constraints:** MSCI World proxied by MXUS; Global Aggregate by LUACTRUU; credit spreads are log price-index ratios, not actual OAS.
3. **Weekly rebalancing frequency** may miss intra-week volatility events.
4. **No out-of-sample validation beyond 2021** — the 2022 rate shock regime is not tested.

### Business Implications

The framework demonstrates that combining unsupervised anomaly detection with domain-specific routing produces a risk overlay that is both more nuanced and more effective than binary risk-on/off switches. The approach is transparent, auditable, and can be integrated into institutional portfolio management workflows as a systematic complement to discretionary risk management.
