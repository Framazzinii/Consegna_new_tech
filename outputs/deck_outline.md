# Pitch Deck Outline — Early-Warning System for Risk-Off Regimes
## 15 Slides

### Slide 1 — Cover
**Title:** Early-Warning System for Risk-Off Regimes with Allocation Routing
**Subtitle:** Anomaly Detection + Domain-Specific Safe-Haven Selection
**Authors / Course / Date**

### Slide 2 — Problem & Business Case
- Institutional portfolios need systematic risk-off triggers
- Single-indicator approaches (VIX > 25) miss regime nuance
- **Key insight:** the *type* of crisis determines the optimal safe haven (USD vs Gold vs MBS)
- Business value: risk overlay for portfolio management, reducible to systematic rules

### Slide 3 — Dataset Overview
- Bloomberg weekly data: 2000-01 → 2021-04 (1,111 obs)
- 43 features: equities (7 MSCI), FX, commodities, bonds, yields, rates, VIX
- Target Y: risk-off weeks (21.3% prevalence)
- Concentration in 2000-02, 2008-09, 2011-12, 2020
- **Figure:** Y=1 distribution by year (bar chart)

### Slide 4 — Pipeline Overview
- **Flow diagram:** Raw Data → Stationarity → Spreads → Walk-Forward Split → 4 Models → Ensemble → Routing → Backtest
- Emphasize: no data leakage, temporal discipline throughout

### Slide 5 — Stationarity & Feature Engineering
- Log-returns for prices, first differences for yields, level for VIX/ECSURPUS
- 42/42 features pass ADF at 5%
- 20 cross-asset spreads: term, credit, equity-bond rotation, VRP, JPY strength
- 7 collinear features removed → 56-dimensional clean feature space
- **Figure:** Correlation heatmap (before/after cleanup)

### Slide 6 — Walk-Forward Cross-Validation Design
- 5-fold expanding window with 4-week embargo
- Development: 2000-2018 | Test holdout: 2019-2021 (sealed, contains COVID)
- Weighted-by-n_pos aggregation (folds 1-2 dominate with 53/64 positives)
- **Figure:** Timeline showing fold boundaries and crisis events

### Slide 7 — Four Anomaly Detection Models
- **MVG** (Ledoit-Wolf): Mahalanobis distance on normals
- **One-Class SVM** (RBF): grid search nu × gamma
- **Autoencoder** (56→6→56): reconstruction MSE, early stopping
- **Isolation Forest** (200 trees): contamination grid
- All trained on Y=0 weeks only (novelty detection paradigm)

### Slide 8 — Ensemble Detection
- Percentile-mapping normalizes scores across models
- Three variants: hard voting (3/4), soft mean, soft median
- **Winner: Soft Mean** (F1=0.650, beats best single 0.615)
- Error correlation 0.66 → ensemble adds genuine value
- **Figure:** Score distribution per model + ensemble threshold

### Slide 9 — Routing Engine Architecture
- **Decision matrix:**
  | Signal | Condition | Allocation |
  |--------|-----------|------------|
  | Risk-on | — | 1.5x Equity |
  | Risk-off | USD score high + DXY↑ | Cash USD |
  | Risk-off | Gold score high | Gold |
  | Risk-off | MBS conditions met | MBS |
  | Risk-off | Default | Cash USD |

### Slide 10 — Domain Sub-Scores
- **USD:** funding stress, DXY, VRP, Treasury flight, US relative (signed z-scores)
- **Gold:** real yield, inverse DXY, JPY strength, equity-bond corr, gold-oil (signed z-scores)
- **MBS:** binary rules (VIX 20-28, positive term spread, moderate DD, no block)
- Thresholds tuned independently per domain on CV folds

### Slide 11 — Threshold Optimization
- Grid search: usd ∈ {0.5-2.0} × oro ∈ {0.5-1.5}
- Objective: Calmar ratio weighted by fold duration
- **Figure:** Calmar heatmap over threshold grid

### Slide 12 — Backtest Results
- CAGR: 35.2% | Sharpe: 1.93 | Max DD: -12.1% | Calmar: 2.92
- 103w equity, 8w gold, 8w cash, 1w MBS
- Transaction costs: 5/2/8/20 bps (differentiated)
- **Figure:** Equity curves — Strategy vs 60/40 vs Buy&Hold

### Slide 13 — COVID Stress Test
- **Crash (Feb-Apr 2020):** Strategy performance vs benchmarks, routing to cash/gold
- **Recovery (Apr-Dec 2020):** Re-entry timing, levered equity capture
- **Reflation (Jan-Apr 2021):** Continued risk-on allocation
- **Figure:** Crisis-zoom timeline with regime shading

### Slide 14 — Limitations & Future Work
- Thin folds 3-5 (2-10 positives) → weighted metrics essential
- MSCI World → MXUS proxy; credit spreads = log ratios not OAS
- Weekly frequency misses intra-week events
- 2022 rate shock not in test window
- **Future:** daily frequency, SHAP-based attribution, dynamic threshold adaptation

### Slide 15 — Q&A / Backup
- Detailed per-fold metrics table
- ADF test results
- Full correlation matrix
- Autoencoder training curves
- Per-regime return decomposition
