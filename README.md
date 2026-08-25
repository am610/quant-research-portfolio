# Quant Research Portfolio

Rigorous, reproducible studies of statistical arbitrage, predictive uncertainty, and market microstructure.

[![Open baseline in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/am610/quant-research-portfolio/blob/main/notebooks/01_pair_trading_baseline.ipynb)
[![Open regime study in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/am610/quant-research-portfolio/blob/main/notebooks/02_regime_filter.ipynb)
[![Open universe study in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/am610/quant-research-portfolio/blob/main/notebooks/03_peer_universe.ipynb)

## Main study

![Peer universe dashboard](docs/assets/universe_dashboard.png)

The expanded experiment begins with twelve liquid stocks in five economically predefined peer groups. Pair relationships are estimated using training data only, all stationarity tests receive a family level false discovery correction, and hedge ratios are updated from trailing observations.

Only BAC with JPM and PEP with KO pass the five percent corrected threshold. Their combined untouched test result has a 1.94 percent annual return, 5.75 percent annual volatility, a 0.36 Sharpe estimate, and a 7.47 percent maximum drawdown. The 95 percent bootstrap Sharpe interval spans negative 0.63 to 1.48, so the evidence remains inconclusive.

![Statistical arbitrage baseline](docs/assets/pair_baseline.png)

## First result

The first locked baseline uses XLE and XOP adjusted daily prices. The hedge relation is estimated only on the training period. A fixed causal rolling rule is evaluated after three basis points of turnover cost.

The untouched test period produced a 1.34 percent annual return, 5.02 percent annual volatility, a Sharpe ratio of 0.29, and a maximum drawdown of 9.40 percent. A stationary block bootstrap gave a wide 95 percent Sharpe interval from negative 0.96 to 1.46. Validation performance was negative.

This is not evidence of a dependable trading strategy. It is an intentionally transparent benchmark and motivates the next research question: can regime information and calibrated uncertainty distinguish temporary mean reversion from structural breakdown?

## Regime result

![Causal regime filter](docs/assets/regime_filter.png)

A causal filter based on rolling spread persistence and innovation volatility was selected using validation data only. On the untouched test period it increased the Sharpe estimate from 0.49 to 0.73 and reduced maximum drawdown from 9.40 percent to 3.78 percent.

The 95 percent bootstrap Sharpe interval still crosses zero, from negative 0.22 to 1.79. The result is therefore encouraging but not conclusive. This uncertainty motivates the calibrated prediction study rather than a profitability claim.

## Research program

### Study 1: Regime aware statistical arbitrage

This study asks whether machine learning can identify more persistent mean reverting portfolios than classical methods after transaction costs, selection effects, and market regime changes are considered.

Planned comparisons include cointegration, principal component residuals, moving band portfolios, and temporal neural models. Evaluation will use chronological splits, realistic costs, block bootstrap uncertainty, and stress tests across market conditions.

### Study 2: Uncertainty aware return prediction

This study asks whether calibrated predictive uncertainty improves trading decisions during distribution shift. Linear models, gradient boosted trees, temporal neural models, and ensemble uncertainty estimates will be compared using both statistical and economic metrics.

### Study 3: Limit order book forecasting

This study asks when accurate short horizon forecasts remain tradable after fees, spread, fill probability, latency assumptions, and adverse selection are included.

## Portfolio principles

1. Every complex model must beat a simple baseline.
2. Every evaluation must preserve time order.
3. Every reported strategy must include realistic trading frictions.
4. Every key result must include uncertainty or stability evidence.
5. Negative results and failure cases are part of the research output.

## Planned public artifacts

1. Reproducible Python package and configuration files
2. Colab notebooks for compact demonstrations
3. Research reports with methods, results, and limitations
4. Interactive visual summaries for recruiters
5. Automated tests for data leakage and accounting logic

## Status

The data validation, cost accounting, risk metrics, bootstrap uncertainty, classical baseline, causal regime filter, training only pair selection, false discovery correction, and walk forward hedge estimation are implemented and tested. Three executed Colab notebooks document the research progression.

## Reproduce the first study

```bash
python -m pip install -e .
PYTHONPATH=src python scripts/run_pair_baseline.py
```

The script downloads adjusted daily observations from Yahoo Finance and records the retrieval cutoff. Generated data files are excluded from version control.
