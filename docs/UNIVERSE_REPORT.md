# Predefined Peer Universe Study

## Research question

Do statistically selected relationships among economically related stocks remain profitable when pair selection, parameter updates, costs, and uncertainty are handled explicitly?

## Universe design

The universe is fixed before testing and contains five peer groups:

1. Integrated energy: XOM, CVX, and COP
2. Beverages: KO and PEP
3. Payments: V and MA
4. Banks: JPM, BAC, and WFC
5. Home improvement: HD and LOW

Only relationships within the same peer group are tested. This prevents arbitrary cross industry searches that may discover meaningless historical coincidences.

## Selection procedure

Each candidate hedge relation is fitted on the training period ending October 2020. Residual stationarity is evaluated with an augmented Dickey Fuller test. Benjamini Hochberg adjustment controls the false discovery rate across the complete family of nine tests.

Two pairs pass the corrected five percent threshold:

1. BAC with JPM, corrected probability 0.0066
2. PEP with KO, corrected probability 0.0123

No test or validation observations influence pair selection.

## Portfolio procedure

Hedge parameters are estimated from the trailing 252 observations. A causal 60 observation rolling score creates entry and exit signals. Each pair receives equal capital, total gross exposure is capped at one, positions affect returns one observation later, and three basis points of cost are charged per unit of turnover.

## Results

Validation performance is negative, with a Sharpe ratio of negative 0.31. The untouched test period produces a 1.94 percent annual return, 5.75 percent annual volatility, a Sharpe estimate of 0.36, and a maximum drawdown of 7.47 percent.

A stationary block bootstrap with 2000 samples gives a 95 percent Sharpe interval from negative 0.63 to 1.48. About 76 percent of samples have a positive Sharpe estimate.

## Interpretation

The positive test point estimate is not sufficient evidence of a dependable strategy. Validation is negative, year level results vary materially, and the bootstrap interval includes zero. The experiment does demonstrate a defensible selection and accounting process.

The next modeling stage will estimate predictive uncertainty and permit exposure only when the expected convergence is large relative to both uncertainty and trading costs.

## Limitations

Yahoo Finance data are convenient and reproducible but are not institutional point in time data. The study uses daily adjusted prices and does not directly model bid and ask spreads, short availability, market impact, corporate action timing, or intraday execution. The predefined groups reduce arbitrary searching but do not eliminate economic selection judgment.
