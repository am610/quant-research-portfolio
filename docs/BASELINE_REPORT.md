# Statistical Arbitrage Baseline

## Research question

Can a simple residual mean reversion rule applied to two related energy exchange traded funds generate stable returns after trading costs?

## Experimental design

XOP is modeled as a linear function of XLE in log price space. The hedge relation is fitted using only the training period. The residual spread is converted to a causal rolling score using a 60 observation window.

The strategy enters when the absolute score reaches two standard deviations and exits when it returns within one half standard deviation of zero. Portfolio weights have one unit of gross exposure. Target positions are delayed by one observation before returns are applied.

The chronological partitions are:

1. Training from January 2012 through October 2020
2. Validation from October 2020 through September 2023
3. Test from September 2023 through August 2026

The return calculation includes three basis points of linear cost per unit of turnover.

## Results

Validation performance was negative, with a Sharpe ratio of negative 0.48 and a maximum drawdown of 15.41 percent.

The untouched test period produced a 1.34 percent annual return, 5.02 percent annual volatility, a Sharpe ratio of 0.29, and a maximum drawdown of 9.40 percent.

A stationary block bootstrap with 1000 samples and an expected block length of 20 observations produced a 95 percent Sharpe interval from negative 0.96 to 1.46. Seventy percent of bootstrap samples had a positive Sharpe ratio.

## Interpretation

The point estimate is mildly positive, but the uncertainty is too large to claim reliable profitability. The negative validation result and persistent movement in the residual spread suggest that the fixed hedge relation and fixed thresholds do not adapt adequately to structural changes.

This is the desired role of the baseline. It establishes what simple modeling can and cannot achieve before additional complexity is introduced.

## Next hypothesis

The next study will test whether observable volatility, correlation, liquidity, and spread persistence features can identify regimes in which mean reversion is more credible. Predictions will be filtered using calibrated uncertainty, and any improvement must survive the same chronological and bootstrap evaluation.

## Limitations

Yahoo Finance adjusted daily observations are convenient and publicly reproducible but are not institutional quality point in time data. The current model does not include bid and ask spread estimates, short availability, market impact, tax effects, or intraday execution. The three basis point cost is a transparent scenario assumption rather than a claim about achievable execution.
