# Principal Component Baseline and Cost Sensitivity

## Research question

Does a broad principal component residual strategy provide stronger statistical arbitrage evidence than selected pairs, and does that evidence survive plausible transaction costs?

## Principal component procedure

At each observation, principal components are fitted to the previous 252 daily returns. No future observations enter factor estimation. The current return is separated into common factor and residual components.

A market neutral reversal portfolio opposes unusually large cumulative residual movements. The number of components, signal window, and entry score are selected using validation data only. The selected model uses three components, a 20 observation signal window, and an entry score of 0.5.

## Test comparison

With no transaction costs, the principal component strategy produces a test Sharpe estimate of 0.55, compared with 0.42 for the selected pair portfolio.

At the standard three basis point assumption, the principal component Sharpe estimate falls to 0.13, while the selected pair portfolio retains a Sharpe estimate of 0.36. The principal component maximum drawdown is 9.12 percent, compared with 7.47 percent for selected pairs.

The principal component 95 percent bootstrap Sharpe interval spans negative 0.60 to 0.93.

## Cost sensitivity

The selected pair Sharpe estimates at zero, one, three, five, and ten basis points are 0.42, 0.40, 0.36, 0.33, and 0.23.

The principal component estimates at the same costs are 0.55, 0.41, 0.13, negative 0.15, and negative 0.86.

## Interpretation

The apparent advantage of the principal component model before costs is not robust to implementation friction. Its diversified residual positions create substantially greater turnover. A model that finds more short horizon structure can therefore have less economic value.

Neither model provides conclusive evidence of a persistent opportunity because both bootstrap intervals include zero. The primary contribution is a reproducible comparison that separates statistical predictability, turnover, costs, and uncertainty.

## Limitations

The linear cost scenarios do not model nonlinear market impact, bid and ask variation, borrow availability, or execution timing. Principal component loadings are estimated from adjusted daily returns, and their economic interpretation may change across regimes. Validation selection across a finite parameter grid can itself introduce selection optimism.
