# Final Research Conclusions

## Objective

This project evaluates whether mean reverting relationships among liquid United States equities remain useful after controlling selection, temporal leakage, turnover, borrowing costs, execution delays, parameter sensitivity, and uncertainty.

## Models

The selected pair model searches only within five predefined economic peer groups. Stationarity tests are performed on training data and corrected across the complete test family. Eligible relationships use trailing hedge estimates and causal threshold signals.

The principal component model estimates common return factors from trailing data and trades market neutral residual reversals. Its parameters are selected on validation data.

## Common test comparison

Under three basis points of turnover cost, 50 basis points of annual short borrow cost, and one day delayed execution, the selected pair model produces a 1.86 percent annual return, 5.75 percent volatility, a Sharpe estimate of 0.35, and a 7.51 percent maximum drawdown.

The principal component model produces a 0.43 percent annual return, 7.60 percent volatility, a Sharpe estimate of 0.09, and a 9.38 percent maximum drawdown.

SPY produces a 21.47 percent annual return and a 1.34 Sharpe estimate during the same period. This does not invalidate the arbitrage comparison because SPY carries direct market exposure. The selected pair and principal component portfolios have estimated market betas near 0.02. It does show that low beta diversification came with a large opportunity cost during this particular rising market.

## Implementation findings

The selected pair strategy averages about 4.38 percent daily turnover and is active on about 58 percent of test days. Its Sharpe estimate remains positive when trading cost rises to five basis points, annual borrow cost doubles, execution is delayed by two days, and those stresses are combined.

The principal component strategy averages about 42.21 percent daily turnover and is continuously active. This explains why its attractive result before costs deteriorates rapidly after costs.

## Robustness findings

The pair result is not invariant to modeling choices. Shorter signal memory performs much better in the test period, while a 504 observation hedge window produces a negative result. These alternatives were not used to replace the locked base configuration after observing the test results.

Changing the training boundary alters which pairs pass selection. The test result remains positive across the three reported partitions, but the available evidence is limited to one market history.

## Conclusion

The project does not establish a deployable profitable strategy. It establishes a transparent research process and demonstrates why apparently promising signals must be tested against selection effects, implementation friction, uncertainty, benchmarks, and parameter instability.

The most defensible empirical conclusion is that sparse selected pairs are more resilient to costs than broad principal component residual trading in this experiment. Evidence for persistent excess performance remains inconclusive.

## Next project

The next portfolio project will focus on calibrated predictive uncertainty. It will ask whether abstaining when model confidence is weak improves economic results under distribution shift.
