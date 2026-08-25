import numpy as np
import pandas as pd

from quant_portfolio.diagnostics import exposure_summary, market_beta


def test_market_beta_recovers_linear_sensitivity() -> None:
    market = pd.Series(np.linspace(-0.02, 0.02, 100))
    strategy = 0.25 * market
    assert np.isclose(market_beta(strategy, market), 0.25)


def test_exposure_summary_reports_market_neutral_weights() -> None:
    weights = pd.DataFrame({"A": [0.5, 0.0], "B": [-0.5, 0.0]})
    returns = pd.Series([0.01, 0.0])
    summary = exposure_summary(weights, returns, returns)
    assert summary["maximum_gross_exposure"] == 1.0
    assert summary["maximum_absolute_net_exposure"] == 0.0
