import numpy as np
import pandas as pd

from quant_portfolio.regimes import filter_positions, regime_mask, rolling_ar1


def test_rolling_ar1_detects_persistent_process() -> None:
    rng = np.random.default_rng(2)
    values = np.zeros(500)
    for index in range(1, len(values)):
        values[index] = 0.8 * values[index - 1] + rng.normal(0, 0.1)
    diagnostics = rolling_ar1(pd.Series(values), window=120)
    assert 0.65 < diagnostics["ar1"].dropna().median() < 0.95


def test_regime_filter_removes_ineligible_positions() -> None:
    diagnostics = pd.DataFrame(
        {"half_life": [5.0, 50.0], "innovation_volatility": [0.1, 0.1]}
    )
    eligible = regime_mask(diagnostics, maximum_half_life=20, maximum_volatility=0.2)
    result = filter_positions(pd.Series([1.0, 1.0]), eligible)
    assert result.tolist() == [1.0, 0.0]

