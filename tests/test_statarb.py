import numpy as np
import pandas as pd

from quant_portfolio.statarb import (
    fit_pair_model,
    run_pair_backtest,
    threshold_positions,
    walk_forward_pair_state,
)


def synthetic_prices(rows: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    common = np.cumsum(rng.normal(0.0003, 0.01, rows))
    spread = np.zeros(rows)
    for index in range(1, rows):
        spread[index] = 0.92 * spread[index - 1] + rng.normal(0, 0.01)
    return pd.DataFrame(
        {"A": np.exp(4.5 + common + spread), "B": np.exp(4.2 + common)},
        index=pd.date_range("2020-01-01", periods=rows, freq="D"),
    )


def test_pair_model_recovers_positive_hedge_relation() -> None:
    prices = synthetic_prices()
    model = fit_pair_model(np.log(prices.iloc[:300]), "A", "B")
    assert 0.7 < model.hedge_ratio < 1.3


def test_pair_backtest_outputs_are_aligned() -> None:
    prices = synthetic_prices()
    model = fit_pair_model(np.log(prices.iloc[:300]), "A", "B")
    result = run_pair_backtest(prices.iloc[300:], model, "A", "B", window=30)
    assert result["pnl"].index.equals(prices.iloc[300:].index)
    assert set(result["pnl"].columns) == {
        "gross_return",
        "trading_cost",
        "borrow_cost",
        "net_return",
    }


def test_threshold_position_exits_near_zero() -> None:
    zscore = pd.Series([0.0, 2.2, 1.0, 0.4, -2.3, -0.2])
    assert threshold_positions(zscore).tolist() == [0.0, -1.0, -1.0, 0.0, 1.0, 0.0]


def test_walk_forward_state_starts_after_lookback() -> None:
    prices = synthetic_prices(200)
    state = walk_forward_pair_state(prices, "A", "B", lookback=60)
    assert state.iloc[:60].isna().all().all()
    assert state.iloc[60:].notna().all().all()
