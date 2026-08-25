import pandas as pd

from quant_portfolio.costs import portfolio_returns, turnover


def test_weights_are_lagged_to_prevent_lookahead() -> None:
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    weights = pd.DataFrame({"A": [0.0, 1.0, 1.0]}, index=index)
    returns = pd.DataFrame({"A": [0.0, 0.5, 0.1]}, index=index)
    result = portfolio_returns(weights, returns)
    assert result["gross_return"].tolist() == [0.0, 0.0, 0.1]


def test_execution_delay_and_borrow_cost_are_applied() -> None:
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    weights = pd.DataFrame({"A": [0.0, -1.0, -1.0, -1.0]}, index=index)
    returns = pd.DataFrame({"A": [0.0, 0.1, 0.2, 0.3]}, index=index)
    result = portfolio_returns(
        weights,
        returns,
        borrow_bps_annual=252.0,
        execution_lag=2,
    )
    assert result.loc[index[2], "gross_return"] == 0.0
    assert result.loc[index[3], "gross_return"] == -0.3
    assert result.loc[index[3], "borrow_cost"] == 0.0001


def test_turnover_counts_target_weight_changes() -> None:
    weights = pd.DataFrame({"A": [0.0, 0.5, -0.5], "B": [0.0, -0.5, 0.5]})
    assert turnover(weights).tolist() == [0.0, 1.0, 2.0]
