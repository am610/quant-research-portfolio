"""Classical pair trading baseline with causal signal construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import portfolio_returns


@dataclass(frozen=True)
class PairModel:
    intercept: float
    hedge_ratio: float


def fit_pair_model(log_prices: pd.DataFrame, dependent: str, independent: str) -> PairModel:
    """Fit an ordinary least squares hedge relation on training data only."""

    sample = log_prices[[dependent, independent]].dropna()
    if len(sample) < 20:
        raise ValueError("At least twenty paired observations are required")
    design = np.column_stack([np.ones(len(sample)), sample[independent].to_numpy()])
    coefficients, *_ = np.linalg.lstsq(design, sample[dependent].to_numpy(), rcond=None)
    return PairModel(intercept=float(coefficients[0]), hedge_ratio=float(coefficients[1]))


def pair_spread(
    log_prices: pd.DataFrame,
    model: PairModel,
    dependent: str,
    independent: str,
) -> pd.Series:
    """Calculate the residual spread implied by a fitted hedge relation."""

    return (
        log_prices[dependent]
        - model.intercept
        - model.hedge_ratio * log_prices[independent]
    ).rename("spread")


def causal_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """Calculate a rolling score using information available at each timestamp."""

    if window < 2:
        raise ValueError("window must be at least two")
    mean = spread.rolling(window, min_periods=window).mean()
    scale = spread.rolling(window, min_periods=window).std(ddof=1)
    return ((spread - mean) / scale.replace(0.0, np.nan)).rename("zscore")


def threshold_positions(
    zscore: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.Series:
    """Create persistent spread positions from entry and exit thresholds."""

    if not 0 <= exit_z < entry_z:
        raise ValueError("Thresholds must satisfy zero <= exit_z < entry_z")
    positions = pd.Series(0.0, index=zscore.index, name="spread_position")
    current = 0.0
    for timestamp, value in zscore.items():
        if pd.isna(value):
            current = 0.0
        elif current == 0.0 and value >= entry_z:
            current = -1.0
        elif current == 0.0 and value <= -entry_z:
            current = 1.0
        elif current != 0.0 and abs(value) <= exit_z:
            current = 0.0
        positions.loc[timestamp] = current
    return positions


def pair_weights(
    position: pd.Series,
    model: PairModel,
    dependent: str,
    independent: str,
) -> pd.DataFrame:
    """Convert spread positions into unit gross exposure portfolio weights."""

    normalizer = 1.0 + abs(model.hedge_ratio)
    return pd.DataFrame(
        {
            dependent: position / normalizer,
            independent: -position * model.hedge_ratio / normalizer,
        },
        index=position.index,
    )


def run_pair_backtest(
    prices: pd.DataFrame,
    model: PairModel,
    dependent: str,
    independent: str,
    window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    cost_bps: float = 3.0,
) -> dict[str, pd.Series | pd.DataFrame]:
    """Run the transparent classical pair strategy on a price sample."""

    selected = prices[[dependent, independent]].dropna()
    if (selected <= 0).any().any():
        raise ValueError("Prices must be strictly positive")
    log_prices = np.log(selected)
    spread = pair_spread(log_prices, model, dependent, independent)
    zscore = causal_zscore(spread, window)
    position = threshold_positions(zscore, entry_z, exit_z)
    weights = pair_weights(position, model, dependent, independent)
    returns = selected.pct_change().fillna(0.0)
    pnl = portfolio_returns(weights, returns, cost_bps)
    return {"spread": spread, "zscore": zscore, "position": position, "weights": weights, "pnl": pnl}

