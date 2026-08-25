"""Portfolio exposure and benchmark diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .costs import turnover


def market_beta(strategy_returns: pd.Series, market_returns: pd.Series) -> float:
    """Estimate ordinary least squares sensitivity to market returns."""

    strategy, market = strategy_returns.align(market_returns, join="inner")
    sample = pd.concat([strategy, market], axis=1).dropna()
    if len(sample) < 2 or sample.iloc[:, 1].var(ddof=1) == 0:
        return float("nan")
    return float(sample.cov().iloc[0, 1] / sample.iloc[:, 1].var(ddof=1))


def exposure_summary(
    weights: pd.DataFrame,
    strategy_returns: pd.Series,
    market_returns: pd.Series,
) -> dict[str, float]:
    """Summarize leverage, direction, concentration, turnover, and beta."""

    gross = weights.abs().sum(axis=1)
    net = weights.sum(axis=1)
    concentration = weights.abs().max(axis=1)
    return {
        "average_gross_exposure": float(gross.mean()),
        "maximum_gross_exposure": float(gross.max()),
        "average_absolute_net_exposure": float(net.abs().mean()),
        "maximum_absolute_net_exposure": float(net.abs().max()),
        "average_largest_position": float(concentration.mean()),
        "average_daily_turnover": float(turnover(weights).mean()),
        "active_day_fraction": float((gross > 0).mean()),
        "market_beta": market_beta(strategy_returns, market_returns),
    }

