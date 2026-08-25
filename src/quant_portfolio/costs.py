"""Transparent trading friction calculations."""

from __future__ import annotations

import pandas as pd


def turnover(weights: pd.DataFrame) -> pd.Series:
    """Calculate one way portfolio turnover from target weights."""

    if weights.empty:
        raise ValueError("weights cannot be empty")
    return weights.diff().abs().sum(axis=1).fillna(weights.iloc[0].abs().sum())


def linear_trading_costs(weights: pd.DataFrame, cost_bps: float) -> pd.Series:
    """Apply a linear cost to each unit of portfolio turnover."""

    if cost_bps < 0:
        raise ValueError("cost_bps cannot be negative")
    return turnover(weights) * cost_bps / 10_000.0


def portfolio_returns(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Calculate gross and net returns using lagged target weights."""

    aligned_weights, aligned_returns = weights.align(asset_returns, join="inner", axis=0)
    aligned_weights, aligned_returns = aligned_weights.align(
        aligned_returns, join="inner", axis=1
    )
    if aligned_weights.empty:
        raise ValueError("weights and returns have no overlapping observations")
    held_weights = aligned_weights.shift(1).fillna(0.0)
    gross = (held_weights * aligned_returns).sum(axis=1)
    costs = linear_trading_costs(aligned_weights, cost_bps)
    return pd.DataFrame({"gross_return": gross, "cost": costs, "net_return": gross - costs})

