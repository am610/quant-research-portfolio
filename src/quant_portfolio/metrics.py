"""Performance measures used consistently across every study."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def equity_curve(returns: pd.Series) -> pd.Series:
    """Convert simple periodic returns into cumulative wealth."""

    return (1.0 + returns.fillna(0.0)).cumprod()


def maximum_drawdown(returns: pd.Series) -> float:
    """Return the most negative peak relative decline."""

    wealth = equity_curve(returns)
    return float((wealth / wealth.cummax() - 1.0).min())


def performance_summary(returns: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    """Calculate a compact set of annualized portfolio statistics."""

    clean = returns.dropna().astype(float)
    if clean.empty:
        raise ValueError("returns cannot be empty")
    years = len(clean) / periods_per_year
    ending_wealth = float(equity_curve(clean).iloc[-1])
    annual_return = ending_wealth ** (1.0 / years) - 1.0 if ending_wealth > 0 else np.nan
    annual_volatility = float(clean.std(ddof=1) * math.sqrt(periods_per_year))
    sharpe = (
        float(clean.mean() / clean.std(ddof=1) * math.sqrt(periods_per_year))
        if clean.std(ddof=1) > 0
        else np.nan
    )
    downside = clean[clean < 0].std(ddof=1)
    sortino = (
        float(clean.mean() / downside * math.sqrt(periods_per_year))
        if pd.notna(downside) and downside > 0
        else np.nan
    )
    return {
        "annual_return": float(annual_return),
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "maximum_drawdown": maximum_drawdown(clean),
        "ending_wealth": ending_wealth,
        "positive_period_fraction": float((clean > 0).mean()),
    }


def stationary_block_bootstrap(
    returns: pd.Series,
    samples: int = 1000,
    expected_block_length: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate uncertainty while approximately preserving serial dependence."""

    values = returns.dropna().to_numpy(dtype=float)
    if len(values) < 2:
        raise ValueError("At least two returns are required")
    if samples < 1 or expected_block_length < 1:
        raise ValueError("samples and expected_block_length must be positive")
    rng = np.random.default_rng(seed)
    restart_probability = 1.0 / expected_block_length
    output = []
    for _ in range(samples):
        indices = np.empty(len(values), dtype=int)
        indices[0] = rng.integers(0, len(values))
        for position in range(1, len(values)):
            if rng.random() < restart_probability:
                indices[position] = rng.integers(0, len(values))
            else:
                indices[position] = (indices[position - 1] + 1) % len(values)
        sample = pd.Series(values[indices])
        output.append(performance_summary(sample))
    return pd.DataFrame(output)

