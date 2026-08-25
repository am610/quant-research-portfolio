"""Causal diagnostics and filters for changing spread behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_ar1(spread: pd.Series, window: int = 120) -> pd.DataFrame:
    """Estimate a rolling first order autoregression using past data only."""

    if window < 20:
        raise ValueError("window must be at least twenty")
    lagged = spread.shift(1)
    output = pd.DataFrame(index=spread.index, columns=["ar1", "innovation_volatility"], dtype=float)
    for end in range(window, len(spread) + 1):
        sample = pd.concat([spread.iloc[end - window : end], lagged.iloc[end - window : end]], axis=1)
        sample.columns = ["current", "lagged"]
        sample = sample.dropna()
        if len(sample) < window - 2:
            continue
        design = np.column_stack([np.ones(len(sample)), sample["lagged"].to_numpy()])
        coefficients, *_ = np.linalg.lstsq(design, sample["current"].to_numpy(), rcond=None)
        residuals = sample["current"].to_numpy() - design @ coefficients
        timestamp = spread.index[end - 1]
        output.loc[timestamp, "ar1"] = coefficients[1]
        output.loc[timestamp, "innovation_volatility"] = residuals.std(ddof=2)
    output["half_life"] = np.where(
        (output["ar1"] > 0) & (output["ar1"] < 1),
        np.log(0.5) / np.log(output["ar1"]),
        np.inf,
    )
    return output


def regime_mask(
    diagnostics: pd.DataFrame,
    maximum_half_life: float,
    maximum_volatility: float,
) -> pd.Series:
    """Identify periods with sufficiently fast and stable mean reversion."""

    if maximum_half_life <= 0 or maximum_volatility <= 0:
        raise ValueError("Regime limits must be positive")
    mask = (
        diagnostics["half_life"].between(0, maximum_half_life, inclusive="both")
        & (diagnostics["innovation_volatility"] <= maximum_volatility)
    )
    return mask.fillna(False).rename("eligible_regime")


def filter_positions(position: pd.Series, eligible: pd.Series) -> pd.Series:
    """Set exposure to zero whenever the causal regime is not eligible."""

    aligned_position, aligned_eligible = position.align(eligible, join="left")
    return aligned_position.where(aligned_eligible.fillna(False), 0.0).rename(position.name)

