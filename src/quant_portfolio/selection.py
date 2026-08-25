"""Training only selection tools for candidate statistical arbitrage pairs."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from .statarb import fit_pair_model, pair_spread


def benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    """Control false discoveries across a family of pair tests."""

    if pvalues.empty:
        return pvalues.astype(float)
    if pvalues.isna().any() or ((pvalues < 0) | (pvalues > 1)).any():
        raise ValueError("pvalues must be finite and lie between zero and one")
    ordered = pvalues.sort_values()
    ranks = np.arange(1, len(ordered) + 1)
    adjusted = ordered.to_numpy() * len(ordered) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    return pd.Series(adjusted, index=ordered.index).reindex(pvalues.index)


def estimate_half_life(spread: pd.Series) -> float:
    """Estimate spread half life from a first order autoregression."""

    sample = pd.concat([spread, spread.shift(1)], axis=1).dropna()
    sample.columns = ["current", "lagged"]
    if len(sample) < 20:
        return float("inf")
    design = np.column_stack([np.ones(len(sample)), sample["lagged"].to_numpy()])
    coefficients, *_ = np.linalg.lstsq(design, sample["current"].to_numpy(), rcond=None)
    phi = float(coefficients[1])
    if not 0 < phi < 1:
        return float("inf")
    return float(np.log(0.5) / np.log(phi))


def screen_pairs(training_prices: pd.DataFrame) -> pd.DataFrame:
    """Rank every pair using only the supplied training observations."""

    if training_prices.shape[1] < 2:
        raise ValueError("At least two assets are required")
    if (training_prices <= 0).any().any():
        raise ValueError("Prices must be strictly positive")

    records: list[dict[str, float | str]] = []
    log_prices = np.log(training_prices)
    for independent, dependent in combinations(training_prices.columns, 2):
        sample = log_prices[[dependent, independent]].dropna()
        model = fit_pair_model(sample, dependent, independent)
        spread = pair_spread(sample, model, dependent, independent)
        statistic, pvalue, *_ = adfuller(spread, regression="c", autolag="AIC")
        records.append(
            {
                "dependent": dependent,
                "independent": independent,
                "adf_statistic": float(statistic),
                "adf_pvalue": float(pvalue),
                "half_life": estimate_half_life(spread),
                "hedge_ratio": model.hedge_ratio,
            }
        )
    table = pd.DataFrame(records)
    table["adjusted_pvalue"] = benjamini_hochberg(table["adf_pvalue"])
    return table.sort_values(["adjusted_pvalue", "half_life"]).reset_index(drop=True)


def select_diverse_pairs(screen: pd.DataFrame, count: int = 3) -> pd.DataFrame:
    """Select ranked pairs while limiting repeated exposure to one asset."""

    if count < 1:
        raise ValueError("count must be positive")
    selected = []
    asset_usage: dict[str, int] = {}
    for _, row in screen.iterrows():
        dependent = str(row["dependent"])
        independent = str(row["independent"])
        if asset_usage.get(dependent, 0) >= 2 or asset_usage.get(independent, 0) >= 2:
            continue
        selected.append(row)
        asset_usage[dependent] = asset_usage.get(dependent, 0) + 1
        asset_usage[independent] = asset_usage.get(independent, 0) + 1
        if len(selected) == count:
            break
    return pd.DataFrame(selected).reset_index(drop=True)

