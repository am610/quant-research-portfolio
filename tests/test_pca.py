import numpy as np
import pandas as pd

from quant_portfolio.pca import residual_reversal_weights, walk_forward_pca_residuals


def sample_returns(rows: int = 160, assets: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    common = rng.normal(0, 0.01, (rows, 1))
    noise = rng.normal(0, 0.004, (rows, assets))
    return pd.DataFrame(common + noise, columns=[f"A{index}" for index in range(assets)])


def test_walk_forward_pca_respects_lookback() -> None:
    returns = sample_returns()
    residuals = walk_forward_pca_residuals(returns, components=2, lookback=60)
    assert residuals.iloc[:60].isna().all().all()
    assert residuals.iloc[60:].notna().all().all()


def test_residual_weights_are_market_neutral_and_bounded() -> None:
    weights = residual_reversal_weights(sample_returns(), signal_window=20, entry_score=0.5)
    active = weights.abs().sum(axis=1) > 0
    assert np.allclose(weights.loc[active].sum(axis=1), 0.0)
    assert (weights.abs().sum(axis=1) <= 1.0 + 1e-12).all()
