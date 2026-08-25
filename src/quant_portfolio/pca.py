"""Walk forward principal component residual baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def walk_forward_pca_residuals(
    returns: pd.DataFrame,
    components: int = 3,
    lookback: int = 252,
) -> pd.DataFrame:
    """Remove trailing common return components without using future data."""

    if lookback < 20:
        raise ValueError("lookback must be at least twenty")
    if not 1 <= components < returns.shape[1]:
        raise ValueError("components must be positive and less than the asset count")
    clean = returns.fillna(0.0).astype(float)
    residuals = pd.DataFrame(np.nan, index=clean.index, columns=clean.columns)
    for location in range(lookback, len(clean)):
        history = clean.iloc[location - lookback : location]
        mean = history.mean()
        scale = history.std(ddof=1).replace(0.0, np.nan)
        if scale.isna().any():
            continue
        standardized_history = (history - mean) / scale
        model = PCA(n_components=components, svd_solver="full")
        model.fit(standardized_history.to_numpy())
        current = ((clean.iloc[location] - mean) / scale).to_numpy().reshape(1, -1)
        reconstruction = model.inverse_transform(model.transform(current))[0]
        residuals.iloc[location] = (current[0] - reconstruction) * scale.to_numpy()
    return residuals


def residual_reversal_weights(
    residual_returns: pd.DataFrame,
    signal_window: int = 20,
    entry_score: float = 1.0,
) -> pd.DataFrame:
    """Create market neutral weights that oppose unusually large residual moves."""

    if signal_window < 2 or entry_score <= 0:
        raise ValueError("signal_window and entry_score must be positive")
    cumulative = residual_returns.rolling(signal_window).sum()
    scale = residual_returns.rolling(signal_window).std(ddof=1) * np.sqrt(signal_window)
    score = cumulative / scale.replace(0.0, np.nan)
    raw = -score.where(score.abs() >= entry_score, 0.0).fillna(0.0)
    raw = raw.sub(raw.mean(axis=1), axis=0)
    gross = raw.abs().sum(axis=1)
    return raw.div(gross.where(gross > 0, 1.0), axis=0)
