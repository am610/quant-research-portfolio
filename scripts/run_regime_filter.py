"""Select a causal spread regime filter and evaluate it on locked test data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_portfolio.costs import portfolio_returns
from quant_portfolio.data import chronological_split, load_yahoo_daily
from quant_portfolio.metrics import equity_curve, performance_summary, stationary_block_bootstrap
from quant_portfolio.regimes import filter_positions, regime_mask, rolling_ar1
from quant_portfolio.statarb import fit_pair_model, pair_weights, run_pair_backtest


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"
OUTPUT_DIR = ROOT / "outputs"


def score_candidate(net_returns: pd.Series) -> float:
    summary = performance_summary(net_returns)
    if not np.isfinite(summary["sharpe_ratio"]) or (net_returns != 0).mean() < 0.02:
        return float("-inf")
    drawdown_penalty = abs(summary["maximum_drawdown"])
    return float(summary["sharpe_ratio"] - drawdown_penalty)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_yahoo_daily(["XLE", "XOP"], "2012-01-01", "2026-08-24")
    prices = pd.concat({symbol: frame["close"] for symbol, frame in frames.items()}, axis=1).dropna()
    split = chronological_split(prices)
    model = fit_pair_model(np.log(split.train), "XOP", "XLE")

    evaluation_prices = pd.concat([split.validation, split.test])
    baseline = run_pair_backtest(
        evaluation_prices, model, "XOP", "XLE", window=60, cost_bps=3.0
    )
    diagnostics = rolling_ar1(baseline["spread"], window=120)
    validation_index = split.validation.index
    test_index = split.test.index
    validation_volatility = diagnostics.loc[validation_index, "innovation_volatility"].dropna()

    candidates = []
    for maximum_half_life in [5.0, 10.0, 20.0, 40.0, 80.0]:
        for volatility_quantile in [0.50, 0.70, 0.90, 1.00]:
            maximum_volatility = float(validation_volatility.quantile(volatility_quantile))
            eligible = regime_mask(diagnostics, maximum_half_life, maximum_volatility)
            filtered = filter_positions(baseline["position"], eligible)
            weights = pair_weights(filtered, model, "XOP", "XLE")
            returns = evaluation_prices.pct_change().fillna(0.0)
            pnl = portfolio_returns(weights, returns, cost_bps=3.0)
            validation_returns = pnl.loc[validation_index, "net_return"]
            validation_exposure = float((filtered.loc[validation_index] != 0).mean())
            candidates.append(
                {
                    "maximum_half_life": maximum_half_life,
                    "volatility_quantile": volatility_quantile,
                    "maximum_volatility": maximum_volatility,
                    "validation_score": score_candidate(validation_returns),
                    "validation_exposure_fraction": validation_exposure,
                }
            )
    selected = max(candidates, key=lambda item: item["validation_score"])
    eligible = regime_mask(
        diagnostics,
        selected["maximum_half_life"],
        selected["maximum_volatility"],
    )
    filtered = filter_positions(baseline["position"], eligible)
    weights = pair_weights(filtered, model, "XOP", "XLE")
    pnl = portfolio_returns(weights, evaluation_prices.pct_change().fillna(0.0), cost_bps=3.0)

    baseline_test = baseline["pnl"].loc[test_index, "net_return"]
    filtered_test = pnl.loc[test_index, "net_return"]
    bootstrap = stationary_block_bootstrap(filtered_test, samples=1000, expected_block_length=20)
    result = {
        "selection_rule": "Highest validation Sharpe minus absolute maximum drawdown",
        "selected_parameters": selected,
        "eligible_test_fraction": float(eligible.loc[test_index].mean()),
        "baseline_test": performance_summary(baseline_test),
        "filtered_test": performance_summary(filtered_test),
        "filtered_test_sharpe_interval_95": [
            float(bootstrap["sharpe_ratio"].quantile(0.025)),
            float(bootstrap["sharpe_ratio"].quantile(0.975)),
        ],
    }
    for destination in [ASSET_DIR / "regime_filter_results.json", OUTPUT_DIR / "regime_filter_results.json"]:
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")

    plt.style.use("dark_background")
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    figure.patch.set_facecolor("#08111f")
    for axis in axes:
        axis.set_facecolor("#0d1b2a")
        axis.grid(color="#29445f", alpha=0.35)
    axes[0].plot(diagnostics.loc[test_index, "half_life"].clip(upper=100), color="#54d6c7")
    axes[0].axhline(selected["maximum_half_life"], color="#f4b942", linestyle=":")
    axes[0].set_title("Causal spread persistence", loc="left", weight="bold", fontsize=14)
    axes[0].set_ylabel("Estimated half life")
    axes[1].plot(equity_curve(baseline_test), label="Fixed baseline", color="#8da2b8")
    axes[1].plot(equity_curve(filtered_test), label="Regime filtered", color="#7aa2ff", linewidth=2)
    axes[1].set_title("Untouched test comparison", loc="left", weight="bold", fontsize=14)
    axes[1].set_ylabel("Growth of one dollar")
    axes[1].legend(frameon=False)
    figure.suptitle("Does a causal regime filter improve robustness?", fontsize=19, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    for destination in [ASSET_DIR / "regime_filter.png", OUTPUT_DIR / "regime_filter.png"]:
        figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
