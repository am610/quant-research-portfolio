"""Produce the final benchmark, exposure, and robustness synthesis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_portfolio.costs import portfolio_returns
from quant_portfolio.data import chronological_split, load_yahoo_daily
from quant_portfolio.diagnostics import exposure_summary
from quant_portfolio.metrics import equity_curve, performance_summary
from quant_portfolio.pca import residual_reversal_weights, walk_forward_pca_residuals
from run_universe_study import UNIVERSE, build_selected_pair_portfolio


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"
OUTPUT_DIR = ROOT / "outputs"


def json_safe(value):
    """Replace nonfinite numerical values with JSON null values."""

    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def model_result(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    test_index: pd.Index,
    market_returns: pd.Series,
    cost_bps: float = 3.0,
    borrow_bps_annual: float = 50.0,
    execution_lag: int = 1,
) -> tuple[pd.Series, dict[str, float], dict[str, float]]:
    net = portfolio_returns(
        weights,
        asset_returns,
        cost_bps=cost_bps,
        borrow_bps_annual=borrow_bps_annual,
        execution_lag=execution_lag,
    ).loc[test_index, "net_return"]
    return (
        net,
        performance_summary(net),
        exposure_summary(weights.loc[test_index], net, market_returns.loc[test_index]),
    )


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = UNIVERSE + ["SPY"]
    frames = load_yahoo_daily(symbols, "2012-01-01", "2026-08-24")
    all_prices = pd.concat({symbol: frames[symbol]["close"] for symbol in symbols}, axis=1).dropna()
    prices = all_prices[UNIVERSE]
    market_returns = all_prices["SPY"].pct_change().fillna(0.0)
    asset_returns = prices.pct_change().fillna(0.0)
    split = chronological_split(prices)

    pair_weights, _ = build_selected_pair_portfolio(prices, split.train)
    residuals = walk_forward_pca_residuals(asset_returns, components=3, lookback=252)
    pca_weights = residual_reversal_weights(residuals, signal_window=20, entry_score=0.5)

    pair_net, pair_summary, pair_exposure = model_result(
        pair_weights, asset_returns, split.test.index, market_returns
    )
    pca_net, pca_summary, pca_exposure = model_result(
        pca_weights, asset_returns, split.test.index, market_returns
    )
    spy_test = market_returns.loc[split.test.index]
    cash_test = pd.Series(0.0, index=split.test.index)

    parameter_cases = [
        ("Base", 252, 60, 2.0),
        ("Hedge 126", 126, 60, 2.0),
        ("Hedge 504", 504, 60, 2.0),
        ("Signal 40", 252, 40, 2.0),
        ("Signal 90", 252, 90, 2.0),
        ("Entry 1.5", 252, 60, 1.5),
        ("Entry 2.5", 252, 60, 2.5),
    ]
    parameter_records = []
    for label, hedge_lookback, signal_window, entry_z in parameter_cases:
        weights, _ = build_selected_pair_portfolio(
            prices,
            split.train,
            hedge_lookback=hedge_lookback,
            signal_window=signal_window,
            entry_z=entry_z,
        )
        _, summary, _ = model_result(weights, asset_returns, split.test.index, market_returns)
        parameter_records.append({"case": label, **summary})

    boundary_records = []
    for label, train_fraction, validation_fraction in [
        ("55 25 20", 0.55, 0.25),
        ("60 20 20", 0.60, 0.20),
        ("65 15 20", 0.65, 0.15),
    ]:
        boundary_split = chronological_split(prices, train_fraction, validation_fraction)
        weights, screen = build_selected_pair_portfolio(prices, boundary_split.train)
        _, summary, _ = model_result(
            weights, asset_returns, boundary_split.test.index, market_returns
        )
        boundary_records.append(
            {
                "boundary": label,
                "selected_pair_count": int((screen["family_adjusted_pvalue"] <= 0.05).sum()),
                **summary,
            }
        )

    execution_records = []
    for label, cost_bps, borrow_bps, lag in [
        ("Base", 3.0, 50.0, 1),
        ("Higher spread", 5.0, 50.0, 1),
        ("Higher borrow", 3.0, 100.0, 1),
        ("Two day delay", 3.0, 50.0, 2),
        ("Combined stress", 5.0, 100.0, 2),
    ]:
        _, summary, _ = model_result(
            pair_weights,
            asset_returns,
            split.test.index,
            market_returns,
            cost_bps,
            borrow_bps,
            lag,
        )
        execution_records.append({"scenario": label, **summary})

    comparison = {
        "Selected pairs": pair_summary,
        "PCA residuals": pca_summary,
        "SPY benchmark": performance_summary(spy_test),
        "Cash": performance_summary(cash_test),
    }
    result = {
        "common_assumptions": {
            "trading_cost_bps": 3.0,
            "borrow_cost_bps_annual": 50.0,
            "execution_lag_days": 1,
        },
        "model_comparison": comparison,
        "exposure_diagnostics": {
            "Selected pairs": pair_exposure,
            "PCA residuals": pca_exposure,
        },
        "parameter_robustness": parameter_records,
        "time_boundary_robustness": boundary_records,
        "execution_stress": execution_records,
    }
    result = json_safe(result)
    for destination in [ASSET_DIR / "final_validation_results.json", OUTPUT_DIR / "final_validation_results.json"]:
        destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")

    parameter_table = pd.DataFrame(parameter_records)
    boundary_table = pd.DataFrame(boundary_records)
    execution_table = pd.DataFrame(execution_records)
    parameter_table.to_csv(ASSET_DIR / "parameter_robustness.csv", index=False)
    boundary_table.to_csv(ASSET_DIR / "boundary_robustness.csv", index=False)
    execution_table.to_csv(ASSET_DIR / "execution_stress.csv", index=False)

    plt.style.use("dark_background")
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    figure.patch.set_facecolor("#08111f")
    for axis in axes.flat:
        axis.set_facecolor("#0d1b2a")
        axis.grid(color="#29445f", alpha=0.35)

    axes[0, 0].plot(equity_curve(pair_net), label="Selected pairs", color="#7aa2ff", linewidth=2)
    axes[0, 0].plot(equity_curve(pca_net), label="PCA residuals", color="#54d6c7", linewidth=2)
    axes[0, 0].plot(equity_curve(spy_test), label="SPY", color="#f4b942", linewidth=1.5)
    axes[0, 0].set_title("Common test period wealth", loc="left", weight="bold")
    axes[0, 0].set_ylabel("Growth of one dollar")
    axes[0, 0].legend(frameon=False)

    model_names = ["Selected pairs", "PCA residuals", "SPY benchmark"]
    sharpes = [comparison[name]["sharpe_ratio"] for name in model_names]
    axes[0, 1].bar(model_names, sharpes, color=["#7aa2ff", "#54d6c7", "#f4b942"])
    axes[0, 1].axhline(0, color="#afc4d8", linewidth=0.8)
    axes[0, 1].set_title("Sharpe estimates under common assumptions", loc="left", weight="bold")
    axes[0, 1].tick_params(axis="x", rotation=15)

    axes[1, 0].bar(parameter_table["case"], parameter_table["sharpe_ratio"], color="#7aa2ff")
    axes[1, 0].axhline(0, color="#afc4d8", linewidth=0.8)
    axes[1, 0].set_title("Pair parameter robustness", loc="left", weight="bold")
    axes[1, 0].set_ylabel("Test Sharpe ratio")
    axes[1, 0].tick_params(axis="x", rotation=28)

    axes[1, 1].bar(execution_table["scenario"], execution_table["sharpe_ratio"], color="#ff8a80")
    axes[1, 1].axhline(0, color="#afc4d8", linewidth=0.8)
    axes[1, 1].set_title("Execution stress", loc="left", weight="bold")
    axes[1, 1].set_ylabel("Test Sharpe ratio")
    axes[1, 1].tick_params(axis="x", rotation=28)

    figure.suptitle("Final validation and benchmark synthesis", fontsize=21, weight="bold", x=0.06, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    for destination in [ASSET_DIR / "final_validation_dashboard.png", OUTPUT_DIR / "final_validation_dashboard.png"]:
        figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
