"""Run the first public statistical arbitrage baseline study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_portfolio.data import chronological_split, load_yahoo_daily
from quant_portfolio.metrics import equity_curve, performance_summary, stationary_block_bootstrap
from quant_portfolio.statarb import fit_pair_model, run_pair_backtest


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"
OUTPUT_DIR = ROOT / "outputs"


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    symbols = ["XLE", "XOP"]
    frames = load_yahoo_daily(symbols, "2012-01-01", "2026-08-24")
    prices = pd.concat({symbol: frames[symbol]["close"] for symbol in symbols}, axis=1).dropna()
    split = chronological_split(prices)

    model = fit_pair_model(np.log(split.train), dependent="XOP", independent="XLE")
    validation = run_pair_backtest(
        split.validation,
        model,
        dependent="XOP",
        independent="XLE",
        window=60,
        entry_z=2.0,
        exit_z=0.5,
        cost_bps=3.0,
    )
    test = run_pair_backtest(
        split.test,
        model,
        dependent="XOP",
        independent="XLE",
        window=60,
        entry_z=2.0,
        exit_z=0.5,
        cost_bps=3.0,
    )

    summary = {
        "symbols": symbols,
        "retrieval_end": "2026-08-24",
        "train_start": str(split.train.index.min().date()),
        "train_end": str(split.train.index.max().date()),
        "validation_start": str(split.validation.index.min().date()),
        "validation_end": str(split.validation.index.max().date()),
        "test_start": str(split.test.index.min().date()),
        "test_end": str(split.test.index.max().date()),
        "hedge_ratio": model.hedge_ratio,
        "validation": performance_summary(validation["pnl"]["net_return"]),
        "test": performance_summary(test["pnl"]["net_return"]),
    }
    bootstrap = stationary_block_bootstrap(
        test["pnl"]["net_return"], samples=1000, expected_block_length=20
    )
    summary["test_sharpe_interval_95"] = [
        float(bootstrap["sharpe_ratio"].quantile(0.025)),
        float(bootstrap["sharpe_ratio"].quantile(0.975)),
    ]
    summary["test_probability_positive_sharpe"] = float(
        (bootstrap["sharpe_ratio"] > 0).mean()
    )

    for destination in [ASSET_DIR / "pair_baseline_results.json", OUTPUT_DIR / "pair_baseline_results.json"]:
        destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")

    plt.style.use("dark_background")
    figure, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    figure.patch.set_facecolor("#08111f")
    for axis in axes:
        axis.set_facecolor("#0d1b2a")
        axis.grid(color="#29445f", alpha=0.35)

    axes[0].plot(test["spread"], color="#54d6c7", linewidth=1.4)
    axes[0].set_title("XOP and XLE residual spread", loc="left", fontsize=14, weight="bold")
    axes[0].set_ylabel("Log residual")

    axes[1].plot(test["zscore"], color="#f4b942", linewidth=1.2)
    axes[1].axhline(2.0, color="#ff6b6b", linestyle=":")
    axes[1].axhline(-2.0, color="#ff6b6b", linestyle=":")
    axes[1].axhline(0.0, color="#afc4d8", linewidth=0.7)
    axes[1].set_title("Causal rolling score", loc="left", fontsize=14, weight="bold")
    axes[1].set_ylabel("Standard deviations")

    wealth = equity_curve(test["pnl"]["net_return"])
    axes[2].plot(wealth, color="#7aa2ff", linewidth=2.0)
    axes[2].fill_between(wealth.index, 1.0, wealth.values, color="#7aa2ff", alpha=0.15)
    axes[2].set_title("Untouched test period net wealth", loc="left", fontsize=14, weight="bold")
    axes[2].set_ylabel("Growth of one dollar")
    axes[2].set_xlabel("Date")

    figure.suptitle(
        "A transparent statistical arbitrage baseline",
        x=0.125,
        y=0.995,
        ha="left",
        fontsize=20,
        weight="bold",
        color="#f1f5f9",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    for destination in [ASSET_DIR / "pair_baseline.png", OUTPUT_DIR / "pair_baseline.png"]:
        figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

