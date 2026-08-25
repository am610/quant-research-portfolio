"""Run a training selected, walk forward statistical arbitrage universe study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_portfolio.costs import portfolio_returns
from quant_portfolio.data import chronological_split, load_yahoo_daily
from quant_portfolio.metrics import equity_curve, performance_summary, stationary_block_bootstrap
from quant_portfolio.selection import benjamini_hochberg, screen_pairs
from quant_portfolio.statarb import (
    causal_zscore,
    dynamic_pair_weights,
    threshold_positions,
    walk_forward_pair_state,
)


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"
OUTPUT_DIR = ROOT / "outputs"
PEER_GROUPS = {
    "Integrated energy": ["XOM", "CVX", "COP"],
    "Beverages": ["KO", "PEP"],
    "Payments": ["V", "MA"],
    "Banks": ["JPM", "BAC", "WFC"],
    "Home improvement": ["HD", "LOW"],
}
UNIVERSE = list(dict.fromkeys(symbol for group in PEER_GROUPS.values() for symbol in group))


def rolling_sharpe(returns: pd.Series, window: int = 126) -> pd.Series:
    mean = returns.rolling(window).mean()
    volatility = returns.rolling(window).std(ddof=1)
    return mean / volatility.replace(0, np.nan) * np.sqrt(252)


def build_pair_weights(
    prices: pd.DataFrame,
    dependent: str,
    independent: str,
) -> pd.DataFrame:
    state = walk_forward_pair_state(prices, dependent, independent, lookback=252)
    score = causal_zscore(state["spread"], window=60)
    position = threshold_positions(score, entry_z=2.0, exit_z=0.5)
    return dynamic_pair_weights(position, state["hedge_ratio"], dependent, independent)


def build_selected_pair_portfolio(
    prices: pd.DataFrame,
    training_prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select eligible peer pairs on training data and assemble their weights."""

    screens = []
    for group_name, symbols in PEER_GROUPS.items():
        group_screen = screen_pairs(training_prices[symbols])
        group_screen["peer_group"] = group_name
        screens.append(group_screen)
    screen = pd.concat(screens, ignore_index=True)
    screen["family_adjusted_pvalue"] = benjamini_hochberg(screen["adf_pvalue"])
    screen = screen.sort_values(["family_adjusted_pvalue", "half_life"]).reset_index(drop=True)
    selected = screen.loc[screen["family_adjusted_pvalue"] <= 0.05].copy()
    if selected.empty:
        raise RuntimeError("No pair passed the predefined false discovery threshold")

    component_weights = []
    for _, pair in selected.iterrows():
        component_weights.append(
            build_pair_weights(prices, str(pair["dependent"]), str(pair["independent"]))
        )
    combined = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for weights in component_weights:
        combined = combined.add(weights.reindex(columns=prices.columns, fill_value=0.0), fill_value=0.0)
    combined /= len(component_weights)
    gross_exposure = combined.abs().sum(axis=1)
    combined = combined.div(gross_exposure.where(gross_exposure > 1.0, 1.0), axis=0)
    return combined, screen


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_yahoo_daily(UNIVERSE, "2012-01-01", "2026-08-24")
    prices = pd.concat({symbol: frames[symbol]["close"] for symbol in UNIVERSE}, axis=1).dropna()
    split = chronological_split(prices)

    combined, screen = build_selected_pair_portfolio(prices, split.train)
    selected = screen.loc[screen["family_adjusted_pvalue"] <= 0.05].copy()

    pnl = portfolio_returns(combined, prices.pct_change().fillna(0.0), cost_bps=3.0)
    validation_returns = pnl.loc[split.validation.index, "net_return"]
    test_returns = pnl.loc[split.test.index, "net_return"]
    bootstrap = stationary_block_bootstrap(test_returns, samples=2000, expected_block_length=20)

    yearly = test_returns.groupby(test_returns.index.year).apply(
        lambda values: pd.Series(performance_summary(values))
    ).unstack()
    selection_records = selected.replace([np.inf, -np.inf], None).to_dict(orient="records")
    result = {
        "universe": UNIVERSE,
        "selection_data_end": str(split.train.index.max().date()),
        "selection_rule": "Predefined peer groups with family false discovery rate at five percent",
        "selected_pairs": selection_records,
        "validation": performance_summary(validation_returns),
        "test": performance_summary(test_returns),
        "test_sharpe_interval_95": [
            float(bootstrap["sharpe_ratio"].quantile(0.025)),
            float(bootstrap["sharpe_ratio"].quantile(0.975)),
        ],
        "test_probability_positive_sharpe": float((bootstrap["sharpe_ratio"] > 0).mean()),
        "test_average_daily_turnover": float(combined.loc[split.test.index].diff().abs().sum(axis=1).mean()),
        "yearly_test_results": yearly.replace([np.inf, -np.inf, np.nan], None).to_dict(orient="index"),
    }
    for destination in [ASSET_DIR / "universe_results.json", OUTPUT_DIR / "universe_results.json"]:
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    screen.to_csv(ASSET_DIR / "pair_screen.csv", index=False)
    screen.to_csv(OUTPUT_DIR / "pair_screen.csv", index=False)

    plt.style.use("dark_background")
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    figure.patch.set_facecolor("#08111f")
    for axis in axes.flat:
        axis.set_facecolor("#0d1b2a")
        axis.grid(color="#29445f", alpha=0.35)

    plotted = screen.replace([np.inf, -np.inf], np.nan).dropna(subset=["half_life"])
    significant = plotted["family_adjusted_pvalue"] <= 0.05
    axes[0, 0].scatter(
        plotted.loc[~significant, "half_life"],
        plotted.loc[~significant, "family_adjusted_pvalue"],
        c="#71869b",
        s=55,
        alpha=0.65,
        label="Not selected",
    )
    axes[0, 0].scatter(
        plotted.loc[significant, "half_life"],
        plotted.loc[significant, "family_adjusted_pvalue"],
        c="#54d6c7",
        edgecolor="#d9fffa",
        linewidth=0.8,
        s=105,
        label="Selected",
    )
    for label_index, (_, row) in enumerate(plotted.loc[significant].iterrows()):
        vertical_offset = 12 if label_index % 2 == 0 else -20
        axes[0, 0].annotate(
            f"{row['dependent']}:{row['independent']}",
            (row["half_life"], row["family_adjusted_pvalue"]),
            xytext=(8, vertical_offset),
            textcoords="offset points",
            fontsize=10,
            weight="bold",
        )
    axes[0, 0].axhline(0.05, color="#ff6b6b", linestyle=":")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_title("Training only pair screen", loc="left", weight="bold")
    axes[0, 0].set_xlabel("Estimated half life")
    axes[0, 0].set_ylabel("Family adjusted stationarity probability")
    axes[0, 0].legend(frameon=False, loc="upper left")

    test_wealth = equity_curve(test_returns)
    axes[0, 1].plot(test_wealth, color="#7aa2ff", linewidth=2)
    axes[0, 1].fill_between(test_wealth.index, 1.0, test_wealth, color="#7aa2ff", alpha=0.15)
    axes[0, 1].set_title("Untouched test wealth", loc="left", weight="bold")
    axes[0, 1].set_ylabel("Growth of one dollar")

    rolling = rolling_sharpe(test_returns)
    axes[1, 0].plot(rolling, color="#f4b942")
    axes[1, 0].axhline(0, color="#afc4d8", linewidth=0.8)
    axes[1, 0].set_title("Rolling six month Sharpe", loc="left", weight="bold")
    axes[1, 0].set_ylabel("Sharpe ratio")

    drawdown = test_wealth / test_wealth.cummax() - 1.0
    axes[1, 1].fill_between(drawdown.index, 0, drawdown, color="#ff6b6b", alpha=0.45)
    axes[1, 1].set_title("Drawdown path", loc="left", weight="bold")
    axes[1, 1].set_ylabel("Drawdown")

    figure.suptitle(
        "Predefined peer statistical arbitrage study",
        fontsize=21,
        weight="bold",
        x=0.06,
        ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    for destination in [ASSET_DIR / "universe_dashboard.png", OUTPUT_DIR / "universe_dashboard.png"]:
        figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
