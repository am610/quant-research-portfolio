"""Compare pair and principal component baselines across trading costs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_portfolio.costs import portfolio_returns
from quant_portfolio.data import chronological_split, load_yahoo_daily
from quant_portfolio.metrics import equity_curve, performance_summary, stationary_block_bootstrap
from quant_portfolio.pca import residual_reversal_weights, walk_forward_pca_residuals
from run_universe_study import PEER_GROUPS, UNIVERSE, build_selected_pair_portfolio


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"
OUTPUT_DIR = ROOT / "outputs"
COST_GRID = [0.0, 1.0, 3.0, 5.0, 10.0]


def selection_score(returns: pd.Series) -> float:
    summary = performance_summary(returns)
    if not np.isfinite(summary["sharpe_ratio"]):
        return float("-inf")
    return float(summary["sharpe_ratio"] - abs(summary["maximum_drawdown"]))


def evaluate_costs(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    test_index: pd.Index,
) -> pd.DataFrame:
    records = []
    for cost_bps in COST_GRID:
        net = portfolio_returns(weights, asset_returns, cost_bps).loc[test_index, "net_return"]
        records.append({"cost_bps": cost_bps, **performance_summary(net)})
    return pd.DataFrame(records)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_yahoo_daily(UNIVERSE, "2012-01-01", "2026-08-24")
    prices = pd.concat({symbol: frames[symbol]["close"] for symbol in UNIVERSE}, axis=1).dropna()
    split = chronological_split(prices)
    asset_returns = prices.pct_change().fillna(0.0)

    pair_weights, screen = build_selected_pair_portfolio(prices, split.train)
    selected_pairs = screen.loc[screen["family_adjusted_pvalue"] <= 0.05]

    candidates = []
    residual_cache = {}
    for components in [1, 2, 3, 4]:
        residuals = walk_forward_pca_residuals(asset_returns, components=components, lookback=252)
        residual_cache[components] = residuals
        for signal_window in [5, 10, 20, 40]:
            for entry_score in [0.5, 1.0, 1.5]:
                weights = residual_reversal_weights(residuals, signal_window, entry_score)
                validation_returns = portfolio_returns(weights, asset_returns, 3.0).loc[
                    split.validation.index, "net_return"
                ]
                candidates.append(
                    {
                        "components": components,
                        "signal_window": signal_window,
                        "entry_score": entry_score,
                        "validation_score": selection_score(validation_returns),
                        "validation_sharpe": performance_summary(validation_returns)["sharpe_ratio"],
                    }
                )
    candidate_table = pd.DataFrame(candidates)
    selected_parameters = candidate_table.loc[candidate_table["validation_score"].idxmax()].to_dict()
    pca_weights = residual_reversal_weights(
        residual_cache[int(selected_parameters["components"])],
        int(selected_parameters["signal_window"]),
        float(selected_parameters["entry_score"]),
    )

    pair_costs = evaluate_costs(pair_weights, asset_returns, split.test.index)
    pca_costs = evaluate_costs(pca_weights, asset_returns, split.test.index)
    pair_test = portfolio_returns(pair_weights, asset_returns, 3.0).loc[
        split.test.index, "net_return"
    ]
    pca_test = portfolio_returns(pca_weights, asset_returns, 3.0).loc[
        split.test.index, "net_return"
    ]
    pca_bootstrap = stationary_block_bootstrap(pca_test, samples=2000, expected_block_length=20)

    result = {
        "peer_groups": PEER_GROUPS,
        "selected_pairs": selected_pairs[
            ["dependent", "independent", "peer_group", "family_adjusted_pvalue"]
        ].to_dict(orient="records"),
        "selected_pca_parameters": selected_parameters,
        "pair_test_at_three_bps": performance_summary(pair_test),
        "pca_test_at_three_bps": performance_summary(pca_test),
        "pca_test_sharpe_interval_95": [
            float(pca_bootstrap["sharpe_ratio"].quantile(0.025)),
            float(pca_bootstrap["sharpe_ratio"].quantile(0.975)),
        ],
        "pair_cost_sensitivity": pair_costs.to_dict(orient="records"),
        "pca_cost_sensitivity": pca_costs.to_dict(orient="records"),
    }
    for destination in [ASSET_DIR / "pca_cost_results.json", OUTPUT_DIR / "pca_cost_results.json"]:
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    candidate_table.to_csv(ASSET_DIR / "pca_validation_grid.csv", index=False)
    candidate_table.to_csv(OUTPUT_DIR / "pca_validation_grid.csv", index=False)

    plt.style.use("dark_background")
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    figure.patch.set_facecolor("#08111f")
    for axis in axes.flat:
        axis.set_facecolor("#0d1b2a")
        axis.grid(color="#29445f", alpha=0.35)

    axes[0, 0].plot(equity_curve(pair_test), label="Selected pairs", color="#7aa2ff", linewidth=2)
    axes[0, 0].plot(equity_curve(pca_test), label="PCA residuals", color="#54d6c7", linewidth=2)
    axes[0, 0].set_title("Untouched test wealth at three basis points", loc="left", weight="bold")
    axes[0, 0].set_ylabel("Growth of one dollar")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(pair_costs["cost_bps"], pair_costs["sharpe_ratio"], marker="o", label="Selected pairs", color="#7aa2ff")
    axes[0, 1].plot(pca_costs["cost_bps"], pca_costs["sharpe_ratio"], marker="o", label="PCA residuals", color="#54d6c7")
    axes[0, 1].axhline(0, color="#afc4d8", linewidth=0.8)
    axes[0, 1].set_title("Trading cost sensitivity", loc="left", weight="bold")
    axes[0, 1].set_xlabel("Cost per unit turnover in basis points")
    axes[0, 1].set_ylabel("Test Sharpe ratio")
    axes[0, 1].legend(frameon=False)

    best_by_components = candidate_table.groupby("components")["validation_sharpe"].max()
    axes[1, 0].bar(best_by_components.index.astype(str), best_by_components.values, color="#f4b942")
    axes[1, 0].axhline(0, color="#afc4d8", linewidth=0.8)
    axes[1, 0].set_title("Best validation Sharpe by factor count", loc="left", weight="bold")
    axes[1, 0].set_xlabel("Principal components")
    axes[1, 0].set_ylabel("Validation Sharpe ratio")

    pair_drawdown = equity_curve(pair_test) / equity_curve(pair_test).cummax() - 1.0
    pca_drawdown = equity_curve(pca_test) / equity_curve(pca_test).cummax() - 1.0
    axes[1, 1].plot(pair_drawdown, label="Selected pairs", color="#7aa2ff")
    axes[1, 1].plot(pca_drawdown, label="PCA residuals", color="#54d6c7")
    axes[1, 1].fill_between(pair_drawdown.index, 0, pair_drawdown, color="#7aa2ff", alpha=0.12)
    axes[1, 1].fill_between(pca_drawdown.index, 0, pca_drawdown, color="#54d6c7", alpha=0.12)
    axes[1, 1].set_title("Test drawdown comparison", loc="left", weight="bold")
    axes[1, 1].set_ylabel("Drawdown")
    axes[1, 1].legend(frameon=False)

    figure.suptitle("Classical baselines and implementation friction", fontsize=21, weight="bold", x=0.06, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    for destination in [ASSET_DIR / "pca_cost_dashboard.png", OUTPUT_DIR / "pca_cost_dashboard.png"]:
        figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
