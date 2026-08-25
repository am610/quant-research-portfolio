import pandas as pd

from quant_portfolio.selection import benjamini_hochberg, select_diverse_pairs


def test_benjamini_hochberg_is_monotone_in_rank() -> None:
    values = pd.Series({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.20})
    adjusted = benjamini_hochberg(values)
    ordered = pd.DataFrame({"raw": values, "adjusted": adjusted}).sort_values("raw")
    assert ordered["adjusted"].is_monotonic_increasing
    assert (adjusted >= values).all()


def test_diverse_selection_limits_asset_reuse() -> None:
    screen = pd.DataFrame(
        {
            "dependent": ["B", "C", "D", "E"],
            "independent": ["A", "A", "A", "B"],
            "adjusted_pvalue": [0.01, 0.02, 0.03, 0.04],
            "half_life": [5, 6, 7, 8],
        }
    )
    selected = select_diverse_pairs(screen, count=3)
    usage = pd.concat([selected["dependent"], selected["independent"]]).value_counts()
    assert usage.max() <= 2
