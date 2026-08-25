import numpy as np
import pandas as pd
import pytest

from quant_portfolio.data import chronological_split, validate_price_frame


def price_frame(rows: int = 20) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=rows, freq="D")
    close = np.linspace(100, 120, rows)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 10},
        index=index,
    )


def test_chronological_split_has_no_overlap() -> None:
    split = chronological_split(price_frame())
    assert len(split.train) == 12
    assert len(split.validation) == 4
    assert len(split.test) == 4
    assert split.train.index.max() < split.validation.index.min() < split.test.index.min()


def test_duplicate_timestamps_are_rejected() -> None:
    frame = price_frame()
    duplicated = pd.concat([frame, frame.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"):
        validate_price_frame(duplicated)

