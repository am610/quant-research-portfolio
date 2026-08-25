"""Market data loading and chronological validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from urllib.request import urlopen

import pandas as pd


REQUIRED_PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class ChronologicalSplit:
    """Contiguous research partitions that preserve time order."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def validate_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a clean price frame or raise on dangerous data problems."""

    clean = frame.copy()
    clean.columns = [str(column).lower() for column in clean.columns]
    missing = sorted(set(REQUIRED_PRICE_COLUMNS).difference(clean.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not isinstance(clean.index, pd.DatetimeIndex):
        raise TypeError("Price frame index must be a DatetimeIndex")
    if clean.index.has_duplicates:
        raise ValueError("Price frame contains duplicate timestamps")
    if not clean.index.is_monotonic_increasing:
        clean = clean.sort_index()
    if clean[list(REQUIRED_PRICE_COLUMNS)].isna().any().any():
        raise ValueError("Price frame contains missing values")
    if (clean[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Prices must be strictly positive")
    if (clean["volume"] < 0).any():
        raise ValueError("Volume cannot be negative")
    return clean.loc[:, list(REQUIRED_PRICE_COLUMNS)].astype(float)


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> ChronologicalSplit:
    """Split observations into contiguous train, validation, and test sets."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must lie between zero and one")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Training and validation fractions must sum to less than one")
    if len(frame) < 10:
        raise ValueError("At least ten observations are required")

    train_end = int(len(frame) * train_fraction)
    validation_end = int(len(frame) * (train_fraction + validation_fraction))
    return ChronologicalSplit(
        train=frame.iloc[:train_end].copy(),
        validation=frame.iloc[train_end:validation_end].copy(),
        test=frame.iloc[validation_end:].copy(),
    )


def load_stooq_daily(symbol: str) -> pd.DataFrame:
    """Download free daily prices from Stooq without an API key."""

    normalized = symbol.strip().lower()
    if not normalized:
        raise ValueError("symbol cannot be empty")
    url = f"https://stooq.com/q/d/l/?s={normalized}&i=d"
    with urlopen(url, timeout=30) as response:
        payload = response.read().decode("utf8")
    frame = pd.read_csv(StringIO(payload), parse_dates=["Date"], index_col="Date")
    return validate_price_frame(frame)


def load_yahoo_daily(
    symbols: str | list[str],
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    """Download adjusted daily prices from Yahoo Finance."""

    import yfinance as yf

    requested = [symbols] if isinstance(symbols, str) else list(symbols)
    if not requested or any(not symbol.strip() for symbol in requested):
        raise ValueError("At least one nonempty symbol is required")
    raw = yf.download(
        requested,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw.empty:
        raise ValueError("Yahoo Finance returned no observations")

    output: dict[str, pd.DataFrame] = {}
    for symbol in requested:
        frame = raw[symbol].copy() if len(requested) > 1 else raw.copy()
        frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
        output[symbol] = validate_price_frame(frame)
    return output
