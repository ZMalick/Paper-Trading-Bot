"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from trading_bot.models import Signal, SignalType


@pytest.fixture
def sample_bars() -> pd.DataFrame:
    """Generate 100 days of realistic OHLCV data using a random walk."""
    np.random.seed(42)
    n = 100
    dates = pd.bdate_range(end=datetime(2026, 3, 1), periods=n)

    # Cumulative random walk starting at 150
    returns = np.random.normal(0.0005, 0.015, n)
    close = 150.0 * np.cumprod(1 + returns)

    high = close * (1 + np.abs(np.random.normal(0, 0.008, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.008, n)))
    open_ = close * (1 + np.random.normal(0, 0.005, n))
    volume = np.random.randint(1_000_000, 5_000_000, n)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def sample_signal() -> Signal:
    """A sample BUY signal for testing."""
    return Signal(
        symbol="AAPL",
        signal_type=SignalType.BUY,
        strategy_name="test",
        confidence=0.8,
    )
