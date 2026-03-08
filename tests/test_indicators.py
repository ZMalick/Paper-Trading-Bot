"""Tests for technical indicators."""

import pandas as pd
from trading_bot.data.indicators import (
    compute_sma,
    compute_ema,
    compute_rsi,
    compute_macd,
    compute_bollinger,
    compute_all_indicators,
)


def test_sma_length(sample_bars):
    sma = compute_sma(sample_bars, window=20)
    assert isinstance(sma, pd.Series)
    assert len(sma) == len(sample_bars)


def test_sma_values(sample_bars):
    sma = compute_sma(sample_bars, window=5)
    # After warmup, SMA should equal the mean of the last 5 closes
    expected = sample_bars["close"].iloc[4:9].mean()
    assert abs(sma.iloc[8] - expected) < 0.01


def test_ema_length(sample_bars):
    ema = compute_ema(sample_bars, window=20)
    assert isinstance(ema, pd.Series)
    assert len(ema) == len(sample_bars)


def test_rsi_range(sample_bars):
    rsi = compute_rsi(sample_bars)
    valid = rsi.dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


def test_macd_columns(sample_bars):
    macd_df = compute_macd(sample_bars)
    assert isinstance(macd_df, pd.DataFrame)
    assert set(macd_df.columns) == {"macd", "macd_signal", "macd_diff"}
    assert len(macd_df) == len(sample_bars)


def test_bollinger_columns(sample_bars):
    bb = compute_bollinger(sample_bars)
    assert isinstance(bb, pd.DataFrame)
    assert set(bb.columns) == {"bb_upper", "bb_middle", "bb_lower"}


def test_bollinger_order(sample_bars):
    bb = compute_bollinger(sample_bars)
    valid = bb.dropna()
    assert (valid["bb_upper"] >= valid["bb_middle"]).all()
    assert (valid["bb_middle"] >= valid["bb_lower"]).all()


def test_compute_all_indicators(sample_bars):
    result = compute_all_indicators(sample_bars)
    expected_cols = {
        "sma_20", "ema_20", "rsi_14",
        "macd", "macd_signal", "macd_diff",
        "bb_upper", "bb_middle", "bb_lower",
    }
    assert expected_cols.issubset(set(result.columns))
    assert len(result) == len(sample_bars)
