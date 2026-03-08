"""Technical indicators using the ta library."""

import pandas as pd
import ta.momentum
import ta.trend
import ta.volatility
from loguru import logger


def compute_sma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Compute Simple Moving Average.

    Args:
        df: DataFrame with a 'close' column.
        window: Lookback period.

    Returns:
        Series with the SMA values.
    """
    logger.debug("Computing SMA with window={}", window)
    return ta.trend.sma_indicator(df["close"], window=window)


def compute_ema(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Compute Exponential Moving Average.

    Args:
        df: DataFrame with a 'close' column.
        window: Lookback period.

    Returns:
        Series with the EMA values.
    """
    logger.debug("Computing EMA with window={}", window)
    return ta.trend.ema_indicator(df["close"], window=window)


def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Compute Relative Strength Index.

    Args:
        df: DataFrame with a 'close' column.
        window: Lookback period.

    Returns:
        Series with RSI values (0-100).
    """
    logger.debug("Computing RSI with window={}", window)
    return ta.momentum.rsi(df["close"], window=window)


def compute_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Compute MACD indicator.

    Args:
        df: DataFrame with a 'close' column.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal line period.

    Returns:
        DataFrame with columns: macd, macd_signal, macd_diff.
    """
    logger.debug(
        "Computing MACD with fast={}, slow={}, signal={}", fast, slow, signal
    )
    close = df["close"]
    return pd.DataFrame(
        {
            "macd": ta.trend.macd(close, window_slow=slow, window_fast=fast),
            "macd_signal": ta.trend.macd_signal(
                close, window_slow=slow, window_fast=fast, window_sign=signal
            ),
            "macd_diff": ta.trend.macd_diff(
                close, window_slow=slow, window_fast=fast, window_sign=signal
            ),
        },
        index=df.index,
    )


def compute_bollinger(
    df: pd.DataFrame, window: int = 20, std_dev: int = 2
) -> pd.DataFrame:
    """Compute Bollinger Bands.

    Args:
        df: DataFrame with a 'close' column.
        window: Lookback period.
        std_dev: Number of standard deviations.

    Returns:
        DataFrame with columns: bb_upper, bb_middle, bb_lower.
    """
    logger.debug(
        "Computing Bollinger Bands with window={}, std_dev={}",
        window,
        std_dev,
    )
    close = df["close"]
    return pd.DataFrame(
        {
            "bb_upper": ta.volatility.bollinger_hband(
                close, window=window, window_dev=std_dev
            ),
            "bb_middle": ta.volatility.bollinger_mavg(close, window=window),
            "bb_lower": ta.volatility.bollinger_lband(
                close, window=window, window_dev=std_dev
            ),
        },
        index=df.index,
    )


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators as new columns to the DataFrame.

    Adds: sma_20, ema_20, rsi_14, macd, macd_signal, macd_diff,
    bb_upper, bb_middle, bb_lower.

    Args:
        df: DataFrame with a 'close' column.

    Returns:
        Copy of the input DataFrame with indicator columns appended.
    """
    logger.info("Computing all indicators for DataFrame with {} rows", len(df))
    result = df.copy()

    result["sma_20"] = compute_sma(df)
    result["ema_20"] = compute_ema(df)
    result["rsi_14"] = compute_rsi(df)

    macd_df = compute_macd(df)
    result["macd"] = macd_df["macd"]
    result["macd_signal"] = macd_df["macd_signal"]
    result["macd_diff"] = macd_df["macd_diff"]

    bb_df = compute_bollinger(df)
    result["bb_upper"] = bb_df["bb_upper"]
    result["bb_middle"] = bb_df["bb_middle"]
    result["bb_lower"] = bb_df["bb_lower"]

    logger.info("All indicators computed successfully")
    return result
