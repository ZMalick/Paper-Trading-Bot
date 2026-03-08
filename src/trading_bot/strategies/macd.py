"""MACD Signal Crossover strategy."""

import pandas as pd
from loguru import logger

from trading_bot.data.indicators import compute_macd
from trading_bot.models import Signal, SignalType
from trading_bot.strategies.base import Strategy


class MACDStrategy(Strategy):
    """Buy when MACD crosses above signal line, sell when it crosses below."""

    name: str = "macd"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def analyze(self, symbol: str, bars: pd.DataFrame) -> Signal:
        """Analyze bars for MACD crossover signals."""
        min_bars = self.slow + self.signal + 2
        if len(bars) < min_bars:
            logger.warning(
                "{}: not enough bars ({}/{}), returning HOLD",
                self.name, len(bars), min_bars,
            )
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                strategy_name=self.name,
                confidence=0.0,
                metadata={"reason": "insufficient_data"},
            )

        macd_df = compute_macd(bars, fast=self.fast, slow=self.slow, signal=self.signal)

        macd_line = macd_df["macd"]
        signal_line = macd_df["macd_signal"]
        histogram = macd_df["macd_diff"]

        curr_macd = macd_line.iloc[-1]
        curr_signal = signal_line.iloc[-1]
        prev_macd = macd_line.iloc[-2]
        prev_signal = signal_line.iloc[-2]
        curr_hist = histogram.iloc[-1]

        if pd.isna(curr_macd) or pd.isna(curr_signal):
            logger.warning("{}: MACD values are NaN for {}, returning HOLD", self.name, symbol)
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                strategy_name=self.name,
                confidence=0.0,
                metadata={"reason": "macd_nan"},
            )

        # Confidence based on histogram magnitude relative to price
        price = bars["close"].iloc[-1]
        confidence = min(abs(curr_hist) / price * 100, 1.0) if price > 0 else 0.0

        # Bullish crossover: MACD crosses above signal
        if prev_macd <= prev_signal and curr_macd > curr_signal:
            signal_type = SignalType.BUY
            logger.info("{}: bullish MACD crossover for {}", self.name, symbol)
        # Bearish crossover: MACD crosses below signal
        elif prev_macd >= prev_signal and curr_macd < curr_signal:
            signal_type = SignalType.SELL
            logger.info("{}: bearish MACD crossover for {}", self.name, symbol)
        else:
            signal_type = SignalType.HOLD
            logger.debug("{}: no MACD crossover for {}", self.name, symbol)

        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            strategy_name=self.name,
            confidence=confidence,
            metadata={
                "macd": curr_macd,
                "macd_signal": curr_signal,
                "histogram": curr_hist,
            },
        )
