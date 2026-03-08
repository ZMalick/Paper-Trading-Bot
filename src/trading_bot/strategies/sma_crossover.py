"""SMA Golden/Death Cross strategy."""

import pandas as pd
from loguru import logger

from trading_bot.data.indicators import compute_sma
from trading_bot.models import Signal, SignalType
from trading_bot.strategies.base import Strategy


class SMACrossoverStrategy(Strategy):
    """Buy on golden cross (SMA20 > SMA50), sell on death cross."""

    name: str = "sma_crossover"

    def __init__(self, short_window: int = 20, long_window: int = 50) -> None:
        self.short_window = short_window
        self.long_window = long_window

    def analyze(self, symbol: str, bars: pd.DataFrame) -> Signal:
        """Analyze bars for SMA crossover signals."""
        min_bars = self.long_window + 2
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

        sma_short = compute_sma(bars, window=self.short_window)
        sma_long = compute_sma(bars, window=self.long_window)

        current_short = sma_short.iloc[-1]
        current_long = sma_long.iloc[-1]
        prev_short = sma_short.iloc[-2]
        prev_long = sma_long.iloc[-2]

        price = bars["close"].iloc[-1]
        distance = abs(current_short - current_long)
        confidence = min(distance / price, 1.0) if price > 0 else 0.0

        # Golden cross: short crosses above long
        if prev_short <= prev_long and current_short > current_long:
            signal_type = SignalType.BUY
            logger.info("{}: golden cross detected for {}", self.name, symbol)
        # Death cross: short crosses below long
        elif prev_short >= prev_long and current_short < current_long:
            signal_type = SignalType.SELL
            logger.info("{}: death cross detected for {}", self.name, symbol)
        else:
            signal_type = SignalType.HOLD
            logger.debug("{}: no crossover for {}", self.name, symbol)

        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            strategy_name=self.name,
            confidence=confidence,
            metadata={
                "sma_short": current_short,
                "sma_long": current_long,
                "price": price,
            },
        )
