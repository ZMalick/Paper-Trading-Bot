"""RSI Oversold/Overbought strategy."""

import pandas as pd
from loguru import logger

from trading_bot.data.indicators import compute_rsi
from trading_bot.models import Signal, SignalType
from trading_bot.strategies.base import Strategy


class RSIStrategy(Strategy):
    """Buy when RSI < 45 (oversold), sell when RSI > 55 (overbought)."""

    name: str = "rsi"

    def __init__(
        self,
        window: int = 14,
        oversold: float = 45.0,
        overbought: float = 55.0,
    ) -> None:
        self.window = window
        self.oversold = oversold
        self.overbought = overbought

    def analyze(self, symbol: str, bars: pd.DataFrame) -> Signal:
        """Analyze bars for RSI oversold/overbought signals."""
        min_bars = self.window + 2
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

        rsi_series = compute_rsi(bars, window=self.window)
        rsi = rsi_series.iloc[-1]

        if pd.isna(rsi):
            logger.warning("{}: RSI is NaN for {}, returning HOLD", self.name, symbol)
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                strategy_name=self.name,
                confidence=0.0,
                metadata={"reason": "rsi_nan"},
            )

        confidence = abs(rsi - 50.0) / 50.0

        if rsi < self.oversold:
            signal_type = SignalType.BUY
            logger.info("{}: RSI={:.1f} oversold for {}", self.name, rsi, symbol)
        elif rsi > self.overbought:
            signal_type = SignalType.SELL
            logger.info("{}: RSI={:.1f} overbought for {}", self.name, rsi, symbol)
        else:
            signal_type = SignalType.HOLD
            logger.debug("{}: RSI={:.1f} neutral for {}", self.name, rsi, symbol)

        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            strategy_name=self.name,
            confidence=confidence,
            metadata={"rsi": rsi},
        )
