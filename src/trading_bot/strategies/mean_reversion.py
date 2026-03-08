"""Bollinger Band Mean Reversion strategy."""

import pandas as pd
from loguru import logger

from trading_bot.data.indicators import compute_bollinger
from trading_bot.models import Signal, SignalType
from trading_bot.strategies.base import Strategy


class MeanReversionStrategy(Strategy):
    """Buy below lower Bollinger Band, sell above upper Bollinger Band."""

    name: str = "mean_reversion"

    def __init__(self, window: int = 20, std_dev: int = 2) -> None:
        self.window = window
        self.std_dev = std_dev

    def analyze(self, symbol: str, bars: pd.DataFrame) -> Signal:
        """Analyze bars for Bollinger Band breakout signals."""
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

        bb_df = compute_bollinger(bars, window=self.window, std_dev=self.std_dev)

        price = bars["close"].iloc[-1]
        upper = bb_df["bb_upper"].iloc[-1]
        middle = bb_df["bb_middle"].iloc[-1]
        lower = bb_df["bb_lower"].iloc[-1]

        if pd.isna(upper) or pd.isna(lower):
            logger.warning(
                "{}: Bollinger values are NaN for {}, returning HOLD",
                self.name, symbol,
            )
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                strategy_name=self.name,
                confidence=0.0,
                metadata={"reason": "bollinger_nan"},
            )

        band_width = upper - lower

        if price < lower:
            signal_type = SignalType.BUY
            distance = lower - price
            confidence = min(distance / band_width, 1.0) if band_width > 0 else 0.0
            logger.info(
                "{}: price {:.2f} below lower band {:.2f} for {}",
                self.name, price, lower, symbol,
            )
        elif price > upper:
            signal_type = SignalType.SELL
            distance = price - upper
            confidence = min(distance / band_width, 1.0) if band_width > 0 else 0.0
            logger.info(
                "{}: price {:.2f} above upper band {:.2f} for {}",
                self.name, price, upper, symbol,
            )
        else:
            signal_type = SignalType.HOLD
            confidence = 0.0
            logger.debug(
                "{}: price {:.2f} within bands [{:.2f}, {:.2f}] for {}",
                self.name, price, lower, upper, symbol,
            )

        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            strategy_name=self.name,
            confidence=confidence,
            metadata={
                "price": price,
                "bb_upper": upper,
                "bb_middle": middle,
                "bb_lower": lower,
                "band_width": band_width,
            },
        )
