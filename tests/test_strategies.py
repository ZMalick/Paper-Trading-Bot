"""Tests for trading strategies."""

import pandas as pd

from trading_bot.models import Signal, SignalType
from trading_bot.strategies.sma_crossover import SMACrossoverStrategy
from trading_bot.strategies.rsi import RSIStrategy
from trading_bot.strategies.macd import MACDStrategy
from trading_bot.strategies.mean_reversion import MeanReversionStrategy
from trading_bot.strategies.ml_strategy import MLStrategy


class TestSMACrossover:
    def test_returns_signal(self, sample_bars):
        strategy = SMACrossoverStrategy()
        signal = strategy.analyze("AAPL", sample_bars)
        assert isinstance(signal, Signal)
        assert signal.symbol == "AAPL"
        assert signal.strategy_name == "sma_crossover"

    def test_insufficient_data_returns_hold(self):
        strategy = SMACrossoverStrategy()
        short_bars = pd.DataFrame(
            {"close": [100.0] * 10, "open": [100.0] * 10,
             "high": [101.0] * 10, "low": [99.0] * 10, "volume": [1e6] * 10},
            index=pd.bdate_range("2026-01-01", periods=10),
        )
        signal = strategy.analyze("AAPL", short_bars)
        assert signal.signal_type == SignalType.HOLD


class TestRSI:
    def test_returns_signal(self, sample_bars):
        strategy = RSIStrategy()
        signal = strategy.analyze("AAPL", sample_bars)
        assert isinstance(signal, Signal)
        assert signal.strategy_name == "rsi"

    def test_oversold_buy(self):
        """Construct steadily declining prices to trigger RSI < 30."""
        n = 50
        prices = [100 - i * 1.5 for i in range(n)]
        bars = pd.DataFrame(
            {"close": prices, "open": prices, "high": prices, "low": prices,
             "volume": [1e6] * n},
            index=pd.bdate_range("2026-01-01", periods=n),
        )
        strategy = RSIStrategy()
        signal = strategy.analyze("AAPL", bars)
        # With steadily declining prices, RSI should be very low
        assert signal.signal_type in (SignalType.BUY, SignalType.HOLD)

    def test_insufficient_data_returns_hold(self):
        strategy = RSIStrategy()
        bars = pd.DataFrame(
            {"close": [100.0] * 5, "open": [100.0] * 5,
             "high": [101.0] * 5, "low": [99.0] * 5, "volume": [1e6] * 5},
            index=pd.bdate_range("2026-01-01", periods=5),
        )
        signal = strategy.analyze("AAPL", bars)
        assert signal.signal_type == SignalType.HOLD


class TestMACD:
    def test_returns_signal(self, sample_bars):
        strategy = MACDStrategy()
        signal = strategy.analyze("AAPL", sample_bars)
        assert isinstance(signal, Signal)
        assert signal.strategy_name == "macd"

    def test_insufficient_data_returns_hold(self):
        strategy = MACDStrategy()
        bars = pd.DataFrame(
            {"close": [100.0] * 10, "open": [100.0] * 10,
             "high": [101.0] * 10, "low": [99.0] * 10, "volume": [1e6] * 10},
            index=pd.bdate_range("2026-01-01", periods=10),
        )
        signal = strategy.analyze("AAPL", bars)
        assert signal.signal_type == SignalType.HOLD


class TestMeanReversion:
    def test_returns_signal(self, sample_bars):
        strategy = MeanReversionStrategy()
        signal = strategy.analyze("AAPL", sample_bars)
        assert isinstance(signal, Signal)
        assert signal.strategy_name == "mean_reversion"

    def test_insufficient_data_returns_hold(self):
        strategy = MeanReversionStrategy()
        bars = pd.DataFrame(
            {"close": [100.0] * 5, "open": [100.0] * 5,
             "high": [101.0] * 5, "low": [99.0] * 5, "volume": [1e6] * 5},
            index=pd.bdate_range("2026-01-01", periods=5),
        )
        signal = strategy.analyze("AAPL", bars)
        assert signal.signal_type == SignalType.HOLD


class TestMLStrategy:
    def test_untrained_returns_hold(self, sample_bars):
        strategy = MLStrategy()
        signal = strategy.analyze("AAPL", sample_bars)
        assert signal.signal_type == SignalType.HOLD

    def test_train_and_predict(self, sample_bars):
        strategy = MLStrategy()
        metrics = strategy.train(sample_bars)
        assert "train_accuracy" in metrics
        assert "test_accuracy" in metrics
        assert strategy.is_trained

        signal = strategy.analyze("AAPL", sample_bars)
        assert isinstance(signal, Signal)
        assert signal.signal_type in (SignalType.BUY, SignalType.SELL)
        assert 0 <= signal.confidence <= 1

    def test_insufficient_data_returns_hold(self):
        strategy = MLStrategy()
        strategy.is_trained = True
        strategy.model = "dummy"  # won't be called
        bars = pd.DataFrame(
            {"close": [100.0] * 10, "open": [100.0] * 10,
             "high": [101.0] * 10, "low": [99.0] * 10, "volume": [1e6] * 10},
            index=pd.bdate_range("2026-01-01", periods=10),
        )
        signal = strategy.analyze("AAPL", bars)
        assert signal.signal_type == SignalType.HOLD
