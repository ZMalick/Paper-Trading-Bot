"""Tests for backtest engine."""

import pandas as pd

from trading_bot.backtest.engine import BacktestEngine, BacktestResult
from trading_bot.strategies.sma_crossover import SMACrossoverStrategy
from trading_bot.strategies.rsi import RSIStrategy


def test_backtest_produces_result(sample_bars):
    strategy = SMACrossoverStrategy()
    engine = BacktestEngine(strategy, initial_capital=100000.0)
    result = engine.run(sample_bars, "AAPL")
    assert isinstance(result, BacktestResult)
    assert not result.equity_curve.empty


def test_equity_curve_length(sample_bars):
    strategy = SMACrossoverStrategy()
    engine = BacktestEngine(strategy)
    result = engine.run(sample_bars, "AAPL")
    # Equity curve should have one entry per bar after warmup
    expected = len(sample_bars) - BacktestEngine.WARMUP_PERIOD
    assert len(result.equity_curve) == expected


def test_insufficient_bars():
    strategy = SMACrossoverStrategy()
    engine = BacktestEngine(strategy)
    short_bars = pd.DataFrame(
        {"close": [100.0] * 20, "open": [100.0] * 20,
         "high": [101.0] * 20, "low": [99.0] * 20, "volume": [1e6] * 20},
        index=pd.bdate_range("2026-01-01", periods=20),
    )
    result = engine.run(short_bars, "AAPL")
    assert result.total_trades == 0
    assert result.equity_curve.empty


def test_compare_strategies(sample_bars):
    strategies = [SMACrossoverStrategy(), RSIStrategy()]
    results = BacktestEngine.compare_strategies(strategies, sample_bars, "AAPL")
    assert "sma_crossover" in results
    assert "rsi" in results
    assert isinstance(results["sma_crossover"], BacktestResult)
    assert isinstance(results["rsi"], BacktestResult)


def test_metrics_reasonable(sample_bars):
    strategy = SMACrossoverStrategy()
    engine = BacktestEngine(strategy)
    result = engine.run(sample_bars, "AAPL")
    # Return should be within reasonable bounds for 100 days
    assert -50 < result.total_return_pct < 100
    assert 0 <= result.max_drawdown <= 100
    assert 0 <= result.win_rate <= 100
