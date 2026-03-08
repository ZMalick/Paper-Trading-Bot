from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from trading_bot.models import Signal, SignalType, TradeRecord
from trading_bot.strategies.base import Strategy


@dataclass
class BacktestResult:
    """Contains the full results of a backtest run."""

    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0


class BacktestEngine:
    """Simulates strategy execution over historical bar data."""

    WARMUP_PERIOD = 50
    POSITION_SIZE_PCT = 0.10

    def __init__(self, strategy: Strategy, initial_capital: float = 100000.0):
        self.strategy = strategy
        self.initial_capital = initial_capital

    def run(self, bars: pd.DataFrame, symbol: str) -> BacktestResult:
        """Run a backtest over the provided bars DataFrame.

        Args:
            bars: DataFrame with OHLCV data indexed by date/datetime.
                  Must contain at least a 'close' column.
            symbol: The ticker symbol being tested.

        Returns:
            A BacktestResult with trades, equity curve, and performance metrics.
        """
        if len(bars) <= self.WARMUP_PERIOD:
            logger.warning(
                f"Not enough bars ({len(bars)}) for warmup period "
                f"({self.WARMUP_PERIOD}). Returning empty result."
            )
            return BacktestResult()

        cash = self.initial_capital
        shares = 0
        trades: list[TradeRecord] = []
        equity_records: list[dict] = []
        peak_equity = self.initial_capital

        logger.info(
            f"Starting backtest: {self.strategy.name} on {symbol}, "
            f"{len(bars)} bars, capital=${self.initial_capital:,.2f}"
        )

        for i in range(self.WARMUP_PERIOD, len(bars)):
            # Slice historical bars up to and including the current day
            historical_bars = bars.iloc[: i + 1]
            current_price = float(bars.iloc[i]["close"])
            current_date = bars.index[i]

            # Get signal from strategy
            try:
                signal = self.strategy.analyze(symbol, historical_bars)
            except Exception as e:
                logger.debug(f"Strategy error on bar {i}: {e}")
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.HOLD,
                    strategy_name=self.strategy.name,
                    confidence=0.0,
                )

            # Execute signal
            if signal.signal_type == SignalType.BUY and shares == 0:
                portfolio_value = cash + shares * current_price
                allocation = portfolio_value * self.POSITION_SIZE_PCT
                qty = int(allocation // current_price)

                if qty > 0 and cash >= qty * current_price:
                    cost = qty * current_price
                    cash -= cost
                    shares += qty
                    trades.append(
                        TradeRecord(
                            symbol=symbol,
                            side="BUY",
                            qty=float(qty),
                            price=current_price,
                            timestamp=_to_datetime(current_date),
                            strategy_name=self.strategy.name,
                            signal_confidence=signal.confidence,
                            order_id=f"bt-{len(trades) + 1}",
                        )
                    )
                    logger.debug(
                        f"BUY {qty} {symbol} @ ${current_price:.2f} "
                        f"on {current_date}"
                    )

            elif signal.signal_type == SignalType.SELL and shares > 0:
                revenue = shares * current_price
                cash += revenue
                trades.append(
                    TradeRecord(
                        symbol=symbol,
                        side="SELL",
                        qty=float(shares),
                        price=current_price,
                        timestamp=_to_datetime(current_date),
                        strategy_name=self.strategy.name,
                        signal_confidence=signal.confidence,
                        order_id=f"bt-{len(trades) + 1}",
                    )
                )
                logger.debug(
                    f"SELL {shares} {symbol} @ ${current_price:.2f} "
                    f"on {current_date}"
                )
                shares = 0

            # Record daily equity
            equity = cash + shares * current_price
            peak_equity = max(peak_equity, equity)
            equity_records.append(
                {"timestamp": _to_datetime(current_date), "equity": equity}
            )

        # Final metrics
        equity_curve = pd.DataFrame(equity_records)
        final_equity = equity_records[-1]["equity"] if equity_records else self.initial_capital

        total_return_pct = (
            (final_equity - self.initial_capital) / self.initial_capital * 100.0
        )
        sharpe_ratio = _compute_sharpe(equity_curve)
        max_drawdown = _compute_max_drawdown(equity_curve)
        win_rate = _compute_win_rate(trades)

        result = BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            total_return_pct=round(total_return_pct, 4),
            sharpe_ratio=round(sharpe_ratio, 4),
            max_drawdown=round(max_drawdown, 4),
            win_rate=round(win_rate, 4),
            total_trades=len(trades),
        )

        logger.info(
            f"Backtest complete: {self.strategy.name} on {symbol} | "
            f"Return={result.total_return_pct:.2f}% | "
            f"Sharpe={result.sharpe_ratio:.2f} | "
            f"MaxDD={result.max_drawdown:.2f}% | "
            f"WinRate={result.win_rate:.1f}% | "
            f"Trades={result.total_trades}"
        )
        return result

    @staticmethod
    def compare_strategies(
        strategies: list[Strategy], bars: pd.DataFrame, symbol: str
    ) -> dict[str, BacktestResult]:
        """Run backtests for multiple strategies and return results keyed by name.

        Args:
            strategies: List of Strategy instances to compare.
            bars: Historical OHLCV DataFrame.
            symbol: Ticker symbol.

        Returns:
            Dict mapping strategy name to its BacktestResult.
        """
        results: dict[str, BacktestResult] = {}
        for strategy in strategies:
            logger.info(f"Running backtest for strategy: {strategy.name}")
            engine = BacktestEngine(strategy)
            results[strategy.name] = engine.run(bars, symbol)
        return results


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _to_datetime(value) -> datetime:
    """Convert various date types to a datetime object."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    try:
        return pd.Timestamp(value).to_pydatetime()
    except Exception:
        return datetime.now()


def _compute_sharpe(equity_curve: pd.DataFrame) -> float:
    """Compute annualized Sharpe ratio from an equity curve DataFrame."""
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0

    equities = equity_curve["equity"].values
    daily_returns = np.diff(equities) / equities[:-1] * 100.0

    mean_ret = np.mean(daily_returns)
    std_ret = np.std(daily_returns, ddof=1)

    if std_ret == 0:
        return 0.0

    return float((mean_ret / std_ret) * np.sqrt(252))


def _compute_max_drawdown(equity_curve: pd.DataFrame) -> float:
    """Compute maximum drawdown percentage from an equity curve DataFrame."""
    if equity_curve.empty:
        return 0.0

    equities = equity_curve["equity"].values
    peak = equities[0]
    max_dd = 0.0

    for eq in equities:
        if eq > peak:
            peak = eq
        drawdown = (peak - eq) / peak * 100.0
        if drawdown > max_dd:
            max_dd = drawdown

    return max_dd


def _compute_win_rate(trades: list[TradeRecord]) -> float:
    """Compute win rate from paired BUY/SELL trades."""
    if not trades:
        return 0.0

    buys: dict[str, list[TradeRecord]] = {}
    wins = 0
    total_round_trips = 0

    for trade in trades:
        if trade.side == "BUY":
            buys.setdefault(trade.symbol, []).append(trade)
        elif trade.side == "SELL" and trade.symbol in buys and buys[trade.symbol]:
            buy_trade = buys[trade.symbol].pop(0)
            total_round_trips += 1
            if trade.price > buy_trade.price:
                wins += 1

    if total_round_trips == 0:
        return 0.0

    return wins / total_round_trips * 100.0
