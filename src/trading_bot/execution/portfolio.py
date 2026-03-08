from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from trading_bot.models import PerformanceSnapshot, TradeRecord


class PortfolioTracker:
    """Tracks P&L, trade history, and computes performance metrics."""

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.trade_history: list[TradeRecord] = []
        self.snapshots: list[PerformanceSnapshot] = []
        self._peak_equity: float = initial_capital

    def record_trade(self, trade: TradeRecord) -> None:
        """Record a completed trade."""
        self.trade_history.append(trade)
        logger.info(
            f"Trade recorded: {trade.side} {trade.qty} {trade.symbol} "
            f"@ ${trade.price:.2f} ({trade.strategy_name})"
        )

    def take_snapshot(
        self, equity: float, cash: float, positions_value: float
    ) -> PerformanceSnapshot:
        """Take a portfolio snapshot and calculate performance metrics.

        Args:
            equity: Total account equity (cash + positions).
            cash: Available cash.
            positions_value: Market value of all open positions.

        Returns:
            A PerformanceSnapshot with computed metrics.
        """
        # Daily return
        if self.snapshots:
            prev_equity = self.snapshots[-1].total_equity
            daily_return_pct = (
                ((equity - prev_equity) / prev_equity * 100.0)
                if prev_equity != 0
                else 0.0
            )
        else:
            daily_return_pct = (
                ((equity - self.initial_capital) / self.initial_capital * 100.0)
                if self.initial_capital != 0
                else 0.0
            )

        # Total return
        total_return_pct = (
            ((equity - self.initial_capital) / self.initial_capital * 100.0)
            if self.initial_capital != 0
            else 0.0
        )

        # Sharpe ratio (annualized, using daily returns)
        sharpe_ratio = self._calculate_sharpe_ratio(daily_return_pct)

        # Max drawdown
        self._peak_equity = max(self._peak_equity, equity)
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - equity) / self._peak_equity * 100.0
        else:
            drawdown = 0.0
        # Track the worst drawdown seen so far
        prev_max_dd = self.snapshots[-1].max_drawdown if self.snapshots else 0.0
        max_drawdown = max(prev_max_dd or 0.0, drawdown)

        # Win rate
        win_rate = self._calculate_win_rate()

        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            total_equity=equity,
            cash=cash,
            positions_value=positions_value,
            daily_return_pct=daily_return_pct,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(self.trade_history),
        )
        self.snapshots.append(snapshot)
        logger.debug(
            f"Snapshot: equity=${equity:.2f}, return={total_return_pct:.2f}%, "
            f"sharpe={sharpe_ratio}, dd={max_drawdown:.2f}%"
        )
        return snapshot

    def _calculate_sharpe_ratio(self, current_daily_return: float) -> Optional[float]:
        """Calculate annualized Sharpe ratio from daily return history."""
        # Gather all daily returns including the current one
        daily_returns = [s.daily_return_pct for s in self.snapshots]
        daily_returns.append(current_daily_return)

        if len(daily_returns) < 2:
            return None

        returns_array = np.array(daily_returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)

        if std_return == 0:
            return None

        # Annualize: multiply by sqrt(252 trading days)
        sharpe = (mean_return / std_return) * np.sqrt(252)
        return round(float(sharpe), 4)

    def _calculate_win_rate(self) -> Optional[float]:
        """Calculate win rate from trade history using paired buy/sell trades."""
        if not self.trade_history:
            return None

        # Pair up BUY and SELL trades by symbol to determine wins
        buys: dict[str, list[TradeRecord]] = {}
        wins = 0
        total_round_trips = 0

        for trade in self.trade_history:
            if trade.side == "BUY":
                buys.setdefault(trade.symbol, []).append(trade)
            elif trade.side == "SELL" and trade.symbol in buys and buys[trade.symbol]:
                buy_trade = buys[trade.symbol].pop(0)
                total_round_trips += 1
                if trade.price > buy_trade.price:
                    wins += 1

        if total_round_trips == 0:
            return None

        return round(wins / total_round_trips * 100.0, 2)

    def get_performance_summary(self) -> dict:
        """Return a summary dict of the latest performance metrics."""
        if not self.snapshots:
            return {
                "total_equity": self.initial_capital,
                "cash": self.initial_capital,
                "positions_value": 0.0,
                "daily_return_pct": 0.0,
                "total_return_pct": 0.0,
                "sharpe_ratio": None,
                "max_drawdown": 0.0,
                "win_rate": None,
                "total_trades": 0,
            }

        latest = self.snapshots[-1]
        return {
            "total_equity": latest.total_equity,
            "cash": latest.cash,
            "positions_value": latest.positions_value,
            "daily_return_pct": latest.daily_return_pct,
            "total_return_pct": latest.total_return_pct,
            "sharpe_ratio": latest.sharpe_ratio,
            "max_drawdown": latest.max_drawdown,
            "win_rate": latest.win_rate,
            "total_trades": latest.total_trades,
        }

    def get_equity_curve(self) -> pd.DataFrame:
        """Return a DataFrame of the equity curve with timestamp and equity columns."""
        if not self.snapshots:
            return pd.DataFrame(columns=["timestamp", "equity"])

        data = [
            {"timestamp": s.timestamp, "equity": s.total_equity}
            for s in self.snapshots
        ]
        return pd.DataFrame(data)
