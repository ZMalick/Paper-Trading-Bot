"""Main trading bot orchestrator."""

import sys
import time

from loguru import logger

from trading_bot.config import settings
from trading_bot.clients import get_trading_client, get_data_client
from trading_bot.data.market_data import MarketDataFetcher
from trading_bot.strategies import get_strategy
from trading_bot.execution.order_manager import OrderManager
from trading_bot.execution.portfolio import PortfolioTracker
from trading_bot.models import SignalType


class TradingBot:
    """Core orchestrator that ties together data fetching, strategy analysis,
    order execution, and portfolio tracking in a continuous loop."""

    def __init__(self) -> None:
        # Configure loguru -----------------------------------------------
        logger.remove()
        logger.add(sys.stderr, level=settings.log_level)
        logger.add(
            "logs/trading_bot_{time}.log",
            rotation="1 day",
            retention="30 days",
            level="DEBUG",
        )

        logger.info("Initializing Trading Bot...")

        # Initialize components ------------------------------------------
        self.trading_client = get_trading_client()
        self.data_client = get_data_client()
        self.data_fetcher = MarketDataFetcher(self.data_client)
        self.strategy = get_strategy(settings.strategy)
        self.order_manager = OrderManager(
            self.trading_client, settings.max_position_pct
        )
        self.portfolio = PortfolioTracker()

        logger.info(f"Strategy: {self.strategy.name}")
        logger.info(f"Symbols: {settings.symbols_list}")

    # ------------------------------------------------------------------ #
    #  Market status                                                      #
    # ------------------------------------------------------------------ #

    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False

    # ------------------------------------------------------------------ #
    #  Single iteration                                                   #
    # ------------------------------------------------------------------ #

    def run_once(self) -> None:
        """Run one iteration of the trading loop."""
        account = self.order_manager.get_account()
        logger.info(
            f"Account equity: ${account['equity']:,.2f} | "
            f"Cash: ${account['cash']:,.2f}"
        )

        for symbol in settings.symbols_list:
            try:
                # Fetch recent bars
                bars = self.data_fetcher.get_latest_bars(symbol, limit=100)
                if bars.empty:
                    logger.warning(f"No data for {symbol}, skipping")
                    continue

                # Generate signal
                signal = self.strategy.analyze(symbol, bars)
                logger.info(
                    f"{symbol}: {signal.signal_type.value} "
                    f"(confidence: {signal.confidence:.2f})"
                )

                # Execute if not HOLD
                if signal.signal_type != SignalType.HOLD:
                    current_price = float(bars["close"].iloc[-1])
                    trade = self.order_manager.execute_signal(signal, current_price)
                    if trade:
                        self.portfolio.record_trade(trade)
                        logger.info(
                            f"Trade executed: {trade.side} {trade.qty} "
                            f"{trade.symbol} @ ${trade.price:.2f}"
                        )

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

        # Take portfolio snapshot ----------------------------------------
        try:
            account = self.order_manager.get_account()
            positions = self.order_manager.get_positions()
            positions_value = (
                sum(float(p.market_value) for p in positions)
                if positions
                else 0.0
            )
            self.portfolio.take_snapshot(
                equity=account["equity"],
                cash=account["cash"],
                positions_value=positions_value,
            )
        except Exception as e:
            logger.error(f"Error taking snapshot: {e}")

    # ------------------------------------------------------------------ #
    #  Main loop                                                          #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Main trading loop."""
        logger.info("Starting trading bot main loop...")

        while True:
            try:
                if self.is_market_open():
                    logger.info("Market is open - running trading cycle")
                    self.run_once()
                else:
                    logger.info("Market is closed - waiting...")

                logger.debug(
                    f"Sleeping for {settings.trading_interval} seconds"
                )
                time.sleep(settings.trading_interval)

            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(settings.trading_interval)


def main() -> None:
    """Entry point for the trading bot."""
    bot = TradingBot()
    bot.run()


if __name__ == "__main__":
    main()
