from datetime import datetime
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from loguru import logger

from trading_bot.models import Signal, SignalType, TradeRecord


class OrderManager:
    """Manages order placement, position sizing, and trade execution for paper trading."""

    def __init__(self, trading_client: TradingClient, max_position_pct: float = 0.1):
        self.trading_client = trading_client
        self.max_position_pct = max_position_pct

    def get_account(self) -> dict:
        """Return account info with equity, cash, and buying_power."""
        try:
            account = self.trading_client.get_account()
            return {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
            }
        except Exception as e:
            logger.error(f"Failed to fetch account info: {e}")
            raise

    def get_positions(self) -> list:
        """Return list of current open positions."""
        try:
            positions = self.trading_client.get_all_positions()
            return positions
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            raise

    def calculate_position_size(self, symbol: str, signal: Signal) -> int:
        """Calculate the number of shares to buy based on max_position_pct of equity.

        Returns 0 if already at max position or if the signal is not a BUY.
        """
        if signal.signal_type != SignalType.BUY:
            return 0

        try:
            account = self.get_account()
            equity = account["equity"]
            max_position_value = equity * self.max_position_pct

            # Check if we already hold a position in this symbol
            positions = self.get_positions()
            for pos in positions:
                if pos.symbol == symbol:
                    current_value = abs(float(pos.market_value))
                    if current_value >= max_position_value:
                        logger.info(
                            f"Already at max position for {symbol}: "
                            f"${current_value:.2f} >= ${max_position_value:.2f}"
                        )
                        return 0
                    # Reduce allocation by existing position value
                    max_position_value -= current_value

            # Get the latest price from the existing position or latest trade
            latest_trade = self.trading_client.get_latest_trade(symbol)
            price = float(latest_trade.price)

            if price <= 0:
                logger.warning(f"Invalid price for {symbol}: {price}")
                return 0

            shares = int(max_position_value // price)
            logger.info(
                f"Position size for {symbol}: {shares} shares "
                f"(price=${price:.2f}, allocation=${max_position_value:.2f})"
            )
            return shares

        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0

    def execute_signal(self, signal: Signal) -> Optional[TradeRecord]:
        """Execute a trading signal and return a TradeRecord, or None for HOLD.

        - BUY: Place a market buy order with calculated position size.
        - SELL: Close existing position if one exists.
        - HOLD: Do nothing.
        """
        symbol = signal.symbol

        if signal.signal_type == SignalType.HOLD:
            logger.debug(f"HOLD signal for {symbol}, no action taken")
            return None

        if signal.signal_type == SignalType.BUY:
            qty = self.calculate_position_size(symbol, signal)
            if qty <= 0:
                logger.info(f"No shares to buy for {symbol} (qty=0)")
                return None

            try:
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
                order = self.trading_client.submit_order(order_request)
                logger.info(
                    f"BUY order submitted: {qty} shares of {symbol} "
                    f"(order_id={order.id})"
                )
                return TradeRecord(
                    symbol=symbol,
                    side="BUY",
                    qty=float(qty),
                    price=float(order.filled_avg_price or 0.0),
                    timestamp=datetime.now(),
                    strategy_name=signal.strategy_name,
                    signal_confidence=signal.confidence,
                    order_id=str(order.id),
                )
            except Exception as e:
                logger.error(f"Failed to execute BUY for {symbol}: {e}")
                return None

        if signal.signal_type == SignalType.SELL:
            try:
                positions = self.get_positions()
                position = None
                for pos in positions:
                    if pos.symbol == symbol:
                        position = pos
                        break

                if position is None:
                    logger.info(f"No position to sell for {symbol}")
                    return None

                qty = abs(float(position.qty))
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
                order = self.trading_client.submit_order(order_request)
                logger.info(
                    f"SELL order submitted: {qty} shares of {symbol} "
                    f"(order_id={order.id})"
                )
                return TradeRecord(
                    symbol=symbol,
                    side="SELL",
                    qty=qty,
                    price=float(order.filled_avg_price or 0.0),
                    timestamp=datetime.now(),
                    strategy_name=signal.strategy_name,
                    signal_confidence=signal.confidence,
                    order_id=str(order.id),
                )
            except Exception as e:
                logger.error(f"Failed to execute SELL for {symbol}: {e}")
                return None

        return None

    def close_all_positions(self) -> list[TradeRecord]:
        """Close all open positions and return a list of TradeRecords."""
        trade_records: list[TradeRecord] = []
        try:
            positions = self.get_positions()
            if not positions:
                logger.info("No open positions to close")
                return trade_records

            for position in positions:
                symbol = position.symbol
                qty = abs(float(position.qty))
                side = OrderSide.SELL if float(position.qty) > 0 else OrderSide.BUY

                try:
                    order_request = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=side,
                        time_in_force=TimeInForce.DAY,
                    )
                    order = self.trading_client.submit_order(order_request)
                    logger.info(f"Closed position: {qty} shares of {symbol}")
                    trade_records.append(
                        TradeRecord(
                            symbol=symbol,
                            side=side.value,
                            qty=qty,
                            price=float(order.filled_avg_price or 0.0),
                            timestamp=datetime.now(),
                            strategy_name="close_all",
                            signal_confidence=1.0,
                            order_id=str(order.id),
                        )
                    )
                except Exception as e:
                    logger.error(f"Failed to close position for {symbol}: {e}")

        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")

        return trade_records
