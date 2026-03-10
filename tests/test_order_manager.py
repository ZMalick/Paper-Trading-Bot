"""Tests for order manager (mocked Alpaca API)."""

from unittest.mock import MagicMock
import pytest

from trading_bot.models import Signal, SignalType
from trading_bot.execution.order_manager import OrderManager


@pytest.fixture
def mock_trading_client():
    client = MagicMock()
    # Mock account
    account = MagicMock()
    account.equity = "100000.0"
    account.cash = "95000.0"
    account.buying_power = "190000.0"
    client.get_account.return_value = account
    # Mock positions (empty)
    client.get_all_positions.return_value = []
    # Mock latest trade
    latest_trade = MagicMock()
    latest_trade.price = 150.0
    client.get_latest_trade.return_value = latest_trade
    # Mock order submission
    order = MagicMock()
    order.id = "test-order-123"
    order.filled_avg_price = "150.0"
    client.submit_order.return_value = order
    return client


@pytest.fixture
def order_manager(mock_trading_client):
    return OrderManager(mock_trading_client, max_position_pct=0.1)


def test_get_account(order_manager):
    account = order_manager.get_account()
    assert "equity" in account
    assert "cash" in account
    assert "buying_power" in account
    assert account["equity"] == 100000.0


def test_get_positions(order_manager):
    positions = order_manager.get_positions()
    assert isinstance(positions, list)


def test_calculate_position_size_buy(order_manager):
    signal = Signal(
        symbol="AAPL", signal_type=SignalType.BUY,
        strategy_name="test", confidence=0.8,
    )
    qty = order_manager.calculate_position_size("AAPL", signal, current_price=150.0)
    # 10% of $100k = $10k, at $150/share = 66 shares
    assert qty == 66


def test_calculate_position_size_non_buy(order_manager):
    signal = Signal(
        symbol="AAPL", signal_type=SignalType.HOLD,
        strategy_name="test", confidence=0.5,
    )
    qty = order_manager.calculate_position_size("AAPL", signal)
    assert qty == 0


def test_execute_signal_buy(order_manager):
    signal = Signal(
        symbol="AAPL", signal_type=SignalType.BUY,
        strategy_name="test", confidence=0.8,
    )
    trade = order_manager.execute_signal(signal, current_price=150.0)
    assert trade is not None
    assert trade.side == "BUY"
    assert trade.symbol == "AAPL"
    assert trade.order_id == "test-order-123"


def test_execute_signal_sell_no_position(order_manager):
    signal = Signal(
        symbol="AAPL", signal_type=SignalType.SELL,
        strategy_name="test", confidence=0.8,
    )
    trade = order_manager.execute_signal(signal)
    assert trade is None


def test_execute_signal_sell_with_position(mock_trading_client):
    position = MagicMock()
    position.symbol = "AAPL"
    position.qty = "10"
    mock_trading_client.get_all_positions.return_value = [position]

    om = OrderManager(mock_trading_client, max_position_pct=0.1)
    signal = Signal(
        symbol="AAPL", signal_type=SignalType.SELL,
        strategy_name="test", confidence=0.8,
    )
    trade = om.execute_signal(signal)
    assert trade is not None
    assert trade.side == "SELL"
    assert trade.qty == 10.0


def test_execute_signal_hold(order_manager):
    signal = Signal(
        symbol="AAPL", signal_type=SignalType.HOLD,
        strategy_name="test", confidence=0.5,
    )
    trade = order_manager.execute_signal(signal)
    assert trade is None
