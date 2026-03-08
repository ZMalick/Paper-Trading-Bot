"""Streamlit dashboard for the Paper Trading Bot."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Paper Trading Bot Dashboard", layout="wide")
st.title("Paper Trading Bot Dashboard")

# ---------------------------------------------------------------------------
# Helpers to connect to Alpaca (gracefully handle missing config)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_account():
    """Fetch account info from Alpaca."""
    from trading_bot.clients import get_trading_client
    client = get_trading_client()
    account = client.get_account()
    return {
        "equity": float(account.equity),
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
        "last_equity": float(account.last_equity),
    }


@st.cache_data(ttl=300)
def load_positions():
    """Fetch open positions from Alpaca."""
    from trading_bot.clients import get_trading_client
    client = get_trading_client()
    positions = client.get_all_positions()
    if not positions:
        return pd.DataFrame()
    rows = []
    for p in positions:
        rows.append({
            "Symbol": p.symbol,
            "Qty": float(p.qty),
            "Avg Entry": float(p.avg_entry_price),
            "Current Price": float(p.current_price),
            "Market Value": float(p.market_value),
            "P&L": float(p.unrealized_pl),
            "P&L %": float(p.unrealized_plpc) * 100,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_bars(symbol: str, days: int = 180):
    """Fetch historical bars for a symbol."""
    from trading_bot.clients import get_data_client
    from trading_bot.data.market_data import MarketDataFetcher
    client = get_data_client()
    fetcher = MarketDataFetcher(client)
    return fetcher.get_bars(symbol, limit=days)


def generate_demo_bars(days: int = 180) -> pd.DataFrame:
    """Generate synthetic price data for demo mode."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq="B")
    price = 150.0
    prices = []
    for _ in range(days):
        price *= 1 + np.random.normal(0.0005, 0.015)
        prices.append(price)
    return pd.DataFrame({
        "open": [p * (1 + np.random.uniform(-0.005, 0.005)) for p in prices],
        "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        "close": prices,
        "volume": [int(np.random.uniform(1e6, 5e6)) for _ in prices],
    }, index=dates)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.header("Configuration")

STRATEGIES = ["sma_crossover", "rsi", "macd", "mean_reversion", "ml"]
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

selected_strategies = st.sidebar.multiselect(
    "Strategies", STRATEGIES, default=["sma_crossover", "rsi"]
)
selected_symbols = st.sidebar.multiselect(
    "Symbols", DEFAULT_SYMBOLS, default=["AAPL"]
)
backtest_days = st.sidebar.slider("Backtest Days", 60, 365, 180)
run_backtest = st.sidebar.button("Run Backtest")

# Detect connection mode
demo_mode = False
try:
    account_info = load_account()
except Exception:
    demo_mode = True
    account_info = {
        "equity": 100000.0,
        "cash": 95000.0,
        "buying_power": 190000.0,
        "last_equity": 99500.0,
    }

if demo_mode:
    st.sidebar.warning("Could not connect to Alpaca API. Showing demo data.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Account Overview", "Strategy Comparison", "Trade Log", "Technical Analysis"
])

# ---- Tab 1: Account Overview ---- #
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    daily_pl = account_info["equity"] - account_info["last_equity"]
    daily_pl_pct = (daily_pl / account_info["last_equity"] * 100) if account_info["last_equity"] else 0

    col1.metric("Total Equity", f"${account_info['equity']:,.2f}")
    col2.metric("Cash", f"${account_info['cash']:,.2f}")
    col3.metric("Buying Power", f"${account_info['buying_power']:,.2f}")
    col4.metric("Today's P&L", f"${daily_pl:,.2f}", f"{daily_pl_pct:+.2f}%")

    st.subheader("Open Positions")
    if demo_mode:
        st.info("Connect your Alpaca API to see live positions.")
    else:
        try:
            positions_df = load_positions()
            if positions_df.empty:
                st.info("No open positions.")
            else:
                st.dataframe(positions_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading positions: {e}")

# ---- Tab 2: Strategy Comparison ---- #
with tab2:
    st.subheader("Backtest Strategy Comparison")

    if run_backtest and selected_strategies and selected_symbols:
        from trading_bot.strategies import get_strategy
        from trading_bot.backtest.engine import BacktestEngine

        symbol = selected_symbols[0]

        # Get bars
        try:
            bars = load_bars(symbol, days=backtest_days)
        except Exception:
            bars = generate_demo_bars(backtest_days)
            st.warning("Using synthetic data for backtest.")

        fig = go.Figure()
        metrics_rows = []

        for strat_name in selected_strategies:
            try:
                strategy = get_strategy(strat_name)
                if strat_name == "ml":
                    strategy.train(bars)
                engine = BacktestEngine(strategy)
                result = engine.run(bars, symbol)

                if not result.equity_curve.empty:
                    fig.add_trace(go.Scatter(
                        x=result.equity_curve["timestamp"],
                        y=result.equity_curve["equity"],
                        mode="lines",
                        name=strat_name,
                    ))

                metrics_rows.append({
                    "Strategy": strat_name,
                    "Return %": f"{result.total_return_pct:.2f}",
                    "Sharpe": f"{result.sharpe_ratio:.2f}",
                    "Max Drawdown %": f"{result.max_drawdown:.2f}",
                    "Win Rate %": f"{result.win_rate:.1f}",
                    "Trades": result.total_trades,
                })
            except Exception as e:
                st.error(f"Error running {strat_name}: {e}")

        fig.update_layout(
            title=f"Equity Curves — {symbol}",
            xaxis_title="Date",
            yaxis_title="Equity ($)",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        if metrics_rows:
            st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True)
    else:
        st.info("Select strategies and symbols, then click 'Run Backtest'.")

# ---- Tab 3: Trade Log ---- #
with tab3:
    st.subheader("Recent Trades")
    if demo_mode:
        demo_trades = pd.DataFrame({
            "Timestamp": pd.date_range(end=datetime.now(), periods=5, freq="D"),
            "Symbol": ["AAPL", "MSFT", "AAPL", "GOOGL", "MSFT"],
            "Side": ["BUY", "BUY", "SELL", "BUY", "SELL"],
            "Qty": [10, 15, 10, 8, 15],
            "Price": [178.50, 415.20, 182.30, 175.80, 420.10],
            "Strategy": ["sma_crossover"] * 5,
        })
        st.dataframe(demo_trades, use_container_width=True)
        st.caption("Demo data — connect Alpaca API for live trades.")
    else:
        try:
            from trading_bot.clients import get_trading_client
            client = get_trading_client()
            orders = client.get_orders(status="filled", limit=50)
            if not orders:
                st.info("No recent trades.")
            else:
                rows = []
                for o in orders:
                    rows.append({
                        "Timestamp": o.filled_at or o.submitted_at,
                        "Symbol": o.symbol,
                        "Side": o.side.value,
                        "Qty": float(o.filled_qty or o.qty),
                        "Price": float(o.filled_avg_price or 0),
                        "Status": o.status.value,
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
        except Exception as e:
            st.error(f"Error loading trades: {e}")

# ---- Tab 4: Technical Analysis ---- #
with tab4:
    st.subheader("Technical Analysis")
    ta_symbol = st.selectbox("Symbol", selected_symbols or DEFAULT_SYMBOLS[:1])

    try:
        bars = load_bars(ta_symbol, days=backtest_days)
    except Exception:
        bars = generate_demo_bars(backtest_days)
        st.warning("Using synthetic data.")

    from trading_bot.data.indicators import (
        compute_sma, compute_rsi, compute_macd, compute_bollinger,
    )

    sma_20 = compute_sma(bars, 20)
    sma_50 = compute_sma(bars, 50)
    bb = compute_bollinger(bars)
    rsi = compute_rsi(bars)
    macd_df = compute_macd(bars)

    # Candlestick + overlays
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=bars.index, open=bars["open"], high=bars["high"],
        low=bars["low"], close=bars["close"], name="Price",
    ))
    fig.add_trace(go.Scatter(x=bars.index, y=sma_20, mode="lines", name="SMA 20", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=bars.index, y=sma_50, mode="lines", name="SMA 50", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=bars.index, y=bb["bb_upper"], mode="lines", name="BB Upper", line=dict(width=1, dash="dash", color="gray")))
    fig.add_trace(go.Scatter(x=bars.index, y=bb["bb_lower"], mode="lines", name="BB Lower", line=dict(width=1, dash="dash", color="gray"), fill="tonexty", fillcolor="rgba(128,128,128,0.1)"))
    fig.update_layout(title=f"{ta_symbol} — Price & Indicators", template="plotly_white", xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # RSI subplot
    col_rsi, col_macd = st.columns(2)
    with col_rsi:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=bars.index, y=rsi, mode="lines", name="RSI"))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig_rsi.update_layout(title="RSI (14)", template="plotly_white", height=300, yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_rsi, use_container_width=True)

    # MACD subplot
    with col_macd:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=bars.index, y=macd_df["macd"], mode="lines", name="MACD"))
        fig_macd.add_trace(go.Scatter(x=bars.index, y=macd_df["macd_signal"], mode="lines", name="Signal"))
        colors = ["green" if v >= 0 else "red" for v in macd_df["macd_diff"]]
        fig_macd.add_trace(go.Bar(x=bars.index, y=macd_df["macd_diff"], name="Histogram", marker_color=colors))
        fig_macd.update_layout(title="MACD", template="plotly_white", height=300)
        st.plotly_chart(fig_macd, use_container_width=True)
