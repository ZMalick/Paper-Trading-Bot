# Paper Trading Bot

AI-driven paper trading bot using Alpaca's API with multiple strategies and a real-time dashboard.

## Features

- **5 Trading Strategies**: SMA Crossover, RSI, MACD, Mean Reversion, ML Ensemble (Random Forest)
- **Paper Trading**: Real-time execution via Alpaca's paper trading API ($100k simulated capital)
- **Backtesting Engine**: Compare strategy performance on historical data
- **Interactive Dashboard**: Streamlit + Plotly for live monitoring and analysis
- **Portfolio Tracking**: Sharpe ratio, max drawdown, win rate, equity curves

## Architecture

```
src/trading_bot/
├── config.py          # Configuration (pydantic-settings)
├── clients.py         # Alpaca API clients + rate limiting
├── models.py          # Pydantic data models
├── bot.py             # Main trading loop orchestrator
├── data/
│   ├── market_data.py # OHLCV data fetching (IEX feed)
│   └── indicators.py  # SMA, EMA, RSI, MACD, Bollinger Bands
├── strategies/
│   ├── base.py            # Abstract strategy interface
│   ├── sma_crossover.py   # Golden/death cross
│   ├── rsi.py             # Oversold/overbought
│   ├── macd.py            # Signal line crossover
│   ├── mean_reversion.py  # Bollinger Band reversion
│   └── ml_strategy.py     # Random Forest classifier
├── execution/
│   ├── order_manager.py   # Order placement + position sizing
│   └── portfolio.py       # P&L tracking + performance metrics
├── backtest/
│   └── engine.py          # Historical strategy replay
└── dashboard/
    └── app.py             # Streamlit dashboard
```

## Quick Start

### Prerequisites
- Python 3.11+
- Free [Alpaca](https://alpaca.markets/) paper trading account

### Setup
```bash
git clone https://github.com/ZMalick/Paper-Trading-Bot.git
cd Paper-Trading-Bot
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your Alpaca API keys
```

### Run the Bot
```bash
python -m trading_bot.bot
```

### Run the Dashboard
```bash
streamlit run src/trading_bot/dashboard/app.py
```

### Run Backtests
```python
from trading_bot.backtest.engine import BacktestEngine
from trading_bot.strategies import get_strategy
from trading_bot.clients import get_data_client
from trading_bot.data.market_data import MarketDataFetcher

# Fetch data
client = get_data_client()
fetcher = MarketDataFetcher(client)
bars = fetcher.get_bars("AAPL", limit=365)

# Run backtest
strategy = get_strategy("sma_crossover")
engine = BacktestEngine(strategy)
result = engine.run(bars, "AAPL")
print(f"Return: {result.total_return_pct:.2f}%")
print(f"Sharpe: {result.sharpe_ratio:.2f}")
```

## Strategies

| Strategy | Signal | Key Parameters |
|----------|--------|---------------|
| **SMA Crossover** | Buy on golden cross (SMA20 > SMA50), sell on death cross | `short_window=20`, `long_window=50` |
| **RSI** | Buy when RSI < 30, sell when RSI > 70 | `window=14`, `oversold=30`, `overbought=70` |
| **MACD** | Buy/sell on MACD-signal line crossovers | `fast=12`, `slow=26`, `signal=9` |
| **Mean Reversion** | Buy below lower Bollinger Band, sell above upper | `window=20`, `std_dev=2` |
| **ML Ensemble** | Random Forest predicts next-day direction using indicator features | `n_estimators=100` |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ALPACA_API_KEY` | *required* | Alpaca API key |
| `ALPACA_SECRET_KEY` | *required* | Alpaca secret key |
| `ALPACA_BASE_URL` | `https://paper-api.alpaca.markets` | API endpoint |
| `TRADING_SYMBOLS` | `AAPL,MSFT,GOOGL,AMZN,TSLA` | Comma-separated symbols |
| `TRADING_INTERVAL` | `60` | Loop interval in seconds |
| `MAX_POSITION_PCT` | `0.1` | Max portfolio % per position |
| `STRATEGY` | `sma_crossover` | Active strategy name |
| `LOG_LEVEL` | `INFO` | Logging level |

## Testing

```bash
pytest tests/ -v
ruff check src/ tests/
```

## License

MIT
