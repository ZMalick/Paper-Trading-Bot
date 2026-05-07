# Paper Trading Bot

> A Python paper trading bot that runs five strategies — including an ML ensemble — against live Alpaca paper accounts, with a Streamlit dashboard and a backtesting engine for historical comparison.

<!-- DEMO: add a screenshot or short GIF of the Streamlit dashboard here. Recruiters spend ~6 seconds on a README - the demo is the hook. -->

## What it does

Connects to Alpaca's paper trading API (free, $100k simulated capital), runs one of five strategies on a configurable list of symbols, and tracks P&L in real time. A separate backtesting engine replays each strategy on historical OHLCV data so you can compare performance before deploying. The Streamlit dashboard surfaces equity curves, Sharpe ratio, max drawdown, and win rate.

## Tech Stack

- **Language:** Python 3.11+
- **Trading API:** alpaca-py (Alpaca paper trading)
- **Data + indicators:** pandas, numpy, ta (technical analysis)
- **ML:** scikit-learn (Random Forest classifier)
- **Dashboard:** Streamlit + Plotly
- **Config:** pydantic-settings (env-driven)

## Strategies

| Strategy | Signal | Key Parameters |
|----------|--------|----------------|
| **SMA Crossover** | Buy on golden cross (SMA20 > SMA50), sell on death cross | `short=20`, `long=50` |
| **RSI** | Buy when RSI < 30, sell when RSI > 70 | `window=14`, `oversold=30`, `overbought=70` |
| **MACD** | Buy/sell on MACD-signal line crossovers | `fast=12`, `slow=26`, `signal=9` |
| **Mean Reversion** | Buy below lower Bollinger Band, sell above upper | `window=20`, `std_dev=2` |
| **ML Ensemble** | Random Forest predicts next-day direction from indicator features | `n_estimators=100` |

## Backtest Results

Run `python run_backtest.py` to compare all five strategies against buy-and-hold across AAPL, MSFT, and TSLA on 365 days of historical data. Output looks like:

```
=== AAPL | 2024-04-23 to 2025-04-23 | 252 days ===
Strategy                Return   Sharpe   Max DD  WinRate  Trades
sma_crossover           +X.XX%     X.XX    X.XX%    XX.X%       N
rsi                     +X.XX%     X.XX    X.XX%    XX.X%       N
...
Buy & Hold              +X.XX%
```

<!-- TODO: Zaid to paste real backtest numbers here once he runs run_backtest.py - this is the section recruiters will scroll to. -->

## Architecture

```
src/trading_bot/
├── config.py             # pydantic-settings env config
├── clients.py            # Alpaca clients + rate limiting
├── models.py             # pydantic data models
├── bot.py                # main trading loop
├── data/                 # market data + indicators
├── strategies/           # 5 strategies behind a base interface
├── execution/            # order manager + portfolio tracking
├── backtest/             # historical replay engine
└── dashboard/            # Streamlit app
```

The `strategies/base.py` abstract class makes it easy to add new strategies — implement `generate_signal(bars) -> Signal` and register the class in `strategies/__init__.py`.

## Setup

```bash
git clone https://github.com/ZMalick/Paper-Trading-Bot.git
cd Paper-Trading-Bot
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your Alpaca paper API keys (free: alpaca.markets)
```

## Usage

**Run live (paper) trading:**
```bash
python -m trading_bot.bot
```

**Launch the dashboard:**
```bash
streamlit run src/trading_bot/dashboard/app.py
```

**Run backtests:**
```bash
python run_backtest.py
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ALPACA_API_KEY` | *required* | Alpaca paper API key |
| `ALPACA_SECRET_KEY` | *required* | Alpaca paper secret |
| `ALPACA_BASE_URL` | `https://paper-api.alpaca.markets` | API endpoint |
| `TRADING_SYMBOLS` | `AAPL,MSFT,GOOGL,AMZN,TSLA` | Symbols to trade |
| `TRADING_INTERVAL` | `60` | Loop interval (seconds) |
| `MAX_POSITION_PCT` | `0.1` | Max portfolio % per position |
| `STRATEGY` | `sma_crossover` | Active strategy |
| `LOG_LEVEL` | `INFO` | Logging level |

## Testing

```bash
pytest tests/ -v
ruff check src/ tests/
```

## Why I built this / What I learned

<!-- TODO: Zaid to fill in 2-3 sentences. Examples of what to include:
  - The constraint or curiosity that drove the project (e.g., "Wanted to understand whether classical TA strategies actually beat buy-and-hold on liquid US equities")
  - One or two technical things you learned (e.g., "Learned how to design a strategy interface that doesn't leak indicator computation into the bot loop", or "Found that the ML ensemble overfits on training windows under 6 months")
  - Avoid generic phrases like "passionate about quant" - be specific
-->

## License

MIT
