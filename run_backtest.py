"""Run backtests on real data and display results."""

import warnings
warnings.filterwarnings("ignore")

from loguru import logger
logger.remove()

from trading_bot.clients import get_data_client
from trading_bot.data.market_data import MarketDataFetcher
from trading_bot.backtest.engine import BacktestEngine
from trading_bot.strategies import get_strategy


def main():
    client = get_data_client()
    fetcher = MarketDataFetcher(client)

    for symbol in ["AAPL", "MSFT", "TSLA"]:
        bars = fetcher.get_bars(symbol, limit=365)
        buy_hold = (bars["close"].iloc[-1] - bars["close"].iloc[0]) / bars["close"].iloc[0] * 100

        print(f"=== {symbol} | {bars.index[0].date()} to {bars.index[-1].date()} | {len(bars)} days ===")
        print(f"Price: ${bars['close'].iloc[0]:.2f} -> ${bars['close'].iloc[-1]:.2f}")
        print()
        header = f"{'Strategy':<20} {'Return':>9} {'Sharpe':>8} {'Max DD':>8} {'WinRate':>9} {'Trades':>7}"
        print(header)
        print("-" * len(header))

        for name in ["sma_crossover", "rsi", "macd", "mean_reversion", "ml"]:
            strategy = get_strategy(name)
            if name == "ml":
                strategy.train(bars)
            engine = BacktestEngine(strategy)
            r = engine.run(bars, symbol)
            print(
                f"{name:<20} {r.total_return_pct:>+8.2f}% {r.sharpe_ratio:>8.2f} "
                f"{r.max_drawdown:>7.2f}% {r.win_rate:>8.1f}% {r.total_trades:>7}"
            )

        print(f"{'Buy & Hold':<20} {buy_hold:>+8.2f}%")
        print()


if __name__ == "__main__":
    main()
