"""Market data fetching from Alpaca API."""

from datetime import datetime, timedelta

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from loguru import logger


class MarketDataFetcher:
    """Fetches OHLCV bar data from the Alpaca API."""

    def __init__(self, client: StockHistoricalDataClient) -> None:
        self.client = client
        logger.info("MarketDataFetcher initialized")

    def get_bars(
        self,
        symbol: str,
        timeframe: TimeFrame = TimeFrame.Day,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for a single symbol.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").
            timeframe: Bar timeframe (default: daily).
            start: Start datetime. Defaults to 6 months ago.
            end: End datetime. Defaults to now.
            limit: Maximum number of bars to return.

        Returns:
            DataFrame with columns: open, high, low, close, volume.
        """
        if start is None:
            start = datetime.now() - timedelta(days=180)
        if end is None:
            end = datetime.now()

        logger.info(
            "Fetching bars for {} | timeframe={} | start={} | end={} | limit={}",
            symbol,
            timeframe,
            start.date(),
            end.date(),
            limit,
        )

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            limit=limit,
            feed=DataFeed.IEX,
        )

        bars = self.client.get_stock_bars(request)
        df = bars.df

        # If multi-index (symbol, timestamp), drop the symbol level
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel("symbol")

        df = df[["open", "high", "low", "close", "volume"]]
        logger.info("Retrieved {} bars for {}", len(df), symbol)
        return df

    def get_latest_bars(self, symbol: str, limit: int = 50) -> pd.DataFrame:
        """Convenience method to fetch the most recent bars.

        Args:
            symbol: Ticker symbol.
            limit: Number of recent bars to fetch.

        Returns:
            DataFrame with columns: open, high, low, close, volume.
        """
        logger.info("Fetching latest {} bars for {}", limit, symbol)
        return self.get_bars(symbol, limit=limit)

    def get_multi_bars(
        self, symbols: list[str], **kwargs
    ) -> dict[str, pd.DataFrame]:
        """Fetch bars for multiple symbols.

        Args:
            symbols: List of ticker symbols.
            **kwargs: Additional arguments passed to get_bars.

        Returns:
            Dictionary mapping each symbol to its DataFrame.
        """
        logger.info("Fetching bars for {} symbols: {}", len(symbols), symbols)
        result: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                result[symbol] = self.get_bars(symbol, **kwargs)
            except Exception:
                logger.exception("Failed to fetch bars for {}", symbol)
        logger.info(
            "Successfully fetched data for {}/{} symbols",
            len(result),
            len(symbols),
        )
        return result
