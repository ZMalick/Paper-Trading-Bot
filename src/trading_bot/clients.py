import time
import threading

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from loguru import logger

from .config import settings


class RateLimiter:
    """Simple rate limiter that tracks API calls and sleeps if approaching the limit."""

    def __init__(self, max_calls: int = 200, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self._calls: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Wait if necessary to stay under the rate limit."""
        with self._lock:
            now = time.time()
            # Remove calls outside the current window
            self._calls = [t for t in self._calls if now - t < self.period]

            if len(self._calls) >= self.max_calls - 10:
                # Approaching the limit — sleep until the oldest call expires
                sleep_time = self.period - (now - self._calls[0])
                if sleep_time > 0:
                    logger.warning(
                        f"Rate limit approaching ({len(self._calls)}/{self.max_calls}), "
                        f"sleeping {sleep_time:.1f}s"
                    )
                    time.sleep(sleep_time)
                    # Clean up again after sleeping
                    now = time.time()
                    self._calls = [t for t in self._calls if now - t < self.period]

            self._calls.append(time.time())


# Module-level rate limiter shared across clients
rate_limiter = RateLimiter(max_calls=200, period=60.0)


def get_trading_client() -> TradingClient:
    """Create and return an Alpaca TradingClient configured for paper trading."""
    logger.info("Initializing Alpaca TradingClient (paper=True)")
    rate_limiter.acquire()
    return TradingClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
        paper=True,
    )


def get_data_client() -> StockHistoricalDataClient:
    """Create and return an Alpaca StockHistoricalDataClient."""
    logger.info("Initializing Alpaca StockHistoricalDataClient")
    rate_limiter.acquire()
    return StockHistoricalDataClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
    )
