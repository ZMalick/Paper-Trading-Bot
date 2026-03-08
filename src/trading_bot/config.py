from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    alpaca_api_key: str = Field(..., description="Alpaca API key")
    alpaca_secret_key: str = Field(..., description="Alpaca secret key")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        description="Alpaca API base URL",
    )
    trading_symbols: str = Field(
        default="AAPL,MSFT,GOOGL,AMZN,TSLA",
        description="Comma-separated symbols to trade",
    )
    trading_interval: int = Field(
        default=60, description="Trading loop interval in seconds"
    )
    log_level: str = Field(default="INFO", description="Logging level")
    max_position_pct: float = Field(
        default=0.1, description="Max portfolio % per position"
    )
    strategy: str = Field(
        default="sma_crossover", description="Active trading strategy"
    )

    @property
    def symbols_list(self) -> List[str]:
        """Parse trading_symbols string into a list."""
        return [s.strip() for s in self.trading_symbols.split(",")]


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Convenience alias for backwards compatibility
settings = get_settings()
