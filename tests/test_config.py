"""Tests for config module."""

import importlib


def _make_settings(monkeypatch, extra_env=None):
    """Create a fresh Settings instance with given env vars."""
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")
    if extra_env:
        for k, v in extra_env.items():
            monkeypatch.setenv(k, v)
    # Need to reimport to avoid cached module-level `settings`
    import trading_bot.config
    importlib.reload(trading_bot.config)
    # Skip .env file so tests only see env vars set by monkeypatch
    return trading_bot.config.Settings(_env_file=None)


def test_settings_loads_from_env(monkeypatch):
    s = _make_settings(monkeypatch)
    assert s.alpaca_api_key == "test_key"
    assert s.alpaca_secret_key == "test_secret"


def test_default_values(monkeypatch):
    s = _make_settings(monkeypatch)
    assert s.alpaca_base_url == "https://paper-api.alpaca.markets"
    assert s.trading_interval == 60
    assert s.log_level == "INFO"
    assert s.max_position_pct == 0.1
    assert s.strategy == "sma_crossover"


def test_trading_symbols_parsing(monkeypatch):
    s = _make_settings(monkeypatch, {"TRADING_SYMBOLS": "AAPL, TSLA, MSFT"})
    assert s.symbols_list == ["AAPL", "TSLA", "MSFT"]


def test_trading_symbols_default_is_list(monkeypatch):
    s = _make_settings(monkeypatch)
    symbols = s.symbols_list
    assert isinstance(symbols, list)
    assert len(symbols) == 5
