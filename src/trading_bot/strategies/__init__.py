from trading_bot.strategies.sma_crossover import SMACrossoverStrategy
from trading_bot.strategies.rsi import RSIStrategy
from trading_bot.strategies.macd import MACDStrategy
from trading_bot.strategies.mean_reversion import MeanReversionStrategy
from trading_bot.strategies.ml_strategy import MLStrategy

STRATEGY_REGISTRY: dict[str, type] = {
    "sma_crossover": SMACrossoverStrategy,
    "rsi": RSIStrategy,
    "macd": MACDStrategy,
    "mean_reversion": MeanReversionStrategy,
    "ml": MLStrategy,
}

def get_strategy(name: str):
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name]()
