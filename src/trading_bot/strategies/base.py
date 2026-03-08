from abc import ABC, abstractmethod
import pandas as pd
from trading_bot.models import Signal

class Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def analyze(self, symbol: str, bars: pd.DataFrame) -> Signal:
        """Analyze bars and return a trading signal."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
