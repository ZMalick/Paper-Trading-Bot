from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Signal(BaseModel):
    symbol: str
    signal_type: SignalType
    strategy_name: str
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TradeRecord(BaseModel):
    symbol: str
    side: str
    qty: float
    price: float
    timestamp: datetime = Field(default_factory=datetime.now)
    strategy_name: str
    signal_confidence: float = Field(ge=0.0, le=1.0)
    order_id: str


class PerformanceSnapshot(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    total_equity: float
    cash: float
    positions_value: float
    daily_return_pct: float
    total_return_pct: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    total_trades: int
