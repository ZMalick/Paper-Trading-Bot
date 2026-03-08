# Trading Strategies

## 1. SMA Crossover (Trend Following)

**Concept**: Uses two Simple Moving Averages of different periods. When the short-term SMA crosses above the long-term SMA, it signals an uptrend (golden cross). The reverse signals a downtrend (death cross).

**Formula**:
```
SMA(n) = (P₁ + P₂ + ... + Pₙ) / n
```

**Signals**:
- **BUY**: SMA(20) crosses above SMA(50)
- **SELL**: SMA(20) crosses below SMA(50)

**Best for**: Trending markets with sustained directional moves. Struggles in sideways/choppy markets.

**Parameters**: `short_window=20`, `long_window=50`

---

## 2. RSI (Momentum)

**Concept**: Relative Strength Index measures the speed and magnitude of price changes on a 0-100 scale.

**Formula**:
```
RSI = 100 - 100 / (1 + RS)
RS = Average Gain / Average Loss (over n periods)
```

**Signals**:
- **BUY**: RSI < 30 (oversold)
- **SELL**: RSI > 70 (overbought)

**Best for**: Range-bound markets. Can generate false signals during strong trends.

**Parameters**: `window=14`, `oversold=30`, `overbought=70`

---

## 3. MACD (Trend + Momentum)

**Concept**: Moving Average Convergence Divergence shows the relationship between two EMAs of a security's price.

**Formula**:
```
MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(9) of MACD Line
Histogram = MACD Line - Signal Line
```

**Signals**:
- **BUY**: MACD line crosses above signal line
- **SELL**: MACD line crosses below signal line

**Best for**: Confirming trend changes. Combines trend-following and momentum aspects.

**Parameters**: `fast=12`, `slow=26`, `signal=9`

---

## 4. Mean Reversion (Bollinger Bands)

**Concept**: Prices tend to revert to their mean. Bollinger Bands create a channel around the moving average; prices touching the bands are expected to revert.

**Formula**:
```
Middle Band = SMA(20)
Upper Band = SMA(20) + 2σ
Lower Band = SMA(20) - 2σ
```

**Signals**:
- **BUY**: Price drops below lower Bollinger Band
- **SELL**: Price rises above upper Bollinger Band

**Best for**: Range-bound, mean-reverting markets. Risky during breakouts.

**Parameters**: `window=20`, `std_dev=2`

---

## 5. ML Ensemble (Random Forest)

**Concept**: Uses a Random Forest classifier trained on technical indicator values to predict next-day price direction.

**Features** (11 total):
- SMA_20, SMA_50
- RSI (14-period)
- MACD, MACD signal line
- Bollinger Bands (upper, middle, lower)
- Derived: price/SMA20 ratio, price/SMA50 ratio, Bollinger Band position

**Target**: Binary — 1 if next-day close > today's close, 0 otherwise

**Training**: Uses time-series aware train/test split (no shuffle) to prevent look-ahead bias.

**Confidence**: Model's predicted probability for the chosen class.

**Parameters**: `n_estimators=100`, `test_size=0.2`
