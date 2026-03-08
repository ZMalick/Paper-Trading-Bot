"""Random Forest ML strategy."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from trading_bot.data.indicators import (
    compute_bollinger,
    compute_macd,
    compute_rsi,
    compute_sma,
)
from trading_bot.models import Signal, SignalType
from trading_bot.strategies.base import Strategy


class MLStrategy(Strategy):
    """Random Forest classifier predicting next-day price direction."""

    name: str = "ml"

    def __init__(
        self,
        n_estimators: int = 100,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.test_size = test_size
        self.random_state = random_state
        self.model: RandomForestClassifier | None = None
        self.is_trained: bool = False
        self.feature_columns: list[str] = []

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _build_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Compute indicator-based features from OHLCV bars."""
        df = bars.copy()

        # Core indicators
        df["SMA_20"] = compute_sma(bars, window=20)
        df["SMA_50"] = compute_sma(bars, window=50)
        df["RSI"] = compute_rsi(bars, window=14)

        macd_df = compute_macd(bars)
        df["MACD"] = macd_df["macd"]
        df["MACD_signal"] = macd_df["macd_signal"]

        bb_df = compute_bollinger(bars)
        df["BB_upper"] = bb_df["bb_upper"]
        df["BB_middle"] = bb_df["bb_middle"]
        df["BB_lower"] = bb_df["bb_lower"]

        # Derived features
        df["price_sma20_ratio"] = df["close"] / df["SMA_20"]
        df["price_sma50_ratio"] = df["close"] / df["SMA_50"]
        bb_range = df["BB_upper"] - df["BB_lower"]
        df["bb_position"] = np.where(
            bb_range > 0,
            (df["close"] - df["BB_lower"]) / bb_range,
            0.5,
        )

        self.feature_columns = [
            "SMA_20",
            "SMA_50",
            "RSI",
            "MACD",
            "MACD_signal",
            "BB_upper",
            "BB_lower",
            "BB_middle",
            "price_sma20_ratio",
            "price_sma50_ratio",
            "bb_position",
        ]

        return df

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, bars: pd.DataFrame) -> dict:
        """Train the Random Forest model on historical bar data.

        Args:
            bars: DataFrame with OHLCV columns.

        Returns:
            Dictionary with training metrics (accuracy on train/test).
        """
        logger.info("{}: starting training on {} bars", self.name, len(bars))

        df = self._build_features(bars)

        # Target: 1 if next-day close is higher, else 0
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

        # Drop rows with NaN values from indicators or target
        df = df.dropna(subset=self.feature_columns + ["target"])

        if len(df) < 50:
            logger.error("{}: not enough rows after feature computation ({})", self.name, len(df))
            raise ValueError(f"Not enough data to train: {len(df)} rows (need >= 50)")

        X = df[self.feature_columns]
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, shuffle=False,
        )

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True

        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)

        logger.info(
            "{}: training complete | train_acc={:.3f} | test_acc={:.3f}",
            self.name, train_acc, test_acc,
        )

        return {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "features": self.feature_columns,
        }

    # ------------------------------------------------------------------
    # Prediction / analyze
    # ------------------------------------------------------------------

    def analyze(self, symbol: str, bars: pd.DataFrame) -> Signal:
        """Predict next-day direction using the trained model."""
        if not self.is_trained or self.model is None:
            logger.warning("{}: model not trained, returning HOLD", self.name)
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                strategy_name=self.name,
                confidence=0.0,
                metadata={"reason": "model_not_trained"},
            )

        min_bars = 52  # Need at least SMA_50 + buffer
        if len(bars) < min_bars:
            logger.warning(
                "{}: not enough bars ({}/{}), returning HOLD",
                self.name, len(bars), min_bars,
            )
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                strategy_name=self.name,
                confidence=0.0,
                metadata={"reason": "insufficient_data"},
            )

        df = self._build_features(bars)

        # Use the latest row for prediction
        latest = df[self.feature_columns].iloc[[-1]]

        if latest.isna().any(axis=1).iloc[0]:
            logger.warning("{}: features contain NaN for {}, returning HOLD", self.name, symbol)
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                strategy_name=self.name,
                confidence=0.0,
                metadata={"reason": "features_nan"},
            )

        prediction = self.model.predict(latest)[0]
        probabilities = self.model.predict_proba(latest)[0]
        confidence = float(probabilities.max())

        if prediction == 1:
            signal_type = SignalType.BUY
            logger.info(
                "{}: predicted UP for {} (confidence={:.3f})",
                self.name, symbol, confidence,
            )
        else:
            signal_type = SignalType.SELL
            logger.info(
                "{}: predicted DOWN for {} (confidence={:.3f})",
                self.name, symbol, confidence,
            )

        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            strategy_name=self.name,
            confidence=confidence,
            metadata={
                "prediction": int(prediction),
                "prob_down": float(probabilities[0]),
                "prob_up": float(probabilities[1]),
            },
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str | Path) -> None:
        """Save the trained model to disk using joblib."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Cannot save: model has not been trained yet.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "feature_columns": self.feature_columns},
            path,
        )
        logger.info("{}: model saved to {}", self.name, path)

    def load_model(self, path: str | Path) -> None:
        """Load a previously trained model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        self.is_trained = True
        logger.info("{}: model loaded from {}", self.name, path)
