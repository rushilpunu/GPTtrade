"""Behavioral feature calculations for per-symbol signals."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


class BehavioralFeatureCalculator:
    """Compute behavioral features from price/volume data."""

    def __init__(self) -> None:
        self._eps = 1e-12

    def return_anomaly_zscore(self, returns: pd.Series, window: int = 20) -> float:
        """Z-score of the latest return vs a rolling window."""
        series = self._as_series(returns)
        return self._safe_zscore(series, window)

    def volume_anomaly_zscore(self, volumes: pd.Series, window: int = 20) -> float:
        """Z-score of the latest volume vs a rolling window."""
        series = self._as_series(volumes)
        return self._safe_zscore(series, window)

    def volatility_regime(self, prices: pd.DataFrame, window: int = 20) -> float:
        """Volatility regime based on ATR ratio to its rolling mean."""
        if not isinstance(prices, pd.DataFrame):
            return float("nan")

        high = self._get_series(prices, "high")
        low = self._get_series(prices, "low")
        close = self._get_series(prices, "close")

        if high is None or low is None or close is None:
            return float("nan")

        frame = pd.concat([high, low, close], axis=1).dropna()
        if len(frame) < window + 1:
            return float("nan")

        high = frame.iloc[:, 0]
        low = frame.iloc[:, 1]
        close = frame.iloc[:, 2]
        prev_close = close.shift(1)

        tr_components = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        )
        true_range = tr_components.max(axis=1)

        atr = true_range.rolling(window=window).mean()
        if atr.isna().iloc[-1]:
            return float("nan")

        atr_baseline = atr.rolling(window=window).mean()
        baseline_value = atr_baseline.iloc[-1]
        if pd.isna(baseline_value) or abs(baseline_value) < self._eps:
            return float("nan")

        return float(atr.iloc[-1] / baseline_value)

    def mean_reversion_score(
        self,
        price: pd.Series | float,
        vwap: pd.Series | float,
        ma: pd.Series | float,
        window: int = 20,
    ) -> float:
        """Score how stretched price is from VWAP/MA (higher = more stretched)."""
        if self._is_scalar(price) and self._is_scalar(vwap) and self._is_scalar(ma):
            baseline = np.nanmean([float(vwap), float(ma)])
            if np.isnan(baseline) or abs(baseline) < self._eps:
                return float("nan")
            return float((float(price) - baseline) / abs(baseline))

        price_series = self._as_series(price)
        vwap_series = self._as_series(vwap)
        ma_series = self._as_series(ma)

        aligned = pd.concat([price_series, vwap_series, ma_series], axis=1).dropna()
        if len(aligned) < window:
            return float("nan")

        price_series = aligned.iloc[:, 0]
        vwap_series = aligned.iloc[:, 1]
        ma_series = aligned.iloc[:, 2]

        z_vwap = self._safe_zscore(price_series - vwap_series, window)
        z_ma = self._safe_zscore(price_series - ma_series, window)

        if np.isnan(z_vwap) and np.isnan(z_ma):
            return float("nan")
        if np.isnan(z_vwap):
            return float(z_ma)
        if np.isnan(z_ma):
            return float(z_vwap)
        return float((z_vwap + z_ma) / 2.0)

    def trend_score(
        self,
        prices: pd.Series,
        short_window: int = 10,
        long_window: int = 50,
    ) -> float:
        """Trend score from short/long moving average crossover."""
        series = self._as_series(prices)
        if len(series.dropna()) < long_window:
            return float("nan")

        short_ma = series.rolling(window=short_window).mean()
        long_ma = series.rolling(window=long_window).mean()
        aligned = pd.concat([short_ma, long_ma], axis=1).dropna()
        if aligned.empty:
            return float("nan")

        short_val = aligned.iloc[-1, 0]
        long_val = aligned.iloc[-1, 1]
        if abs(long_val) < self._eps:
            return float("nan")

        return float((short_val - long_val) / long_val)

    def compute_all_features(self, symbol: str, ohlcv_df: pd.DataFrame) -> Dict[str, float]:
        """Compute all behavioral features for a symbol."""
        if not isinstance(ohlcv_df, pd.DataFrame) or ohlcv_df.empty:
            return {
                "symbol": symbol,
                "return_anomaly_zscore": float("nan"),
                "volume_anomaly_zscore": float("nan"),
                "volatility_regime": float("nan"),
                "mean_reversion_score": float("nan"),
                "trend_score": float("nan"),
            }

        close = self._get_series(ohlcv_df, "close")
        volume = self._get_series(ohlcv_df, "volume")

        returns = close.pct_change() if close is not None else pd.Series(dtype=float)

        vwap = self._get_series(ohlcv_df, "vwap")
        if vwap is None and close is not None:
            high = self._get_series(ohlcv_df, "high")
            low = self._get_series(ohlcv_df, "low")
            if high is not None and low is not None:
                vwap = (high + low + close) / 3.0

        ma = close.rolling(window=20).mean() if close is not None else pd.Series(dtype=float)

        return {
            "symbol": symbol,
            "return_anomaly_zscore": self.return_anomaly_zscore(returns),
            "volume_anomaly_zscore": self.volume_anomaly_zscore(volume)
            if volume is not None
            else float("nan"),
            "volatility_regime": self.volatility_regime(ohlcv_df),
            "mean_reversion_score": self.mean_reversion_score(
                close if close is not None else float("nan"),
                vwap if vwap is not None else float("nan"),
                ma,
            ),
            "trend_score": self.trend_score(close) if close is not None else float("nan"),
        }

    def _safe_zscore(self, series: pd.Series, window: int) -> float:
        series = self._as_series(series).dropna()
        if len(series) < window:
            return float("nan")

        window_slice = series.iloc[-window:]
        mean = window_slice.mean()
        std = window_slice.std(ddof=0)
        if pd.isna(std) or abs(std) < self._eps:
            return float("nan")

        return float((window_slice.iloc[-1] - mean) / std)

    @staticmethod
    def _get_series(frame: pd.DataFrame, column: str) -> Optional[pd.Series]:
        if column in frame.columns:
            return frame[column].astype(float)
        return None

    @staticmethod
    def _as_series(values: pd.Series | np.ndarray | list | float) -> pd.Series:
        if isinstance(values, pd.Series):
            return values.astype(float)
        if isinstance(values, np.ndarray):
            return pd.Series(values.astype(float))
        if isinstance(values, list):
            return pd.Series([float(v) if v is not None else np.nan for v in values])
        if isinstance(values, (int, float, np.floating)):
            return pd.Series([float(values)])
        return pd.Series(dtype=float)

    @staticmethod
    def _is_scalar(value: object) -> bool:
        return isinstance(value, (int, float, np.floating))
