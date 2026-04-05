"""Feature engineering from market history (aligned train/inference)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["price", "price_change", "rsi", "sma", "volume", "volatility"]


def _history_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _compute_volatility(close: pd.Series, window: int = 5) -> pd.Series:
    ret = close.pct_change()
    return ret.rolling(window=window, min_periods=1).std()


def build_training_matrix(
    rows: List[Dict[str, Any]],
    volatility_window: int = 5,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build X, y for binary classification: next-day UP (1) vs DOWN (0).
    Drops last row (no future price).
    """
    df = _history_to_dataframe(rows)
    if len(df) < 3:
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series(dtype=int)

    price = df["price"].astype(float)
    price_change = price.pct_change().fillna(0.0)
    rsi = df["rsi"].astype(float) if "rsi" in df else pd.Series(50.0, index=df.index)
    sma = df["sma"].astype(float) if "sma" in df else price.rolling(5, min_periods=1).mean()
    volume = df["volume"].astype(float) if "volume" in df else pd.Series(0.0, index=df.index)
    volatility = _compute_volatility(price, window=volatility_window)

    feat = pd.DataFrame(
        {
            "price": price,
            "price_change": price_change,
            "rsi": rsi,
            "sma": sma,
            "volume": volume,
            "volatility": volatility.fillna(0.0),
        }
    )

    next_up = (price.shift(-1) > price).astype(int)
    feat = feat.iloc[:-1].reset_index(drop=True)
    y = next_up.iloc[:-1].reset_index(drop=True)

    combined = pd.concat([feat, y.rename("_y")], axis=1).dropna()
    if combined.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series(dtype=int)

    return combined[FEATURE_COLUMNS].reset_index(drop=True), combined["_y"].astype(int).reset_index(drop=True)


def latest_features_from_history(
    rows: List[Dict[str, Any]],
    volatility_window: int = 5,
) -> Optional[Dict[str, float]]:
    """Single-row feature dict for the latest observation (inference)."""
    df = _history_to_dataframe(rows)
    if df.empty:
        return None
    price = df["price"].astype(float)
    price_change = price.pct_change().fillna(0.0)
    rsi = df["rsi"].astype(float) if "rsi" in df else pd.Series(50.0, index=df.index)
    sma = df["sma"].astype(float) if "sma" in df else price.rolling(5, min_periods=1).mean()
    volume = df["volume"].astype(float) if "volume" in df else pd.Series(0.0, index=df.index)
    volatility = _compute_volatility(price, window=volatility_window)

    out = {
        "price": float(price.iloc[-1]),
        "price_change": float(price_change.iloc[-1]),
        "rsi": float(rsi.iloc[-1]),
        "sma": float(sma.iloc[-1]),
        "volume": float(volume.iloc[-1]),
        "volatility": float(volatility.fillna(0.0).iloc[-1]),
    }
    return out
