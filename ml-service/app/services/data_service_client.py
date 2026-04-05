"""REST client for Data Service: retries, timeout, in-memory cache fallback."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from infrastructure.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def _synthetic_history(symbol: str, n: int = 40) -> Dict[str, Any]:
    """Deterministic pseudo history for local/dev when Data Service is unavailable."""
    h = int(hashlib.sha256(symbol.encode()).hexdigest()[:8], 16)
    rng_seed = h % (2**31)
    import numpy as np

    rng = np.random.default_rng(rng_seed)
    base = 20.0 + (h % 100) / 10.0
    price = base + np.cumsum(rng.normal(0, 0.3, size=n))
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        p = float(price[i])
        sma = float(np.mean(price[max(0, i - 4) : i + 1]))
        rows.append(
            {
                "price": p,
                "rsi": float(35 + (h + i) % 40),
                "sma": sma,
                "volume": float(800_000 + (h + i) % 200_000),
                "timestamp": f"2026-03-{(i % 28) + 1:02d}",
            }
        )
    return {"symbol": symbol.upper(), "data": rows}


class DataServiceClient:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._ttl_seconds = 300.0

    def _get_cached(self, symbol: str) -> Optional[Dict[str, Any]]:
        entry = self._cache.get(symbol.upper())
        if not entry:
            return None
        ts, payload = entry
        if time.time() - ts > self._ttl_seconds:
            return None
        logger.info("data_service_cache_hit", extra={"symbol": symbol})
        return payload

    def _set_cache(self, symbol: str, payload: Dict[str, Any]) -> None:
        self._cache[symbol.upper()] = (time.time(), payload)

    def _fetch(self, symbol: str) -> Dict[str, Any]:
        url = (
            f"{self.settings.data_service_base_url.rstrip('/')}"
            f"/market-data/{symbol}/history"
        )
        with httpx.Client(timeout=self.settings.data_service_timeout_seconds) as client:
            r = client.get(url)
            r.raise_for_status()
            return r.json()

    def get_market_history(self, symbol: str) -> Dict[str, Any]:
        sym = symbol.strip().upper()
        cached = self._get_cached(sym)
        if cached is not None:
            return cached

        attempts = max(1, self.settings.data_service_max_retries)
        last_err: Optional[Exception] = None
        for i in range(attempts):
            try:
                payload = self._fetch(sym)
                self._set_cache(sym, payload)
                return payload
            except Exception as e:
                last_err = e
                logger.warning(
                    "data_service_fetch_failed",
                    extra={"symbol": sym, "attempt": i + 1, "error": str(e)},
                )
                if i < attempts - 1:
                    time.sleep(self.settings.data_service_retry_backoff_seconds)

        stale = self._cache.get(sym)
        if stale:
            logger.error(
                "data_service_fallback_cache",
                extra={"symbol": sym, "error": str(last_err)},
            )
            return stale[1]

        if self.settings.synthetic_fallback:
            logger.warning(
                "data_service_synthetic_fallback",
                extra={"symbol": sym, "error": str(last_err)},
            )
            payload = _synthetic_history(sym)
            self._set_cache(sym, payload)
            return payload

        raise RuntimeError(f"Data Service unavailable and no cache for {sym}") from last_err
