"""
Rolling entropy window manager backed by Redis sorted sets.

Maintains a per-language sliding window of (timestamp, entropy_bits) observations
over the last N minutes. Used by SpikePredictor to compute entropy velocity.

Redis data structure:
    Key: lens:entropy_window:{language}:{customer_id}
    Type: Sorted Set — score=unix_timestamp_ms, member=entropy_bits_as_string
"""

import asyncio
import time
from datetime import datetime, timedelta
from loguru import logger
from lens.config import Config

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis not available — using in-memory rolling windows")


class RollingEntropyWindow:
    """
    Per-language rolling entropy window.

    Stores entropy observations in Redis sorted sets (score = timestamp).
    Automatically expires old observations beyond the rolling window duration.
    Falls back to in-memory deques when Redis is unavailable.
    """

    def __init__(self, config: Config):
        self.config = config
        self._redis: "aioredis.Redis | None" = None
        self._redis_connected = False
        self._redis_attempted = False
        # In-memory fallback: {key: list[(datetime, float)]}
        self._memory_store: dict[str, list[tuple[datetime, float]]] = {}

    async def _get_redis(self):
        if not REDIS_AVAILABLE:
            return None
        if self._redis_attempted and not self._redis_connected:
            return None
        if self._redis_connected and self._redis:
            return self._redis
        self._redis_attempted = True
        try:
            self._redis = aioredis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            await self._redis.ping()
            self._redis_connected = True
            logger.info("Redis connected for rolling windows")
            return self._redis
        except Exception as e:
            logger.warning("Redis unavailable: {} — using in-memory fallback", e)
            self._redis_connected = False
            return None

    def _make_key(self, language: str, customer_id: str = "global") -> str:
        return f"lens:entropy_window:{customer_id}:{language}"

    async def update(
        self,
        language: str,
        entropy_bits: float,
        timestamp: datetime,
        customer_id: str = "global",
    ) -> None:
        """Add a new entropy observation to the window."""
        ts_ms = int(timestamp.timestamp() * 1000)
        key = self._make_key(language, customer_id)
        window_ms = self.config.rolling_window_minutes * 60 * 1000
        cutoff_ms = ts_ms - window_ms

        r = await self._get_redis()
        if r:
            try:
                pipe = r.pipeline()
                pipe.zadd(key, {str(entropy_bits): ts_ms})
                pipe.zremrangebyscore(key, 0, cutoff_ms)
                pipe.expire(key, self.config.rolling_window_minutes * 120)
                await pipe.execute()
                return
            except Exception as e:
                logger.debug("Redis write failed: {}", e)

        # In-memory fallback
        if key not in self._memory_store:
            self._memory_store[key] = []
        self._memory_store[key].append((timestamp, entropy_bits))
        # Trim to window
        cutoff = timestamp - timedelta(minutes=self.config.rolling_window_minutes)
        self._memory_store[key] = [
            (ts, e) for ts, e in self._memory_store[key] if ts >= cutoff
        ]

    async def get(
        self,
        language: str,
        customer_id: str = "global",
    ) -> list[tuple[datetime, float]] | None:
        """
        Retrieve the rolling window as a list of (timestamp, entropy_bits) tuples,
        sorted by timestamp ascending.
        """
        key = self._make_key(language, customer_id)
        r = await self._get_redis()

        if r:
            try:
                now_ms = int(time.time() * 1000)
                cutoff_ms = now_ms - (self.config.rolling_window_minutes * 60 * 1000)
                # Get (member, score) pairs within the window
                items = await r.zrangebyscore(key, cutoff_ms, "+inf", withscores=True)
                if not items:
                    return None
                result = []
                for member, score in items:
                    ts = datetime.fromtimestamp(score / 1000)
                    try:
                        entropy = float(member)
                    except ValueError:
                        continue
                    result.append((ts, entropy))
                result.sort(key=lambda x: x[0])
                return result if len(result) >= 3 else None
            except Exception as e:
                logger.debug("Redis read failed: {}", e)

        # In-memory fallback
        data = self._memory_store.get(key)
        if not data or len(data) < 3:
            return None
        return sorted(data, key=lambda x: x[0])

    async def get_all_languages(self, customer_id: str = "global") -> list[str]:
        """Return all languages that have active rolling windows."""
        r = await self._get_redis()
        if r:
            try:
                pattern = f"lens:entropy_window:{customer_id}:*"
                keys = await r.keys(pattern)
                prefix = f"lens:entropy_window:{customer_id}:"
                return [k.replace(prefix, "") for k in keys]
            except Exception:
                pass
        # In-memory fallback
        prefix = self._make_key("", customer_id).rstrip(":")
        langs = []
        for key in self._memory_store:
            if key.startswith(prefix):
                langs.append(key.split(":")[-1])
        return langs

    async def clear(self, language: str, customer_id: str = "global") -> None:
        """Clear the rolling window for a language (for testing)."""
        key = self._make_key(language, customer_id)
        r = await self._get_redis()
        if r:
            try:
                await r.delete(key)
                return
            except Exception:
                pass
        self._memory_store.pop(key, None)
