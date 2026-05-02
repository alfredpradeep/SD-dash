"""
Redis cache for LENS — request-level result caching and TTL management.

Cached items:
  - EntropyProfile for identical (text_hash, model, language) — TTL 5 min
  - Language detection results — TTL 1 hour (language rarely changes for a customer)
  - Tokenizer arbitrage results — TTL 15 min

Falls back to in-memory LRU dict when Redis is unavailable.
"""

import hashlib
import json
import asyncio
from loguru import logger
from lens.config import Config

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class _LRUCache:
    """Minimal in-memory LRU cache (fallback)."""
    def __init__(self, max_size: int = 1000):
        self._store: dict = {}
        self._max = max_size

    def get(self, key: str):
        return self._store.get(key)

    def set(self, key: str, value, ttl: int = None):
        if len(self._store) >= self._max:
            oldest = next(iter(self._store))
            del self._store[oldest]
        self._store[key] = value

    def delete(self, key: str):
        self._store.pop(key, None)


class RedisCache:
    """Redis-backed cache with in-memory LRU fallback."""

    TTL_PROFILE = 300       # 5 minutes
    TTL_LANGUAGE = 3600     # 1 hour
    TTL_ARBITRAGE = 900     # 15 minutes

    def __init__(self, config: Config):
        self.config = config
        self._redis = None
        self._redis_ok = False
        self._fallback = _LRUCache()

    async def _get_redis(self):
        if not REDIS_AVAILABLE:
            return None
        if self._redis_ok is False and self._redis is not None:
            return None  # Previously failed
        if self._redis_ok and self._redis:
            return self._redis
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
            self._redis_ok = True
            return self._redis
        except Exception as e:
            logger.debug("Redis cache unavailable: {} — using in-memory fallback", e)
            self._redis_ok = False
            return None

    def _hash(self, *parts: str) -> str:
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    # ---- Profile caching ----

    def _profile_key(self, text: str, model: str, language: str) -> str:
        return f"lens:profile:{self._hash(text[:200], model, language)}"

    async def get_profile(self, text: str, model: str, language: str) -> dict | None:
        key = self._profile_key(text, model, language)
        r = await self._get_redis()
        if r:
            try:
                raw = await r.get(key)
                return json.loads(raw) if raw else None
            except Exception:
                pass
        return self._fallback.get(key)

    async def set_profile(self, text: str, model: str, language: str, data: dict) -> None:
        key = self._profile_key(text, model, language)
        r = await self._get_redis()
        if r:
            try:
                await r.setex(key, self.TTL_PROFILE, json.dumps(data, default=str))
                return
            except Exception:
                pass
        self._fallback.set(key, data, self.TTL_PROFILE)

    # ---- Language detection caching ----

    def _lang_key(self, text: str) -> str:
        return f"lens:lang:{self._hash(text[:100])}"

    async def get_language(self, text: str) -> tuple[str, float] | None:
        key = self._lang_key(text)
        r = await self._get_redis()
        if r:
            try:
                raw = await r.get(key)
                if raw:
                    parts = raw.split(":")
                    return parts[0], float(parts[1])
            except Exception:
                pass
        cached = self._fallback.get(key)
        return cached

    async def set_language(self, text: str, language: str, confidence: float) -> None:
        key = self._lang_key(text)
        r = await self._get_redis()
        if r:
            try:
                await r.setex(key, self.TTL_LANGUAGE, f"{language}:{confidence}")
                return
            except Exception:
                pass
        self._fallback.set(key, (language, confidence))

    # ---- Arbitrage caching ----

    def _arbitrage_key(self, text: str, language: str) -> str:
        return f"lens:arb:{self._hash(text[:200], language)}"

    async def get_arbitrage(self, text: str, language: str) -> dict | None:
        key = self._arbitrage_key(text, language)
        r = await self._get_redis()
        if r:
            try:
                raw = await r.get(key)
                return json.loads(raw) if raw else None
            except Exception:
                pass
        return self._fallback.get(key)

    async def set_arbitrage(self, text: str, language: str, data: dict) -> None:
        key = self._arbitrage_key(text, language)
        r = await self._get_redis()
        if r:
            try:
                await r.setex(key, self.TTL_ARBITRAGE, json.dumps(data, default=str))
                return
            except Exception:
                pass
        self._fallback.set(key, data)

    async def health(self) -> str:
        r = await self._get_redis()
        if r:
            return "healthy"
        return "degraded (in-memory fallback)"
