"""
Redis-backed cache for compression results.

Cache key = SHA-256 hash of (text, source_language, target_tokenizer).
Gracefully degrades: cache failures log warnings but never block compression.
"""

import hashlib
import json
import redis.asyncio as redis
from compress.lattice.structures import CompressionResult
from compress.config import Config
from loguru import logger


class CompressionCache:
    """
    Redis-backed cache for compression results.
    """

    def __init__(self, config: Config):
        self.config = config
        self._pool: redis.Redis | None = None
        self._available = True

    async def _get_pool(self) -> redis.Redis | None:
        """Lazy-init Redis connection pool."""
        if self._pool is None:
            try:
                self._pool = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    decode_responses=True,
                    socket_connect_timeout=2.0,
                    socket_timeout=2.0,
                )
                await self._pool.ping()
            except Exception as e:
                logger.warning("Redis unavailable ({}), caching disabled", e)
                self._available = False
                self._pool = None
        return self._pool

    def make_key(self, text: str, language: str, tokenizer: str) -> str:
        """Generate deterministic cache key."""
        raw = f"{text}|{language}|{tokenizer}"
        return f"scl:compress:{hashlib.sha256(raw.encode()).hexdigest()}"

    async def get(self, key: str) -> CompressionResult | None:
        """Retrieve cached result. Returns None on miss or error."""
        if not self._available:
            return None
        try:
            pool = await self._get_pool()
            if pool is None:
                return None
            raw = await pool.get(key)
            if raw is None:
                return None
            data = json.loads(raw)
            return CompressionResult(**data)
        except Exception as e:
            logger.warning("Cache get failed for key {}: {}", key[:20], e)
            return None

    async def set(
        self,
        key: str,
        result: CompressionResult,
        ttl_seconds: int = 3600,
    ) -> None:
        """Store result in cache. Silently fails on error."""
        if not self._available:
            return
        try:
            pool = await self._get_pool()
            if pool is None:
                return
            data = json.dumps(vars(result))
            await pool.set(key, data, ex=ttl_seconds)
        except Exception as e:
            logger.warning("Cache set failed for key {}: {}", key[:20], e)

    async def health(self) -> str:
        """Return cache health status."""
        if not self._available:
            return "degraded (redis unavailable)"
        try:
            pool = await self._get_pool()
            if pool:
                await pool.ping()
                return "healthy"
            return "degraded"
        except Exception:
            return "degraded"
