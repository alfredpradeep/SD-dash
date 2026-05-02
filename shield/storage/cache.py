"""
ScanCache: Redis-backed caching for probe results.

Implements SHA-256 based keying and graceful degradation.
"""

import hashlib
import json
from typing import Optional, Dict, Any
from dataclasses import asdict

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from loguru import logger


class ProbeResult:
    """Cached probe result."""

    def __init__(
        self,
        probe_text: str,
        language: str,
        endpoint: str,
        response: str,
        judgment: Dict[str, Any],
        harm_vector: Dict[str, float],
    ):
        """Initialize probe result."""
        self.probe_text = probe_text
        self.language = language
        self.endpoint = endpoint
        self.response = response
        self.judgment = judgment
        self.harm_vector = harm_vector

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "probe_text": self.probe_text,
            "language": self.language,
            "endpoint": self.endpoint,
            "response": self.response,
            "judgment": self.judgment,
            "harm_vector": self.harm_vector,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ProbeResult":
        """Create from dict."""
        return ProbeResult(
            probe_text=data["probe_text"],
            language=data["language"],
            endpoint=data["endpoint"],
            response=data["response"],
            judgment=data["judgment"],
            harm_vector=data["harm_vector"],
        )


class ScanCache:
    """
    Redis-backed cache for scan results.

    Uses SHA-256 keying on (probe_text + language + endpoint).
    Gracefully degrades if Redis is unavailable.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        ttl_seconds: int = 3600,
    ):
        """Initialize cache."""
        self.ttl_seconds = ttl_seconds
        self.redis_client = None
        self.degraded = False

        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    decode_responses=True,
                    socket_connect_timeout=5,
                )
                self.redis_client.ping()
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Redis connection failed, running degraded: {e}")
                self.degraded = True
        else:
            logger.warning("Redis not available, running in degraded mode")
            self.degraded = True

    @staticmethod
    def _make_key(probe_text: str, language: str, endpoint: str) -> str:
        """Create cache key from probe components."""
        combined = f"{probe_text}:{language}:{endpoint}"
        return f"shield:probe:{hashlib.sha256(combined.encode()).hexdigest()}"

    async def get_result(
        self,
        probe_text: str,
        language: str,
        endpoint: str,
    ) -> Optional[ProbeResult]:
        """Get cached probe result."""
        if self.degraded:
            return None

        try:
            key = self._make_key(probe_text, language, endpoint)
            data = self.redis_client.get(key)
            if data:
                result_dict = json.loads(data)
                logger.debug(f"Cache hit for probe: {key}")
                return ProbeResult.from_dict(result_dict)
            return None
        except Exception as e:
            logger.warning(f"Cache retrieval failed, continuing: {e}")
            self.degraded = True
            return None

    async def set_result(
        self,
        probe_text: str,
        language: str,
        endpoint: str,
        result: ProbeResult,
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache a probe result."""
        if self.degraded:
            return False

        try:
            key = self._make_key(probe_text, language, endpoint)
            ttl_seconds = ttl or self.ttl_seconds
            data = json.dumps(result.to_dict())
            self.redis_client.setex(key, ttl_seconds, data)
            logger.debug(f"Cached probe result: {key}")
            return True
        except Exception as e:
            logger.warning(f"Cache write failed, continuing: {e}")
            self.degraded = True
            return False

    async def clear(self) -> bool:
        """Clear all cache entries."""
        if self.degraded:
            return False

        try:
            self.redis_client.delete(
                *self.redis_client.keys("shield:probe:*")
            )
            logger.info("Cache cleared")
            return True
        except Exception as e:
            logger.warning(f"Cache clear failed: {e}")
            return False

    def is_available(self) -> bool:
        """Check if cache is available."""
        return not self.degraded

    async def close(self):
        """Close cache connection."""
        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception as e:
                logger.warning(f"Error closing cache: {e}")
