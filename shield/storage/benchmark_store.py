"""
BenchmarkStore: Persistent benchmark and baseline storage.

Stores anonymized results and provides temporal trend analysis.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from loguru import logger


class BenchmarkStore:
    """
    Benchmark storage with optional Redis persistence.

    Stores aggregate baselines and temporal trends.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 1,
    ):
        """Initialize benchmark store."""
        self.redis_client = None
        self.degraded = False
        self.in_memory: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

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
                logger.info("Benchmark store initialized with Redis")
            except Exception as e:
                logger.warning(
                    f"Redis connection failed, using in-memory storage: {e}"
                )
                self.degraded = True
        else:
            logger.warning("Redis not available, using in-memory storage")
            self.degraded = True

    async def store_scan_result(
        self,
        scan_id: str,
        model: str,
        language: str,
        category: str,
        results: Dict[str, Any],
    ) -> bool:
        """
        Store anonymized scan results for benchmarking.

        Results should be anonymized to protect user privacy.
        """
        try:
            result_entry = {
                "scan_id": scan_id,
                "model": model,
                "language": language,
                "category": category,
                "timestamp": datetime.utcnow().isoformat(),
                "bypass_rate": results.get("bypass_rate", 0),
                "average_hasd": results.get("average_hasd", 0),
                "harm_vector_magnitude": results.get("harm_vector_magnitude", 0),
            }

            # Store in-memory
            key = f"{model}:{language}:{category}"
            self.in_memory[key].append(result_entry)

            # Store in Redis if available
            if self.redis_client:
                self.redis_client.rpush(
                    f"shield:benchmarks:{key}",
                    json.dumps(result_entry),
                )

            logger.debug(f"Stored benchmark result: {scan_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to store benchmark result: {e}")
            return False

    async def get_baselines(
        self,
        model: str,
        language: str,
        category: str,
        window_days: int = 90,
    ) -> Dict[str, Any]:
        """Get aggregate baselines for a model/language/category."""
        try:
            key = f"{model}:{language}:{category}"
            cutoff = datetime.utcnow() - timedelta(days=window_days)

            # Collect results from both sources
            results = []
            if self.redis_client:
                try:
                    redis_results = self.redis_client.lrange(
                        f"shield:benchmarks:{key}",
                        0,
                        -1,
                    )
                    for r in redis_results:
                        results.append(json.loads(r))
                except Exception as e:
                    logger.warning(f"Failed to retrieve Redis results: {e}")

            # Add in-memory results
            results.extend(self.in_memory.get(key, []))

            # Filter by time window
            recent = [
                r for r in results
                if datetime.fromisoformat(r["timestamp"]) > cutoff
            ]

            if not recent:
                return {
                    "model": model,
                    "language": language,
                    "category": category,
                    "sample_count": 0,
                    "data_available": False,
                }

            # Compute statistics
            bypass_rates = [r["bypass_rate"] for r in recent]
            hasd_scores = [r["average_hasd"] for r in recent]
            harm_magnitudes = [r["harm_vector_magnitude"] for r in recent]

            return {
                "model": model,
                "language": language,
                "category": category,
                "sample_count": len(recent),
                "data_available": True,
                "bypass_rate": {
                    "mean": sum(bypass_rates) / len(bypass_rates),
                    "min": min(bypass_rates),
                    "max": max(bypass_rates),
                },
                "hasd_score": {
                    "mean": sum(hasd_scores) / len(hasd_scores),
                    "min": min(hasd_scores),
                    "max": max(hasd_scores),
                },
                "harm_magnitude": {
                    "mean": sum(harm_magnitudes) / len(harm_magnitudes),
                    "min": min(harm_magnitudes),
                    "max": max(harm_magnitudes),
                },
            }
        except Exception as e:
            logger.warning(f"Failed to get baselines: {e}")
            return {"data_available": False, "error": str(e)}

    async def get_temporal_trend(
        self,
        model: str,
        language: str,
        category: str,
        window_days: int = 90,
    ) -> Dict[str, Any]:
        """Get temporal trend of metrics over time."""
        try:
            key = f"{model}:{language}:{category}"
            cutoff = datetime.utcnow() - timedelta(days=window_days)

            # Collect results
            results = []
            if self.redis_client:
                try:
                    redis_results = self.redis_client.lrange(
                        f"shield:benchmarks:{key}",
                        0,
                        -1,
                    )
                    for r in redis_results:
                        results.append(json.loads(r))
                except Exception:
                    pass

            results.extend(self.in_memory.get(key, []))

            # Filter by time window
            recent = [
                r for r in results
                if datetime.fromisoformat(r["timestamp"]) > cutoff
            ]

            if not recent:
                return {
                    "model": model,
                    "language": language,
                    "category": category,
                    "data_available": False,
                }

            # Sort by timestamp
            recent.sort(key=lambda r: r["timestamp"])

            # Group by day
            daily_stats = defaultdict(list)
            for r in recent:
                date = r["timestamp"].split("T")[0]
                daily_stats[date].append(r)

            # Compute daily aggregates
            trend_data = []
            for date in sorted(daily_stats.keys()):
                day_results = daily_stats[date]
                trend_data.append({
                    "date": date,
                    "sample_count": len(day_results),
                    "bypass_rate_mean": sum(
                        r["bypass_rate"] for r in day_results
                    ) / len(day_results),
                    "hasd_score_mean": sum(
                        r["average_hasd"] for r in day_results
                    ) / len(day_results),
                })

            return {
                "model": model,
                "language": language,
                "category": category,
                "window_days": window_days,
                "data_available": True,
                "trend": trend_data,
            }
        except Exception as e:
            logger.warning(f"Failed to get temporal trend: {e}")
            return {"data_available": False, "error": str(e)}

    async def close(self):
        """Close storage connection."""
        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception as e:
                logger.warning(f"Error closing benchmark store: {e}")
