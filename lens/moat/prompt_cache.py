import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict


@dataclass
class CacheEntry:
    prompt_hash: str           # SHA-256 of exact text
    semantic_hash: str         # Locality-sensitive hash for similarity
    prompt_text: str
    response_text: str
    model: str
    tokens_saved: int          # If this is a cache hit
    cost_saved: float
    created_at: datetime
    last_hit_at: datetime
    hit_count: int
    similarity_score: float    # How similar was the query to this entry


@dataclass
class CacheResult:
    hit: bool
    entry: Optional[CacheEntry]
    similarity: float          # 0-1
    tokens_saved: int
    cost_saved: float
    method: str                # "exact_match", "semantic_match", "miss"


@dataclass
class CacheStats:
    total_queries: int
    exact_hits: int
    semantic_hits: int
    misses: int
    hit_rate: float
    total_tokens_saved: int
    total_cost_saved: float
    cache_size: int
    avg_similarity_on_hit: float


class SemanticPromptCache:
    """
    Semantic caching for AI prompts.

    Level 1: Exact match (SHA-256 hash) — instant, 100% confidence
    Level 2: Semantic match (TF-IDF cosine similarity > threshold) — fast, configurable confidence
    Level 3: Miss — no match found

    Designed for single-provider use: cache responses from GPT-4o,
    detect when the same question is asked 500 times a day in interviews.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_cache_size: int = 10000,
        ttl_hours: int = 24,
    ):
        self._exact_cache: dict[str, CacheEntry] = {}
        self._entries: list[CacheEntry] = []
        self._similarity_threshold = similarity_threshold
        self._max_size = max_cache_size
        self._ttl_hours = ttl_hours
        self._stats = CacheStats(
            total_queries=0,
            exact_hits=0,
            semantic_hits=0,
            misses=0,
            hit_rate=0.0,
            total_tokens_saved=0,
            total_cost_saved=0.0,
            cache_size=0,
            avg_similarity_on_hit=0.0,
        )
        self._vectorizer = None
        self._tfidf_matrix = None
        self._similarity_scores_on_hit = []

        # Try to import sklearn
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self._sklearn_available = True
            self._TfidfVectorizer = TfidfVectorizer
            self._cosine_similarity = cosine_similarity
        except ImportError:
            self._sklearn_available = False

    def lookup(self, prompt: str, model: str) -> CacheResult:
        """Look up a prompt in the cache."""
        self._stats.total_queries += 1

        # Evict expired entries
        self._evict_expired()

        # 1. Try exact hash match
        exact_hash = self._exact_hash(prompt)
        if exact_hash in self._exact_cache:
            entry = self._exact_cache[exact_hash]
            # Update hit metadata
            entry.last_hit_at = datetime.now()
            entry.hit_count += 1

            result = CacheResult(
                hit=True,
                entry=entry,
                similarity=1.0,
                tokens_saved=entry.tokens_saved,
                cost_saved=entry.cost_saved,
                method="exact_match",
            )
            self._stats.exact_hits += 1
            self._similarity_scores_on_hit.append(1.0)
            self._stats.total_tokens_saved += entry.tokens_saved
            self._stats.total_cost_saved += entry.cost_saved
            self._update_stats()
            return result

        # 2. Try semantic similarity match
        if self._entries:
            best_similarity = 0.0
            best_entry = None

            for entry in self._entries:
                if entry.model != model:
                    continue

                similarity = self._compute_similarity(prompt, entry.prompt_text)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_entry = entry

            if best_similarity >= self._similarity_threshold and best_entry:
                best_entry.last_hit_at = datetime.now()
                best_entry.hit_count += 1

                result = CacheResult(
                    hit=True,
                    entry=best_entry,
                    similarity=best_similarity,
                    tokens_saved=int(best_entry.tokens_saved * best_similarity),
                    cost_saved=best_entry.cost_saved * best_similarity,
                    method="semantic_match",
                )
                self._stats.semantic_hits += 1
                self._similarity_scores_on_hit.append(best_similarity)
                self._stats.total_tokens_saved += result.tokens_saved
                self._stats.total_cost_saved += result.cost_saved
                self._update_stats()
                return result

        # 3. Miss
        self._stats.misses += 1
        self._update_stats()
        return CacheResult(
            hit=False,
            entry=None,
            similarity=0.0,
            tokens_saved=0,
            cost_saved=0.0,
            method="miss",
        )

    def store(self, prompt: str, response: str, model: str, tokens_used: int, cost: float):
        """Store a prompt-response pair."""
        # Check if exact match already exists
        exact_hash = self._exact_hash(prompt)
        if exact_hash in self._exact_cache:
            # Just update hit count
            self._exact_cache[exact_hash].hit_count += 1
            return

        # Evict if needed
        if len(self._entries) >= self._max_size:
            self._evict_lru()

        semantic_hash = self._semantic_hash(prompt)
        now = datetime.now()

        entry = CacheEntry(
            prompt_hash=exact_hash,
            semantic_hash=semantic_hash,
            prompt_text=prompt,
            response_text=response,
            model=model,
            tokens_saved=tokens_used,
            cost_saved=cost,
            created_at=now,
            last_hit_at=now,
            hit_count=0,
            similarity_score=1.0,
        )

        self._entries.append(entry)
        self._exact_cache[exact_hash] = entry
        self._stats.cache_size = len(self._entries)

    def get_stats(self) -> CacheStats:
        """Return cache performance statistics."""
        return self._stats

    def get_top_duplicates(self, n: int = 10) -> list[dict]:
        """Return most frequently hit cache entries (biggest waste sources)."""
        # Sort by hit_count * tokens_saved
        ranked = sorted(
            self._entries,
            key=lambda e: e.hit_count * e.tokens_saved,
            reverse=True,
        )

        result = []
        for i, entry in enumerate(ranked[:n]):
            result.append(
                {
                    "rank": i + 1,
                    "prompt_preview": entry.prompt_text[:100] + ("..." if len(entry.prompt_text) > 100 else ""),
                    "hit_count": entry.hit_count,
                    "tokens_per_hit": entry.tokens_saved,
                    "cost_per_hit": entry.cost_saved,
                    "total_tokens_wasted": entry.hit_count * entry.tokens_saved,
                    "total_cost_wasted": entry.hit_count * entry.cost_saved,
                    "model": entry.model,
                }
            )

        return result

    def _exact_hash(self, text: str) -> str:
        """SHA-256 hash of exact text."""
        return hashlib.sha256(text.encode()).hexdigest()

    def _semantic_hash(self, text: str) -> str:
        """Locality-sensitive hash for semantic similarity."""
        # Simple approach: lowercase, split, and hash significant words
        words = set(text.lower().split())
        # Remove common stop words
        stop = {"the", "a", "an", "and", "or", "is", "are", "was", "were", "be", "been", "of", "to", "in"}
        words = words - stop
        # Sort and hash
        sorted_words = sorted(list(words))
        combined = "|".join(sorted_words)
        return hashlib.sha256(combined.encode()).hexdigest()

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """TF-IDF cosine similarity between two texts."""
        if self._sklearn_available:
            return self._compute_tfidf_similarity(text1, text2)
        else:
            return self._compute_jaccard_similarity(text1, text2)

    def _compute_tfidf_similarity(self, text1: str, text2: str) -> float:
        """TF-IDF based similarity using sklearn."""
        try:
            vectorizer = self._TfidfVectorizer(lowercase=True, stop_words="english", max_features=100)
            vectors = vectorizer.fit_transform([text1, text2])
            similarity = self._cosine_similarity(vectors[0], vectors[1])[0][0]
            return float(similarity)
        except Exception:
            # Fallback to Jaccard
            return self._compute_jaccard_similarity(text1, text2)

    def _compute_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Jaccard similarity (word overlap) as fallback."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _evict_expired(self):
        """Remove entries older than TTL."""
        now = datetime.now()
        cutoff = now - timedelta(hours=self._ttl_hours)

        expired = [e for e in self._entries if e.created_at < cutoff]

        for entry in expired:
            self._entries.remove(entry)
            if entry.prompt_hash in self._exact_cache:
                del self._exact_cache[entry.prompt_hash]

        self._stats.cache_size = len(self._entries)

    def _evict_lru(self):
        """Remove least recently used entries when cache is full."""
        if not self._entries:
            return

        # Find LRU entry (by last_hit_at, or created_at if never hit)
        lru_entry = min(self._entries, key=lambda e: e.last_hit_at)

        self._entries.remove(lru_entry)
        if lru_entry.prompt_hash in self._exact_cache:
            del self._exact_cache[lru_entry.prompt_hash]

        self._stats.cache_size = len(self._entries)

    def _update_stats(self):
        """Update aggregate statistics."""
        if self._stats.total_queries > 0:
            self._stats.hit_rate = (
                (self._stats.exact_hits + self._stats.semantic_hits)
                / self._stats.total_queries
            )

        if self._similarity_scores_on_hit:
            self._stats.avg_similarity_on_hit = sum(self._similarity_scores_on_hit) / len(
                self._similarity_scores_on_hit
            )
