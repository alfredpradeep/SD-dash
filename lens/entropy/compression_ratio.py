"""
Gap 8 — Kolmogorov Complexity Approximation via Compression Ratio.

Compresses raw text with multiple algorithms to estimate true information
content independent of any tokenizer. The ratio compressed_bytes/token_count
reveals objective tokenizer waste.
"""

import zlib
import lzma
from dataclasses import dataclass, field
from typing import List, Optional
from loguru import logger

try:
    import brotli
    BROTLI_AVAILABLE = True
except ImportError:
    BROTLI_AVAILABLE = False
    logger.debug("brotli not available — using zlib + lzma only")


@dataclass
class CompressionProfile:
    """Result of multi-algorithm compression analysis."""
    raw_bytes: int
    zlib_bytes: int
    lzma_bytes: int
    brotli_bytes: Optional[int]
    best_compressed_bytes: int
    best_algorithm: str
    token_count: int
    # Derived metrics
    compression_ratio: float          # raw / best_compressed
    information_density: float        # best_compressed / raw (0-1, higher = more info)
    tokenizer_inflation: float        # token_bytes / best_compressed (>1 = waste)
    objective_waste_pct: float        # % of tokens that carry no info
    kolmogorov_estimate_bits: float   # best_compressed * 8


class CompressionAnalyzer:
    """
    Estimates true information content using algorithmic compression.

    Theory: Kolmogorov complexity K(x) is the length of the shortest program
    that produces x. While K(x) is uncomputable, compression algorithms provide
    an upper bound. The ratio between compressed size and tokenizer output
    reveals how much waste the tokenizer introduces.
    """

    AVG_BYTES_PER_TOKEN = {
        "gpt-4o": 3.8, "gpt-4o-mini": 3.8, "gpt-3.5-turbo": 3.5,
        "claude-opus-4-6": 4.0, "claude-sonnet-4-6": 4.0,
        "claude-3-5-sonnet": 4.0, "claude-3-haiku": 4.0,
        "gemini-1.5-pro": 3.9, "llama-3-70b": 3.6, "llama-3-8b": 3.6,
        "mistral-7b": 3.6,
    }

    def analyze(self, text: str, token_count: int, model_name: str = "gpt-4o") -> CompressionProfile:
        """Run multi-algorithm compression and compute objective waste metrics."""
        raw = text.encode("utf-8")
        raw_bytes = len(raw)

        # Compress with multiple algorithms
        zlib_compressed = zlib.compress(raw, level=9)
        zlib_bytes = len(zlib_compressed)

        lzma_compressed = lzma.compress(raw, preset=9)
        lzma_bytes = len(lzma_compressed)

        brotli_bytes = None
        if BROTLI_AVAILABLE:
            try:
                brotli_compressed = brotli.compress(raw, quality=11)
                brotli_bytes = len(brotli_compressed)
            except Exception:
                pass

        # Find best compression
        candidates = [("zlib", zlib_bytes), ("lzma", lzma_bytes)]
        if brotli_bytes is not None:
            candidates.append(("brotli", brotli_bytes))

        best_algo, best_bytes = min(candidates, key=lambda x: x[1])

        # Compute derived metrics
        avg_token_bytes = self.AVG_BYTES_PER_TOKEN.get(model_name, 3.8)
        estimated_token_bytes = token_count * avg_token_bytes

        compression_ratio = raw_bytes / best_bytes if best_bytes > 0 else 1.0
        information_density = best_bytes / raw_bytes if raw_bytes > 0 else 0.0
        tokenizer_inflation = estimated_token_bytes / best_bytes if best_bytes > 0 else 1.0

        # Objective waste: what % of token representation is beyond true information
        objective_waste_pct = max(0.0, (1.0 - best_bytes / estimated_token_bytes) * 100) if estimated_token_bytes > 0 else 0.0

        return CompressionProfile(
            raw_bytes=raw_bytes,
            zlib_bytes=zlib_bytes,
            lzma_bytes=lzma_bytes,
            brotli_bytes=brotli_bytes,
            best_compressed_bytes=best_bytes,
            best_algorithm=best_algo,
            token_count=token_count,
            compression_ratio=round(compression_ratio, 3),
            information_density=round(information_density, 4),
            tokenizer_inflation=round(tokenizer_inflation, 3),
            objective_waste_pct=round(objective_waste_pct, 2),
            kolmogorov_estimate_bits=round(best_bytes * 8, 1),
        )
