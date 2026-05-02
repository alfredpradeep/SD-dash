"""
Tests for SCLEngine — the core compression orchestrator.

Covers:
  - Supported language/tokenizer validation
  - Cache hit path
  - Full pipeline (extract → search → verify)
  - No-candidate fallback
  - Batch compression
  - Health check
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from compress.engine import SCLEngine
from compress.lattice.structures import CompressionResult
from compress.exceptions import UnsupportedLanguageError


@pytest.mark.asyncio
class TestSCLEngineCompress:
    """Tests for SCLEngine.compress()."""

    async def test_compress_returns_result(self, mock_engine, sample_candidate):
        """Full pipeline returns a valid CompressionResult."""
        result = await mock_engine.compress(
            text="The ML model achieved great results.",
            source_language="en",
            target_tokenizer="gpt-4o",
        )
        assert isinstance(result, CompressionResult)
        assert result.compression_applied is True
        assert result.compressed_text == sample_candidate.text
        assert result.processing_ms > 0

    async def test_compress_unsupported_language_raises(self, mock_engine):
        """Unsupported language code raises UnsupportedLanguageError."""
        with pytest.raises(UnsupportedLanguageError, match="not supported"):
            await mock_engine.compress(
                text="Test", source_language="xx", target_tokenizer="gpt-4o"
            )

    async def test_compress_unsupported_tokenizer_raises(self, mock_engine):
        """Unsupported tokenizer raises UnsupportedLanguageError."""
        with pytest.raises(UnsupportedLanguageError, match="not supported"):
            await mock_engine.compress(
                text="Test", source_language="en", target_tokenizer="fake-model"
            )

    async def test_compress_cache_hit(self, mock_engine, sample_result):
        """Cached result is returned without running pipeline."""
        mock_engine.cache.get = AsyncMock(return_value=sample_result)

        result = await mock_engine.compress(
            text="Cached text", source_language="en", target_tokenizer="gpt-4o"
        )

        assert result is sample_result
        mock_engine.metrics.record_cache_hit.assert_called_once_with("en")
        mock_engine.extractor.extract.assert_not_awaited()

    async def test_compress_no_candidate_passes(self, mock_engine):
        """When no candidate passes gate, original text is returned."""
        mock_engine.gate.verify = AsyncMock(
            return_value=(False, "Below threshold")
        )

        result = await mock_engine.compress(
            text="Original text that cannot be compressed well.",
            source_language="en",
            target_tokenizer="gpt-4o",
        )

        assert result.compression_applied is False
        assert result.compressed_text == result.original_text
        assert result.reduction_ratio == 0.0
        assert "No candidate" in result.rejection_reason

    async def test_compress_records_metrics(self, mock_engine):
        """Successful compression records metrics."""
        await mock_engine.compress(
            text="Record this.", source_language="en", target_tokenizer="gpt-4o"
        )
        mock_engine.metrics.record_compression.assert_called_once()

    async def test_compress_sets_cache(self, mock_engine):
        """Result is stored in cache after compression."""
        await mock_engine.compress(
            text="Cache this.", source_language="en", target_tokenizer="gpt-4o"
        )
        mock_engine.cache.set.assert_awaited_once()


@pytest.mark.asyncio
class TestSCLEngineBatch:
    """Tests for SCLEngine.batch_compress()."""

    async def test_batch_compress_multiple(self, mock_engine):
        """Batch compresses multiple items concurrently."""
        items = [
            {"text": "First text.", "language": "en", "tokenizer": "gpt-4o"},
            {"text": "Second text.", "language": "en", "tokenizer": "gpt-4o"},
            {"text": "Third text.", "language": "ta", "tokenizer": "gpt-4o"},
        ]
        results = await mock_engine.batch_compress(texts=items)
        assert len(results) == 3
        assert all(isinstance(r, CompressionResult) for r in results)

    async def test_batch_compress_empty(self, mock_engine):
        """Empty batch returns empty list."""
        results = await mock_engine.batch_compress(texts=[])
        assert results == []


@pytest.mark.asyncio
class TestSCLEngineHealth:
    """Tests for SCLEngine.health_check()."""

    async def test_health_check_all_healthy(self, mock_engine):
        """Health check aggregates sub-component statuses."""
        status = await mock_engine.health_check()
        assert status["status"] == "healthy"
        assert status["extractor"] == "healthy"
        assert status["searcher"] == "healthy"
        assert status["gate"] == "healthy"
        assert status["cache"] == "healthy"


class TestSCLEngineConstants:
    """Tests for SCLEngine class-level constants."""

    def test_supported_languages_count(self):
        assert len(SCLEngine.SUPPORTED_LANGUAGES) == 20

    def test_supported_tokenizers_count(self):
        assert len(SCLEngine.SUPPORTED_TOKENIZERS) == 10

    def test_all_expected_languages_present(self, expected_outputs):
        for lang in expected_outputs["supported_languages"]:
            assert lang in SCLEngine.SUPPORTED_LANGUAGES

    def test_all_expected_tokenizers_present(self, expected_outputs):
        for tok in expected_outputs["supported_tokenizers"]:
            assert tok in SCLEngine.SUPPORTED_TOKENIZERS
