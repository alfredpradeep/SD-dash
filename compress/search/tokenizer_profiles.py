"""
Unified token counting across all supported tokenizer families.

Supports three backends:
- cl100k_base via tiktoken (OpenAI: GPT-4o, GPT-4o-mini, GPT-3.5-turbo)
- Anthropic tokenizer (Claude models)
- SentencePiece (Llama, Mistral)

Each backend is loaded lazily and cached for reuse.
"""

import tiktoken
from loguru import logger


class TokenizerProfiles:
    """
    Unified token counting across all supported tokenizer families.
    """

    PROFILES: dict[str, dict] = {
        "gpt-4o": {"backend": "tiktoken", "encoding": "cl100k_base"},
        "gpt-4o-mini": {"backend": "tiktoken", "encoding": "cl100k_base"},
        "gpt-3.5-turbo": {"backend": "tiktoken", "encoding": "cl100k_base"},
        "claude-3-5-sonnet": {
            "backend": "anthropic", "model": "claude-3-5-sonnet",
        },
        "claude-3-haiku": {
            "backend": "anthropic", "model": "claude-3-haiku",
        },
        "claude-3-opus": {
            "backend": "anthropic", "model": "claude-3-opus",
        },
        "llama-3-8b": {
            "backend": "sentencepiece",
            "model_path": "models/llama3-tokenizer.model",
        },
        "llama-3-70b": {
            "backend": "sentencepiece",
            "model_path": "models/llama3-tokenizer.model",
        },
        "mistral-7b": {
            "backend": "sentencepiece",
            "model_path": "models/mistral-tokenizer.model",
        },
        "mistral-8x7b": {
            "backend": "sentencepiece",
            "model_path": "models/mistral-tokenizer.model",
        },
    }

    def __init__(self):
        self._cache: dict = {}

    def count_tokens(self, text: str, tokenizer_name: str) -> int:
        """Count tokens using the appropriate backend for the given model."""
        profile = self.PROFILES.get(tokenizer_name)
        if not profile:
            logger.warning(
                "Unknown tokenizer '{}', falling back to cl100k_base",
                tokenizer_name,
            )
            profile = {"backend": "tiktoken", "encoding": "cl100k_base"}

        backend = profile["backend"]

        if backend == "tiktoken":
            return self._count_tiktoken(text, profile["encoding"])
        elif backend == "anthropic":
            return self._count_anthropic(text, profile["model"])
        elif backend == "sentencepiece":
            return self._count_sentencepiece(text, profile["model_path"])
        else:
            raise ValueError(f"Unknown tokenizer backend: {backend}")

    def _count_tiktoken(self, text: str, encoding_name: str) -> int:
        """Count tokens using OpenAI's tiktoken library."""
        if encoding_name not in self._cache:
            self._cache[encoding_name] = tiktoken.get_encoding(encoding_name)
        return len(self._cache[encoding_name].encode(text))

    def _count_anthropic(self, text: str, model: str) -> int:
        """Count tokens using Anthropic's tokenizer."""
        cache_key = f"anthropic:{model}"
        if cache_key not in self._cache:
            try:
                from anthropic import Anthropic
                client = Anthropic()
                self._cache[cache_key] = client
            except ImportError:
                logger.warning(
                    "anthropic package not installed, "
                    "falling back to cl100k approximation"
                )
                return self._count_tiktoken(text, "cl100k_base")
        try:
            client = self._cache[cache_key]
            result = client.count_tokens(text)
            return result
        except Exception as e:
            logger.warning(
                "Anthropic token count failed ({}), using cl100k fallback", e
            )
            return self._count_tiktoken(text, "cl100k_base")

    def _count_sentencepiece(self, text: str, model_path: str) -> int:
        """Count tokens using SentencePiece model."""
        if model_path not in self._cache:
            try:
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor()
                sp.Load(model_path)
                self._cache[model_path] = sp
            except (ImportError, OSError) as e:
                logger.warning(
                    "SentencePiece model '{}' unavailable ({}), "
                    "using cl100k fallback",
                    model_path,
                    e,
                )
                return self._count_tiktoken(text, "cl100k_base")
        sp = self._cache[model_path]
        return len(sp.EncodeAsIds(text))
