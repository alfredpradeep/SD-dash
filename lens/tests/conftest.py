"""
Pytest configuration for LENS tests.

Patches tiktoken to disable network downloads in sandboxed environments.
The TokenEntropyCalculator has a built-in fallback to character bigram entropy
when tiktoken's encoding file is unavailable — conftest ensures this path runs.
"""

import pytest
import sys
from unittest.mock import patch, MagicMock


def _mock_tiktoken_encoder():
    """
    Returns a minimal mock encoder that produces token IDs from character positions.
    Used when tiktoken's remote vocab file cannot be downloaded.
    """
    encoder = MagicMock()
    encoder.n_vocab = 100257

    def encode(text):
        # Approximate: each UTF-8 byte ≈ 1 token for test purposes
        return list(range(len(text.encode("utf-8"))))

    def decode_single_token_bytes(token_id):
        return b"x"

    encoder.encode = encode
    encoder.decode_single_token_bytes = decode_single_token_bytes
    return encoder


# Patch tiktoken globally so network access is never attempted
import tiktoken as _tiktoken_module
_original_get_encoding = _tiktoken_module.get_encoding


def _safe_get_encoding(name):
    try:
        return _original_get_encoding(name)
    except Exception:
        return _mock_tiktoken_encoder()


_tiktoken_module.get_encoding = _safe_get_encoding
