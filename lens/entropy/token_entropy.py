"""
Per-token Shannon entropy (surprisal) calculator.

Gap 1 Fix: Uses actual LM log-probabilities via a lightweight local model
(distilgpt2) instead of the broken Zipf rank approximation.

Surprisal of token t_i given context t_1...t_{i-1}:
    surprisal(t_i | context) = -log2(P(t_i | t_1...t_{i-1}))

For non-English scripts where distilgpt2 has limited coverage, we fall back to
character-level bigram entropy computed from the actual text — a much better
approximation than Zipf for fragmented scripts.
"""

import numpy as np
import asyncio
from loguru import logger
from lens.entropy.structures import TokenEntropyRecord
from lens.config import Config

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available — token counting will use character estimate")

try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch/transformers not available — falling back to character entropy")


class TokenEntropyCalculator:
    """
    Computes per-token Shannon entropy (surprisal) for input text.

    Strategy:
      - English / Latin scripts: distilgpt2 actual log-probabilities (Gap 1 fix)
      - Non-Latin scripts: character-level bigram entropy (far better than Zipf)
      - Hybrid: for code-switched text, splits by script and applies per-segment strategy
    """

    # HIGH_FREQ_RANK_THRESHOLD kept for legacy compatibility
    HIGH_FREQ_RANK_THRESHOLD = 1000

    TOKENIZER_ENCODING_MAP = {
        "gpt-4o":            "cl100k_base",
        "gpt-4o-mini":       "cl100k_base",
        "gpt-3.5-turbo":     "cl100k_base",
        "claude-3-5-sonnet": "cl100k_base",
        "claude-3-haiku":    "cl100k_base",
        "llama-3-8b":        "cl100k_base",
        "llama-3-70b":       "cl100k_base",
        "mistral-7b":        "cl100k_base",
    }

    # Scripts that benefit from character bigram entropy rather than LM probs
    NON_LATIN_LANGS = {"ta", "ml", "te", "kn", "hi", "bn", "mr", "gu", "pa", "ur",
                       "ar", "ja", "zh", "ko"}

    # Language-specific entropy adjustment multipliers (still used as scaling, not approximation)
    LANG_SCALE = {
        "ta": 1.45, "ml": 1.45, "te": 1.40, "kn": 1.40,
        "hi": 1.30, "bn": 1.30, "mr": 1.30, "gu": 1.30,
        "pa": 1.25, "ur": 1.25,
        "ar": 1.35,
        "ja": 1.20, "zh": 1.15, "ko": 1.20,
        "en": 1.00, "es": 1.02, "fr": 1.02,
        "de": 1.05, "pt": 1.02,
    }

    def __init__(self, config: Config):
        self.config = config
        self._encoders: dict = {}
        self._lm_model = None
        self._lm_tokenizer = None
        self._lm_loaded = False
        self._bigram_cache: dict = {}

    def _load_lm(self):
        """Lazy-load distilgpt2 for actual log-probability computation."""
        if self._lm_loaded:
            return self._lm_model is not None
        self._lm_loaded = True
        if not TORCH_AVAILABLE:
            logger.info("Torch not available — using character bigram entropy for all languages")
            return False
        try:
            logger.info("Loading distilgpt2 for actual surprisal computation...")
            self._lm_tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
            self._lm_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
            self._lm_model.eval()
            if self.config.device != "cpu":
                self._lm_model = self._lm_model.to(self.config.device)
            logger.info("distilgpt2 loaded successfully")
            return True
        except Exception as e:
            logger.warning("Failed to load distilgpt2: {}. Using character bigram fallback.", e)
            return False

    def _get_encoder(self, enc_name: str):
        if not TIKTOKEN_AVAILABLE:
            return None
        if enc_name not in self._encoders:
            try:
                import tiktoken
                self._encoders[enc_name] = tiktoken.get_encoding(enc_name)
            except Exception as e:
                logger.warning("tiktoken encoding '{}' unavailable: {} — using character fallback", enc_name, e)
                self._encoders[enc_name] = None
        return self._encoders[enc_name]

    def _character_bigram_entropy(self, text: str) -> dict[str, float]:
        """
        Character-level bigram entropy for non-Latin script tokens.
        Returns a mapping from character position to estimated surprisal in bits.

        Much more accurate than Zipf for scripts like Tamil, Arabic, Devanagari
        where token IDs have no rank relationship to frequency.
        """
        if not text:
            return {}
        chars = list(text)
        n = len(chars)
        if n < 2:
            return {0: 4.0}

        # Compute character unigram and bigram frequencies
        unigram: dict[str, int] = {}
        bigram: dict[tuple, int] = {}
        for i, c in enumerate(chars):
            unigram[c] = unigram.get(c, 0) + 1
            if i > 0:
                bg = (chars[i-1], c)
                bigram[bg] = bigram.get(bg, 0) + 1

        total_chars = sum(unigram.values())
        total_bigrams = sum(bigram.values())

        # Compute surprisal per character using bigram conditional probability
        # P(c_i | c_{i-1}) = count(c_{i-1}, c_i) / count(c_{i-1})
        surprisals: dict[int, float] = {}
        for i, c in enumerate(chars):
            if i == 0:
                p_unigram = unigram.get(c, 1) / total_chars
                surprisals[i] = float(-np.log2(max(p_unigram, 1e-10)))
            else:
                bg = (chars[i-1], c)
                p_bg = bigram.get(bg, 0)
                p_prev = unigram.get(chars[i-1], 1)
                p_cond = (p_bg + 1) / (p_prev + len(unigram))  # Laplace smoothing
                surprisals[i] = float(-np.log2(max(p_cond, 1e-10)))
        return surprisals

    def _lm_surprisal(self, text: str) -> list[float]:
        """
        Compute actual token surprisal using distilgpt2.
        Returns list of surprisal values (bits) per token.
        Gap 1 fix — replaces Zipf approximation.
        """
        if self._lm_model is None or self._lm_tokenizer is None:
            return []
        try:
            inputs = self._lm_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs["input_ids"]
            with torch.no_grad():
                outputs = self._lm_model(input_ids, labels=input_ids)
                logits = outputs.logits  # [1, seq_len, vocab_size]

            # Shift to get P(t_i | t_1..t_{i-1})
            shift_logits = logits[0, :-1, :]
            shift_labels = input_ids[0, 1:]

            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs[range(len(shift_labels)), shift_labels]
            # Convert nats to bits
            surprisals = (-token_log_probs / np.log(2)).tolist()
            # Prepend a small value for the first token (no context)
            first_token_surprisal = float(-np.log2(1.0 / self._lm_model.config.vocab_size))
            return [first_token_surprisal] + surprisals
        except Exception as e:
            logger.debug("LM surprisal failed: {}", e)
            return []

    def _classify_token(self, token_text: str, language: str) -> str:
        """
        Classify a token by type for Gap 5 (cause attribution).
        Returns: "fragment" | "entity" | "codesw" | "oov" | "normal"
        """
        if not token_text.strip():
            return "normal"
        # Fragment: single character from non-Latin script
        if len(token_text.strip()) <= 2 and language in self.NON_LATIN_LANGS:
            return "fragment"
        # Code-switching: Latin characters appearing in non-Latin text
        if language in self.NON_LATIN_LANGS:
            latin_chars = sum(1 for c in token_text if c.isascii() and c.isalpha())
            if latin_chars > len(token_text) * 0.5:
                return "codesw"
        # Named entity heuristic: title case, all-caps in context
        if token_text.istitle() or token_text.isupper():
            return "entity"
        return "normal"

    async def decompose(
        self,
        text: str,
        model_name: str,
        language: str,
    ) -> list[TokenEntropyRecord]:
        """
        Decompose text into per-token entropy records.

        Strategy selection:
          - Latin languages: attempt distilgpt2 LM surprisal (Gap 1 fix)
          - Non-Latin: character bigram entropy per token
          - Fallback: Zipf approximation (legacy, only if all else fails)
        """
        enc_name = self.TOKENIZER_ENCODING_MAP.get(model_name, "cl100k_base")
        encoder = self._get_encoder(enc_name)
        lang_scale = self.LANG_SCALE.get(language, 1.10)

        if encoder is None:
            # No tiktoken — estimate from text length
            return self._estimate_without_tokenizer(text, language, lang_scale)

        token_ids = encoder.encode(text)
        n_tokens = len(token_ids)

        # Decode all tokens up front
        token_texts = []
        token_bytes_list = []
        for tid in token_ids:
            try:
                tb = encoder.decode_single_token_bytes(tid)
                tt = tb.decode("utf-8", errors="replace")
            except Exception:
                tt = f"<tok_{tid}>"
                tb = b""
            token_texts.append(tt)
            token_bytes_list.append(tb)

        # --- Surprisal strategy selection ---
        surprisals: list[float] = []

        use_lm = language not in self.NON_LATIN_LANGS and self._load_lm()
        if use_lm:
            surprisals = self._lm_surprisal(text)

        if len(surprisals) != n_tokens:
            # Fallback: character bigram entropy per token
            char_surprisals = self._character_bigram_entropy(text)
            surprisals = []
            char_pos = 0
            for tt in token_texts:
                token_char_surprisals = [
                    char_surprisals.get(char_pos + i, 4.0)
                    for i in range(max(len(tt), 1))
                ]
                surprisals.append(float(np.mean(token_char_surprisals)) * lang_scale)
                char_pos += len(tt)

        records = []
        threshold = self.config.high_entropy_bit_threshold
        for position, (tid, tt, tb, surp) in enumerate(
            zip(token_ids, token_texts, token_bytes_list, surprisals)
        ):
            adjusted = surp * lang_scale if language not in self.NON_LATIN_LANGS else surp
            records.append(TokenEntropyRecord(
                token_id=tid,
                token_text=tt,
                token_bytes=tb,
                character_count=len(tt),
                surprisal_bits=adjusted,
                is_high_entropy=adjusted > threshold,
                position=position,
                token_type=self._classify_token(tt, language),
            ))

        return records

    def _estimate_without_tokenizer(
        self, text: str, language: str, lang_scale: float
    ) -> list[TokenEntropyRecord]:
        """Fallback when tiktoken is not available."""
        words = text.split()
        records = []
        char_surprisals = self._character_bigram_entropy(text)
        threshold = self.config.high_entropy_bit_threshold
        for i, word in enumerate(words):
            surp = float(np.mean([char_surprisals.get(j, 4.0) for j in range(len(word))])) * lang_scale
            records.append(TokenEntropyRecord(
                token_id=i,
                token_text=word,
                token_bytes=word.encode("utf-8", errors="replace"),
                character_count=len(word),
                surprisal_bits=surp,
                is_high_entropy=surp > threshold,
                position=i,
                token_type=self._classify_token(word, language),
            ))
        return records
