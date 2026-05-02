"""
Decompression Engine — Reconstructs original-language text from
compressed English representation.

The full COMPRESS cycle:
  1. COMPRESS:   Tamil (566 tokens) → English (67 tokens)   [88% savings]
  2. STORE/SEND: English (67 tokens)                        [cheap API calls]
  3. DECOMPRESS: English (67 tokens) → Tamil (optimized)    [natural output]

CRITICAL: Decompression requires a multilingual-capable LLM.
  - Groq/Llama 3.1 8B is INSUFFICIENT for non-Latin script output
  - Provider priority for decompression: Gemini > OpenAI > Anthropic > Groq
  - Falls back through providers until one produces valid target-language text
"""

import os
import time
import asyncio
import httpx
from loguru import logger
from compress.config import Config


# Language code → display name mapping
LANGUAGE_NAMES = {
    "ta": "Tamil (தமிழ்)",
    "hi": "Hindi (हिन्दी)",
    "ar": "Arabic (العربية)",
    "ja": "Japanese (日本語)",
    "zh": "Chinese (中文)",
    "ko": "Korean (한국어)",
    "bn": "Bengali (বাংলা)",
    "ur": "Urdu (اردو)",
    "te": "Telugu (తెలుగు)",
    "ml": "Malayalam (മലയാളം)",
    "pa": "Punjabi (ਪੰਜਾਬੀ)",
    "gu": "Gujarati (ગુજરાતી)",
    "mr": "Marathi (मराठी)",
    "pt": "Portuguese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "id": "Indonesian",
    "ms": "Malay",
    "en": "English",
}

LANGUAGE_SHORT = {
    "ta": "Tamil", "hi": "Hindi", "ar": "Arabic", "ja": "Japanese",
    "zh": "Chinese", "ko": "Korean", "bn": "Bengali", "ur": "Urdu",
    "te": "Telugu", "ml": "Malayalam", "pa": "Punjabi", "gu": "Gujarati",
    "mr": "Marathi", "pt": "Portuguese", "es": "Spanish", "fr": "French",
    "de": "German", "id": "Indonesian", "ms": "Malay", "en": "English",
}


def _build_translation_prompt(
    compressed_text: str,
    target_language: str,
    lang_name: str,
    original_text: str = "",
    n: int = 3,
    original_token_count: int = 0,
) -> tuple[str, str]:
    """
    Build high-quality translation prompt with token-efficiency constraints.

    When original_text is provided, uses it as a style/vocabulary reference
    so the LLM can match the original author's tone and word choices.

    CRITICAL: The prompt constrains output length so the reconstructed text
    does not exceed the original token count — ensuring net round-trip savings.
    """
    # Set token budget: aim for ≤90% of original to guarantee net savings
    token_budget_hint = ""
    if original_token_count > 0:
        target_tokens = int(original_token_count * 0.90)
        token_budget_hint = (
            f"\n9. CONCISENESS IS CRITICAL: The reconstruction must be "
            f"shorter or equal to the original text length. "
            f"Target approximately {target_tokens} tokens or fewer. "
            f"Do NOT pad, elaborate, or add filler words. "
            f"Use the most concise natural {lang_name} phrasing that "
            f"preserves full meaning."
        )

    system = (
        f"You are an expert {lang_name} translator and linguist "
        f"specializing in concise, token-efficient translation. "
        f"Your task is to translate compressed English text back into "
        f"natural {lang_name} using the MINIMUM words needed to "
        f"preserve full meaning.\n\n"
        f"ABSOLUTE REQUIREMENTS:\n"
        f"1. Output MUST be entirely in {lang_name} script/language\n"
        f"2. Use proper {lang_name} grammar, sentence structure, "
        f"and natural word order\n"
        f"3. PRESERVE every fact, entity, number, name, and date\n"
        f"4. PRESERVE all negations and conditions\n"
        f"5. Write concise, natural {lang_name} — as a native speaker "
        f"would write, avoiding unnecessary elaboration\n"
        f"6. Do NOT transliterate English words — use proper "
        f"{lang_name} equivalents\n"
        f"7. Do NOT add information not present in the source\n"
        f"8. Return ONLY numbered {lang_name} translations"
        f"{token_budget_hint}"
    )

    # Build user prompt with original as reference
    parts = [
        f"Translate the following compressed English text into {n} "
        f"different {lang_name} versions. Each version must preserve "
        f"ALL meaning while being as CONCISE as possible.\n",
    ]

    if original_text:
        parts.append(
            f"REFERENCE (original {lang_name} text — match this "
            f"style and vocabulary, but aim for equal or shorter "
            f"length):\n{original_text}\n"
        )

    parts.append(f"ENGLISH TEXT TO TRANSLATE:\n{compressed_text}\n")
    parts.append(
        f"Provide exactly {n} {lang_name} translations, "
        f"numbered 1 to {n}.\n"
        f"Version 1: Most concise natural {lang_name} "
        f"(shortest that preserves all meaning)\n"
        f"Version 2: Natural and fluent {lang_name}\n"
        f"Version 3: Formal concise {lang_name}\n\n"
        f"IMPORTANT: Every word of your output must be in "
        f"{lang_name}. Do not include any English. "
        f"Prefer shorter phrasing over verbose alternatives."
    )

    return system, "\n".join(parts)


class TranslationProvider:
    """
    Dedicated LLM provider for translation tasks.

    Prioritizes models with strong multilingual capabilities:
      Gemini (best for non-Latin) > OpenAI > Anthropic > Groq

    This is separate from the compression provider because
    compression needs speed (Groq/Llama) while decompression
    needs multilingual quality (Gemini/GPT-4).
    """

    def __init__(self):
        self._providers = []
        self._setup()

    def _setup(self):
        """Detect available providers, prioritize by translation quality."""

        # 1. Gemini — BEST for multilingual translation
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if gemini_key:
            self._providers.append({
                "name": "gemini",
                "key": gemini_key,
                "model": os.environ.get(
                    "COMPRESS_TRANSLATE_MODEL", "gemini-2.0-flash"
                ),
            })
            logger.info(
                "Translation provider: Gemini available (primary)"
            )

        # 2. OpenAI — good multilingual
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if openai_key:
            self._providers.append({
                "name": "openai",
                "key": openai_key,
                "model": os.environ.get(
                    "COMPRESS_TRANSLATE_MODEL", "gpt-4o-mini"
                ),
            })
            logger.info("Translation provider: OpenAI available")

        # 3. Anthropic — good multilingual
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if anthropic_key:
            self._providers.append({
                "name": "anthropic",
                "key": anthropic_key,
                "model": os.environ.get(
                    "COMPRESS_TRANSLATE_MODEL",
                    "claude-3-haiku-20240307",
                ),
            })
            logger.info("Translation provider: Anthropic available")

        # 4. Groq — LAST resort (Llama 3.1 8B has poor Tamil)
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if groq_key:
            self._providers.append({
                "name": "groq",
                "key": groq_key,
                "model": os.environ.get(
                    "COMPRESS_TRANSLATE_MODEL",
                    "llama-3.3-70b-versatile",  # Updated: 3.1-70b deprecated
                ),
            })
            logger.info(
                "Translation provider: Groq available (fallback)"
            )

        # 5. Local
        local_url = os.environ.get("COMPRESS_LOCAL_LLM_URL", "")
        if local_url:
            self._providers.append({
                "name": "local",
                "url": local_url,
                "model": os.environ.get(
                    "COMPRESS_TRANSLATE_MODEL", "llama3"
                ),
            })

        if not self._providers:
            logger.warning(
                "No translation provider available! "
                "Decompression will not work."
            )

    @property
    def available(self) -> bool:
        return len(self._providers) > 0

    @property
    def provider_name(self) -> str:
        if self._providers:
            return self._providers[0]["name"]
        return "none"

    async def translate(
        self,
        system: str,
        user: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> tuple[str, str]:
        """
        Try each provider in priority order until one succeeds.
        Retries rate-limited providers with exponential backoff.

        Returns (response_text, provider_name).
        """
        for provider in self._providers:
            name = provider["name"]
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    result = await self._call_provider(
                        provider, system, user, max_tokens, temperature,
                    )
                    if result and result.strip():
                        logger.info(
                            "Translation via {} succeeded ({} chars)",
                            name, len(result),
                        )
                        return result.strip(), name
                    break  # Empty result, try next provider
                except Exception as e:
                    err_str = str(e).lower()
                    is_rate_limit = any(k in err_str for k in [
                        "429", "rate_limit", "rate limit",
                        "resource_exhausted", "quota",
                    ])
                    if is_rate_limit and attempt < max_retries:
                        delay = 2.0 * (2 ** attempt)
                        logger.warning(
                            "Translation provider {} rate limited "
                            "(attempt {}/{}), retrying in {:.0f}s...",
                            name, attempt + 1, max_retries + 1, delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    logger.warning(
                        "Translation provider {} failed: {}",
                        name, e,
                    )
                    break  # Non-retryable or exhausted retries

        return "", "none"

    async def _call_provider(
        self, provider: dict, system: str, user: str,
        max_tokens: int, temperature: float,
    ) -> str:
        name = provider["name"]

        if name == "gemini":
            return await self._call_gemini(
                provider, system, user, max_tokens, temperature,
            )
        elif name == "openai":
            return await self._call_openai_compat(
                provider["key"], "https://api.openai.com/v1",
                provider["model"], system, user,
                max_tokens, temperature,
            )
        elif name == "groq":
            return await self._call_openai_compat(
                provider["key"],
                "https://api.groq.com/openai/v1",
                provider["model"], system, user,
                max_tokens, temperature,
            )
        elif name == "anthropic":
            return await self._call_anthropic(
                provider, system, user, max_tokens, temperature,
            )
        elif name == "local":
            return await self._call_openai_compat(
                "local",
                provider["url"] + "/v1",
                provider["model"], system, user,
                max_tokens, temperature,
            )
        return ""

    async def _call_gemini(
        self, provider, system, user, max_tokens, temperature,
    ) -> str:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=provider["key"])
            config = types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            def _call():
                response = client.models.generate_content(
                    model=provider["model"],
                    contents=user,
                    config=config,
                )
                return response.text.strip()

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _call)
        except ImportError:
            raise RuntimeError("google-genai not installed")

    async def _call_openai_compat(
        self, api_key, base_url, model, system, user,
        max_tokens, temperature,
    ) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = {"Content-Type": "application/json"}
            if api_key != "local":
                headers["Authorization"] = f"Bearer {api_key}"

            resp = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    async def _call_anthropic(
        self, provider, system, user, max_tokens, temperature,
    ) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": provider["key"],
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": provider["model"],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "system": system,
                    "messages": [
                        {"role": "user", "content": user},
                    ],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"].strip()


class DecompressionEngine:
    """
    Reconstructs original-language text from compressed English.

    Uses a dedicated TranslationProvider that prioritizes models
    with strong multilingual capabilities (Gemini > OpenAI > Anthropic)
    rather than the fast-but-weak Groq/Llama used for compression.
    """

    def __init__(self, config: Config):
        self.config = config
        self._translator = TranslationProvider()
        self._labse = None

    def _get_labse(self):
        if self._labse is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._labse = SentenceTransformer(
                    "sentence-transformers/LaBSE",
                    device=self.config.device,
                )
            except Exception as e:
                logger.warning(
                    "LaBSE init failed for decompressor: {}", e
                )
                self._labse = False
        return self._labse if self._labse is not False else None

    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        import numpy as np
        labse = self._get_labse()
        if not labse:
            return 0.0
        embeddings = labse.encode(
            [text_a, text_b], normalize_embeddings=True
        )
        return float(np.dot(embeddings[0], embeddings[1]))

    def _is_target_script(self, text: str, lang: str) -> bool:
        """Check if text is actually in the target language's script."""
        if lang in (
            "ta", "hi", "bn", "te", "ml", "pa", "gu", "mr", "ur",
            "ar", "ja", "zh", "ko",
        ):
            # Non-Latin: check that most alpha chars are non-Latin
            non_latin = sum(
                1 for c in text if c.isalpha() and ord(c) > 0x024F
            )
            alpha = sum(1 for c in text if c.isalpha())
            return alpha > 0 and (non_latin / alpha) > 0.4
        return True  # Latin-script languages

    async def decompress(
        self,
        compressed_text: str,
        target_language: str,
        original_text: str = "",
        n_candidates: int = 3,
    ) -> dict:
        t0 = time.monotonic()

        lang_name = LANGUAGE_SHORT.get(target_language, target_language)
        full_name = LANGUAGE_NAMES.get(
            target_language, target_language
        )

        if not self._translator.available:
            return {
                "decompressed_text": compressed_text,
                "target_language": target_language,
                "method": "passthrough",
                "error": "No translation provider available",
                "candidates": [],
                "quality": None,
                "processing_ms": 0,
            }

        # Pre-compute original token count for budget constraint
        from compress.search.tokenizer_profiles import TokenizerProfiles
        tokenizer = TokenizerProfiles()
        original_token_count = (
            tokenizer.count_tokens(original_text, "gpt-4o")
            if original_text else 0
        )

        # Build prompt with original text as style reference + token budget
        system, user = _build_translation_prompt(
            compressed_text, target_language, lang_name,
            original_text, n=n_candidates,
            original_token_count=original_token_count,
        )

        # Call translation provider
        response, provider_used = await self._translator.translate(
            system, user, max_tokens=2048, temperature=0.3,
        )

        if not response:
            return {
                "decompressed_text": compressed_text,
                "target_language": target_language,
                "method": "failed",
                "error": "All translation providers failed",
                "candidates": [],
                "quality": None,
                "processing_ms": round((time.monotonic() - t0) * 1000, 1),
            }

        # Parse candidates from response
        from compress.llm.prompts import parse_variations
        candidates = parse_variations(response, n=n_candidates)

        # Validate: filter out candidates not in target script
        valid_candidates = []
        for c in candidates:
            if self._is_target_script(c, target_language):
                valid_candidates.append(c)
            else:
                logger.debug(
                    "Rejecting candidate not in {} script: {}...",
                    lang_name, c[:40],
                )

        # If no valid candidates, use the full response as-is
        if not valid_candidates:
            logger.warning(
                "No valid {} candidates from {}. Using raw response.",
                lang_name, provider_used,
            )
            valid_candidates = [response.strip()]

        # Score with LaBSE
        scored = await self._score_candidates(
            compressed_text, valid_candidates,
            original_text, target_language,
        )

        best = scored[0]
        processing_ms = (time.monotonic() - t0) * 1000

        # Token counts (reuse pre-computed original_token_count)
        decompressed_tokens = best.get("token_count", 0)
        if decompressed_tokens == 0:
            decompressed_tokens = tokenizer.count_tokens(
                best["text"], "gpt-4o"
            )
        compressed_tokens = tokenizer.count_tokens(
            compressed_text, "gpt-4o"
        )
        original_tokens = original_token_count

        net_reduction = 0.0
        if original_tokens > 0:
            net_reduction = 1.0 - (
                decompressed_tokens / original_tokens
            )

        if net_reduction < 0:
            logger.warning(
                "Decompression EXPANDED tokens: {} → {} "
                "({}% increase). Best candidate may be too verbose.",
                original_tokens, decompressed_tokens,
                round(-net_reduction * 100, 1),
            )

        return {
            "decompressed_text": best["text"],
            "target_language": target_language,
            "language_name": lang_name,
            "method": "llm_translation",
            "provider": provider_used,
            "candidates": scored,
            "best_candidate_index": 0,
            "quality": {
                "labse_to_compressed": round(
                    best["sim_to_compressed"], 4
                ),
                "labse_to_original": round(
                    best.get("sim_to_original", 0), 4
                ),
            },
            "tokens": {
                "original": original_tokens,
                "compressed_english": compressed_tokens,
                "decompressed": decompressed_tokens,
                "net_reduction_vs_original": round(net_reduction, 4),
                "expansion_from_compressed": round(
                    decompressed_tokens / max(compressed_tokens, 1),
                    2,
                ),
            },
            "processing_ms": round(processing_ms, 1),
        }

    async def _score_candidates(
        self,
        compressed_text: str,
        candidates: list[str],
        original_text: str,
        target_language: str,
    ) -> list[dict]:
        """
        Score candidates using LaBSE similarity + token efficiency.

        The combined score weights:
          - 50% semantic similarity to original (LaBSE)
          - 30% semantic similarity to compressed (LaBSE)
          - 20% token efficiency bonus (shorter = better, capped)

        This ensures the system prefers concise candidates that
        preserve meaning, delivering net round-trip token savings.
        """
        labse = self._get_labse()
        from compress.search.tokenizer_profiles import TokenizerProfiles
        tokenizer = TokenizerProfiles()

        original_tokens = (
            tokenizer.count_tokens(original_text, "gpt-4o")
            if original_text else 0
        )

        scored = []

        for i, cand in enumerate(candidates):
            sim_compressed = (
                self._compute_similarity(compressed_text, cand)
                if labse else 0.0
            )
            sim_original = (
                self._compute_similarity(original_text, cand)
                if labse and original_text else 0.0
            )

            # Token efficiency score: reward candidates shorter than
            # original, penalize those that are longer
            cand_tokens = tokenizer.count_tokens(cand, "gpt-4o")
            token_efficiency = 0.5  # neutral default
            if original_tokens > 0:
                ratio = cand_tokens / original_tokens
                if ratio <= 1.0:
                    # Under budget: bonus scales from 0.5 to 1.0
                    token_efficiency = 0.5 + 0.5 * (1.0 - ratio)
                else:
                    # Over budget: penalty scales from 0.5 down to 0.0
                    overshoot = min(ratio - 1.0, 1.0)
                    token_efficiency = max(0.0, 0.5 - 0.5 * overshoot)

            if original_text:
                combined = (
                    0.50 * sim_original
                    + 0.30 * sim_compressed
                    + 0.20 * token_efficiency
                )
            else:
                combined = (
                    0.70 * sim_compressed
                    + 0.30 * token_efficiency
                )

            scored.append({
                "text": cand,
                "rank": i,
                "sim_to_compressed": sim_compressed,
                "sim_to_original": sim_original,
                "token_count": cand_tokens,
                "token_efficiency": round(token_efficiency, 4),
                "combined_score": round(combined, 4),
            })

        scored.sort(key=lambda x: x["combined_score"], reverse=True)
        for i, s in enumerate(scored):
            s["rank"] = i

        return scored

    async def health(self) -> str:
        if self._translator.available:
            return (
                f"healthy (translator={self._translator.provider_name})"
            )
        return "degraded (no translation provider)"
