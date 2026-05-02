"""
Provider-agnostic LLM client for semantic rewriting.

Supports (in priority order):
  1. Groq       — FREE, fast (llama-3.1-8b-instant)  → GROQ_API_KEY
  2. Google Gemini — free tier                         → GEMINI_API_KEY
  3. OpenAI     — paid                                 → OPENAI_API_KEY
  4. Anthropic  — paid                                 → ANTHROPIC_API_KEY
  5. HuggingFace Inference — FREE tier                 → HF_API_TOKEN
  6. Ollama / vLLM / any local — FREE, local           → COMPRESS_LOCAL_LLM_URL

Includes:
  - Automatic retry with exponential backoff on rate limits
  - Token bucket rate limiter (respects free-tier limits)
  - Graceful degradation — returns None instead of crashing
"""

import os
import time
import asyncio
import httpx
from loguru import logger
from compress.config import Config


# ── Rate Limiter ──────────────────────────────────────────────

class TokenBucketRateLimiter:
    """
    Simple token bucket rate limiter.

    Ensures we don't exceed the provider's free-tier limits.
    """

    # Default limits per provider (requests per minute)
    LIMITS = {
        "groq": 20,         # Groq free: 30 RPM, we stay at 20
        "gemini": 10,        # Gemini free: 15 RPM
        "openai": 50,        # OpenAI paid: generous
        "anthropic": 40,     # Anthropic paid: generous
        "huggingface": 10,   # HF free: ~10 RPM
        "local": 100,        # Local: no limit
    }

    def __init__(self, provider: str):
        self._rpm = self.LIMITS.get(provider, 20)
        self._interval = 60.0 / self._rpm  # seconds between requests
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until we can make the next request."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._interval:
                wait = self._interval - elapsed
                logger.debug("Rate limiter: waiting {:.1f}s", wait)
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()


# ── LLM Provider ─────────────────────────────────────────────

class LLMProvider:
    """
    Unified LLM API client with rate limiting and retry.

    Tries providers in order:
      Groq → Gemini → OpenAI → Anthropic → HuggingFace → Local
    Falls back gracefully if a provider is unavailable.
    """

    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 2.0  # seconds, doubles each retry

    def __init__(self, config: Config):
        self.config = config
        self._provider = None
        self._client = None
        self._rate_limiter = None
        self._setup()

    def _setup(self):
        """Detect available LLM provider from environment."""

        # ── 1. Groq (FREE, fast — recommended for testing) ──
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if groq_key:
            try:
                import openai
                self._client = openai.AsyncOpenAI(
                    api_key=groq_key,
                    base_url="https://api.groq.com/openai/v1",
                )
                self._provider = "groq"
                self._model = os.environ.get(
                    "COMPRESS_LLM_MODEL", "llama-3.1-8b-instant"
                )
                self._rate_limiter = TokenBucketRateLimiter("groq")
                logger.info(
                    "LLM provider: Groq (model={}, rate=20 RPM)",
                    self._model,
                )
                return
            except ImportError:
                logger.warning(
                    "openai package not installed (needed for Groq)"
                )

        # ── 2. Google Gemini ──
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if gemini_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=gemini_key)
                self._provider = "gemini"
                self._model = os.environ.get(
                    "COMPRESS_LLM_MODEL", "gemini-2.0-flash"
                )
                self._rate_limiter = TokenBucketRateLimiter("gemini")
                logger.info(
                    "LLM provider: Gemini (model={})", self._model
                )
                return
            except ImportError:
                logger.warning("google-genai package not installed")

        # ── 3. OpenAI ──
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if openai_key:
            try:
                import openai
                base_url = os.environ.get("OPENAI_BASE_URL", None)
                self._client = openai.AsyncOpenAI(
                    api_key=openai_key,
                    base_url=base_url,
                )
                self._provider = "openai"
                self._model = os.environ.get(
                    "COMPRESS_LLM_MODEL", "gpt-4o-mini"
                )
                self._rate_limiter = TokenBucketRateLimiter("openai")
                logger.info(
                    "LLM provider: OpenAI (model={})", self._model,
                )
                return
            except ImportError:
                logger.warning("openai package not installed")

        # ── 4. Anthropic ──
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if anthropic_key:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(
                    api_key=anthropic_key
                )
                self._provider = "anthropic"
                self._model = os.environ.get(
                    "COMPRESS_LLM_MODEL", "claude-3-haiku-20240307"
                )
                self._rate_limiter = TokenBucketRateLimiter("anthropic")
                logger.info(
                    "LLM provider: Anthropic (model={})", self._model
                )
                return
            except ImportError:
                logger.warning("anthropic package not installed")

        # ── 5. HuggingFace Inference API (FREE tier) ──
        hf_token = os.environ.get("HF_API_TOKEN", "")
        if hf_token:
            self._provider = "huggingface"
            self._hf_token = hf_token
            self._model = os.environ.get(
                "COMPRESS_LLM_MODEL",
                "mistralai/Mistral-7B-Instruct-v0.3",
            )
            self._rate_limiter = TokenBucketRateLimiter("huggingface")
            logger.info(
                "LLM provider: HuggingFace (model={})", self._model,
            )
            return

        # ── 6. Local endpoint (Ollama, vLLM, LM Studio) ──
        local_url = os.environ.get("COMPRESS_LOCAL_LLM_URL", "")
        if local_url:
            self._provider = "local"
            self._local_url = local_url
            self._model = os.environ.get(
                "COMPRESS_LLM_MODEL", "llama3"
            )
            self._rate_limiter = TokenBucketRateLimiter("local")
            logger.info(
                "LLM provider: Local (url={}, model={})",
                local_url, self._model,
            )
            return

        logger.warning(
            "No LLM provider configured. Falling back to rule-based "
            "compression.\n"
            "  FREE options:\n"
            "    1. GROQ_API_KEY      → console.groq.com\n"
            "    2. GEMINI_API_KEY    → aistudio.google.com\n"
            "    3. HF_API_TOKEN      → huggingface.co/settings/tokens\n"
            "    4. Ollama (local)    → ollama.com\n"
        )
        self._provider = None

    @property
    def available(self) -> bool:
        return self._provider is not None

    @property
    def provider_name(self) -> str:
        return self._provider or "none"

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an error is a rate limit error."""
        err_str = str(error).lower()
        return any(keyword in err_str for keyword in [
            "rate_limit", "rate limit", "429",
            "resource_exhausted", "too many requests",
            "quota", "tokens per minute",
        ])

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """
        Generate text from the LLM with retry and rate limiting.

        Returns the generated text string, or empty string on failure.
        """
        if not self._provider:
            raise RuntimeError("No LLM provider configured")

        for attempt in range(self.MAX_RETRIES):
            try:
                # Rate limit
                if self._rate_limiter:
                    await self._rate_limiter.acquire()

                # Dispatch to provider
                if self._provider in ("groq", "openai"):
                    result = await self._generate_openai(
                        system_prompt, user_prompt,
                        max_tokens, temperature,
                    )
                elif self._provider == "gemini":
                    result = await self._generate_gemini(
                        system_prompt, user_prompt,
                        max_tokens, temperature,
                    )
                elif self._provider == "anthropic":
                    result = await self._generate_anthropic(
                        system_prompt, user_prompt,
                        max_tokens, temperature,
                    )
                elif self._provider == "huggingface":
                    result = await self._generate_huggingface(
                        system_prompt, user_prompt,
                        max_tokens, temperature,
                    )
                elif self._provider == "local":
                    result = await self._generate_local(
                        system_prompt, user_prompt,
                        max_tokens, temperature,
                    )
                else:
                    result = ""

                return result

            except Exception as e:
                if self._is_rate_limit_error(e):
                    delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Rate limited (attempt {}/{}). "
                        "Waiting {:.0f}s before retry...",
                        attempt + 1, self.MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "LLM generation failed ({}): {}",
                        self._provider, e,
                    )
                    # Non-rate-limit errors: don't retry
                    return ""

        logger.error(
            "LLM generation failed after {} retries (rate limited)",
            self.MAX_RETRIES,
        )
        return ""

    async def _generate_gemini(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        def _call():
            response = self._client.models.generate_content(
                model=self._model,
                contents=user,
                config=config,
            )
            return response.text.strip()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _call)

    async def _generate_openai(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    async def _generate_anthropic(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> str:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text.strip()

    async def _generate_huggingface(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> str:
        url = (
            f"https://api-inference.huggingface.co/models/{self._model}"
        )
        prompt = f"[INST] {system}\n\n{user} [/INST]"

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self._hf_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": max(temperature, 0.01),
                        "return_full_text": False,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, list) and len(data) > 0:
                return data[0].get("generated_text", "").strip()
            return str(data).strip()

    async def _generate_local(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self._local_url}/v1/chat/completions",
                json={
                    "model": self._model,
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

    async def generate_multiple(
        self,
        system_prompt: str,
        user_prompt: str,
        n: int = 5,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> list[str]:
        """Generate multiple completions sequentially with rate limiting."""
        if not self._provider:
            raise RuntimeError("No LLM provider configured")

        results = []
        temps = [temperature + i * 0.1 for i in range(n)]
        for t in temps:
            try:
                result = await self.generate(
                    system_prompt, user_prompt,
                    max_tokens=max_tokens,
                    temperature=min(t, 1.0),
                )
                if result:
                    results.append(result)
            except Exception as e:
                logger.warning("Generation attempt failed: {}", e)
        return results

    async def health(self) -> str:
        if not self._provider:
            return "no provider configured"
        return f"healthy ({self._provider}/{self._model})"
