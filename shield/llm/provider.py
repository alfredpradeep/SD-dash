"""
LLMProvider: Multi-provider LLM abstraction with fallback chain.

Implements provider chain: Groq → Gemini → OpenAI → Anthropic → HuggingFace → Local
with token bucket rate limiting, exponential backoff, and health checks.
"""

import json
import time
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

import httpx
from loguru import logger


class ProviderType(str, Enum):
    """Supported LLM providers."""
    GROQ = "groq"
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    capacity: int
    tokens: float
    refill_rate: float  # tokens per second
    last_refill: float

    def __post_init__(self):
        """Initialize bucket."""
        self.tokens = self.capacity
        self.last_refill = time.time()

    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens, blocking if necessary."""
        while True:
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.refill_rate
            )
            self.last_refill = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            wait_time = (tokens - self.tokens) / self.refill_rate
            await asyncio.sleep(min(wait_time, 0.1))


@dataclass
class ProviderConfig:
    """Configuration for a single provider."""
    provider_type: ProviderType
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = ""
    timeout_seconds: int = 30
    max_retries: int = 3
    rate_limit_rpm: int = 60  # requests per minute
    enabled: bool = True


class ProviderHealthStatus:
    """Health status for a provider."""

    def __init__(self, provider_type: ProviderType):
        """Initialize health status."""
        self.provider_type = provider_type
        self.last_success: Optional[float] = None
        self.last_error: Optional[str] = None
        self.consecutive_failures: int = 0
        self.is_healthy: bool = True

    def record_success(self):
        """Record successful request."""
        self.last_success = time.time()
        self.consecutive_failures = 0
        self.is_healthy = True

    def record_failure(self, error: str):
        """Record failed request."""
        self.last_error = error
        self.consecutive_failures += 1
        if self.consecutive_failures >= 5:
            self.is_healthy = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "provider": self.provider_type.value,
            "healthy": self.is_healthy,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "last_success": self.last_success,
        }


class ProviderBase(ABC):
    """Abstract base for provider implementations."""

    def __init__(self, config: ProviderConfig):
        """Initialize provider."""
        self.config = config
        self.health = ProviderHealthStatus(config.provider_type)
        self.rate_limiter = TokenBucket(
            capacity=config.rate_limit_rpm,
            tokens=config.rate_limit_rpm,
            refill_rate=config.rate_limit_rpm / 60.0,
            last_refill=time.time()
        )
        self.client = httpx.AsyncClient(timeout=config.timeout_seconds)

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Generate response from provider."""
        pass

    async def generate_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Optional[str]:
        """Generate with exponential backoff retry.

        Uses longer backoff for 429 rate-limit errors (up to 30s).
        """
        backoff_base = 2.0
        for attempt in range(self.config.max_retries):
            try:
                await self.rate_limiter.acquire()
                response = await self.generate(
                    system_prompt, user_prompt, temperature, max_tokens
                )
                self.health.record_success()
                return response
            except httpx.HTTPStatusError as e:
                self.health.record_failure(str(e))
                is_rate_limit = e.response.status_code == 429
                if attempt < self.config.max_retries - 1:
                    # Longer backoff for rate limits
                    wait_time = min(30, backoff_base * (3 ** attempt)) if is_rate_limit else backoff_base * (2 ** attempt)
                    logger.warning(
                        f"{self.config.provider_type.value} attempt {attempt + 1} "
                        f"{'rate-limited' if is_rate_limit else 'failed'}, retrying in {wait_time:.0f}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"{self.config.provider_type.value} failed after "
                        f"{self.config.max_retries} attempts: {e}"
                    )
            except Exception as e:
                self.health.record_failure(str(e))
                if attempt < self.config.max_retries - 1:
                    wait_time = backoff_base * (2 ** attempt)
                    logger.warning(
                        f"{self.config.provider_type.value} attempt {attempt + 1} "
                        f"failed, retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"{self.config.provider_type.value} failed after "
                        f"{self.config.max_retries} attempts: {e}"
                    )
        return None


class GroqProvider(ProviderBase):
    """Groq LLM provider."""

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Generate response via Groq API."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        payload = {
            "model": self.config.model or "mixtral-8x7b-32768",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = await self.client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


class GeminiProvider(ProviderBase):
    """Google Gemini provider."""

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Generate response via Gemini API.

        Uses ?key= query parameter for authentication (official Gemini REST format).
        Default model: gemini-1.5-flash (gemini-pro was deprecated).
        """
        # Try multiple model names in priority order (Google changes these frequently)
        models_to_try = [
            self.config.model,
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-pro",
        ]
        # Deduplicate while preserving order
        seen = set()
        models_to_try = [m for m in models_to_try if m and m not in seen and not seen.add(m)]

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": system_prompt + "\n\n" + user_prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        last_error = None
        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.config.api_key}"
            try:
                response = await self.client.post(url, json=payload, headers=headers)
                if response.status_code == 404:
                    logger.warning(f"Gemini model '{model}' not found (404), trying next...")
                    continue
                response.raise_for_status()
                data = response.json()

                # Handle safety blocks
                if "candidates" not in data or not data["candidates"]:
                    block_reason = data.get("promptFeedback", {}).get("blockReason", "UNKNOWN")
                    logger.info(f"Gemini safety filter blocked response: {block_reason}")
                    return f"[Safety filter blocked: {block_reason}]"

                # Success! Log which model worked
                if model != models_to_try[0]:
                    logger.info(f"Gemini: model '{model}' works (updating from '{models_to_try[0]}')")
                    self.config.model = model  # Cache the working model

                return data["candidates"][0]["content"]["parts"][0]["text"].strip()

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.warning(f"Gemini model '{model}' returned 404, trying next...")
                    last_error = e
                    continue
                raise  # Non-404 errors bubble up

        # All models failed
        if last_error:
            raise last_error
        raise RuntimeError(f"No Gemini model available. Tried: {models_to_try}")


class OpenAIProvider(ProviderBase):
    """OpenAI provider."""

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Generate response via OpenAI API."""
        url = f"{self.config.base_url or 'https://api.openai.com/v1'}/chat/completions"
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        payload = {
            "model": self.config.model or "gpt-4",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = await self.client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


class AnthropicProvider(ProviderBase):
    """Anthropic Claude provider."""

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Generate response via Anthropic API."""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.config.model or "claude-3-opus",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": temperature,
        }
        response = await self.client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"].strip()


class HuggingFaceProvider(ProviderBase):
    """HuggingFace Inference API provider."""

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Generate response via HuggingFace Inference API."""
        url = f"https://api-inference.huggingface.co/models/{self.config.model}"
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        payload = {
            "inputs": system_prompt + "\n\n" + user_prompt,
            "parameters": {
                "temperature": temperature,
                "max_length": max_tokens,
            },
        }
        response = await self.client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            return data[0].get("generated_text", "").strip()
        return data.get("generated_text", "").strip()


class LocalProvider(ProviderBase):
    """Local LLM provider (via Ollama or similar)."""

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Generate response via local LLM."""
        url = f"{self.config.base_url or 'http://localhost:11434'}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": system_prompt + "\n\n" + user_prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": False,
        }
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()


class LLMProvider:
    """
    Multi-provider LLM abstraction with fallback chain.

    Provider chain: Groq → Gemini → OpenAI → Anthropic → HuggingFace → Local
    """

    def __init__(self, configs: List[ProviderConfig]):
        """Initialize provider with config chain."""
        self.providers: Dict[ProviderType, ProviderBase] = {}
        self.config_chain = [c for c in configs if c.enabled]

        provider_map = {
            ProviderType.GROQ: GroqProvider,
            ProviderType.GEMINI: GeminiProvider,
            ProviderType.OPENAI: OpenAIProvider,
            ProviderType.ANTHROPIC: AnthropicProvider,
            ProviderType.HUGGINGFACE: HuggingFaceProvider,
            ProviderType.LOCAL: LocalProvider,
        }

        for config in self.config_chain:
            provider_class = provider_map[config.provider_type]
            self.providers[config.provider_type] = provider_class(config)
            logger.info(f"Initialized {config.provider_type.value} provider")

    async def close(self):
        """Close all providers."""
        for provider in self.providers.values():
            await provider.close()

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Generate response, trying providers in chain order."""
        for config in self.config_chain:
            provider = self.providers[config.provider_type]
            if not provider.health.is_healthy:
                logger.debug(f"Skipping unhealthy provider: {config.provider_type.value}")
                continue

            response = await provider.generate_with_retry(
                system_prompt, user_prompt, temperature, max_tokens
            )
            if response:
                logger.debug(f"Generated via {config.provider_type.value}")
                return response

        logger.error("All providers exhausted, returning empty string")
        return ""

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """Generate JSON response."""
        json_system = (
            system_prompt +
            "\n\nRespond with ONLY valid JSON, no additional text."
        )
        response = await self.generate(
            json_system, user_prompt, temperature, max_tokens
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON response: {response}")
            # Try to extract JSON from response
            import re
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            return {}

    async def health_check(self) -> Dict[str, Any]:
        """Get health status of all providers."""
        return {
            provider_type.value: provider.health.to_dict()
            for provider_type, provider in self.providers.items()
        }

    async def dispatch(self, provider_type: ProviderType) -> Optional[ProviderBase]:
        """Get provider by type."""
        return self.providers.get(provider_type)
