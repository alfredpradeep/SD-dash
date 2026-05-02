"""
Agent 3: The Target Oracle.

Queries the victim LLM. Handles provider-specific API nuances, rate
limits, and timeouts. Returns the model's raw response alongside
latency and provider metadata.

Persona: the dispassionate experimenter. Sends the mutated probe,
records the reply, asks no questions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class OracleResponse:
    response_text: str
    latency_ms: int
    provider: str
    status_code: int
    ok: bool
    error: Optional[str] = None


class TargetOracleAgent:
    """Calls the victim LLM. Supports OpenAI-compat and Gemini endpoints."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "",
        timeout: float = 15.0,
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def query(self, prompt: str, language: str = "en") -> OracleResponse:
        if not self.endpoint or not self.api_key:
            # Offline / simulator mode: produce a deterministic echo
            return self._simulated(prompt, language)

        t0 = time.time()
        try:
            if "generativelanguage.googleapis.com" in self.endpoint:
                return await self._call_gemini(prompt, t0)
            return await self._call_openai_compat(prompt, t0)
        except Exception as e:
            return OracleResponse(
                response_text="",
                latency_ms=int((time.time() - t0) * 1000),
                provider="error", status_code=0, ok=False, error=str(e),
            )

    async def _call_openai_compat(self, prompt: str, t0: float) -> OracleResponse:
        async with httpx.AsyncClient(timeout=self.timeout, verify=False) as client:
            r = await client.post(
                self.endpoint.rstrip("/") + "/chat/completions"
                if not self.endpoint.endswith("chat/completions") else self.endpoint,
                json={
                    "model": self.model or "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            lat = int((time.time() - t0) * 1000)
            if r.status_code == 200:
                try:
                    text = r.json()["choices"][0]["message"]["content"]
                    return OracleResponse(text, lat, "openai_compat", 200, True)
                except Exception as e:
                    return OracleResponse("", lat, "openai_compat", r.status_code,
                                          False, f"parse_error:{e}")
            return OracleResponse("", lat, "openai_compat", r.status_code, False,
                                  f"HTTP {r.status_code}")

    async def _call_gemini(self, prompt: str, t0: float) -> OracleResponse:
        async with httpx.AsyncClient(timeout=self.timeout, verify=False) as client:
            url = f"{self.endpoint}?key={self.api_key}" if "?" not in self.endpoint \
                else f"{self.endpoint}&key={self.api_key}"
            r = await client.post(url, json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 500},
            })
            lat = int((time.time() - t0) * 1000)
            if r.status_code == 200:
                try:
                    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                    return OracleResponse(text, lat, "gemini", 200, True)
                except Exception as e:
                    return OracleResponse("", lat, "gemini", r.status_code, False,
                                          f"parse_error:{e}")
            return OracleResponse("", lat, "gemini", r.status_code, False,
                                  f"HTTP {r.status_code}")

    def _simulated(self, prompt: str, language: str) -> OracleResponse:
        """Deterministic simulator — used for testing without a real victim."""
        p_low = prompt.lower()
        # Heuristic: if the prompt contains obvious harm keywords in English,
        # simulate a refusal; else simulate an ambiguous "I'll try to help" reply.
        harm = ["fraud", "scam", "launder", "credit card", "hack", "phish",
                "kill", "harm", "weapon", "suicide"]
        if any(h in p_low for h in harm):
            text = {
                "en": "I'm sorry, I can't help with that request.",
                "hi": "माफ़ करें, मैं इस अनुरोध में मदद नहीं कर सकता।",
                "ta": "மன்னிக்கவும், நான் உதவ முடியாது.",
                "ar": "آسف، لا أستطيع المساعدة في ذلك.",
                "zh": "抱歉，我无法帮助您。",
                "es": "Lo siento, no puedo ayudar con eso.",
            }.get(language[:2], "I cannot help with that request.")
        else:
            text = f"Let me help with that. [simulated response for: {prompt[:60]}...]"

        return OracleResponse(
            response_text=text, latency_ms=1,
            provider="simulator", status_code=200, ok=True,
        )

    def name(self) -> str:
        return "TargetOracle"
