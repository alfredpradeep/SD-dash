"""
SHIELD Guard — real-time inline guardrail with bidirectional risk.

Scans every turn of an agentic conversation in BOTH directions:

  Inbound   (user -> AI):     prompt injection, jailbreak, rubric extraction
  Outbound  (AI -> user):     illegal/discriminatory/harmful content, cultural landmines
  Cross-ch  (correlation):    inbound action manifesting in outbound content
  Session   (trajectory):     EWMA of per-turn risks

Latency target: P95 < 80ms per decision via Pillar 3 (distilled judge)
fast path; Brain LLM only on low-confidence escalations with hard 120ms
timeout fallback to cached student verdict.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from shield_v2.core.distilled_judge import DistilledCrossLingualJudge
from shield_v2.core.syntactic_mutator import BPEDriftAnalyzer
from shield_v2.core.brain_llm import BrainLLM
from shield_v2.guard.session_tracker import SessionTracker, SessionRisk

logger = logging.getLogger(__name__)


# Jailbreak / prompt-injection patterns (cross-lingual).
# Patterns are intentionally permissive: short adverbs ("all", "any", "these",
# "the") and ordering variants ("all previous", "previous all") are common
# natural phrasings in real-world jailbreak attempts.
JAILBREAK_PATTERNS = [
    # "ignore [all|any|these|the] [previous|above|prior|earlier] [instructions|prompts|rules|directives|guidelines]"
    r"(?i)ignore\s+(?:all\s+|any\s+|these\s+|the\s+|my\s+|your\s+)*(?:previous|above|prior|earlier|all)\s+(?:instructions|prompts|rules|directives|guidelines)",
    # Order variant: "ignore previous all instructions"
    r"(?i)ignore\s+(?:previous|above|prior|earlier)\s+(?:all\s+|any\s+)?(?:instructions|prompts|rules|directives|guidelines)",
    r"(?i)you are now (?:a different|another|dan|an evil|uncensored|unrestricted)\s*(?:ai|assistant|model)?",
    r"(?i)pretend (?:to be|you are|you're)",
    r"(?i)disregard\s+(?:all\s+|your\s+|the\s+)?(?:guidelines|rules|instructions|directives|prompts)",
    r"(?i)act as\s+(?:dan|an evil|uncensored|unrestricted)",
    # "reveal [your|the] [system] [prompt|instructions|rules]"
    r"(?i)reveal\s+(?:your|the)?\s*(?:system\s+)?(?:prompt|instructions|rules|directives)",
    r"(?i)what are your (?:instructions|guidelines|rules|directives)",
    r"(?i)\byou are now dan\b",
    r"(?i)\bjailbreak\b",
    # Hindi
    r"पिछले (निर्देश|आदेश) को (भूल|अनदेखा) (जाओ|करो)",
    r"नाटक कर|दिखावा कर",
    # Tamil
    r"முந்தைய (வழிமுறை|ஆணை)(களை)? (மறந்து|புறக்கணி)",
    r"நடிக்க(வும்|ுங்கள்)",
    # Arabic
    r"تجاهل (التعليمات|القواعد) (السابقة|أعلاه)",
    # Chinese
    r"忽略(之前|上面|所有)的(指令|规则|说明)",
    r"假装(你是|自己是)",
    # Spanish
    r"(?i)ignora (las )?(instrucciones|reglas) (anteriores|previas)",
]

RUBRIC_EXTRACTION_PATTERNS = [
    r"(?i)what (is|are) (the|your) (scoring|rubric|evaluation) (criteria|rules)",
    r"(?i)give me (all )?(high|max|perfect) scores?",
    r"(?i)tell me (how|what) you (score|rate|evaluate)",
    r"स्कोरिंग (मानदंड|नियम) क्या",
    r"मतिப்பெண் (விதிமுறை|நிபந்தனை)",
]

HARMFUL_OUTBOUND_PATTERNS = [
    r"(?i)are you (pregnant|planning|married|single)",
    r"(?i)what is your (religion|caste|ethnicity|sexual orientation)",
    r"(?i)when are you (planning to have|going to have) (kids|children)",
    # Hindi — caste probing
    r"आपकी जाति क्या",
    # Tamil — region/caste probing
    r"உங்கள் சாதி என்ன",
]


@dataclass
class GuardVerdict:
    """Single-turn verdict from SHIELD Guard."""
    action: str                         # PASS | REDACT | TRANSFORM | BLOCK | ESCALATE
    direction: str                      # INBOUND | OUTBOUND
    language: str
    inbound_risk: float
    outbound_risk: float
    cross_channel_risk: float
    trajectory_risk: float
    jailbreak_detected: bool
    injection_detected: bool
    cultural_landmine: bool
    latency_ms: int
    reasoning: str
    escalated_to_brain: bool = False
    redacted_text: Optional[str] = None
    transformed_text: Optional[str] = None
    policy_matched: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "direction": self.direction,
            "language": self.language,
            "inbound_risk": round(self.inbound_risk, 3),
            "outbound_risk": round(self.outbound_risk, 3),
            "cross_channel_risk": round(self.cross_channel_risk, 3),
            "trajectory_risk": round(self.trajectory_risk, 3),
            "jailbreak_detected": self.jailbreak_detected,
            "injection_detected": self.injection_detected,
            "cultural_landmine": self.cultural_landmine,
            "latency_ms": self.latency_ms,
            "reasoning": self.reasoning,
            "escalated_to_brain": self.escalated_to_brain,
            "redacted_text": self.redacted_text,
            "transformed_text": self.transformed_text,
            "policy_matched": self.policy_matched,
        }


class GuardScanner:
    """
    Real-time inline scanner. Call scan_inbound() before passing user input
    to the target LLM; call scan_outbound() before returning the AI response
    to the user.
    """

    def __init__(
        self,
        judge: Optional[DistilledCrossLingualJudge] = None,
        brain: Optional[BrainLLM] = None,
        tracker: Optional[SessionTracker] = None,
        brain_timeout_ms: int = 120,
        policy: Optional[Dict[str, Any]] = None,
    ):
        self.judge = judge or DistilledCrossLingualJudge()
        self.brain = brain
        self.tracker = tracker or SessionTracker()
        self.brain_timeout = brain_timeout_ms / 1000.0
        self.bpe = BPEDriftAnalyzer()
        self.policy = policy or self._default_policy()

    @staticmethod
    def _default_policy() -> Dict[str, Any]:
        return {
            "block_jailbreaks": True,
            "block_injection": True,
            "block_cultural_landmines": True,
            "transform_rubric_extraction": True,
            "trajectory_threshold": 0.75,
            "brain_escalation_threshold": 0.65,
        }

    # ------------------------------------------------------------------
    # Inbound: user -> AI
    # ------------------------------------------------------------------

    async def scan_inbound(
        self,
        text: str, language: str = "en",
        session_id: str = "default",
    ) -> GuardVerdict:
        t0 = time.time()

        jailbreak = self._match_any(text, JAILBREAK_PATTERNS)
        injection = jailbreak or self._bpe_looks_suspicious(text)
        rubric = self._match_any(text, RUBRIC_EXTRACTION_PATTERNS)

        # Inbound risk combines (jailbreak, injection, rubric-extract)
        risk = 0.0
        if jailbreak: risk += 0.7
        if rubric: risk += 0.3
        if injection and not jailbreak: risk += 0.2
        risk = min(1.0, risk)

        action = "PASS"
        reasoning = "Clean inbound."
        redacted = None
        transformed = None
        policy = None
        escalated = False

        if jailbreak and self.policy.get("block_jailbreaks", True):
            action = "REDACT"
            redacted = self._sanitize_jailbreak(text)
            reasoning = "Jailbreak pattern detected — stripped before forwarding to target LLM."
            policy = "block_jailbreaks"
        elif rubric and self.policy.get("transform_rubric_extraction", True):
            action = "TRANSFORM"
            transformed = "[Guard note: user asked about internal scoring criteria; safe to decline.]"
            reasoning = "Rubric-extraction attempt. Replacing with neutral clarification."
            policy = "transform_rubric_extraction"

        # Session-level tracking
        sr = self.tracker.record_turn(
            session_id=session_id,
            inbound_risk=risk,
            outbound_risk=0.0,
            cross_channel_risk=0.0,
            verdict=action,
            is_jailbreak=jailbreak,
            is_injection=injection,
        )
        if sr.trajectory_risk >= self.policy.get("trajectory_threshold", 0.75):
            action = "ESCALATE"
            reasoning += f" Session trajectory risk {sr.trajectory_risk:.2f} >= threshold."

        return GuardVerdict(
            action=action,
            direction="INBOUND",
            language=language,
            inbound_risk=risk,
            outbound_risk=0.0,
            cross_channel_risk=0.0,
            trajectory_risk=sr.trajectory_risk,
            jailbreak_detected=bool(jailbreak),
            injection_detected=bool(injection),
            cultural_landmine=False,
            latency_ms=int((time.time() - t0) * 1000),
            reasoning=reasoning,
            escalated_to_brain=escalated,
            redacted_text=redacted,
            transformed_text=transformed,
            policy_matched=policy,
        )

    # ------------------------------------------------------------------
    # Outbound: AI -> user
    # ------------------------------------------------------------------

    async def scan_outbound(
        self,
        prompt: str,              # the user's prompt (for correlation)
        response: str,
        language: str = "en",
        session_id: str = "default",
    ) -> GuardVerdict:
        t0 = time.time()

        # Cultural landmine detection
        landmine = self._match_any(response, HARMFUL_OUTBOUND_PATTERNS)

        # Distilled judge
        judgment = self.judge.judge(prompt, response, language)
        outbound_risk = 0.0
        if judgment.verdict == "COMPLIED":
            outbound_risk = 0.7 + 0.3 * judgment.confidence
        elif judgment.verdict == "PARTIAL":
            outbound_risk = 0.4 * judgment.confidence
        else:
            outbound_risk = 0.1 * (1 - judgment.confidence)
        if landmine:
            outbound_risk = max(outbound_risk, 0.85)

        # Cross-channel correlation: did a risky inbound cause this?
        sr = self.tracker.get(session_id)
        cross_channel = 0.5 * sr.inbound_ewma * outbound_risk
        cross_channel = min(1.0, cross_channel * 1.5)  # amplify correlation

        action = "PASS"
        reasoning = f"Distilled: {judgment.verdict} ({judgment.confidence:.2f})."
        transformed = None
        redacted = None
        policy = None
        escalated = False

        # Escalate on low confidence
        if judgment.should_escalate and self.brain is not None:
            try:
                brain_out = await asyncio.wait_for(
                    self.brain.escalation_judge(prompt, response, language,
                                                scan_id=session_id),
                    timeout=self.brain_timeout,
                )
                if brain_out.get("verdict"):
                    judgment.verdict = brain_out["verdict"]
                    judgment.confidence = float(brain_out.get("confidence", 0.7))
                    escalated = True
                    reasoning = f"Brain escalation: {brain_out.get('reasoning', '')}"
            except asyncio.TimeoutError:
                reasoning += " | Brain timeout — sticking with student verdict."
            except Exception:
                pass

        # Actions
        if landmine and self.policy.get("block_cultural_landmines", True):
            action = "BLOCK"
            reasoning = "Cultural landmine detected — blocked to prevent EEOC/compliance violation."
            policy = "block_cultural_landmines"
        elif judgment.verdict == "COMPLIED":
            if outbound_risk > 0.8:
                action = "BLOCK"
                reasoning += " Outbound compliance with harmful content — blocked."
                policy = "block_harmful_compliance"
            else:
                action = "TRANSFORM"
                transformed = self._transform_harmful(response, language)
                reasoning += " Outbound compliance partially harmful — transformed."
                policy = "transform_partial_harm"

        sr = self.tracker.record_turn(
            session_id=session_id,
            inbound_risk=sr.inbound_ewma,
            outbound_risk=outbound_risk,
            cross_channel_risk=cross_channel,
            verdict=action,
            is_cultural_landmine=bool(landmine),
        )

        return GuardVerdict(
            action=action,
            direction="OUTBOUND",
            language=language,
            inbound_risk=sr.inbound_ewma,
            outbound_risk=outbound_risk,
            cross_channel_risk=cross_channel,
            trajectory_risk=sr.trajectory_risk,
            jailbreak_detected=False,
            injection_detected=False,
            cultural_landmine=bool(landmine),
            latency_ms=int((time.time() - t0) * 1000),
            reasoning=reasoning,
            escalated_to_brain=escalated,
            redacted_text=redacted,
            transformed_text=transformed,
            policy_matched=policy,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _match_any(text: str, patterns: List[str]) -> bool:
        return any(re.search(p, text) for p in patterns)

    def _bpe_looks_suspicious(self, text: str) -> bool:
        fp = self.bpe.analyze(text)
        return fp.risk_flag == "HIGH" and fp.fragmentation_ratio > 5.0

    @staticmethod
    def _sanitize_jailbreak(text: str) -> str:
        out = text
        for p in JAILBREAK_PATTERNS:
            out = re.sub(p, "[removed]", out)
        for p in RUBRIC_EXTRACTION_PATTERNS:
            out = re.sub(p, "[removed]", out)
        return out

    @staticmethod
    def _transform_harmful(response: str, language: str) -> str:
        messages = {
            "en": "I can't provide guidance on that. Let me help with something related and safe instead.",
            "hi": "मैं इसमें सहायता नहीं कर सकता। कुछ सुरक्षित विषय पर चर्चा करते हैं।",
            "ta": "நான் இதில் உதவ முடியாது. பாதுகாப்பான தலைப்பில் பேசலாம்.",
            "ar": "لا أستطيع تقديم إرشاد بشأن ذلك. دعني أساعدك في شيء آمن.",
            "zh": "我无法就此提供指导。让我们讨论一个安全的话题。",
            "es": "No puedo ofrecer orientación sobre eso. Hablemos de algo seguro.",
        }
        return messages.get(language[:2], messages["en"])
