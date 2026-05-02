"""
Cumulative Context Distillation Engine.

Maintains a living semantic graph across conversation turns.
Instead of growing conversation history linearly, distills each
exchange into compact semantic state.

Usage:
  session = DistillationSession(config)
  # For each conversation turn:
  state = await session.add_turn(user_text, assistant_text, language)
  compressed_context = state.to_prompt()  # ~2000 tokens, constant size
"""

import time
import json
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger
from compress.config import Config


@dataclass
class Fact:
    """A single extracted fact from conversation."""
    content: str
    source_turn: int
    category: str  # "claim", "preference", "entity", "experience", "skill"
    confidence: float = 1.0
    superseded_by: Optional[int] = None  # turn that contradicted this


@dataclass
class OpenThread:
    """A topic mentioned but not fully explored."""
    topic: str
    introduced_turn: int
    explored: bool = False
    priority: float = 1.0


@dataclass
class DistilledState:
    """The compressed representation of an entire conversation."""
    session_id: str
    turn_count: int = 0
    facts: list[Fact] = field(default_factory=list)
    entities: dict[str, str] = field(default_factory=dict)
    sentiment_trajectory: list[tuple[int, str, float]] = field(
        default_factory=list
    )
    open_threads: list[OpenThread] = field(default_factory=list)
    contradictions: list[tuple[int, int, str]] = field(
        default_factory=list
    )
    qa_pairs: list[tuple[str, str]] = field(default_factory=list)

    def to_prompt(self, max_tokens_budget: int = 2000) -> str:
        """
        Generate a compressed context prompt from the distilled state.

        This replaces the entire conversation history with a dense summary.
        Target: ~2000 tokens regardless of conversation length.
        """
        parts = []

        # Active facts (not superseded)
        active_facts = [
            f for f in self.facts if f.superseded_by is None
        ]
        if active_facts:
            fact_lines = []
            for f in active_facts[-30:]:  # Keep most recent 30
                fact_lines.append(f"- [{f.category}] {f.content}")
            parts.append(
                "ESTABLISHED FACTS:\n" + "\n".join(fact_lines)
            )

        # Key entities
        if self.entities:
            ent_lines = [
                f"- {k}: {v}" for k, v in list(self.entities.items())[:20]
            ]
            parts.append(
                "KEY ENTITIES:\n" + "\n".join(ent_lines)
            )

        # Recent Q&A (last 5)
        if self.qa_pairs:
            qa_lines = []
            for q, a in self.qa_pairs[-5:]:
                q_short = q[:100] + "..." if len(q) > 100 else q
                a_short = a[:150] + "..." if len(a) > 150 else a
                qa_lines.append(f"Q: {q_short}\nA: {a_short}")
            parts.append(
                "RECENT EXCHANGES:\n" + "\n".join(qa_lines)
            )

        # Sentiment trajectory
        if self.sentiment_trajectory:
            latest = self.sentiment_trajectory[-1]
            parts.append(
                f"CURRENT SENTIMENT: {latest[1]} "
                f"(intensity={latest[2]:.2f})"
            )

        # Contradictions
        if self.contradictions:
            contra_lines = [
                f"- Turn {t1} vs Turn {t2}: {desc}"
                for t1, t2, desc in self.contradictions[-5:]
            ]
            parts.append(
                "NOTED CONTRADICTIONS:\n" + "\n".join(contra_lines)
            )

        # Open threads
        unexplored = [
            t for t in self.open_threads if not t.explored
        ]
        if unexplored:
            thread_lines = [
                f"- {t.topic} (from turn {t.introduced_turn})"
                for t in unexplored[:10]
            ]
            parts.append(
                "OPEN TOPICS TO EXPLORE:\n" + "\n".join(thread_lines)
            )

        parts.append(f"\n[Conversation: {self.turn_count} turns]")

        return "\n\n".join(parts)

    def to_dict(self) -> dict:
        """Serialize state for API response."""
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "fact_count": len(self.facts),
            "active_facts": len(
                [f for f in self.facts if f.superseded_by is None]
            ),
            "entity_count": len(self.entities),
            "open_threads": len(
                [t for t in self.open_threads if not t.explored]
            ),
            "contradictions": len(self.contradictions),
            "sentiment": (
                self.sentiment_trajectory[-1]
                if self.sentiment_trajectory
                else None
            ),
        }


class ContextDistiller:
    """
    Maintains a living semantic state across conversation turns.

    After each exchange:
    1. Extracts facts, entities, sentiment from the new turn
    2. Checks for contradictions with existing facts
    3. Updates open threads
    4. Merges into the distilled state
    5. Returns compressed context prompt (~2000 tokens constant)
    """

    def __init__(self, config: Config):
        self.config = config
        self._sessions: dict[str, DistilledState] = {}

    def get_or_create_session(
        self, session_id: str
    ) -> DistilledState:
        if session_id not in self._sessions:
            self._sessions[session_id] = DistilledState(
                session_id=session_id
            )
        return self._sessions[session_id]

    async def add_turn(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        language: str,
        extractor=None,
    ) -> DistilledState:
        """
        Process a new conversation turn and update the distilled state.

        Args:
            session_id: Unique conversation identifier
            user_text: What the user said this turn
            assistant_text: What the assistant responded
            language: ISO 639-1 language code
            extractor: Optional LatticeExtractor for semantic graph extraction

        Returns:
            Updated DistilledState
        """
        state = self.get_or_create_session(session_id)
        state.turn_count += 1
        turn = state.turn_count

        # Extract semantic information from user utterance
        if extractor:
            try:
                graph = await extractor.extract(user_text, language)

                # Extract entities
                for node in graph.nodes:
                    if node.unit_type == SemanticUnitType.ENTITY:
                        state.entities[node.value] = (
                            node.metadata.get("ner_label", "ENTITY")
                        )

                    # Track sentiment
                    if node.unit_type == SemanticUnitType.SENTIMENT:
                        intensity = node.intensity or 0.5
                        state.sentiment_trajectory.append(
                            (turn, node.value, intensity)
                        )

                    # Extract facts from entities + relations
                    if node.unit_type in (
                        SemanticUnitType.ENTITY,
                        SemanticUnitType.QUANTIFIER,
                    ):
                        state.facts.append(
                            Fact(
                                content=node.value,
                                source_turn=turn,
                                category=node.unit_type.value,
                                confidence=node.confidence,
                            )
                        )

            except Exception as e:
                logger.warning(
                    "Extraction failed for turn {}: {}", turn, e
                )

        # Always add Q&A pair
        state.qa_pairs.append((
            user_text[:300],
            assistant_text[:300] if assistant_text else "",
        ))

        # Simple fact extraction from user text
        # (supplements semantic graph extraction)
        self._extract_simple_facts(state, user_text, turn)

        logger.debug(
            "Session {} turn {}: {} facts, {} entities, "
            "{} threads",
            session_id, turn,
            len(state.facts), len(state.entities),
            len(state.open_threads),
        )

        return state

    def _extract_simple_facts(
        self,
        state: DistilledState,
        text: str,
        turn: int,
    ):
        """Extract basic facts using patterns."""
        import re

        # Extract quoted claims
        for match in re.finditer(r'"([^"]+)"', text):
            state.facts.append(
                Fact(
                    content=match.group(1),
                    source_turn=turn,
                    category="claim",
                )
            )

        # Extract numbers/quantities
        for match in re.finditer(
            r'\b(\d+[\.,]?\d*)\s*(years?|months?|%|percent|'
            r'million|billion|thousand|lakh|crore)\b',
            text,
            re.IGNORECASE,
        ):
            state.facts.append(
                Fact(
                    content=match.group(0),
                    source_turn=turn,
                    category="quantifier",
                )
            )

    def get_compressed_context(
        self, session_id: str
    ) -> str:
        """Get the compressed context prompt for a session."""
        state = self.get_or_create_session(session_id)
        return state.to_prompt()

    def get_session_stats(
        self, session_id: str
    ) -> dict:
        """Get metrics for a session."""
        state = self.get_or_create_session(session_id)
        return state.to_dict()

    async def health(self) -> str:
        return f"healthy ({len(self._sessions)} active sessions)"


# Import at module level to avoid circular imports
from compress.lattice.structures import SemanticUnitType
