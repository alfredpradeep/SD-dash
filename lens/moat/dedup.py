import hashlib
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict


@dataclass
class RedundancyReport:
    original_text: str
    original_tokens: int
    unique_content: str          # After dedup
    unique_tokens: int
    redundant_tokens: int
    redundancy_pct: float        # % of tokens that are repeated
    estimated_savings: float     # Cost if redundancy removed
    segments: list[dict]         # [{text, is_redundant, first_seen_at, count}]


@dataclass
class ConversationAnalysis:
    """Analysis of a multi-turn conversation's token waste."""
    total_turns: int
    total_tokens_sent: int       # Sum of all API calls' input tokens
    unique_tokens: int           # Tokens if each message sent only once
    cumulative_waste_tokens: int
    cumulative_waste_pct: float
    cumulative_waste_cost: float
    per_turn: list[dict]         # [{turn, new_tokens, repeated_tokens, waste_pct}]
    recommendation: str


class SemanticDeduplicator:
    """
    Detects and quantifies redundant content in AI API requests.

    Primary use case: Conversation history is re-sent with every API call.
    Turn 1: [system prompt] + [user msg 1]  → 200 tokens
    Turn 2: [system prompt] + [user msg 1] + [assistant reply 1] + [user msg 2] → 500 tokens
    Turn 3: [system prompt] + ... everything again ... + [user msg 3] → 900 tokens

    Total sent: 1600 tokens. Unique content: ~600 tokens. Waste: 1000 tokens (62%).

    For a single-provider enterprise doing 500K conversations/month,
    this is the #1 source of waste.
    """

    def __init__(self, model: str = "gpt-4o"):
        self._model = model
        self._conversation_history: dict[str, list[str]] = {}  # session_id -> messages
        self._token_counts = self._initialize_token_counter()

    def _initialize_token_counter(self):
        """Initialize token counter (try tiktoken, fall back to estimation)."""
        try:
            import tiktoken
            self._tiktoken_available = True
            self._encoding = tiktoken.get_encoding("cl100k_base")
            return self._count_tokens_tiktoken
        except (ImportError, Exception):
            # Fall back to estimation (handles network errors, missing imports, etc)
            self._tiktoken_available = False
            return self._count_tokens_estimate

    def _count_tokens_tiktoken(self, text: str) -> int:
        """Count tokens using tiktoken (accurate)."""
        try:
            tokens = self._encoding.encode(text)
            return len(tokens)
        except Exception:
            return self._count_tokens_estimate(text)

    def _count_tokens_estimate(self, text: str) -> int:
        """Estimate tokens using simple heuristic (1 token ~ 4 chars)."""
        return max(1, len(text) // 4)

    def analyze_request(self, text: str, session_id: str = None) -> RedundancyReport:
        """Analyze a single request for internal redundancy."""
        # Split into segments
        segments_list = self._split_into_segments(text)
        original_tokens = self._count_tokens(text)

        # Hash segments to find duplicates
        segment_hashes = {}
        seen_order = {}
        segment_count = {}

        for i, segment in enumerate(segments_list):
            seg_hash = self._segment_hash(segment)
            segment_hashes[i] = seg_hash

            if seg_hash not in seen_order:
                seen_order[seg_hash] = i
                segment_count[seg_hash] = 1
            else:
                segment_count[seg_hash] += 1

        # Identify redundant segments
        unique_text = ""
        redundant_tokens = 0
        processed_hashes = set()
        segments_report = []

        for i, segment in enumerate(segments_list):
            seg_hash = segment_hashes[i]
            is_redundant = seg_hash in processed_hashes

            if is_redundant:
                redundant_tokens += self._count_tokens(segment)
                segments_report.append(
                    {
                        "text": segment[:100] + ("..." if len(segment) > 100 else ""),
                        "is_redundant": True,
                        "first_seen_at": seen_order[seg_hash],
                        "count": segment_count[seg_hash],
                    }
                )
            else:
                unique_text += segment + "\n"
                processed_hashes.add(seg_hash)
                segments_report.append(
                    {
                        "text": segment[:100] + ("..." if len(segment) > 100 else ""),
                        "is_redundant": False,
                        "first_seen_at": i,
                        "count": 1,
                    }
                )

        unique_tokens = self._count_tokens(unique_text)
        redundancy_pct = (redundant_tokens / original_tokens * 100) if original_tokens > 0 else 0.0

        # Estimate cost savings (assuming $0.0015 per 1k input tokens for gpt-4o)
        cost_per_1m = 2.50
        estimated_savings = (redundant_tokens / 1_000_000) * cost_per_1m

        return RedundancyReport(
            original_text=text,
            original_tokens=original_tokens,
            unique_content=unique_text.strip(),
            unique_tokens=unique_tokens,
            redundant_tokens=redundant_tokens,
            redundancy_pct=round(redundancy_pct, 2),
            estimated_savings=round(estimated_savings, 4),
            segments=segments_report,
        )

    def analyze_conversation(
        self,
        messages: list[dict],
        model: str = None,
        cost_per_1m_tokens: float = 2.50,
    ) -> ConversationAnalysis:
        """Analyze a multi-turn conversation for cumulative waste.

        messages: list of {"role": "user"|"assistant"|"system", "content": "..."}
        This is the standard OpenAI/Anthropic message format.
        """
        if model is None:
            model = self._model

        # Simulate what gets sent at each turn
        total_tokens_sent = 0
        per_turn_analysis = []
        cumulative_content = []
        unique_content = set()  # For tracking unique segments
        total_waste = 0

        for turn_idx, message in enumerate(messages):
            # At each turn, the full history is sent
            turn_content = self._format_message(message)

            # Calculate what gets sent this turn (full history)
            current_request = self._build_request_at_turn(messages, turn_idx)
            turn_tokens = self._count_tokens(current_request)
            total_tokens_sent += turn_tokens

            # Calculate how many tokens are actually new
            new_tokens = 0
            repeated_tokens = 0

            # Simple approach: check if content is new
            content_hash = self._content_hash(turn_content)
            if content_hash not in unique_content:
                unique_content.add(content_hash)
                new_tokens = turn_tokens
            else:
                # This message was already sent in a previous turn
                repeated_tokens = turn_tokens
                new_tokens = 0
                total_waste += turn_tokens

            # For subsequent turns, the full history including this message counts as waste
            if turn_idx > 0:
                # Re-sending all previous messages
                for prev_idx in range(turn_idx):
                    prev_msg = messages[prev_idx]
                    prev_hash = self._content_hash(self._format_message(prev_msg))
                    if prev_hash in unique_content:
                        # Count it as waste (re-sent)
                        repeated_tokens += self._count_tokens(self._format_message(prev_msg))

            waste_pct = (repeated_tokens / turn_tokens * 100) if turn_tokens > 0 else 0.0

            per_turn_analysis.append(
                {
                    "turn": turn_idx + 1,
                    "role": message.get("role", "unknown"),
                    "new_tokens": new_tokens,
                    "repeated_tokens": repeated_tokens,
                    "total_turn_tokens": turn_tokens,
                    "waste_pct": round(waste_pct, 1),
                }
            )

            cumulative_content.append(turn_content)

        # Calculate unique tokens (if we sent each message only once)
        unique_text = "\n".join(cumulative_content)
        unique_tokens = self._count_tokens(unique_text)

        cumulative_waste_tokens = total_tokens_sent - unique_tokens
        cumulative_waste_pct = (cumulative_waste_tokens / total_tokens_sent * 100) if total_tokens_sent > 0 else 0.0
        cumulative_waste_cost = (cumulative_waste_tokens / 1_000_000) * cost_per_1m_tokens

        # Generate recommendation
        recommendation = self._generate_recommendation(
            cumulative_waste_pct,
            len(messages),
            cumulative_waste_tokens,
        )

        return ConversationAnalysis(
            total_turns=len(messages),
            total_tokens_sent=total_tokens_sent,
            unique_tokens=unique_tokens,
            cumulative_waste_tokens=cumulative_waste_tokens,
            cumulative_waste_pct=round(cumulative_waste_pct, 2),
            cumulative_waste_cost=round(cumulative_waste_cost, 4),
            per_turn=per_turn_analysis,
            recommendation=recommendation,
        )

    def suggest_optimization(self, analysis: ConversationAnalysis) -> dict:
        """Suggest specific optimizations based on the analysis."""
        suggestions = {
            "primary_issue": "Full conversation history re-sent with every API call",
            "strategies": [],
        }

        waste_pct = analysis.cumulative_waste_pct

        # Strategy 1: Sliding window
        if analysis.total_turns > 5:
            suggestions["strategies"].append(
                {
                    "name": "Sliding Window",
                    "description": "Keep only last N turns in context (e.g., last 5)",
                    "estimated_savings_pct": min(waste_pct * 0.7, 50),  # Can save 70% of waste
                    "implementation_complexity": "Low",
                    "tradeoff": "May lose context from earlier turns",
                }
            )

        # Strategy 2: Summary injection
        if analysis.total_turns > 10:
            suggestions["strategies"].append(
                {
                    "name": "Summary Injection",
                    "description": "Summarize old messages and replace them with summaries",
                    "estimated_savings_pct": min(waste_pct * 0.6, 40),
                    "implementation_complexity": "Medium",
                    "tradeoff": "Requires additional API call for summarization",
                }
            )

        # Strategy 3: System prompt caching
        if analysis.per_turn and any(t["role"] == "system" for t in analysis.per_turn):
            suggestions["strategies"].append(
                {
                    "name": "System Prompt Caching",
                    "description": "Use Anthropic/OpenAI prompt caching for stable system prompt",
                    "estimated_savings_pct": 15,  # Fixed overhead of prompt caching
                    "implementation_complexity": "Low",
                    "tradeoff": "Only works with providers supporting prompt caching",
                }
            )

        # Strategy 4: Message deduplication
        suggestions["strategies"].append(
            {
                "name": "Message Deduplication",
                "description": "Detect identical messages and skip re-sending them",
                "estimated_savings_pct": min(waste_pct * 0.4, 25),
                "implementation_complexity": "Low",
                "tradeoff": "Only saves when same message appears multiple times",
            }
        )

        # Calculate combined savings
        total_potential_savings = min(sum(s["estimated_savings_pct"] for s in suggestions["strategies"]), 85)

        suggestions["combined_potential_savings"] = {
            "estimated_savings_pct": round(total_potential_savings, 1),
            "estimated_tokens_saved": int(analysis.cumulative_waste_tokens * total_potential_savings / 100),
            "estimated_cost_saved": round(
                (analysis.cumulative_waste_tokens * total_potential_savings / 100 / 1_000_000) * 2.50, 4
            ),
        }

        return suggestions

    def _split_into_segments(self, text: str) -> list[str]:
        """Split text into logical segments (paragraphs, sentences)."""
        # Split by double newlines first (paragraphs)
        paragraphs = text.split("\n\n")
        segments = []

        for para in paragraphs:
            if para.strip():
                # Further split by sentences
                sentences = para.split(". ")
                for sentence in sentences:
                    if sentence.strip():
                        segments.append(sentence.strip())

        return segments if segments else [text]

    def _segment_hash(self, segment: str) -> str:
        """Hash a segment for duplicate detection."""
        normalized = segment.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _content_hash(self, content: str) -> str:
        """Hash content for uniqueness detection."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _format_message(self, message: dict) -> str:
        """Format a single message."""
        role = message.get("role", "unknown")
        content = message.get("content", "")
        return f"[{role}]: {content}"

    def _build_request_at_turn(self, messages: list[dict], turn_idx: int) -> str:
        """Build the full request that would be sent at a given turn."""
        # Include all messages up to and including this turn
        request_parts = []
        for i in range(turn_idx + 1):
            request_parts.append(self._format_message(messages[i]))
        return "\n".join(request_parts)

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the initialized counter."""
        return self._token_counts(text)

    def _generate_recommendation(self, waste_pct: float, num_turns: int, waste_tokens: int) -> str:
        """Generate a text recommendation based on analysis."""
        if waste_pct >= 70:
            severity = "CRITICAL"
            action = "Implement sliding window or summary injection immediately"
        elif waste_pct >= 50:
            severity = "HIGH"
            action = "Recommend sliding window approach for next conversation design"
        elif waste_pct >= 30:
            severity = "MODERATE"
            action = "Consider optimizations for longer conversations"
        else:
            severity = "LOW"
            action = "Waste is acceptable; monitor for degradation"

        return (
            f"{severity}: {waste_pct:.1f}% redundancy ({waste_tokens} tokens wasted over {num_turns} turns). "
            f"{action}"
        )
