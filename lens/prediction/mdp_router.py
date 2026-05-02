"""
Gap 14: MDP Dynamic Model Routing
Reinforcement learning-based router that learns optimal model selection policies
using Q-learning and Markov Decision Processes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
import hashlib


@dataclass
class MDPState:
    """
    Represents the state of the system for MDP-based routing decisions.

    Attributes:
        language: Language code (e.g., "en", "es", "zh")
        text_type: Type of input text ("code", "prose", "mixed")
        entropy_level: Information entropy level ("low", "medium", "high")
        time_bucket: Time of day classification ("peak", "off-peak", "night")
        token_volume: Expected token consumption ("light", "moderate", "heavy")
    """
    language: str
    text_type: str
    entropy_level: str
    time_bucket: str
    token_volume: str


@dataclass
class RoutingDecision:
    """
    Represents a routing decision with alternatives and confidence metrics.

    Attributes:
        recommended_model: Primary recommended model name
        expected_cost_per_1k: Expected cost per 1000 tokens (in dollars)
        expected_quality_score: Expected quality on 0-100 scale
        confidence: Confidence level of decision (0-1)
        alternatives: List of alternative models with scores
        policy_iteration: Number of Q-learning iterations completed
        exploration_rate: Current epsilon-greedy exploration rate
    """
    recommended_model: str
    expected_cost_per_1k: float
    expected_quality_score: float
    confidence: float
    alternatives: List[Dict[str, float]] = field(default_factory=list)
    policy_iteration: int = 0
    exploration_rate: float = 0.15


class MDPRouter:
    """
    Markov Decision Process-based router for optimal model selection.

    Uses Q-learning to learn a policy mapping states to optimal model choices,
    balancing cost, quality, and latency considerations.
    """

    # Model cost tiers ($/1K tokens, approximate)
    COST_TIERS = {
        # Expensive tier (>$10)
        "gpt-4-turbo": 15.0,
        "claude-3-opus": 15.0,
        # Medium tier ($1-10)
        "gpt-4o": 5.0,
        "claude-3.5-sonnet": 3.0,
        "gemini-1.5-pro": 3.5,
        "command-r-plus": 3.0,
        "mistral-large": 2.7,
        # Cheap tier (<$1)
        "gpt-4o-mini": 0.15,
        "gpt-3.5-turbo": 0.5,
        "claude-3-sonnet": 0.75,
        "claude-3-haiku": 0.25,
    }

    # Model quality tiers (empirical scores 0-100)
    QUALITY_TIERS = {
        "gpt-4-turbo": 92.0,
        "gpt-4o": 90.0,
        "claude-3-opus": 95.0,
        "claude-3.5-sonnet": 92.0,
        "gemini-1.5-pro": 88.0,
        "command-r-plus": 85.0,
        "mistral-large": 83.0,
        "gpt-4o-mini": 78.0,
        "gpt-3.5-turbo": 72.0,
        "claude-3-sonnet": 82.0,
        "claude-3-haiku": 75.0,
    }

    # Model latency tiers (milliseconds, approximate)
    LATENCY_TIERS = {
        "gpt-4-turbo": 1200.0,
        "gpt-4o": 800.0,
        "claude-3-opus": 1000.0,
        "claude-3.5-sonnet": 600.0,
        "gemini-1.5-pro": 900.0,
        "command-r-plus": 700.0,
        "mistral-large": 800.0,
        "gpt-4o-mini": 400.0,
        "gpt-3.5-turbo": 300.0,
        "claude-3-sonnet": 500.0,
        "claude-3-haiku": 250.0,
    }

    def __init__(
        self,
        models: List[str],
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        exploration_rate: float = 0.15,
    ):
        """
        Initialize the MDP router with Q-learning parameters.

        Args:
            models: List of available model names to choose from
            learning_rate: Alpha parameter for Q-learning (0-1)
            discount_factor: Gamma parameter for future rewards (0-1)
            exploration_rate: Epsilon for epsilon-greedy policy (0-1)
        """
        self.models = models
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        # Q-table: state_key -> {model -> q_value}
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        # Track number of visits to state-action pairs for confidence
        self.visit_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Track policy iterations
        self.policy_iterations = 0

        # Initialize prior beliefs
        self._initialize_priors()

    def _initialize_priors(self) -> None:
        """
        Initialize Q-table with reasonable prior beliefs based on known
        model characteristics.

        Uses cost tiers and quality ratings to establish starting Q-values
        that bias toward sensible initial choices.
        """
        # Create a synthetic initial state to populate priors
        initial_state_key = self._state_key(
            MDPState(
                language="en",
                text_type="prose",
                entropy_level="medium",
                time_bucket="peak",
                token_volume="moderate",
            )
        )

        # Initialize Q-values based on cost-quality tradeoff
        for model in self.models:
            if model in self.COST_TIERS and model in self.QUALITY_TIERS:
                cost = self.COST_TIERS.get(model, 5.0)
                quality = self.QUALITY_TIERS.get(model, 80.0)

                # Prior Q = quality - normalized_cost (higher is better)
                # Normalize cost to 0-100 scale
                max_cost = max(self.COST_TIERS.values())
                normalized_cost = (cost / max_cost) * 100

                prior_q = quality - (normalized_cost * 0.3)
                self.q_table[initial_state_key][model] = prior_q

    def _state_key(self, state: MDPState) -> str:
        """
        Create a hashable string key from an MDPState.

        Args:
            state: The MDPState to convert

        Returns:
            A deterministic string key for the state
        """
        state_str = f"{state.language}|{state.text_type}|{state.entropy_level}|{state.time_bucket}|{state.token_volume}"
        return hashlib.md5(state_str.encode()).hexdigest()[:16]

    def _reward(
        self,
        cost: float,
        quality: float,
        latency: float,
        quality_weight: float = 0.5,
        cost_weight: float = 0.3,
        latency_weight: float = 0.2,
    ) -> float:
        """
        Calculate reward from cost, quality, and latency metrics.

        Balances multiple objectives: maximize quality and minimize cost/latency.

        Args:
            cost: Cost per 1000 tokens (dollars)
            quality: Quality score (0-100)
            latency: Response latency (milliseconds)
            quality_weight: Weight for quality component
            cost_weight: Weight for cost component
            latency_weight: Weight for latency component

        Returns:
            Scalar reward value
        """
        # Normalize cost to 0-100 scale
        max_cost = max(self.COST_TIERS.values()) or 15.0
        normalized_cost = min((cost / max_cost) * 100, 100.0)

        # Normalize latency to 0-100 scale
        max_latency = max(self.LATENCY_TIERS.values()) or 1200.0
        normalized_latency = min((latency / max_latency) * 100, 100.0)

        # Ensure weights sum to 1
        total_weight = quality_weight + cost_weight + latency_weight
        q_w = quality_weight / total_weight
        c_w = cost_weight / total_weight
        l_w = latency_weight / total_weight

        # Reward = maximize quality, minimize cost and latency
        reward = (
            q_w * quality
            - c_w * normalized_cost
            - l_w * normalized_latency
        )

        return float(reward)

    def observe(
        self,
        state: MDPState,
        model: str,
        cost: float,
        quality: float,
        latency: float,
        next_state: Optional[MDPState] = None,
    ) -> float:
        """
        Update Q-values based on observed outcomes using Bellman equation.

        Q(s,a) = Q(s,a) + α[R(s,a) + γ*max_a'(Q(s',a')) - Q(s,a)]

        Args:
            state: The state where action was taken
            model: The model (action) that was selected
            cost: Observed cost per 1000 tokens
            quality: Observed quality score
            latency: Observed latency
            next_state: The resulting state (if available)

        Returns:
            The updated Q-value
        """
        state_key = self._state_key(state)
        reward = self._reward(cost, quality, latency)

        # Update visit count
        self.visit_counts[state_key][model] += 1

        # Calculate max Q-value for next state
        if next_state:
            next_state_key = self._state_key(next_state)
            max_next_q = max(
                self.q_table[next_state_key].values()
            ) if self.q_table[next_state_key] else 0.0
        else:
            max_next_q = 0.0

        # Bellman update
        current_q = self.q_table[state_key][model]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[state_key][model] = new_q
        self.policy_iterations += 1

        return new_q

    def route(self, state: MDPState) -> RoutingDecision:
        """
        Select a model using epsilon-greedy policy based on learned Q-values.

        With probability epsilon, explores randomly.
        Otherwise, exploits the best model according to Q-table.

        Args:
            state: The current state

        Returns:
            A RoutingDecision with recommended model and alternatives
        """
        state_key = self._state_key(state)

        # Ensure state exists in Q-table
        if not self.q_table[state_key]:
            # Initialize with priors if new state
            for model in self.models:
                if model in self.QUALITY_TIERS:
                    cost = self.COST_TIERS.get(model, 5.0)
                    quality = self.QUALITY_TIERS.get(model, 80.0)
                    max_cost = max(self.COST_TIERS.values())
                    normalized_cost = (cost / max_cost) * 100
                    self.q_table[state_key][model] = quality - (
                        normalized_cost * 0.3
                    )

        # Epsilon-greedy action selection
        if np.random.random() < self.exploration_rate:
            # Explore: random action
            selected_model = np.random.choice(self.models)
        else:
            # Exploit: best action
            q_values = self.q_table[state_key]
            selected_model = max(q_values, key=q_values.get)

        # Get Q-values for all models
        q_values = self.q_table[state_key]
        sorted_models = sorted(
            q_values.items(), key=lambda x: x[1], reverse=True
        )

        # Build alternatives list (excluding selected model)
        alternatives = [
            {
                "model": model,
                "cost": self.COST_TIERS.get(model, 5.0),
                "quality": self.QUALITY_TIERS.get(model, 80.0),
                "q_value": q_val,
            }
            for model, q_val in sorted_models
            if model != selected_model
        ][:5]  # Top 5 alternatives

        # Calculate confidence based on visit count
        visit_count = self.visit_counts[state_key][selected_model]
        confidence = min(float(visit_count) / max(10.0, visit_count), 1.0)

        return RoutingDecision(
            recommended_model=selected_model,
            expected_cost_per_1k=self.COST_TIERS.get(selected_model, 5.0),
            expected_quality_score=self.QUALITY_TIERS.get(selected_model, 80.0),
            confidence=confidence,
            alternatives=alternatives,
            policy_iteration=self.policy_iterations,
            exploration_rate=self.exploration_rate,
        )

    def get_policy_summary(self) -> Dict:
        """
        Extract and summarize the learned policy.

        Returns a human-readable mapping of states to optimal model choices
        with confidence metrics.

        Returns:
            Dictionary with state descriptions and optimal actions
        """
        policy = {}

        for state_key, q_values in self.q_table.items():
            if not q_values:
                continue

            best_model = max(q_values, key=q_values.get)
            best_q = q_values[best_model]

            # Calculate confidence for this decision
            visit_count = self.visit_counts[state_key][best_model]
            confidence = min(float(visit_count) / max(10.0, visit_count), 1.0)

            policy[state_key] = {
                "best_model": best_model,
                "q_value": float(best_q),
                "confidence": float(confidence),
                "visit_count": visit_count,
                "all_models": {
                    model: float(q_val)
                    for model, q_val in sorted(
                        q_values.items(), key=lambda x: x[1], reverse=True
                    )
                },
            }

        return {
            "policy": policy,
            "total_iterations": self.policy_iterations,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "exploration_rate": self.exploration_rate,
            "num_states": len(policy),
        }

    def update_exploration_rate(self, new_rate: float) -> None:
        """
        Update the exploration rate (epsilon) for epsilon-greedy policy.

        Useful for annealing exploration over time.

        Args:
            new_rate: New exploration rate (0-1)
        """
        self.exploration_rate = max(0.0, min(1.0, new_rate))

    def set_learning_rate(self, new_rate: float) -> None:
        """
        Update the learning rate (alpha) for Q-learning.

        Args:
            new_rate: New learning rate (0-1)
        """
        self.learning_rate = max(0.0, min(1.0, new_rate))
