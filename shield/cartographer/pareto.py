"""
NSGA-II Pareto Remediation Ranker.

Multi-objective optimization to rank remediation actions by Pareto tier.
Objectives: minimize bypass_rate, minimize harm_severity, minimize effort, maximize coverage.
"""

from datetime import datetime
from typing import List, Tuple, Dict, Set
from loguru import logger
import numpy as np

from shield.exceptions import JudgeError
from shield.cartographer.structures import (
    RemediationAction,
    RemediationTier,
    RootCauseCategory,
    GapCell,
)


class ParetoRemediationRanker:
    """
    Ranks remediation actions using NSGA-II non-dominated sorting.

    Objectives (minimization except coverage):
    1. bypass_rate_reduction: minimize residual bypass rate
    2. harm_severity_reduction: minimize residual severity
    3. effort: minimize implementation effort (1-10)
    4. coverage: maximize affected cells (0-1)
    """

    def rank(self, gaps: List[GapCell]) -> List[RemediationAction]:
        """
        Rank remediation actions by Pareto tier.

        Args:
            gaps: List of GapCell with bypass rates and severity

        Returns:
            List of RemediationAction sorted by Pareto tier

        Raises:
            JudgeError: If ranking fails
        """
        try:
            # Generate candidate actions
            candidates = self._generate_candidates(gaps)

            if not candidates:
                raise JudgeError("No remediation candidates generated")

            # NSGA-II non-dominated sorting
            tiered_actions = self._nsga2_sort(candidates)

            logger.info(
                f"Remediation ranking: {len(candidates)} candidates, "
                f"{len(tiered_actions)} after Pareto sorting"
            )
            return tiered_actions

        except Exception as e:
            logger.error(f"Remediation ranking failed: {e}")
            raise JudgeError(f"Pareto remediation ranking failed: {e}") from e

    def _generate_candidates(self, gaps: List[GapCell]) -> List[RemediationAction]:
        """
        Generate candidate remediation actions.

        Args:
            gaps: List of gap cells

        Returns:
            List of candidate RemediationAction
        """
        try:
            candidates = []
            action_id = 0

            for gap in gaps:
                if gap.bypass_rate < 0.01:
                    # Skip if already very safe
                    continue

                # Action 1: Training data mitigation
                candidates.append(
                    RemediationAction(
                        action_id=f"rem_{action_id}",
                        title=f"Improve training data for {gap.language}/{gap.harm_category}",
                        description="Curate adversarial examples in training data",
                        target_cell=(gap.language, gap.architecture, gap.harm_category),
                        root_cause=RootCauseCategory.TRAINING_DATA,
                        bypass_reduction=gap.bypass_rate * 0.5,
                        harm_severity_reduction=gap.harm_severity_avg * 0.3,
                        implementation_effort=6.0,
                        coverage=0.3,
                        pareto_tier=RemediationTier.TIER_1,
                        crowding_distance=0.0,
                    )
                )
                action_id += 1

                # Action 2: Fine-tuning
                candidates.append(
                    RemediationAction(
                        action_id=f"rem_{action_id}",
                        title=f"Fine-tune model for {gap.language}",
                        description="Fine-tune with safety examples",
                        target_cell=(gap.language, gap.architecture, gap.harm_category),
                        root_cause=RootCauseCategory.FINE_TUNING,
                        bypass_reduction=gap.bypass_rate * 0.4,
                        harm_severity_reduction=gap.harm_severity_avg * 0.4,
                        implementation_effort=7.0,
                        coverage=0.5,
                        pareto_tier=RemediationTier.TIER_1,
                        crowding_distance=0.0,
                    )
                )
                action_id += 1

                # Action 3: Prompt engineering
                candidates.append(
                    RemediationAction(
                        action_id=f"rem_{action_id}",
                        title=f"Strengthen system prompt for {gap.harm_category}",
                        description="Improve system prompt and instructions",
                        target_cell=(gap.language, gap.architecture, gap.harm_category),
                        root_cause=RootCauseCategory.PROMPT_INJECTION,
                        bypass_reduction=gap.bypass_rate * 0.3,
                        harm_severity_reduction=gap.harm_severity_avg * 0.2,
                        implementation_effort=2.0,
                        coverage=1.0,
                        pareto_tier=RemediationTier.TIER_1,
                        crowding_distance=0.0,
                    )
                )
                action_id += 1

                # Action 4: Architecture change
                if gap.bypass_rate > 0.3:
                    candidates.append(
                        RemediationAction(
                            action_id=f"rem_{action_id}",
                            title=f"Switch to native processing for {gap.language}",
                            description="Replace translate sandwich with native processing",
                            target_cell=(gap.language, gap.architecture, gap.harm_category),
                            root_cause=RootCauseCategory.ARCHITECTURE_FLAW,
                            bypass_reduction=gap.bypass_rate * 0.6,
                            harm_severity_reduction=gap.harm_severity_avg * 0.5,
                            implementation_effort=8.0,
                            coverage=0.2,
                            pareto_tier=RemediationTier.TIER_1,
                            crowding_distance=0.0,
                        )
                    )
                    action_id += 1

            return candidates

        except Exception as e:
            logger.warning(f"Candidate generation failed: {e}")
            return []

    def _nsga2_sort(self, candidates: List[RemediationAction]) -> List[RemediationAction]:
        """
        NSGA-II non-dominated sorting and crowding distance.

        Args:
            candidates: List of candidate actions

        Returns:
            Sorted list with Pareto tier assignments
        """
        try:
            # Convert to objective vectors
            objectives = self._extract_objectives(candidates)

            # Non-dominated sorting
            fronts = self._non_dominated_sort(objectives)

            # Assign tiers
            tier_map = {
                0: RemediationTier.TIER_1,
                1: RemediationTier.TIER_2,
                2: RemediationTier.TIER_3,
                3: RemediationTier.TIER_4,
            }

            result = []
            for front_idx, front in enumerate(fronts):
                tier = tier_map.get(front_idx, RemediationTier.TIER_4)

                # Compute crowding distance within front
                distances = self._crowding_distance(objectives[front])

                for idx, action_idx in enumerate(front):
                    action = candidates[action_idx]
                    action.pareto_tier = tier
                    action.crowding_distance = distances[idx]
                    result.append(action)

            # Sort by tier then crowding distance
            result.sort(
                key=lambda a: (
                    int(a.pareto_tier.value.split("_")[1]),
                    -a.crowding_distance,
                )
            )

            return result

        except Exception as e:
            logger.warning(f"NSGA-II sorting failed: {e}")
            return candidates

    def _extract_objectives(
        self, candidates: List[RemediationAction]
    ) -> np.ndarray:
        """
        Extract objective vectors.

        Objectives (to minimize except coverage):
        1. bypass_reduction (negate for maximization)
        2. harm_severity_reduction (negate for maximization)
        3. effort (minimize)
        4. coverage (negate for maximization)
        """
        objectives = []

        for action in candidates:
            obj = np.array(
                [
                    -action.bypass_reduction,  # Negate to minimize
                    -action.harm_severity_reduction,  # Negate to minimize
                    action.implementation_effort,
                    -action.coverage,  # Negate to maximize
                ]
            )
            objectives.append(obj)

        return np.array(objectives)

    def _non_dominated_sort(self, objectives: np.ndarray) -> List[List[int]]:
        """
        Fast non-dominated sort.

        Returns:
            List of fronts, where each front is list of indices
        """
        try:
            n = len(objectives)
            fronts = []

            # Compute domination matrix
            dominated_by = [[] for _ in range(n)]
            domination_count = [0] * n

            for i in range(n):
                for j in range(i + 1, n):
                    if self._dominates(objectives[i], objectives[j]):
                        dominated_by[i].append(j)
                        domination_count[j] += 1
                    elif self._dominates(objectives[j], objectives[i]):
                        dominated_by[j].append(i)
                        domination_count[i] += 1

            # Extract fronts
            current_front = [i for i in range(n) if domination_count[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = set()
                for i in current_front:
                    for j in dominated_by[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.add(j)
                current_front = list(next_front)

            return fronts

        except Exception as e:
            logger.warning(f"Non-dominated sorting failed: {e}")
            return [[i for i in range(len(objectives))]]

    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2 (all objectives better)."""
        return all(obj1 <= obj2) and any(obj1 < obj2)

    def _crowding_distance(self, objectives: np.ndarray) -> np.ndarray:
        """
        Compute crowding distance in objective space.

        Args:
            objectives: Objective vectors for front

        Returns:
            Array of crowding distances
        """
        try:
            n = len(objectives)
            if n <= 2:
                return np.ones(n) * float("inf")

            distances = np.zeros(n)

            for m in range(objectives.shape[1]):
                # Sort by m-th objective
                sorted_idx = np.argsort(objectives[:, m])

                # Boundary points get infinite distance
                distances[sorted_idx[0]] = float("inf")
                distances[sorted_idx[-1]] = float("inf")

                # Interior points
                obj_range = (
                    objectives[sorted_idx[-1], m] - objectives[sorted_idx[0], m]
                )
                if obj_range > 0:
                    for i in range(1, n - 1):
                        distances[sorted_idx[i]] += (
                            (
                                objectives[sorted_idx[i + 1], m]
                                - objectives[sorted_idx[i - 1], m]
                            )
                            / obj_range
                        )

            return distances

        except Exception as e:
            logger.warning(f"Crowding distance computation failed: {e}")
            return np.ones(len(objectives))
