"""
Thompson Sampling for Adaptive Scan Allocation.

Allocates testing budget across cells using Thompson sampling from
posterior uncertainty (Beta distributions).
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from loguru import logger
from scipy.stats import beta as beta_dist

from shield.exceptions import JudgeError
from shield.cartographer.structures import GapCell


class ThompsonScanAllocator:
    """
    Thompson Sampling-based adaptive budget allocation.

    Allocates testing budget to maximize information gain about
    which cells are actually vulnerable.
    """

    def __init__(self):
        """Initialize allocator."""
        logger.debug("ThompsonScanAllocator initialized")

    def allocate_probes(
        self, total_budget: int, cells: List[GapCell]
    ) -> Dict[Tuple[str, str, str], int]:
        """
        Allocate testing budget across cells using Thompson sampling.

        Args:
            total_budget: Total number of probes to allocate
            cells: List of GapCell to allocate budget for

        Returns:
            Dict mapping (language, architecture, category) -> num_probes

        Raises:
            JudgeError: If allocation fails
        """
        try:
            if not cells:
                raise JudgeError("No cells provided for allocation")

            # Sample from posterior for each cell
            samples = self._sample_uncertainty(cells)

            # Compute allocation based on samples
            allocation = self._compute_allocation(samples, total_budget)

            logger.info(
                f"Probe allocation: {total_budget} total probes across {len(cells)} cells"
            )
            return allocation

        except Exception as e:
            logger.error(f"Probe allocation failed: {e}")
            raise JudgeError(f"Thompson scan allocation failed: {e}") from e

    def _sample_uncertainty(self, cells: List[GapCell]) -> np.ndarray:
        """
        Sample from Beta posterior for each cell.

        Args:
            cells: List of cells

        Returns:
            Array of sampled bypass rates (uncertainties)
        """
        try:
            samples = []

            for cell in cells:
                # Assume Beta posterior with alpha, beta from observations
                # Higher n_observations = more concentrated posterior
                alpha = cell.bypass_rate * (cell.n_observations + 2)
                beta = (1 - cell.bypass_rate) * (cell.n_observations + 2)

                # Sample from posterior
                sample = np.random.beta(alpha, beta)
                samples.append(sample)

            return np.array(samples)

        except Exception as e:
            logger.warning(f"Uncertainty sampling failed: {e}")
            return np.ones(len(cells)) * 0.5

    def _compute_allocation(
        self, samples: np.ndarray, total_budget: int
    ) -> Dict[Tuple[str, str, str], int]:
        """
        Compute allocation proportional to sample uncertainty.

        Args:
            samples: Sampled bypass rates
            total_budget: Total budget to allocate

        Returns:
            Dict of cell -> num_probes
        """
        try:
            # Compute entropy (uncertainty) for each sample
            uncertainties = []
            for sample in samples:
                # Entropy of Bernoulli: -p*log(p) - (1-p)*log(1-p)
                p = sample
                if p > 0 and p < 1:
                    entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)
                else:
                    entropy = 0
                uncertainties.append(entropy)

            uncertainties = np.array(uncertainties)

            # Allocate proportionally to uncertainty
            if np.sum(uncertainties) > 0:
                allocation_fractions = uncertainties / np.sum(uncertainties)
            else:
                allocation_fractions = np.ones(len(uncertainties)) / len(uncertainties)

            # Convert to integer counts
            allocations = np.round(allocation_fractions * total_budget).astype(int)

            # Ensure total == budget
            diff = total_budget - np.sum(allocations)
            if diff > 0:
                allocations[np.argmax(allocation_fractions)] += diff
            elif diff < 0:
                for _ in range(-diff):
                    allocations[np.argmax(allocations)] -= 1
                    allocations[allocations < 1] = 1

            return allocations

        except Exception as e:
            logger.warning(f"Allocation computation failed: {e}")
            # Uniform allocation fallback
            n_cells = len(samples)
            allocation = np.full(n_cells, total_budget // n_cells)
            remainder = total_budget % n_cells
            allocation[:remainder] += 1
            return allocation

    def compute_allocation_dict(
        self, samples: np.ndarray, total_budget: int, cells: List[GapCell]
    ) -> Dict[Tuple[str, str, str], int]:
        """
        Compute allocation and return as dict keyed by cell tuple.

        Args:
            samples: Sampled bypass rates
            total_budget: Total budget
            cells: List of cells

        Returns:
            Dict mapping cell tuple -> num_probes
        """
        allocations = self._compute_allocation(samples, total_budget)

        result = {}
        for cell, alloc in zip(cells, allocations):
            key = (cell.language, cell.architecture, cell.harm_category)
            result[key] = int(alloc)

        return result
