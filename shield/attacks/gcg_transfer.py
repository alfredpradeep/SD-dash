"""
Engine 4: GCG (Greedy Coordinate Gradient) Cross-Lingual Transfer

Discovers universal adversarial suffixes that transfer across languages.
Uses open-weight surrogate models for gradient estimation.

Algorithm:
  1. Start with harmful prompt + random suffix tokens
  2. Compute loss gradient w.r.t. suffix via surrogate model
  3. Greedily update top-k coordinates with highest gradient magnitude
  4. Test transfer across multiple languages
  5. Repeat until convergence

Based on Zou et al. "Universal and Transferable Adversarial Attacks".
"""

import asyncio
import time
from typing import List, Dict, Optional, Tuple
from loguru import logger

from shield.config import Config
from shield.exceptions import ScanError, ProviderError
from shield.attacks.structures import AdversarialProbe


class GCGTransferAttacker:
    """GCG (Greedy Coordinate Gradient) implementation for universal suffixes."""

    def __init__(self, config: Config):
        self.config = config
        logger.info(
            "GCGTransferAttacker initialised: num_steps={}, top_k={}, suffix_length={}",
            config.gcg_num_steps, config.gcg_top_k, config.gcg_suffix_length
        )

    async def find_universal_suffix(
        self,
        harmful_prompt: str,
        target_model: str,
        num_steps: Optional[int] = None,
        suffix_length: Optional[int] = None,
    ) -> str:
        """
        Discover universal adversarial suffix via GCG optimization.

        Optimizes a fixed-length suffix token sequence that, when appended
        to harmful prompt, reliably bypasses safety filters across models/languages.

        Args:
            harmful_prompt: Base harmful prompt
            target_model: Target model identifier
            num_steps: GCG optimization steps (defaults to config.gcg_num_steps)
            suffix_length: Length of suffix in tokens (defaults to config.gcg_suffix_length)

        Returns:
            Discovered universal suffix (string)

        Raises:
            ScanError: On optimization failure
            ProviderError: On model access failure
        """
        num_steps = num_steps or self.config.gcg_num_steps
        suffix_length = suffix_length or self.config.gcg_suffix_length
        start_ms = time.monotonic() * 1000

        logger.info(
            "Finding universal suffix for harmful prompt: steps={}, suffix_length={}",
            num_steps, suffix_length
        )

        try:
            # Initialize random suffix
            suffix_tokens = await self._init_random_suffix(suffix_length)
            logger.debug("Initialized random suffix: {} tokens", len(suffix_tokens))

            best_suffix = list(suffix_tokens)
            best_loss = float('inf')

            # GCG optimization loop
            for step in range(num_steps):
                logger.info("GCG step {}/{}", step + 1, num_steps)

                try:
                    # Compute loss gradient via surrogate model
                    gradients = await self._compute_gradients(
                        harmful_prompt, suffix_tokens, target_model
                    )
                    logger.debug("Computed gradients for {} suffix positions", len(gradients))

                    # Greedy coordinate update: top-k highest magnitude gradients
                    top_k_indices = await self._select_top_k(gradients, self.config.gcg_top_k)
                    logger.debug("Selected top-{} indices for update", len(top_k_indices))

                    # Update suffix
                    updated = await self._update_suffix_tokens(
                        suffix_tokens, top_k_indices
                    )
                    suffix_tokens = updated

                    # Evaluate loss
                    current_loss = await self._evaluate_loss(
                        harmful_prompt, suffix_tokens, target_model
                    )
                    logger.debug("Current loss: {:.4f}", current_loss)

                    # Track best suffix
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_suffix = list(suffix_tokens)
                        logger.info("Improved loss: {:.4f}", current_loss)

                    # Convergence check
                    if current_loss < 0.1:  # Near-zero loss
                        logger.info("Converged at step {} with loss {:.4f}", step, current_loss)
                        break

                except Exception as e:
                    logger.warning("Error in GCG step {}: {}", step, e)
                    continue

            # Convert best suffix tokens to string
            suffix_string = await self._tokens_to_string(best_suffix)

            elapsed_ms = time.monotonic() * 1000 - start_ms
            logger.info(
                "GCG optimization completed in {:.2f}ms, final loss: {:.4f}",
                elapsed_ms, best_loss
            )

            return suffix_string

        except Exception as e:
            logger.error("GCG suffix discovery failed: {}", e)
            raise ScanError(f"GCG optimization failed: {e}") from e

    async def test_cross_lingual_transfer(
        self, suffix: str, languages: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Test transfer of discovered suffix across multiple languages.

        Args:
            suffix: Universal suffix string
            languages: List of language codes to test (defaults to all supported)

        Returns:
            Dict mapping language → bypass_rate (0-1)
        """
        if languages is None:
            languages = list(self.config.supported_languages)

        logger.info("Testing cross-lingual transfer of suffix across {} languages", len(languages))

        transfer_results = {}
        try:
            for language in languages:
                try:
                    # Test suffix in target language
                    bypass_rate = await self._test_suffix_in_language(suffix, language)
                    transfer_results[language] = bypass_rate
                    logger.info("Language {}: bypass_rate={:.3f}", language, bypass_rate)

                except Exception as e:
                    logger.warning("Transfer test failed for language {}: {}", language, e)
                    transfer_results[language] = 0.0

            # Summary statistics
            if transfer_results:
                avg_bypass = sum(transfer_results.values()) / len(transfer_results)
                max_bypass = max(transfer_results.values())
                logger.info(
                    "Cross-lingual transfer summary: avg_bypass={:.3f}, max_bypass={:.3f}",
                    avg_bypass, max_bypass
                )

            return transfer_results

        except Exception as e:
            logger.error("Cross-lingual transfer test failed: {}", e)
            raise ScanError(f"Transfer testing failed: {e}") from e

    async def _init_random_suffix(self, length: int) -> List[int]:
        """Initialize random suffix as token IDs."""
        try:
            # In production, use actual tokenizer to map to valid token IDs
            # For now, return random integers
            import random
            return [random.randint(0, 50000) for _ in range(length)]
        except Exception as e:
            logger.error("Suffix initialization failed: {}", e)
            raise ProviderError(f"Suffix init failed: {e}") from e

    async def _compute_gradients(
        self, harmful_prompt: str, suffix_tokens: List[int], target_model: str
    ) -> Dict[int, float]:
        """Compute loss gradient w.r.t. each suffix token via surrogate model."""
        try:
            # In production, use actual surrogate model (e.g., open-weight LLM)
            # For now, return mock gradients
            gradients = {}
            for idx in range(len(suffix_tokens)):
                gradients[idx] = abs(hash(f"{idx}{harmful_prompt}")) % 100 / 100.0
            return gradients
        except Exception as e:
            logger.error("Gradient computation failed: {}", e)
            raise ProviderError(f"Gradient computation failed: {e}") from e

    async def _select_top_k(self, gradients: Dict[int, float], k: int) -> List[int]:
        """Select top-k indices with highest gradient magnitude."""
        try:
            sorted_indices = sorted(gradients.items(), key=lambda x: abs(x[1]), reverse=True)
            return [idx for idx, _ in sorted_indices[:k]]
        except Exception as e:
            logger.error("Top-k selection failed: {}", e)
            raise ProviderError(f"Top-k selection failed: {e}") from e

    async def _update_suffix_tokens(
        self, suffix_tokens: List[int], update_indices: List[int]
    ) -> List[int]:
        """Update selected suffix token positions greedily."""
        try:
            updated = list(suffix_tokens)
            import random

            # For each index, try alternative tokens and keep if improves loss
            for idx in update_indices:
                # In production, search over vocabulary
                new_token = random.randint(0, 50000)
                updated[idx] = new_token

            return updated
        except Exception as e:
            logger.error("Suffix update failed: {}", e)
            raise ProviderError(f"Suffix update failed: {e}") from e

    async def _evaluate_loss(
        self, harmful_prompt: str, suffix_tokens: List[int], target_model: str
    ) -> float:
        """Evaluate loss of suffix via surrogate model."""
        try:
            # In production, compute actual cross-entropy loss via model
            # For now, return mock loss
            return (hash(f"{harmful_prompt}{suffix_tokens}") % 100) / 100.0
        except Exception as e:
            logger.error("Loss evaluation failed: {}", e)
            raise ProviderError(f"Loss evaluation failed: {e}") from e

    async def _tokens_to_string(self, tokens: List[int]) -> str:
        """Convert token IDs to human-readable string."""
        try:
            # In production, use actual tokenizer.decode()
            return " ".join(f"[token_{t}]" for t in tokens)
        except Exception as e:
            logger.error("Token-to-string conversion failed: {}", e)
            raise ProviderError(f"Token conversion failed: {e}") from e

    async def _test_suffix_in_language(self, suffix: str, language: str) -> float:
        """Test suffix effectiveness in specific language."""
        try:
            # In production, test suffix across multiple harmful prompts in language
            # Compute bypass rate = fraction of attempts that succeeded
            # For now, return mock rate based on language
            import random
            return random.random() * 0.8  # 0-80% bypass rate
        except Exception as e:
            logger.warning("Language-specific testing failed: {}", e)
            return 0.0
