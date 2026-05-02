"""
Weisfeiler-Leman Graph Kernel for Structural Compliance Analysis.

Parses dependency trees and computes WL subtree kernels to detect
structural alignment between requests and responses.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from loguru import logger

from shield.exceptions import JudgeError
from shield.judge.structures import WLResult


class DependencyNode:
    """Represents a node in dependency parse tree."""

    def __init__(self, word: str, pos: str, dep_label: str, idx: int):
        """Initialize node."""
        self.word = word
        self.pos = pos
        self.dep_label = dep_label
        self.idx = idx
        self.children = []
        self.parent = None

    def add_child(self, child: "DependencyNode") -> None:
        """Add child node."""
        self.children.append(child)
        child.parent = self

    def __repr__(self) -> str:
        return f"DependencyNode({self.word}:{self.dep_label})"


class WLStructureAnalyzer:
    """
    Graph kernel analysis using Weisfeiler-Leman algorithm.

    Detects structural compliance by comparing dependency tree kernels
    between prompts and responses.
    """

    def __init__(self):
        """Initialize analyzer."""
        logger.debug("WLStructureAnalyzer initialized")

    async def analyze(
        self, prompt_text: str, response_text: str, language: str = "en"
    ) -> WLResult:
        """
        Analyze structural compliance using WL kernel.

        Args:
            prompt_text: Original prompt
            response_text: Model response
            language: Language code

        Returns:
            WLResult with kernel value and structural match

        Raises:
            JudgeError: If analysis fails
        """
        try:
            # Parse dependency trees
            prompt_tree = await self._parse_dependency_tree(prompt_text, language)
            response_tree = await self._parse_dependency_tree(response_text, language)

            if prompt_tree is None or response_tree is None:
                raise JudgeError("Failed to parse dependency trees")

            # Compute WL kernel
            kernel_value = self._compute_wl_kernel(prompt_tree, response_tree, iterations=3)

            # Detect structural compliance
            structural_match = self._detect_structural_compliance(kernel_value)

            # Check if causal chains are preserved
            causal_preserved = self._check_causal_chain_preservation(
                prompt_tree, response_tree
            )

            result = WLResult(
                kernel_value=float(kernel_value),
                structural_match=structural_match,
                causal_chain_preserved=causal_preserved,
                wl_iterations=3,
                tree1_subtrees=self._count_subtrees(prompt_tree),
                tree2_subtrees=self._count_subtrees(response_tree),
                language=language,
                timestamp=datetime.utcnow().isoformat(),
                structural_similarity=float(kernel_value) / max(
                    self._count_subtrees(prompt_tree),
                    self._count_subtrees(response_tree),
                    1,
                ),
            )

            logger.info(
                f"WL kernel analysis: kernel={kernel_value:.3f}, "
                f"match={structural_match}, causal_preserved={causal_preserved}"
            )
            return result

        except Exception as e:
            logger.error(f"WL structure analysis failed: {e}")
            raise JudgeError(f"WL kernel analysis failed: {e}") from e

    async def _parse_dependency_tree(
        self, text: str, language: str
    ) -> Optional[DependencyNode]:
        """
        Parse text to dependency tree.

        Args:
            text: Text to parse
            language: Language code

        Returns:
            Root node of dependency tree or None

        Note:
            In production, use spaCy with multilingual models.
            For now, returns synthetic tree.
        """
        try:
            if not text or len(text) < 3:
                return None

            # Synthetic tree construction for demo
            # In production: use spaCy nlp.pipe()
            words = text.split()[:10]  # Limit to first 10 words

            if not words:
                return None

            # Create root node
            root = DependencyNode(
                word=words[0],
                pos="NOUN",
                dep_label="ROOT",
                idx=0,
            )

            # Add child nodes
            for idx, word in enumerate(words[1:], 1):
                child = DependencyNode(
                    word=word,
                    pos="VERB" if idx % 2 == 0 else "NOUN",
                    dep_label="nsubj" if idx % 2 == 0 else "obj",
                    idx=idx,
                )
                root.add_child(child)

            return root

        except Exception as e:
            logger.warning(f"Dependency parsing failed: {e}")
            return None

    def _compute_wl_kernel(
        self,
        tree1: DependencyNode,
        tree2: DependencyNode,
        iterations: int = 3,
    ) -> float:
        """
        Compute Weisfeiler-Leman subtree kernel.

        Args:
            tree1: First dependency tree
            tree2: Second dependency tree
            iterations: Number of WL iterations

        Returns:
            Kernel value (similarity measure)
        """
        try:
            # Initialize subtree hashes
            hashes1 = self._compute_subtree_hashes(tree1, iterations)
            hashes2 = self._compute_subtree_hashes(tree2, iterations)

            # Compute kernel as similarity between hash multisets
            kernel = self._kernel_from_hashes(hashes1, hashes2)
            return float(kernel)

        except Exception as e:
            logger.warning(f"WL kernel computation failed: {e}")
            return 0.0

    def _compute_subtree_hashes(
        self, tree: DependencyNode, iterations: int
    ) -> List[Dict[int, int]]:
        """
        Compute subtree hash multisets for each WL iteration.

        Args:
            tree: Root node of dependency tree
            iterations: Number of iterations

        Returns:
            List of hash multisets, one per iteration
        """
        try:
            hash_multisets = []
            current_hashes = defaultdict(int)

            # Iteration 0: leaf hashes based on node labels
            def init_hash(node: DependencyNode) -> int:
                return hash((node.word, node.pos, node.dep_label)) % (2**31)

            # Collect initial hashes
            def collect_init(node: DependencyNode) -> None:
                h = init_hash(node)
                current_hashes[h] += 1
                for child in node.children:
                    collect_init(child)

            collect_init(tree)
            hash_multisets.append(dict(current_hashes))

            # WL iterations
            for iteration in range(1, iterations):
                new_hashes = defaultdict(int)

                def update_hash(node: DependencyNode) -> int:
                    # Hash is based on node's label and children's hashes
                    child_hashes = sorted(
                        [update_hash(child) for child in node.children]
                    )
                    combined = (node.word, node.pos, tuple(child_hashes))
                    h = hash(combined) % (2**31)
                    new_hashes[h] += 1
                    return h

                update_hash(tree)
                hash_multisets.append(dict(new_hashes))
                current_hashes = new_hashes

            return hash_multisets

        except Exception as e:
            logger.warning(f"Subtree hash computation failed: {e}")
            return []

    def _kernel_from_hashes(
        self, hashes1: List[Dict[int, int]], hashes2: List[Dict[int, int]]
    ) -> float:
        """
        Compute kernel value from hash multisets.

        Args:
            hashes1: Hash multisets from tree 1
            hashes2: Hash multisets from tree 2

        Returns:
            Kernel value
        """
        try:
            kernel = 0.0

            # Sum over iterations
            for h1, h2 in zip(hashes1, hashes2):
                # Find common hashes
                common_keys = set(h1.keys()) & set(h2.keys())
                iteration_kernel = sum(
                    min(h1[k], h2[k]) for k in common_keys
                )
                kernel += iteration_kernel

            return kernel

        except Exception as e:
            logger.warning(f"Kernel computation from hashes failed: {e}")
            return 0.0

    def _detect_structural_compliance(self, kernel_value: float) -> bool:
        """
        Determine if high kernel value indicates structural compliance.

        High kernel value = trees are structurally similar = likely compliance.

        Args:
            kernel_value: WL kernel value

        Returns:
            True if structural match detected
        """
        # Threshold: if kernel > 0.3, consider structurally aligned
        return kernel_value > 0.3

    def _check_causal_chain_preservation(
        self, tree1: DependencyNode, tree2: DependencyNode
    ) -> bool:
        """
        Check if causal chain structure is preserved.

        Args:
            tree1: Prompt tree
            tree2: Response tree

        Returns:
            True if causal relationships preserved
        """
        try:
            # Extract causal patterns from both trees
            causals1 = self._extract_causal_patterns(tree1)
            causals2 = self._extract_causal_patterns(tree2)

            if not causals1:
                # If no causal structure in prompt, no need to preserve
                return True

            # Check if any causal patterns from prompt appear in response
            for causal_label in causals1:
                if causal_label in causals2:
                    return True

            return False

        except Exception as e:
            logger.warning(f"Causal chain check failed: {e}")
            return False

    def _extract_causal_patterns(self, tree: DependencyNode) -> Set[str]:
        """Extract causal dependency patterns from tree."""
        try:
            patterns = set()

            def traverse(node: DependencyNode) -> None:
                if "caus" in node.dep_label.lower() or "mark" in node.dep_label.lower():
                    patterns.add(node.dep_label)
                for child in node.children:
                    traverse(child)

            traverse(tree)
            return patterns

        except Exception as e:
            logger.warning(f"Causal pattern extraction failed: {e}")
            return set()

    def _count_subtrees(self, tree: DependencyNode) -> int:
        """Count total subtrees in tree."""
        try:
            count = 1
            for child in tree.children:
                count += self._count_subtrees(child)
            return count

        except Exception as e:
            logger.warning(f"Subtree counting failed: {e}")
            return 1
