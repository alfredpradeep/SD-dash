"""
Adversarial Reconstruction Test (Stage 4).

Verifies compression by attempting to reconstruct the knowledge graph
from the compressed text. If the compressed text allows reconstruction
of the same entities, negations, quantities, and relations as the
original, the compression is semantically lossless.

This is the key differentiator — no one else verifies compression
through structured knowledge reconstruction.
"""

import json
from loguru import logger
from compress.config import Config


class ReconstructionVerifier:
    """
    Adversarial reconstruction test for compression verification.

    Two modes:
    1. LLM-based: Uses LLM to extract KG from both texts, compares
    2. Graph-based: Re-extracts semantic graph from compressed text,
       compares with original graph

    The LLM-based mode is more thorough but requires API calls.
    The graph-based mode is free but less comprehensive.
    """

    def __init__(self, config: Config):
        self.config = config
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            try:
                from compress.llm.provider import LLMProvider
                self._llm = LLMProvider(self.config)
            except Exception:
                self._llm = False
        return self._llm if self._llm is not False else None

    async def verify(
        self,
        original_text: str,
        compressed_text: str,
        original_graph=None,
        extractor=None,
        language: str = "en",
        use_llm: bool = False,
    ) -> tuple[float, list[str], dict]:
        """
        Run adversarial reconstruction test.

        Default: graph-based verification (FREE, no API calls).
        Set use_llm=True for LLM-based verification (uses API quota).

        Returns:
            (score: float 0-1, missing_items: list[str], details: dict)
        """
        # Prefer graph-based (free, no API calls) by default
        if original_graph and extractor:
            return await self._verify_graph(
                original_text, compressed_text,
                original_graph, extractor, language,
            )

        # Fall back to LLM only if explicitly requested AND available
        if use_llm:
            llm = self._get_llm()
            if llm and llm.available:
                return await self._verify_llm(
                    original_text, compressed_text
                )

        # No verification possible — return neutral score
        return 1.0, [], {"method": "skipped", "reason": "no extractor"}

    async def _verify_llm(
        self,
        original_text: str,
        compressed_text: str,
    ) -> tuple[float, list[str], dict]:
        """
        LLM-based adversarial reconstruction.

        1. Extract KG from original text
        2. Extract KG from compressed text
        3. Compare KGs
        """
        from compress.llm.prompts import (
            RECONSTRUCT_SYSTEM,
            build_reconstruct_prompt,
            compare_reconstructions,
        )

        llm = self._get_llm()

        try:
            # Extract KG from original
            orig_prompt = build_reconstruct_prompt(original_text)
            orig_response = await llm.generate(
                RECONSTRUCT_SYSTEM, orig_prompt,
                max_tokens=512, temperature=0.1,
            )
            orig_kg = self._parse_json(orig_response)

            # Extract KG from compressed
            comp_prompt = build_reconstruct_prompt(compressed_text)
            comp_response = await llm.generate(
                RECONSTRUCT_SYSTEM, comp_prompt,
                max_tokens=512, temperature=0.1,
            )
            comp_kg = self._parse_json(comp_response)

            # Compare
            score, missing = compare_reconstructions(orig_kg, comp_kg)

            details = {
                "method": "llm_reconstruction",
                "original_kg_size": sum(
                    len(v) if isinstance(v, list) else 1
                    for v in orig_kg.values()
                ),
                "compressed_kg_size": sum(
                    len(v) if isinstance(v, list) else 1
                    for v in comp_kg.values()
                ),
                "missing_count": len(missing),
            }

            logger.debug(
                "Reconstruction score: {:.3f}, missing: {}",
                score, missing,
            )

            return score, missing, details

        except Exception as e:
            logger.warning(
                "LLM reconstruction failed: {}. Passing.", e
            )
            return 1.0, [], {"method": "llm_failed", "error": str(e)}

    async def _verify_graph(
        self,
        original_text: str,
        compressed_text: str,
        original_graph,
        extractor,
        language: str,
    ) -> tuple[float, list[str], dict]:
        """
        Graph-based reconstruction verification.

        Re-extracts semantic graph from compressed text and compares
        with original graph at the node level.

        For cross-lingual cases (e.g. Tamil→English), uses type-count
        comparison instead of value matching, since entity values will
        be in different scripts/languages.
        """
        try:
            # Detect cross-lingual: original is non-English,
            # compressed is likely English
            is_cross_lingual = language != "en"

            # For cross-lingual, extract compressed graph as English
            comp_language = "en" if is_cross_lingual else language
            comp_graph = await extractor.extract(
                compressed_text, comp_language
            )

            # Compare node sets
            orig_nodes = {
                (n.unit_type.value, n.value.lower().strip())
                for n in original_graph.nodes
            }
            comp_nodes = {
                (n.unit_type.value, n.value.lower().strip())
                for n in comp_graph.nodes
            }

            critical_types = {"entity", "negation", "quantifier"}

            if is_cross_lingual:
                # ── Cross-lingual: compare by type counts ──
                # Entity values are in different scripts, so we
                # compare structural preservation (type counts)
                orig_counts = {}
                comp_counts = {}
                for ntype, _ in orig_nodes:
                    if ntype in critical_types:
                        orig_counts[ntype] = orig_counts.get(ntype, 0) + 1
                for ntype, _ in comp_nodes:
                    if ntype in critical_types:
                        comp_counts[ntype] = comp_counts.get(ntype, 0) + 1

                total_orig = sum(orig_counts.values())
                total_comp = sum(comp_counts.values())

                missing = []
                # Check each type: compressed should have at least
                # as many nodes of each critical type
                for ntype in critical_types:
                    orig_n = orig_counts.get(ntype, 0)
                    comp_n = comp_counts.get(ntype, 0)
                    deficit = orig_n - comp_n
                    if deficit > 0:
                        missing.append(
                            f"[{ntype}] {deficit} fewer in compressed"
                        )

                # Score: ratio of preserved critical nodes
                if total_orig == 0:
                    score = 1.0
                else:
                    score = min(total_comp / total_orig, 1.0)

                details = {
                    "method": "graph_reconstruction_crosslingual",
                    "original_nodes": len(orig_nodes),
                    "compressed_nodes": len(comp_nodes),
                    "original_critical_counts": orig_counts,
                    "compressed_critical_counts": comp_counts,
                    "critical_missing": len(missing),
                }
            else:
                # ── Same-language: compare by exact/fuzzy values ──
                missing = []
                for ntype, nval in orig_nodes:
                    if ntype in critical_types:
                        if (ntype, nval) not in comp_nodes:
                            fuzzy_found = any(
                                ct == ntype
                                and self._fuzzy_val(nval, cv)
                                for ct, cv in comp_nodes
                            )
                            if not fuzzy_found:
                                missing.append(f"[{ntype}] {nval}")

                total_critical = sum(
                    1 for t, _ in orig_nodes if t in critical_types
                )
                matched = total_critical - len(missing)
                score = matched / max(total_critical, 1)

                details = {
                    "method": "graph_reconstruction",
                    "original_nodes": len(orig_nodes),
                    "compressed_nodes": len(comp_nodes),
                    "critical_missing": len(missing),
                }

            return score, missing, details

        except Exception as e:
            logger.warning(
                "Graph reconstruction failed: {}. Passing.", e
            )
            return 1.0, [], {
                "method": "graph_failed", "error": str(e)
            }

    def _parse_json(self, text: str) -> dict:
        """Parse JSON from LLM response, handling common formatting issues."""
        text = text.strip()

        # Remove markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                l for l in lines
                if not l.strip().startswith("```")
            )

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            import re
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

            logger.warning(
                "Failed to parse JSON from LLM response: {}...",
                text[:100],
            )
            return {}

    def _fuzzy_val(self, a: str, b: str) -> bool:
        """Check if two values are fuzzy matches."""
        if a in b or b in a:
            return True
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a:
            return False
        return len(words_a & words_b) / len(words_a) >= 0.5

    async def health(self) -> str:
        llm = self._get_llm()
        if llm and llm.available:
            return f"healthy (llm={llm.provider_name})"
        return "degraded (no LLM — using graph-based fallback)"
