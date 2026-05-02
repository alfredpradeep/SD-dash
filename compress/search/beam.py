"""
LLM-powered semantic compression search (v3.0).

Stage 2 of the COMPRESS pipeline.

Uses an LLM API to generate intelligent rewrites, then scores each
candidate with LaBSE cross-lingual embeddings. The LLM proposes
token-efficient rephrasings; the verification gate disposes of any
that don't meet semantic preservation thresholds.

Falls back to rule-based compression when no LLM is available.
"""

import re
import itertools
import numpy as np
from sentence_transformers import SentenceTransformer
from compress.lattice.structures import (
    SemanticGraph,
    CompressionCandidate,
    GraphNode,
    SemanticUnitType,
)
from compress.search.scorer import STESScorer
from compress.config import Config
from compress.exceptions import SearchError
from loguru import logger


# ── Language-specific function words (safe to remove in fallback mode) ──
REMOVABLE_WORDS = {
    "en": {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "that", "which", "who",
        "whom", "this", "these", "those", "very", "really", "quite",
        "just", "also", "then", "than", "both", "each", "every", "all",
        "most", "other", "some", "such", "only", "own", "same", "so",
        "too", "as", "but", "or", "yet", "for", "nor", "not", "at",
        "by", "from", "in", "into", "of", "on", "to", "with", "it",
    },
    "ta": {
        "இந்த", "அந்த", "ஒரு", "மற்றும்", "என்று", "ஆகும்", "உள்ள",
        "அது", "இது", "எல்லாம்", "போன்ற", "போது", "பற்றி", "மிகவும்",
        "என்ற", "என்பது", "ஆனால்", "ஆகவே", "எனவே", "கூட", "மட்டும்",
        "தான்", "ஆக", "ஏனெனில்", "அவர்", "அவர்கள்",
    },
    "hi": {
        "है", "हैं", "का", "की", "के", "में", "एक", "और", "यह", "वह",
        "को", "से", "पर", "ने", "जो", "कि", "या", "भी", "तो", "इस",
        "उस", "था", "थी", "थे", "सभी", "बहुत", "अपने", "किसी", "कोई",
    },
    "ar": {
        "في", "من", "إلى", "على", "هو", "هي", "التي", "الذي", "هذا",
        "هذه", "تلك", "ذلك", "عن", "مع", "أو", "ثم", "بعد", "قبل",
        "كان", "كانت", "أن", "لا", "قد", "ما", "لم", "كل",
    },
    "ja": {
        "は", "が", "を", "に", "で", "の", "と", "も", "や", "へ",
        "から", "まで", "より", "ため", "こと", "もの", "です", "ます",
    },
    "zh": {
        "的", "了", "在", "是", "和", "有", "不", "也", "都", "就",
        "与", "而", "对", "为", "被", "把", "从", "到", "很", "这",
        "那", "所", "以", "可以", "已经", "正在",
    },
    "ko": {
        "은", "는", "이", "가", "을", "를", "의", "에", "에서", "으로",
        "와", "과", "도", "만", "부터", "까지", "하는", "하고",
    },
    "de": {
        "der", "die", "das", "ein", "eine", "und", "ist", "im", "von",
        "zu", "den", "dem", "mit", "auf", "für", "an", "auch", "es",
    },
    "fr": {
        "le", "la", "les", "un", "une", "des", "de", "du", "est", "et",
        "en", "que", "qui", "dans", "pour", "par", "sur", "avec",
    },
    "es": {
        "el", "la", "los", "las", "un", "una", "de", "en", "es", "y",
        "que", "del", "al", "se", "por", "con", "para", "su",
    },
    "pt": {
        "o", "a", "os", "as", "um", "uma", "de", "em", "é", "e",
        "que", "do", "da", "no", "na", "se", "por", "com",
    },
    "bn": {"হয়", "এবং", "একটি", "এই", "সেই", "তার", "যে", "থেকে"},
    "ur": {"ہے", "اور", "کا", "کی", "میں", "ایک", "سے", "کو"},
    "te": {"మరియు", "ఒక", "ఈ", "ఆ", "చేసి", "కూడా", "అయితే", "కానీ"},
    "ml": {"ആണ്", "ഒരു", "ഈ", "അത്", "എന്ന", "കൂടി", "മാത്രം"},
    "pa": {"ਹੈ", "ਅਤੇ", "ਇੱਕ", "ਦਾ", "ਦੀ", "ਨੂੰ", "ਵਿੱਚ", "ਨੇ"},
    "gu": {"છે", "અને", "એક", "આ", "તે", "માટે", "પણ", "કે"},
    "mr": {"आहे", "आणि", "एक", "हा", "ती", "या", "त्या", "ते"},
    "id": {
        "yang", "dan", "di", "itu", "ini", "dengan", "untuk", "dari",
        "pada", "ke", "oleh", "juga", "akan", "telah", "sudah",
        "tidak", "ada", "dapat", "adalah", "atau", "sebagai",
    },
    "ms": {
        "yang", "dan", "di", "itu", "ini", "dengan", "untuk", "dari",
        "pada", "ke", "oleh", "juga", "akan", "telah", "tidak",
    },
}


def _graph_summary(graph: SemanticGraph) -> str:
    """Build a human-readable summary of the semantic graph for LLM prompts."""
    parts = []
    for node in graph.nodes:
        label = node.unit_type.value.upper()
        val = node.value
        extra = ""
        if node.intensity is not None:
            extra += f" (intensity={node.intensity:.2f})"
        if node.certainty is not None:
            extra += f" (certainty={node.certainty:.2f})"
        parts.append(f"- {label}: {val}{extra}")

    if not parts:
        return "No semantic units extracted"
    return "\n".join(parts)


class BeamSearcher:
    """
    LLM-powered semantic compression search.

    Primary: LLM generates intelligent rewrites.
    Fallback: Rule-based word removal (when no LLM available).
    Both: Scored by LaBSE for semantic similarity.
    """

    def __init__(self, config: Config):
        self.config = config
        self.stes = STESScorer(config)

        # LaBSE for cross-lingual semantic similarity
        logger.info("Loading LaBSE model...")
        self.labse = SentenceTransformer(
            "sentence-transformers/LaBSE", device=config.device
        )
        logger.info("LaBSE loaded successfully")

        # LLM provider (lazy-loaded)
        self._llm = None

    def _get_llm(self):
        """Lazy-load LLM provider."""
        if self._llm is None:
            try:
                from compress.llm.provider import LLMProvider
                self._llm = LLMProvider(self.config)
            except Exception as e:
                logger.warning("LLM provider init failed: {}", e)
                self._llm = False  # sentinel: tried and failed
        return self._llm if self._llm is not False else None

    def count_tokens(self, text: str, tokenizer_name: str) -> int:
        return self.stes.count_tokens(text, tokenizer_name)

    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute LaBSE cosine similarity between two texts."""
        embeddings = self.labse.encode(
            [text_a, text_b], normalize_embeddings=True
        )
        return float(np.dot(embeddings[0], embeddings[1]))

    async def search(
        self,
        graph: SemanticGraph,
        target_tokenizer: str,
        beam_width: int = 12,
        max_candidates: int = 8,
    ) -> list[CompressionCandidate]:
        """
        Generate and score compression candidates.

        Tries LLM generation first. Falls back to rule-based
        if no LLM is available.
        """
        try:
            text = graph.source_text
            language = graph.source_language
            original_tokens = self.count_tokens(text, target_tokenizer)

            # Try LLM-powered generation first
            llm = self._get_llm()
            if llm and llm.available:
                raw_candidates = await self._generate_llm(
                    text, language, graph, target_tokenizer
                )
                logger.info(
                    "LLM generated {} candidates", len(raw_candidates)
                )
            else:
                raw_candidates = []

            # Add rule-based candidates as supplementary
            rule_candidates = self._generate_rule_based(
                text, language, graph
            )
            raw_candidates.extend(rule_candidates)

            if not raw_candidates:
                logger.warning(
                    "No candidates generated for: {}...", text[:60]
                )
                return []

            # Score all candidates with LaBSE
            return await self._score_candidates(
                text, language, target_tokenizer,
                original_tokens, raw_candidates, max_candidates,
            )

        except Exception as e:
            logger.exception("Beam search failed")
            raise SearchError(f"Beam search failed: {e}") from e

    async def _generate_llm(
        self,
        text: str,
        language: str,
        graph: SemanticGraph,
        target_tokenizer: str,
    ) -> list[str]:
        """
        Generate candidates using a SINGLE LLM API call.

        For non-English: forces translation to English (3-4x token savings).
        For English: generates shorter rephrasings.
        """
        from compress.llm.prompts import parse_variations

        llm = self._get_llm()
        candidates = []
        graph_summary = _graph_summary(graph)

        if language != "en":
            # ── NON-ENGLISH: Translate to English (the big win) ──
            combined_system = (
                "You are a translation and compression engine. "
                "Your job: translate non-English text into concise "
                "English that preserves ALL meaning.\n\n"
                "CRITICAL RULES:\n"
                "1. Output MUST be in English\n"
                "2. PRESERVE every entity, number, name, date\n"
                "3. PRESERVE all negations (not, never, no)\n"
                "4. PRESERVE all relationships and actions\n"
                "5. Use the shortest English phrasing possible\n"
                "6. Keep proper nouns as-is (transliterate if needed)\n"
                "7. Return ONLY numbered translations, no explanations"
            )

            combined_user = (
                f"Translate this {language} text into concise English. "
                f"Give 5 different English versions, from shortest to "
                f"longest. ALL versions must be in English.\n\n"
                f"INPUT ({language}):\n{text}\n\n"
                f"KEY INFORMATION TO PRESERVE:\n{graph_summary}\n\n"
                f"Return exactly 5 English translations, numbered 1-5.\n"
                f"REMEMBER: Output must be ENGLISH, not {language}:"
            )
        else:
            # ── ENGLISH: Compress/rephrase ──
            combined_system = (
                "You are a text compression engine. Rewrite text to "
                "use fewer words while preserving ALL meaning.\n\n"
                "RULES:\n"
                "1. PRESERVE all entities, numbers, names, negations\n"
                "2. NEVER drop negation or change quantities\n"
                "3. Remove filler words, redundancies, verbose phrases\n"
                "4. Combine sentences where possible\n"
                "5. Return ONLY numbered rewrites, no explanations"
            )

            combined_user = (
                f"Rewrite this text in 5 shorter versions. "
                f"Order from most compressed to least.\n\n"
                f"INPUT:\n{text}\n\n"
                f"SEMANTIC UNITS TO PRESERVE:\n{graph_summary}\n\n"
                f"Return exactly 5 rewrites, numbered 1-5:"
            )

        try:
            response = await llm.generate(
                combined_system, combined_user,
                max_tokens=1024, temperature=0.3,
            )
            if response:
                parsed = parse_variations(response, n=5)
                candidates.extend(parsed)
                logger.info(
                    "LLM generated {} candidates in single call "
                    "(lang={})",
                    len(parsed), language,
                )
        except Exception as e:
            logger.warning("LLM candidate generation failed: {}", e)

        return candidates

    def _generate_rule_based(
        self,
        text: str,
        language: str,
        graph: SemanticGraph,
    ) -> list[str]:
        """
        Fallback: Generate candidates through rule-based word removal.

        Used as supplementary candidates even when LLM is available.
        """
        candidates = set()
        words = text.split()

        if len(words) <= 3:
            return []

        # Identify protected words (entities, numbers, negation)
        protected = set()
        for node in graph.nodes:
            if node.unit_type in (
                SemanticUnitType.ENTITY,
                SemanticUnitType.NEGATION,
                SemanticUnitType.QUANTIFIER,
            ):
                protected.add(node.value.lower().strip())

        removable = REMOVABLE_WORDS.get(language, set())

        # Remove all function words at once
        removable_indices = []
        for i, word in enumerate(words):
            clean = word.strip(".,;:!?\"'()[]{}。、！？").lower()
            if clean in removable and clean not in protected:
                removable_indices.append(i)

        if removable_indices:
            filtered = [
                w for i, w in enumerate(words)
                if i not in removable_indices
            ]
            if len(filtered) >= max(2, len(words) * 0.4):
                candidates.add(" ".join(filtered))

        # Progressive removal by importance
        word_scores = []
        for i, word in enumerate(words):
            clean = word.strip(".,;:!?\"'()[]{}。、！？").lower()
            if clean in protected:
                score = 10.0
            elif any(c.isdigit() for c in word):
                score = 8.0
            elif clean in removable:
                score = 1.0
            elif i == 0 or i == len(words) - 1:
                score = 5.0
            else:
                score = 3.0
            word_scores.append((i, score))

        word_scores.sort(key=lambda x: x[1])
        for n_remove in range(1, max(2, len(words) // 3)):
            if n_remove > len(word_scores):
                break
            remove_set = set(idx for idx, _ in word_scores[:n_remove])
            filtered = [
                w for i, w in enumerate(words)
                if i not in remove_set
            ]
            if len(filtered) >= max(2, len(words) * 0.3):
                candidates.add(" ".join(filtered))
            if len(candidates) > 10:
                break

        candidates.discard(text)
        candidates.discard("")
        return list(candidates)

    def _is_non_latin(self, text: str) -> bool:
        """Check if text is predominantly non-Latin script."""
        if not text:
            return False
        non_latin = sum(
            1 for c in text
            if c.isalpha() and ord(c) > 0x024F  # beyond Latin Extended
        )
        alpha = sum(1 for c in text if c.isalpha())
        return alpha > 0 and (non_latin / alpha) > 0.5

    async def _score_candidates(
        self,
        original_text: str,
        language: str,
        target_tokenizer: str,
        original_tokens: int,
        raw_candidates: list[str],
        max_candidates: int,
    ) -> list[CompressionCandidate]:
        """Score all candidates with LaBSE and STES."""
        # Encode original once
        original_emb = self.labse.encode(
            [original_text], normalize_embeddings=True
        )[0]

        # Deduplicate and filter
        seen = set()
        unique_candidates = []
        for c in raw_candidates:
            c = c.strip()
            if not c:
                continue
            norm = c.lower().strip()
            if norm in seen or norm == original_text.lower().strip():
                continue
            # For cross-lingual: reject candidates still in source
            # language (non-Latin script when source is non-English)
            if language != "en" and self._is_non_latin(c):
                logger.debug(
                    "Rejecting non-English candidate: {}...", c[:40]
                )
                continue
            seen.add(norm)
            unique_candidates.append(c)

        if not unique_candidates:
            return []

        # Batch encode all candidates
        cand_embeddings = self.labse.encode(
            unique_candidates, normalize_embeddings=True,
            batch_size=32, show_progress_bar=False,
        )

        candidates = []
        for rank, (cand_text, cand_emb) in enumerate(
            zip(unique_candidates, cand_embeddings)
        ):
            # LaBSE semantic similarity
            semantic_sim = float(np.dot(original_emb, cand_emb))

            candidate_tokens = self.count_tokens(
                cand_text, target_tokenizer
            )

            # Use semantic similarity as proxy for graph preservation
            graph_preservation = semantic_sim

            stes = self.stes.compute_stes(
                graph_preservation=graph_preservation,
                token_count=candidate_tokens,
                source_language=language,
                baseline_token_count=original_tokens,
            )

            candidates.append(
                CompressionCandidate(
                    text=cand_text,
                    token_count=candidate_tokens,
                    stes_score=stes,
                    semantic_similarity=semantic_sim,
                    graph_jaccard=graph_preservation,
                    rank=rank,
                )
            )

        # Sort by STES descending
        candidates.sort(key=lambda c: c.stes_score, reverse=True)

        logger.info(
            "Scored {} candidates, best STES={:.3f}, best sim={:.3f}",
            len(candidates),
            candidates[0].stes_score if candidates else 0,
            candidates[0].semantic_similarity if candidates else 0,
        )

        return candidates[:max_candidates]

    async def health(self) -> str:
        llm = self._get_llm()
        llm_status = (
            await llm.health() if llm else "no LLM provider"
        )
        return f"healthy (labse=loaded, llm={llm_status})"
