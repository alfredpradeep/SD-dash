"""
Development / Demo SCL Engine.

Drop-in replacement for SCLEngine that runs without heavy ML dependencies
(torch, transformers, sentence-transformers, amrlib, spacy).

Simulates the full compression pipeline with realistic latency and scores.
Returns rich pipeline trace data for the visualization UI.

Usage:
    COMPRESS_DEV_MODE=1 python -m compress.main
"""

import asyncio
import hashlib
import math
import random
import re
import time
import unicodedata
from loguru import logger
from compress.config import Config
from compress.lattice.structures import CompressionResult


# ── Language-aware word splitters ─────────────────────────────
def split_words(text, language):
    """Split text into semantic units respecting language rules."""
    if language in ("ja", "zh"):
        # CJK: each character cluster is roughly a word
        return [c for c in text if not c.isspace()]
    elif language in ("ta", "hi", "bn", "te", "ml", "mr", "gu", "pa", "ur"):
        # Indic: split on spaces but also on certain punctuation
        return [w for w in re.split(r'[\s,;।\u0964]+', text) if w]
    elif language in ("ar",):
        return [w for w in re.split(r'[\s,;،]+', text) if w]
    elif language in ("ko",):
        return [w for w in re.split(r'[\s,;.]+', text) if w]
    else:
        return text.split()


# ── Semantic node extraction (simulated) ──────────────────────
NODE_TYPES = ["entity", "event", "attribute", "relation", "negation",
              "quantifier", "temporal", "modal", "sentiment", "condition"]

EDGE_TYPES = ["agent", "patient", "theme", "location", "time", "manner",
              "cause", "purpose", "instrument", "source", "goal", "mod"]


def extract_simulated_graph(text, language):
    """
    Build a plausible semantic graph from text.
    Uses word-level heuristics to assign node types.
    """
    words = split_words(text, language)
    nodes = []
    edges = []

    # Assign node types based on position and character patterns
    for i, word in enumerate(words):
        if len(word) < 2:
            continue

        # Heuristic node type assignment
        clean = word.strip(".,;:!?\"'()[]{}。、！？")
        if not clean:
            continue

        # Detect node type
        if i == 0 or (clean[0].isupper() if clean[0].isascii() else False):
            ntype = "entity"
        elif any(c.isdigit() for c in clean):
            ntype = "quantifier"
        elif clean.lower() in ("not", "no", "never", "neither", "nor", "இல்லை", "नहीं", "لا", "ない"):
            ntype = "negation"
        elif clean.lower() in ("was", "were", "had", "ago", "yesterday", "tomorrow", "before", "after"):
            ntype = "temporal"
        elif clean.lower() in ("can", "could", "may", "might", "should", "must", "would"):
            ntype = "modal"
        elif clean.lower() in ("good", "bad", "great", "terrible", "excellent", "poor", "happy", "sad"):
            ntype = "sentiment"
        elif clean.lower() in ("if", "unless", "when", "while", "although"):
            ntype = "condition"
        elif i < len(words) * 0.3:
            ntype = "entity"
        elif i < len(words) * 0.5:
            ntype = "event"
        else:
            ntype = random.choice(["attribute", "relation", "entity"])

        conf = round(random.uniform(0.78, 0.98), 3)
        nodes.append({
            "id": f"n{i}",
            "type": ntype,
            "value": clean[:30],
            "confidence": conf,
            "intensity": round(random.uniform(0.5, 1.0), 2),
            "specificity": round(random.uniform(0.4, 0.95), 2),
            "source": random.choice(["amr", "nli", "ner"]),
            "x": None,  # Will be set by frontend
            "y": None,
        })

    # Create edges between adjacent and related nodes
    for i in range(len(nodes) - 1):
        etype = EDGE_TYPES[i % len(EDGE_TYPES)]
        edges.append({
            "source": nodes[i]["id"],
            "target": nodes[i + 1]["id"],
            "type": etype,
            "weight": round(random.uniform(0.6, 1.0), 2),
        })
        # Add some cross-connections for graph richness
        if i + 2 < len(nodes) and random.random() > 0.6:
            edges.append({
                "source": nodes[i]["id"],
                "target": nodes[i + 2]["id"],
                "type": random.choice(EDGE_TYPES),
                "weight": round(random.uniform(0.4, 0.8), 2),
            })

    return {"nodes": nodes, "edges": edges}


def generate_embedding_viz(original, compressed, similarity):
    """Generate 2D embedding coordinates for visualization."""
    # Simulate a 2D PCA/t-SNE projection
    # Original at center, compressed nearby based on similarity
    angle = random.uniform(0, 2 * math.pi)
    distance = (1.0 - similarity) * 3.0  # Closer = more similar

    orig_x, orig_y = 0.0, 0.0
    comp_x = distance * math.cos(angle)
    comp_y = distance * math.sin(angle)

    # Generate some "other" points in the embedding space for context
    context_points = []
    for i in range(8):
        a = random.uniform(0, 2 * math.pi)
        d = random.uniform(1.5, 4.0)
        context_points.append({
            "x": d * math.cos(a),
            "y": d * math.sin(a),
            "label": f"candidate_{i+1}",
            "similarity": round(random.uniform(0.65, 0.89), 3),
        })

    return {
        "original": {"x": orig_x, "y": orig_y, "label": "original"},
        "compressed": {"x": comp_x, "y": comp_y, "label": "compressed"},
        "context_points": context_points,
        "similarity_radius": 1.0 - similarity,
    }


def generate_candidates_trace(compressed_text, num_candidates):
    """Generate simulated beam search candidates."""
    candidates = []
    words = compressed_text.split() if compressed_text else ["..."]
    for i in range(num_candidates):
        # Create variations
        if len(words) > 3:
            variant = " ".join(random.sample(words, min(len(words), len(words) - random.randint(0, 2))))
        else:
            variant = compressed_text

        candidates.append({
            "rank": i + 1,
            "text": variant[:80] + ("..." if len(variant) > 80 else ""),
            "stes_score": round(random.uniform(0.8, 3.5), 3),
            "passed": i == 0,  # Only first candidate passes
            "semantic_sim": round(random.uniform(0.85, 0.98), 3),
            "graph_jacc": round(random.uniform(0.82, 0.96), 3),
        })

    # Sort by STES descending
    candidates.sort(key=lambda c: c["stes_score"], reverse=True)
    for i, c in enumerate(candidates):
        c["rank"] = i + 1
        c["passed"] = i == 0

    return candidates


class DevSCLEngine:
    """
    Simulated SCL Engine for development and UI testing.
    Returns rich pipeline trace data for visualization.
    """

    SUPPORTED_LANGUAGES = {
        "en", "ta", "hi", "ar", "ja", "zh", "ko", "pt", "es",
        "fr", "de", "id", "ms", "bn", "ur", "te", "ml", "pa", "gu", "mr",
    }

    SUPPORTED_TOKENIZERS = {
        "gpt-4o": "cl100k_base",
        "gpt-4o-mini": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "claude-3-5-sonnet": "anthropic",
        "claude-3-haiku": "anthropic",
        "claude-3-opus": "anthropic",
        "llama-3-8b": "sentencepiece_llama3",
        "llama-3-70b": "sentencepiece_llama3",
        "mistral-7b": "sentencepiece_mistral",
        "mistral-8x7b": "sentencepiece_mistral",
    }

    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "that", "which", "who",
        "whom", "this", "these", "those", "very", "really", "quite",
        "just", "also", "then", "than", "both", "each", "every", "all",
        "most", "other", "some", "such", "only", "own", "same", "so",
        "too", "as", "but", "or", "yet", "for", "nor", "not", "at",
        "by", "from", "in", "into", "of", "on", "to", "with",
    }

    ABBREVIATIONS = {
        "natural language processing": "NLP",
        "machine learning": "ML",
        "artificial intelligence": "AI",
        "deep learning": "DL",
        "state-of-the-art": "SOTA",
        "state of the art": "SOTA",
        "large language model": "LLM",
        "large language models": "LLMs",
        "application programming interface": "API",
        "semantic compression lattice": "SCL",
    }

    # Whole-word removable particles/stopwords per language
    REMOVABLE_WORDS = {
        "ta": {"என்று", "ஆகும்", "உள்ள", "அந்த", "இந்த", "மற்றும்", "ஒரு",
               "அது", "இது", "அவர்", "எல்லாம்", "போன்ற", "போது", "பற்றி",
               "மிகவும்", "என்ற", "என்பது", "ஆனால்", "ஆகவே", "எனவே",
               "கூட", "மட்டும்", "தான்", "ஆக", "ஏனெனில்"},
        "hi": {"है", "हैं", "का", "की", "के", "में", "एक", "और", "यह", "वह",
               "को", "से", "पर", "ने", "जो", "कि", "या", "भी", "तो", "इस",
               "उस", "जा", "रहा", "रही", "रहे", "गया", "गई", "हुए", "था",
               "थी", "थे", "सभी", "बहुत", "अपने", "किसी", "कोई"},
        "ar": {"في", "من", "إلى", "على", "هو", "هي", "التي", "الذي", "هذا",
               "هذه", "تلك", "ذلك", "عن", "مع", "أو", "ثم", "بعد", "قبل",
               "كان", "كانت", "أن", "لا", "قد", "ما", "لم", "كل"},
        "ja": {"は", "が", "を", "に", "で", "の", "と", "も", "や", "へ",
               "から", "まで", "より", "ため", "こと", "もの", "ない", "ある",
               "いる", "する", "れる", "です", "ます", "した", "その", "この"},
        "zh": {"的", "了", "在", "是", "和", "有", "不", "也", "都", "就",
               "与", "而", "对", "为", "被", "把", "从", "到", "很", "这",
               "那", "所", "以", "可以", "已经", "正在"},
        "ko": {"은", "는", "이", "가", "을", "를", "의", "에", "에서", "으로",
               "와", "과", "도", "만", "부터", "까지", "하는", "하고", "있는"},
        "de": {"der", "die", "das", "ein", "eine", "und", "ist", "im", "von",
               "zu", "den", "dem", "mit", "auf", "für", "an", "auch", "es",
               "als", "nicht", "sich", "aus", "war", "aber", "noch", "wie"},
        "fr": {"le", "la", "les", "un", "une", "des", "de", "du", "est", "et",
               "en", "que", "qui", "dans", "pour", "par", "sur", "avec", "ce",
               "cette", "ces", "son", "sa", "ses", "au", "aux", "pas", "plus"},
        "es": {"el", "la", "los", "las", "un", "una", "de", "en", "es", "y",
               "que", "del", "al", "se", "por", "con", "para", "su", "más",
               "no", "como", "pero", "sus", "le", "ya", "este", "esta"},
        "pt": {"o", "a", "os", "as", "um", "uma", "de", "em", "é", "e",
               "que", "do", "da", "no", "na", "se", "por", "com", "para",
               "não", "mais", "como", "mas", "seu", "sua", "ao", "dos"},
        "bn": {"হয়", "এবং", "একটি", "এই", "সেই", "তার", "যে", "থেকে",
               "জন্য", "করা", "হয়েছে", "আছে", "নেই", "তবে", "কিন্তু"},
        "ur": {"ہے", "اور", "کا", "کی", "میں", "ایک", "سے", "کو", "نے",
               "پر", "تھا", "یہ", "وہ", "بھی", "جو", "ہیں", "کے"},
        "te": {"మరియు", "ఒక", "ఈ", "ఆ", "చేసి", "కూడా", "అయితే", "కానీ",
               "లో", "కు", "నుండి", "తో", "ఇది", "అది"},
        "ml": {"ആണ്", "ഒരു", "ഈ", "അത്", "എന്ന", "കൂടി", "മാത്രം",
               "ആയി", "ആയ", "ഇത്", "പക്ഷേ", "എന്നാല്"},
        "pa": {"ਹੈ", "ਅਤੇ", "ਇੱਕ", "ਦਾ", "ਦੀ", "ਨੂੰ", "ਵਿੱਚ", "ਨੇ",
               "ਤੋਂ", "ਉਹ", "ਇਹ", "ਜੋ", "ਕਿ", "ਦੇ"},
        "gu": {"છે", "અને", "એક", "આ", "તે", "માટે", "પણ", "કે",
               "થી", "માં", "ને", "પર", "જે"},
        "mr": {"आहे", "आणि", "एक", "हा", "ती", "या", "त्या", "ते",
               "की", "पण", "मात्र", "करून", "असे"},
        "id": {"yang", "dan", "di", "itu", "ini", "dengan", "untuk", "dari",
               "pada", "ke", "oleh", "juga", "akan", "telah", "sudah",
               "tidak", "ada", "dapat", "adalah", "atau", "sebagai", "mereka"},
        "ms": {"yang", "dan", "di", "itu", "ini", "dengan", "untuk", "dari",
               "pada", "ke", "oleh", "juga", "akan", "telah", "tidak",
               "ada", "adalah", "atau", "sebagai"},
    }

    # Suffix patterns to strip from agglutinative language words
    # These are common case markers, postpositions, and inflectional suffixes
    SUFFIX_PATTERNS = {
        "ta": [
            "களுக்கு", "கின்றன", "கின்றது", "களுக்கு", "களிலேயே",
            "படுகின்றன", "ப்படுகின்றன", "ப்படும்", "க்கப்படும்",
            "படையிலேயே", "யிலேயே", "களில்", "களுக்கு", "க்கான",
            "த்தில்", "த்தின்", "த்தை", "களை", "கள்", "க்கு",
            "யில்", "இல்", "ஆல்", "ஆக", "ஆன", "உடன்",
            "ன்று", "ின்", "த்து", "ப்பு",
        ],
        "hi": ["ियों", "ाओं", "ियां", "ताओं", "पन", "वाला", "वाली",
               "वाले", "कर", "ना", "ने", "ता", "ती", "ते"],
        "bn": ["গুলি", "গুলো", "দের", "তে", "তার", "কে", "রা"],
        "te": ["లకు", "లో", "గా", "తో", "కు", "ని", "లు"],
        "ml": ["കള്", "ത്തി", "ത്തെ", "യുടെ", "ത്തില്", "ന്റെ"],
        "mr": ["ांना", "ांचे", "ांची", "ातील", "ामध्ये"],
    }

    def __init__(self, config: Config):
        self.config = config
        self._try_tiktoken = None
        logger.warning(
            "DEV MODE — Using simulated compression engine. "
            "Set COMPRESS_DEV_MODE=0 and install ML deps for production."
        )

    def _estimate_tokens(self, text: str, tokenizer: str) -> int:
        """Estimate token count. Uses tiktoken if available, else heuristic."""
        if self._try_tiktoken is None:
            try:
                import tiktoken
                self._try_tiktoken = True
            except ImportError:
                self._try_tiktoken = False

        if self._try_tiktoken:
            import tiktoken
            try:
                enc = tiktoken.get_encoding("cl100k_base")
                return len(enc.encode(text))
            except Exception:
                pass

        if any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff'
               or '\uac00' <= c <= '\ud7af' for c in text):
            return max(1, len(text) // 2)
        return max(1, len(text.split()) + len(text) // 10)

    def _strip_suffixes(self, word: str, language: str) -> str:
        """Strip inflectional suffixes from agglutinative language words."""
        suffixes = self.SUFFIX_PATTERNS.get(language, [])
        if not suffixes:
            return word

        # Try longest suffix first
        sorted_sfx = sorted(suffixes, key=len, reverse=True)
        for sfx in sorted_sfx:
            if word.endswith(sfx) and len(word) > len(sfx) + 2:
                return word[: -len(sfx)]
        return word

    def _compress_text(self, text: str, language: str) -> str:
        """
        Multi-pass language-aware compression.

        Pass 1: Abbreviation substitution (English) or whole-word removal
        Pass 2: Suffix stripping for agglutinative languages
        Pass 3: Duplicate/redundant word merging
        Pass 4: Structural trimming (remove every Nth low-info word) if still needed
        """
        if language == "en":
            result = text
            # Abbreviation pass
            for phrase, abbr in self.ABBREVIATIONS.items():
                result = re.sub(re.escape(phrase), abbr, result, flags=re.IGNORECASE)
            # Stopword removal
            words = result.split()
            filtered = []
            for i, word in enumerate(words):
                clean = word.strip(".,;:!?\"'()[]").lower()
                if i == 0 or i == len(words) - 1 or clean not in self.STOPWORDS:
                    filtered.append(word)
            result = re.sub(r'\s+', ' ', " ".join(filtered)).strip()
            return result if len(result) < len(text) * 0.92 else text[:int(len(text) * 0.75)]

        # ── Non-English multi-pass compression ──
        sep = "" if language in ("ja", "zh") else " "
        words = split_words(text, language)

        if len(words) <= 2:
            return text

        # Pass 1: Remove whole-word stopwords/particles
        removable = self.REMOVABLE_WORDS.get(language, set())
        if removable:
            words = [w for w in words if w not in removable]

        # Pass 2: Suffix stripping (agglutinative languages)
        if language in self.SUFFIX_PATTERNS:
            words = [self._strip_suffixes(w, language) for w in words]

        # Pass 3: Deduplicate consecutive identical stems
        deduped = [words[0]] if words else []
        for i in range(1, len(words)):
            if words[i] != words[i - 1]:
                deduped.append(words[i])
        words = deduped

        # Check if we've compressed enough (compare char length as proxy)
        compressed = sep.join(words)
        ratio = len(compressed) / max(len(text), 1)

        # Pass 4: If not enough compression, structurally trim
        # Remove every 3rd word from the middle (keep first 2 and last 2)
        if ratio > 0.82 and len(words) > 5:
            keep = []
            keep.append(words[0])
            keep.append(words[1])
            mid = words[2:-2]
            for i, w in enumerate(mid):
                if i % 3 != 0:  # keep 2 out of every 3
                    keep.append(w)
            keep.append(words[-2])
            keep.append(words[-1])
            words = keep
            compressed = sep.join(words)
            ratio = len(compressed) / max(len(text), 1)

        # Pass 5: Last resort — if still too long, truncate to ~70% of words
        if ratio > 0.82 and len(words) > 4:
            target_len = max(3, int(len(words) * 0.65))
            # Keep first word, last word, and evenly sample the rest
            if target_len < len(words):
                indices = [0]
                step = (len(words) - 1) / max(target_len - 1, 1)
                for i in range(1, target_len - 1):
                    indices.append(int(i * step))
                indices.append(len(words) - 1)
                words = [words[i] for i in sorted(set(indices))]
                compressed = sep.join(words)

        return compressed

    async def compress(
        self,
        text: str,
        source_language: str,
        target_tokenizer: str = "gpt-4o",
        customer_id: str = "default",
        min_reduction_threshold: float = 0.15,
        semantic_threshold: float = 0.91,
        graph_threshold: float = 0.88,
    ) -> CompressionResult:
        """Simulate compression with realistic timing and scores."""
        start_ms = time.monotonic() * 1000

        if source_language not in self.SUPPORTED_LANGUAGES:
            from compress.exceptions import UnsupportedLanguageError
            raise UnsupportedLanguageError(f"Language '{source_language}' not supported")
        if target_tokenizer not in self.SUPPORTED_TOKENIZERS:
            from compress.exceptions import UnsupportedLanguageError
            raise UnsupportedLanguageError(f"Tokenizer '{target_tokenizer}' not supported")

        await asyncio.sleep(random.uniform(0.05, 0.2))

        original_tokens = self._estimate_tokens(text, target_tokenizer)
        compressed_text = self._compress_text(text, source_language)
        compressed_tokens = self._estimate_tokens(compressed_text, target_tokenizer)

        reduction_ratio = 1.0 - (compressed_tokens / max(original_tokens, 1))

        if original_tokens <= 3:
            processing_ms = (time.monotonic() * 1000) - start_ms
            return CompressionResult(
                original_text=text, compressed_text=text,
                source_language=source_language, target_tokenizer=target_tokenizer,
                original_token_count=original_tokens, compressed_token_count=original_tokens,
                reduction_ratio=0.0, semantic_similarity=1.0, graph_jaccard=1.0,
                stes_score=0.0, compression_applied=False,
                rejection_reason="Input too short for compression (3 tokens or fewer)",
                processing_ms=processing_ms, candidates_evaluated=0,
            )

        candidates_evaluated = random.randint(3, 8)
        semantic_sim = round(random.uniform(max(semantic_threshold, 0.91), 0.98), 4)
        graph_jacc = round(random.uniform(max(graph_threshold, 0.88), 0.96), 4)
        stes = round((graph_jacc ** 1.0) / (max(compressed_tokens / original_tokens, 0.01) ** 0.65), 4)
        processing_ms = (time.monotonic() * 1000) - start_ms

        # Check if reduction meets threshold
        if reduction_ratio < min_reduction_threshold:
            return CompressionResult(
                original_text=text, compressed_text=text,
                source_language=source_language, target_tokenizer=target_tokenizer,
                original_token_count=original_tokens, compressed_token_count=original_tokens,
                reduction_ratio=round(reduction_ratio, 4),
                semantic_similarity=semantic_sim, graph_jaccard=graph_jacc,
                stes_score=0.0, compression_applied=False,
                rejection_reason=f"Reduction {reduction_ratio*100:.1f}% below threshold {min_reduction_threshold*100:.0f}%",
                processing_ms=round(processing_ms, 1), candidates_evaluated=candidates_evaluated,
            )

        return CompressionResult(
            original_text=text, compressed_text=compressed_text,
            source_language=source_language, target_tokenizer=target_tokenizer,
            original_token_count=original_tokens, compressed_token_count=compressed_tokens,
            reduction_ratio=round(reduction_ratio, 4),
            semantic_similarity=semantic_sim, graph_jaccard=graph_jacc,
            stes_score=round(stes, 4), compression_applied=True, rejection_reason=None,
            processing_ms=round(processing_ms, 1), candidates_evaluated=candidates_evaluated,
        )

    async def compress_with_trace(
        self,
        text: str,
        source_language: str,
        target_tokenizer: str = "gpt-4o",
        customer_id: str = "default",
        min_reduction_threshold: float = 0.15,
        semantic_threshold: float = 0.91,
        graph_threshold: float = 0.88,
    ) -> dict:
        """
        Compress with full pipeline trace for visualization.
        Returns result + semantic graph + embedding viz + candidates.
        """
        result = await self.compress(
            text, source_language, target_tokenizer, customer_id,
            min_reduction_threshold, semantic_threshold, graph_threshold,
        )

        # Build pipeline trace
        graph = extract_simulated_graph(text, source_language)
        embedding_viz = generate_embedding_viz(
            text, result.compressed_text, result.semantic_similarity
        )
        candidates = generate_candidates_trace(
            result.compressed_text, result.candidates_evaluated
        )

        # Pipeline stages with timing
        total_ms = result.processing_ms
        stages = [
            {
                "name": "Semantic Graph Extraction",
                "subtitle": "AMR + NLI + NER hybrid pipeline",
                "duration_ms": round(total_ms * 0.4, 1),
                "status": "complete",
                "details": {
                    "nodes_extracted": len(graph["nodes"]),
                    "edges_extracted": len(graph["edges"]),
                    "amr_confidence": round(random.uniform(0.82, 0.95), 3),
                    "nli_confidence": round(random.uniform(0.78, 0.92), 3),
                    "ner_confidence": round(random.uniform(0.85, 0.97), 3),
                },
            },
            {
                "name": "Beam Search",
                "subtitle": f"mT5 generation, beam_width={self.config.beam_width}",
                "duration_ms": round(total_ms * 0.45, 1),
                "status": "complete",
                "details": {
                    "candidates_generated": result.candidates_evaluated,
                    "beam_width": self.config.beam_width,
                    "best_stes": result.stes_score,
                    "tokenizer": target_tokenizer,
                },
            },
            {
                "name": "Triple Verification Gate",
                "subtitle": "LaBSE + Jaccard + Reduction",
                "duration_ms": round(total_ms * 0.15, 1),
                "status": "complete" if result.compression_applied else "rejected",
                "details": {
                    "labse_score": result.semantic_similarity,
                    "labse_threshold": semantic_threshold,
                    "labse_passed": result.semantic_similarity >= semantic_threshold,
                    "jaccard_score": result.graph_jaccard,
                    "jaccard_threshold": graph_threshold,
                    "jaccard_passed": result.graph_jaccard >= graph_threshold,
                    "reduction_score": result.reduction_ratio,
                    "reduction_threshold": min_reduction_threshold,
                    "reduction_passed": result.reduction_ratio >= min_reduction_threshold,
                },
            },
        ]

        return {
            "result": {
                "original_text": result.original_text,
                "compressed_text": result.compressed_text,
                "language": result.source_language,
                "target_tokenizer": result.target_tokenizer,
                "original_tokens": result.original_token_count,
                "compressed_tokens": result.compressed_token_count,
                "reduction_ratio": result.reduction_ratio,
                "semantic_similarity": result.semantic_similarity,
                "graph_jaccard": result.graph_jaccard,
                "stes_score": result.stes_score,
                "compression_applied": result.compression_applied,
                "rejection_reason": result.rejection_reason,
                "processing_ms": result.processing_ms,
                "candidates_evaluated": result.candidates_evaluated,
            },
            "pipeline": {
                "stages": stages,
                "total_ms": result.processing_ms,
            },
            "semantic_graph": graph,
            "embedding_visualization": embedding_viz,
            "beam_candidates": candidates,
        }

    async def batch_compress(self, texts: list, customer_id: str = "default", concurrency: int = 10) -> list:
        semaphore = asyncio.Semaphore(concurrency)
        async def compress_one(item: dict) -> CompressionResult:
            async with semaphore:
                return await self.compress(
                    text=item["text"], source_language=item["language"],
                    target_tokenizer=item.get("tokenizer", "gpt-4o"), customer_id=customer_id,
                )
        return await asyncio.gather(*[compress_one(item) for item in texts])

    async def health_check(self) -> dict:
        return {
            "status": "healthy (dev mode)", "extractor": "simulated",
            "searcher": "simulated", "gate": "simulated", "cache": "disabled",
        }
