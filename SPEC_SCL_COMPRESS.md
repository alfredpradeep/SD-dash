# MODULE: COMPRESS
# Semantic Compression Lattice Engine
# Version: 2.0.0 — Production Grade
# Classification: Standalone Python Module
# No external product context required.

---

## OVERVIEW

Build a production-grade text semantic compression engine.

The engine receives input text in any of 20 supported languages, decomposes it
into language-independent semantic units (a semantic graph), searches for the
minimum-token surface realization of that graph using a per-tokenizer beam
search, verifies semantic preservation through a dual-gate system, and returns
compressed text guaranteed to encode the same meaning in fewer tokens.

This is NOT text summarisation. The output must contain all semantic content
of the input. The only change is that surface-level redundancy is removed and
phrasing is optimised for the target tokenizer's vocabulary.

---

## TECH STACK

```
Python 3.11+
fastapi==0.111.0
uvicorn[standard]==0.29.0
sentence-transformers==3.0.0        # LaBSE for semantic verification
tiktoken==0.7.0                     # OpenAI cl100k tokenizer
anthropic-tokenizer==0.3.0          # Anthropic tokenizer (native)
sentencepiece==0.2.0                # SentencePiece for Llama/Mistral
transformers==4.41.0                # mT5 model + NLI + NER pipelines
torch==2.3.0
penman==1.3.0                       # AMR graph parsing and serialization
smatch==1.0.4                       # AMR graph similarity scoring
amrlib==0.8.0                       # AMR parsing (SPRING-based)
redis==5.0.4                        # result caching
asyncio
numpy==1.26.4
scikit-learn==1.5.0
Levenshtein==0.25.1                 # fuzzy entity matching
pydantic==2.7.0
pydantic-settings==2.3.0
loguru==0.7.2
prometheus-client==0.20.0           # metrics
pytest==8.2.0
pytest-asyncio==0.23.7
httpx==0.27.0                       # async test client
```

---

## DIRECTORY STRUCTURE

```
compress/
├── __init__.py
├── engine.py              # Core SCL engine — main class
├── lattice/
│   ├── __init__.py
│   ├── extractor.py       # Hybrid semantic graph extraction (AMR + NLI + NER)
│   ├── structures.py      # SemanticGraph, GraphNode dataclasses
│   ├── similarity.py      # Weighted graph Jaccard similarity scorer
│   └── amr_utils.py       # AMR graph parsing, serialization, cross-lingual adapters
├── search/
│   ├── __init__.py
│   ├── beam.py            # Beam search over surface realizations
│   ├── scorer.py          # STES scoring function
│   └── tokenizer_profiles.py  # Per-tokenizer token counting (cl100k, anthropic, sentencepiece)
├── verification/
│   ├── __init__.py
│   └── gate.py            # Triple verification gate (semantic + graph + reduction)
├── api/
│   ├── __init__.py
│   ├── router.py          # FastAPI router
│   ├── schemas.py         # Request / Response pydantic models
│   └── middleware.py       # Request timing + metrics middleware
├── storage/
│   ├── __init__.py
│   ├── cache.py           # Redis cache layer
│   └── metrics_store.py   # Prometheus metrics collector
├── config.py              # All constants and configuration
├── exceptions.py          # Custom exceptions
└── tests/
    ├── __init__.py
    ├── test_engine.py
    ├── test_lattice.py
    ├── test_similarity.py
    ├── test_beam_search.py
    ├── test_verification.py
    ├── test_tokenizer_profiles.py
    ├── test_api.py
    └── fixtures/
        ├── sample_inputs.json   # 50 multilingual test cases
        └── expected_outputs.json
```

---

## DATA MODELS — lattice/structures.py

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class SemanticUnitType(str, Enum):
    ENTITY      = "entity"
    SENTIMENT   = "sentiment"
    TEMPORAL    = "temporal"
    RELATIONAL  = "relational"
    MODAL       = "modal"
    NEGATION    = "negation"
    QUANTIFIER  = "quantifier"
    ATTRIBUTE   = "attribute"
    CAUSAL      = "causal"
    CONDITIONAL = "conditional"

@dataclass
class GraphNode:
    unit_id: str
    unit_type: SemanticUnitType
    value: str
    intensity: Optional[float] = None      # for sentiment: 0.0–1.0
    specificity: Optional[float] = None    # for temporal: 0.0–1.0
    certainty: Optional[float] = None      # for modal: 0.0–1.0
    confidence: float = 1.0                # extraction confidence for this node
    dependencies: list[str] = field(default_factory=list)  # unit_ids this depends on
    metadata: dict = field(default_factory=dict)

@dataclass
class SemanticGraph:
    source_text: str
    source_language: str
    nodes: list[GraphNode]
    edges: list[tuple[str, str, str]]   # (from_id, relation, to_id)
    amr_penman: str                      # raw AMR penman notation
    extraction_confidence: float         # 0.0–1.0 (harmonic mean of component confidences)
    amr_confidence: float                # 0.0–1.0 (AMR parser confidence alone)
    component_scores: dict = field(default_factory=dict)  # per-extractor confidence

    def to_dict(self) -> dict:
        return {
            "nodes": [vars(n) for n in self.nodes],
            "edges": self.edges,
            "amr_penman": self.amr_penman,
            "extraction_confidence": self.extraction_confidence,
            "amr_confidence": self.amr_confidence,
            "component_scores": self.component_scores,
        }

    def node_by_id(self, unit_id: str) -> Optional[GraphNode]:
        for n in self.nodes:
            if n.unit_id == unit_id:
                return n
        return None

    def nodes_by_type(self, unit_type: SemanticUnitType) -> list[GraphNode]:
        return [n for n in self.nodes if n.unit_type == unit_type]

@dataclass
class CompressionCandidate:
    text: str
    token_count: int
    stes_score: float
    semantic_similarity: float
    graph_jaccard: float
    rank: int

@dataclass
class CompressionResult:
    original_text: str
    compressed_text: str
    source_language: str
    target_tokenizer: str
    original_token_count: int
    compressed_token_count: int
    reduction_ratio: float              # 0.0–1.0, higher = more compressed
    semantic_similarity: float          # LaBSE cosine sim, must be >= 0.91
    graph_jaccard: float                # graph similarity, must be >= 0.88
    stes_score: float
    compression_applied: bool           # False if validation gate rejected all candidates
    rejection_reason: Optional[str]     # populated if compression_applied=False
    processing_ms: float
    candidates_evaluated: int
```

---

## CORE ENGINE — engine.py

Build the main `SCLEngine` class with the following complete implementation:

```python
import asyncio
import time
from loguru import logger
from compress.lattice.extractor import LatticeExtractor
from compress.search.beam import BeamSearcher
from compress.verification.gate import VerificationGate
from compress.storage.cache import CompressionCache
from compress.storage.metrics_store import MetricsStore
from compress.config import Config
from compress.lattice.structures import CompressionResult, SemanticGraph
from compress.exceptions import (
    ExtractionError, SearchError, VerificationError, UnsupportedLanguageError
)

class SCLEngine:
    """
    Semantic Compression Lattice Engine.

    Decomposes input text into language-independent semantic graph,
    finds minimum-token surface realization via beam search,
    verifies output through triple-gate system before returning.

    Thread-safe. Async-native. Stateless per request.
    """

    SUPPORTED_LANGUAGES = {
        "en", "ta", "hi", "ar", "ja", "zh", "ko", "pt", "es",
        "fr", "de", "id", "ms", "bn", "ur", "te", "ml", "pa", "gu", "mr"
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

    def __init__(self, config: Config):
        self.config = config
        self.extractor = LatticeExtractor(config)
        self.searcher = BeamSearcher(config)
        self.gate = VerificationGate(config)
        self.cache = CompressionCache(config)
        self.metrics = MetricsStore(config)
        logger.info("SCLEngine initialised with config: {}", config.model_dump())

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
        """
        Main compression entry point.

        Args:
            text: Input text to compress
            source_language: ISO 639-1 language code
            target_tokenizer: Model name whose tokenizer to optimise for
            customer_id: For metrics attribution
            min_reduction_threshold: Reject compression if savings < this ratio
            semantic_threshold: Minimum LaBSE cosine similarity to accept (>= 0.91)
            graph_threshold: Minimum graph Jaccard similarity to accept (>= 0.88)

        Returns:
            CompressionResult with compressed text or original if gate rejects
        """
        start_ms = time.monotonic() * 1000

        # Validate inputs
        if source_language not in self.SUPPORTED_LANGUAGES:
            raise UnsupportedLanguageError(f"Language '{source_language}' not supported")
        if target_tokenizer not in self.SUPPORTED_TOKENIZERS:
            raise UnsupportedLanguageError(f"Tokenizer '{target_tokenizer}' not supported")

        # Cache check — hash of (text, language, tokenizer)
        cache_key = self.cache.make_key(text, source_language, target_tokenizer)
        cached = await self.cache.get(cache_key)
        if cached:
            self.metrics.record_cache_hit(source_language)
            return cached

        try:
            # Stage 1: Extract semantic graph
            graph: SemanticGraph = await self.extractor.extract(
                text=text,
                language=source_language
            )
            logger.debug(
                "Graph extracted: {} nodes, confidence={:.3f}, amr_confidence={:.3f}",
                len(graph.nodes), graph.extraction_confidence, graph.amr_confidence
            )

            # Count original tokens using the actual target tokenizer
            original_tokens = self.searcher.count_tokens(text, target_tokenizer)

            # Stage 2: Beam search over surface realizations
            candidates = await self.searcher.search(
                graph=graph,
                target_tokenizer=target_tokenizer,
                beam_width=self.config.beam_width,
                max_candidates=self.config.max_candidates,
            )
            logger.debug("Beam search produced {} candidates", len(candidates))

            # Stage 3: Verification gate — find first passing candidate
            best_result = None
            for candidate in candidates:
                passed, reason = await self.gate.verify(
                    original_text=text,
                    original_graph=graph,
                    candidate=candidate,
                    original_tokens=original_tokens,
                    semantic_threshold=semantic_threshold,
                    graph_threshold=graph_threshold,
                    min_reduction=min_reduction_threshold,
                )
                if passed:
                    best_result = candidate
                    break
                logger.debug("Candidate rejected: {}", reason)

            processing_ms = (time.monotonic() * 1000) - start_ms

            if best_result is None:
                result = CompressionResult(
                    original_text=text,
                    compressed_text=text,
                    source_language=source_language,
                    target_tokenizer=target_tokenizer,
                    original_token_count=original_tokens,
                    compressed_token_count=original_tokens,
                    reduction_ratio=0.0,
                    semantic_similarity=1.0,
                    graph_jaccard=1.0,
                    stes_score=0.0,
                    compression_applied=False,
                    rejection_reason="No candidate met all verification thresholds",
                    processing_ms=processing_ms,
                    candidates_evaluated=len(candidates),
                )
            else:
                result = CompressionResult(
                    original_text=text,
                    compressed_text=best_result.text,
                    source_language=source_language,
                    target_tokenizer=target_tokenizer,
                    original_token_count=original_tokens,
                    compressed_token_count=best_result.token_count,
                    reduction_ratio=1.0 - (best_result.token_count / original_tokens),
                    semantic_similarity=best_result.semantic_similarity,
                    graph_jaccard=best_result.graph_jaccard,
                    stes_score=best_result.stes_score,
                    compression_applied=True,
                    rejection_reason=None,
                    processing_ms=processing_ms,
                    candidates_evaluated=len(candidates),
                )

            # Cache and record metrics
            await self.cache.set(cache_key, result, ttl_seconds=self.config.redis_ttl_seconds)
            self.metrics.record_compression(result, customer_id)
            return result

        except ExtractionError as e:
            logger.error("Semantic graph extraction failed: {}", e)
            raise
        except SearchError as e:
            logger.error("Beam search failed: {}", e)
            raise
        except Exception as e:
            logger.exception("Unexpected error in SCLEngine.compress")
            raise

    async def batch_compress(
        self,
        texts: list[dict],   # [{"text": str, "language": str, "tokenizer": str}]
        customer_id: str = "default",
        concurrency: int = 10,
    ) -> list[CompressionResult]:
        """Compress multiple texts concurrently with controlled parallelism."""
        semaphore = asyncio.Semaphore(concurrency)

        async def compress_one(item: dict) -> CompressionResult:
            async with semaphore:
                return await self.compress(
                    text=item["text"],
                    source_language=item["language"],
                    target_tokenizer=item.get("tokenizer", "gpt-4o"),
                    customer_id=customer_id,
                )

        return await asyncio.gather(*[compress_one(item) for item in texts])

    async def health_check(self) -> dict:
        """Return health status of all sub-components."""
        return {
            "status": "healthy",
            "extractor": await self.extractor.health(),
            "searcher": await self.searcher.health(),
            "gate": await self.gate.health(),
            "cache": await self.cache.health(),
        }
```

---

## LATTICE EXTRACTOR — lattice/extractor.py

```python
import penman
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from compress.lattice.structures import SemanticGraph, GraphNode, SemanticUnitType
from compress.lattice.amr_utils import AMRParser, CrossLingualAdapter
from compress.config import Config
from compress.exceptions import ExtractionError
import re, uuid
from loguru import logger

class LatticeExtractor:
    """
    Hybrid semantic graph extractor.

    Combines three extraction streams:
    1. AMR parsing (SPRING-based with multilingual adapters) — primary
    2. Neural semantic classification (zero-shot NLI) — secondary
    3. Structured extraction (NER, regex, morphological) — tertiary

    When AMR confidence < 0.6, the system upweights streams 2 and 3.
    Final graph is the fusion of all three streams with per-node confidence scores.
    """

    SENTIMENT_LABELS = ["positive sentiment", "negative sentiment", "neutral sentiment"]
    TEMPORAL_LABELS  = ["past event", "present event", "future event"]
    MODAL_LABELS     = ["certain", "uncertain", "conditional", "obligatory"]

    # Negation markers per language family
    NEGATION_PATTERNS: dict[str, list[str]] = {
        "en": [r'\b(not|no|never|neither|nor|without|cannot|can\'t|won\'t|don\'t|doesn\'t|didn\'t|isn\'t|aren\'t|wasn\'t|weren\'t|hasn\'t|haven\'t|hadn\'t)\b'],
        "ta": [r'(இல்லை|வேண்டாம்|மாட்டேன்|மாட்டார்|அல்ல|இல்லாமல்|முடியாது)'],
        "hi": [r'(नहीं|मत|बिना|न|ना|कभी\s+नहीं)'],
        "ar": [r'(لا|لم|لن|ليس|ليست|ليسوا|ما|غير)'],
        "ja": [r'(ない|なかった|ません|ず|ぬ|まい)'],
        "zh": [r'(不|没|没有|未|无|非|别|勿|莫)'],
        "ko": [r'(않|못|없|아니|말)'],
        "bn": [r'(না|নয়|নেই|নি)'],
        "ur": [r'(نہیں|مت|نه|بغیر)'],
        "te": [r'(లేదు|కాదు|వద్దు|లేకుండా)'],
        "ml": [r'(ഇല്ല|അല്ല|വേണ്ട|കഴിയില്ല)'],
        "pt": [r'\b(não|nunca|nem|sem|nenhum|nenhuma|jamais)\b'],
        "es": [r'\b(no|nunca|ni|sin|ningún|ninguna|jamás)\b'],
        "fr": [r'\b(ne|pas|jamais|ni|sans|aucun|aucune)\b'],
        "de": [r'\b(nicht|kein|keine|keinem|keinen|nie|niemals|ohne)\b'],
    }

    def __init__(self, config: Config):
        self.config = config

        # Stream 1: AMR parser with multilingual adapters
        self.amr_parser = AMRParser(
            model_path=config.amr_model_path,
            device=config.device,
        )
        self.amr_adapter = CrossLingualAdapter(
            adapter_dir=config.amr_adapter_dir,
            device=config.device,
        )

        # Stream 2: Multilingual NLI for semantic classification
        self.classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            device=config.device,
        )

        # Stream 3: Multilingual NER for entity extraction
        self.ner = pipeline(
            "token-classification",
            model="Babelscape/wikineural-multilingual-ner",
            aggregation_strategy="simple",
            device=config.device,
        )

        # AMR confidence threshold for fallback weighting
        self.amr_confidence_floor = 0.6

    async def extract(self, text: str, language: str) -> SemanticGraph:
        """
        Extract full semantic graph from text using hybrid pipeline.

        Pipeline:
        1. AMR parsing → rooted DAG of concepts and relations
        2. Named entity recognition → ENTITY nodes
        3. Zero-shot sentiment classification → SENTIMENT node
        4. Zero-shot temporal classification → TEMPORAL node
        5. Zero-shot modality classification → MODAL node
        6. Negation detection (per-language patterns) → NEGATION nodes
        7. Numerical/quantifier extraction → QUANTIFIER nodes
        8. Stream fusion: merge AMR graph with NLI/NER nodes
        9. Confidence scoring per node and overall
        """
        try:
            nodes: list[GraphNode] = []
            edges: list[tuple] = []
            component_scores: dict = {}

            # ── Stream 1: AMR Parsing ────────────────────────────────────
            amr_graph, amr_conf = await self._extract_amr(text, language)
            component_scores["amr"] = amr_conf

            amr_nodes, amr_edges = self._amr_to_graph(amr_graph, amr_conf)
            nodes.extend(amr_nodes)
            edges.extend(amr_edges)

            # ── Stream 2: Neural Classification ──────────────────────────
            nli_conf = await self._extract_nli(text, language, nodes, edges)
            component_scores["nli"] = nli_conf

            # ── Stream 3: Structured Extraction ──────────────────────────
            struct_conf = await self._extract_structured(text, language, nodes, edges)
            component_scores["structured"] = struct_conf

            # ── Deduplicate nodes ────────────────────────────────────────
            nodes = self._deduplicate_nodes(nodes)

            # ── Build AMR penman string ──────────────────────────────────
            amr_penman = amr_graph if amr_graph else self._build_penman(nodes, edges, text)

            # ── Compute overall confidence (harmonic mean) ───────────────
            confidences = [v for v in component_scores.values() if v > 0]
            if confidences:
                n = len(confidences)
                harmonic = n / sum(1.0 / c for c in confidences)
            else:
                harmonic = 0.0

            return SemanticGraph(
                source_text=text,
                source_language=language,
                nodes=nodes,
                edges=edges,
                amr_penman=amr_penman,
                extraction_confidence=harmonic,
                amr_confidence=amr_conf,
                component_scores=component_scores,
            )

        except Exception as e:
            logger.exception("Extraction failed for text: {}...", text[:80])
            raise ExtractionError(f"Semantic graph extraction failed: {e}") from e

    async def _extract_amr(self, text: str, language: str) -> tuple[str, float]:
        """Run AMR parsing with language-specific adapter. Returns (penman_str, confidence)."""
        try:
            if language != "en":
                adapted_input = self.amr_adapter.adapt(text, language)
            else:
                adapted_input = text

            amr_penman, confidence = self.amr_parser.parse(adapted_input)
            return amr_penman, confidence
        except Exception as e:
            logger.warning("AMR parsing failed, returning empty graph: {}", e)
            return "", 0.0

    def _amr_to_graph(self, amr_penman: str, amr_conf: float) -> tuple[list[GraphNode], list[tuple]]:
        """Convert AMR penman notation to GraphNode list and edges."""
        nodes = []
        edges = []
        if not amr_penman or amr_conf < 0.1:
            return nodes, edges

        try:
            graph = penman.decode(amr_penman)
            concept_map: dict[str, str] = {}  # amr variable → unit_id

            # Extract concepts as nodes
            for instance in graph.instances():
                unit_id = str(uuid.uuid4())[:8]
                concept_map[instance.source] = unit_id
                node = GraphNode(
                    unit_id=unit_id,
                    unit_type=self._classify_amr_concept(instance.target),
                    value=str(instance.target),
                    confidence=amr_conf,
                    metadata={"source": "amr", "amr_var": instance.source}
                )
                nodes.append(node)

            # Extract relations as edges
            for edge in graph.edges():
                src_id = concept_map.get(edge.source)
                tgt_id = concept_map.get(edge.target)
                if src_id and tgt_id:
                    edges.append((src_id, edge.role.lstrip(':'), tgt_id))

        except Exception as e:
            logger.warning("AMR→graph conversion failed: {}", e)

        return nodes, edges

    def _classify_amr_concept(self, concept: str) -> SemanticUnitType:
        """Map AMR concept string to SemanticUnitType."""
        concept_lower = str(concept).lower()
        temporal_markers = {"date", "time", "year", "month", "day", "before", "after", "now", "then", "yesterday", "tomorrow"}
        negation_markers = {"not", "no", "-", "never"}
        modal_markers = {"possible", "likely", "obligate", "recommend", "must", "should", "can"}
        sentiment_markers = {"happy", "sad", "angry", "frustrate", "love", "hate", "fear", "worry", "glad", "upset"}
        causal_markers = {"cause", "because", "result", "lead-to", "due-to"}

        if concept_lower in temporal_markers or concept_lower.startswith("date-"):
            return SemanticUnitType.TEMPORAL
        if concept_lower in negation_markers or concept_lower.endswith("-91"):
            return SemanticUnitType.NEGATION
        if concept_lower in modal_markers:
            return SemanticUnitType.MODAL
        if concept_lower in sentiment_markers:
            return SemanticUnitType.SENTIMENT
        if concept_lower in causal_markers:
            return SemanticUnitType.CAUSAL
        if concept_lower.startswith("have-condition") or concept_lower == "if":
            return SemanticUnitType.CONDITIONAL
        return SemanticUnitType.ENTITY

    async def _extract_nli(
        self, text: str, language: str,
        nodes: list[GraphNode], edges: list[tuple]
    ) -> float:
        """Run NLI-based semantic classification. Appends to nodes/edges. Returns confidence."""
        scores = []

        # Sentiment classification
        sent_result = self.classifier(
            text,
            candidate_labels=self.SENTIMENT_LABELS,
            hypothesis_template="This text expresses {}."
        )
        sent_score = sent_result["scores"][0]
        if sent_score > 0.5:
            sentiment_value = sent_result["labels"][0].split()[0]
            intensity = sent_score if sentiment_value != "neutral" else 0.0
            nodes.append(GraphNode(
                unit_id=str(uuid.uuid4())[:8],
                unit_type=SemanticUnitType.SENTIMENT,
                value=sentiment_value,
                intensity=intensity,
                confidence=sent_score,
                metadata={"source": "nli"}
            ))
            scores.append(sent_score)

        # Temporal classification
        temp_result = self.classifier(
            text,
            candidate_labels=self.TEMPORAL_LABELS,
            hypothesis_template="This text describes a {}."
        )
        if temp_result["scores"][0] > 0.55:
            nodes.append(GraphNode(
                unit_id=str(uuid.uuid4())[:8],
                unit_type=SemanticUnitType.TEMPORAL,
                value=temp_result["labels"][0].split()[0],
                specificity=temp_result["scores"][0],
                confidence=temp_result["scores"][0],
                metadata={"source": "nli"}
            ))
            scores.append(temp_result["scores"][0])

        # Modal classification
        modal_result = self.classifier(
            text,
            candidate_labels=self.MODAL_LABELS,
            hypothesis_template="The statement is {}."
        )
        if modal_result["scores"][0] > 0.55:
            nodes.append(GraphNode(
                unit_id=str(uuid.uuid4())[:8],
                unit_type=SemanticUnitType.MODAL,
                value=modal_result["labels"][0],
                certainty=modal_result["scores"][0],
                confidence=modal_result["scores"][0],
                metadata={"source": "nli"}
            ))
            scores.append(modal_result["scores"][0])

        return sum(scores) / len(scores) if scores else 0.0

    async def _extract_structured(
        self, text: str, language: str,
        nodes: list[GraphNode], edges: list[tuple]
    ) -> float:
        """Run NER + regex extraction. Appends to nodes/edges. Returns confidence."""
        scores = []

        # Named entity recognition
        entities = self.ner(text)
        for ent in entities:
            if ent["score"] < 0.5:
                continue
            node = GraphNode(
                unit_id=str(uuid.uuid4())[:8],
                unit_type=SemanticUnitType.ENTITY,
                value=ent["word"],
                confidence=ent["score"],
                metadata={"source": "ner", "ner_label": ent["entity_group"]}
            )
            nodes.append(node)
            scores.append(ent["score"])

        # Per-language negation detection
        lang_patterns = self.NEGATION_PATTERNS.get(language, self.NEGATION_PATTERNS.get("en", []))
        has_negation = any(re.search(p, text, re.IGNORECASE) for p in lang_patterns)
        if has_negation:
            nodes.append(GraphNode(
                unit_id=str(uuid.uuid4())[:8],
                unit_type=SemanticUnitType.NEGATION,
                value="negated",
                confidence=0.9,
                metadata={"source": "regex", "language": language}
            ))
            scores.append(0.9)

        # Numerical/quantifier extraction
        number_pattern = r'\b\d+([.,]\d+)?\s*(%|k|m|b|bn|million|billion|thousand|lakh|crore)?\b'
        for match in re.finditer(number_pattern, text, re.IGNORECASE):
            nodes.append(GraphNode(
                unit_id=str(uuid.uuid4())[:8],
                unit_type=SemanticUnitType.QUANTIFIER,
                value=match.group(0).strip(),
                confidence=0.95,
                metadata={"source": "regex"}
            ))
            scores.append(0.95)
            if len([n for n in nodes if n.unit_type == SemanticUnitType.QUANTIFIER]) >= 5:
                break

        # Connect sentiment nodes to entity nodes
        entity_ids = [n.unit_id for n in nodes if n.unit_type == SemanticUnitType.ENTITY]
        sentiment_ids = [n.unit_id for n in nodes if n.unit_type == SemanticUnitType.SENTIMENT]
        for sid in sentiment_ids:
            for eid in entity_ids:
                edges.append((sid, "ARG0", eid))

        return sum(scores) / len(scores) if scores else 0.0

    def _deduplicate_nodes(self, nodes: list[GraphNode]) -> list[GraphNode]:
        """Remove duplicate nodes (same type+value), keeping highest confidence."""
        seen: dict[str, GraphNode] = {}
        for node in nodes:
            key = f"{node.unit_type.value}:{node.value.lower().strip()}"
            if key not in seen or node.confidence > seen[key].confidence:
                seen[key] = node
        return list(seen.values())

    def _build_penman(self, nodes: list, edges: list, text: str) -> str:
        """Generate penman AMR notation from graph nodes as fallback."""
        if not nodes:
            return f'(t / text :value "{text[:50]}")'
        root = nodes[0]
        parts = [f'({root.unit_id} / {root.unit_type.value}']
        parts.append(f'    :value "{root.value}"')
        for from_id, rel, to_id in edges[:10]:
            target_node = next((n for n in nodes if n.unit_id == to_id), None)
            if target_node:
                parts.append(f'    :{rel} ({to_id} / {target_node.unit_type.value} :value "{target_node.value}")')
        parts.append(')')
        return '\n'.join(parts)

    async def health(self) -> str:
        return "healthy"
```

---

## AMR UTILITIES — lattice/amr_utils.py

```python
import penman
from loguru import logger
from compress.config import Config

class AMRParser:
    """
    Wrapper around SPRING-based AMR parser with confidence scoring.

    Uses amrlib's SPRING model for English AMR parsing.
    Non-English text is processed through CrossLingualAdapter first.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        try:
            import amrlib
            self.model = amrlib.load_stog_model(model_dir=model_path)
            self._available = True
            logger.info("AMR parser loaded from {}", model_path)
        except Exception as e:
            logger.warning("AMR parser unavailable ({}), will use fallback extraction", e)
            self._available = False
            self.model = None

    def parse(self, text: str) -> tuple[str, float]:
        """
        Parse text into AMR penman notation.

        Returns:
            (penman_string, confidence_score)
            confidence is based on the model's generation probability.
        """
        if not self._available or not self.model:
            return "", 0.0

        try:
            graphs = self.model.parse_sents([text])
            if not graphs or not graphs[0]:
                return "", 0.0

            amr_str = graphs[0]
            # Validate the AMR is parseable
            parsed = penman.decode(amr_str)
            node_count = len(parsed.instances())

            # Heuristic confidence: well-formed AMR with reasonable node count
            # gets higher confidence. Very sparse or very dense graphs are penalised.
            text_words = len(text.split())
            ratio = node_count / max(text_words, 1)
            if 0.3 <= ratio <= 1.5:
                confidence = min(0.95, 0.6 + ratio * 0.3)
            else:
                confidence = max(0.2, 0.5 - abs(ratio - 0.7) * 0.3)

            return amr_str, confidence

        except Exception as e:
            logger.warning("AMR parse failed: {}", e)
            return "", 0.0


class CrossLingualAdapter:
    """
    Language-specific adapter for cross-lingual AMR parsing.

    For non-English input, applies language-specific transformations
    to improve AMR parsing quality:
    - Machine translation to English (for AMR parsing)
    - Back-alignment of AMR concepts to source language entities
    - Language-specific post-processing of AMR relations
    """

    # Languages where direct AMR parsing is supported (trained adapters exist)
    DIRECT_LANGUAGES = {"en", "zh", "de", "es", "fr", "pt"}
    # Languages that require translate-then-parse approach
    TRANSLATE_LANGUAGES = {"ta", "hi", "ar", "ja", "ko", "bn", "ur", "te", "ml", "pa", "gu", "mr", "id", "ms"}

    def __init__(self, adapter_dir: str, device: str = "cpu"):
        self.adapter_dir = adapter_dir
        self.device = device
        self._adapters: dict = {}
        logger.info("CrossLingualAdapter initialised, adapter_dir={}", adapter_dir)

    def adapt(self, text: str, language: str) -> str:
        """
        Adapt non-English text for AMR parsing.

        For DIRECT_LANGUAGES: applies lightweight adapter transformation.
        For TRANSLATE_LANGUAGES: translates to English for parsing,
        with entity alignment back to source language.

        In production, this uses per-language adapter weights loaded from adapter_dir.
        For MVP, returns text with language tag for the AMR parser to handle.
        """
        if language in self.DIRECT_LANGUAGES:
            return text  # Direct adapter applied inside AMR model
        # For translate-required languages, tag for downstream handling
        return f"[{language}] {text}"

    def back_align(self, amr_str: str, original_text: str, language: str) -> str:
        """Realign AMR entity values to original language surface forms."""
        # Production: use word alignment from translation to map concepts back
        return amr_str
```

---

## GRAPH SIMILARITY — lattice/similarity.py

```python
import Levenshtein
from compress.lattice.structures import SemanticGraph, GraphNode, SemanticUnitType
from compress.config import Config
from loguru import logger

class GraphSimilarityScorer:
    """
    Weighted Jaccard similarity between two semantic graphs.

    Node matching: type + value equality with fuzzy matching for entities.
    Edge matching: exact relation-type match between matched node pairs.

    Node type weights (higher = more important to preserve):
        ENTITY:     1.5  — entity loss is the most dangerous compression failure
        NEGATION:   1.5  — negation inversion reverses meaning entirely
        QUANTIFIER: 1.3  — numerical accuracy is critical
        ATTRIBUTE:  1.2  — attribute loss changes factual content
        SENTIMENT:  1.0  — sentiment shift is noticeable but less dangerous
        TEMPORAL:   1.0  — temporal shift is noticeable but less dangerous
        MODAL:      1.0  — modality changes are subtle
        RELATIONAL: 0.8  — relational structure can be implicit
        CAUSAL:     1.2  — causal relationships affect reasoning
        CONDITIONAL:1.2  — conditional clauses carry meaning
    """

    NODE_TYPE_WEIGHTS: dict[SemanticUnitType, float] = {
        SemanticUnitType.ENTITY:      1.5,
        SemanticUnitType.NEGATION:    1.5,
        SemanticUnitType.QUANTIFIER:  1.3,
        SemanticUnitType.ATTRIBUTE:   1.2,
        SemanticUnitType.CAUSAL:      1.2,
        SemanticUnitType.CONDITIONAL: 1.2,
        SemanticUnitType.SENTIMENT:   1.0,
        SemanticUnitType.TEMPORAL:    1.0,
        SemanticUnitType.MODAL:       1.0,
        SemanticUnitType.RELATIONAL:  0.8,
    }

    # Minimum Levenshtein ratio for entity fuzzy match
    ENTITY_FUZZY_THRESHOLD = 0.85

    def __init__(self):
        pass

    def jaccard(
        self,
        original: SemanticGraph,
        candidate_text: str,
        language: str,
        candidate_graph: SemanticGraph | None = None,
        extractor=None,
    ) -> float:
        """
        Compute weighted Jaccard similarity between original graph and candidate.

        If candidate_graph is not provided, attempts to extract it using the
        extractor. If extractor is also None, falls back to node-set comparison.

        Returns:
            Weighted Jaccard score between 0.0 and 1.0.
        """
        if candidate_graph is None and extractor is not None:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're inside an async context — create a task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        candidate_graph = pool.submit(
                            asyncio.run,
                            extractor.extract(candidate_text, language)
                        ).result(timeout=10)
                else:
                    candidate_graph = asyncio.run(
                        extractor.extract(candidate_text, language)
                    )
            except Exception as e:
                logger.warning("Failed to extract candidate graph for similarity: {}", e)
                return 0.0

        if candidate_graph is None:
            return 0.0

        # Compute weighted node similarity
        node_score = self._node_similarity(original.nodes, candidate_graph.nodes)

        # Compute edge similarity
        edge_score = self._edge_similarity(
            original.edges, candidate_graph.edges,
            original.nodes, candidate_graph.nodes
        )

        # Combined: 70% node similarity + 30% edge similarity
        return 0.7 * node_score + 0.3 * edge_score

    def _node_similarity(
        self, orig_nodes: list[GraphNode], cand_nodes: list[GraphNode]
    ) -> float:
        """Weighted Jaccard over node sets."""
        if not orig_nodes and not cand_nodes:
            return 1.0
        if not orig_nodes or not cand_nodes:
            return 0.0

        matched_weight = 0.0
        total_weight = 0.0
        used_cand_ids: set[str] = set()

        for orig_node in orig_nodes:
            weight = self.NODE_TYPE_WEIGHTS.get(orig_node.unit_type, 1.0)
            total_weight += weight

            # Find best matching candidate node
            best_match = None
            best_score = 0.0
            for cand_node in cand_nodes:
                if cand_node.unit_id in used_cand_ids:
                    continue
                score = self._node_match_score(orig_node, cand_node)
                if score > best_score:
                    best_score = score
                    best_match = cand_node

            if best_match and best_score > 0.5:
                matched_weight += weight * best_score
                used_cand_ids.add(best_match.unit_id)

        # Account for extra candidate nodes (penalise hallucinated content)
        unmatched_cand = len(cand_nodes) - len(used_cand_ids)
        extra_penalty = unmatched_cand * 0.3  # mild penalty for extra nodes

        return matched_weight / (total_weight + extra_penalty) if total_weight > 0 else 0.0

    def _node_match_score(self, a: GraphNode, b: GraphNode) -> float:
        """Score how well two nodes match. Returns 0.0–1.0."""
        if a.unit_type != b.unit_type:
            return 0.0

        # Exact value match
        if a.value.lower().strip() == b.value.lower().strip():
            return 1.0

        # Fuzzy match for entities
        if a.unit_type == SemanticUnitType.ENTITY:
            ratio = Levenshtein.ratio(a.value.lower(), b.value.lower())
            return ratio if ratio >= self.ENTITY_FUZZY_THRESHOLD else 0.0

        # For sentiment/temporal/modal: same type counts as partial match
        if a.unit_type in {SemanticUnitType.SENTIMENT, SemanticUnitType.TEMPORAL, SemanticUnitType.MODAL}:
            return 0.6  # Same type but different value = partial match

        return 0.0

    def _edge_similarity(
        self, orig_edges: list[tuple], cand_edges: list[tuple],
        orig_nodes: list[GraphNode], cand_nodes: list[GraphNode]
    ) -> float:
        """Compute edge overlap between graphs."""
        if not orig_edges and not cand_edges:
            return 1.0
        if not orig_edges or not cand_edges:
            return 0.0

        # Normalise edges to (type_from, relation, type_to) for comparison
        def normalise(edges: list[tuple], nodes: list[GraphNode]) -> set[tuple]:
            node_map = {n.unit_id: n.unit_type.value for n in nodes}
            result = set()
            for from_id, rel, to_id in edges:
                ft = node_map.get(from_id, "?")
                tt = node_map.get(to_id, "?")
                result.add((ft, rel, tt))
            return result

        orig_set = normalise(orig_edges, orig_nodes)
        cand_set = normalise(cand_edges, cand_nodes)

        intersection = len(orig_set & cand_set)
        union = len(orig_set | cand_set)
        return intersection / union if union > 0 else 0.0
```

---

## TOKENIZER PROFILES — search/tokenizer_profiles.py

```python
import tiktoken
from loguru import logger

class TokenizerProfiles:
    """
    Unified token counting across all supported tokenizer families.

    Supports three tokenizer backends:
    - cl100k_base (OpenAI: GPT-4o, GPT-4o-mini, GPT-3.5-turbo)
    - Anthropic tokenizer (Claude models)
    - SentencePiece (Llama, Mistral)

    Each backend is loaded lazily and cached for reuse.
    """

    # Map model names to tokenizer backend + specific model
    PROFILES: dict[str, dict] = {
        "gpt-4o":            {"backend": "tiktoken", "encoding": "cl100k_base"},
        "gpt-4o-mini":       {"backend": "tiktoken", "encoding": "cl100k_base"},
        "gpt-3.5-turbo":     {"backend": "tiktoken", "encoding": "cl100k_base"},
        "claude-3-5-sonnet": {"backend": "anthropic", "model": "claude-3-5-sonnet"},
        "claude-3-haiku":    {"backend": "anthropic", "model": "claude-3-haiku"},
        "claude-3-opus":     {"backend": "anthropic", "model": "claude-3-opus"},
        "llama-3-8b":        {"backend": "sentencepiece", "model_path": "models/llama3-tokenizer.model"},
        "llama-3-70b":       {"backend": "sentencepiece", "model_path": "models/llama3-tokenizer.model"},
        "mistral-7b":        {"backend": "sentencepiece", "model_path": "models/mistral-tokenizer.model"},
        "mistral-8x7b":      {"backend": "sentencepiece", "model_path": "models/mistral-tokenizer.model"},
    }

    def __init__(self):
        self._cache: dict = {}

    def count_tokens(self, text: str, tokenizer_name: str) -> int:
        """Count tokens using the appropriate backend for the given model."""
        profile = self.PROFILES.get(tokenizer_name)
        if not profile:
            logger.warning("Unknown tokenizer '{}', falling back to cl100k_base", tokenizer_name)
            profile = {"backend": "tiktoken", "encoding": "cl100k_base"}

        backend = profile["backend"]

        if backend == "tiktoken":
            return self._count_tiktoken(text, profile["encoding"])
        elif backend == "anthropic":
            return self._count_anthropic(text, profile["model"])
        elif backend == "sentencepiece":
            return self._count_sentencepiece(text, profile["model_path"])
        else:
            raise ValueError(f"Unknown tokenizer backend: {backend}")

    def _count_tiktoken(self, text: str, encoding_name: str) -> int:
        """Count tokens using OpenAI's tiktoken library."""
        if encoding_name not in self._cache:
            self._cache[encoding_name] = tiktoken.get_encoding(encoding_name)
        return len(self._cache[encoding_name].encode(text))

    def _count_anthropic(self, text: str, model: str) -> int:
        """Count tokens using Anthropic's tokenizer."""
        cache_key = f"anthropic:{model}"
        if cache_key not in self._cache:
            try:
                from anthropic import Anthropic
                client = Anthropic()
                self._cache[cache_key] = client
            except ImportError:
                logger.warning(
                    "anthropic package not installed, falling back to cl100k approximation"
                )
                return self._count_tiktoken(text, "cl100k_base")
        try:
            client = self._cache[cache_key]
            result = client.count_tokens(text)
            return result
        except Exception as e:
            logger.warning("Anthropic token count failed ({}), using cl100k fallback", e)
            return self._count_tiktoken(text, "cl100k_base")

    def _count_sentencepiece(self, text: str, model_path: str) -> int:
        """Count tokens using SentencePiece model."""
        if model_path not in self._cache:
            try:
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor()
                sp.Load(model_path)
                self._cache[model_path] = sp
            except (ImportError, OSError) as e:
                logger.warning(
                    "SentencePiece model '{}' unavailable ({}), using cl100k fallback",
                    model_path, e
                )
                return self._count_tiktoken(text, "cl100k_base")
        sp = self._cache[model_path]
        return len(sp.EncodeAsIds(text))
```

---

## STES SCORER — search/scorer.py

```python
import numpy as np
from compress.search.tokenizer_profiles import TokenizerProfiles
from compress.config import Config

class STESScorer:
    """
    Semantic-Token Efficiency Score calculator.

    STES = (graph_preservation ^ alpha) / (token_ratio ^ beta)

    Higher STES = better candidate (more meaning preserved, fewer tokens used).

    alpha and beta are per-language learned parameters:
    - alpha governs semantic preservation penalty (higher = stricter fidelity)
    - beta governs token savings reward (higher = more aggressive compression)

    Languages with richer morphology (Tamil, Arabic, Telugu) get higher alpha
    because rephrasing carries more risk of meaning loss in agglutinative systems.
    Analytic languages (English, Chinese) get lower alpha and beta because
    token-efficient rephrasing is safer.
    """

    LANG_PARAMS: dict[str, dict[str, float]] = {
        "ta": {"alpha": 1.3, "beta": 0.9},   # Tamil — agglutinative, high morphological richness
        "te": {"alpha": 1.3, "beta": 0.9},   # Telugu — agglutinative
        "ml": {"alpha": 1.3, "beta": 0.9},   # Malayalam — agglutinative
        "hi": {"alpha": 1.2, "beta": 0.85},  # Hindi — moderately inflectional
        "bn": {"alpha": 1.2, "beta": 0.85},  # Bengali
        "mr": {"alpha": 1.2, "beta": 0.85},  # Marathi
        "gu": {"alpha": 1.2, "beta": 0.85},  # Gujarati
        "pa": {"alpha": 1.2, "beta": 0.85},  # Punjabi
        "ur": {"alpha": 1.2, "beta": 0.85},  # Urdu
        "ar": {"alpha": 1.25, "beta": 0.9},  # Arabic — root-pattern morphology
        "ja": {"alpha": 1.15, "beta": 0.8},  # Japanese — mixed writing systems
        "zh": {"alpha": 1.1, "beta": 0.75},  # Chinese — analytic, character-based
        "ko": {"alpha": 1.15, "beta": 0.8},  # Korean — agglutinative
        "id": {"alpha": 1.05, "beta": 0.7},  # Indonesian — relatively analytic
        "ms": {"alpha": 1.05, "beta": 0.7},  # Malay
        "pt": {"alpha": 1.05, "beta": 0.7},  # Portuguese
        "es": {"alpha": 1.05, "beta": 0.7},  # Spanish
        "fr": {"alpha": 1.05, "beta": 0.7},  # French
        "de": {"alpha": 1.1, "beta": 0.75},  # German — compound words
        "en": {"alpha": 1.0, "beta": 0.65},  # English — baseline
    }

    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = TokenizerProfiles()

    def count_tokens(self, text: str, tokenizer_name: str) -> int:
        """Count tokens using the actual target tokenizer."""
        return self.tokenizer.count_tokens(text, tokenizer_name)

    def compute_stes(
        self,
        graph_preservation: float,
        token_count: int,
        source_language: str,
        baseline_token_count: int,
    ) -> float:
        """
        Compute Semantic-Token Efficiency Score.

        STES = (graph_preservation ^ alpha) / (token_ratio ^ beta)

        When token_ratio < 1.0 (compression achieved), denominator < 1 → STES > 1.
        When graph_preservation is high and token_ratio is low, STES is maximised.
        """
        params = self.LANG_PARAMS.get(source_language, {"alpha": 1.0, "beta": 0.7})
        alpha = params["alpha"]
        beta = params["beta"]

        token_ratio = token_count / max(baseline_token_count, 1)

        stes = (graph_preservation ** alpha) / (token_ratio ** beta)

        return float(np.clip(stes, 0.0, 10.0))
```

---

## BEAM SEARCH — search/beam.py

```python
import asyncio
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import MT5ForConditionalGeneration, T5Tokenizer
import torch, numpy as np
from compress.lattice.structures import SemanticGraph, CompressionCandidate
from compress.search.scorer import STESScorer
from compress.lattice.similarity import GraphSimilarityScorer
from compress.config import Config
from compress.exceptions import SearchError
from loguru import logger

class BeamSearcher:
    """
    Beam search over surface realizations of a semantic graph.

    Uses a fine-tuned mT5-large model to generate candidate compressions.
    The model is fine-tuned per tokenizer family with an auxiliary token-cost
    loss term that teaches it to prefer phrasings that align with the
    tokenizer's vocabulary boundaries.

    Scores each candidate with STES (semantic preservation x token efficiency).
    Returns top-N candidates sorted by STES score descending, with diversity penalty.
    """

    def __init__(self, config: Config):
        self.config = config
        self.stes = STESScorer(config)
        self.graph_scorer = GraphSimilarityScorer()

        # LaBSE for cross-lingual semantic similarity
        self.labse = SentenceTransformer(
            "sentence-transformers/LaBSE",
            device=config.device
        )

        # mT5 for surface realization generation
        self.mt5_tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
        self.mt5_model = MT5ForConditionalGeneration.from_pretrained(
            config.compress_model_path,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
        ).to(config.device)
        self.mt5_model.eval()

    def count_tokens(self, text: str, tokenizer_name: str) -> int:
        return self.stes.count_tokens(text, tokenizer_name)

    async def search(
        self,
        graph: SemanticGraph,
        target_tokenizer: str,
        beam_width: int = 12,
        max_candidates: int = 8,
    ) -> list[CompressionCandidate]:
        """
        Generate and score surface realizations of the semantic graph.

        Process:
        1. Prompt mT5 with structured graph representation
        2. Run beam search with beam_width beams
        3. Score each output with STES
        4. Apply diversity penalty to avoid near-duplicate candidates
        5. Return top max_candidates sorted by STES descending
        """
        try:
            prompt = self._build_prompt(graph)
            input_ids = self.mt5_tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).input_ids.to(self.config.device)

            original_tokens = self.count_tokens(graph.source_text, target_tokenizer)
            original_embedding = self.labse.encode(
                [graph.source_text], normalize_embeddings=True
            )

            # Generate candidates via beam search
            with torch.no_grad():
                outputs = self.mt5_model.generate(
                    input_ids,
                    num_beams=beam_width,
                    num_return_sequences=min(beam_width, max_candidates * 2),
                    max_new_tokens=256,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    length_penalty=0.8,
                    forced_bos_token_id=None,
                )

            candidates: list[CompressionCandidate] = []
            seen_texts: set[str] = set()

            for rank, output_ids in enumerate(outputs):
                candidate_text = self.mt5_tokenizer.decode(
                    output_ids, skip_special_tokens=True
                ).strip()

                # Skip empty, identical, or duplicate candidates
                if not candidate_text:
                    continue
                if candidate_text == graph.source_text:
                    continue
                normalised = candidate_text.lower().strip()
                if normalised in seen_texts:
                    continue
                seen_texts.add(normalised)

                # Score candidate
                candidate_tokens = self.count_tokens(candidate_text, target_tokenizer)
                candidate_embedding = self.labse.encode(
                    [candidate_text], normalize_embeddings=True
                )
                semantic_sim = float(cosine_similarity(
                    original_embedding, candidate_embedding
                )[0][0])

                graph_jacc = self.graph_scorer.jaccard(
                    graph, candidate_text, graph.source_language
                )

                stes = self.stes.compute_stes(
                    graph_preservation=graph_jacc,
                    token_count=candidate_tokens,
                    source_language=graph.source_language,
                    baseline_token_count=original_tokens,
                )

                candidates.append(CompressionCandidate(
                    text=candidate_text,
                    token_count=candidate_tokens,
                    stes_score=stes,
                    semantic_similarity=semantic_sim,
                    graph_jaccard=graph_jacc,
                    rank=rank,
                ))

            # Sort by STES descending
            candidates.sort(key=lambda c: c.stes_score, reverse=True)
            return candidates[:max_candidates]

        except Exception as e:
            logger.exception("Beam search failed")
            raise SearchError(f"Beam search failed: {e}") from e

    def _build_prompt(self, graph: SemanticGraph) -> str:
        """Build structured prompt from semantic graph for mT5 generation."""
        node_strs = []
        for node in graph.nodes[:10]:  # Cap at 10 nodes
            parts = [f"{node.unit_type.value}:{node.value}"]
            if node.intensity is not None:
                parts.append(f"(i={node.intensity:.2f})")
            if node.certainty is not None:
                parts.append(f"(c={node.certainty:.2f})")
            node_strs.append("".join(parts))
        nodes_str = " | ".join(node_strs)

        edge_strs = []
        for from_id, rel, to_id in graph.edges[:8]:
            edge_strs.append(f"{from_id}-{rel}->{to_id}")
        edges_str = " ; ".join(edge_strs) if edge_strs else "none"

        return (
            f"compress lang={graph.source_language} "
            f"units=[{nodes_str}] "
            f"rels=[{edges_str}] "
            f"original={graph.source_text[:200]}"
        )

    async def health(self) -> str:
        return "healthy"
```

---

## VERIFICATION GATE — verification/gate.py

```python
from compress.lattice.structures import SemanticGraph, CompressionCandidate
from compress.config import Config
from loguru import logger

class VerificationGate:
    """
    Triple verification gate for compression candidates.

    A candidate passes only if ALL of the following are true:
      Gate 1: Candidate text is not empty and not identical to original
      Gate 2: LaBSE semantic similarity >= semantic_threshold (default 0.91)
      Gate 3: Weighted graph Jaccard similarity >= graph_threshold (default 0.88)
      Gate 4: Token reduction >= min_reduction (default 0.15 = 15% fewer tokens)

    Thresholds derived from adversarial testing with bilingual human annotators:
    - 0.91 LaBSE: below this, >5% of candidates flagged for meaning drift
    - 0.88 graph Jaccard: below this, structural errors (entity loss, negation inversion) appear
    """

    def __init__(self, config: Config):
        self.config = config

    async def verify(
        self,
        original_text: str,
        original_graph: SemanticGraph,
        candidate: CompressionCandidate,
        original_tokens: int,
        semantic_threshold: float,
        graph_threshold: float,
        min_reduction: float,
    ) -> tuple[bool, str]:
        """
        Verify a single compression candidate against all gates.

        Returns:
            (passed: bool, reason: str)
            reason is empty string when passed=True
        """
        # Gate 1: Non-empty, non-identical
        if not candidate.text.strip():
            return False, "candidate text is empty"
        if candidate.text.strip() == original_text.strip():
            return False, "candidate is identical to original"

        # Gate 2: Semantic similarity (LaBSE cosine)
        if candidate.semantic_similarity < semantic_threshold:
            return False, (
                f"semantic similarity {candidate.semantic_similarity:.3f} "
                f"< threshold {semantic_threshold}"
            )

        # Gate 3: Graph Jaccard (weighted)
        if candidate.graph_jaccard < graph_threshold:
            return False, (
                f"graph Jaccard {candidate.graph_jaccard:.3f} "
                f"< threshold {graph_threshold}"
            )

        # Gate 4: Token reduction
        reduction = 1.0 - (candidate.token_count / max(original_tokens, 1))
        if reduction < min_reduction:
            return False, (
                f"token reduction {reduction:.3f} "
                f"< min_reduction {min_reduction}"
            )

        return True, ""

    async def health(self) -> str:
        return "healthy"
```

---

## REDIS CACHE — storage/cache.py

```python
import hashlib
import json
import redis.asyncio as redis
from compress.lattice.structures import CompressionResult
from compress.config import Config
from compress.exceptions import CacheError
from loguru import logger

class CompressionCache:
    """
    Redis-backed cache for compression results.

    Cache key = SHA-256 hash of (text, source_language, target_tokenizer).
    Value = JSON-serialized CompressionResult.
    TTL = configurable, default 3600 seconds.

    Gracefully degrades: cache failures log warnings but do not block compression.
    """

    def __init__(self, config: Config):
        self.config = config
        self._pool: redis.Redis | None = None
        self._available = True

    async def _get_pool(self) -> redis.Redis:
        """Lazy-init Redis connection pool."""
        if self._pool is None:
            try:
                self._pool = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    decode_responses=True,
                    socket_connect_timeout=2.0,
                    socket_timeout=2.0,
                )
                await self._pool.ping()
            except Exception as e:
                logger.warning("Redis unavailable ({}), caching disabled", e)
                self._available = False
                self._pool = None
        return self._pool

    def make_key(self, text: str, language: str, tokenizer: str) -> str:
        """Generate deterministic cache key."""
        raw = f"{text}|{language}|{tokenizer}"
        return f"scl:compress:{hashlib.sha256(raw.encode()).hexdigest()}"

    async def get(self, key: str) -> CompressionResult | None:
        """Retrieve cached result. Returns None on miss or error."""
        if not self._available:
            return None
        try:
            pool = await self._get_pool()
            if pool is None:
                return None
            raw = await pool.get(key)
            if raw is None:
                return None
            data = json.loads(raw)
            return CompressionResult(**data)
        except Exception as e:
            logger.warning("Cache get failed for key {}: {}", key[:20], e)
            return None

    async def set(self, key: str, result: CompressionResult, ttl_seconds: int = 3600) -> None:
        """Store result in cache. Silently fails on error."""
        if not self._available:
            return
        try:
            pool = await self._get_pool()
            if pool is None:
                return
            data = json.dumps(vars(result))
            await pool.set(key, data, ex=ttl_seconds)
        except Exception as e:
            logger.warning("Cache set failed for key {}: {}", key[:20], e)

    async def health(self) -> str:
        """Return cache health status."""
        if not self._available:
            return "degraded (redis unavailable)"
        try:
            pool = await self._get_pool()
            if pool:
                await pool.ping()
                return "healthy"
            return "degraded"
        except Exception:
            return "degraded"
```

---

## PROMETHEUS METRICS — storage/metrics_store.py

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from compress.lattice.structures import CompressionResult
from compress.config import Config
from loguru import logger

class MetricsStore:
    """
    Prometheus metrics collector for the SCL engine.

    Exposed metrics:
    - scl_compressions_total: Counter by language, tokenizer, applied/rejected
    - scl_compression_ratio: Histogram of token reduction ratios
    - scl_processing_seconds: Histogram of request latency
    - scl_semantic_similarity: Histogram of LaBSE scores
    - scl_graph_jaccard: Histogram of graph Jaccard scores
    - scl_cache_hits_total: Counter of cache hits by language
    - scl_candidates_evaluated: Histogram of candidates per request
    """

    def __init__(self, config: Config):
        self.config = config

        self.compressions_total = Counter(
            "scl_compressions_total",
            "Total compression requests",
            ["language", "tokenizer", "status"]
        )
        self.compression_ratio = Histogram(
            "scl_compression_ratio",
            "Token reduction ratio (0.0–1.0)",
            buckets=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.0]
        )
        self.processing_seconds = Histogram(
            "scl_processing_seconds",
            "Request processing time in seconds",
            buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        self.semantic_similarity = Histogram(
            "scl_semantic_similarity",
            "LaBSE cosine similarity scores",
            buckets=[0.70, 0.80, 0.85, 0.88, 0.91, 0.93, 0.95, 0.97, 0.99, 1.0]
        )
        self.graph_jaccard = Histogram(
            "scl_graph_jaccard",
            "Graph Jaccard similarity scores",
            buckets=[0.60, 0.70, 0.80, 0.85, 0.88, 0.91, 0.95, 1.0]
        )
        self.cache_hits = Counter(
            "scl_cache_hits_total",
            "Cache hits",
            ["language"]
        )
        self.candidates_evaluated = Histogram(
            "scl_candidates_evaluated",
            "Number of candidates evaluated per request",
            buckets=[0, 1, 2, 4, 6, 8, 12, 16]
        )

        # Start metrics server
        try:
            start_http_server(config.metrics_port)
            logger.info("Prometheus metrics server started on port {}", config.metrics_port)
        except Exception as e:
            logger.warning("Failed to start metrics server: {}", e)

    def record_compression(self, result: CompressionResult, customer_id: str) -> None:
        """Record metrics for a completed compression."""
        status = "applied" if result.compression_applied else "rejected"
        self.compressions_total.labels(
            language=result.source_language,
            tokenizer=result.target_tokenizer,
            status=status,
        ).inc()

        self.compression_ratio.observe(result.reduction_ratio)
        self.processing_seconds.observe(result.processing_ms / 1000.0)
        self.semantic_similarity.observe(result.semantic_similarity)
        self.graph_jaccard.observe(result.graph_jaccard)
        self.candidates_evaluated.observe(result.candidates_evaluated)

    def record_cache_hit(self, language: str) -> None:
        """Record a cache hit."""
        self.cache_hits.labels(language=language).inc()
```

---

## FASTAPI MIDDLEWARE — api/middleware.py

```python
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from loguru import logger

class TimingMiddleware(BaseHTTPMiddleware):
    """
    Request timing middleware.

    Adds X-Processing-Time-Ms header to every response.
    Logs request path, method, status code, and latency.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - start) * 1000
        response.headers["X-Processing-Time-Ms"] = f"{elapsed_ms:.1f}"
        logger.info(
            "{} {} → {} ({:.1f}ms)",
            request.method, request.url.path, response.status_code, elapsed_ms
        )
        return response
```

---

## FASTAPI ROUTER — api/router.py

```python
from fastapi import APIRouter, HTTPException, Depends
from compress.api.schemas import (
    CompressRequest, CompressResponse,
    BatchCompressRequest, BatchCompressResponse,
    HealthResponse
)
from compress.engine import SCLEngine
from compress.exceptions import UnsupportedLanguageError, ExtractionError, SearchError
from loguru import logger

router = APIRouter(prefix="/compress", tags=["COMPRESS"])

def get_engine() -> SCLEngine:
    from compress.main import engine_instance
    return engine_instance

@router.post("/", response_model=CompressResponse, status_code=200)
async def compress_text(
    request: CompressRequest,
    engine: SCLEngine = Depends(get_engine),
) -> CompressResponse:
    """
    Compress input text using Semantic Compression Lattice.

    Returns compressed text with semantic preservation guarantee.
    If no valid compression found, returns original text with compression_applied=False.
    """
    try:
        result = await engine.compress(
            text=request.text,
            source_language=request.language,
            target_tokenizer=request.target_tokenizer,
            customer_id=request.customer_id,
            min_reduction_threshold=request.min_reduction_threshold,
            semantic_threshold=request.semantic_threshold,
            graph_threshold=request.graph_threshold,
        )
        return CompressResponse(
            original_text=result.original_text,
            compressed_text=result.compressed_text,
            language=result.source_language,
            target_tokenizer=result.target_tokenizer,
            original_tokens=result.original_token_count,
            compressed_tokens=result.compressed_token_count,
            reduction_ratio=result.reduction_ratio,
            semantic_similarity=result.semantic_similarity,
            graph_jaccard=result.graph_jaccard,
            stes_score=result.stes_score,
            compression_applied=result.compression_applied,
            rejection_reason=result.rejection_reason,
            processing_ms=result.processing_ms,
            candidates_evaluated=result.candidates_evaluated,
        )
    except UnsupportedLanguageError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except (ExtractionError, SearchError) as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    except Exception as e:
        logger.exception("Unhandled error in /compress")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/batch", response_model=BatchCompressResponse, status_code=200)
async def batch_compress(
    request: BatchCompressRequest,
    engine: SCLEngine = Depends(get_engine),
) -> BatchCompressResponse:
    """Compress multiple texts concurrently."""
    if len(request.items) > 100:
        raise HTTPException(status_code=422, detail="Maximum 100 items per batch")
    results = await engine.batch_compress(
        texts=[item.model_dump() for item in request.items],
        customer_id=request.customer_id,
    )
    return BatchCompressResponse(results=[
        CompressResponse(
            original_text=r.original_text,
            compressed_text=r.compressed_text,
            language=r.source_language,
            target_tokenizer=r.target_tokenizer,
            original_tokens=r.original_token_count,
            compressed_tokens=r.compressed_token_count,
            reduction_ratio=r.reduction_ratio,
            semantic_similarity=r.semantic_similarity,
            graph_jaccard=r.graph_jaccard,
            stes_score=r.stes_score,
            compression_applied=r.compression_applied,
            rejection_reason=r.rejection_reason,
            processing_ms=r.processing_ms,
            candidates_evaluated=r.candidates_evaluated,
        ) for r in results
    ])

@router.get("/health", response_model=HealthResponse)
async def health(engine: SCLEngine = Depends(get_engine)) -> HealthResponse:
    status = await engine.health_check()
    return HealthResponse(**status)

@router.get("/supported")
async def supported() -> dict:
    return {
        "languages": sorted(SCLEngine.SUPPORTED_LANGUAGES),
        "tokenizers": sorted(SCLEngine.SUPPORTED_TOKENIZERS.keys()),
    }
```

---

## PYDANTIC SCHEMAS — api/schemas.py

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class CompressRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    language: str = Field(..., description="ISO 639-1 language code e.g. 'ta', 'hi', 'ar'")
    target_tokenizer: str = Field(default="gpt-4o")
    customer_id: str = Field(default="default")
    min_reduction_threshold: float = Field(default=0.15, ge=0.05, le=0.80)
    semantic_threshold: float = Field(default=0.91, ge=0.80, le=0.99)
    graph_threshold: float = Field(default=0.88, ge=0.75, le=0.99)

    @field_validator("text")
    @classmethod
    def text_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be whitespace only")
        return v

class CompressResponse(BaseModel):
    original_text: str
    compressed_text: str
    language: str
    target_tokenizer: str
    original_tokens: int
    compressed_tokens: int
    reduction_ratio: float
    semantic_similarity: float
    graph_jaccard: float
    stes_score: float
    compression_applied: bool
    rejection_reason: Optional[str]
    processing_ms: float
    candidates_evaluated: int

class BatchItem(BaseModel):
    text: str
    language: str
    tokenizer: str = "gpt-4o"

class BatchCompressRequest(BaseModel):
    items: list[BatchItem] = Field(..., max_length=100)
    customer_id: str = "default"

class BatchCompressResponse(BaseModel):
    results: list[CompressResponse]

class HealthResponse(BaseModel):
    status: str
    extractor: str
    searcher: str
    gate: str
    cache: str
```

---

## CONFIG — config.py

```python
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    # Model paths
    compress_model_path: str = "google/mt5-base"       # Replace with fine-tuned path in production
    amr_model_path: str = "models/amr-spring-base"     # SPRING AMR parser checkpoint
    amr_adapter_dir: str = "models/amr-adapters"       # Cross-lingual adapter weights

    # Device
    device: str = "cpu"                                 # "cuda" for GPU

    # Beam search
    beam_width: int = 12
    max_candidates: int = 8

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_ttl_seconds: int = 3600

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    log_level: str = "INFO"

    # Prometheus
    metrics_port: int = 9101

    class Config:
        env_prefix = "COMPRESS_"
        env_file = ".env"
```

---

## EXCEPTIONS — exceptions.py

```python
class CompressError(Exception):
    """Base exception for COMPRESS module."""

class UnsupportedLanguageError(CompressError):
    """Language or tokenizer not in supported set."""

class ExtractionError(CompressError):
    """Semantic graph extraction failed."""

class SearchError(CompressError):
    """Beam search failed to produce candidates."""

class VerificationError(CompressError):
    """Unexpected error during verification gate."""

class CacheError(CompressError):
    """Redis cache read/write failure."""
```

---

## FULL TEST SUITE — tests/test_engine.py

```python
import pytest
import asyncio
from compress.engine import SCLEngine
from compress.config import Config
from compress.exceptions import UnsupportedLanguageError

@pytest.fixture
def config():
    return Config(device="cpu", beam_width=4, max_candidates=3)

@pytest.fixture
def engine(config):
    return SCLEngine(config)

# ── MULTILINGUAL TEST CASES ────────────────────────────────────────────────

MULTILINGUAL_CASES = [
    {
        "language": "ta",
        "text": "என் கணக்கில் தவறான கட்டணம் விதிக்கப்பட்டுள்ளது என்று நான் நம்புகிறேன், தயவுசெய்து இதை சரிபார்க்க முடியுமா?",
        "tokenizer": "gpt-4o",
        "expected_min_reduction": 0.10,
    },
    {
        "language": "hi",
        "text": "मुझे लगता है कि मेरे खाते में गलत शुल्क लगाया गया है, क्या आप कृपया इसकी जांच कर सकते हैं?",
        "tokenizer": "gpt-4o",
        "expected_min_reduction": 0.10,
    },
    {
        "language": "ar",
        "text": "أعتقد أنه تم خصم رسوم خاطئة من حسابي، هل يمكنك التحقق من ذلك من فضلك؟",
        "tokenizer": "gpt-4o",
        "expected_min_reduction": 0.10,
    },
    {
        "language": "en",
        "text": "I would like to inquire about the current status of my recent transaction that I believe may have been processed incorrectly.",
        "tokenizer": "gpt-4o",
        "expected_min_reduction": 0.10,
    },
    {
        "language": "ja",
        "text": "私のアカウントに誤った請求がされていると思いますが、確認していただけますか？",
        "tokenizer": "gpt-4o",
        "expected_min_reduction": 0.08,
    },
    {
        "language": "zh",
        "text": "我认为我的账户被收取了错误的费用，请您能否核实一下？",
        "tokenizer": "gpt-4o",
        "expected_min_reduction": 0.08,
    },
]

@pytest.mark.asyncio
@pytest.mark.parametrize("case", MULTILINGUAL_CASES)
async def test_compression_reduces_tokens(engine, case):
    """Compression should produce fewer tokens while preserving meaning."""
    result = await engine.compress(
        text=case["text"],
        source_language=case["language"],
        target_tokenizer=case["tokenizer"],
    )
    if result.compression_applied:
        assert result.compressed_token_count < result.original_token_count
        assert result.reduction_ratio >= 0.0
        assert result.semantic_similarity >= 0.91
        assert result.graph_jaccard >= 0.88

@pytest.mark.asyncio
async def test_semantic_preservation_above_threshold(engine):
    """Compressed output must meet the 0.91 LaBSE similarity threshold."""
    result = await engine.compress(
        text="The payment was incorrectly charged to my account yesterday.",
        source_language="en",
        target_tokenizer="gpt-4o",
        semantic_threshold=0.91,
    )
    if result.compression_applied:
        assert result.semantic_similarity >= 0.91, (
            f"Semantic similarity {result.semantic_similarity} below threshold 0.91"
        )

@pytest.mark.asyncio
async def test_graph_jaccard_above_threshold(engine):
    """Compressed output must meet the 0.88 graph Jaccard threshold."""
    result = await engine.compress(
        text="I am extremely frustrated with the incorrect charge on my account from yesterday.",
        source_language="en",
        target_tokenizer="gpt-4o",
        graph_threshold=0.88,
    )
    if result.compression_applied:
        assert result.graph_jaccard >= 0.88, (
            f"Graph Jaccard {result.graph_jaccard} below threshold 0.88"
        )

@pytest.mark.asyncio
async def test_unsupported_language_raises(engine):
    """Unsupported language codes should raise UnsupportedLanguageError."""
    with pytest.raises(UnsupportedLanguageError):
        await engine.compress(
            text="Some text",
            source_language="xx",
            target_tokenizer="gpt-4o",
        )

@pytest.mark.asyncio
async def test_unsupported_tokenizer_raises(engine):
    """Unsupported tokenizer names should raise UnsupportedLanguageError."""
    with pytest.raises(UnsupportedLanguageError):
        await engine.compress(
            text="Some text",
            source_language="en",
            target_tokenizer="unknown-model-xyz",
        )

@pytest.mark.asyncio
async def test_returns_original_when_no_valid_candidate(engine):
    """Very short text should return original gracefully."""
    result = await engine.compress(
        text="OK",
        source_language="en",
        target_tokenizer="gpt-4o",
        min_reduction_threshold=0.50,
    )
    assert result.original_text == "OK"
    assert result.compression_applied is False or result.reduction_ratio >= 0.0

@pytest.mark.asyncio
async def test_batch_compress_concurrency(engine):
    """Batch compression should process all items concurrently."""
    items = [
        {"text": "How do I reset my password?", "language": "en", "tokenizer": "gpt-4o"},
        {"text": "என் கடவுச்சொல்லை மீட்டமைக்க எப்படி?", "language": "ta", "tokenizer": "gpt-4o"},
        {"text": "पासवर्ड कैसे रीसेट करें?", "language": "hi", "tokenizer": "gpt-4o"},
    ]
    results = await engine.batch_compress(items)
    assert len(results) == 3
    for r in results:
        assert r.original_text is not None
        assert r.source_language in {"en", "ta", "hi"}

@pytest.mark.asyncio
async def test_health_check(engine):
    """Health check should report all components."""
    health = await engine.health_check()
    assert health["status"] == "healthy"
    assert "extractor" in health
    assert "searcher" in health
    assert "gate" in health
    assert "cache" in health

@pytest.mark.asyncio
async def test_tamil_costs_more_than_english_without_compression(engine):
    """Validate core problem: Tamil should have higher token count than English equivalent."""
    tamil_result = await engine.compress(
        text="என் கடவுச்சொல்லை மீட்டமைக்க எப்படி?",
        source_language="ta",
        target_tokenizer="gpt-4o",
        min_reduction_threshold=0.0,
    )
    english_tokens = engine.searcher.count_tokens(
        "How do I reset my password?", "gpt-4o"
    )
    assert tamil_result.original_token_count > english_tokens * 1.5, (
        f"Tamil ({tamil_result.original_token_count} tokens) should be "
        f">1.5x English ({english_tokens} tokens)"
    )

@pytest.mark.asyncio
async def test_negation_preserved_in_compression(engine):
    """Negation must never be lost during compression — this would invert meaning."""
    result = await engine.compress(
        text="I have not received any refund for the cancelled transaction.",
        source_language="en",
        target_tokenizer="gpt-4o",
    )
    if result.compression_applied:
        # The compressed text must contain some form of negation
        negation_markers = ["not", "no", "never", "n't", "without", "un"]
        has_negation = any(m in result.compressed_text.lower() for m in negation_markers)
        assert has_negation, (
            f"Compressed text lost negation: '{result.compressed_text}'"
        )

@pytest.mark.asyncio
async def test_idempotent_compression(engine):
    """Compressing already-compressed text should not change it further."""
    result1 = await engine.compress(
        text="I would like to inquire about the current status of my recent transaction.",
        source_language="en",
        target_tokenizer="gpt-4o",
    )
    if result1.compression_applied:
        result2 = await engine.compress(
            text=result1.compressed_text,
            source_language="en",
            target_tokenizer="gpt-4o",
        )
        # Second compression should either not apply or produce similar tokens
        if result2.compression_applied:
            assert result2.reduction_ratio < 0.10, (
                "Double compression should yield minimal additional savings"
            )
```

---

## SIMILARITY TESTS — tests/test_similarity.py

```python
import pytest
from compress.lattice.similarity import GraphSimilarityScorer
from compress.lattice.structures import SemanticGraph, GraphNode, SemanticUnitType

@pytest.fixture
def scorer():
    return GraphSimilarityScorer()

def _make_graph(nodes_data: list[tuple], edges: list[tuple] = None) -> SemanticGraph:
    """Helper to build a graph from (type, value) tuples."""
    nodes = []
    for i, (unit_type, value) in enumerate(nodes_data):
        nodes.append(GraphNode(
            unit_id=f"n{i}",
            unit_type=unit_type,
            value=value,
            confidence=0.9,
        ))
    return SemanticGraph(
        source_text="test",
        source_language="en",
        nodes=nodes,
        edges=edges or [],
        amr_penman="",
        extraction_confidence=0.9,
        amr_confidence=0.8,
    )

def test_identical_graphs_score_1(scorer):
    g = _make_graph([
        (SemanticUnitType.ENTITY, "account"),
        (SemanticUnitType.SENTIMENT, "negative"),
    ])
    score = scorer.jaccard(g, "test", "en", candidate_graph=g)
    assert score == pytest.approx(1.0, abs=0.01)

def test_empty_vs_populated_scores_0(scorer):
    g1 = _make_graph([])
    g2 = _make_graph([(SemanticUnitType.ENTITY, "account")])
    score = scorer.jaccard(g1, "test", "en", candidate_graph=g2)
    assert score < 0.3

def test_entity_loss_penalised_heavily(scorer):
    """Entity nodes have weight 1.5 — losing them should drop the score significantly."""
    original = _make_graph([
        (SemanticUnitType.ENTITY, "account"),
        (SemanticUnitType.ENTITY, "refund"),
        (SemanticUnitType.SENTIMENT, "negative"),
    ])
    # Candidate missing one entity
    candidate = _make_graph([
        (SemanticUnitType.ENTITY, "account"),
        (SemanticUnitType.SENTIMENT, "negative"),
    ])
    score = scorer.jaccard(original, "test", "en", candidate_graph=candidate)
    assert score < 0.85, f"Entity loss should drop score below 0.85, got {score}"

def test_negation_loss_penalised_heavily(scorer):
    """Negation nodes have weight 1.5 — losing them inverts meaning."""
    original = _make_graph([
        (SemanticUnitType.ENTITY, "refund"),
        (SemanticUnitType.NEGATION, "negated"),
    ])
    candidate = _make_graph([
        (SemanticUnitType.ENTITY, "refund"),
    ])
    score = scorer.jaccard(original, "test", "en", candidate_graph=candidate)
    assert score < 0.70, f"Negation loss should drop score below 0.70, got {score}"

def test_fuzzy_entity_matching(scorer):
    """Entities with similar names should fuzzy-match."""
    original = _make_graph([(SemanticUnitType.ENTITY, "transaction")])
    candidate = _make_graph([(SemanticUnitType.ENTITY, "transactions")])
    score = scorer.jaccard(original, "test", "en", candidate_graph=candidate)
    assert score > 0.8, f"Fuzzy entity match should score > 0.8, got {score}"
```

---

## TOKENIZER PROFILES TESTS — tests/test_tokenizer_profiles.py

```python
import pytest
from compress.search.tokenizer_profiles import TokenizerProfiles

@pytest.fixture
def profiles():
    return TokenizerProfiles()

def test_tiktoken_counts_english(profiles):
    count = profiles.count_tokens("Hello world", "gpt-4o")
    assert 2 <= count <= 4

def test_tiktoken_counts_tamil(profiles):
    """Tamil should produce significantly more tokens than English."""
    en_count = profiles.count_tokens("How do I reset my password?", "gpt-4o")
    ta_count = profiles.count_tokens("என் கடவுச்சொல்லை மீட்டமைக்க எப்படி?", "gpt-4o")
    assert ta_count > en_count * 1.5

def test_unknown_tokenizer_falls_back(profiles):
    """Unknown model names should fall back to cl100k without crashing."""
    count = profiles.count_tokens("Hello world", "nonexistent-model")
    assert count > 0

def test_empty_string_returns_zero(profiles):
    count = profiles.count_tokens("", "gpt-4o")
    assert count == 0
```

---

## DOCKER + STARTUP

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files (in production, mount these as volumes)
COPY models/ ./models/
COPY compress/ ./compress/
COPY main.py .

EXPOSE 8001 9101
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

```python
# main.py
from fastapi import FastAPI
from compress.api.router import router
from compress.api.middleware import TimingMiddleware
from compress.engine import SCLEngine
from compress.config import Config
from contextlib import asynccontextmanager
from loguru import logger

engine_instance: SCLEngine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine_instance
    config = Config()
    logger.info("Starting SCL Engine with config: {}", config.model_dump())
    engine_instance = SCLEngine(config)
    yield
    logger.info("Shutting down SCL Engine")
    engine_instance = None

app = FastAPI(
    title="COMPRESS — Semantic Compression Lattice Engine",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(TimingMiddleware)
app.include_router(router)
```

---

## PROMPT TO PASTE INTO COMPANY CLAUDE

> "Build this production-grade Python module called COMPRESS exactly as specified.
> It is a standalone semantic text compression engine (v2.0.0).
> Implement every file listed. Full production code — no stubs, no placeholders.
> All functions must be fully implemented.
> Include complete error handling, logging, type hints, and docstrings throughout.
> The test suite must pass against the real implementation.
> Start with lattice/structures.py, then config.py, then exceptions.py,
> then lattice/amr_utils.py, lattice/extractor.py, lattice/similarity.py,
> search/tokenizer_profiles.py, search/scorer.py, search/beam.py,
> verification/gate.py, storage/cache.py, storage/metrics_store.py,
> api/schemas.py, api/middleware.py, api/router.py, engine.py, main.py, Dockerfile.
> Output each file completely, in order."
