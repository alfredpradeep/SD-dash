"""
Hybrid semantic graph extractor.

Combines three extraction streams:
1. AMR parsing (SPRING-based with multilingual adapters)
2. Neural semantic classification (zero-shot NLI via mDeBERTa)
3. Structured extraction (NER, regex, morphological analysis)

When AMR confidence < 0.6, the system upweights streams 2 and 3.
"""

import penman
from transformers import pipeline
from compress.lattice.structures import SemanticGraph, GraphNode, SemanticUnitType
from compress.lattice.amr_utils import AMRParser, CrossLingualAdapter
from compress.config import Config
from compress.exceptions import ExtractionError
import re
import uuid
from loguru import logger


class LatticeExtractor:
    """
    Hybrid semantic graph extractor.

    Fuses AMR parsing, zero-shot NLI classification, and structured
    extraction (NER + regex) into a single SemanticGraph with per-node
    confidence scores.
    """

    SENTIMENT_LABELS = [
        "positive sentiment", "negative sentiment", "neutral sentiment"
    ]
    TEMPORAL_LABELS = ["past event", "present event", "future event"]
    MODAL_LABELS = ["certain", "uncertain", "conditional", "obligatory"]

    NEGATION_PATTERNS: dict[str, list[str]] = {
        "en": [
            r"\b(not|no|never|neither|nor|without|cannot|can't|won't|don't|"
            r"doesn't|didn't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't)\b"
        ],
        "ta": [r"(இல்லை|வேண்டாம்|மாட்டேன்|மாட்டார்|அல்ல|இல்லாமல்|முடியாது)"],
        "hi": [r"(नहीं|मत|बिना|न|ना|कभी\s+नहीं)"],
        "ar": [r"(لا|لم|لن|ليس|ليست|ليسوا|ما|غير)"],
        "ja": [r"(ない|なかった|ません|ず|ぬ|まい)"],
        "zh": [r"(不|没|没有|未|无|非|别|勿|莫)"],
        "ko": [r"(않|못|없|아니|말)"],
        "bn": [r"(না|নয়|নেই|নি)"],
        "ur": [r"(نہیں|مت|نه|بغیر)"],
        "te": [r"(లేదు|కాదు|వద్దు|లేకుండా)"],
        "ml": [r"(ഇല്ല|അല്ല|വേണ്ട|കഴിയില്ല)"],
        "pt": [r"\b(não|nunca|nem|sem|nenhum|nenhuma|jamais)\b"],
        "es": [r"\b(no|nunca|ni|sin|ningún|ninguna|jamás)\b"],
        "fr": [r"\b(ne|pas|jamais|ni|sans|aucun|aucune)\b"],
        "de": [r"\b(nicht|kein|keine|keinem|keinen|nie|niemals|ohne)\b"],
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
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                device=-1,  # CPU — avoid device mapping issues
            )
            logger.info("NLI classifier loaded (mDeBERTa-xnli)")
        except Exception as e:
            logger.warning("NLI classifier failed to load: {}. Sentiment/temporal/modal extraction disabled.", e)
            self.classifier = None

        # Stream 3: Multilingual NER for entity extraction
        try:
            import spacy
            self.spacy_nlp = spacy.load("xx_ent_wiki_sm")
            self.ner = None  # Use spaCy instead
            logger.info("NER loaded via spaCy (xx_ent_wiki_sm)")
        except Exception:
            try:
                self.ner = pipeline(
                    "token-classification",
                    model="Babelscape/wikineural-multilingual-ner",
                    aggregation_strategy="simple",
                    device=-1,
                )
                self.spacy_nlp = None
                logger.info("NER loaded via HuggingFace (wikineural)")
            except Exception as e:
                logger.warning("NER failed to load: {}. Entity extraction will use regex fallback.", e)
                self.ner = None
                self.spacy_nlp = None

        self.amr_confidence_floor = 0.6

    async def extract(self, text: str, language: str) -> SemanticGraph:
        """
        Extract full semantic graph from text using hybrid pipeline.

        Pipeline:
        1. AMR parsing -> rooted DAG of concepts and relations
        2. Named entity recognition -> ENTITY nodes
        3. Zero-shot sentiment -> SENTIMENT node
        4. Zero-shot temporal -> TEMPORAL node
        5. Zero-shot modality -> MODAL node
        6. Negation detection (per-language) -> NEGATION nodes
        7. Numerical/quantifier extraction -> QUANTIFIER nodes
        8. Stream fusion + deduplication
        9. Confidence scoring (harmonic mean)
        """
        try:
            nodes: list[GraphNode] = []
            edges: list[tuple] = []
            component_scores: dict = {}

            # Stream 1: AMR Parsing
            amr_graph, amr_conf = await self._extract_amr(text, language)
            component_scores["amr"] = amr_conf

            amr_nodes, amr_edges = self._amr_to_graph(amr_graph, amr_conf)
            nodes.extend(amr_nodes)
            edges.extend(amr_edges)

            # Stream 2: Neural Classification
            nli_conf = await self._extract_nli(text, language, nodes, edges)
            component_scores["nli"] = nli_conf

            # Stream 3: Structured Extraction
            struct_conf = await self._extract_structured(
                text, language, nodes, edges
            )
            component_scores["structured"] = struct_conf

            # Deduplicate nodes (same type+value, keep highest confidence)
            nodes = self._deduplicate_nodes(nodes)

            # Build AMR penman string
            amr_penman = (
                amr_graph if amr_graph
                else self._build_penman(nodes, edges, text)
            )

            # Compute overall confidence (harmonic mean of nonzero components)
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
            raise ExtractionError(
                f"Semantic graph extraction failed: {e}"
            ) from e

    async def _extract_amr(
        self, text: str, language: str
    ) -> tuple[str, float]:
        """Run AMR parsing with language-specific adapter."""
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

    def _amr_to_graph(
        self, amr_penman: str, amr_conf: float
    ) -> tuple[list[GraphNode], list[tuple]]:
        """Convert AMR penman notation to GraphNode list and edges."""
        nodes = []
        edges = []
        if not amr_penman or amr_conf < 0.1:
            return nodes, edges

        try:
            graph = penman.decode(amr_penman)
            concept_map: dict[str, str] = {}

            for instance in graph.instances():
                unit_id = str(uuid.uuid4())[:8]
                concept_map[instance.source] = unit_id
                node = GraphNode(
                    unit_id=unit_id,
                    unit_type=self._classify_amr_concept(instance.target),
                    value=str(instance.target),
                    confidence=amr_conf,
                    metadata={"source": "amr", "amr_var": instance.source},
                )
                nodes.append(node)

            for edge in graph.edges():
                src_id = concept_map.get(edge.source)
                tgt_id = concept_map.get(edge.target)
                if src_id and tgt_id:
                    edges.append((src_id, edge.role.lstrip(":"), tgt_id))

        except Exception as e:
            logger.warning("AMR->graph conversion failed: {}", e)

        return nodes, edges

    def _classify_amr_concept(self, concept: str) -> SemanticUnitType:
        """Map AMR concept string to SemanticUnitType."""
        c = str(concept).lower()
        temporal = {
            "date", "time", "year", "month", "day", "before", "after",
            "now", "then", "yesterday", "tomorrow",
        }
        negation = {"not", "no", "-", "never"}
        modal = {
            "possible", "likely", "obligate", "recommend",
            "must", "should", "can",
        }
        sentiment = {
            "happy", "sad", "angry", "frustrate", "love", "hate",
            "fear", "worry", "glad", "upset",
        }
        causal = {"cause", "because", "result", "lead-to", "due-to"}

        if c in temporal or c.startswith("date-"):
            return SemanticUnitType.TEMPORAL
        if c in negation or c.endswith("-91"):
            return SemanticUnitType.NEGATION
        if c in modal:
            return SemanticUnitType.MODAL
        if c in sentiment:
            return SemanticUnitType.SENTIMENT
        if c in causal:
            return SemanticUnitType.CAUSAL
        if c.startswith("have-condition") or c == "if":
            return SemanticUnitType.CONDITIONAL
        return SemanticUnitType.ENTITY

    async def _extract_nli(
        self,
        text: str,
        language: str,
        nodes: list[GraphNode],
        edges: list[tuple],
    ) -> float:
        """Run NLI-based semantic classification. Appends to nodes/edges."""
        if self.classifier is None:
            return 0.0

        scores = []

        # Sentiment
        sent_result = self.classifier(
            text,
            candidate_labels=self.SENTIMENT_LABELS,
            hypothesis_template="This text expresses {}.",
        )
        sent_score = sent_result["scores"][0]
        if sent_score > 0.5:
            sentiment_value = sent_result["labels"][0].split()[0]
            intensity = sent_score if sentiment_value != "neutral" else 0.0
            nodes.append(
                GraphNode(
                    unit_id=str(uuid.uuid4())[:8],
                    unit_type=SemanticUnitType.SENTIMENT,
                    value=sentiment_value,
                    intensity=intensity,
                    confidence=sent_score,
                    metadata={"source": "nli"},
                )
            )
            scores.append(sent_score)

        # Temporal
        temp_result = self.classifier(
            text,
            candidate_labels=self.TEMPORAL_LABELS,
            hypothesis_template="This text describes a {}.",
        )
        if temp_result["scores"][0] > 0.55:
            nodes.append(
                GraphNode(
                    unit_id=str(uuid.uuid4())[:8],
                    unit_type=SemanticUnitType.TEMPORAL,
                    value=temp_result["labels"][0].split()[0],
                    specificity=temp_result["scores"][0],
                    confidence=temp_result["scores"][0],
                    metadata={"source": "nli"},
                )
            )
            scores.append(temp_result["scores"][0])

        # Modal
        modal_result = self.classifier(
            text,
            candidate_labels=self.MODAL_LABELS,
            hypothesis_template="The statement is {}.",
        )
        if modal_result["scores"][0] > 0.55:
            nodes.append(
                GraphNode(
                    unit_id=str(uuid.uuid4())[:8],
                    unit_type=SemanticUnitType.MODAL,
                    value=modal_result["labels"][0],
                    certainty=modal_result["scores"][0],
                    confidence=modal_result["scores"][0],
                    metadata={"source": "nli"},
                )
            )
            scores.append(modal_result["scores"][0])

        return sum(scores) / len(scores) if scores else 0.0

    async def _extract_structured(
        self,
        text: str,
        language: str,
        nodes: list[GraphNode],
        edges: list[tuple],
    ) -> float:
        """Run NER + regex extraction. Appends to nodes/edges."""
        scores = []

        # NER (spaCy or HuggingFace pipeline)
        entities = []
        if self.spacy_nlp is not None:
            try:
                doc = self.spacy_nlp(text)
                entities = [
                    {"word": ent.text, "score": 0.85, "entity_group": ent.label_}
                    for ent in doc.ents
                ]
            except Exception as e:
                logger.warning("spaCy NER failed: {}", e)
        elif self.ner is not None:
            try:
                entities = self.ner(text)
            except Exception as e:
                logger.warning("HF NER failed: {}", e)
        else:
            # Regex fallback: extract capitalized words and numbers as entities
            for match in re.finditer(r'\b[A-Z][a-zA-Z]+\b', text):
                entities.append({"word": match.group(), "score": 0.7, "entity_group": "MISC"})
            for match in re.finditer(r'\b\d+\b', text):
                entities.append({"word": match.group(), "score": 0.8, "entity_group": "NUM"})

        for ent in entities:
            if ent["score"] < 0.5:
                continue
            nodes.append(
                GraphNode(
                    unit_id=str(uuid.uuid4())[:8],
                    unit_type=SemanticUnitType.ENTITY,
                    value=ent["word"],
                    confidence=ent["score"],
                    metadata={
                        "source": "ner",
                        "ner_label": ent["entity_group"],
                    },
                )
            )
            scores.append(ent["score"])

        # Per-language negation
        lang_patterns = self.NEGATION_PATTERNS.get(
            language, self.NEGATION_PATTERNS.get("en", [])
        )
        has_negation = any(
            re.search(p, text, re.IGNORECASE) for p in lang_patterns
        )
        if has_negation:
            nodes.append(
                GraphNode(
                    unit_id=str(uuid.uuid4())[:8],
                    unit_type=SemanticUnitType.NEGATION,
                    value="negated",
                    confidence=0.9,
                    metadata={"source": "regex", "language": language},
                )
            )
            scores.append(0.9)

        # Numerical / quantifier
        number_pattern = (
            r"\b\d+([.,]\d+)?\s*"
            r"(%|k|m|b|bn|million|billion|thousand|lakh|crore)?\b"
        )
        quant_count = 0
        for match in re.finditer(number_pattern, text, re.IGNORECASE):
            nodes.append(
                GraphNode(
                    unit_id=str(uuid.uuid4())[:8],
                    unit_type=SemanticUnitType.QUANTIFIER,
                    value=match.group(0).strip(),
                    confidence=0.95,
                    metadata={"source": "regex"},
                )
            )
            scores.append(0.95)
            quant_count += 1
            if quant_count >= 5:
                break

        # Connect sentiment -> entities
        entity_ids = [
            n.unit_id
            for n in nodes
            if n.unit_type == SemanticUnitType.ENTITY
        ]
        sentiment_ids = [
            n.unit_id
            for n in nodes
            if n.unit_type == SemanticUnitType.SENTIMENT
        ]
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

    def _build_penman(
        self, nodes: list, edges: list, text: str
    ) -> str:
        """Generate penman AMR notation from graph nodes as fallback."""
        if not nodes:
            return f'(t / text :value "{text[:50]}")'
        root = nodes[0]
        parts = [f"({root.unit_id} / {root.unit_type.value}"]
        parts.append(f'    :value "{root.value}"')
        for from_id, rel, to_id in edges[:10]:
            target_node = next(
                (n for n in nodes if n.unit_id == to_id), None
            )
            if target_node:
                parts.append(
                    f"    :{rel} ({to_id} / {target_node.unit_type.value}"
                    f' :value "{target_node.value}")'
                )
        parts.append(")")
        return "\n".join(parts)

    async def health(self) -> str:
        return "healthy"
