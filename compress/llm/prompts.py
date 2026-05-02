"""
Prompt engineering for LLM-powered semantic compression.

Two prompt types:
  1. REWRITE — Generate token-efficient rewrites of input text
  2. RECONSTRUCT — Adversarial reconstruction from compressed text

The prompts are the least defensible part of the system.
The verification pipeline is the IP.
"""


REWRITE_SYSTEM = """You are a semantic compression engine. Your job is to rewrite text to use fewer tokens while preserving EVERY piece of meaning.

RULES:
1. PRESERVE all entities, numbers, names, dates, negations, and relationships
2. NEVER drop negation — "not received" must stay negative
3. NEVER change quantities or numbers
4. NEVER add information not in the original
5. Use shorter synonyms and simpler grammar
6. Remove redundant words, filler phrases, and unnecessary qualifiers
7. Combine sentences where possible without losing meaning
8. For non-English input: rewrite in English if it preserves all meaning (the target system understands English better and it uses fewer tokens)
9. If the text is already concise, return it unchanged — don't force compression

OUTPUT FORMAT: Return ONLY the rewritten text, nothing else. No explanations, no labels, no quotes."""


def build_rewrite_prompt(
    text: str,
    language: str,
    semantic_graph_summary: str,
    target_tokenizer: str = "gpt-4o",
) -> str:
    """
    Build the user prompt for rewrite generation.

    Includes the semantic graph summary so the LLM knows what
    must be preserved (entities, negations, quantities, etc.)
    """
    lang_instruction = ""
    if language != "en":
        lang_instruction = (
            f"\n\nIMPORTANT: The input is in {language}. You may rewrite in English "
            f"if it preserves all meaning — English uses ~3-4x fewer tokens in {target_tokenizer}. "
            f"If the meaning is culturally specific or has no direct English equivalent, "
            f"keep those parts in {language}."
        )

    return f"""Rewrite this text to use minimum tokens while preserving ALL meaning.

INPUT TEXT ({language}):
{text}

SEMANTIC UNITS THAT MUST BE PRESERVED:
{semantic_graph_summary}

TARGET TOKENIZER: {target_tokenizer} (optimise for this tokenizer's vocabulary){lang_instruction}

Rewrite:"""


REWRITE_VARIATIONS_SYSTEM = """You are a semantic compression engine. Generate {n} different rewrites of the input text, each using fewer tokens while preserving ALL meaning.

RULES:
1. Each rewrite must preserve ALL entities, numbers, negations, and relationships
2. Each rewrite should be different — vary sentence structure, word choice, level of compression
3. Order from most compressed to least compressed
4. For non-English input: at least one rewrite should be in English (if meaning is fully preserved)
5. NEVER drop negation, quantities, or named entities

OUTPUT FORMAT: Return exactly {n} rewrites, one per line, numbered 1-{n}. No explanations."""


def build_variations_prompt(
    text: str,
    language: str,
    semantic_graph_summary: str,
    n: int = 5,
    target_tokenizer: str = "gpt-4o",
) -> tuple[str, str]:
    """
    Build system + user prompt for generating N rewrite variations.

    Returns (system_prompt, user_prompt).
    """
    system = REWRITE_VARIATIONS_SYSTEM.format(n=n)

    lang_note = ""
    if language != "en":
        lang_note = (
            f"\n\nThe input is in {language}. Include at least one English rewrite "
            f"and at least one {language} rewrite."
        )

    user = f"""Generate {n} different rewrites of this text, each shorter than the original.

INPUT TEXT ({language}):
{text}

SEMANTIC UNITS TO PRESERVE:
{semantic_graph_summary}

TARGET TOKENIZER: {target_tokenizer}{lang_note}

Rewrites:"""

    return system, user


def parse_variations(response: str, n: int = 5) -> list[str]:
    """
    Parse numbered variations from LLM response.

    Handles formats like:
    1. First rewrite
    2. Second rewrite
    Or:
    1) First rewrite
    2) Second rewrite
    """
    import re
    lines = response.strip().split("\n")
    results = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Remove numbering: "1. ", "1) ", "1: ", etc.
        cleaned = re.sub(r'^\d+[\.\)\:]\s*', '', line).strip()
        # Remove quotes if present
        cleaned = cleaned.strip('"\'')
        if cleaned and len(cleaned) > 5:
            results.append(cleaned)

    return results[:n]


# ── Adversarial Reconstruction Prompts ──

RECONSTRUCT_SYSTEM = """You are a knowledge extraction engine. Given a piece of text, extract ALL factual information as structured JSON.

Extract these categories:
- entities: named things (people, places, organizations, products)
- quantities: numbers, amounts, percentages, dates
- negations: things that are NOT true, denied, absent
- sentiments: emotional valence (positive/negative/neutral + intensity)
- temporal: time references (past/present/future, specific dates)
- relations: who did what to whom, cause-effect, conditions
- claims: factual assertions made in the text

OUTPUT FORMAT: Valid JSON only, no markdown, no explanations."""


def build_reconstruct_prompt(text: str) -> str:
    """Build user prompt for adversarial reconstruction."""
    return f"""Extract ALL factual information from this text as JSON:

TEXT:
{text}

JSON:"""


def compare_reconstructions(
    original_kg: dict,
    compressed_kg: dict,
) -> tuple[float, list[str]]:
    """
    Compare two knowledge graphs extracted from original and compressed text.

    Returns:
        (score: float 0-1, missing_items: list[str])

    Score = (matched items) / (total original items)
    """
    missing = []
    total = 0
    matched = 0

    for category in [
        "entities", "quantities", "negations", "sentiments",
        "temporal", "relations", "claims",
    ]:
        orig_items = original_kg.get(category, [])
        comp_items = compressed_kg.get(category, [])

        if isinstance(orig_items, list):
            total += len(orig_items)
            # Flatten to strings for comparison
            orig_strs = {
                str(item).lower().strip() for item in orig_items
            }
            comp_strs = {
                str(item).lower().strip() for item in comp_items
            }

            for orig_str in orig_strs:
                # Check for exact or fuzzy match
                if orig_str in comp_strs:
                    matched += 1
                elif any(
                    _fuzzy_match(orig_str, cs) for cs in comp_strs
                ):
                    matched += 1
                else:
                    missing.append(f"[{category}] {orig_str}")
        elif isinstance(orig_items, str):
            total += 1
            if (
                str(orig_items).lower().strip()
                == str(comp_items).lower().strip()
            ):
                matched += 1
            elif _fuzzy_match(
                str(orig_items).lower(), str(comp_items).lower()
            ):
                matched += 1
            else:
                missing.append(
                    f"[{category}] {orig_items}"
                )

    score = matched / max(total, 1)
    return score, missing


def _fuzzy_match(a: str, b: str) -> bool:
    """Check if two strings are fuzzy matches."""
    if not a or not b:
        return False
    # Containment check
    if a in b or b in a:
        return True
    # Word overlap
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a:
        return False
    overlap = len(words_a & words_b) / len(words_a)
    return overlap >= 0.6
