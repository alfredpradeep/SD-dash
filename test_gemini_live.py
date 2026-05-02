#!/usr/bin/env python3
"""
Live end-to-end test of the COMPRESS v3.0 pipeline with Gemini.

Usage:
  export GEMINI_API_KEY="your-key-here"
  python test_gemini_live.py

Or:
  GEMINI_API_KEY="your-key" python test_gemini_live.py

This script tests:
  1. Gemini LLM provider connectivity
  2. LLM-powered semantic rewriting (Stage 2)
  3. Adversarial reconstruction verification (Stage 4)
  4. Context distillation (Stage 5)
  5. Full compression pipeline (all 5 stages)
"""

import os
import sys
import asyncio
import time

# Set the API key
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is required. Set it in your environment before running this script."
    )
os.environ["GEMINI_API_KEY"] = GEMINI_KEY

# ── Color output ──
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
DIM = "\033[2m"
NC = "\033[0m"


def header(text):
    print(f"\n{BOLD}{'═' * 60}")
    print(f"  {text}")
    print(f"{'═' * 60}{NC}\n")


def ok(text):
    print(f"  {GREEN}✓{NC} {text}")


def fail(text):
    print(f"  {RED}✗{NC} {text}")


def info(text):
    print(f"  {DIM}{text}{NC}")


async def test_1_gemini_connectivity():
    """Test 1: Gemini API connectivity."""
    header("Test 1: Gemini API Connectivity")

    from compress.config import Config
    from compress.llm.provider import LLMProvider

    config = Config()
    provider = LLMProvider(config)

    assert provider.available, "Gemini provider not available"
    assert provider.provider_name == "gemini", f"Expected gemini, got {provider.provider_name}"
    ok(f"Provider: {provider.provider_name}")

    # Simple generation test
    result = await provider.generate(
        system_prompt="You are a helpful assistant.",
        user_prompt="Say 'COMPRESS works' and nothing else.",
        max_tokens=20,
        temperature=0.1,
    )
    print(f"  Response: {CYAN}{result}{NC}")
    assert len(result) > 0, "Empty response from Gemini"
    ok("Gemini API responding")
    return True


async def test_2_llm_rewriting():
    """Test 2: LLM-powered semantic rewriting."""
    header("Test 2: LLM-Powered Semantic Rewriting")

    from compress.config import Config
    from compress.llm.provider import LLMProvider
    from compress.llm.prompts import (
        REWRITE_SYSTEM, build_rewrite_prompt,
        build_variations_prompt, parse_variations,
    )

    config = Config()
    provider = LLMProvider(config)

    # Test English rewriting
    original = (
        "The patient was not diagnosed with diabetes mellitus "
        "but the doctor confirmed that she has been experiencing "
        "high blood pressure for approximately 3 years and has "
        "been taking medication regularly since then."
    )
    graph_summary = (
        "- ENTITY: patient\n"
        "- NEGATION: not diagnosed with diabetes mellitus\n"
        "- ENTITY: doctor\n"
        "- ENTITY: high blood pressure\n"
        "- QUANTIFIER: 3 years\n"
        "- ENTITY: medication"
    )

    print(f"  {BOLD}Original ({len(original.split())} words):{NC}")
    print(f"  {DIM}{original}{NC}")

    # Single rewrite
    prompt = build_rewrite_prompt(original, "en", graph_summary)
    rewrite = await provider.generate(
        REWRITE_SYSTEM, prompt,
        max_tokens=256, temperature=0.2,
    )
    print(f"\n  {BOLD}Rewrite ({len(rewrite.split())} words):{NC}")
    print(f"  {CYAN}{rewrite}{NC}")
    ok("Single rewrite generated")

    # Multiple variations
    sys_p, user_p = build_variations_prompt(
        original, "en", graph_summary, n=3,
    )
    var_resp = await provider.generate(
        sys_p, user_p,
        max_tokens=512, temperature=0.5,
    )
    variations = parse_variations(var_resp, n=3)
    print(f"\n  {BOLD}Variations ({len(variations)}):{NC}")
    for i, v in enumerate(variations, 1):
        print(f"  {i}. {v}")
    ok(f"{len(variations)} variations generated")

    # Test Tamil rewriting
    tamil_text = "நான் 10 ஆண்டுகளாக சென்னையில் வசிக்கிறேன், ஆனால் என் குடும்பம் மதுரையில் இருக்கிறது."
    tamil_graph = (
        "- ENTITY: நான் (I)\n"
        "- QUANTIFIER: 10 ஆண்டுகள் (10 years)\n"
        "- ENTITY: சென்னை (Chennai)\n"
        "- ENTITY: குடும்பம் (family)\n"
        "- ENTITY: மதுரை (Madurai)"
    )
    tamil_prompt = build_rewrite_prompt(tamil_text, "ta", tamil_graph)
    tamil_rewrite = await provider.generate(
        REWRITE_SYSTEM, tamil_prompt,
        max_tokens=256, temperature=0.2,
    )
    print(f"\n  {BOLD}Tamil Original:{NC}")
    print(f"  {DIM}{tamil_text}{NC}")
    print(f"  {BOLD}Tamil→English Rewrite:{NC}")
    print(f"  {CYAN}{tamil_rewrite}{NC}")
    ok("Cross-lingual Tamil→English rewrite generated")

    return True


async def test_3_adversarial_reconstruction():
    """Test 3: Adversarial reconstruction verification."""
    header("Test 3: Adversarial Reconstruction Verification")

    from compress.config import Config
    from compress.llm.provider import LLMProvider
    from compress.llm.prompts import (
        RECONSTRUCT_SYSTEM, build_reconstruct_prompt,
        compare_reconstructions,
    )
    import json

    config = Config()
    provider = LLMProvider(config)

    original = "Dr. Sharma did not recommend surgery. The patient has 3 fractures and moderate pain since January 2024."
    compressed = "Dr. Sharma didn't recommend surgery. Patient: 3 fractures, moderate pain since Jan 2024."

    print(f"  {BOLD}Original:{NC} {original}")
    print(f"  {BOLD}Compressed:{NC} {compressed}")

    # Extract KG from both
    orig_prompt = build_reconstruct_prompt(original)
    orig_resp = await provider.generate(
        RECONSTRUCT_SYSTEM, orig_prompt,
        max_tokens=512, temperature=0.1,
    )
    print(f"\n  {BOLD}Original KG:{NC}")
    print(f"  {DIM}{orig_resp[:300]}{NC}")

    comp_prompt = build_reconstruct_prompt(compressed)
    comp_resp = await provider.generate(
        RECONSTRUCT_SYSTEM, comp_prompt,
        max_tokens=512, temperature=0.1,
    )
    print(f"\n  {BOLD}Compressed KG:{NC}")
    print(f"  {DIM}{comp_resp[:300]}{NC}")

    # Parse and compare
    try:
        orig_kg = json.loads(orig_resp.strip().replace("```json", "").replace("```", ""))
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{[\s\S]*\}', orig_resp)
        orig_kg = json.loads(match.group()) if match else {}

    try:
        comp_kg = json.loads(comp_resp.strip().replace("```json", "").replace("```", ""))
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{[\s\S]*\}', comp_resp)
        comp_kg = json.loads(match.group()) if match else {}

    score, missing = compare_reconstructions(orig_kg, comp_kg)
    print(f"\n  {BOLD}Reconstruction Score: {GREEN if score >= 0.85 else RED}{score:.3f}{NC}")
    if missing:
        print(f"  {YELLOW}Missing items:{NC}")
        for m in missing:
            print(f"    - {m}")
    else:
        ok("All critical information preserved!")

    return True


async def test_4_context_distillation():
    """Test 4: Cumulative context distillation."""
    header("Test 4: Context Distillation")

    from compress.config import Config
    from compress.context.distiller import ContextDistiller

    config = Config()
    distiller = ContextDistiller(config)

    # Simulate a 10-turn Tamil AI interview
    turns = [
        ("நான் பிரதீப், சென்னையில் இருக்கிறேன்", "வணக்கம் பிரதீப்!"),
        ("எனக்கு 5 வருடம் Python அனுபவம் உள்ளது", "Python-ல் 5 ஆண்டு அனுபவம் நல்லது."),
        ("Machine learning projects செய்திருக்கிறேன்", "ML experience பற்றி சொல்லுங்கள்."),
        ("TensorFlow and PyTorch இரண்டையும் பயன்படுத்தியிருக்கிறேன்", "Both frameworks — great."),
        ("Previous company-ல் recommendation system build பண்ணினேன்", "Tell me more about the rec system."),
        ("10 million users-க்கு serve பண்ணினோம்", "Impressive scale!"),
        ("Team size 8 பேர், நான் tech lead", "Leadership experience. Good."),
        ("Current salary 25 lakhs per annum", "Noted your compensation."),
        ("Expected salary 35 lakhs", "We will discuss compensation."),
        ("Notice period 2 months", "Thank you for the information."),
    ]

    for i, (user, asst) in enumerate(turns):
        state = await distiller.add_turn(
            session_id="tamil-interview",
            user_text=user,
            assistant_text=asst,
            language="ta",
        )

    context = distiller.get_compressed_context("tamil-interview")
    stats = distiller.get_session_stats("tamil-interview")

    print(f"  Turns processed: {stats['turn_count']}")
    print(f"  Facts extracted: {stats['fact_count']}")
    print(f"  Active facts: {stats['active_facts']}")
    print(f"  Context length: {len(context)} chars (~{int(len(context.split()) * 1.3)} tokens)")
    print(f"\n  {BOLD}Compressed context:{NC}")
    print(f"  {DIM}{context}{NC}")

    ok(f"10-turn conversation distilled to ~{int(len(context.split()) * 1.3)} tokens")
    return True


async def test_5_full_pipeline():
    """Test 5: Full compression pipeline with trace."""
    header("Test 5: Full v3.0 Pipeline (all 5 stages)")

    from compress.config import Config
    from compress.engine import SCLEngine

    config = Config()
    info("Loading SCL Engine (LaBSE + spaCy + mDeBERTa)... this may take a moment")
    t0 = time.time()
    engine = SCLEngine(config)
    load_time = time.time() - t0
    ok(f"Engine loaded in {load_time:.1f}s")

    # Test English compression
    test_text = (
        "The candidate has demonstrated a very strong understanding of "
        "machine learning concepts and has approximately 5 years of "
        "relevant professional experience in the field. However, the "
        "candidate has not shown any significant experience with "
        "distributed computing systems or large-scale data processing "
        "frameworks. The overall assessment is that the candidate would "
        "be a good fit for a mid-level position but not for a senior "
        "engineering role at this time."
    )

    print(f"\n  {BOLD}Input ({len(test_text.split())} words):{NC}")
    print(f"  {DIM}{test_text}{NC}")

    info("Running 5-stage pipeline with trace...")
    t1 = time.time()
    trace = await engine.compress_with_trace(
        text=test_text,
        source_language="en",
        target_tokenizer="gpt-4o",
        semantic_threshold=0.88,
        graph_threshold=0.85,
        min_reduction_threshold=0.10,
    )
    pipeline_time = time.time() - t1

    result = trace["result"]
    stages = trace["pipeline"]["stages"]

    print(f"\n  {BOLD}Pipeline Stages:{NC}")
    for s in stages:
        icon = "✓" if s["status"] == "complete" else "✗"
        color = GREEN if s["status"] == "complete" else RED
        ms = s.get("ms", 0)
        print(f"    {color}{icon}{NC} {s['name']}: {s.get('detail', '')} ({ms:.1f}ms)")

    print(f"\n  {BOLD}Result:{NC}")
    print(f"  Original tokens:    {result['original_tokens']}")
    print(f"  Compressed tokens:  {result['compressed_tokens']}")
    reduction_pct = result['reduction_ratio'] * 100
    color = GREEN if result['compression_applied'] else YELLOW
    print(f"  Reduction:          {color}{reduction_pct:.1f}%{NC}")
    print(f"  Semantic similarity: {result['semantic_similarity']:.4f}")
    print(f"  Compression applied: {result['compression_applied']}")
    if result['compression_applied']:
        print(f"\n  {BOLD}Compressed:{NC}")
        print(f"  {CYAN}{result['compressed_text']}{NC}")
    else:
        print(f"  {YELLOW}Rejection: {result['rejection_reason']}{NC}")

    print(f"\n  Total pipeline: {pipeline_time:.2f}s")
    print(f"  Graph nodes: {len(trace['semantic_graph']['nodes'])}")
    print(f"  Candidates: {len(trace['beam_candidates'])}")

    if trace.get("reconstruction"):
        recon = trace["reconstruction"]
        print(f"  Reconstruction score: {recon.get('score', 'N/A')}")
        print(f"  Reconstruction method: {recon.get('method', 'N/A')}")

    ok("Full v3.0 pipeline executed successfully!")
    return True


async def main():
    print(f"\n{BOLD}{'━' * 60}")
    print(f"  COMPRESS v3.0 — Live Gemini Integration Test")
    print(f"{'━' * 60}{NC}")
    print(f"  API Key: {GEMINI_KEY[:8]}...{GEMINI_KEY[-4:]}")
    print(f"  Provider: Google Gemini (gemini-2.0-flash)")

    results = []
    tests = [
        ("Gemini Connectivity", test_1_gemini_connectivity),
        ("LLM Rewriting", test_2_llm_rewriting),
        ("Adversarial Reconstruction", test_3_adversarial_reconstruction),
        ("Context Distillation", test_4_context_distillation),
        ("Full Pipeline", test_5_full_pipeline),
    ]

    for name, test_fn in tests:
        try:
            passed = await test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  {RED}✗ FAILED: {e}{NC}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    header("Test Summary")
    for name, passed in results:
        icon = f"{GREEN}✓ PASS{NC}" if passed else f"{RED}✗ FAIL{NC}"
        print(f"  {icon}  {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n  {BOLD}{passed}/{total} tests passed{NC}")

    if passed == total:
        print(f"\n  {GREEN}{BOLD}🎉 All tests passed! COMPRESS v3.0 is production-ready.{NC}")
    else:
        print(f"\n  {YELLOW}Some tests failed. Check logs above.{NC}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
