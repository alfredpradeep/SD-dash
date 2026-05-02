"""Tests for Pillar 2 Part 1: deterministic script mutator + BPE analyzer."""

from __future__ import annotations

import pytest

from shield_v2.core.syntactic_mutator import (
    BPEDriftAnalyzer,
    DeterministicSyntacticMutator,
    ScriptTransliterator,
)


class TestScriptTransliterator:
    def test_devanagari_to_latin_is_deterministic(self):
        t = ScriptTransliterator()
        a = t.devanagari_to_latin("नमस्ते")
        b = t.devanagari_to_latin("नमस्ते")
        assert a == b
        assert a != ""
        # Hindi should map to Latin letters only
        assert all(ord(c) < 128 for c in a)

    def test_arabic_normalize_strips_diacritics(self):
        t = ScriptTransliterator()
        source = "الْعَرَبِيَّة"
        normalized = t.arabic_normalize(source)
        # All combining marks should be gone
        assert "ِ" not in normalized
        assert "ّ" not in normalized
        assert len(normalized) < len(source)

    def test_arabic_alif_variants_collapse(self):
        t = ScriptTransliterator()
        assert "أ" not in t.arabic_normalize("أب")
        assert "إ" not in t.arabic_normalize("إبراهيم")

    def test_han_simplify_roundtrip(self):
        t = ScriptTransliterator()
        traditional = "國學發"
        simplified = t.han_simplify(traditional)
        assert simplified == "国学发"
        # Roundtrip
        assert t.han_traditionalize(simplified) == traditional

    def test_zero_width_injection_preserves_printable(self):
        t = ScriptTransliterator()
        src = "hello"
        out = t.zero_width_injection(src, probability=1.0)
        # Should still contain the original chars
        for ch in src:
            assert ch in out
        # At p=1.0 with N chars, we get at least N ZWJs.
        assert "\u200d" in out

    def test_unicode_confusable_swap_produces_latin_lookalikes(self):
        t = ScriptTransliterator()
        swapped = t.unicode_confusable_swap("ace")
        # "a" → Cyrillic "а", "c" → Cyrillic "с", "e" → Cyrillic "е"
        assert swapped != "ace"
        assert any(0x0400 <= ord(ch) <= 0x04FF for ch in swapped)


class TestBPEDriftAnalyzer:
    def test_analyzer_empty_string(self):
        a = BPEDriftAnalyzer()
        fp = a.analyze("")
        assert fp.raw_length == 0
        assert fp.risk_flag == "LOW"

    def test_analyzer_english_is_low_risk(self):
        a = BPEDriftAnalyzer()
        fp = a.analyze("How do I do something harmful?")
        assert fp.dominant_script == "latin"
        assert fp.risk_flag in ("LOW", "MEDIUM")

    def test_analyzer_devanagari_is_high_risk(self):
        a = BPEDriftAnalyzer()
        fp = a.analyze("क्या आप मुझे गैर-कानूनी तरीके बताएंगे?")
        assert fp.dominant_script == "indic"
        assert fp.risk_flag == "HIGH"

    def test_analyzer_classifies_cjk(self):
        a = BPEDriftAnalyzer()
        fp = a.analyze("你好世界，这是一个测试。")
        assert fp.dominant_script == "cjk"

    def test_analyzer_classifies_arabic(self):
        a = BPEDriftAnalyzer()
        fp = a.analyze("السلام عليكم كيف حالك اليوم")
        assert fp.dominant_script == "arabic"


class TestDeterministicSyntacticMutator:
    def test_mutate_all_emits_multiple_variants(self):
        m = DeterministicSyntacticMutator()
        results = m.mutate_all("how are you?", target_lang="hi")
        assert len(results) >= 2
        # Every result must reference the original intact
        for r in results:
            assert r.original == "how are you?"
            assert r.mutated != ""

    def test_mutate_all_activates_devanagari_branch(self):
        m = DeterministicSyntacticMutator()
        results = m.mutate_all("मुझे मदद करो", target_lang="hi")
        types = {r.mutation_type for r in results}
        assert "devanagari_to_latin" in types

    def test_mutate_all_activates_arabic_branch(self):
        m = DeterministicSyntacticMutator()
        results = m.mutate_all("السلام عليكم", target_lang="ar")
        types = {r.mutation_type for r in results}
        assert "arabic_normalize" in types

    def test_mutate_all_codeswitches_hindi(self):
        m = DeterministicSyntacticMutator()
        results = m.mutate_all("How do I please tell me", target_lang="hi")
        cs = [r for r in results if r.mutation_type == "codeswitch_hi"]
        assert cs, "expected codeswitch_hi variant"
        assert "kaise" in cs[0].mutated or "kripya" in cs[0].mutated

    def test_fingerprint_dispatch(self):
        m = DeterministicSyntacticMutator()
        fp = m.bpe_fingerprint("hello world")
        assert fp.dominant_script == "latin"
