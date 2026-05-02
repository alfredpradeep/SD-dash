"""
Gap 16: Adversarial Tokenizer Probing

Finds worst-case token count inputs via optimization to discover tokenizer inefficiencies.
Tests tokenizer robustness against adversarial inputs designed to maximize token inflation.

Uses multiple attack strategies:
- Unicode substitution (visually similar characters)
- Whitespace injection (invisible characters)
- Encoding exploits (combining diacriticals, normalization)
- Repetition amplification (expensive token patterns)
- Mixed script attacks (cross-script tokenization breaks)
"""

from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Dict, Any
import unicodedata
import re


@dataclass
class AdversarialResult:
    """Result of a single adversarial attack on input text."""
    original_text: str
    adversarial_text: str
    original_tokens: int
    adversarial_tokens: int
    token_inflation_pct: float
    cost_inflation_pct: float
    vulnerability_type: str
    severity: str  # "critical" / "high" / "medium" / "low"
    explanation: str
    mitigation: str


@dataclass
class ProbeReport:
    """Comprehensive report of adversarial probing results."""
    results: List[AdversarialResult]
    worst_case_inflation: float
    average_inflation: float
    critical_count: int
    high_count: int
    model_vulnerability_score: float  # 0-100
    recommendations: List[str]


class AdversarialProber:
    """
    Probes tokenizer weaknesses by applying adversarial transformations.

    Implements multiple attack strategies to discover tokenizer inefficiencies
    and token count inflation vulnerabilities.
    """

    def __init__(self, models: List[str]):
        """
        Initialize the adversarial prober.

        Args:
            models: List of model identifiers to probe
        """
        self.models = models

        # Unicode substitution map: ASCII -> visually similar Unicode
        self.unicode_map = {
            'a': 'а',  # Cyrillic 'a'
            'c': 'с',  # Cyrillic 'c'
            'e': 'е',  # Cyrillic 'e'
            'h': 'һ',  # Cyrillic 'h'
            'o': 'ο',  # Greek 'o'
            'p': 'р',  # Cyrillic 'p'
            'x': 'х',  # Cyrillic 'x'
            'y': 'у',  # Cyrillic 'y'
            'A': 'А',  # Cyrillic 'A'
            'B': 'В',  # Cyrillic 'B'
            'C': 'С',  # Cyrillic 'C'
            'E': 'Е',  # Cyrillic 'E'
            'H': 'Н',  # Cyrillic 'H'
            'K': 'К',  # Cyrillic 'K'
            'M': 'М',  # Cyrillic 'M'
            'O': 'О',  # Cyrillic 'O'
            'P': 'Р',  # Cyrillic 'P'
            'T': 'Т',  # Cyrillic 'T'
            'X': 'Х',  # Cyrillic 'X'
        }

    def _unicode_substitution(self, text: str) -> List[Tuple[str, str]]:
        """
        Replace ASCII characters with visually similar Unicode characters.

        These often tokenize as unknown/multi-byte tokens, inflating token count.

        Args:
            text: Original input text

        Returns:
            List of (adversarial_text, vulnerability_type) tuples
        """
        results = []

        # Strategy 1: Replace individual ASCII chars with visually similar Unicode
        adversarial = text
        for ascii_char, unicode_char in self.unicode_map.items():
            adversarial = adversarial.replace(ascii_char, unicode_char)

        if adversarial != text:
            results.append((adversarial, "unicode_substitution"))

        # Strategy 2: Partial substitution (every other occurrence)
        adversarial = ""
        replace_next = True
        for char in text:
            if char in self.unicode_map and replace_next:
                adversarial += self.unicode_map[char]
                replace_next = False
            else:
                adversarial += char
                if char in self.unicode_map:
                    replace_next = True

        if adversarial != text:
            results.append((adversarial, "unicode_substitution_partial"))

        return results

    def _whitespace_injection(self, text: str) -> List[Tuple[str, str]]:
        """
        Insert zero-width and invisible characters between words.

        Tests if tokenizer handles invisible Unicode efficiently.

        Args:
            text: Original input text

        Returns:
            List of (adversarial_text, vulnerability_type) tuples
        """
        results = []

        # Zero-width space (U+200B)
        zero_width_space = '\u200b'
        # Zero-width joiner (U+200D)
        zero_width_joiner = '\u200d'
        # Soft hyphen (U+00AD)
        soft_hyphen = '\u00ad'
        # Zero-width non-joiner (U+200C)
        zero_width_non_joiner = '\u200c'

        # Strategy 1: Insert zero-width spaces between words
        adversarial = re.sub(r'(\w)(\s+)', r'\1' + zero_width_space + r'\2', text)
        if adversarial != text:
            results.append((adversarial, "zwsp_injection"))

        # Strategy 2: Insert soft hyphens in long words
        adversarial = ""
        for word in text.split():
            if len(word) > 4:
                # Insert soft hyphen every 3 chars
                new_word = ""
                for i, char in enumerate(word):
                    new_word += char
                    if (i + 1) % 3 == 0 and i < len(word) - 1:
                        new_word += soft_hyphen
                adversarial += new_word + " "
            else:
                adversarial += word + " "

        adversarial = adversarial.rstrip()
        if adversarial != text:
            results.append((adversarial, "soft_hyphen_injection"))

        # Strategy 3: Mix of zero-width joiners
        adversarial = re.sub(r'(\w)(\w)', r'\1' + zero_width_joiner + '\2', text)
        if adversarial != text:
            results.append((adversarial, "zwj_injection"))

        return results

    def _encoding_exploit(self, text: str) -> List[Tuple[str, str]]:
        """
        Use combining diacriticals and Unicode normalization variants.

        Different normalization forms (NFC vs NFD) can change token boundaries.

        Args:
            text: Original input text

        Returns:
            List of (adversarial_text, vulnerability_type) tuples
        """
        results = []

        # Strategy 1: NFD normalization (decomposed form)
        # e → e + combining acute accent
        adversarial_nfd = unicodedata.normalize('NFD', text)
        if adversarial_nfd != text:
            results.append((adversarial_nfd, "nfd_normalization"))

        # Strategy 2: Add combining diacriticals to vowels
        combining_map = {
            'a': 'a\u0308',  # a + diaeresis
            'e': 'e\u0308',  # e + diaeresis
            'i': 'i\u0308',  # i + diaeresis
            'o': 'o\u0308',  # o + diaeresis
            'u': 'u\u0308',  # u + diaeresis
            'A': 'A\u0308',  # A + diaeresis
            'E': 'E\u0308',  # E + diaeresis
            'I': 'I\u0308',  # I + diaeresis
            'O': 'O\u0308',  # O + diaeresis
            'U': 'U\u0308',  # U + diaeresis
        }

        adversarial = text
        for char, combined in combining_map.items():
            adversarial = adversarial.replace(char, combined)

        if adversarial != text:
            results.append((adversarial, "combining_diacriticals"))

        # Strategy 3: NFKD normalization (compatibility decomposition)
        adversarial_nfkd = unicodedata.normalize('NFKD', text)
        if adversarial_nfkd != text:
            results.append((adversarial_nfkd, "nfkd_normalization"))

        return results

    def _repetition_amplification(self, text: str) -> List[Tuple[str, str]]:
        """
        Strategically repeat patterns known to tokenize inefficiently.

        Focuses on creating expensive token sequences.

        Args:
            text: Original input text

        Returns:
            List of (adversarial_text, vulnerability_type) tuples
        """
        results = []

        # Strategy 1: Repetition of long numbers (expensive tokens)
        if re.search(r'\d+', text):
            adversarial = re.sub(r'(\d+)', r'\1\1', text)  # Double numbers
            results.append((adversarial, "number_repetition"))

        # Strategy 2: Repetition of rare subword patterns
        adversarial = ""
        words = text.split()
        for word in words:
            # Repeat word 3 times with spacing variations
            if len(word) > 3:
                adversarial += word + " " + word + " " + word + " "
            else:
                adversarial += word + " "

        adversarial = adversarial.rstrip()
        if adversarial != text:
            results.append((adversarial, "word_repetition"))

        # Strategy 3: Character-level repetition with spaces (breaks merging)
        adversarial = ""
        for word in text.split():
            # Add space between each character
            spaced_word = " ".join(word)
            adversarial += spaced_word + " "

        adversarial = adversarial.rstrip()
        if adversarial != text:
            results.append((adversarial, "character_spacing"))

        return results

    def _mixed_script_attack(self, text: str) -> List[Tuple[str, str]]:
        """
        Mix multiple scripts (Latin, CJK, Arabic, Cyrillic) in ways that break merge rules.

        Cross-script transitions often cause inefficient tokenization.

        Args:
            text: Original input text

        Returns:
            List of (adversarial_text, vulnerability_type) tuples
        """
        results = []

        # Strategy 1: Interleave Latin with CJK characters
        adversarial = ""
        cjk_chars = ['中', '文', '字', '符', '号']
        word_list = text.split()

        for i, word in enumerate(word_list):
            adversarial += word
            if i < len(word_list) - 1:
                # Add CJK character between words
                adversarial += cjk_chars[i % len(cjk_chars)] + " "

        if adversarial != text:
            results.append((adversarial, "latin_cjk_mix"))

        # Strategy 2: Interleave Latin with Arabic
        adversarial = ""
        arabic_chars = ['ع', 'ب', 'ت', 'ث', 'ج']
        word_list = text.split()

        for i, word in enumerate(word_list):
            adversarial += word
            if i < len(word_list) - 1:
                adversarial += arabic_chars[i % len(arabic_chars)] + " "

        if adversarial != text:
            results.append((adversarial, "latin_arabic_mix"))

        # Strategy 3: Mixed script within words (script switching)
        adversarial = ""
        for word in text.split():
            # Insert Cyrillic 'о' in the middle of English words
            if len(word) > 2:
                mid = len(word) // 2
                mixed_word = word[:mid] + 'о' + word[mid:]
                adversarial += mixed_word + " "
            else:
                adversarial += word + " "

        adversarial = adversarial.rstrip()
        if adversarial != text:
            results.append((adversarial, "intraword_script_mixing"))

        return results

    def _classify_severity(self, inflation_pct: float) -> str:
        """
        Classify severity based on token inflation percentage.

        Args:
            inflation_pct: Percentage of token inflation (0-100+)

        Returns:
            Severity level: "critical", "high", "medium", or "low"
        """
        if inflation_pct > 50:
            return "critical"
        elif inflation_pct > 25:
            return "high"
        elif inflation_pct > 10:
            return "medium"
        else:
            return "low"

    def _generate_recommendations(self, results: List[AdversarialResult]) -> List[str]:
        """
        Generate context-aware mitigation recommendations.

        Args:
            results: List of adversarial results

        Returns:
            List of mitigation recommendations
        """
        recommendations = []
        vuln_types = set(r.vulnerability_type for r in results)
        severity_levels = [r.severity for r in results]

        # Add Unicode normalization recommendation
        if any('unicode' in vt or 'normalization' in vt for vt in vuln_types):
            recommendations.append(
                "Normalize input text to NFC form before tokenization to handle "
                "Unicode variant issues"
            )

        # Add whitespace handling recommendation
        if any('whitespace' in vt or 'zwsp' in vt or 'zwj' in vt for vt in vuln_types):
            recommendations.append(
                "Strip zero-width and invisible characters (U+200B, U+200C, U+200D, U+00AD) "
                "from user input before processing"
            )

        # Add repetition safeguard recommendation
        if 'word_repetition' in vuln_types or 'character_spacing' in vuln_types:
            recommendations.append(
                "Implement input deduplication and normalize excessive spacing to prevent "
                "repetition-based token inflation attacks"
            )

        # Add mixed-script recommendation
        if any('mix' in vt for vt in vuln_types):
            recommendations.append(
                "Consider script-aware tokenization or validation to flag inputs with "
                "excessive script mixing as potentially adversarial"
            )

        # Critical/high severity recommendations
        if "critical" in severity_levels or "high" in severity_levels:
            recommendations.append(
                "Implement token budget validation at input time to reject or truncate "
                "inputs that exceed expected token expansion ratios"
            )
            recommendations.append(
                "Monitor token count inflation in production; flag requests with >30% "
                "token inflation for further inspection"
            )

        # General recommendations
        if not recommendations:
            recommendations.append(
                "Tokenizer shows reasonable robustness. Continue monitoring for new attack vectors."
            )

        return recommendations

    def probe(
        self,
        text: str,
        model: str,
        token_count_fn: Callable[[str, str], int]
    ) -> ProbeReport:
        """
        Run all attack strategies and generate comprehensive report.

        Args:
            text: Original input text to probe
            model: Model identifier to use for tokenization
            token_count_fn: Callable that returns token count for (text, model)
                           Must be compatible with tokenizer setup (tiktoken, transformers, etc.)

        Returns:
            ProbeReport with results, statistics, and recommendations
        """
        results = []
        original_token_count = token_count_fn(text, model)

        # Collect all attack strategies
        attack_strategies = [
            (self._unicode_substitution, "Unicode Substitution"),
            (self._whitespace_injection, "Whitespace Injection"),
            (self._encoding_exploit, "Encoding Exploit"),
            (self._repetition_amplification, "Repetition Amplification"),
            (self._mixed_script_attack, "Mixed Script Attack"),
        ]

        # Run each attack strategy
        for attack_fn, strategy_name in attack_strategies:
            adversarial_texts = attack_fn(text)

            for adversarial_text, vulnerability_type in adversarial_texts:
                adversarial_token_count = token_count_fn(adversarial_text, model)

                if adversarial_token_count > original_token_count:
                    inflation_count = adversarial_token_count - original_token_count
                    inflation_pct = (inflation_count / original_token_count) * 100

                    # Estimate cost inflation (assuming linear pricing model)
                    cost_inflation_pct = inflation_pct

                    severity = self._classify_severity(inflation_pct)

                    # Generate explanation
                    if inflation_pct > 50:
                        explanation = (
                            f"Severe token inflation detected via {vulnerability_type}. "
                            f"Input tokenizes {inflation_pct:.1f}% less efficiently than original. "
                            f"This vulnerability could be exploited for token exhaustion attacks."
                        )
                    elif inflation_pct > 25:
                        explanation = (
                            f"Significant token inflation via {vulnerability_type}. "
                            f"Adversarial input generates {inflation_pct:.1f}% more tokens. "
                            f"May impact cost efficiency and model behavior."
                        )
                    else:
                        explanation = (
                            f"Minor token inflation via {vulnerability_type}. "
                            f"Adversarial input generates {inflation_pct:.1f}% more tokens."
                        )

                    # Generate mitigation
                    mitigation_map = {
                        "unicode_substitution": "Validate and normalize Unicode characters to canonical forms",
                        "unicode_substitution_partial": "Apply comprehensive Unicode normalization (NFC form)",
                        "zwsp_injection": "Strip zero-width spaces before tokenization",
                        "soft_hyphen_injection": "Remove soft hyphens and normalize whitespace",
                        "zwj_injection": "Remove zero-width joiners from input",
                        "nfd_normalization": "Normalize input to NFC form before tokenization",
                        "combining_diacriticals": "Remove or normalize combining diacritical marks",
                        "nfkd_normalization": "Apply NFC normalization to handle compatibility variants",
                        "number_repetition": "Detect and deduplicate repeated numeric patterns",
                        "word_repetition": "Implement deduplication of repeated word sequences",
                        "character_spacing": "Normalize excessive spacing in input text",
                        "latin_cjk_mix": "Validate script mixing ratios; flag excessive mixing as suspicious",
                        "latin_arabic_mix": "Monitor cross-script token boundaries for inefficiency",
                        "intraword_script_mixing": "Reject or normalize words with mixed scripts",
                    }
                    mitigation = mitigation_map.get(
                        vulnerability_type,
                        "Implement input validation and normalization"
                    )

                    result = AdversarialResult(
                        original_text=text,
                        adversarial_text=adversarial_text,
                        original_tokens=original_token_count,
                        adversarial_tokens=adversarial_token_count,
                        token_inflation_pct=inflation_pct,
                        cost_inflation_pct=cost_inflation_pct,
                        vulnerability_type=vulnerability_type,
                        severity=severity,
                        explanation=explanation,
                        mitigation=mitigation,
                    )
                    results.append(result)

        # Calculate statistics
        if results:
            inflations = [r.token_inflation_pct for r in results]
            worst_case_inflation = max(inflations)
            average_inflation = sum(inflations) / len(inflations)
            critical_count = sum(1 for r in results if r.severity == "critical")
            high_count = sum(1 for r in results if r.severity == "high")

            # Calculate vulnerability score (0-100)
            # Based on worst case inflation and number of critical/high vulnerabilities
            base_score = min(100, worst_case_inflation)
            critical_penalty = critical_count * 15
            high_penalty = high_count * 5
            vulnerability_score = min(100, base_score + critical_penalty + high_penalty)
        else:
            worst_case_inflation = 0.0
            average_inflation = 0.0
            critical_count = 0
            high_count = 0
            vulnerability_score = 0.0

        # Generate recommendations
        recommendations = self._generate_recommendations(results)

        return ProbeReport(
            results=results,
            worst_case_inflation=worst_case_inflation,
            average_inflation=average_inflation,
            critical_count=critical_count,
            high_count=high_count,
            model_vulnerability_score=vulnerability_score,
            recommendations=recommendations,
        )
