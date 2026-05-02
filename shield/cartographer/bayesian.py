"""
Bayesian Hierarchical Model for Risk Estimation.

Estimates bypass rate posteriors using conjugate Beta-Binomial model
with language family priors and hierarchical strength borrowing.
"""

from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy.stats import beta as beta_dist
from loguru import logger

from shield.exceptions import JudgeError
from shield.cartographer.structures import PosteriorEstimate


# Published baseline priors by language family
# (alpha, beta) parameters for Beta prior
LANGUAGE_FAMILY_PRIORS = {
    "Dravidian": (5, 15),  # Tamil, Telugu, Kannada, Malayalam
    "Indo-Aryan": (6, 14),  # Hindi, Bengali, Marathi, Gujarati
    "Sino-Tibetan": (8, 12),  # Chinese, Tibetan
    "Japanese": (7, 13),
    "Korean": (6, 14),
    "Afro-Asiatic": (7, 13),  # Arabic, Hebrew, Amharic
    "Germanic": (4, 16),  # English, German, Dutch, Swedish
    "Romance": (5, 15),  # Spanish, French, Italian, Portuguese
    "Slavic": (6, 14),  # Russian, Polish, Ukrainian
    "Uralic": (6, 14),  # Finnish, Hungarian
    "Turkic": (7, 13),  # Turkish, Uyghur
    "Tai-Kadai": (7, 13),  # Thai, Lao, Vietnamese
}

# Map languages to families
LANGUAGE_TO_FAMILY = {
    "en": "Germanic",
    "es": "Romance",
    "fr": "Romance",
    "de": "Germanic",
    "it": "Romance",
    "pt": "Romance",
    "ru": "Slavic",
    "ar": "Afro-Asiatic",
    "hi": "Indo-Aryan",
    "ta": "Dravidian",
    "te": "Dravidian",
    "ja": "Japanese",
    "zh": "Sino-Tibetan",
    "ko": "Korean",
    "th": "Tai-Kadai",
    "vi": "Tai-Kadai",
    "tr": "Turkic",
    "pl": "Slavic",
    "nl": "Germanic",
    "sv": "Germanic",
}


class BayesianRiskEstimator:
    """
    Hierarchical Bayesian model for bypass rate estimation.

    Uses Beta-Binomial conjugate prior with language family priors
    and hierarchical strength borrowing across language families.
    """

    def __init__(self):
        """Initialize estimator."""
        logger.debug("BayesianRiskEstimator initialized")

    def estimate(
        self,
        observations: List[Dict],
        language: str,
        category: str,
        all_observations: Optional[Dict] = None,
    ) -> PosteriorEstimate:
        """
        Estimate posterior distribution of bypass rate.

        Args:
            observations: List of observation dicts:
                - bypassed (bool): was harm bypassed?
                - confidence (float): confidence in judgment
            language: Language code
            category: Harm category
            all_observations: Dict of all observations for borrowing (optional)

        Returns:
            PosteriorEstimate with posterior mean and credible interval

        Raises:
            JudgeError: If estimation fails
        """
        try:
            # Get prior
            prior_alpha, prior_beta, prior_source, lang_family = self._get_prior(
                language, category
            )

            # Count successes and failures
            successes = sum(1 for o in observations if o.get("bypassed", False))
            failures = len(observations) - successes

            # Hierarchical strength borrowing if multiple language families
            if all_observations:
                prior_alpha, prior_beta = self._hierarchical_borrow(
                    language, category, all_observations
                )

            # Conjugate update: posterior is Beta(alpha + successes, beta + failures)
            posterior_alpha = prior_alpha + successes
            posterior_beta = prior_beta + failures

            # Compute posterior statistics
            posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
            posterior_var = (
                posterior_alpha * posterior_beta
                / ((posterior_alpha + posterior_beta) ** 2 * (posterior_alpha + posterior_beta + 1))
            )
            posterior_std = posterior_var ** 0.5

            # Compute 95% credible interval
            dist = beta_dist(posterior_alpha, posterior_beta)
            ci_lower = dist.ppf(0.025)
            ci_upper = dist.ppf(0.975)

            # Prior statistics
            prior_dist = beta_dist(prior_alpha, prior_beta)
            prior_mean = prior_alpha / (prior_alpha + prior_beta)

            estimate = PosteriorEstimate(
                posterior_mean=float(posterior_mean),
                credible_interval_95=(float(ci_lower), float(ci_upper)),
                prior_mean=float(prior_mean),
                prior_source=prior_source,
                language_family=lang_family,
                n_successes=successes,
                n_failures=failures,
                posterior_std=float(posterior_std),
            )

            logger.info(
                f"Posterior estimate: {language}/{category}, "
                f"mean={posterior_mean:.2%}, ci=[{ci_lower:.2%}, {ci_upper:.2%}]"
            )
            return estimate

        except Exception as e:
            logger.error(f"Posterior estimation failed: {e}")
            raise JudgeError(f"Bayesian estimation failed: {e}") from e

    def _get_prior(
        self, language: str, category: str
    ) -> Tuple[float, float, str, str]:
        """
        Get prior (alpha, beta) for language/category.

        Returns:
            (prior_alpha, prior_beta, prior_source, language_family)
        """
        # Determine language family
        lang_family = LANGUAGE_TO_FAMILY.get(language.lower(), "Germanic")

        # Get language family prior
        if lang_family in LANGUAGE_FAMILY_PRIORS:
            alpha, beta = LANGUAGE_FAMILY_PRIORS[lang_family]
            return alpha, beta, "published_baseline", lang_family

        # Default prior (weak)
        return 5.0, 15.0, "default", "Unknown"

    def _hierarchical_borrow(
        self, language: str, category: str, all_observations: Dict
    ) -> Tuple[float, float]:
        """
        Borrow strength from language family.

        If current language has few observations, use empirical data
        from other languages in the same family to update prior.

        Args:
            language: Language code
            category: Harm category
            all_observations: Dict of all observations keyed by language

        Returns:
            Updated (alpha, beta) incorporating language family data
        """
        try:
            # Get language family
            lang_family = LANGUAGE_TO_FAMILY.get(language.lower(), "Germanic")

            # Collect observations from same family
            family_successes = 0
            family_failures = 0

            for lang, obs_list in all_observations.items():
                if LANGUAGE_TO_FAMILY.get(lang.lower(), "Germanic") == lang_family:
                    family_successes += sum(1 for o in obs_list if o.get("bypassed", False))
                    family_failures += len(obs_list) - family_successes

            # If family has observations, use empirical Bayes update
            if family_successes + family_failures > 5:
                empirical_rate = family_successes / (family_successes + family_failures)
                # Empirical Bayes: update prior toward family rate
                # Soft weighting
                alpha = 5 + empirical_rate * 10
                beta = 15 + (1 - empirical_rate) * 10
                return alpha, beta

            # Otherwise use family prior
            return LANGUAGE_FAMILY_PRIORS.get(lang_family, (5.0, 15.0))

        except Exception as e:
            logger.warning(f"Hierarchical borrowing failed: {e}")
            return 5.0, 15.0
