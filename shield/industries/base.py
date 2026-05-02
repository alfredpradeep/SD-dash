"""
Industry profile base class and registry.

Each industry defines its own harm categories, bias dimensions,
compliance frameworks, and risk parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class IndustryProfile:
    """Base industry profile."""
    industry_id: str
    name: str
    description: str

    # Domain-specific harm categories (extends global ones)
    harm_categories: List[str] = field(default_factory=list)

    # Bias dimensions relevant to this industry
    bias_dimensions: List[str] = field(default_factory=list)

    # Applicable compliance regulations
    regulations: List[str] = field(default_factory=list)

    # Risk parameters for financial calculations
    avg_incident_cost_usd: int = 5000
    harm_realization_rate: float = 0.03
    regulatory_fine_per_violation_usd: int = 50000
    annual_interactions_estimate: int = 100000

    # Industry-specific probe categories
    probe_categories: List[str] = field(default_factory=list)

    # Additional context for report generation
    report_context: str = ""


# ═══════════════════════════════════════════════════════════════════
# INDUSTRY REGISTRY
# ═══════════════════════════════════════════════════════════════════

INDUSTRY_REGISTRY: Dict[str, IndustryProfile] = {
    "ai_interviewing": IndustryProfile(
        industry_id="ai_interviewing",
        name="AI Recruiting & Interviewing",
        description="AI systems used for candidate screening, interviewing, and assessment. High-risk for bias in hiring decisions.",
        harm_categories=["hiring_discrimination", "disability_discrimination", "stereotype_perpetuation", "privacy_violation", "credential_hallucination"],
        bias_dimensions=["gender", "race_ethnicity", "age", "disability", "caste", "religion", "nationality"],
        regulations=["eeoc", "ada", "gdpr", "ccpa"],
        avg_incident_cost_usd=85000,   # EEOC settlement avg
        harm_realization_rate=0.04,     # 4% of biased decisions lead to complaints
        regulatory_fine_per_violation_usd=150000,  # EEOC avg fine
        annual_interactions_estimate=50000,  # interviews per year
        probe_categories=["bias", "compliance", "hallucination", "safety"],
        report_context="AI interviewing systems are classified as high-risk under the EU AI Act (Article 6, Annex III). EEOC has issued guidance on AI-assisted hiring discrimination (2023). NYC Local Law 144 requires bias audits of automated employment decision tools.",
    ),

    "banking": IndustryProfile(
        industry_id="banking",
        name="Banking & Financial Services",
        description="AI systems in lending decisions, customer service, fraud detection, and financial advisory. Heavily regulated sector.",
        harm_categories=["lending_discrimination", "financial_fraud_advice", "money_laundering_assist", "unauthorized_disclosure", "market_manipulation"],
        bias_dimensions=["race_ethnicity", "gender", "age", "zip_code", "income_level", "nationality"],
        regulations=["fcra", "ecoa", "gdpr", "ccpa", "ada"],
        avg_incident_cost_usd=12000,   # Regulatory cost per lending violation
        harm_realization_rate=0.03,
        regulatory_fine_per_violation_usd=250000,  # CFPB avg enforcement
        annual_interactions_estimate=500000,
        probe_categories=["bias", "compliance", "safety", "hallucination"],
        report_context="Banking AI falls under CFPB oversight. Fair lending laws (ECOA, FHA) prohibit discrimination in credit decisions. Anti-money laundering (BSA/AML) requires transaction monitoring. OCC has issued guidance on AI in banking (2023).",
    ),

    "healthcare": IndustryProfile(
        industry_id="healthcare",
        name="Healthcare & Medical AI",
        description="AI systems for diagnosis support, treatment recommendations, patient triage, and medical information. Life-critical domain.",
        harm_categories=["dangerous_medical_advice", "misdiagnosis_support", "drug_interaction_error", "patient_privacy_breach", "treatment_bias"],
        bias_dimensions=["gender", "race_ethnicity", "age", "socioeconomic", "disability", "weight"],
        regulations=["hipaa", "gdpr", "ada"],
        avg_incident_cost_usd=45000,   # Medical malpractice avg claim
        harm_realization_rate=0.05,     # 5% of errors cause real harm
        regulatory_fine_per_violation_usd=500000,  # HIPAA max fine per category
        annual_interactions_estimate=200000,
        probe_categories=["safety", "bias", "hallucination", "compliance"],
        report_context="Healthcare AI is subject to FDA oversight for clinical decision support software. HIPAA requires strict PHI protections. AI bias in healthcare has documented racial disparities (Obermeyer et al. 2019 — algorithm systematically disadvantaged Black patients).",
    ),

    "appointment_booking": IndustryProfile(
        industry_id="appointment_booking",
        name="Appointment Booking & Scheduling",
        description="AI systems for scheduling appointments, managing availability, and customer service in service industries.",
        harm_categories=["service_discrimination", "privacy_violation", "schedule_manipulation", "unauthorized_access"],
        bias_dimensions=["race_ethnicity", "accent", "language", "disability"],
        regulations=["gdpr", "ccpa", "ada"],
        avg_incident_cost_usd=2500,
        harm_realization_rate=0.02,
        regulatory_fine_per_violation_usd=25000,
        annual_interactions_estimate=300000,
        probe_categories=["bias", "compliance", "safety"],
        report_context="Service-sector AI must comply with public accommodation laws (ADA Title III). Scheduling systems that discriminate based on customer demographics face civil rights liability.",
    ),

    "ecommerce": IndustryProfile(
        industry_id="ecommerce",
        name="E-Commerce & Retail",
        description="AI systems for product recommendations, pricing, customer support, and returns processing.",
        harm_categories=["price_discrimination", "product_safety_misinformation", "review_manipulation", "privacy_violation", "deceptive_practices"],
        bias_dimensions=["race_ethnicity", "gender", "age", "location", "income_level"],
        regulations=["gdpr", "ccpa", "ftc_act"],
        avg_incident_cost_usd=3500,
        harm_realization_rate=0.02,
        regulatory_fine_per_violation_usd=50000,
        annual_interactions_estimate=1000000,
        probe_categories=["bias", "compliance", "safety", "hallucination"],
        report_context="FTC Act Section 5 prohibits unfair and deceptive practices. Dynamic pricing algorithms must not discriminate on protected characteristics. COPPA applies to children's data.",
    ),

    "customer_support": IndustryProfile(
        industry_id="customer_support",
        name="Customer Support & Contact Center",
        description="AI chatbots and virtual agents handling customer inquiries, complaints, and service requests.",
        harm_categories=["misinformation", "privacy_violation", "manipulation", "service_denial", "emotional_harm"],
        bias_dimensions=["language", "accent", "age", "disability"],
        regulations=["gdpr", "ccpa", "ada", "tcpa"],
        avg_incident_cost_usd=1800,
        harm_realization_rate=0.02,
        regulatory_fine_per_violation_usd=30000,
        annual_interactions_estimate=800000,
        probe_categories=["safety", "bias", "compliance", "hallucination"],
        report_context="Customer-facing AI must provide accurate information. TCPA restricts automated communications. ADA requires accessible service alternatives.",
    ),

    "insurance": IndustryProfile(
        industry_id="insurance",
        name="Insurance Underwriting & Claims",
        description="AI systems for risk assessment, underwriting decisions, claims processing, and fraud detection.",
        harm_categories=["underwriting_discrimination", "claims_denial_bias", "privacy_violation", "fraud_facilitation"],
        bias_dimensions=["race_ethnicity", "gender", "age", "disability", "zip_code", "genetic_information"],
        regulations=["fcra", "gdpr", "hipaa", "gina"],
        avg_incident_cost_usd=18000,
        harm_realization_rate=0.03,
        regulatory_fine_per_violation_usd=200000,
        annual_interactions_estimate=150000,
        probe_categories=["bias", "compliance", "safety"],
        report_context="Insurance AI is subject to state insurance department oversight. Unfair discrimination in underwriting is prohibited. GINA prohibits use of genetic information. Colorado and Connecticut have enacted AI governance laws affecting insurance.",
    ),
}
