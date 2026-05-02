"""
Shadow Bill Reconciliation Engine

Tracks every request that flows through LENS, independently counts tokens,
and at any point can generate a reconciliation report comparing LENS
measurements vs expected provider charges.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict
from collections import defaultdict
from io import StringIO
import uuid

from lens.moat.token_verifier import TokenVerificationEngine


@dataclass
class BilledRequest:
    """Single request with independent token verification."""
    request_id: str
    timestamp: datetime
    model: str
    input_text: str
    output_text: str = ""             # if available
    input_tokens_measured: int = 0    # LENS's independent count
    output_tokens_measured: int = 0
    input_tokens_billed: int = 0      # What provider says (if known, else 0)
    output_tokens_billed: int = 0
    cost_measured: float = 0.0         # What LENS calculates
    cost_billed: float = 0.0           # What provider charges (if known)
    discrepancy_tokens: int = 0        # billed - measured
    discrepancy_cost: float = 0.0
    team: str = "default"              # For cost attribution
    feature: str = "default"
    user_id: str = "anonymous"
    metadata: Dict = field(default_factory=dict)  # Flexible tags


@dataclass
class BillingReconciliation:
    """Complete reconciliation report for a time period."""
    period_start: datetime
    period_end: datetime
    total_requests: int
    total_tokens_measured: int
    total_tokens_billed: int
    total_cost_measured: float
    total_cost_billed: float
    discrepancy_tokens: int
    discrepancy_cost: float
    discrepancy_pct: float
    overcharge_requests: int      # Where billed > measured
    undercharge_requests: int
    top_discrepancies: List[BilledRequest] = field(default_factory=list)  # Worst offenders
    by_model: Dict = field(default_factory=dict)  # Breakdown per model
    by_team: Dict = field(default_factory=dict)   # Breakdown per team
    recommendations: List[str] = field(default_factory=list)


class BillingReconciler:
    """
    Tracks every request with independent token verification.
    Generates reconciliation reports comparing LENS measurements vs provider charges.
    """

    def __init__(self, verifier: Optional[TokenVerificationEngine] = None):
        """Initialize reconciler with optional token verifier."""
        self._verifier = verifier or TokenVerificationEngine()
        self._ledger: List[BilledRequest] = []

    def record(
        self,
        model: str,
        input_text: str,
        output_text: str = "",
        billed_input_tokens: int = 0,
        billed_output_tokens: int = 0,
        billed_cost: float = 0.0,
        team: str = "default",
        feature: str = "default",
        user_id: str = "anonymous",
        metadata: Optional[Dict] = None,
        request_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> BilledRequest:
        """
        Record a request with independent token verification.
        Uses TokenVerificationEngine to count input tokens.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())[:8]

        if timestamp is None:
            timestamp = datetime.utcnow()

        if metadata is None:
            metadata = {}

        # Independently verify input tokens
        try:
            input_verification = self._verifier.verify(input_text, model)
            input_tokens_measured = input_verification.measured_tokens
        except Exception as e:
            # Fallback: use character estimate
            input_tokens_measured = max(1, int(len(input_text) / 3.5))

        # Independently verify output tokens (if provided)
        output_tokens_measured = 0
        if output_text:
            try:
                output_verification = self._verifier.verify(output_text, model)
                output_tokens_measured = output_verification.measured_tokens
            except Exception as e:
                # Fallback: use character estimate
                output_tokens_measured = max(1, int(len(output_text) / 3.5))

        # Calculate measured cost
        pricing = self._verifier.PRICING
        price_per_1m_input = pricing.get(model, 1.0)
        measured_cost = (input_tokens_measured / 1_000_000) * price_per_1m_input

        # Calculate discrepancies
        total_billed_tokens = billed_input_tokens + billed_output_tokens
        total_measured_tokens = input_tokens_measured + output_tokens_measured
        discrepancy_tokens = total_billed_tokens - total_measured_tokens
        discrepancy_cost = billed_cost - measured_cost

        billed_request = BilledRequest(
            request_id=request_id,
            timestamp=timestamp,
            model=model,
            input_text=input_text,
            output_text=output_text,
            input_tokens_measured=input_tokens_measured,
            output_tokens_measured=output_tokens_measured,
            input_tokens_billed=billed_input_tokens,
            output_tokens_billed=billed_output_tokens,
            cost_measured=measured_cost,
            cost_billed=billed_cost,
            discrepancy_tokens=discrepancy_tokens,
            discrepancy_cost=discrepancy_cost,
            team=team,
            feature=feature,
            user_id=user_id,
            metadata=metadata,
        )

        self._ledger.append(billed_request)
        return billed_request

    def reconcile(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> BillingReconciliation:
        """
        Generate reconciliation report for a time period.
        Filters ledger by start/end timestamps, aggregates totals,
        identifies discrepancies, generates recommendations.
        """
        if not self._ledger:
            return BillingReconciliation(
                period_start=start or datetime.utcnow(),
                period_end=end or datetime.utcnow(),
                total_requests=0,
                total_tokens_measured=0,
                total_tokens_billed=0,
                total_cost_measured=0.0,
                total_cost_billed=0.0,
                discrepancy_tokens=0,
                discrepancy_cost=0.0,
                discrepancy_pct=0.0,
                overcharge_requests=0,
                undercharge_requests=0,
                top_discrepancies=[],
                by_model={},
                by_team={},
                recommendations=[]
            )

        # Filter by time range
        if start is None:
            start = min(req.timestamp for req in self._ledger)
        if end is None:
            end = max(req.timestamp for req in self._ledger)

        filtered_ledger = [
            req for req in self._ledger
            if start <= req.timestamp <= end
        ]

        if not filtered_ledger:
            return BillingReconciliation(
                period_start=start,
                period_end=end,
                total_requests=0,
                total_tokens_measured=0,
                total_tokens_billed=0,
                total_cost_measured=0.0,
                total_cost_billed=0.0,
                discrepancy_tokens=0,
                discrepancy_cost=0.0,
                discrepancy_pct=0.0,
                overcharge_requests=0,
                undercharge_requests=0,
                top_discrepancies=[],
                by_model={},
                by_team={},
                recommendations=[]
            )

        # Aggregate totals
        total_requests = len(filtered_ledger)
        total_tokens_measured = sum(
            req.input_tokens_measured + req.output_tokens_measured
            for req in filtered_ledger
        )
        total_tokens_billed = sum(
            req.input_tokens_billed + req.output_tokens_billed
            for req in filtered_ledger
        )
        total_cost_measured = sum(req.cost_measured for req in filtered_ledger)
        total_cost_billed = sum(req.cost_billed for req in filtered_ledger)

        discrepancy_tokens = total_tokens_billed - total_tokens_measured
        discrepancy_cost = total_cost_billed - total_cost_measured

        discrepancy_pct = (
            (abs(discrepancy_cost) / total_cost_measured * 100)
            if total_cost_measured > 0 else 0.0
        )

        # Count overcharge/undercharge requests
        overcharge_requests = sum(
            1 for req in filtered_ledger
            if req.discrepancy_cost > 0.01  # Threshold to avoid floating-point noise
        )
        undercharge_requests = sum(
            1 for req in filtered_ledger
            if req.discrepancy_cost < -0.01
        )

        # Top discrepancies (by cost)
        top_discrepancies = sorted(
            filtered_ledger,
            key=lambda r: abs(r.discrepancy_cost),
            reverse=True
        )[:10]

        # Breakdown by model
        by_model = self._breakdown_by_model(filtered_ledger)

        # Breakdown by team
        by_team = self._breakdown_by_team(filtered_ledger)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            total_requests,
            discrepancy_cost,
            discrepancy_pct,
            overcharge_requests,
            by_model
        )

        return BillingReconciliation(
            period_start=start,
            period_end=end,
            total_requests=total_requests,
            total_tokens_measured=total_tokens_measured,
            total_tokens_billed=total_tokens_billed,
            total_cost_measured=total_cost_measured,
            total_cost_billed=total_cost_billed,
            discrepancy_tokens=discrepancy_tokens,
            discrepancy_cost=discrepancy_cost,
            discrepancy_pct=discrepancy_pct,
            overcharge_requests=overcharge_requests,
            undercharge_requests=undercharge_requests,
            top_discrepancies=top_discrepancies,
            by_model=by_model,
            by_team=by_team,
            recommendations=recommendations,
        )

    def _breakdown_by_model(self, ledger: List[BilledRequest]) -> Dict:
        """Break down costs and discrepancies by model."""
        breakdown = defaultdict(lambda: {
            "requests": 0,
            "tokens_measured": 0,
            "tokens_billed": 0,
            "cost_measured": 0.0,
            "cost_billed": 0.0,
            "discrepancy_cost": 0.0,
        })

        for req in ledger:
            model_key = req.model
            breakdown[model_key]["requests"] += 1
            breakdown[model_key]["tokens_measured"] += (
                req.input_tokens_measured + req.output_tokens_measured
            )
            breakdown[model_key]["tokens_billed"] += (
                req.input_tokens_billed + req.output_tokens_billed
            )
            breakdown[model_key]["cost_measured"] += req.cost_measured
            breakdown[model_key]["cost_billed"] += req.cost_billed
            breakdown[model_key]["discrepancy_cost"] += req.discrepancy_cost

        return dict(breakdown)

    def _breakdown_by_team(self, ledger: List[BilledRequest]) -> Dict:
        """Break down costs and discrepancies by team."""
        breakdown = defaultdict(lambda: {
            "requests": 0,
            "cost_measured": 0.0,
            "cost_billed": 0.0,
            "discrepancy_cost": 0.0,
            "tokens_measured": 0,
        })

        for req in ledger:
            team_key = req.team
            breakdown[team_key]["requests"] += 1
            breakdown[team_key]["cost_measured"] += req.cost_measured
            breakdown[team_key]["cost_billed"] += req.cost_billed
            breakdown[team_key]["discrepancy_cost"] += req.discrepancy_cost
            breakdown[team_key]["tokens_measured"] += (
                req.input_tokens_measured + req.output_tokens_measured
            )

        return dict(breakdown)

    def _generate_recommendations(
        self,
        total_requests: int,
        discrepancy_cost: float,
        discrepancy_pct: float,
        overcharge_requests: int,
        by_model: Dict,
    ) -> List[str]:
        """Generate actionable recommendations based on reconciliation data."""
        recommendations = []

        if abs(discrepancy_cost) < 0.01:
            recommendations.append("Reconciliation excellent: measured costs match billed costs.")
            return recommendations

        if discrepancy_cost > 0.01:
            recommendations.append(
                f"OVERCHARGE DETECTED: Provider billed ${abs(discrepancy_cost):.2f} "
                f"more than measured ({discrepancy_pct:.2f}%). "
                f"{overcharge_requests} requests show overcharges. "
                f"Request refund and audit with provider."
            )
        else:
            recommendations.append(
                f"UNDERCHARGE FOUND: Provider billed ${abs(discrepancy_cost):.2f} "
                f"less than measured ({abs(discrepancy_pct):.2f}%). "
                f"Verify accuracy of billing integration."
            )

        # Per-model recommendations
        most_expensive = max(
            by_model.items(),
            key=lambda item: item[1]["cost_measured"],
            default=None
        )
        if most_expensive:
            model, stats = most_expensive
            pct_total = (stats["cost_measured"] / sum(m["cost_measured"] for m in by_model.values()) * 100)
            if pct_total > 40:
                recommendations.append(
                    f"{model} represents {pct_total:.1f}% of costs ({stats['requests']} requests). "
                    f"Consider cross-provider comparison."
                )

        if discrepancy_pct > 5:
            recommendations.append(
                "Large discrepancy percentage suggests systematic billing issue. "
                "Investigate token counting or pricing integration."
            )

        return recommendations

    def get_by_team(self, team: str) -> List[BilledRequest]:
        """Get all requests for a specific team."""
        return [req for req in self._ledger if req.team == team]

    def get_by_feature(self, feature: str) -> List[BilledRequest]:
        """Get all requests for a specific feature."""
        return [req for req in self._ledger if req.feature == feature]

    def get_by_model(self, model: str) -> List[BilledRequest]:
        """Get all requests for a specific model."""
        return [req for req in self._ledger if req.model == model]

    def get_by_user(self, user_id: str) -> List[BilledRequest]:
        """Get all requests from a specific user."""
        return [req for req in self._ledger if req.user_id == user_id]

    def export_csv(self) -> str:
        """Export ledger as CSV for accounting and audit."""
        if not self._ledger:
            return "request_id,timestamp,model,input_tokens_measured,output_tokens_measured," \
                   "input_tokens_billed,output_tokens_billed,cost_measured,cost_billed," \
                   "discrepancy_tokens,discrepancy_cost,team,feature,user_id\n"

        output = StringIO()

        # Header
        headers = [
            "request_id",
            "timestamp",
            "model",
            "input_tokens_measured",
            "output_tokens_measured",
            "input_tokens_billed",
            "output_tokens_billed",
            "cost_measured",
            "cost_billed",
            "discrepancy_tokens",
            "discrepancy_cost",
            "team",
            "feature",
            "user_id",
        ]
        output.write(",".join(headers) + "\n")

        # Rows
        for req in self._ledger:
            row = [
                req.request_id,
                req.timestamp.isoformat(),
                req.model,
                str(req.input_tokens_measured),
                str(req.output_tokens_measured),
                str(req.input_tokens_billed),
                str(req.output_tokens_billed),
                f"{req.cost_measured:.6f}",
                f"{req.cost_billed:.6f}",
                str(req.discrepancy_tokens),
                f"{req.discrepancy_cost:.6f}",
                req.team,
                req.feature,
                req.user_id,
            ]
            output.write(",".join(row) + "\n")

        return output.getvalue()

    def get_ledger_size(self) -> int:
        """Get total number of requests in ledger."""
        return len(self._ledger)

    def clear_ledger(self) -> int:
        """Clear all records and return count of cleared items."""
        count = len(self._ledger)
        self._ledger = []
        return count


def main_demo():
    """Demo the billing reconciliation engine."""
    print("=" * 80)
    print("BILLING RECONCILIATION ENGINE DEMO")
    print("=" * 80)

    # Create engine
    verifier = TokenVerificationEngine()
    reconciler = BillingReconciler(verifier)

    # Simulate some requests
    print("\nRecording sample requests...")
    print("-" * 80)

    sample_inputs = [
        "What is machine learning?" * 5,
        "Explain quantum computing in simple terms." * 10,
        "Write a Python function for fibonacci sequence." * 3,
    ]

    sample_outputs = [
        "Machine learning is a subset of artificial intelligence..." * 3,
        "Quantum computing uses quantum mechanical phenomena..." * 5,
        "Here is a fibonacci function: def fib(n): ..." * 2,
    ]

    models = ["gpt-4o", "claude-3-5-sonnet", "gpt-4o-mini"]
    teams = ["data-science", "research", "engineering"]
    features = ["chat", "analysis", "code-generation"]

    for i in range(12):
        model = models[i % len(models)]
        team = teams[i % len(teams)]
        feature = features[i % len(features)]
        input_text = sample_inputs[i % len(sample_inputs)]
        output_text = sample_outputs[i % len(sample_outputs)]

        # Simulate some billing discrepancies
        billed_input = int(len(input_text) / 3.2) if i % 3 == 0 else 0  # Only some have billing
        billed_output = int(len(output_text) / 3.2) if i % 3 == 0 else 0
        billed_cost = (billed_input + billed_output) * 0.000001 if i % 3 == 0 else 0

        req = reconciler.record(
            model=model,
            input_text=input_text,
            output_text=output_text,
            billed_input_tokens=billed_input,
            billed_output_tokens=billed_output,
            billed_cost=billed_cost,
            team=team,
            feature=feature,
            user_id=f"user_{i % 4}",
            metadata={"experiment": "demo"}
        )
        print(f"  [{i+1}] {req.request_id}: {model:20} | {team:15} | "
              f"Measured: {req.input_tokens_measured:4} tokens, "
              f"${req.cost_measured:.6f}")

    print("\n" + "-" * 80)
    print("Generating reconciliation report...")
    print("-" * 80)

    # Generate reconciliation
    report = reconciler.reconcile()

    print(f"\nPeriod: {report.period_start.date()} to {report.period_end.date()}")
    print(f"Total Requests: {report.total_requests}")
    print(f"Total Tokens (Measured): {report.total_tokens_measured:,}")
    print(f"Total Tokens (Billed): {report.total_tokens_billed:,}")
    print(f"Total Cost (Measured): ${report.total_cost_measured:.6f}")
    print(f"Total Cost (Billed): ${report.total_cost_billed:.6f}")
    print(f"Discrepancy: ${report.discrepancy_cost:.6f} ({report.discrepancy_pct:.2f}%)")
    print(f"Overcharge Requests: {report.overcharge_requests}")
    print(f"Undercharge Requests: {report.undercharge_requests}")

    print("\n" + "-" * 80)
    print("Breakdown by Model:")
    print("-" * 80)
    for model, stats in sorted(report.by_model.items(), key=lambda x: x[1]["cost_measured"], reverse=True):
        print(f"  {model:20} | Requests: {stats['requests']:3} | "
              f"Measured: ${stats['cost_measured']:.6f} | "
              f"Billed: ${stats['cost_billed']:.6f} | "
              f"Discrepancy: ${stats['discrepancy_cost']:.6f}")

    print("\n" + "-" * 80)
    print("Breakdown by Team:")
    print("-" * 80)
    for team, stats in sorted(report.by_team.items(), key=lambda x: x[1]["cost_measured"], reverse=True):
        print(f"  {team:20} | Requests: {stats['requests']:3} | "
              f"Measured: ${stats['cost_measured']:.6f} | "
              f"Discrepancy: ${stats['discrepancy_cost']:.6f}")

    if report.top_discrepancies:
        print("\n" + "-" * 80)
        print("Top 3 Discrepancies:")
        print("-" * 80)
        for i, req in enumerate(report.top_discrepancies[:3], 1):
            print(f"  {i}. {req.request_id}: {req.model} | "
                  f"Discrepancy: ${req.discrepancy_cost:.6f}")

    if report.recommendations:
        print("\n" + "-" * 80)
        print("Recommendations:")
        print("-" * 80)
        for rec in report.recommendations:
            print(f"  • {rec}")

    # Export CSV sample
    print("\n" + "-" * 80)
    print("CSV Export (first 3 rows):")
    print("-" * 80)
    csv_data = reconciler.export_csv()
    csv_lines = csv_data.split("\n")
    for line in csv_lines[:4]:  # Header + first 3 rows
        if line:
            print(f"  {line[:100]}...")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main_demo()
