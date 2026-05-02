"""
Per-Feature / Per-Team / Per-User Cost Attribution

Tags every request and provides cost breakdowns by any dimension.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional
import statistics


@dataclass
class AttributionTag:
    """Tags for attributing a cost to business dimensions."""
    team: str = "default"
    feature: str = "default"
    user_id: str = "anonymous"
    environment: str = "production"  # production, staging, development
    session_id: str = ""
    custom_tags: dict = field(default_factory=dict)


@dataclass
class AttributionEntry:
    """A single cost-attributed request."""
    request_id: str
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    tags: AttributionTag
    language: str = "en"


@dataclass
class CostBreakdown:
    """Cost breakdown for a specific dimension."""
    dimension: str            # "team", "feature", "user_id", "model", "language"
    period_start: datetime
    period_end: datetime
    total_cost: float
    total_requests: int
    total_tokens: int
    breakdown: list[dict] = field(default_factory=list)  # [{name, cost, requests, tokens, pct_of_total, avg_cost_per_request}]
    top_spender: str = ""
    top_spender_pct: float = 0.0


class CostAttributionEngine:
    """
    Attributes AI costs to teams, features, users, and custom dimensions.

    For Eightfold AI Interviewer:
    - Team: "recruiting-us", "recruiting-india", "engineering"
    - Feature: "screening", "technical-interview", "offer-letter"
    - User: candidate ID or recruiter ID
    - Session: interview session ID

    Enables: "The India recruiting team spent $12,400 on Tamil interviews
    this month. Technical interviews cost 3x more than screening."
    """

    def __init__(self):
        self._entries: list[AttributionEntry] = []
        self._budgets: dict[str, float] = {}  # team -> monthly budget

    def record(self, request_id: str, model: str, input_tokens: int, output_tokens: int,
               cost_usd: float, tags: AttributionTag, language: str = "en") -> AttributionEntry:
        """Record a cost-attributed request."""

        entry = AttributionEntry(
            request_id=request_id,
            timestamp=datetime.utcnow(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            tags=tags,
            language=language
        )

        self._entries.append(entry)

        return entry

    def set_budget(self, team: str, monthly_budget: float):
        """Set monthly budget for a team."""

        self._budgets[team] = monthly_budget

    def get_breakdown(self, dimension: str, start: datetime = None, end: datetime = None) -> CostBreakdown:
        """
        Get cost breakdown by dimension.

        Args:
            dimension: "team", "feature", "user_id", "model", "language", or "environment"
            start: Start time (default: 30 days ago)
            end: End time (default: now)

        Returns:
            CostBreakdown object
        """

        if end is None:
            end = datetime.utcnow()
        if start is None:
            start = end - timedelta(days=30)

        # Filter entries by time window
        entries = [e for e in self._entries if start <= e.timestamp <= end]

        if not entries:
            return CostBreakdown(
                dimension=dimension,
                period_start=start,
                period_end=end,
                total_cost=0.0,
                total_requests=0,
                total_tokens=0,
                breakdown=[],
                top_spender="",
                top_spender_pct=0.0
            )

        # Group by dimension
        groups: dict[str, list[AttributionEntry]] = defaultdict(list)

        for entry in entries:
            if dimension == "team":
                key = entry.tags.team
            elif dimension == "feature":
                key = entry.tags.feature
            elif dimension == "user_id":
                key = entry.tags.user_id
            elif dimension == "model":
                key = entry.model
            elif dimension == "language":
                key = entry.language
            elif dimension == "environment":
                key = entry.tags.environment
            else:
                key = "unknown"

            groups[key].append(entry)

        # Compute breakdown
        total_cost = sum(e.cost_usd for e in entries)
        total_requests = len(entries)
        total_tokens = sum(e.input_tokens + e.output_tokens for e in entries)

        breakdown = []
        top_spender = ""
        top_spender_cost = 0.0

        for name, group_entries in sorted(groups.items()):
            group_cost = sum(e.cost_usd for e in group_entries)
            group_requests = len(group_entries)
            group_tokens = sum(e.input_tokens + e.output_tokens for e in group_entries)
            avg_cost_per_request = group_cost / group_requests if group_requests > 0 else 0
            pct_of_total = (group_cost / total_cost * 100) if total_cost > 0 else 0

            breakdown.append({
                "name": name,
                "cost": round(group_cost, 2),
                "requests": group_requests,
                "tokens": group_tokens,
                "pct_of_total": round(pct_of_total, 2),
                "avg_cost_per_request": round(avg_cost_per_request, 4)
            })

            if group_cost > top_spender_cost:
                top_spender = name
                top_spender_cost = group_cost

        # Sort by cost descending
        breakdown = sorted(breakdown, key=lambda x: x["cost"], reverse=True)

        top_spender_pct = (top_spender_cost / total_cost * 100) if total_cost > 0 else 0

        return CostBreakdown(
            dimension=dimension,
            period_start=start,
            period_end=end,
            total_cost=round(total_cost, 2),
            total_requests=total_requests,
            total_tokens=total_tokens,
            breakdown=breakdown,
            top_spender=top_spender,
            top_spender_pct=round(top_spender_pct, 2)
        )

    def get_team_budget_status(self, team: str) -> dict:
        """Check team's budget utilization."""

        if team not in self._budgets:
            return {
                "team": team,
                "budget": None,
                "spent": 0,
                "remaining": None,
                "pct_used": None,
                "projected_monthly": 0,
                "on_track": None,
                "message": "No budget set for this team"
            }

        budget = self._budgets[team]

        # Get current month's spending
        now = datetime.utcnow()
        month_start = datetime(now.year, now.month, 1)
        month_end = month_start + timedelta(days=31)

        # Handle month rollover
        if month_end.month == 1:
            month_end = month_end.replace(day=1)

        spent = sum(e.cost_usd for e in self._entries
                   if month_start <= e.timestamp <= month_end and e.tags.team == team)

        # Project to end of month
        days_elapsed = (now - month_start).days + 1
        days_in_month = 30  # Approximate
        projected_monthly = spent * (days_in_month / days_elapsed) if days_elapsed > 0 else spent

        remaining = budget - spent
        pct_used = (spent / budget * 100) if budget > 0 else 0
        on_track = projected_monthly <= budget

        return {
            "team": team,
            "budget": budget,
            "spent": round(spent, 2),
            "remaining": round(remaining, 2),
            "pct_used": round(pct_used, 1),
            "projected_monthly": round(projected_monthly, 2),
            "on_track": on_track,
            "message": f"Team has spent ${spent:.2f} of ${budget:.2f} budget ({pct_used:.1f}%). Projected monthly: ${projected_monthly:.2f}."
        }

    def get_top_consumers(self, n: int = 10, dimension: str = "user_id") -> list[dict]:
        """Get top N cost consumers by dimension."""

        breakdown = self.get_breakdown(dimension)

        return breakdown.breakdown[:n]

    def get_cost_trend(self, dimension: str, value: str, days: int = 30) -> list[dict]:
        """Daily cost trend for a specific team/feature/user."""

        # Filter entries matching dimension and value
        filtered = []
        for entry in self._entries:
            if dimension == "team" and entry.tags.team != value:
                continue
            elif dimension == "feature" and entry.tags.feature != value:
                continue
            elif dimension == "user_id" and entry.tags.user_id != value:
                continue
            elif dimension == "model" and entry.model != value:
                continue
            elif dimension == "language" and entry.language != value:
                continue
            elif dimension == "environment" and entry.tags.environment != value:
                continue

            filtered.append(entry)

        # Group by day
        cutoff = datetime.utcnow() - timedelta(days=days)
        filtered = [e for e in filtered if e.timestamp >= cutoff]

        daily_costs: dict[str, float] = defaultdict(float)
        daily_requests: dict[str, int] = defaultdict(int)
        daily_tokens: dict[str, int] = defaultdict(int)

        for entry in filtered:
            day = entry.timestamp.date().isoformat()
            daily_costs[day] += entry.cost_usd
            daily_requests[day] += 1
            daily_tokens[day] += entry.input_tokens + entry.output_tokens

        # Generate all days in range (fill gaps)
        trend = []
        current = datetime.utcnow().date()
        for i in range(days):
            day = (current - timedelta(days=i)).isoformat()
            trend.append({
                "date": day,
                "cost": round(daily_costs.get(day, 0), 2),
                "requests": daily_requests.get(day, 0),
                "tokens": daily_tokens.get(day, 0),
                "avg_cost_per_request": round(daily_costs.get(day, 0) / daily_requests.get(day, 1), 4)
            })

        return sorted(trend, key=lambda x: x["date"])

    def export_summary(self) -> dict:
        """Export summary statistics."""

        if not self._entries:
            return {
                "total_requests": 0,
                "total_cost": 0,
                "total_tokens": 0,
                "avg_cost_per_request": 0,
                "avg_tokens_per_request": 0,
                "unique_teams": 0,
                "unique_features": 0,
                "unique_users": 0
            }

        total_cost = sum(e.cost_usd for e in self._entries)
        total_requests = len(self._entries)
        total_tokens = sum(e.input_tokens + e.output_tokens for e in self._entries)
        avg_cost_per_request = total_cost / total_requests if total_requests > 0 else 0
        avg_tokens_per_request = total_tokens / total_requests if total_requests > 0 else 0

        unique_teams = len(set(e.tags.team for e in self._entries))
        unique_features = len(set(e.tags.feature for e in self._entries))
        unique_users = len(set(e.tags.user_id for e in self._entries))

        return {
            "total_requests": total_requests,
            "total_cost": round(total_cost, 2),
            "total_tokens": total_tokens,
            "avg_cost_per_request": round(avg_cost_per_request, 4),
            "avg_tokens_per_request": round(avg_tokens_per_request, 1),
            "unique_teams": unique_teams,
            "unique_features": unique_features,
            "unique_users": unique_users
        }
