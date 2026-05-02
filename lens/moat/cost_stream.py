import time
import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional, Callable


@dataclass
class CostEvent:
    """Single cost event in the stream."""
    request_id: str
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    language: str
    team: str
    feature: str
    cumulative_cost_today: float  # Running total for the day
    velocity_per_minute: float   # Current spend rate


@dataclass
class CostAlert:
    """Alert when spending exceeds thresholds."""
    alert_type: str  # "velocity_spike", "daily_budget_warning", "daily_budget_exceeded", "unusual_pattern"
    severity: str    # "info", "warning", "critical"
    message: str
    current_value: float
    threshold: float
    timestamp: datetime


class CostStream:
    """
    Real-time cost accumulator with alerting.

    Tracks:
    - Cumulative cost per day/hour/minute
    - Spend velocity ($/min rolling average)
    - Budget thresholds with alerts
    - Per-team and per-feature breakdowns
    - Anomaly detection on spend patterns
    """

    def __init__(self, daily_budget: float = 1000.0, alert_callback: Optional[Callable] = None):
        self._events: list[CostEvent] = []
        self._daily_budget = daily_budget
        self._alert_callback = alert_callback
        self._alerts: list[CostAlert] = []
        self._subscribers: list[asyncio.Queue] = []  # For WebSocket streaming
        self._cumulative_today = 0.0
        self._velocity_window: list[tuple[float, float]] = []  # (timestamp, cost)
        self._by_team: dict[str, float] = defaultdict(float)
        self._by_feature: dict[str, float] = defaultdict(float)
        self._by_model: dict[str, float] = defaultdict(float)
        self._day_started: Optional[datetime] = None
        self._velocity_history_30m: list[tuple[float, float]] = []  # For 30-min avg
        self._thresholds_hit: dict[str, bool] = {
            "50%": False,
            "80%": False,
            "100%": False,
        }

    def record(
        self,
        request_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        language: str = "en",
        team: str = "default",
        feature: str = "default",
    ) -> CostEvent:
        """Record a cost event and check thresholds."""
        now = datetime.now()

        # Initialize day if needed
        if self._day_started is None:
            self._day_started = now
            self._cumulative_today = 0.0
            self._thresholds_hit = {"50%": False, "80%": False, "100%": False}
        elif now.date() != self._day_started.date():
            # New day, reset
            self._day_started = now
            self._cumulative_today = 0.0
            self._thresholds_hit = {"50%": False, "80%": False, "100%": False}
            self._events = []
            self._by_team = defaultdict(float)
            self._by_feature = defaultdict(float)
            self._by_model = defaultdict(float)
            self._velocity_window = []

        # Update cumulative
        self._cumulative_today += cost_usd

        # Update velocity window (5-minute rolling)
        now_timestamp = now.timestamp()
        self._velocity_window.append((now_timestamp, cost_usd))
        self._velocity_window = [(ts, cost) for ts, cost in self._velocity_window
                                if now_timestamp - ts <= 300]  # 5 minutes

        # Update 30-minute history for baseline
        self._velocity_history_30m.append((now_timestamp, cost_usd))
        self._velocity_history_30m = [(ts, cost) for ts, cost in self._velocity_history_30m
                                      if now_timestamp - ts <= 1800]  # 30 minutes

        # Update breakdowns
        self._by_team[team] += cost_usd
        self._by_feature[feature] += cost_usd
        self._by_model[model] += cost_usd

        # Calculate current velocity
        velocity = self.get_velocity(window_minutes=5)

        # Create event
        event = CostEvent(
            request_id=request_id,
            timestamp=now,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            language=language,
            team=team,
            feature=feature,
            cumulative_cost_today=self._cumulative_today,
            velocity_per_minute=velocity,
        )

        self._events.append(event)

        # Check alerts
        self._check_alerts(event)

        # Notify subscribers (only if event loop is running)
        try:
            asyncio.create_task(self._notify_subscribers(event))
        except RuntimeError:
            # No event loop running; skip async notification
            pass

        return event

    def subscribe(self) -> asyncio.Queue:
        """Subscribe to live cost events (for WebSocket)."""
        queue = asyncio.Queue()
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue):
        """Unsubscribe from events."""
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def get_velocity(self, window_minutes: int = 5) -> float:
        """Current spend velocity in $/minute."""
        if not self._velocity_window:
            return 0.0

        total_cost = sum(cost for _, cost in self._velocity_window)
        window_seconds = window_minutes * 60
        velocity = total_cost / (window_seconds / 60.0) if self._velocity_window else 0.0
        return velocity

    def get_summary(self) -> dict:
        """Current spending summary."""
        velocity = self.get_velocity(window_minutes=5)
        remaining = max(0.0, self._daily_budget - self._cumulative_today)
        budget_pct = (self._cumulative_today / self._daily_budget * 100) if self._daily_budget > 0 else 0

        # Project daily total (if velocity holds constant)
        hours_left = (23 - datetime.now().hour)
        minutes_left = hours_left * 60 + (60 - datetime.now().minute)
        if minutes_left > 0:
            projected = self._cumulative_today + (velocity * minutes_left)
        else:
            projected = self._cumulative_today

        return {
            "today_total": round(self._cumulative_today, 2),
            "velocity_per_minute": round(velocity, 4),
            "budget_remaining": round(remaining, 2),
            "budget_percent": round(budget_pct, 1),
            "projected_daily_total": round(projected, 2),
            "by_team": dict(self._by_team),
            "by_feature": dict(self._by_feature),
            "by_model": dict(self._by_model),
            "alerts": [
                {
                    "alert_type": a.alert_type,
                    "severity": a.severity,
                    "message": a.message,
                    "timestamp": a.timestamp.isoformat(),
                }
                for a in self._alerts[-10:]  # Last 10 alerts
            ],
        }

    def get_hourly_breakdown(self) -> list[dict]:
        """Cost per hour for today."""
        hourly: dict[int, float] = defaultdict(float)

        for event in self._events:
            hour = event.timestamp.hour
            hourly[hour] += event.cost_usd

        result = []
        for hour in range(24):
            result.append(
                {
                    "hour": hour,
                    "cost": round(hourly.get(hour, 0.0), 2),
                }
            )

        return result

    def _check_alerts(self, event: CostEvent):
        """Check and generate alerts."""
        now = datetime.now()

        # Budget threshold alerts (50%, 80%, 100%)
        thresholds = [
            (0.50, "50%", "info", "Spent 50% of daily budget"),
            (0.80, "80%", "warning", "Spent 80% of daily budget"),
            (1.00, "100%", "critical", "Daily budget exceeded"),
        ]

        for threshold_pct, threshold_key, severity, msg in thresholds:
            threshold_amount = self._daily_budget * threshold_pct
            if (
                self._cumulative_today >= threshold_amount
                and not self._thresholds_hit[threshold_key]
            ):
                self._thresholds_hit[threshold_key] = True
                alert = CostAlert(
                    alert_type="daily_budget_warning",
                    severity=severity,
                    message=msg,
                    current_value=self._cumulative_today,
                    threshold=threshold_amount,
                    timestamp=now,
                )
                self._alerts.append(alert)
                if self._alert_callback:
                    self._alert_callback(alert)

        # Velocity spike alert (>2x 30-minute average)
        if len(self._velocity_history_30m) > 0:
            avg_30m = sum(cost for _, cost in self._velocity_history_30m) / len(self._velocity_history_30m)
            current_velocity = self.get_velocity(window_minutes=5)

            # Spike is when current > 2x average (but only if average > 0)
            if avg_30m > 0 and current_velocity > 2.0 * avg_30m:
                alert = CostAlert(
                    alert_type="velocity_spike",
                    severity="warning",
                    message=f"Spending spike detected: ${current_velocity:.2f}/min (avg: ${avg_30m:.2f}/min)",
                    current_value=current_velocity,
                    threshold=2.0 * avg_30m,
                    timestamp=now,
                )
                self._alerts.append(alert)
                if self._alert_callback:
                    self._alert_callback(alert)

    async def _notify_subscribers(self, event: CostEvent):
        """Notify all subscribers of a new event."""
        # Remove closed queues
        dead_queues = []
        for queue in self._subscribers:
            try:
                queue.put_nowait(event)
            except (asyncio.QueueFull, RuntimeError):
                dead_queues.append(queue)

        for queue in dead_queues:
            if queue in self._subscribers:
                self._subscribers.remove(queue)
