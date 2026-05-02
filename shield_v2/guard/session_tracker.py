"""
Session-level trajectory risk tracking.

The fourth dimension of bidirectional risk: Session Trajectory Risk.
A single turn may be safe in isolation but the accumulated conversation
context may drift the AI into unsafe behavior on turn N.

We track an exponentially-weighted moving average of per-turn risk
scores, flag sessions whose cumulative risk crosses thresholds, and
emit real-time signals that the Guard uses to decide whether to
preemptively block the next turn.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass
class TurnRisk:
    turn_id: int
    timestamp: float
    inbound_risk: float
    outbound_risk: float
    cross_channel_risk: float
    verdict: str


@dataclass
class SessionRisk:
    session_id: str
    inbound_ewma: float = 0.0
    outbound_ewma: float = 0.0
    cross_channel_ewma: float = 0.0
    trajectory_risk: float = 0.0
    turn_count: int = 0
    jailbreak_attempts: int = 0
    injection_attempts: int = 0
    cultural_landmines: int = 0
    turns: List[TurnRisk] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "inbound_ewma": round(self.inbound_ewma, 3),
            "outbound_ewma": round(self.outbound_ewma, 3),
            "cross_channel_ewma": round(self.cross_channel_ewma, 3),
            "trajectory_risk": round(self.trajectory_risk, 3),
            "turn_count": self.turn_count,
            "jailbreak_attempts": self.jailbreak_attempts,
            "injection_attempts": self.injection_attempts,
            "cultural_landmines": self.cultural_landmines,
        }


class SessionTracker:
    """
    Maintains bidirectional risk state across turns for each session.
    EWMA decay α = 0.35: recent turns weigh more than old turns.
    """

    def __init__(self, alpha: float = 0.35, traj_threshold: float = 0.6):
        self.alpha = alpha
        self.traj_threshold = traj_threshold
        self._sessions: Dict[str, SessionRisk] = {}

    def get(self, session_id: str) -> SessionRisk:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionRisk(session_id=session_id)
        return self._sessions[session_id]

    def record_turn(
        self,
        session_id: str,
        inbound_risk: float,
        outbound_risk: float,
        cross_channel_risk: float,
        verdict: str,
        is_jailbreak: bool = False,
        is_injection: bool = False,
        is_cultural_landmine: bool = False,
    ) -> SessionRisk:
        sr = self.get(session_id)
        sr.turn_count += 1
        sr.inbound_ewma = self.alpha * inbound_risk + (1 - self.alpha) * sr.inbound_ewma
        sr.outbound_ewma = self.alpha * outbound_risk + (1 - self.alpha) * sr.outbound_ewma
        sr.cross_channel_ewma = self.alpha * cross_channel_risk + (1 - self.alpha) * sr.cross_channel_ewma
        # Trajectory = weighted max of the three streams + escalation bonus
        sr.trajectory_risk = max(
            sr.inbound_ewma, sr.outbound_ewma, sr.cross_channel_ewma
        ) * (1.0 + 0.05 * (sr.jailbreak_attempts + sr.injection_attempts))
        sr.trajectory_risk = min(1.0, sr.trajectory_risk)

        if is_jailbreak:
            sr.jailbreak_attempts += 1
        if is_injection:
            sr.injection_attempts += 1
        if is_cultural_landmine:
            sr.cultural_landmines += 1

        sr.turns.append(TurnRisk(
            turn_id=sr.turn_count, timestamp=time.time(),
            inbound_risk=inbound_risk, outbound_risk=outbound_risk,
            cross_channel_risk=cross_channel_risk, verdict=verdict,
        ))
        # Keep last 100 turns
        if len(sr.turns) > 100:
            sr.turns = sr.turns[-100:]
        return sr

    def is_session_hot(self, session_id: str) -> bool:
        return self.get(session_id).trajectory_risk >= self.traj_threshold

    def reset(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]

    def snapshot_all(self) -> List[Dict]:
        return [sr.as_dict() for sr in self._sessions.values()]
