"""Simulator mode definitions."""

from enum import Enum


class SimulatorMode(str, Enum):
    """Simulator operating modes."""
    DISABLED = "disabled"       # Use real endpoint (production)
    DEMO = "demo"               # Lightweight synthetic for demos
    BENCHMARK = "benchmark"     # Match known model behavior profiles
