"""Canonical GridWise constants and shared dataclasses.

Everything in this module is derived directly from the Problem Statement
(Sections 04, 07, 09, 10) and is the single source of truth for the rest of
the service.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

HORIZON = 24
HOURS = tuple(range(HORIZON))

# Section 04.1 -- the complete, closed set of directive types.
DIRECTIVE_SOLAR_REDUCTION = "solar_reduction"
DIRECTIVE_MIN_RESERVE = "minimum_battery_reserve"
DIRECTIVE_NO_CHARGE = "no_charge_window"
DIRECTIVE_NO_DISCHARGE = "no_discharge_window"
DIRECTIVE_MAX_GRID = "max_grid_window"
DIRECTIVE_NO_OP = "no_op"

DIRECTIVE_TYPES = (
    DIRECTIVE_SOLAR_REDUCTION,
    DIRECTIVE_MIN_RESERVE,
    DIRECTIVE_NO_CHARGE,
    DIRECTIVE_NO_DISCHARGE,
    DIRECTIVE_MAX_GRID,
    DIRECTIVE_NO_OP,
)

# Required structured_adjustment payload keys per directive type.
REQUIRED_ADJUSTMENT_KEYS = {
    DIRECTIVE_SOLAR_REDUCTION: ("hours", "factor"),
    DIRECTIVE_MIN_RESERVE: ("hours", "minimum_energy_kwh"),
    DIRECTIVE_NO_CHARGE: ("hours",),
    DIRECTIVE_NO_DISCHARGE: ("hours",),
    DIRECTIVE_MAX_GRID: ("hours", "max_grid_kwh"),
    DIRECTIVE_NO_OP: (),
}

ACTION_CHARGE = "charge"
ACTION_DISCHARGE = "discharge"
ACTION_IDLE = "idle"
BATTERY_ACTIONS = (ACTION_CHARGE, ACTION_DISCHARGE, ACTION_IDLE)

# Section 11.5 -- judge tolerance. We keep our own slack an order of
# magnitude tighter so rounding never pushes us over the judge's line.
JUDGE_TOLERANCE = 0.01
INTERNAL_TOLERANCE = 1e-6
# Safety margin applied inside the optimizer so that a schedule which is
# optimal to the solver is still comfortably valid after rounding to 4dp.
ROUNDING_DECIMALS = 4


@dataclass
class Battery:
    """Section 7.3 battery object."""

    capacity_kwh: float
    initial_energy_kwh: float
    minimum_energy_kwh: float
    max_charge_kwh_per_hour: float
    max_discharge_kwh_per_hour: float


@dataclass
class HourInput:
    """Section 7.2 hour entry."""

    hour: int
    demand_kwh: float
    solar_kwh: float
    tariff_bdt_per_kwh: float


@dataclass
class Scenario:
    scenario_id: str
    operator_notes: List[str]
    hours: List[HourInput]
    battery: Battery

    def demand(self) -> List[float]:
        return [h.demand_kwh for h in self.hours]

    def solar(self) -> List[float]:
        return [h.solar_kwh for h in self.hours]

    def tariff(self) -> List[float]:
        return [h.tariff_bdt_per_kwh for h in self.hours]


@dataclass
class Directive:
    """A single guardrail-validated interpretation entry.

    ``applies``/``directive_type``/``structured_adjustment`` mirror the
    response contract exactly (Section 10.2); ``explanation`` is free text
    that the judge does not match byte-for-byte.
    """

    note_index: int
    applies: bool
    directive_type: str
    structured_adjustment: Optional[Dict]
    explanation: str
    # Diagnostics only -- never serialised into the API response.
    warnings: List[str] = field(default_factory=list)

    def to_response(self) -> Dict:
        return {
            "note_index": self.note_index,
            "applies": self.applies,
            "directive_type": self.directive_type,
            "structured_adjustment": self.structured_adjustment,
            "explanation": self.explanation,
        }

    @property
    def hours(self) -> List[int]:
        if not self.structured_adjustment:
            return []
        return list(self.structured_adjustment.get("hours") or [])


@dataclass
class ConstraintSet:
    """Deterministic constraints handed to the optimizer.

    This is the *only* channel through which LLM output can influence the
    mathematical model -- it is built exclusively from directives that have
    already passed the guardrails.
    """

    effective_solar: List[float]
    reserve_floor: List[float]
    no_charge_hours: set
    no_discharge_hours: set
    max_grid: List[Optional[float]]

    @classmethod
    def baseline(cls, scenario: Scenario) -> "ConstraintSet":
        return cls(
            effective_solar=list(scenario.solar()),
            reserve_floor=[scenario.battery.minimum_energy_kwh] * HORIZON,
            no_charge_hours=set(),
            no_discharge_hours=set(),
            max_grid=[None] * HORIZON,
        )
