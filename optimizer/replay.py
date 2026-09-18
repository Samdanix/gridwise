"""Final replay validator (Problem Statement Section 08 "Final replay").

The judge independently replays our published ``hourly_plan`` hour by hour
against the ground-truth directives and the GridWise energy rules. This
module runs that same replay before we answer, so an invalid plan is never
returned as a success.
"""
from typing import Dict, List, Tuple

from .schema import (
    ACTION_CHARGE,
    ACTION_DISCHARGE,
    ACTION_IDLE,
    BATTERY_ACTIONS,
    HORIZON,
    JUDGE_TOLERANCE,
    ConstraintSet,
    Scenario,
)

TOL = JUDGE_TOLERANCE


def replay(
    scenario: Scenario, constraints: ConstraintSet, plan: List[Dict]
) -> Tuple[bool, List[str]]:
    """Return ``(is_valid, violations)`` for a candidate plan."""
    problems: List[str] = []
    battery = scenario.battery

    if len(plan) != HORIZON:
        return False, ["hourly_plan must contain exactly %d entries" % HORIZON]

    seen = sorted(entry.get("hour") for entry in plan)
    if seen != list(range(HORIZON)):
        return False, ["hourly_plan hours must be the unique integers 0..23"]

    by_hour = {entry["hour"]: entry for entry in plan}
    energy = float(battery.initial_energy_kwh)

    for hour in range(HORIZON):
        entry = by_hour[hour]
        grid = float(entry["grid_kwh"])
        solar_used = float(entry["solar_used_kwh"])
        action = entry["battery_action"]
        magnitude = float(entry["battery_kwh"])
        reported = float(entry["battery_energy_after_kwh"])

        if grid < -TOL or solar_used < -TOL or magnitude < -TOL:
            problems.append("hour %d: negative energy value" % hour)
        if action not in BATTERY_ACTIONS:
            problems.append("hour %d: invalid battery_action %r" % (hour, action))
            continue
        if action == ACTION_IDLE and abs(magnitude) > TOL:
            problems.append("hour %d: idle hour must have battery_kwh 0" % hour)

        # Section 9.4 -- solar usage may not exceed effective solar.
        available = constraints.effective_solar[hour]
        if solar_used > available + TOL:
            problems.append(
                "hour %d: solar_used %.4f exceeds effective solar %.4f"
                % (hour, solar_used, available)
            )

        charge = magnitude if action == ACTION_CHARGE else 0.0
        discharge = magnitude if action == ACTION_DISCHARGE else 0.0

        # Section 9.3 -- hourly rate limits.
        if charge > battery.max_charge_kwh_per_hour + TOL:
            problems.append("hour %d: charge exceeds hourly charge limit" % hour)
        if discharge > battery.max_discharge_kwh_per_hour + TOL:
            problems.append("hour %d: discharge exceeds hourly discharge limit" % hour)

        # Directive constraints checked directly against the plan.
        if hour in constraints.no_charge_hours and charge > TOL:
            problems.append("hour %d: charging inside a no_charge_window" % hour)
        if hour in constraints.no_discharge_hours and discharge > TOL:
            problems.append("hour %d: discharging inside a no_discharge_window" % hour)
        cap = constraints.max_grid[hour]
        if cap is not None and grid > cap + TOL:
            problems.append(
                "hour %d: grid import %.4f exceeds cap %.4f" % (hour, grid, cap)
            )

        # Section 9.5 -- energy balance.
        supply = grid + solar_used + discharge
        draw = scenario.hours[hour].demand_kwh + charge
        if abs(supply - draw) > TOL:
            problems.append(
                "hour %d: energy balance off by %.4f kWh" % (hour, supply - draw)
            )

        # Section 9.1/9.2 -- state transition and bounds.
        energy = energy + charge - discharge
        if abs(energy - reported) > TOL:
            problems.append(
                "hour %d: battery_energy_after_kwh %.4f does not follow the transition (%.4f)"
                % (hour, reported, energy)
            )
            energy = reported
        floor = constraints.reserve_floor[hour]
        if energy < floor - TOL:
            problems.append(
                "hour %d: battery energy %.4f below required floor %.4f"
                % (hour, energy, floor)
            )
        if energy > battery.capacity_kwh + TOL:
            problems.append("hour %d: battery energy above capacity" % hour)

    # Section 9.6 -- end-of-day neutrality.
    if abs(energy - battery.initial_energy_kwh) > TOL:
        problems.append(
            "final battery energy %.4f does not equal initial %.4f"
            % (energy, battery.initial_energy_kwh)
        )

    return (not problems), problems


def verify_totals(scenario: Scenario, plan: List[Dict], response: Dict) -> List[str]:
    """Confirm reported aggregates match a recalculation from the plan."""
    tariff = scenario.tariff()
    expected_grid = sum(float(e["grid_kwh"]) for e in plan)
    expected_cost = sum(float(e["grid_kwh"]) * tariff[e["hour"]] for e in plan)
    expected_peak = max((float(e["grid_kwh"]) for e in plan), default=0.0)

    problems: List[str] = []
    for field, expected in (
        ("total_grid_kwh", expected_grid),
        ("total_cost_bdt", expected_cost),
        ("peak_grid_kwh", expected_peak),
    ):
        if abs(float(response[field]) - expected) > TOL:
            problems.append(
                "%s (%.4f) disagrees with recalculation (%.4f)"
                % (field, float(response[field]), expected)
            )
    return problems
