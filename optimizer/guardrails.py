"""Deterministic guardrails (Problem Statement Section 08).

LLM output is untrusted structured data. Nothing reaches the optimizer until
it has survived every check in this module. Anything that cannot be repaired
deterministically degrades to ``no_op`` -- we never invent a directive type
and we never raise out of here.
"""
import math
from typing import Any, Dict, List, Optional, Tuple

from .schema import (
    DIRECTIVE_MAX_GRID,
    DIRECTIVE_MIN_RESERVE,
    DIRECTIVE_NO_CHARGE,
    DIRECTIVE_NO_DISCHARGE,
    DIRECTIVE_NO_OP,
    DIRECTIVE_SOLAR_REDUCTION,
    DIRECTIVE_TYPES,
    HORIZON,
    ConstraintSet,
    Directive,
    Scenario,
)

NO_OP_EXPLANATION = "This note does not affect today's 24-hour energy schedule."


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _normalise_hours(raw: Any) -> Tuple[List[int], List[str]]:
    """Coerce an hours payload into unique ascending ints in 0..23.

    Returns the cleaned list plus any warnings. An empty result means the
    directive cannot be applied and must collapse to no_op.
    """
    warnings: List[str] = []
    if not isinstance(raw, (list, tuple)):
        return [], ["hours was not a list"]

    cleaned = set()
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float, str)):
            warnings.append("dropped non-numeric hour %r" % (item,))
            continue
        try:
            value = float(item)
        except (TypeError, ValueError):
            warnings.append("dropped unparseable hour %r" % (item,))
            continue
        if not math.isfinite(value) or value != int(value):
            warnings.append("dropped non-integer hour %r" % (item,))
            continue
        hour = int(value)
        if hour < 0 or hour >= HORIZON:
            warnings.append("dropped out-of-range hour %d" % hour)
            continue
        cleaned.add(hour)

    if not cleaned:
        warnings.append("no valid hours remained")
    return sorted(cleaned), warnings


def _as_no_op(note_index: int, explanation: str, warnings: List[str]) -> Directive:
    return Directive(
        note_index=note_index,
        applies=False,
        directive_type=DIRECTIVE_NO_OP,
        structured_adjustment=None,
        explanation=explanation or NO_OP_EXPLANATION,
        warnings=warnings,
    )


def validate_entry(raw: Any, note_index: int, scenario: Scenario) -> Directive:
    """Validate one raw interpretation entry for a known note index."""
    warnings: List[str] = []

    if not isinstance(raw, dict):
        return _as_no_op(note_index, NO_OP_EXPLANATION, ["entry was not an object"])

    explanation = raw.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        explanation = ""
    else:
        explanation = explanation.strip()[:400]

    directive_type = raw.get("directive_type")
    if not isinstance(directive_type, str):
        return _as_no_op(note_index, explanation, ["directive_type was not a string"])
    directive_type = directive_type.strip().lower()

    # Guardrail: allowed types only. An unsupported type is never passed
    # through and never renamed into something adjacent -- it becomes no_op.
    if directive_type not in DIRECTIVE_TYPES:
        return _as_no_op(
            note_index, explanation, ["unsupported directive_type %r" % directive_type]
        )

    if directive_type == DIRECTIVE_NO_OP:
        return _as_no_op(note_index, explanation, warnings)

    adjustment = raw.get("structured_adjustment")
    if not isinstance(adjustment, dict):
        return _as_no_op(
            note_index, explanation, ["structured_adjustment missing for non-no_op"]
        )

    hours, hour_warnings = _normalise_hours(adjustment.get("hours"))
    warnings.extend(hour_warnings)
    if not hours:
        return _as_no_op(note_index, explanation, warnings)

    battery = scenario.battery
    clean: Dict[str, Any] = {"hours": hours}

    if directive_type == DIRECTIVE_SOLAR_REDUCTION:
        factor = adjustment.get("factor")
        if not _is_finite_number(factor):
            return _as_no_op(note_index, explanation, warnings + ["factor not numeric"])
        factor = float(factor)
        # Tolerate a model that answered in percent (e.g. 20 meaning 0.2).
        if 1.0 < factor <= 100.0:
            warnings.append("factor %g interpreted as a percentage" % factor)
            factor = factor / 100.0
        if factor < 0.0 or factor > 1.0:
            return _as_no_op(
                note_index, explanation, warnings + ["factor outside 0..1"]
            )
        clean["factor"] = factor

    elif directive_type == DIRECTIVE_MIN_RESERVE:
        reserve = adjustment.get("minimum_energy_kwh")
        if not _is_finite_number(reserve):
            return _as_no_op(
                note_index, explanation, warnings + ["minimum_energy_kwh not numeric"]
            )
        reserve = float(reserve)
        if reserve < 0.0:
            return _as_no_op(
                note_index, explanation, warnings + ["negative reserve"]
            )
        if reserve > battery.capacity_kwh:
            # Guardrail: reserve may not exceed capacity. Clamping keeps the
            # scenario feasible rather than discarding an applicable rule.
            warnings.append(
                "reserve %g clamped to capacity %g" % (reserve, battery.capacity_kwh)
            )
            reserve = float(battery.capacity_kwh)
        clean["minimum_energy_kwh"] = reserve

    elif directive_type == DIRECTIVE_MAX_GRID:
        cap = adjustment.get("max_grid_kwh")
        if not _is_finite_number(cap):
            return _as_no_op(
                note_index, explanation, warnings + ["max_grid_kwh not numeric"]
            )
        cap = float(cap)
        if cap < 0.0:
            return _as_no_op(note_index, explanation, warnings + ["negative grid cap"])
        clean["max_grid_kwh"] = cap

    return Directive(
        note_index=note_index,
        applies=True,
        directive_type=directive_type,
        structured_adjustment=clean,
        explanation=explanation or "Interpreted as %s." % directive_type,
        warnings=warnings,
    )


def validate_interpretation(raw_entries: Any, scenario: Scenario) -> List[Directive]:
    """Build exactly one validated directive per operator note.

    Guarantees, regardless of what the model returned:
      * length == len(operator_notes)
      * one entry per note_index, ascending, no gaps, no duplicates
      * applies is False only for no_op, and then adjustment is None
    """
    note_count = len(scenario.operator_notes)
    by_index: Dict[int, Any] = {}

    if isinstance(raw_entries, dict):
        raw_entries = raw_entries.get("directive_interpretation")

    if isinstance(raw_entries, list):
        for position, entry in enumerate(raw_entries):
            index = None
            if isinstance(entry, dict):
                candidate = entry.get("note_index")
                if not isinstance(candidate, bool) and isinstance(
                    candidate, (int, float)
                ):
                    if float(candidate) == int(candidate):
                        index = int(candidate)
            if index is None or index < 0 or index >= note_count:
                # Fall back to list position when the model omitted or
                # invented a note_index.
                index = position
            if index >= note_count or index in by_index:
                continue
            by_index[index] = entry

    directives: List[Directive] = []
    for note_index in range(note_count):
        raw = by_index.get(note_index)
        if raw is None:
            directives.append(
                _as_no_op(note_index, NO_OP_EXPLANATION, ["no entry returned for note"])
            )
        else:
            directives.append(validate_entry(raw, note_index, scenario))
    return directives


def build_constraints(scenario: Scenario, directives: List[Directive]) -> ConstraintSet:
    """Translate validated directives into optimizer constraints (Section 5.3).

    Multiple directives of the same type compose conservatively: the
    strictest requirement wins, so a schedule satisfying the result
    satisfies every individual directive.
    """
    constraints = ConstraintSet.baseline(scenario)

    for directive in directives:
        if not directive.applies or directive.directive_type == DIRECTIVE_NO_OP:
            continue
        adjustment = directive.structured_adjustment or {}
        hours = adjustment.get("hours") or []

        if directive.directive_type == DIRECTIVE_SOLAR_REDUCTION:
            factor = float(adjustment["factor"])
            for hour in hours:
                new_solar = scenario.hours[hour].solar_kwh * factor
                constraints.effective_solar[hour] = min(
                    constraints.effective_solar[hour], new_solar
                )

        elif directive.directive_type == DIRECTIVE_MIN_RESERVE:
            reserve = float(adjustment["minimum_energy_kwh"])
            for hour in hours:
                constraints.reserve_floor[hour] = max(
                    constraints.reserve_floor[hour], reserve
                )

        elif directive.directive_type == DIRECTIVE_NO_CHARGE:
            constraints.no_charge_hours.update(hours)

        elif directive.directive_type == DIRECTIVE_NO_DISCHARGE:
            constraints.no_discharge_hours.update(hours)

        elif directive.directive_type == DIRECTIVE_MAX_GRID:
            cap = float(adjustment["max_grid_kwh"])
            for hour in hours:
                current = constraints.max_grid[hour]
                constraints.max_grid[hour] = cap if current is None else min(current, cap)

    return constraints


def stacked_solar(scenario: Scenario, directives: List[Directive]) -> List[float]:
    """Effective solar per hour -- exposed for the replay validator."""
    return build_constraints(scenario, directives).effective_solar


def collect_warnings(directives: List[Directive]) -> List[str]:
    out: List[str] = []
    for directive in directives:
        for warning in directive.warnings:
            out.append("note %d: %s" % (directive.note_index, warning))
    return out


def optional_float(value: Any) -> Optional[float]:
    return float(value) if _is_finite_number(value) else None
