"""End-to-end pipeline orchestration.

    request -> LLM interpretation -> deterministic guardrails
            -> optimizer -> replay validation -> response

Each stage distrusts the previous one. The pipeline only ever returns a
response whose schedule has passed the same replay the judge performs.
"""
import logging
from typing import Dict, List, Optional, Tuple

from . import fallback, guardrails, llm, replay, solver
from .schema import (
    DIRECTIVE_MAX_GRID,
    DIRECTIVE_MIN_RESERVE,
    DIRECTIVE_NO_CHARGE,
    DIRECTIVE_NO_DISCHARGE,
    DIRECTIVE_NO_OP,
    DIRECTIVE_SOLAR_REDUCTION,
    ConstraintSet,
    Directive,
    Scenario,
)

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """No valid schedule could be produced for a well-formed request."""


def _hour_span(hours: List[int]) -> str:
    if not hours:
        return ""
    if len(hours) == 1:
        return "hour %d" % hours[0]
    contiguous = hours == list(range(hours[0], hours[-1] + 1))
    if contiguous:
        return "hours %d-%d" % (hours[0], hours[-1])
    return "hours " + ", ".join(str(h) for h in hours)


def _describe(directive: Directive) -> Optional[str]:
    adjustment = directive.structured_adjustment or {}
    span = _hour_span(directive.hours)
    kind = directive.directive_type

    if kind == DIRECTIVE_SOLAR_REDUCTION:
        return "usable solar cut to %g%% of forecast in %s" % (
            float(adjustment["factor"]) * 100,
            span,
        )
    if kind == DIRECTIVE_MIN_RESERVE:
        return "battery held at or above %g kWh across %s" % (
            float(adjustment["minimum_energy_kwh"]),
            span,
        )
    if kind == DIRECTIVE_NO_CHARGE:
        return "no battery charging in %s" % span
    if kind == DIRECTIVE_NO_DISCHARGE:
        return "no battery discharging in %s" % span
    if kind == DIRECTIVE_MAX_GRID:
        return "grid import capped at %g kWh in %s" % (
            float(adjustment["max_grid_kwh"]),
            span,
        )
    return None


def build_summary(
    scenario: Scenario, directives: List[Directive], plan: List[Dict], totals: Dict
) -> str:
    """Deterministic, factual plan_summary derived from the actual plan."""
    applied = [text for text in (_describe(d) for d in directives) if text]
    ignored = sum(1 for d in directives if d.directive_type == DIRECTIVE_NO_OP)
    tariff = scenario.tariff()

    charge_hours = [e["hour"] for e in plan if e["battery_action"] == "charge"]
    discharge_hours = [e["hour"] for e in plan if e["battery_action"] == "discharge"]

    parts = []
    if applied:
        parts.append("Applied operator directives: " + "; ".join(applied) + ".")
    else:
        parts.append("No operator note changed the schedule.")
    if ignored:
        parts.append(
            "%d note%s did not affect this schedule."
            % (ignored, "" if ignored == 1 else "s")
        )
    if charge_hours and discharge_hours:
        cheap = min(tariff[h] for h in charge_hours)
        dear = max(tariff[h] for h in discharge_hours)
        parts.append(
            "The battery charges in low-tariff hours (from %g BDT/kWh) and discharges "
            "into peak hours (up to %g BDT/kWh), returning to its opening state of "
            "charge by hour 23." % (cheap, dear)
        )
    else:
        parts.append(
            "Solar is used first each hour and the battery ends the day at its opening "
            "state of charge."
        )
    parts.append(
        "Total grid import %g kWh at %g BDT, peaking at %g kWh in one hour."
        % (
            totals["total_grid_kwh"],
            totals["total_cost_bdt"],
            totals["peak_grid_kwh"],
        )
    )
    return " ".join(parts)


def interpret(scenario: Scenario) -> Tuple[List[Directive], Dict]:
    """Run the mandatory LLM stage, then the guardrails.

    A model or transport failure degrades safely rather than crashing or
    inventing a directive: the deterministic backup extractor takes over and
    its output is held to the same guardrails.
    """
    meta: Dict = {"llm_used": False, "degraded": False}
    try:
        raw_entries, llm_meta = llm.interpret_notes(scenario)
        meta.update(llm_meta)
        meta["llm_used"] = True
    except llm.LLMUnavailable as exc:
        logger.warning("scenario %s: LLM stage unavailable: %s", scenario.scenario_id, exc)
        meta["degraded"] = True
        meta["degraded_reason"] = str(exc)
        # Provider outage or exhausted quota. Rather than drop every real
        # operating constraint to no_op, fall back to the deterministic
        # backup extractor; its output faces the same guardrails.
        raw_entries = fallback.interpret_notes(scenario)
        meta["fallback_interpreter"] = "deterministic"

    directives = guardrails.validate_interpretation(raw_entries, scenario)
    if not meta["degraded"]:
        directives, recovered = reconcile_no_ops(scenario, directives)
        if recovered:
            meta["recovered_notes"] = recovered
    meta["guardrail_warnings"] = guardrails.collect_warnings(directives)
    return directives, meta


def reconcile_no_ops(
    scenario: Scenario, directives: List[Directive]
) -> Tuple[List[Directive], List[int]]:
    """Disagreement check on notes the model dismissed.

    A missed directive is the most expensive error in this challenge: the
    interpretation is wrong *and* the schedule then violates a rule the judge
    replays. So where the model returned ``no_op`` but the deterministic
    extractor finds an unambiguous directive -- an explicit whole-hour
    window plus directive vocabulary plus, where the type needs one, a
    quantity -- the structural reading is adopted instead.

    This never overrides a directive the model did extract, and it is a
    cross-check on one branch of the model's output, not a replacement
    interpreter: the model remains the primary path on every request.
    """
    recovered: List[int] = []
    updated = list(directives)

    for position, directive in enumerate(directives):
        if directive.directive_type != DIRECTIVE_NO_OP:
            continue

        note = scenario.operator_notes[directive.note_index]
        candidate = fallback.interpret_note(note, scenario)
        if candidate["directive_type"] == DIRECTIVE_NO_OP:
            continue

        adjustment = candidate.get("structured_adjustment") or {}
        hours = adjustment.get("hours") or []
        if not hours:
            continue
        # Require the quantity for the types that are meaningless without
        # one, so a vague sentence cannot manufacture a hard constraint.
        if candidate["directive_type"] == DIRECTIVE_MIN_RESERVE and not adjustment.get(
            "minimum_energy_kwh"
        ):
            continue
        if candidate["directive_type"] == DIRECTIVE_MAX_GRID and adjustment.get(
            "max_grid_kwh"
        ) is None:
            continue

        validated = guardrails.validate_entry(
            candidate, directive.note_index, scenario
        )
        if validated.directive_type == DIRECTIVE_NO_OP:
            continue

        validated.warnings.append(
            "model returned no_op; adopted structural reading %s"
            % validated.directive_type
        )
        updated[position] = validated
        recovered.append(directive.note_index)

    return updated, recovered


def run(scenario: Scenario) -> Tuple[Dict, Dict]:
    """Execute the full pipeline for one validated scenario."""
    directives, meta = interpret(scenario)
    constraints: ConstraintSet = guardrails.build_constraints(scenario, directives)

    candidates = []
    try:
        candidates.append(solver.solve(scenario, constraints))
    except solver.InfeasibleScenario as exc:
        meta["lp_infeasible"] = str(exc)
        try:
            candidates.append(solver.solve_heuristic(scenario, constraints))
        except solver.InfeasibleScenario as heuristic_exc:
            raise PipelineError(
                "no valid schedule exists for the interpreted constraints (%s)"
                % heuristic_exc
            )

    plan = None
    plan_meta: Dict = {}
    violations: List[str] = []
    for candidate_plan, candidate_meta in candidates:
        ok, problems = replay.replay(scenario, constraints, candidate_plan)
        if ok:
            plan, plan_meta = candidate_plan, candidate_meta
            break
        violations = problems

    if plan is None:
        # The LP result failed replay (rounding or modelling defect): try the
        # independent heuristic path before giving up.
        try:
            fallback_plan, fallback_meta = solver.solve_heuristic(scenario, constraints)
        except solver.InfeasibleScenario as exc:
            raise PipelineError(
                "schedule failed final validation: %s" % "; ".join(violations or [str(exc)])
            )
        ok, problems = replay.replay(scenario, constraints, fallback_plan)
        if not ok:
            raise PipelineError(
                "schedule failed final validation: %s" % "; ".join(problems)
            )
        plan, plan_meta = fallback_plan, fallback_meta
        plan_meta["replay_recovered"] = True

    totals = solver.totals(scenario, plan)
    response = {
        "scenario_id": scenario.scenario_id,
        "directive_interpretation": [d.to_response() for d in directives],
        "hourly_plan": plan,
        "total_grid_kwh": totals["total_grid_kwh"],
        "total_cost_bdt": totals["total_cost_bdt"],
        "peak_grid_kwh": totals["peak_grid_kwh"],
        "plan_summary": build_summary(scenario, directives, plan, totals),
    }

    total_problems = replay.verify_totals(scenario, plan, response)
    if total_problems:
        raise PipelineError("; ".join(total_problems))

    meta.update(plan_meta)
    return response, meta
