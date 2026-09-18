"""Optimizer and replay-validator behaviour.

Every schedule produced here is checked with the same replay the judge runs,
so these tests assert validity first and cost second.
"""
import pytest

from optimizer import guardrails, solver
from optimizer.replay import replay, verify_totals
from optimizer.schema import (
    DIRECTIVE_MAX_GRID,
    DIRECTIVE_MIN_RESERVE,
    DIRECTIVE_NO_CHARGE,
    DIRECTIVE_NO_DISCHARGE,
    DIRECTIVE_SOLAR_REDUCTION,
    Battery,
    ConstraintSet,
    Directive,
    HourInput,
    Scenario,
)

# A day with a cheap night, a solar midday, and an expensive evening peak --
# the shape that makes battery arbitrage worthwhile.
DEMAND = [90, 85, 80, 80, 85, 95, 110, 130, 150, 165, 175, 180,
          185, 180, 170, 165, 170, 185, 205, 215, 205, 175, 140, 110]
SOLAR = [0, 0, 0, 0, 0, 0, 5, 20, 50, 90, 130, 160,
         180, 170, 140, 90, 45, 10, 0, 0, 0, 0, 0, 0]
TARIFF = [6, 6, 5, 5, 5, 6, 8, 10, 12, 14, 16, 16,
          15, 14, 13, 14, 18, 22, 28, 30, 26, 20, 12, 8]


def make_scenario(notes=None, **battery_overrides):
    battery_args = {
        "capacity_kwh": 300.0,
        "initial_energy_kwh": 150.0,
        "minimum_energy_kwh": 50.0,
        "max_charge_kwh_per_hour": 60.0,
        "max_discharge_kwh_per_hour": 60.0,
    }
    battery_args.update(battery_overrides)
    return Scenario(
        scenario_id="SOLVE-1",
        operator_notes=notes or ["note"],
        hours=[
            HourInput(
                hour=h,
                demand_kwh=float(DEMAND[h]),
                solar_kwh=float(SOLAR[h]),
                tariff_bdt_per_kwh=float(TARIFF[h]),
            )
            for h in range(24)
        ],
        battery=Battery(**battery_args),
    )


def directive(kind, adjustment, index=0):
    return Directive(
        note_index=index,
        applies=True,
        directive_type=kind,
        structured_adjustment=adjustment,
        explanation="test",
    )


def solve_and_validate(scenario, directives=()):
    constraints = guardrails.build_constraints(scenario, list(directives))
    plan, meta = solver.solve(scenario, constraints)
    ok, problems = replay(scenario, constraints, plan)
    assert ok, problems
    return plan, constraints, meta


class TestBaselineValidity:
    def test_plan_is_valid_and_complete(self):
        scenario = make_scenario()
        plan, _, _ = solve_and_validate(scenario)
        assert [entry["hour"] for entry in plan] == list(range(24))
        assert all(entry["battery_action"] in ("charge", "discharge", "idle") for entry in plan)

    def test_idle_hours_report_zero_magnitude(self):
        scenario = make_scenario()
        plan, _, _ = solve_and_validate(scenario)
        for entry in plan:
            if entry["battery_action"] == "idle":
                assert entry["battery_kwh"] == 0

    def test_end_of_day_neutrality(self):
        scenario = make_scenario()
        plan, _, _ = solve_and_validate(scenario)
        assert plan[-1]["battery_energy_after_kwh"] == pytest.approx(
            scenario.battery.initial_energy_kwh, abs=0.01
        )

    def test_totals_reconcile_with_plan(self):
        scenario = make_scenario()
        plan, _, _ = solve_and_validate(scenario)
        totals = solver.totals(scenario, plan)
        response = dict(totals)
        assert verify_totals(scenario, plan, response) == []

    def test_battery_shifts_energy_into_the_peak(self):
        scenario = make_scenario()
        plan, _, _ = solve_and_validate(scenario)
        peak_hours = [18, 19, 20]
        discharged = sum(
            entry["battery_kwh"]
            for entry in plan
            if entry["hour"] in peak_hours and entry["battery_action"] == "discharge"
        )
        assert discharged > 0

    def test_battery_use_beats_doing_nothing(self):
        """Arbitrage must actually pay: the plan should cost less than the
        same day served without any battery movement."""
        scenario = make_scenario()
        plan, _, _ = solve_and_validate(scenario)
        optimised = solver.totals(scenario, plan)["total_cost_bdt"]
        idle_cost = sum(
            max(0.0, DEMAND[h] - SOLAR[h]) * TARIFF[h] for h in range(24)
        )
        assert optimised < idle_cost


class TestDirectiveApplication:
    def test_solar_reduction_is_respected(self):
        scenario = make_scenario()
        plan, constraints, _ = solve_and_validate(
            scenario,
            [directive(DIRECTIVE_SOLAR_REDUCTION, {"hours": [11, 12], "factor": 0.25})],
        )
        for hour in (11, 12):
            assert plan[hour]["solar_used_kwh"] <= SOLAR[hour] * 0.25 + 0.01

    def test_no_charge_window_is_respected(self):
        scenario = make_scenario()
        plan, _, _ = solve_and_validate(
            scenario, [directive(DIRECTIVE_NO_CHARGE, {"hours": [2, 3, 4]})]
        )
        for hour in (2, 3, 4):
            assert plan[hour]["battery_action"] != "charge"

    def test_no_discharge_window_is_respected(self):
        scenario = make_scenario()
        plan, _, _ = solve_and_validate(
            scenario, [directive(DIRECTIVE_NO_DISCHARGE, {"hours": [18, 19]})]
        )
        for hour in (18, 19):
            assert plan[hour]["battery_action"] != "discharge"

    def test_reserve_floor_is_respected(self):
        scenario = make_scenario()
        plan, _, _ = solve_and_validate(
            scenario,
            [
                directive(
                    DIRECTIVE_MIN_RESERVE,
                    {"hours": [18, 19, 20], "minimum_energy_kwh": 200},
                )
            ],
        )
        for hour in (18, 19, 20):
            assert plan[hour]["battery_energy_after_kwh"] >= 200 - 0.01

    def test_grid_cap_is_respected(self):
        scenario = make_scenario()
        plan, _, _ = solve_and_validate(
            scenario,
            [directive(DIRECTIVE_MAX_GRID, {"hours": [18, 19], "max_grid_kwh": 160})],
        )
        for hour in (18, 19):
            assert plan[hour]["grid_kwh"] <= 160 + 0.01

    def test_two_simultaneous_hard_directives(self):
        scenario = make_scenario()
        plan, _, _ = solve_and_validate(
            scenario,
            [
                directive(
                    DIRECTIVE_MIN_RESERVE,
                    {"hours": [18, 19, 20, 21], "minimum_energy_kwh": 120},
                    index=0,
                ),
                directive(
                    DIRECTIVE_MAX_GRID,
                    {"hours": [19, 20], "max_grid_kwh": 180},
                    index=1,
                ),
            ],
        )
        for hour in (19, 20):
            assert plan[hour]["grid_kwh"] <= 180.01
            assert plan[hour]["battery_energy_after_kwh"] >= 119.99

    def test_directive_costs_no_more_than_unconstrained(self):
        """A hard directive can only ever make the day more expensive."""
        scenario = make_scenario()
        free_plan, _, _ = solve_and_validate(scenario)
        capped_plan, _, _ = solve_and_validate(
            scenario,
            # 170 kWh is reachable at hours 18-19 (demand 205/215 less the
            # 60 kWh discharge limit), so the day stays feasible.
            [directive(DIRECTIVE_MAX_GRID, {"hours": [18, 19], "max_grid_kwh": 170})],
        )
        assert (
            solver.totals(scenario, capped_plan)["total_cost_bdt"]
            >= solver.totals(scenario, free_plan)["total_cost_bdt"] - 0.01
        )


class TestEdgeCases:
    def test_zero_solar_day(self):
        scenario = make_scenario()
        for hour in scenario.hours:
            hour.solar_kwh = 0.0
        solve_and_validate(scenario)

    def test_solar_exceeding_demand_is_curtailed(self):
        scenario = make_scenario()
        for hour in scenario.hours:
            hour.solar_kwh = 10_000.0
        plan, _, _ = solve_and_validate(scenario)
        for entry in plan:
            assert entry["grid_kwh"] == pytest.approx(0.0, abs=0.01)

    def test_flat_tariff_still_valid(self):
        scenario = make_scenario()
        for hour in scenario.hours:
            hour.tariff_bdt_per_kwh = 9.0
        solve_and_validate(scenario)

    def test_immovable_battery(self):
        scenario = make_scenario(
            max_charge_kwh_per_hour=0.0, max_discharge_kwh_per_hour=0.0
        )
        plan, _, _ = solve_and_validate(scenario)
        assert all(entry["battery_action"] == "idle" for entry in plan)

    def test_battery_starting_at_capacity(self):
        scenario = make_scenario(initial_energy_kwh=300.0)
        solve_and_validate(scenario)

    def test_full_reserve_pins_the_battery(self):
        """A reserve equal to the starting level leaves no room to discharge."""
        scenario = make_scenario(initial_energy_kwh=150.0)
        plan, _, _ = solve_and_validate(
            scenario,
            [
                directive(
                    DIRECTIVE_MIN_RESERVE,
                    {"hours": list(range(24)), "minimum_energy_kwh": 150},
                )
            ],
        )
        for entry in plan:
            assert entry["battery_energy_after_kwh"] >= 149.99

    def test_infeasible_grid_cap_is_reported(self):
        """A cap below what demand requires must fail loudly, not silently
        return an invalid plan."""
        scenario = make_scenario(
            max_discharge_kwh_per_hour=0.0, max_charge_kwh_per_hour=0.0
        )
        constraints = guardrails.build_constraints(scenario, [])
        constraints.max_grid[0] = 1.0  # demand is 90 kWh with no solar at hour 0
        with pytest.raises(solver.InfeasibleScenario):
            solver.solve(scenario, constraints)


class TestHeuristicFallback:
    """The fallback exists for solver-unavailable environments; it must
    produce plans that pass the same replay."""

    def test_baseline(self):
        scenario = make_scenario()
        constraints = guardrails.build_constraints(scenario, [])
        plan, _ = solver.solve_heuristic(scenario, constraints)
        ok, problems = replay(scenario, constraints, plan)
        assert ok, problems

    def test_with_every_directive_family(self):
        scenario = make_scenario(notes=["a", "b", "c", "d"])
        directives = [
            directive(DIRECTIVE_SOLAR_REDUCTION, {"hours": [11, 12], "factor": 0.3}, 0),
            directive(DIRECTIVE_NO_CHARGE, {"hours": [2, 3]}, 1),
            directive(DIRECTIVE_NO_DISCHARGE, {"hours": [18]}, 2),
            directive(
                DIRECTIVE_MIN_RESERVE, {"hours": [20, 21], "minimum_energy_kwh": 120}, 3
            ),
        ]
        constraints = guardrails.build_constraints(scenario, directives)
        plan, _ = solver.solve_heuristic(scenario, constraints)
        ok, problems = replay(scenario, constraints, plan)
        assert ok, problems

    def test_lp_is_at_least_as_cheap_as_heuristic(self):
        scenario = make_scenario()
        constraints = guardrails.build_constraints(scenario, [])
        lp_plan, _ = solver.solve_lp(scenario, constraints)
        heuristic_plan, _ = solver.solve_heuristic(scenario, constraints)
        assert (
            solver.totals(scenario, lp_plan)["total_cost_bdt"]
            <= solver.totals(scenario, heuristic_plan)["total_cost_bdt"] + 0.01
        )


class TestReplayCatchesViolations:
    """The replay validator is the last line of defence, so prove it
    actually rejects each class of bad plan."""

    def _valid_plan(self, scenario):
        constraints = guardrails.build_constraints(scenario, [])
        plan, _ = solver.solve(scenario, constraints)
        return plan, constraints

    def test_detects_broken_energy_balance(self):
        scenario = make_scenario()
        plan, constraints = self._valid_plan(scenario)
        plan[5]["grid_kwh"] += 25
        ok, problems = replay(scenario, constraints, plan)
        assert not ok and any("balance" in p for p in problems)

    def test_detects_solar_overuse(self):
        scenario = make_scenario()
        plan, constraints = self._valid_plan(scenario)
        plan[12]["solar_used_kwh"] += 500
        ok, problems = replay(scenario, constraints, plan)
        assert not ok and any("solar" in p for p in problems)

    def test_detects_broken_neutrality(self):
        scenario = make_scenario()
        plan, constraints = self._valid_plan(scenario)
        plan[23]["battery_energy_after_kwh"] -= 30
        ok, problems = replay(scenario, constraints, plan)
        assert not ok

    def test_detects_missing_hour(self):
        scenario = make_scenario()
        plan, constraints = self._valid_plan(scenario)
        plan.pop()
        ok, _ = replay(scenario, constraints, plan)
        assert not ok

    def test_detects_idle_with_movement(self):
        scenario = make_scenario()
        plan, constraints = self._valid_plan(scenario)
        plan[0]["battery_action"] = "idle"
        plan[0]["battery_kwh"] = 30
        ok, problems = replay(scenario, constraints, plan)
        assert not ok

    def test_detects_rate_limit_breach(self):
        scenario = make_scenario()
        plan, constraints = self._valid_plan(scenario)
        plan[19]["battery_action"] = "discharge"
        plan[19]["battery_kwh"] = 5_000
        ok, problems = replay(scenario, constraints, plan)
        assert not ok and any("limit" in p for p in problems)
