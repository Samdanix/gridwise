"""Guardrail behaviour: hostile and malformed model output must never reach
the optimizer, and must never crash the service.
"""
import pytest

from optimizer import guardrails
from optimizer.schema import (
    DIRECTIVE_MAX_GRID,
    DIRECTIVE_MIN_RESERVE,
    DIRECTIVE_NO_OP,
    DIRECTIVE_SOLAR_REDUCTION,
    Battery,
    HourInput,
    Scenario,
)


def make_scenario(notes=None, capacity=200.0):
    return Scenario(
        scenario_id="T-1",
        operator_notes=notes if notes is not None else ["a note"],
        hours=[
            HourInput(hour=h, demand_kwh=100.0, solar_kwh=50.0, tariff_bdt_per_kwh=10.0)
            for h in range(24)
        ],
        battery=Battery(
            capacity_kwh=capacity,
            initial_energy_kwh=120.0,
            minimum_energy_kwh=40.0,
            max_charge_kwh_per_hour=50.0,
            max_discharge_kwh_per_hour=50.0,
        ),
    )


def entry(**kwargs):
    base = {
        "note_index": 0,
        "applies": True,
        "directive_type": DIRECTIVE_SOLAR_REDUCTION,
        "structured_adjustment": {"hours": [13, 14], "factor": 0.2},
        "explanation": "test",
    }
    base.update(kwargs)
    return base


class TestShapeEnforcement:
    def test_one_entry_per_note_in_order(self):
        scenario = make_scenario(["a", "b", "c"])
        # Model returned them shuffled and with a duplicate index.
        raw = [
            entry(note_index=2, directive_type=DIRECTIVE_NO_OP, applies=False,
                  structured_adjustment=None),
            entry(note_index=0),
            entry(note_index=0, directive_type=DIRECTIVE_NO_OP, applies=False,
                  structured_adjustment=None),
        ]
        result = guardrails.validate_interpretation(raw, scenario)
        assert [d.note_index for d in result] == [0, 1, 2]
        assert result[0].directive_type == DIRECTIVE_SOLAR_REDUCTION
        assert result[1].directive_type == DIRECTIVE_NO_OP  # never returned
        assert result[2].directive_type == DIRECTIVE_NO_OP

    def test_missing_entries_become_no_op(self):
        scenario = make_scenario(["a", "b"])
        result = guardrails.validate_interpretation([], scenario)
        assert len(result) == 2
        assert all(d.directive_type == DIRECTIVE_NO_OP for d in result)
        assert all(d.applies is False for d in result)
        assert all(d.structured_adjustment is None for d in result)

    @pytest.mark.parametrize(
        "raw", [None, "nonsense", 42, {"unexpected": True}, [], [None], [[]]]
    )
    def test_garbage_never_raises(self, raw):
        scenario = make_scenario(["a"])
        result = guardrails.validate_interpretation(raw, scenario)
        assert len(result) == 1
        assert result[0].directive_type == DIRECTIVE_NO_OP

    def test_no_op_always_has_null_adjustment(self):
        scenario = make_scenario(["a"])
        raw = [entry(directive_type=DIRECTIVE_NO_OP, applies=True,
                     structured_adjustment={"hours": [1]})]
        result = guardrails.validate_interpretation(raw, scenario)
        assert result[0].applies is False
        assert result[0].structured_adjustment is None

    def test_applies_is_forced_true_for_real_directives(self):
        scenario = make_scenario(["a"])
        raw = [entry(applies=False)]
        result = guardrails.validate_interpretation(raw, scenario)
        assert result[0].applies is True


class TestTypeSafety:
    @pytest.mark.parametrize(
        "bad_type",
        ["shed_load", "SOLAR_REDUCTION_V2", "", None, 7, "no_op_but_different"],
    )
    def test_unsupported_types_collapse_to_no_op(self, bad_type):
        scenario = make_scenario(["a"])
        result = guardrails.validate_interpretation(
            [entry(directive_type=bad_type)], scenario
        )
        assert result[0].directive_type == DIRECTIVE_NO_OP

    def test_uppercase_type_is_normalised(self):
        scenario = make_scenario(["a"])
        result = guardrails.validate_interpretation(
            [entry(directive_type="Solar_Reduction")], scenario
        )
        assert result[0].directive_type == DIRECTIVE_SOLAR_REDUCTION


class TestHourNormalisation:
    def test_hours_are_deduplicated_and_sorted(self):
        scenario = make_scenario(["a"])
        result = guardrails.validate_interpretation(
            [entry(structured_adjustment={"hours": [14, 13, 14, 13], "factor": 0.5})],
            scenario,
        )
        assert result[0].structured_adjustment["hours"] == [13, 14]

    def test_out_of_range_hours_are_dropped(self):
        scenario = make_scenario(["a"])
        result = guardrails.validate_interpretation(
            [entry(structured_adjustment={"hours": [-1, 5, 24, 99], "factor": 0.5})],
            scenario,
        )
        assert result[0].structured_adjustment["hours"] == [5]

    def test_all_hours_invalid_becomes_no_op(self):
        scenario = make_scenario(["a"])
        result = guardrails.validate_interpretation(
            [entry(structured_adjustment={"hours": [99, "x", None], "factor": 0.5})],
            scenario,
        )
        assert result[0].directive_type == DIRECTIVE_NO_OP

    def test_string_hours_are_coerced(self):
        scenario = make_scenario(["a"])
        result = guardrails.validate_interpretation(
            [entry(structured_adjustment={"hours": ["13", 14.0], "factor": 0.5})],
            scenario,
        )
        assert result[0].structured_adjustment["hours"] == [13, 14]


class TestNumericGuardrails:
    @pytest.mark.parametrize("factor,expected", [(0.0, 0.0), (1.0, 1.0), (0.25, 0.25)])
    def test_valid_factors_pass(self, factor, expected):
        scenario = make_scenario(["a"])
        result = guardrails.validate_interpretation(
            [entry(structured_adjustment={"hours": [1], "factor": factor})], scenario
        )
        assert result[0].structured_adjustment["factor"] == expected

    def test_percentage_style_factor_is_rescaled(self):
        scenario = make_scenario(["a"])
        result = guardrails.validate_interpretation(
            [entry(structured_adjustment={"hours": [1], "factor": 20})], scenario
        )
        assert result[0].structured_adjustment["factor"] == pytest.approx(0.2)

    @pytest.mark.parametrize("factor", [-0.5, 101, float("inf"), float("nan"), "half", None])
    def test_invalid_factor_becomes_no_op(self, factor):
        scenario = make_scenario(["a"])
        result = guardrails.validate_interpretation(
            [entry(structured_adjustment={"hours": [1], "factor": factor})], scenario
        )
        assert result[0].directive_type == DIRECTIVE_NO_OP

    def test_reserve_above_capacity_is_clamped(self):
        scenario = make_scenario(["a"], capacity=200.0)
        result = guardrails.validate_interpretation(
            [
                entry(
                    directive_type=DIRECTIVE_MIN_RESERVE,
                    structured_adjustment={"hours": [18], "minimum_energy_kwh": 9999},
                )
            ],
            scenario,
        )
        assert result[0].directive_type == DIRECTIVE_MIN_RESERVE
        assert result[0].structured_adjustment["minimum_energy_kwh"] == 200.0

    def test_negative_reserve_becomes_no_op(self):
        scenario = make_scenario(["a"])
        result = guardrails.validate_interpretation(
            [
                entry(
                    directive_type=DIRECTIVE_MIN_RESERVE,
                    structured_adjustment={"hours": [18], "minimum_energy_kwh": -5},
                )
            ],
            scenario,
        )
        assert result[0].directive_type == DIRECTIVE_NO_OP

    def test_negative_grid_cap_becomes_no_op(self):
        scenario = make_scenario(["a"])
        result = guardrails.validate_interpretation(
            [
                entry(
                    directive_type=DIRECTIVE_MAX_GRID,
                    structured_adjustment={"hours": [18], "max_grid_kwh": -1},
                )
            ],
            scenario,
        )
        assert result[0].directive_type == DIRECTIVE_NO_OP

    def test_zero_grid_cap_is_legitimate(self):
        scenario = make_scenario(["a"])
        result = guardrails.validate_interpretation(
            [
                entry(
                    directive_type=DIRECTIVE_MAX_GRID,
                    structured_adjustment={"hours": [18], "max_grid_kwh": 0},
                )
            ],
            scenario,
        )
        assert result[0].structured_adjustment["max_grid_kwh"] == 0.0


class TestConstraintComposition:
    def test_solar_reduction_scales_only_listed_hours(self):
        scenario = make_scenario(["a"])
        directives = guardrails.validate_interpretation(
            [entry(structured_adjustment={"hours": [13, 14], "factor": 0.25})], scenario
        )
        constraints = guardrails.build_constraints(scenario, directives)
        assert constraints.effective_solar[13] == pytest.approx(12.5)
        assert constraints.effective_solar[12] == pytest.approx(50.0)

    def test_strictest_reserve_wins(self):
        scenario = make_scenario(["a", "b"])
        raw = [
            entry(
                note_index=0,
                directive_type=DIRECTIVE_MIN_RESERVE,
                structured_adjustment={"hours": [18], "minimum_energy_kwh": 90},
            ),
            entry(
                note_index=1,
                directive_type=DIRECTIVE_MIN_RESERVE,
                structured_adjustment={"hours": [18], "minimum_energy_kwh": 150},
            ),
        ]
        directives = guardrails.validate_interpretation(raw, scenario)
        constraints = guardrails.build_constraints(scenario, directives)
        assert constraints.reserve_floor[18] == 150.0

    def test_tightest_grid_cap_wins(self):
        scenario = make_scenario(["a", "b"])
        raw = [
            entry(
                note_index=0,
                directive_type=DIRECTIVE_MAX_GRID,
                structured_adjustment={"hours": [19], "max_grid_kwh": 180},
            ),
            entry(
                note_index=1,
                directive_type=DIRECTIVE_MAX_GRID,
                structured_adjustment={"hours": [19], "max_grid_kwh": 150},
            ),
        ]
        directives = guardrails.validate_interpretation(raw, scenario)
        constraints = guardrails.build_constraints(scenario, directives)
        assert constraints.max_grid[19] == 150.0

    def test_base_minimum_is_never_lowered_by_a_directive(self):
        scenario = make_scenario(["a"])
        directives = guardrails.validate_interpretation(
            [
                entry(
                    directive_type=DIRECTIVE_MIN_RESERVE,
                    structured_adjustment={"hours": [3], "minimum_energy_kwh": 10},
                )
            ],
            scenario,
        )
        constraints = guardrails.build_constraints(scenario, directives)
        assert constraints.reserve_floor[3] == 40.0  # the base minimum
