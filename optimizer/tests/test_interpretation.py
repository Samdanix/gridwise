"""Offline interpretation tests: response reshaping and the deterministic
backup extractor (time-window convention, quantities, distractors).
"""
import pytest

from optimizer import fallback, llm
from optimizer.schema import (
    DIRECTIVE_MAX_GRID,
    DIRECTIVE_MIN_RESERVE,
    DIRECTIVE_NO_CHARGE,
    DIRECTIVE_NO_DISCHARGE,
    DIRECTIVE_NO_OP,
    DIRECTIVE_SOLAR_REDUCTION,
    Battery,
    HourInput,
    Scenario,
)


def make_scenario(notes, capacity=200.0):
    return Scenario(
        scenario_id="INT-1",
        operator_notes=notes,
        hours=[
            HourInput(hour=h, demand_kwh=100.0, solar_kwh=80.0, tariff_bdt_per_kwh=10.0)
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


class TestModelOutputReshaping:
    def test_flat_keys_are_folded_into_structured_adjustment(self):
        entries = llm.normalise_entries(
            {
                "directive_interpretation": [
                    {
                        "note_index": 0,
                        "applies": True,
                        "directive_type": "solar_reduction",
                        "hours": [13, 14],
                        "factor": 0.2,
                        "explanation": "x",
                    }
                ]
            }
        )
        assert entries[0]["structured_adjustment"] == {"hours": [13, 14], "factor": 0.2}

    def test_already_nested_output_is_accepted(self):
        entries = llm.normalise_entries(
            [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "max_grid_window",
                    "structured_adjustment": {"hours": [19], "max_grid_kwh": 180},
                    "explanation": "x",
                }
            ]
        )
        assert entries[0]["structured_adjustment"]["max_grid_kwh"] == 180

    def test_keys_foreign_to_the_type_are_stripped(self):
        entries = llm.normalise_entries(
            [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "no_charge_window",
                    "hours": [2, 3],
                    "factor": 0.5,
                    "max_grid_kwh": 100,
                    "explanation": "x",
                }
            ]
        )
        assert entries[0]["structured_adjustment"] == {"hours": [2, 3]}

    def test_no_op_adjustment_is_nulled(self):
        entries = llm.normalise_entries(
            [
                {
                    "note_index": 0,
                    "applies": False,
                    "directive_type": "no_op",
                    "hours": [1],
                    "explanation": "x",
                }
            ]
        )
        assert entries[0]["structured_adjustment"] is None

    @pytest.mark.parametrize("payload", [None, "text", 5, {}, {"other": 1}, [1, 2]])
    def test_unusable_payloads_do_not_raise(self, payload):
        assert isinstance(llm.normalise_entries(payload), list)

    def test_fenced_json_is_extracted(self):
        parsed = llm._extract_json('```json\n{"directive_interpretation": []}\n```')
        assert parsed == {"directive_interpretation": []}

    def test_json_embedded_in_prose_is_extracted(self):
        parsed = llm._extract_json('Sure! {"a": 1} hope that helps')
        assert parsed == {"a": 1}


class TestWindowConvention:
    """Start hour inclusive, end hour exclusive, on a 0-23 clock."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("from 1 PM to 3 PM", [13, 14]),
            ("between 2 PM and 4 PM", [14, 15]),
            ("from 6 PM until 9 PM", [18, 19, 20]),
            ("from 2 AM until 5 AM", [2, 3, 4]),
            ("from noon until 2 PM", [12, 13]),
            ("from 10 AM until noon", [10, 11]),
            ("between 11 AM and 2 PM", [11, 12, 13]),
            ("between 09:00 and 12:00", [9, 10, 11]),
            ("from 17:00 to 20:00", [17, 18, 19]),
            ("from 6 PM until 10 PM", [18, 19, 20, 21]),
            ("during the 1-3 PM maintenance window", [13, 14]),
            ("from midnight until 3 AM", [0, 1, 2]),
            ("from 8 PM through to 11 PM", [20, 21, 22]),
            ("from ten in the morning to one in the afternoon", [10, 11, 12]),
        ],
    )
    def test_windows(self, text, expected):
        assert fallback.extract_hours(text) == expected

    def test_afternoon_default_for_bare_working_hours(self):
        # "one until three" in an operations note means the afternoon.
        assert fallback.extract_hours("panel washing from one until three") == [13, 14]


class TestBackupExtractor:
    @pytest.mark.parametrize(
        "note,kind,adjustment",
        [
            (
                "Solar output will drop to about 20% from 1 PM to 3 PM.",
                DIRECTIVE_SOLAR_REDUCTION,
                {"hours": [13, 14], "factor": 0.2},
            ),
            (
                "Expect an 80% reduction in rooftop solar during the 1-3 PM window.",
                DIRECTIVE_SOLAR_REDUCTION,
                {"hours": [13, 14], "factor": 0.2},
            ),
            (
                "Cloud cover will leave about half of forecast solar from 10 AM until noon.",
                DIRECTIVE_SOLAR_REDUCTION,
                {"hours": [10, 11], "factor": 0.5},
            ),
            (
                "Do not charge the battery between 2 PM and 4 PM.",
                DIRECTIVE_NO_CHARGE,
                {"hours": [14, 15]},
            ),
            (
                "The battery must not discharge from 6 PM until 8 PM.",
                DIRECTIVE_NO_DISCHARGE,
                {"hours": [18, 19]},
            ),
            (
                "Keep at least 120 kWh in reserve from 6 PM until 9 PM.",
                DIRECTIVE_MIN_RESERVE,
                {"hours": [18, 19, 20], "minimum_energy_kwh": 120},
            ),
            (
                "Grid import must not exceed 155 kWh in any hour from 6 PM until 9 PM.",
                DIRECTIVE_MAX_GRID,
                {"hours": [18, 19, 20], "max_grid_kwh": 155},
            ),
        ],
    )
    def test_core_directives(self, note, kind, adjustment):
        scenario = make_scenario([note])
        entry = fallback.interpret_note(note, scenario)
        assert entry["directive_type"] == kind
        for key, value in adjustment.items():
            assert entry["structured_adjustment"][key] == pytest.approx(value)

    def test_relative_reserve_resolves_against_capacity(self):
        note = "Keep at least 50% of battery capacity stored from 6 PM until 9 PM."
        scenario = make_scenario([note], capacity=200.0)
        entry = fallback.interpret_note(note, scenario)
        assert entry["directive_type"] == DIRECTIVE_MIN_RESERVE
        assert entry["structured_adjustment"]["minimum_energy_kwh"] == pytest.approx(100)

    @pytest.mark.parametrize(
        "note",
        [
            "The cafeteria menu changes tomorrow.",
            "The sports office moved next month's registration deadline.",
            "The library is extending book-return hours next week.",
            "A seminar room booking was moved to next week.",
            "The student affairs office will publish club notices tomorrow.",
        ],
    )
    def test_distractors_are_no_op(self, note):
        scenario = make_scenario([note])
        entry = fallback.interpret_note(note, scenario)
        assert entry["directive_type"] == DIRECTIVE_NO_OP
        assert entry["applies"] is False
        assert entry["structured_adjustment"] is None

    def test_note_without_a_window_is_no_op(self):
        note = "Solar output may be lower than usual at some point."
        scenario = make_scenario([note])
        assert fallback.interpret_note(note, scenario)["directive_type"] == DIRECTIVE_NO_OP

    def test_spelled_out_quantity(self):
        note = "Between 17:00 and 20:00 the bank may not be drawn below one hundred and twenty kilowatt-hours."
        scenario = make_scenario([note], capacity=400)
        entry = fallback.interpret_note(note, scenario)
        assert entry["directive_type"] == DIRECTIVE_MIN_RESERVE
        assert entry["structured_adjustment"]["minimum_energy_kwh"] == pytest.approx(120)

    def test_every_note_gets_exactly_one_entry(self):
        notes = ["Do not charge from 2 AM until 4 AM.", "Menu changes tomorrow.", "x"]
        scenario = make_scenario(notes)
        entries = fallback.interpret_notes(scenario)
        assert [e["note_index"] for e in entries] == [0, 1, 2]


class TestCache:
    def test_repeat_scenarios_reuse_the_first_answer(self, monkeypatch):
        scenario = make_scenario(["Do not charge from 2 AM until 4 AM."])
        calls = {"count": 0}
        payload = [
            {
                "note_index": 0,
                "applies": True,
                "directive_type": DIRECTIVE_NO_CHARGE,
                "structured_adjustment": {"hours": [2, 3]},
                "explanation": "x",
            }
        ]

        def fake_call(model, body, headers, timeout):
            calls["count"] += 1
            return payload, "", 200

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(llm, "_call_model", fake_call)
        llm._CACHE.clear()

        first, meta_first = llm.interpret_notes(scenario)
        second, meta_second = llm.interpret_notes(scenario)

        assert calls["count"] == 1
        assert meta_first["cached"] is False
        assert meta_second["cached"] is True
        assert first == second

    def test_cache_returns_a_copy(self, monkeypatch):
        scenario = make_scenario(["Do not charge from 3 AM until 5 AM."])

        def fake_call(model, body, headers, timeout):
            return [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": DIRECTIVE_NO_CHARGE,
                    "structured_adjustment": {"hours": [3, 4]},
                    "explanation": "x",
                }
            ], "", 200

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(llm, "_call_model", fake_call)
        llm._CACHE.clear()

        first, _ = llm.interpret_notes(scenario)
        first[0]["structured_adjustment"]["hours"].append(99)
        second, _ = llm.interpret_notes(scenario)
        assert second[0]["structured_adjustment"]["hours"] == [3, 4]


class TestCascade:
    def test_quota_error_moves_to_the_next_model(self, monkeypatch):
        scenario = make_scenario(["Do not charge from 1 AM until 2 AM."])
        seen = []

        def fake_call(model, body, headers, timeout):
            seen.append(model)
            if len(seen) == 1:
                return None, "http 429", 429
            return [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": DIRECTIVE_NO_CHARGE,
                    "structured_adjustment": {"hours": [1]},
                    "explanation": "x",
                }
            ], "", 200

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(llm, "_call_model", fake_call)
        llm._CACHE.clear()

        entries, meta = llm.interpret_notes(scenario)
        assert len(seen) == 2 and seen[0] != seen[1]
        assert meta["model"] == seen[1]
        assert entries

    def test_total_failure_raises_unavailable(self, monkeypatch):
        scenario = make_scenario(["Do not charge from 1 AM until 2 AM."])
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(
            llm, "_call_model", lambda *a, **k: (None, "http 503", 503)
        )
        llm._CACHE.clear()
        with pytest.raises(llm.LLMUnavailable):
            llm.interpret_notes(scenario)

    def test_missing_key_raises_immediately(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "")
        with pytest.raises(llm.LLMUnavailable):
            llm.interpret_notes(make_scenario(["x"]))
