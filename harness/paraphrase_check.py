#!/usr/bin/env python3
"""Paraphrase-robustness probe (Problem Statement Section 11.4).

The public pack teaches the contract; the hidden set rewords it. This probe
feeds the service operator notes written in deliberately different registers
-- 24-hour clock, spelled-out numbers, fractions, passive maintenance
phrasing, percentage-reduction vs percentage-remaining, and distractors that
merely *sound* electrical -- and checks the structured directive that comes
back.

Notes here are intentionally unlike the published samples. Usage:
    python3 harness/paraphrase_check.py [--base-url URL]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.run_samples import compare_adjustment  # noqa: E402

BATTERY = {
    "capacity_kwh": 400,
    "initial_energy_kwh": 160,
    "minimum_energy_kwh": 40,
    "max_charge_kwh_per_hour": 80,
    "max_discharge_kwh_per_hour": 80,
}

HOURS = [
    {
        "hour": h,
        "demand_kwh": 120 + 70 * (1 if 8 <= h <= 21 else 0),
        "solar_kwh": [0, 0, 0, 0, 0, 0, 8, 30, 70, 110, 150, 180, 195, 185, 150, 100,
                      55, 15, 0, 0, 0, 0, 0, 0][h],
        "tariff_bdt_per_kwh": [6, 6, 5, 5, 5, 6, 8, 10, 12, 14, 16, 16, 15, 14, 13, 14,
                               18, 22, 28, 30, 26, 20, 12, 8][h],
    }
    for h in range(24)
]

# (note, expected_directive_type, expected_structured_adjustment)
PROBES = [
    # -- solar_reduction, varied quantification --------------------------
    (
        "Between 09:00 and 12:00 the array will be derated to one third of forecast while technicians re-torque the mounts.",
        "solar_reduction",
        {"hours": [9, 10, 11], "factor": 1.0 / 3.0},
    ),
    (
        "Heavy haze is expected to knock sixty percent off photovoltaic yield from ten in the morning to one in the afternoon.",
        "solar_reduction",
        {"hours": [10, 11, 12], "factor": 0.4},
    ),
    (
        "Inverter number two stays offline between 13:00 and 16:00, so treat rooftop generation as nil for that stretch.",
        "solar_reduction",
        {"hours": [13, 14, 15], "factor": 0.0},
    ),
    # -- minimum_battery_reserve, absolute and relative ------------------
    (
        "Between 17:00 and 20:00 the storage bank may not be drawn below one hundred and twenty kilowatt-hours.",
        "minimum_battery_reserve",
        {"hours": [17, 18, 19], "minimum_energy_kwh": 120},
    ),
    (
        "Critical care load requires a quarter of the pack to stay in reserve from 8 PM through to 11 PM.",
        "minimum_battery_reserve",
        {"hours": [20, 21, 22], "minimum_energy_kwh": 100},
    ),
    # -- no_charge_window ------------------------------------------------
    (
        "Rectifier cubicle is racked out for testing between 03:00 and 06:00; no energy may be pushed into the bank then.",
        "no_charge_window",
        {"hours": [3, 4, 5]},
    ),
    (
        "Please refrain from topping up the battery from four in the morning until seven.",
        "no_charge_window",
        {"hours": [4, 5, 6]},
    ),
    # -- no_discharge_window ---------------------------------------------
    (
        "While the protection scheme is being proven from 16:00 to 18:00, the bank must stay off-load and export nothing.",
        "no_discharge_window",
        {"hours": [16, 17]},
    ),
    (
        "Hold the battery back from supplying load between seven and nine in the evening for the relay trial.",
        "no_discharge_window",
        {"hours": [19, 20]},
    ),
    # -- max_grid_window --------------------------------------------------
    (
        "Utility has asked that we stay at or beneath 210 kWh of intake per hour from 18:00 until 21:00.",
        "max_grid_window",
        {"hours": [18, 19, 20], "max_grid_kwh": 210},
    ),
    (
        "Incoming supply is throttled to a ceiling of 175 kilowatt-hours each hour between 9 PM and midnight.",
        "max_grid_window",
        {"hours": [21, 22, 23], "max_grid_kwh": 175},
    ),
    # -- no_op, including electrical-sounding distractors ----------------
    (
        "The estates team will repaint the substation fence railings sometime next quarter.",
        "no_op",
        None,
    ),
    (
        "Reminder: quarterly electrical safety training for facilities staff has moved to the second Thursday of next month.",
        "no_op",
        None,
    ),
    (
        "A generator load-bank test is being planned for the Eid holiday shutdown; details to follow.",
        "no_op",
        None,
    ),
    (
        "Hostel residents have asked for longer laundry room hours starting next week.",
        "no_op",
        None,
    ),
]


def build_scenario(scenario_id, notes):
    return {
        "scenario_id": scenario_id,
        "operator_notes": notes,
        "hours": HOURS,
        "battery": BATTERY,
    }


def run(base_url, timeout=45, batch_size=3):
    url = base_url.rstrip("/") + "/optimize-energy"
    passed = 0
    failures = []

    for start in range(0, len(PROBES), batch_size):
        batch = PROBES[start : start + batch_size]
        payload = build_scenario(
            "PARAPHRASE-%02d" % (start // batch_size + 1), [p[0] for p in batch]
        )
        response = requests.post(url, json=payload, timeout=timeout)
        if response.status_code != 200:
            for note, _, _ in batch:
                failures.append((note, "http %d" % response.status_code))
            continue

        body = response.json()
        entries = body.get("directive_interpretation") or []
        for offset, (note, expected_type, expected_adjustment) in enumerate(batch):
            entry = entries[offset] if offset < len(entries) else {}
            actual_type = entry.get("directive_type")
            problems = []
            if actual_type != expected_type:
                problems.append("type %s != %s" % (actual_type, expected_type))
            else:
                ok, detail = compare_adjustment(
                    expected_adjustment, entry.get("structured_adjustment")
                )
                if not ok:
                    problems.append(detail)
            if problems:
                failures.append((note, "; ".join(problems)))
                print("FAIL  %-22s %s" % (expected_type, note[:78]))
                print("        %s" % "; ".join(problems))
            else:
                passed += 1
                print("pass  %-22s %s" % (expected_type, note[:78]))

    print("\n%d/%d paraphrase probes passed" % (passed, len(PROBES)))
    return 0 if passed == len(PROBES) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("GRIDWISE_BASE_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument("--timeout", type=float, default=45)
    args = parser.parse_args()
    sys.exit(run(args.base_url, args.timeout))


if __name__ == "__main__":
    main()
