#!/usr/bin/env python3
"""Public sample-case validation harness.

Runs every case in the public pack against a live service and reports, per
case: directive-interpretation match against the published reference, an
independent replay of the returned schedule, and cost quality versus the
reference optimum (the same ratio the rubric uses).

Usage:
    python3 harness/run_samples.py                     # http://127.0.0.1:8000
    python3 harness/run_samples.py --base-url https://your-host
    python3 harness/run_samples.py --case SAMPLE-03 --verbose
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizer.guardrails import build_constraints  # noqa: E402
from optimizer.replay import replay, verify_totals  # noqa: E402
from optimizer.schema import (  # noqa: E402
    DIRECTIVE_NO_OP,
    JUDGE_TOLERANCE,
    Battery,
    Directive,
    HourInput,
    Scenario,
)

DEFAULT_CASES = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "public_sample_cases.json"
)

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"


def scenario_from_input(payload):
    hours = sorted(payload["hours"], key=lambda e: e["hour"])
    return Scenario(
        scenario_id=payload["scenario_id"],
        operator_notes=list(payload["operator_notes"]),
        hours=[
            HourInput(
                hour=e["hour"],
                demand_kwh=float(e["demand_kwh"]),
                solar_kwh=float(e["solar_kwh"]),
                tariff_bdt_per_kwh=float(e["tariff_bdt_per_kwh"]),
            )
            for e in hours
        ],
        battery=Battery(**{k: float(v) for k, v in payload["battery"].items()}),
    )


def directives_from_response(entries):
    out = []
    for entry in entries:
        out.append(
            Directive(
                note_index=entry.get("note_index", 0),
                applies=bool(entry.get("applies")),
                directive_type=entry.get("directive_type", DIRECTIVE_NO_OP),
                structured_adjustment=entry.get("structured_adjustment"),
                explanation=entry.get("explanation", ""),
            )
        )
    return out


def compare_adjustment(expected, actual):
    """Numeric-tolerant comparison of two structured_adjustment payloads."""
    if expected is None and actual is None:
        return True, ""
    if expected is None or actual is None:
        return False, "one adjustment is null"

    exp_hours = list(expected.get("hours") or [])
    act_hours = list(actual.get("hours") or [])
    if exp_hours != act_hours:
        return False, "hours %s != expected %s" % (act_hours, exp_hours)

    for key in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
        if key in expected:
            if key not in actual:
                return False, "missing %s" % key
            if abs(float(expected[key]) - float(actual[key])) > JUDGE_TOLERANCE:
                return False, "%s %s != expected %s" % (key, actual[key], expected[key])
    return True, ""


def check_interpretation(expected_entries, actual_entries):
    """Score the interpretation exactly the way Section 11.1 describes."""
    problems = []
    if len(actual_entries) != len(expected_entries):
        problems.append(
            "expected %d entries, got %d" % (len(expected_entries), len(actual_entries))
        )
        return problems

    indices = [entry.get("note_index") for entry in actual_entries]
    if indices != list(range(len(expected_entries))):
        problems.append("note_index order %s is not 0..N-1" % indices)

    for expected, actual in zip(expected_entries, actual_entries):
        note = expected.get("note_index")
        exp_type = expected.get("directive_type")
        act_type = actual.get("directive_type")
        if exp_type != act_type:
            problems.append("note %s: type %s != expected %s" % (note, act_type, exp_type))
            continue
        if bool(expected.get("applies")) != bool(actual.get("applies")):
            problems.append("note %s: applies mismatch" % note)
        if act_type == DIRECTIVE_NO_OP:
            if actual.get("structured_adjustment") is not None:
                problems.append("note %s: no_op must have null adjustment" % note)
            if actual.get("applies"):
                problems.append("note %s: no_op must have applies=false" % note)
            continue
        ok, detail = compare_adjustment(
            expected.get("structured_adjustment"), actual.get("structured_adjustment")
        )
        if not ok:
            problems.append("note %s: %s" % (note, detail))
        if not isinstance(actual.get("explanation"), str) or not actual["explanation"]:
            problems.append("note %s: missing explanation" % note)
    return problems


def run(base_url, cases_path, only=None, verbose=False, timeout=45):
    pack = json.loads(Path(cases_path).read_text())
    cases = pack["cases"]
    if only:
        wanted = {c.strip().upper() for c in only}
        cases = [c for c in cases if c["id"].upper() in wanted]
        if not cases:
            print("no matching cases")
            return 1

    health = requests.get(base_url.rstrip("/") + "/health", timeout=timeout)
    print(
        "health: %s %s"
        % (health.status_code, json.dumps(health.json(), separators=(",", ":")))
    )

    passed = 0
    ratios = []
    latencies = []
    failures = []

    for case in cases:
        scenario = scenario_from_input(case["input"])
        expected = case["expected_output"]

        started = time.time()
        response = requests.post(
            base_url.rstrip("/") + "/optimize-energy",
            json=case["input"],
            timeout=timeout,
        )
        elapsed = time.time() - started
        latencies.append(elapsed)

        label = "%-10s %-34s" % (case["id"], case["label"][:34])
        if response.status_code != 200:
            failures.append((case["id"], ["http %d: %s" % (response.status_code, response.text[:200])]))
            print("%s %sFAIL%s  http %d  %.2fs" % (label, RED, RESET, response.status_code, elapsed))
            continue

        body = response.json()
        problems = []

        if body.get("scenario_id") != scenario.scenario_id:
            problems.append("scenario_id not echoed")

        problems.extend(
            check_interpretation(
                expected["directive_interpretation"],
                body.get("directive_interpretation") or [],
            )
        )

        plan = body.get("hourly_plan") or []
        # Replay against OUR OWN interpretation-independent ground truth: the
        # reference directives from the pack. This is what the judge does.
        reference_directives = directives_from_response(
            expected["directive_interpretation"]
        )
        constraints = build_constraints(scenario, reference_directives)
        ok, violations = replay(scenario, constraints, plan)
        if not ok:
            problems.extend(violations)
        problems.extend(verify_totals(scenario, plan, body))

        reference_cost = float(expected["total_cost_bdt"])
        team_cost = float(body.get("total_cost_bdt", 0) or 0)
        ratio = 1.0
        if not problems:
            if team_cost <= JUDGE_TOLERANCE and reference_cost <= JUDGE_TOLERANCE:
                ratio = 1.0
            elif team_cost > 0:
                ratio = min(1.0, reference_cost / team_cost)
            else:
                ratio = 1.0
            ratios.append(ratio)

        if problems:
            failures.append((case["id"], problems))
            print("%s %sFAIL%s  %.2fs" % (label, RED, RESET, elapsed))
            for problem in problems[:6]:
                print("    %s- %s%s" % (DIM, problem, RESET))
            if len(problems) > 6:
                print("    %s- ... %d more%s" % (DIM, len(problems) - 6, RESET))
        else:
            passed += 1
            colour = GREEN if ratio >= 0.999 else YELLOW
            print(
                "%s %sPASS%s  cost %9.2f vs ref %9.2f  quality %s%.4f%s  %.2fs"
                % (
                    label,
                    GREEN,
                    RESET,
                    team_cost,
                    reference_cost,
                    colour,
                    ratio,
                    RESET,
                    elapsed,
                )
            )
        if verbose:
            print(
                json.dumps(body.get("directive_interpretation"), indent=2)
            )
            print("    summary: %s" % body.get("plan_summary"))

    total = len(cases)
    print("\n%d/%d cases passed" % (passed, total))
    if ratios:
        print("mean cost quality ratio: %.4f" % (sum(ratios) / len(ratios)))
    if latencies:
        ordered = sorted(latencies)
        index = max(0, int(round(0.95 * (len(ordered) - 1))))
        print(
            "latency: mean %.2fs  p95 %.2fs  max %.2fs"
            % (sum(ordered) / len(ordered), ordered[index], ordered[-1])
        )
    if failures:
        print("\nfailed cases: %s" % ", ".join(case_id for case_id, _ in failures))
    return 0 if passed == total else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("GRIDWISE_BASE_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--case", action="append", dest="only")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--timeout", type=float, default=45)
    args = parser.parse_args()
    sys.exit(
        run(args.base_url, args.cases, args.only, args.verbose, args.timeout)
    )


if __name__ == "__main__":
    main()
