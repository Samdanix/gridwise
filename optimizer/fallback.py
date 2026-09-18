"""Deterministic backup interpreter.

IMPORTANT SCOPE: this is NOT the primary interpretation path. The language
model in ``llm.py`` interprets every operator note in normal operation, and
that is the path whose output reaches the optimizer. This module runs only
when the whole model cascade is unavailable (provider outage, exhausted
quota, network failure), where the alternative would be marking every note
``no_op`` and silently dropping real operating constraints.

It is a safety net, not a shortcut: it matches structural language patterns
(time expressions, quantities, constraint vocabulary) rather than any
specific published note wording, and its output goes through exactly the
same guardrails as the model's.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from .schema import (
    DIRECTIVE_MAX_GRID,
    DIRECTIVE_MIN_RESERVE,
    DIRECTIVE_NO_CHARGE,
    DIRECTIVE_NO_DISCHARGE,
    DIRECTIVE_NO_OP,
    DIRECTIVE_SOLAR_REDUCTION,
    HORIZON,
    Scenario,
)

WORD_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "midnight": 0,
    "noon": 12,
    "midday": 12,
}

PERCENT_WORDS = {
    "ten": 10,
    "fifteen": 15,
    "twenty": 20,
    "twenty five": 25,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "seventy five": 75,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}

# Phrases that fix an otherwise ambiguous clock reading to am or pm.
MERIDIEM_PHRASES = (
    ("in the morning", "am"),
    ("this morning", "am"),
    ("before noon", "am"),
    ("in the afternoon", "pm"),
    ("in the evening", "pm"),
    ("this evening", "pm"),
    ("at night", "pm"),
    ("tonight", "pm"),
    ("overnight", "am"),
)

FRACTION_WORDS = {
    "half": 0.5,
    "a half": 0.5,
    "one half": 0.5,
    "a third": 1.0 / 3.0,
    "one third": 1.0 / 3.0,
    "two thirds": 2.0 / 3.0,
    "a quarter": 0.25,
    "one quarter": 0.25,
    "three quarters": 0.75,
    "a fifth": 0.2,
    "one fifth": 0.2,
    "a tenth": 0.1,
    "one tenth": 0.1,
}

SOLAR_WORDS = (
    "solar",
    "pv",
    "photovoltaic",
    "panel",
    "rooftop",
    "inverter",
    "array",
    "irradiance",
    "generation",
    "yield",
    "string",
)
REDUCTION_WORDS = (
    "reduc",
    "drop",
    "derate",
    "curtail",
    "cloud",
    "overcast",
    "haze",
    "wash",
    "clean",
    "shade",
    "dust",
    "soil",
    "offline",
    "out of service",
    "nil",
    "knock",
    "off forecast",
    "treat",
    "inspect",
    "maintenance",
    "output will",
    "only",
    "limited to",
)
CHARGE_WORDS = (
    "charg",
    "charger",
    "top up",
    "topping up",
    "topped up",
    "rectifier",
    "pushed into the bank",
    "energy into the",
    "into the battery",
    "replenish",
)
DISCHARGE_WORDS = (
    "discharg",
    "off-load",
    "off load",
    "supplying load",
    "supply load",
    "drawn from the battery",
    "draw from the battery",
    "battery back",
    "export",
)
BLOCK_WORDS = (
    "not ",
    "no ",
    "never",
    "avoid",
    "disable",
    "unavailable",
    "isolat",
    "offline",
    "prohibit",
    "forbid",
    "inhibit",
    "suspend",
    "block",
    "must not",
    "cannot",
    "do not",
    "don't",
    "halt",
    "stop",
    "lock",
    "refrain",
    "hold",
    "racked out",
    "may not",
    "will not",
    "withheld",
    "nothing",
    "off-load",
    "off load",
    "out of service",
)
# Phrases that state a floor on stored energy. "may not be drawn below X"
# reads as a prohibition, so these are checked before the block vocabulary.
RESERVE_FLOOR_WORDS = (
    "below",
    "beneath",
    "under",
    "lower than",
    "less than",
)
BATTERY_WORDS = (
    "battery",
    "storage",
    "bank",
    "pack",
    "bess",
    "accumulator",
    "stored",
)
RESERVE_WORDS = (
    "reserve",
    "at least",
    "minimum",
    "no lower than",
    "not fall below",
    "keep",
    "maintain",
    "retain",
    "remain in the battery",
    "stored in the battery",
    "floor",
)
GRID_WORDS = (
    "grid",
    "import",
    "intake",
    "feeder",
    "transformer",
    "substation",
    "utility",
    "incoming supply",
    "mains",
    "purchased",
)
CAP_WORDS = (
    "not exceed",
    "no more than",
    "at or below",
    "at or beneath",
    "beneath",
    "cap",
    "limit",
    "throttl",
    "restrict",
    "maximum",
    "max ",
    "ceiling",
    "up to",
    "stay below",
    "stay at or",
    "under",
)
IRRELEVANT_WORDS = (
    "menu",
    "cafeteria",
    "registration",
    "deadline",
    "library",
    "book",
    "seminar",
    "booking",
    "notice",
    "club",
    "sports",
    "exam",
    "holiday",
    "newsletter",
    "announcement",
    "next week",
    "next month",
    "tomorrow",
)


def _to_hour(value: float, meridiem: Optional[str], reference: Optional[str]) -> int:
    """Convert a clock reading into a 0-23 hour."""
    hour = int(value) % 24
    tag = (meridiem or reference or "").lower()
    if "pm" in tag:
        if hour < 12:
            hour += 12
    elif "am" in tag:
        if hour == 12:
            hour = 0
    return hour % 24


def _numeric_tokens(text: str) -> List[Tuple[int, float, Optional[str]]]:
    """Every clock-like token with its position and explicit meridiem."""
    tokens: List[Tuple[int, float, Optional[str]]] = []
    pattern = re.compile(
        r"(?P<num>\d{1,2})(?::(?P<min>\d{2}))?\s*(?P<mer>a\.?m\.?|p\.?m\.?)?",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        value = float(match.group("num"))
        if value > 24:
            continue
        meridiem = match.group("mer")
        meridiem = meridiem.replace(".", "").lower() if meridiem else None
        tokens.append((match.start(), value, meridiem))

    for word, value in WORD_NUMBERS.items():
        for match in re.finditer(r"\b%s\b" % word, text, re.IGNORECASE):
            fixed = word in ("midnight", "noon", "midday")
            tokens.append((match.start(), float(value), "fixed" if fixed else None))

    tokens.sort(key=lambda item: item[0])
    return tokens


CLOCK_SIDE = (
    r"(?:\d{1,2}(?::\d{2})?|midnight|noon|midday|"
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r"(?:\s*(?:o'clock|hours|hrs))?"
    r"(?:\s*(?:a\.?m\.?|p\.?m\.?))?"
)

# "through to", "until about", "up until" all connect two clock readings.
RANGE_CONNECTOR = (
    r"(?:-|--|–|to|until|til|till|through|thru|and)"
    r"(?:\s+(?:to|about|around|approximately))?"
)


def normalise_time_prose(text: str) -> str:
    """Rewrite prose meridiem cues into compact am/pm markers.

    "ten in the morning to one in the afternoon" becomes "ten am to one pm",
    which both fixes the clock reading and removes the filler words that
    would otherwise sit between the two ends of the range.
    """
    lowered = text.lower()
    for phrase, tag in MERIDIEM_PHRASES:
        lowered = lowered.replace(phrase, " " + tag)
    return re.sub(r"\s+", " ", lowered)


def extract_hours(text: str) -> List[int]:
    """Parse a whole-hour window, start inclusive and end exclusive."""
    lowered = normalise_time_prose(text)
    tokens = _numeric_tokens(lowered)
    if not tokens:
        return []

    # Prefer an explicit range connector between two clock readings.
    range_pattern = re.compile(
        r"(?:from|between|starting|during)?\s*"
        r"(?P<a>%s)\s*%s\s*(?P<b>%s)" % (CLOCK_SIDE, RANGE_CONNECTOR, CLOCK_SIDE),
        re.IGNORECASE,
    )

    def parse_side(raw: str, other: str) -> Optional[int]:
        raw = raw.strip()
        found = re.search(r"a\.?m\.?|p\.?m\.?", raw, re.IGNORECASE)
        meridiem = found.group(0).replace(".", "").lower() if found else None
        body = re.sub(r"\s*(?:a\.?m\.?|p\.?m\.?)\s*$", "", raw, flags=re.IGNORECASE)
        body = re.sub(r"\s*(?:o'clock|hours|hrs)\s*$", "", body).strip()

        if body == "midnight":
            return 0
        if body in ("noon", "midday"):
            return 12

        word = WORD_NUMBERS.get(body)
        if word is not None:
            value = float(word)
        else:
            match = re.match(r"(\d{1,2})(?::\d{2})?$", body)
            if not match:
                return None
            value = float(match.group(1))
            # A 24-hour reading such as 17:00 or 20 needs no meridiem.
            if value >= 13:
                return int(value) % 24
        return _to_hour(value, meridiem, other)

    for match in range_pattern.finditer(lowered):
        raw_a, raw_b = match.group("a"), match.group("b")
        # A meridiem stated only once applies to both ends of the range.
        shared = ""
        for candidate in (raw_a, raw_b):
            found = re.search(r"a\.?m\.?|p\.?m\.?", candidate, re.IGNORECASE)
            if found:
                shared = found.group(0).replace(".", "").lower()
        if not shared:
            # "from ten in the morning to one in the afternoon" carries its
            # meridiem in prose rather than in the clock reading itself.
            for phrase, tag in MERIDIEM_PHRASES:
                if phrase in lowered:
                    shared = tag
                    break
        start = parse_side(raw_a, shared)
        end = parse_side(raw_b, shared)
        if start is None or end is None:
            continue
        # "one until three" with no meridiem in an operations note means the
        # working afternoon, not 01:00-03:00.
        if not shared and start < 6 and end < 6 and start < end:
            start, end = start + 12, end + 12
        if end <= start:
            end += 12 if end + 12 > start else 24
            end = min(end, HORIZON)
        hours = [h for h in range(start, end) if 0 <= h < HORIZON]
        if hours:
            return hours

    # Single stated hour.
    position, value, meridiem = tokens[0]
    hour = _to_hour(value, meridiem, lowered)
    if re.search(r"\b(at|by|during)\b", lowered):
        return [hour]
    return []


REDUCTION_CONTEXT = (
    "reduc",
    "drop by",
    "drop of",
    "less",
    "loss",
    "lose",
    "cut",
    "down by",
    "knock",
    "off ",
    "shave",
    "decrease",
)


def _percentage(text: str) -> Optional[Tuple[float, bool]]:
    """Return ``(fraction, is_reduction)`` from percentage or fraction words."""
    match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent|per cent)", text)
    if match is None:
        # Spelled-out percentages: "sixty percent", "twenty five per cent".
        for phrase, value in sorted(PERCENT_WORDS.items(), key=lambda i: -len(i[0])):
            spelled = re.search(
                r"\b%s\b\s*(?:%%|percent|per cent)" % re.escape(phrase), text
            )
            if spelled:
                window = text[max(0, spelled.start() - 45) : spelled.end() + 45]
                is_reduction = any(word in window for word in REDUCTION_CONTEXT)
                return min(max(value / 100.0, 0.0), 1.0), is_reduction
    if match:
        value = float(match.group(1)) / 100.0
        window = text[max(0, match.start() - 45) : match.end() + 45]
        is_reduction = any(word in window for word in REDUCTION_CONTEXT)
        return min(max(value, 0.0), 1.0), is_reduction

    for phrase, value in sorted(
        FRACTION_WORDS.items(), key=lambda item: -len(item[0])
    ):
        if phrase in text:
            window = text[max(0, text.find(phrase) - 40) :]
            is_reduction = any(
                word in window for word in ("reduc", "loss", "less by", "cut by")
            )
            return value, is_reduction
    return None


SPELLED_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
SPELLED_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}


def _spelled_number(phrase: str) -> Optional[float]:
    """Parse "one hundred and twenty" style quantities."""
    total = 0.0
    current = 0.0
    seen = False
    for token in re.split(r"[\s-]+", phrase.strip()):
        token = token.strip(",")
        if not token or token == "and":
            continue
        if token in SPELLED_UNITS:
            current += SPELLED_UNITS[token]
            seen = True
        elif token in SPELLED_TENS:
            current += SPELLED_TENS[token]
            seen = True
        elif token == "hundred":
            current = (current or 1) * 100
            seen = True
        elif token == "thousand":
            total += (current or 1) * 1000
            current = 0.0
            seen = True
        else:
            return None
    return (total + current) if seen else None


UNIT_PATTERN = r"(?:kwh|kw h|kw-h|kilowatt[- ]?hours?|units?)"


def _kwh_value(text: str) -> Optional[float]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%s" % UNIT_PATTERN, text)
    if match:
        return float(match.group(1))
    spelled = re.search(
        r"((?:(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
        r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|and)"
        r"[\s-]*){1,8})%s" % UNIT_PATTERN,
        text,
    )
    if spelled:
        return _spelled_number(spelled.group(1))
    return None


def interpret_note(note: str, scenario: Scenario) -> Dict[str, Any]:
    """Best-effort structured directive for one note."""
    text = note.lower().strip()
    hours = extract_hours(text)
    capacity = scenario.battery.capacity_kwh

    has_solar = any(word in text for word in SOLAR_WORDS)
    has_charge = any(word in text for word in CHARGE_WORDS)
    has_discharge = any(word in text for word in DISCHARGE_WORDS)
    has_grid = any(word in text for word in GRID_WORDS)
    has_battery = any(word in text for word in BATTERY_WORDS)
    blocked = any(word in text for word in BLOCK_WORDS)
    # "may not be drawn below 120 kWh" is a floor, not an outage: a stated
    # quantity plus floor language outranks the prohibition vocabulary.
    states_floor = any(word in text for word in RESERVE_FLOOR_WORDS)

    def entry(directive_type: str, adjustment: Optional[Dict[str, Any]], why: str):
        return {
            "note_index": None,
            "applies": directive_type != DIRECTIVE_NO_OP,
            "directive_type": directive_type,
            "structured_adjustment": adjustment,
            "explanation": why,
        }

    if not hours:
        return entry(
            DIRECTIVE_NO_OP, None, "No operating window was stated in this note."
        )

    if any(word in text for word in IRRELEVANT_WORDS) and not (
        has_solar or has_charge or has_discharge or has_grid
    ):
        return entry(
            DIRECTIVE_NO_OP, None, "Administrative notice with no energy effect."
        )

    # Solar reduction.
    if has_solar and any(word in text for word in REDUCTION_WORDS):
        percentage = _percentage(text)
        factor = 0.0
        if percentage is not None:
            value, is_reduction = percentage
            factor = (1.0 - value) if is_reduction else value
        return entry(
            DIRECTIVE_SOLAR_REDUCTION,
            {"hours": hours, "factor": round(min(max(factor, 0.0), 1.0), 6)},
            "Usable solar is reduced during the stated window.",
        )

    # Grid import cap.
    if has_grid and any(word in text for word in CAP_WORDS):
        cap = _kwh_value(text)
        if cap is not None:
            return entry(
                DIRECTIVE_MAX_GRID,
                {"hours": hours, "max_grid_kwh": cap},
                "Grid import is capped during the stated window.",
            )

    # Battery reserve floor. Accepted when reserve language is present and
    # not contradicted by an outage, or when a floor is stated explicitly
    # against a quantity ("not be drawn below 120 kWh").
    reserve_language = any(word in text for word in RESERVE_WORDS)
    if (reserve_language or (has_battery and states_floor)) and not (
        has_charge or has_discharge
    ):
        reserve = _kwh_value(text)
        if reserve is None:
            percentage = _percentage(text)
            if percentage is not None and capacity:
                reserve = percentage[0] * capacity
        if reserve is not None:
            return entry(
                DIRECTIVE_MIN_RESERVE,
                {"hours": hours, "minimum_energy_kwh": round(reserve, 6)},
                "A minimum battery reserve applies during the stated window.",
            )

    # Charge / discharge outages.
    if has_discharge and blocked:
        return entry(
            DIRECTIVE_NO_DISCHARGE,
            {"hours": hours},
            "Battery discharging is unavailable during the stated window.",
        )
    if has_charge and blocked:
        return entry(
            DIRECTIVE_NO_CHARGE,
            {"hours": hours},
            "Battery charging is unavailable during the stated window.",
        )

    return entry(
        DIRECTIVE_NO_OP, None, "This note does not map to a supported directive."
    )


def interpret_notes(scenario: Scenario) -> List[Dict[str, Any]]:
    entries = []
    for index, note in enumerate(scenario.operator_notes):
        entry = interpret_note(note, scenario)
        entry["note_index"] = index
        entries.append(entry)
    return entries
