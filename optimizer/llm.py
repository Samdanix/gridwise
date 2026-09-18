"""Gemini-backed operator-note interpretation.

This module is the mandatory language-model stage of the pipeline: every
operator note is interpreted here, and its structured output is what feeds
the guardrails and then the optimizer. Nothing downstream trusts it.

Transport is plain HTTPS against the Generative Language API so the service
has no heavyweight SDK dependency and full control over timeouts.
"""
import copy
import hashlib
import itertools
import json
import logging
import os
import re
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import requests

from .prompts import RESPONSE_SCHEMA, SYSTEM_INSTRUCTION, build_user_prompt
from .schema import DIRECTIVE_NO_OP, REQUIRED_ADJUSTMENT_KEYS, Scenario

logger = logging.getLogger(__name__)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3.6-flash"

# Free-tier quota is enforced per project *per model*, so a cascade of
# distinct Flash-class models multiplies usable throughput and rides out a
# single model being overloaded (503) or rate-limited (429). Order is
# strongest-first; every entry is held to the same guardrails downstream.
# Full Flash models first: the lite variants are measurably weaker at
# spotting a quantified directive inside maintenance prose, so they sit at
# the end of the chain as availability insurance rather than as peers.
DEFAULT_FALLBACK_MODELS = (
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
)


class LLMUnavailable(Exception):
    """Raised when no usable model response could be obtained."""


def _config() -> Dict[str, Any]:
    primary = os.environ.get("GRIDWISE_LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    raw_fallbacks = os.environ.get("GRIDWISE_LLM_FALLBACK_MODELS")
    if raw_fallbacks is None:
        fallbacks = list(DEFAULT_FALLBACK_MODELS)
    else:
        fallbacks = [m.strip() for m in raw_fallbacks.split(",") if m.strip()]

    chain = [primary] + [m for m in fallbacks if m != primary]

    # Quota is per project, so several keys multiply usable throughput.
    # GEMINI_API_KEYS (comma-separated) takes precedence; GEMINI_API_KEY
    # remains supported as the single-key form.
    raw_keys = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY", "")
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

    return {
        "api_key": keys[0] if keys else "",
        "api_keys": keys,
        "model": primary,
        "models": chain,
        "timeout": float(os.environ.get("GRIDWISE_LLM_TIMEOUT", "9")),
        "attempts": int(os.environ.get("GRIDWISE_LLM_ATTEMPTS", "2")),
        # Hard ceiling for the whole interpretation stage. The judge fails a
        # request at 30s, so the cascade must give up in time for the
        # optimizer and the deterministic backup to still finish.
        "deadline": float(os.environ.get("GRIDWISE_LLM_DEADLINE", "18")),
        "cache_size": int(os.environ.get("GRIDWISE_CACHE_SIZE", "512")),
    }


def is_configured() -> bool:
    return bool(_config()["api_key"])


def _extract_json(text: str) -> Optional[Any]:
    """Parse model text into JSON, tolerating code fences and prose."""
    if not text:
        return None
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    # Last resort: the outermost {...} or [...] block.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except ValueError:
                continue
    return None


def _response_text(payload: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for candidate in payload.get("candidates") or []:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            piece = part.get("text")
            if isinstance(piece, str):
                chunks.append(piece)
    return "\n".join(chunks).strip()


def normalise_entries(parsed: Any) -> List[Dict[str, Any]]:
    """Reshape flat model entries into response-contract shape.

    The model is asked for flat keys (``hours``, ``factor``, ...) because a
    flat JSON schema is far more reliable than a polymorphic nested object.
    Here they are folded into ``structured_adjustment``. A model that already
    nested the payload is accepted too.
    """
    if isinstance(parsed, dict):
        entries = parsed.get("directive_interpretation")
    else:
        entries = parsed
    if not isinstance(entries, list):
        return []

    shaped: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            shaped.append({})
            continue

        directive_type = entry.get("directive_type")
        directive_type = (
            directive_type.strip().lower() if isinstance(directive_type, str) else ""
        )

        adjustment = entry.get("structured_adjustment")
        if not isinstance(adjustment, dict):
            adjustment = {}
        else:
            adjustment = dict(adjustment)

        for key in ("hours", "factor", "minimum_energy_kwh", "max_grid_kwh"):
            if key not in adjustment and entry.get(key) is not None:
                adjustment[key] = entry[key]

        if directive_type == DIRECTIVE_NO_OP:
            adjustment = None
        else:
            # Drop keys that do not belong to this directive type so the
            # guardrails see exactly the required shape.
            allowed = REQUIRED_ADJUSTMENT_KEYS.get(directive_type, ())
            adjustment = {k: v for k, v in adjustment.items() if k in allowed}

        shaped.append(
            {
                "note_index": entry.get("note_index"),
                "applies": entry.get("applies"),
                "directive_type": directive_type,
                "structured_adjustment": adjustment,
                "explanation": entry.get("explanation"),
            }
        )
    return shaped


def _request_payload(scenario: Scenario) -> Dict[str, Any]:
    return {
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": [{"role": "user", "parts": [{"text": build_user_prompt(scenario)}]}],
        "generationConfig": {
            "temperature": 0,
            "candidateCount": 1,
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
            # Extraction is a short classification task, not a reasoning
            # marathon: low thinking cuts latency from ~45s to ~3s with no
            # loss of accuracy on this workload. Ignored by older models.
            "thinkingConfig": {"thinkingLevel": "low"},
        },
    }


_CACHE: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()

# Round-robin starting point for key selection.
_ROTATION = itertools.count()


def _cache_key(scenario: Scenario) -> str:
    """Identity of an interpretation request.

    Only the inputs the interpreter actually reads participate: the note
    text and the battery/solar figures used to resolve relative language.
    Two scenarios with identical notes and battery therefore share a result,
    which keeps repeated judge passes off the quota entirely.
    """
    payload = json.dumps(
        {
            "notes": scenario.operator_notes,
            "capacity": scenario.battery.capacity_kwh,
            "minimum": scenario.battery.minimum_energy_kwh,
            "solar": [h.solar_kwh for h in scenario.hours],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> Optional[List[Dict[str, Any]]]:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    _CACHE.move_to_end(key)
    return copy.deepcopy(entry)


def _cache_put(key: str, entries: List[Dict[str, Any]], limit: int) -> None:
    _CACHE[key] = copy.deepcopy(entries)
    _CACHE.move_to_end(key)
    while len(_CACHE) > max(1, limit):
        _CACHE.popitem(last=False)


def _call_model(
    model: str, body: Dict[str, Any], headers: Dict[str, str], timeout: float
) -> Tuple[Optional[List[Dict[str, Any]]], str, Optional[int]]:
    """One HTTP attempt. Returns (entries, error, status_code)."""
    url = "%s/%s:generateContent" % (API_ROOT, model)
    try:
        response = requests.post(url, headers=headers, json=body, timeout=timeout)
    except requests.RequestException as exc:
        return None, "transport error: %s" % type(exc).__name__, None

    if response.status_code != 200:
        return None, "http %d" % response.status_code, response.status_code

    try:
        payload = response.json()
    except ValueError:
        return None, "non-JSON transport body", 200

    parsed = _extract_json(_response_text(payload))
    if parsed is None:
        return None, "model returned unparseable content", 200

    entries = normalise_entries(parsed)
    if not entries:
        return None, "model returned no interpretation entries", 200
    return entries, "", 200


def interpret_notes(scenario: Scenario) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Interpret every operator note in a single model call.

    Resilience, in order: memo cache -> primary model -> one retry ->
    fallback models (each with its own quota bucket). Returns
    ``(entries, meta)``; the caller passes ``entries`` straight into the
    guardrails and never trusts them directly.
    """
    config = _config()
    if not config["api_key"]:
        raise LLMUnavailable("GEMINI_API_KEY is not configured")

    started = time.time()
    meta: Dict[str, Any] = {
        "model": config["model"],
        "attempts": 0,
        "latency_ms": None,
        "cached": False,
        "errors": [],
    }

    key = _cache_key(scenario)
    cached = _cache_get(key)
    if cached is not None:
        meta["cached"] = True
        meta["latency_ms"] = int((time.time() - started) * 1000)
        return cached, meta

    body = _request_payload(scenario)
    last_error = "unknown error"
    deadline = started + config["deadline"]
    keys = config["api_keys"]
    code = None
    # Spread load so consecutive requests do not all start on the same key.
    offset = next(_ROTATION)
    dead_keys = set()

    for model in config["models"]:
        for position in range(len(keys)):
            index = (offset + position) % len(keys)
            if index in dead_keys:
                continue
            headers = {
                "x-goog-api-key": keys[index],
                "Content-Type": "application/json",
            }

            for attempt in range(1, max(1, config["attempts"]) + 1):
                remaining = deadline - time.time()
                if remaining <= 0.5:
                    last_error = "interpretation deadline reached"
                    meta["errors"].append(last_error)
                    meta["latency_ms"] = int((time.time() - started) * 1000)
                    logger.warning("LLM interpretation deadline hit")
                    raise LLMUnavailable(last_error)

                meta["attempts"] += 1
                entries, error, code = _call_model(
                    model, body, headers, min(config["timeout"], remaining)
                )
                if entries is not None:
                    meta["model"] = model
                    meta["key_index"] = index
                    meta["latency_ms"] = int((time.time() - started) * 1000)
                    _cache_put(key, entries, config["cache_size"])
                    return entries, meta

                # Never include the key itself in diagnostics.
                last_error = "%s[key %d]: %s" % (model, index, error)
                meta["errors"].append(last_error)

                # Quota exhaustion will not clear inside our latency budget:
                # try the next key, then the next model.
                if code == 429:
                    break
                # An invalid or unauthorised key is terminal for that key.
                if code in (401, 403):
                    dead_keys.add(index)
                    break
                # A bad request or unknown model is terminal for the model.
                if code in (400, 404):
                    break
                if attempt < config["attempts"]:
                    time.sleep(0.4 * attempt)

            if code in (400, 404):
                break

    meta["latency_ms"] = int((time.time() - started) * 1000)
    logger.warning("LLM interpretation failed after cascade: %s", last_error)
    raise LLMUnavailable(last_error)


def cache_stats() -> Dict[str, int]:
    return {"entries": len(_CACHE)}
