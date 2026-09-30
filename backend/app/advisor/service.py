"""Generating one advisory, and refusing to show a bad one.

The pipeline, and where it can stop
-----------------------------------
``facts -> prompt -> model -> parse -> validate -> advisory``, with a
deterministic advisory available at every point after the first. A failure never
produces an error page: it produces the advisory this system could write on its
own, labelled as such, with the reason the model path was not used recorded in
``limitations``. An operator who wanted to know what is happening at a junction
gets an answer either way, and is never misled about where it came from.

The validation gate
-------------------
A generated advisory is scanned before it is shown, using the Phase 1 machinery
in :mod:`app.domain.decision`:

* **numbers** -- :func:`~app.domain.decision.scan_for_unsupported_numbers` finds
  numeric literals in the prose that no fact supports. One is enough to reject.
* **evidence ids** -- any cited id absent from the fact table is a claim to have
  used evidence that does not exist.

Both are rejections rather than repairs. Stripping an invented number leaves a
sentence built around it; quietly dropping a bad citation leaves prose that
still rests on it. The whole advisory goes, and the deterministic one is shown
in its place with the reason attached.

What this module does not do
----------------------------
It computes no traffic quantity. Everything it knows arrives through
:func:`~app.advisor.facts.build_facts` from results other phases already
produced, which is what keeps this an explanation layer rather than a second
opinion competing with the pipeline.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any

from app.advisor.facts import TrafficAdvisorFacts
from app.advisor.fallback import deterministic_advisory
from app.advisor.prompt import SYSTEM_PROMPT, user_prompt
from app.advisor.provider import AdvisorProviderError, TrafficAdvisorLLM
from app.advisor.schemas import AdvisoryMode, TrafficAdvisory
from app.core.logging import get_logger
from app.domain.decision import NarrativeValidation, scan_for_unsupported_numbers

logger = get_logger(__name__)

#: Caps on what a model may return, applied before the text reaches a schema
#: with its own length limits. A model that returns forty suggestions has
#: misunderstood the task; truncating is kinder than a 422.
MAX_LIST_ITEMS = 8
MAX_ITEM_CHARS = 600

#: Matches a ```json ... ``` fence, which several providers add despite being
#: asked for bare JSON.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def generate_advisory(
    facts: TrafficAdvisorFacts,
    provider: TrafficAdvisorLLM | None,
    *,
    now: dt.datetime,
) -> TrafficAdvisory:
    """Produce an advisory for these facts, by whatever route is available."""
    if provider is None:
        return deterministic_advisory(
            facts,
            now=now,
            extra_limitations=(
                "No language model is configured, so this advisory was composed "
                "from the recorded facts by fixed rules rather than generated.",
            ),
        )

    try:
        raw = provider.complete(SYSTEM_PROMPT, user_prompt(facts))
    except AdvisorProviderError as exc:
        logger.info(
            "advisor provider unavailable; falling back",
            # Not "message": that key is reserved on a LogRecord, and passing it
            # through `extra` raises inside logging -- which would crash the
            # fallback path at exactly the moment it is needed.
            extra={"reason": exc.error_code, "detail": exc.message},
        )
        return deterministic_advisory(
            facts,
            now=now,
            extra_limitations=(
                f"The language model was not used: {exc.message} This advisory "
                "was composed from the recorded facts by fixed rules.",
            ),
        )

    parsed = _parse(raw)
    if parsed is None:
        logger.info("advisor model returned unparseable content; falling back")
        return deterministic_advisory(
            facts,
            now=now,
            extra_limitations=(
                "The language model's reply could not be read as the required "
                "JSON advisory, so it was discarded. This advisory was composed "
                "from the recorded facts by fixed rules.",
            ),
        )

    advisory, validation = _validated(parsed, facts, provider.model_id, now=now)
    if advisory is not None:
        return advisory

    logger.warning(
        "advisor narrative rejected by validation",
        extra={
            "unsupported_numbers": list(validation.unsupported_numbers),
            "unknown_fact_ids": list(validation.unknown_fact_ids),
            "model": provider.model_id,
        },
    )
    return deterministic_advisory(
        facts,
        now=now,
        extra_limitations=(_rejection_note(validation),),
    )


def _parse(raw: str) -> dict[str, Any] | None:
    """Read the model's reply as a JSON object, tolerating a code fence.

    Tolerant about packaging, strict about shape. A reply that is not an object
    with the required text fields is discarded rather than patched: guessing
    what a malformed advisory meant is how an advisory stops meaning anything.
    """
    text = raw.strip()
    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    for required in ("summary", "traffic_assessment", "route_advice"):
        value = payload.get(required)
        if not isinstance(value, str) or not value.strip():
            return None
    return payload


def _strings(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    """Read a list-of-strings field, dropping anything that is not usable."""
    value = payload.get(key)
    if not isinstance(value, list):
        return ()
    items = [
        item.strip()[:MAX_ITEM_CHARS]
        for item in value
        if isinstance(item, str) and item.strip()
    ]
    return tuple(items[:MAX_LIST_ITEMS])


def _text(payload: dict[str, Any], key: str, limit: int) -> str:
    """Read a required text field, already known to be present and non-empty."""
    value = payload[key]
    assert isinstance(value, str)  # noqa: S101 - established by _parse
    return value.strip()[:limit]


def _validated(
    payload: dict[str, Any],
    facts: TrafficAdvisorFacts,
    model_id: str,
    *,
    now: dt.datetime,
) -> tuple[TrafficAdvisory | None, NarrativeValidation]:
    """Build the advisory and check it against the fact table.

    Returns ``(None, validation)`` when it fails, so the caller can report
    exactly what was wrong rather than only that something was.
    """
    summary = _text(payload, "summary", 1500)
    assessment = _text(payload, "traffic_assessment", 1500)
    route_advice = _text(payload, "route_advice", 1000)
    suggestions = _strings(payload, "suggestions")
    safety = _strings(payload, "safety_considerations")
    limitations = _strings(payload, "limitations")
    cited = _strings(payload, "evidence_ids")

    prose = "\n".join(
        [summary, assessment, route_advice, *suggestions, *safety, *limitations]
    )
    unsupported = scan_for_unsupported_numbers(prose, facts.fact_table)
    known = facts.fact_table.fact_ids
    unknown = tuple(sorted({item for item in cited if item not in known}))

    validation = NarrativeValidation(
        unsupported_numbers=unsupported,
        unknown_fact_ids=unknown,
        passed=not (unsupported or unknown),
    )
    if not validation.passed:
        return None, validation

    # Recorded absences are appended rather than trusted to the model: an
    # advisory that omits what it could not see reads more confident than it is.
    merged = tuple(
        dict.fromkeys(
            [
                *limitations,
                *(
                    f"{item.label} is unavailable: {item.reason}."
                    for item in facts.unavailable
                ),
            ]
        )
    )[:MAX_LIST_ITEMS]

    advisory = TrafficAdvisory(
        mode=AdvisoryMode.LLM,
        observation_area_id=facts.observation_area_id,
        summary=summary,
        traffic_assessment=assessment,
        suggestions=suggestions,
        safety_considerations=safety,
        route_advice=route_advice,
        limitations=merged,
        evidence_ids=cited,
        model_id=model_id,
        generated_at=now,
        validation=validation,
    )
    return advisory, validation


def _rejection_note(validation: NarrativeValidation) -> str:
    """Say precisely why a generated advisory was discarded."""
    reasons: list[str] = []
    if validation.unsupported_numbers:
        values = ", ".join(f"{value:g}" for value in validation.unsupported_numbers)
        reasons.append(f"it stated figures the facts do not support ({values})")
    if validation.unknown_fact_ids:
        reasons.append(
            "it cited evidence that does not exist "
            f"({', '.join(validation.unknown_fact_ids)})"
        )
    return (
        "The language model's advisory was rejected and not shown because "
        f"{' and '.join(reasons)}. This advisory was composed from the recorded "
        "facts by fixed rules instead."
    )
