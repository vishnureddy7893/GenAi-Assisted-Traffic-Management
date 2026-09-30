"""The advisory this system can write without a language model.

Why this exists at all
----------------------
Three situations produce it, and the third is the interesting one:

* no provider is configured -- an offline demo, or a deployment that has not
  been given a key;
* the provider failed -- timeout, HTTP error, unparseable reply;
* the provider answered and the answer was **rejected** by validation.

That last case is the reason this is not a stub. When a generated advisory cites
a number the facts do not support, the correct action is to discard it, and
discarding it leaves the operator with nothing unless something else can answer.
This can, because every sentence below is assembled from a fact that exists.

What it is not
--------------
It is not an LLM, and it never claims to be: the service labels it
``deterministic_fallback`` and :class:`~app.advisor.schemas.TrafficAdvisory`
refuses to attach a model id to it. It is also not a second analysis pipeline --
it classifies nothing, computes nothing, and only reads the band the traffic
state system already decided.
"""

from __future__ import annotations

import datetime as dt

from app.advisor.facts import TrafficAdvisorFacts
from app.advisor.schemas import AdvisoryMode, TrafficAdvisory
from app.domain.enums import CongestionLevel

#: One line per band, phrased as an observation plus what it licenses. Every one
#: of them stops short of naming a road or a route, because the band alone is
#: not evidence about either.
_ASSESSMENT: dict[CongestionLevel, str] = {
    CongestionLevel.FREE_FLOW: (
        "No significant congestion is indicated by the available evidence."
    ),
    CongestionLevel.LIGHT: (
        "Traffic is light on the monitored area. Continue routine monitoring."
    ),
    CongestionLevel.MODERATE: (
        "Traffic is moderate on the monitored area. Continue monitoring and "
        "evaluate alternate routes if it increases."
    ),
    CongestionLevel.HEAVY: (
        "Heavy traffic is indicated on the monitored area. Evaluate an alternate "
        "route using current routing information before directing additional "
        "demand toward the affected segment."
    ),
    CongestionLevel.SEVERE: (
        "Severe congestion is indicated on the monitored area. Evaluate alternate "
        "routes and review any declared incident information."
    ),
    CongestionLevel.GRIDLOCK: (
        "Gridlock is indicated by the traffic-state system. Avoid directing "
        "additional demand into the affected area where a verified alternative "
        "exists."
    ),
    CongestionLevel.UNKNOWN: (
        "The traffic-state system could not classify this observation, so no "
        "congestion assessment is available. This is missing information, not a "
        "report of clear roads."
    ),
}

_SUGGESTIONS: dict[CongestionLevel, tuple[str, ...]] = {
    CongestionLevel.FREE_FLOW: (
        "No action is indicated by the current evidence.",
        "Continue monitoring this observation area.",
    ),
    CongestionLevel.LIGHT: (
        "Continue monitoring this observation area.",
        "No demand-management action is indicated by the current evidence.",
    ),
    CongestionLevel.MODERATE: (
        "Continue monitoring and re-assess if the observed state worsens.",
        "Evaluate alternate routes using current routing information before "
        "congestion increases further.",
    ),
    CongestionLevel.HEAVY: (
        "Evaluate alternate routes using current routing information before "
        "directing additional demand toward the affected area.",
        "Re-run the analysis on more recent footage to confirm the state persists.",
        "Check whether any incident has been declared that would explain the "
        "observed congestion.",
    ),
    CongestionLevel.SEVERE: (
        "Evaluate alternate routes and hold additional demand away from the "
        "affected area where a verified alternative exists.",
        "Review declared incidents for a cause; none is inferred from the footage.",
        "Re-run the analysis on more recent footage to confirm the state persists.",
    ),
    CongestionLevel.GRIDLOCK: (
        "Avoid directing additional demand into the affected area where a "
        "verified alternative exists.",
        "Review declared incidents and emergency responses for context.",
        "Escalate to an operator with authority over the affected corridor.",
    ),
    CongestionLevel.UNKNOWN: (
        "Re-run analysis and interpretation on this media to obtain a state.",
        "Do not treat an unclassified observation as clear road.",
    ),
}

#: Bands at which a pedestrian count, where one exists, is worth raising.
_CROWDED = frozenset(
    {CongestionLevel.HEAVY, CongestionLevel.SEVERE, CongestionLevel.GRIDLOCK}
)


def deterministic_advisory(
    facts: TrafficAdvisorFacts,
    *,
    now: dt.datetime,
    extra_limitations: tuple[str, ...] = (),
) -> TrafficAdvisory:
    """Compose an advisory from the facts alone.

    ``extra_limitations`` carries why the model path was not used, so an
    operator reading a fallback advisory learns whether it was offline, failed
    or was rejected.
    """
    level = facts.traffic_state
    assessment = (
        _ASSESSMENT[level]
        if level is not None
        else (
            "No traffic state has been interpreted for this observation area, so "
            "no congestion assessment can be made. This is an absence of "
            "evidence, not a report of clear roads."
        )
    )

    return TrafficAdvisory(
        mode=AdvisoryMode.DETERMINISTIC_FALLBACK,
        observation_area_id=facts.observation_area_id,
        summary=_summary(facts, level),
        traffic_assessment=assessment,
        suggestions=_suggestions(facts, level),
        safety_considerations=_safety(facts, level),
        route_advice=_route_advice(facts),
        limitations=tuple(extra_limitations) + _limitations(facts),
        evidence_ids=tuple(sorted(facts.fact_table.fact_ids)),
        model_id=None,
        generated_at=now,
        validation=None,
    )


def _where(facts: TrafficAdvisorFacts) -> str:
    """Name the place using only what a human confirmed or recorded."""
    if facts.confirmed_segment_id is not None:
        return (
            f"segment {facts.confirmed_segment_id} "
            f"({facts.confirmed_segment_direction} at {facts.confirmed_junction_id})"
        )
    if facts.confirmed_junction_id is not None:
        return f"junction {facts.confirmed_junction_id}"
    if facts.site_label is not None:
        return facts.site_label
    return f"observation area {facts.observation_area_id}"


def _summary(facts: TrafficAdvisorFacts, level: CongestionLevel | None) -> str:
    """One paragraph, built only from values that are present."""
    where = _where(facts)
    if level is None:
        return (
            f"No interpreted traffic state is available for {where}. The advisor "
            "has no congestion evidence to report."
        )

    parts = [f"The traffic-state system classifies {where} as {level.value}."]
    if facts.vehicle_count is not None:
        parts.append(
            f"{facts.vehicle_count} vehicles were counted in the observation area."
        )
    if facts.congestion_index is not None:
        parts.append(f"The congestion index is {facts.congestion_index:.2f}.")
    if facts.pedestrian_count:
        parts.append(f"{facts.pedestrian_count} pedestrians were tracked.")
    if facts.has_incident:
        parts.append(
            f"{len(facts.incident_summaries)} incident(s) are declared in the area."
        )
    return " ".join(parts)


def _suggestions(
    facts: TrafficAdvisorFacts, level: CongestionLevel | None
) -> tuple[str, ...]:
    """Band-appropriate actions, plus anything the evidence gaps imply."""
    if level is None:
        return (
            "Run vision analysis and traffic-state interpretation on this media "
            "to obtain an assessment.",
            "Do not treat the absence of a state as an absence of congestion.",
        )

    suggestions = list(_SUGGESTIONS[level])
    if facts.confirmed_segment_id is None and level not in {
        CongestionLevel.FREE_FLOW,
        CongestionLevel.UNKNOWN,
    }:
        suggestions.append(
            "Confirm which directed segment this camera watches so the observed "
            "congestion can be attributed to one direction of travel."
        )
    return tuple(suggestions)


def _safety(
    facts: TrafficAdvisorFacts, level: CongestionLevel | None
) -> tuple[str, ...]:
    """Safety notes, none of which assume anything not in the facts."""
    notes = [
        "This advisory is decision support for a human operator. The system has "
        "no control path to any traffic signal, vehicle or infrastructure.",
    ]
    if facts.pedestrian_count and level in _CROWDED:
        notes.append(
            f"{facts.pedestrian_count} pedestrians were tracked in this view "
            "while congestion is elevated; pedestrian clearance must not be "
            "shortened to improve vehicle flow."
        )
    if not facts.has_emergency:
        notes.append(
            "No emergency response is declared. An emergency is never inferred "
            "from footage, so this reflects what has been reported, not what is "
            "happening."
        )
    if level is CongestionLevel.UNKNOWN or level is None:
        notes.append(
            "An unclassified observation must not be acted on as if it were a "
            "clear road."
        )
    return tuple(notes)


def _route_advice(facts: TrafficAdvisorFacts) -> str:
    """The rule that matters most: no route claim without route evidence."""
    if not facts.has_route_comparison:
        return (
            "No route comparison is available, so no route can be described as "
            "faster. An alternate route should be evaluated using current "
            "routing information before demand is redirected."
        )

    baseline = facts.route_baseline_s
    alternative = facts.route_alternative_s
    improvement = facts.route_improvement_pct
    if baseline is None or alternative is None:  # pragma: no cover - guarded upstream
        return (
            "Route candidates exist but carry no comparable travel times, so no "
            "route can be described as faster."
        )

    if improvement is not None and abs(improvement) < 1.0:
        return (
            f"The available route comparison does not show a meaningful "
            f"improvement: the recommended route's predicted travel time "
            f"({alternative:.0f} s) is close to the baseline ({baseline:.0f} s)."
        )

    direction = "lower" if alternative < baseline else "higher"
    advice = (
        f"Under the supplied routing evidence, the recommended route "
        f"{facts.route_selected_id or ''} has a {direction} predicted travel "
        f"time ({alternative:.0f} s) than the baseline ({baseline:.0f} s)."
    ).replace("  ", " ")
    if improvement is not None:
        advice += f" That is a predicted difference of {improvement:.1f}%."
    return advice + " These are predicted values, not measurements."


def _limitations(facts: TrafficAdvisorFacts) -> tuple[str, ...]:
    """Everything the advisory could not see, in the words already recorded."""
    limitations = [f"{item.label} is unavailable: {item.reason}." for item in facts.unavailable]

    if facts.image_occupancy_ratio is not None:
        limitations.append(
            "Image occupancy is a share of frame area covered by detection "
            "boxes. It is not road occupancy or traffic density, both of which "
            "require road geometry this observation does not have."
        )
    if facts.has_simulation:
        limitations.append(
            "Simulation figures describe a model under stated assumptions and "
            "are not measurements of the road."
        )
    limitations.append(
        "This advisory was composed from the recorded facts by fixed rules. No "
        "language model was involved."
    )
    return tuple(limitations)
