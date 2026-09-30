"""The instructions the model gets, and the facts it gets them with.

The prompt is a request, not a control
--------------------------------------
Everything below is worth writing and none of it is load-bearing. A language
model asked not to invent numbers will still invent them, most often as round
figures in exactly the place a reader is least likely to check. The control is
:func:`~app.domain.decision.scan_for_unsupported_numbers`, which runs after
generation and rejects the advisory. The prompt exists to make the right answer
easy, not to make the wrong one impossible.

Which is why the payload is shaped the way it is. Every number the model may use
is pre-rendered in the fact table, including unit conversions, so the model never
has to compute anything. Every missing value is listed with the reason it is
missing, so "unknown" is an available answer rather than a gap to fill.
"""

from __future__ import annotations

import json

from app.advisor.facts import TrafficAdvisorFacts

#: The one instruction that is genuinely enforced elsewhere is repeated here
#: anyway: a model that has been told the rule and breaks it is cheaper to
#: diagnose than one that was never told.
SYSTEM_PROMPT = """\
You are the Traffic Advisor for an advisory-only traffic management system.
You explain traffic facts that a deterministic pipeline has already computed, \
and you suggest actions a human operator may choose to take.

You are not the source of truth. The system is.

ABSOLUTE RULES

1. Use only the facts supplied in the FACTS payload. Invent nothing.
2. Never state a vehicle count, congestion level, road name, coordinate, \
incident or emergency that is not in the payload.
3. Every number you write must appear in the fact table. Prefer quoting a \
display_variant verbatim. Do not convert units; the conversions you need are \
already provided.
4. Distinguish how each value was produced, using the provenance on each fact:
   - observed  -> may be stated as measured fact
   - derived   -> state as computed by the system
   - predicted -> state as a prediction
   - simulated -> state as a modelled outcome under stated assumptions
   - assumed   -> state as an assumption
5. Traffic state bands (free_flow, light, moderate, heavy, severe, gridlock, \
unknown) are system classifications. Report them; do not re-derive or dispute \
them.
6. Never infer an emergency, an accident, or any other incident from detections \
or from congestion. If the payload declares none, say none is declared.
7. ROUTE ADVICE. If has_route_comparison is false, you must NOT name a route as \
faster or better. Say that an alternate route should be evaluated using current \
routing information. If has_route_comparison is true, you may compare the \
supplied durations and must label them as predicted values.
8. SIMULATION. Simulated figures describe a model, never the road. Write "the \
simulation models ..." and never "the road will take ...".
9. If the evidence is insufficient to answer something, say so explicitly.
10. Recommend only. Never claim the system will act, execute, control a signal, \
redirect a vehicle, or override any decision. There is no control path.
11. Anything listed in unavailable is unknown. It is not zero and not "none".
12. Cite the fact ids you relied on in evidence_ids. Use only ids present in \
the fact table.

OUTPUT

Reply with a single JSON object and nothing else. No prose before or after, no \
code fences. Shape:

{
  "summary": "one short paragraph",
  "traffic_assessment": "one short paragraph",
  "suggestions": ["...", "..."],
  "safety_considerations": ["..."],
  "route_advice": "one short paragraph",
  "limitations": ["..."],
  "evidence_ids": ["state.level", "..."]
}

Keep every field plain text. Be concise and specific. An operator reads this \
while doing something else.\
"""


def facts_payload(facts: TrafficAdvisorFacts) -> dict[str, object]:
    """Render the facts as the JSON object the model is given.

    Deliberately flat and verbose. Provenance travels with each fact rather than
    being summarised once at the top, because a model that has to look up which
    section a number came from will eventually not bother.
    """
    fact_entries = [
        {
            "fact_id": fact.fact_id,
            "label": fact.label,
            "value": fact.value,
            "unit": fact.unit,
            "source": fact.provenance.source.value,
            "may_be_stated_as_fact": fact.is_factual,
            "display_variants": list(fact.display_variants),
        }
        for fact in facts.fact_table.facts
    ]

    return {
        "observation_area_id": facts.observation_area_id,
        "site_label": facts.site_label,
        "confirmed_junction_id": facts.confirmed_junction_id,
        "confirmed_segment_id": facts.confirmed_segment_id,
        "confirmed_segment_direction": facts.confirmed_segment_direction,
        "traffic_state": (
            facts.traffic_state.value if facts.traffic_state is not None else None
        ),
        "has_state": facts.has_state,
        "vehicle_count": facts.vehicle_count,
        "vehicle_class_counts": facts.vehicle_class_counts,
        "pedestrian_count": facts.pedestrian_count,
        "cyclist_count": facts.cyclist_count,
        "image_occupancy_ratio": facts.image_occupancy_ratio,
        "image_occupancy_note": (
            "image-space only; this is the share of frame area covered by "
            "detection boxes and is not road occupancy or density"
        ),
        "quality_grade": facts.quality_grade,
        "declared_incidents": list(facts.incident_summaries),
        "declared_emergencies": list(facts.emergency_summaries),
        "incident_note": (
            "an incident exists only because somebody declared it; never infer "
            "one from detections or congestion"
        ),
        "has_route_comparison": facts.has_route_comparison,
        "route_baseline_s": facts.route_baseline_s,
        "route_alternative_s": facts.route_alternative_s,
        "route_improvement_pct": facts.route_improvement_pct,
        "has_simulation": facts.has_simulation,
        "simulation_travel_time_delta_s": facts.simulation_travel_time_delta_s,
        "simulation_improvement_pct": facts.simulation_improvement_pct,
        "simulation_note": (
            "simulated figures describe a model under stated assumptions, never "
            "a measurement of the road"
        ),
        "fact_table": fact_entries,
        "unavailable": [
            {"fact_id": item.fact_id, "label": item.label, "reason": item.reason}
            for item in facts.unavailable
        ],
    }


def user_prompt(facts: TrafficAdvisorFacts) -> str:
    """The user-turn content: the facts, and what to do with them."""
    payload = json.dumps(facts_payload(facts), indent=2, default=str)
    return (
        "FACTS\n"
        f"{payload}\n\n"
        "Write the advisory as a single JSON object following the required "
        "shape. Use only the facts above. Where something is listed under "
        "'unavailable', say it is unknown rather than estimating it."
    )
