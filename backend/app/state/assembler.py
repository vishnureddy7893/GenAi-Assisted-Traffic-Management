"""Assembly of interpreted state, for one area and across many.

Two steps, kept separate because they fail differently:

* :func:`interpret` turns one measurement set into one interpreted state. It
  cannot fail for reasons of timing -- it either has evidence or says it does
  not.
* :func:`assemble` fuses many interpreted states into a snapshot, and *timing*
  is its whole problem. Cameras are analysed at different moments, and a fuse
  that discards that fact produces a snapshot whose elements are minutes apart
  while looking simultaneous.

Both are pure functions: every timestamp is passed in rather than read from a
clock, so a test can pin the exact instant and a replay reproduces the snapshot
it is checking against.

The reference instant is the newest element, not now
----------------------------------------------------
Element ages are measured against the newest element in the snapshot rather than
against wall-clock time. Recorded footage carries media time -- analysing a clip
filmed last year produces states dated last year -- and measuring those against
``now()`` would mark every element expired the instant it was computed, which
says nothing about whether the elements agree with each other. Measuring against
the newest element answers the question that actually matters for fusion: how
far apart are these observations?
"""

from __future__ import annotations

import datetime as dt
import hashlib

from app.analytics.schemas import AnalyticsProvenance, TrafficAnalytics
from app.core.config import Settings
from app.domain.enums import FreshnessStatus
from app.domain.state import StateAssemblyPolicy
from app.state.classifier import classify, vehicle_count_for
from app.state.schemas import (
    IntersectionStateElement,
    IntersectionTrafficState,
    NetworkTrafficState,
)

STATE_SCHEMA_VERSION = "1.0"

#: Named so a snapshot records how multiple analyses of one area were reconciled.
FUSION_STRATEGY = "latest_per_observation_area"


def observed_at(provenance: AnalyticsProvenance) -> dt.datetime:
    """The instant a measurement set describes.

    Prefers the end of an analysed range over its start: a congestion verdict
    over a 30-second clip describes the traffic as it was by the end of that
    clip, and dating it from the beginning would systematically report state
    that is one window stale.
    """
    if provenance.time_range is not None:
        return provenance.time_range[1]
    if provenance.timestamp is not None:
        return provenance.timestamp
    return provenance.computed_at


def _state_id(observation_area_id: str, as_of: dt.datetime) -> str:
    """A deterministic identifier for one interpreted state.

    Derived rather than random so that re-interpreting the same measurements
    yields the same id, which is what lets a stored decision be matched against
    a re-run of the state that produced it. The area id is hashed because it may
    be far longer than an identifier field allows.
    """
    digest = hashlib.sha256(observation_area_id.encode("utf-8")).hexdigest()[:12]
    return f"state-{digest}-{int(as_of.timestamp() * 1000)}"


def interpret(
    analytics: TrafficAnalytics,
    settings: Settings,
    *,
    assessed_at: dt.datetime,
) -> IntersectionTrafficState:
    """Interpret one measurement set as traffic state for its observation area."""
    assessment = classify(analytics, settings)
    as_of = observed_at(analytics.provenance)
    stationary = analytics.stationary_vehicle_count

    return IntersectionTrafficState(
        state_id=_state_id(analytics.observation_area_id, as_of),
        observation_area_id=analytics.observation_area_id,
        mode=analytics.mode,
        as_of=as_of,
        assessed_at=assessed_at,
        congestion=assessment,
        vehicle_count=vehicle_count_for(analytics),
        stationary_vehicle_count=stationary.value if stationary.is_available else None,
        quality_grade=analytics.quality.grade,
        analytics_provenance=analytics.provenance,
        schema_version=STATE_SCHEMA_VERSION,
    )


def policy_from(settings: Settings) -> StateAssemblyPolicy:
    """Build the staleness policy in force from configuration."""
    return StateAssemblyPolicy(
        max_staleness_s=settings.max_observation_staleness_s,
        drop_after_s=settings.drop_observation_after_s,
        fusion_strategy=FUSION_STRATEGY,
    )


def freshness_for(age_s: float, policy: StateAssemblyPolicy) -> FreshnessStatus:
    """Classify an element's age against the budget.

    ``UNKNOWN`` is never produced here: it means no observation has ever been
    received for an element, which cannot be true of a state that exists.
    """
    if age_s <= policy.max_staleness_s:
        return FreshnessStatus.FRESH
    if age_s <= policy.drop_after_s:
        return FreshnessStatus.STALE
    return FreshnessStatus.EXPIRED


def assemble(
    states: tuple[IntersectionTrafficState, ...] | list[IntersectionTrafficState],
    settings: Settings,
    *,
    assembled_at: dt.datetime,
) -> NetworkTrafficState:
    """Fuse interpreted states into one snapshot under the staleness budget.

    Where an observation area has been interpreted more than once, the newest
    state wins -- the fusion strategy recorded in the policy. The older ones are
    dropped rather than averaged: two analyses of the same camera at different
    instants are not two samples of one quantity, and blending them would
    produce a state that existed at neither moment.
    """
    policy = policy_from(settings)

    latest: dict[str, IntersectionTrafficState] = {}
    for state in states:
        existing = latest.get(state.observation_area_id)
        if existing is None or state.as_of > existing.as_of:
            latest[state.observation_area_id] = state

    selected = sorted(latest.values(), key=lambda item: item.observation_area_id)
    as_of = max((item.as_of for item in selected), default=assembled_at)

    elements = tuple(
        IntersectionStateElement(
            state=item,
            freshness=freshness_for(
                max(0.0, (as_of - item.as_of).total_seconds()), policy
            ),
            evidence_age_s=round(max(0.0, (as_of - item.as_of).total_seconds()), 3),
        )
        for item in selected
    )

    fingerprint = hashlib.sha256(
        "|".join(item.state.state_id for item in elements).encode("utf-8")
    ).hexdigest()[:12]

    return NetworkTrafficState(
        state_id=f"netstate-{fingerprint}-{int(as_of.timestamp() * 1000)}",
        as_of=as_of,
        assembled_at=assembled_at,
        policy=policy,
        elements=elements,
        schema_version=STATE_SCHEMA_VERSION,
    )
