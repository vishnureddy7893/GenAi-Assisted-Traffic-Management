"""Pricing one segment traversal, in equivalent seconds.

Two questions, kept apart because they fail differently:

* **How long does this take?** Answered from the best time basis available, and
  answered with *nothing* when no basis exists. There is no fallback speed here,
  because a fallback speed is an invented measurement no matter what it is
  called.
* **How much worse is it than a clear road?** Answered from congestion,
  incidents, soft safety constraints and evidence quality, each priced by a
  configured weight and each recorded separately so the total can be taken
  apart.

Unknown is priced, and not as zero
----------------------------------
The single most consequential rule in this module. A segment nobody is watching
has no congestion band, and charging it nothing would make the optimizer prefer
exactly the roads it knows least about -- it would route an ambulance down the
unmonitored street *because* it is unmonitored. So an absent or unclassified
band draws an uncertainty penalty instead, and congestion is priced only where a
band genuinely exists.
"""

from __future__ import annotations

import datetime as dt

from app.domain.enums import CongestionLevel, FreshnessStatus
from app.domain.network import RoadSegment
from app.domain.provenance import (
    FloatMeasurement,
    Provenance,
    Unit,
    ValueSource,
)
from app.domain.routing import TimeBasis
from app.domain.state import RoadSegmentState
from app.routing.schemas import EvidenceStatus, SegmentEvidence
from app.routing.weights import RouteWeights, RoutingMode
from app.state.schemas import IntersectionStateElement

#: Identifier recorded in the provenance of every derived traversal time.
COSTING_PRODUCER = "routing.segment_costing"
COSTING_VERSION = "1.0"


class TraversalTime:
    """A leg's time and the basis it rests on. ``seconds`` is ``None`` when none."""

    __slots__ = ("basis", "reason", "seconds")

    def __init__(
        self, seconds: float | None, basis: TimeBasis, reason: str | None = None
    ) -> None:
        self.seconds = seconds
        self.basis = basis
        self.reason = reason

    @property
    def is_available(self) -> bool:
        """Whether a time could be established at all."""
        return self.seconds is not None


def traversal_time(
    segment: RoadSegment,
    state: RoadSegmentState | None,
    *,
    mode: RoutingMode,
) -> TraversalTime:
    """Resolve how long this segment takes, preferring the strongest basis.

    The order is deliberate. An emergency-specific estimate, where one exists,
    was produced by its own model against its own ground truth -- it is not the
    nominal time scaled by a constant, and this module will not manufacture one
    if it is missing. A nominal estimate comes next. Failing both, length over
    the segment's *surveyed* free-flow speed is a derivation from two measured
    quantities rather than an assumption, though it describes an empty road and
    is therefore optimistic wherever there is traffic.

    When none of the three is available the answer is ``None``. That is not a
    failure of this function; it is the honest state of the data, and the route
    containing this segment will report no ETA because of it.
    """
    estimate = state.travel_time if state is not None else None

    if mode is RoutingMode.EMERGENCY and estimate is not None:
        emergency = estimate.emergency_travel_time_s
        if emergency is not None:
            return TraversalTime(emergency.value, TimeBasis.MEASURED_EMERGENCY)

    if estimate is not None and estimate.nominal_travel_time_s is not None:
        reason = None
        if mode is RoutingMode.EMERGENCY:
            reason = (
                "No emergency-specific travel time exists for this segment, so the "
                "nominal time is used unchanged. No emergency advantage is modelled: "
                "scaling nominal time by a constant would be optimistic in free flow "
                "and badly pessimistic in the stop-and-go conditions that decide "
                "which route an ambulance takes."
            )
        return TraversalTime(
            estimate.nominal_travel_time_s.value, TimeBasis.MEASURED_NOMINAL, reason
        )

    speed = segment.free_flow_speed_kph
    if speed is not None and speed.value > 0:
        seconds = segment.length_m.value / (speed.value / 3.6)
        return TraversalTime(
            seconds,
            TimeBasis.DERIVED_FROM_FREE_FLOW,
            "Derived as length over the segment's surveyed free-flow speed. It "
            "describes an empty road, so it is optimistic wherever there is traffic.",
        )

    return TraversalTime(
        None,
        TimeBasis.UNAVAILABLE,
        f"Segment {segment.segment_id!r} carries no travel-time estimate and no "
        "free-flow speed, so no traversal time can be established. No speed is "
        "assumed in its place.",
    )


def time_measurement(
    time: TraversalTime, segment: RoadSegment, *, at: dt.datetime
) -> FloatMeasurement | None:
    """Wrap a resolved time in a measurement carrying its own provenance."""
    if time.seconds is None:
        return None
    source = (
        ValueSource.DERIVED
        if time.basis is TimeBasis.DERIVED_FROM_FREE_FLOW
        else ValueSource.OBSERVED
    )
    return FloatMeasurement(
        value=round(time.seconds, 3),
        unit=Unit.SECOND,
        provenance=Provenance(
            source=source,
            producer=COSTING_PRODUCER,
            producer_version=COSTING_VERSION,
            observed_at=at,
            computed_at=at,
            input_refs=(segment.segment_id,),
            notes=time.reason,
        ),
    )


def segment_evidence(
    segment_id: str,
    element: IntersectionStateElement | None,
    observation_area_id: str | None,
    *,
    weights: RouteWeights,
    degraded_by: tuple[str, ...] = (),
) -> SegmentEvidence:
    """Price one segment's traffic evidence, or its absence.

    ``element`` is the interpreted state of the observation area bound to this
    segment. ``None`` means either nothing is bound or the bound area has not
    been interpreted -- both are ``UNMAPPED`` from the optimizer's point of view,
    because neither yields a usable band, and both are named in the result.
    """
    incident_penalty = weights.incident_degraded_s if degraded_by else 0.0

    if element is None:
        return SegmentEvidence(
            segment_id=segment_id,
            status=EvidenceStatus.UNMAPPED,
            congestion_level=CongestionLevel.UNKNOWN,
            observation_area_id=observation_area_id,
            uncertainty_penalty_s=weights.uncertainty_unmapped_s,
            incident_penalty_s=incident_penalty,
            reason=(
                "No observation area is bound to this segment, so no traffic "
                "evidence can be attributed to it. Priced as unobserved rather "
                "than as clear."
                if observation_area_id is None
                else (
                    f"Observation area {observation_area_id!r} is bound to this "
                    "segment but has not been interpreted, so no band exists yet."
                )
            ),
        )

    state = element.state
    assessment = state.congestion
    provenance = state.analytics_provenance

    common = {
        "segment_id": segment_id,
        "observation_area_id": state.observation_area_id,
        "traffic_state_id": state.state_id,
        "freshness": element.freshness,
        "evidence_age_s": element.evidence_age_s,
        "media_id": provenance.media_id,
        "frame_range": provenance.frame_range,
        "time_range": provenance.time_range,
        "detector_model": provenance.model.name,
        "incident_penalty_s": incident_penalty,
    }

    if assessment.level is CongestionLevel.UNKNOWN:
        return SegmentEvidence(
            **common,
            status=EvidenceStatus.MAPPED_UNCLASSIFIED,
            congestion_level=CongestionLevel.UNKNOWN,
            uncertainty_penalty_s=weights.uncertainty_unclassified_s,
            reason=(
                "Traffic was observed here but could not be classified: "
                f"{assessment.reason or 'no reason recorded'} Priced as uncertain "
                "rather than as clear."
            ),
        )

    stale = element.freshness is not FreshnessStatus.FRESH
    return SegmentEvidence(
        **common,
        status=EvidenceStatus.MAPPED_STALE if stale else EvidenceStatus.MAPPED_CLASSIFIED,
        congestion_level=assessment.level,
        congestion_index=assessment.index,
        congestion_penalty_s=weights.congestion_penalty(assessment.level),
        uncertainty_penalty_s=weights.uncertainty_stale_s if stale else 0.0,
        reason=(
            (
                f"Evidence is {element.freshness.value} at "
                f"{element.evidence_age_s:.1f} s old. The band is still used -- "
                "withholding a route because one camera lagged would be its own "
                "kind of failure -- with an uncertainty charge alongside."
            )
            if stale
            else None
        ),
    )


def distance_penalty(segment: RoadSegment, weights: RouteWeights) -> float:
    """Distance priced as a stated preference, zero unless configured."""
    if weights.distance_penalty_s_per_100m <= 0.0:
        return 0.0
    return (segment.length_m.value / 100.0) * weights.distance_penalty_s_per_100m
