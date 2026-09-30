"""Dynamic network state: the fused, network-wide snapshot at one instant.

This is the assembled view that the decision engine reasons over. It is built
from many :class:`~app.domain.observation.JunctionObservation` records plus
environmental and incident context, and it is immutable once built.

Time alignment is the hard part
-------------------------------
Cameras do not report in lockstep. Junction J11 might have a 200 ms-old
observation while J14's is 45 s old. A naive "latest observation per junction"
fuse produces a snapshot whose elements are minutes apart, and nothing in the
resulting object records that fact. Routing an ambulance across such a snapshot
means routing partly on the present and partly on the past.

So every element carries its own :class:`~app.domain.enums.FreshnessStatus`,
computed against an explicit, configured budget that is recorded in
:class:`StateAssemblyPolicy`. Downstream code can then require freshness where
it matters, and a decision built on stale evidence is marked degraded rather
than presented with false confidence.

Absent is not zero
------------------
Every dynamic quantity is optional. A junction with no camera has *unknown*
queues, not empty ones. Defaulting to zero would let a routing engine treat an
unmonitored gridlocked road as the fastest path on the network.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Annotated

from pydantic import Field, model_validator

from app.domain.base import DomainModel
from app.domain.enums import (
    CongestionLevel,
    DataQualityFlag,
    FreshnessStatus,
    LevelOfService,
    RoadSurfaceCondition,
    SignalIndication,
    VisibilityLevel,
    WeatherCondition,
)
from app.domain.incident import Incident
from app.domain.network import ApproachId, JunctionId, MovementId, SegmentId
from app.domain.observation import ObservationId, SourceId
from app.domain.provenance import FloatMeasurement, IntMeasurement, Provenance

StateId = Annotated[str, Field(min_length=1, max_length=64)]


class EmergencyTraversalAssumption(StrEnum):
    """What an emergency travel-time estimate assumed in order to be valid.

    Colocated with :class:`TravelTimeEstimate` because it is meaningless apart
    from it. An emergency ETA is always conditional -- it depends on operational
    behaviour that has not happened yet and that this system can only recommend,
    never guarantee. Recording the conditions makes the estimate falsifiable and
    makes the generated narrative honest: "3 minutes *if* priority is granted at
    J14" is a defensible statement; "3 minutes" is not.
    """

    SIGNAL_PRIORITY_GRANTED = "signal_priority_granted"
    """Assumes an operator acts on the recommended priority at each junction."""

    OPPOSING_TRAFFIC_YIELDS = "opposing_traffic_yields"
    QUEUE_JUMPING_PERMITTED = "queue_jumping_permitted"
    """Assumes the vehicle can pass a stationary queue using opposing or shoulder
    space -- only defensible where the segment geometry allows it."""

    CONTRAFLOW_USED = "contraflow_used"
    NO_SIGNAL_PRIORITY = "no_signal_priority"
    """The conservative case: emergency vehicle obeys signals as normal traffic."""

    CLEAR_RUN_ASSUMED = "clear_run_assumed"


class TravelTimeEstimate(DomainModel):
    """Travel time across a segment, for normal and for emergency traffic.

    The two fields are independent by construction, and this module intentionally
    provides **no function that converts one into the other**.

    Multiplying a nominal travel time by a constant is the standard shortcut and
    it is wrong in a way that matters. An emergency vehicle's advantage is not
    proportional to normal travel time -- it comes from discrete mechanisms:
    passing stopped queues, receiving signal priority, using opposing lanes.
    On a free-flowing segment the advantage is near zero; in stop-and-go
    conditions it can be several-fold; on a fully blocked segment it is
    undefined for both. A constant ratio gets the free-flow case optimistic and
    the gridlock case badly pessimistic, which is precisely the case that
    decides which route an ambulance takes.

    Each estimate is therefore produced and validated by its own model, against
    its own ground truth, and carries its own provenance.
    """

    nominal_travel_time_s: FloatMeasurement | None = None
    """Expected traversal time for general traffic."""

    emergency_travel_time_s: FloatMeasurement | None = None
    """Expected traversal time for an emergency vehicle under the stated
    assumptions. Never derived from ``nominal_travel_time_s``."""

    emergency_assumptions: tuple[EmergencyTraversalAssumption, ...] = ()
    """Required whenever an emergency estimate is present."""

    @model_validator(mode="after")
    def _emergency_estimate_states_its_assumptions(self) -> TravelTimeEstimate:
        if self.emergency_travel_time_s is not None and not self.emergency_assumptions:
            msg = (
                "emergency_travel_time_s requires emergency_assumptions; an emergency "
                "ETA is conditional on operational behaviour this system can only "
                "recommend, and an unconditional emergency ETA overstates certainty"
            )
            raise ValueError(msg)
        return self

    @property
    def emergency_advantage_ratio(self) -> float | None:
        """Observed ratio of emergency to nominal time, for reporting only.

        Computed *from* two independent estimates as a sanity check. It is never
        used to produce either one; a value near or above 1.0 is a legitimate
        signal that this segment offers an emergency vehicle no advantage.
        """
        if self.nominal_travel_time_s is None or self.emergency_travel_time_s is None:
            return None
        nominal = self.nominal_travel_time_s.value
        if nominal <= 0:
            return None
        return self.emergency_travel_time_s.value / nominal


class EnvironmentConditions(DomainModel):
    """Environmental context applying to an area of the network.

    Scoped rather than global: fog in a river valley does not affect the whole
    modelled area, and applying it network-wide would penalise routes that are
    genuinely clear.
    """

    weather: WeatherCondition = WeatherCondition.UNKNOWN
    visibility: VisibilityLevel = VisibilityLevel.UNKNOWN
    surface_condition: RoadSurfaceCondition = RoadSurfaceCondition.UNKNOWN
    temperature_c: FloatMeasurement | None = None
    precipitation_mm_per_hour: FloatMeasurement | None = None
    applies_to_segment_ids: frozenset[str] | None = None
    """``None`` means network-wide. An explicit set scopes it to those segments."""

    provenance: Provenance
    observed_at: dt.datetime

    @model_validator(mode="after")
    def _tz_aware(self) -> EnvironmentConditions:
        if self.observed_at.tzinfo is None:
            msg = "observed_at must be timezone-aware"
            raise ValueError(msg)
        return self


class RoadSegmentState(DomainModel):
    """Dynamic condition of one directed segment at the snapshot instant."""

    segment_id: SegmentId
    freshness: FreshnessStatus = FreshnessStatus.UNKNOWN
    evidence_age_s: float | None = None
    """Age of the newest evidence supporting this state, on the snapshot clock."""

    flow_veh_per_hour: FloatMeasurement | None = None
    mean_speed_kph: FloatMeasurement | None = None
    density_veh_per_km_per_lane: FloatMeasurement | None = None
    volume_to_capacity_ratio: FloatMeasurement | None = None
    """Only computable where the segment has a *calibrated* capacity. Absent
    otherwise, rather than computed against a guessed denominator."""

    queue_length_m: FloatMeasurement | None = None
    queue_length_veh: IntMeasurement | None = None

    congestion_level: CongestionLevel = CongestionLevel.UNKNOWN
    level_of_service: LevelOfService | None = None

    travel_time: TravelTimeEstimate = TravelTimeEstimate()

    is_blocked: bool = False
    """Hard constraint input. Set only from an incident or an authoritative
    closure record -- never inferred from heavy congestion. A jammed road is
    slow; a blocked road is impassable, and conflating them either strands an
    ambulance or discards a usable route."""

    blocked_by_incident_ids: frozenset[str] = frozenset()
    surface_condition: RoadSurfaceCondition = RoadSurfaceCondition.UNKNOWN
    is_emergency_accessible: bool | None = None
    """``None`` means not assessed. Assessment happens in the safety engine
    against the specific responding vehicle type, since accessibility is
    vehicle-dependent."""

    @model_validator(mode="after")
    def _blocked_implies_reason(self) -> RoadSegmentState:
        if self.is_blocked and not self.blocked_by_incident_ids:
            msg = (
                f"segment {self.segment_id!r} is marked blocked with no causing "
                "incident; a blockage with no provenance cannot be explained to an "
                "operator or cleared when the cause resolves"
            )
            raise ValueError(msg)
        return self


class ApproachState(DomainModel):
    """Dynamic condition of one approach at a junction."""

    approach_id: ApproachId
    incoming_segment_id: SegmentId
    queue_length_m: FloatMeasurement | None = None
    queue_length_veh: IntMeasurement | None = None
    mean_waiting_time_s: FloatMeasurement | None = None
    stopped_vehicle_count: IntMeasurement | None = None
    congestion_level: CongestionLevel = CongestionLevel.UNKNOWN
    level_of_service: LevelOfService | None = None
    observed_signal_indication: SignalIndication = SignalIndication.UNKNOWN
    pedestrians_waiting: IntMeasurement | None = None
    pedestrian_density_person_per_m2: FloatMeasurement | None = None


class JunctionState(DomainModel):
    """Dynamic condition of one junction at the snapshot instant."""

    junction_id: JunctionId
    freshness: FreshnessStatus = FreshnessStatus.UNKNOWN
    evidence_age_s: float | None = None
    contributing_observation_ids: tuple[ObservationId, ...] = ()
    contributing_source_ids: tuple[SourceId, ...] = ()
    """Which sources produced this state. Two cameras disagreeing about the same
    junction is a condition worth surfacing, not silently averaging."""

    approaches: tuple[ApproachState, ...] = ()
    quality_flags: frozenset[DataQualityFlag] = frozenset()
    active_incident_ids: frozenset[str] = frozenset()
    emergency_active: bool = False
    """Whether a declared emergency is currently traversing or approaching."""

    signal_indications: dict[MovementId, SignalIndication] = Field(default_factory=dict)
    """*Observed* indications only. Recommendations live in
    :class:`~app.domain.decision.SignalAdvisory`, in a different module and a
    different type, so that a fact about the world and a proposal for changing
    it can never be confused for one another in code or on the wire."""

    @property
    def is_usable_evidence(self) -> bool:
        """Whether this junction's state may support a decision."""
        return self.freshness in (FreshnessStatus.FRESH, FreshnessStatus.STALE)


class StateAssemblyPolicy(DomainModel):
    """The time-alignment policy used to build a snapshot.

    Recorded inside the snapshot rather than read from config at decision time,
    so that a recommendation can be audited against the exact freshness rules
    that were in force when it was produced.
    """

    max_staleness_s: Annotated[float, Field(gt=0)]
    """Observations older than this are usable but flagged STALE."""

    drop_after_s: Annotated[float, Field(gt=0)]
    """Observations older than this are discarded; the element becomes UNKNOWN."""

    fusion_strategy: Annotated[str, Field(min_length=1, max_length=100)] = "latest_per_junction"
    """Named strategy for reconciling multiple sources on one junction."""

    @model_validator(mode="after")
    def _ordering(self) -> StateAssemblyPolicy:
        if self.drop_after_s < self.max_staleness_s:
            msg = "drop_after_s must be >= max_staleness_s"
            raise ValueError(msg)
        return self


class NetworkState(DomainModel):
    """Fused, network-wide snapshot at a single instant.

    The unit of input to the decision engine and the unit of reproducibility:
    given the same ``NetworkState``, the same network version and the same
    policy versions, a decision must be identical.
    """

    state_id: StateId
    network_id: Annotated[str, Field(min_length=1, max_length=64)]
    network_version: Annotated[str, Field(min_length=1, max_length=50)]
    schema_version: Annotated[str, Field(min_length=1, max_length=20)]

    as_of: dt.datetime
    """The instant this snapshot claims to describe, on the source clock. For a
    recorded replay this is media time, so a snapshot from a year-old video is
    correctly dated a year ago and cannot masquerade as current."""

    assembled_at: dt.datetime
    policy: StateAssemblyPolicy

    junction_states: tuple[JunctionState, ...] = ()
    segment_states: tuple[RoadSegmentState, ...] = ()
    active_incidents: tuple[Incident, ...] = ()
    environment: tuple[EnvironmentConditions, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> NetworkState:
        if self.as_of.tzinfo is None or self.assembled_at.tzinfo is None:
            msg = "snapshot timestamps must be timezone-aware"
            raise ValueError(msg)
        jids = [j.junction_id for j in self.junction_states]
        if len(set(jids)) != len(jids):
            msg = "duplicate junction_id in network state"
            raise ValueError(msg)
        sids = [s.segment_id for s in self.segment_states]
        if len(set(sids)) != len(sids):
            msg = "duplicate segment_id in network state"
            raise ValueError(msg)
        return self

    # --- freshness summary ------------------------------------------------
    @property
    def stale_junction_ids(self) -> frozenset[str]:
        """Junctions whose evidence exceeded the staleness budget."""
        return frozenset(
            j.junction_id for j in self.junction_states if j.freshness is FreshnessStatus.STALE
        )

    @property
    def unknown_junction_ids(self) -> frozenset[str]:
        """Junctions with no usable evidence at all."""
        return frozenset(
            j.junction_id
            for j in self.junction_states
            if j.freshness in (FreshnessStatus.UNKNOWN, FreshnessStatus.EXPIRED)
        )

    @property
    def is_degraded(self) -> bool:
        """Whether any element lacks fresh evidence.

        A degraded snapshot does not block a decision -- withholding a route from
        an ambulance because one camera is offline would be its own kind of
        failure -- but it must lower reported confidence and appear as a warning
        on the recommendation.
        """
        return bool(self.stale_junction_ids or self.unknown_junction_ids)

    @property
    def blocked_segment_ids(self) -> frozenset[str]:
        """Segments that are impassable in this snapshot.

        The hard-constraint input to route pruning. Derived from segment state
        and from every active incident, so a blockage cannot be missed because
        only one of the two sources recorded it.
        """
        from_state = frozenset(s.segment_id for s in self.segment_states if s.is_blocked)
        from_incidents: frozenset[str] = frozenset()
        for inc in self.active_incidents:
            from_incidents |= inc.hard_blocked_segment_ids
        return from_state | from_incidents

    def junction_state(self, junction_id: str) -> JunctionState:
        """Return one junction's state, raising ``KeyError`` if absent."""
        for js in self.junction_states:
            if js.junction_id == junction_id:
                return js
        raise KeyError(junction_id)

    def segment_state(self, segment_id: str) -> RoadSegmentState:
        """Return one segment's state, raising ``KeyError`` if absent."""
        for ss in self.segment_states:
            if ss.segment_id == segment_id:
                return ss
        raise KeyError(segment_id)
