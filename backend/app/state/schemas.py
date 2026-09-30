"""Interpreted traffic state schemas.

Phase 4 answers *what was measured*. These types answer *what it means*, and the
separation is deliberate enough to be visible in the type names: nothing here is
a measurement, and every field is derived from one.

Why an interpretation carries its own inputs
--------------------------------------------
A congestion level on its own is unarguable. "Heavy" is either accepted or
rejected on trust, and trust is exactly what a traffic operator should not have
to extend to a number that will influence an ambulance's route.

So :class:`CongestionAssessment` carries the signals it used, the score each
one contributed, the weights, and the exact threshold set in force -- enough to
recompute the verdict by hand. A disagreement then becomes a specific argument
about a threshold rather than a general suspicion about the system.

Why level of service is absent
------------------------------
:class:`~app.domain.enums.LevelOfService` is *not* assigned here, and the
omission is deliberate. At a signalized junction LOS is a function of control
delay per vehicle. Nothing in this layer measures control delay, and mapping a
vehicle count or an occupancy ratio onto an HCM letter grade would put a
familiar, authoritative-looking label on a quantity that never entered the
calculation. It stays ``None`` until a component genuinely computes delay.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from itertools import pairwise
from typing import Annotated

from pydantic import Field, model_validator

from app.analytics.schemas import (
    AnalyticsProvenance,
    MetricAvailability,
    ObservationMode,
    QualityGrade,
)
from app.domain.base import DomainModel
from app.domain.enums import CongestionLevel, FreshnessStatus
from app.domain.state import StateAssemblyPolicy

StateId = Annotated[str, Field(min_length=1, max_length=64)]


class EvidenceBasis(StrEnum):
    """What kind of evidence a congestion verdict rests on.

    Reported because the two bases are not interchangeable. A physical basis
    rests on metres and km/h and can be compared against published traffic
    engineering thresholds. An image-space basis rests on how much of a frame is
    covered and how fast things move across it, which is camera-specific: the
    same road under the same conditions produces different numbers from a
    different mounting. Presenting the two as one number would make an
    uncalibrated camera look as authoritative as a surveyed one.
    """

    PHYSICAL = "physical"
    """Derived from calibrated quantities: density in veh/km/lane, speed in km/h."""

    IMAGE_SPACE = "image_space"
    """Derived from frame-relative quantities only. Camera-specific, and never
    to be compared across cameras as though it were a physical measurement."""

    INSUFFICIENT = "insufficient"
    """Not enough usable evidence to classify. Produces ``UNKNOWN``, never a
    default band."""


class SignalName(StrEnum):
    """The contributing signals a verdict may be built from."""

    DENSITY = "density"
    """Physical. Vehicles per km per lane against a jam-density reference."""

    MEAN_SPEED = "mean_speed"
    """Physical. Mean speed against a free-flow reference."""

    IMAGE_OCCUPANCY = "image_occupancy"
    """Image-space. Union of road-user box area over frame area."""

    IMAGE_SPEED = "image_speed"
    """Image-space. Frame diagonals per second against a free-flow reference."""

    STATIONARY_FRACTION = "stationary_fraction"
    """Basis-independent. Stationary vehicles over observed vehicles."""


class CongestionSignal(DomainModel):
    """One contributing signal, with its raw value and normalised score.

    Retained even when unavailable. An assessment built on two signals out of
    four is a weaker claim than one built on all four, and the only way a reader
    can tell the difference is if the missing ones are still listed with their
    reason.
    """

    name: SignalName
    availability: MetricAvailability
    raw_value: float | None = None
    unit: Annotated[str, Field(max_length=40)] | None = None

    score: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    """Normalised contribution: 0.0 is free-flowing, 1.0 is maximally congested."""

    weight: Annotated[float, Field(ge=0.0)] = 0.0
    """Relative influence on the index. Zero when the signal is unavailable."""

    reason: Annotated[str, Field(max_length=300)] | None = None
    is_image_space: bool = False

    @model_validator(mode="after")
    def _availability_matches_content(self) -> CongestionSignal:
        if self.availability is MetricAvailability.AVAILABLE:
            if self.score is None or self.raw_value is None:
                msg = f"signal {self.name.value!r} is AVAILABLE but carries no value or score"
                raise ValueError(msg)
        else:
            if self.score is not None or self.raw_value is not None:
                msg = (
                    f"signal {self.name.value!r} is unavailable but carries a value; an "
                    "unavailable signal must not contribute a number"
                )
                raise ValueError(msg)
            if not self.reason:
                msg = f"signal {self.name.value!r} is unavailable and must state why"
                raise ValueError(msg)
            if self.weight != 0.0:
                msg = f"signal {self.name.value!r} is unavailable and must carry zero weight"
                raise ValueError(msg)
        return self


class CongestionThresholds(DomainModel):
    """The exact band cut points and reference values used for one verdict.

    Recorded *inside* the assessment rather than read from configuration when the
    result is displayed. A verdict produced under one threshold set and rendered
    against another is unreproducible, and the drift would be invisible.

    Cut points are lower bounds: a band holds from its own cut point up to the
    next one, so ``index == light`` is LIGHT and not FREE_FLOW.
    """

    light: Annotated[float, Field(ge=0.0, le=1.0)]
    moderate: Annotated[float, Field(ge=0.0, le=1.0)]
    heavy: Annotated[float, Field(ge=0.0, le=1.0)]
    severe: Annotated[float, Field(ge=0.0, le=1.0)]
    gridlock: Annotated[float, Field(ge=0.0, le=1.0)]

    occupancy_reference: Annotated[float, Field(gt=0.0, le=1.0)]
    """Image occupancy ratio treated as fully congested."""

    stationary_reference: Annotated[float, Field(gt=0.0, le=1.0)]
    """Stationary fraction treated as fully congested."""

    image_speed_reference: Annotated[float, Field(gt=0.0)]
    """Image speed, in frame diagonals per second, treated as free-flowing."""

    density_reference_veh_per_km_per_lane: Annotated[float, Field(gt=0.0)]
    """Jam density: the density treated as fully congested."""

    free_flow_speed_kph: Annotated[float, Field(gt=0.0)]
    """Speed treated as fully free-flowing."""

    @model_validator(mode="after")
    def _bands_are_strictly_increasing(self) -> CongestionThresholds:
        bands = (
            ("light", self.light),
            ("moderate", self.moderate),
            ("heavy", self.heavy),
            ("severe", self.severe),
            ("gridlock", self.gridlock),
        )
        for (lower_name, lower), (upper_name, upper) in pairwise(bands):
            if upper <= lower:
                msg = (
                    f"congestion band cut points must strictly increase, but "
                    f"{upper_name}={upper} does not exceed {lower_name}={lower}; "
                    "overlapping bands would make the classification ambiguous"
                )
                raise ValueError(msg)
        return self


class CongestionAssessment(DomainModel):
    """A congestion verdict, with everything needed to recompute it."""

    level: CongestionLevel
    index: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    """Weighted congestion index in [0, 1]. ``None`` exactly when the level is
    ``UNKNOWN``: an index without a band, or a band without an index, would be a
    verdict that cannot be checked."""

    basis: EvidenceBasis
    signals: tuple[CongestionSignal, ...] = ()
    thresholds: CongestionThresholds

    confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    reason: Annotated[str, Field(max_length=300)] | None = None
    """Required when the level is ``UNKNOWN``. An unexplained UNKNOWN is
    indistinguishable from a crash."""

    method: Annotated[str, Field(min_length=1, max_length=120)]
    method_version: Annotated[str, Field(min_length=1, max_length=40)]

    @model_validator(mode="after")
    def _unknown_is_explained_and_unscored(self) -> CongestionAssessment:
        if self.level is CongestionLevel.UNKNOWN:
            if self.index is not None:
                msg = (
                    "an UNKNOWN congestion level must not carry an index; a scored "
                    "UNKNOWN invites a reader to treat it as a band"
                )
                raise ValueError(msg)
            if not self.reason:
                msg = "an UNKNOWN congestion level must state why it could not be classified"
                raise ValueError(msg)
        elif self.index is None:
            msg = f"congestion level {self.level.value!r} requires the index that produced it"
            raise ValueError(msg)
        return self

    @property
    def is_classified(self) -> bool:
        """Whether a usable band was produced."""
        return self.level is not CongestionLevel.UNKNOWN


class IntersectionTrafficState(DomainModel):
    """Interpreted traffic state for one observation area at one instant.

    Named *intersection* rather than *junction* on purpose. A
    :class:`~app.domain.network.Junction` is a surveyed node with named
    approaches; an observation area is one camera's field of view. Until a
    network mapping exists they are not the same thing, and using the junction
    vocabulary here would quietly claim a survey that has not happened.
    """

    state_id: StateId
    observation_area_id: Annotated[str, Field(min_length=1, max_length=120)]
    mode: ObservationMode

    as_of: dt.datetime
    """The instant the underlying measurements describe, on the source clock.
    For a recorded replay this is media time, so an interpretation of year-old
    footage is dated a year ago and cannot masquerade as current."""

    assessed_at: dt.datetime

    congestion: CongestionAssessment
    vehicle_count: Annotated[int, Field(ge=0)]
    stationary_vehicle_count: Annotated[int, Field(ge=0)] | None = None
    quality_grade: QualityGrade

    analytics_provenance: AnalyticsProvenance
    """The measurement set this interpretation was computed from, so the chain
    back to a model, a media file and a frame range stays unbroken."""

    schema_version: Annotated[str, Field(min_length=1, max_length=40)]

    @model_validator(mode="after")
    def _tz_aware(self) -> IntersectionTrafficState:
        if self.as_of.tzinfo is None or self.assessed_at.tzinfo is None:
            msg = "interpreted state timestamps must be timezone-aware"
            raise ValueError(msg)
        return self


class IntersectionStateElement(DomainModel):
    """One intersection's state inside a fused snapshot, with its freshness."""

    state: IntersectionTrafficState
    freshness: FreshnessStatus
    evidence_age_s: Annotated[float, Field(ge=0.0)]
    """Age of this element's evidence relative to the snapshot instant."""

    @property
    def is_usable_evidence(self) -> bool:
        """Whether this element may support a decision."""
        return self.freshness in (FreshnessStatus.FRESH, FreshnessStatus.STALE)


class NetworkTrafficState(DomainModel):
    """Every interpreted observation area, fused at one instant.

    Distinct from :class:`~app.domain.state.NetworkState`, and deliberately so.
    That type describes a *surveyed network*: junctions and directed segments
    from a road graph. This one describes the set of observation areas that
    happen to have been analysed, which is all that can be honestly assembled
    before a network graph is loaded.

    The staleness policy is reused unchanged from
    :class:`~app.domain.state.StateAssemblyPolicy`, because the problem is
    identical: elements observed at different instants must not be fused into a
    snapshot that hides the spread.
    """

    state_id: StateId
    as_of: dt.datetime
    """The newest element's instant. Element ages are measured against this
    rather than against wall-clock time, so replaying recorded footage does not
    mark every element expired the moment it is analysed."""

    assembled_at: dt.datetime
    policy: StateAssemblyPolicy
    elements: tuple[IntersectionStateElement, ...] = ()
    schema_version: Annotated[str, Field(min_length=1, max_length=40)]

    @model_validator(mode="after")
    def _validate(self) -> NetworkTrafficState:
        if self.as_of.tzinfo is None or self.assembled_at.tzinfo is None:
            msg = "snapshot timestamps must be timezone-aware"
            raise ValueError(msg)
        area_ids = [element.state.observation_area_id for element in self.elements]
        if len(set(area_ids)) != len(area_ids):
            msg = "duplicate observation_area_id in network traffic state"
            raise ValueError(msg)
        return self

    @property
    def stale_area_ids(self) -> frozenset[str]:
        """Areas whose evidence exceeded the staleness budget but is still usable."""
        return frozenset(
            element.state.observation_area_id
            for element in self.elements
            if element.freshness is FreshnessStatus.STALE
        )

    @property
    def expired_area_ids(self) -> frozenset[str]:
        """Areas whose evidence is too old to use."""
        return frozenset(
            element.state.observation_area_id
            for element in self.elements
            if element.freshness in (FreshnessStatus.EXPIRED, FreshnessStatus.UNKNOWN)
        )

    @property
    def unclassified_area_ids(self) -> frozenset[str]:
        """Areas whose congestion could not be determined.

        Surfaced separately from staleness: fresh evidence that was too poor to
        interpret and good evidence that arrived too late are different faults
        with different fixes.
        """
        return frozenset(
            element.state.observation_area_id
            for element in self.elements
            if not element.state.congestion.is_classified
        )

    @property
    def is_degraded(self) -> bool:
        """Whether any element lacks fresh, classified evidence.

        Does not block a decision -- withholding a route because one camera
        lagged would be its own kind of failure -- but must lower reported
        confidence and appear as a warning.
        """
        return bool(self.stale_area_ids or self.expired_area_ids or self.unclassified_area_ids)

    def element(self, observation_area_id: str) -> IntersectionStateElement:
        """Return one element, raising ``KeyError`` if absent."""
        for item in self.elements:
            if item.state.observation_area_id == observation_area_id:
                return item
        raise KeyError(observation_area_id)
