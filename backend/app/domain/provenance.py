"""Provenance-carrying measurements: the structural defence against fabrication.

The requirement is "do not fabricate numerical values". Enforcing that by
discipline alone does not survive contact with a deadline, an LLM, or a
plausible-looking default. So it is enforced by the type system instead:

**Every quantity that could be invented is a** :class:`Measurement` **, and a
Measurement cannot be constructed without a** :class:`Provenance` **that states
where the number came from.**

There is deliberately no ``MOCK``, ``PLACEHOLDER`` or ``DEFAULT`` member of
:class:`ValueSource`. If a component does not know a value, the correct
representation is ``None`` -- an explicitly absent measurement -- not a zero.
Zero is a legitimate traffic reading (an empty approach at 04:00) and must never
be used to mean "unknown"; conflating the two is how a routing engine concludes
that a gridlocked road is empty.

The distinction between ``OBSERVED``, ``DERIVED``, ``PREDICTED``, ``SIMULATED``,
``STATIC`` and ``ASSUMED`` is not bookkeeping. It determines what the system is
permitted to say: a recommendation narrative may state an observed count as
fact, must hedge a predicted one, and must label a simulated one as a modelled
outcome. Phase 12 relies on exactly this field to keep the LLM honest.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Annotated, Generic, TypeVar

from pydantic import Field, model_validator

from app.domain.base import DomainModel

T = TypeVar("T")


class ValueSource(StrEnum):
    """How a value came to exist. Closed set, by design."""

    OBSERVED = "observed"
    """Measured directly from input data (CV detection, dataset column, sensor)."""

    DERIVED = "derived"
    """Computed deterministically from other values by a documented formula.
    Requires ``input_refs`` naming what it was computed from."""

    PREDICTED = "predicted"
    """Output of a trained model. Requires ``producer`` and ``producer_version``
    identifying the exact model artifact, and should carry a confidence."""

    SIMULATED = "simulated"
    """Produced by the traffic simulator. Never presentable as a measurement of
    the real world -- it is the modelled outcome of a hypothetical policy."""

    STATIC = "static"
    """From the map, survey or configuration (segment length, lane count,
    design capacity). Does not change between snapshots."""

    ASSUMED = "assumed"
    """An explicit modelling assumption that no data supports. Requires ``notes``
    justifying it. Surfaced to the user as an assumption, never as a fact."""


class Unit(StrEnum):
    """Physical unit of a measurement. SI where a choice exists.

    Units are explicit because the classic failure here is mixing km/h with m/s
    in a travel-time computation and getting an ETA wrong by 3.6x -- an error
    that looks entirely plausible in a dashboard.
    """

    METER = "m"
    KILOMETER = "km"
    SECOND = "s"
    MINUTE = "min"
    HOUR = "h"
    METER_PER_SECOND = "m/s"
    KILOMETER_PER_HOUR = "km/h"
    VEHICLE = "veh"
    VEHICLE_PER_HOUR = "veh/h"
    VEHICLE_PER_HOUR_PER_LANE = "veh/h/ln"
    VEHICLE_PER_KILOMETER = "veh/km"
    VEHICLE_PER_KILOMETER_PER_LANE = "veh/km/ln"
    PERSON = "person"
    PERSON_PER_SQUARE_METER = "person/m2"
    RATIO = "ratio"
    """Dimensionless 0..1, e.g. volume-to-capacity or occupancy."""

    PERCENT = "%"
    PIXEL = "px"
    PIXEL_PER_SECOND = "px/s"
    """Only valid when camera calibration is unavailable. A pixel rate must never
    be presented as a speed; it exists so an uncalibrated pipeline can report
    what it actually measured instead of inventing metric units."""

    COUNT = "count"


class Provenance(DomainModel):
    """Where a value came from, who produced it, and when."""

    source: ValueSource
    producer: Annotated[str, Field(min_length=1, max_length=200)]
    """Stable identifier of the producing component, e.g. ``cv.yolo_detector``,
    ``analytics.queue_estimator``, ``map.osm_import``, ``sim.sumo_runner``."""

    producer_version: Annotated[str, Field(min_length=1, max_length=100)]
    """Version of that component or model artifact. Required for replay."""

    observed_at: dt.datetime
    """When the underlying *evidence* was captured -- the frame timestamp, not
    the processing time. Staleness is measured against this."""

    computed_at: dt.datetime
    """When this value was produced from that evidence."""

    confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    """Producer-reported confidence, where the producer has a calibrated notion
    of one. ``None`` means "not reported", which is not the same as low."""

    input_refs: tuple[str, ...] = ()
    """Identifiers of the upstream facts this value was computed from. Makes the
    derivation chain auditable end to end."""

    notes: Annotated[str, Field(max_length=1000)] | None = None
    """Required for ``ASSUMED``: the justification for the assumption."""

    @model_validator(mode="after")
    def _check_source_obligations(self) -> Provenance:
        """Enforce the obligations that each source class carries."""
        if self.observed_at.tzinfo is None or self.computed_at.tzinfo is None:
            msg = "provenance timestamps must be timezone-aware"
            raise ValueError(msg)

        if self.source is ValueSource.ASSUMED and not self.notes:
            msg = (
                "ValueSource.ASSUMED requires 'notes' justifying the assumption; "
                "an unjustified assumption is indistinguishable from a fabricated value"
            )
            raise ValueError(msg)

        if self.source is ValueSource.DERIVED and not self.input_refs:
            msg = (
                "ValueSource.DERIVED requires 'input_refs' identifying what the "
                "value was derived from, so the derivation chain stays auditable"
            )
            raise ValueError(msg)

        return self

    @property
    def is_factual(self) -> bool:
        """Whether this value may be stated as fact about the real world.

        Predicted, simulated and assumed values may not: they must be presented
        with their modality intact ("predicted to", "in simulation", "assuming").
        """
        return self.source in (ValueSource.OBSERVED, ValueSource.DERIVED, ValueSource.STATIC)


class Measurement(DomainModel, Generic[T]):
    """A value, its unit, and its provenance. The only way to carry a quantity."""

    value: T
    unit: Unit
    provenance: Provenance

    @property
    def is_factual(self) -> bool:
        """Delegates to provenance; convenience for narrative generation."""
        return self.provenance.is_factual

    def age_seconds_at(self, when: dt.datetime) -> float:
        """Age of the underlying evidence at instant ``when``.

        Negative values indicate the evidence is timestamped in the future,
        which means clock skew between the source and this host. Surfaced rather
        than clamped, because silently clamping hides a real fault.
        """
        if when.tzinfo is None:
            msg = "reference instant must be timezone-aware"
            raise ValueError(msg)
        return (when - self.provenance.observed_at).total_seconds()


#: Convenience aliases for the shapes used most often across the domain.
FloatMeasurement = Measurement[float]
IntMeasurement = Measurement[int]


def make_provenance(
    *,
    source: ValueSource,
    producer: str,
    producer_version: str,
    observed_at: dt.datetime,
    computed_at: dt.datetime | None = None,
    confidence: float | None = None,
    input_refs: tuple[str, ...] = (),
    notes: str | None = None,
) -> Provenance:
    """Build a :class:`Provenance`, defaulting ``computed_at`` to ``observed_at``.

    ``computed_at`` deliberately does not default to "now": a recorded pipeline
    replaying last month's video must stamp values on the media timeline, not on
    wall-clock time. Callers that want wall-clock time pass it explicitly from
    their :class:`~app.core.clock.Clock`.
    """
    return Provenance(
        source=source,
        producer=producer,
        producer_version=producer_version,
        observed_at=observed_at,
        computed_at=computed_at if computed_at is not None else observed_at,
        confidence=confidence,
        input_refs=input_refs,
        notes=notes,
    )
