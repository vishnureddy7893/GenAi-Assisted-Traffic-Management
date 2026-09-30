"""Where an observation area physically is.

The gap this closes
-------------------
An ``observation_area_id`` identifies *a camera's field of view* -- the thing
Phase 4 keys its analytics by and Phase 5 keys its verdicts by. It deliberately
says nothing about geography: two cameras with the same id would be the same
view, but an id alone cannot be placed on a map.

An :class:`ObservationSite` is the record that places one. It is the surveyed
answer to "where is the junction this footage came from?", and nothing else.

Why it is a separate model from MediaAsset
------------------------------------------
A media asset describes a *file*: its bytes, its codec, its dimensions. A site
describes a *place*. The two have different lifetimes and different cardinality
-- one junction outlives many uploads of it, and re-uploading tomorrow's footage
of the same junction must not create a second, contradictory location. Folding
latitude onto ``MediaAsset`` would tie the place to the file, so deleting the
file would delete the survey, and two files of one junction could disagree about
where it is.

Why it is a survey, not an inference
------------------------------------
The same discipline as :mod:`app.routing.mapping`: a location is recorded by
somebody who knows it, never guessed from a filename, an upload order or an
image's contents. A junction pinned to the wrong coordinates is a specific false
claim, and every downstream use of it -- distance, nearest-facility, and any
future real-world routing -- would inherit that claim silently.

Nothing here routes, geocodes or contacts any mapping provider. This module
records a coordinate and its provenance; that is its whole scope.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Annotated

from pydantic import Field, model_validator

from app.domain.base import DomainModel
from app.domain.network import GeoPoint
from app.domain.provenance import Provenance

#: Identifier of a camera's field of view. Same width as the analytics and
#: mapping fields it must join against, so a value valid in one is valid in all.
ObservationAreaId = Annotated[str, Field(min_length=1, max_length=120)]

#: Prefix under which an observation area is named after the media that produced
#: it. Defined once here because analytics, the media API and the site registry
#: must all agree on it: a drift of one character silently unjoins the three.
OBSERVATION_AREA_PREFIX = "observation-area"

#: Upper bound on a claimed position accuracy, in metres. Beyond roughly ten
#: kilometres a fix cannot distinguish one junction from any other in the same
#: city, so a value that large is a unit mistake rather than a weak GPS lock.
MAX_ACCURACY_M = 10_000.0


def observation_area_id_for_media(media_id: str) -> str:
    """Observation area id for footage ingested under ``media_id``.

    One camera, one upload, one field of view: the analytics pipeline names the
    area after the media it was computed from, and this is that same rule stated
    once so a site created at upload lands on exactly the id the verdicts will
    later carry.
    """
    return f"{OBSERVATION_AREA_PREFIX}:{media_id}"


class ObservationSite(DomainModel):
    """The surveyed location of the junction or camera an observation area sees.

    Keyed by ``observation_area_id`` and nothing else. That key is not changed,
    reinterpreted or widened here -- a site is additional knowledge *about* an
    existing identifier, never a new meaning for it.
    """

    observation_area_id: ObservationAreaId
    """The area this location belongs to. Matches
    ``TrafficAnalytics.observation_area_id`` and
    ``ObservationBinding.observation_area_id`` exactly, so the three join
    without a translation step."""

    location: GeoPoint
    """WGS-84 position of the junction or camera. Required: a site whose whole
    purpose is to say where something is cannot omit where it is. Reuses the
    network's :class:`GeoPoint`, so a latitude is bounded identically here and
    on every segment polyline."""

    bearing_degrees: Annotated[float, Field(ge=0.0, lt=360.0)] | None = None
    """Compass direction the camera faces: 0 = true north, increasing clockwise.
    360 is excluded because it is 0, and admitting both would make two equal
    bearings compare unequal. ``None`` means the direction was not surveyed,
    which is not the same as north."""

    accuracy_m: Annotated[float, Field(gt=0.0, le=MAX_ACCURACY_M)] | None = None
    """Radius, in metres, within which the true position lies. Strictly positive:
    a zero-metre accuracy is a claim of perfect knowledge that no fix supports.
    ``None`` means unreported -- never assume it is good."""

    label: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    """Human name for the place, e.g. ``Benz Circle, Vijayawada``. Display only;
    nothing is ever resolved, matched or geocoded from it."""

    provenance: Provenance
    """Who recorded this position, from what, and when. Required for the same
    reason every measurement in this system carries it: a coordinate asserted
    with no source is indistinguishable from one that was guessed, and a guessed
    junction location is exactly the failure this model exists to prevent."""

    recorded_at: dt.datetime
    """When this record was first written."""

    updated_at: dt.datetime
    """When it was last replaced. Equal to ``recorded_at`` until a re-survey
    moves the site, so "has this ever been corrected?" stays answerable."""

    @property
    def latitude(self) -> float:
        """Convenience accessor for the surveyed latitude."""
        return self.location.latitude

    @property
    def longitude(self) -> float:
        """Convenience accessor for the surveyed longitude."""
        return self.location.longitude

    @property
    def was_updated(self) -> bool:
        """Whether this site has been re-recorded since it was first written."""
        return self.updated_at > self.recorded_at


class BindingConfirmationMethod(StrEnum):
    """How a human arrived at a site-to-junction binding.

    Recorded because the three differ in how much they are worth as evidence,
    and a reviewer must be able to tell them apart. There is deliberately no
    ``AUTOMATIC`` member: a binding nothing human decided is exactly what this
    design refuses to create.
    """

    ACCEPTED_NEAREST = "accepted_nearest"
    """The operator accepted the nearest-junction candidate as offered."""

    ALIGNED_WITH_BEARING = "aligned_with_bearing"
    """The operator confirmed a directed segment that the camera's surveyed
    bearing points along.

    Records that the geometry agreed, not that the geometry decided. It cannot
    have decided: the opposite direction of the same road shares that line, so
    this value is reachable for either of them and says only that the operator's
    choice is consistent with where the camera is pointed."""

    CHOSE_DIFFERENT = "chose_different"
    """The operator confirmed a junction that was *not* the nearest. Kept
    distinct because it is the case where human judgement overrode the geometry,
    and the reason for that is worth having on record."""

    SURVEYED = "surveyed"
    """Established independently of any candidate -- from a site visit or a
    mounting record -- and entered directly."""


class SegmentDirection(StrEnum):
    """Which way a directed segment runs relative to the junction it touches.

    Recorded on a confirmed binding rather than left to be re-derived. A segment
    id is directed by construction, but "SEG-21 at J1" reads as a road name to a
    human, and the two directions of one road are the whole difference between a
    queue arriving at the junction and a queue leaving it. Stating it makes the
    record answerable without the map in hand.
    """

    INCOMING = "incoming"
    """Traffic on this segment travels *toward* the junction: the segment's
    ``to_junction_id`` is the confirmed junction. This is what an approach is."""

    OUTGOING = "outgoing"
    """Traffic travels *away* from the junction: ``from_junction_id`` is it."""


class BindingConfirmation(DomainModel):
    """The human act that turned a candidate into a binding.

    Separate from :class:`~app.domain.provenance.Provenance` rather than folded
    into it, because the two answer different questions. Provenance says how the
    *value* came to exist; this says who decided to accept it and what they were
    looking at when they did. A later reviewer asking "was this binding checked,
    or did it fall out of a distance sort?" is answered here and nowhere else.
    """

    confirmed_by: Annotated[str, Field(min_length=1, max_length=200)]
    """Who confirmed it. An operator id or name -- never a component name, which
    would mean no human was involved."""

    confirmed_at: dt.datetime

    method: BindingConfirmationMethod

    candidate_distance_m: Annotated[float, Field(ge=0.0)] | None = None
    """Distance from the site to the confirmed junction at the moment of
    confirmation -- the number the operator was shown. Retained so the decision
    can be reviewed against the evidence it was made on, even after the site is
    re-surveyed and the distance changes."""

    site_accuracy_m: Annotated[float, Field(gt=0.0, le=MAX_ACCURACY_M)] | None = None
    """The site's position accuracy at confirmation time. ``None`` means it was
    unreported, which is not the same as good: a confirmation made against an
    unquantified position is weaker evidence and stays visibly so."""

    was_nearest_candidate: bool | None = None
    """Whether the confirmed junction was the nearest one. ``None`` when no
    candidate evaluation backed the confirmation at all."""

    notes: Annotated[str, Field(max_length=1000)] | None = None
    """Why this junction, in the operator's words. The thing a reviewer needs in
    order to disagree."""

    @model_validator(mode="after")
    def _overriding_the_geometry_needs_a_reason(self) -> BindingConfirmation:
        if self.method is BindingConfirmationMethod.CHOSE_DIFFERENT and not self.notes:
            msg = (
                "confirming a junction that was not the nearest requires notes "
                "saying why; an unexplained override is indistinguishable from a "
                "misclick, and it is the case a reviewer is most likely to question"
            )
            raise ValueError(msg)
        return self


class ConfirmedSiteBinding(DomainModel):
    """One observation area bound to one junction, by a human, on one network.

    This is the *record* of the decision. The binding it produces for the
    routing layer is an ordinary
    :class:`~app.routing.mapping.ObservationBinding`; keeping the record
    separate means the evidence and the confirmation survive even though the
    binding itself carries only what the routing layer needs.
    """

    observation_area_id: ObservationAreaId
    junction_id: Annotated[str, Field(min_length=1, max_length=64)]

    network_id: Annotated[str, Field(min_length=1, max_length=64)]
    """Which network this junction id means something on. A binding carried onto
    a different map would name a junction that is somewhere else entirely, so
    the network is recorded and checked rather than assumed."""

    network_version: Annotated[str, Field(max_length=50)] | None = None
    """Version of that network at confirmation time. A mismatch does not
    invalidate the binding -- a re-surveyed map usually keeps its junction ids --
    but it is reported, because "confirmed against an older map" is something a
    reviewer is entitled to know."""

    site_location: GeoPoint
    """Where the site was when this was confirmed. Kept so that re-surveying the
    site later does not silently rewrite the evidence behind a past decision."""

    confirmation: BindingConfirmation
    provenance: Provenance
    """Source ``observed``: a human decision, directly recorded. Never
    ``derived`` -- the *candidate* is derived from distance, but the binding is
    not, and labelling it derived would credit the geometry with a choice it did
    not make."""

    segment_id: Annotated[str, Field(min_length=1, max_length=64)] | None = None
    """The directed segment the operator confirmed this camera watches, once one
    has been confirmed.

    Optional because it is a *refinement*: the junction binding is complete and
    useful on its own, and a camera overlooking a whole intersection may never
    gain one. ``None`` means the segment has not been decided yet -- never that
    the camera watches no road."""

    segment_direction: SegmentDirection | None = None
    """Whether that segment runs toward or away from the confirmed junction.
    Present exactly when ``segment_id`` is."""

    approach_id: Annotated[str, Field(min_length=1, max_length=64)] | None = None
    """The approach the operator additionally named, when they named one. Only
    an incoming segment has an approach. Left ``None`` rather than looked up:
    filling it in automatically would put a value on an ``observed`` record that
    no human actually stated."""

    segment_confirmation: BindingConfirmation | None = None
    """The second human act, kept separate from the junction's.

    Two decisions were made, potentially by different people on different days,
    and each carries its own method, notes and evidence. Overwriting the
    junction confirmation with the segment one would erase the record of who
    placed the camera at this intersection in the first place."""

    @model_validator(mode="after")
    def _segment_fields_travel_together(self) -> ConfirmedSiteBinding:
        """A confirmed segment must carry its direction and its confirmation.

        The three describe one decision. A ``segment_id`` with no confirmation
        would be a binding nobody made; a confirmation with no segment would be
        a decision about nothing. Either alone is a record that cannot be
        reviewed, which is the failure this model exists to prevent.
        """
        present = (
            self.segment_id is not None,
            self.segment_direction is not None,
            self.segment_confirmation is not None,
        )
        if any(present) and not all(present):
            msg = (
                f"binding for {self.observation_area_id!r} has an incomplete "
                "segment refinement: segment_id, segment_direction and "
                "segment_confirmation are one decision and must be present "
                "together or absent together"
            )
            raise ValueError(msg)

        if self.approach_id is not None and self.segment_id is None:
            msg = (
                f"binding for {self.observation_area_id!r} names an approach "
                "without the segment it belongs to; an approach is one direction "
                "of one road, and naming it alone says which lane without saying "
                "which road"
            )
            raise ValueError(msg)

        if (
            self.approach_id is not None
            and self.segment_direction is SegmentDirection.OUTGOING
        ):
            msg = (
                f"binding for {self.observation_area_id!r} names an approach on an "
                "outgoing segment; an approach is traffic arriving at the junction, "
                "so only an incoming segment has one"
            )
            raise ValueError(msg)

        return self

    @property
    def is_segment_bound(self) -> bool:
        """Whether the junction binding has been refined to one directed road."""
        return self.segment_id is not None

    @property
    def coverage_note(self) -> str:
        """One-line summary of the decision, for the binding it produces."""
        distance = self.confirmation.candidate_distance_m
        measured = f"{distance:.1f} m from the site" if distance is not None else (
            "no distance recorded"
        )
        summary = (
            f"Confirmed by {self.confirmation.confirmed_by} "
            f"({self.confirmation.method.value}); {measured}."
        )
        if self.segment_confirmation is not None and self.segment_direction is not None:
            summary += (
                f" Segment {self.segment_id} ({self.segment_direction.value} at "
                f"{self.junction_id}) confirmed by "
                f"{self.segment_confirmation.confirmed_by}."
            )
        return summary
