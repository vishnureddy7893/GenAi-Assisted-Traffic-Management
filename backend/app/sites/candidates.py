"""Which junction is nearest a site -- offered, never applied.

What this computes
------------------
Given an :class:`~app.domain.site.ObservationSite` and the loaded
:class:`~app.domain.network.RoadNetwork`, the great-circle distance from the
site to every junction that has a surveyed position, and the nearest of them.

What it deliberately does not do
--------------------------------
It does not bind anything. Nearest is not the same as watched: a camera on a
mast between two closely spaced junctions is nearest to one and pointed at the
other, and a coordinate typed one digit wrong is nearest to a junction on a
different road. Proximity is *evidence for* a binding and the whole of the
argument against inferring one -- see :mod:`app.routing.mapping`, where the rule
that evidence reaches a road only through an explicit survey is set out.

So the output is a :class:`SiteBindingCandidate`: a proposal, carrying the
distance it was ranked by and every reason it might be wrong, addressed to a
human who will accept it or not. ``requires_human_confirmation`` is fixed True
and cannot be set otherwise, the same way a signal advisory cannot declare
itself non-advisory.

Uncertainty is carried, not collapsed
-------------------------------------
Two facts routinely make a nearest-junction ranking meaningless, and both are
reported rather than hidden:

* the distance is inside the site's own position accuracy, so the site could
  equally be standing on the junction or ``accuracy_m`` away from it; and
* the runner-up is no further away than the position accuracy, so which of the
  two is "nearest" is decided by noise.

Either one clears ``is_discriminating``. The candidate is still offered -- an
operator who knows the junction can confirm it regardless -- but it is offered
as a question, not an answer.

Distance
--------
``haversine_m`` from :mod:`app.emergency.zones` is used as-is. It is imported
rather than copied because a second implementation of a distance would
eventually disagree with the first, and rather than moved because relocating it
would edit the exclusion-zone module for no behavioural reason.

Segment candidates, and why bearing does not rank them
------------------------------------------------------
:func:`evaluate_segments` lists the directed segments incident to a junction an
operator has already confirmed. Where the site records ``bearing_degrees`` and
the junction has a position, each segment also carries the compass bearing from
the junction along it and the angle between that and the way the camera faces.

That angle is evidence and nothing more, and the listing is deliberately ordered
by segment id rather than by it. Ordering by bearing would put one segment at
the top of the operator's list, which is selecting by another name -- and the
case where it would be most confidently wrong is the commonest one on the
network: the two directions of a two-way road lie on the same line, so their
bearings from the junction differ by roughly 180 degrees and the *closer* of the
two is decided by which way the camera happens to be pointed, not by which
direction of travel it can actually see. A camera facing oncoming traffic and a
camera filming the backs of departing vehicles are photographed from the same
spot. Only a person who has looked at the footage can say which.

So no segment is pre-selected anywhere in this module, and
``requires_human_confirmation`` is fixed True on the result for the same reason
it is on a junction candidate.
"""

from __future__ import annotations

import datetime as dt
import math
from enum import StrEnum
from typing import Annotated

from pydantic import Field, model_validator

from app.core.versions import APP_VERSION
from app.domain.base import DomainModel
from app.domain.network import GeoPoint, Junction, RoadNetwork, RoadSegment
from app.domain.provenance import Provenance, ValueSource
from app.domain.site import ObservationSite, SegmentDirection
from app.emergency.zones import haversine_m

#: How many alternatives to report alongside the nearest. Enough for an operator
#: to see whether the ranking was close, few enough to stay readable.
RUNNER_UP_COUNT = 3

#: Producer recorded in the candidate's provenance.
CANDIDATE_PRODUCER = "sites.nearest_junction"

#: Producer recorded in a segment candidate set's provenance.
SEGMENT_CANDIDATE_PRODUCER = "sites.incident_segments"

#: Below this angular separation two segment bearings are treated as the same
#: line. The two directions of one road differ by about 180 degrees, so a pair
#: within this tolerance of each other or of 180 degrees cannot be told apart by
#: bearing at all -- which is reported rather than resolved.
COLLINEAR_TOLERANCE_DEG = 15.0


class CandidateStatus(StrEnum):
    """Outcome of evaluating one site against the loaded network.

    Every value is a *complete* answer. None of them stands in for "something
    went wrong": a missing network and a missing site are refused at the API
    boundary with their own distinct error codes, never folded in here as a
    candidate that happens to be empty.
    """

    CANDIDATE_FOUND = "candidate_found"
    """A positioned junction lies within the acceptable range."""

    OUT_OF_RANGE = "out_of_range"
    """The nearest positioned junction is further away than the configured
    range. It is still reported, with its distance, because the useful
    conclusion is usually "the coordinate is wrong" rather than "no junction"."""

    NO_POSITIONED_JUNCTION = "no_positioned_junction"
    """The network has no junction carrying a surveyed position, so nothing can
    be measured against. Distinct from "nothing is near": an unsurveyed map
    cannot answer the question at all."""


class JunctionCandidate(DomainModel):
    """One junction, and how far the site is from it."""

    junction_id: Annotated[str, Field(min_length=1, max_length=64)]
    junction_name: Annotated[str, Field(max_length=200)] | None = None
    distance_m: Annotated[float, Field(ge=0.0)]

    within_site_accuracy: bool = False
    """Whether the distance is inside the site's reported accuracy radius. When
    true the site's position cannot distinguish standing on this junction from
    standing ``accuracy_m`` away from it."""


class SiteBindingCandidate(DomainModel):
    """A proposed site-to-junction binding, pending human confirmation."""

    observation_area_id: Annotated[str, Field(min_length=1, max_length=120)]
    network_id: Annotated[str, Field(min_length=1, max_length=64)]
    network_version: Annotated[str, Field(min_length=1, max_length=50)]

    status: CandidateStatus

    nearest: JunctionCandidate | None = None
    """The closest positioned junction. Present for ``candidate_found`` and for
    ``out_of_range`` alike -- naming the too-distant junction is what lets an
    operator recognise a mistyped coordinate."""

    runners_up: tuple[JunctionCandidate, ...] = ()
    """The next nearest junctions, in order. Offered so the operator can see how
    close the decision was rather than being handed one id to trust."""

    max_distance_m: Annotated[float, Field(gt=0.0)]
    """The acceptable range this evaluation was judged against."""

    site_accuracy_m: Annotated[float, Field(gt=0.0)] | None = None
    """The site's reported position accuracy. ``None`` means unreported, which
    is not the same as good."""

    positioned_junction_count: Annotated[int, Field(ge=0)] = 0
    unpositioned_junction_ids: tuple[str, ...] = ()
    """Junctions with no surveyed position. They were not measured and are not
    candidates -- named rather than dropped, because the nearest junction among
    those that *could* be tested is not necessarily the nearest junction."""

    is_discriminating: bool = True
    """Whether the ranking is decided by more than the position's own noise.
    False when the distance is inside the site accuracy, or when the runner-up
    is within that accuracy of the nearest."""

    requires_human_confirmation: bool = True
    """Always true, and structurally not settable otherwise. A candidate is a
    question put to an operator; nothing downstream may read it as a binding."""

    provenance: Provenance
    """``derived``: this is a distance computation over the site coordinate and
    the network's junction positions, and ``input_refs`` names both."""

    warnings: tuple[str, ...] = ()
    note: str = ""

    @model_validator(mode="after")
    def _confirmation_cannot_be_waived(self) -> SiteBindingCandidate:
        if not self.requires_human_confirmation:
            msg = (
                "requires_human_confirmation cannot be false: a nearest-junction "
                "result is proximity evidence, not a binding, and a system that "
                "could mark it self-confirming would bind cameras to roads nobody "
                "ever checked"
            )
            raise ValueError(msg)
        return self

    @property
    def is_bindable(self) -> bool:
        """Whether there is a junction a human could reasonably confirm."""
        return self.status is CandidateStatus.CANDIDATE_FOUND


def _candidate_provenance(
    site: ObservationSite, network: RoadNetwork, *, computed_at: dt.datetime
) -> Provenance:
    """Provenance for a derived proximity result."""
    return Provenance(
        source=ValueSource.DERIVED,
        producer=CANDIDATE_PRODUCER,
        producer_version=APP_VERSION,
        observed_at=site.provenance.observed_at,
        computed_at=computed_at,
        input_refs=(
            f"observation_site:{site.observation_area_id}",
            f"network:{network.network_id}@{network.version}",
        ),
        notes=(
            "Great-circle distance from the surveyed site coordinate to each "
            "positioned junction. Proximity only; it asserts nothing about what "
            "the camera is pointed at."
        ),
    )


def evaluate_site(
    site: ObservationSite,
    network: RoadNetwork,
    *,
    max_distance_m: float,
    computed_at: dt.datetime,
) -> SiteBindingCandidate:
    """Rank the loaded network's junctions by distance from ``site``.

    Returns a proposal. It never writes anything and never returns a binding.
    """
    measured: list[JunctionCandidate] = []
    unpositioned: list[str] = []
    accuracy = site.accuracy_m

    for junction in network.junctions:
        if junction.position is None:
            unpositioned.append(junction.junction_id)
            continue
        distance = haversine_m(site.location, junction.position)
        measured.append(
            JunctionCandidate(
                junction_id=junction.junction_id,
                junction_name=junction.name,
                distance_m=distance,
                within_site_accuracy=accuracy is not None and distance <= accuracy,
            )
        )

    measured.sort(key=lambda item: (item.distance_m, item.junction_id))
    provenance = _candidate_provenance(site, network, computed_at=computed_at)
    warnings: list[str] = []

    if unpositioned:
        warnings.append(
            f"{len(unpositioned)} junction(s) carry no surveyed position and were "
            "not measured; the nearest junction among those that could be tested "
            "is not necessarily the nearest junction."
        )

    if not measured:
        return SiteBindingCandidate(
            observation_area_id=site.observation_area_id,
            network_id=network.network_id,
            network_version=network.version,
            status=CandidateStatus.NO_POSITIONED_JUNCTION,
            max_distance_m=max_distance_m,
            site_accuracy_m=accuracy,
            positioned_junction_count=0,
            unpositioned_junction_ids=tuple(unpositioned),
            provenance=provenance,
            warnings=tuple(warnings),
            note=(
                f"No junction on network {network.network_id!r} carries a surveyed "
                "position, so no distance could be computed. This is a gap in the "
                "map, not a statement that the site is far from everything."
            ),
        )

    nearest = measured[0]
    runners_up = tuple(measured[1 : 1 + RUNNER_UP_COUNT])

    # The ranking is only meaningful if it survives the position's own noise.
    ambiguous_runner_up = (
        accuracy is not None
        and bool(runners_up)
        and (runners_up[0].distance_m - nearest.distance_m) <= accuracy
    )
    is_discriminating = not (nearest.within_site_accuracy or ambiguous_runner_up)

    if nearest.within_site_accuracy:
        warnings.append(
            f"The site is {nearest.distance_m:.1f} m from {nearest.junction_id}, "
            f"inside its own {accuracy:.1f} m position accuracy. The coordinate "
            "cannot distinguish this junction from anything else within that "
            "radius."
        )
    if ambiguous_runner_up:
        warnings.append(
            f"{runners_up[0].junction_id} is within the site's position accuracy "
            f"of {nearest.junction_id} ({runners_up[0].distance_m:.1f} m against "
            f"{nearest.distance_m:.1f} m). Which one is nearest is decided by "
            "measurement noise, not by geography."
        )

    if nearest.distance_m > max_distance_m:
        warnings.append(
            f"The nearest junction is {nearest.distance_m:.1f} m away, beyond the "
            f"{max_distance_m:.0f} m range within which proximity is treated as "
            "evidence. A distance this large usually means the coordinate is "
            "wrong or the junction is not on this map."
        )
        status = CandidateStatus.OUT_OF_RANGE
        note = (
            f"Nearest junction {nearest.junction_id} is {nearest.distance_m:.1f} m "
            f"from the site, outside the {max_distance_m:.0f} m evidentiary range. "
            "It is reported so the distance can be checked, and it is not offered "
            "as a binding."
        )
    else:
        status = CandidateStatus.CANDIDATE_FOUND
        note = (
            f"Nearest junction {nearest.junction_id} is {nearest.distance_m:.1f} m "
            "from the site. This is proximity evidence only: nothing is bound "
            "until an operator confirms it."
        )

    return SiteBindingCandidate(
        observation_area_id=site.observation_area_id,
        network_id=network.network_id,
        network_version=network.version,
        status=status,
        nearest=nearest,
        runners_up=runners_up,
        max_distance_m=max_distance_m,
        site_accuracy_m=accuracy,
        positioned_junction_count=len(measured),
        unpositioned_junction_ids=tuple(unpositioned),
        is_discriminating=is_discriminating,
        provenance=provenance,
        warnings=tuple(warnings),
        note=note,
    )


def bearing_degrees_between(origin: GeoPoint, target: GeoPoint) -> float:
    """Initial great-circle bearing from ``origin`` to ``target``, in degrees.

    0 is true north, increasing clockwise, normalised to ``[0, 360)`` so it is
    directly comparable with ``ObservationSite.bearing_degrees`` and
    ``Approach.bearing_degrees``, which use the same convention.

    Defined here rather than in :mod:`app.emergency.zones` because that module
    is not part of this step's scope and has no need of a bearing; defined once
    here so the dashboard, the API and the tests cannot each grow their own.
    """
    lat1 = math.radians(origin.latitude)
    lat2 = math.radians(target.latitude)
    dlon = math.radians(target.longitude - origin.longitude)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return math.degrees(math.atan2(x, y)) % 360.0


def angular_separation_degrees(a: float, b: float) -> float:
    """Smallest angle between two compass bearings, in ``[0, 180]``."""
    difference = abs(a - b) % 360.0
    return min(difference, 360.0 - difference)


class SegmentCandidate(DomainModel):
    """One directed segment incident to a junction, offered for selection."""

    segment_id: Annotated[str, Field(min_length=1, max_length=64)]
    segment_name: Annotated[str, Field(max_length=200)] | None = None
    direction: SegmentDirection
    """Whether traffic on it runs toward or away from the junction. The whole
    point of offering directed segments rather than roads."""

    from_junction_id: Annotated[str, Field(min_length=1, max_length=64)]
    to_junction_id: Annotated[str, Field(min_length=1, max_length=64)]
    other_junction_id: Annotated[str, Field(min_length=1, max_length=64)]
    """The junction at the far end. What an operator actually recognises: "the
    arm toward J2", not "SEG-12"."""

    length_m: Annotated[float, Field(gt=0.0)] | None = None

    approach_id: Annotated[str, Field(max_length=64)] | None = None
    """The approach this segment feeds, when the junction surveys one. Only an
    incoming segment has an approach."""

    approach_bearing_degrees: Annotated[float, Field(ge=0.0, lt=360.0)] | None = None
    """The bearing the *survey* recorded for that approach, when it recorded
    one. Surveyed data, distinct from the bearing computed from geometry below,
    and reported separately so the two can disagree visibly."""

    bearing_from_junction_degrees: Annotated[float, Field(ge=0.0, lt=360.0)] | None = (
        None
    )
    """Compass bearing from the junction toward the segment's far end, computed
    from surveyed positions. ``None`` when either end lacks a position."""

    camera_bearing_offset_degrees: Annotated[float, Field(ge=0.0, le=180.0)] | None = (
        None
    )
    """Angle between the way the camera faces and this segment's line. ``None``
    when the site records no bearing or the geometry is unavailable.

    A small offset means the camera is pointed along this road. It does **not**
    mean the camera watches this direction of travel: the opposite segment of
    the same road lies on the same line."""

    @property
    def has_bearing_evidence(self) -> bool:
        """Whether a camera-to-segment angle could be computed at all."""
        return self.camera_bearing_offset_degrees is not None


class SegmentCandidateSet(DomainModel):
    """The directed segments an operator may choose between at one junction."""

    observation_area_id: Annotated[str, Field(min_length=1, max_length=120)]
    junction_id: Annotated[str, Field(min_length=1, max_length=64)]
    junction_name: Annotated[str, Field(max_length=200)] | None = None
    network_id: Annotated[str, Field(min_length=1, max_length=64)]
    network_version: Annotated[str, Field(min_length=1, max_length=50)]

    candidates: tuple[SegmentCandidate, ...] = ()
    """Every directed segment touching the junction, ordered by segment id.

    Ordered by id on purpose: any ordering derived from bearing would present
    one segment as the answer, and the point of this list is that the geometry
    does not know the answer."""

    site_bearing_degrees: Annotated[float, Field(ge=0.0, lt=360.0)] | None = None
    bearing_evidence_available: bool = False
    """Whether any segment carries a camera-to-segment angle. False when the
    site has no surveyed bearing, or the junction has no position -- in which
    case the operator is choosing on local knowledge alone, and should be told
    so rather than shown an empty column."""

    selected_segment_id: None = None
    """Structurally always absent. This type cannot carry a selection, so no
    caller can mistake the offer for a decision."""

    requires_human_confirmation: bool = True
    """Always true, and not settable otherwise."""

    provenance: Provenance
    """``derived``: incidence comes from the network topology and the angles
    from surveyed coordinates. The selection made from this list is a separate,
    human act with its own ``observed`` provenance."""

    warnings: tuple[str, ...] = ()
    note: str = ""

    @model_validator(mode="after")
    def _confirmation_cannot_be_waived(self) -> SegmentCandidateSet:
        if not self.requires_human_confirmation:
            msg = (
                "requires_human_confirmation cannot be false: which direction of "
                "a road a camera watches is not derivable from where the camera "
                "is, and a system that could mark this self-confirming would bind "
                "queues to the wrong side of the street"
            )
            raise ValueError(msg)
        return self

    def candidate(self, segment_id: str) -> SegmentCandidate | None:
        """One offered segment by id, or ``None`` when it was not offered."""
        for item in self.candidates:
            if item.segment_id == segment_id:
                return item
        return None


def incident_segments(
    network: RoadNetwork, junction_id: str
) -> tuple[tuple[RoadSegment, SegmentDirection], ...]:
    """Directed segments touching a junction, with which way each one runs.

    Both directions are returned. A camera at an intersection may be watching
    the queue arriving on one arm or the traffic discharging onto another, and
    excluding the outgoing segments would decide that for the operator.
    """
    found: list[tuple[RoadSegment, SegmentDirection]] = []
    for segment in network.segments:
        if segment.to_junction_id == junction_id:
            found.append((segment, SegmentDirection.INCOMING))
        elif segment.from_junction_id == junction_id:
            found.append((segment, SegmentDirection.OUTGOING))
    return tuple(sorted(found, key=lambda item: item[0].segment_id))


def _approach_for(junction: Junction, segment_id: str) -> tuple[str | None, float | None]:
    """The junction's approach fed by this segment, and its surveyed bearing."""
    for approach in junction.approaches:
        if approach.incoming_segment_id == segment_id:
            return approach.approach_id, approach.bearing_degrees
    return None, None


def evaluate_segments(
    site: ObservationSite,
    network: RoadNetwork,
    junction_id: str,
    *,
    computed_at: dt.datetime,
) -> SegmentCandidateSet:
    """List the directed segments at ``junction_id`` for an operator to choose from.

    Computes bearing evidence where the data allows it and selects nothing.
    """
    junction = next(
        (item for item in network.junctions if item.junction_id == junction_id), None
    )
    if junction is None:  # pragma: no cover - callers validate first
        msg = f"junction {junction_id!r} is not on network {network.network_id!r}"
        raise ValueError(msg)

    positions = {
        item.junction_id: item.position
        for item in network.junctions
        if item.position is not None
    }
    origin = positions.get(junction_id)
    site_bearing = site.bearing_degrees

    candidates: list[SegmentCandidate] = []
    for segment, direction in incident_segments(network, junction_id):
        other = (
            segment.from_junction_id
            if direction is SegmentDirection.INCOMING
            else segment.to_junction_id
        )
        approach_id, approach_bearing = (
            _approach_for(junction, segment.segment_id)
            if direction is SegmentDirection.INCOMING
            else (None, None)
        )

        far_end = positions.get(other)
        bearing = (
            bearing_degrees_between(origin, far_end)
            if origin is not None and far_end is not None
            else None
        )
        offset = (
            angular_separation_degrees(site_bearing, bearing)
            if bearing is not None and site_bearing is not None
            else None
        )

        candidates.append(
            SegmentCandidate(
                segment_id=segment.segment_id,
                segment_name=segment.name,
                direction=direction,
                from_junction_id=segment.from_junction_id,
                to_junction_id=segment.to_junction_id,
                other_junction_id=other,
                length_m=segment.length_m.value,
                approach_id=approach_id,
                approach_bearing_degrees=approach_bearing,
                bearing_from_junction_degrees=bearing,
                camera_bearing_offset_degrees=offset,
            )
        )

    warnings: list[str] = []
    has_evidence = any(item.has_bearing_evidence for item in candidates)

    if site_bearing is None:
        warnings.append(
            "The site records no camera bearing, so no geometric evidence could "
            "be computed. Choose from local knowledge of what the camera sees."
        )
    elif origin is None:
        warnings.append(
            f"Junction {junction_id} carries no surveyed position, so no segment "
            "bearing could be computed from geometry."
        )

    # Two arms of one road lie on the same line. Naming the pairs is the useful
    # thing to say: it is exactly where the bearing stops discriminating.
    collinear: list[str] = []
    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
            if (
                first.bearing_from_junction_degrees is None
                or second.bearing_from_junction_degrees is None
            ):
                continue
            separation = angular_separation_degrees(
                first.bearing_from_junction_degrees,
                second.bearing_from_junction_degrees,
            )
            if separation <= COLLINEAR_TOLERANCE_DEG:
                collinear.append(f"{first.segment_id} and {second.segment_id}")
    if collinear:
        warnings.append(
            "Bearing cannot separate "
            + "; ".join(collinear)
            + ". These run along the same line from the junction, which is what "
            "the two directions of one road look like: a camera facing oncoming "
            "traffic and one filming departing vehicles stand in the same place."
        )

    if not candidates:
        note = (
            f"No segment touches junction {junction_id} on network "
            f"{network.network_id!r}. Nothing can be bound here, which is a gap "
            "in the map rather than a property of the camera."
        )
    else:
        note = (
            f"{len(candidates)} directed segment(s) meet junction {junction_id}. "
            "Bearing, where present, says which line the camera points along -- "
            "never which direction of travel it watches. Nothing is bound until "
            "an operator names a segment."
        )

    return SegmentCandidateSet(
        observation_area_id=site.observation_area_id,
        junction_id=junction_id,
        junction_name=junction.name,
        network_id=network.network_id,
        network_version=network.version,
        candidates=tuple(candidates),
        site_bearing_degrees=site_bearing,
        bearing_evidence_available=has_evidence,
        provenance=Provenance(
            source=ValueSource.DERIVED,
            producer=SEGMENT_CANDIDATE_PRODUCER,
            producer_version=APP_VERSION,
            observed_at=site.provenance.observed_at,
            computed_at=computed_at,
            input_refs=(
                f"observation_site:{site.observation_area_id}",
                f"network:{network.network_id}@{network.version}",
                f"junction:{junction_id}",
            ),
            notes=(
                "Segments incident to the confirmed junction, with bearings "
                "computed from surveyed positions. Topology and geometry only; "
                "it asserts nothing about which road the camera watches."
            ),
        ),
        warnings=tuple(warnings),
        note=note,
    )
