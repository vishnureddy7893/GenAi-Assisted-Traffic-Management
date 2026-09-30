"""Resolving an exclusion zone onto the network graph.

An :class:`~app.domain.incident.ExclusionZone` is declared as a centre and a
radius, but route pruning cannot afford a geometric query in the inner loop of a
search. So the zone is resolved **once, at declaration** into the set of segment
and junction ids it covers, and pruning becomes a set membership test.

Distance is measured to the segment, not to its endpoints
---------------------------------------------------------
The obvious implementation tests each junction's position and excludes a segment
when an endpoint falls inside. It has a specific and dangerous failure: a long
segment running straight through the zone, with both endpoints outside it, is
not excluded. That is exactly the road past a fire.

So each segment is treated as a polyline -- its surveyed ``geometry`` when
present, its two endpoint positions otherwise -- and the *minimum* distance from
the zone centre to that polyline decides. Coordinates are projected onto a local
equirectangular plane centred on the zone, which is accurate to well under a
metre at the few-kilometre scale an exclusion zone operates over, and avoids
trigonometry per candidate point.

What cannot be resolved is reported, not skipped
------------------------------------------------
A network element with no coordinates cannot be tested. Silently leaving it out
would produce a zone that looks resolved and prunes less than it claims -- the
failure mode where a hard safety constraint quietly stops constraining. Every
untestable element is counted and named in :class:`ZoneResolution`, and the
caller decides whether that is acceptable.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Annotated

from pydantic import Field

from app.domain.base import DomainModel
from app.domain.network import GeoPoint, RoadNetwork

#: IUGG mean Earth radius, in metres.
EARTH_RADIUS_M = 6_371_008.8


class ZoneResolution(DomainModel):
    """Which network elements an exclusion zone covers, and what could not be tested."""

    excluded_segment_ids: frozenset[str] = frozenset()
    excluded_junction_ids: frozenset[str] = frozenset()

    tested_segment_count: Annotated[int, Field(ge=0)] = 0
    untestable_segment_ids: frozenset[str] = frozenset()
    """Segments with neither surveyed geometry nor positioned endpoints. These
    are *not* excluded, and that is the point of naming them: a caller must be
    able to see that the zone's coverage is incomplete."""

    untestable_junction_ids: frozenset[str] = frozenset()
    warnings: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        """Whether every element could be tested against the zone."""
        return not self.untestable_segment_ids and not self.untestable_junction_ids


def _project(centre: GeoPoint, point: GeoPoint) -> tuple[float, float]:
    """Project a coordinate onto a local plane in metres about ``centre``."""
    cos_lat = math.cos(math.radians(centre.latitude))
    x = math.radians(point.longitude - centre.longitude) * cos_lat * EARTH_RADIUS_M
    y = math.radians(point.latitude - centre.latitude) * EARTH_RADIUS_M
    return x, y


def haversine_m(a: GeoPoint, b: GeoPoint) -> float:
    """Great-circle distance between two coordinates, in metres.

    Used for point-to-point checks where the small-angle projection is not
    appropriate, and for reporting distances an operator may sanity-check
    against a map.
    """
    lat1, lat2 = math.radians(a.latitude), math.radians(b.latitude)
    dlat = lat2 - lat1
    dlon = math.radians(b.longitude - a.longitude)
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def _distance_to_polyline(points: list[tuple[float, float]]) -> float:
    """Shortest distance from the projected origin to a projected polyline."""
    if not points:
        return math.inf
    if len(points) == 1:
        return math.hypot(points[0][0], points[0][1])

    best = math.inf
    for (ax, ay), (bx, by) in pairwise(points):
        dx, dy = bx - ax, by - ay
        length_squared = dx * dx + dy * dy
        if length_squared == 0.0:
            best = min(best, math.hypot(ax, ay))
            continue
        # Projection parameter of the origin onto the segment, clamped to it.
        t = -(ax * dx + ay * dy) / length_squared
        t = max(0.0, min(1.0, t))
        best = min(best, math.hypot(ax + t * dx, ay + t * dy))
    return best


def resolve_zone(
    network: RoadNetwork,
    *,
    centre: GeoPoint,
    radius_m: float,
) -> ZoneResolution:
    """Resolve a centre-and-radius exclusion zone onto network elements."""
    positions = {j.junction_id: j.position for j in network.junctions}

    excluded_junctions: set[str] = set()
    untestable_junctions: set[str] = set()
    for junction in network.junctions:
        if junction.position is None:
            untestable_junctions.add(junction.junction_id)
            continue
        x, y = _project(centre, junction.position)
        if math.hypot(x, y) <= radius_m:
            excluded_junctions.add(junction.junction_id)

    excluded_segments: set[str] = set()
    untestable_segments: set[str] = set()
    tested = 0
    for segment in network.segments:
        if segment.geometry:
            polyline = [_project(centre, point) for point in segment.geometry]
        else:
            endpoints = [
                positions.get(segment.from_junction_id),
                positions.get(segment.to_junction_id),
            ]
            if any(point is None for point in endpoints):
                untestable_segments.add(segment.segment_id)
                continue
            polyline = [_project(centre, point) for point in endpoints if point is not None]

        tested += 1
        if _distance_to_polyline(polyline) <= radius_m:
            excluded_segments.add(segment.segment_id)

    warnings: list[str] = []
    if untestable_junctions:
        warnings.append(
            f"{len(untestable_junctions)} junction(s) carry no surveyed position and "
            "could not be tested against the zone."
        )
    if untestable_segments:
        warnings.append(
            f"{len(untestable_segments)} segment(s) have neither surveyed geometry nor "
            "positioned endpoints and could not be tested; the zone's coverage is "
            "incomplete and those segments remain admissible."
        )
    if not excluded_segments and not excluded_junctions:
        warnings.append(
            "The zone covers no network element. A zone that excludes nothing "
            "constrains nothing, which is usually a mistaken centre or radius."
        )

    return ZoneResolution(
        excluded_segment_ids=frozenset(excluded_segments),
        excluded_junction_ids=frozenset(excluded_junctions),
        tested_segment_count=tested,
        untestable_segment_ids=frozenset(untestable_segments),
        untestable_junction_ids=frozenset(untestable_junctions),
        warnings=tuple(warnings),
    )
