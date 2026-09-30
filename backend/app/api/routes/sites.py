"""Observation site API: where the junction behind an observation area is.

Scope
-----
Recording and reading coordinates. Nothing here plans a route, measures a
distance, geocodes a name or contacts any mapping provider -- a site is a
surveyed fact, and this endpoint is the place it is written down.

Why the id is an observation area id
------------------------------------
A site is keyed by the ``observation_area_id`` that analytics and the routing
mapping already use, so a location joins to a congestion verdict without a
translation table. The key's meaning is unchanged: it still identifies one
camera's field of view. A site adds *where that view is*, and nothing else.

Recording twice is an update
----------------------------
``POST /api/sites`` with an id already on record replaces the location and
answers 200 rather than 201. Refusing it would be wrong -- a camera is
re-surveyed, or a first coordinate was mistyped, and the correction has to be
able to win. The original ``recorded_at`` is preserved so the correction is
visible as a correction.

Absence is 404, unreadable is 502
---------------------------------
"No location has been recorded for this area" is a genuine answer and is a 404.
"The file holding the locations will not open" is a different thing entirely and
is a 502: answering it with 404, or with an empty list, would present missing
knowledge as established fact.

Candidates are proposed by GET, bindings are created only by POST
-----------------------------------------------------------------
``GET /api/sites/{id}/candidate`` computes the nearest junction and returns it
as a proposal. It is a GET because it changes nothing, and that is a structural
guarantee rather than a convention: there is no code path from evaluating a
candidate to recording a binding.

``POST /api/sites/{id}/binding`` is the only thing that creates one, and it
refuses unless the caller sets ``confirm`` **and** names the junction id it is
confirming. Naming the junction is what makes the confirmation about a specific
road rather than about whatever happens to be nearest at the moment the request
lands -- if the map has changed underneath the operator, the ids no longer match
and the confirmation is refused instead of silently applied to a new junction.

Each failure mode is distinguishable by error code: ``network_not_loaded``
(no map), ``not_found`` (no such site, or no such confirmed binding),
``validation_failed`` (the junction is not on this map, or the confirmation was
not explicit), ``data_source_error`` (a registry file will not open).

A segment refines a junction; it never replaces one
----------------------------------------------------
``GET /api/sites/{id}/segments`` lists the directed segments meeting the
junction an operator already confirmed, with bearing evidence where the survey
allows it. ``POST /api/sites/{id}/binding/segment`` records which one they chose.

The same read/write split applies: the listing is a GET that writes nothing and
pre-selects nothing, and the confirmation requires ``confirm: true`` plus both
the junction and the segment named explicitly. Naming the junction again is not
redundant -- it is how the operator states which decision they are refining, and
a mismatch with what is on record is refused rather than silently redirected.

The refinement is written back onto the existing ``ConfirmedSiteBinding``, so
the junction decision and whoever made it survive intact alongside it.

The effective mapping is derived, not stored
--------------------------------------------
``GET /api/sites/effective-mapping`` composes the mapping in force from the
loaded mapping file and the confirmations, under the fixed precedence rule in
:mod:`app.routing.effective`, and reports its deterministic content hash. It is
recomputed on every request: there is no effective-mapping registry, so it
cannot drift from its inputs and cannot become a third source of truth.

It is read-only and changes nothing, including route costs -- the optimizer
still reads the mapping file alone, and the response says so rather than
leaving it to be inferred.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated

from fastapi import APIRouter, Response
from pydantic import Field

from app.api.deps import (
    BindingRegistryDep,
    MappingStoreDep,
    NetworkStoreDep,
    SettingsDep,
    SiteRegistryDep,
)
from app.api.schemas import ApiModel, ErrorResponse
from app.core.errors import ResourceNotFoundError, ValidationFailedError
from app.core.logging import get_logger
from app.core.versions import APP_VERSION
from app.domain.network import GeoPoint, RoadNetwork
from app.domain.provenance import Provenance, ValueSource
from app.domain.site import (
    MAX_ACCURACY_M,
    BindingConfirmation,
    BindingConfirmationMethod,
    ConfirmedSiteBinding,
    ObservationSite,
)
from app.routing.effective import (
    EffectiveObservationMapping,
    compose_effective_mapping,
)
from app.routing.mapping import ObservationBinding
from app.sites.bindings import (
    ConfirmedBindingRegistry,
    binding_for,
    validate_approach,
    validate_junction_exists,
    validate_network_version,
    validate_segment_incident,
)
from app.sites.candidates import (
    COLLINEAR_TOLERANCE_DEG,
    CandidateStatus,
    SegmentCandidateSet,
    SiteBindingCandidate,
    evaluate_segments,
    evaluate_site,
)
from app.sites.registry import ObservationSiteRegistry, utc_now

logger = get_logger(__name__)
router = APIRouter(prefix="/sites", tags=["sites"])

_ERRORS: dict[int | str, dict[str, object]] = {
    404: {"model": ErrorResponse, "description": "No site recorded for this area"},
    422: {"model": ErrorResponse, "description": "The location is not valid"},
    502: {"model": ErrorResponse, "description": "The site registry could not be read or written"},
}

#: Stated on every listing of confirmed bindings, so the inertness of a
#: segment binding is something a client reads rather than infers from an
#: unchanged route.
ROUTE_COST_NOTE = (
    "A confirmed binding does not change any route cost. The optimizer is "
    "handed the single ObservationMapping held by the mapping store and pins "
    "its id and version into every result for replay; this registry is neither "
    "versioned nor immutable, so feeding it in would leave that pin describing "
    "bindings that were not used. The association is recorded and reported "
    "instead."
)

#: Sources an operator-facing survey may claim. ``DERIVED`` and ``PREDICTED``
#: carry obligations (input refs, a model artifact) an HTTP caller cannot
#: satisfy honestly, and a *predicted* junction location is precisely the
#: fabrication this model exists to keep out.
_RECORDABLE_SOURCES = frozenset(
    {ValueSource.OBSERVED, ValueSource.STATIC, ValueSource.ASSUMED}
)


class SiteRequest(ApiModel):
    """A surveyed location for one observation area.

    Latitude and longitude are given flat here because that is how a surveyor
    reads them off a device; they are stored as the network's
    :class:`~app.domain.network.GeoPoint`, so the bounds enforced on a site are
    the same ones enforced on every segment polyline in the system.
    """

    observation_area_id: Annotated[str, Field(min_length=1, max_length=120)]
    """The area being located, e.g. ``observation-area:<media_id>``. Existing id,
    existing meaning -- a site does not mint a new identifier space."""

    latitude: Annotated[float, Field(ge=-90.0, le=90.0)]
    longitude: Annotated[float, Field(ge=-180.0, le=180.0)]

    bearing_degrees: Annotated[float, Field(ge=0.0, lt=360.0)] | None = None
    """Direction the camera faces: 0 = true north, clockwise. Optional, and
    ``None`` means unsurveyed rather than north."""

    accuracy_m: Annotated[float, Field(gt=0.0, le=MAX_ACCURACY_M)] | None = None
    """Radius within which the true position lies. Optional; strictly positive,
    because a zero-metre accuracy claims perfection no fix provides."""

    label: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    """Human name for the place. Display only; nothing is resolved from it."""

    recorded_by: Annotated[str, Field(min_length=1, max_length=200)] = "operator"
    """Who recorded this position. Kept as the provenance producer, because a
    coordinate read off a survey sheet is stronger evidence than one eyeballed
    from a map, and a reviewer is entitled to tell them apart."""

    source: ValueSource = ValueSource.OBSERVED
    source_notes: Annotated[str, Field(max_length=1000)] | None = None
    """Required when ``source`` is ``assumed``: an unjustified assumption about
    where a junction is cannot be told apart from a fabricated one."""


class SiteResponse(ApiModel):
    """One observation site, and whether recording it replaced an earlier one."""

    site: ObservationSite
    replaced_existing: bool = False
    """True when this recording corrected a location already on file. Surfaced
    so a caller who expected to create a site learns that it moved one."""


class SiteListResponse(ApiModel):
    """Recorded observation sites."""

    sites: tuple[ObservationSite, ...] = ()
    count: int = 0
    note: str = ""
    """Why the collection is what it is. An empty registry means nobody has
    surveyed a camera yet -- a complete and true statement about our records,
    not an absence of knowledge, which is why this answers 200 rather than 501."""


class CandidateResponse(ApiModel):
    """A proposed site-to-junction binding, and whether one is already confirmed."""

    site: ObservationSite
    candidate: SiteBindingCandidate
    confirmed: ConfirmedSiteBinding | None = None
    """The binding already in force for this area on this network, when there is
    one. Present so a caller can see that confirming again would replace a
    decision rather than make a first one."""

    note: str = ""


class ConfirmBindingRequest(ApiModel):
    """An operator confirming that a camera watches a particular junction."""

    junction_id: Annotated[str, Field(min_length=1, max_length=64)]
    """The junction being confirmed. Required and never defaulted from the
    candidate: a confirmation has to be *about* a junction the operator named,
    or it is only a click on whatever happened to sort first."""

    confirm: bool = False
    """Must be true. A separate, explicit flag rather than consent inferred from
    the request existing at all -- the whole point of this step is that a
    binding is never created by accident."""

    confirmed_by: Annotated[str, Field(min_length=1, max_length=200)] = "operator"
    """Who is confirming. Recorded as the provenance producer, because a binding
    nobody is named against is a binding nobody is accountable for."""

    notes: Annotated[str, Field(max_length=1000)] | None = None
    """Why this junction. Required when the confirmed junction is not the
    nearest one, since that is the case a reviewer is most likely to question."""


class ConfirmedBindingResponse(ApiModel):
    """One confirmed binding, and the mapping binding it produces."""

    confirmed: ConfirmedSiteBinding
    binding: ObservationBinding
    """The same decision in the routing layer's own type. A junction binding,
    which by existing mapping semantics prices and prunes nothing -- knowing
    which junction a camera watches is not yet knowing which directed segment
    its queue belongs to."""

    replaced_existing: bool = False
    note: str = ""


class SegmentCandidatesResponse(ApiModel):
    """Directed segments at the confirmed junction, offered for selection."""

    site: ObservationSite
    confirmed: ConfirmedSiteBinding
    """The junction binding being refined. Returned so the caller can see which
    decision these segments belong to, and whether one is already chosen."""

    segments: SegmentCandidateSet
    note: str = ""


class ConfirmSegmentRequest(ApiModel):
    """An operator naming the directed segment a camera watches."""

    junction_id: Annotated[str, Field(min_length=1, max_length=64)]
    """The junction whose binding is being refined. Required and checked against
    what is on record: a mismatch means the operator is looking at a different
    decision from the one they would be changing."""

    segment_id: Annotated[str, Field(min_length=1, max_length=64)]
    """The directed segment. Directed on purpose -- the two ids of a two-way
    road are the difference between a queue arriving and a queue departing."""

    approach_id: Annotated[str, Field(min_length=1, max_length=64)] | None = None
    """Optionally, the approach as well. Never filled in automatically, even
    where the lookup is unambiguous: this record is ``observed``, and every
    value on it should be one a person actually stated."""

    confirm: bool = False
    """Must be true. Bearing evidence never selects a segment, so nothing here
    happens without an explicit act."""

    confirmed_by: Annotated[str, Field(min_length=1, max_length=200)] = "operator"

    notes: Annotated[str, Field(max_length=1000)] | None = None
    """Why this segment. Required when the choice cannot be read off the
    geometry -- see the endpoint for exactly when."""


class EffectiveMappingResponse(ApiModel):
    """The composed mapping in force, with both source identities.

    Read-only and derived. Nothing here is stored, and asking for it changes
    nothing -- including route costs, which ``effective_mapping_in_use`` states
    plainly rather than leaving to be inferred from a route that did not move.
    """

    effective: EffectiveObservationMapping

    effective_mapping_in_use: bool = False
    """Whether route costing currently uses this composition. Always false in
    this step; the optimizer reads the loaded mapping file alone."""

    optimizer_mapping_id: Annotated[str, Field(max_length=64)] | None = None
    optimizer_mapping_version: Annotated[str, Field(max_length=50)] | None = None
    """What the optimizer *does* pin into a route result today. Reported beside
    the effective identity so the two are never confused: a route result
    carrying ``optimizer_mapping_version`` was computed from the file, not from
    this composition."""

    note: str = ""


class ConfirmedBindingListResponse(ApiModel):
    """Confirmed bindings, for the loaded network or for every network."""

    bindings: tuple[ConfirmedSiteBinding, ...] = ()
    count: int = 0
    network_id: str | None = None
    """The network these bindings are in force on, when one is loaded."""

    stale_version_area_ids: tuple[str, ...] = ()
    """Areas confirmed against a different version of the loaded network.
    Reported, never auto-invalidated: a re-surveyed map usually keeps its
    junction ids, and whether that holds here is the operator's call."""

    segment_bound_count: int = 0
    """How many of these have been refined to a directed segment."""

    segment_bound_observation_area_ids: tuple[str, ...] = ()
    """Areas whose evidence is now associated with one directed road."""

    affects_route_cost: bool = False
    """Whether these bindings currently change any route cost. Always false, and
    stated rather than left to be inferred from an unchanged route."""

    route_cost_note: str = ""
    """Why not, in one sentence, so nobody has to read the source to find out."""

    note: str = ""


def _build_site(
    request: SiteRequest, *, now: dt.datetime, recorded_at: dt.datetime
) -> ObservationSite:
    """Turn a request into a validated site, stamping provenance server-side."""
    if request.source not in _RECORDABLE_SOURCES:
        msg = (
            f"source {request.source.value!r} cannot be claimed for a surveyed "
            f"location; choose one of "
            f"{sorted(item.value for item in _RECORDABLE_SOURCES)}. A derived or "
            "predicted junction position is a guess wearing a measurement's clothes."
        )
        raise ValidationFailedError(msg, source=request.source.value)

    try:
        return ObservationSite(
            observation_area_id=request.observation_area_id,
            location=GeoPoint(latitude=request.latitude, longitude=request.longitude),
            bearing_degrees=request.bearing_degrees,
            accuracy_m=request.accuracy_m,
            label=request.label,
            provenance=Provenance(
                source=request.source,
                producer=request.recorded_by,
                producer_version=APP_VERSION,
                observed_at=recorded_at,
                computed_at=now,
                notes=request.source_notes,
            ),
            recorded_at=recorded_at,
            updated_at=now,
        )
    except ValueError as exc:
        # The one cross-field rule that pydantic's field bounds cannot express:
        # ValueSource.ASSUMED demands a justification.
        msg = f"The site could not be recorded: {exc}"
        raise ValidationFailedError(
            msg, observation_area_id=request.observation_area_id
        ) from exc


def record_site(
    registry: ObservationSiteRegistry, request: SiteRequest, *, now: dt.datetime
) -> tuple[ObservationSite, bool]:
    """Record a site from a request. Shared with the media upload path."""
    site = _build_site(request, now=now, recorded_at=now)
    return registry.record(site)


@router.get("", response_model=SiteListResponse, responses=_ERRORS, summary="Recorded sites")
def list_sites(registry: SiteRegistryDep) -> SiteListResponse:
    """Return every recorded observation site, most recently updated first."""
    sites = registry.list()
    return SiteListResponse(
        sites=sites,
        count=len(sites),
        note=(
            "Observation sites are surveyed locations, persisted to "
            f"{registry.path.name} and independent of the media store: they "
            "survive a restart and are unaffected by deleting the footage that "
            "prompted them. An empty list means no camera has been located yet."
        ),
    )


@router.post(
    "",
    response_model=SiteResponse,
    responses={201: {"model": SiteResponse}, **_ERRORS},
    summary="Record an observation site",
)
def create_site(
    request: SiteRequest, registry: SiteRegistryDep, response: Response
) -> SiteResponse:
    """Record where an observation area is, or correct where it was.

    Answers 201 on a first recording and 200 on a correction; the body says
    which in ``replaced_existing`` either way, so a client that ignores the
    status code still cannot mistake a move for a creation.
    """
    stored, replaced = record_site(registry, request, now=utc_now())
    response.status_code = 200 if replaced else 201
    return SiteResponse(site=stored, replaced_existing=replaced)


def _distance_to(candidate: SiteBindingCandidate, junction_id: str) -> float | None:
    """Distance to one junction from an evaluation, when it was measured.

    ``None`` when the junction carries no surveyed position, which is why the
    confirmation records the distance as absent rather than as zero.
    """
    for entry in (candidate.nearest, *candidate.runners_up):
        if entry is not None and entry.junction_id == junction_id:
            return entry.distance_m
    return None


def _evaluate(
    observation_area_id: str,
    sites: ObservationSiteRegistry,
    network: RoadNetwork,
    *,
    max_distance_m: float,
) -> tuple[ObservationSite, SiteBindingCandidate]:
    """Load the site and rank the network's junctions by distance from it."""
    site = sites.get(observation_area_id)
    candidate = evaluate_site(
        site, network, max_distance_m=max_distance_m, computed_at=utc_now()
    )
    return site, candidate


@router.get(
    "/confirmed",
    response_model=ConfirmedBindingListResponse,
    responses=_ERRORS,
    summary="Human-confirmed site-to-junction bindings",
)
def list_confirmed_bindings(
    bindings: BindingRegistryDep,
    store: NetworkStoreDep,
    all_networks: bool = False,
) -> ConfirmedBindingListResponse:
    """Return confirmed bindings for the loaded network.

    Answers 200 with an empty list when nothing has been confirmed: a binding
    exists only because somebody made it, so "none confirmed" is a complete and
    true description of the record rather than an absence of knowledge.

    Set ``all_networks`` to include confirmations made against maps that are not
    currently loaded. They are retained rather than deleted when the map
    changes, because switching networks and switching back must not silently
    discard decisions.
    """
    network = store.network
    if all_networks or network is None:
        items = bindings.list()
        stale: tuple[str, ...] = ()
    else:
        items = bindings.list(network.network_id)
        stale = bindings.stale_version_ids(network)

    if network is None:
        note = (
            "No network is loaded, so every confirmed binding is listed without "
            "regard to the map it was made on. A junction id means something "
            "different on a different map."
        )
    elif not items:
        note = (
            f"No binding has been confirmed on network {network.network_id!r}. "
            "A camera is bound to a junction only when an operator says so; "
            "nearest-junction proximity never binds anything on its own."
        )
    else:
        note = (
            f"{len(items)} binding(s) confirmed on network {network.network_id!r}. "
            "These are junction-level associations: they record which "
            "intersection each camera watches and do not price or prune any "
            "route, because a junction is not a directed segment."
        )

    segment_bound = tuple(
        sorted(item.observation_area_id for item in items if item.is_segment_bound)
    )
    return ConfirmedBindingListResponse(
        bindings=items,
        count=len(items),
        network_id=network.network_id if network else None,
        stale_version_area_ids=stale,
        segment_bound_count=len(segment_bound),
        segment_bound_observation_area_ids=segment_bound,
        affects_route_cost=False,
        route_cost_note=ROUTE_COST_NOTE,
        note=note,
    )


@router.get(
    "/{observation_area_id:path}/candidate",
    response_model=CandidateResponse,
    responses=_ERRORS,
    summary="Nearest junction to a site (a proposal, not a binding)",
)
def site_candidate(
    observation_area_id: str,
    sites: SiteRegistryDep,
    bindings: BindingRegistryDep,
    store: NetworkStoreDep,
    settings: SettingsDep,
) -> CandidateResponse:
    """Propose the nearest surveyed junction to this site.

    Creates nothing. The result is proximity evidence addressed to an operator,
    who binds it with ``POST /api/sites/{id}/binding`` or does not.

    Answers 404 with ``network_not_loaded`` when there is no map -- distinct
    from the 404 with ``not_found`` that means the site itself was never
    recorded, because the two call for completely different responses.
    """
    network = store.require()
    site, candidate = _evaluate(
        observation_area_id,
        sites,
        network,
        max_distance_m=settings.site_binding_max_distance_m,
    )
    existing = bindings.find(network.network_id, observation_area_id)

    if existing is not None:
        note = (
            f"This area is already bound to junction {existing.junction_id} by "
            f"{existing.confirmation.confirmed_by}. Confirming again would "
            "replace that decision."
        )
    elif candidate.status is CandidateStatus.CANDIDATE_FOUND:
        note = (
            "Nothing is bound yet. This is a proposal; it becomes a binding only "
            "when an operator confirms it by naming the junction."
        )
    else:
        note = candidate.note

    return CandidateResponse(
        site=site, candidate=candidate, confirmed=existing, note=note
    )


@router.post(
    "/{observation_area_id:path}/binding",
    response_model=ConfirmedBindingResponse,
    responses={201: {"model": ConfirmedBindingResponse}, **_ERRORS},
    summary="Confirm a site-to-junction binding",
)
def confirm_binding(
    observation_area_id: str,
    request: ConfirmBindingRequest,
    sites: SiteRegistryDep,
    bindings: BindingRegistryDep,
    store: NetworkStoreDep,
    settings: SettingsDep,
    response: Response,
) -> ConfirmedBindingResponse:
    """Record that an operator confirms this camera watches this junction.

    The only operation in the system that creates a binding from a site. It
    requires ``confirm: true`` and an explicit ``junction_id``; neither is
    inferred, and the candidate endpoint cannot reach this code.

    Answers 201 on a first confirmation and 200 when it replaces one.
    """
    network = store.require()
    if not request.confirm:
        msg = (
            "A binding is created only on an explicit confirmation: set "
            "confirm=true. Nearest-junction proximity is evidence an operator "
            "weighs, never a decision the system may take on their behalf."
        )
        raise ValidationFailedError(
            msg,
            observation_area_id=observation_area_id,
            junction_id=request.junction_id,
        )

    site, candidate = _evaluate(
        observation_area_id,
        sites,
        network,
        max_distance_m=settings.site_binding_max_distance_m,
    )
    validate_junction_exists(request.junction_id, network)

    nearest = candidate.nearest
    was_nearest = nearest.junction_id == request.junction_id if nearest else None
    distance = _distance_to(candidate, request.junction_id)

    if was_nearest is True:
        method = BindingConfirmationMethod.ACCEPTED_NEAREST
    elif was_nearest is False:
        method = BindingConfirmationMethod.CHOSE_DIFFERENT
    else:
        method = BindingConfirmationMethod.SURVEYED

    now = utc_now()
    try:
        confirmation = BindingConfirmation(
            confirmed_by=request.confirmed_by,
            confirmed_at=now,
            method=method,
            candidate_distance_m=distance,
            site_accuracy_m=site.accuracy_m,
            was_nearest_candidate=was_nearest,
            notes=request.notes,
        )
    except ValueError as exc:
        msg = (
            f"Junction {request.junction_id!r} is not the nearest to this site "
            f"({nearest.junction_id if nearest else 'none'} is), so the "
            f"confirmation needs notes saying why. {exc}"
        )
        raise ValidationFailedError(
            msg,
            observation_area_id=observation_area_id,
            junction_id=request.junction_id,
        ) from exc

    confirmed = ConfirmedSiteBinding(
        observation_area_id=observation_area_id,
        junction_id=request.junction_id,
        network_id=network.network_id,
        network_version=network.version,
        site_location=site.location,
        confirmation=confirmation,
        provenance=Provenance(
            source=ValueSource.OBSERVED,
            producer=request.confirmed_by,
            producer_version=APP_VERSION,
            observed_at=now,
            computed_at=now,
            notes=(
                "Human confirmation of a site-to-junction binding. The candidate "
                "was derived from distance; this decision was not."
            ),
        ),
    )

    stored, replaced = bindings.confirm(confirmed)
    response.status_code = 200 if replaced else 201
    return ConfirmedBindingResponse(
        confirmed=stored,
        binding=binding_for(stored),
        replaced_existing=replaced,
        note=(
            f"Observation area {observation_area_id!r} is bound to junction "
            f"{stored.junction_id} on network {network.network_id!r}. This is a "
            "junction-level binding: it records which intersection the camera "
            "watches and does not by itself price or prune any route."
        ),
    )


def _require_confirmed(
    observation_area_id: str,
    bindings: ConfirmedBindingRegistry,
    network: RoadNetwork,
) -> ConfirmedSiteBinding:
    """The junction binding being refined, or a 404 explaining what is missing.

    Distinct from "no such site": the camera may be perfectly well located and
    simply not yet bound to a junction, and telling an operator to record a
    location they already recorded would send them to the wrong place.
    """
    existing = bindings.find(network.network_id, observation_area_id)
    if existing is None:
        msg = (
            f"No junction binding has been confirmed for observation area "
            f"{observation_area_id!r} on network {network.network_id!r}. A "
            "directed segment is a refinement of a junction decision, so confirm "
            "the junction first."
        )
        raise ResourceNotFoundError(
            msg,
            observation_area_id=observation_area_id,
            network_id=network.network_id,
        )
    return existing


@router.get(
    "/effective-mapping",
    response_model=EffectiveMappingResponse,
    responses=_ERRORS,
    summary="The observation mapping in force, composed from both sources",
)
def effective_mapping(
    bindings: BindingRegistryDep,
    mappings: MappingStoreDep,
    store: NetworkStoreDep,
) -> EffectiveMappingResponse:
    """Compose the mapping file and the confirmed bindings into one mapping.

    Requires a loaded network, because the composition is network-scoped: which
    confirmations are in force, and which are stale and excluded, is decided
    against the loaded map and its version. Without one there is no question to
    answer, so this answers 404 ``network_not_loaded`` rather than composing
    something that applies to no map.

    Recomputed on each call from the loaded mapping and the registry. Identical
    inputs give an identical ``effective_mapping_hash``.
    """
    network = store.require()
    composed = compose_effective_mapping(
        network=network,
        mapping=mappings.mapping,
        mapping_source=mappings.source,
        confirmed=bindings.list(network.network_id),
        binding_for=binding_for,
        composed_at=utc_now(),
    )
    return EffectiveMappingResponse(
        effective=composed,
        effective_mapping_in_use=False,
        optimizer_mapping_id=mappings.mapping.mapping_id if mappings.mapping else None,
        optimizer_mapping_version=mappings.mapping.version if mappings.mapping else None,
        note=(
            "Derived on request from the loaded mapping file and the confirmed "
            "binding registry; nothing is stored. Route costs are unaffected: "
            "the optimizer is handed the mapping file alone, and pinning this "
            "hash onto a route result would claim the route was computed from "
            "bindings it never saw."
        ),
    )


@router.get(
    "/{observation_area_id:path}/segments",
    response_model=SegmentCandidatesResponse,
    responses=_ERRORS,
    summary="Directed segments at a confirmed junction (offered, not selected)",
)
def site_segment_candidates(
    observation_area_id: str,
    sites: SiteRegistryDep,
    bindings: BindingRegistryDep,
    store: NetworkStoreDep,
) -> SegmentCandidatesResponse:
    """List the directed segments meeting this site's confirmed junction.

    Both directions of every arm are listed, ordered by segment id. Where the
    site records a bearing and the junctions carry positions, each segment also
    carries its bearing from the junction and the angle to the camera's facing.

    Nothing is selected and nothing is written. The angle says which *line* the
    camera points along; it cannot say which direction of travel it watches,
    because the two segments of one road lie on the same line.
    """
    network = store.require()
    site = sites.get(observation_area_id)
    existing = _require_confirmed(observation_area_id, bindings, network)

    segments = evaluate_segments(
        site, network, existing.junction_id, computed_at=utc_now()
    )

    if existing.is_segment_bound:
        note = (
            f"This area is already bound to segment {existing.segment_id} "
            f"({existing.segment_direction.value if existing.segment_direction else '?'}"
            f" at {existing.junction_id}). Confirming another would replace that."
        )
    else:
        note = (
            f"Junction {existing.junction_id} is confirmed; no segment has been "
            "chosen yet. Choosing one records which directed road this camera's "
            "evidence belongs to."
        )

    return SegmentCandidatesResponse(
        site=site, confirmed=existing, segments=segments, note=note
    )


@router.post(
    "/{observation_area_id:path}/binding/segment",
    response_model=ConfirmedBindingResponse,
    responses={201: {"model": ConfirmedBindingResponse}, **_ERRORS},
    summary="Confirm the directed segment a site observes",
)
def confirm_segment_binding(
    observation_area_id: str,
    request: ConfirmSegmentRequest,
    sites: SiteRegistryDep,
    bindings: BindingRegistryDep,
    store: NetworkStoreDep,
    response: Response,
) -> ConfirmedBindingResponse:
    """Refine a confirmed junction binding to one directed segment.

    Refuses unless: the site exists, a junction binding exists, the named
    junction is the one on record, that junction binding was confirmed against
    the loaded network version, the segment exists, and the segment actually
    touches the junction. Each of those fails with its own message, because each
    means something different to the person who has to fix it.

    Answers 201 the first time a segment is chosen and 200 when it replaces an
    earlier choice.
    """
    network = store.require()
    site = sites.get(observation_area_id)
    existing = _require_confirmed(observation_area_id, bindings, network)

    if not request.confirm:
        msg = (
            "A segment binding is created only on an explicit confirmation: set "
            "confirm=true. Bearing evidence narrows the choice and never makes "
            "it, because the two directions of one road share a line."
        )
        raise ValidationFailedError(
            msg,
            observation_area_id=observation_area_id,
            segment_id=request.segment_id,
        )

    if request.junction_id != existing.junction_id:
        msg = (
            f"This area is bound to junction {existing.junction_id!r}, not "
            f"{request.junction_id!r}. A segment confirmation refines an existing "
            "junction decision; to move the camera to a different junction, "
            "confirm that junction first."
        )
        raise ValidationFailedError(
            msg,
            observation_area_id=observation_area_id,
            confirmed_junction_id=existing.junction_id,
            requested_junction_id=request.junction_id,
        )

    validate_network_version(existing, network)
    validate_junction_exists(request.junction_id, network)
    segment, direction = validate_segment_incident(
        request.segment_id, request.junction_id, network
    )
    if request.approach_id is not None:
        validate_approach(
            request.approach_id, request.segment_id, request.junction_id, network
        )

    candidates = evaluate_segments(
        site, network, existing.junction_id, computed_at=utc_now()
    )
    offered = candidates.candidate(request.segment_id)
    offset = offered.camera_bearing_offset_degrees if offered else None

    # Whether the geometry backs this choice at all decides what has to be
    # written down. With no usable bearing, or with the camera pointed along a
    # different line, the only record of *why* is the operator's own words.
    if offset is None:
        # No usable bearing. The operator is working from local knowledge, and
        # that is what the record should say.
        method = BindingConfirmationMethod.SURVEYED
    elif offset <= COLLINEAR_TOLERANCE_DEG:
        method = BindingConfirmationMethod.ALIGNED_WITH_BEARING
    else:
        method = BindingConfirmationMethod.CHOSE_DIFFERENT

    try:
        confirmation = BindingConfirmation(
            confirmed_by=request.confirmed_by,
            confirmed_at=utc_now(),
            method=method,
            candidate_distance_m=existing.confirmation.candidate_distance_m,
            site_accuracy_m=site.accuracy_m,
            was_nearest_candidate=None,
            notes=request.notes,
        )
    except ValueError as exc:
        msg = (
            f"The camera's bearing points {offset:.0f} degrees away from segment "
            f"{request.segment_id!r}, so the geometry does not support this "
            f"choice and the confirmation needs notes saying why. {exc}"
        )
        raise ValidationFailedError(
            msg,
            observation_area_id=observation_area_id,
            segment_id=request.segment_id,
            camera_bearing_offset_degrees=offset,
        ) from exc

    refined = ConfirmedSiteBinding.model_validate(
        existing.model_dump()
        | {
            "segment_id": segment.segment_id,
            "segment_direction": direction.value,
            "approach_id": request.approach_id,
            "segment_confirmation": confirmation.model_dump(),
        }
    )
    stored = bindings.refine(refined)
    response.status_code = 200 if existing.is_segment_bound else 201

    return ConfirmedBindingResponse(
        confirmed=stored,
        binding=binding_for(stored),
        replaced_existing=existing.is_segment_bound,
        note=(
            f"Observation area {observation_area_id!r} is bound to segment "
            f"{stored.segment_id} ({direction.value} at {stored.junction_id}) on "
            f"network {network.network_id!r}. The junction confirmation by "
            f"{existing.confirmation.confirmed_by} is unchanged. {ROUTE_COST_NOTE}"
        ),
    )


@router.delete(
    "/{observation_area_id:path}/binding",
    response_model=ConfirmedBindingResponse,
    responses=_ERRORS,
    summary="Withdraw a confirmed binding",
)
def revoke_binding(
    observation_area_id: str,
    bindings: BindingRegistryDep,
    store: NetworkStoreDep,
) -> ConfirmedBindingResponse:
    """Withdraw a confirmation, returning what was withdrawn.

    A binding an operator no longer stands behind has to be removable, or the
    only way to correct a mistake would be to assert a different one over it.
    """
    network = store.require()
    removed = bindings.revoke(network.network_id, observation_area_id)
    return ConfirmedBindingResponse(
        confirmed=removed,
        binding=binding_for(removed),
        replaced_existing=False,
        note=(
            f"The binding from {observation_area_id!r} to junction "
            f"{removed.junction_id} has been withdrawn. The area is unbound "
            "again, and its evidence is attributable to no junction."
        ),
    )


@router.get(
    "/{observation_area_id:path}",
    response_model=ObservationSite,
    responses=_ERRORS,
    summary="One observation site",
)
def get_site(observation_area_id: str, registry: SiteRegistryDep) -> ObservationSite:
    """Return one site, or 404 when no location has been recorded for it.

    The path is matched greedily because an observation area id contains a
    colon (``observation-area:<media_id>``) and may be passed unencoded.
    """
    return registry.get(observation_area_id)
