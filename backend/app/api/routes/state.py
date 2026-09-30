"""Interpreted traffic state API.

Phase 5 scope: **what the Phase 4 measurements mean**. A congestion level per
observation area, the signals and thresholds behind it, and a snapshot fusing
every interpreted area under an explicit staleness budget.

What no endpoint here returns: travel time, ETA, route, signal advisory or
emergency decision. Those act *on* an interpreted state rather than producing
one, and belong to later phases.

Interpretation never re-measures
--------------------------------
Every endpoint reads a measurement set the analytics layer already computed and
cached. None of them run a detector, a tracker or the analytics engine. A
request for a traffic state whose analytics do not exist is a **404 naming the
analytics endpoint to call first**, not an invitation to quietly recompute
perception -- which would be slow, and would give the system a second chance to
disagree with itself about what was in the frame.

An unclassifiable state is a 200, not an error
----------------------------------------------
Poor measurement quality or missing signals produce ``CongestionLevel.UNKNOWN``
with a stated reason, returned successfully. That is a true answer about the
traffic, not a fault in the service, and reporting it as an error would train an
operator to dismiss the one condition that most needs their attention.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Any

from fastapi import APIRouter
from pydantic import Field

from app.analytics.schemas import TrafficAnalytics
from app.api.deps import (
    AnalyticsStoreDep,
    BindingRegistryDep,
    MappingStoreDep,
    NetworkStoreDep,
    SettingsDep,
    StateStoreDep,
)
from app.api.schemas import ApiModel, ErrorResponse
from app.core.config import Settings
from app.core.errors import DataSourceError, ResourceNotFoundError
from app.core.logging import get_logger
from app.domain.state import StateAssemblyPolicy
from app.network.store import NetworkStore
from app.routing.attribution import EvidenceAttributionReport, attribute_evidence
from app.routing.effective import compose_effective_mapping
from app.routing.mapping import MappingStore
from app.sites.bindings import ConfirmedBindingRegistry, binding_for
from app.state.assembler import (
    FUSION_STRATEGY,
    STATE_SCHEMA_VERSION,
    assemble,
    interpret,
    policy_from,
)
from app.state.classifier import METHOD, METHOD_VERSION, thresholds_from
from app.state.schemas import (
    CongestionThresholds,
    IntersectionStateElement,
    IntersectionTrafficState,
    NetworkTrafficState,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/state", tags=["state"])

_ERRORS: dict[int | str, dict[str, object]] = {
    404: {"model": ErrorResponse, "description": "No analytics or state for this id"},
}


class StateConfigResponse(ApiModel):
    """The interpretation policy in force.

    Exposed for the same reason the analytics thresholds are: a congestion band
    is meaningless without the cut points that produced it, and an operator who
    disagrees with a verdict needs something specific to disagree with.
    """

    method: str
    method_version: str
    schema_version: str
    thresholds: CongestionThresholds
    weight_density: float
    weight_speed: float
    weight_occupancy: float
    weight_stationary: float
    max_staleness_s: float
    drop_after_s: float
    fusion_strategy: str


@router.get("/config", response_model=StateConfigResponse, summary="Interpretation policy")
def state_config(settings: SettingsDep) -> StateConfigResponse:
    """Return the thresholds, weights and staleness budget behind every verdict."""
    policy = policy_from(settings)
    return StateConfigResponse(
        method=METHOD,
        method_version=METHOD_VERSION,
        schema_version=STATE_SCHEMA_VERSION,
        thresholds=thresholds_from(settings),
        weight_density=settings.congestion_weight_density,
        weight_speed=settings.congestion_weight_speed,
        weight_occupancy=settings.congestion_weight_occupancy,
        weight_stationary=settings.congestion_weight_stationary,
        max_staleness_s=policy.max_staleness_s,
        drop_after_s=policy.drop_after_s,
        fusion_strategy=FUSION_STRATEGY,
    )


def _analytics_or_404(
    analytics_store: dict[str, Any],
    key: str,
    *,
    kind: str,
    identifier: str,
    endpoint: str,
) -> TrafficAnalytics:
    """Fetch a cached measurement set, or explain which endpoint produces it."""
    result = analytics_store.get(key)
    if result is None:
        msg = (
            f"No traffic analytics have been computed for {kind} {identifier!r}, so "
            f"there is nothing to interpret. Compute them first with POST {endpoint}."
        )
        raise ResourceNotFoundError(msg, **{f"{kind}_id": identifier})
    return result


def _interpret_and_store(
    analytics: TrafficAnalytics,
    settings: Settings,
    states: dict[str, Any],
    key: str,
) -> IntersectionTrafficState:
    """Interpret one measurement set and cache the result."""
    state = interpret(analytics, settings, assessed_at=dt.datetime.now(tz=dt.UTC))
    states[key] = state
    logger.info(
        "traffic state interpreted",
        extra={
            "observation_area_id": state.observation_area_id,
            "congestion_level": state.congestion.level.value,
            "congestion_index": state.congestion.index,
            "basis": state.congestion.basis.value,
        },
    )
    return state


# ---------------------------------------------------------------------------
# Image-derived state
# ---------------------------------------------------------------------------
@router.post(
    "/images/{media_id}",
    response_model=IntersectionTrafficState,
    responses=_ERRORS,
    summary="Interpret an image's measurements as traffic state",
)
def interpret_image_state(
    media_id: str,
    settings: SettingsDep,
    analytics_store: AnalyticsStoreDep,
    states: StateStoreDep,
) -> IntersectionTrafficState:
    """Interpret the cached analytics for an image.

    A single frame carries no time, so speed, queue persistence and flow are
    unavailable by construction and the verdict rests on occupancy alone. The
    reported confidence reflects that rather than hiding it.
    """
    analytics = _analytics_or_404(
        analytics_store,
        media_id,
        kind="media",
        identifier=media_id,
        endpoint=f"/api/analytics/images/{media_id}",
    )
    return _interpret_and_store(analytics, settings, states, media_id)


@router.get(
    "/images/{media_id}",
    response_model=IntersectionTrafficState,
    responses=_ERRORS,
    summary="Most recent interpreted state for an image",
)
def get_image_state(media_id: str, states: StateStoreDep) -> IntersectionTrafficState:
    """Return the cached interpretation, or 404 if none has been computed."""
    state = states.get(media_id)
    if state is None:
        msg = f"No traffic state has been interpreted for image {media_id!r}."
        raise ResourceNotFoundError(msg, media_id=media_id)
    return state


# ---------------------------------------------------------------------------
# Video-derived state
# ---------------------------------------------------------------------------
@router.post(
    "/jobs/{job_id}",
    response_model=IntersectionTrafficState,
    responses=_ERRORS,
    summary="Interpret a video job's measurements as traffic state",
)
def interpret_job_state(
    job_id: str,
    settings: SettingsDep,
    analytics_store: AnalyticsStoreDep,
    states: StateStoreDep,
) -> IntersectionTrafficState:
    """Interpret the cached analytics for a completed video analysis job."""
    analytics = _analytics_or_404(
        analytics_store,
        f"job:{job_id}",
        kind="job",
        identifier=job_id,
        endpoint=f"/api/analytics/jobs/{job_id}",
    )
    return _interpret_and_store(analytics, settings, states, f"job:{job_id}")


@router.get(
    "/jobs/{job_id}",
    response_model=IntersectionTrafficState,
    responses=_ERRORS,
    summary="Most recent interpreted state for a video job",
)
def get_job_state(job_id: str, states: StateStoreDep) -> IntersectionTrafficState:
    """Return the cached interpretation, or 404 if none has been computed."""
    state = states.get(f"job:{job_id}")
    if state is None:
        msg = f"No traffic state has been interpreted for job {job_id!r}."
        raise ResourceNotFoundError(msg, job_id=job_id)
    return state


# ---------------------------------------------------------------------------
# Fused snapshot
# ---------------------------------------------------------------------------
class NetworkTrafficStateResponse(ApiModel):
    """The fused snapshot, plus where each area's evidence lands on the network.

    Every field of :class:`~app.state.schemas.NetworkTrafficState` is repeated
    here at the top level, unchanged, so a client written before attribution
    existed reads exactly what it always read. ``attribution`` is the only
    addition, and it is optional because it genuinely may not exist: resolving a
    camera to a road needs a loaded network, and traffic state does not.
    """

    # --- the snapshot, verbatim -------------------------------------------
    state_id: str
    as_of: dt.datetime
    assembled_at: dt.datetime
    policy: StateAssemblyPolicy
    elements: tuple[IntersectionStateElement, ...] = ()
    schema_version: str

    # --- additive ----------------------------------------------------------
    attribution: EvidenceAttributionReport | None = None
    """Where each interpreted area's evidence reaches the road network, through
    the effective mapping and through nothing else. ``None`` when it could not
    be resolved, with the reason below."""

    attribution_unavailable_reason: Annotated[str, Field(max_length=800)] | None = None
    """Why attribution is absent. Stated rather than left as a bare null: an
    operator seeing no attribution needs to know whether nothing is bound or
    whether the question could not be asked.

    Roomy on purpose. It carries a registry's own failure message verbatim, and
    those are deliberately full sentences naming the file and how to fix it;
    truncating one to fit a tidier limit would cut exactly the part that makes
    it actionable."""



@router.get(
    "/network",
    response_model=NetworkTrafficStateResponse,
    summary="Fused traffic state across every interpreted observation area",
)
def network_traffic_state(
    settings: SettingsDep,
    states: StateStoreDep,
    store: NetworkStoreDep,
    mappings: MappingStoreDep,
    bindings: BindingRegistryDep,
) -> NetworkTrafficStateResponse:
    """Fuse every interpreted observation area into one snapshot, and place it.

    Answers 200 with an empty element list when nothing has been interpreted,
    and the distinction from ``/api/network/state`` is deliberate. This endpoint
    reports **our own records**: "no area has been interpreted" is a true and
    knowable statement about this process. ``/api/network/state`` would have to
    describe a surveyed road network, and an empty answer there would assert
    that a network exists and is clear -- a claim about the world that no
    measurement supports.

    Elements are aged against the newest element rather than against wall-clock
    time, so interpreting recorded footage does not mark every element expired.

    The snapshot itself is computed exactly as before; nothing about traffic
    state interpretation changed. ``attribution`` is added beside it, resolving
    each area to a directed segment through the effective mapping -- and through
    nothing else. An area with no effective binding stays unmapped rather than
    being assigned to whatever road happens to be nearest.
    """
    snapshot = assemble(
        tuple(states.values()),
        settings,
        assembled_at=dt.datetime.now(tz=dt.UTC),
    )
    attribution, unavailable = _attribution(snapshot, store, mappings, bindings)

    logger.info(
        "network traffic state assembled",
        extra={
            "elements": len(snapshot.elements),
            "degraded": snapshot.is_degraded,
            "unclassified": len(snapshot.unclassified_area_ids),
            "attributed": attribution.attributed_count if attribution else 0,
        },
    )
    return NetworkTrafficStateResponse(
        **snapshot.model_dump(),
        attribution=attribution,
        attribution_unavailable_reason=unavailable,
    )


def _attribution(
    snapshot: NetworkTrafficState,
    store: NetworkStore,
    mappings: MappingStore,
    bindings: ConfirmedBindingRegistry,
) -> tuple[EvidenceAttributionReport | None, str | None]:
    """Resolve attribution, or say why it could not be resolved.

    Never raises. Traffic state is the primary payload here and must survive a
    problem with the mapping registry: a corrupt bindings file is a real fault
    and is reported as one, but taking down the interpreted state along with it
    would hide working evidence behind an unrelated failure.
    """
    network = store.network
    if network is None:
        return None, (
            "No road network is loaded, so no observation area can be resolved "
            "to a road. The interpreted traffic state above is unaffected."
        )

    try:
        effective = compose_effective_mapping(
            network=network,
            mapping=mappings.mapping,
            mapping_source=mappings.source,
            confirmed=bindings.list(network.network_id),
            binding_for=binding_for,
            composed_at=dt.datetime.now(tz=dt.UTC),
        )
    except DataSourceError as exc:
        return None, (
            f"The effective mapping could not be composed: {exc.message} "
            "Traffic state is unaffected; no evidence has been attributed."
        )

    return attribute_evidence(snapshot, effective), None
