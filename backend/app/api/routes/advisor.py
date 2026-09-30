"""AI Traffic Advisor API: explanation over facts this system already computed.

What this endpoint is
---------------------
A reader for results other phases produced, a prompt, and a validator. It runs
no detection, computes no analytics, classifies no congestion and plans no
route. Given a media id it looks up the interpreted traffic state, the analytics
set, the site, the confirmed binding, the declared incidents and emergencies,
and -- only if they exist -- a route result and a simulation run. Everything it
says rests on those.

That constraint is what keeps this additive. If this module ever computed a
traffic quantity there would be two pipelines producing the same number, and the
first time they disagreed nobody would know which one the dashboard was showing.

Absence is answered, not errored
--------------------------------
An advisory can be produced for media with no traffic state: it says no state
has been interpreted and suggests running the analysis, which is more useful
than a 404. The endpoint fails only when the media id itself is unknown, because
then there is nothing to be advisory *about*.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Any

from fastapi import APIRouter
from pydantic import Field

from app.advisor.facts import TrafficAdvisorFacts, build_facts
from app.advisor.provider import build_provider
from app.advisor.schemas import AdvisoryMode, TrafficAdvisory
from app.advisor.service import generate_advisory
from app.api.deps import (
    AnalyticsStoreDep,
    BindingRegistryDep,
    EmergencyRegistryDep,
    IncidentRegistryDep,
    MappingStoreDep,
    NetworkStoreDep,
    SettingsDep,
    SimulationStoreDep,
    SiteRegistryDep,
    StateStoreDep,
)
from app.api.routing_context import prepare_routing
from app.api.schemas import ApiModel, ErrorResponse
from app.core.errors import (
    ResourceNotFoundError,
    TrafficSystemError,
    ValidationFailedError,
)
from app.core.logging import get_logger
from app.domain.site import observation_area_id_for_media
from app.routing.engine import optimize
from app.routing.schemas import RouteOptimizationResult
from app.routing.weights import RoutingMode

logger = get_logger(__name__)
router = APIRouter(prefix="/advisor", tags=["advisor"])

_ERRORS: dict[int | str, dict[str, object]] = {
    404: {"model": ErrorResponse, "description": "Nothing analysed under this id"},
    422: {"model": ErrorResponse, "description": "The request names neither media nor a job"},
}


class AdvisorStatusResponse(ApiModel):
    """Whether a model is configured, and which mode an advisory will use.

    Served so the panel can label itself before anyone presses a button. It
    reports configuration only: no key, no key prefix, no base URL, nothing that
    could identify the credential.
    """

    provider_configured: bool
    mode: AdvisoryMode
    """The mode the next advisory would use if the provider behaves. A
    configured provider that then fails still yields a fallback advisory."""

    model_id: Annotated[str, Field(max_length=200)] | None = None
    """The configured model name. Never the key."""

    advisory_only: bool = True
    note: str = ""


class AdvisorRequest(ApiModel):
    """Which already-analysed thing to explain."""

    media_id: Annotated[str, Field(min_length=1, max_length=64)] | None = None
    """An uploaded image or video. Its traffic state and analytics are looked up
    from the caches the existing endpoints populate."""

    job_id: Annotated[str, Field(min_length=1, max_length=64)] | None = None
    """A video analysis job, for the same purpose. Exactly one of these two."""

    origin_junction_id: Annotated[str, Field(min_length=1, max_length=64)] | None = None
    destination_junction_id: Annotated[str, Field(min_length=1, max_length=64)] | None = (
        None
    )
    """Both, to include route evidence. When given and a network is loaded, the
    **existing** optimizer is run and its ranked candidates become the route
    facts. Omitted, there is no route comparison and the advisory says so
    instead of naming a faster route -- which is the rule that matters most
    here, because "take route B" with no comparison behind it is exactly the
    kind of confident fabrication this layer exists to prevent."""

    simulation_run_id: Annotated[str, Field(min_length=1, max_length=64)] | None = None
    """A completed run to include as *modelled* context. Never presented as a
    measurement of the road."""


class AdvisorResponse(ApiModel):
    """The advisory, and the facts it was built from."""

    advisory: TrafficAdvisory
    facts: TrafficAdvisorFacts
    """Returned alongside so the panel can show the evidence, and so a reviewer
    can check every sentence against what the model was actually given."""


def _state_key(request: AdvisorRequest) -> str:
    """The cache key the analytics and state endpoints already use."""
    return request.media_id if request.media_id else f"job:{request.job_id}"


@router.get(
    "/status",
    response_model=AdvisorStatusResponse,
    summary="Whether the advisor has a language model configured",
)
def advisor_status(settings: SettingsDep) -> AdvisorStatusResponse:
    """Report the advisor's configuration without revealing any credential."""
    configured = settings.genai_configured
    return AdvisorStatusResponse(
        provider_configured=configured,
        mode=AdvisoryMode.LLM if configured else AdvisoryMode.DETERMINISTIC_FALLBACK,
        model_id=settings.openrouter_model if configured else None,
        note=(
            "A language model is configured. Generated advisories are validated "
            "against the fact table before being shown, and a rejected one is "
            "replaced by the deterministic advisory."
            if configured
            else "No language model is configured. Advisories are composed from "
            "the recorded facts by fixed rules and are labelled as such."
        ),
    )


@router.post(
    "/analyze",
    response_model=AdvisorResponse,
    responses=_ERRORS,
    summary="Explain the current traffic facts and suggest operator actions",
)
def analyze(
    request: AdvisorRequest,
    settings: SettingsDep,
    states: StateStoreDep,
    analytics_store: AnalyticsStoreDep,
    sites: SiteRegistryDep,
    bindings: BindingRegistryDep,
    incidents: IncidentRegistryDep,
    emergencies: EmergencyRegistryDep,
    network: NetworkStoreDep,
    simulations: SimulationStoreDep,
    mappings: MappingStoreDep,
) -> AdvisorResponse:
    """Produce an advisory over the facts already computed for this media.

    Reads; never writes and never recomputes. The advisory is advisory: there is
    no control path from this endpoint to any signal, vehicle or piece of
    infrastructure, and the response says so on every path.
    """
    if (request.media_id is None) == (request.job_id is None):
        msg = (
            "Name exactly one of media_id or job_id: the advisor explains one "
            "analysed observation area, and cannot tell which one is meant "
            "from both or neither."
        )
        raise ValidationFailedError(msg)

    key = _state_key(request)
    state = states.get(key)
    analytics = analytics_store.get(key)

    if state is None and analytics is None:
        msg = (
            f"Nothing has been analysed under {key!r}. Upload the media, run "
            "vision analysis and traffic analytics, then ask for an advisory."
        )
        raise ResourceNotFoundError(msg, key=key)

    area_id = _area_id(state, analytics, request)
    site = sites.find(area_id)
    binding = _binding(bindings, network, area_id)

    simulation = (
        simulations.get(request.simulation_run_id)
        if request.simulation_run_id is not None
        else None
    )

    now = dt.datetime.now(tz=dt.UTC)
    route_result = _route_evidence(
        request,
        store=network,
        settings=settings,
        incidents=incidents,
        emergencies=emergencies,
        states=states,
        mappings=mappings,
        at=now,
    )

    facts = build_facts(
        area_id,
        state=state,
        analytics=analytics,
        site=site,
        binding=binding,
        incidents=incidents.list(),
        emergencies=emergencies.list(),
        route_result=route_result,
        simulation=simulation,
        now=now,
    )

    advisory = generate_advisory(facts, build_provider(settings), now=now)
    logger.info(
        "traffic advisory generated",
        extra={
            "observation_area_id": area_id,
            "mode": advisory.mode.value,
            "has_state": facts.has_state,
            "facts": len(facts.fact_table.facts),
            "unavailable": len(facts.unavailable),
        },
    )
    return AdvisorResponse(advisory=advisory, facts=facts)


def _area_id(state: Any, analytics: Any, request: AdvisorRequest) -> str:
    """The observation area these results belong to.

    Read off whichever result exists rather than reconstructed, so the advisory
    is keyed by exactly the id the verdicts carry.
    """
    if state is not None:
        return str(state.observation_area_id)
    if analytics is not None:
        return str(analytics.observation_area_id)
    # Unreachable while the caller checks for both being absent, but the
    # convention is defined in one place and this uses it rather than a literal.
    return observation_area_id_for_media(request.media_id or request.job_id or "")


def _binding(bindings: Any, network: Any, area_id: str) -> Any:
    """The confirmed binding for this area on the loaded network, if any.

    Requires a loaded network because a binding is per map. No network means no
    binding to report, which is a working state rather than an error.
    """
    loaded = network.network
    if loaded is None:
        return None
    return bindings.find(loaded.network_id, area_id)


def _route_evidence(
    request: AdvisorRequest,
    *,
    store: Any,
    settings: Any,
    incidents: Any,
    emergencies: Any,
    states: Any,
    mappings: Any,
    at: dt.datetime,
) -> RouteOptimizationResult | None:
    """Run the existing optimizer, when the caller asked for route evidence.

    Reuses ``prepare_routing`` and ``optimize`` exactly as ``/api/routes/optimize``
    does, so the advisory talks about the same ranking an operator would see on
    the routing panel. Nothing about the optimizer changes; this is a caller.

    Returns ``None`` when no origin and destination were given, when no network
    is loaded, or when the request cannot be resolved. An advisory with no route
    evidence is a normal outcome and says so; failing the whole advisory because
    a junction id was mistyped would be worse.
    """
    if request.origin_junction_id is None or request.destination_junction_id is None:
        return None
    if not store.is_loaded:
        return None

    try:
        context = prepare_routing(
            store=store,
            settings=settings,
            incidents=incidents,
            emergencies=emergencies,
            states=states,
            mappings=mappings,
            mode=RoutingMode.NORMAL,
            origin_junction_id=request.origin_junction_id,
            destination_junction_id=request.destination_junction_id,
            destination_facility_id=None,
            emergency_id=None,
            max_candidates=None,
            at=at,
        )
        return optimize(
            context.request,
            context.optimizer,
            destination_junction_id=context.destination_junction_id,
            at=context.at,
        )
    except TrafficSystemError as exc:
        logger.info(
            "advisor route evidence unavailable",
            # "message" is reserved on a LogRecord; "detail" carries the text.
            extra={"error_code": exc.error_code, "detail": exc.message},
        )
        return None
