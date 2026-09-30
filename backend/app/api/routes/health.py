"""Health and metadata endpoints.

Health here answers two separate questions rather than one merged one:

* **Liveness** -- is this process serving requests? (``status``)
* **Pipeline readiness** -- can an end-to-end decision be produced? (``pipeline_ready``)

Through Phase 8 those answers are ``ok`` and ``false``: ingestion,
perception, analytics, congestion interpretation, the road graph, emergency
management, route optimization and controlled simulation all work, but the
decision loop that records an approved recommendation does not exist yet, so no
end-to-end decision can be produced. A single flag would have to misreport one of
them, and the failure mode of merging them is specific: a green health tick on a
dashboard whose panels are empty reads as "no traffic detected" rather than "no
pipeline exists".
"""

from __future__ import annotations

import datetime as dt
import time

from fastapi import APIRouter

from app.api.deps import NetworkStoreDep
from app.api.schemas import (
    ComponentHealth,
    ComponentStatus,
    HealthResponse,
    PhaseInfo,
    PhasesResponse,
    VersionsResponse,
)
from app.core.config import Settings, get_settings
from app.core.errors import ValidationFailedError
from app.core.phases import PHASES
from app.core.versions import (
    APP_VERSION,
    IMPLEMENTED_PHASE,
    IMPLEMENTED_PHASE_NAME,
    SCHEMA_VERSION,
)
from app.ingestion.registry import source_registry
from app.network.store import NetworkStore
from app.vision.registry import BackendSpec, resolve_backend

router = APIRouter(tags=["health"])

#: Process start, for uptime reporting.
_STARTED_MONOTONIC = time.monotonic()


def _detector_backend(settings: Settings) -> BackendSpec | None:
    """Probe for a usable detector backend, tolerating a bad configuration.

    Reported rather than assumed: Phase 3 is implemented, but whether detection
    can actually run depends on which optional dependencies are installed in
    *this* environment. A misconfigured backend name is a configuration fault
    surfaced by the vision endpoints, not a reason for /health to fail.
    """
    try:
        return resolve_backend(settings)
    except ValidationFailedError:
        return None


def _components(settings: Settings, network: NetworkStore) -> tuple[ComponentHealth, ...]:
    """Report each subsystem's real state, including the ones that do not exist yet."""
    implemented_sources = source_registry.implemented_kinds()
    detector_backend = _detector_backend(settings)

    return (
        ComponentHealth(
            name="configuration",
            status=ComponentStatus.OK,
            detail=f"environment={settings.app_env.value}",
        ),
        ComponentHealth(
            name="logging",
            status=ComponentStatus.OK,
            detail=f"level={settings.log_level} format={settings.log_format}",
        ),
        ComponentHealth(
            name="domain_schemas",
            status=ComponentStatus.OK,
            detail=f"schema_version={SCHEMA_VERSION}",
        ),
        ComponentHealth(
            name="safety_catalog",
            status=ComponentStatus.OK,
            detail=(
                f"version={settings.safety_rules_version}; rule definitions only, "
                "soft-constraint penalty weights are calibrated in phase 11"
            ),
        ),
        ComponentHealth(
            name="data_sources",
            status=(
                ComponentStatus.OK
                if implemented_sources
                else ComponentStatus.NOT_IMPLEMENTED
            ),
            detail=(
                f"{len(implemented_sources)} source implementation(s) registered; "
                "interfaces and capability profiles are defined"
            ),
            phase=None if implemented_sources else 2,
        ),
        ComponentHealth(
            name="computer_vision",
            status=(
                ComponentStatus.OK
                if detector_backend is not None
                else ComponentStatus.NOT_CONFIGURED
            ),
            detail=(
                f"detector backend {detector_backend.name!r} available; "
                "ByteTrack-style tracking implemented"
                if detector_backend is not None
                else "detection and tracking are implemented, but no detector "
                "backend's dependencies are installed in this environment"
            ),
        ),
        ComponentHealth(
            name="traffic_analytics",
            status=ComponentStatus.OK,
            detail="measurements derived from detections and tracks, with provenance",
        ),
        ComponentHealth(
            name="traffic_state",
            status=ComponentStatus.OK,
            detail=(
                "congestion interpreted from analytics against configured "
                f"thresholds; staleness budget {settings.max_observation_staleness_s}s"
            ),
        ),
        ComponentHealth(
            name="road_network",
            status=(
                ComponentStatus.OK if network.is_loaded else ComponentStatus.NOT_CONFIGURED
            ),
            detail=(
                f"network {network.source.network_id!r} version "
                f"{network.source.version} loaded "
                f"({network.source.junction_count} junctions, "
                f"{network.source.segment_count} segments)"
                if network.source is not None
                else "loading is implemented; no network file has been loaded yet"
            ),
        ),
        ComponentHealth(
            name="emergency_management",
            status=(
                ComponentStatus.OK if network.is_loaded else ComponentStatus.NOT_CONFIGURED
            ),
            detail=(
                "incident and emergency declaration with lifecycle tracking"
                if network.is_loaded
                else "declaration requires a loaded network to anchor against"
            ),
        ),
        ComponentHealth(
            name="route_optimizer",
            status=(
                ComponentStatus.OK if network.is_loaded else ComponentStatus.NOT_CONFIGURED
            ),
            detail=(
                "movement-aware Dijkstra with Yen's alternatives; costs in "
                f"equivalent seconds, weights version {settings.route_weights_version}"
                if network.is_loaded
                else "route search is implemented; it needs a loaded network to run on"
            ),
        ),
        ComponentHealth(
            name="simulation",
            status=(
                ComponentStatus.OK if network.is_loaded else ComponentStatus.NOT_CONFIGURED
            ),
            detail=(
                "deterministic point-queue model over a "
                f"{settings.sim_horizon_s:g}s horizon at {settings.sim_step_s:g}s "
                f"steps; parameters version {settings.sim_parameters_version}"
                if network.is_loaded
                else "the simulator is implemented; it needs a loaded network to run on"
            ),
        ),
        ComponentHealth(
            name="ml_models",
            status=ComponentStatus.NOT_IMPLEMENTED,
            detail="no model trained or loaded; predictions are unavailable, not defaulted",
            phase=10,
        ),
        ComponentHealth(
            name="genai",
            status=(
                ComponentStatus.OK
                if settings.genai_configured
                else ComponentStatus.NOT_CONFIGURED
            ),
            detail=(
                "OpenRouter credentials present"
                if settings.genai_configured
                else "no OpenRouter credentials configured; narrative generation is disabled"
            ),
            phase=12,
        ),
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service liveness and pipeline readiness",
)
def health(network: NetworkStoreDep) -> HealthResponse:
    """Report liveness, pipeline readiness, and per-subsystem status."""
    settings = get_settings()
    components = _components(settings, network)

    blockers = tuple(
        f"{c.name}: {c.detail or c.status.value}"
        + (f" (phase {c.phase})" if c.phase else "")
        for c in components
        if c.status is ComponentStatus.NOT_IMPLEMENTED
    )

    return HealthResponse(
        status=ComponentStatus.OK,
        pipeline_ready=not blockers,
        pipeline_blockers=blockers,
        app_version=APP_VERSION,
        schema_version=SCHEMA_VERSION,
        implemented_phase=IMPLEMENTED_PHASE,
        implemented_phase_name=IMPLEMENTED_PHASE_NAME,
        environment=settings.app_env.value,
        server_time=dt.datetime.now(tz=dt.UTC),
        uptime_s=round(time.monotonic() - _STARTED_MONOTONIC, 3),
        components=components,
    )


@router.get(
    "/meta/versions",
    response_model=VersionsResponse,
    summary="Version pins recorded into every decision",
)
def versions() -> VersionsResponse:
    """Return the version pins that make a decision reproducible."""
    settings = get_settings()
    return VersionsResponse(
        app_version=APP_VERSION,
        schema_version=SCHEMA_VERSION,
        safety_rules_version=settings.safety_rules_version,
        route_weights_version=settings.route_weights_version,
        implemented_phase=IMPLEMENTED_PHASE,
        implemented_phase_name=IMPLEMENTED_PHASE_NAME,
        model_versions={},
    )


@router.get(
    "/meta/phases",
    response_model=PhasesResponse,
    summary="Development phase plan and current position",
)
def phases() -> PhasesResponse:
    """Return the phase plan so the UI can label gaps precisely."""
    return PhasesResponse(
        current_phase=IMPLEMENTED_PHASE,
        phases=tuple(
            PhaseInfo(
                phase=p.number,
                name=p.name,
                implemented=p.implemented,
                summary=p.summary,
            )
            for p in PHASES
        ),
    )
