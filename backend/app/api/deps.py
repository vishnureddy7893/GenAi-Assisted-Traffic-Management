"""Request-scoped dependencies.

These exist to fix a specific class of bug: a route that calls
``get_settings()`` reads the *process-wide* singleton, not the settings the
application was actually built with. An application constructed by
``create_app(settings)`` with an overridden media directory or upload limit
would silently ignore them, and a test asserting on those overrides would pass
or fail for reasons unrelated to what it was testing.

Resolving both settings and the media store from ``request.app.state`` means a
route always sees the configuration of the app instance serving it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Depends
from starlette.requests import Request

from app.core.config import Settings, get_settings
from app.emergency.incidents import IncidentRegistry
from app.emergency.responses import EmergencyRegistry
from app.ingestion.storage import MediaStore
from app.network.store import NetworkStore
from app.routing.mapping import MappingStore
from app.simulation.store import SimulationStore
from app.sites.bindings import ConfirmedBindingRegistry
from app.sites.registry import ObservationSiteRegistry

if TYPE_CHECKING:
    from app.vision.analysis import DetectorProvider, JobRegistry


def app_settings(request: Request) -> Settings:
    """Settings belonging to the application serving this request."""
    settings: Settings | None = getattr(request.app.state, "settings", None)
    return settings if settings is not None else get_settings()


def media_store(request: Request) -> MediaStore:
    """Media store belonging to the application serving this request."""
    store: MediaStore | None = getattr(request.app.state, "media_store", None)
    if store is None:
        store = MediaStore(app_settings(request).media_dir)
        request.app.state.media_store = store
    return store


def site_registry(request: Request) -> ObservationSiteRegistry:
    """Observation site registry belonging to the application serving this request.

    Resolved exactly like the media store, and for the same reason -- an app
    built with an overridden ``sites_file`` must actually use it, or a test
    would read and write the developer's real registry.

    It is a *separate* object from the media store on purpose: a site outlives
    the media that prompted it, and coupling the two would make clearing media
    discard surveyed locations.
    """
    registry: ObservationSiteRegistry | None = getattr(
        request.app.state, "site_registry", None
    )
    if registry is None:
        registry = ObservationSiteRegistry(app_settings(request).sites_file)
        request.app.state.site_registry = registry
    return registry


def binding_registry(request: Request) -> ConfirmedBindingRegistry:
    """Confirmed site-to-junction bindings for this application instance.

    Resolved from app state like the site registry, and held in a *separate*
    file: a confirmation is a decision about a site, and re-surveying the site
    must not rewrite the decision.
    """
    registry: ConfirmedBindingRegistry | None = getattr(
        request.app.state, "binding_registry", None
    )
    if registry is None:
        registry = ConfirmedBindingRegistry(app_settings(request).confirmed_bindings_file)
        request.app.state.binding_registry = registry
    return registry


SettingsDep = Annotated[Settings, Depends(app_settings)]
MediaStoreDep = Annotated[MediaStore, Depends(media_store)]
SiteRegistryDep = Annotated[ObservationSiteRegistry, Depends(site_registry)]
BindingRegistryDep = Annotated[ConfirmedBindingRegistry, Depends(binding_registry)]


def detector_provider(request: Request) -> DetectorProvider:
    """Detector provider belonging to this application instance.

    Cached on app state so the weights are loaded once per process rather than
    once per request -- a YOLO load costs roughly a second, which would
    otherwise dominate every inference.
    """
    from app.vision.analysis import DetectorProvider

    provider: DetectorProvider | None = getattr(request.app.state, "detector_provider", None)
    if provider is None:
        provider = DetectorProvider(app_settings(request))
        request.app.state.detector_provider = provider
    return provider


def job_registry(request: Request) -> JobRegistry:
    """Video analysis job registry for this application instance."""
    from app.vision.analysis import JobRegistry

    registry: JobRegistry | None = getattr(request.app.state, "job_registry", None)
    if registry is None:
        registry = JobRegistry()
        request.app.state.job_registry = registry
    return registry


def image_analysis_store(request: Request) -> dict[str, tuple[Any, bytes]]:
    """Cache of image analyses and their rendered overlays, keyed by media id.

    In-memory and per-process: an analysis is cheap to recompute and pinning it
    to a database before the schema has settled would be premature.
    """
    store: dict[str, tuple[Any, bytes]] | None = getattr(
        request.app.state, "image_analyses", None
    )
    if store is None:
        store = {}
        request.app.state.image_analyses = store
    return store


ProviderDep = Annotated["DetectorProvider", Depends(detector_provider)]
JobRegistryDep = Annotated["JobRegistry", Depends(job_registry)]
ImageAnalysisStoreDep = Annotated[dict[str, tuple[Any, bytes]], Depends(image_analysis_store)]


def analytics_store(request: Request) -> dict[str, Any]:
    """Cache of computed traffic analytics, keyed by media id or ``job:<id>``.

    Per-process and in-memory: analytics are cheap to recompute from stored
    perception output, and committing to a persistence schema before the
    measurement set has settled would be premature.
    """
    store: dict[str, Any] | None = getattr(request.app.state, "analytics_results", None)
    if store is None:
        store = {}
        request.app.state.analytics_results = store
    return store


AnalyticsStoreDep = Annotated[dict[str, Any], Depends(analytics_store)]


def state_store(request: Request) -> dict[str, Any]:
    """Cache of interpreted traffic states, keyed by media id or ``job:<id>``.

    Keyed identically to the analytics cache on purpose: an interpretation is a
    pure function of one measurement set, so the two caches stay in step and a
    state can always be traced back to the analytics entry that produced it.
    """
    store: dict[str, Any] | None = getattr(request.app.state, "traffic_states", None)
    if store is None:
        store = {}
        request.app.state.traffic_states = store
    return store


StateStoreDep = Annotated[dict[str, Any], Depends(state_store)]


def network_store(request: Request) -> NetworkStore:
    """The road network loaded for this application instance.

    One store per app, holding at most one network. Created empty: a fresh
    process has no map until one is loaded, and every endpoint that needs one
    says so rather than inventing an empty network.
    """
    store: NetworkStore | None = getattr(request.app.state, "network_store", None)
    if store is None:
        store = NetworkStore()
        request.app.state.network_store = store
    return store


def incident_registry(request: Request) -> IncidentRegistry:
    """Declared incidents for this application instance."""
    registry: IncidentRegistry | None = getattr(request.app.state, "incidents", None)
    if registry is None:
        registry = IncidentRegistry()
        request.app.state.incidents = registry
    return registry


def emergency_registry(request: Request) -> EmergencyRegistry:
    """Declared emergency responses for this application instance."""
    registry: EmergencyRegistry | None = getattr(request.app.state, "emergencies", None)
    if registry is None:
        registry = EmergencyRegistry()
        request.app.state.emergencies = registry
    return registry


def mapping_store(request: Request) -> MappingStore:
    """The observation-to-network mapping in force for this application.

    Created empty. An empty mapping is a working state, not a broken one: every
    segment is simply unobserved, priced by the uncertainty weights, and the
    optimizer says so. A mapping is needed to *use* traffic evidence, never to
    produce a route.
    """
    store: MappingStore | None = getattr(request.app.state, "mapping_store", None)
    if store is None:
        store = MappingStore()
        request.app.state.mapping_store = store
    return store


def simulation_store(request: Request) -> SimulationStore:
    """Completed simulation runs retained by this application instance.

    Bounded by configuration and held in memory. A run is cheap to reproduce --
    identical inputs give an identical run under an identical id -- so retaining
    it is a convenience for retrieval, never the system of record.
    """
    store: SimulationStore | None = getattr(request.app.state, "simulation_store", None)
    if store is None:
        store = SimulationStore(max_runs=app_settings(request).sim_max_stored_runs)
        request.app.state.simulation_store = store
    return store


NetworkStoreDep = Annotated[NetworkStore, Depends(network_store)]
SimulationStoreDep = Annotated[SimulationStore, Depends(simulation_store)]
MappingStoreDep = Annotated[MappingStore, Depends(mapping_store)]
IncidentRegistryDep = Annotated[IncidentRegistry, Depends(incident_registry)]
EmergencyRegistryDep = Annotated[EmergencyRegistry, Depends(emergency_registry)]
