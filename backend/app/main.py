"""FastAPI application factory.

An explicit ``create_app`` factory rather than a module-level ``app`` singleton:
tests need to build an application against overridden settings, and a singleton
constructed at import time forecloses that.
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.context import bind_context, new_request_id
from app.core.errors import TrafficSystemError
from app.core.logging import configure_logging, get_logger
from app.core.versions import APP_VERSION, IMPLEMENTED_PHASE_NAME, SCHEMA_VERSION
from app.ingestion.registry import register_default_sources
from app.ingestion.storage import MediaStore
from app.sites.bindings import ConfirmedBindingRegistry
from app.sites.registry import ObservationSiteRegistry

logger = get_logger(__name__)

DESCRIPTION = """
Decision-support backend for the GenAI-Assisted Traffic Management System.

**Advisory only.** This service analyses recorded and simulated traffic data and
produces recommendations for human operators. It has no control path to any real
traffic signal, vehicle or piece of infrastructure, and none may be added.

Capabilities belonging to a later development phase return **501 Not
Implemented** with the owning phase attached, rather than an empty success
payload -- an empty result would be indistinguishable from a genuine observation
of no traffic.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging at startup and report the served configuration."""
    settings: Settings = app.state.settings
    configure_logging(level=settings.log_level, fmt=settings.log_format, force=True)
    logger.info(
        "application starting",
        extra={
            "app_version": APP_VERSION,
            "schema_version": SCHEMA_VERSION,
            "phase": IMPLEMENTED_PHASE_NAME,
            "environment": settings.app_env.value,
            "genai_configured": settings.genai_configured,
        },
    )
    _autoload_network(app, settings)
    yield
    logger.info("application stopped")


def _autoload_network(app: FastAPI, settings: Settings) -> None:
    """Load the configured road network, if one is configured.

    A load failure is logged and the application still starts. The alternative
    -- refusing to boot -- would take down ingestion, perception and analytics
    because of a bad map path, and every endpoint that needs a network already
    reports its absence accurately. ``/api/network/status`` carries the reason
    so the failure is visible rather than merely survived.
    """
    if not settings.network_file:
        return

    from app.ingestion.storage import resolve_within
    from app.network.loader import load_network
    from app.network.store import NetworkStore

    try:
        path = resolve_within(settings.networks_dir, settings.network_file)
        network, source = load_network(
            path, settings.network_file, loaded_at=dt.datetime.now(tz=dt.UTC)
        )
    except Exception as exc:  # startup must survive a bad map
        app.state.network_load_error = str(exc)
        logger.warning(
            "configured road network could not be loaded",
            extra={"network_file": settings.network_file, "error": str(exc)},
        )
        return

    store: NetworkStore = getattr(app.state, "network_store", None) or NetworkStore()
    store.load(network, source)
    app.state.network_store = store
    app.state.network_load_error = None
    logger.info(
        "road network loaded at startup",
        extra={
            "network_id": network.network_id,
            "version": network.version,
            "junctions": len(network.junctions),
            "segments": len(network.segments),
        },
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    Parameters
    ----------
    settings:
        Overrides the process settings singleton. Used by tests to exercise
        alternative configurations without mutating global state.
    """
    resolved = settings or get_settings()
    configure_logging(level=resolved.log_level, fmt=resolved.log_format)

    # Register the source implementations this build actually contains, so
    # /api/sources/catalog reports availability from the live registry rather
    # than from a hand-maintained list.
    register_default_sources()

    app = FastAPI(
        title=resolved.app_name,
        description=DESCRIPTION,
        version=APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.state.settings = resolved
    # The media store belongs to this application instance, not to the process,
    # so an app built with an overridden media directory actually uses it.
    app.state.media_store = MediaStore(resolved.media_dir)
    # Surveyed site locations are persistent and deliberately *not* part of the
    # media store: they outlive the footage that prompted them, and an app built
    # with an overridden sites file must read and write that file, not a global.
    app.state.site_registry = ObservationSiteRegistry(resolved.sites_file)
    # Confirmed bindings persist alongside sites, in their own file: a human
    # decision about a site outlives any correction to the site's coordinate.
    app.state.binding_registry = ConfirmedBindingRegistry(resolved.confirmed_bindings_file)

    # CORS is scoped to the configured origins. Not "*": the dev server origin is
    # known, and a permissive default has a way of surviving into deployment.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.api_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def correlation_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[JSONResponse]]
    ) -> JSONResponse:
        """Bind a correlation id to the request and log its outcome.

        The id is echoed as ``X-Request-ID`` so a user reporting a problem can
        quote a value that appears verbatim in the server logs.
        """
        request_id = request.headers.get("X-Request-ID") or new_request_id()
        started = time.perf_counter()
        with bind_context(request_id=request_id):
            response = await call_next(request)
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            response.headers["X-Request-ID"] = request_id
            logger.info(
                "request completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                },
            )
            return response

    @app.exception_handler(TrafficSystemError)
    async def handle_domain_error(_: Request, exc: TrafficSystemError) -> JSONResponse:
        """Map application errors onto their declared status codes."""
        # Severity follows `is_expected`, not the status code. A 501 raised by a
        # phase boundary is a 5xx that is working exactly as designed; logging it
        # at ERROR made a healthy backend look like it was failing in the console
        # while the HTTP responses were entirely correct.
        if exc.is_expected:
            logger.info(
                "request declined: capability not available yet",
                extra={"error_code": exc.error_code, "status_code": exc.status_code},
            )
        else:
            logger.error(
                "request failed",
                extra={"error_code": exc.error_code, "status_code": exc.status_code},
            )
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Return validation failures in the same envelope as every other error."""
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_failed",
                    "message": "Request validation failed.",
                    "details": {"errors": exc.errors()},
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Wrap framework HTTP errors in the uniform error envelope."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "http_error",
                    "message": str(exc.detail),
                    "details": {},
                }
            },
        )

    app.include_router(api_router, prefix=resolved.api_prefix)
    return app


app = create_app()
