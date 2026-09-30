"""Aggregate router mounted under the configured API prefix."""

from fastapi import APIRouter

from app.api.routes import (
    advisor,
    analytics,
    emergency,
    health,
    incidents,
    media,
    network,
    routes,
    simulation,
    sites,
    sources,
    state,
    vision,
)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(sources.router)
api_router.include_router(media.router)
api_router.include_router(sites.router)
api_router.include_router(vision.router)
api_router.include_router(analytics.router)
api_router.include_router(state.router)
api_router.include_router(network.router)
api_router.include_router(incidents.router)
api_router.include_router(routes.router)
api_router.include_router(emergency.router)
api_router.include_router(simulation.router)
api_router.include_router(advisor.router)

__all__ = ["api_router"]
