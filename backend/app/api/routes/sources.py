"""Data source endpoints: the two input modes, described honestly.

These endpoints back the dashboard's Recorded and Live input options. They
return the *capability catalogue* -- design metadata about what each input kind
is and which phase implements it -- and the list of sources currently
configured in this process, which is empty until a source is opened.

Describing an unimplemented source kind is not fabrication: it is a statement
about this system's architecture, not a claim about traffic. What would be
fabrication is returning a configured source that does not exist, or an
observation from a pipeline that was never run. Neither happens here: attempting
to open a source raises :class:`~app.core.errors.NotAvailableYetError`, which
the client sees as 501 with the owning phase attached.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import ApiModel
from app.ingestion.base import SourceDescriptor, SourceKind
from app.ingestion.registry import (
    SOURCE_KIND_PROFILES,
    SourceKindProfile,
    live_profiles,
    recorded_profiles,
    source_registry,
)

router = APIRouter(prefix="/sources", tags=["sources"])


class SourceModeGroup(ApiModel):
    """One input mode and the source kinds it contains."""

    mode: str
    title: str
    description: str
    is_primary: bool
    profiles: tuple[SourceKindProfile, ...]


class SourceCatalogResponse(ApiModel):
    """The full input catalogue, grouped by mode."""

    modes: tuple[SourceModeGroup, ...]
    implemented_kinds: tuple[SourceKind, ...]
    """Kinds with a working implementation right now."""

    unavailable_kinds: tuple[SourceKind, ...] = ()
    """Kinds with no implementation registered. Stated explicitly rather than
    left to be inferred from an empty ``implemented_kinds``: a client that only
    reads the positive list would have no way to distinguish "nothing is
    implemented" from "the catalogue failed to load"."""

    any_source_available: bool = False
    """Whether *any* input can currently be opened."""

    note: str = ""
    """Operator-facing summary of catalogue availability."""


class ConfiguredSourcesResponse(ApiModel):
    """Sources currently configured in this process."""

    sources: tuple[SourceDescriptor, ...]
    count: int


@router.get(
    "/catalog",
    response_model=SourceCatalogResponse,
    summary="Input kinds for Mode A (recorded) and Mode B (live)",
)
def catalog() -> SourceCatalogResponse:
    """Return the input catalogue with real implementation status per kind."""
    implemented = source_registry.implemented_kinds()
    reconciled = {p.kind: p for p in source_registry.profiles()}

    return SourceCatalogResponse(
        modes=(
            SourceModeGroup(
                mode="A",
                title="Recorded real-world traffic",
                description=(
                    "The primary capstone input. Real captured traffic data -- video, "
                    "detection datasets or count datasets -- replayed in a controlled "
                    "environment. Bounded, replayable and seekable, which is what makes "
                    "a decision change attributable to the change that caused it."
                ),
                is_primary=True,
                profiles=tuple(reconciled[p.kind] for p in recorded_profiles()),
            ),
            SourceModeGroup(
                mode="B",
                title="Live traffic",
                description=(
                    "Continuous camera or feed input. Unbounded, lossy and "
                    "non-replayable, so it requires an explicit backpressure policy. "
                    "Converges on the same TrafficState and the same decision pipeline "
                    "as recorded input."
                ),
                is_primary=False,
                profiles=tuple(reconciled[p.kind] for p in live_profiles()),
            ),
        ),
        implemented_kinds=tuple(sorted(implemented, key=lambda k: k.value)),
        unavailable_kinds=tuple(
            sorted(
                (p.kind for p in SOURCE_KIND_PROFILES if p.kind not in implemented),
                key=lambda k: k.value,
            )
        ),
        any_source_available=bool(implemented),
        note=(
            "The capability profiles below describe the designed input paths. "
            f"{len(implemented)} of {len(SOURCE_KIND_PROFILES)} source kinds have a "
            "registered implementation, so no traffic data can be ingested yet. "
            "Each profile states the phase that will implement it."
        ),
    )


@router.get(
    "",
    response_model=ConfiguredSourcesResponse,
    summary="Currently configured data sources",
)
def configured_sources() -> ConfiguredSourcesResponse:
    """Return configured sources.

    Genuinely empty until a source is configured in this process, so there is
    nothing to describe. The frontend renders this as an explicit "no source
    configured" state rather than as an idle pipeline.
    """
    return ConfiguredSourcesResponse(sources=(), count=0)
