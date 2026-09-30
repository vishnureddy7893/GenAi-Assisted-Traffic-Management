"""Registry of source kinds and their implementation status.

Two things live here, and the split matters:

* :data:`SOURCE_KIND_PROFILES` -- the *architectural catalogue*. What each input
  kind is, which capabilities it will declare, and which phase implements it.
  This is design metadata, not traffic data: describing what an RTSP source will
  be able to do is a statement about this system, not a claim about the world.

* :class:`SourceRegistry` -- the *runtime registry* of concrete implementations.
  It holds only the kinds that genuinely work, and the API says which those are.

Keeping them apart is what lets the dashboard offer a Recorded and a Live input
option, each labelled with its real status, without any part of the system
implying that an unimplemented pipeline works. The frontend renders from the
profile catalogue and disables anything whose ``implemented`` flag is false --
so the UI cannot drift out of step with the backend, because it is reading the
backend's own account of itself.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from app.core.errors import NotAvailableYetError
from app.domain.base import DomainModel
from app.ingestion.base import (
    LIVE_KINDS,
    RECORDED_KINDS,
    PayloadKind,
    SourceKind,
    TrafficDataSource,
)


class SourceKindProfile(DomainModel):
    """Design-time description of one input kind."""

    kind: SourceKind
    mode: Annotated[str, Field(pattern="^[AB]$")]
    """``A`` = recorded real-world data (primary), ``B`` = live traffic."""

    display_name: Annotated[str, Field(min_length=1, max_length=100)]
    description: Annotated[str, Field(min_length=1, max_length=500)]
    payload_kind: PayloadKind

    seekable: bool
    bounded: bool
    replayable: bool
    realtime: bool
    requires_backpressure: bool

    implemented: bool
    """Whether a working implementation exists *right now*. False for every kind
    whose phase has not landed. The frontend must not offer one as usable."""

    available_from_phase: Annotated[int, Field(ge=1, le=14)]
    required_dependencies: tuple[str, ...] = ()
    """External libraries the kind will need. Surfaced so the operational cost of
    each input path is visible before it is built."""


#: The architectural catalogue. Ordered so that the recorded kinds -- the primary
#: capstone input -- come first.
SOURCE_KIND_PROFILES: tuple[SourceKindProfile, ...] = (
    SourceKindProfile(
        kind=SourceKind.RECORDED_IMAGE,
        mode="A",
        display_name="Recorded traffic image",
        description=(
            "A single traffic still (JPEG, PNG or WebP). Bounded, replayable and "
            "one sample long. Ingested as raw input: the image is decoded and "
            "measured, never interpreted."
        ),
        payload_kind=PayloadKind.VIDEO_FRAME,
        seekable=True,
        bounded=True,
        replayable=True,
        realtime=False,
        requires_backpressure=False,
        implemented=True,
        available_from_phase=2,
        required_dependencies=("pillow",),
    ),
    SourceKindProfile(
        kind=SourceKind.RECORDED_COUNT_DATASET,
        mode="A",
        display_name="Recorded count / flow dataset",
        description=(
            "Tabular vehicle counts, flows or speeds from loop detectors or an "
            "agency open-data release. No computer vision required, so it "
            "exercises analytics and decision-making independently of detector "
            "quality."
        ),
        payload_kind=PayloadKind.TABULAR_RECORD,
        seekable=True,
        bounded=True,
        replayable=True,
        realtime=False,
        requires_backpressure=False,
        implemented=False,
        available_from_phase=2,
        required_dependencies=("pandas",),
    ),
    SourceKindProfile(
        kind=SourceKind.RECORDED_OBSERVATIONS,
        mode="A",
        display_name="Recorded observation replay",
        description=(
            "Serialised JunctionObservation records previously produced by this "
            "system. The regression-testing format: a fixed input against which "
            "a decision-logic change can be attributed."
        ),
        payload_kind=PayloadKind.OBSERVATION,
        seekable=True,
        bounded=True,
        replayable=True,
        realtime=False,
        requires_backpressure=False,
        implemented=False,
        available_from_phase=2,
    ),
    SourceKindProfile(
        kind=SourceKind.RECORDED_DETECTION_DATASET,
        mode="A",
        display_name="Recorded detection / tracking dataset",
        description=(
            "Pre-annotated detections or tracks. Skips detection and exercises "
            "tracking, analytics and decision-making against ground truth."
        ),
        payload_kind=PayloadKind.DETECTION_RECORD,
        seekable=True,
        bounded=True,
        replayable=True,
        realtime=False,
        requires_backpressure=False,
        implemented=False,
        available_from_phase=2,
    ),
    SourceKindProfile(
        kind=SourceKind.RECORDED_VIDEO,
        mode="A",
        display_name="Recorded intersection video",
        description=(
            "Video of a real intersection (MP4, MOV or AVI). Decoded frame by "
            "frame with seek, reset and replay support. Raw ingestion only: "
            "detection is phase 3 and traffic-state extraction phase 4."
        ),
        payload_kind=PayloadKind.VIDEO_FRAME,
        seekable=True,
        bounded=True,
        replayable=True,
        realtime=False,
        requires_backpressure=False,
        implemented=True,
        available_from_phase=2,
        required_dependencies=("opencv-python-headless",),
    ),
    SourceKindProfile(
        kind=SourceKind.LIVE_RTSP,
        mode="B",
        display_name="Live RTSP camera",
        description=(
            "Continuous RTSP feed from a fixed traffic camera. Unbounded, lossy "
            "and non-replayable; requires an explicit backpressure policy."
        ),
        payload_kind=PayloadKind.VIDEO_FRAME,
        seekable=False,
        bounded=False,
        replayable=False,
        realtime=True,
        requires_backpressure=True,
        implemented=False,
        available_from_phase=14,
        required_dependencies=("opencv-python", "a detector backend"),
    ),
    SourceKindProfile(
        kind=SourceKind.LIVE_HTTP_STREAM,
        mode="B",
        display_name="Live HTTP / MJPEG stream",
        description="HTTP-delivered video stream from a public or private camera.",
        payload_kind=PayloadKind.VIDEO_FRAME,
        seekable=False,
        bounded=False,
        replayable=False,
        realtime=True,
        requires_backpressure=True,
        implemented=False,
        available_from_phase=14,
        required_dependencies=("opencv-python",),
    ),
    SourceKindProfile(
        kind=SourceKind.LIVE_WEBCAM,
        mode="B",
        display_name="Local webcam",
        description=(
            "Locally attached camera. Primarily a development aid for exercising "
            "the live path without access to traffic infrastructure."
        ),
        payload_kind=PayloadKind.VIDEO_FRAME,
        seekable=False,
        bounded=False,
        replayable=False,
        realtime=True,
        requires_backpressure=True,
        implemented=False,
        available_from_phase=14,
        required_dependencies=("opencv-python",),
    ),
    SourceKindProfile(
        kind=SourceKind.LIVE_API_FEED,
        mode="B",
        display_name="Live traffic API feed",
        description=(
            "Polled REST feed of counts or speeds from a traffic authority. "
            "Unbounded like any live source, but far cheaper than video."
        ),
        payload_kind=PayloadKind.TABULAR_RECORD,
        seekable=False,
        bounded=False,
        replayable=False,
        realtime=True,
        requires_backpressure=True,
        implemented=False,
        available_from_phase=14,
    ),
)


def profile_for(kind: SourceKind) -> SourceKindProfile:
    """Return the catalogue profile for a kind, raising ``KeyError`` if absent."""
    for profile in SOURCE_KIND_PROFILES:
        if profile.kind is kind:
            return profile
    raise KeyError(kind)


class SourceRegistry:
    """Runtime registry mapping a source kind to its implementation.

    A kind with no registered implementation makes :meth:`create` raise
    :class:`~app.core.errors.NotAvailableYetError`, which the API surfaces as
    **501** with the owning phase attached -- so an attempt to start an
    unimplemented pipeline produces an accurate error rather than an empty
    success that looks like "no traffic detected".
    """

    def __init__(self) -> None:
        self._factories: dict[SourceKind, type[TrafficDataSource]] = {}

    def register(self, kind: SourceKind, factory: type[TrafficDataSource]) -> None:
        """Register an implementation for a kind."""
        if kind in self._factories:
            msg = f"a source implementation is already registered for {kind.value}"
            raise ValueError(msg)
        self._factories[kind] = factory

    def is_implemented(self, kind: SourceKind) -> bool:
        """Whether a working implementation is registered for this kind."""
        return kind in self._factories

    def implemented_kinds(self) -> frozenset[SourceKind]:
        """Kinds with a registered implementation."""
        return frozenset(self._factories)

    def factory(self, kind: SourceKind) -> type[TrafficDataSource]:
        """Return the implementation class for a kind, or raise 501-mapped error."""
        try:
            return self._factories[kind]
        except KeyError:
            profile = profile_for(kind)
            msg = (
                f"{profile.display_name} is not implemented yet "
                f"(planned for phase {profile.available_from_phase})"
            )
            raise NotAvailableYetError(
                msg,
                phase=profile.available_from_phase,
                phase_name=f"Phase {profile.available_from_phase}",
                source_kind=kind.value,
            ) from None

    def profiles(self) -> tuple[SourceKindProfile, ...]:
        """Catalogue profiles with ``implemented`` reflecting the live registry.

        The catalogue's own flag is the design intent; this reconciles it with
        what is actually registered, so the two can never disagree on the wire.
        """
        return tuple(
            profile.model_copy(update={"implemented": self.is_implemented(profile.kind)})
            for profile in SOURCE_KIND_PROFILES
        )


#: Process-wide registry. Concrete sources register here as phases land.
source_registry = SourceRegistry()


def register_default_sources(registry: SourceRegistry | None = None) -> None:
    """Register the source implementations that exist in this build.

    Called from the application factory rather than executed at import time, so
    that importing the registry has no side effects and a test can construct an
    empty registry when it wants one. Idempotent, because the factory may run
    more than once in a single process (every TestClient builds an app).
    """
    from app.ingestion.image_source import RecordedImageDataSource
    from app.ingestion.video_source import RecordedVideoDataSource

    target = registry if registry is not None else source_registry
    pairs: tuple[tuple[SourceKind, type[TrafficDataSource]], ...] = (
        (SourceKind.RECORDED_IMAGE, RecordedImageDataSource),
        (SourceKind.RECORDED_VIDEO, RecordedVideoDataSource),
    )
    for kind, factory in pairs:
        if not target.is_implemented(kind):
            target.register(kind, factory)


def recorded_profiles() -> tuple[SourceKindProfile, ...]:
    """Mode A profiles, in catalogue order."""
    return tuple(p for p in SOURCE_KIND_PROFILES if p.kind in RECORDED_KINDS)


def live_profiles() -> tuple[SourceKindProfile, ...]:
    """Mode B profiles, in catalogue order."""
    return tuple(p for p in SOURCE_KIND_PROFILES if p.kind in LIVE_KINDS)
