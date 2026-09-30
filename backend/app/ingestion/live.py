"""Mode B: live traffic. Architecture reserved; no streaming until Phase 14.

A live source is unbounded, non-replayable, real-time and not seekable, and it
runs on a :class:`~app.core.clock.WallClock`. Each of those has an operational
consequence that must be designed for rather than discovered:

**Unbounded.** There is no "read it all, then process". Every consumer is a
streaming consumer, and any accumulator must have a bounded window.

**Real-time with backpressure.** The camera produces frames whether or not we
are ready. When the consumer falls behind, something must give, and the only
honest options are to drop the oldest frames, drop the newest, or block the
producer. Blocking accumulates latency without bound -- a decision computed from
a 40-second-old frame is worse than no decision -- so live sources default to
dropping the oldest and *report* what they dropped rather than hiding it.

**Not replayable.** A live incident cannot be re-examined. This is the strongest
practical argument for the recorded pipeline being primary: development and
evaluation need reproducibility that live input structurally cannot provide.

**Not seekable.** :meth:`seek` is implemented here as a hard rejection rather
than left unimplemented, so the failure is a clear error at the call site
instead of an ``AttributeError`` somewhere further in.
"""

from __future__ import annotations

import abc
from enum import StrEnum
from typing import Annotated

from pydantic import Field

from app.core.clock import WallClock
from app.core.errors import SourceCapabilityError
from app.domain.base import DomainModel
from app.ingestion.base import (
    LIVE_KINDS,
    SourceCapabilities,
    SourceKind,
    TrafficDataSource,
)


class BackpressurePolicy(StrEnum):
    """What to discard when the consumer cannot keep up."""

    DROP_OLDEST = "drop_oldest"
    """Keep the newest frames. Correct for traffic: a decision must reflect the
    present, and a stale frame has no value once a newer one exists."""

    DROP_NEWEST = "drop_newest"
    """Keep the oldest. Appropriate only when continuity matters more than
    recency -- e.g. tracking that would lose identity across a gap."""

    BLOCK = "block"
    """Never drop; slow the producer. Available for completeness. For a live
    camera this converts frame loss into unbounded latency, which is the worse
    failure, so it is never the default."""


class LiveSourceConfig(DomainModel):
    """Runtime configuration for a live source."""

    uri: Annotated[str, Field(min_length=1, max_length=2000)]
    """Connection URI. Held only in memory, sourced from the environment. RTSP
    URIs frequently embed credentials, so this value must never be logged or
    returned by the API -- the descriptor exposes the source id, not the URI."""

    target_fps: Annotated[float, Field(gt=0)] = 10.0
    """Processing rate. Usually well below camera fps: traffic state changes on
    a scale of seconds, so analysing every frame of 30 fps video multiplies cost
    without improving any decision."""

    backpressure: BackpressurePolicy = BackpressurePolicy.DROP_OLDEST
    max_queue_frames: Annotated[int, Field(gt=0)] = 60
    reconnect_initial_delay_s: Annotated[float, Field(gt=0)] = 1.0
    reconnect_max_delay_s: Annotated[float, Field(gt=0)] = 30.0
    """Exponential backoff ceiling. A camera that drops out during an incident
    is exactly when a tight reconnect loop would saturate the network it is
    trying to reach."""

    max_consecutive_errors: Annotated[int, Field(gt=0)] = 10
    """After this many failures the source is marked FAILED and stops
    contributing. A source that reconnects forever while producing nothing looks
    healthy on a dashboard while the junction it covers goes dark."""


class LiveTrafficDataSource(TrafficDataSource, abc.ABC):
    """Base class for every Mode B source.

    Reserved for Phase 14. The contract exists now so that the ingestion,
    analytics and decision layers built in Phases 2-13 are written against an
    interface that already accounts for unbounded, lossy, non-replayable input --
    rather than being written for files and retrofitted later, which is how a
    system ends up with an unbounded buffer in its hot path.
    """

    def __init__(
        self,
        *,
        source_id: str,
        config: LiveSourceConfig,
        junction_id: str | None = None,
    ) -> None:
        super().__init__(source_id=source_id, junction_id=junction_id)
        self._config = config
        self._clock = WallClock()

    @property
    def config(self) -> LiveSourceConfig:
        """Runtime configuration for this source."""
        return self._config

    @property
    def clock(self) -> WallClock:
        """Real time. A live source has no other meaningful time base."""
        return self._clock

    @abc.abstractmethod
    def connect(self) -> None:
        """Establish the underlying connection."""

    @abc.abstractmethod
    def reconnect(self) -> None:
        """Re-establish a dropped connection, respecting the configured backoff."""

    @property
    @abc.abstractmethod
    def queue_depth(self) -> int:
        """Samples currently buffered. A persistently rising depth means the
        consumer is losing the race and drops are imminent."""

    @property
    @abc.abstractmethod
    def dropped_count(self) -> int:
        """Total samples discarded by backpressure since opening."""

    def seek(self, media_time_s: float) -> None:
        """Always rejected: a live stream has no timeline to seek within."""
        msg = (
            f"cannot seek a live source to {media_time_s}s: live input is "
            "unbounded and has no seekable timeline"
        )
        raise SourceCapabilityError(msg, capability="seekable", source_id=self.source_id)

    @staticmethod
    def validate_capabilities(kind: SourceKind, capabilities: SourceCapabilities) -> None:
        """Assert that a declared capability set is coherent for Mode B."""
        if kind not in LIVE_KINDS:
            msg = f"{kind.value} is not a live (Mode B) source kind"
            raise ValueError(msg)
        if capabilities.bounded:
            msg = "a live source cannot be bounded; there is no end to a camera feed"
            raise ValueError(msg)
        if capabilities.seekable:
            msg = "a live source cannot be seekable"
            raise ValueError(msg)
        if capabilities.replayable:
            msg = (
                "a live source cannot be replayable; claiming otherwise would let "
                "a regression test silently depend on traffic that will never recur"
            )
            raise ValueError(msg)
        if not capabilities.realtime:
            msg = "a live source must declare realtime=True"
            raise ValueError(msg)
        if not capabilities.requires_backpressure:
            msg = (
                "a live source must declare requires_backpressure=True; without an "
                "explicit drop policy a slow consumer grows the queue without bound "
                "and latency rises until the decisions are worthless"
            )
            raise ValueError(msg)
