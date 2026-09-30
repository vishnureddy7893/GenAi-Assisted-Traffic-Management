"""Traffic data source abstraction.

The convergence requirement
---------------------------
Recorded and live inputs must reach the same decision pipeline. The tempting way
to achieve that is to pretend they are the same thing -- one ``Stream`` type,
and live sources simply never seek. That pretence fails in both directions:

* Code written for a live stream calls ``time.time()``, and replaying a recorded
  video at 10x silently multiplies every speed and ETA by 10.
* Code written for a recorded file assumes it can rewind, buffer without limit,
  and that a read always eventually returns. A live camera honours none of that.

So the sources are *not* the same, and this module says so explicitly through
:class:`SourceCapabilities`. What converges is the **output**: both kinds emit
:class:`SourceSample` values on a clock they supply, and everything downstream
consumes only those two things. The differences are declared, checked, and
confined to this layer.

Why samples are dataclasses and not pydantic models
---------------------------------------------------
A ``SourceSample`` carries a decoded video frame and is produced tens of times
per second per camera. Pydantic validation on that path buys nothing -- the
payload is a numpy array whose contents no schema can check -- and costs real
throughput. Schema validation belongs at the boundary where structured meaning
first appears, which is :class:`~app.domain.observation.JunctionObservation`.
"""

from __future__ import annotations

import abc
import datetime as dt
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from types import TracebackType
from typing import Annotated, Any, Self

from pydantic import Field

from app.core.clock import Clock
from app.core.errors import SourceCapabilityError
from app.domain.base import DomainModel

SourceId = Annotated[str, Field(min_length=1, max_length=64)]


class SourceKind(StrEnum):
    """Concrete kind of input, within Mode A (recorded) or Mode B (live)."""

    # --- Mode A: recorded -------------------------------------------------
    RECORDED_IMAGE = "recorded_image"
    """A single traffic still. The degenerate recorded source: one sample,
    bounded, replayable, trivially seekable. Useful on its own for calibration
    and single-frame inspection."""

    RECORDED_VIDEO = "recorded_video"
    """A video file of an intersection. Requires the full CV pipeline."""

    RECORDED_DETECTION_DATASET = "recorded_detection_dataset"
    """Pre-annotated detections or tracks. Skips detection, keeps tracking and
    analytics -- the fastest path to end-to-end validation, because it isolates
    decision quality from detector quality."""

    RECORDED_COUNT_DATASET = "recorded_count_dataset"
    """Tabular vehicle counts / flows / speeds, e.g. loop-detector or agency
    open data. No CV at all; feeds analytics directly."""

    RECORDED_OBSERVATIONS = "recorded_observations"
    """Serialised JunctionObservation records previously produced by this
    system. The replay format for regression testing a decision change against
    a fixed input."""

    # --- Mode B: live -----------------------------------------------------
    LIVE_RTSP = "live_rtsp"
    LIVE_HTTP_STREAM = "live_http_stream"
    LIVE_WEBCAM = "live_webcam"
    LIVE_API_FEED = "live_api_feed"
    """A polled JSON/REST traffic feed rather than video."""


#: Kinds that belong to Mode A. Membership is what the capability invariants and
#: the registry check against, so a new kind cannot be added without deciding
#: which mode it belongs to.
RECORDED_KINDS: frozenset[SourceKind] = frozenset(
    {
        SourceKind.RECORDED_IMAGE,
        SourceKind.RECORDED_VIDEO,
        SourceKind.RECORDED_DETECTION_DATASET,
        SourceKind.RECORDED_COUNT_DATASET,
        SourceKind.RECORDED_OBSERVATIONS,
    }
)

#: Kinds that belong to Mode B.
LIVE_KINDS: frozenset[SourceKind] = frozenset(
    {
        SourceKind.LIVE_RTSP,
        SourceKind.LIVE_HTTP_STREAM,
        SourceKind.LIVE_WEBCAM,
        SourceKind.LIVE_API_FEED,
    }
)


class PayloadKind(StrEnum):
    """What a sample carries."""

    VIDEO_FRAME = "video_frame"
    DETECTION_RECORD = "detection_record"
    TABULAR_RECORD = "tabular_record"
    OBSERVATION = "observation"


class SourceStatus(StrEnum):
    """Lifecycle state of a source."""

    CLOSED = "closed"
    OPENING = "opening"
    READY = "ready"
    DEGRADED = "degraded"
    """Still producing, but dropping samples or reconnecting. Distinct from
    ``FAILED``: a live source that drops 5% of frames is usable with reduced
    confidence, and treating that as failure would discard usable evidence."""

    EXHAUSTED = "exhausted"
    """Bounded source reached its end. Only reachable for recorded sources."""

    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SourceSample:
    """One unit read from a source.

    ``media_time_s`` and ``captured_at`` are both present and both meaningful:

    * ``media_time_s`` is the position on the source timeline. Defined for
      recorded sources; ``None`` for live ones, which have no timeline to be at
      a position on.
    * ``captured_at`` is the absolute instant the evidence was captured, derived
      from the source clock. For a recorded source this comes from the clock
      epoch plus media time, so a replay of last year's video produces
      observations correctly dated last year rather than dated now.
    """

    source_id: str
    sequence: int
    """Monotonic index within this source. A gap indicates dropped samples."""

    payload_kind: PayloadKind
    payload: Any
    """Decoded frame (ndarray, Phase 3), parsed record, or observation. Typed
    loosely on purpose: constraining it here would force this module to import
    numpy, which the tabular and observation paths never need."""

    captured_at: dt.datetime
    media_time_s: float | None = None
    dropped_before: int = 0
    """Samples discarded by backpressure immediately before this one. Carried on
    the sample rather than only counted globally, so an observation built from a
    gappy stretch can be flagged with
    :attr:`~app.domain.enums.DataQualityFlag.DROPPED_FRAMES` at the point of
    use, not merely tallied in a metric nobody reads."""

    metadata: dict[str, Any] = field(default_factory=dict)


class SourceCapabilities(DomainModel):
    """What a source can and cannot do. Declared, not assumed.

    Serialised to the API so the frontend can enable a seek control for a video
    file and hide it for an RTSP stream -- driven by the source's own declaration
    rather than by the UI guessing from a kind string.
    """

    seekable: bool
    """Supports repositioning to an arbitrary point. Live sources never do."""

    bounded: bool
    """Has a defined end. Recorded sources do; live sources do not, which is why
    'read everything then process' is not an available strategy for them."""

    replayable: bool
    """Produces identical samples on re-read. The prerequisite for regression
    testing a decision change, and false for anything live."""

    realtime: bool
    """Produces samples at wall-clock rate and cannot be slowed down. Implies
    that a slow consumer loses data rather than delaying the producer."""

    supports_variable_rate: bool = False
    """Can be replayed faster or slower than real time."""

    requires_backpressure: bool = False
    """Consumer must apply an explicit drop policy or the queue grows without
    bound. True for every live source."""

    may_drop_samples: bool = False
    nominal_fps: Annotated[float, Field(gt=0)] | None = None
    duration_s: Annotated[float, Field(ge=0)] | None = None
    """``None`` for unbounded sources -- the honest value, distinct from 0."""

    sample_count: Annotated[int, Field(ge=0)] | None = None
    payload_kind: PayloadKind

    def require_seekable(self) -> None:
        """Raise unless this source supports seeking.

        Called by seek entry points so an unsupported seek fails loudly. A
        silently ignored seek is the dangerous case: the caller believes it is
        analysing 14:05 and is in fact analysing whatever is playing now.
        """
        if not self.seekable:
            msg = "this source is not seekable"
            raise SourceCapabilityError(msg, capability="seekable")


class SourceHealth(DomainModel):
    """Runtime health of an open source."""

    status: SourceStatus
    samples_read: Annotated[int, Field(ge=0)] = 0
    samples_dropped: Annotated[int, Field(ge=0)] = 0
    last_sample_at: dt.datetime | None = None
    latency_s: float | None = None
    """Live sources only: capture-to-availability delay. Meaningless for
    recorded input, where it would measure disk speed rather than anything
    about the traffic."""

    consecutive_errors: Annotated[int, Field(ge=0)] = 0
    message: Annotated[str, Field(max_length=500)] | None = None

    @property
    def drop_rate(self) -> float | None:
        """Fraction of samples lost, or ``None`` when nothing has been read."""
        total = self.samples_read + self.samples_dropped
        return None if total == 0 else self.samples_dropped / total


class SourceDescriptor(DomainModel):
    """API-facing description of a configured source."""

    source_id: SourceId
    kind: SourceKind
    display_name: Annotated[str, Field(max_length=200)] | None = None
    junction_id: Annotated[str, Field(max_length=64)] | None = None
    """Which junction this source observes. A source not bound to a junction
    cannot contribute to network state."""

    capabilities: SourceCapabilities
    status: SourceStatus = SourceStatus.CLOSED

    @property
    def mode(self) -> str:
        """``"A"`` for recorded input, ``"B"`` for live input."""
        return "A" if self.kind in RECORDED_KINDS else "B"


class TrafficDataSource(abc.ABC):
    """Base class for every input to the system.

    Subclasses implement :meth:`open`, :meth:`close` and :meth:`read`. Everything
    downstream of ingestion depends only on this interface plus the clock, which
    is what makes Mode A and Mode B interchangeable to the decision pipeline.
    """

    def __init__(self, *, source_id: str, junction_id: str | None = None) -> None:
        self._source_id = source_id
        self._junction_id = junction_id
        self._status = SourceStatus.CLOSED
        self._sequence = 0

    # --- identity ---------------------------------------------------------
    @property
    def source_id(self) -> str:
        """Stable identifier for this source, recorded in every observation."""
        return self._source_id

    @property
    def junction_id(self) -> str | None:
        """Junction this source observes, if bound to one."""
        return self._junction_id

    @property
    def status(self) -> SourceStatus:
        """Current lifecycle state."""
        return self._status

    @property
    @abc.abstractmethod
    def kind(self) -> SourceKind:
        """Concrete kind of this source."""

    @property
    @abc.abstractmethod
    def capabilities(self) -> SourceCapabilities:
        """Declared capabilities. Must be stable for the life of the source."""

    @property
    @abc.abstractmethod
    def clock(self) -> Clock:
        """Clock this source's timestamps are expressed on.

        Consumers must read time from here rather than from the wall clock. This
        is the single mechanism that keeps recorded replay dimensionally correct.
        """

    # --- lifecycle --------------------------------------------------------
    @abc.abstractmethod
    def open(self) -> None:
        """Acquire the underlying resource and make the source readable."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release the underlying resource. Must be idempotent."""

    @abc.abstractmethod
    def read(self) -> SourceSample | None:
        """Return the next sample, or ``None`` when a bounded source is exhausted.

        An unbounded source never returns ``None``; it blocks or raises. Callers
        must therefore not use ``None`` as a general loop-termination condition
        without first checking :attr:`SourceCapabilities.bounded`.
        """

    @abc.abstractmethod
    def health(self) -> SourceHealth:
        """Current health. Cheap enough to poll on every decision cycle."""

    # --- convenience ------------------------------------------------------
    def describe(self) -> SourceDescriptor:
        """Serialisable description for the API."""
        return SourceDescriptor(
            source_id=self.source_id,
            kind=self.kind,
            junction_id=self.junction_id,
            capabilities=self.capabilities,
            status=self.status,
        )

    def __iter__(self) -> Iterator[SourceSample]:
        """Iterate samples until a bounded source is exhausted."""
        while True:
            sample = self.read()
            if sample is None:
                return
            yield sample

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _next_sequence(self) -> int:
        """Allocate the next monotonic sample index."""
        value = self._sequence
        self._sequence += 1
        return value
