"""Mode A: recorded real-world traffic. The primary capstone input.

A recorded source is bounded, replayable and usually seekable, and it runs on a
:class:`~app.core.clock.VirtualClock`. Those four properties are what make it
useful for engineering rather than merely convenient:

* **Bounded** -- a run terminates, so a scenario is a finite, comparable unit.
* **Replayable** -- the same input produces the same samples, which is the
  precondition for regression-testing a decision change. Without it, "the route
  changed" can never be attributed to the change we made.
* **Seekable** -- a specific incident can be examined directly instead of
  waiting for it to recur.
* **Virtual clock** -- replay rate does not contaminate any measured quantity.

The epoch problem
-----------------
A recorded file has a timeline starting at zero; the world does not. Mapping
media time to absolute time requires knowing when the capture began, and video
files frequently do not record it. That value must therefore be supplied
explicitly and, when it is not genuinely known, marked
:attr:`~app.domain.provenance.ValueSource.ASSUMED` on everything derived from
it. It must never quietly default to "now": that turns a two-year-old recording
into apparently live traffic, and every staleness check downstream would then
pass on stale evidence.
"""

from __future__ import annotations

import abc
import datetime as dt

from app.core.clock import VirtualClock
from app.core.errors import SourceCapabilityError
from app.ingestion.base import (
    RECORDED_KINDS,
    SourceCapabilities,
    SourceKind,
    TrafficDataSource,
)


class RecordedTrafficDataSource(TrafficDataSource, abc.ABC):
    """Base class for every Mode A source.

    Concrete implementations arrive in Phase 2 (datasets and observation replay)
    and Phase 3 (video decoding). This class fixes the contract they satisfy and
    enforces the capability invariants that make Mode A meaningful.
    """

    def __init__(
        self,
        *,
        source_id: str,
        epoch: dt.datetime,
        junction_id: str | None = None,
        rate: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        epoch:
            Absolute instant corresponding to media time zero. Required, never
            defaulted -- see the module docstring.
        rate:
            Wall-clock replay speed. Affects only how fast samples are handed
            out; it does not affect any timestamp, because those come from the
            media timeline.
        """
        super().__init__(source_id=source_id, junction_id=junction_id)
        if epoch.tzinfo is None:
            msg = "recorded source epoch must be timezone-aware"
            raise ValueError(msg)
        self._clock = VirtualClock(epoch=epoch, rate=rate)

    @property
    def clock(self) -> VirtualClock:
        """The media-timeline clock this source drives."""
        return self._clock

    @property
    def position_s(self) -> float:
        """Current position on the media timeline, in seconds."""
        return self._clock.media_time_s

    @property
    @abc.abstractmethod
    def duration_s(self) -> float | None:
        """Total length of the recording, when the container reports one."""

    @abc.abstractmethod
    def seek(self, media_time_s: float) -> None:
        """Reposition to an absolute point on the media timeline.

        Implementations must call ``self.capabilities.require_seekable()`` first,
        so that a source which cannot seek (a streamed archive, a compressed
        record file) rejects the request rather than silently ignoring it.
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """Return to the start so the run can be replayed identically."""

    def set_rate(self, rate: float) -> None:
        """Change wall-clock replay speed.

        Rejected when the source has not declared variable-rate support, rather
        than accepted and ignored: a caller that believes it is replaying at 4x
        and is not will draw wrong conclusions about throughput.
        """
        if not self.capabilities.supports_variable_rate:
            msg = "this source does not support variable-rate replay"
            raise SourceCapabilityError(msg, capability="supports_variable_rate")
        self._clock.set_rate(rate)

    @staticmethod
    def validate_capabilities(kind: SourceKind, capabilities: SourceCapabilities) -> None:
        """Assert that a declared capability set is coherent for Mode A.

        Called by implementations at construction. Catching an incoherent
        declaration here is far better than discovering it when a pipeline that
        trusted ``replayable`` produces different results on the second run.
        """
        if kind not in RECORDED_KINDS:
            msg = f"{kind.value} is not a recorded (Mode A) source kind"
            raise ValueError(msg)
        if not capabilities.bounded:
            msg = "a recorded source must be bounded; an endless recording is a live stream"
            raise ValueError(msg)
        if not capabilities.replayable:
            msg = (
                "a recorded source must be replayable; without deterministic "
                "re-reads it cannot support regression testing, which is the main "
                "reason to prefer recorded input"
            )
            raise ValueError(msg)
        if capabilities.realtime:
            msg = (
                "a recorded source must not declare realtime=True; it is not "
                "constrained to wall-clock rate and does not lose data when the "
                "consumer is slow"
            )
            raise ValueError(msg)
        if capabilities.requires_backpressure:
            msg = (
                "a recorded source must not require backpressure; the reader "
                "controls the pace, so the queue cannot grow without bound"
            )
            raise ValueError(msg)
