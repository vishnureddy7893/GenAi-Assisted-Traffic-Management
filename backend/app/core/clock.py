"""Time abstraction shared by recorded and live pipelines.

The problem this solves
-----------------------
Every downstream traffic computation is a rate: speed is metres per second,
flow is vehicles per hour, queue growth is vehicles per second, ETA is seconds
into the future. If any of that code calls :func:`time.time` directly, replaying
a recorded video at 10x produces speeds that are wrong by 10x and ETAs that are
wrong by 10x -- silently, with no error anywhere.

So *the data source owns the clock*, and every consumer reads time from the
clock the source hands it:

* :class:`WallClock`    -- live sources. Time is real and advances on its own.
* :class:`VirtualClock` -- recorded sources. Time is the media timeline and only
  advances when a sample is decoded. Replay rate has no effect on any computed
  traffic quantity, which is exactly the property we need for reproducibility.

This is what makes the requirement "recorded and live converge on the same
downstream pipeline" actually true rather than aspirational: the pipeline is
written once against :class:`Clock` and is agnostic to which one it received.
"""

from __future__ import annotations

import abc
import datetime as dt
import time
from enum import StrEnum

EPOCH_UTC = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)


class ClockKind(StrEnum):
    """Which time base a clock represents."""

    WALL = "wall"
    """Real elapsed time. Advances by itself. Used by live sources."""

    VIRTUAL = "virtual"
    """Media/simulation timeline. Advances only when explicitly driven."""


class Clock(abc.ABC):
    """Read-only time source consumed by the analytics and decision pipeline."""

    kind: ClockKind

    @abc.abstractmethod
    def now(self) -> dt.datetime:
        """Return the current timezone-aware timestamp on this clock's time base."""

    @abc.abstractmethod
    def monotonic(self) -> float:
        """Return a non-decreasing seconds value suitable for measuring durations.

        Never derive a duration by subtracting two :meth:`now` values in code
        that could run against a wall clock: NTP steps and DST make that unsafe.
        """

    def elapsed_since(self, earlier: dt.datetime) -> float:
        """Seconds elapsed on this clock since ``earlier``.

        Used for observation staleness. A negative result means the observation
        is timestamped in the future, which indicates clock skew between a camera
        and this host -- a real condition worth surfacing rather than clamping.
        """
        if earlier.tzinfo is None:
            msg = "timestamps must be timezone-aware; got a naive datetime"
            raise ValueError(msg)
        return (self.now() - earlier).total_seconds()


class WallClock(Clock):
    """Real time. The only correct clock for a live, unbounded source."""

    kind = ClockKind.WALL

    def now(self) -> dt.datetime:
        return dt.datetime.now(tz=dt.UTC)

    def monotonic(self) -> float:
        return time.monotonic()


class VirtualClock(Clock):
    """Media-timeline time, driven explicitly by the source that decodes samples.

    Parameters
    ----------
    epoch:
        Wall-clock instant that media time ``0.0`` corresponds to. For a recorded
        capture this should be the true capture start time when the dataset
        provides one, so that derived observations carry meaningful absolute
        timestamps. When the dataset does not record it, the caller must pass an
        explicit value and mark downstream provenance as ``ASSUMED`` -- it must
        never be silently defaulted to "now".
    rate:
        Wall-clock replay speed multiplier. Recorded purely for telemetry: it
        deliberately does **not** affect :meth:`now`, because the media timeline
        is the ground truth regardless of how fast we walk through it.
    """

    kind = ClockKind.VIRTUAL

    def __init__(self, *, epoch: dt.datetime, rate: float = 1.0) -> None:
        if epoch.tzinfo is None:
            msg = "VirtualClock epoch must be timezone-aware"
            raise ValueError(msg)
        if rate <= 0:
            msg = "VirtualClock rate must be > 0"
            raise ValueError(msg)
        self._epoch = epoch
        self._rate = rate
        self._media_time_s = 0.0

    @property
    def epoch(self) -> dt.datetime:
        """Wall-clock instant corresponding to media time zero."""
        return self._epoch

    @property
    def rate(self) -> float:
        """Replay rate multiplier (telemetry only; does not affect time base)."""
        return self._rate

    @property
    def media_time_s(self) -> float:
        """Current position on the media timeline, in seconds."""
        return self._media_time_s

    def set_rate(self, rate: float) -> None:
        """Change the replay rate without disturbing the timeline position."""
        if rate <= 0:
            msg = "VirtualClock rate must be > 0"
            raise ValueError(msg)
        self._rate = rate

    def advance_to(self, media_time_s: float) -> None:
        """Move the clock to an absolute position on the media timeline.

        Rewinding is permitted only through :meth:`seek_to`, so that an
        accidental out-of-order sample is caught rather than quietly reordering
        the timeline underneath the analytics.
        """
        if media_time_s < self._media_time_s:
            msg = (
                "VirtualClock cannot move backwards via advance_to "
                f"({media_time_s} < {self._media_time_s}); use seek_to for an "
                "intentional seek"
            )
            raise ValueError(msg)
        self._media_time_s = media_time_s

    def seek_to(self, media_time_s: float) -> None:
        """Reposition the clock, forwards or backwards (seekable sources only)."""
        if media_time_s < 0:
            msg = "media time cannot be negative"
            raise ValueError(msg)
        self._media_time_s = media_time_s

    def now(self) -> dt.datetime:
        return self._epoch + dt.timedelta(seconds=self._media_time_s)

    def monotonic(self) -> float:
        return self._media_time_s
