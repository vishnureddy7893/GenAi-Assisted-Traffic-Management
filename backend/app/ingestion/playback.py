"""Open video playback sessions.

A :class:`~app.ingestion.video_source.RecordedVideoDataSource` holds an OS-level
decoder handle, so sessions are pooled and explicitly closed rather than being
created per request. An unbounded pool would exhaust file descriptors under
repeated use, so the manager evicts the least recently used session once the cap
is reached.

The media epoch
---------------
A recorded file has a timeline starting at zero; the world does not. Mapping one
onto the other needs the capture time, which video containers frequently omit.
This module uses the file's modification time and treats it as an **assumption**,
not a measurement -- documented here and carried into provenance by whatever
consumes the samples.

What it must never do is default to "now". That would make a two-year-old clip
look like live traffic, and every staleness check downstream would then pass on
stale evidence.
"""

from __future__ import annotations

import datetime as dt
from collections import OrderedDict
from pathlib import Path

from app.core.logging import get_logger
from app.ingestion.assets import VideoMetadata
from app.ingestion.video_source import RecordedVideoDataSource

logger = get_logger(__name__)

#: Maximum simultaneously open decoders.
MAX_OPEN_SESSIONS = 8


def epoch_for(path: Path) -> dt.datetime:
    """Assumed capture instant for a recorded file.

    Uses the file's modification time. This is an assumption and is documented
    as one; it is not presented anywhere as a measured capture time.
    """
    try:
        stamp = path.stat().st_mtime
    except OSError:
        # A file we cannot stat is about to fail to open anyway; the epoch value
        # is irrelevant at that point, but it must still be timezone-aware.
        return dt.datetime(1970, 1, 1, tzinfo=dt.UTC)
    return dt.datetime.fromtimestamp(stamp, tz=dt.UTC)


class PlaybackManager:
    """LRU pool of open video sources, keyed by media id."""

    def __init__(self, max_sessions: int = MAX_OPEN_SESSIONS) -> None:
        self._sessions: OrderedDict[str, RecordedVideoDataSource] = OrderedDict()
        self._max_sessions = max_sessions

    def open(
        self,
        media_id: str,
        path: Path,
        metadata: VideoMetadata | None = None,
    ) -> RecordedVideoDataSource:
        """Open (or return) a session for this media id."""
        existing = self._sessions.get(media_id)
        if existing is not None:
            self._sessions.move_to_end(media_id)
            return existing

        source = RecordedVideoDataSource(
            source_id=f"video:{media_id}",
            path=path,
            epoch=epoch_for(path),
            metadata=metadata,
        )
        source.open()

        self._sessions[media_id] = source
        self._sessions.move_to_end(media_id)
        self._evict_if_needed()
        return source

    def get(self, media_id: str) -> RecordedVideoDataSource | None:
        """Return an open session, or ``None`` if there is not one."""
        source = self._sessions.get(media_id)
        if source is not None:
            self._sessions.move_to_end(media_id)
        return source

    def close(self, media_id: str) -> None:
        """Close and discard a session. Safe to call when none is open."""
        source = self._sessions.pop(media_id, None)
        if source is not None:
            source.close()

    def close_all(self) -> None:
        """Close every open session. Used at shutdown and by tests."""
        for source in list(self._sessions.values()):
            source.close()
        self._sessions.clear()

    @property
    def open_count(self) -> int:
        """Number of currently open sessions."""
        return len(self._sessions)

    def _evict_if_needed(self) -> None:
        """Close least recently used sessions until within the cap."""
        while len(self._sessions) > self._max_sessions:
            media_id, source = self._sessions.popitem(last=False)
            source.close()
            logger.info(
                "playback session evicted",
                extra={"media_id": media_id, "reason": "session cap reached"},
            )


#: Process-wide session pool.
playback_manager = PlaybackManager()
