"""Mode A: a recorded traffic video.

The canonical recorded source: bounded, seekable, replayable and driven by a
:class:`~app.core.clock.VirtualClock`, so frame timestamps come from the media
timeline rather than from wall-clock time. Replaying a clip at 4x therefore
produces identical timestamps to replaying it at 1x, which is what keeps every
downstream rate calculation correct.

Frames are yielded as numpy arrays in **RGB** order. OpenCV decodes to BGR, so
the conversion happens here, once, at the boundary. Leaving BGR to leak
downstream is a classic source of silently wrong colour handling, and it would
also make this source's payload differ from the image source's for no reason.

**On OpenCV.** Used strictly as a container/codec reader: open, read, seek,
query properties, release. No vision algorithm is invoked anywhere in this file.
Decoding a video is not computer vision.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from app.core.errors import DataSourceError, SourceCapabilityError
from app.core.logging import get_logger
from app.ingestion.assets import VideoMetadata
from app.ingestion.base import (
    PayloadKind,
    SourceCapabilities,
    SourceHealth,
    SourceKind,
    SourceSample,
    SourceStatus,
)
from app.ingestion.probe import probe_video
from app.ingestion.recorded import RecordedTrafficDataSource

logger = get_logger(__name__)

#: Assumed frame rate when a container declares none. Used only to give seeking
#: a usable axis; it is never reported as measured metadata, and
#: `VideoMetadata.fps` stays `None` so nothing downstream mistakes it for fact.
FALLBACK_SEEK_FPS = 25.0


class RecordedVideoDataSource(RecordedTrafficDataSource):
    """A recorded video file (MP4, MOV or AVI) presented as a data source."""

    def __init__(
        self,
        *,
        source_id: str,
        path: Path,
        epoch: dt.datetime,
        metadata: VideoMetadata | None = None,
        junction_id: str | None = None,
        rate: float = 1.0,
    ) -> None:
        super().__init__(
            source_id=source_id, epoch=epoch, junction_id=junction_id, rate=rate
        )
        self._path = Path(path)
        self._metadata = metadata
        self._capture: Any = None
        self._frame_index = 0
        self._samples_read = 0
        self._capabilities: SourceCapabilities | None = None

    # --- identity ---------------------------------------------------------
    @property
    def kind(self) -> SourceKind:
        return SourceKind.RECORDED_VIDEO

    @property
    def path(self) -> Path:
        """File this source reads from."""
        return self._path

    @property
    def metadata(self) -> VideoMetadata:
        """Measured container metadata. Probed on first access."""
        if self._metadata is None:
            self._metadata = probe_video(self._path)
        return self._metadata

    @property
    def capabilities(self) -> SourceCapabilities:
        """Declared capabilities, derived from the measured container."""
        if self._capabilities is None:
            meta = self.metadata
            self._capabilities = SourceCapabilities(
                seekable=True,
                bounded=True,
                replayable=True,
                realtime=False,
                supports_variable_rate=True,
                requires_backpressure=False,
                may_drop_samples=False,
                nominal_fps=meta.fps,
                duration_s=meta.duration_s,
                sample_count=meta.frame_count,
                payload_kind=PayloadKind.VIDEO_FRAME,
            )
            RecordedTrafficDataSource.validate_capabilities(self.kind, self._capabilities)
        return self._capabilities

    @property
    def duration_s(self) -> float | None:
        """Container duration, or ``None`` when it cannot be computed."""
        return self.metadata.duration_s

    @property
    def _seek_fps(self) -> float:
        """Frame rate used for time/frame conversion during seeks."""
        return self.metadata.fps or FALLBACK_SEEK_FPS

    # --- lifecycle --------------------------------------------------------
    def open(self) -> None:
        """Open the container and make it readable. Idempotent."""
        if self._status is SourceStatus.READY and self._capture is not None:
            return
        self._status = SourceStatus.OPENING

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - dependency-gated
            self._status = SourceStatus.FAILED
            msg = "Video ingestion requires OpenCV (opencv-python-headless)."
            raise DataSourceError(msg, missing_dependency="cv2") from exc

        if not self._path.exists():
            self._status = SourceStatus.FAILED
            msg = f"Video file not found: {self._path.name}"
            raise DataSourceError(msg, path=self._path.name)

        # Probe before opening for reading: a corrupt container must fail with a
        # clear message rather than yielding a capture that returns no frames.
        self._metadata = probe_video(self._path)

        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            capture.release()
            self._status = SourceStatus.FAILED
            msg = f"Could not open video {self._path.name}."
            raise DataSourceError(msg, path=self._path.name)

        self._capture = capture
        self._frame_index = 0
        self._clock.seek_to(0.0)
        self._status = SourceStatus.READY
        logger.info(
            "video source opened",
            extra={
                "source_id": self.source_id,
                "width": self._metadata.width,
                "height": self._metadata.height,
                "fps": self._metadata.fps,
                "frame_count": self._metadata.frame_count,
            },
        )

    def close(self) -> None:
        """Release the container. Idempotent."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._status = SourceStatus.CLOSED

    def read(self) -> SourceSample | None:
        """Return the next frame, or ``None`` at end of file."""
        if self._status not in (SourceStatus.READY, SourceStatus.EXHAUSTED):
            msg = "Source must be opened before reading."
            raise DataSourceError(msg, source_id=self.source_id)
        if self._capture is None:
            msg = "Source must be opened before reading."
            raise DataSourceError(msg, source_id=self.source_id)

        ok, frame_bgr = self._capture.read()
        if not ok or frame_bgr is None:
            self._status = SourceStatus.EXHAUSTED
            return None

        import cv2

        # Convert once, at the boundary. BGR leaking downstream is a silent
        # correctness bug and would make this payload differ from the image
        # source's for no reason.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        media_time = self._frame_index / self._seek_fps
        self._clock.seek_to(media_time)

        sample = SourceSample(
            source_id=self.source_id,
            sequence=self._next_sequence(),
            payload_kind=PayloadKind.VIDEO_FRAME,
            payload=frame_rgb,
            captured_at=self._clock.now(),
            media_time_s=media_time,
            metadata={"frame_index": self._frame_index},
        )
        self._frame_index += 1
        self._samples_read += 1
        return sample

    # --- recorded-source controls ----------------------------------------
    def seek(self, media_time_s: float) -> None:
        """Reposition to an absolute point on the media timeline."""
        self.capabilities.require_seekable()
        if media_time_s < 0:
            msg = "Cannot seek to a negative position."
            raise SourceCapabilityError(msg, capability="seekable", position=media_time_s)
        if self._capture is None:
            msg = "Source must be opened before seeking."
            raise DataSourceError(msg, source_id=self.source_id)

        import cv2

        target_frame = round(media_time_s * self._seek_fps)
        total = self.metadata.frame_count
        if total is not None:
            target_frame = min(target_frame, total)

        self._capture.set(cv2.CAP_PROP_POS_FRAMES, float(target_frame))
        self._frame_index = target_frame
        self._clock.seek_to(target_frame / self._seek_fps)

        if self._status is SourceStatus.EXHAUSTED and (
            total is None or target_frame < total
        ):
            self._status = SourceStatus.READY

    def seek_to_frame(self, frame_index: int) -> None:
        """Reposition by frame index rather than by time."""
        if frame_index < 0:
            msg = "Frame index cannot be negative."
            raise SourceCapabilityError(msg, capability="seekable", frame=frame_index)
        self.seek(frame_index / self._seek_fps)

    def reset(self) -> None:
        """Return to the first frame so the run replays identically."""
        if self._capture is None:
            self._frame_index = 0
            self._clock.seek_to(0.0)
            return
        self.seek(0.0)
        self._status = SourceStatus.READY

    @property
    def current_frame_index(self) -> int:
        """Index of the frame that :meth:`read` will return next."""
        return self._frame_index

    @property
    def position_s(self) -> float:
        """Current playback position on the media timeline."""
        return self._clock.media_time_s

    def health(self) -> SourceHealth:
        """Current health of this source."""
        return SourceHealth(
            status=self._status,
            samples_read=self._samples_read,
            samples_dropped=0,
            message=None,
        )
