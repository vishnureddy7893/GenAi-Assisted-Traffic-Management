"""Mode A: a single recorded traffic image.

A still image is the degenerate recorded source: bounded, replayable, trivially
seekable, and one sample long. Modelling it as a
:class:`~app.ingestion.recorded.RecordedTrafficDataSource` rather than as a
special case is what lets the same downstream pipeline consume a photograph, a
video clip and (later) a live feed without knowing which it received.

Frames are yielded as numpy arrays in RGB order -- the same payload type the
video source produces -- so a consumer genuinely cannot tell the two apart. That
is the whole point of the abstraction, and it would be defeated by handing back
a PIL object here and an array there.

Nothing in this module interprets the image. It is decoded and handed on. No
detection, no classification, no counting.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from app.core.errors import DataSourceError, SourceCapabilityError
from app.core.logging import get_logger
from app.ingestion.assets import ImageMetadata
from app.ingestion.base import (
    PayloadKind,
    SourceCapabilities,
    SourceHealth,
    SourceKind,
    SourceSample,
    SourceStatus,
)
from app.ingestion.probe import probe_image
from app.ingestion.recorded import RecordedTrafficDataSource

logger = get_logger(__name__)

#: Nominal spacing between frames of an animated still, in seconds. Applies only
#: to multi-frame images, where the container's own timing is not read in Phase 2.
ANIMATED_FRAME_INTERVAL_S = 1.0


class RecordedImageDataSource(RecordedTrafficDataSource):
    """A single image file (JPEG, PNG or WebP) presented as a data source."""

    def __init__(
        self,
        *,
        source_id: str,
        path: Path,
        epoch: dt.datetime,
        metadata: ImageMetadata | None = None,
        junction_id: str | None = None,
    ) -> None:
        super().__init__(source_id=source_id, epoch=epoch, junction_id=junction_id)
        self._path = Path(path)
        self._metadata = metadata
        self._frame_index = 0
        self._samples_read = 0
        self._image: Any = None
        self._capabilities: SourceCapabilities | None = None

    # --- identity ---------------------------------------------------------
    @property
    def kind(self) -> SourceKind:
        return SourceKind.RECORDED_IMAGE

    @property
    def path(self) -> Path:
        """File this source reads from."""
        return self._path

    @property
    def metadata(self) -> ImageMetadata:
        """Measured image metadata. Probed on first access."""
        if self._metadata is None:
            self._metadata = probe_image(self._path)
        return self._metadata

    @property
    def frame_count(self) -> int:
        """Frames available. 1 for a still, ``n_frames`` for an animated image."""
        return self.metadata.frame_count or 1

    @property
    def capabilities(self) -> SourceCapabilities:
        """Declared capabilities, derived from the measured image."""
        if self._capabilities is None:
            count = self.frame_count
            self._capabilities = SourceCapabilities(
                seekable=True,
                bounded=True,
                replayable=True,
                realtime=False,
                supports_variable_rate=False,
                requires_backpressure=False,
                may_drop_samples=False,
                nominal_fps=None,
                # A still has no duration. An animated image is given a nominal
                # one so that seeking has a meaningful axis; it is not read from
                # the container, so it is not presented as measured timing.
                duration_s=(count - 1) * ANIMATED_FRAME_INTERVAL_S if count > 1 else None,
                sample_count=count,
                payload_kind=PayloadKind.VIDEO_FRAME,
            )
            RecordedTrafficDataSource.validate_capabilities(self.kind, self._capabilities)
        return self._capabilities

    @property
    def duration_s(self) -> float | None:
        """Nominal span of the frame sequence; ``None`` for a still image."""
        return self.capabilities.duration_s

    # --- lifecycle --------------------------------------------------------
    def open(self) -> None:
        """Decode the image and make it readable. Idempotent."""
        if self._status is SourceStatus.READY:
            return
        self._status = SourceStatus.OPENING
        try:
            self._metadata = probe_image(self._path)
            self._image = self._load_frames()
        except DataSourceError:
            self._status = SourceStatus.FAILED
            raise
        self._frame_index = 0
        self._clock.seek_to(0.0)
        self._status = SourceStatus.READY
        logger.info(
            "image source opened",
            extra={
                "source_id": self.source_id,
                "width": self._metadata.width,
                "height": self._metadata.height,
                "format": self._metadata.format,
                "frames": self.frame_count,
            },
        )

    def _load_frames(self) -> list[Any]:
        """Decode every frame to an RGB array.

        A still decodes to a one-element list, so :meth:`read` has a single code
        path rather than branching on whether the image happens to be animated.
        """
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - dependency-gated
            msg = "Image ingestion requires Pillow and numpy."
            raise DataSourceError(msg, missing_dependency=str(exc)) from exc

        frames: list[Any] = []
        with Image.open(self._path) as img:
            total = self.frame_count
            for index in range(total):
                if total > 1:
                    img.seek(index)
                # Normalise to RGB so every consumer receives the same shape,
                # regardless of whether the source was greyscale or had alpha.
                frames.append(np.asarray(img.convert("RGB")))
        return frames

    def close(self) -> None:
        """Release the decoded image. Idempotent."""
        self._image = None
        self._status = SourceStatus.CLOSED

    def read(self) -> SourceSample | None:
        """Return the next frame, or ``None`` once the image is exhausted."""
        if self._status is not SourceStatus.READY:
            msg = "Source must be opened before reading."
            raise DataSourceError(msg, source_id=self.source_id)
        if self._image is None or self._frame_index >= len(self._image):
            self._status = SourceStatus.EXHAUSTED
            return None

        media_time = self._frame_index * ANIMATED_FRAME_INTERVAL_S
        self._clock.seek_to(media_time)
        frame = self._image[self._frame_index]
        sample = SourceSample(
            source_id=self.source_id,
            sequence=self._next_sequence(),
            payload_kind=PayloadKind.VIDEO_FRAME,
            payload=frame,
            captured_at=self._clock.now(),
            media_time_s=media_time,
            metadata={"frame_index": self._frame_index, "still_image": self.frame_count == 1},
        )
        self._frame_index += 1
        self._samples_read += 1
        return sample

    # --- recorded-source controls ----------------------------------------
    def seek(self, media_time_s: float) -> None:
        """Reposition within the frame sequence."""
        self.capabilities.require_seekable()
        if media_time_s < 0:
            msg = "Cannot seek to a negative position."
            raise SourceCapabilityError(msg, capability="seekable", position=media_time_s)
        index = round(media_time_s / ANIMATED_FRAME_INTERVAL_S)
        self._frame_index = min(index, self.frame_count)
        self._clock.seek_to(media_time_s)
        if self._status is SourceStatus.EXHAUSTED and self._frame_index < self.frame_count:
            self._status = SourceStatus.READY

    def reset(self) -> None:
        """Return to the first frame so the run replays identically."""
        self._frame_index = 0
        self._clock.seek_to(0.0)
        if self._status is SourceStatus.EXHAUSTED:
            self._status = SourceStatus.READY

    @property
    def current_frame_index(self) -> int:
        """Index of the frame that :meth:`read` will return next."""
        return self._frame_index

    def health(self) -> SourceHealth:
        """Current health of this source."""
        return SourceHealth(
            status=self._status,
            samples_read=self._samples_read,
            samples_dropped=0,
            message=None,
        )
