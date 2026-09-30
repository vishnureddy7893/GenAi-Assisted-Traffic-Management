"""Metadata for ingested media assets.

Every field is optional wherever the container may genuinely fail to report it,
and the extraction code maps "not reported" onto ``None`` rather than onto a
plausible number. This matters more than it looks:

OpenCV returns ``0`` for FPS and ``0`` or ``-1`` for frame count when a
container has no such metadata. Passing those through would produce a video
claiming zero frames per second, and any duration derived from it would be
either zero or a division error. So the extractor converts non-positive
readings to ``None``, and duration is computed **only** when both inputs are
genuinely present and positive.

This is the same rule the rest of the system follows: absence is ``None``, never
zero, because zero is a legitimate value and cannot also mean "unknown".
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated

from pydantic import Field

from app.domain.base import DomainModel
from app.ingestion.media import MediaKind

MediaId = Annotated[str, Field(min_length=1, max_length=64)]


class ImageMetadata(DomainModel):
    """Measured properties of a still image. Nothing here is interpreted."""

    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    format: Annotated[str, Field(min_length=1, max_length=20)]

    mode: Annotated[str, Field(max_length=20)] | None = None
    """Pillow colour mode, e.g. ``RGB``, ``RGBA``, ``L``. ``None`` when the
    decoder does not report one."""

    channels: Annotated[int, Field(gt=0)] | None = None
    """Number of colour bands, derived from ``mode`` when it is known."""

    is_animated: bool | None = None
    frame_count: Annotated[int, Field(gt=0)] | None = None
    """Frames in an animated image. ``None`` for a still, which is different
    from 1: a still has no frame sequence at all."""

    @property
    def megapixels(self) -> float:
        """Resolution in megapixels, for display."""
        return (self.width * self.height) / 1_000_000


class VideoMetadata(DomainModel):
    """Measured properties of a video container.

    Container metadata only. No frame has been analysed and no content has been
    interpreted -- this describes the file, not the traffic in it.
    """

    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]

    fps: Annotated[float, Field(gt=0)] | None = None
    """``None`` when the container does not declare a frame rate. Never 0."""

    frame_count: Annotated[int, Field(gt=0)] | None = None
    """``None`` when not declared. Some containers report an estimate; it is
    reported as given rather than being verified by decoding the whole file."""

    duration_s: Annotated[float, Field(gt=0)] | None = None
    """Computed only when both ``fps`` and ``frame_count`` are known. Left
    ``None`` otherwise rather than guessed from file size."""

    codec_fourcc: Annotated[str, Field(max_length=8)] | None = None

    @property
    def aspect_ratio(self) -> float:
        """Width divided by height."""
        return self.width / self.height


class MediaAsset(DomainModel):
    """An ingested file plus everything measured about it."""

    media_id: MediaId
    """Server-generated identifier. Also the on-disk filename stem, which is why
    no client-supplied string ever reaches the filesystem."""

    kind: MediaKind
    format: Annotated[str, Field(min_length=1, max_length=20)]
    media_type: Annotated[str, Field(min_length=1, max_length=100)]

    original_filename: Annotated[str, Field(max_length=255)]
    """The client's filename, retained for display only. Never used to build a
    path, and stripped of directory components before storage."""

    size_bytes: Annotated[int, Field(ge=0)]
    ingested_at: dt.datetime

    image: ImageMetadata | None = None
    video: VideoMetadata | None = None

    @property
    def source_id(self) -> str:
        """Identifier this asset presents as when opened as a data source."""
        return f"{self.kind.value}:{self.media_id}"
