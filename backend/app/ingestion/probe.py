"""Metadata extraction for images and videos.

Both probes measure the *container*. Nothing here interprets content: no
detection, no classification, no counting. An image is width, height, format and
colour mode; a video is dimensions, frame rate and length. What is in the
picture is a later phase's problem.

Decoder libraries are imported lazily so that a deployment ingesting only
tabular data does not need Pillow or OpenCV installed, and so a missing decoder
produces a clear error at the point of use rather than an import failure at
startup.

**On OpenCV.** It is used strictly as a container/codec reader -- open, read a
frame, seek, query properties. No vision algorithm is invoked. That distinction
is deliberate: Phase 2 is raw ingestion, and the decoding of a video file is not
computer vision any more than parsing a CSV is statistics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.errors import DataSourceError
from app.ingestion.assets import ImageMetadata, VideoMetadata


def _require_pillow() -> Any:
    """Import Pillow, or raise a clear error explaining what is missing."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised only without Pillow
        msg = (
            "Image ingestion requires Pillow, which is not installed. "
            "Install it with: pip install pillow"
        )
        raise DataSourceError(msg, missing_dependency="pillow") from exc
    return Image


def _require_cv2() -> Any:
    """Import OpenCV, or raise a clear error explaining what is missing."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised only without OpenCV
        msg = (
            "Video ingestion requires OpenCV, which is not installed. "
            "Install it with: pip install opencv-python-headless"
        )
        raise DataSourceError(msg, missing_dependency="opencv-python-headless") from exc
    return cv2


def _positive_or_none(value: float | None) -> float | None:
    """Map a non-positive or non-finite reading onto ``None``.

    Decoders report ``0`` or ``-1`` for properties a container does not declare.
    Passing those through would produce a video claiming zero frames per second,
    and any duration derived from it would be zero or a division error. Unknown
    must be ``None``, never a number.
    """
    if value is None:
        return None
    if value != value:  # NaN
        return None
    if value <= 0 or value == float("inf"):
        return None
    return value


def probe_image(path: Path) -> ImageMetadata:
    """Measure a still image.

    Raises :class:`DataSourceError` for a missing, unreadable or corrupt file.
    """
    if not path.exists():
        msg = f"Image file not found: {path.name}"
        raise DataSourceError(msg, path=path.name)

    image_module = _require_pillow()

    try:
        with image_module.open(path) as img:
            width, height = img.size
            fmt = (img.format or "unknown").lower()
            mode = img.mode or None
            is_animated = bool(getattr(img, "is_animated", False))
            raw_frames = getattr(img, "n_frames", None)
            frame_count = int(raw_frames) if is_animated and raw_frames else None

            channels: int | None = None
            if mode:
                try:
                    # getmodebands returns the band count directly, not a sequence.
                    channels = int(image_module.getmodebands(mode))
                except (KeyError, ValueError, TypeError):
                    channels = None

            # Force a full decode so truncated files fail here rather than
            # later. Reading the header alone would report metadata for an
            # image that cannot actually be displayed.
            img.load()
    except DataSourceError:
        raise
    except Exception as exc:
        msg = (
            f"Could not decode image {path.name}: the file appears to be corrupt "
            f"or is not a valid image ({type(exc).__name__})."
        )
        raise DataSourceError(msg, path=path.name, reason=type(exc).__name__) from exc

    if width <= 0 or height <= 0:
        msg = f"Image {path.name} reports non-positive dimensions ({width}x{height})."
        raise DataSourceError(msg, path=path.name)

    return ImageMetadata(
        width=width,
        height=height,
        format=fmt,
        mode=mode,
        channels=channels,
        is_animated=is_animated,
        frame_count=frame_count,
    )


def probe_video(path: Path) -> VideoMetadata:
    """Measure a video container.

    Raises :class:`DataSourceError` for a missing, unreadable or corrupt file.
    """
    if not path.exists():
        msg = f"Video file not found: {path.name}"
        raise DataSourceError(msg, path=path.name)

    cv2 = _require_cv2()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            msg = (
                f"Could not open video {path.name}. The file may be corrupt, "
                "truncated, or use a codec this build cannot decode."
            )
            raise DataSourceError(msg, path=path.name)

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = _positive_or_none(capture.get(cv2.CAP_PROP_FPS))
        raw_count = _positive_or_none(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = int(raw_count) if raw_count is not None else None
        fourcc_raw = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)

        # A container can open yet declare no usable dimensions; that is a
        # corrupt file rather than a video with unknown size.
        if width <= 0 or height <= 0:
            msg = (
                f"Video {path.name} opened but declares no usable frame size "
                f"({width}x{height}); treating it as corrupt."
            )
            raise DataSourceError(msg, path=path.name)
    finally:
        capture.release()

    codec = None
    if fourcc_raw > 0:
        chars = [chr((fourcc_raw >> (8 * i)) & 0xFF) for i in range(4)]
        candidate = "".join(chars).strip()
        codec = candidate if candidate.isprintable() and candidate else None

    # Duration only when both inputs are genuinely known. Never estimated from
    # file size, and never zero-filled.
    duration_s: float | None = None
    if fps is not None and frame_count is not None:
        duration_s = frame_count / fps

    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_s=duration_s,
        codec_fourcc=codec,
    )
