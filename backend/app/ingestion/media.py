"""Media format identification and validation.

Formats are identified by **content**, not by filename extension or by the
``Content-Type`` header the client sent. Both of those are client-controlled and
therefore untrustworthy: a file named ``traffic.png`` that is actually an
executable is trivial to produce, and a decoder handed unexpected bytes is
exactly where parser vulnerabilities live.

So every upload is sniffed against a magic-number table before anything else
touches it, and the declared extension must agree with what the bytes actually
are. A mismatch is rejected rather than silently corrected -- a file whose name
disagrees with its content is either a mistake worth surfacing or an attempt
worth refusing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from app.core.errors import ValidationFailedError


class MediaKind(StrEnum):
    """Whether a media asset is a still image or a video."""

    IMAGE = "image"
    VIDEO = "video"


@dataclass(frozen=True, slots=True)
class MediaFormat:
    """One supported container/codec family."""

    name: str
    kind: MediaKind
    extensions: tuple[str, ...]
    media_type: str
    """MIME type this service will serve the bytes back as. Ours, not theirs."""


JPEG = MediaFormat("jpeg", MediaKind.IMAGE, (".jpg", ".jpeg"), "image/jpeg")
PNG = MediaFormat("png", MediaKind.IMAGE, (".png",), "image/png")
WEBP = MediaFormat("webp", MediaKind.IMAGE, (".webp",), "image/webp")

MP4 = MediaFormat("mp4", MediaKind.VIDEO, (".mp4",), "video/mp4")
MOV = MediaFormat("mov", MediaKind.VIDEO, (".mov",), "video/quicktime")
AVI = MediaFormat("avi", MediaKind.VIDEO, (".avi",), "video/x-msvideo")

SUPPORTED_FORMATS: Final[tuple[MediaFormat, ...]] = (JPEG, PNG, WEBP, MP4, MOV, AVI)

IMAGE_FORMATS: Final[tuple[MediaFormat, ...]] = tuple(
    f for f in SUPPORTED_FORMATS if f.kind is MediaKind.IMAGE
)
VIDEO_FORMATS: Final[tuple[MediaFormat, ...]] = tuple(
    f for f in SUPPORTED_FORMATS if f.kind is MediaKind.VIDEO
)

#: Bytes needed to identify every supported format. The ISO base-media brand
#: sits at offset 8, so 16 is comfortably sufficient.
SNIFF_LENGTH: Final[int] = 16

#: ``ftyp`` brands that indicate QuickTime rather than MP4. Both are ISO
#: base-media files and share the same box structure, so the brand is the only
#: thing that separates them.
_QUICKTIME_BRANDS: Final[frozenset[bytes]] = frozenset({b"qt  "})


def sniff_format(header: bytes) -> MediaFormat | None:
    """Identify a format from a file's leading bytes.

    Returns ``None`` for anything unrecognised. Callers must treat ``None`` as a
    rejection: an unidentified file is never passed to a decoder on the hope
    that it will cope.
    """
    if len(header) < 4:
        return None

    if header[:3] == b"\xff\xd8\xff":
        return JPEG
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        return PNG

    if header[:4] == b"RIFF" and len(header) >= 12:
        riff_type = header[8:12]
        if riff_type == b"WEBP":
            return WEBP
        if riff_type == b"AVI ":
            return AVI
        return None

    # ISO base media (MP4 / MOV): a `ftyp` box at offset 4, brand at offset 8.
    if len(header) >= 12 and header[4:8] == b"ftyp":
        brand = header[8:12]
        return MOV if brand in _QUICKTIME_BRANDS else MP4

    # QuickTime files occasionally lead with `moov` or `mdat` instead of `ftyp`.
    if len(header) >= 8 and header[4:8] in (b"moov", b"mdat"):
        return MOV

    return None


def format_for_extension(extension: str) -> MediaFormat | None:
    """Look up a format by file extension (case-insensitive)."""
    lowered = extension.lower()
    for fmt in SUPPORTED_FORMATS:
        if lowered in fmt.extensions:
            return fmt
    return None


def validate_upload(
    *,
    filename: str,
    header: bytes,
    expected_kind: MediaKind | None = None,
) -> MediaFormat:
    """Identify and validate an upload, or raise :class:`ValidationFailedError`.

    Three checks, in order of how much they can be trusted:

    1. the bytes are a format we support (content, trustworthy);
    2. the extension agrees with the bytes (cross-check, catches mislabelling);
    3. the format is the kind the endpoint expects (an image is not a video).
    """
    detected = sniff_format(header)
    if detected is None:
        supported = ", ".join(sorted({f.name for f in SUPPORTED_FORMATS}))
        msg = (
            "Unrecognised file content. The uploaded bytes do not match any "
            f"supported format ({supported}). Files are identified by content, "
            "not by filename, so renaming a file will not make it acceptable."
        )
        raise ValidationFailedError(msg, detected_format=None, filename=filename)

    suffix = ""
    if "." in filename:
        suffix = filename[filename.rfind(".") :].lower()
    if suffix and suffix not in detected.extensions:
        declared = format_for_extension(suffix)
        msg = (
            f"File extension {suffix!r} does not match the actual content, which "
            f"is {detected.name}. A file whose name disagrees with its bytes is "
            "rejected rather than silently accepted."
        )
        raise ValidationFailedError(
            msg,
            declared_format=declared.name if declared else None,
            detected_format=detected.name,
            filename=filename,
        )

    if expected_kind is not None and detected.kind is not expected_kind:
        msg = (
            f"Expected {expected_kind.value} content but the file is "
            f"{detected.name}, which is a {detected.kind.value}."
        )
        raise ValidationFailedError(
            msg,
            detected_format=detected.name,
            detected_kind=detected.kind.value,
            expected_kind=expected_kind.value,
        )

    return detected


def formats_for_kind(kind: MediaKind) -> tuple[MediaFormat, ...]:
    """Supported formats for one media kind."""
    return IMAGE_FORMATS if kind is MediaKind.IMAGE else VIDEO_FORMATS
