"""Media ingestion API: images, videos, and video playback control.

Phase 2 scope: **raw input only**. These endpoints accept a file, measure it,
store it, and hand back what was measured. Nothing here detects, classifies,
counts or predicts anything, and no endpoint returns a traffic quantity.

Security posture
----------------
* Uploads are streamed to disk with a running size check, so an oversized file
  is refused partway through rather than after being buffered whole in memory.
* Format is decided by sniffing content, never by the client's filename or
  ``Content-Type``.
* The stored filename is a server-generated UUID. No client string ever becomes
  a path component.
* Library selection resolves against the media root and rejects anything that
  escapes it, checked after resolution so ``..``, symlinks, absolute paths and
  UNC forms are all covered.

Location is optional, and separate
----------------------------------
An upload may carry the coordinates of the junction it was filmed at. When it
does, an :class:`~app.domain.site.ObservationSite` is recorded in the site
registry -- a different store, with a different lifetime. ``MediaAsset`` gains
no geographic field: the file and the place are not the same thing, one junction
outlives many uploads of it, and deleting footage must not delete a survey. An
upload with no coordinates behaves exactly as it did before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, Query, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import Field
from starlette.requests import Request

from app.api.deps import MediaStoreDep, SettingsDep, SiteRegistryDep
from app.api.routes.sites import SiteRequest, record_site
from app.api.schemas import ApiModel, ErrorResponse
from app.core.config import Settings
from app.core.errors import DataSourceError, NotAvailableYetError, ValidationFailedError
from app.core.logging import get_logger
from app.core.phases import phase
from app.domain.site import MAX_ACCURACY_M, observation_area_id_for_media
from app.ingestion.assets import MediaAsset
from app.ingestion.media import (
    SNIFF_LENGTH,
    MediaKind,
    formats_for_kind,
    validate_upload,
)
from app.ingestion.playback import playback_manager
from app.ingestion.probe import probe_image, probe_video
from app.ingestion.storage import MediaStore, sanitize_filename, utc_now
from app.sites.registry import ObservationSiteRegistry

logger = get_logger(__name__)
router = APIRouter(prefix="/media", tags=["media"])

_CHUNK_BYTES = 1024 * 1024

_NOT_IMPLEMENTED_RESPONSES: dict[int | str, dict[str, object]] = {
    501: {"model": ErrorResponse, "description": "Capability belongs to a later phase"}
}

# ---------------------------------------------------------------------------
# Optional upload location
# ---------------------------------------------------------------------------
# An upload may carry where the camera was. It is optional in the strict sense:
# omitting it changes nothing about ingestion, and supplying it creates or
# corrects an ObservationSite in a registry that has nothing to do with this
# store. No geographic field is added to MediaAsset -- an asset describes a
# file, a site describes a place, and one junction outlives many files of it.
#
# The bounds are declared on the form fields, so an out-of-range coordinate is
# refused by request validation before a single byte reaches the media
# directory, rather than after a video has been written and probed.
LatitudeForm = Annotated[
    float | None,
    Form(ge=-90.0, le=90.0, description="Latitude of the junction, WGS-84 degrees."),
]
LongitudeForm = Annotated[
    float | None,
    Form(ge=-180.0, le=180.0, description="Longitude of the junction, WGS-84 degrees."),
]
BearingForm = Annotated[
    float | None,
    Form(ge=0.0, lt=360.0, description="Camera bearing: 0 = true north, clockwise."),
]
AccuracyForm = Annotated[
    float | None,
    Form(gt=0.0, le=MAX_ACCURACY_M, description="Position accuracy radius, metres."),
]
SiteLabelForm = Annotated[
    str | None,
    Form(min_length=1, max_length=200, description="Human name for the junction."),
]

#: Recorded as the provenance producer for a site created during an upload, so
#: a location that arrived alongside a file is distinguishable from one a
#: surveyor posted deliberately to /api/sites.
_UPLOAD_PRODUCER = "operator:media-upload"


# ---------------------------------------------------------------------------
# Store wiring
# ---------------------------------------------------------------------------
def reset_store() -> None:
    """Close all playback sessions. Tests only.

    The media store itself lives on ``app.state`` and is discarded with the
    application, so there is no singleton left to clear.
    """
    playback_manager.close_all()


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------
class SupportedFormat(ApiModel):
    """One format this service accepts."""

    name: str
    kind: MediaKind
    extensions: list[str]
    media_type: str


class FormatsResponse(ApiModel):
    """Everything the ingestion API will accept.

    Served so the frontend's file picker is configured by the backend rather
    than by a duplicated list that can drift out of step with what is actually
    validated.
    """

    formats: list[SupportedFormat]
    max_image_upload_bytes: int
    max_video_upload_bytes: int


class MediaListResponse(ApiModel):
    """Assets currently held by the store."""

    assets: list[MediaAsset]
    count: int


class LibraryEntry(ApiModel):
    """A file already present under the media directory, available to select."""

    relative_path: str
    size_bytes: int
    suggested_kind: MediaKind


class LibraryResponse(ApiModel):
    """Selectable files found under the media directory."""

    entries: list[LibraryEntry]
    count: int
    media_dir_exists: bool


class SelectRequest(ApiModel):
    """Request to ingest a file already present under the media directory."""

    relative_path: Annotated[str, Field(min_length=1, max_length=1024)]
    expected_kind: MediaKind | None = None


class PlaybackState(ApiModel):
    """Current replay position of an opened video source.

    Reports position only. No frame pixels are returned: serving decoded frames
    over JSON would be wasteful, and the Phase 3 analysis pipeline decodes the
    stored file itself rather than consuming frames through this endpoint.
    """

    media_id: str
    status: str
    is_open: bool
    current_frame_index: int
    position_s: float
    frames_read: int
    exhausted: bool
    fps: float | None
    frame_count: int | None
    duration_s: float | None
    last_read_succeeded: bool | None = None


class SeekRequest(ApiModel):
    """Seek target, in seconds or by frame index."""

    position_s: Annotated[float, Field(ge=0)] | None = None
    frame_index: Annotated[int, Field(ge=0)] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _store_upload(
    upload: UploadFile,
    kind: MediaKind,
    settings: Settings,
    store: MediaStore,
) -> MediaAsset:
    """Validate and persist an upload, returning the measured asset."""
    filename = sanitize_filename(upload.filename or "unnamed")
    limit = (
        settings.max_image_upload_bytes
        if kind is MediaKind.IMAGE
        else settings.max_video_upload_bytes
    )

    header = await upload.read(SNIFF_LENGTH)
    if not header:
        msg = "The uploaded file is empty."
        raise ValidationFailedError(msg, filename=filename)

    # Content decides the format. The filename is only cross-checked against it.
    media_format = validate_upload(filename=filename, header=header, expected_kind=kind)

    media_id, destination = store.allocate(media_format)
    total = len(header)
    try:
        with destination.open("wb") as handle:
            handle.write(header)
            while True:
                chunk = await upload.read(_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    msg = (
                        f"File exceeds the {limit // (1024 * 1024)} MiB limit for "
                        f"{kind.value} uploads."
                    )
                    raise ValidationFailedError(msg, limit_bytes=limit, filename=filename)
                handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    try:
        if kind is MediaKind.IMAGE:
            image_meta = probe_image(destination)
            video_meta = None
        else:
            image_meta = None
            video_meta = probe_video(destination)
    except DataSourceError:
        # A file that sniffed correctly but will not decode is corrupt. Do not
        # keep it: a stored asset that cannot be opened is worse than none.
        destination.unlink(missing_ok=True)
        raise

    asset = MediaAsset(
        media_id=media_id,
        kind=kind,
        format=media_format.name,
        media_type=media_format.media_type,
        original_filename=filename,
        size_bytes=total,
        ingested_at=utc_now(),
        image=image_meta,
        video=video_meta,
    )
    return store.register(asset)


def _check_location_pairing(latitude: float | None, longitude: float | None) -> bool:
    """Decide whether a location was supplied, rejecting half of one.

    A latitude without a longitude is not a weak location; it is not a location
    at all. Silently ignoring the half that arrived would record nothing while
    the caller believed the junction had been placed, which is the failure this
    whole feature exists to avoid.
    """
    if latitude is None and longitude is None:
        return False
    if latitude is None or longitude is None:
        missing = "longitude" if longitude is None else "latitude"
        msg = (
            f"A location needs both latitude and longitude; {missing} is missing. "
            "Half a coordinate cannot place a junction, and recording it would "
            "put the camera somewhere it is not."
        )
        raise ValidationFailedError(msg, missing=missing)
    return True


async def _ingest(
    upload: UploadFile,
    kind: MediaKind,
    settings: Settings,
    store: MediaStore,
    sites: ObservationSiteRegistry,
    *,
    latitude: float | None,
    longitude: float | None,
    bearing_degrees: float | None,
    accuracy_m: float | None,
    site_label: str | None,
) -> MediaAsset:
    """Store an upload and, when a location came with it, record the site.

    Ordering is deliberate. The location is checked first, so a malformed one
    costs nothing and leaves nothing behind. The site is recorded last, and if
    that write fails the stored asset is removed: the caller asked for a located
    upload, and an upload whose location was silently dropped is exactly the
    half-applied result that makes a dashboard confidently wrong.
    """
    has_location = _check_location_pairing(latitude, longitude)

    asset = await _store_upload(upload, kind, settings, store)
    if not has_location:
        return asset

    # Narrowing for the type checker; _check_location_pairing has already
    # established that both are present.
    assert latitude is not None and longitude is not None  # noqa: S101

    site_request = SiteRequest(
        observation_area_id=observation_area_id_for_media(asset.media_id),
        latitude=latitude,
        longitude=longitude,
        bearing_degrees=bearing_degrees,
        accuracy_m=accuracy_m,
        label=site_label,
        recorded_by=_UPLOAD_PRODUCER,
    )
    try:
        site, replaced = record_site(sites, site_request, now=utc_now())
    except Exception:
        playback_manager.close(asset.media_id)
        store.delete(asset.media_id)
        raise

    logger.info(
        "media ingested with a location",
        extra={
            "media_id": asset.media_id,
            "observation_area_id": site.observation_area_id,
            "replaced_existing_site": replaced,
        },
    )
    return asset


def _require_kind(asset: MediaAsset, kind: MediaKind) -> None:
    """Reject an asset of the wrong kind for the endpoint being used."""
    if asset.kind is not kind:
        msg = f"Media {asset.media_id!r} is a {asset.kind.value}, not a {kind.value}."
        raise ValidationFailedError(
            msg, media_id=asset.media_id, actual_kind=asset.kind.value
        )


def _playback_state(
    store: MediaStore, media_id: str, last_read: bool | None = None
) -> PlaybackState:
    """Build the playback state for a video, opened or not."""
    asset = store.get(media_id)
    _require_kind(asset, MediaKind.VIDEO)
    video = asset.video

    source = playback_manager.get(media_id)
    if source is None:
        return PlaybackState(
            media_id=media_id,
            status="closed",
            is_open=False,
            current_frame_index=0,
            position_s=0.0,
            frames_read=0,
            exhausted=False,
            fps=video.fps if video else None,
            frame_count=video.frame_count if video else None,
            duration_s=video.duration_s if video else None,
            last_read_succeeded=last_read,
        )

    health = source.health()
    return PlaybackState(
        media_id=media_id,
        status=source.status.value,
        is_open=True,
        current_frame_index=source.current_frame_index,
        position_s=source.position_s,
        frames_read=health.samples_read,
        exhausted=source.status.value == "exhausted",
        fps=video.fps if video else None,
        frame_count=video.frame_count if video else None,
        duration_s=video.duration_s if video else None,
        last_read_succeeded=last_read,
    )


def _ranged_file_response(request: Request, path: Path, media_type: str) -> Response:
    """Serve a file, honouring a single HTTP Range request.

    Implemented explicitly because a browser ``<video>`` element needs 206
    responses to seek. Without range support the element can usually still play
    a small clip from the start but cannot scrub, which would make the replay
    controls appear broken for reasons unrelated to the backend.
    """
    file_size = path.stat().st_size
    range_header = request.headers.get("range")

    if range_header is None or not range_header.startswith("bytes="):
        return FileResponse(
            path,
            media_type=media_type,
            headers={"Accept-Ranges": "bytes", "Cache-Control": "no-store"},
        )

    spec = range_header.removeprefix("bytes=").split(",")[0].strip()
    start_text, _, end_text = spec.partition("-")
    try:
        start = int(start_text) if start_text else 0
        end = int(end_text) if end_text else file_size - 1
    except ValueError:
        start, end = 0, file_size - 1

    start = max(0, start)
    end = min(end, file_size - 1)
    if start > end:
        return Response(
            status_code=416,
            headers={"Content-Range": f"bytes */{file_size}"},
        )

    with path.open("rb") as handle:
        handle.seek(start)
        payload = handle.read(end - start + 1)

    return Response(
        content=payload,
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(len(payload)),
            "Cache-Control": "no-store",
        },
    )


# ---------------------------------------------------------------------------
# Catalogue endpoints
# ---------------------------------------------------------------------------
@router.get("/formats", response_model=FormatsResponse, summary="Accepted media formats")
def formats(settings: SettingsDep) -> FormatsResponse:
    """Return the formats and size limits this service enforces."""
    entries = [
        SupportedFormat(
            name=fmt.name,
            kind=fmt.kind,
            extensions=list(fmt.extensions),
            media_type=fmt.media_type,
        )
        for kind in (MediaKind.IMAGE, MediaKind.VIDEO)
        for fmt in formats_for_kind(kind)
    ]
    return FormatsResponse(
        formats=entries,
        max_image_upload_bytes=settings.max_image_upload_bytes,
        max_video_upload_bytes=settings.max_video_upload_bytes,
    )


@router.get("", response_model=MediaListResponse, summary="Ingested media assets")
def list_media(
    store: MediaStoreDep,
    kind: Annotated[MediaKind | None, Query(description="Filter by media kind")] = None,
) -> MediaListResponse:
    """List ingested assets. Genuinely empty until something is ingested."""
    assets = list(store.list(kind))
    return MediaListResponse(assets=assets, count=len(assets))


@router.get("/library", response_model=LibraryResponse, summary="Selectable local files")
def library(store: MediaStoreDep) -> LibraryResponse:
    """List supported files already present under the media directory."""
    allowed = frozenset(
        ext
        for kind in (MediaKind.IMAGE, MediaKind.VIDEO)
        for fmt in formats_for_kind(kind)
        for ext in fmt.extensions
    )
    entries: list[LibraryEntry] = []
    for path in store.library_files(allowed):
        suffix = path.suffix.lower()
        kind = (
            MediaKind.IMAGE
            if any(suffix in f.extensions for f in formats_for_kind(MediaKind.IMAGE))
            else MediaKind.VIDEO
        )
        entries.append(
            LibraryEntry(
                relative_path=path.relative_to(store.root).as_posix(),
                size_bytes=path.stat().st_size,
                suggested_kind=kind,
            )
        )
    return LibraryResponse(
        entries=entries, count=len(entries), media_dir_exists=store.root.exists()
    )


# ---------------------------------------------------------------------------
# Ingestion endpoints
# ---------------------------------------------------------------------------
@router.post("/images", response_model=MediaAsset, status_code=201, summary="Upload an image")
async def upload_image(
    file: Annotated[UploadFile, File()],
    settings: SettingsDep,
    store: MediaStoreDep,
    sites: SiteRegistryDep,
    latitude: LatitudeForm = None,
    longitude: LongitudeForm = None,
    bearing_degrees: BearingForm = None,
    accuracy_m: AccuracyForm = None,
    site_label: SiteLabelForm = None,
) -> MediaAsset:
    """Ingest a traffic still. Raw input: the image is measured, not interpreted.

    Supplying ``latitude`` and ``longitude`` also records where the junction is,
    as an :class:`~app.domain.site.ObservationSite` keyed by
    ``observation-area:<media_id>`` and retrievable from ``GET /api/sites``. The
    returned asset is unchanged either way: a media asset describes the file,
    never the place.
    """
    return await _ingest(
        file,
        MediaKind.IMAGE,
        settings,
        store,
        sites,
        latitude=latitude,
        longitude=longitude,
        bearing_degrees=bearing_degrees,
        accuracy_m=accuracy_m,
        site_label=site_label,
    )


@router.post("/videos", response_model=MediaAsset, status_code=201, summary="Upload a video")
async def upload_video(
    file: Annotated[UploadFile, File()],
    settings: SettingsDep,
    store: MediaStoreDep,
    sites: SiteRegistryDep,
    latitude: LatitudeForm = None,
    longitude: LongitudeForm = None,
    bearing_degrees: BearingForm = None,
    accuracy_m: AccuracyForm = None,
    site_label: SiteLabelForm = None,
) -> MediaAsset:
    """Ingest a recorded video. Raw input: the container is measured, not analysed.

    Takes the same optional location as the image endpoint, with identical
    meaning: it places the camera, not the file.
    """
    return await _ingest(
        file,
        MediaKind.VIDEO,
        settings,
        store,
        sites,
        latitude=latitude,
        longitude=longitude,
        bearing_degrees=bearing_degrees,
        accuracy_m=accuracy_m,
        site_label=site_label,
    )


@router.post("/select", response_model=MediaAsset, status_code=201, summary="Select a local file")
def select_media(request: SelectRequest, store: MediaStoreDep) -> MediaAsset:
    """Ingest a file already present under the media directory.

    The path is resolved against the media root and rejected if it escapes,
    which is checked on the *resolved* path so ``..`` segments, symlinks and
    absolute paths are all covered.
    """
    relative = request.relative_path

    suffix = Path(relative).suffix.lower()
    from app.ingestion.media import format_for_extension

    declared = format_for_extension(suffix)
    if declared is None:
        msg = f"Unsupported file extension {suffix!r}."
        raise ValidationFailedError(msg, relative_path=relative)
    if request.expected_kind is not None and declared.kind is not request.expected_kind:
        msg = (
            f"Expected a {request.expected_kind.value} but {relative!r} is a "
            f"{declared.kind.value}."
        )
        raise ValidationFailedError(msg, relative_path=relative)

    media_id, destination = store.import_from_library(relative, declared)

    try:
        with destination.open("rb") as handle:
            header = handle.read(SNIFF_LENGTH)
        # The copied file is sniffed too: a `.png` containing something else is
        # rejected here rather than handed to a decoder.
        media_format = validate_upload(
            filename=Path(relative).name, header=header, expected_kind=declared.kind
        )
        if media_format.kind is MediaKind.IMAGE:
            image_meta = probe_image(destination)
            video_meta = None
        else:
            image_meta = None
            video_meta = probe_video(destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    asset = MediaAsset(
        media_id=media_id,
        kind=media_format.kind,
        format=media_format.name,
        media_type=media_format.media_type,
        original_filename=sanitize_filename(Path(relative).name),
        size_bytes=destination.stat().st_size,
        ingested_at=utc_now(),
        image=image_meta,
        video=video_meta,
    )
    return store.register(asset)


# ---------------------------------------------------------------------------
# Asset endpoints
# ---------------------------------------------------------------------------
@router.get("/{media_id}", response_model=MediaAsset, summary="Media metadata")
def get_media(media_id: str, store: MediaStoreDep) -> MediaAsset:
    """Return everything measured about one asset."""
    return store.get(media_id)


@router.get("/{media_id}/content", summary="Raw media bytes")
def get_content(media_id: str, request: Request, store: MediaStoreDep) -> Response:
    """Serve the stored file, honouring Range requests for video scrubbing."""
    asset = store.get(media_id)
    path = store.path_for(media_id)
    return _ranged_file_response(request, path, asset.media_type)


@router.delete("/{media_id}", status_code=204, summary="Delete a media asset")
def delete_media(media_id: str, store: MediaStoreDep) -> Response:
    """Remove an asset and its file, closing any open playback session first."""
    store.get(media_id)
    playback_manager.close(media_id)
    store.delete(media_id)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Video playback control
# ---------------------------------------------------------------------------
@router.get(
    "/videos/{media_id}/playback",
    response_model=PlaybackState,
    summary="Video replay state",
)
def playback_state(media_id: str, store: MediaStoreDep) -> PlaybackState:
    """Return the current replay position, whether or not a session is open."""
    return _playback_state(store, media_id)


@router.post(
    "/videos/{media_id}/playback/{action}",
    response_model=PlaybackState,
    summary="Control video replay",
)
def playback_control(
    media_id: str,
    action: Literal["open", "read", "seek", "reset", "close"],
    store: MediaStoreDep,
    request: SeekRequest | None = None,
) -> PlaybackState:
    """Drive a recorded video source: open, read, seek, reset, close.

    These are exactly the operations the recorded-source contract defines. The
    read action advances by one frame and reports the new position; the decoded
    frame is deliberately not returned, because the analysis pipeline decodes
    the stored file directly.
    """
    asset = store.get(media_id)
    _require_kind(asset, MediaKind.VIDEO)
    path = store.path_for(media_id)

    if action == "open":
        playback_manager.open(media_id, path, asset.video)
        return _playback_state(store, media_id)

    if action == "close":
        playback_manager.close(media_id)
        return _playback_state(store, media_id)

    source = playback_manager.get(media_id)
    if source is None:
        source = playback_manager.open(media_id, path, asset.video)

    if action == "read":
        sample = source.read()
        return _playback_state(store, media_id, last_read=sample is not None)

    if action == "reset":
        source.reset()
        return _playback_state(store, media_id)

    # action == "seek"
    if request is None or (request.position_s is None and request.frame_index is None):
        msg = "A seek requires either position_s or frame_index."
        raise ValidationFailedError(msg, media_id=media_id)
    if request.frame_index is not None:
        source.seek_to_frame(request.frame_index)
    elif request.position_s is not None:
        source.seek(request.position_s)
    return _playback_state(store, media_id)


# ---------------------------------------------------------------------------
# Live ingestion — architecture preserved, deliberately not implemented
# ---------------------------------------------------------------------------
@router.post(
    "/live/connect",
    responses=_NOT_IMPLEMENTED_RESPONSES,
    summary="Connect a live traffic source (phase 14)",
)
def connect_live() -> None:
    """Reject live connection attempts, naming the phase that will provide it.

    The interface and capability profile for live sources are defined and
    unchanged; only the implementation is absent. Nothing here opens a socket or
    simulates a connection, because a fake live source would be indistinguishable
    from a real one that is producing nothing.
    """
    target = phase(14)
    msg = (
        "Live traffic ingestion is not implemented. The LiveTrafficDataSource "
        "interface, its capability profile and its backpressure policy are "
        "defined, but no stream is connected and none is simulated. Recorded "
        "images and videos are the supported inputs in this phase."
    )
    raise NotAvailableYetError(msg, phase=target.number, phase_name=target.name)
