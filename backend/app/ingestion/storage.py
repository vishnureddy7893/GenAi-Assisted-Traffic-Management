"""Media storage, and the containment rules that keep it safe.

Threat model
------------
Two client-controlled strings could otherwise reach the filesystem: the upload
filename, and the relative path used to select an already-present file. Both are
handled so that traversal is structurally impossible rather than merely
validated against:

* **Uploads never use the client filename.** The stored name is a
  server-generated UUID plus the extension implied by the *sniffed* format. The
  original name is kept as display metadata only. There is no code path in which
  client text becomes a path component.

* **Selection is confined by resolution, not by string inspection.**
  :func:`resolve_within` resolves the candidate against the root and then checks
  containment on the resolved result. Blacklisting ``..`` is not enough --
  ``....//``, URL-encoded variants, absolute paths, Windows drive letters, UNC
  paths and symlinks all defeat string matching. Resolving first and comparing
  after defeats all of them at once, including the symlink case, because
  ``Path.resolve`` follows links before the comparison happens.
"""

from __future__ import annotations

import datetime as dt
import re
import shutil
import uuid
from pathlib import Path, PurePath, PureWindowsPath

from app.core.errors import DataSourceError, ResourceNotFoundError, ValidationFailedError
from app.core.logging import get_logger
from app.ingestion.assets import MediaAsset
from app.ingestion.media import MediaFormat, MediaKind

logger = get_logger(__name__)

#: Characters that must never appear in a path we are asked to resolve.
_FORBIDDEN_IN_PATH = ("\x00",)


def sanitize_filename(filename: str) -> str:
    """Reduce a client filename to a display-safe leaf name.

    Strips directory components using both POSIX and Windows separator rules,
    because an upload may originate from either and a server parsing only its
    own convention will happily keep ``..\\..\\evil`` intact as a "leaf".

    The result is used for display and never as a path component, so this is
    defence in depth rather than the primary control.
    """
    if not filename:
        return "unnamed"
    cleaned = filename.replace("\x00", "")
    # Take the leaf under both conventions; whichever is shorter has stripped more.
    leaf = PurePath(cleaned).name
    windows_leaf = PureWindowsPath(cleaned).name
    if len(windows_leaf) < len(leaf):
        leaf = windows_leaf
    leaf = leaf.strip().strip(".")
    return leaf[:255] or "unnamed"


def resolve_within(root: Path, candidate: str) -> Path:
    """Resolve ``candidate`` beneath ``root``, or raise.

    Raises :class:`ValidationFailedError` if the candidate is absolute, names a
    drive or UNC share, contains a NUL byte, or resolves anywhere outside
    ``root``.
    """
    if not candidate or not candidate.strip():
        msg = "A path must be provided."
        raise ValidationFailedError(msg)

    for forbidden in _FORBIDDEN_IN_PATH:
        if forbidden in candidate:
            msg = "Path contains an illegal character."
            raise ValidationFailedError(msg)

    # Reject anything that is absolute under either convention, plus Windows
    # drive-relative ("C:foo") and UNC ("\\\\host\\share") forms, before we let
    # pathlib join it -- joining an absolute path silently discards the root.
    pure_posix = PurePath(candidate)
    pure_windows = PureWindowsPath(candidate)
    if (
        pure_posix.is_absolute()
        or pure_windows.is_absolute()
        or pure_windows.drive != ""
        or candidate.startswith(("/", "\\"))
    ):
        msg = "Absolute paths are not accepted; provide a path relative to the media directory."
        raise ValidationFailedError(msg, candidate=candidate)

    resolved_root = root.resolve()
    try:
        target = (resolved_root / candidate).resolve()
    except (OSError, RuntimeError) as exc:  # pragma: no cover - platform dependent
        msg = "Path could not be resolved."
        raise ValidationFailedError(msg, candidate=candidate) from exc

    # The containment check happens on the *resolved* path, so `..` segments and
    # symlinks have already been collapsed and cannot smuggle anything out.
    if target != resolved_root and resolved_root not in target.parents:
        msg = (
            "Path escapes the media directory. Only files inside the configured "
            "media directory can be opened."
        )
        raise ValidationFailedError(msg, candidate=candidate)

    return target


class MediaStore:
    """Filesystem-backed store for ingested media.

    The in-memory index is rebuilt from disk on construction, so a restart does
    not orphan previously ingested files. Phase 2 keeps metadata in memory
    deliberately: introducing a database before the ingestion contract has
    settled would be committing to a schema we are still learning the shape of.

    What the rebuild can and cannot recover
    ---------------------------------------
    The bytes are the record. A managed asset is stored as ``<media_id><ext>``
    with a server-generated UUID stem, so the id is recoverable from the
    filename and everything measurable is recoverable by re-probing the file:
    format by sniffing content, dimensions, frame rate, duration.

    Two fields are not on disk and are reconstructed honestly rather than
    invented. ``original_filename`` becomes the stored name, because the
    client's display name was never written anywhere; and ``ingested_at``
    becomes the file's modification time, which is genuinely when this process
    wrote it. Both are display metadata. Nothing measurable is guessed.

    Only files whose stem is a managed id are adopted. A file a user dropped
    into the media directory is *source material* and must keep appearing in
    the selectable library; adopting it as a managed asset would make it vanish
    from the picker and claim this system ingested something it did not.
    """

    #: Stem of a managed asset: the hex form of the UUID ``allocate`` generates.
    #: Matched exactly so no hand-placed file is ever mistaken for one.
    _MANAGED_STEM = re.compile(r"^[0-9a-f]{32}$")

    def __init__(self, root: Path, *, rebuild: bool = True) -> None:
        self._root = Path(root)
        self._assets: dict[str, MediaAsset] = {}
        if rebuild:
            self.rebuild_from_disk()

    def rebuild_from_disk(self) -> int:
        """Adopt managed files already under the root, returning how many.

        Idempotent, and never overwrites an asset already in the index: a live
        record carries the client's real filename and ingest time, and replacing
        it with a reconstruction would lose information for no gain.

        A file that will not sniff or will not decode is skipped and logged
        rather than raising. One corrupt file must not stop a process from
        starting, and the file stays on disk for someone to look at.
        """
        if not self._root.exists():
            return 0

        adopted = 0
        for path in sorted(self._root.rglob("*")):
            if not path.is_file():
                continue
            media_id = path.stem
            if not self._MANAGED_STEM.match(media_id) or media_id in self._assets:
                continue
            asset = self._asset_from_file(path, media_id)
            if asset is None:
                continue
            self._assets[media_id] = asset
            adopted += 1

        if adopted:
            logger.info(
                "media index rebuilt from disk",
                extra={"root": str(self._root), "assets": adopted},
            )
        return adopted

    def _asset_from_file(self, path: Path, media_id: str) -> MediaAsset | None:
        """Re-measure one managed file, or return ``None`` if it will not read.

        Format is decided by sniffing content, exactly as it is on upload. The
        extension is not trusted here either: a file renamed on disk must not
        change what this system believes it holds.
        """
        from app.ingestion.media import SNIFF_LENGTH, sniff_format
        from app.ingestion.probe import probe_image, probe_video

        try:
            with path.open("rb") as handle:
                header = handle.read(SNIFF_LENGTH)
            media_format = sniff_format(header)
            if media_format is None:
                logger.warning(
                    "managed media file has unrecognised content; not adopted",
                    extra={"media_id": media_id},
                )
                return None

            if media_format.kind is MediaKind.IMAGE:
                image_meta, video_meta = probe_image(path), None
            else:
                image_meta, video_meta = None, probe_video(path)

            stat = path.stat()
            return MediaAsset(
                media_id=media_id,
                kind=media_format.kind,
                format=media_format.name,
                media_type=media_format.media_type,
                # Not the client's original name: that was never stored. The
                # stored name is what this system actually has.
                original_filename=path.name,
                size_bytes=stat.st_size,
                ingested_at=dt.datetime.fromtimestamp(stat.st_mtime, tz=dt.UTC),
                image=image_meta,
                video=video_meta,
            )
        except (DataSourceError, OSError, ValueError) as exc:
            logger.warning(
                "managed media file could not be re-measured; not adopted",
                extra={"media_id": media_id, "error": str(exc)},
            )
            return None

    @property
    def root(self) -> Path:
        """Directory containing all managed media."""
        return self._root

    def ensure_root(self) -> None:
        """Create the media directory if it does not exist."""
        self._root.mkdir(parents=True, exist_ok=True)

    # --- writing ----------------------------------------------------------
    def allocate(self, media_format: MediaFormat) -> tuple[str, Path]:
        """Reserve a new media id and its on-disk path.

        The path is built entirely from server-controlled values: a UUID stem
        and the extension implied by the sniffed format.
        """
        self.ensure_root()
        media_id = uuid.uuid4().hex
        extension = media_format.extensions[0]
        return media_id, self._root / f"{media_id}{extension}"

    def register(self, asset: MediaAsset) -> MediaAsset:
        """Record an asset in the index."""
        self._assets[asset.media_id] = asset
        logger.info(
            "media asset ingested",
            extra={
                "media_id": asset.media_id,
                "kind": asset.kind.value,
                "format": asset.format,
                "size_bytes": asset.size_bytes,
            },
        )
        return asset

    # --- reading ----------------------------------------------------------
    def get(self, media_id: str) -> MediaAsset:
        """Return an asset by id, or raise :class:`ResourceNotFoundError` (404)."""
        asset = self._assets.get(media_id)
        if asset is None:
            msg = f"No media asset with id {media_id!r}."
            raise ResourceNotFoundError(msg, media_id=media_id)
        return asset

    def has(self, media_id: str) -> bool:
        """Whether an asset with this id is registered."""
        return media_id in self._assets

    def list(self, kind: MediaKind | None = None) -> tuple[MediaAsset, ...]:
        """Registered assets, newest first, optionally filtered by kind."""
        assets = [a for a in self._assets.values() if kind is None or a.kind is kind]
        return tuple(sorted(assets, key=lambda a: a.ingested_at, reverse=True))

    def path_for(self, media_id: str) -> Path:
        """On-disk path of a registered asset.

        Derived from the registered asset rather than from any client string, so
        this cannot be pointed anywhere by a caller.
        """
        asset = self.get(media_id)
        extension = f".{asset.format}" if asset.format != "jpeg" else ".jpg"
        candidate = self._root / f"{asset.media_id}{extension}"
        if candidate.exists():
            return candidate
        # Fall back to a glob on the id stem: the extension recorded on the
        # asset is the format name, which does not always equal the extension.
        matches = sorted(self._root.glob(f"{media_id}.*"))
        if not matches:
            msg = (
                f"Media asset {media_id!r} is registered but its file is missing from "
                "disk. It may have been deleted outside the application."
            )
            raise DataSourceError(msg, media_id=media_id)
        return matches[0]

    def delete(self, media_id: str) -> None:
        """Remove an asset and its file."""
        try:
            path = self.path_for(media_id)
        except DataSourceError:
            path = None
        self._assets.pop(media_id, None)
        if path is not None and path.exists():
            path.unlink()

    def clear(self) -> None:
        """Drop the in-memory index. Used by tests; leaves files untouched.

        Does not rebuild: a test that clears the index is asking for an empty
        one, and quietly re-adopting the files on disk would defeat that.
        """
        self._assets.clear()

    # --- library ----------------------------------------------------------
    def library_files(self, allowed_extensions: frozenset[str]) -> tuple[Path, ...]:
        """Files already present under the media root, for selection.

        Only files with a supported extension are listed. Managed assets (whose
        stem is a registered media id) are excluded so the library shows source
        material rather than the store's own copies.
        """
        if not self._root.exists():
            return ()
        found: list[Path] = []
        for path in sorted(self._root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in allowed_extensions:
                continue
            if self.has(path.stem):
                continue
            found.append(path)
        return tuple(found)

    def import_from_library(
        self,
        relative_path: str,
        media_format: MediaFormat,
    ) -> tuple[str, Path]:
        """Copy a library file into managed storage under a new media id.

        Copying rather than referencing in place means the managed asset cannot
        be changed underneath us by whatever else writes to the media directory,
        which keeps a replayed run reproducible.
        """
        source = resolve_within(self._root, relative_path)
        if not source.exists() or not source.is_file():
            msg = f"No file at {relative_path!r} inside the media directory."
            raise DataSourceError(msg, relative_path=relative_path)

        media_id, destination = self.allocate(media_format)
        shutil.copy2(source, destination)
        return media_id, destination


def utc_now() -> dt.datetime:
    """Current wall-clock instant.

    Ingestion timestamps are genuinely wall-clock events -- when *this system*
    received a file -- as distinct from the media timeline a recorded source is
    replayed on. Keeping the two apart is why this does not read a source clock.
    """
    return dt.datetime.now(tz=dt.UTC)
