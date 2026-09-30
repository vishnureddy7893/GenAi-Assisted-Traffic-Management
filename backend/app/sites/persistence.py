"""One validated JSON document on disk, read and written safely.

Extracted from the observation site registry when a second store -- confirmed
bindings -- needed exactly the same discipline. Both keep survey-grade facts
that somebody established by hand, and both must behave identically when the
file is missing, unreadable or half-written. Two copies of that logic would
drift, and the half that drifted would be the error path nobody exercises.

The three rules this enforces, in one place:

**An absent file is an empty document, not a failure.** Nothing has been
recorded yet, which is a working state.

**A file that will not parse is never reported as empty and never overwritten.**
It goes degraded: the load error is retained, reads raise, writes are refused.
Reporting empty would hide records that are still in the file, and the first
successful write would then destroy them.

**A write replaces the whole file atomically** -- a temporary file in the same
directory, then ``os.replace``, which is atomic on POSIX and Windows alike. A
crash mid-write leaves the previous file intact instead of truncating away every
record in order to save one.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import ValidationError

from app.core.errors import DataSourceError
from app.core.logging import get_logger
from app.domain.base import DomainModel

logger = get_logger(__name__)

#: Cap on a registry file. These hold one entry per surveyed camera; anything
#: larger is a mistaken path rather than a registry.
MAX_REGISTRY_FILE_BYTES = 4 * 1024 * 1024

DocumentT = TypeVar("DocumentT", bound=DomainModel)


class JsonDocumentFile(Generic[DocumentT]):
    """Read and write one pydantic-validated JSON document.

    ``description`` names the store in operator-facing errors, so a 502 says
    *which* registry will not open rather than only where it lives.
    """

    __slots__ = ("_description", "_document_type", "_path", "_schema_version")

    def __init__(
        self,
        path: Path,
        *,
        document_type: type[DocumentT],
        schema_version: str,
        description: str,
    ) -> None:
        self._path = Path(path)
        self._document_type = document_type
        self._schema_version = schema_version
        self._description = description

    @property
    def path(self) -> Path:
        """File this document is stored in."""
        return self._path

    @property
    def description(self) -> str:
        """Operator-facing name of the store."""
        return self._description

    def read(self) -> tuple[DocumentT | None, str | None]:
        """Load the document.

        Returns ``(document, None)`` when it loaded, ``(None, None)`` when the
        file is absent, and ``(None, reason)`` when it exists but will not load.
        The reason is a sentence fragment, phrased to follow "could not be read:".
        """
        if not self._path.exists():
            return None, None

        try:
            size = self._path.stat().st_size
            if size > MAX_REGISTRY_FILE_BYTES:
                return None, (
                    f"the file is {size} bytes, above the "
                    f"{MAX_REGISTRY_FILE_BYTES}-byte limit."
                )
            raw = self._path.read_bytes()
        except OSError as exc:
            return None, f"{exc}."

        if not raw.strip():
            # An empty file is a completed truncation, not a valid empty
            # document: an empty document is an *absent* file.
            return None, "the file is empty."

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return None, f"it is not valid UTF-8 JSON ({exc})."

        try:
            document = self._document_type.model_validate(payload)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in item['loc']) or '(root)'}: {item['msg']}"
                for item in exc.errors()[:6]
            )
            return None, f"it is not a valid {self._description} ({problems})."

        declared = getattr(document, "schema_version", None)
        if declared != self._schema_version:
            return None, (
                f"it declares schema version {declared!r}, but this build reads "
                f"version {self._schema_version!r}."
            )

        logger.info(
            "registry loaded",
            extra={"path": str(self._path), "store": self._description},
        )
        return document, None

    def write(self, document: DocumentT) -> None:
        """Replace the stored file with ``document``, atomically."""
        payload = document.model_dump_json(indent=2) + "\n"

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Written into the destination directory so the replace stays on one
            # filesystem; a cross-device replace is not atomic.
            handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed below
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            )
            temporary = Path(handle.name)
            try:
                with handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self._path)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        except OSError as exc:
            msg = (
                f"The {self._description} at {self._path} could not be written: "
                f"{exc}. Nothing was recorded."
            )
            raise DataSourceError(msg, path=str(self._path)) from exc

    def unreadable_error(self, load_error: str) -> DataSourceError:
        """The 502 raised while the stored file cannot be read."""
        msg = (
            f"The {self._description} at {self._path} could not be read: "
            f"{load_error} Nothing can be reported or recorded until it is "
            "repaired or moved, because reporting an empty registry would hide "
            "records that are still in the file."
        )
        return DataSourceError(msg, path=str(self._path))
