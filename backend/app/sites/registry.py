"""Durable storage for observation site locations.

Why this is not held in memory
------------------------------
Every other registry in this service is per-process by design, and each says so:
an incident, an analytics result or a playback session is cheap to reproduce and
means nothing after a restart. A surveyed location is the opposite. Somebody
stood at a junction, or read a coordinate off a map, and recorded it. Losing
that on restart would mean re-surveying, and a survey that is expensive to
repeat is one that gets guessed instead.

So this registry writes to a JSON file and reads it back at construction. That
is the same shape the road network and the observation mapping already use -- a
validated JSON document under a configured directory -- with the one difference
that this service is allowed to *write* it, because a site is recorded through
the API rather than hand-authored.

Independent of the media store
------------------------------
The file lives outside the media directory and is keyed by observation area, not
by media id. Deleting a video, clearing the media store, or restarting the
process leaves the sites untouched: the place did not stop existing because the
footage of it did.

Reading and writing the file
----------------------------
Delegated to :class:`~app.sites.persistence.JsonDocumentFile`, which is where
the atomic-write and never-overwrite-an-unreadable-file rules live. Confirmed
bindings keep the same rules through the same helper, so the error paths of the
two stores cannot drift apart.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Annotated

from pydantic import Field

from app.core.errors import DataSourceError, ResourceNotFoundError
from app.core.logging import get_logger
from app.domain.base import DomainModel
from app.domain.site import ObservationSite
from app.sites.persistence import MAX_REGISTRY_FILE_BYTES, JsonDocumentFile

logger = get_logger(__name__)

#: Version of the on-disk document, not of the domain model. Bumped only when
#: the file layout changes in a way an older reader could misinterpret.
SITES_FILE_SCHEMA_VERSION = "1"

#: Retained under its original name; the limit itself is shared with every other
#: registry file.
MAX_SITES_FILE_BYTES = MAX_REGISTRY_FILE_BYTES


class SiteRegistryDocument(DomainModel):
    """The whole registry, as it is stored on disk."""

    schema_version: Annotated[str, Field(min_length=1, max_length=20)]
    sites: tuple[ObservationSite, ...] = ()


class ObservationSiteRegistry:
    """Observation site locations, persisted to a JSON file.

    Construction reads the file if it exists. An absent file is an empty
    registry, which is a working state: no site has been recorded yet, and
    nothing downstream needs one in order to function.
    """

    __slots__ = ("_file", "_load_error", "_sites")

    def __init__(self, path: Path) -> None:
        self._file: JsonDocumentFile[SiteRegistryDocument] = JsonDocumentFile(
            path,
            document_type=SiteRegistryDocument,
            schema_version=SITES_FILE_SCHEMA_VERSION,
            description="observation site registry",
        )
        self._sites: dict[str, ObservationSite] = {}
        self._load_error: str | None = None
        self._read()

    # --- state ------------------------------------------------------------
    @property
    def path(self) -> Path:
        """File this registry is stored in."""
        return self._file.path

    @property
    def load_error(self) -> str | None:
        """Why the stored file could not be read, when it could not be."""
        return self._load_error

    @property
    def is_degraded(self) -> bool:
        """Whether the stored file exists but could not be read.

        A degraded registry answers no question about sites, because the honest
        answer to "which sites are recorded?" is "the file that knows will not
        open", not "none".
        """
        return self._load_error is not None

    def __len__(self) -> int:
        """Number of sites currently held."""
        return len(self._sites)

    # --- reading ----------------------------------------------------------
    def _require_readable(self) -> None:
        """Refuse to answer while the stored file is unreadable."""
        if self._load_error is None:
            return
        raise self._file.unreadable_error(self._load_error)

    def get(self, observation_area_id: str) -> ObservationSite:
        """Return one site, or raise :class:`ResourceNotFoundError` (404)."""
        self._require_readable()
        site = self._sites.get(observation_area_id)
        if site is None:
            msg = (
                "No location has been recorded for observation area "
                f"{observation_area_id!r}."
            )
            raise ResourceNotFoundError(msg, observation_area_id=observation_area_id)
        return site

    def find(self, observation_area_id: str) -> ObservationSite | None:
        """Return one site, or ``None`` when none is recorded for this area.

        The non-raising counterpart to :meth:`get`, for callers where an absent
        site is an ordinary outcome rather than a failed lookup -- the advisor
        can describe traffic at a camera whose location nobody has surveyed.
        """
        self._require_readable()
        return self._sites.get(observation_area_id)

    def has(self, observation_area_id: str) -> bool:
        """Whether a site is recorded for this observation area."""
        self._require_readable()
        return observation_area_id in self._sites

    def list(self) -> tuple[ObservationSite, ...]:
        """Every recorded site, most recently updated first."""
        self._require_readable()
        return tuple(
            sorted(
                self._sites.values(),
                key=lambda site: (site.updated_at, site.observation_area_id),
                reverse=True,
            )
        )

    # --- writing ----------------------------------------------------------
    def record(self, site: ObservationSite) -> tuple[ObservationSite, bool]:
        """Create or replace the site for an observation area.

        Returns the stored site and whether it replaced an existing one.

        Re-recording an area is an update, not a duplicate: a camera is
        re-surveyed, or the first coordinate was wrong, and the corrected
        position must win. ``recorded_at`` is carried over from the site being
        replaced so the original record's age survives the correction, while
        ``updated_at`` marks the change.
        """
        self._require_readable()
        existing = self._sites.get(site.observation_area_id)
        stored = (
            site.model_copy(update={"recorded_at": existing.recorded_at})
            if existing is not None
            else site
        )

        previous = dict(self._sites)
        self._sites[stored.observation_area_id] = stored
        try:
            self._write()
        except DataSourceError:
            # The in-memory view must not survive a failed write, or a caller
            # would be told the site was recorded while the file says otherwise.
            self._sites = previous
            raise

        logger.info(
            "observation site recorded",
            extra={
                "observation_area_id": stored.observation_area_id,
                "replaced": existing is not None,
                "has_bearing": stored.bearing_degrees is not None,
            },
        )
        return stored, existing is not None

    # --- persistence ------------------------------------------------------
    def _read(self) -> None:
        """Load the stored document, recording why if it will not load."""
        document, error = self._file.read()
        self._load_error = error
        if document is not None:
            self._sites = {site.observation_area_id: site for site in document.sites}

    def _write(self) -> None:
        """Serialise the whole registry over the stored file, atomically."""
        self._file.write(
            SiteRegistryDocument(
                schema_version=SITES_FILE_SCHEMA_VERSION,
                sites=tuple(
                    sorted(
                        self._sites.values(), key=lambda site: site.observation_area_id
                    )
                ),
            )
        )


def utc_now() -> dt.datetime:
    """Current wall-clock instant, for stamping a record's time.

    A survey is recorded when somebody records it, so this is genuinely
    wall-clock -- unlike a media timeline, which is replayed.
    """
    return dt.datetime.now(tz=dt.UTC)
