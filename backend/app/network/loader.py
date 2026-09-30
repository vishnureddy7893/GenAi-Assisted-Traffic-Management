"""Loading a surveyed road network from a file.

Validation happens at load, not at use
--------------------------------------
:class:`~app.domain.network.RoadNetwork` already validates referential integrity
in its own model validator: dangling segment references, approaches on segments
that do not terminate at their junction, movements that do not connect. This
module's job is to get bytes into that constructor and to turn whatever goes
wrong into an error an operator can act on.

That ordering matters. An unchecked dangling reference does not fail at import
-- it fails much later, inside route relaxation, as a ``KeyError`` on an
emergency request. Failing at load is strictly better, and failing at load with
the offending id named is better still.

Identity is a checksum, not a filename
--------------------------------------
A loaded network is recorded with the SHA-256 of the file it came from, beside
the ``network_id`` and ``version`` the file declares. The declared version is
what a decision record pins; the checksum is what proves the file behind that
version was not edited underneath it. Two networks claiming version ``1.0`` with
different checksums is a condition worth being able to detect.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Annotated

from pydantic import Field, ValidationError

from app.core.errors import DataSourceError, ResourceNotFoundError, ValidationFailedError
from app.domain.base import DomainModel
from app.domain.network import RoadNetwork

#: Cap on a network file, so a mistaken path at a multi-gigabyte file fails
#: fast instead of being read into memory.
MAX_NETWORK_FILE_BYTES = 64 * 1024 * 1024


class NetworkSource(DomainModel):
    """Where a loaded network came from, and what exactly was read."""

    relative_path: Annotated[str, Field(min_length=1, max_length=1024)]
    sha256: Annotated[str, Field(pattern="^[0-9a-f]{64}$")]
    size_bytes: Annotated[int, Field(ge=0)]
    loaded_at: dt.datetime
    network_id: Annotated[str, Field(min_length=1, max_length=64)]
    version: Annotated[str, Field(min_length=1, max_length=50)]

    junction_count: Annotated[int, Field(ge=0)]
    segment_count: Annotated[int, Field(ge=0)]
    facility_count: Annotated[int, Field(ge=0)]


def _format_validation_error(error: ValidationError, relative_path: str) -> str:
    """Render pydantic's report as something an operator can act on."""
    problems = []
    for item in error.errors()[:8]:
        location = ".".join(str(part) for part in item["loc"]) or "(root)"
        problems.append(f"{location}: {item['msg']}")
    listed = "; ".join(problems)
    more = "" if len(error.errors()) <= 8 else f" (and {len(error.errors()) - 8} more)"
    return f"Network file {relative_path!r} is not a valid road network. {listed}{more}"


def load_network(path: Path, relative_path: str, *, loaded_at: dt.datetime) -> tuple[
    RoadNetwork, NetworkSource
]:
    """Read and validate a road network file.

    Three failures, kept distinct because they call for different responses:

    * :class:`~app.core.errors.ResourceNotFoundError` (404) -- nothing at that
      path. A caller naming a file that is not there has made a mistake; it is
      not an upstream failure, and reporting it as one would send an operator to
      check a decoder that is working fine.
    * :class:`~app.core.errors.DataSourceError` (502) -- the file is there but
      cannot be read.
    * :class:`~app.core.errors.ValidationFailedError` (422) -- it was read, and
      it is not a coherent network.
    """
    if not path.exists() or not path.is_file():
        msg = (
            f"No network file at {relative_path!r} inside the networks directory. "
            "GET /api/network/status lists the files that are there."
        )
        raise ResourceNotFoundError(msg, relative_path=relative_path)

    size = path.stat().st_size
    if size > MAX_NETWORK_FILE_BYTES:
        msg = (
            f"Network file {relative_path!r} is {size} bytes, above the "
            f"{MAX_NETWORK_FILE_BYTES}-byte limit."
        )
        raise ValidationFailedError(msg, relative_path=relative_path, size_bytes=size)

    try:
        raw = path.read_bytes()
    except OSError as exc:
        msg = f"Network file {relative_path!r} exists but could not be read: {exc}."
        raise DataSourceError(msg, relative_path=relative_path) from exc
    digest = hashlib.sha256(raw).hexdigest()

    try:
        payload = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        msg = f"Network file {relative_path!r} is not valid UTF-8."
        raise ValidationFailedError(msg, relative_path=relative_path) from exc
    except json.JSONDecodeError as exc:
        msg = (
            f"Network file {relative_path!r} is not valid JSON: {exc.msg} "
            f"at line {exc.lineno}, column {exc.colno}."
        )
        raise ValidationFailedError(msg, relative_path=relative_path) from exc

    if not isinstance(payload, dict):
        msg = (
            f"Network file {relative_path!r} contains a {type(payload).__name__}, "
            "but a road network must be a JSON object."
        )
        raise ValidationFailedError(msg, relative_path=relative_path)

    try:
        network = RoadNetwork.model_validate(payload)
    except ValidationError as exc:
        raise ValidationFailedError(
            _format_validation_error(exc, relative_path), relative_path=relative_path
        ) from exc

    source = NetworkSource(
        relative_path=relative_path,
        sha256=digest,
        size_bytes=size,
        loaded_at=loaded_at,
        network_id=network.network_id,
        version=network.version,
        junction_count=len(network.junctions),
        segment_count=len(network.segments),
        facility_count=len(network.facilities),
    )
    return network, source


def list_available(networks_dir: Path) -> tuple[str, ...]:
    """Relative paths of the JSON files that could be loaded.

    Lists candidates without opening them: a file appearing here is a statement
    that it exists and ends in ``.json``, not that it parses as a network.
    """
    if not networks_dir.exists() or not networks_dir.is_dir():
        return ()
    return tuple(
        sorted(
            item.relative_to(networks_dir).as_posix()
            for item in networks_dir.rglob("*.json")
            if item.is_file()
        )
    )
