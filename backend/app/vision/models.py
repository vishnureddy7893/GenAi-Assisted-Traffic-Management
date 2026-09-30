"""Model weight management and identity.

Rules this module enforces:

* **Weights never enter Git.** They live under the configured ``models_dir``,
  which ``.gitignore`` excludes. Nothing here writes into the source tree.
* **The model name is configuration, never a literal at a call site.**
* **Identity is reported, and it is measured.** The SHA-256 of the actual file
  on disk is what gets attached to inference results, not the configured name.
  Two runs of "yolov8n" can use different bytes; a checksum makes that visible.
* **A model that is not loaded is never reported as loaded.** Absence produces
  an explicit unavailable status naming the reason, and inference then fails
  loudly rather than returning an empty detection list that would read as
  "nothing on the road".
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated

from pydantic import Field

from app.core.errors import DataSourceError, ValidationFailedError
from app.core.logging import get_logger
from app.domain.base import DomainModel

logger = get_logger(__name__)

#: Read size for checksumming. Weights are small, but streaming avoids loading
#: an unexpectedly large model file into memory just to hash it.
_HASH_CHUNK = 1024 * 1024


class ModelIdentity(DomainModel):
    """Exactly which model produced a set of detections."""

    name: Annotated[str, Field(min_length=1, max_length=120)]
    backend: Annotated[str, Field(min_length=1, max_length=40)]
    file_name: Annotated[str, Field(max_length=255)] | None = None

    sha256: Annotated[str, Field(min_length=64, max_length=64)] | None = None
    """Checksum of the weights actually loaded. ``None`` only when the backend
    holds no local file. This is the field that makes a result reproducible."""

    size_bytes: Annotated[int, Field(ge=0)] | None = None
    task: Annotated[str, Field(max_length=40)] | None = None
    class_count: Annotated[int, Field(ge=0)] | None = None

    source: Annotated[str, Field(max_length=300)] | None = None
    """Where the weights came from, e.g. a release URL or ``local file``."""

    @property
    def short_hash(self) -> str | None:
        """First 12 hex characters of the checksum, for display."""
        return self.sha256[:12] if self.sha256 else None


def compute_sha256(path: Path) -> str:
    """Stream a file through SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_file(models_dir: Path, model_name: str) -> Path:
    """Resolve a configured model name to a path inside ``models_dir``.

    Containment is enforced the same way media selection is: resolve first,
    then check. Model names come from configuration rather than from a request,
    so this is defence in depth -- but a deployment that templates the model
    name from an environment variable is one indirection away from being
    request-influenced, and the check costs nothing.
    """
    if not model_name or not model_name.strip():
        msg = "No detector model is configured."
        raise ValidationFailedError(msg)
    if "\x00" in model_name:
        msg = "Model name contains an illegal character."
        raise ValidationFailedError(msg, model=model_name)

    candidate = Path(model_name)
    if candidate.is_absolute() or candidate.drive:
        msg = (
            "Detector model must be a name inside the models directory, not an "
            "absolute path."
        )
        raise ValidationFailedError(msg, model=model_name)

    root = models_dir.resolve()
    target = (root / candidate).resolve()
    if target != root and root not in target.parents:
        msg = "Detector model path escapes the models directory."
        raise ValidationFailedError(msg, model=model_name)
    return target


def describe_weights(path: Path, *, name: str, backend: str, source: str) -> ModelIdentity:
    """Build a :class:`ModelIdentity` from a weights file on disk."""
    if not path.exists():
        msg = f"Model weights not found at {path.name}."
        raise DataSourceError(msg, model=name)
    stat = path.stat()
    return ModelIdentity(
        name=name,
        backend=backend,
        file_name=path.name,
        sha256=compute_sha256(path),
        size_bytes=stat.st_size,
        source=source,
    )


def ensure_weights(models_dir: Path, model_name: str) -> Path:
    """Return the path to the model weights, downloading them if necessary.

    Downloads go into ``models_dir`` and nowhere else. Ultralytics otherwise
    writes to the process working directory, which would scatter weight files
    through the source tree and eventually into a commit.

    Raises :class:`DataSourceError` with an actionable message when the weights
    are absent and cannot be fetched -- never a silent empty model.
    """
    target = resolve_model_file(models_dir, model_name)
    if target.exists():
        return target

    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics.utils.downloads import attempt_download_asset
    except ImportError as exc:
        msg = (
            f"Model weights {model_name!r} are not present in the models directory "
            f"and cannot be downloaded because ultralytics is not installed. "
            f"Either install it, or place the weights at {target}."
        )
        raise DataSourceError(msg, model=model_name, models_dir=str(models_dir)) from exc

    logger.info(
        "downloading model weights",
        extra={"model": model_name, "destination": str(target)},
    )
    try:
        # Ultralytics resolves the asset by filename and writes it next to the
        # path we hand it, which is why the target is inside models_dir.
        downloaded = attempt_download_asset(str(target))
    except Exception as exc:
        msg = (
            f"Could not download model weights {model_name!r}: "
            f"{type(exc).__name__}. Place the file at {target} manually, or "
            f"configure a different model."
        )
        raise DataSourceError(msg, model=model_name) from exc

    result = Path(downloaded)
    if not result.exists():
        msg = (
            f"Model weights {model_name!r} were reported as downloaded but no file "
            f"exists at {result}."
        )
        raise DataSourceError(msg, model=model_name)

    if result != target:
        # Ultralytics may resolve to its own cache; move it under our control so
        # every weight file this application uses lives in one auditable place.
        result.replace(target)

    logger.info(
        "model weights ready",
        extra={"model": model_name, "path": str(target), "bytes": target.stat().st_size},
    )
    return target
