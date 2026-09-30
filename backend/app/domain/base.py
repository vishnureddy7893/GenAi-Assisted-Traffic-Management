"""Shared base model for every domain schema.

Three settings, each chosen for a specific failure it prevents:

``frozen=True``
    Domain objects are *snapshots*. A ``NetworkState`` describes the network at
    one instant; if a later stage could mutate it in place, we would lose the
    ability to replay a decision against the exact state that produced it, and
    two subsystems holding the same object would silently disagree. Immutability
    makes a snapshot mean what it says.

``extra="forbid"``
    A typo in a field name would otherwise be accepted and silently ignored,
    producing a model that validates cleanly while missing the value the caller
    thought they set. For safety-relevant fields (``is_blocked``, ``tier``) that
    is the worst possible failure mode.

``validate_assignment=True``
    Belt and braces alongside ``frozen``: a field set after construction is
    re-validated rather than trusted.

    It does **not** cover ``model_copy(update=...)``, which writes fields and
    returns without running model validators. Anything applying an update that
    must satisfy a cross-field invariant -- a ``CLEARED`` incident recording
    ``cleared_at``, a blocked segment naming its cause -- has to round-trip
    through ``model_validate`` instead. The lifecycle registries in
    :mod:`app.emergency` do exactly that, and a test pins the behaviour so this
    note cannot quietly become wrong.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DomainModel(BaseModel):
    """Immutable, strictly-validated base for all domain schemas."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
        str_strip_whitespace=True,
        populate_by_name=True,
    )
