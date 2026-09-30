"""Composing one mapping out of two sources, and naming the result.

Why there are two sources
-------------------------
:mod:`app.routing.mapping` loads a surveyed mapping *file*: somebody walked the
network, decided which road each camera watches, and wrote it down under a
version. Steps 2 and 3 added a second way to establish the same fact: an
operator is shown the nearest junction and the directed segments meeting it, and
confirms one. Both are human survey work. Neither is a guess.

They can disagree, and the disagreement is informative -- a file written six
months ago against a camera that has since been remounted will disagree with the
operator who just looked at the footage. So the two are not merged silently.
They are *composed*, by a stated rule, into an
:class:`EffectiveObservationMapping` that records what each source said and why
one won.

The precedence rule
-------------------
**A human-confirmed binding supersedes a file binding for the same observation
area.** The confirmation is the more recent act by a person who was looking at
this camera, and the file is a record of an earlier such act. Where only one
source has an opinion, that opinion stands. Where both agree, the result carries
the confirmed provenance and says the sources agree, because "two independent
surveys concur" is stronger evidence than either alone and losing that would
throw away the only corroboration this system ever gets.

Nothing is combined. A binding comes from exactly one source, and every area
where the sources differed is listed in :attr:`EffectiveObservationMapping.overrides`.

Identity, and what it is for
----------------------------
The effective mapping carries a SHA-256 over a canonical rendering of its own
content: the file identity, the confirmed-binding identity, every effective
binding and every precedence decision. Two logically identical compositions hash
identically regardless of dictionary ordering, insertion order or when they were
computed. No timestamp, uuid or wall-clock value enters the hash.

That identity is what will eventually let a route result answer "which exact
mapping produced this?". It does not answer it yet, and this module does not
pretend otherwise -- see :attr:`EffectiveObservationMapping.used_by_route_costing`,
which is fixed ``False`` and cannot be set true. The optimizer still reads the
mapping *file* alone. Wiring the identity onto a route result before the
optimizer consumes the effective mapping would assert that the route was
computed against it, which would be false.

Derived, never stored
---------------------
There is no effective-mapping registry and no file. It is recomputed from the
loaded mapping and the confirmed-binding registry whenever it is asked for,
which is cheap and has the property that matters: it cannot drift from its
inputs, and it cannot become a third source of truth that somebody edits.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from enum import StrEnum
from typing import Annotated, Any

from pydantic import Field, model_validator

from app.core.versions import APP_VERSION
from app.domain.base import DomainModel
from app.domain.network import RoadNetwork
from app.domain.provenance import Provenance, ValueSource
from app.domain.site import ConfirmedSiteBinding
from app.routing.mapping import MappingSource, ObservationBinding, ObservationMapping

#: Version of the *canonical rendering* used for the content hash, not of any
#: mapping. Bumped only when the hashed shape changes, so an old hash is never
#: silently compared against a new rendering of the same content.
CANONICAL_FORM_VERSION = "1"

#: Producer recorded in the composed mapping's provenance.
COMPOSER_PRODUCER = "routing.effective_mapping"

#: How many hex characters of the content hash form the human-facing version.
VERSION_PREFIX_LENGTH = 16


class BindingSource(StrEnum):
    """Which survey a binding in the effective mapping came from."""

    MAPPING_FILE = "mapping_file"
    """A versioned file loaded through ``POST /api/routes/mapping/load``."""

    CONFIRMED_BINDING = "confirmed_binding"
    """An operator confirmation recorded through the sites API."""


class BindingResolution(StrEnum):
    """Why this binding is the effective one for its observation area."""

    FILE_ONLY = "file_only"
    """Only the mapping file bound this area."""

    CONFIRMED_ONLY = "confirmed_only"
    """Only an operator confirmation bound this area."""

    SOURCES_AGREE = "sources_agree"
    """Both sources named the same target. The confirmed provenance is kept and
    the corroboration is recorded: two independent surveys concurring is the
    strongest evidence available here, and collapsing it to "confirmed" would
    discard the only cross-check this system gets."""

    CONFIRMED_BINDING_PRECEDENCE = "confirmed_binding_precedence"
    """The sources disagreed and the confirmation won."""


class ExclusionReason(StrEnum):
    """Why a confirmed binding did not reach the effective mapping."""

    DIFFERENT_NETWORK = "different_network"
    """Confirmed against another map entirely. Its ids mean something else
    here."""

    STALE_NETWORK_VERSION = "stale_network_version"
    """Confirmed against a different version of *this* map. Segment ids,
    directions and which arms meet which node are exactly what a re-survey
    redraws, so the binding requires reconfirmation rather than migration."""

    UNKNOWN_NETWORK_ELEMENT = "unknown_network_element"
    """Names a junction or segment the loaded network does not have. Would bind
    evidence to nothing while looking complete."""


class EffectiveBinding(DomainModel):
    """One observation area's effective binding, and how it got there."""

    observation_area_id: Annotated[str, Field(min_length=1, max_length=120)]
    binding: ObservationBinding
    """The binding itself, in the routing layer's own type, so the composed
    mapping is directly usable by whatever consumes mappings."""

    source: BindingSource
    resolution: BindingResolution

    confirmed: ConfirmedSiteBinding | None = None
    """The whole confirmation record when one is behind this binding.

    Carried in full rather than flattened into ``ObservationBinding``. The
    confirmation knows things a mapping binding has no field for -- the network
    version it was made against, the site coordinate at the time, the direction,
    the method and who confirmed -- and dropping any of it to fit the older
    shape would lose provenance for no benefit. The mapping type is extended
    only where routing itself needs the value."""

    superseded_segment_id: Annotated[str, Field(max_length=64)] | None = None
    superseded_junction_id: Annotated[str, Field(max_length=64)] | None = None
    """What the losing source said, when a source lost. Kept so an override can
    be reviewed without re-reading the file it overrode."""


class MappingOverride(DomainModel):
    """One observation area where the two sources did not simply align."""

    observation_area_id: Annotated[str, Field(min_length=1, max_length=120)]
    file_binding: Annotated[str, Field(max_length=64)] | None = None
    confirmed_binding: Annotated[str, Field(max_length=64)] | None = None
    effective_binding: Annotated[str, Field(max_length=64)] | None = None
    resolution: BindingResolution
    note: Annotated[str, Field(max_length=400)] = ""


class ExcludedConfirmedBinding(DomainModel):
    """A confirmation that exists and is deliberately not in force."""

    observation_area_id: Annotated[str, Field(min_length=1, max_length=120)]
    junction_id: Annotated[str, Field(max_length=64)] | None = None
    segment_id: Annotated[str, Field(max_length=64)] | None = None
    network_id: Annotated[str, Field(max_length=64)] | None = None
    network_version: Annotated[str, Field(max_length=50)] | None = None
    reason: ExclusionReason
    detail: Annotated[str, Field(min_length=1, max_length=400)]
    """Phrased for whoever has to fix it. Exclusions are never silent: a
    confirmation that quietly stopped applying is worse than one that never
    existed, because somebody believes it is working."""


class EffectiveObservationMapping(DomainModel):
    """The mapping in force, composed from the file and the confirmations."""

    # --- source identities ----------------------------------------------
    mapping_id: Annotated[str, Field(max_length=64)] | None = None
    mapping_version: Annotated[str, Field(max_length=50)] | None = None
    mapping_sha256: Annotated[str, Field(pattern="^[0-9a-f]{64}$")] | None = None
    """Identity of the loaded mapping *file*. All three are ``None`` when no
    file is loaded, which is a working state: the confirmations stand alone."""

    mapping_source: MappingSource | None = None

    confirmed_bindings_hash: Annotated[str, Field(pattern="^[0-9a-f]{64}$")]
    """Content identity of the confirmed set this was composed from, so the
    second input is as identifiable as the first."""

    confirmed_binding_count: Annotated[int, Field(ge=0)] = 0

    # --- network context --------------------------------------------------
    network_id: Annotated[str, Field(min_length=1, max_length=64)]
    network_version: Annotated[str, Field(min_length=1, max_length=50)]

    # --- the composition --------------------------------------------------
    mapping: ObservationMapping
    """The composed result as an ordinary ``ObservationMapping``.

    Built through the existing type on purpose: its ``_unique_areas`` validator
    is what guarantees at most one binding per observation area, so the
    no-duplicates rule is enforced by the same code that has always enforced it
    rather than by a new check that could disagree."""

    bindings: tuple[EffectiveBinding, ...] = ()
    """The same bindings with their resolution and provenance, sorted by
    observation area id."""

    overrides: tuple[MappingOverride, ...] = ()
    excluded: tuple[ExcludedConfirmedBinding, ...] = ()

    # --- identity ---------------------------------------------------------
    effective_mapping_hash: Annotated[str, Field(pattern="^[0-9a-f]{64}$")]
    effective_mapping_version: Annotated[str, Field(min_length=1, max_length=50)]
    """First 16 hex characters of the hash. A short form for display and for
    pinning; the full hash remains the identity."""

    provenance: Provenance

    used_by_route_costing: bool = False
    """Fixed ``False``, and structurally not settable otherwise.

    The optimizer is still handed the mapping *file* alone. Until it consumes
    the effective mapping, anything that recorded this identity against a route
    result would be asserting the route was computed from it."""

    warnings: tuple[str, ...] = ()
    note: str = ""

    @model_validator(mode="after")
    def _cannot_claim_to_be_in_force(self) -> EffectiveObservationMapping:
        if self.used_by_route_costing:
            msg = (
                "used_by_route_costing cannot be true: the route optimizer reads "
                "the loaded mapping file, not this composition, so marking it in "
                "force would attribute every route to bindings it never saw"
            )
            raise ValueError(msg)
        return self

    @property
    def binding_count(self) -> int:
        """Number of observation areas bound in the effective mapping."""
        return len(self.bindings)

    @property
    def override_count(self) -> int:
        """Number of areas where a confirmation superseded a file binding."""
        return sum(
            1
            for item in self.overrides
            if item.resolution is BindingResolution.CONFIRMED_BINDING_PRECEDENCE
        )

    def binding_for(self, observation_area_id: str) -> EffectiveBinding | None:
        """The effective binding for one area, or ``None``."""
        for item in self.bindings:
            if item.observation_area_id == observation_area_id:
                return item
        return None


# ---------------------------------------------------------------------------
# Canonical hashing
# ---------------------------------------------------------------------------
def _fingerprint(payload: dict[str, Any]) -> str:
    """SHA-256 over a canonical rendering of everything that changes the answer.

    Sorted keys and compact separators, so two structurally identical payloads
    hash identically regardless of the order their parts were assembled in.

    The same convention as ``app.simulation.scenario.compute_fingerprint``,
    written out here rather than imported: routing must not acquire a dependency
    on the simulator, and the simulator is out of scope for this change. The two
    are three lines each and the shared rule -- canonical JSON, then SHA-256 --
    is stated in both places.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def confirmed_bindings_fingerprint(
    confirmed: tuple[ConfirmedSiteBinding, ...],
) -> str:
    """Content identity of a set of confirmations.

    Hashes what the confirmations *say*, not when they were said. Two operators
    recording the same decision on different days produce the same mapping, and
    a mapping identity that changed because somebody re-confirmed an unchanged
    binding would be unusable for the replay question it exists to answer.

    ``confirmed_by`` and ``method`` are included because they are evidence: a
    binding accepted from a bearing alignment and one surveyed on site are
    different claims about the same road.
    """
    body = sorted(
        (
            {
                "observation_area_id": item.observation_area_id,
                "junction_id": item.junction_id,
                "segment_id": item.segment_id,
                "segment_direction": (
                    item.segment_direction.value if item.segment_direction else None
                ),
                "approach_id": item.approach_id,
                "network_id": item.network_id,
                "network_version": item.network_version,
                "confirmed_by": item.confirmation.confirmed_by,
                "method": item.confirmation.method.value,
                "segment_confirmed_by": (
                    item.segment_confirmation.confirmed_by
                    if item.segment_confirmation
                    else None
                ),
                "segment_method": (
                    item.segment_confirmation.method.value
                    if item.segment_confirmation
                    else None
                ),
            }
            for item in confirmed
        ),
        key=lambda entry: (
            str(entry["network_id"]),
            str(entry["observation_area_id"]),
        ),
    )
    return _fingerprint({"form": CANONICAL_FORM_VERSION, "confirmed": body})


def _binding_body(binding: ObservationBinding) -> dict[str, Any]:
    """The routing-relevant content of one binding, for hashing."""
    return {
        "observation_area_id": binding.observation_area_id,
        "segment_id": binding.segment_id,
        "junction_id": binding.junction_id,
        "approach_id": binding.approach_id,
    }


def effective_fingerprint(
    *,
    mapping_id: str | None,
    mapping_version: str | None,
    mapping_sha256: str | None,
    confirmed_hash: str,
    network_id: str,
    network_version: str,
    bindings: tuple[EffectiveBinding, ...],
    overrides: tuple[MappingOverride, ...],
    excluded: tuple[ExcludedConfirmedBinding, ...],
) -> str:
    """Deterministic content identity of one composition.

    Everything that could change which road an observation is attributed to is
    in here, including the precedence decisions: two compositions that bind the
    same areas to the same roads *for different reasons* are not the same
    mapping, and a reviewer replaying one needs to know which it was.
    """
    payload: dict[str, Any] = {
        "form": CANONICAL_FORM_VERSION,
        "mapping_id": mapping_id,
        "mapping_version": mapping_version,
        "mapping_sha256": mapping_sha256,
        "confirmed_bindings_hash": confirmed_hash,
        "network_id": network_id,
        "network_version": network_version,
        "bindings": sorted(
            (
                {
                    **_binding_body(item.binding),
                    "source": item.source.value,
                    "resolution": item.resolution.value,
                    "superseded_segment_id": item.superseded_segment_id,
                    "superseded_junction_id": item.superseded_junction_id,
                }
                for item in bindings
            ),
            key=lambda entry: str(entry["observation_area_id"]),
        ),
        "overrides": sorted(
            (
                {
                    "observation_area_id": item.observation_area_id,
                    "file_binding": item.file_binding,
                    "confirmed_binding": item.confirmed_binding,
                    "effective_binding": item.effective_binding,
                    "resolution": item.resolution.value,
                }
                for item in overrides
            ),
            key=lambda entry: str(entry["observation_area_id"]),
        ),
        "excluded": sorted(
            (
                {
                    "observation_area_id": item.observation_area_id,
                    "junction_id": item.junction_id,
                    "segment_id": item.segment_id,
                    "network_id": item.network_id,
                    "network_version": item.network_version,
                    "reason": item.reason.value,
                }
                for item in excluded
            ),
            key=lambda entry: (
                str(entry["observation_area_id"]),
                str(entry["reason"]),
            ),
        ),
    }
    return _fingerprint(payload)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------
def _target(binding: ObservationBinding) -> str | None:
    """What a binding names, for the conflict report."""
    return binding.segment_id or binding.junction_id


def _sources_agree(
    file_binding: ObservationBinding, confirmed_binding: ObservationBinding
) -> bool:
    """Whether the two sources make the same claim about which road.

    Compared at the level each source actually speaks at, not field by field. A
    mapping file binds a camera to a segment and usually records no junction;
    the confirmation records both. Requiring every field to match would report a
    disagreement every time a confirmation merely added the junction the file
    had omitted, which is extra detail rather than a contradiction.

    So: two segment bindings agree when they name the same segment. Two
    junction-only bindings agree when they name the same junction. A segment
    binding and a junction-only binding never agree -- one says which road and
    the other does not, and treating that as agreement would let a vaguer claim
    pass as corroboration of a more specific one.
    """
    if file_binding.segment_id is not None and confirmed_binding.segment_id is not None:
        return file_binding.segment_id == confirmed_binding.segment_id
    if file_binding.segment_id is None and confirmed_binding.segment_id is None:
        return file_binding.junction_id == confirmed_binding.junction_id
    return False


def _admit(
    confirmed: ConfirmedSiteBinding, network: RoadNetwork
) -> ExcludedConfirmedBinding | None:
    """Decide whether a confirmation may take effect on the loaded network.

    Returns the exclusion when it may not. Nothing is migrated and nothing is
    assumed: a binding made against a different map, or a different version of
    this one, is reported as needing reconfirmation exactly as the segment
    confirmation endpoint already requires.
    """
    if confirmed.network_id != network.network_id:
        return ExcludedConfirmedBinding(
            observation_area_id=confirmed.observation_area_id,
            junction_id=confirmed.junction_id,
            segment_id=confirmed.segment_id,
            network_id=confirmed.network_id,
            network_version=confirmed.network_version,
            reason=ExclusionReason.DIFFERENT_NETWORK,
            detail=(
                f"Confirmed against network {confirmed.network_id!r}, but network "
                f"{network.network_id!r} is loaded. Junction and segment ids mean "
                "different things on different maps."
            ),
        )

    if (
        confirmed.network_version is not None
        and confirmed.network_version != network.version
    ):
        return ExcludedConfirmedBinding(
            observation_area_id=confirmed.observation_area_id,
            junction_id=confirmed.junction_id,
            segment_id=confirmed.segment_id,
            network_id=confirmed.network_id,
            network_version=confirmed.network_version,
            reason=ExclusionReason.STALE_NETWORK_VERSION,
            detail=(
                f"Confirmed against network version {confirmed.network_version!r}, "
                f"but version {network.version!r} is loaded. Re-confirm this "
                "binding against the loaded network; it is not migrated."
            ),
        )

    junction_ids = {item.junction_id for item in network.junctions}
    segment_ids = {item.segment_id for item in network.segments}
    missing: list[str] = []
    if confirmed.junction_id not in junction_ids:
        missing.append(f"junction {confirmed.junction_id!r}")
    if confirmed.segment_id is not None and confirmed.segment_id not in segment_ids:
        missing.append(f"segment {confirmed.segment_id!r}")
    if missing:
        return ExcludedConfirmedBinding(
            observation_area_id=confirmed.observation_area_id,
            junction_id=confirmed.junction_id,
            segment_id=confirmed.segment_id,
            network_id=confirmed.network_id,
            network_version=confirmed.network_version,
            reason=ExclusionReason.UNKNOWN_NETWORK_ELEMENT,
            detail=(
                f"Names {' and '.join(missing)}, which network "
                f"{network.network_id!r} does not have. A binding to an element "
                "that does not exist would look complete while attributing "
                "evidence to nothing."
            ),
        )

    return None


def compose_effective_mapping(
    *,
    network: RoadNetwork,
    mapping: ObservationMapping | None,
    mapping_source: MappingSource | None,
    confirmed: tuple[ConfirmedSiteBinding, ...],
    binding_for: Any,
    composed_at: dt.datetime,
) -> EffectiveObservationMapping:
    """Compose the mapping in force from a loaded file and confirmations.

    ``binding_for`` renders a :class:`ConfirmedSiteBinding` as an
    :class:`~app.routing.mapping.ObservationBinding`; it is passed in rather
    than imported so that routing does not depend on the sites package, which
    already depends on routing.

    Pure. Reads nothing, writes nothing, and derives its identity entirely from
    its arguments.
    """
    file_bindings = {
        item.observation_area_id: item for item in (mapping.bindings if mapping else ())
    }

    admitted: dict[str, ConfirmedSiteBinding] = {}
    excluded: list[ExcludedConfirmedBinding] = []
    for item in confirmed:
        exclusion = _admit(item, network)
        if exclusion is not None:
            excluded.append(exclusion)
            continue
        admitted[item.observation_area_id] = item

    effective: list[EffectiveBinding] = []
    overrides: list[MappingOverride] = []
    warnings: list[str] = []

    for area_id in sorted(set(file_bindings) | set(admitted)):
        file_binding = file_bindings.get(area_id)
        confirmation = admitted.get(area_id)

        if confirmation is None:
            # File only. Its provenance is the mapping's own, unchanged.
            assert file_binding is not None  # noqa: S101 - one of the two must exist
            effective.append(
                EffectiveBinding(
                    observation_area_id=area_id,
                    binding=file_binding,
                    source=BindingSource.MAPPING_FILE,
                    resolution=BindingResolution.FILE_ONLY,
                )
            )
            continue

        confirmed_binding = binding_for(confirmation)
        confirmed_target = _target(confirmed_binding)

        if file_binding is None:
            effective.append(
                EffectiveBinding(
                    observation_area_id=area_id,
                    binding=confirmed_binding,
                    source=BindingSource.CONFIRMED_BINDING,
                    resolution=BindingResolution.CONFIRMED_ONLY,
                    confirmed=confirmation,
                )
            )
            continue

        file_target = _target(file_binding)
        agree = _sources_agree(file_binding, confirmed_binding)
        resolution = (
            BindingResolution.SOURCES_AGREE
            if agree
            else BindingResolution.CONFIRMED_BINDING_PRECEDENCE
        )

        effective.append(
            EffectiveBinding(
                observation_area_id=area_id,
                binding=confirmed_binding,
                source=BindingSource.CONFIRMED_BINDING,
                resolution=resolution,
                confirmed=confirmation,
                superseded_segment_id=None if agree else file_binding.segment_id,
                superseded_junction_id=None if agree else file_binding.junction_id,
            )
        )

        note = (
            "Both sources name the same target. The confirmed provenance is "
            "kept and the corroboration is recorded."
            if agree
            else "A human confirmation supersedes the mapping file for this area."
        )
        if (
            not agree
            and file_binding.segment_id is not None
            and confirmed_binding.segment_id is None
        ):
            # The confirmation wins, but it is a weaker statement: it says which
            # intersection, not which road. Said out loud rather than left for
            # somebody to notice a segment binding had vanished.
            note = (
                "A human confirmation supersedes the mapping file for this area. "
                "The confirmation names a junction only, so the file's segment "
                "attribution is withdrawn until a directed segment is confirmed."
            )
            warnings.append(
                f"Observation area {area_id!r}: the confirmed binding names "
                f"junction {confirmed_binding.junction_id} only and supersedes "
                f"the file's segment {file_binding.segment_id}. Confirm a "
                "directed segment to restore road-level attribution."
            )

        overrides.append(
            MappingOverride(
                observation_area_id=area_id,
                file_binding=file_target,
                confirmed_binding=confirmed_target,
                effective_binding=confirmed_target,
                resolution=resolution,
                note=note,
            )
        )

    bindings = tuple(effective)
    override_records = tuple(overrides)
    exclusion_records = tuple(
        sorted(excluded, key=lambda item: (item.observation_area_id, item.reason.value))
    )
    confirmed_hash = confirmed_bindings_fingerprint(confirmed)

    content_hash = effective_fingerprint(
        mapping_id=mapping.mapping_id if mapping else None,
        mapping_version=mapping.version if mapping else None,
        mapping_sha256=mapping_source.sha256 if mapping_source else None,
        confirmed_hash=confirmed_hash,
        network_id=network.network_id,
        network_version=network.version,
        bindings=bindings,
        overrides=override_records,
        excluded=exclusion_records,
    )
    short_version = content_hash[:VERSION_PREFIX_LENGTH]

    if exclusion_records:
        warnings.append(
            f"{len(exclusion_records)} confirmed binding(s) are excluded and are "
            "not in force. They require reconfirmation against the loaded "
            "network; none has been migrated."
        )

    composed = ObservationMapping(
        # Identity derived from content, so two identical compositions address
        # the same mapping. The `_unique_areas` validator on this type is what
        # enforces one binding per observation area.
        mapping_id=f"effective-{short_version}"[:64],
        version=short_version,
        network_id=network.network_id,
        network_version=network.version,
        provenance=Provenance(
            source=ValueSource.DERIVED,
            producer=COMPOSER_PRODUCER,
            producer_version=APP_VERSION,
            observed_at=composed_at,
            computed_at=composed_at,
            input_refs=tuple(
                ref
                for ref in (
                    f"mapping:{mapping.mapping_id}@{mapping.version}" if mapping else None,
                    f"confirmed_bindings:{confirmed_hash}",
                    f"network:{network.network_id}@{network.version}",
                )
                if ref is not None
            ),
            notes=(
                "Composed from a surveyed mapping file and human-confirmed site "
                "bindings under a fixed precedence rule. Not used for route "
                "costing."
            ),
        ),
        bindings=tuple(item.binding for item in bindings),
    )

    return EffectiveObservationMapping(
        mapping_id=mapping.mapping_id if mapping else None,
        mapping_version=mapping.version if mapping else None,
        mapping_sha256=mapping_source.sha256 if mapping_source else None,
        mapping_source=mapping_source,
        confirmed_bindings_hash=confirmed_hash,
        confirmed_binding_count=len(confirmed),
        network_id=network.network_id,
        network_version=network.version,
        mapping=composed,
        bindings=bindings,
        overrides=override_records,
        excluded=exclusion_records,
        effective_mapping_hash=content_hash,
        effective_mapping_version=short_version,
        provenance=composed.provenance,
        warnings=tuple(warnings),
        note=_note(mapping, bindings, override_records, exclusion_records),
    )


def _note(
    mapping: ObservationMapping | None,
    bindings: tuple[EffectiveBinding, ...],
    overrides: tuple[MappingOverride, ...],
    excluded: tuple[ExcludedConfirmedBinding, ...],
) -> str:
    """One paragraph an operator can read instead of the structure."""
    if not bindings:
        base = (
            "No observation area is bound: no mapping file is loaded and no "
            "confirmation applies to this network. Every segment is priced as "
            "unobserved."
        )
    else:
        sources = "a mapping file and human confirmations" if mapping else (
            "human confirmations only"
        )
        superseded = sum(
            1
            for item in overrides
            if item.resolution is BindingResolution.CONFIRMED_BINDING_PRECEDENCE
        )
        base = (
            f"{len(bindings)} observation area(s) bound from {sources}. "
            f"{superseded} confirmation(s) superseded a file binding"
        )
        agreed = sum(
            1 for item in overrides if item.resolution is BindingResolution.SOURCES_AGREE
        )
        base += f"; {agreed} area(s) had both sources agree." if agreed else "."
    if excluded:
        base += (
            f" {len(excluded)} confirmed binding(s) are excluded and need "
            "reconfirmation."
        )
    return (
        base
        + " This composition is not used for route costing: the optimizer still "
        "reads the loaded mapping file alone."
    )
