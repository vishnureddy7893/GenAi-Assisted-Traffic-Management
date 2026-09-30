"""Binding observation areas to network elements.

The gap this closes
-------------------
Phase 5 interprets congestion for an **observation area** -- one camera's field
of view. Phase 6 loaded a surveyed graph of **segments and junctions**. Nothing
connected the two, so network state reported ``unknown`` congestion everywhere,
and Phase 6 said so rather than guessing which road a camera was pointed at.

This module is that connection, and it is a *survey artifact*: somebody has to
look at the camera and record which road it watches. It cannot be inferred, and
inferring it is the one thing that must not happen here -- a congestion band
attached to the wrong segment is a specific false claim that a route would be
planned around.

Why it is configuration, not code
---------------------------------
The mapping is loaded from a file and validated against the loaded network, for
the same reason the network itself is. Hard-coding ``camera_3 -> SEG-12`` inside
the optimizer would make the binding invisible to review, impossible to version,
and wrong the moment a camera is remounted.

Unmapped is a reported state, not a silent one
----------------------------------------------
An observation area with no mapping does not contribute to any segment's cost.
That is the whole point: its evidence is real but unattributable, and pretending
otherwise would be worse than ignoring it. Every unmapped area is named in the
optimization result so an operator can see that traffic was observed and
deliberately not used.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Annotated

from pydantic import Field, ValidationError, model_validator

from app.core.errors import ResourceNotFoundError, ValidationFailedError
from app.domain.base import DomainModel
from app.domain.network import RoadNetwork
from app.domain.provenance import Provenance
from app.domain.site import BindingConfirmation

#: Cap on a mapping file. A mapping is one line per camera; anything larger is a
#: mistaken path rather than a survey.
MAX_MAPPING_FILE_BYTES = 4 * 1024 * 1024


class ObservationBinding(DomainModel):
    """One observation area bound to one network element."""

    observation_area_id: Annotated[str, Field(min_length=1, max_length=120)]
    """Matches ``IntersectionTrafficState.observation_area_id`` exactly. Not a
    camera name or a media id: the id Phase 5 actually keys its verdicts by, so
    the join cannot silently miss."""

    segment_id: Annotated[str, Field(max_length=64)] | None = None
    """The directed segment this camera watches. Directed on purpose: a camera
    facing one way down a two-way road observes one of the two segments, and
    binding it to both would apply an inbound queue to outbound traffic."""

    junction_id: Annotated[str, Field(max_length=64)] | None = None
    """The junction this camera watches, when it covers the intersection rather
    than one approach."""

    approach_id: Annotated[str, Field(max_length=64)] | None = None
    """Optional finer binding, recorded for later phases. Not used for costing:
    a route traverses segments, not approaches."""

    coverage_note: Annotated[str, Field(max_length=500)] | None = None
    """What the camera actually sees, in the surveyor's words. The thing a
    reviewer needs in order to disagree with the binding."""

    confirmation: BindingConfirmation | None = None
    """Present when this binding came from an operator confirming a proposed
    site-to-junction association rather than from a hand-authored survey file.

    Optional, so every mapping file written before this existed still validates
    unchanged, and a binding without it means exactly what it always did: it was
    surveyed. Its value is that the two are now distinguishable -- "a surveyor
    wrote this down" and "an operator accepted a nearest-junction proposal" are
    different strengths of evidence, and a reviewer must be able to tell which
    one a binding rests on."""

    @model_validator(mode="after")
    def _requires_a_target(self) -> ObservationBinding:
        if self.segment_id is None and self.junction_id is None:
            msg = (
                f"binding for {self.observation_area_id!r} names neither a segment "
                "nor a junction; a binding to nothing is an unmapped observation "
                "wearing a mapping's clothes"
            )
            raise ValueError(msg)
        if self.approach_id is not None and self.junction_id is None:
            msg = (
                f"binding for {self.observation_area_id!r} names an approach without "
                "the junction it belongs to"
            )
            raise ValueError(msg)
        return self


class ObservationMapping(DomainModel):
    """A versioned set of observation-to-network bindings."""

    mapping_id: Annotated[str, Field(min_length=1, max_length=64)]
    version: Annotated[str, Field(min_length=1, max_length=50)]
    """Pinned into every route result, so a recommendation can be replayed
    against the bindings that produced it."""

    network_id: Annotated[str, Field(min_length=1, max_length=64)]
    """Which network these bindings are expressed against. A binding names
    segment ids, and those mean something different on a different map."""

    network_version: Annotated[str, Field(max_length=50)] | None = None
    """Optional pin. When present it is enforced on load, so a re-surveyed
    network cannot silently keep bindings drawn against the old one."""

    provenance: Provenance
    """Who surveyed these bindings and when. A mapping asserted with no evidence
    is indistinguishable from a guess, and a guess here is exactly the failure
    this module exists to prevent."""

    bindings: tuple[ObservationBinding, ...] = ()

    @model_validator(mode="after")
    def _unique_areas(self) -> ObservationMapping:
        ids = [b.observation_area_id for b in self.bindings]
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        if duplicates:
            msg = (
                f"observation area(s) {duplicates} are bound more than once; two "
                "bindings for one camera would make the segment it affects depend "
                "on iteration order"
            )
            raise ValueError(msg)
        return self

    @property
    def area_ids(self) -> frozenset[str]:
        """Every observation area this mapping binds."""
        return frozenset(b.observation_area_id for b in self.bindings)

    def segment_for(self, observation_area_id: str) -> str | None:
        """The segment an observation area is bound to, if any."""
        for binding in self.bindings:
            if binding.observation_area_id == observation_area_id:
                return binding.segment_id
        return None

    def bindings_by_segment(self) -> dict[str, tuple[ObservationBinding, ...]]:
        """Bindings grouped by the segment they target.

        A segment may legitimately be watched by more than one camera. Grouping
        rather than collapsing keeps that visible; the caller decides how to
        reconcile two verdicts on one road instead of silently taking whichever
        came last.
        """
        grouped: dict[str, list[ObservationBinding]] = {}
        for binding in self.bindings:
            if binding.segment_id is not None:
                grouped.setdefault(binding.segment_id, []).append(binding)
        return {
            key: tuple(sorted(value, key=lambda b: b.observation_area_id))
            for key, value in grouped.items()
        }


class MappingSource(DomainModel):
    """Where a loaded mapping came from, and what exactly was read."""

    relative_path: Annotated[str, Field(min_length=1, max_length=1024)]
    sha256: Annotated[str, Field(pattern="^[0-9a-f]{64}$")]
    size_bytes: Annotated[int, Field(ge=0)]
    loaded_at: dt.datetime
    mapping_id: Annotated[str, Field(min_length=1, max_length=64)]
    version: Annotated[str, Field(min_length=1, max_length=50)]
    binding_count: Annotated[int, Field(ge=0)]


def validate_against_network(mapping: ObservationMapping, network: RoadNetwork) -> None:
    """Reject a mapping that names elements the network does not have.

    Checked rather than trusted, because the failure is silent in the worst
    direction: a binding to a segment that does not exist contributes nothing,
    so the road it *was meant* to describe stays unobserved and picks up an
    uncertainty penalty, while the mapping file looks complete.
    """
    if mapping.network_id != network.network_id:
        msg = (
            f"Mapping {mapping.mapping_id!r} is expressed against network "
            f"{mapping.network_id!r}, but network {network.network_id!r} is loaded. "
            "Segment ids mean different things on different maps."
        )
        raise ValidationFailedError(
            msg, mapping_id=mapping.mapping_id, network_id=network.network_id
        )

    if (
        mapping.network_version is not None
        and mapping.network_version != network.version
    ):
        msg = (
            f"Mapping {mapping.mapping_id!r} pins network version "
            f"{mapping.network_version!r}, but version {network.version!r} is loaded. "
            "Re-survey the bindings or drop the pin deliberately."
        )
        raise ValidationFailedError(msg, mapping_id=mapping.mapping_id)

    segment_ids = {s.segment_id for s in network.segments}
    junction_ids = {j.junction_id for j in network.junctions}
    approach_ids = {
        approach.approach_id for j in network.junctions for approach in j.approaches
    }

    unknown_segments: set[str] = set()
    unknown_junctions: set[str] = set()
    unknown_approaches: set[str] = set()
    for binding in mapping.bindings:
        if binding.segment_id is not None and binding.segment_id not in segment_ids:
            unknown_segments.add(binding.segment_id)
        if binding.junction_id is not None and binding.junction_id not in junction_ids:
            unknown_junctions.add(binding.junction_id)
        if binding.approach_id is not None and binding.approach_id not in approach_ids:
            unknown_approaches.add(binding.approach_id)

    problems = []
    if unknown_segments:
        problems.append(f"segment(s) {sorted(unknown_segments)}")
    if unknown_junctions:
        problems.append(f"junction(s) {sorted(unknown_junctions)}")
    if unknown_approaches:
        problems.append(f"approach(es) {sorted(unknown_approaches)}")
    if problems:
        msg = (
            f"Mapping {mapping.mapping_id!r} references unknown "
            f"{' and '.join(problems)} on network {network.network_id!r}. A binding "
            "to an element that does not exist leaves the road it described "
            "unobserved while the mapping looks complete."
        )
        raise ValidationFailedError(
            msg,
            mapping_id=mapping.mapping_id,
            unknown_segment_ids=sorted(unknown_segments),
            unknown_junction_ids=sorted(unknown_junctions),
        )


def load_mapping(
    path: Path, relative_path: str, *, loaded_at: dt.datetime
) -> tuple[ObservationMapping, MappingSource]:
    """Read and validate an observation mapping file."""
    if not path.exists() or not path.is_file():
        msg = (
            f"No observation mapping at {relative_path!r} inside the networks "
            "directory."
        )
        raise ResourceNotFoundError(msg, relative_path=relative_path)

    size = path.stat().st_size
    if size > MAX_MAPPING_FILE_BYTES:
        msg = (
            f"Mapping file {relative_path!r} is {size} bytes, above the "
            f"{MAX_MAPPING_FILE_BYTES}-byte limit."
        )
        raise ValidationFailedError(msg, relative_path=relative_path)

    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = f"Mapping file {relative_path!r} is not valid UTF-8 JSON: {exc}."
        raise ValidationFailedError(msg, relative_path=relative_path) from exc

    if not isinstance(payload, dict):
        msg = (
            f"Mapping file {relative_path!r} contains a {type(payload).__name__}, "
            "but a mapping must be a JSON object."
        )
        raise ValidationFailedError(msg, relative_path=relative_path)

    try:
        mapping = ObservationMapping.model_validate(payload)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or '(root)'}: {item['msg']}"
            for item in exc.errors()[:6]
        )
        msg = f"Mapping file {relative_path!r} is not a valid mapping. {problems}"
        raise ValidationFailedError(msg, relative_path=relative_path) from exc

    source = MappingSource(
        relative_path=relative_path,
        sha256=digest,
        size_bytes=size,
        loaded_at=loaded_at,
        mapping_id=mapping.mapping_id,
        version=mapping.version,
        binding_count=len(mapping.bindings),
    )
    return mapping, source


class MappingStore:
    """Holds the observation mapping in force for this process.

    Empty by default, and that is a working state rather than a broken one: with
    no mapping every segment is simply unobserved, priced by the uncertainty
    weights, and the result says so. The optimizer never needs a mapping to
    produce a route -- it needs one to use traffic evidence.
    """

    __slots__ = ("_mapping", "_source")

    def __init__(self) -> None:
        self._mapping: ObservationMapping | None = None
        self._source: MappingSource | None = None

    def load(self, mapping: ObservationMapping, source: MappingSource) -> None:
        """Replace the mapping in force."""
        self._mapping = mapping
        self._source = source

    def clear(self) -> None:
        """Drop the mapping. Every segment becomes unobserved."""
        self._mapping = None
        self._source = None

    @property
    def is_loaded(self) -> bool:
        """Whether a mapping is in force."""
        return self._mapping is not None

    @property
    def mapping(self) -> ObservationMapping | None:
        """The mapping in force, or ``None``."""
        return self._mapping

    @property
    def source(self) -> MappingSource | None:
        """Provenance of the loaded mapping."""
        return self._source

    def empty_mapping_reason(self) -> str:
        """Why no observation is being used, phrased for an operator."""
        return (
            "No observation mapping is loaded, so no camera is bound to a road "
            "segment. Traffic evidence exists but cannot be attributed to any "
            "element of the network, and every segment is therefore priced as "
            "unobserved rather than as clear."
        )
