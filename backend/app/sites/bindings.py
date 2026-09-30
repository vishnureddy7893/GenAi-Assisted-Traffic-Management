"""Confirmed site-to-junction bindings, and how they reach the mapping layer.

A confirmed binding is a human decision
---------------------------------------
Nothing in this module computes which junction a camera watches. The nearest
junction is computed in :mod:`app.sites.candidates` and offered; this module
stores what an operator did about it. The distinction is the point: the
candidate carries ``ValueSource.DERIVED`` because it falls out of a distance
sort, and a confirmed binding carries ``ValueSource.OBSERVED`` naming the person
who decided. Labelling the decision derived would credit the geometry with a
choice it did not make.

Why it persists
---------------
For the same reason a surveyed site does. A confirmation is somebody looking at
a camera's coordinates and a map and saying "yes, that one". Losing it on
restart would mean asking them again, and a check that is tedious to repeat is a
check that gets rubber-stamped. It uses the same
:class:`~app.sites.persistence.JsonDocumentFile` as the site registry, so both
behave identically when the file is missing or unreadable.

A segment confirmation refines; it does not replace
---------------------------------------------------
Confirming a directed segment writes back to the *same*
:class:`~app.domain.site.ConfirmedSiteBinding`, adding ``segment_id``,
``segment_direction`` and a second ``segment_confirmation``. The junction
decision and whoever made it survive untouched. They are two acts, often by
different people on different days, and collapsing them would erase the record
of who placed the camera at this intersection in the first place.

How it integrates with ObservationMapping
-----------------------------------------
A confirmed binding renders to an ordinary
:class:`~app.routing.mapping.ObservationBinding`. Before a segment is confirmed
it carries ``junction_id`` alone -- already a supported form, meaning "this
camera covers the intersection rather than one approach". After a segment is
confirmed it carries ``segment_id`` as well, which is the form the optimizer's
costing understands.

Why a segment binding still changes no route
--------------------------------------------
It is not connected to the optimizer, deliberately, and the reason is specific
rather than cautious. :class:`~app.routing.engine.RouteOptimizer` is handed the
*one* ``ObservationMapping`` held by :class:`~app.routing.mapping.MappingStore`,
and pins that mapping's ``mapping_id`` and ``version`` into every route result
so a recommendation can be replayed against the bindings that produced it.

This registry is neither versioned nor immutable: it changes the moment an
operator confirms anything. Feeding it into the optimizer would leave every
route result pinned to a mapping identifier that no longer describes the
bindings actually used -- the replay guarantee would still be printed and would
silently be false. On top of that, ``ObservationMapping`` forbids one
observation area being bound twice, and a file mapping plus a confirmation for
the same area would need a precedence rule nobody has decided.

So the association is recorded and reported, and route costs are unchanged.
Closing that gap means deciding how a confirmed set is versioned and how it
composes with a loaded file, which is a decision worth making on its own rather
than as a side effect of this step.

Bindings are per network
------------------------
A junction id means something different on a different map, so every record
names the network it was confirmed against and is only reported as in force
while that network is loaded. Confirmations for other networks are retained, not
deleted -- switching maps and switching back must not lose them.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Annotated

from pydantic import Field

from app.core.errors import DataSourceError, ResourceNotFoundError, ValidationFailedError
from app.core.logging import get_logger
from app.domain.base import DomainModel
from app.domain.network import RoadNetwork, RoadSegment
from app.domain.site import ConfirmedSiteBinding, SegmentDirection
from app.routing.mapping import ObservationBinding
from app.sites.persistence import JsonDocumentFile

logger = get_logger(__name__)

#: Version of the on-disk document. Bumped only when the file layout changes in
#: a way an older reader could misinterpret.
BINDINGS_FILE_SCHEMA_VERSION = "1"


class BindingRegistryDocument(DomainModel):
    """The whole confirmed-binding registry, as stored on disk."""

    schema_version: Annotated[str, Field(min_length=1, max_length=20)]
    bindings: tuple[ConfirmedSiteBinding, ...] = ()


def binding_for(confirmed: ConfirmedSiteBinding) -> ObservationBinding:
    """Render a confirmation as the mapping layer's own binding type.

    Deliberately a function here rather than a method on the domain model: the
    domain layer must not import the routing layer, and the conversion is a
    detail of how confirmations reach routing rather than a property of the
    confirmation itself.
    """
    return ObservationBinding(
        observation_area_id=confirmed.observation_area_id,
        junction_id=confirmed.junction_id,
        segment_id=confirmed.segment_id,
        approach_id=confirmed.approach_id,
        coverage_note=confirmed.coverage_note,
        confirmation=confirmed.segment_confirmation or confirmed.confirmation,
    )


def validate_junction_exists(junction_id: str, network: RoadNetwork) -> None:
    """Reject a confirmation naming a junction the loaded network does not have.

    Checked rather than trusted, because the failure is silent in the worst
    direction: a binding to a junction that does not exist would sit in the
    registry looking confirmed while describing nothing at all.
    """
    if any(junction.junction_id == junction_id for junction in network.junctions):
        return
    msg = (
        f"Junction {junction_id!r} does not exist on network "
        f"{network.network_id!r} (version {network.version}). A binding to a "
        "junction the map does not have would look confirmed while describing "
        "nothing."
    )
    raise ValidationFailedError(
        msg, junction_id=junction_id, network_id=network.network_id
    )


def validate_segment_incident(
    segment_id: str, junction_id: str, network: RoadNetwork
) -> tuple[RoadSegment, SegmentDirection]:
    """Check a segment exists and touches the junction, and say which way it runs.

    Both halves matter and they fail differently. A segment that does not exist
    is a typo. A segment that exists but does not touch this junction is worse:
    it is a real road somewhere else on the map, so the binding would look
    entirely plausible in every listing while attributing a camera's queue to a
    street it cannot see.

    The direction comes back rather than being recomputed by the caller, so the
    record of "incoming at J1" is established once, here, from the topology.
    """
    segment = next(
        (item for item in network.segments if item.segment_id == segment_id), None
    )
    if segment is None:
        msg = (
            f"Segment {segment_id!r} does not exist on network "
            f"{network.network_id!r} (version {network.version})."
        )
        raise ValidationFailedError(
            msg, segment_id=segment_id, network_id=network.network_id
        )

    if segment.to_junction_id == junction_id:
        return segment, SegmentDirection.INCOMING
    if segment.from_junction_id == junction_id:
        return segment, SegmentDirection.OUTGOING

    msg = (
        f"Segment {segment_id!r} does not touch junction {junction_id!r}: it runs "
        f"from {segment.from_junction_id!r} to {segment.to_junction_id!r}. A "
        "binding to a road the camera's junction does not meet would look "
        "plausible everywhere while describing a street the camera cannot see."
    )
    raise ValidationFailedError(
        msg,
        segment_id=segment_id,
        junction_id=junction_id,
        from_junction_id=segment.from_junction_id,
        to_junction_id=segment.to_junction_id,
    )


def validate_approach(
    approach_id: str, segment_id: str, junction_id: str, network: RoadNetwork
) -> None:
    """Check an approach belongs to this junction and is fed by this segment."""
    junction = next(
        (item for item in network.junctions if item.junction_id == junction_id), None
    )
    approaches = junction.approaches if junction is not None else ()
    approach = next(
        (item for item in approaches if item.approach_id == approach_id), None
    )
    if approach is None:
        msg = (
            f"Approach {approach_id!r} is not an approach of junction "
            f"{junction_id!r} on network {network.network_id!r}."
        )
        raise ValidationFailedError(
            msg, approach_id=approach_id, junction_id=junction_id
        )
    if approach.incoming_segment_id != segment_id:
        msg = (
            f"Approach {approach_id!r} carries traffic from segment "
            f"{approach.incoming_segment_id!r}, not {segment_id!r}. An approach is "
            "one direction of one road, so the two cannot disagree."
        )
        raise ValidationFailedError(
            msg,
            approach_id=approach_id,
            segment_id=segment_id,
            approach_segment_id=approach.incoming_segment_id,
        )


def validate_network_version(
    confirmed: ConfirmedSiteBinding, network: RoadNetwork
) -> None:
    """Refuse to refine a junction binding made against a different map version.

    Stricter than the junction level, where a version mismatch is reported and
    left to the operator. The difference is what a re-survey changes: junction
    ids usually survive it, while segment ids, directions and which arms meet
    which node are exactly what gets redrawn. Refining a stale junction decision
    would attach a direction that was chosen by looking at a different map.

    Re-confirming the junction against the loaded network clears it.
    """
    if confirmed.network_version is None:
        return
    if confirmed.network_version == network.version:
        return
    msg = (
        f"The junction binding for {confirmed.observation_area_id!r} was confirmed "
        f"against network version {confirmed.network_version!r}, but version "
        f"{network.version!r} is loaded. Segment ids and directions are what a "
        "re-survey redraws, so confirm the junction again against the loaded "
        "network before refining it to a segment."
    )
    raise ValidationFailedError(
        msg,
        observation_area_id=confirmed.observation_area_id,
        confirmed_network_version=confirmed.network_version,
        loaded_network_version=network.version,
    )


class ConfirmedBindingRegistry:
    """Human-confirmed site-to-junction bindings, persisted to a JSON file.

    Keyed by ``(network_id, observation_area_id)``: one camera has one confirmed
    junction per map. Re-confirming replaces the earlier decision, because a
    correction has to be able to win -- and the replaced record's confirmation
    time is *not* carried over, since a new decision was genuinely made.
    """

    __slots__ = ("_bindings", "_file", "_load_error")

    def __init__(self, path: Path) -> None:
        self._file: JsonDocumentFile[BindingRegistryDocument] = JsonDocumentFile(
            path,
            document_type=BindingRegistryDocument,
            schema_version=BINDINGS_FILE_SCHEMA_VERSION,
            description="confirmed binding registry",
        )
        self._bindings: dict[tuple[str, str], ConfirmedSiteBinding] = {}
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
        """Whether the stored file exists but could not be read."""
        return self._load_error is not None

    def __len__(self) -> int:
        """Number of confirmed bindings held, across every network."""
        return len(self._bindings)

    def _require_readable(self) -> None:
        """Refuse to answer while the stored file is unreadable."""
        if self._load_error is None:
            return
        raise self._file.unreadable_error(self._load_error)

    # --- reading ----------------------------------------------------------
    def get(self, network_id: str, observation_area_id: str) -> ConfirmedSiteBinding:
        """Return one confirmed binding, or raise a 404."""
        self._require_readable()
        found = self._bindings.get((network_id, observation_area_id))
        if found is None:
            msg = (
                f"No binding has been confirmed for observation area "
                f"{observation_area_id!r} on network {network_id!r}."
            )
            raise ResourceNotFoundError(
                msg,
                observation_area_id=observation_area_id,
                network_id=network_id,
            )
        return found

    def find(
        self, network_id: str, observation_area_id: str
    ) -> ConfirmedSiteBinding | None:
        """Return one confirmed binding, or ``None`` when there is none."""
        self._require_readable()
        return self._bindings.get((network_id, observation_area_id))

    def list(self, network_id: str | None = None) -> tuple[ConfirmedSiteBinding, ...]:
        """Confirmed bindings, most recently confirmed first.

        ``network_id`` filters to the bindings in force on one map. Passing
        ``None`` returns every record, including confirmations made against maps
        that are not currently loaded.
        """
        self._require_readable()
        items = [
            binding
            for binding in self._bindings.values()
            if network_id is None or binding.network_id == network_id
        ]
        return tuple(
            sorted(
                items,
                key=lambda b: (b.confirmation.confirmed_at, b.observation_area_id),
                reverse=True,
            )
        )

    def bindings_for(self, network: RoadNetwork) -> tuple[ObservationBinding, ...]:
        """Mapping-layer bindings for every confirmation on this network."""
        return tuple(
            binding_for(confirmed)
            for confirmed in sorted(
                self.list(network.network_id),
                key=lambda b: b.observation_area_id,
            )
        )

    def stale_version_ids(self, network: RoadNetwork) -> tuple[str, ...]:
        """Areas confirmed against a different version of the loaded network.

        Reported rather than invalidated. A re-surveyed map usually keeps its
        junction ids, so the binding is probably still right -- but "probably"
        is the operator's call to make, not this module's.
        """
        return tuple(
            sorted(
                binding.observation_area_id
                for binding in self.list(network.network_id)
                if binding.network_version is not None
                and binding.network_version != network.version
            )
        )

    # --- writing ----------------------------------------------------------
    def confirm(self, binding: ConfirmedSiteBinding) -> tuple[ConfirmedSiteBinding, bool]:
        """Record a confirmation, returning it and whether it replaced one."""
        self._require_readable()
        key = (binding.network_id, binding.observation_area_id)
        existing = self._bindings.get(key)

        previous = dict(self._bindings)
        self._bindings[key] = binding
        try:
            self._write()
        except DataSourceError:
            # The in-memory view must not survive a failed write, or a caller
            # would be told a binding was confirmed while the file says otherwise.
            self._bindings = previous
            raise

        logger.info(
            "site binding confirmed",
            extra={
                "observation_area_id": binding.observation_area_id,
                "junction_id": binding.junction_id,
                "network_id": binding.network_id,
                "method": binding.confirmation.method.value,
                "replaced": existing is not None,
            },
        )
        return binding, existing is not None

    def refine(self, binding: ConfirmedSiteBinding) -> ConfirmedSiteBinding:
        """Replace an existing record with a segment-refined version of it.

        Requires the junction binding to already exist. A segment confirmation
        is a refinement of a decision somebody made about a junction, and one
        arriving with no junction decision behind it would be an unrelated
        binding wearing a refinement's clothes.
        """
        self._require_readable()
        key = (binding.network_id, binding.observation_area_id)
        if key not in self._bindings:
            msg = (
                f"No junction binding has been confirmed for observation area "
                f"{binding.observation_area_id!r} on network "
                f"{binding.network_id!r}, so there is nothing to refine. Confirm "
                "the junction first."
            )
            raise ResourceNotFoundError(
                msg,
                observation_area_id=binding.observation_area_id,
                network_id=binding.network_id,
            )

        previous = dict(self._bindings)
        self._bindings[key] = binding
        try:
            self._write()
        except DataSourceError:
            self._bindings = previous
            raise

        logger.info(
            "site segment binding confirmed",
            extra={
                "observation_area_id": binding.observation_area_id,
                "junction_id": binding.junction_id,
                "segment_id": binding.segment_id,
                "direction": (
                    binding.segment_direction.value
                    if binding.segment_direction
                    else None
                ),
                "network_id": binding.network_id,
            },
        )
        return binding

    def segment_bound(self, network_id: str) -> tuple[ConfirmedSiteBinding, ...]:
        """Bindings refined to a directed segment, on one network."""
        return tuple(
            item for item in self.list(network_id) if item.is_segment_bound
        )

    def revoke(self, network_id: str, observation_area_id: str) -> ConfirmedSiteBinding:
        """Remove a confirmation, returning what was removed.

        A binding an operator no longer stands behind must be removable, or the
        only way to correct a mistake would be to assert a different one.
        """
        removed = self.get(network_id, observation_area_id)
        previous = dict(self._bindings)
        del self._bindings[(network_id, observation_area_id)]
        try:
            self._write()
        except DataSourceError:
            self._bindings = previous
            raise
        logger.info(
            "site binding revoked",
            extra={
                "observation_area_id": observation_area_id,
                "network_id": network_id,
            },
        )
        return removed

    # --- persistence ------------------------------------------------------
    def _read(self) -> None:
        """Load the stored document, recording why if it will not load."""
        document, error = self._file.read()
        self._load_error = error
        if document is not None:
            self._bindings = {
                (binding.network_id, binding.observation_area_id): binding
                for binding in document.bindings
            }

    def _write(self) -> None:
        """Serialise the whole registry over the stored file, atomically."""
        self._file.write(
            BindingRegistryDocument(
                schema_version=BINDINGS_FILE_SCHEMA_VERSION,
                bindings=tuple(
                    sorted(
                        self._bindings.values(),
                        key=lambda b: (b.network_id, b.observation_area_id),
                    )
                ),
            )
        )


def utc_now() -> dt.datetime:
    """Current wall-clock instant, for stamping a confirmation's time."""
    return dt.datetime.now(tz=dt.UTC)
