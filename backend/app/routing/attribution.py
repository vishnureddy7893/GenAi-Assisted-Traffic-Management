"""Which road each camera's evidence belongs to, and which has no road at all.

The question this answers
-------------------------
"Does the right camera's evidence land on the right road?" -- and nothing else.
Not whether that evidence should change a route cost, which is a separate
question with separate consequences and is deliberately not answered here.

Separating them is the point. Attribution can be checked by a person who knows
the junction: they look at the camera, look at the segment it resolved to, and
say yes or no. Costing cannot be checked that way, and wiring both at once would
mean a wrong binding and a wrong penalty arriving together, with no way to tell
which produced the surprising route.

Only an explicit mapping attributes anything
--------------------------------------------
Evidence reaches a segment through the
:class:`~app.routing.effective.EffectiveObservationMapping` and through nothing
else. There is no proximity rule here, no bearing rule, no fallback to the
nearest road and no inference from image coordinates or detection boxes. An
observation area with no effective binding resolves to ``segment_id = None`` and
is named as unmapped.

That is the rule :mod:`app.routing.mapping` has enforced since Phase 7, applied
one layer further out: a congestion band attached to the wrong road is a
specific false claim that a route would then be planned around, and the only
defence against it is that a person said so.

Unmapped is not one thing
-------------------------
Four outcomes, because they call for four different actions:

* **attributed** -- a directed segment is in force, evidence lands on it;
* **junction only** -- an operator confirmed the intersection but not which
  arm, so the evidence belongs to a junction and to no direction of travel;
* **excluded** -- a confirmation exists and is not in force, because it was made
  against another network or another version of this one;
* **unmapped** -- nothing binds this area at all.

Collapsing them into "mapped / unmapped" would tell an operator that four
different problems are the same problem. Only the first attributes evidence to a
road; the other three leave ``segment_id`` null and say why.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field, model_validator

from app.domain.base import DomainModel
from app.domain.enums import CongestionLevel, FreshnessStatus
from app.domain.network import GeoPoint
from app.domain.provenance import Provenance
from app.routing.effective import (
    BindingResolution,
    BindingSource,
    EffectiveBinding,
    EffectiveObservationMapping,
    ExclusionReason,
)
from app.state.schemas import NetworkTrafficState


class AttributionStatus(StrEnum):
    """Whether one observation area's evidence reached a road, and if not why."""

    ATTRIBUTED = "attributed"
    """A directed segment is in force for this area."""

    JUNCTION_ONLY = "junction_only"
    """An operator confirmed the intersection but not which arm. The evidence
    belongs to a junction and to no direction of travel, so it attributes to no
    segment -- which is correct, not a gap to fill by guessing."""

    EXCLUDED = "excluded"
    """A confirmation exists and is not in force: made against another network,
    or another version of this one. Reported, never migrated."""

    UNMAPPED = "unmapped"
    """Nothing binds this area. Its evidence is real and unattributable."""


class EvidenceAttribution(DomainModel):
    """One observation area's evidence, and the road it reaches -- or does not."""

    observation_area_id: Annotated[str, Field(min_length=1, max_length=120)]
    """Also the site key: a site is recorded under the observation area it
    locates, so the camera, the area and the site are one identifier rather than
    three that could disagree."""

    status: AttributionStatus

    segment_id: Annotated[str, Field(max_length=64)] | None = None
    """``None`` for everything except ``attributed``. Never filled from
    proximity, bearing or any other inference."""

    junction_id: Annotated[str, Field(max_length=64)] | None = None
    approach_id: Annotated[str, Field(max_length=64)] | None = None
    segment_direction: Annotated[str, Field(max_length=20)] | None = None

    site_location: GeoPoint | None = None
    """Where the camera was when its binding was confirmed, when a confirmation
    is behind this attribution. Keeps the chain camera -> area -> segment
    traceable back to a coordinate."""

    network_id: Annotated[str, Field(min_length=1, max_length=64)]
    network_version: Annotated[str, Field(min_length=1, max_length=50)]
    """The loaded network this attribution was resolved against."""

    source: BindingSource | None = None
    resolution: BindingResolution | None = None
    """How the binding came to be in force. Kept distinct rather than flattened
    into "mapped": a surveyed file binding, a confirmation, and two sources
    agreeing are different strengths of evidence about the same road."""

    exclusion_reason: ExclusionReason | None = None
    provenance: Provenance | None = None
    """The binding's own provenance, carried through unchanged."""

    congestion_level: CongestionLevel | None = None
    freshness: FreshnessStatus | None = None
    vehicle_count: Annotated[int, Field(ge=0)] | None = None
    """The evidence being attributed. Read off the interpreted state; nothing
    here recomputes or reinterprets it."""

    reason: Annotated[str, Field(min_length=1, max_length=400)]
    """Why this area is where it is. Always present, including when attribution
    succeeded, so a reviewer never has to infer the rule from the outcome."""

    @model_validator(mode="after")
    def _only_attributed_names_a_segment(self) -> EvidenceAttribution:
        if self.status is AttributionStatus.ATTRIBUTED and self.segment_id is None:
            msg = (
                "an attributed observation area must name the segment its "
                "evidence reached"
            )
            raise ValueError(msg)
        if self.status is not AttributionStatus.ATTRIBUTED and self.segment_id:
            msg = (
                f"observation area {self.observation_area_id!r} is "
                f"{self.status.value} but names segment {self.segment_id!r}; only "
                "an attributed area may name a road, or a null attribution would "
                "read as a binding"
            )
            raise ValueError(msg)
        return self


class EvidenceAttributionReport(DomainModel):
    """Where every interpreted observation area's evidence landed."""

    network_id: Annotated[str, Field(min_length=1, max_length=64)]
    network_version: Annotated[str, Field(min_length=1, max_length=50)]

    effective_mapping_hash: Annotated[str, Field(pattern="^[0-9a-f]{64}$")]
    effective_mapping_version: Annotated[str, Field(min_length=1, max_length=50)]
    """Identity of the composition this attribution was resolved against, so
    "which mapping put this camera on this road?" is answerable from the
    response alone."""

    attributions: tuple[EvidenceAttribution, ...] = ()
    """One entry per interpreted observation area, sorted by area id. Exactly
    one: an area attributed twice would let the same evidence count twice."""

    attributed_area_ids: tuple[str, ...] = ()
    unmapped_observation_area_ids: tuple[str, ...] = ()
    """Areas whose evidence reached no segment, under the same name the route
    optimizer already uses for the same condition. Includes junction-only and
    excluded areas: all three have evidence that cannot be attributed to a
    directed road, which is what the name means."""

    excluded_area_ids: tuple[str, ...] = ()
    junction_only_area_ids: tuple[str, ...] = ()
    """The finer breakdown, because the three call for different fixes."""

    attributed_segment_ids: tuple[str, ...] = ()
    """Distinct segments receiving evidence. Two cameras may watch one segment;
    that is reconciled by whoever consumes the evidence, not collapsed here."""

    used_by_route_costing: bool = False
    """Fixed. Attribution is reported; the optimizer is unchanged and still
    reads the loaded mapping file alone."""

    note: str = ""

    @model_validator(mode="after")
    def _one_attribution_per_area_and_no_costing_claim(
        self,
    ) -> EvidenceAttributionReport:
        ids = [item.observation_area_id for item in self.attributions]
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        if duplicates:
            msg = (
                f"observation area(s) {duplicates} are attributed more than once; "
                "the same evidence would then reach a road twice"
            )
            raise ValueError(msg)
        if self.used_by_route_costing:
            msg = (
                "used_by_route_costing cannot be true: this report describes "
                "where evidence would land, and the optimizer has not been "
                "changed to use it"
            )
            raise ValueError(msg)
        return self

    @property
    def attributed_count(self) -> int:
        """Observation areas whose evidence reached a directed segment."""
        return len(self.attributed_area_ids)

    def attribution_for(self, observation_area_id: str) -> EvidenceAttribution | None:
        """One area's attribution, or ``None`` when it was not interpreted."""
        for item in self.attributions:
            if item.observation_area_id == observation_area_id:
                return item
        return None


def attribute_evidence(
    snapshot: NetworkTrafficState,
    effective: EffectiveObservationMapping,
) -> EvidenceAttributionReport:
    """Resolve each interpreted observation area to its effective segment.

    Pure. Reads the snapshot and the composition, writes nothing, and resolves
    every area through the effective mapping alone -- there is no path in this
    function from a coordinate, a bearing or a detection to a segment id.
    """
    excluded_by_area = {item.observation_area_id: item for item in effective.excluded}

    attributions: list[EvidenceAttribution] = []
    for element in snapshot.elements:
        state = element.state
        area_id = state.observation_area_id
        common: dict[str, object] = {
            "observation_area_id": area_id,
            "network_id": effective.network_id,
            "network_version": effective.network_version,
            "congestion_level": state.congestion.level,
            "freshness": element.freshness,
            "vehicle_count": state.vehicle_count,
        }

        binding = effective.binding_for(area_id)
        if binding is not None:
            attributions.append(
                _from_binding(binding, common, effective.effective_mapping_version)
            )
            continue

        exclusion = excluded_by_area.get(area_id)
        if exclusion is not None:
            attributions.append(
                EvidenceAttribution.model_validate(
                    common
                    | {
                        "status": AttributionStatus.EXCLUDED,
                        "junction_id": exclusion.junction_id,
                        "exclusion_reason": exclusion.reason,
                        "reason": (
                            f"A confirmed binding exists but is not in force: "
                            f"{exclusion.detail} Evidence from this area is not "
                            "attributed to any road."
                        ),
                    }
                )
            )
            continue

        attributions.append(
            EvidenceAttribution.model_validate(
                common
                | {
                    "status": AttributionStatus.UNMAPPED,
                    "reason": (
                        "No effective binding exists for this observation area, "
                        "so its evidence is real and unattributable. It is not "
                        "assigned to the nearest road: which road a camera "
                        "watches is established by a survey, never by proximity."
                    ),
                }
            )
        )

    ordered = tuple(sorted(attributions, key=lambda item: item.observation_area_id))
    attributed = tuple(
        item.observation_area_id
        for item in ordered
        if item.status is AttributionStatus.ATTRIBUTED
    )
    junction_only = tuple(
        item.observation_area_id
        for item in ordered
        if item.status is AttributionStatus.JUNCTION_ONLY
    )
    excluded = tuple(
        item.observation_area_id
        for item in ordered
        if item.status is AttributionStatus.EXCLUDED
    )
    unmapped = tuple(
        item.observation_area_id
        for item in ordered
        if item.status is not AttributionStatus.ATTRIBUTED
    )
    segments = tuple(
        sorted({item.segment_id for item in ordered if item.segment_id is not None})
    )

    return EvidenceAttributionReport(
        network_id=effective.network_id,
        network_version=effective.network_version,
        effective_mapping_hash=effective.effective_mapping_hash,
        effective_mapping_version=effective.effective_mapping_version,
        attributions=ordered,
        attributed_area_ids=attributed,
        unmapped_observation_area_ids=unmapped,
        excluded_area_ids=excluded,
        junction_only_area_ids=junction_only,
        attributed_segment_ids=segments,
        note=_note(ordered, attributed, unmapped, segments),
    )


def _from_binding(
    entry: EffectiveBinding,
    common: dict[str, object],
    mapping_version: str,
) -> EvidenceAttribution:
    """Build an attribution from an effective binding, preserving its provenance."""
    rendered = entry.binding
    confirmed = entry.confirmed

    shared = common | {
        "junction_id": rendered.junction_id,
        "approach_id": rendered.approach_id,
        "source": entry.source,
        "resolution": entry.resolution,
        # The confirmation's own provenance, carried through rather than
        # restated: a file binding has none of its own here and says so with
        # None, which is different from claiming an unknown source.
        "provenance": confirmed.provenance if confirmed is not None else None,
        "site_location": confirmed.site_location if confirmed is not None else None,
        "segment_direction": (
            confirmed.segment_direction.value
            if confirmed is not None and confirmed.segment_direction is not None
            else None
        ),
    }

    if rendered.segment_id is None:
        return EvidenceAttribution.model_validate(
            shared
            | {
                "status": AttributionStatus.JUNCTION_ONLY,
                "reason": (
                    f"Bound to junction {rendered.junction_id} but to no directed "
                    "segment, so the evidence belongs to an intersection and to "
                    "no direction of travel. Confirm a segment to attribute it "
                    "to a road."
                ),
            }
        )

    return EvidenceAttribution.model_validate(
        shared
        | {
            "status": AttributionStatus.ATTRIBUTED,
            "segment_id": rendered.segment_id,
            "reason": (
                f"Evidence is attributed to segment {rendered.segment_id} by the "
                f"effective mapping ({entry.resolution.value}, mapping version "
                f"{mapping_version})."
            ),
        }
    )


def _note(
    attributions: tuple[EvidenceAttribution, ...],
    attributed: tuple[str, ...],
    unmapped: tuple[str, ...],
    segments: tuple[str, ...],
) -> str:
    """One paragraph an operator can read instead of the structure."""
    if not attributions:
        return (
            "No observation area has been interpreted, so there is no evidence "
            "to attribute. This describes our records, not the roads."
        )
    base = (
        f"{len(attributed)} of {len(attributions)} interpreted observation "
        f"area(s) reach a directed segment, across {len(segments)} segment(s)."
    )
    if unmapped:
        base += (
            f" {len(unmapped)} area(s) have evidence that reaches no road and "
            "are listed as unmapped; none is assigned to a nearest segment."
        )
    return (
        base
        + " Attribution is reported only: route costs are unchanged, and the "
        "optimizer still reads the loaded mapping file alone."
    )
