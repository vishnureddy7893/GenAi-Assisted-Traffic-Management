"""Decision outputs, advisories, and the mechanism that keeps GenAI honest.

Everything here is **advisory**. There is no actuation path from any type in
this module to any real signal controller, vehicle or piece of infrastructure,
and none may be added: the capstone scope is decision support over recorded and
simulated data.

The fact table
--------------
Phase 12 hands structured state to a pretrained LLM and asks for a narrative.
The instruction "do not invent numbers" is not a control -- it is a request, and
language models comply with it unevenly, most often failing by producing
plausible round numbers exactly where a reader is least likely to check.

So the numbers are pinned to a :class:`FactTable` built by this system. Every
quantity the narrative is permitted to mention exists there with an id, a value
and a provenance. The generated text must cite fact ids, and
:func:`scan_for_unsupported_numbers` then checks mechanically that no numeric
literal appears in the prose that is not backed by a fact. A validator that runs
after generation is a control; a sentence in a prompt is not.

The scan is a lint, not a proof: it catches invented quantities, not invented
causal claims. Those are constrained separately, by giving the model only
structured state and requiring the reason to reference the cost breakdown that
routing actually produced.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Annotated, Any

from pydantic import Field, model_validator

from app.domain.base import DomainModel
from app.domain.enums import SignalAdvisoryAction
from app.domain.provenance import Provenance, ValueSource
from app.domain.routing import RouteComparison
from app.domain.safety import SafetyConstraint

DecisionId = Annotated[str, Field(min_length=1, max_length=64)]
FactId = Annotated[str, Field(min_length=1, max_length=64)]


# ---------------------------------------------------------------------------
# Advisories
# ---------------------------------------------------------------------------
class SignalAdvisory(DomainModel):
    """A recommended signal action at one junction. Advisory only.

    Deliberately a different type from the observed
    :class:`~app.domain.enums.SignalIndication` carried on
    :class:`~app.domain.state.JunctionState`. One is a fact about the world, the
    other is a proposal for changing it. Sharing a type would make it possible to
    read a recommendation as an observation -- and there is no plausible reading
    of this project's scope in which that confusion is survivable.
    """

    junction_id: Annotated[str, Field(min_length=1, max_length=64)]
    action: SignalAdvisoryAction
    target_movement_ids: frozenset[str] = frozenset()
    conflicting_movement_ids: frozenset[str] = frozenset()
    """Movements that must be held. Stated explicitly so a human operator can
    verify the conflict set rather than trusting it."""

    rationale: Annotated[str, Field(min_length=1, max_length=500)]
    earliest_apply_at: dt.datetime | None = None
    expected_duration_s: Annotated[float, Field(ge=0.0)] | None = None
    pedestrian_clearance_required_s: Annotated[float, Field(ge=0.0)] | None = None
    """Minimum clearance interval before conflicting movements may be cut. Never
    omitted for a junction with an active crossing: truncating a pedestrian
    clearance to speed an emergency vehicle trades one life for another."""

    advisory_only: bool = True

    @model_validator(mode="after")
    def _must_remain_advisory(self) -> SignalAdvisory:
        if not self.advisory_only:
            msg = (
                "SignalAdvisory.advisory_only cannot be False: this system issues "
                "recommendations to human operators and has no control authority "
                "over real infrastructure"
            )
            raise ValueError(msg)
        return self


class CorridorAdvisory(DomainModel):
    """An ordered set of signal advisories forming an emergency green corridor.

    Ordered because a corridor is a sequence in time, not a set: granting
    priority at every junction simultaneously holds cross traffic far longer than
    necessary and can gridlock the surrounding network. Each junction carries its
    own ``earliest_apply_at``, staged against predicted arrival.
    """

    emergency_id: Annotated[str, Field(min_length=1, max_length=64)]
    route_id: Annotated[str, Field(min_length=1, max_length=64)]
    advisories: tuple[SignalAdvisory, ...]
    recompute_if_conditions_change: bool = True
    """A corridor is a prediction about arrival times. If downstream conditions
    change, the staging is wrong and must be recomputed rather than followed."""

    @model_validator(mode="after")
    def _non_empty(self) -> CorridorAdvisory:
        if not self.advisories:
            msg = "a corridor advisory must contain at least one junction advisory"
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Fact table
# ---------------------------------------------------------------------------
class Fact(DomainModel):
    """One citable quantity or statement made available to narrative generation."""

    fact_id: FactId
    label: Annotated[str, Field(min_length=1, max_length=200)]
    value: float | int | str | bool
    unit: Annotated[str, Field(max_length=30)] | None = None
    provenance: Provenance
    display_variants: tuple[str, ...] = ()
    """Pre-rendered forms the narrative may use verbatim -- ``"187 s"``,
    ``"3.1 min"``, ``"about 3 minutes"``. Supplied by this system so the model
    never has to perform a unit conversion. An LLM converting 187 seconds to
    minutes is an arithmetic step performed by a component with no arithmetic
    guarantees, and it is unnecessary: we already know the answer."""

    @property
    def is_factual(self) -> bool:
        """Whether the narrative may state this as fact rather than as a model output."""
        return self.provenance.is_factual


class FactTable(DomainModel):
    """The complete set of quantities a narrative is permitted to mention."""

    facts: tuple[Fact, ...] = ()
    entity_tokens: frozenset[str] = frozenset()
    """Identifiers that legitimately contain digits -- ``J11``, ``Lane 1``,
    ``Route B``. Excluded from the numeric scan so an identifier is not mistaken
    for an invented measurement."""

    @model_validator(mode="after")
    def _unique_ids(self) -> FactTable:
        ids = [f.fact_id for f in self.facts]
        if len(set(ids)) != len(ids):
            msg = "duplicate fact_id in fact table"
            raise ValueError(msg)
        return self

    def fact(self, fact_id: str) -> Fact:
        """Return a fact by id, raising ``KeyError`` if absent."""
        for f in self.facts:
            if f.fact_id == fact_id:
                return f
        raise KeyError(fact_id)

    @property
    def fact_ids(self) -> frozenset[str]:
        """All citable fact ids."""
        return frozenset(f.fact_id for f in self.facts)

    def allowed_numeric_values(self) -> set[float]:
        """Numeric values a narrative may legitimately state.

        Includes numbers appearing inside ``display_variants`` so a pre-rendered
        ``"3.1 min"`` does not trip the scan that exists to catch invented values.
        """
        allowed: set[float] = set()
        for f in self.facts:
            if isinstance(f.value, bool):
                continue
            if isinstance(f.value, int | float):
                allowed.add(float(f.value))
            for variant in f.display_variants:
                allowed.update(_extract_numbers(variant))
        return allowed


#: Numeric literal not glued to a letter or underscore. The lookbehind keeps
#: ``J11`` and ``P1`` out; standalone identifiers such as ``Lane 1`` are handled
#: by stripping :attr:`FactTable.entity_tokens` before scanning.
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?![A-Za-z0-9_])")


def _extract_numbers(text: str) -> set[float]:
    """Extract numeric literals from free text."""
    out: set[float] = set()
    for match in _NUMBER_RE.finditer(text):
        try:
            out.add(float(match.group().replace(",", "")))
        except ValueError:  # pragma: no cover - regex guarantees parseability
            continue
    return out


def scan_for_unsupported_numbers(
    text: str,
    fact_table: FactTable,
    *,
    tolerance: float = 0.05,
) -> tuple[float, ...]:
    """Return numeric literals in ``text`` that no fact supports.

    Used as a post-generation gate on LLM narrative: a non-empty result means the
    model produced a quantity this system never computed, and the narrative must
    be rejected or regenerated rather than shown.

    ``tolerance`` is relative and exists only to absorb display rounding (a fact
    of ``187.4`` narrated as ``187``). It is not a licence to approximate -- the
    intended path is that the model quotes a
    :attr:`Fact.display_variants` entry verbatim.

    Ordinals and small integers (``1``, ``2``, ``3``) are still checked. That
    produces occasional noise on phrases like "the first two junctions", which is
    the correct trade: an over-eager lint is cheap, an invented ETA is not.
    """
    scrubbed = text
    for token in sorted(fact_table.entity_tokens, key=len, reverse=True):
        scrubbed = scrubbed.replace(token, " ")

    allowed = fact_table.allowed_numeric_values()
    unsupported: list[float] = []
    for value in sorted(_extract_numbers(scrubbed)):
        if any(
            abs(value - ok) <= tolerance * max(abs(ok), 1.0) for ok in allowed
        ):
            continue
        unsupported.append(value)
    return tuple(unsupported)


# ---------------------------------------------------------------------------
# GenAI output
# ---------------------------------------------------------------------------
class GenAIRecommendation(DomainModel):
    """Structured narrative produced by the LLM. Phase 12.

    Structured rather than free prose so each part can be validated and rendered
    independently, and so a failure in one part does not discard the rest.
    """

    recommendation: Annotated[str, Field(min_length=1, max_length=2000)]
    reason: Annotated[str, Field(min_length=1, max_length=4000)]
    safety_precautions: tuple[str, ...] = ()
    expected_outcome: Annotated[str, Field(max_length=2000)] | None = None
    potential_risks: tuple[str, ...] = ()
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    """The model's self-reported confidence. Reported as such and never merged
    into the system's own confidence: a language model's stated certainty is not
    calibrated evidence about traffic."""

    cited_fact_ids: frozenset[str] = frozenset()
    model_id: Annotated[str, Field(min_length=1, max_length=200)]
    generated_at: dt.datetime

    @model_validator(mode="after")
    def _tz(self) -> GenAIRecommendation:
        if self.generated_at.tzinfo is None:
            msg = "generated_at must be timezone-aware"
            raise ValueError(msg)
        return self

    def narrative_text(self) -> str:
        """All generated prose concatenated, for validation scanning."""
        parts = [self.recommendation, self.reason, *self.safety_precautions, *self.potential_risks]
        if self.expected_outcome:
            parts.append(self.expected_outcome)
        return "\n".join(parts)


class NarrativeValidation(DomainModel):
    """Result of validating generated narrative against the fact table."""

    unsupported_numbers: tuple[float, ...] = ()
    unknown_fact_ids: tuple[str, ...] = ()
    passed: bool

    @model_validator(mode="after")
    def _consistent(self) -> NarrativeValidation:
        expected = not (self.unsupported_numbers or self.unknown_fact_ids)
        if self.passed != expected:
            msg = "passed flag contradicts the recorded findings"
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Audit record
# ---------------------------------------------------------------------------
class RunManifest(DomainModel):
    """Everything needed to reproduce one decision.

    A recommendation without a manifest is an anecdote. With one, a result can be
    replayed months later against the same inputs and the same policy versions,
    and a behaviour change can be attributed to the component that actually
    caused it rather than guessed at.
    """

    scenario_id: Annotated[str, Field(min_length=1, max_length=64)]
    input_source_id: Annotated[str, Field(min_length=1, max_length=64)]
    input_source_kind: Annotated[str, Field(min_length=1, max_length=50)]
    app_version: Annotated[str, Field(min_length=1, max_length=50)]
    schema_version: Annotated[str, Field(min_length=1, max_length=50)]
    network_version: Annotated[str, Field(min_length=1, max_length=50)]
    safety_rules_version: Annotated[str, Field(min_length=1, max_length=50)]
    route_weights_version: Annotated[str, Field(min_length=1, max_length=50)]
    model_versions: dict[str, str] = Field(default_factory=dict)
    """Every ML artifact that contributed, keyed by role: ``travel_time``,
    ``congestion``, ``queue_growth``. Empty until Phase 10 -- and empty is the
    honest value, not a placeholder version string."""

    random_seed: int | None = None
    created_at: dt.datetime

    @model_validator(mode="after")
    def _tz(self) -> RunManifest:
        if self.created_at.tzinfo is None:
            msg = "created_at must be timezone-aware"
            raise ValueError(msg)
        return self


class DecisionRecord(DomainModel):
    """The complete, auditable output of one decision cycle.

    Assembled in Phase 9. Defined now so every phase built before it targets a
    known output shape rather than inventing its own.
    """

    decision_id: DecisionId
    manifest: RunManifest
    network_state_id: Annotated[str, Field(min_length=1, max_length=64)]
    emergency_id: Annotated[str, Field(max_length=64)] | None = None

    route_comparison: RouteComparison | None = None
    signal_advisories: tuple[SignalAdvisory, ...] = ()
    corridor_advisory: CorridorAdvisory | None = None
    applied_constraints: tuple[SafetyConstraint, ...] = ()

    fact_table: FactTable = FactTable()
    genai: GenAIRecommendation | None = None
    narrative_validation: NarrativeValidation | None = None

    evidence_degraded: bool = False
    degradation_reasons: tuple[str, ...] = ()
    created_at: dt.datetime

    @model_validator(mode="after")
    def _validate(self) -> DecisionRecord:
        if self.created_at.tzinfo is None:
            msg = "created_at must be timezone-aware"
            raise ValueError(msg)
        if self.genai is not None and self.narrative_validation is None:
            msg = (
                "a decision carrying generated narrative must also carry its "
                "validation result; unvalidated model prose must never reach an "
                "operator"
            )
            raise ValueError(msg)
        if self.evidence_degraded and not self.degradation_reasons:
            msg = "evidence_degraded requires at least one stated reason"
            raise ValueError(msg)
        return self

    def validate_narrative(self) -> NarrativeValidation:
        """Check generated prose against the fact table.

        Verifies both directions: that every cited fact id exists, and that every
        numeric literal in the prose is backed by a fact.
        """
        if self.genai is None:
            return NarrativeValidation(passed=True)
        unknown = tuple(sorted(self.genai.cited_fact_ids - self.fact_table.fact_ids))
        unsupported = scan_for_unsupported_numbers(
            self.genai.narrative_text(), self.fact_table
        )
        return NarrativeValidation(
            unsupported_numbers=unsupported,
            unknown_fact_ids=unknown,
            passed=not (unsupported or unknown),
        )

    @property
    def has_predicted_content(self) -> bool:
        """Whether any cited fact is a model prediction rather than an observation.

        Drives presentation: predicted quantities must be shown with their
        modality intact, never rendered identically to measured ones.
        """
        return any(
            f.provenance.source in (ValueSource.PREDICTED, ValueSource.SIMULATED)
            for f in self.fact_table.facts
        )


def build_fact(
    *,
    fact_id: str,
    label: str,
    value: Any,
    provenance: Provenance,
    unit: str | None = None,
    display_variants: tuple[str, ...] = (),
) -> Fact:
    """Construct a :class:`Fact`. Thin helper kept for call-site readability."""
    return Fact(
        fact_id=fact_id,
        label=label,
        value=value,
        unit=unit,
        provenance=provenance,
        display_variants=display_variants,
    )
