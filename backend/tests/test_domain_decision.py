"""Routing cost structure, advisory constraints, and the GenAI fact-table gate."""

from __future__ import annotations

import datetime as dt

import pytest
from domain_fixtures import T0, measure_float
from pydantic import ValidationError

from app.domain.decision import (
    CorridorAdvisory,
    DecisionRecord,
    Fact,
    FactTable,
    GenAIRecommendation,
    NarrativeValidation,
    RunManifest,
    SignalAdvisory,
    scan_for_unsupported_numbers,
)
from app.domain.enums import SignalAdvisoryAction
from app.domain.provenance import Unit, ValueSource, make_provenance
from app.domain.routing import RouteCandidate, RouteComparison, RouteCostBreakdown, RouteLeg

PROV = make_provenance(
    source=ValueSource.OBSERVED, producer="test", producer_version="1", observed_at=T0
)
PREDICTED = make_provenance(
    source=ValueSource.PREDICTED,
    producer="model.travel_time",
    producer_version="0.3.1",
    observed_at=T0,
    confidence=0.72,
)


def _leg(segment: str, frm: str, to: str, seconds: float) -> RouteLeg:
    return RouteLeg(
        segment_id=segment,
        from_junction_id=frm,
        to_junction_id=to,
        length_m=measure_float(500.0, Unit.METER, source=ValueSource.STATIC),
        traversal_time_s=measure_float(seconds, Unit.SECOND),
    )


def _route(route_id: str, legs: tuple[RouteLeg, ...], cost: RouteCostBreakdown) -> RouteCandidate:
    return RouteCandidate(
        route_id=route_id, legs=legs, cost=cost, computed_at=T0, provenance=PROV
    )


class TestRouteCost:
    def test_total_is_a_risk_adjusted_time_in_seconds(self) -> None:
        """Every term is in equivalent seconds, so the total is interpretable."""
        cost = RouteCostBreakdown(
            base_travel_time_s=310.0,
            congestion_penalty_s=40.0,
            safety_penalty_s=60.0,
            incident_penalty_s=42.0,
            weights_version="0.1.0",
        )
        assert cost.total_s == pytest.approx(452.0)

    def test_penalty_share_exposes_policy_dominated_rankings(self) -> None:
        """A high share means the ranking rests on weights, not measured traffic."""
        cost = RouteCostBreakdown(
            base_travel_time_s=100.0, safety_penalty_s=300.0, weights_version="0.1.0"
        )
        assert cost.penalty_share == pytest.approx(0.75)

    def test_negative_penalties_are_rejected(self) -> None:
        """A negative penalty would be a reward for being unsafe."""
        with pytest.raises(ValidationError):
            RouteCostBreakdown(
                base_travel_time_s=10.0, safety_penalty_s=-5.0, weights_version="0.1.0"
            )

    def test_weights_version_is_mandatory(self) -> None:
        """Two runs with different weights are not comparable."""
        with pytest.raises(ValidationError):
            RouteCostBreakdown(base_travel_time_s=10.0)  # type: ignore[call-arg]


class TestRouteCandidate:
    def test_a_discontinuous_route_is_rejected(self) -> None:
        cost = RouteCostBreakdown(base_travel_time_s=10.0, weights_version="0.1.0")
        with pytest.raises(ValidationError, match="discontinuous"):
            _route(
                "R1",
                (_leg("S1", "J11", "J12", 60.0), _leg("S2", "J14", "J18", 60.0)),
                cost,
            )

    def test_junction_sequence_reads_as_a_corridor(self) -> None:
        cost = RouteCostBreakdown(base_travel_time_s=120.0, weights_version="0.1.0")
        route = _route(
            "R1",
            (_leg("S1", "J11", "J14", 60.0), _leg("S2", "J14", "J18", 60.0)),
            cost,
        )
        assert route.junction_sequence == ("J11", "J14", "J18")
        assert route.origin_junction_id == "J11"
        assert route.destination_junction_id == "J18"

    def test_an_empty_route_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least one leg"):
            _route("R1", (), RouteCostBreakdown(base_travel_time_s=0.0, weights_version="1"))


class TestRouteComparison:
    def _comparison(self) -> RouteComparison:
        fast = _route(
            "R-A",
            (_leg("S1", "J11", "J14", 150.0),),
            RouteCostBreakdown(base_travel_time_s=150.0, weights_version="0.1.0"),
        )
        slow = _route(
            "R-B",
            (_leg("S2", "J11", "J12", 300.0),),
            RouteCostBreakdown(base_travel_time_s=300.0, weights_version="0.1.0"),
        )
        return RouteComparison(
            candidates=(fast, slow),
            selected_route_id="R-A",
            pruned_segment_ids=frozenset({"S9"}),
        )

    def test_rejected_alternatives_are_retained(self) -> None:
        """'Why not the short route?' is the first question anyone asks."""
        comparison = self._comparison()
        assert len(comparison.candidates) == 2
        assert comparison.selected.route_id == "R-A"

    def test_pruned_segments_are_recorded_for_explanation(self) -> None:
        assert self._comparison().pruned_segment_ids == {"S9"}

    def test_the_selection_must_be_one_of_the_candidates(self) -> None:
        with pytest.raises(ValidationError, match="not among the"):
            RouteComparison(candidates=self._comparison().candidates, selected_route_id="R-Z")


class TestSignalAdvisory:
    def test_advisories_cannot_be_marked_non_advisory(self) -> None:
        """There is no reading of this project's scope in which control is allowed."""
        with pytest.raises(ValidationError, match="cannot be False"):
            SignalAdvisory(
                junction_id="J11",
                action=SignalAdvisoryAction.GRANT_EMERGENCY_PRIORITY,
                rationale="ambulance approaching",
                advisory_only=False,
            )

    def test_an_advisory_defaults_to_advisory(self) -> None:
        advisory = SignalAdvisory(
            junction_id="J11",
            action=SignalAdvisoryAction.PREPARE_EMERGENCY_PRIORITY,
            rationale="staged ahead of predicted arrival",
        )
        assert advisory.advisory_only is True

    def test_a_corridor_must_contain_at_least_one_junction(self) -> None:
        with pytest.raises(ValidationError, match="at least one junction advisory"):
            CorridorAdvisory(emergency_id="EMG-1", route_id="R-A", advisories=())

    def test_a_corridor_is_ordered_and_recomputable(self) -> None:
        """Simultaneous priority everywhere holds cross traffic far too long."""
        corridor = CorridorAdvisory(
            emergency_id="EMG-1",
            route_id="R-A",
            advisories=(
                SignalAdvisory(
                    junction_id="J11",
                    action=SignalAdvisoryAction.GRANT_EMERGENCY_PRIORITY,
                    rationale="vehicle present",
                    earliest_apply_at=T0,
                ),
                SignalAdvisory(
                    junction_id="J14",
                    action=SignalAdvisoryAction.PREPARE_EMERGENCY_PRIORITY,
                    rationale="predicted arrival downstream",
                    earliest_apply_at=T0 + dt.timedelta(seconds=90),
                ),
            ),
        )
        assert [a.junction_id for a in corridor.advisories] == ["J11", "J14"]
        assert corridor.recompute_if_conditions_change is True


class TestFactTableGate:
    def _table(self) -> FactTable:
        return FactTable(
            facts=(
                Fact(
                    fact_id="eta.selected",
                    label="Predicted emergency arrival time",
                    value=187.0,
                    unit="s",
                    provenance=PREDICTED,
                    display_variants=("187 s", "3.1 min"),
                ),
                Fact(
                    fact_id="queue.j11.a1",
                    label="Queue length on J11 approach 1",
                    value=85.0,
                    unit="m",
                    provenance=PROV,
                    display_variants=("85 m",),
                ),
            ),
            entity_tokens=frozenset({"J11", "J14", "Lane 1", "Route B"}),
        )

    def test_supported_numbers_pass(self) -> None:
        text = "Queue on J11 Lane 1 is 85 m; predicted arrival 187 s via Route B."
        assert scan_for_unsupported_numbers(text, self._table()) == ()

    def test_an_invented_number_is_caught(self) -> None:
        """The characteristic LLM failure: a plausible round number."""
        text = "Recommended route saves approximately 240 seconds."
        assert 240.0 in scan_for_unsupported_numbers(text, self._table())

    def test_identifiers_are_not_mistaken_for_measurements(self) -> None:
        text = "Proceed J11 to J14 using Route B."
        assert scan_for_unsupported_numbers(text, self._table()) == ()

    def test_pre_rendered_unit_conversions_are_accepted(self) -> None:
        """The model quotes our conversion instead of performing arithmetic."""
        text = "Estimated arrival in 3.1 min."
        assert scan_for_unsupported_numbers(text, self._table()) == ()

    def test_display_rounding_is_tolerated_but_invention_is_not(self) -> None:
        assert scan_for_unsupported_numbers("Arrival 187 s", self._table()) == ()
        assert scan_for_unsupported_numbers("Arrival 210 s", self._table()) == (210.0,)

    def test_an_empty_fact_table_rejects_every_number(self) -> None:
        assert scan_for_unsupported_numbers("It will take 90 seconds", FactTable()) == (90.0,)


class TestDecisionRecord:
    def _manifest(self) -> RunManifest:
        return RunManifest(
            scenario_id="SC-1",
            input_source_id="src-1",
            input_source_kind="recorded_video",
            app_version="0.1.0",
            schema_version="0.1.0",
            network_version="1.0",
            safety_rules_version="0.1.0",
            route_weights_version="0.1.0",
            created_at=T0,
        )

    def _record(self, genai: GenAIRecommendation | None, table: FactTable) -> DecisionRecord:
        return DecisionRecord(
            decision_id="D-1",
            manifest=self._manifest(),
            network_state_id="ns-1",
            fact_table=table,
            genai=genai,
            narrative_validation=NarrativeValidation(passed=True) if genai else None,
            created_at=T0,
        )

    def _genai(self, reason: str, cited: frozenset[str]) -> GenAIRecommendation:
        return GenAIRecommendation(
            recommendation="Use Route B.",
            reason=reason,
            cited_fact_ids=cited,
            model_id="test/model",
            generated_at=T0,
        )

    def test_narrative_must_be_validated_before_it_can_be_carried(self) -> None:
        """Unvalidated model prose must never reach an operator."""
        with pytest.raises(ValidationError, match="must also carry its"):
            DecisionRecord(
                decision_id="D-1",
                manifest=self._manifest(),
                network_state_id="ns-1",
                genai=self._genai("because", frozenset()),
                created_at=T0,
            )

    def test_validation_catches_invented_numbers(self) -> None:
        table = FactTable(
            facts=(
                Fact(
                    fact_id="eta",
                    label="ETA",
                    value=187.0,
                    unit="s",
                    provenance=PREDICTED,
                    display_variants=("187 s",),
                ),
            )
        )
        record = self._record(self._genai("Saves 240 s of travel.", frozenset({"eta"})), table)
        result = record.validate_narrative()
        assert result.passed is False
        assert 240.0 in result.unsupported_numbers

    def test_validation_catches_citations_of_facts_that_do_not_exist(self) -> None:
        table = FactTable(
            facts=(Fact(fact_id="eta", label="ETA", value=187.0, provenance=PREDICTED),)
        )
        record = self._record(self._genai("Fastest safe route.", frozenset({"ghost"})), table)
        result = record.validate_narrative()
        assert result.passed is False
        assert result.unknown_fact_ids == ("ghost",)

    def test_a_clean_narrative_passes(self) -> None:
        table = FactTable(
            facts=(
                Fact(
                    fact_id="eta",
                    label="ETA",
                    value=187.0,
                    unit="s",
                    provenance=PREDICTED,
                    display_variants=("187 s",),
                ),
            )
        )
        record = self._record(
            self._genai("Predicted arrival is 187 s.", frozenset({"eta"})), table
        )
        assert record.validate_narrative().passed is True

    def test_predicted_content_is_flagged_for_presentation(self) -> None:
        """Predicted quantities must not render identically to measured ones."""
        table = FactTable(
            facts=(Fact(fact_id="eta", label="ETA", value=187.0, provenance=PREDICTED),)
        )
        assert self._record(None, table).has_predicted_content is True

    def test_a_validation_result_cannot_contradict_itself(self) -> None:
        with pytest.raises(ValidationError, match="contradicts"):
            NarrativeValidation(unsupported_numbers=(240.0,), passed=True)

    def test_degraded_evidence_must_state_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="at least one stated reason"):
            DecisionRecord(
                decision_id="D-1",
                manifest=self._manifest(),
                network_state_id="ns-1",
                evidence_degraded=True,
                created_at=T0,
            )

    def test_manifest_model_versions_are_empty_not_placeholder(self) -> None:
        assert self._manifest().model_versions == {}
