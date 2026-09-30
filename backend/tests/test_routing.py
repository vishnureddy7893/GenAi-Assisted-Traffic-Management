"""Phase 7 route optimization tests.

Networks are built programmatically with the exact geometry each assertion needs,
so a ranking can be checked against arithmetic rather than against whatever a
shared fixture happened to contain.

The properties under test are mostly about what must *not* happen. A blocked
segment must never appear in a candidate. An observation bound to nothing must
never move a cost. A missing travel time must never become a zero. And two runs
over identical inputs must produce byte-identical results, because a
recommendation that cannot be replayed cannot be audited.
"""

from __future__ import annotations

import datetime as dt
from itertools import pairwise

import pytest
from fastapi.testclient import TestClient
from network_fixtures import (
    make_junction,
    make_network,
    make_segment,
    square_network,
)

from app.analytics.schemas import (
    AnalyticsProvenance,
    ObservationMode,
    QualityGrade,
)
from app.core.config import Settings
from app.domain.enums import (
    CongestionLevel,
    EmergencyVehicleType,
    FreshnessStatus,
    IncidentSeverity,
    IncidentStatus,
    IncidentType,
    MovementType,
)
from app.domain.incident import Incident, IncidentLocation
from app.domain.network import TurnMovement
from app.domain.provenance import Provenance, ValueSource
from app.domain.routing import TimeBasis
from app.domain.state import StateAssemblyPolicy
from app.main import create_app
from app.network.assembler import assemble_network_state
from app.routing.engine import RouteOptimizer, optimize
from app.routing.mapping import (
    ObservationBinding,
    ObservationMapping,
    validate_against_network,
)
from app.routing.pruning import build_subgraph
from app.routing.schemas import (
    EvidenceStatus,
    OptimizationStatus,
    ReasonCode,
    RejectionReason,
    RouteRequest,
)
from app.routing.weights import RoutingMode, weights_for
from app.state.classifier import thresholds_from
from app.state.schemas import (
    CongestionAssessment,
    EvidenceBasis,
    IntersectionStateElement,
    IntersectionTrafficState,
    NetworkTrafficState,
)
from app.vision.models import ModelIdentity

T0 = dt.datetime(2024, 3, 14, 8, 30, tzinfo=dt.UTC)
MODEL = ModelIdentity(name="test-detector", backend="test", sha256="d" * 64, size_bytes=1)

PROV = Provenance(
    source=ValueSource.OBSERVED,
    producer="tests.routing",
    producer_version="1.0",
    observed_at=T0,
    computed_at=T0,
)

POLICY = StateAssemblyPolicy(max_staleness_s=5.0, drop_after_s=30.0)


@pytest.fixture
def route_settings() -> Settings:
    """Deterministic settings with the default routing policy."""
    return Settings(
        app_env="local",
        log_level="WARNING",
        max_observation_staleness_s=5.0,
        drop_observation_after_s=30.0,
    )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def make_incident(
    incident_id: str = "INC-1",
    *,
    blocks: frozenset[str] = frozenset(),
    affected: frozenset[str] = frozenset(),
    status: IncidentStatus = IncidentStatus.CONFIRMED,
) -> Incident:
    """An incident blocking or degrading specific segments."""
    return Incident(
        incident_id=incident_id,
        incident_type=IncidentType.ACCIDENT,
        severity=IncidentSeverity.MAJOR,
        status=status,
        location=IncidentLocation(segment_id=next(iter(blocks | affected), "SEG-12")),
        reported_at=T0,
        cleared_at=T0 if status is IncidentStatus.CLEARED else None,
        provenance=PROV,
        blocks_segment_ids=blocks,
        affected_segment_ids=affected,
    )


def make_traffic_state(
    settings: Settings,
    bands: dict[str, CongestionLevel],
    *,
    freshness: FreshnessStatus = FreshnessStatus.FRESH,
    unclassified: frozenset[str] = frozenset(),
) -> NetworkTrafficState:
    """A Phase 5 snapshot with a chosen band per observation area."""
    elements = []
    for area_id, level in sorted(bands.items()):
        is_unknown = area_id in unclassified or level is CongestionLevel.UNKNOWN
        assessment = CongestionAssessment(
            level=CongestionLevel.UNKNOWN if is_unknown else level,
            index=None if is_unknown else 0.5,
            basis=(
                EvidenceBasis.INSUFFICIENT if is_unknown else EvidenceBasis.IMAGE_SPACE
            ),
            thresholds=thresholds_from(settings),
            reason="Measurement quality is poor." if is_unknown else None,
            method="test",
            method_version="1.0",
        )
        state = IntersectionTrafficState(
            state_id=f"state-{area_id}",
            observation_area_id=area_id,
            mode=ObservationMode.SEQUENCE,
            as_of=T0,
            assessed_at=T0,
            congestion=assessment,
            vehicle_count=12,
            quality_grade=QualityGrade.GOOD,
            analytics_provenance=AnalyticsProvenance(
                media_id=f"media-{area_id}",
                source_id=area_id,
                frame_range=(0, 90),
                model=MODEL,
                tracker="test-tracker",
                method="test",
                method_version="1.0",
                analytics_schema_version="1.0",
                computed_at=T0,
            ),
            schema_version="1.0",
        )
        elements.append(
            IntersectionStateElement(
                state=state,
                freshness=freshness,
                evidence_age_s=0.0 if freshness is FreshnessStatus.FRESH else 12.0,
            )
        )
    return NetworkTrafficState(
        state_id="netstate-test",
        as_of=T0,
        assembled_at=T0,
        policy=POLICY,
        elements=tuple(elements),
        schema_version="1.0",
    )


def make_mapping(bindings: dict[str, str], *, network_id: str = "test-net") -> ObservationMapping:
    """Bind observation areas to segments."""
    return ObservationMapping(
        mapping_id="map-test",
        version="1.0",
        network_id=network_id,
        provenance=Provenance(
            source=ValueSource.ASSUMED,
            producer="tests.survey",
            producer_version="1.0",
            observed_at=T0,
            computed_at=T0,
            notes="Synthetic binding for tests. Not a survey of anything real.",
        ),
        bindings=tuple(
            ObservationBinding(observation_area_id=area, segment_id=segment)
            for area, segment in sorted(bindings.items())
        ),
    )


def run_optimize(
    settings: Settings,
    *,
    network=None,
    origin: str = "J1",
    destination: str = "J3",
    mode: RoutingMode = RoutingMode.NORMAL,
    incidents: tuple[Incident, ...] = (),
    traffic_state: NetworkTrafficState | None = None,
    mapping: ObservationMapping | None = None,
    vehicle_type: EmergencyVehicleType | None = None,
    max_candidates: int | None = None,
    request_id: str = "req-test",
):
    """Optimize over a network, assembling the Phase 6 state the way the API does."""
    net = network if network is not None else square_network(free_flow_speed_kph=45.0)
    state = assemble_network_state(
        net,
        incidents=incidents,
        policy=POLICY,
        schema_version="0.1.0",
        as_of=T0,
        assembled_at=T0,
    )
    optimizer = RouteOptimizer(
        net,
        settings=settings,
        mode=mode,
        network_state=state,
        incidents=incidents,
        traffic_state=traffic_state,
        mapping=mapping,
        vehicle_type=vehicle_type,
    )
    request = RouteRequest(
        request_id=request_id,
        mode=mode,
        origin_junction_id=origin,
        destination_junction_id=destination,
        vehicle_type=vehicle_type,
        max_candidates=max_candidates,
        requested_at=T0,
    )
    return optimize(request, optimizer, destination_junction_id=destination, at=T0)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
class TestShortestPath:
    def test_a_simple_shortest_path(self, route_settings: Settings) -> None:
        """A -> B -> C on a chain, with the only path found."""
        network = make_network(
            junctions=(make_junction("A"), make_junction("B"), make_junction("C")),
            segments=(
                make_segment("S-AB", "A", "B", length_m=100.0, free_flow_speed_kph=36.0),
                make_segment("S-BC", "B", "C", length_m=200.0, free_flow_speed_kph=36.0),
            ),
        )
        result = run_optimize(
            route_settings, network=network, origin="A", destination="C"
        )
        assert result.status is OptimizationStatus.ROUTES_FOUND
        route = result.routes[0]
        assert route.segment_sequence == ("S-AB", "S-BC")
        assert route.junction_sequence == ("A", "B", "C")
        # 36 km/h is 10 m/s, so 300 m takes 30 s.
        assert route.estimated_travel_time_s == pytest.approx(30.0)
        assert route.total_distance_m == pytest.approx(300.0)

    def test_the_cheaper_of_two_paths_wins(self, route_settings: Settings) -> None:
        """Weighted, not merely shortest by hop count."""
        network = make_network(
            junctions=(
                make_junction("A"),
                make_junction("B"),
                make_junction("C"),
                make_junction("D"),
            ),
            segments=(
                make_segment("S-AB", "A", "B", length_m=100.0, free_flow_speed_kph=36.0),
                make_segment("S-BD", "B", "D", length_m=100.0, free_flow_speed_kph=36.0),
                make_segment("S-AC", "A", "C", length_m=900.0, free_flow_speed_kph=36.0),
                make_segment("S-CD", "C", "D", length_m=900.0, free_flow_speed_kph=36.0),
            ),
        )
        result = run_optimize(
            route_settings, network=network, origin="A", destination="D"
        )
        assert result.routes[0].segment_sequence == ("S-AB", "S-BD")
        assert result.routes[0].estimated_travel_time_s == pytest.approx(20.0)

    def test_a_faster_road_beats_a_shorter_one(self, route_settings: Settings) -> None:
        """Ranking is on time, not distance."""
        network = make_network(
            junctions=(make_junction("A"), make_junction("B"), make_junction("C")),
            segments=(
                make_segment("S-AC", "A", "C", length_m=500.0, free_flow_speed_kph=18.0),
                make_segment("S-AB", "A", "B", length_m=600.0, free_flow_speed_kph=108.0),
                make_segment("S-BC", "B", "C", length_m=600.0, free_flow_speed_kph=108.0),
            ),
        )
        result = run_optimize(
            route_settings, network=network, origin="A", destination="C"
        )
        assert result.routes[0].segment_sequence == ("S-AB", "S-BC")
        assert result.routes[0].total_distance_m > result.routes[1].total_distance_m

    def test_roads_are_directed(self, route_settings: Settings) -> None:
        """A one-way road cannot be driven backwards."""
        network = make_network(
            junctions=(make_junction("A"), make_junction("B")),
            segments=(
                make_segment("S-AB", "A", "B", free_flow_speed_kph=36.0),
            ),
        )
        forward = run_optimize(
            route_settings, network=network, origin="A", destination="B"
        )
        backward = run_optimize(
            route_settings, network=network, origin="B", destination="A"
        )
        assert forward.status is OptimizationStatus.ROUTES_FOUND
        assert backward.status is OptimizationStatus.NO_ADMISSIBLE_ROUTE
        assert backward.rejections[-1].reason is RejectionReason.NETWORK_DISCONNECTED

    def test_alternatives_are_returned_ranked(self, route_settings: Settings) -> None:
        """"Why not the other way?" is answerable only if alternatives survive."""
        result = run_optimize(route_settings, max_candidates=4)
        assert len(result.routes) >= 2
        costs = [route.candidate.cost.total_s for route in result.routes]
        assert costs == sorted(costs)
        assert [route.rank for route in result.routes] == list(
            range(1, len(result.routes) + 1)
        )

    def test_candidate_count_is_capped_by_the_request(
        self, route_settings: Settings
    ) -> None:
        assert len(run_optimize(route_settings, max_candidates=1).routes) == 1

    def test_a_disconnected_network_reports_the_map_not_a_constraint(
        self, route_settings: Settings
    ) -> None:
        network = make_network(
            junctions=(make_junction("A"), make_junction("B"), make_junction("Z")),
            segments=(make_segment("S-AB", "A", "B", free_flow_speed_kph=36.0),),
        )
        result = run_optimize(
            route_settings, network=network, origin="A", destination="Z"
        )
        assert result.status is OptimizationStatus.NO_ADMISSIBLE_ROUTE
        assert result.rejections[-1].reason is RejectionReason.NETWORK_DISCONNECTED
        assert "does not connect" in result.rejections[-1].message


# ---------------------------------------------------------------------------
# Hard constraints
# ---------------------------------------------------------------------------
class TestHardConstraints:
    def test_a_blocked_road_is_never_selected(self, route_settings: Settings) -> None:
        """The single most important property in this phase."""
        incident = make_incident(blocks=frozenset({"SEG-12"}))
        result = run_optimize(route_settings, incidents=(incident,))

        assert result.status is OptimizationStatus.ROUTES_FOUND
        for route in result.routes:
            assert "SEG-12" not in route.segment_sequence
        assert "SEG-12" in result.pruned_segment_ids

    def test_a_blocked_road_is_absent_from_every_candidate_not_merely_ranked_last(
        self, route_settings: Settings
    ) -> None:
        """Candidates are generated on the pruned subgraph, so it cannot appear."""
        incident = make_incident(blocks=frozenset({"SEG-12", "SEG-23"}))
        result = run_optimize(route_settings, incidents=(incident,), max_candidates=10)
        every_segment = {
            segment for route in result.routes for segment in route.segment_sequence
        }
        assert not every_segment & {"SEG-12", "SEG-23"}

    def test_an_exclusion_zone_prunes_as_hard_as_a_blockage(
        self, route_settings: Settings
    ) -> None:
        from app.domain.incident import ExclusionZone
        from app.domain.provenance import FloatMeasurement, Unit

        incident = Incident(
            incident_id="INC-FIRE",
            incident_type=IncidentType.FIRE,
            severity=IncidentSeverity.CRITICAL,
            status=IncidentStatus.CONFIRMED,
            location=IncidentLocation(junction_id="J2"),
            reported_at=T0,
            provenance=PROV,
            exclusion_zone=ExclusionZone(
                radius_m=FloatMeasurement(value=200.0, unit=Unit.METER, provenance=PROV),
                excluded_segment_ids=frozenset({"SEG-12"}),
                reason="Active fire",
            ),
        )
        result = run_optimize(route_settings, incidents=(incident,))
        assert "SEG-12" in result.pruned_segment_ids
        for route in result.routes:
            assert "SEG-12" not in route.segment_sequence

    def test_a_cleared_incident_stops_pruning(self, route_settings: Settings) -> None:
        incident = make_incident(
            blocks=frozenset({"SEG-12"}), status=IncidentStatus.CLEARED
        )
        result = run_optimize(route_settings, incidents=(incident,))
        assert "SEG-12" not in result.pruned_segment_ids

    def test_a_vehicle_excluded_road_is_pruned_for_that_responder(
        self, route_settings: Settings
    ) -> None:
        """A ladder truck under a low bridge is a hard constraint, not traffic."""
        network = make_network(
            junctions=(make_junction("A"), make_junction("B"), make_junction("C")),
            segments=(
                make_segment(
                    "S-AC",
                    "A",
                    "C",
                    length_m=100.0,
                    free_flow_speed_kph=36.0,
                    excluded_vehicle_types=frozenset({EmergencyVehicleType.FIRE_ENGINE}),
                ),
                make_segment("S-AB", "A", "B", length_m=900.0, free_flow_speed_kph=36.0),
                make_segment("S-BC", "B", "C", length_m=900.0, free_flow_speed_kph=36.0),
            ),
        )
        fire = run_optimize(
            route_settings,
            network=network,
            origin="A",
            destination="C",
            mode=RoutingMode.EMERGENCY,
            vehicle_type=EmergencyVehicleType.FIRE_ENGINE,
        )
        assert fire.routes[0].segment_sequence == ("S-AB", "S-BC")
        assert "S-AC" in fire.pruned_segment_ids

        ambulance = run_optimize(
            route_settings,
            network=network,
            origin="A",
            destination="C",
            mode=RoutingMode.EMERGENCY,
            vehicle_type=EmergencyVehicleType.AMBULANCE,
        )
        assert ambulance.routes[0].segment_sequence == ("S-AC",)

    def test_a_prohibited_turn_cannot_be_taken(self, route_settings: Settings) -> None:
        """The U-turns the square network bans are never used."""
        result = run_optimize(route_settings, origin="J1", destination="J3", max_candidates=6)
        for route in result.routes:
            legs = route.candidate.legs
            for previous, following in pairwise(legs):
                assert previous.from_junction_id != following.to_junction_id, (
                    "a banned U-turn was routed through"
                )

    def test_an_undeclared_turn_is_treated_as_prohibited(
        self, route_settings: Settings
    ) -> None:
        """Omission is indistinguishable from an incomplete map import.

        A junction that declares some turns out of an approach but not this one
        has not said the turn is allowed, and "unknown" must not read as "yes".
        """
        segments = (
            make_segment("S-AB", "A", "B", free_flow_speed_kph=36.0),
            make_segment("S-BC", "B", "C", free_flow_speed_kph=36.0),
            make_segment("S-BD", "B", "D", free_flow_speed_kph=36.0),
        )
        junction_b = make_junction(
            "B",
            movements=(
                TurnMovement(
                    movement_id="MV-AB-BD",
                    junction_id="B",
                    from_segment_id="S-AB",
                    to_segment_id="S-BD",
                    movement_type=MovementType.THROUGH,
                    is_permitted=True,
                ),
            ),
        )
        network = make_network(
            junctions=(
                make_junction("A"),
                junction_b,
                make_junction("C"),
                make_junction("D"),
            ),
            segments=segments,
        )
        subgraph = build_subgraph(network)
        assert subgraph.permits_movement("B", "S-AB", "S-BD") is True
        assert subgraph.permits_movement("B", "S-AB", "S-BC") is False

        result = run_optimize(
            route_settings, network=network, origin="A", destination="C"
        )
        assert result.status is OptimizationStatus.NO_ADMISSIBLE_ROUTE

    def test_congestion_is_never_a_hard_constraint(
        self, route_settings: Settings
    ) -> None:
        """A gridlocked road is slow, not impassable."""
        traffic = make_traffic_state(route_settings, {"Z-12": CongestionLevel.GRIDLOCK})
        mapping = make_mapping({"Z-12": "SEG-12"})
        result = run_optimize(
            route_settings, traffic_state=traffic, mapping=mapping, max_candidates=6
        )
        assert "SEG-12" not in result.pruned_segment_ids
        assert any("SEG-12" in route.segment_sequence for route in result.routes)


class TestNoAdmissibleRoute:
    def test_pruning_that_severs_the_graph_is_reported_as_such(
        self, route_settings: Settings
    ) -> None:
        """Distinct from a disconnected map: this one an operator can act on."""
        incident = make_incident(blocks=frozenset({"SEG-12", "SEG-14"}))
        result = run_optimize(route_settings, incidents=(incident,))

        assert result.status is OptimizationStatus.NO_ADMISSIBLE_ROUTE
        assert result.routes == ()
        assert result.recommendation is None
        final = result.rejections[-1]
        assert final.reason is RejectionReason.DISCONNECTED_BY_PRUNING
        assert final.incident_ids == ("INC-1",)

    def test_the_refusal_names_the_segments_that_caused_it(
        self, route_settings: Settings
    ) -> None:
        incident = make_incident(blocks=frozenset({"SEG-12", "SEG-14"}))
        result = run_optimize(route_settings, incidents=(incident,))
        blocked = next(
            r for r in result.rejections if r.reason is RejectionReason.SEGMENT_BLOCKED
        )
        assert set(blocked.segment_ids) == {"SEG-12", "SEG-14"}

    def test_no_route_is_invented_for_an_emergency(
        self, route_settings: Settings
    ) -> None:
        """The one output a person might act on must never be fabricated."""
        incident = make_incident(blocks=frozenset({"SEG-12", "SEG-14"}))
        result = run_optimize(
            route_settings,
            incidents=(incident,),
            mode=RoutingMode.EMERGENCY,
            vehicle_type=EmergencyVehicleType.AMBULANCE,
        )
        assert result.status is OptimizationStatus.NO_ADMISSIBLE_ROUTE
        assert result.routes == ()
        assert result.comparison is None


# ---------------------------------------------------------------------------
# Soft costs and evidence
# ---------------------------------------------------------------------------
class TestSoftCosts:
    def test_congestion_moves_the_ranking_without_pruning(
        self, route_settings: Settings
    ) -> None:
        """The example from the specification, on a network built for it."""
        clean = run_optimize(route_settings, max_candidates=4)
        assert clean.routes[0].segment_sequence == ("SEG-12", "SEG-23")

        traffic = make_traffic_state(route_settings, {"Z-12": CongestionLevel.SEVERE})
        congested = run_optimize(
            route_settings,
            traffic_state=traffic,
            mapping=make_mapping({"Z-12": "SEG-12"}),
            max_candidates=4,
        )
        assert congested.routes[0].segment_sequence == ("SEG-14", "SEG-43")
        reasons = {reason.code for reason in congested.recommendation.reasons}
        assert ReasonCode.LOWER_TOTAL_COST in reasons

    def test_the_congestion_penalty_matches_the_configured_weight(
        self, route_settings: Settings
    ) -> None:
        traffic = make_traffic_state(route_settings, {"Z-12": CongestionLevel.HEAVY})
        result = run_optimize(
            route_settings,
            traffic_state=traffic,
            mapping=make_mapping({"Z-12": "SEG-12"}),
            max_candidates=6,
        )
        route = next(
            r for r in result.routes if "SEG-12" in r.segment_sequence
        )
        assert route.candidate.cost.congestion_penalty_s == pytest.approx(
            route_settings.route_congestion_heavy_s
        )

    def test_an_incident_degrades_without_blocking(
        self, route_settings: Settings
    ) -> None:
        incident = make_incident(affected=frozenset({"SEG-12"}))
        result = run_optimize(route_settings, incidents=(incident,), max_candidates=6)
        assert "SEG-12" not in result.pruned_segment_ids
        route = next(r for r in result.routes if "SEG-12" in r.segment_sequence)
        assert route.candidate.cost.incident_penalty_s == pytest.approx(
            route_settings.route_incident_degraded_s
        )

    def test_a_turn_costs_more_than_going_straight(
        self, route_settings: Settings
    ) -> None:
        result = run_optimize(route_settings, max_candidates=6)
        assert all(
            route.candidate.cost.turn_penalty_s >= 0.0 for route in result.routes
        )

    def test_soft_conditions_produce_warnings_not_rejections(
        self, route_settings: Settings
    ) -> None:
        incident = make_incident(affected=frozenset({"SEG-12"}))
        result = run_optimize(route_settings, incidents=(incident,), max_candidates=6)
        route = next(r for r in result.routes if "SEG-12" in r.segment_sequence)
        assert any("degraded" in warning for warning in route.candidate.warnings)


class TestEvidenceMapping:
    def test_a_mapped_observation_affects_the_cost(
        self, route_settings: Settings
    ) -> None:
        traffic = make_traffic_state(route_settings, {"Z-12": CongestionLevel.SEVERE})
        result = run_optimize(
            route_settings,
            traffic_state=traffic,
            mapping=make_mapping({"Z-12": "SEG-12"}),
            max_candidates=6,
        )
        route = next(r for r in result.routes if "SEG-12" in r.segment_sequence)
        evidence = next(e for e in route.evidence if e.segment_id == "SEG-12")
        assert evidence.status is EvidenceStatus.MAPPED_CLASSIFIED
        assert evidence.congestion_level is CongestionLevel.SEVERE
        assert evidence.congestion_penalty_s > 0.0
        assert evidence.observation_area_id == "Z-12"

    def test_an_unmapped_observation_does_not_affect_any_cost(
        self, route_settings: Settings
    ) -> None:
        """The property the whole mapping contract exists to guarantee."""
        traffic = make_traffic_state(route_settings, {"Z-99": CongestionLevel.GRIDLOCK})

        without = run_optimize(route_settings, max_candidates=6)
        with_unmapped = run_optimize(
            route_settings, traffic_state=traffic, mapping=None, max_candidates=6
        )

        assert [r.candidate.cost.total_s for r in without.routes] == [
            r.candidate.cost.total_s for r in with_unmapped.routes
        ]
        assert with_unmapped.unmapped_observation_area_ids == ("Z-99",)
        for route in with_unmapped.routes:
            assert all(e.congestion_penalty_s == 0.0 for e in route.evidence)

    def test_an_observation_bound_to_another_segment_leaves_this_one_alone(
        self, route_settings: Settings
    ) -> None:
        traffic = make_traffic_state(route_settings, {"Z-43": CongestionLevel.GRIDLOCK})
        result = run_optimize(
            route_settings,
            traffic_state=traffic,
            mapping=make_mapping({"Z-43": "SEG-43"}),
            max_candidates=6,
        )
        route = next(r for r in result.routes if "SEG-12" in r.segment_sequence)
        evidence = next(e for e in route.evidence if e.segment_id == "SEG-12")
        assert evidence.status is EvidenceStatus.UNMAPPED
        assert evidence.congestion_penalty_s == 0.0

    def test_unmapped_segments_are_priced_as_uncertain_not_as_clear(
        self, route_settings: Settings
    ) -> None:
        """Missing must not become zero, or the optimizer prefers the dark."""
        result = run_optimize(route_settings, max_candidates=6)
        route = result.routes[0]
        assert route.candidate.cost.uncertainty_penalty_s > 0.0
        assert all(e.status is EvidenceStatus.UNMAPPED for e in route.evidence)
        assert all(e.uncertainty_penalty_s > 0.0 for e in route.evidence)

    def test_an_unclassified_observation_is_priced_as_uncertain(
        self, route_settings: Settings
    ) -> None:
        traffic = make_traffic_state(
            route_settings,
            {"Z-12": CongestionLevel.UNKNOWN},
            unclassified=frozenset({"Z-12"}),
        )
        result = run_optimize(
            route_settings,
            traffic_state=traffic,
            mapping=make_mapping({"Z-12": "SEG-12"}),
            max_candidates=6,
        )
        route = next(r for r in result.routes if "SEG-12" in r.segment_sequence)
        evidence = next(e for e in route.evidence if e.segment_id == "SEG-12")
        assert evidence.status is EvidenceStatus.MAPPED_UNCLASSIFIED
        assert evidence.congestion_penalty_s == 0.0
        assert evidence.uncertainty_penalty_s == pytest.approx(
            route_settings.route_uncertainty_unclassified_s
        )

    def test_stale_evidence_is_used_and_charged_for(
        self, route_settings: Settings
    ) -> None:
        """Withholding a route because one camera lagged is its own failure."""
        traffic = make_traffic_state(
            route_settings,
            {"Z-12": CongestionLevel.MODERATE},
            freshness=FreshnessStatus.STALE,
        )
        result = run_optimize(
            route_settings,
            traffic_state=traffic,
            mapping=make_mapping({"Z-12": "SEG-12"}),
            max_candidates=6,
        )
        route = next(r for r in result.routes if "SEG-12" in r.segment_sequence)
        evidence = next(e for e in route.evidence if e.segment_id == "SEG-12")
        assert evidence.status is EvidenceStatus.MAPPED_STALE
        assert evidence.congestion_penalty_s > 0.0
        assert evidence.uncertainty_penalty_s == pytest.approx(
            route_settings.route_uncertainty_stale_s
        )

    def test_evidence_carries_its_chain_back_to_the_media(
        self, route_settings: Settings
    ) -> None:
        traffic = make_traffic_state(route_settings, {"Z-12": CongestionLevel.LIGHT})
        result = run_optimize(
            route_settings,
            traffic_state=traffic,
            mapping=make_mapping({"Z-12": "SEG-12"}),
            max_candidates=6,
        )
        route = next(r for r in result.routes if "SEG-12" in r.segment_sequence)
        evidence = next(e for e in route.evidence if e.segment_id == "SEG-12")
        assert evidence.media_id == "media-Z-12"
        assert evidence.frame_range == (0, 90)
        assert evidence.detector_model == "test-detector"
        assert evidence.traffic_state_id == "state-Z-12"


class TestMappingValidation:
    def test_a_binding_to_an_unknown_segment_is_refused(self) -> None:
        with pytest.raises(Exception, match="unknown"):
            validate_against_network(
                make_mapping({"Z-1": "SEG-GHOST"}), square_network()
            )

    def test_a_mapping_for_another_network_is_refused(self) -> None:
        with pytest.raises(Exception, match="expressed against network"):
            validate_against_network(
                make_mapping({"Z-1": "SEG-12"}, network_id="other-net"),
                square_network(),
            )

    def test_a_binding_to_nothing_is_refused(self) -> None:
        with pytest.raises(Exception, match="neither a segment nor a junction"):
            ObservationBinding(observation_area_id="Z-1")

    def test_one_area_cannot_be_bound_twice(self) -> None:
        with pytest.raises(Exception, match="bound more than once"):
            ObservationMapping(
                mapping_id="m",
                version="1",
                network_id="test-net",
                provenance=PROV,
                bindings=(
                    ObservationBinding(observation_area_id="Z-1", segment_id="SEG-12"),
                    ObservationBinding(observation_area_id="Z-1", segment_id="SEG-23"),
                ),
            )


# ---------------------------------------------------------------------------
# Travel time
# ---------------------------------------------------------------------------
class TestTravelTime:
    def test_a_segment_with_no_speed_yields_no_eta(
        self, route_settings: Settings
    ) -> None:
        """No fallback speed is invented in its place."""
        result = run_optimize(route_settings, network=square_network())
        route = result.routes[0]
        assert route.travel_time_available is False
        assert route.estimated_travel_time_s is None
        assert route.unpriced_segment_ids
        assert all(basis is TimeBasis.UNAVAILABLE for basis in route.time_bases)

    def test_missing_travel_time_falls_back_to_distance_and_says_so(
        self, route_settings: Settings
    ) -> None:
        result = run_optimize(route_settings, network=square_network())
        assert result.status is OptimizationStatus.ROUTES_FOUND
        assert any("ranked by distance" in w for w in result.warnings)
        codes = {reason.code for reason in result.recommendation.reasons}
        assert ReasonCode.TRAVEL_TIME_UNAVAILABLE_RANKED_BY_DISTANCE in codes

    def test_a_derived_time_is_labelled_as_derived(
        self, route_settings: Settings
    ) -> None:
        result = run_optimize(route_settings)
        route = result.routes[0]
        assert all(
            basis is TimeBasis.DERIVED_FROM_FREE_FLOW for basis in route.time_bases
        )
        assert any("free-flow speed" in w for w in route.candidate.warnings)

    def test_a_leg_time_carries_its_own_provenance(
        self, route_settings: Settings
    ) -> None:
        result = run_optimize(route_settings)
        leg = result.routes[0].candidate.legs[0]
        assert leg.traversal_time_s is not None
        assert leg.traversal_time_s.provenance.source is ValueSource.DERIVED
        assert leg.traversal_time_s.provenance.input_refs == (leg.segment_id,)

    def test_an_eta_is_never_a_partial_sum(self, route_settings: Settings) -> None:
        """A route with one untimed leg reports no ETA, not the sum of the rest."""
        network = make_network(
            junctions=(make_junction("A"), make_junction("B"), make_junction("C")),
            segments=(
                make_segment("S-AB", "A", "B", length_m=100.0, free_flow_speed_kph=36.0),
                make_segment("S-BC", "B", "C", length_m=100.0),
            ),
        )
        result = run_optimize(
            route_settings, network=network, origin="A", destination="C"
        )
        route = result.routes[0]
        assert route.estimated_travel_time_s is None
        assert route.unpriced_segment_ids == ("S-BC",)


# ---------------------------------------------------------------------------
# Emergency routing
# ---------------------------------------------------------------------------
class TestEmergencyRouting:
    def test_the_emergency_fallback_is_stated_not_invented(
        self, route_settings: Settings
    ) -> None:
        """No emergency-specific time exists, so none is manufactured."""
        result = run_optimize(
            route_settings,
            mode=RoutingMode.EMERGENCY,
            vehicle_type=EmergencyVehicleType.AMBULANCE,
        )
        assumptions = result.routes[0].candidate.emergency_assumptions
        assert assumptions
        assert "no emergency advantage is modelled" in assumptions[0]
        assert "not scaled by a constant" in assumptions[0]

    def test_normal_routing_states_no_emergency_assumptions(
        self, route_settings: Settings
    ) -> None:
        result = run_optimize(route_settings, mode=RoutingMode.NORMAL)
        assert result.routes[0].candidate.emergency_assumptions == ()

    def test_the_weight_set_records_its_mode(self, route_settings: Settings) -> None:
        emergency = run_optimize(route_settings, mode=RoutingMode.EMERGENCY)
        assert emergency.weights.mode is RoutingMode.EMERGENCY
        assert emergency.weights.weights_version == route_settings.route_weights_version

    def test_emergency_weights_default_to_the_normal_ones(
        self, route_settings: Settings
    ) -> None:
        """Any divergence must be an operator decision, not one invented in code."""
        normal = weights_for(route_settings, RoutingMode.NORMAL)
        emergency = weights_for(route_settings, RoutingMode.EMERGENCY)
        assert normal.congestion_s == emergency.congestion_s
        assert normal.incident_degraded_s == emergency.incident_degraded_s

    def test_reconfigured_emergency_weights_change_the_ranking(self) -> None:
        settings = Settings(
            app_env="local",
            log_level="WARNING",
            route_emergency_congestion_severe_s=5.0,
            route_congestion_severe_s=900.0,
        )
        traffic = make_traffic_state(settings, {"Z-12": CongestionLevel.SEVERE})
        mapping = make_mapping({"Z-12": "SEG-12"})

        normal = run_optimize(
            settings, traffic_state=traffic, mapping=mapping, max_candidates=4
        )
        emergency = run_optimize(
            settings,
            traffic_state=traffic,
            mapping=mapping,
            mode=RoutingMode.EMERGENCY,
            max_candidates=4,
        )
        assert normal.routes[0].segment_sequence == ("SEG-14", "SEG-43")
        assert emergency.routes[0].segment_sequence == ("SEG-12", "SEG-23")


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------
class TestResultStructure:
    def test_the_cost_breakdown_sums_to_the_total(
        self, route_settings: Settings
    ) -> None:
        result = run_optimize(route_settings)
        cost = result.routes[0].candidate.cost
        assert cost.total_s == pytest.approx(
            cost.base_travel_time_s
            + cost.congestion_penalty_s
            + cost.incident_penalty_s
            + cost.safety_penalty_s
            + cost.accessibility_penalty_s
            + cost.turn_penalty_s
            + cost.uncertainty_penalty_s
            + cost.distance_penalty_s
        )

    def test_the_result_pins_everything_needed_to_replay_it(
        self, route_settings: Settings
    ) -> None:
        traffic = make_traffic_state(route_settings, {"Z-12": CongestionLevel.LIGHT})
        result = run_optimize(
            route_settings, traffic_state=traffic, mapping=make_mapping({"Z-12": "SEG-12"})
        )
        assert result.network_id == "test-net"
        assert result.network_version == "1.0"
        assert result.mapping_id == "map-test"
        assert result.mapping_version == "1.0"
        assert result.traffic_state_id == "netstate-test"
        assert result.algorithm == "dijkstra+yen-k-shortest"
        assert result.weights.weights_version == route_settings.route_weights_version

    def test_the_comparison_contract_is_produced_for_the_decision_engine(
        self, route_settings: Settings
    ) -> None:
        result = run_optimize(route_settings, max_candidates=3)
        assert result.comparison is not None
        assert result.comparison.selected_route_id == result.routes[0].candidate.route_id
        assert len(result.comparison.candidates) == len(result.routes)
        assert result.comparison.selected is result.comparison.candidates[0]

    def test_the_recommendation_gives_checkable_reasons(
        self, route_settings: Settings
    ) -> None:
        traffic = make_traffic_state(route_settings, {"Z-12": CongestionLevel.SEVERE})
        result = run_optimize(
            route_settings,
            traffic_state=traffic,
            mapping=make_mapping({"Z-12": "SEG-12"}),
            max_candidates=4,
        )
        reasons = result.recommendation.reasons
        assert reasons
        comparative = [r for r in reasons if r.comparison_value is not None]
        assert comparative, "a comparative claim must carry the figure it beat"
        assert result.recommendation.cost_margin_s is not None

    def test_a_single_candidate_wins_by_default_and_says_so(
        self, route_settings: Settings
    ) -> None:
        result = run_optimize(route_settings, max_candidates=1)
        codes = {reason.code for reason in result.recommendation.reasons}
        assert ReasonCode.ONLY_ADMISSIBLE_ROUTE in codes
        assert result.recommendation.cost_margin_s is None

    def test_a_refusal_cannot_carry_a_route(self, route_settings: Settings) -> None:
        """The invariant that keeps "no safe route" from quietly returning one."""
        from pydantic import ValidationError

        from app.routing.schemas import RouteOptimizationResult

        payload = run_optimize(route_settings).model_dump()
        payload["status"] = OptimizationStatus.NO_ADMISSIBLE_ROUTE.value

        with pytest.raises(ValidationError, match="still carries a route"):
            RouteOptimizationResult.model_validate(payload)

    def test_a_refusal_must_state_why(self, route_settings: Settings) -> None:
        from pydantic import ValidationError

        from app.routing.schemas import RouteOptimizationResult

        payload = run_optimize(route_settings).model_dump()
        payload["status"] = OptimizationStatus.NO_ADMISSIBLE_ROUTE.value
        payload["routes"] = []
        payload["comparison"] = None
        payload["recommendation"] = None
        payload["rejections"] = []

        with pytest.raises(ValidationError, match="must state why"):
            RouteOptimizationResult.model_validate(payload)

    def test_an_eta_cannot_accompany_unavailable_travel_time(
        self, route_settings: Settings
    ) -> None:
        """A partial sum must never be dressed up as an ETA."""
        from pydantic import ValidationError

        from app.routing.schemas import RankedRoute

        real = run_optimize(route_settings).routes[0]
        payload = real.model_dump()
        payload["travel_time_available"] = False
        payload["estimated_travel_time_s"] = 10.0

        with pytest.raises(ValidationError, match="must not be presented as an ETA"):
            RankedRoute.model_validate(payload)

    def test_an_available_travel_time_requires_a_figure(
        self, route_settings: Settings
    ) -> None:
        from pydantic import ValidationError

        from app.routing.schemas import RankedRoute

        payload = run_optimize(route_settings).routes[0].model_dump()
        payload["travel_time_available"] = True
        payload["estimated_travel_time_s"] = None

        with pytest.raises(ValidationError, match="no estimated_travel_time_s"):
            RankedRoute.model_validate(payload)


class TestDeterminism:
    def test_repeated_optimization_is_byte_identical(
        self, route_settings: Settings
    ) -> None:
        """A recommendation that cannot be replayed cannot be audited."""
        traffic = make_traffic_state(route_settings, {"Z-12": CongestionLevel.MODERATE})
        mapping = make_mapping({"Z-12": "SEG-12"})
        runs = [
            run_optimize(
                route_settings,
                traffic_state=traffic,
                mapping=mapping,
                max_candidates=4,
            ).model_dump()
            for _ in range(5)
        ]
        assert all(run == runs[0] for run in runs)

    def test_ties_resolve_the_same_way_every_time(
        self, route_settings: Settings
    ) -> None:
        """The square network has two identically-priced routes J1 to J3."""
        selections = {
            run_optimize(route_settings, max_candidates=4).routes[0].segment_sequence
            for _ in range(10)
        }
        assert len(selections) == 1

    def test_candidate_order_is_stable_across_runs(
        self, route_settings: Settings
    ) -> None:
        first = [r.segment_sequence for r in run_optimize(route_settings, max_candidates=4).routes]
        second = [r.segment_sequence for r in run_optimize(route_settings, max_candidates=4).routes]
        assert first == second


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@pytest.fixture
def routing_client(tmp_path):
    """A client with the square network loaded and a mapping file available."""
    import json

    network = square_network(free_flow_speed_kph=45.0)
    (tmp_path / "net.json").write_text(
        json.dumps(network.model_dump(mode="json")), encoding="utf-8"
    )
    (tmp_path / "map.json").write_text(
        json.dumps(make_mapping({"Z-12": "SEG-12"}).model_dump(mode="json")),
        encoding="utf-8",
    )
    settings = Settings(app_env="local", log_level="WARNING", networks_dir=tmp_path)
    with TestClient(create_app(settings)) as client:
        client.post("/api/network/load", json={"relative_path": "net.json"})
        yield client


class TestRoutingApi:
    def test_config_exposes_both_weight_sets(self, routing_client: TestClient) -> None:
        body = routing_client.get("/api/routes/config").json()
        assert body["algorithm"] == "dijkstra+yen-k-shortest"
        assert body["normal"]["mode"] == "normal"
        assert body["emergency"]["mode"] == "emergency"
        assert body["normal"]["congestion_s"]["heavy"] == 110.0

    def test_optimizing_returns_ranked_routes(self, routing_client: TestClient) -> None:
        response = routing_client.post(
            "/api/routes/optimize",
            json={"origin_junction_id": "J1", "destination_junction_id": "J3"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "routes_found"
        assert len(body["routes"]) >= 2
        assert body["routes"][0]["junction_sequence"][0] == "J1"
        assert body["routes"][0]["junction_sequence"][-1] == "J3"

    def test_routing_to_a_facility_resolves_its_access_junction(
        self, routing_client: TestClient
    ) -> None:
        body = routing_client.post(
            "/api/routes/optimize",
            json={"origin_junction_id": "J1", "destination_facility_id": "FAC-HOSP-1"},
        ).json()
        assert body["status"] == "routes_found"
        assert body["request"]["destination_facility_id"] == "FAC-HOSP-1"
        assert body["routes"][0]["junction_sequence"][-1] == "J3"

    def test_an_unknown_origin_is_refused(self, routing_client: TestClient) -> None:
        response = routing_client.post(
            "/api/routes/optimize",
            json={"origin_junction_id": "J-GHOST", "destination_junction_id": "J3"},
        )
        assert response.status_code == 422
        assert "not on network" in response.json()["error"]["message"]

    def test_a_missing_destination_is_refused(self, routing_client: TestClient) -> None:
        response = routing_client.post(
            "/api/routes/optimize", json={"origin_junction_id": "J1"}
        )
        assert response.status_code == 422

    def test_optimizing_without_a_network_is_refused(self, tmp_path) -> None:
        settings = Settings(app_env="local", log_level="WARNING", networks_dir=tmp_path)
        with TestClient(create_app(settings)) as client:
            response = client.post(
                "/api/routes/optimize",
                json={"origin_junction_id": "J1", "destination_junction_id": "J3"},
            )
            assert response.status_code == 404
            assert response.json()["error"]["code"] == "network_not_loaded"

    def test_loading_a_mapping_then_reading_it_back(
        self, routing_client: TestClient
    ) -> None:
        loaded = routing_client.post(
            "/api/routes/mapping/load", json={"relative_path": "map.json"}
        )
        assert loaded.status_code == 200
        body = loaded.json()
        assert body["loaded"] is True
        assert body["bound_segment_ids"] == ["SEG-12"]
        assert body["bound_observation_area_ids"] == ["Z-12"]

        status = routing_client.get("/api/routes/mapping").json()
        assert status["source"]["mapping_id"] == "map-test"

    def test_mapping_status_reports_the_empty_case_plainly(
        self, routing_client: TestClient
    ) -> None:
        body = routing_client.get("/api/routes/mapping").json()
        assert body["loaded"] is False
        assert "No observation mapping is loaded" in body["note"]

    def test_a_blocked_road_never_appears_in_an_api_route(
        self, routing_client: TestClient
    ) -> None:
        routing_client.post(
            "/api/incidents",
            json={
                "incident_type": "road_blockage",
                "severity": "critical",
                "location": {"segment_id": "SEG-12"},
                "blocks_segment_ids": ["SEG-12"],
            },
        )
        body = routing_client.post(
            "/api/routes/optimize",
            json={"origin_junction_id": "J1", "destination_junction_id": "J3"},
        ).json()
        assert body["status"] == "routes_found"
        for route in body["routes"]:
            assert "SEG-12" not in route["segment_sequence"]
        assert "SEG-12" in body["pruned_segment_ids"]

    def test_no_admissible_route_is_a_200_carrying_its_causes(
        self, routing_client: TestClient
    ) -> None:
        """A severed network is a complete answer, not a service fault."""
        routing_client.post(
            "/api/incidents",
            json={
                "incident_type": "road_blockage",
                "severity": "critical",
                "location": {"segment_id": "SEG-12"},
                "blocks_segment_ids": ["SEG-12", "SEG-14"],
            },
        )
        response = routing_client.post(
            "/api/routes/optimize",
            json={"origin_junction_id": "J1", "destination_junction_id": "J3"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "no_admissible_route"
        assert body["routes"] == []
        assert body["recommendation"] is None
        assert any(
            r["reason"] == "disconnected_by_pruning" for r in body["rejections"]
        )

    def test_routing_for_a_declared_emergency(self, routing_client: TestClient) -> None:
        declared = routing_client.post(
            "/api/emergency/declare",
            json={
                "vehicle": {"vehicle_id": "AMB-1", "vehicle_type": "ambulance"},
                "priority": "P1_immediate",
                "current_position": {"junction_id": "J1"},
                "destination": {"facility_id": "FAC-HOSP-1"},
            },
        ).json()
        emergency_id = declared["emergency"]["emergency_id"]

        body = routing_client.post(
            "/api/routes/optimize", json={"emergency_id": emergency_id}
        ).json()
        assert body["status"] == "routes_found"
        assert body["request"]["mode"] == "emergency"
        assert body["request"]["emergency_id"] == emergency_id
        assert body["request"]["vehicle_type"] == "ambulance"
        assert body["routes"][0]["junction_sequence"][-1] == "J3"

    def test_the_endpoint_returns_no_prose_recommendation(
        self, routing_client: TestClient
    ) -> None:
        """Facts only. Narrative is Phase 12, against a fact table."""
        body = routing_client.post(
            "/api/routes/optimize",
            json={"origin_junction_id": "J1", "destination_junction_id": "J3"},
        ).json()
        for reason in body["recommendation"]["reasons"]:
            assert reason["code"] in {item.value for item in ReasonCode}

    def test_no_approval_or_execution_endpoint_exists(
        self, routing_client: TestClient
    ) -> None:
        """Human approval and execution belong to later phases."""
        paths = routing_client.get("/openapi.json").json()["paths"]
        routing_paths = [p for p in paths if p.startswith("/api/routes")]
        for path in routing_paths:
            assert not any(
                token in path for token in ("approve", "execute", "dispatch", "apply")
            )

    def test_repeated_api_calls_agree(self, routing_client: TestClient) -> None:
        def selection() -> list[str]:
            body = routing_client.post(
                "/api/routes/optimize",
                json={"origin_junction_id": "J1", "destination_junction_id": "J3"},
            ).json()
            return body["routes"][0]["segment_sequence"]

        assert selection() == selection() == selection()
