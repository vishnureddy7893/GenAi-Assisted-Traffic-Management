"""Phase 8 tests: the traffic simulation layer.

Three things are being checked, and they are not the same thing:

* **The model is right.** Vehicles are conserved, a queue forms exactly when
  demand exceeds capacity, and the delay a vehicle experiences is the queue
  ahead of it divided by the rate the exit discharges at.
* **The model is deterministic.** Identical inputs produce byte-identical
  output, because a comparison that cannot be reproduced cannot be reviewed.
* **The model does not invent anything.** Every figure it produces is labelled
  ``SIMULATED`` or ``DERIVED``, never ``OBSERVED``; every missing input produces
  an absent output with a reason; and no default demand, capacity or speed
  exists anywhere in the package.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from network_fixtures import square_network

from app.analytics.schemas import (
    AnalyticsProvenance,
    ClassTallyOut,
    MeasurementQuality,
    MetricAvailability,
    MetricValue,
    ObservationMode,
    QualityGrade,
    SpatialDistribution,
    TrafficAnalytics,
    available,
    unavailable,
)
from app.core.config import Settings
from app.domain.enums import VehicleClass
from app.domain.provenance import Provenance, ValueSource
from app.domain.routing import TimeBasis
from app.domain.state import NetworkState, RoadSegmentState, StateAssemblyPolicy
from app.main import create_app
from app.routing.mapping import ObservationBinding, ObservationMapping
from app.routing.weights import RoutingMode
from app.simulation import model as engine
from app.simulation import runner
from app.simulation.scenario import (
    ObservedSegmentInput,
    StatedSegmentInput,
    build_links,
    build_scenario,
    modellable_links,
    observed_inputs,
)
from app.simulation.schemas import (
    ArmRole,
    ArmStatus,
    BaselineStrategy,
    BetterArm,
    InputBasis,
    LinkInputs,
    LinkStatus,
    RouteArmInput,
    RouteSource,
    SimulationParameters,
    SimulationStatus,
    UnavailableReason,
)
from app.simulation.store import SimulationStore
from app.vision.models import ModelIdentity

T0 = dt.datetime(2024, 3, 14, 8, 30, tzinfo=dt.UTC)

#: 45 km/h over a 550 m segment is 44 s; short enough that a route completes
#: inside a default horizon and long enough that a lag of several steps matters.
SPEED_KPH = 45.0


def parameters(
    *,
    horizon_s: float = 900.0,
    step_s: float = 5.0,
    departure_offset_s: float = 0.0,
    bypass: float = 0.0,
    use_observed: bool = False,
    baseline: BaselineStrategy = BaselineStrategy.SHORTEST_DISTANCE,
) -> SimulationParameters:
    """Parameters for a test run. Every value is stated, none defaulted here."""
    return SimulationParameters(
        horizon_s=horizon_s,
        step_s=step_s,
        step_count=int(horizon_s // step_s),
        departure_offset_s=departure_offset_s,
        emergency_queue_bypass_fraction=bypass,
        use_observed_demand=use_observed,
        baseline_strategy=baseline,
        parameters_version="test-1",
    )


def stated(
    segment_id: str,
    *,
    demand: float | None = None,
    capacity: float | None = None,
    queue: float | None = None,
) -> StatedSegmentInput:
    """A stated assumption for one segment."""
    return StatedSegmentInput(
        segment_id=segment_id,
        note="Stated by the test to exercise a known regime.",
        demand_veh_per_hour=demand,
        capacity_veh_per_hour=capacity,
        initial_queue_veh=queue,
    )


def blocked_state(network_id: str, segment_id: str, incident_id: str) -> NetworkState:
    """A snapshot in which exactly one segment is impassable."""
    return NetworkState(
        state_id="state-blocked",
        network_id=network_id,
        network_version="1.0",
        schema_version="0.1.0",
        as_of=T0,
        assembled_at=T0,
        policy=StateAssemblyPolicy(max_staleness_s=5.0, drop_after_s=30.0),
        segment_states=(
            RoadSegmentState(
                segment_id=segment_id,
                is_blocked=True,
                blocked_by_incident_ids=frozenset({incident_id}),
            ),
        ),
    )


def arm(
    segment_ids: tuple[str, ...],
    *,
    role: ArmRole = ArmRole.EVALUATED,
    origin: str = "J1",
    destination: str = "J3",
    mode: RoutingMode = RoutingMode.NORMAL,
    route_id: str = "route-test",
) -> RouteArmInput:
    """One evaluated route."""
    return RouteArmInput(
        role=role,
        route_id=route_id,
        label="Route under test",
        source=RouteSource.EXPLICIT,
        mode=mode,
        segment_ids=segment_ids,
        origin_junction_id=origin,
        destination_junction_id=destination,
    )


def observed_analytics(
    *,
    area: str,
    arrivals_per_minute: float | None = None,
    stationary: int | None = None,
) -> TrafficAnalytics:
    """A measurement set carrying only the signals the simulator reads.

    Everything else is explicitly unavailable, which is what an uncalibrated
    camera actually produces: counts and arrival rates need no calibration,
    speeds and densities do.
    """
    absent_float: MetricValue[float] = unavailable(
        MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA, "Not measured by this test."
    )
    absent_int: MetricValue[int] = unavailable(
        MetricAvailability.UNAVAILABLE_INSUFFICIENT_DATA, "Not measured by this test."
    )
    tally = ClassTallyOut(by_class={VehicleClass.CAR: 6}, total=6, vehicles=6)
    return TrafficAnalytics(
        analytics_id="analytics-sim",
        mode=ObservationMode.SEQUENCE,
        observation_area_id=area,
        provenance=AnalyticsProvenance(
            media_id="media-1",
            source_id=area,
            timestamp=T0,
            model=ModelIdentity(name="test-detector", backend="test"),
            tracker="test-tracker",
            method="test-method",
            method_version="1.0",
            analytics_schema_version="1.0",
            computed_at=T0,
        ),
        quality=MeasurementQuality(grade=QualityGrade.GOOD, frames_analysed=30),
        frame_width=800,
        frame_height=600,
        detection_tally=tally,
        tracked_object_tally=tally,
        active_tracked_objects=available(6, unit="vehicles"),
        spatial_distribution=SpatialDistribution(
            spread_x_norm=absent_float,
            spread_y_norm=absent_float,
            grid_rows=1,
            grid_cols=1,
            grid_counts=((0,),),
            image_occupancy_ratio=absent_float,
        ),
        mean_image_speed_diagonals_per_s=absent_float,
        mean_trajectory_duration_s=absent_float,
        arrival_rate_per_minute=(
            available(arrivals_per_minute, unit="veh/min")
            if arrivals_per_minute is not None
            else absent_float
        ),
        stationary_vehicle_count=(
            available(stationary, unit="vehicles") if stationary is not None else absent_int
        ),
        total_queue_length_m=absent_float,
        traffic_density_veh_per_km_per_lane=absent_float,
        mean_speed_kph=absent_float,
    )


def mapping_for(area_id: str, segment_id: str = "SEG-12") -> ObservationMapping:
    """A survey binding one observation area to one directed segment."""
    return ObservationMapping(
        mapping_id="map-sim",
        version="1.0",
        network_id="test-net",
        provenance=Provenance(
            source=ValueSource.ASSUMED,
            producer="tests.test_simulation",
            producer_version="1.0",
            observed_at=T0,
            computed_at=T0,
            notes="Synthetic binding for a test; not a survey of anything real.",
        ),
        bindings=(
            ObservationBinding(
                observation_area_id=area_id,
                segment_id=segment_id,
                coverage_note="Test binding.",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class TestPointQueueModel:
    def test_no_queue_forms_below_capacity(self) -> None:
        trace = engine.simulate_link(
            engine.Link(
                segment_id="S1",
                free_flow_traversal_s=40.0,
                capacity_veh_per_hour=1800.0,
                demand_veh_per_hour=600.0,
                initial_queue_veh=0.0,
            ),
            step_s=5.0,
            step_count=180,
        )
        assert trace.peak_queue_veh == 0.0
        assert trace.final_queue_veh == 0.0
        assert trace.total_delay_veh_s == 0.0

    def test_a_queue_grows_when_demand_exceeds_capacity(self) -> None:
        """The regime the horizon exists to bound: an unbounded queue."""
        trace = engine.simulate_link(
            engine.Link(
                segment_id="S1",
                free_flow_traversal_s=40.0,
                capacity_veh_per_hour=900.0,
                demand_veh_per_hour=1400.0,
                initial_queue_veh=0.0,
            ),
            step_s=5.0,
            step_count=180,
        )
        assert trace.saturation_ratio > 1.0
        assert trace.peak_queue_veh > 0.0
        assert trace.final_queue_veh == trace.peak_queue_veh, "queue never recedes"
        assert trace.queue_veh == tuple(sorted(trace.queue_veh)), "monotonic growth"

    def test_vehicles_are_conserved(self) -> None:
        """Everything released is queued, discharged, or still running.

        The single check that would catch a whole class of arithmetic error:
        a model that quietly loses vehicles reports a throughput it never
        achieved and a queue that never formed.
        """
        trace = engine.simulate_link(
            engine.Link(
                segment_id="S1",
                free_flow_traversal_s=40.0,
                capacity_veh_per_hour=900.0,
                demand_veh_per_hour=1400.0,
                initial_queue_veh=12.0,
            ),
            step_s=5.0,
            step_count=180,
        )
        accounted = (
            trace.throughput_veh + trace.final_queue_veh + trace.in_transit_veh
        )
        assert accounted == pytest.approx(trace.entered_veh + 12.0, abs=1e-6)

    def test_an_initial_queue_discharges_at_capacity(self) -> None:
        trace = engine.simulate_link(
            engine.Link(
                segment_id="S1",
                free_flow_traversal_s=10.0,
                capacity_veh_per_hour=3600.0,
                demand_veh_per_hour=0.0,
                initial_queue_veh=10.0,
            ),
            step_s=1.0,
            step_count=20,
        )
        # 3600 veh/h is one vehicle per second, so ten vehicles take ten steps.
        assert trace.queue_veh[9] == 0.0
        assert trace.queue_veh[8] == pytest.approx(1.0)
        assert trace.throughput_veh == pytest.approx(10.0)

    def test_throughput_never_exceeds_capacity(self) -> None:
        trace = engine.simulate_link(
            engine.Link(
                segment_id="S1",
                free_flow_traversal_s=5.0,
                capacity_veh_per_hour=900.0,
                demand_veh_per_hour=5000.0,
                initial_queue_veh=0.0,
            ),
            step_s=5.0,
            step_count=100,
        )
        per_step = 900.0 / 3600.0 * 5.0
        assert max(trace.discharged_veh) <= per_step + 1e-9

    def test_the_delay_is_the_queue_over_the_discharge_rate(self) -> None:
        trace = engine.simulate_link(
            engine.Link(
                segment_id="S1",
                free_flow_traversal_s=5.0,
                capacity_veh_per_hour=900.0,
                demand_veh_per_hour=1800.0,
                initial_queue_veh=0.0,
            ),
            step_s=5.0,
            step_count=60,
        )
        queue = trace.queue_at(100.0)
        assert trace.discharge_delay_at(100.0) == pytest.approx(queue / 0.25)

    def test_the_queue_before_the_first_step_is_the_initial_one(self) -> None:
        trace = engine.simulate_link(
            engine.Link(
                segment_id="S1",
                free_flow_traversal_s=5.0,
                capacity_veh_per_hour=900.0,
                demand_veh_per_hour=900.0,
                initial_queue_veh=4.0,
            ),
            step_s=5.0,
            step_count=10,
        )
        assert trace.queue_at(0.0) == 4.0
        assert trace.queue_at(-1.0) == 4.0

    def test_a_degenerate_step_is_refused(self) -> None:
        link = engine.Link(
            segment_id="S1",
            free_flow_traversal_s=5.0,
            capacity_veh_per_hour=900.0,
            demand_veh_per_hour=900.0,
            initial_queue_veh=0.0,
        )
        with pytest.raises(ValueError, match="at least one step"):
            engine.simulate_link(link, step_s=5.0, step_count=0)
        with pytest.raises(ValueError, match="step_s must be positive"):
            engine.simulate_link(link, step_s=0.0, step_count=5)

    def test_network_order_does_not_depend_on_insertion_order(self) -> None:
        def link(segment_id: str) -> engine.Link:
            return engine.Link(
                segment_id=segment_id,
                free_flow_traversal_s=10.0,
                capacity_veh_per_hour=900.0,
                demand_veh_per_hour=600.0,
                initial_queue_veh=0.0,
            )

        forward = engine.simulate_network(
            {"B": link("B"), "A": link("A")}, step_s=5.0, step_count=10
        )
        backward = engine.simulate_network(
            {"A": link("A"), "B": link("B")}, step_s=5.0, step_count=10
        )
        assert list(forward) == list(backward) == ["A", "B"]


class TestProbeVehicle:
    def test_an_unmodelled_link_adds_no_delay_and_says_so(self) -> None:
        result = engine.simulate_probe(
            (engine.ProbeLeg("S1", 40.0), engine.ProbeLeg("S2", 30.0)),
            {},
            horizon_s=900.0,
        )
        assert result.travel_time_s == 70.0
        assert result.queue_delay_s == 0.0
        assert result.modelled_leg_count == 0
        assert [leg.queue_delay_s for leg in result.legs] == [None, None], (
            "an unmodelled leg reports no delay, which is not the same as zero"
        )

    def test_a_standing_queue_delays_the_vehicle_by_its_discharge_time(self) -> None:
        """Ten vehicles ahead at 900 veh/h is forty seconds of waiting."""
        traces = engine.simulate_network(
            {
                "S1": engine.Link(
                    segment_id="S1",
                    free_flow_traversal_s=40.0,
                    # 900 veh/h discharges 1.25 vehicles per five-second step.
                    capacity_veh_per_hour=900.0,
                    demand_veh_per_hour=0.0,
                    initial_queue_veh=100.0,
                )
            },
            step_s=5.0,
            step_count=180,
        )
        result = engine.simulate_probe(
            (engine.ProbeLeg("S1", 40.0),), traces, horizon_s=900.0
        )
        assert result.legs[0].modelled is True
        # Arriving at t=40 s, after eight steps have each discharged 1.25.
        assert result.legs[0].queue_at_arrival_veh == pytest.approx(90.0)
        assert result.queue_delay_s == pytest.approx(90.0 / 0.25)
        assert result.travel_time_s == pytest.approx(400.0)

    def test_a_vehicle_departing_later_meets_the_queue_that_has_built(self) -> None:
        """The point of a horizon: when you leave changes what you meet."""
        traces = engine.simulate_network(
            {
                "S1": engine.Link(
                    segment_id="S1",
                    free_flow_traversal_s=40.0,
                    capacity_veh_per_hour=900.0,
                    demand_veh_per_hour=1000.0,
                    initial_queue_veh=0.0,
                )
            },
            step_s=5.0,
            step_count=180,
        )
        legs = (engine.ProbeLeg("S1", 40.0),)
        early = engine.simulate_probe(legs, traces, horizon_s=900.0)
        late = engine.simulate_probe(
            legs, traces, horizon_s=900.0, departure_offset_s=600.0
        )
        assert early.queue_delay_s == 0.0, "nothing has queued yet at t=0"
        assert late.queue_delay_s > 0.0
        assert late.completed_within_horizon is True
        assert late.travel_time_s is not None
        assert late.travel_time_s > 40.0

    def test_the_bypass_fraction_reduces_delay_proportionally(self) -> None:
        traces = engine.simulate_network(
            {
                "S1": engine.Link(
                    segment_id="S1",
                    free_flow_traversal_s=40.0,
                    capacity_veh_per_hour=900.0,
                    demand_veh_per_hour=1800.0,
                    initial_queue_veh=0.0,
                )
            },
            step_s=5.0,
            step_count=180,
        )
        legs = (engine.ProbeLeg("S1", 40.0),)
        full = engine.simulate_probe(legs, traces, horizon_s=900.0)
        half = engine.simulate_probe(
            legs, traces, horizon_s=900.0, queue_bypass_fraction=0.5
        )
        clear = engine.simulate_probe(
            legs, traces, horizon_s=900.0, queue_bypass_fraction=1.0
        )
        assert half.queue_delay_s == pytest.approx(full.queue_delay_s / 2)
        assert clear.queue_delay_s == 0.0

    def test_an_out_of_range_bypass_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            engine.simulate_probe(
                (engine.ProbeLeg("S1", 10.0),),
                {},
                horizon_s=100.0,
                queue_bypass_fraction=1.5,
            )

    def test_a_journey_past_the_horizon_reports_no_travel_time(self) -> None:
        result = engine.simulate_probe(
            (engine.ProbeLeg("S1", 400.0),), {}, horizon_s=100.0
        )
        assert result.completed_within_horizon is False
        assert result.travel_time_s is None
        assert result.completion_time_s is None

    def test_a_departure_offset_shifts_arrival_but_not_travel_time(self) -> None:
        result = engine.simulate_probe(
            (engine.ProbeLeg("S1", 40.0),),
            {},
            horizon_s=900.0,
            departure_offset_s=120.0,
        )
        assert result.travel_time_s == 40.0
        assert result.completion_time_s == 160.0


# ---------------------------------------------------------------------------
# Scenario inputs
# ---------------------------------------------------------------------------
class TestScenarioInputs:
    def test_a_segment_without_capacity_is_unmodellable_and_says_why(self) -> None:
        network = square_network(free_flow_speed_kph=SPEED_KPH)
        links = build_links(network, network_state=None, mode=RoutingMode.NORMAL)
        link = next(item for item in links if item.segment_id == "SEG-12")
        assert link.capacity_veh_per_hour is None
        assert link.capacity_basis is InputBasis.UNAVAILABLE
        assert "no surveyed capacity" in (link.capacity_reason or "")
        assert link.is_modellable is False

    def test_a_segment_without_a_speed_has_no_traversal_time(self) -> None:
        """No free-flow speed, no travel time. No speed is assumed in its place."""
        network = square_network()
        links = build_links(network, network_state=None, mode=RoutingMode.NORMAL)
        assert all(item.free_flow_traversal_s is None for item in links)
        assert all(item.time_basis is TimeBasis.UNAVAILABLE for item in links)

    def test_a_stated_input_makes_a_link_modellable(self) -> None:
        network = square_network(free_flow_speed_kph=SPEED_KPH)
        links = build_links(
            network,
            network_state=None,
            mode=RoutingMode.NORMAL,
            stated=(stated("SEG-12", demand=600.0, capacity=1800.0),),
        )
        link = next(item for item in links if item.segment_id == "SEG-12")
        assert link.is_modellable
        assert link.capacity_basis is InputBasis.STATED
        assert link.demand_basis is InputBasis.STATED
        assert "Stated by the test" in (link.demand_reason or "")

    def test_a_stated_capacity_overrides_a_surveyed_one_visibly(self) -> None:
        from network_fixtures import measure

        from app.domain.provenance import Unit

        base = square_network(free_flow_speed_kph=SPEED_KPH)
        surveyed = tuple(
            segment.model_copy(
                update={
                    "capacity_veh_per_hour": measure(1800.0, Unit.VEHICLE_PER_HOUR)
                }
            )
            for segment in base.segments
        )
        network = base.model_copy(update={"segments": surveyed})

        plain = build_links(network, network_state=None, mode=RoutingMode.NORMAL)
        assert next(
            item for item in plain if item.segment_id == "SEG-12"
        ).capacity_basis is InputBasis.SURVEYED

        overridden = build_links(
            network,
            network_state=None,
            mode=RoutingMode.NORMAL,
            stated=(stated("SEG-12", capacity=600.0),),
        )
        link = next(item for item in overridden if item.segment_id == "SEG-12")
        assert link.capacity_veh_per_hour == 600.0
        assert link.capacity_basis is InputBasis.STATED
        assert "overriding the surveyed 1800 veh/h" in (link.capacity_reason or "")

    def test_stated_demand_wins_over_observed_and_both_are_recorded(self) -> None:
        network = square_network(free_flow_speed_kph=SPEED_KPH)
        observed = {
            "SEG-12": ObservedSegmentInput(
                segment_id="SEG-12",
                observation_area_id="area-1",
                demand_veh_per_hour=300.0,
                demand_note="observed",
            )
        }
        links = build_links(
            network,
            network_state=None,
            mode=RoutingMode.NORMAL,
            stated=(stated("SEG-12", demand=900.0),),
            observed=observed,
        )
        link = next(item for item in links if item.segment_id == "SEG-12")
        assert link.demand_veh_per_hour == 900.0
        assert link.demand_basis is InputBasis.STATED
        assert link.observation_area_id == "area-1", "the binding is still reported"

    def test_observed_demand_can_be_switched_off(self) -> None:
        network = square_network(free_flow_speed_kph=SPEED_KPH)
        observed = {
            "SEG-12": ObservedSegmentInput(
                segment_id="SEG-12",
                observation_area_id="area-1",
                demand_veh_per_hour=300.0,
                demand_note="observed",
            )
        }
        links = build_links(
            network,
            network_state=None,
            mode=RoutingMode.NORMAL,
            observed=observed,
            use_observed_demand=False,
        )
        link = next(item for item in links if item.segment_id == "SEG-12")
        assert link.demand_veh_per_hour is None
        assert link.demand_basis is InputBasis.UNAVAILABLE
        assert "switched off" in (link.demand_reason or "")

    def test_observations_reach_a_segment_only_through_the_mapping(self) -> None:
        """No mapping, no attribution. The Phase 7 rule, unchanged."""
        assert observed_inputs((), None) == {}

    def test_an_observed_arrival_rate_becomes_demand_with_its_caveat(self) -> None:
        """The one path by which measured traffic reaches the model."""
        measured = observed_analytics(
            area="observation-area:cam-1", arrivals_per_minute=12.0, stationary=4
        )
        resolved = observed_inputs((measured,), mapping_for("observation-area:cam-1"))
        assert set(resolved) == {"SEG-12"}
        entry = resolved["SEG-12"]
        assert entry.demand_veh_per_hour == 720.0, "twelve a minute is 720 an hour"
        assert entry.initial_queue_veh == 4.0
        assert "not a counting-line flow" in (entry.demand_note or "")

    def test_an_unbound_observation_contributes_nothing(self) -> None:
        measured = observed_analytics(
            area="observation-area:unbound", arrivals_per_minute=12.0
        )
        assert observed_inputs((measured,), mapping_for("observation-area:cam-1")) == {}

    def test_an_observation_with_no_arrival_rate_contributes_nothing(self) -> None:
        measured = observed_analytics(area="observation-area:cam-1")
        assert observed_inputs((measured,), mapping_for("observation-area:cam-1")) == {}

    def test_observed_demand_reaches_the_link_and_is_labelled_observed(self) -> None:
        network = square_network(free_flow_speed_kph=SPEED_KPH)
        measured = observed_analytics(
            area="observation-area:cam-1", arrivals_per_minute=10.0, stationary=3
        )
        links = build_links(
            network,
            network_state=None,
            mode=RoutingMode.NORMAL,
            observed=observed_inputs((measured,), mapping_for("observation-area:cam-1")),
            use_observed_demand=True,
        )
        link = next(item for item in links if item.segment_id == "SEG-12")
        assert link.demand_veh_per_hour == 600.0
        assert link.demand_basis is InputBasis.OBSERVED
        assert link.initial_queue_basis is InputBasis.OBSERVED
        assert link.observation_area_id == "observation-area:cam-1"

    def test_a_blocked_segment_is_carried_into_the_link_inputs(self) -> None:
        network = square_network(free_flow_speed_kph=SPEED_KPH)
        state = blocked_state(network.network_id, "SEG-23", "INC-1")
        links = build_links(network, network_state=state, mode=RoutingMode.NORMAL)
        link = next(item for item in links if item.segment_id == "SEG-23")
        assert link.is_blocked is True
        assert link.blocked_by_incident_ids == ("INC-1",)

    def test_a_value_without_a_basis_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unavailable basis"):
            LinkInputs(
                segment_id="S1",
                length_m=100.0,
                lane_count=1,
                demand_veh_per_hour=600.0,
                demand_basis=InputBasis.UNAVAILABLE,
            )

    def test_a_basis_without_a_value_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="claims basis"):
            LinkInputs(
                segment_id="S1",
                length_m=100.0,
                lane_count=1,
                capacity_basis=InputBasis.SURVEYED,
            )

    def test_an_unmodellable_link_never_reaches_the_model(self) -> None:
        network = square_network(free_flow_speed_kph=SPEED_KPH)
        links = build_links(
            network,
            network_state=None,
            mode=RoutingMode.NORMAL,
            stated=(stated("SEG-12", demand=600.0, capacity=1800.0),),
        )
        runnable = modellable_links(links)
        assert set(runnable) == {"SEG-12"}


class TestFingerprint:
    def test_identical_inputs_hash_identically(self) -> None:
        network = square_network(free_flow_speed_kph=SPEED_KPH)
        links = build_links(network, network_state=None, mode=RoutingMode.NORMAL)
        arms = (arm(("SEG-12", "SEG-23")),)
        first = build_scenario(
            network=network, parameters=parameters(), links=links, arms=arms
        )
        second = build_scenario(
            network=network, parameters=parameters(), links=links, arms=arms
        )
        assert first.input_fingerprint == second.input_fingerprint
        assert first.scenario_id == second.scenario_id

    def test_a_changed_assumption_changes_the_fingerprint(self) -> None:
        network = square_network(free_flow_speed_kph=SPEED_KPH)
        arms = (arm(("SEG-12", "SEG-23")),)
        quiet = build_scenario(
            network=network,
            parameters=parameters(),
            links=build_links(
                network,
                network_state=None,
                mode=RoutingMode.NORMAL,
                stated=(stated("SEG-12", demand=200.0, capacity=1800.0),),
            ),
            arms=arms,
        )
        busy = build_scenario(
            network=network,
            parameters=parameters(),
            links=build_links(
                network,
                network_state=None,
                mode=RoutingMode.NORMAL,
                stated=(stated("SEG-12", demand=1900.0, capacity=1800.0),),
            ),
            arms=arms,
        )
        assert quiet.input_fingerprint != busy.input_fingerprint

    def test_the_traffic_state_id_does_not_disturb_the_fingerprint(self) -> None:
        """Context, not input: the model never reads an interpreted band.

        The id embeds the instant its snapshot was assembled, so hashing it
        would make two identical scenarios differ purely because a clock moved.
        What an observation actually contributes -- a demand, an initial queue --
        is resolved onto the links, and those are hashed.
        """
        network = square_network(free_flow_speed_kph=SPEED_KPH)
        links = build_links(network, network_state=None, mode=RoutingMode.NORMAL)
        arms = (arm(("SEG-12", "SEG-23")),)
        first = build_scenario(
            network=network,
            parameters=parameters(),
            links=links,
            arms=arms,
            traffic_state_id="netstate-aaa-1710404000000",
        )
        second = build_scenario(
            network=network,
            parameters=parameters(),
            links=links,
            arms=arms,
            traffic_state_id="netstate-bbb-1710407600000",
        )
        assert first.input_fingerprint == second.input_fingerprint
        assert first.traffic_state_id != second.traffic_state_id

    def test_a_changed_horizon_changes_the_fingerprint(self) -> None:
        network = square_network(free_flow_speed_kph=SPEED_KPH)
        links = build_links(network, network_state=None, mode=RoutingMode.NORMAL)
        arms = (arm(("SEG-12", "SEG-23")),)
        short = build_scenario(
            network=network,
            parameters=parameters(horizon_s=300.0),
            links=links,
            arms=arms,
        )
        long = build_scenario(
            network=network,
            parameters=parameters(horizon_s=900.0),
            links=links,
            arms=arms,
        )
        assert short.input_fingerprint != long.input_fingerprint


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def build_run(
    *,
    network_state: NetworkState | None = None,
    stated_inputs: tuple[StatedSegmentInput, ...] = (),
    arms: tuple[RouteArmInput, ...] | None = None,
    params: SimulationParameters | None = None,
    at: dt.datetime = T0,
    speed_kph: float | None = SPEED_KPH,
):
    """Assemble and run a scenario over the square network."""
    network = square_network(free_flow_speed_kph=speed_kph)
    resolved = params or parameters()
    links = build_links(
        network,
        network_state=network_state,
        mode=RoutingMode.NORMAL,
        stated=stated_inputs,
    )
    scenario = build_scenario(
        network=network,
        parameters=resolved,
        links=links,
        arms=arms if arms is not None else (arm(("SEG-12", "SEG-23")),),
    )
    return runner.run(scenario, network.build_index(), at=at)


#: A congested southern route and a clear northern one, for comparisons.
CONGESTED = (
    stated("SEG-14", demand=500.0, capacity=1800.0),
    stated("SEG-43", demand=1600.0, capacity=900.0),
    stated("SEG-12", demand=600.0, capacity=1800.0),
    stated("SEG-23", demand=600.0, capacity=1800.0),
)


class TestSimulationRun:
    def test_a_run_completes_and_reports_a_travel_time(self) -> None:
        result = build_run(stated_inputs=CONGESTED)
        assert result.status is SimulationStatus.COMPLETED
        evaluated = result.routes[0]
        assert evaluated.status is ArmStatus.SIMULATED
        assert evaluated.travel_time_s is not None
        assert evaluated.travel_time_s.value > 0

    def test_the_travel_time_is_free_flow_plus_the_modelled_delay(self) -> None:
        result = build_run(
            stated_inputs=CONGESTED, arms=(arm(("SEG-14", "SEG-43")),)
        )
        route = result.routes[0]
        assert route.travel_time_s is not None
        assert route.free_flow_travel_time_s is not None
        assert route.delay_s is not None
        assert route.travel_time_s.value == pytest.approx(
            route.free_flow_travel_time_s.value + route.delay_s.value
        )
        assert route.delay_s.value > 0, "the congested route must be delayed"

    def test_the_completion_time_includes_the_departure_offset(self) -> None:
        result = build_run(
            stated_inputs=CONGESTED, params=parameters(departure_offset_s=100.0)
        )
        route = result.routes[0]
        assert route.completion_time_s is not None
        assert route.travel_time_s is not None
        assert route.completion_time_s.value == pytest.approx(
            route.travel_time_s.value + 100.0
        )

    def test_queue_and_throughput_are_reported_where_supported(self) -> None:
        result = build_run(stated_inputs=CONGESTED)
        modelled = {
            item.segment_id: item
            for item in result.links
            if item.status is LinkStatus.MODELLED
        }
        assert set(modelled) == {"SEG-12", "SEG-14", "SEG-23", "SEG-43"}
        congested = modelled["SEG-43"]
        assert congested.peak_queue_veh is not None
        assert congested.peak_queue_veh.value > 0
        assert congested.throughput_veh is not None
        assert congested.throughput_veh.value > 0
        assert congested.saturation_ratio is not None
        assert congested.saturation_ratio.value > 1.0
        assert len(congested.queue_series_veh) == 180

    def test_a_link_without_inputs_reports_no_queue_at_all(self) -> None:
        result = build_run(stated_inputs=CONGESTED)
        silent = next(item for item in result.links if item.segment_id == "SEG-21")
        assert silent.status is LinkStatus.NOT_MODELLED
        assert silent.peak_queue_veh is None
        assert silent.throughput_veh is None
        assert silent.reason and "no surveyed capacity" in silent.reason

    def test_the_network_summary_scopes_itself_to_modelled_links(self) -> None:
        result = build_run(stated_inputs=CONGESTED)
        assert result.network is not None
        assert result.network.modelled_link_count == 4
        assert result.network.unmodelled_link_count == 4
        assert result.network.oversaturated_segment_ids == ("SEG-43",)
        assert "not over it" in result.network.note

    def test_a_run_with_no_modellable_link_says_so_without_figures(self) -> None:
        result = build_run()
        assert result.network is not None
        assert result.network.modelled_link_count == 0
        assert result.network.modelled_throughput_veh is None
        assert "No link could be modelled" in result.network.note
        assert result.routes[0].status is ArmStatus.SIMULATED, (
            "a route still runs at free flow when nothing queues"
        )
        assert result.routes[0].unmodelled_segment_ids == ("SEG-12", "SEG-23")


class TestEmergencyArm:
    """The one emergency-specific term in the model, and its default."""

    def test_no_emergency_advantage_is_modelled_by_default(self) -> None:
        normal = build_run(
            stated_inputs=CONGESTED,
            arms=(arm(("SEG-14", "SEG-43"), mode=RoutingMode.NORMAL),),
            params=parameters(departure_offset_s=300.0),
        )
        emergency = build_run(
            stated_inputs=CONGESTED,
            arms=(arm(("SEG-14", "SEG-43"), mode=RoutingMode.EMERGENCY),),
            params=parameters(departure_offset_s=300.0),
        )
        assert normal.routes[0].travel_time_s is not None
        assert emergency.routes[0].travel_time_s is not None
        assert (
            emergency.routes[0].travel_time_s.value
            == normal.routes[0].travel_time_s.value
        ), "an emergency vehicle gets no modelled advantage nobody has measured"

    def test_a_stated_bypass_applies_only_to_the_emergency_arm(self) -> None:
        params = parameters(departure_offset_s=300.0, bypass=0.5)
        normal = build_run(
            stated_inputs=CONGESTED,
            arms=(arm(("SEG-14", "SEG-43"), mode=RoutingMode.NORMAL),),
            params=params,
        )
        emergency = build_run(
            stated_inputs=CONGESTED,
            arms=(arm(("SEG-14", "SEG-43"), mode=RoutingMode.EMERGENCY),),
            params=params,
        )
        assert normal.routes[0].delay_s is not None
        assert emergency.routes[0].delay_s is not None
        assert emergency.routes[0].delay_s.value == pytest.approx(
            normal.routes[0].delay_s.value / 2
        )

    def test_a_stated_bypass_is_surfaced_as_an_assumption(self) -> None:
        result = build_run(
            stated_inputs=CONGESTED,
            arms=(arm(("SEG-14", "SEG-43"), mode=RoutingMode.EMERGENCY),),
            params=parameters(departure_offset_s=300.0, bypass=0.5),
        )
        assert any(
            "queue bypass of 50%" in warning for warning in result.routes[0].warnings
        ), "a modelled advantage must be stated as an assumption, not applied silently"


class TestDeterminism:
    def test_two_runs_of_identical_input_are_byte_identical(self) -> None:
        first = build_run(stated_inputs=CONGESTED, at=T0)
        second = build_run(stated_inputs=CONGESTED, at=T0)
        assert first.model_dump_json() == second.model_dump_json()

    def test_the_run_id_is_the_fingerprint_of_the_inputs(self) -> None:
        first = build_run(stated_inputs=CONGESTED, at=T0)
        later = build_run(
            stated_inputs=CONGESTED, at=T0 + dt.timedelta(days=400)
        )
        assert first.run_id == later.run_id
        assert first.run_id.endswith(first.scenario.input_fingerprint[:12])

    def test_only_the_timestamps_move_when_the_clock_does(self) -> None:
        """A later run of the same scenario reports the same traffic."""
        first = build_run(stated_inputs=CONGESTED, at=T0)
        later = build_run(stated_inputs=CONGESTED, at=T0 + dt.timedelta(hours=9))

        def figures(result: object) -> str:
            payload = json.loads(result.model_dump_json())  # type: ignore[attr-defined]

            def strip(node: object) -> object:
                if isinstance(node, dict):
                    return {
                        key: strip(value)
                        for key, value in node.items()
                        if key not in ("observed_at", "computed_at")
                    }
                if isinstance(node, list):
                    return [strip(item) for item in node]
                return node

            return json.dumps(strip(payload), sort_keys=True)

        assert figures(first) == figures(later)

    def test_a_different_assumption_gives_a_different_answer(self) -> None:
        """Determinism is not constancy: the model must still respond to input."""
        quiet = build_run(
            stated_inputs=(
                stated("SEG-12", demand=100.0, capacity=1800.0),
                stated("SEG-23", demand=100.0, capacity=1800.0),
            )
        )
        jammed = build_run(
            stated_inputs=(
                stated("SEG-12", demand=3000.0, capacity=900.0),
                stated("SEG-23", demand=3000.0, capacity=900.0),
            )
        )
        assert quiet.run_id != jammed.run_id
        assert quiet.routes[0].travel_time_s is not None
        assert jammed.routes[0].travel_time_s is not None
        assert jammed.routes[0].travel_time_s.value > quiet.routes[0].travel_time_s.value

    def test_the_model_declares_itself_deterministic(self) -> None:
        result = build_run()
        assert result.scenario.parameters.is_stochastic is False
        assert "seed" not in result.model_dump_json().lower()


class TestUnavailableInputs:
    def test_a_route_over_a_blocked_segment_produces_no_travel_time(self) -> None:
        network = square_network(free_flow_speed_kph=SPEED_KPH)
        result = build_run(
            network_state=blocked_state(network.network_id, "SEG-23", "INC-9"),
            stated_inputs=CONGESTED,
        )
        route = result.routes[0]
        assert route.status is ArmStatus.UNAVAILABLE
        assert route.travel_time_s is None
        assert route.rejections[0].reason is UnavailableReason.SEGMENT_BLOCKED
        assert route.rejections[0].segment_ids == ("SEG-23",)
        assert route.rejections[0].incident_ids == ("INC-9",)

    def test_a_route_with_no_time_basis_produces_no_travel_time(self) -> None:
        result = build_run(speed_kph=None)
        route = result.routes[0]
        assert route.status is ArmStatus.UNAVAILABLE
        assert route.travel_time_s is None
        assert route.rejections[0].reason is UnavailableReason.NO_TRAVERSAL_TIME_BASIS
        assert "No speed was assumed" in route.rejections[0].message

    def test_a_journey_past_the_horizon_reports_no_arrival(self) -> None:
        result = build_run(
            stated_inputs=CONGESTED, params=parameters(horizon_s=30.0, step_s=5.0)
        )
        route = result.routes[0]
        assert route.status is ArmStatus.UNAVAILABLE
        assert route.travel_time_s is None
        assert route.completion_time_s is None
        assert route.rejections[0].reason is UnavailableReason.HORIZON_EXCEEDED
        assert route.legs, "how far it got is still reported"

    def test_a_route_off_the_network_is_refused(self) -> None:
        result = build_run(arms=(arm(("SEG-12", "SEG-NOWHERE")),))
        route = result.routes[0]
        assert route.status is ArmStatus.UNAVAILABLE
        assert route.rejections[0].reason is UnavailableReason.ROUTE_NOT_ON_NETWORK
        assert route.rejections[0].segment_ids == ("SEG-NOWHERE",)

    def test_a_discontinuous_route_is_refused(self) -> None:
        result = build_run(arms=(arm(("SEG-12", "SEG-34")),))
        route = result.routes[0]
        assert route.status is ArmStatus.UNAVAILABLE
        assert route.rejections[0].reason is UnavailableReason.ROUTE_DISCONTINUOUS

    def test_a_route_that_does_not_start_where_it_claims_is_refused(self) -> None:
        result = build_run(arms=(arm(("SEG-23",), origin="J1", destination="J3"),))
        route = result.routes[0]
        assert route.status is ArmStatus.UNAVAILABLE
        assert route.rejections[0].reason is UnavailableReason.ROUTE_DISCONTINUOUS


class TestComparison:
    def test_the_faster_alternative_wins_with_a_stated_margin(self) -> None:
        result = build_run(
            stated_inputs=CONGESTED,
            arms=(
                arm(
                    ("SEG-14", "SEG-43"),
                    role=ArmRole.BASELINE,
                    route_id="route-baseline",
                ),
                arm(
                    ("SEG-12", "SEG-23"),
                    role=ArmRole.RECOMMENDED,
                    route_id="route-recommended",
                ),
            ),
        )
        comparison = result.comparison
        assert comparison is not None
        assert comparison.comparable is True
        assert comparison.better is BetterArm.ALTERNATIVE
        assert comparison.travel_time_delta_s is not None
        assert comparison.travel_time_delta_s.value > 0
        assert comparison.improvement_pct is not None
        assert comparison.improvement_pct > 0

    def test_the_delta_is_the_difference_between_the_two_arms(self) -> None:
        result = build_run(
            stated_inputs=CONGESTED,
            arms=(
                arm(("SEG-14", "SEG-43"), role=ArmRole.BASELINE, route_id="r-base"),
                arm(("SEG-12", "SEG-23"), role=ArmRole.RECOMMENDED, route_id="r-alt"),
            ),
        )
        baseline = result.arm(ArmRole.BASELINE)
        recommended = result.arm(ArmRole.RECOMMENDED)
        assert baseline is not None and recommended is not None
        assert baseline.travel_time_s is not None
        assert recommended.travel_time_s is not None
        assert result.comparison is not None
        assert result.comparison.travel_time_delta_s is not None
        assert result.comparison.travel_time_delta_s.value == pytest.approx(
            baseline.travel_time_s.value - recommended.travel_time_s.value
        )

    def test_the_same_road_twice_reports_no_difference(self) -> None:
        result = build_run(
            stated_inputs=CONGESTED,
            arms=(
                arm(("SEG-12", "SEG-23"), role=ArmRole.BASELINE, route_id="r-base"),
                arm(("SEG-12", "SEG-23"), role=ArmRole.RECOMMENDED, route_id="r-alt"),
            ),
        )
        assert result.comparison is not None
        assert result.comparison.better is BetterArm.NO_DIFFERENCE
        assert result.comparison.improvement_pct == pytest.approx(0.0)

    def test_a_slower_alternative_is_reported_as_worse(self) -> None:
        result = build_run(
            stated_inputs=CONGESTED,
            arms=(
                arm(("SEG-12", "SEG-23"), role=ArmRole.BASELINE, route_id="r-base"),
                arm(("SEG-14", "SEG-43"), role=ArmRole.RECOMMENDED, route_id="r-alt"),
            ),
        )
        assert result.comparison is not None
        assert result.comparison.better is BetterArm.BASELINE
        assert result.comparison.improvement_pct is not None
        assert result.comparison.improvement_pct < 0, (
            "a worse recommendation must read as worse, not be clamped at zero"
        )

    def test_an_unavailable_arm_produces_no_improvement_figure(self) -> None:
        network = square_network(free_flow_speed_kph=SPEED_KPH)
        result = build_run(
            network_state=blocked_state(network.network_id, "SEG-43", "INC-3"),
            stated_inputs=CONGESTED,
            arms=(
                arm(("SEG-14", "SEG-43"), role=ArmRole.BASELINE, route_id="r-base"),
                arm(("SEG-12", "SEG-23"), role=ArmRole.RECOMMENDED, route_id="r-alt"),
            ),
        )
        comparison = result.comparison
        assert comparison is not None
        assert comparison.comparable is False
        assert comparison.travel_time_delta_s is None
        assert comparison.improvement_pct is None
        assert comparison.better is BetterArm.UNKNOWN
        assert comparison.rejections[0].reason is UnavailableReason.ARM_UNAVAILABLE


class TestNoFabricatedValues:
    """The point of the phase: a modelled number must never read as a measured one."""

    @staticmethod
    def _measurements(node: object) -> list[dict]:
        found: list[dict] = []
        if isinstance(node, dict):
            if {"value", "unit", "provenance"} <= set(node):
                found.append(node)
            for value in node.values():
                found.extend(TestNoFabricatedValues._measurements(value))
        elif isinstance(node, list):
            for item in node:
                found.extend(TestNoFabricatedValues._measurements(item))
        return found

    def test_no_output_claims_to_be_observed(self) -> None:
        result = build_run(
            stated_inputs=CONGESTED,
            arms=(
                arm(("SEG-14", "SEG-43"), role=ArmRole.BASELINE, route_id="r-base"),
                arm(("SEG-12", "SEG-23"), role=ArmRole.RECOMMENDED, route_id="r-alt"),
            ),
        )
        payload = json.loads(result.model_dump_json())
        sources = {
            item["provenance"]["source"] for item in self._measurements(payload)
        }
        assert sources, "the run must produce measurements to check"
        assert sources <= {ValueSource.SIMULATED.value, ValueSource.DERIVED.value}, (
            f"a simulated run must not claim observed or predicted values: {sources}"
        )

    def test_every_outcome_is_simulated_rather_than_factual(self) -> None:
        result = build_run(stated_inputs=CONGESTED)
        route = result.routes[0]
        assert route.travel_time_s is not None
        assert route.travel_time_s.provenance.source is ValueSource.SIMULATED
        assert route.travel_time_s.is_factual is False
        assert result.provenance.source is ValueSource.SIMULATED

    def test_the_free_flow_reference_is_derived_not_simulated(self) -> None:
        """The reference is arithmetic over surveyed values; the outcome is not."""
        result = build_run(stated_inputs=CONGESTED)
        route = result.routes[0]
        assert route.free_flow_travel_time_s is not None
        assert route.free_flow_travel_time_s.provenance.source is ValueSource.DERIVED

    def test_an_unavailable_arm_carries_no_derived_travel_figure(self) -> None:
        result = build_run(speed_kph=None)
        route = result.routes[0]
        assert route.travel_time_s is None
        assert route.delay_s is None
        assert route.completion_time_s is None
        assert route.mean_speed_kph is None

    def test_absent_inputs_are_named_rather_than_defaulted(self) -> None:
        result = build_run(stated_inputs=CONGESTED)
        absent = [
            link
            for link in result.scenario.links
            if link.demand_basis is InputBasis.UNAVAILABLE
        ]
        assert absent, "the square network has segments with no stated demand"
        for link in absent:
            assert link.demand_veh_per_hour is None
            assert link.demand_reason
            assert "not a finding that the road is empty" in link.demand_reason

    def test_the_run_warns_that_unmodelled_links_understate_congestion(self) -> None:
        result = build_run(stated_inputs=CONGESTED)
        assert any("understates congestion" in item for item in result.warnings)


class TestSimulationStore:
    def test_a_run_can_be_retrieved_by_its_id(self) -> None:
        store = SimulationStore(max_runs=5)
        result = build_run(stated_inputs=CONGESTED)
        store.put(result)
        assert store.get(result.run_id).run_id == result.run_id
        assert store.count == 1

    def test_an_identical_run_replaces_rather_than_duplicates(self) -> None:
        store = SimulationStore(max_runs=5)
        store.put(build_run(stated_inputs=CONGESTED, at=T0))
        store.put(build_run(stated_inputs=CONGESTED, at=T0 + dt.timedelta(hours=1)))
        assert store.count == 1

    def test_the_oldest_run_is_evicted_first(self) -> None:
        store = SimulationStore(max_runs=1)
        first = build_run(
            stated_inputs=(stated("SEG-12", demand=100.0, capacity=1800.0),)
        )
        second = build_run(
            stated_inputs=(stated("SEG-12", demand=200.0, capacity=1800.0),)
        )
        store.put(first)
        store.put(second)
        assert store.count == 1
        assert store.list()[0].run_id == second.run_id

    def test_an_unknown_id_explains_itself(self) -> None:
        from app.core.errors import ResourceNotFoundError

        store = SimulationStore()
        with pytest.raises(ResourceNotFoundError, match="kept in memory"):
            store.get("sim-nope")

    def test_a_degenerate_ceiling_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            SimulationStore(max_runs=0)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@pytest.fixture
def sim_client(tmp_path: Path) -> Iterator[TestClient]:
    """A client with the square network loaded, ready to simulate."""
    network = square_network(free_flow_speed_kph=SPEED_KPH)
    (tmp_path / "net.json").write_text(
        json.dumps(network.model_dump(mode="json")), encoding="utf-8"
    )
    settings = Settings(app_env="local", log_level="WARNING", networks_dir=tmp_path)
    with TestClient(create_app(settings)) as client:
        client.post("/api/network/load", json={"relative_path": "net.json"})
        yield client


#: The same congested-south / clear-north scenario, as a request body.
SEGMENT_INPUTS = [
    {
        "segment_id": "SEG-14",
        "demand_veh_per_hour": 500.0,
        "capacity_veh_per_hour": 1800.0,
        "note": "Stated off-peak demand for this what-if.",
    },
    {
        "segment_id": "SEG-43",
        "demand_veh_per_hour": 1600.0,
        "capacity_veh_per_hour": 900.0,
        "note": "Stated peak demand against a lane closure for this what-if.",
    },
    {
        "segment_id": "SEG-12",
        "demand_veh_per_hour": 600.0,
        "capacity_veh_per_hour": 1800.0,
        "note": "Stated off-peak demand for this what-if.",
    },
    {
        "segment_id": "SEG-23",
        "demand_veh_per_hour": 600.0,
        "capacity_veh_per_hour": 1800.0,
        "note": "Stated off-peak demand for this what-if.",
    },
]


class TestSimulationApi:
    def test_config_states_the_model_and_its_limitations(
        self, sim_client: TestClient
    ) -> None:
        body = sim_client.get("/api/simulation/config").json()
        assert body["model_name"] == "deterministic-point-queue"
        assert body["is_stochastic"] is False
        assert body["horizon_s"] > 0
        assert any("vertical" in item for item in body["limitations"])
        assert any("origin-destination" in item for item in body["limitations"])

    def test_simulating_without_a_network_is_a_404(self, client: TestClient) -> None:
        response = client.post("/api/simulation/run", json={})
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "network_not_loaded"

    def test_an_explicit_route_is_simulated(self, sim_client: TestClient) -> None:
        response = sim_client.post(
            "/api/simulation/run",
            json={
                "route_segment_ids": ["SEG-14", "SEG-43"],
                "segment_inputs": SEGMENT_INPUTS,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "completed"
        route = body["routes"][0]
        assert route["role"] == "evaluated"
        assert route["status"] == "simulated"
        assert route["travel_time_s"]["provenance"]["source"] == "simulated"
        assert route["delay_s"]["value"] > 0

    def test_a_run_with_nothing_to_evaluate_is_unavailable_not_empty(
        self, sim_client: TestClient
    ) -> None:
        body = sim_client.post("/api/simulation/run", json={}).json()
        assert body["status"] == "unavailable"
        assert body["routes"] == []
        assert body["rejections"][0]["reason"] == "no_route_supplied"

    def test_the_optimizer_supplies_the_route_when_asked(
        self, sim_client: TestClient
    ) -> None:
        body = sim_client.post(
            "/api/simulation/run",
            json={
                "origin_junction_id": "J1",
                "destination_junction_id": "J3",
                "segment_inputs": SEGMENT_INPUTS,
            },
        ).json()
        assert body["status"] == "completed"
        assert body["routes"][0]["source"] == "optimizer_selected"
        assert body["scenario"]["traffic_state_id"] is not None

    def test_comparing_explicit_routes_reports_the_difference(
        self, sim_client: TestClient
    ) -> None:
        body = sim_client.post(
            "/api/simulation/compare",
            json={
                "baseline_segment_ids": ["SEG-14", "SEG-43"],
                "alternative_segment_ids": ["SEG-12", "SEG-23"],
                "segment_inputs": SEGMENT_INPUTS,
            },
        ).json()
        assert body["status"] == "completed"
        assert {route["role"] for route in body["routes"]} == {
            "baseline",
            "recommended",
        }
        comparison = body["comparison"]
        assert comparison["comparable"] is True
        assert comparison["better"] == "alternative"
        assert comparison["improvement_pct"] > 0
        assert comparison["travel_time_delta_s"]["provenance"]["source"] == "simulated"

    def test_comparing_against_the_shortest_distance_baseline(
        self, sim_client: TestClient
    ) -> None:
        body = sim_client.post(
            "/api/simulation/compare",
            json={
                "origin_junction_id": "J1",
                "destination_junction_id": "J3",
                "segment_inputs": SEGMENT_INPUTS,
            },
        ).json()
        assert body["status"] == "completed"
        baseline = next(r for r in body["routes"] if r["role"] == "baseline")
        assert baseline["source"] == "shortest_distance_search"
        assert body["scenario"]["parameters"]["baseline_strategy"] == "shortest_distance"

    def test_a_blocked_route_reports_no_travel_time(
        self, sim_client: TestClient
    ) -> None:
        declared = sim_client.post(
            "/api/incidents",
            json={
                "incident_type": "accident",
                "severity": "major",
                "location": {"segment_id": "SEG-43"},
                "blocks_segment_ids": ["SEG-43"],
                "reported_by": "operator.test",
            },
        )
        assert declared.status_code == 201

        body = sim_client.post(
            "/api/simulation/run",
            json={
                "route_segment_ids": ["SEG-14", "SEG-43"],
                "segment_inputs": SEGMENT_INPUTS,
            },
        ).json()
        route = body["routes"][0]
        assert route["status"] == "unavailable"
        assert route["travel_time_s"] is None
        assert route["rejections"][0]["reason"] == "segment_blocked"
        assert route["rejections"][0]["segment_ids"] == ["SEG-43"]

    def test_an_emergency_route_is_simulated_from_its_own_record(
        self, sim_client: TestClient
    ) -> None:
        declared = sim_client.post(
            "/api/emergency/declare",
            json={
                "vehicle": {"vehicle_id": "AMB-1", "vehicle_type": "ambulance"},
                "priority": "P1_immediate",
                "current_position": {"junction_id": "J1"},
                "destination": {"facility_id": "FAC-HOSP-1"},
                "declared_by": "operator.test",
            },
        )
        assert declared.status_code == 201, declared.text
        emergency_id = declared.json()["emergency"]["emergency_id"]

        body = sim_client.post(
            "/api/simulation/run",
            json={"emergency_id": emergency_id, "segment_inputs": SEGMENT_INPUTS},
        ).json()
        assert body["status"] == "completed"
        route = body["routes"][0]
        assert route["mode"] == "emergency"
        assert route["status"] == "simulated"
        assert (
            body["scenario"]["parameters"]["emergency_queue_bypass_fraction"] == 0.0
        ), "no emergency advantage is modelled unless an operator states one"

    def test_repeated_identical_requests_return_the_same_run(
        self, sim_client: TestClient
    ) -> None:
        body = {
            "route_segment_ids": ["SEG-14", "SEG-43"],
            "segment_inputs": SEGMENT_INPUTS,
        }
        first = sim_client.post("/api/simulation/run", json=body).json()
        second = sim_client.post("/api/simulation/run", json=body).json()
        assert first["run_id"] == second["run_id"]
        assert (
            first["routes"][0]["travel_time_s"]["value"]
            == second["routes"][0]["travel_time_s"]["value"]
        )
        assert sim_client.get("/api/simulation/runs").json()["count"] == 1

    def test_repeated_optimizer_driven_requests_return_the_same_run(
        self, sim_client: TestClient
    ) -> None:
        """The path that once drifted: a snapshot id must not move the id."""
        body = {
            "origin_junction_id": "J1",
            "destination_junction_id": "J3",
            "segment_inputs": SEGMENT_INPUTS,
        }
        first = sim_client.post("/api/simulation/compare", json=body).json()
        second = sim_client.post("/api/simulation/compare", json=body).json()
        assert first["run_id"] == second["run_id"]
        assert (
            first["scenario"]["input_fingerprint"]
            == second["scenario"]["input_fingerprint"]
        )
        assert first["comparison"]["improvement_pct"] == (
            second["comparison"]["improvement_pct"]
        )

        def outcome(payload: dict) -> str:
            """The modelled result, with the clock stripped out.

            Timestamps are expected to move between two calls, and so is the
            traffic-state id, which names the instant its snapshot was
            assembled. Everything the model produced must not.
            """

            def strip(node: object) -> object:
                if isinstance(node, dict):
                    return {
                        key: strip(value)
                        for key, value in node.items()
                        if key not in ("observed_at", "computed_at")
                    }
                if isinstance(node, list):
                    return [strip(item) for item in node]
                return node

            return json.dumps(
                strip(
                    {
                        key: payload[key]
                        for key in ("routes", "links", "network", "comparison")
                    }
                ),
                sort_keys=True,
            )

        assert outcome(first) == outcome(second), (
            "two identical requests must produce the same modelled outcome"
        )

    def test_a_stored_run_can_be_retrieved(self, sim_client: TestClient) -> None:
        run_id = sim_client.post(
            "/api/simulation/run",
            json={
                "route_segment_ids": ["SEG-12", "SEG-23"],
                "segment_inputs": SEGMENT_INPUTS,
            },
        ).json()["run_id"]

        listed = sim_client.get("/api/simulation/runs").json()
        assert listed["runs"][0]["run_id"] == run_id
        assert listed["runs"][0]["status"] == "completed"

        fetched = sim_client.get(f"/api/simulation/runs/{run_id}")
        assert fetched.status_code == 200
        assert fetched.json()["run_id"] == run_id

    def test_an_unknown_run_is_a_404(self, sim_client: TestClient) -> None:
        response = sim_client.get("/api/simulation/runs/sim-nothing")
        assert response.status_code == 404
        assert "kept in memory" in response.json()["error"]["message"]

    def test_a_stated_input_must_justify_itself(self, sim_client: TestClient) -> None:
        response = sim_client.post(
            "/api/simulation/run",
            json={
                "route_segment_ids": ["SEG-12", "SEG-23"],
                "segment_inputs": [
                    {"segment_id": "SEG-12", "demand_veh_per_hour": 900.0}
                ],
            },
        )
        assert response.status_code == 422, "an assumption with no note is refused"

    def test_a_stated_input_off_the_network_is_refused(
        self, sim_client: TestClient
    ) -> None:
        response = sim_client.post(
            "/api/simulation/run",
            json={
                "route_segment_ids": ["SEG-12", "SEG-23"],
                "segment_inputs": [
                    {
                        "segment_id": "SEG-GHOST",
                        "demand_veh_per_hour": 900.0,
                        "note": "A segment that does not exist.",
                    }
                ],
            },
        )
        assert response.status_code == 422
        assert "not on network" in response.json()["error"]["message"]

    def test_two_stated_inputs_for_one_segment_are_refused(
        self, sim_client: TestClient
    ) -> None:
        response = sim_client.post(
            "/api/simulation/run",
            json={
                "route_segment_ids": ["SEG-12", "SEG-23"],
                "segment_inputs": [
                    {
                        "segment_id": "SEG-12",
                        "demand_veh_per_hour": 900.0,
                        "note": "First.",
                    },
                    {
                        "segment_id": "SEG-12",
                        "capacity_veh_per_hour": 900.0,
                        "note": "Second.",
                    },
                ],
            },
        )
        assert response.status_code == 422
        assert "at most one stated input" in response.json()["error"]["message"]

    def test_a_step_longer_than_the_horizon_is_refused(
        self, sim_client: TestClient
    ) -> None:
        response = sim_client.post(
            "/api/simulation/run",
            json={
                "route_segment_ids": ["SEG-12", "SEG-23"],
                "horizon_s": 10.0,
                "step_s": 60.0,
            },
        )
        assert response.status_code == 422
        assert "less than one step" in response.json()["error"]["message"]

    def test_too_many_steps_are_refused(self, sim_client: TestClient) -> None:
        response = sim_client.post(
            "/api/simulation/run",
            json={
                "route_segment_ids": ["SEG-12", "SEG-23"],
                "horizon_s": 86_400.0,
                "step_s": 0.1,
            },
        )
        assert response.status_code == 422
        assert "ceiling" in response.json()["error"]["message"]

    def test_a_departure_beyond_the_horizon_is_refused(
        self, sim_client: TestClient
    ) -> None:
        response = sim_client.post(
            "/api/simulation/run",
            json={
                "route_segment_ids": ["SEG-12", "SEG-23"],
                "horizon_s": 100.0,
                "departure_offset_s": 100.0,
            },
        )
        assert response.status_code == 422
        assert "never depart" in response.json()["error"]["message"]

    def test_there_is_no_approve_or_execute_endpoint(
        self, sim_client: TestClient
    ) -> None:
        """A simulation exists so a person can check a recommendation first."""
        spec = sim_client.get("/openapi.json").json()
        simulation_paths = [p for p in spec["paths"] if p.startswith("/api/simulation")]
        assert simulation_paths
        forbidden = ("approve", "execute", "dispatch", "apply", "actuate", "signal")
        assert not [
            path
            for path in simulation_paths
            if any(token in path.lower() for token in forbidden)
        ]


class TestBlockedNetworkSimulation:
    def test_a_severed_network_yields_no_route_to_simulate(
        self, sim_client: TestClient
    ) -> None:
        """Both ways out of J1 blocked: the optimizer refuses, so nothing runs."""
        for segment_id in ("SEG-12", "SEG-14"):
            declared = sim_client.post(
                "/api/incidents",
                json={
                    "incident_type": "road_blockage",
                    "severity": "critical",
                    "location": {"segment_id": segment_id},
                    "blocks_segment_ids": [segment_id],
                    "reported_by": "operator.test",
                },
            )
            assert declared.status_code == 201, declared.text

        body = sim_client.post(
            "/api/simulation/compare",
            json={"origin_junction_id": "J1", "destination_junction_id": "J3"},
        ).json()
        assert body["status"] == "unavailable"
        assert body["routes"] == []
        assert body["comparison"] is None
        assert body["rejections"][0]["reason"] == "no_admissible_route"


class TestSimulationSchemaGuards:
    def test_a_run_cannot_refuse_and_still_carry_a_route(self) -> None:
        result = build_run(stated_inputs=CONGESTED)
        with pytest.raises(ValueError, match="refusal that still carries an outcome"):
            result.model_copy(update={"status": SimulationStatus.UNAVAILABLE}).model_validate(
                result.model_dump() | {"status": "unavailable"}
            )

    def test_an_unmodelled_segment_network_is_still_a_valid_run(self) -> None:
        result = build_run()
        assert result.status is SimulationStatus.COMPLETED
        assert result.network is not None

    def test_the_simulation_package_does_not_import_perception_or_genai(self) -> None:
        """Phase 8 is separate from YOLO, tracking and the language model."""
        root = Path(__file__).resolve().parents[1] / "app" / "simulation"
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for forbidden in ("app.vision", "ultralytics", "openrouter", "openai"):
                assert forbidden not in text, f"{path.name} imports {forbidden}"

    def test_no_module_trains_or_loads_a_model(self) -> None:
        root = Path(__file__).resolve().parents[1] / "app" / "simulation"
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            for forbidden in (".fit(", "train(", "torch", "sklearn"):
                assert forbidden not in text, f"{path.name} contains {forbidden}"
