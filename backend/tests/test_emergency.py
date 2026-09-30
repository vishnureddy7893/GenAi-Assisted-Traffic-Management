"""Phase 6 emergency tests: exclusion zones, incidents, and declared responses.

The properties under test are about refusing the wrong thing at the right time.
An exclusion zone must cover the road running past the fire, not merely the
junctions near it. An incident must not name a segment the map does not have. A
responder must not be dispatched to a destination class it cannot serve. And a
terminal record must stay terminal, so "what was blocked at 14:05?" keeps one
answer.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from network_fixtures import (
    make_facility,
    make_junction,
    make_network,
    make_segment,
    square_network,
)

from app.core.config import Settings
from app.core.errors import ResourceNotFoundError, ValidationFailedError
from app.domain.emergency import (
    EmergencyDestination,
    EmergencyState,
    EmergencyVehicle,
    NetworkPosition,
)
from app.domain.enums import (
    EmergencyPriority,
    EmergencyStatus,
    EmergencyVehicleType,
    FacilityType,
    IncidentSeverity,
    IncidentStatus,
    IncidentType,
)
from app.domain.incident import Incident, IncidentLocation
from app.domain.network import GeoPoint
from app.domain.provenance import Provenance, ValueSource
from app.emergency.incidents import IncidentRegistry, revalidated
from app.emergency.responses import EmergencyRegistry
from app.emergency.zones import haversine_m, resolve_zone
from app.main import create_app

T0 = dt.datetime(2024, 3, 14, 8, 30, tzinfo=dt.UTC)

PROV = Provenance(
    source=ValueSource.OBSERVED,
    producer="tests",
    producer_version="1.0",
    observed_at=T0,
    computed_at=T0,
)


def make_incident(
    incident_id: str = "INC-1",
    *,
    status: IncidentStatus = IncidentStatus.REPORTED,
    segment_id: str | None = "SEG-12",
    junction_id: str | None = None,
    blocks: frozenset[str] = frozenset(),
    affected: frozenset[str] = frozenset(),
) -> Incident:
    """A plain incident."""
    return Incident(
        incident_id=incident_id,
        incident_type=IncidentType.ACCIDENT,
        severity=IncidentSeverity.MAJOR,
        status=status,
        location=IncidentLocation(segment_id=segment_id, junction_id=junction_id),
        reported_at=T0,
        cleared_at=T0 if status is IncidentStatus.CLEARED else None,
        provenance=PROV,
        blocks_segment_ids=blocks,
        affected_segment_ids=affected,
    )


def make_emergency(
    emergency_id: str = "EMG-1",
    *,
    vehicle_type: EmergencyVehicleType = EmergencyVehicleType.AMBULANCE,
    position: NetworkPosition | None = None,
    destination: EmergencyDestination | None = None,
    triggering_incident_id: str | None = None,
    status: EmergencyStatus = EmergencyStatus.DECLARED,
) -> EmergencyState:
    """A declared response."""
    return EmergencyState(
        emergency_id=emergency_id,
        vehicle=EmergencyVehicle(vehicle_id="V-1", vehicle_type=vehicle_type),
        priority=EmergencyPriority.P1_IMMEDIATE,
        status=status,
        declared_at=T0,
        updated_at=T0,
        current_position=position or NetworkPosition(junction_id="J1"),
        destination=destination or EmergencyDestination(facility_id="FAC-HOSP-1"),
        triggering_incident_id=triggering_incident_id,
        provenance=PROV,
    )


# ---------------------------------------------------------------------------
# Exclusion zone geometry
# ---------------------------------------------------------------------------
class TestExclusionZones:
    def test_a_road_running_through_the_zone_is_excluded(self) -> None:
        """The failure this design exists to prevent.

        A long segment passing straight through an exclusion zone, with both
        endpoints far outside it, is exactly the road past a fire. Testing only
        the endpoints would leave it admissible.
        """
        west = make_junction("W", position=GeoPoint(latitude=16.5, longitude=80.59))
        east = make_junction("E", position=GeoPoint(latitude=16.5, longitude=80.61))
        through = make_segment(
            "SEG-WE",
            "W",
            "E",
            geometry=(
                GeoPoint(latitude=16.5, longitude=80.59),
                GeoPoint(latitude=16.5, longitude=80.61),
            ),
        )
        network = make_network(junctions=(west, east), segments=(through,))
        centre = GeoPoint(latitude=16.5, longitude=80.60)

        # Both endpoints are roughly a kilometre from the centre.
        assert haversine_m(centre, GeoPoint(latitude=16.5, longitude=80.59)) > 900
        assert haversine_m(centre, GeoPoint(latitude=16.5, longitude=80.61)) > 900

        resolution = resolve_zone(network, centre=centre, radius_m=200.0)
        assert resolution.excluded_segment_ids == {"SEG-WE"}
        assert resolution.excluded_junction_ids == frozenset()

    def test_a_distant_road_is_not_excluded(self) -> None:
        network = square_network()
        far = GeoPoint(latitude=16.6, longitude=80.7)
        resolution = resolve_zone(network, centre=far, radius_m=200.0)
        assert resolution.excluded_segment_ids == frozenset()
        assert any("covers no network element" in w for w in resolution.warnings)

    def test_a_zone_on_a_junction_takes_the_roads_that_meet_it(self) -> None:
        network = square_network()
        resolution = resolve_zone(
            network, centre=GeoPoint(latitude=16.5050, longitude=80.6050), radius_m=200.0
        )
        assert resolution.excluded_junction_ids == {"J2"}
        assert resolution.excluded_segment_ids == {"SEG-12", "SEG-21", "SEG-23", "SEG-32"}

    def test_untestable_elements_are_reported_not_skipped(self) -> None:
        """A zone that looks resolved and covers less than it claims is worse.

        Silently omitting elements with no coordinates produces a hard safety
        constraint that quietly stops constraining.
        """
        network = square_network(positioned=False)
        resolution = resolve_zone(
            network, centre=GeoPoint(latitude=16.505, longitude=80.605), radius_m=500.0
        )
        assert resolution.is_complete is False
        assert resolution.untestable_junction_ids == {"J1", "J2", "J3", "J4"}
        assert len(resolution.untestable_segment_ids) == 8
        assert resolution.excluded_segment_ids == frozenset()
        assert any("could not be tested" in w for w in resolution.warnings)

    def test_a_complete_resolution_says_so(self) -> None:
        resolution = resolve_zone(
            square_network(),
            centre=GeoPoint(latitude=16.5050, longitude=80.6050),
            radius_m=200.0,
        )
        assert resolution.is_complete is True
        assert resolution.tested_segment_count == 8

    def test_haversine_matches_a_known_separation(self) -> None:
        """0.01 degrees of latitude is about 1.11 km anywhere."""
        a = GeoPoint(latitude=16.50, longitude=80.60)
        b = GeoPoint(latitude=16.51, longitude=80.60)
        assert haversine_m(a, b) == pytest.approx(1112.0, abs=5.0)


# ---------------------------------------------------------------------------
# Incident lifecycle
# ---------------------------------------------------------------------------
class TestIncidentRegistry:
    def test_declaring_and_reading_back(self) -> None:
        registry = IncidentRegistry()
        registry.declare(make_incident(), square_network())
        assert registry.get("INC-1").incident_id == "INC-1"
        assert len(registry) == 1

    def test_an_unknown_id_is_a_404(self) -> None:
        with pytest.raises(ResourceNotFoundError):
            IncidentRegistry().get("nope")

    def test_a_duplicate_id_is_refused(self) -> None:
        """Re-declaring an id would overwrite the record of what was blocked."""
        registry = IncidentRegistry()
        registry.declare(make_incident(), square_network())
        with pytest.raises(ValidationFailedError, match="already exists"):
            registry.declare(make_incident(), square_network())

    def test_an_unknown_segment_is_refused(self) -> None:
        """It would appear active in every listing while pruning nothing."""
        registry = IncidentRegistry()
        with pytest.raises(ValidationFailedError, match="unknown"):
            registry.declare(
                make_incident(blocks=frozenset({"SEG-DOES-NOT-EXIST"})),
                square_network(),
            )

    def test_an_unknown_junction_is_refused(self) -> None:
        registry = IncidentRegistry()
        with pytest.raises(ValidationFailedError, match="unknown"):
            registry.declare(
                make_incident(segment_id=None, junction_id="J-NOWHERE"),
                square_network(),
            )

    @pytest.mark.parametrize(
        ("start", "target"),
        [
            (IncidentStatus.REPORTED, IncidentStatus.CONFIRMED),
            (IncidentStatus.REPORTED, IncidentStatus.FALSE_ALARM),
            (IncidentStatus.CONFIRMED, IncidentStatus.RESPONDING),
            (IncidentStatus.RESPONDING, IncidentStatus.CONTAINED),
            (IncidentStatus.CONTAINED, IncidentStatus.CLEARED),
        ],
    )
    def test_permitted_transitions(
        self, start: IncidentStatus, target: IncidentStatus
    ) -> None:
        registry = IncidentRegistry()
        registry.declare(make_incident(status=start), square_network())
        assert registry.transition("INC-1", target, at=T0).status is target

    @pytest.mark.parametrize(
        ("start", "target"),
        [
            (IncidentStatus.CONTAINED, IncidentStatus.REPORTED),
            (IncidentStatus.RESPONDING, IncidentStatus.FALSE_ALARM),
            (IncidentStatus.CONTAINED, IncidentStatus.CONFIRMED),
        ],
    )
    def test_refused_transitions(
        self, start: IncidentStatus, target: IncidentStatus
    ) -> None:
        registry = IncidentRegistry()
        registry.declare(make_incident(status=start), square_network())
        with pytest.raises(ValidationFailedError, match="cannot move"):
            registry.transition("INC-1", target, at=T0)

    def test_a_cleared_incident_stays_cleared(self) -> None:
        """Reopening would make 'was this blocked at 14:05?' unanswerable."""
        registry = IncidentRegistry()
        registry.declare(make_incident(status=IncidentStatus.CONTAINED), square_network())
        registry.transition("INC-1", IncidentStatus.CLEARED, at=T0)
        with pytest.raises(ValidationFailedError, match="cannot change status"):
            registry.transition("INC-1", IncidentStatus.REPORTED, at=T0)

    def test_clearing_records_when_it_happened(self) -> None:
        registry = IncidentRegistry()
        registry.declare(make_incident(), square_network())
        later = T0 + dt.timedelta(hours=2)
        cleared = registry.transition("INC-1", IncidentStatus.CLEARED, at=later)
        assert cleared.cleared_at == later
        assert cleared.is_active is False

    def test_confirming_records_when_it_happened(self) -> None:
        registry = IncidentRegistry()
        registry.declare(make_incident(), square_network())
        confirmed = registry.transition("INC-1", IncidentStatus.CONFIRMED, at=T0)
        assert confirmed.confirmed_at == T0

    def test_transitions_revalidate_rather_than_trusting_model_copy(self) -> None:
        """``model_copy(update=...)`` does not re-run validators in pydantic v2.

        Without an explicit round-trip a CLEARED incident could be written with
        no ``cleared_at``, which the domain forbids precisely because it makes
        the clearance unauditable.
        """
        incident = make_incident()
        loose = incident.model_copy(update={"status": IncidentStatus.CLEARED})
        assert loose.cleared_at is None  # the bug this guards against

        with pytest.raises(Exception, match="cleared_at"):
            revalidated(incident, {"status": "cleared"})

    def test_active_filters_out_finished_incidents(self) -> None:
        registry = IncidentRegistry()
        registry.declare(make_incident("INC-1"), square_network())
        registry.declare(make_incident("INC-2"), square_network())
        registry.transition("INC-2", IncidentStatus.CLEARED, at=T0)

        assert {i.incident_id for i in registry.active()} == {"INC-1"}
        assert len(registry.list()) == 2

    def test_multiple_simultaneous_incidents_are_tracked_independently(self) -> None:
        registry = IncidentRegistry()
        registry.declare(
            make_incident("INC-1", blocks=frozenset({"SEG-12"})), square_network()
        )
        registry.declare(
            make_incident("INC-2", blocks=frozenset({"SEG-23"})), square_network()
        )
        registry.transition("INC-1", IncidentStatus.CLEARED, at=T0)

        blocked: set[str] = set()
        for incident in registry.active():
            blocked |= incident.hard_blocked_segment_ids
        assert blocked == {"SEG-23"}


# ---------------------------------------------------------------------------
# Emergency lifecycle
# ---------------------------------------------------------------------------
class TestEmergencyRegistry:
    def _declare(self, registry: EmergencyRegistry, **kwargs) -> EmergencyState:
        return registry.declare(
            make_emergency(**kwargs), square_network(), known_incident_ids=frozenset()
        )

    def test_declaring_and_reading_back(self) -> None:
        registry = EmergencyRegistry()
        self._declare(registry)
        assert registry.get("EMG-1").emergency_id == "EMG-1"

    def test_a_duplicate_id_is_refused(self) -> None:
        registry = EmergencyRegistry()
        self._declare(registry)
        with pytest.raises(ValidationFailedError, match="already exists"):
            self._declare(registry)

    def test_an_unknown_junction_is_refused(self) -> None:
        registry = EmergencyRegistry()
        with pytest.raises(ValidationFailedError, match="unknown junction"):
            self._declare(registry, position=NetworkPosition(junction_id="J-NOWHERE"))

    def test_an_unknown_facility_is_refused(self) -> None:
        registry = EmergencyRegistry()
        with pytest.raises(ValidationFailedError, match="unknown facility"):
            self._declare(
                registry, destination=EmergencyDestination(facility_id="FAC-NOWHERE")
            )

    def test_an_ambulance_cannot_be_sent_to_a_fire_station(self) -> None:
        """Checked against the real facility record, not only the declared type."""
        registry = EmergencyRegistry()
        with pytest.raises(ValidationFailedError, match="not a valid destination"):
            self._declare(
                registry, destination=EmergencyDestination(facility_id="FAC-FIRE-1")
            )

    def test_a_facility_that_does_not_accept_the_responder_is_refused(self) -> None:
        network = make_network(
            junctions=(make_junction("J1"), make_junction("J2")),
            segments=(make_segment("SEG-12", "J1", "J2"),),
            facilities=(
                make_facility(
                    "FAC-HOSP-1",
                    "J1",
                    facility_type=FacilityType.HOSPITAL,
                    accepts=frozenset({EmergencyVehicleType.RESCUE}),
                ),
            ),
        )
        registry = EmergencyRegistry()
        with pytest.raises(ValidationFailedError, match="does not accept"):
            registry.declare(
                make_emergency(), network, known_incident_ids=frozenset()
            )

    def test_an_undeclared_triggering_incident_is_refused(self) -> None:
        registry = EmergencyRegistry()
        with pytest.raises(ValidationFailedError, match="has not been declared"):
            self._declare(registry, triggering_incident_id="INC-GHOST")

    def test_a_declared_triggering_incident_is_accepted(self) -> None:
        registry = EmergencyRegistry()
        stored = registry.declare(
            make_emergency(triggering_incident_id="INC-1"),
            square_network(),
            known_incident_ids=frozenset({"INC-1"}),
        )
        assert stored.triggering_incident_id == "INC-1"

    @pytest.mark.parametrize(
        ("start", "target"),
        [
            (EmergencyStatus.DECLARED, EmergencyStatus.EN_ROUTE),
            (EmergencyStatus.DECLARED, EmergencyStatus.CANCELLED),
            (EmergencyStatus.EN_ROUTE, EmergencyStatus.ARRIVED),
            (EmergencyStatus.ARRIVED, EmergencyStatus.RETURNING),
            (EmergencyStatus.RETURNING, EmergencyStatus.COMPLETED),
        ],
    )
    def test_permitted_transitions(
        self, start: EmergencyStatus, target: EmergencyStatus
    ) -> None:
        registry = EmergencyRegistry()
        self._declare(registry, status=start)
        assert registry.transition("EMG-1", target, at=T0).status is target

    def test_a_response_cannot_skip_to_completed(self) -> None:
        registry = EmergencyRegistry()
        self._declare(registry)
        with pytest.raises(ValidationFailedError, match="cannot move"):
            registry.transition("EMG-1", EmergencyStatus.COMPLETED, at=T0)

    def test_a_completed_response_stays_completed(self) -> None:
        registry = EmergencyRegistry()
        self._declare(registry, status=EmergencyStatus.RETURNING)
        registry.transition("EMG-1", EmergencyStatus.COMPLETED, at=T0)
        with pytest.raises(ValidationFailedError, match="cannot change status"):
            registry.transition("EMG-1", EmergencyStatus.EN_ROUTE, at=T0)

    def test_completion_records_when_it_happened(self) -> None:
        registry = EmergencyRegistry()
        self._declare(registry, status=EmergencyStatus.ARRIVED)
        later = T0 + dt.timedelta(minutes=12)
        done = registry.transition("EMG-1", EmergencyStatus.COMPLETED, at=later)
        assert done.completed_at == later
        assert done.is_active is False

    def test_position_updates_are_validated_against_the_network(self) -> None:
        registry = EmergencyRegistry()
        self._declare(registry)
        moved = registry.update_position(
            "EMG-1", NetworkPosition(junction_id="J2"), square_network(), at=T0
        )
        assert moved.current_position.junction_id == "J2"

        with pytest.raises(ValidationFailedError, match="unknown junction"):
            registry.update_position(
                "EMG-1", NetworkPosition(junction_id="J-NOWHERE"), square_network(), at=T0
            )

    def test_a_finished_response_does_not_move(self) -> None:
        registry = EmergencyRegistry()
        self._declare(registry, status=EmergencyStatus.RETURNING)
        registry.transition("EMG-1", EmergencyStatus.COMPLETED, at=T0)
        with pytest.raises(ValidationFailedError, match="does not move"):
            registry.update_position(
                "EMG-1", NetworkPosition(junction_id="J2"), square_network(), at=T0
            )

    def test_only_p1_warrants_corridor_preemption(self) -> None:
        """Pre-emption imposes real delay and risk on every conflicting movement."""
        p1 = make_emergency()
        assert p1.warrants_corridor_preemption is True
        p3 = revalidated_emergency(p1, {"priority": "P3_routine"})
        assert p3.warrants_corridor_preemption is False


def revalidated_emergency(emergency: EmergencyState, updates: dict) -> EmergencyState:
    """Helper mirroring the registry's revalidating update."""
    from app.emergency.responses import revalidated as _revalidated

    return _revalidated(emergency, updates)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@pytest.fixture
def loaded_client(tmp_path: Path):
    """A client with the square network already loaded."""
    import json

    (tmp_path / "net.json").write_text(
        json.dumps(square_network().model_dump(mode="json")), encoding="utf-8"
    )
    settings = Settings(app_env="local", log_level="WARNING", networks_dir=tmp_path)
    with TestClient(create_app(settings)) as client:
        client.post("/api/network/load", json={"relative_path": "net.json"})
        yield client


class TestIncidentApi:
    def test_declaring_assigns_an_id_and_stamps_provenance(
        self, loaded_client: TestClient
    ) -> None:
        response = loaded_client.post(
            "/api/incidents",
            json={
                "incident_type": "accident",
                "severity": "major",
                "location": {"segment_id": "SEG-12"},
                "reported_by": "operator.test",
            },
        )
        assert response.status_code == 201
        incident = response.json()["incident"]
        assert incident["incident_id"].startswith("incident-")
        assert incident["provenance"]["producer"] == "operator.test"
        assert incident["provenance"]["source"] == "observed"

    def test_a_zone_declared_from_a_centre_is_resolved(
        self, loaded_client: TestClient
    ) -> None:
        response = loaded_client.post(
            "/api/incidents",
            json={
                "incident_type": "fire",
                "severity": "critical",
                "location": {"junction_id": "J2"},
                "exclusion_zone": {
                    "radius_m": 200.0,
                    "reason": "Active fire",
                    "centre": {"latitude": 16.5050, "longitude": 80.6050},
                },
            },
        )
        assert response.status_code == 201
        body = response.json()
        zone = body["incident"]["exclusion_zone"]
        assert sorted(zone["excluded_segment_ids"]) == [
            "SEG-12",
            "SEG-21",
            "SEG-23",
            "SEG-32",
        ]
        assert body["zone_resolution"]["tested_segment_count"] == 8

    def test_an_oversized_radius_is_refused(self, loaded_client: TestClient) -> None:
        """A hard constraint cannot be outbid, so an implausible radius is caught."""
        response = loaded_client.post(
            "/api/incidents",
            json={
                "incident_type": "fire",
                "severity": "critical",
                "location": {"junction_id": "J2"},
                "exclusion_zone": {
                    "radius_m": 50_000.0,
                    "reason": "typo",
                    "centre": {"latitude": 16.505, "longitude": 80.605},
                },
            },
        )
        assert response.status_code == 422
        assert "maximum" in response.json()["error"]["message"]

    def test_a_zone_with_neither_centre_nor_members_is_refused(
        self, loaded_client: TestClient
    ) -> None:
        response = loaded_client.post(
            "/api/incidents",
            json={
                "incident_type": "fire",
                "severity": "critical",
                "location": {"junction_id": "J2"},
                "exclusion_zone": {"radius_m": 100.0, "reason": "nothing covered"},
            },
        )
        assert response.status_code == 422

    def test_an_incident_cannot_be_declared_already_cleared(
        self, loaded_client: TestClient
    ) -> None:
        """It would be terminal on arrival and could never have constrained anything."""
        response = loaded_client.post(
            "/api/incidents",
            json={
                "incident_type": "accident",
                "severity": "minor",
                "location": {"segment_id": "SEG-12"},
                "status": "cleared",
            },
        )
        assert response.status_code == 422

    def test_an_unknown_segment_is_refused_with_the_id_named(
        self, loaded_client: TestClient
    ) -> None:
        response = loaded_client.post(
            "/api/incidents",
            json={
                "incident_type": "accident",
                "severity": "minor",
                "location": {"segment_id": "SEG-12"},
                "blocks_segment_ids": ["SEG-GHOST"],
            },
        )
        assert response.status_code == 422
        assert "SEG-GHOST" in response.json()["error"]["message"]

    def test_lifecycle_through_the_api(self, loaded_client: TestClient) -> None:
        created = loaded_client.post(
            "/api/incidents",
            json={
                "incident_type": "accident",
                "severity": "major",
                "location": {"segment_id": "SEG-12"},
                "blocks_segment_ids": ["SEG-12"],
            },
        ).json()
        incident_id = created["incident"]["incident_id"]
        assert "confirmed" in created["permitted_next_statuses"]

        confirmed = loaded_client.post(
            f"/api/incidents/{incident_id}/status", json={"status": "confirmed"}
        )
        assert confirmed.json()["incident"]["status"] == "confirmed"

        assert loaded_client.get("/api/network/blocked").json()[
            "blocked_segment_ids"
        ] == ["SEG-12"]

        cleared = loaded_client.post(
            f"/api/incidents/{incident_id}/status", json={"status": "cleared"}
        )
        assert cleared.json()["incident"]["status"] == "cleared"
        assert cleared.json()["permitted_next_statuses"] == []
        assert loaded_client.get("/api/network/blocked").json()["blocked_segment_ids"] == []

    def test_declaring_without_a_network_is_refused(self, tmp_path: Path) -> None:
        settings = Settings(app_env="local", log_level="WARNING", networks_dir=tmp_path)
        with TestClient(create_app(settings)) as client:
            response = client.post(
                "/api/incidents",
                json={
                    "incident_type": "accident",
                    "severity": "minor",
                    "location": {"segment_id": "SEG-12"},
                },
            )
            assert response.status_code == 404
            assert response.json()["error"]["code"] == "network_not_loaded"

    def test_the_empty_list_explains_itself(self, loaded_client: TestClient) -> None:
        body = loaded_client.get("/api/incidents").json()
        assert body["incidents"] == []
        assert body["count"] == 0
        assert "discarded when a different network is loaded" in body["note"]


class TestEmergencyApi:
    def _declare(self, client: TestClient, **overrides) -> dict:
        payload = {
            "vehicle": {"vehicle_id": "AMB-1", "vehicle_type": "ambulance"},
            "priority": "P1_immediate",
            "current_position": {"junction_id": "J1"},
            "destination": {"facility_id": "FAC-HOSP-1"},
        }
        payload.update(overrides)
        return client.post("/api/emergency/declare", json=payload).json()

    def test_declaring_a_response(self, loaded_client: TestClient) -> None:
        body = self._declare(loaded_client)
        assert body["emergency"]["emergency_id"].startswith("emergency-")
        assert body["warrants_corridor_preemption"] is True
        assert body["permitted_next_statuses"] == ["cancelled", "en_route"]
        assert body["elapsed_response_time_s"] is not None

    def test_scenarios_reflect_declarations(self, loaded_client: TestClient) -> None:
        self._declare(loaded_client)
        body = loaded_client.get("/api/emergency/scenarios").json()
        assert body["count"] == 1
        assert body["active_count"] == 1
        assert body["declaration_supported"] is True
        assert len(body["scenarios"]) == 1

    def test_a_future_declaration_time_is_refused(self, loaded_client: TestClient) -> None:
        """A future start would report negative elapsed time on every ETA."""
        future = (dt.datetime.now(tz=dt.UTC) + dt.timedelta(hours=1)).isoformat()
        response = loaded_client.post(
            "/api/emergency/declare",
            json={
                "vehicle": {"vehicle_id": "AMB-9", "vehicle_type": "ambulance"},
                "priority": "P2_urgent",
                "current_position": {"junction_id": "J1"},
                "destination": {"facility_id": "FAC-HOSP-1"},
                "declared_at": future,
            },
        )
        assert response.status_code == 422

    def test_lifecycle_and_position_through_the_api(
        self, loaded_client: TestClient
    ) -> None:
        emergency_id = self._declare(loaded_client)["emergency"]["emergency_id"]

        moved = loaded_client.post(
            f"/api/emergency/{emergency_id}/status", json={"status": "en_route"}
        )
        assert moved.json()["emergency"]["status"] == "en_route"

        positioned = loaded_client.post(
            f"/api/emergency/{emergency_id}/position",
            json={"position": {"junction_id": "J2"}},
        )
        assert positioned.json()["emergency"]["current_position"]["junction_id"] == "J2"

        assert (
            loaded_client.post(
                f"/api/emergency/{emergency_id}/status", json={"status": "completed"}
            ).status_code
            == 422
        )

    def test_an_active_emergency_marks_its_junction_in_network_state(
        self, loaded_client: TestClient
    ) -> None:
        emergency_id = self._declare(loaded_client)["emergency"]["emergency_id"]
        loaded_client.post(
            f"/api/emergency/{emergency_id}/status", json={"status": "en_route"}
        )
        body = loaded_client.get("/api/network/state").json()
        flagged = [
            j["junction_id"] for j in body["junction_states"] if j["emergency_active"]
        ]
        assert flagged == ["J1"]

    def test_declaring_without_a_network_is_refused(self, tmp_path: Path) -> None:
        settings = Settings(app_env="local", log_level="WARNING", networks_dir=tmp_path)
        with TestClient(create_app(settings)) as client:
            response = client.post(
                "/api/emergency/declare",
                json={
                    "vehicle": {"vehicle_id": "AMB-1", "vehicle_type": "ambulance"},
                    "priority": "P1_immediate",
                    "current_position": {"junction_id": "J1"},
                    "destination": {"facility_id": "FAC-HOSP-1"},
                },
            )
            assert response.status_code == 404

            body = client.get("/api/emergency/scenarios").json()
            assert body["declaration_supported"] is False
            assert body["declaration_phase"] == 6

    def test_no_route_is_invented_by_any_phase_6_endpoint(
        self, loaded_client: TestClient
    ) -> None:
        """The one output a person might act on must never be fabricated."""
        body = self._declare(loaded_client)
        emergency_id = body["emergency"]["emergency_id"]
        assert body["emergency"]["assigned_route_id"] is None

        corridor = loaded_client.get("/api/emergency/corridor")
        assert corridor.status_code == 501
        assert corridor.json()["error"]["details"]["phase"] == 9

        moved = loaded_client.get("/api/emergency/recommendation")
        assert moved.status_code == 422
        assert "routes/optimize" in moved.json()["error"]["message"]

        rendered = repr(loaded_client.get(f"/api/emergency/{emergency_id}").json()).lower()
        for token in ("route_comparison", "eta", "travel_time", "signal_advisor"):
            assert token not in rendered, f"phase 6 must not emit {token!r}"
