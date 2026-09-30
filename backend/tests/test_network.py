"""Phase 6 road network tests: loading, holding, and per-instant state.

The properties under test are mostly about what must fail, and when. A broken
map must fail at load with the offending id named, not inside route relaxation
on a live emergency request. An unloaded network must be reported as such, never
as an empty one. And a state assembled over the graph must carry blockages
without inventing congestion.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from network_fixtures import make_junction, make_network, make_segment, square_network

from app.core.config import Settings
from app.core.errors import (
    NetworkNotLoadedError,
    ResourceNotFoundError,
    ValidationFailedError,
)
from app.domain.enums import (
    CongestionLevel,
    FreshnessStatus,
    IncidentSeverity,
    IncidentStatus,
    IncidentType,
)
from app.domain.incident import ExclusionZone, Incident, IncidentLocation
from app.domain.provenance import FloatMeasurement, Provenance, Unit, ValueSource
from app.domain.state import StateAssemblyPolicy
from app.main import create_app
from app.network.assembler import assemble_network_state, degraded_segment_ids
from app.network.loader import list_available, load_network
from app.network.store import NetworkStore

T0 = dt.datetime(2024, 3, 14, 8, 30, tzinfo=dt.UTC)

PROV = Provenance(
    source=ValueSource.OBSERVED,
    producer="tests",
    producer_version="1.0",
    observed_at=T0,
    computed_at=T0,
)

POLICY = StateAssemblyPolicy(max_staleness_s=5.0, drop_after_s=30.0)


def make_incident(
    incident_id: str = "INC-1",
    *,
    status: IncidentStatus = IncidentStatus.CONFIRMED,
    blocks: frozenset[str] = frozenset(),
    affected: frozenset[str] = frozenset(),
    junction_id: str | None = None,
    segment_id: str | None = "SEG-12",
    zone_segments: frozenset[str] = frozenset(),
    zone_junctions: frozenset[str] = frozenset(),
) -> Incident:
    """An incident with exactly the coverage a test needs."""
    zone = None
    if zone_segments or zone_junctions:
        zone = ExclusionZone(
            radius_m=FloatMeasurement(value=200.0, unit=Unit.METER, provenance=PROV),
            excluded_segment_ids=zone_segments,
            excluded_junction_ids=zone_junctions,
            reason="test zone",
        )
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
        exclusion_zone=zone,
    )


def write_network(tmp_path: Path, name: str = "net.json") -> Path:
    """Write a valid network file and return its path."""
    path = tmp_path / name
    path.write_text(
        json.dumps(square_network().model_dump(mode="json")), encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
class TestLoader:
    def test_a_valid_file_loads_with_its_checksum(self, tmp_path: Path) -> None:
        path = write_network(tmp_path)
        network, source = load_network(path, "net.json", loaded_at=T0)

        assert network.network_id == "test-net"
        assert len(source.sha256) == 64
        assert source.junction_count == 4
        assert source.segment_count == 8
        assert source.size_bytes == path.stat().st_size

    def test_the_checksum_identifies_the_bytes_not_the_version(
        self, tmp_path: Path
    ) -> None:
        """Two files claiming one version with different content must differ.

        The declared version is what a decision pins; the checksum is what proves
        the file behind that version was not edited underneath it.
        """
        first = write_network(tmp_path, "a.json")
        payload = json.loads(first.read_text())
        payload["name"] = "edited in place"
        second = tmp_path / "b.json"
        second.write_text(json.dumps(payload), encoding="utf-8")

        _, source_a = load_network(first, "a.json", loaded_at=T0)
        _, source_b = load_network(second, "b.json", loaded_at=T0)
        assert source_a.version == source_b.version
        assert source_a.sha256 != source_b.sha256

    def test_a_missing_file_is_a_client_mistake_not_an_upstream_failure(
        self, tmp_path: Path
    ) -> None:
        """404, not 502.

        Reporting "you named a file that is not there" as an upstream failure
        would send an operator to check a decoder that is working fine.
        """
        with pytest.raises(ResourceNotFoundError, match="No network file") as excinfo:
            load_network(tmp_path / "nope.json", "nope.json", loaded_at=T0)
        assert excinfo.value.status_code == 404

    def test_invalid_json_is_rejected_with_its_position(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text('{"network_id": ', encoding="utf-8")
        with pytest.raises(ValidationFailedError, match="not valid JSON"):
            load_network(path, "bad.json", loaded_at=T0)

    def test_a_json_array_is_not_a_network(self, tmp_path: Path) -> None:
        path = tmp_path / "arr.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValidationFailedError, match="must be a JSON object"):
            load_network(path, "arr.json", loaded_at=T0)

    def test_a_dangling_segment_reference_fails_at_load(self, tmp_path: Path) -> None:
        """Failing here beats failing inside route relaxation on a live request."""
        payload = square_network().model_dump(mode="json")
        payload["segments"][0]["to_junction_id"] = "J-DOES-NOT-EXIST"
        path = tmp_path / "dangling.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValidationFailedError) as excinfo:
            load_network(path, "dangling.json", loaded_at=T0)
        assert "not a valid road network" in str(excinfo.value)

    def test_list_available_finds_json_without_opening_it(self, tmp_path: Path) -> None:
        write_network(tmp_path, "one.json")
        (tmp_path / "nested").mkdir()
        (tmp_path / "nested" / "two.json").write_text("not a network", encoding="utf-8")
        (tmp_path / "ignored.txt").write_text("x", encoding="utf-8")

        assert list_available(tmp_path) == ("nested/two.json", "one.json")

    def test_list_available_tolerates_a_missing_directory(self, tmp_path: Path) -> None:
        assert list_available(tmp_path / "absent") == ()


# ---------------------------------------------------------------------------
# Holding
# ---------------------------------------------------------------------------
class TestStore:
    def test_an_unloaded_store_refuses_rather_than_inventing_a_network(self) -> None:
        store = NetworkStore()
        assert store.is_loaded is False
        with pytest.raises(NetworkNotLoadedError, match="POST /api/network/load"):
            store.require()

    def test_the_refusal_is_a_designed_state_not_a_fault(self) -> None:
        """A fresh service has no map. Logging that at ERROR trains people to ignore it."""
        assert NetworkNotLoadedError("x").is_expected is True
        assert NetworkNotLoadedError("x").status_code == 404
        assert NetworkNotLoadedError("x").error_code == "network_not_loaded"

    def test_loading_builds_the_index(self, tmp_path: Path) -> None:
        store = NetworkStore()
        network, source = load_network(write_network(tmp_path), "net.json", loaded_at=T0)
        store.load(network, source)

        index = store.require_index()
        assert index.junction_ids == {"J1", "J2", "J3", "J4"}
        assert {s.segment_id for s in index.outgoing_segments("J1")} == {"SEG-12", "SEG-14"}

    def test_clearing_unloads(self, tmp_path: Path) -> None:
        store = NetworkStore()
        network, source = load_network(write_network(tmp_path), "net.json", loaded_at=T0)
        store.load(network, source)
        store.clear()
        assert store.is_loaded is False
        assert store.source is None


# ---------------------------------------------------------------------------
# Per-instant state
# ---------------------------------------------------------------------------
class TestAssembler:
    def _state(self, incidents=(), **kwargs):
        return assemble_network_state(
            square_network(),
            incidents=tuple(incidents),
            policy=POLICY,
            schema_version="0.1.0",
            as_of=T0,
            assembled_at=T0,
            **kwargs,
        )

    def test_every_element_gets_a_state_including_the_silent_ones(self) -> None:
        """Omitting unobserved elements makes 'clear' and 'absent' the same lookup."""
        state = self._state()
        assert len(state.segment_states) == 8
        assert len(state.junction_states) == 4
        assert all(s.freshness is FreshnessStatus.UNKNOWN for s in state.segment_states)

    def test_congestion_is_never_invented_from_an_observation_area(self) -> None:
        """A band on the wrong segment is a false claim a route is planned around."""
        state = self._state()
        assert all(
            s.congestion_level is CongestionLevel.UNKNOWN for s in state.segment_states
        )
        assert all(s.level_of_service is None for s in state.segment_states)

    def test_an_incident_blocks_its_segments_and_names_itself(self) -> None:
        state = self._state([make_incident(blocks=frozenset({"SEG-12"}))])
        blocked = state.segment_state("SEG-12")
        assert blocked.is_blocked is True
        assert blocked.blocked_by_incident_ids == {"INC-1"}
        assert state.blocked_segment_ids == {"SEG-12"}

    def test_an_exclusion_zone_blocks_as_hard_as_an_explicit_blockage(self) -> None:
        """A route mostly outside a fire zone is not safer, it is unsafe."""
        state = self._state(
            [make_incident(zone_segments=frozenset({"SEG-23", "SEG-32"}))]
        )
        assert state.blocked_segment_ids == {"SEG-23", "SEG-32"}

    def test_a_cleared_incident_stops_blocking(self) -> None:
        state = self._state(
            [
                make_incident(
                    status=IncidentStatus.CLEARED, blocks=frozenset({"SEG-12"})
                )
            ]
        )
        assert state.blocked_segment_ids == frozenset()
        assert state.active_incidents == ()
        assert state.segment_state("SEG-12").is_blocked is False

    def test_two_incidents_blocking_one_segment_both_appear(self) -> None:
        """Multi-incident scenarios: clearing one must not imply the road reopened."""
        state = self._state(
            [
                make_incident("INC-1", blocks=frozenset({"SEG-12"})),
                make_incident("INC-2", blocks=frozenset({"SEG-12"})),
            ]
        )
        assert state.segment_state("SEG-12").blocked_by_incident_ids == {"INC-1", "INC-2"}

    def test_degraded_is_kept_apart_from_blocked(self) -> None:
        """A jammed road is slow; a blocked road is impassable."""
        incident = make_incident(
            blocks=frozenset({"SEG-12"}), affected=frozenset({"SEG-23"})
        )
        state = self._state([incident])
        assert state.blocked_segment_ids == {"SEG-12"}
        assert state.segment_state("SEG-23").is_blocked is False
        assert degraded_segment_ids(square_network(), (incident,)) == {"SEG-23"}

    def test_a_blocked_segment_always_names_a_cause(self) -> None:
        """The domain forbids an unexplainable closure; the assembler must comply."""
        state = self._state([make_incident(blocks=frozenset({"SEG-12"}))])
        for segment in state.segment_states:
            if segment.is_blocked:
                assert segment.blocked_by_incident_ids

    def test_a_junction_records_the_incidents_touching_it(self) -> None:
        state = self._state(
            [make_incident(segment_id=None, junction_id="J2")]
        )
        assert state.junction_state("J2").active_incident_ids == {"INC-1"}
        assert state.junction_state("J1").active_incident_ids == frozenset()

    def test_an_active_emergency_marks_its_junction(self) -> None:
        state = self._state(emergency_junction_ids=frozenset({"J3"}))
        assert state.junction_state("J3").emergency_active is True
        assert state.junction_state("J1").emergency_active is False

    def test_the_snapshot_pins_the_network_it_describes(self) -> None:
        """A recommendation without the map version behind it is an anecdote."""
        state = self._state()
        assert state.network_id == "test-net"
        assert state.network_version == "1.0"
        assert state.policy.max_staleness_s == 5.0

    def test_assembly_is_deterministic(self) -> None:
        first = self._state([make_incident(blocks=frozenset({"SEG-12"}))])
        second = self._state([make_incident(blocks=frozenset({"SEG-12"}))])
        assert first.model_dump() == second.model_dump()


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@pytest.fixture
def net_client(tmp_path: Path):
    """A client whose networks directory holds one valid network file."""
    write_network(tmp_path)
    settings = Settings(
        app_env="local",
        log_level="WARNING",
        networks_dir=tmp_path,
        max_observation_staleness_s=5.0,
        drop_observation_after_s=30.0,
    )
    with TestClient(create_app(settings)) as client:
        yield client


class TestNetworkApi:
    def test_status_answers_before_anything_is_loaded(self, net_client: TestClient) -> None:
        body = net_client.get("/api/network/status").json()
        assert body["loaded"] is False
        assert body["available_files"] == ["net.json"]
        assert "POST /api/network/load" in body["note"]

    def test_loading_then_serving(self, net_client: TestClient) -> None:
        loaded = net_client.post("/api/network/load", json={"relative_path": "net.json"})
        assert loaded.status_code == 200
        assert loaded.json()["junction_count"] == 4

        network = net_client.get("/api/network")
        assert network.status_code == 200
        assert network.json()["network_id"] == "test-net"

    def test_a_traversal_attempt_is_refused(self, net_client: TestClient) -> None:
        """An endpoint accepting absolute paths would be a filesystem read primitive."""
        for candidate in ("../secrets.json", "/etc/passwd", "C:\\\\windows\\\\win.ini"):
            response = net_client.post(
                "/api/network/load", json={"relative_path": candidate}
            )
            assert response.status_code in (404, 422), candidate
            assert net_client.get("/api/network").status_code == 404

    def test_loading_an_unknown_file_is_a_404(self, net_client: TestClient) -> None:
        response = net_client.post(
            "/api/network/load", json={"relative_path": "absent.json"}
        )
        assert response.status_code == 404

    def test_loading_discards_incidents_from_the_previous_network(
        self, net_client: TestClient
    ) -> None:
        """Segment ids mean something different on a different map."""
        net_client.post("/api/network/load", json={"relative_path": "net.json"})
        declared = net_client.post(
            "/api/incidents",
            json={
                "incident_type": "accident",
                "severity": "major",
                "location": {"segment_id": "SEG-12"},
                "blocks_segment_ids": ["SEG-12"],
            },
        )
        assert declared.status_code == 201

        reloaded = net_client.post(
            "/api/network/load", json={"relative_path": "net.json"}
        )
        assert reloaded.json()["cleared_incident_count"] == 1
        assert net_client.get("/api/incidents").json()["count"] == 0

    def test_network_state_carries_blockages_but_not_congestion(
        self, net_client: TestClient
    ) -> None:
        net_client.post("/api/network/load", json={"relative_path": "net.json"})
        net_client.post(
            "/api/incidents",
            json={
                "incident_type": "road_blockage",
                "severity": "critical",
                "location": {"segment_id": "SEG-12"},
                "blocks_segment_ids": ["SEG-12"],
            },
        )
        body = net_client.get("/api/network/state").json()
        blocked = [s["segment_id"] for s in body["segment_states"] if s["is_blocked"]]
        assert blocked == ["SEG-12"]
        assert all(
            s["congestion_level"] == "unknown" for s in body["segment_states"]
        )

    def test_blocked_endpoint_separates_the_two_safety_tiers(
        self, net_client: TestClient
    ) -> None:
        net_client.post("/api/network/load", json={"relative_path": "net.json"})
        net_client.post(
            "/api/incidents",
            json={
                "incident_type": "accident",
                "severity": "moderate",
                "location": {"segment_id": "SEG-12"},
                "blocks_segment_ids": ["SEG-12"],
                "affected_segment_ids": ["SEG-23"],
            },
        )
        body = net_client.get("/api/network/blocked").json()
        assert body["blocked_segment_ids"] == ["SEG-12"]
        assert body["degraded_segment_ids"] == ["SEG-23"]
        assert body["active_incident_count"] == 1

    def test_facilities_can_be_filtered_by_responder_type(
        self, net_client: TestClient
    ) -> None:
        """Filtering here stops a caller building a dispatch the validator rejects."""
        net_client.post("/api/network/load", json={"relative_path": "net.json"})
        body = net_client.get("/api/network/facilities?accepts=ambulance").json()
        assert [f["facility_id"] for f in body["facilities"]] == ["FAC-HOSP-1"]

    def test_every_network_endpoint_refuses_without_a_map(
        self, net_client: TestClient
    ) -> None:
        for path in ("/api/network", "/api/network/state", "/api/network/blocked",
                     "/api/network/facilities"):
            response = net_client.get(path)
            assert response.status_code == 404, path
            assert response.json()["error"]["code"] == "network_not_loaded"

    def test_the_safety_catalog_is_independent_of_the_network(
        self, net_client: TestClient
    ) -> None:
        """Rules are policy, served since Phase 1, and owe nothing to a loaded map."""
        assert net_client.get("/api/safety/catalog").status_code == 200


class TestStartupAutoload:
    def test_a_configured_network_loads_at_startup(self, tmp_path: Path) -> None:
        write_network(tmp_path)
        settings = Settings(
            app_env="local",
            log_level="WARNING",
            networks_dir=tmp_path,
            network_file="net.json",
        )
        with TestClient(create_app(settings)) as client:
            assert client.get("/api/network").status_code == 200
            assert client.get("/api/network/status").json()["loaded"] is True

    def test_a_bad_configured_network_does_not_stop_the_service(
        self, tmp_path: Path
    ) -> None:
        """A bad map path must not take down ingestion, perception and analytics."""
        settings = Settings(
            app_env="local",
            log_level="WARNING",
            networks_dir=tmp_path,
            network_file="missing.json",
        )
        with TestClient(create_app(settings)) as client:
            assert client.get("/api/health").status_code == 200
            assert client.get("/api/network").status_code == 404


class TestNetworkFixtureFile:
    def test_the_shipped_example_network_is_valid(self) -> None:
        """The example must load, and must say plainly that it is synthetic."""
        path = Path(__file__).resolve().parents[2] / "datasets" / "networks" / "example-grid.json"
        if not path.exists():  # pragma: no cover - example is optional
            pytest.skip("example network not present")
        network, _ = load_network(path, "example-grid.json", loaded_at=T0)
        assert network.provenance.source is ValueSource.ASSUMED
        assert network.provenance.notes
        assert "synthetic" in network.provenance.notes.lower()


class TestUnusedBuilders:
    """The fixture builders themselves must produce valid networks."""

    def test_default_network_and_bare_junctions_are_valid(self) -> None:
        network = make_network(
            junctions=(make_junction("A"), make_junction("B")),
            segments=(make_segment("S-AB", "A", "B"),),
        )
        assert network.build_index().segment_ids == {"S-AB"}
