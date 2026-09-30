"""API tests: health, metadata, source catalogue, emergency scenarios, honest 501s."""

from __future__ import annotations

import json
import logging
import re

import pytest
from fastapi.testclient import TestClient

from app.core.errors import DataSourceError, NotAvailableYetError, TrafficSystemError
from app.core.versions import APP_VERSION, IMPLEMENTED_PHASE, SCHEMA_VERSION


class TestHealth:
    def test_health_reports_liveness_ok(self, client: TestClient) -> None:
        response = client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["app_version"] == APP_VERSION
        assert body["schema_version"] == SCHEMA_VERSION
        assert body["implemented_phase"] == IMPLEMENTED_PHASE

    def test_pipeline_is_not_ready_before_the_decision_loop(
        self, client: TestClient
    ) -> None:
        """Liveness and pipeline readiness are separate questions.

        Reporting a single merged flag would have to misreport one of them, and a
        green tick beside an empty dashboard reads as "no traffic", not "no
        pipeline".
        """
        body = client.get("/api/health").json()
        assert body["pipeline_ready"] is False
        assert body["pipeline_blockers"], "unimplemented subsystems must be stated"

    def test_unimplemented_components_are_labelled_with_their_phase(
        self, client: TestClient
    ) -> None:
        components = {c["name"]: c for c in client.get("/api/health").json()["components"]}
        assert components["ml_models"]["status"] == "not_implemented"
        assert components["ml_models"]["phase"] == 10

    def test_implemented_subsystems_are_not_labelled_unimplemented(
        self, client: TestClient
    ) -> None:
        """Phases 1-8 shipped, so their subsystems must not report as missing."""
        components = {c["name"]: c for c in client.get("/api/health").json()["components"]}
        assert components["data_sources"]["status"] == "ok"
        assert components["computer_vision"]["status"] != "not_implemented"
        assert components["computer_vision"]["phase"] is None
        assert components["traffic_analytics"]["status"] == "ok"
        assert components["traffic_state"]["status"] == "ok"
        assert components["road_network"]["status"] != "not_implemented"
        assert components["emergency_management"]["status"] != "not_implemented"
        assert components["route_optimizer"]["status"] != "not_implemented"
        assert components["simulation"]["status"] != "not_implemented"

    def test_genai_is_reported_as_unconfigured(self, client: TestClient) -> None:
        components = {c["name"]: c for c in client.get("/api/health").json()["components"]}
        assert components["genai"]["status"] == "not_configured"

    def test_correlation_id_is_echoed(self, client: TestClient) -> None:
        response = client.get("/api/health", headers={"X-Request-ID": "abc123"})
        assert response.headers["X-Request-ID"] == "abc123"

    def test_correlation_id_is_generated_when_absent(self, client: TestClient) -> None:
        assert client.get("/api/health").headers.get("X-Request-ID")


class TestMeta:
    def test_versions_are_independently_pinned(self, client: TestClient) -> None:
        body = client.get("/api/meta/versions").json()
        for key in (
            "app_version",
            "schema_version",
            "safety_rules_version",
            "route_weights_version",
        ):
            assert body[key]

    def test_model_versions_are_empty_not_placeholder(self, client: TestClient) -> None:
        """No model exists, so the honest value is an empty mapping."""
        assert client.get("/api/meta/versions").json()["model_versions"] == {}

    def test_phases_report_phases_1_to_8_implemented(self, client: TestClient) -> None:
        body = client.get("/api/meta/phases").json()
        assert body["current_phase"] == 8
        implemented = [p["phase"] for p in body["phases"] if p["implemented"]]
        assert implemented == [1, 2, 3, 4, 5, 6, 7, 8]
        assert len(body["phases"]) == 14

    def test_dataset_inspection_is_phase_2(self, client: TestClient) -> None:
        """Dataset inspection precedes CV, feature design and model selection."""
        phases = {p["phase"]: p for p in client.get("/api/meta/phases").json()["phases"]}
        assert "Inspect the real dataset" in phases[2]["summary"]
        assert phases[3]["name"] == "Computer Vision"

    def test_analytics_precedes_traffic_state(self, client: TestClient) -> None:
        """Traffic state interprets what analytics produces, so analytics is first."""
        phases = {p["phase"]: p["name"] for p in client.get("/api/meta/phases").json()["phases"]}
        assert phases[4] == "Traffic Analytics"
        assert phases[5] == "Traffic State"
        assert phases[6] == "Road Network + Emergency Management"


class TestSourceCatalog:
    def test_both_input_modes_are_offered(self, client: TestClient) -> None:
        body = client.get("/api/sources/catalog").json()
        modes = {m["mode"]: m for m in body["modes"]}
        assert set(modes) == {"A", "B"}
        assert modes["A"]["is_primary"] is True
        assert modes["B"]["is_primary"] is False

    def test_implemented_kinds_are_exactly_the_recorded_media_sources(
        self, client: TestClient
    ) -> None:
        """Phase 2 implements recorded image and video ingestion, nothing else.

        Asserting the exact set rather than a count, so a source quietly gaining
        an implementation is a test failure rather than a silent capability.
        """
        body = client.get("/api/sources/catalog").json()
        assert set(body["implemented_kinds"]) == {"recorded_image", "recorded_video"}

    def test_every_other_kind_still_reports_unimplemented(
        self, client: TestClient
    ) -> None:
        body = client.get("/api/sources/catalog").json()
        implemented = {"recorded_image", "recorded_video"}
        for mode in body["modes"]:
            for profile in mode["profiles"]:
                expected = profile["kind"] in implemented
                assert profile["implemented"] is expected

    def test_no_live_kind_is_implemented(self, client: TestClient) -> None:
        """Live traffic remains a future capability with no implementation."""
        body = client.get("/api/sources/catalog").json()
        live = next(m for m in body["modes"] if m["mode"] == "B")
        assert all(profile["implemented"] is False for profile in live["profiles"])

    def test_capability_declarations_match_mode_semantics(self, client: TestClient) -> None:
        """Recorded input is bounded and replayable; live input is neither."""
        modes = {m["mode"]: m for m in client.get("/api/sources/catalog").json()["modes"]}
        for profile in modes["A"]["profiles"]:
            assert profile["bounded"] is True
            assert profile["replayable"] is True
            assert profile["realtime"] is False
        for profile in modes["B"]["profiles"]:
            assert profile["bounded"] is False
            assert profile["seekable"] is False
            assert profile["requires_backpressure"] is True

    def test_no_sources_are_configured(self, client: TestClient) -> None:
        body = client.get("/api/sources").json()
        assert body["count"] == 0
        assert body["sources"] == []

    def test_catalog_returns_200_and_never_errors(self, client: TestClient) -> None:
        """Regression: the catalogue is static metadata and must always resolve.

        Describing an input path is a statement about this system's architecture,
        not a claim about traffic, so there is no condition under which it should
        fail or 501.
        """
        response = client.get("/api/sources/catalog")
        assert response.status_code == 200
        body = response.json()
        assert len(body["modes"]) == 2
        assert sum(len(m["profiles"]) for m in body["modes"]) == len(
            body["implemented_kinds"]
        ) + len(body["unavailable_kinds"])

    def test_catalog_states_unavailability_positively(self, client: TestClient) -> None:
        """An empty `implemented_kinds` alone is ambiguous.

        A client reading only the positive list cannot distinguish "nothing is
        implemented" from "the catalogue failed to load", so unavailability is
        stated explicitly.
        """
        body = client.get("/api/sources/catalog").json()
        # Two recorded media sources exist from phase 2, so something *is*
        # available; the unavailable set must still be stated positively.
        assert body["any_source_available"] is True
        assert len(body["unavailable_kinds"]) >= 1
        assert set(body["implemented_kinds"]).isdisjoint(body["unavailable_kinds"])

    def test_catalog_is_stable_across_repeated_calls(self, client: TestClient) -> None:
        """Guards against shared-registry mutation between requests."""
        first = client.get("/api/sources/catalog").json()
        for _ in range(5):
            assert client.get("/api/sources/catalog").json() == first


class TestSafetyCatalogEndpoint:
    def test_catalog_is_served(self, client: TestClient) -> None:
        body = client.get("/api/safety/catalog").json()
        assert body["version"]
        assert len(body["rules"]) >= 10

    def test_soft_rules_ship_without_invented_penalty_weights(
        self, client: TestClient
    ) -> None:
        """Penalty weights are empirical and are calibrated in Phase 11."""
        rules = client.get("/api/safety/catalog").json()["rules"]
        assert all(r["default_penalty_equivalent_s"] is None for r in rules)


class TestNotAvailableYet:
    """Unimplemented capabilities must 501, never return an empty success.

    An empty 200 renders as a network with no congestion and no incidents --
    a specific false claim about the world.
    """

    def test_network_reports_that_none_is_loaded(self, client: TestClient) -> None:
        """Loading landed in phase 6, so this is 404-no-map, not 501-no-feature.

        Still never an empty network: an empty RoadNetwork cannot be built
        without inventing an id and a version, and a caller receiving one
        could not tell "no map is loaded" from "the map is genuinely empty".
        """
        response = client.get("/api/network")
        assert response.status_code == 404
        error = response.json()["error"]
        assert error["code"] == "network_not_loaded"
        assert "POST /api/network/load" in error["message"]

    def test_network_state_reports_that_none_is_loaded(self, client: TestClient) -> None:
        response = client.get("/api/network/state")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "network_not_loaded"

    def test_the_old_recommendation_path_points_at_the_optimizer(
        self, client: TestClient
    ) -> None:
        """Routing exists now, so claiming it does not would be its own lie.

        This endpoint predates route optimization and names no emergency, so it
        cannot route anything itself -- but a 501 would understate the system.
        """
        response = client.get("/api/emergency/recommendation")
        assert response.status_code == 422
        assert "/api/routes/optimize" in response.json()["error"]["message"]

    def test_corridor_staging_is_still_a_phase_boundary(
        self, client: TestClient
    ) -> None:
        """Staging a route into signal advisories proposes changing the world."""
        response = client.get("/api/emergency/corridor")
        assert response.status_code == 501
        assert response.json()["error"]["details"]["phase"] == 9


class TestEmergencyScenarios:
    """An empty collection of *declared events* is a true statement.

    Distinct from `/api/network/state`, which must still 501: traffic exists
    whether or not we observe it, so an empty network state would assert "no
    congestion, no incidents" -- a claim about the world. A scenario exists only
    because someone declared it, so "none declared" is a complete answer.
    """

    def test_scenarios_return_200_with_an_empty_collection(
        self, client: TestClient
    ) -> None:
        response = client.get("/api/emergency/scenarios")
        assert response.status_code == 200
        body = response.json()
        assert body["scenarios"] == []
        assert body["count"] == 0
        assert body["active_count"] == 0

    def test_emptiness_is_explained_so_it_cannot_be_misread(
        self, client: TestClient
    ) -> None:
        """An unexplained empty list could read as "the emergency was cleared"."""
        body = client.get("/api/emergency/scenarios").json()
        assert body["declaration_supported"] is False
        assert body["declaration_phase"] == 6
        assert "No road network is loaded" in body["note"]

    def test_no_emergency_is_invented(self, client: TestClient) -> None:
        """The one output a person might act on must never be fabricated.

        No sample scenario, no demonstration route, no illustrative ETA.
        """
        body = client.get("/api/emergency/scenarios").json()
        assert body["scenarios"] == []
        assert body["count"] == 0
        assert body["active_count"] == 0
        # No numeric literal with a unit may appear anywhere in the payload.
        assert re.search(r"\d+(\.\d+)?\s*(s|sec|min|%|km/h|km)\b", json.dumps(body)) is None

    def test_network_state_never_answers_empty(self, client: TestClient) -> None:
        """The empty-collection reasoning must not leak to observational state.

        A scenario list may be empty because nothing was declared. A network
        state may not: it would assert "no congestion, no incidents" about a
        network that has not even been loaded.
        """
        assert client.get("/api/network/state").status_code == 404


class TestExpectedErrorLogging:
    def test_phase_boundaries_are_not_logged_as_failures(
        self, client: TestClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Regression: a 501 was logged at ERROR as "request failed".

        The response was always correct, but the console made a healthy backend
        look like it was failing, which is how this was reported as a 500.
        """
        with caplog.at_level(logging.INFO, logger="app.main"):
            assert client.get("/api/emergency/corridor").status_code == 501

        records = [r for r in caplog.records if r.name == "app.main"]
        assert records, "the declined request should still be logged"
        assert not [r for r in records if r.levelno >= logging.ERROR], (
            "an expected phase boundary must not be logged at ERROR"
        )

    def test_an_unloaded_network_is_not_logged_as_a_failure(
        self, client: TestClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A freshly started service has no map. That is normal, not a fault.

        Same reasoning as the phase boundary above: an ERROR for the ordinary
        starting state trains an operator to ignore ERROR.
        """
        with caplog.at_level(logging.INFO, logger="app.main"):
            assert client.get("/api/network").status_code == 404

        records = [r for r in caplog.records if r.name == "app.main"]
        assert records, "the declined request should still be logged"
        assert not [r for r in records if r.levelno >= logging.ERROR]

    def test_genuine_faults_are_still_logged_as_errors(self) -> None:
        """The fix must not silence real failures."""
        assert NotAvailableYetError("x", phase=4, phase_name="n").is_expected is True
        assert TrafficSystemError("boom").is_expected is False
        assert DataSourceError("camera gone").is_expected is False


class TestErrorEnvelope:
    def test_unknown_route_uses_the_uniform_envelope(self, client: TestClient) -> None:
        response = client.get("/api/does-not-exist")
        assert response.status_code == 404
        assert "error" in response.json()
        assert set(response.json()["error"]) == {"code", "message", "details"}


class TestOpenApi:
    def test_schema_is_generated(self, client: TestClient) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        paths = response.json()["paths"]
        assert "/api/health" in paths

    def test_advisory_scope_is_documented_in_the_api_description(
        self, client: TestClient
    ) -> None:
        """The no-control-authority constraint belongs in the published contract."""
        description = client.get("/openapi.json").json()["info"]["description"]
        assert "Advisory only" in description
