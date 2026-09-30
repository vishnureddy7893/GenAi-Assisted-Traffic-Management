"""Site-to-junction candidate binding and its human confirmation.

The property under test throughout is the one the feature exists to protect:
**proximity proposes, a human disposes**. A nearest-junction result is evidence
and never a binding, the only operation that creates a binding requires an
explicit confirmation naming a junction, and every way the question can fail to
have an answer -- no map, no site, no positioned junction, nothing near enough --
is a distinguishable outcome rather than a plausible-looking guess.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from network_fixtures import make_junction, make_network, make_segment, square_network

from app.core.config import Settings
from app.core.errors import DataSourceError, ResourceNotFoundError, ValidationFailedError
from app.domain.network import GeoPoint, RoadNetwork
from app.domain.provenance import Provenance, ValueSource
from app.domain.site import (
    BindingConfirmation,
    BindingConfirmationMethod,
    ConfirmedSiteBinding,
    ObservationSite,
)
from app.emergency.zones import haversine_m
from app.main import create_app
from app.routing.mapping import ObservationBinding
from app.sites.bindings import (
    ConfirmedBindingRegistry,
    binding_for,
    validate_junction_exists,
)
from app.sites.candidates import CandidateStatus, evaluate_site
from app.sites.registry import utc_now

AREA = "observation-area:cam-1"

#: Junction positions used by ``square_network``. J1 is the north-west corner.
J1 = GeoPoint(latitude=16.5050, longitude=80.6000)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def binding_settings(tmp_path: Path) -> Settings:
    """Settings with both registries in this test's temporary directory."""
    return Settings(
        app_env="local",
        log_level="ERROR",
        media_dir=tmp_path / "media",
        sites_file=tmp_path / "observation-sites.json",
        confirmed_bindings_file=tmp_path / "confirmed-bindings.json",
    )


@pytest.fixture
def bindings_file(binding_settings: Settings) -> Path:
    """Where confirmed bindings are persisted for this test."""
    return binding_settings.confirmed_bindings_file


@pytest.fixture
def client(binding_settings: Settings) -> Iterator[TestClient]:
    """API client with a loaded square network and temporary registries."""
    app = create_app(binding_settings)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        _load_network(app, square_network())
        yield test_client


@pytest.fixture
def networkless_client(binding_settings: Settings) -> Iterator[TestClient]:
    """API client with no network loaded at all."""
    with TestClient(create_app(binding_settings), raise_server_exceptions=False) as c:
        yield c


def _load_network(app: object, network: RoadNetwork) -> None:
    """Put a network into the app's store without touching the filesystem."""
    import datetime as dt

    from app.network.loader import NetworkSource
    from app.network.store import NetworkStore

    source = NetworkSource(
        relative_path="test.json",
        sha256="0" * 64,
        size_bytes=1,
        loaded_at=dt.datetime.now(tz=dt.UTC),
        network_id=network.network_id,
        version=network.version,
        junction_count=len(network.junctions),
        segment_count=len(network.segments),
        facility_count=len(network.facilities),
    )
    store = NetworkStore()
    store.load(network, source)
    app.state.network_store = store  # type: ignore[attr-defined]


def a_site(
    *,
    latitude: float = 16.5051,
    longitude: float = 80.6001,
    accuracy_m: float | None = None,
    area_id: str = AREA,
) -> ObservationSite:
    """A site, built directly rather than through the API."""
    now = utc_now()
    return ObservationSite(
        observation_area_id=area_id,
        location=GeoPoint(latitude=latitude, longitude=longitude),
        accuracy_m=accuracy_m,
        provenance=Provenance(
            source=ValueSource.OBSERVED,
            producer="test.fixture",
            producer_version="1.0",
            observed_at=now,
            computed_at=now,
        ),
        recorded_at=now,
        updated_at=now,
    )


def a_confirmation(
    *,
    junction_id: str = "J1",
    network_id: str = "test-net",
    area_id: str = AREA,
    method: BindingConfirmationMethod = BindingConfirmationMethod.ACCEPTED_NEAREST,
) -> ConfirmedSiteBinding:
    """A confirmed binding, built directly."""
    now = utc_now()
    return ConfirmedSiteBinding(
        observation_area_id=area_id,
        junction_id=junction_id,
        network_id=network_id,
        network_version="1.0",
        site_location=J1,
        confirmation=BindingConfirmation(
            confirmed_by="tester",
            confirmed_at=now,
            method=method,
            candidate_distance_m=12.0,
        ),
        provenance=Provenance(
            source=ValueSource.OBSERVED,
            producer="tester",
            producer_version="1.0",
            observed_at=now,
            computed_at=now,
        ),
    )


def record_site(client: TestClient, **overrides: object) -> str:
    """Record a site through the API and return its observation area id."""
    payload: dict[str, object] = {
        "observation_area_id": AREA,
        "latitude": 16.5051,
        "longitude": 80.6001,
    }
    payload.update(overrides)
    response = client.post("/api/sites", json=payload)
    assert response.status_code in (200, 201), response.text
    return str(payload["observation_area_id"])


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------
class TestDistance:
    def test_the_existing_haversine_is_reused(self) -> None:
        """One distance implementation, or two that will eventually disagree."""
        site = a_site()
        candidate = evaluate_site(
            site, square_network(), max_distance_m=250.0, computed_at=utc_now()
        )
        assert candidate.nearest is not None
        assert candidate.nearest.distance_m == pytest.approx(
            haversine_m(site.location, J1)
        )

    def test_the_distance_is_metres_and_plausible(self) -> None:
        """0.0001 deg of latitude is about 11 m; the result must be in that scale."""
        site = a_site(latitude=16.5051, longitude=80.6000)
        candidate = evaluate_site(
            site, square_network(), max_distance_m=250.0, computed_at=utc_now()
        )
        assert candidate.nearest is not None
        assert 10.0 < candidate.nearest.distance_m < 13.0

    def test_a_site_exactly_on_a_junction_measures_zero(self) -> None:
        site = a_site(latitude=J1.latitude, longitude=J1.longitude)
        candidate = evaluate_site(
            site, square_network(), max_distance_m=250.0, computed_at=utc_now()
        )
        assert candidate.nearest is not None
        assert candidate.nearest.distance_m == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------
class TestCandidateEvaluation:
    def test_the_nearest_junction_is_found_and_the_rest_are_ranked(self) -> None:
        candidate = evaluate_site(
            a_site(), square_network(), max_distance_m=250.0, computed_at=utc_now()
        )
        assert candidate.status is CandidateStatus.CANDIDATE_FOUND
        assert candidate.nearest is not None
        assert candidate.nearest.junction_id == "J1"
        assert candidate.nearest.junction_name == "Junction J1"
        assert [item.junction_id for item in candidate.runners_up] == ["J2", "J4", "J3"]
        distances = [item.distance_m for item in candidate.runners_up]
        assert distances == sorted(distances)

    def test_a_candidate_is_never_a_binding(self) -> None:
        """The flag is fixed true and cannot be turned off."""
        candidate = evaluate_site(
            a_site(), square_network(), max_distance_m=250.0, computed_at=utc_now()
        )
        assert candidate.requires_human_confirmation is True
        with pytest.raises(Exception, match="requires_human_confirmation"):
            candidate.model_validate(
                candidate.model_dump() | {"requires_human_confirmation": False}
            )

    def test_the_candidate_is_derived_and_names_its_inputs(self) -> None:
        """Proximity is computed, so it carries derived provenance -- not observed."""
        site = a_site()
        network = square_network()
        candidate = evaluate_site(
            site, network, max_distance_m=250.0, computed_at=utc_now()
        )
        assert candidate.provenance.source is ValueSource.DERIVED
        assert f"observation_site:{site.observation_area_id}" in (
            candidate.provenance.input_refs
        )
        assert f"network:{network.network_id}@{network.version}" in (
            candidate.provenance.input_refs
        )

    def test_a_junction_beyond_the_range_is_reported_not_offered(self) -> None:
        """Naming the too-distant junction is what reveals a mistyped coordinate."""
        far = a_site(latitude=16.6000, longitude=80.7000)
        candidate = evaluate_site(
            far, square_network(), max_distance_m=250.0, computed_at=utc_now()
        )
        assert candidate.status is CandidateStatus.OUT_OF_RANGE
        assert candidate.is_bindable is False
        assert candidate.nearest is not None
        assert candidate.nearest.distance_m > 250.0
        assert any("beyond the" in warning for warning in candidate.warnings)

    def test_the_range_boundary_is_inclusive(self) -> None:
        site = a_site()
        distance = haversine_m(site.location, J1)
        inside = evaluate_site(
            site, square_network(), max_distance_m=distance, computed_at=utc_now()
        )
        outside = evaluate_site(
            site,
            square_network(),
            max_distance_m=distance - 0.001,
            computed_at=utc_now(),
        )
        assert inside.status is CandidateStatus.CANDIDATE_FOUND
        assert outside.status is CandidateStatus.OUT_OF_RANGE

    def test_an_unsurveyed_network_cannot_answer_the_question(self) -> None:
        """Distinct from "nothing is near": nothing could be measured at all."""
        candidate = evaluate_site(
            a_site(),
            square_network(positioned=False),
            max_distance_m=250.0,
            computed_at=utc_now(),
        )
        assert candidate.status is CandidateStatus.NO_POSITIONED_JUNCTION
        assert candidate.nearest is None
        assert candidate.positioned_junction_count == 0
        assert len(candidate.unpositioned_junction_ids) == 4

    def test_unpositioned_junctions_are_named_rather_than_dropped(self) -> None:
        """The nearest of those that could be tested is not the nearest junction."""
        network = make_network(
            junctions=(
                make_junction("J1", position=J1),
                make_junction("J-UNKNOWN"),
            ),
            segments=(
                make_segment("SEG-1", "J1", "J-UNKNOWN"),
                make_segment("SEG-2", "J-UNKNOWN", "J1"),
            ),
        )
        candidate = evaluate_site(
            a_site(), network, max_distance_m=250.0, computed_at=utc_now()
        )
        assert candidate.unpositioned_junction_ids == ("J-UNKNOWN",)
        assert candidate.positioned_junction_count == 1
        assert any("not measured" in warning for warning in candidate.warnings)


# ---------------------------------------------------------------------------
# Accuracy as evidence
# ---------------------------------------------------------------------------
class TestAccuracyContext:
    def test_a_distance_inside_the_accuracy_is_not_discriminating(self) -> None:
        """The coordinate cannot tell this junction from anything in that radius."""
        site = a_site(accuracy_m=100.0)
        candidate = evaluate_site(
            site, square_network(), max_distance_m=250.0, computed_at=utc_now()
        )
        assert candidate.nearest is not None
        assert candidate.nearest.within_site_accuracy is True
        assert candidate.is_discriminating is False
        assert any("position accuracy" in warning for warning in candidate.warnings)

    def test_a_runner_up_inside_the_accuracy_makes_the_ranking_noise(self) -> None:
        """J1 and J2 are ~550 m apart; a 600 m accuracy cannot separate them."""
        site = a_site(accuracy_m=600.0)
        candidate = evaluate_site(
            site, square_network(), max_distance_m=2000.0, computed_at=utc_now()
        )
        assert candidate.is_discriminating is False
        assert any(
            "measurement noise" in warning for warning in candidate.warnings
        )

    def test_a_tight_accuracy_leaves_the_ranking_decisive(self) -> None:
        site = a_site(accuracy_m=5.0)
        candidate = evaluate_site(
            site, square_network(), max_distance_m=250.0, computed_at=utc_now()
        )
        assert candidate.is_discriminating is True
        assert candidate.warnings == ()

    def test_an_unreported_accuracy_is_not_treated_as_good(self) -> None:
        """No accuracy means no accuracy claim -- never a tacit perfect one."""
        candidate = evaluate_site(
            a_site(accuracy_m=None),
            square_network(),
            max_distance_m=250.0,
            computed_at=utc_now(),
        )
        assert candidate.site_accuracy_m is None
        assert candidate.nearest is not None
        assert candidate.nearest.within_site_accuracy is False


# ---------------------------------------------------------------------------
# Confirmed binding registry
# ---------------------------------------------------------------------------
class TestBindingRegistry:
    def test_an_absent_file_is_an_empty_registry(self, bindings_file: Path) -> None:
        registry = ConfirmedBindingRegistry(bindings_file)
        assert registry.is_degraded is False
        assert registry.list() == ()

    def test_a_confirmation_is_written_and_read_back(self, bindings_file: Path) -> None:
        ConfirmedBindingRegistry(bindings_file).confirm(a_confirmation())
        reopened = ConfirmedBindingRegistry(bindings_file)
        assert reopened.get("test-net", AREA).junction_id == "J1"

    def test_re_confirming_replaces_rather_than_duplicates(
        self, bindings_file: Path
    ) -> None:
        registry = ConfirmedBindingRegistry(bindings_file)
        registry.confirm(a_confirmation(junction_id="J1"))
        _, replaced = registry.confirm(
            a_confirmation(
                junction_id="J2", method=BindingConfirmationMethod.SURVEYED
            )
        )
        assert replaced is True
        assert len(registry) == 1
        assert registry.get("test-net", AREA).junction_id == "J2"

    def test_bindings_are_kept_per_network(self, bindings_file: Path) -> None:
        """A junction id means something different on a different map."""
        registry = ConfirmedBindingRegistry(bindings_file)
        registry.confirm(a_confirmation(network_id="net-a"))
        registry.confirm(a_confirmation(network_id="net-b", junction_id="J2"))
        assert len(registry) == 2
        assert len(registry.list("net-a")) == 1
        assert registry.get("net-b", AREA).junction_id == "J2"

    def test_an_unconfirmed_area_is_a_404(self, bindings_file: Path) -> None:
        registry = ConfirmedBindingRegistry(bindings_file)
        with pytest.raises(ResourceNotFoundError):
            registry.get("test-net", AREA)
        assert registry.find("test-net", AREA) is None

    def test_a_binding_can_be_withdrawn(self, bindings_file: Path) -> None:
        registry = ConfirmedBindingRegistry(bindings_file)
        registry.confirm(a_confirmation())
        registry.revoke("test-net", AREA)
        assert len(registry) == 0
        assert len(ConfirmedBindingRegistry(bindings_file)) == 0

    def test_an_unreadable_file_is_not_reported_as_empty_or_overwritten(
        self, bindings_file: Path
    ) -> None:
        bindings_file.parent.mkdir(parents=True, exist_ok=True)
        bindings_file.write_text("{ not json", encoding="utf-8")

        registry = ConfirmedBindingRegistry(bindings_file)
        assert registry.is_degraded is True
        with pytest.raises(DataSourceError):
            registry.list()
        with pytest.raises(DataSourceError):
            registry.confirm(a_confirmation())
        assert bindings_file.read_text(encoding="utf-8") == "{ not json"

    def test_a_stale_network_version_is_reported_not_invalidated(
        self, bindings_file: Path
    ) -> None:
        registry = ConfirmedBindingRegistry(bindings_file)
        registry.confirm(a_confirmation(network_id="test-net"))
        resurveyed = make_network(network_id="test-net", version="2.0")
        assert registry.stale_version_ids(resurveyed) == (AREA,)
        # Still present: a re-surveyed map usually keeps its junction ids.
        assert len(registry.list("test-net")) == 1


# ---------------------------------------------------------------------------
# Mapping integration
# ---------------------------------------------------------------------------
class TestMappingIntegration:
    def test_a_confirmation_renders_to_the_existing_binding_type(self) -> None:
        binding = binding_for(a_confirmation())
        assert isinstance(binding, ObservationBinding)
        assert binding.observation_area_id == AREA
        assert binding.junction_id == "J1"
        assert binding.segment_id is None
        assert binding.coverage_note is not None
        assert "tester" in binding.coverage_note

    def test_a_junction_binding_prices_no_segment(self) -> None:
        """Existing mapping semantics: costing groups only segment bindings.

        This is why confirming a junction cannot change a route. Knowing which
        intersection a camera watches is not knowing which directed segment its
        queue belongs to.
        """
        from app.routing.mapping import ObservationMapping

        now = utc_now()
        mapping = ObservationMapping(
            mapping_id="m1",
            version="1",
            network_id="test-net",
            provenance=Provenance(
                source=ValueSource.OBSERVED,
                producer="tester",
                producer_version="1",
                observed_at=now,
                computed_at=now,
            ),
            bindings=(binding_for(a_confirmation()),),
        )
        assert mapping.bindings_by_segment() == {}
        assert mapping.segment_for(AREA) is None
        assert AREA in mapping.area_ids

    def test_an_existing_mapping_file_without_confirmation_still_validates(
        self,
    ) -> None:
        """Compatibility: the new field is optional and old bindings are unchanged."""
        binding = ObservationBinding(
            observation_area_id=AREA, segment_id="SEG-12", coverage_note="surveyed"
        )
        assert binding.confirmation is None
        assert binding.segment_id == "SEG-12"

    def test_a_binding_can_carry_its_confirmation(self) -> None:
        confirmed = a_confirmation()
        binding = ObservationBinding(
            observation_area_id=AREA,
            junction_id="J1",
            confirmation=confirmed.confirmation,
        )
        assert binding.confirmation is not None
        assert binding.confirmation.confirmed_by == "tester"

    def test_a_junction_the_network_lacks_is_refused(self) -> None:
        with pytest.raises(ValidationFailedError, match="does not exist"):
            validate_junction_exists("J-NOPE", square_network())

    def test_a_junction_the_network_has_is_accepted(self) -> None:
        validate_junction_exists("J1", square_network())


# ---------------------------------------------------------------------------
# Confirmation domain rules
# ---------------------------------------------------------------------------
class TestConfirmationRules:
    def test_overriding_the_nearest_junction_requires_a_reason(self) -> None:
        now = utc_now()
        with pytest.raises(Exception, match="notes saying why"):
            BindingConfirmation(
                confirmed_by="tester",
                confirmed_at=now,
                method=BindingConfirmationMethod.CHOSE_DIFFERENT,
            )

    def test_an_override_with_a_reason_is_accepted(self) -> None:
        now = utc_now()
        confirmation = BindingConfirmation(
            confirmed_by="tester",
            confirmed_at=now,
            method=BindingConfirmationMethod.CHOSE_DIFFERENT,
            notes="the camera faces the next junction north",
        )
        assert confirmation.notes is not None

    def test_there_is_no_automatic_confirmation_method(self) -> None:
        """A binding nothing human decided is what this design refuses to create."""
        values = {item.value for item in BindingConfirmationMethod}
        assert "automatic" not in values
        assert "inferred" not in values

    def test_a_confirmation_is_observed_not_derived(self) -> None:
        """The candidate is derived from distance; the decision is not."""
        assert a_confirmation().provenance.source is ValueSource.OBSERVED


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
class TestCandidateApi:
    def test_no_network_loaded_is_its_own_error_code(
        self, networkless_client: TestClient
    ) -> None:
        """Distinct from an unknown site: the two call for different responses."""
        networkless_client.post(
            "/api/sites",
            json={
                "observation_area_id": AREA,
                "latitude": 16.5051,
                "longitude": 80.6001,
            },
        )
        response = networkless_client.get(f"/api/sites/{AREA}/candidate")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "network_not_loaded"

    def test_an_unknown_site_is_a_plain_not_found(self, client: TestClient) -> None:
        response = client.get("/api/sites/observation-area:never-surveyed/candidate")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"

    def test_a_candidate_is_returned_with_its_distance(
        self, client: TestClient
    ) -> None:
        record_site(client)
        body = client.get(f"/api/sites/{AREA}/candidate").json()

        assert body["candidate"]["status"] == "candidate_found"
        assert body["candidate"]["nearest"]["junction_id"] == "J1"
        assert body["candidate"]["nearest"]["distance_m"] > 0
        assert body["candidate"]["requires_human_confirmation"] is True
        assert body["confirmed"] is None
        assert "proposal" in body["note"]

    def test_the_site_is_returned_alongside_the_candidate(
        self, client: TestClient
    ) -> None:
        record_site(client, label="NW mast")
        body = client.get(f"/api/sites/{AREA}/candidate").json()
        assert body["site"]["label"] == "NW mast"
        assert body["site"]["observation_area_id"] == AREA

    def test_accuracy_is_carried_into_the_candidate(self, client: TestClient) -> None:
        record_site(client, accuracy_m=100.0)
        body = client.get(f"/api/sites/{AREA}/candidate").json()
        assert body["candidate"]["site_accuracy_m"] == 100.0
        assert body["candidate"]["is_discriminating"] is False
        assert body["candidate"]["warnings"]

    def test_an_out_of_range_site_offers_no_binding(self, client: TestClient) -> None:
        record_site(client, latitude=16.7, longitude=80.9)
        body = client.get(f"/api/sites/{AREA}/candidate").json()
        assert body["candidate"]["status"] == "out_of_range"
        assert body["candidate"]["nearest"] is not None
        assert "outside the" in body["note"]

    def test_evaluating_a_candidate_creates_nothing(self, client: TestClient) -> None:
        """The refusal to auto-bind, stated as a test.

        Evaluating is a GET, and there is no path from it to the registry.
        """
        record_site(client)
        for _ in range(3):
            assert client.get(f"/api/sites/{AREA}/candidate").status_code == 200
        assert client.get("/api/sites/confirmed").json()["count"] == 0


class TestConfirmationApi:
    def test_a_binding_needs_an_explicit_confirmation(
        self, client: TestClient
    ) -> None:
        record_site(client)
        response = client.post(
            f"/api/sites/{AREA}/binding", json={"junction_id": "J1"}
        )
        assert response.status_code == 422
        assert "confirm=true" in response.json()["error"]["message"]
        assert client.get("/api/sites/confirmed").json()["count"] == 0

    def test_a_confirmed_binding_is_created(self, client: TestClient) -> None:
        record_site(client, accuracy_m=5.0)
        response = client.post(
            f"/api/sites/{AREA}/binding",
            json={"junction_id": "J1", "confirm": True, "confirmed_by": "surveyor.a"},
        )
        assert response.status_code == 201
        body = response.json()

        assert body["confirmed"]["junction_id"] == "J1"
        assert body["confirmed"]["network_id"] == "test-net"
        assert body["confirmed"]["confirmation"]["confirmed_by"] == "surveyor.a"
        assert body["confirmed"]["confirmation"]["method"] == "accepted_nearest"
        assert body["confirmed"]["confirmation"]["was_nearest_candidate"] is True
        assert body["confirmed"]["confirmation"]["candidate_distance_m"] > 0
        assert body["confirmed"]["confirmation"]["site_accuracy_m"] == 5.0
        assert body["confirmed"]["provenance"]["source"] == "observed"
        assert body["binding"]["junction_id"] == "J1"
        assert body["binding"]["segment_id"] is None

    def test_the_site_location_at_confirmation_is_kept(
        self, client: TestClient
    ) -> None:
        """Re-surveying the site later must not rewrite past evidence."""
        record_site(client)
        confirmed = client.post(
            f"/api/sites/{AREA}/binding", json={"junction_id": "J1", "confirm": True}
        ).json()["confirmed"]
        assert confirmed["site_location"]["latitude"] == 16.5051

        client.post(
            "/api/sites",
            json={
                "observation_area_id": AREA,
                "latitude": 16.5000,
                "longitude": 80.6000,
            },
        )
        stored = client.get("/api/sites/confirmed").json()["bindings"][0]
        assert stored["site_location"]["latitude"] == 16.5051

    def test_confirming_a_junction_that_is_not_nearest_needs_a_reason(
        self, client: TestClient
    ) -> None:
        record_site(client)
        refused = client.post(
            f"/api/sites/{AREA}/binding", json={"junction_id": "J3", "confirm": True}
        )
        assert refused.status_code == 422
        assert "not the nearest" in refused.json()["error"]["message"]

        accepted = client.post(
            f"/api/sites/{AREA}/binding",
            json={
                "junction_id": "J3",
                "confirm": True,
                "notes": "the camera faces south-east across the flyover",
            },
        )
        assert accepted.status_code == 201
        confirmation = accepted.json()["confirmed"]["confirmation"]
        assert confirmation["method"] == "chose_different"
        assert confirmation["was_nearest_candidate"] is False

    def test_a_junction_the_network_lacks_is_refused(self, client: TestClient) -> None:
        record_site(client)
        response = client.post(
            f"/api/sites/{AREA}/binding",
            json={"junction_id": "J-NOPE", "confirm": True, "notes": "typo"},
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_failed"
        assert client.get("/api/sites/confirmed").json()["count"] == 0

    def test_confirming_without_a_network_is_refused(
        self, networkless_client: TestClient
    ) -> None:
        networkless_client.post(
            "/api/sites",
            json={
                "observation_area_id": AREA,
                "latitude": 16.5051,
                "longitude": 80.6001,
            },
        )
        response = networkless_client.post(
            f"/api/sites/{AREA}/binding", json={"junction_id": "J1", "confirm": True}
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "network_not_loaded"

    def test_confirming_an_unknown_site_is_refused(self, client: TestClient) -> None:
        response = client.post(
            "/api/sites/observation-area:ghost/binding",
            json={"junction_id": "J1", "confirm": True},
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"

    def test_re_confirming_replaces_and_answers_200(self, client: TestClient) -> None:
        record_site(client)
        first = client.post(
            f"/api/sites/{AREA}/binding", json={"junction_id": "J1", "confirm": True}
        )
        second = client.post(
            f"/api/sites/{AREA}/binding",
            json={
                "junction_id": "J2",
                "confirm": True,
                "notes": "remounted facing the east junction",
            },
        )
        assert first.status_code == 201
        assert second.status_code == 200
        assert second.json()["replaced_existing"] is True
        assert client.get("/api/sites/confirmed").json()["count"] == 1

    def test_a_binding_can_be_withdrawn(self, client: TestClient) -> None:
        record_site(client)
        client.post(
            f"/api/sites/{AREA}/binding", json={"junction_id": "J1", "confirm": True}
        )
        response = client.request("DELETE", f"/api/sites/{AREA}/binding")
        assert response.status_code == 200
        assert "withdrawn" in response.json()["note"]
        assert client.get("/api/sites/confirmed").json()["count"] == 0

    def test_withdrawing_a_binding_that_was_never_made_is_a_404(
        self, client: TestClient
    ) -> None:
        response = client.request("DELETE", f"/api/sites/{AREA}/binding")
        assert response.status_code == 404

    def test_the_candidate_reports_an_existing_binding(
        self, client: TestClient
    ) -> None:
        record_site(client)
        client.post(
            f"/api/sites/{AREA}/binding", json={"junction_id": "J1", "confirm": True}
        )
        body = client.get(f"/api/sites/{AREA}/candidate").json()
        assert body["confirmed"] is not None
        assert body["confirmed"]["junction_id"] == "J1"
        assert "replace" in body["note"]


class TestConfirmedListingApi:
    def test_nothing_confirmed_is_an_empty_200(self, client: TestClient) -> None:
        response = client.get("/api/sites/confirmed")
        assert response.status_code == 200
        body = response.json()
        assert body["bindings"] == []
        assert body["count"] == 0
        assert "only when an operator says so" in body["note"]

    def test_bindings_are_listed_for_the_loaded_network(
        self, client: TestClient
    ) -> None:
        record_site(client)
        client.post(
            f"/api/sites/{AREA}/binding", json={"junction_id": "J1", "confirm": True}
        )
        body = client.get("/api/sites/confirmed").json()
        assert body["count"] == 1
        assert body["network_id"] == "test-net"
        assert body["bindings"][0]["observation_area_id"] == AREA

    def test_the_site_route_is_not_shadowed_by_the_sub_routes(
        self, client: TestClient
    ) -> None:
        """An area id contains a colon and the paths must still resolve."""
        record_site(client)
        assert client.get(f"/api/sites/{AREA}").status_code == 200
        assert client.get(f"/api/sites/{AREA}/candidate").status_code == 200
        assert client.get("/api/sites").json()["count"] == 1


class TestMappingStatusReporting:
    def test_confirmed_bindings_are_reported_beside_the_mapping(
        self, client: TestClient
    ) -> None:
        record_site(client)
        client.post(
            f"/api/sites/{AREA}/binding", json={"junction_id": "J1", "confirm": True}
        )
        body = client.get("/api/routes/mapping").json()

        assert body["confirmed_binding_count"] == 1
        assert body["confirmed_observation_area_ids"] == [AREA]
        # The loaded mapping itself is untouched: a confirmation is not merged
        # into the surveyed file, because they are not the same evidence.
        assert body["loaded"] is False
        assert body["bound_observation_area_ids"] == []

    def test_mapping_status_is_unchanged_when_nothing_is_confirmed(
        self, client: TestClient
    ) -> None:
        body = client.get("/api/routes/mapping").json()
        assert body["confirmed_binding_count"] == 0
        assert body["confirmed_observation_area_ids"] == []
        assert body["loaded"] is False
        assert body["note"]


class TestPersistenceAcrossRestart:
    def test_a_confirmed_binding_survives_a_restart(
        self, binding_settings: Settings
    ) -> None:
        app = create_app(binding_settings)
        with TestClient(app) as first:
            _load_network(app, square_network())
            first.post(
                "/api/sites",
                json={
                    "observation_area_id": AREA,
                    "latitude": 16.5051,
                    "longitude": 80.6001,
                },
            )
            confirmed = first.post(
                f"/api/sites/{AREA}/binding",
                json={"junction_id": "J1", "confirm": True, "confirmed_by": "surveyor"},
            )
            assert confirmed.status_code == 201

        rebuilt = create_app(binding_settings)
        with TestClient(rebuilt) as second:
            _load_network(rebuilt, square_network())
            body = second.get("/api/sites/confirmed").json()
            assert body["count"] == 1
            assert body["bindings"][0]["junction_id"] == "J1"
            assert body["bindings"][0]["confirmation"]["confirmed_by"] == "surveyor"

    def test_the_two_registries_are_stored_separately(
        self, binding_settings: Settings
    ) -> None:
        """A correction to a coordinate must not rewrite a decision about it."""
        assert binding_settings.sites_file != binding_settings.confirmed_bindings_file

        app = create_app(binding_settings)
        with TestClient(app) as c:
            _load_network(app, square_network())
            c.post(
                "/api/sites",
                json={
                    "observation_area_id": AREA,
                    "latitude": 16.5051,
                    "longitude": 80.6001,
                },
            )
            c.post(
                f"/api/sites/{AREA}/binding",
                json={"junction_id": "J1", "confirm": True},
            )

        sites_document = json.loads(
            binding_settings.sites_file.read_text(encoding="utf-8")
        )
        bindings_document = json.loads(
            binding_settings.confirmed_bindings_file.read_text(encoding="utf-8")
        )
        assert "sites" in sites_document
        assert "bindings" in bindings_document
        assert len(bindings_document["bindings"]) == 1


# ---------------------------------------------------------------------------
# Regression: nothing that existed before this step behaves differently
# ---------------------------------------------------------------------------
def _ranking(result: dict[str, object]) -> list[tuple[list[str], object]]:
    """The roads each ranked route uses and what it costs, in rank order."""
    routes = result["routes"]
    assert isinstance(routes, list)
    return [
        (
            [leg["segment_id"] for leg in route["candidate"]["legs"]],
            route["candidate"]["cost"],
        )
        for route in routes
    ]



class TestExistingMappingRegression:
    def test_loading_a_mapping_file_still_works(self, client: TestClient) -> None:
        """The committed example mapping loads against its own network.

        Guards the file-loaded path against the ``confirmation`` field that was
        added to ``ObservationBinding``.
        """
        from app.api.routes import network as network_routes  # noqa: F401

        loaded = client.post(
            "/api/network/load", json={"relative_path": "example-grid.json"}
        )
        assert loaded.status_code == 200

        response = client.post(
            "/api/routes/mapping/load",
            json={"relative_path": "example-grid-mapping.json"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["loaded"] is True
        assert body["mapping"]["bindings"]
        assert all(
            binding["confirmation"] is None for binding in body["mapping"]["bindings"]
        )

    def test_route_optimization_is_unaffected_by_a_confirmed_binding(
        self, client: TestClient
    ) -> None:
        """A junction binding prices nothing, so a route cannot change."""
        client.post("/api/network/load", json={"relative_path": "example-grid.json"})
        request = {
            "origin_junction_id": "J1",
            "destination_junction_id": "J3",
            "mode": "normal",
        }
        before = client.post("/api/routes/optimize", json=request)
        assert before.status_code == 200

        client.post(
            "/api/sites",
            json={
                "observation_area_id": AREA,
                "latitude": 16.5051,
                "longitude": 80.6001,
            },
        )
        confirmed = client.post(
            f"/api/sites/{AREA}/binding", json={"junction_id": "J1", "confirm": True}
        )
        assert confirmed.status_code == 201

        after = client.post("/api/routes/optimize", json=request)
        assert after.status_code == 200
        # Route ids are freshly generated per request, so the comparison is on
        # what the ranking actually says: the same roads, at the same cost.
        assert after.json()["status"] == before.json()["status"]
        assert _ranking(after.json()) == _ranking(before.json())
        assert after.json()["unmapped_observation_area_ids"] == (
            before.json()["unmapped_observation_area_ids"]
        )

    def test_the_example_grid_remains_the_offline_demo_network(
        self, client: TestClient
    ) -> None:
        loaded = client.post(
            "/api/network/load", json={"relative_path": "example-grid.json"}
        )
        assert loaded.status_code == 200
        assert loaded.json()["source"]["network_id"] == "example-grid"

    def test_recording_a_site_still_works_unchanged(self, client: TestClient) -> None:
        response = client.post(
            "/api/sites",
            json={
                "observation_area_id": AREA,
                "latitude": 16.5051,
                "longitude": 80.6001,
            },
        )
        assert response.status_code == 201
        assert response.json()["replaced_existing"] is False
