"""Refining a confirmed junction binding to one directed road segment.

Step 2 established that proximity proposes and a human disposes. This step keeps
that rule under a harder case: at a junction, the two directions of every road
lie on the same line, so the geometry can narrow the choice to a *line* and can
never narrow it to a *direction of travel*. A camera filming oncoming traffic and
one filming the backs of departing vehicles stand in the same place and point the
same way.

So the tests below assert that bearing is reported, that it never selects, that
the listing is not even ordered by it, and that a segment confirmation refines
the junction decision rather than quietly replacing it. The last group asserts
the thing most easily broken by accident: a segment binding still changes no
route cost.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from network_fixtures import make_junction, make_network, make_segment, square_network

from app.core.config import Settings
from app.core.errors import ValidationFailedError
from app.domain.network import GeoPoint, RoadNetwork
from app.domain.provenance import Provenance, ValueSource
from app.domain.site import (
    BindingConfirmation,
    BindingConfirmationMethod,
    ConfirmedSiteBinding,
    ObservationSite,
    SegmentDirection,
)
from app.main import create_app
from app.sites.bindings import (
    ConfirmedBindingRegistry,
    binding_for,
    validate_approach,
    validate_network_version,
    validate_segment_incident,
)
from app.sites.candidates import (
    angular_separation_degrees,
    bearing_degrees_between,
    evaluate_segments,
    incident_segments,
)
from app.sites.registry import utc_now

AREA = "observation-area:cam-1"

#: ``square_network`` corners. J1 north-west, J2 north-east (due east of J1),
#: J4 south-west (due south of J1).
J1 = GeoPoint(latitude=16.5050, longitude=80.6000)
J2 = GeoPoint(latitude=16.5050, longitude=80.6050)
J4 = GeoPoint(latitude=16.5000, longitude=80.6000)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def binding_settings(tmp_path: Path) -> Settings:
    return Settings(
        app_env="local",
        log_level="ERROR",
        media_dir=tmp_path / "media",
        sites_file=tmp_path / "observation-sites.json",
        confirmed_bindings_file=tmp_path / "confirmed-bindings.json",
    )


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


@pytest.fixture
def client(binding_settings: Settings) -> Iterator[TestClient]:
    app = create_app(binding_settings)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        _load_network(app, square_network())
        yield test_client


def a_site(
    *,
    bearing_degrees: float | None = 90.0,
    accuracy_m: float | None = None,
) -> ObservationSite:
    """A site just south-east of J1, facing due east by default."""
    now = utc_now()
    return ObservationSite(
        observation_area_id=AREA,
        location=GeoPoint(latitude=16.5051, longitude=80.6001),
        bearing_degrees=bearing_degrees,
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


def a_junction_binding(
    *,
    network_version: str | None = "1.0",
    junction_id: str = "J1",
) -> ConfirmedSiteBinding:
    """A confirmed junction binding, with no segment chosen yet."""
    now = utc_now()
    return ConfirmedSiteBinding(
        observation_area_id=AREA,
        junction_id=junction_id,
        network_id="test-net",
        network_version=network_version,
        site_location=J1,
        confirmation=BindingConfirmation(
            confirmed_by="surveyor.junction",
            confirmed_at=now,
            method=BindingConfirmationMethod.ACCEPTED_NEAREST,
            candidate_distance_m=15.4,
        ),
        provenance=Provenance(
            source=ValueSource.OBSERVED,
            producer="surveyor.junction",
            producer_version="1.0",
            observed_at=now,
            computed_at=now,
        ),
    )


def a_segment_confirmation(
    *,
    method: BindingConfirmationMethod = BindingConfirmationMethod.ALIGNED_WITH_BEARING,
    notes: str | None = None,
) -> BindingConfirmation:
    return BindingConfirmation(
        confirmed_by="surveyor.segment",
        confirmed_at=utc_now(),
        method=method,
        notes=notes,
    )


def set_up(
    client: TestClient,
    *,
    bearing_degrees: float | None = 90.0,
    junction_id: str = "J1",
) -> None:
    """Record a site and confirm its junction, through the API."""
    payload: dict[str, object] = {
        "observation_area_id": AREA,
        "latitude": 16.5051,
        "longitude": 80.6001,
    }
    if bearing_degrees is not None:
        payload["bearing_degrees"] = bearing_degrees
    assert client.post("/api/sites", json=payload).status_code == 201
    confirmed = client.post(
        f"/api/sites/{AREA}/binding",
        json={
            "junction_id": junction_id,
            "confirm": True,
            "confirmed_by": "surveyor.junction",
            **({"notes": "junction chosen by survey"} if junction_id != "J1" else {}),
        },
    )
    assert confirmed.status_code == 201, confirmed.text


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
class TestBearingGeometry:
    def test_due_east_is_ninety_degrees(self) -> None:
        assert bearing_degrees_between(J1, J2) == pytest.approx(90.0, abs=0.01)

    def test_due_south_is_one_hundred_and_eighty(self) -> None:
        assert bearing_degrees_between(J1, J4) == pytest.approx(180.0, abs=0.01)

    def test_the_reverse_bearing_is_the_opposite_direction(self) -> None:
        """The two ends of one road differ by 180 degrees. That is the problem."""
        out = bearing_degrees_between(J1, J2)
        back = bearing_degrees_between(J2, J1)
        assert angular_separation_degrees(out, back) == pytest.approx(180.0, abs=0.01)

    def test_bearings_are_normalised_to_a_single_turn(self) -> None:
        assert 0.0 <= bearing_degrees_between(J2, J1) < 360.0

    def test_angular_separation_wraps_the_short_way(self) -> None:
        assert angular_separation_degrees(350.0, 10.0) == pytest.approx(20.0)
        assert angular_separation_degrees(10.0, 350.0) == pytest.approx(20.0)
        assert angular_separation_degrees(0.0, 180.0) == pytest.approx(180.0)


# ---------------------------------------------------------------------------
# Incidence
# ---------------------------------------------------------------------------
class TestIncidentSegments:
    def test_both_directions_of_every_arm_are_offered(self) -> None:
        """Excluding the outgoing arms would decide the direction for the operator."""
        found = incident_segments(square_network(), "J1")
        by_id = {segment.segment_id: direction for segment, direction in found}
        assert by_id == {
            "SEG-12": SegmentDirection.OUTGOING,
            "SEG-14": SegmentDirection.OUTGOING,
            "SEG-21": SegmentDirection.INCOMING,
            "SEG-41": SegmentDirection.INCOMING,
        }

    def test_segments_elsewhere_on_the_map_are_not_incident(self) -> None:
        ids = {segment.segment_id for segment, _ in incident_segments(square_network(), "J1")}
        assert "SEG-23" not in ids
        assert "SEG-34" not in ids

    def test_a_junction_with_no_segments_offers_nothing(self) -> None:
        network = make_network(
            junctions=(make_junction("J1", position=J1), make_junction("J9", position=J2)),
            segments=(
                make_segment("SEG-19", "J1", "J9"),
                make_segment("SEG-91", "J9", "J1"),
            ),
        )
        assert incident_segments(network, "J-ABSENT") == ()


# ---------------------------------------------------------------------------
# Segment candidate listing
# ---------------------------------------------------------------------------
class TestSegmentCandidates:
    def test_every_incident_segment_is_listed_with_its_direction(self) -> None:
        result = evaluate_segments(
            a_site(), square_network(), "J1", computed_at=utc_now()
        )
        assert [item.segment_id for item in result.candidates] == [
            "SEG-12",
            "SEG-14",
            "SEG-21",
            "SEG-41",
        ]
        assert result.candidate("SEG-21") is not None
        assert result.candidate("SEG-21").direction is SegmentDirection.INCOMING
        assert result.candidate("SEG-12").direction is SegmentDirection.OUTGOING

    def test_the_listing_is_ordered_by_id_and_not_by_bearing(self) -> None:
        """Ordering by bearing would put one segment at the top: that is selecting."""
        result = evaluate_segments(
            a_site(bearing_degrees=90.0), square_network(), "J1", computed_at=utc_now()
        )
        ids = [item.segment_id for item in result.candidates]
        assert ids == sorted(ids)
        offsets = [item.camera_bearing_offset_degrees for item in result.candidates]
        assert offsets != sorted(offsets)

    def test_nothing_is_selected(self) -> None:
        result = evaluate_segments(
            a_site(), square_network(), "J1", computed_at=utc_now()
        )
        assert result.selected_segment_id is None
        assert result.requires_human_confirmation is True

    def test_confirmation_cannot_be_waived(self) -> None:
        result = evaluate_segments(
            a_site(), square_network(), "J1", computed_at=utc_now()
        )
        with pytest.raises(Exception, match="requires_human_confirmation"):
            result.model_validate(
                result.model_dump() | {"requires_human_confirmation": False}
            )

    def test_an_incoming_segment_carries_its_approach(self) -> None:
        result = evaluate_segments(
            a_site(), square_network(), "J1", computed_at=utc_now()
        )
        incoming = result.candidate("SEG-21")
        assert incoming is not None
        assert incoming.approach_id == "J1-A-SEG-21"

    def test_an_outgoing_segment_has_no_approach(self) -> None:
        """An approach is traffic arriving; departing traffic does not have one."""
        result = evaluate_segments(
            a_site(), square_network(), "J1", computed_at=utc_now()
        )
        outgoing = result.candidate("SEG-12")
        assert outgoing is not None
        assert outgoing.approach_id is None

    def test_the_far_end_is_named(self) -> None:
        """Operators recognise "the arm toward J2", not "SEG-12"."""
        result = evaluate_segments(
            a_site(), square_network(), "J1", computed_at=utc_now()
        )
        assert result.candidate("SEG-12").other_junction_id == "J2"
        assert result.candidate("SEG-21").other_junction_id == "J2"
        assert result.candidate("SEG-14").other_junction_id == "J4"

    def test_the_candidate_set_is_derived_and_names_its_inputs(self) -> None:
        network = square_network()
        result = evaluate_segments(a_site(), network, "J1", computed_at=utc_now())
        assert result.provenance.source is ValueSource.DERIVED
        assert f"observation_site:{AREA}" in result.provenance.input_refs
        assert "junction:J1" in result.provenance.input_refs
        assert (
            f"network:{network.network_id}@{network.version}"
            in result.provenance.input_refs
        )


# ---------------------------------------------------------------------------
# Bearing as evidence only
# ---------------------------------------------------------------------------
class TestBearingEvidence:
    def test_the_angle_to_the_camera_is_reported(self) -> None:
        result = evaluate_segments(
            a_site(bearing_degrees=90.0), square_network(), "J1", computed_at=utc_now()
        )
        east = result.candidate("SEG-12")
        south = result.candidate("SEG-14")
        assert east.bearing_from_junction_degrees == pytest.approx(90.0, abs=0.1)
        assert east.camera_bearing_offset_degrees == pytest.approx(0.0, abs=0.1)
        assert south.camera_bearing_offset_degrees == pytest.approx(90.0, abs=0.1)
        assert result.bearing_evidence_available is True

    def test_bearing_cannot_separate_the_two_directions_of_one_road(self) -> None:
        """The core limitation, reported rather than resolved."""
        result = evaluate_segments(
            a_site(bearing_degrees=90.0), square_network(), "J1", computed_at=utc_now()
        )
        outgoing = result.candidate("SEG-12").camera_bearing_offset_degrees
        incoming = result.candidate("SEG-21").camera_bearing_offset_degrees
        assert outgoing == pytest.approx(incoming, abs=0.1)
        assert any(
            "SEG-12 and SEG-21" in warning for warning in result.warnings
        )
        assert any("same line" in warning for warning in result.warnings)

    def test_a_site_with_no_bearing_gets_no_geometric_evidence(self) -> None:
        result = evaluate_segments(
            a_site(bearing_degrees=None), square_network(), "J1", computed_at=utc_now()
        )
        assert result.site_bearing_degrees is None
        assert result.bearing_evidence_available is False
        assert all(
            item.camera_bearing_offset_degrees is None for item in result.candidates
        )
        assert any("no camera bearing" in warning for warning in result.warnings)

    def test_an_unpositioned_junction_yields_no_bearing(self) -> None:
        result = evaluate_segments(
            a_site(), square_network(positioned=False), "J1", computed_at=utc_now()
        )
        assert result.bearing_evidence_available is False
        assert all(
            item.bearing_from_junction_degrees is None for item in result.candidates
        )
        assert any("no surveyed position" in warning for warning in result.warnings)

    def test_the_surveyed_approach_bearing_is_reported_separately(self) -> None:
        """Survey data and computed geometry are kept apart so they can disagree."""
        result = evaluate_segments(
            a_site(), square_network(), "J1", computed_at=utc_now()
        )
        incoming = result.candidate("SEG-21")
        assert incoming is not None
        assert (
            incoming.approach_bearing_degrees is None
            or 0.0 <= incoming.approach_bearing_degrees < 360.0
        )
        assert incoming.bearing_from_junction_degrees is not None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
class TestValidation:
    def test_an_incident_segment_resolves_with_its_direction(self) -> None:
        segment, direction = validate_segment_incident("SEG-21", "J1", square_network())
        assert segment.segment_id == "SEG-21"
        assert direction is SegmentDirection.INCOMING

        segment, direction = validate_segment_incident("SEG-12", "J1", square_network())
        assert direction is SegmentDirection.OUTGOING

    def test_a_segment_that_does_not_exist_is_refused(self) -> None:
        with pytest.raises(ValidationFailedError, match="does not exist"):
            validate_segment_incident("SEG-NOPE", "J1", square_network())

    def test_a_segment_that_does_not_touch_the_junction_is_refused(self) -> None:
        """A real road elsewhere on the map is the dangerous case, not a typo."""
        with pytest.raises(ValidationFailedError, match="does not touch junction"):
            validate_segment_incident("SEG-23", "J1", square_network())

    def test_an_approach_must_belong_to_the_junction(self) -> None:
        with pytest.raises(ValidationFailedError, match="not an approach of junction"):
            validate_approach("J2-A-SEG-12", "SEG-21", "J1", square_network())

    def test_an_approach_must_match_its_segment(self) -> None:
        with pytest.raises(ValidationFailedError, match="carries traffic from segment"):
            validate_approach("J1-A-SEG-21", "SEG-41", "J1", square_network())

    def test_a_matching_approach_is_accepted(self) -> None:
        validate_approach("J1-A-SEG-21", "SEG-21", "J1", square_network())

    def test_a_stale_network_version_blocks_refinement(self) -> None:
        resurveyed = make_network(network_id="test-net", version="2.0")
        with pytest.raises(ValidationFailedError, match="confirm the junction again"):
            validate_network_version(a_junction_binding(network_version="1.0"), resurveyed)

    def test_a_matching_network_version_is_accepted(self) -> None:
        validate_network_version(a_junction_binding(network_version="1.0"), make_network())

    def test_a_binding_with_no_pinned_version_is_accepted(self) -> None:
        validate_network_version(a_junction_binding(network_version=None), make_network())


# ---------------------------------------------------------------------------
# Domain rules on the refined record
# ---------------------------------------------------------------------------
class TestRefinedRecord:
    def test_the_segment_fields_travel_together(self) -> None:
        base = a_junction_binding()
        with pytest.raises(Exception, match="incomplete segment refinement"):
            base.model_validate(base.model_dump() | {"segment_id": "SEG-21"})

    def test_a_segment_confirmation_without_a_segment_is_refused(self) -> None:
        base = a_junction_binding()
        with pytest.raises(Exception, match="incomplete segment refinement"):
            base.model_validate(
                base.model_dump()
                | {"segment_confirmation": a_segment_confirmation().model_dump()}
            )

    def test_an_approach_without_a_segment_is_refused(self) -> None:
        base = a_junction_binding()
        with pytest.raises(Exception, match="without the segment it belongs to"):
            base.model_validate(base.model_dump() | {"approach_id": "J1-A-SEG-21"})

    def test_an_approach_on_an_outgoing_segment_is_refused(self) -> None:
        """An approach is traffic arriving; an outgoing segment has none."""
        base = a_junction_binding()
        with pytest.raises(Exception, match="approach on an outgoing segment"):
            base.model_validate(
                base.model_dump()
                | {
                    "segment_id": "SEG-12",
                    "segment_direction": "outgoing",
                    "approach_id": "J1-A-SEG-21",
                    "segment_confirmation": a_segment_confirmation().model_dump(),
                }
            )

    def test_a_complete_refinement_validates(self) -> None:
        base = a_junction_binding()
        refined = base.model_validate(
            base.model_dump()
            | {
                "segment_id": "SEG-21",
                "segment_direction": "incoming",
                "approach_id": "J1-A-SEG-21",
                "segment_confirmation": a_segment_confirmation().model_dump(),
            }
        )
        assert refined.is_segment_bound is True
        assert refined.confirmation.confirmed_by == "surveyor.junction"
        assert refined.segment_confirmation is not None
        assert refined.segment_confirmation.confirmed_by == "surveyor.segment"

    def test_a_junction_only_binding_is_not_segment_bound(self) -> None:
        assert a_junction_binding().is_segment_bound is False

    def test_overriding_the_geometry_still_requires_a_reason(self) -> None:
        with pytest.raises(Exception, match="notes saying why"):
            a_segment_confirmation(method=BindingConfirmationMethod.CHOSE_DIFFERENT)

    def test_there_is_still_no_automatic_method(self) -> None:
        values = {item.value for item in BindingConfirmationMethod}
        assert "automatic" not in values
        assert "inferred" not in values
        assert "aligned_with_bearing" in values


# ---------------------------------------------------------------------------
# Mapping integration
# ---------------------------------------------------------------------------
class TestMappingIntegration:
    def test_a_segment_binding_renders_with_its_segment(self) -> None:
        base = a_junction_binding()
        refined = base.model_validate(
            base.model_dump()
            | {
                "segment_id": "SEG-21",
                "segment_direction": "incoming",
                "approach_id": "J1-A-SEG-21",
                "segment_confirmation": a_segment_confirmation().model_dump(),
            }
        )
        binding = binding_for(refined)
        assert binding.segment_id == "SEG-21"
        assert binding.junction_id == "J1"
        assert binding.approach_id == "J1-A-SEG-21"
        assert binding.confirmation is not None
        assert binding.confirmation.confirmed_by == "surveyor.segment"

    def test_a_junction_only_binding_still_names_no_segment(self) -> None:
        binding = binding_for(a_junction_binding())
        assert binding.segment_id is None
        assert binding.junction_id == "J1"
        assert binding.confirmation is not None
        assert binding.confirmation.confirmed_by == "surveyor.junction"

    def test_the_coverage_note_records_both_decisions(self) -> None:
        base = a_junction_binding()
        refined = base.model_validate(
            base.model_dump()
            | {
                "segment_id": "SEG-21",
                "segment_direction": "incoming",
                "segment_confirmation": a_segment_confirmation().model_dump(),
            }
        )
        note = refined.coverage_note
        assert "surveyor.junction" in note
        assert "surveyor.segment" in note
        assert "SEG-21" in note
        assert "incoming at J1" in note


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class TestRegistryRefinement:
    def test_refining_requires_an_existing_junction_binding(
        self, binding_settings: Settings
    ) -> None:
        """A refinement with nothing behind it would be an unrelated binding."""
        from app.core.errors import ResourceNotFoundError

        registry = ConfirmedBindingRegistry(binding_settings.confirmed_bindings_file)
        base = a_junction_binding()
        refined = base.model_validate(
            base.model_dump()
            | {
                "segment_id": "SEG-21",
                "segment_direction": "incoming",
                "segment_confirmation": a_segment_confirmation().model_dump(),
            }
        )
        with pytest.raises(ResourceNotFoundError, match="nothing to refine"):
            registry.refine(refined)

    def test_a_refinement_replaces_in_place(self, binding_settings: Settings) -> None:
        registry = ConfirmedBindingRegistry(binding_settings.confirmed_bindings_file)
        base = a_junction_binding()
        registry.confirm(base)
        refined = base.model_validate(
            base.model_dump()
            | {
                "segment_id": "SEG-21",
                "segment_direction": "incoming",
                "segment_confirmation": a_segment_confirmation().model_dump(),
            }
        )
        registry.refine(refined)

        assert len(registry) == 1
        stored = registry.get("test-net", AREA)
        assert stored.segment_id == "SEG-21"
        assert stored.confirmation.confirmed_by == "surveyor.junction"

    def test_segment_bound_lists_only_refined_bindings(
        self, binding_settings: Settings
    ) -> None:
        registry = ConfirmedBindingRegistry(binding_settings.confirmed_bindings_file)
        base = a_junction_binding()
        registry.confirm(base)
        assert registry.segment_bound("test-net") == ()

        registry.refine(
            base.model_validate(
                base.model_dump()
                | {
                    "segment_id": "SEG-21",
                    "segment_direction": "incoming",
                    "segment_confirmation": a_segment_confirmation().model_dump(),
                }
            )
        )
        assert len(registry.segment_bound("test-net")) == 1


# ---------------------------------------------------------------------------
# API: listing
# ---------------------------------------------------------------------------
class TestSegmentListingApi:
    def test_segments_need_a_confirmed_junction_first(
        self, client: TestClient
    ) -> None:
        client.post(
            "/api/sites",
            json={
                "observation_area_id": AREA,
                "latitude": 16.5051,
                "longitude": 80.6001,
            },
        )
        response = client.get(f"/api/sites/{AREA}/segments")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"
        assert "confirm the junction first" in response.json()["error"]["message"]

    def test_an_unknown_site_is_refused(self, client: TestClient) -> None:
        response = client.get("/api/sites/observation-area:ghost/segments")
        assert response.status_code == 404

    def test_the_incident_segments_are_listed(self, client: TestClient) -> None:
        set_up(client)
        body = client.get(f"/api/sites/{AREA}/segments").json()

        assert [item["segment_id"] for item in body["segments"]["candidates"]] == [
            "SEG-12",
            "SEG-14",
            "SEG-21",
            "SEG-41",
        ]
        assert body["segments"]["junction_id"] == "J1"
        assert body["confirmed"]["junction_id"] == "J1"
        assert body["segments"]["selected_segment_id"] is None
        assert "no segment has been chosen yet" in body["note"]

    def test_bearing_evidence_is_exposed(self, client: TestClient) -> None:
        set_up(client, bearing_degrees=90.0)
        body = client.get(f"/api/sites/{AREA}/segments").json()["segments"]

        assert body["site_bearing_degrees"] == 90.0
        assert body["bearing_evidence_available"] is True
        east = next(
            item for item in body["candidates"] if item["segment_id"] == "SEG-12"
        )
        assert east["camera_bearing_offset_degrees"] == pytest.approx(0.0, abs=0.1)

    def test_listing_segments_creates_nothing(self, client: TestClient) -> None:
        set_up(client)
        for _ in range(3):
            assert client.get(f"/api/sites/{AREA}/segments").status_code == 200
        listed = client.get("/api/sites/confirmed").json()
        assert listed["segment_bound_count"] == 0


# ---------------------------------------------------------------------------
# API: confirmation
# ---------------------------------------------------------------------------
class TestSegmentConfirmationApi:
    def test_a_segment_binding_needs_an_explicit_confirmation(
        self, client: TestClient
    ) -> None:
        set_up(client)
        response = client.post(
            f"/api/sites/{AREA}/binding/segment",
            json={"junction_id": "J1", "segment_id": "SEG-12"},
        )
        assert response.status_code == 422
        assert "confirm=true" in response.json()["error"]["message"]
        assert client.get("/api/sites/confirmed").json()["segment_bound_count"] == 0

    def test_a_segment_is_confirmed_with_its_direction(
        self, client: TestClient
    ) -> None:
        set_up(client)
        response = client.post(
            f"/api/sites/{AREA}/binding/segment",
            json={
                "junction_id": "J1",
                "segment_id": "SEG-21",
                "approach_id": "J1-A-SEG-21",
                "confirm": True,
                "confirmed_by": "surveyor.segment",
            },
        )
        assert response.status_code == 201
        confirmed = response.json()["confirmed"]

        assert confirmed["segment_id"] == "SEG-21"
        assert confirmed["segment_direction"] == "incoming"
        assert confirmed["approach_id"] == "J1-A-SEG-21"
        assert confirmed["segment_confirmation"]["confirmed_by"] == "surveyor.segment"
        assert confirmed["segment_confirmation"]["method"] == "aligned_with_bearing"

    def test_the_junction_confirmation_survives_the_refinement(
        self, client: TestClient
    ) -> None:
        """Requirement 7: extend the binding, never silently replace it."""
        set_up(client)
        before = client.get("/api/sites/confirmed").json()["bindings"][0]
        client.post(
            f"/api/sites/{AREA}/binding/segment",
            json={
                "junction_id": "J1",
                "segment_id": "SEG-12",
                "confirm": True,
                "confirmed_by": "surveyor.segment",
            },
        )
        after = client.get("/api/sites/confirmed").json()["bindings"][0]

        assert after["confirmation"] == before["confirmation"]
        assert after["junction_id"] == before["junction_id"]
        assert after["site_location"] == before["site_location"]
        assert after["provenance"] == before["provenance"]
        assert client.get("/api/sites/confirmed").json()["count"] == 1

    def test_the_direction_of_the_chosen_segment_is_preserved(
        self, client: TestClient
    ) -> None:
        """The two ids of one road are the difference between arriving and leaving."""
        set_up(client)
        outgoing = client.post(
            f"/api/sites/{AREA}/binding/segment",
            json={"junction_id": "J1", "segment_id": "SEG-12", "confirm": True},
        )
        assert outgoing.json()["confirmed"]["segment_direction"] == "outgoing"

        incoming = client.post(
            f"/api/sites/{AREA}/binding/segment",
            json={"junction_id": "J1", "segment_id": "SEG-21", "confirm": True},
        )
        assert incoming.json()["confirmed"]["segment_direction"] == "incoming"
        assert incoming.json()["confirmed"]["segment_id"] == "SEG-21"

    def test_a_segment_not_touching_the_junction_is_refused(
        self, client: TestClient
    ) -> None:
        set_up(client)
        response = client.post(
            f"/api/sites/{AREA}/binding/segment",
            json={"junction_id": "J1", "segment_id": "SEG-23", "confirm": True},
        )
        assert response.status_code == 422
        assert "does not touch junction" in response.json()["error"]["message"]
        assert client.get("/api/sites/confirmed").json()["segment_bound_count"] == 0

    def test_a_segment_that_does_not_exist_is_refused(
        self, client: TestClient
    ) -> None:
        set_up(client)
        response = client.post(
            f"/api/sites/{AREA}/binding/segment",
            json={"junction_id": "J1", "segment_id": "SEG-NOPE", "confirm": True},
        )
        assert response.status_code == 422
        assert "does not exist" in response.json()["error"]["message"]

    def test_naming_a_different_junction_is_refused(self, client: TestClient) -> None:
        set_up(client)
        response = client.post(
            f"/api/sites/{AREA}/binding/segment",
            json={"junction_id": "J2", "segment_id": "SEG-23", "confirm": True},
        )
        assert response.status_code == 422
        assert "is bound to junction 'J1'" in response.json()["error"]["message"]

    def test_a_mismatched_approach_is_refused(self, client: TestClient) -> None:
        set_up(client)
        response = client.post(
            f"/api/sites/{AREA}/binding/segment",
            json={
                "junction_id": "J1",
                "segment_id": "SEG-41",
                "approach_id": "J1-A-SEG-21",
                "confirm": True,
            },
        )
        assert response.status_code == 422
        assert "carries traffic from segment" in response.json()["error"]["message"]

    def test_choosing_against_the_bearing_requires_a_reason(
        self, client: TestClient
    ) -> None:
        """The camera faces east; SEG-14 runs due south."""
        set_up(client, bearing_degrees=90.0)
        refused = client.post(
            f"/api/sites/{AREA}/binding/segment",
            json={"junction_id": "J1", "segment_id": "SEG-14", "confirm": True},
        )
        assert refused.status_code == 422
        assert "does not support this choice" in refused.json()["error"]["message"]

        accepted = client.post(
            f"/api/sites/{AREA}/binding/segment",
            json={
                "junction_id": "J1",
                "segment_id": "SEG-14",
                "confirm": True,
                "notes": "the mast points east but the lens is angled down the south arm",
            },
        )
        assert accepted.status_code == 201
        assert (
            accepted.json()["confirmed"]["segment_confirmation"]["method"]
            == "chose_different"
        )

    def test_with_no_bearing_the_choice_is_recorded_as_surveyed(
        self, client: TestClient
    ) -> None:
        """No geometry to agree or disagree with, so the operator's word is it."""
        set_up(client, bearing_degrees=None)
        response = client.post(
            f"/api/sites/{AREA}/binding/segment",
            json={"junction_id": "J1", "segment_id": "SEG-14", "confirm": True},
        )
        assert response.status_code == 201
        assert (
            response.json()["confirmed"]["segment_confirmation"]["method"] == "surveyed"
        )

    def test_re_confirming_a_segment_replaces_and_answers_200(
        self, client: TestClient
    ) -> None:
        set_up(client)
        first = client.post(
            f"/api/sites/{AREA}/binding/segment",
            json={"junction_id": "J1", "segment_id": "SEG-12", "confirm": True},
        )
        second = client.post(
            f"/api/sites/{AREA}/binding/segment",
            json={"junction_id": "J1", "segment_id": "SEG-21", "confirm": True},
        )
        assert first.status_code == 201
        assert second.status_code == 200
        assert second.json()["replaced_existing"] is True
        assert client.get("/api/sites/confirmed").json()["count"] == 1

    def test_confirming_a_segment_needs_a_junction_binding(
        self, client: TestClient
    ) -> None:
        client.post(
            "/api/sites",
            json={
                "observation_area_id": AREA,
                "latitude": 16.5051,
                "longitude": 80.6001,
            },
        )
        response = client.post(
            f"/api/sites/{AREA}/binding/segment",
            json={"junction_id": "J1", "segment_id": "SEG-12", "confirm": True},
        )
        assert response.status_code == 404
        assert "confirm the junction first" in response.json()["error"]["message"]

    def test_confirming_without_a_network_is_refused(
        self, binding_settings: Settings
    ) -> None:
        with TestClient(
            create_app(binding_settings), raise_server_exceptions=False
        ) as c:
            c.post(
                "/api/sites",
                json={
                    "observation_area_id": AREA,
                    "latitude": 16.5051,
                    "longitude": 80.6001,
                },
            )
            response = c.post(
                f"/api/sites/{AREA}/binding/segment",
                json={"junction_id": "J1", "segment_id": "SEG-12", "confirm": True},
            )
            assert response.status_code == 404
            assert response.json()["error"]["code"] == "network_not_loaded"

    def test_the_segment_confirmation_is_observed_not_derived(
        self, client: TestClient
    ) -> None:
        """Bearing narrowed the choice; it did not become the fact."""
        set_up(client)
        body = client.post(
            f"/api/sites/{AREA}/binding/segment",
            json={"junction_id": "J1", "segment_id": "SEG-12", "confirm": True},
        ).json()
        assert body["confirmed"]["provenance"]["source"] == "observed"
        assert body["binding"]["confirmation"]["confirmed_by"] == "operator"

    def test_the_candidate_set_stays_derived(self, client: TestClient) -> None:
        set_up(client)
        body = client.get(f"/api/sites/{AREA}/segments").json()
        assert body["segments"]["provenance"]["source"] == "derived"
        assert body["segments"]["provenance"]["producer"] == "sites.incident_segments"

    def test_a_confirmed_segment_is_reported_in_the_listing(
        self, client: TestClient
    ) -> None:
        set_up(client)
        client.post(
            f"/api/sites/{AREA}/binding/segment",
            json={"junction_id": "J1", "segment_id": "SEG-12", "confirm": True},
        )
        body = client.get(f"/api/sites/{AREA}/segments").json()
        assert "already bound to segment SEG-12" in body["note"]
        assert body["confirmed"]["segment_id"] == "SEG-12"

    def test_the_routes_are_not_shadowed_by_one_another(
        self, client: TestClient
    ) -> None:
        """Area ids contain a colon; every sub-path must still resolve."""
        set_up(client)
        assert client.get(f"/api/sites/{AREA}").status_code == 200
        assert client.get(f"/api/sites/{AREA}/candidate").status_code == 200
        assert client.get(f"/api/sites/{AREA}/segments").status_code == 200
        assert (
            client.post(
                f"/api/sites/{AREA}/binding/segment",
                json={"junction_id": "J1", "segment_id": "SEG-12", "confirm": True},
            ).status_code
            == 201
        )


# ---------------------------------------------------------------------------
# Route cost inertness
# ---------------------------------------------------------------------------
class TestRouteCostIsUnchanged:
    def _ranking(self, result: dict[str, object]) -> list[tuple[list[str], object]]:
        routes = result["routes"]
        assert isinstance(routes, list)
        return [
            (
                [leg["segment_id"] for leg in route["candidate"]["legs"]],
                route["candidate"]["cost"],
            )
            for route in routes
        ]

    def test_a_segment_binding_does_not_change_the_ranking(
        self, client: TestClient
    ) -> None:
        """The property most easily broken by accident, pinned."""
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
                "bearing_degrees": 90.0,
            },
        )
        client.post(
            f"/api/sites/{AREA}/binding", json={"junction_id": "J1", "confirm": True}
        )
        confirmed = client.post(
            f"/api/sites/{AREA}/binding/segment",
            json={"junction_id": "J1", "segment_id": "SEG-12", "confirm": True},
        )
        assert confirmed.status_code == 201

        after = client.post("/api/routes/optimize", json=request)
        assert after.status_code == 200
        assert self._ranking(after.json()) == self._ranking(before.json())

    def test_the_listing_states_that_costs_are_unaffected(
        self, client: TestClient
    ) -> None:
        """Stated, not left to be inferred from an unchanged route."""
        set_up(client)
        client.post(
            f"/api/sites/{AREA}/binding/segment",
            json={"junction_id": "J1", "segment_id": "SEG-12", "confirm": True},
        )
        body = client.get("/api/sites/confirmed").json()

        assert body["affects_route_cost"] is False
        assert body["segment_bound_count"] == 1
        assert body["segment_bound_observation_area_ids"] == [AREA]
        assert "does not change any route cost" in body["route_cost_note"]

    def test_the_loaded_mapping_is_still_untouched(self, client: TestClient) -> None:
        set_up(client)
        client.post(
            f"/api/sites/{AREA}/binding/segment",
            json={"junction_id": "J1", "segment_id": "SEG-12", "confirm": True},
        )
        body = client.get("/api/routes/mapping").json()

        assert body["loaded"] is False
        assert body["bound_segment_ids"] == []
        assert body["bound_observation_area_ids"] == []
        assert body["confirmed_observation_area_ids"] == [AREA]


# ---------------------------------------------------------------------------
# Persistence and regression
# ---------------------------------------------------------------------------
class TestPersistence:
    def test_a_segment_binding_survives_a_restart(
        self, binding_settings: Settings
    ) -> None:
        app = create_app(binding_settings)
        with TestClient(app) as first:
            _load_network(app, square_network())
            set_up(first)
            confirmed = first.post(
                f"/api/sites/{AREA}/binding/segment",
                json={
                    "junction_id": "J1",
                    "segment_id": "SEG-21",
                    "approach_id": "J1-A-SEG-21",
                    "confirm": True,
                    "confirmed_by": "surveyor.segment",
                },
            )
            assert confirmed.status_code == 201

        rebuilt = create_app(binding_settings)
        with TestClient(rebuilt) as second:
            _load_network(rebuilt, square_network())
            body = second.get("/api/sites/confirmed").json()
            assert body["count"] == 1
            stored = body["bindings"][0]
            assert stored["segment_id"] == "SEG-21"
            assert stored["segment_direction"] == "incoming"
            assert stored["approach_id"] == "J1-A-SEG-21"
            assert stored["confirmation"]["confirmed_by"] == "surveyor.junction"
            assert stored["segment_confirmation"]["confirmed_by"] == "surveyor.segment"


class TestJunctionBindingRegression:
    def test_a_junction_binding_still_works_on_its_own(
        self, client: TestClient
    ) -> None:
        set_up(client)
        body = client.get("/api/sites/confirmed").json()
        assert body["count"] == 1
        assert body["bindings"][0]["segment_id"] is None
        assert body["segment_bound_count"] == 0

    def test_the_junction_candidate_endpoint_is_unchanged(
        self, client: TestClient
    ) -> None:
        set_up(client)
        body = client.get(f"/api/sites/{AREA}/candidate").json()
        assert body["candidate"]["nearest"]["junction_id"] == "J1"
        assert body["candidate"]["requires_human_confirmation"] is True
        assert body["confirmed"]["junction_id"] == "J1"

    def test_re_confirming_the_junction_still_works(self, client: TestClient) -> None:
        set_up(client)
        response = client.post(
            f"/api/sites/{AREA}/binding",
            json={
                "junction_id": "J2",
                "confirm": True,
                "notes": "remounted facing the east junction",
            },
        )
        assert response.status_code == 200
        assert response.json()["confirmed"]["junction_id"] == "J2"

    def test_withdrawing_a_binding_removes_the_segment_with_it(
        self, client: TestClient
    ) -> None:
        set_up(client)
        client.post(
            f"/api/sites/{AREA}/binding/segment",
            json={"junction_id": "J1", "segment_id": "SEG-12", "confirm": True},
        )
        assert client.request("DELETE", f"/api/sites/{AREA}/binding").status_code == 200
        assert client.get("/api/sites/confirmed").json()["count"] == 0

    def test_an_existing_mapping_file_still_loads(self, client: TestClient) -> None:
        """Guards the file path against the fields added to ObservationBinding."""
        assert (
            client.post(
                "/api/network/load", json={"relative_path": "example-grid.json"}
            ).status_code
            == 200
        )
        response = client.post(
            "/api/routes/mapping/load",
            json={"relative_path": "example-grid-mapping.json"},
        )
        assert response.status_code == 200
        assert response.json()["loaded"] is True
        assert all(
            binding["confirmation"] is None
            for binding in response.json()["mapping"]["bindings"]
        )

    def test_the_example_grid_is_still_the_offline_demo_network(
        self, client: TestClient
    ) -> None:
        loaded = client.post(
            "/api/network/load", json={"relative_path": "example-grid.json"}
        )
        assert loaded.status_code == 200
        assert loaded.json()["source"]["network_id"] == "example-grid"
