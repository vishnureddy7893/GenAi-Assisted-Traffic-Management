"""Where each camera's evidence lands on the road network.

One question: does the right camera's evidence reach the right road? Not whether
it should change a route cost — that is deliberately not connected, and a
regression test here pins that it has not been.

Most of these assertions are about what attribution refuses to do. It must never
place an area on the nearest road, never use a bearing to pick a segment, never
attribute through a stale binding, and never attribute the same area twice. An
area with no effective binding stays unmapped and says why.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from network_fixtures import square_network

from app.analytics.schemas import AnalyticsProvenance, ModelIdentity, ObservationMode
from app.core.config import Settings
from app.domain.enums import CongestionLevel, FreshnessStatus
from app.domain.network import GeoPoint, RoadNetwork
from app.domain.provenance import Provenance, ValueSource
from app.domain.site import (
    BindingConfirmation,
    BindingConfirmationMethod,
    ConfirmedSiteBinding,
    SegmentDirection,
)
from app.main import create_app
from app.routing.attribution import (
    AttributionStatus,
    EvidenceAttribution,
    attribute_evidence,
)
from app.routing.effective import (
    BindingResolution,
    BindingSource,
    compose_effective_mapping,
)
from app.routing.mapping import MappingSource, ObservationBinding, ObservationMapping
from app.sites.bindings import binding_for
from app.state.classifier import thresholds_from
from app.state.schemas import (
    CongestionAssessment,
    EvidenceBasis,
    IntersectionStateElement,
    IntersectionTrafficState,
    NetworkTrafficState,
    QualityGrade,
    StateAssemblyPolicy,
)

T0 = dt.datetime(2026, 9, 22, 9, 0, tzinfo=dt.UTC)
AREA_A = "observation-area:camera-A"
AREA_B = "observation-area:camera-B"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def provenance(source: ValueSource = ValueSource.OBSERVED) -> Provenance:
    return Provenance(
        source=source,
        producer="tests.attribution",
        producer_version="1.0",
        observed_at=T0,
        computed_at=T0,
    )


def make_state(
    area_id: str,
    *,
    level: CongestionLevel = CongestionLevel.HEAVY,
    vehicle_count: int = 46,
    settings: Settings | None = None,
) -> IntersectionTrafficState:
    """An interpreted state, built with the real domain objects."""
    resolved = settings or Settings(app_env="local", log_level="ERROR")
    return IntersectionTrafficState(
        state_id=f"state-{area_id}",
        observation_area_id=area_id,
        mode=ObservationMode.SNAPSHOT,
        as_of=T0,
        assessed_at=T0,
        congestion=CongestionAssessment(
            level=level,
            index=0.74,
            basis=EvidenceBasis.IMAGE_SPACE,
            thresholds=thresholds_from(resolved),
            method="occupancy_and_density",
            method_version="1.0",
        ),
        vehicle_count=vehicle_count,
        quality_grade=QualityGrade.GOOD,
        analytics_provenance=AnalyticsProvenance(
            media_id=area_id,
            source_id=f"image:{area_id}",
            frame_index=0,
            timestamp=T0,
            model=ModelIdentity(name="yolov8s", backend="ultralytics"),
            method="image_analytics",
            method_version="1.0",
            analytics_schema_version="1.0",
            computed_at=T0,
        ),
        schema_version="1.0",
    )


def snapshot(*states: IntersectionTrafficState) -> NetworkTrafficState:
    """A fused snapshot over the given states, all fresh."""
    return NetworkTrafficState(
        state_id="snapshot-1",
        as_of=T0,
        assembled_at=T0,
        policy=StateAssemblyPolicy(
            max_staleness_s=5.0, drop_after_s=30.0, fusion_strategy="newest_per_area"
        ),
        elements=tuple(
            IntersectionStateElement(
                state=item, freshness=FreshnessStatus.FRESH, evidence_age_s=0.0
            )
            for item in states
        ),
        schema_version="1.0",
    )


def file_mapping(*bindings: ObservationBinding) -> ObservationMapping:
    return ObservationMapping(
        mapping_id="survey-1",
        version="1.0",
        network_id="test-net",
        provenance=provenance(ValueSource.STATIC),
        bindings=bindings,
    )


def file_source(mapping: ObservationMapping) -> MappingSource:
    return MappingSource(
        relative_path="survey.json",
        sha256="a" * 64,
        size_bytes=100,
        loaded_at=T0,
        mapping_id=mapping.mapping_id,
        version=mapping.version,
        binding_count=len(mapping.bindings),
    )


def confirmed(
    *,
    area_id: str = AREA_A,
    junction_id: str = "J1",
    segment_id: str | None = "SEG-21",
    network_version: str | None = "1.0",
    network_id: str = "test-net",
    approach_id: str | None = None,
) -> ConfirmedSiteBinding:
    base: dict[str, Any] = {
        "observation_area_id": area_id,
        "junction_id": junction_id,
        "network_id": network_id,
        "network_version": network_version,
        "site_location": GeoPoint(latitude=16.505, longitude=80.600),
        "confirmation": BindingConfirmation(
            confirmed_by="surveyor.a",
            confirmed_at=T0,
            method=BindingConfirmationMethod.ACCEPTED_NEAREST,
        ),
        "provenance": provenance(),
    }
    if segment_id is not None:
        base |= {
            "segment_id": segment_id,
            "segment_direction": SegmentDirection.INCOMING,
            "approach_id": approach_id,
            "segment_confirmation": BindingConfirmation(
                confirmed_by="surveyor.b",
                confirmed_at=T0,
                method=BindingConfirmationMethod.ALIGNED_WITH_BEARING,
            ),
        }
    return ConfirmedSiteBinding.model_validate(base)


def attribute(
    *,
    states: tuple[IntersectionTrafficState, ...],
    mapping: ObservationMapping | None = None,
    confirmations: tuple[ConfirmedSiteBinding, ...] = (),
    network: RoadNetwork | None = None,
) -> Any:
    """Compose the effective mapping and attribute a snapshot through it."""
    resolved = network or square_network()
    effective = compose_effective_mapping(
        network=resolved,
        mapping=mapping,
        mapping_source=file_source(mapping) if mapping else None,
        confirmed=confirmations,
        binding_for=binding_for,
        composed_at=T0,
    )
    return attribute_evidence(snapshot(*states), effective)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
class TestResolution:
    def test_a_confirmed_binding_attributes_evidence_to_its_segment(self) -> None:
        report = attribute(
            states=(make_state(AREA_A),), confirmations=(confirmed(),)
        )
        entry = report.attribution_for(AREA_A)
        assert entry is not None
        assert entry.status is AttributionStatus.ATTRIBUTED
        assert entry.segment_id == "SEG-21"
        assert entry.junction_id == "J1"
        assert entry.source is BindingSource.CONFIRMED_BINDING
        assert entry.resolution is BindingResolution.CONFIRMED_ONLY
        assert report.attributed_segment_ids == ("SEG-21",)

    def test_a_file_only_mapping_attributes_evidence(self) -> None:
        report = attribute(
            states=(make_state(AREA_A),),
            mapping=file_mapping(
                ObservationBinding(observation_area_id=AREA_A, segment_id="SEG-14")
            ),
        )
        entry = report.attribution_for(AREA_A)
        assert entry is not None
        assert entry.status is AttributionStatus.ATTRIBUTED
        assert entry.segment_id == "SEG-14"
        assert entry.source is BindingSource.MAPPING_FILE
        assert entry.resolution is BindingResolution.FILE_ONLY
        # A file binding carries no confirmation provenance and says so.
        assert entry.provenance is None
        assert entry.site_location is None

    def test_agreeing_sources_attribute_with_the_agreement_recorded(self) -> None:
        report = attribute(
            states=(make_state(AREA_A),),
            mapping=file_mapping(
                ObservationBinding(observation_area_id=AREA_A, segment_id="SEG-21")
            ),
            confirmations=(confirmed(segment_id="SEG-21"),),
        )
        entry = report.attribution_for(AREA_A)
        assert entry is not None
        assert entry.segment_id == "SEG-21"
        assert entry.resolution is BindingResolution.SOURCES_AGREE
        assert entry.source is BindingSource.CONFIRMED_BINDING

    def test_a_confirmed_override_attributes_to_the_confirmed_segment(self) -> None:
        """The file said one road; a person said another. The person wins."""
        report = attribute(
            states=(make_state(AREA_A),),
            mapping=file_mapping(
                ObservationBinding(observation_area_id=AREA_A, segment_id="SEG-41")
            ),
            confirmations=(confirmed(segment_id="SEG-21"),),
        )
        entry = report.attribution_for(AREA_A)
        assert entry is not None
        assert entry.segment_id == "SEG-21"
        assert entry.resolution is BindingResolution.CONFIRMED_BINDING_PRECEDENCE
        assert "SEG-41" not in report.attributed_segment_ids

    def test_the_approach_is_carried_when_one_was_confirmed(self) -> None:
        report = attribute(
            states=(make_state(AREA_A),),
            confirmations=(confirmed(approach_id="J1-A-SEG-21"),),
        )
        entry = report.attribution_for(AREA_A)
        assert entry is not None
        assert entry.approach_id == "J1-A-SEG-21"
        assert entry.segment_direction == "incoming"

    def test_multiple_areas_resolve_independently(self) -> None:
        report = attribute(
            states=(make_state(AREA_A), make_state(AREA_B, level=CongestionLevel.LIGHT)),
            mapping=file_mapping(
                ObservationBinding(observation_area_id=AREA_B, segment_id="SEG-12")
            ),
            confirmations=(confirmed(area_id=AREA_A, segment_id="SEG-21"),),
        )
        by_area = {item.observation_area_id: item for item in report.attributions}
        assert by_area[AREA_A].segment_id == "SEG-21"
        assert by_area[AREA_A].source is BindingSource.CONFIRMED_BINDING
        assert by_area[AREA_B].segment_id == "SEG-12"
        assert by_area[AREA_B].source is BindingSource.MAPPING_FILE
        assert sorted(report.attributed_segment_ids) == ["SEG-12", "SEG-21"]

    def test_two_cameras_may_watch_one_segment(self) -> None:
        """Reconciled by whoever consumes the evidence, not collapsed here."""
        report = attribute(
            states=(make_state(AREA_A), make_state(AREA_B)),
            confirmations=(
                confirmed(area_id=AREA_A, segment_id="SEG-21"),
                confirmed(area_id=AREA_B, segment_id="SEG-21"),
            ),
        )
        assert len(report.attributions) == 2
        assert report.attributed_segment_ids == ("SEG-21",)

    def test_an_area_is_attributed_exactly_once(self) -> None:
        report = attribute(
            states=(make_state(AREA_A),),
            mapping=file_mapping(
                ObservationBinding(observation_area_id=AREA_A, segment_id="SEG-41")
            ),
            confirmations=(confirmed(segment_id="SEG-21"),),
        )
        ids = [item.observation_area_id for item in report.attributions]
        assert ids == sorted(set(ids))
        assert len(ids) == 1

    def test_duplicate_attributions_are_structurally_refused(self) -> None:
        report = attribute(states=(make_state(AREA_A),), confirmations=(confirmed(),))
        doubled = report.model_dump()
        doubled["attributions"] = [*doubled["attributions"], doubled["attributions"][0]]
        with pytest.raises(Exception, match="attributed more than once"):
            report.model_validate(doubled)


# ---------------------------------------------------------------------------
# Unmapped
# ---------------------------------------------------------------------------
class TestUnmapped:
    def test_an_unbound_area_stays_unmapped(self) -> None:
        report = attribute(states=(make_state(AREA_A),))
        entry = report.attribution_for(AREA_A)
        assert entry is not None
        assert entry.status is AttributionStatus.UNMAPPED
        assert entry.segment_id is None
        assert entry.junction_id is None
        assert report.unmapped_observation_area_ids == (AREA_A,)
        assert report.attributed_area_ids == ()

    def test_an_unbound_area_is_not_placed_on_the_nearest_road(self) -> None:
        """No geo-guessing. The square network has four positioned junctions."""
        report = attribute(states=(make_state(AREA_A),))
        entry = report.attribution_for(AREA_A)
        assert entry is not None
        assert "never by proximity" in entry.reason
        assert report.attributed_segment_ids == ()

    def test_a_junction_only_binding_attributes_to_no_segment(self) -> None:
        report = attribute(
            states=(make_state(AREA_A),),
            confirmations=(confirmed(segment_id=None),),
        )
        entry = report.attribution_for(AREA_A)
        assert entry is not None
        assert entry.status is AttributionStatus.JUNCTION_ONLY
        assert entry.segment_id is None
        assert entry.junction_id == "J1"
        assert "no direction of travel" in entry.reason
        assert report.junction_only_area_ids == (AREA_A,)
        assert AREA_A in report.unmapped_observation_area_ids

    def test_a_non_attributed_entry_cannot_name_a_segment(self) -> None:
        """Structural: a null attribution must not read as a binding."""
        report = attribute(states=(make_state(AREA_A),))
        entry = report.attributions[0]
        with pytest.raises(Exception, match="only an attributed area may name a road"):
            EvidenceAttribution.model_validate(
                entry.model_dump() | {"segment_id": "SEG-21"}
            )

    def test_an_attributed_entry_must_name_its_segment(self) -> None:
        report = attribute(states=(make_state(AREA_A),), confirmations=(confirmed(),))
        entry = report.attributions[0]
        with pytest.raises(Exception, match="must name the segment"):
            EvidenceAttribution.model_validate(
                entry.model_dump() | {"segment_id": None}
            )

    def test_an_area_with_no_interpreted_state_is_simply_absent(self) -> None:
        """Attribution describes evidence, so an area with none has no entry."""
        report = attribute(states=(), confirmations=(confirmed(),))
        assert report.attributions == ()
        assert report.attribution_for(AREA_A) is None
        assert "no evidence to attribute" in report.note


# ---------------------------------------------------------------------------
# Stale bindings
# ---------------------------------------------------------------------------
class TestStaleBindings:
    def test_a_stale_binding_does_not_attribute_evidence(self) -> None:
        report = attribute(
            states=(make_state(AREA_A),),
            confirmations=(confirmed(network_version="0.9"),),
        )
        entry = report.attribution_for(AREA_A)
        assert entry is not None
        assert entry.status is AttributionStatus.EXCLUDED
        assert entry.segment_id is None
        assert entry.exclusion_reason is not None
        assert entry.exclusion_reason.value == "stale_network_version"
        assert "Re-confirm" in entry.reason
        assert report.excluded_area_ids == (AREA_A,)
        assert AREA_A in report.unmapped_observation_area_ids

    def test_a_binding_for_another_network_does_not_attribute(self) -> None:
        report = attribute(
            states=(make_state(AREA_A),),
            confirmations=(confirmed(network_id="other-net"),),
        )
        entry = report.attribution_for(AREA_A)
        assert entry is not None
        assert entry.status is AttributionStatus.EXCLUDED
        assert entry.exclusion_reason is not None
        assert entry.exclusion_reason.value == "different_network"

    def test_a_stale_binding_does_not_supersede_a_file_binding(self) -> None:
        """The file binding survives and still attributes the evidence."""
        report = attribute(
            states=(make_state(AREA_A),),
            mapping=file_mapping(
                ObservationBinding(observation_area_id=AREA_A, segment_id="SEG-14")
            ),
            confirmations=(confirmed(segment_id="SEG-21", network_version="0.9"),),
        )
        entry = report.attribution_for(AREA_A)
        assert entry is not None
        assert entry.status is AttributionStatus.ATTRIBUTED
        assert entry.segment_id == "SEG-14"
        assert entry.source is BindingSource.MAPPING_FILE

    def test_a_stale_binding_is_never_migrated(self) -> None:
        report = attribute(
            states=(make_state(AREA_A),),
            confirmations=(confirmed(segment_id="SEG-21", network_version="0.9"),),
        )
        assert report.attributed_segment_ids == ()
        assert "SEG-21" not in str(report.attributed_area_ids)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
class TestProvenance:
    def test_the_confirmation_provenance_survives_attribution(self) -> None:
        report = attribute(states=(make_state(AREA_A),), confirmations=(confirmed(),))
        entry = report.attribution_for(AREA_A)
        assert entry is not None
        assert entry.provenance is not None
        assert entry.provenance.source is ValueSource.OBSERVED
        assert entry.provenance.producer == "tests.attribution"

    def test_the_site_coordinate_survives_attribution(self) -> None:
        """Keeps the chain camera -> area -> segment traceable to a coordinate."""
        report = attribute(states=(make_state(AREA_A),), confirmations=(confirmed(),))
        entry = report.attribution_for(AREA_A)
        assert entry is not None
        assert entry.site_location is not None
        assert entry.site_location.latitude == 16.505

    def test_the_evidence_itself_is_carried_unchanged(self) -> None:
        report = attribute(
            states=(make_state(AREA_A, level=CongestionLevel.SEVERE, vehicle_count=91),),
            confirmations=(confirmed(),),
        )
        entry = report.attribution_for(AREA_A)
        assert entry is not None
        assert entry.congestion_level is CongestionLevel.SEVERE
        assert entry.vehicle_count == 91
        assert entry.freshness is FreshnessStatus.FRESH

    def test_sources_are_not_flattened_into_a_generic_mapped_value(self) -> None:
        """Four resolutions reach the response distinctly."""
        seen = set()
        for mapping, confirmations in (
            (
                file_mapping(
                    ObservationBinding(observation_area_id=AREA_A, segment_id="SEG-14")
                ),
                (),
            ),
            (None, (confirmed(),)),
            (
                file_mapping(
                    ObservationBinding(observation_area_id=AREA_A, segment_id="SEG-21")
                ),
                (confirmed(segment_id="SEG-21"),),
            ),
            (
                file_mapping(
                    ObservationBinding(observation_area_id=AREA_A, segment_id="SEG-41")
                ),
                (confirmed(segment_id="SEG-21"),),
            ),
        ):
            report = attribute(
                states=(make_state(AREA_A),),
                mapping=mapping,
                confirmations=confirmations,
            )
            entry = report.attribution_for(AREA_A)
            assert entry is not None
            assert entry.resolution is not None
            seen.add(entry.resolution)
        assert seen == {
            BindingResolution.FILE_ONLY,
            BindingResolution.CONFIRMED_ONLY,
            BindingResolution.SOURCES_AGREE,
            BindingResolution.CONFIRMED_BINDING_PRECEDENCE,
        }

    def test_the_effective_mapping_identity_is_exposed(self) -> None:
        report = attribute(states=(make_state(AREA_A),), confirmations=(confirmed(),))
        assert len(report.effective_mapping_hash) == 64
        assert report.effective_mapping_version == report.effective_mapping_hash[:16]
        assert report.network_id == "test-net"

    def test_attribution_never_claims_to_drive_costing(self) -> None:
        report = attribute(states=(make_state(AREA_A),), confirmations=(confirmed(),))
        assert report.used_by_route_costing is False
        with pytest.raises(Exception, match="used_by_route_costing cannot be true"):
            report.model_validate(report.model_dump() | {"used_by_route_costing": True})


# ---------------------------------------------------------------------------
# No inference
# ---------------------------------------------------------------------------
class TestNoInference:
    def test_a_bearing_never_selects_a_segment(self) -> None:
        """A site bearing is evidence for a human choice, never an attribution."""
        report = attribute(states=(make_state(AREA_A),))
        assert report.attributed_segment_ids == ()
        assert report.attributions[0].status is AttributionStatus.UNMAPPED

    def test_attribution_reads_nothing_from_the_analytics_geometry(self) -> None:
        """Image coordinates cannot reach a segment id: there is no such path."""
        source = Path("backend/app/routing/attribution.py")
        if not source.exists():  # pragma: no cover - path differs under some runners
            source = Path(__file__).resolve().parents[1] / "app/routing/attribution.py"
        text = source.read_text(encoding="utf-8")
        for forbidden in (
            "haversine",
            "bearing_degrees_between",
            "spatial_distribution",
            "object_positions",
            "evaluate_site",
            "evaluate_segments",
        ):
            assert forbidden not in text, forbidden

    def test_an_unpositioned_network_changes_nothing(self) -> None:
        """Attribution does not depend on geometry at all."""
        positioned = attribute(
            states=(make_state(AREA_A),),
            confirmations=(confirmed(),),
            network=square_network(),
        )
        unpositioned = attribute(
            states=(make_state(AREA_A),),
            confirmations=(confirmed(),),
            network=square_network(positioned=False),
        )
        assert positioned.attributions[0].segment_id == (
            unpositioned.attributions[0].segment_id
        )


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@pytest.fixture
def attribution_settings(tmp_path: Path) -> Settings:
    return Settings(
        app_env="local",
        log_level="ERROR",
        media_dir=tmp_path / "media",
        sites_file=tmp_path / "sites.json",
        confirmed_bindings_file=tmp_path / "bindings.json",
        network_file="example-grid.json",
    )


@pytest.fixture
def client(attribution_settings: Settings) -> Iterator[TestClient]:
    app = create_app(attribution_settings)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def seed_state(client: TestClient, *areas: str, settings: Settings) -> None:
    """Put interpreted states into the cache the state endpoints populate."""
    client.app.state.traffic_states = {  # type: ignore[attr-defined]
        f"media-{index}": make_state(area, settings=settings)
        for index, area in enumerate(areas)
    }


def confirm_segment(client: TestClient, area: str, junction: str, segment: str) -> None:
    client.post(
        "/api/sites",
        json={"observation_area_id": area, "latitude": 16.505, "longitude": 80.600},
    )
    client.post(f"/api/sites/{area}/binding", json={"junction_id": junction, "confirm": True})
    response = client.post(
        f"/api/sites/{area}/binding/segment",
        json={"junction_id": junction, "segment_id": segment, "confirm": True},
    )
    assert response.status_code in (200, 201), response.text


class TestNetworkStateApi:
    def test_the_existing_response_shape_is_preserved(
        self, client: TestClient, attribution_settings: Settings
    ) -> None:
        """A client written before attribution reads exactly what it always read."""
        seed_state(client, AREA_A, settings=attribution_settings)
        body = client.get("/api/state/network").json()

        for key in (
            "state_id",
            "as_of",
            "assembled_at",
            "policy",
            "elements",
            "schema_version",
        ):
            assert key in body
        assert len(body["elements"]) == 1
        assert body["elements"][0]["state"]["observation_area_id"] == AREA_A

    def test_the_interpreted_values_are_unchanged(
        self, client: TestClient, attribution_settings: Settings
    ) -> None:
        """Attribution reports; it does not reinterpret."""
        seed_state(client, AREA_A, settings=attribution_settings)
        confirm_segment(client, AREA_A, "J1", "SEG-21")
        state = client.get("/api/state/network").json()["elements"][0]["state"]

        assert state["congestion"]["level"] == "heavy"
        assert state["vehicle_count"] == 46
        assert state["quality_grade"] == "good"

    def test_a_confirmed_area_is_attributed(
        self, client: TestClient, attribution_settings: Settings
    ) -> None:
        seed_state(client, AREA_A, settings=attribution_settings)
        confirm_segment(client, AREA_A, "J1", "SEG-21")

        attribution = client.get("/api/state/network").json()["attribution"]
        entry = attribution["attributions"][0]
        assert entry["observation_area_id"] == AREA_A
        assert entry["segment_id"] == "SEG-21"
        assert entry["junction_id"] == "J1"
        assert entry["status"] == "attributed"
        assert entry["source"] == "confirmed_binding"
        assert entry["congestion_level"] == "heavy"
        assert attribution["attributed_segment_ids"] == ["SEG-21"]

    def test_an_unbound_area_is_reported_unmapped(
        self, client: TestClient, attribution_settings: Settings
    ) -> None:
        seed_state(client, AREA_A, AREA_B, settings=attribution_settings)
        confirm_segment(client, AREA_A, "J1", "SEG-21")

        attribution = client.get("/api/state/network").json()["attribution"]
        assert attribution["attributed_area_ids"] == [AREA_A]
        assert attribution["unmapped_observation_area_ids"] == [AREA_B]
        entry = next(
            item
            for item in attribution["attributions"]
            if item["observation_area_id"] == AREA_B
        )
        assert entry["segment_id"] is None
        assert entry["status"] == "unmapped"

    def test_the_effective_mapping_hash_is_exposed(
        self, client: TestClient, attribution_settings: Settings
    ) -> None:
        seed_state(client, AREA_A, settings=attribution_settings)
        confirm_segment(client, AREA_A, "J1", "SEG-21")

        body = client.get("/api/state/network").json()
        composed = client.get("/api/sites/effective-mapping").json()["effective"]
        assert (
            body["attribution"]["effective_mapping_hash"]
            == composed["effective_mapping_hash"]
        )

    def test_no_network_loaded_gives_state_without_attribution(
        self, tmp_path: Path
    ) -> None:
        """Traffic state does not need a map; attribution does, and says so."""
        settings = Settings(
            app_env="local",
            log_level="ERROR",
            media_dir=tmp_path / "media",
            sites_file=tmp_path / "sites.json",
            confirmed_bindings_file=tmp_path / "bindings.json",
        )
        with TestClient(create_app(settings), raise_server_exceptions=False) as c:
            seed_state(c, AREA_A, settings=settings)
            body = c.get("/api/state/network").json()

        assert len(body["elements"]) == 1
        assert body["attribution"] is None
        assert "No road network is loaded" in body["attribution_unavailable_reason"]

    def test_an_unreadable_binding_registry_does_not_take_down_traffic_state(
        self, tmp_path: Path
    ) -> None:
        """A corrupt mapping file must not hide working evidence."""
        bindings_file = tmp_path / "bindings.json"
        bindings_file.write_text("{ not json", encoding="utf-8")
        settings = Settings(
            app_env="local",
            log_level="ERROR",
            media_dir=tmp_path / "media",
            sites_file=tmp_path / "sites.json",
            confirmed_bindings_file=bindings_file,
            network_file="example-grid.json",
        )
        with TestClient(create_app(settings), raise_server_exceptions=False) as c:
            seed_state(c, AREA_A, settings=settings)
            response = c.get("/api/state/network")

        assert response.status_code == 200
        body = response.json()
        assert len(body["elements"]) == 1
        assert body["attribution"] is None
        assert "could not be composed" in body["attribution_unavailable_reason"]

    def test_an_empty_state_still_answers(self, client: TestClient) -> None:
        body = client.get("/api/state/network").json()
        assert body["elements"] == []
        assert body["attribution"]["attributions"] == []

    def test_the_response_states_costing_is_unaffected(
        self, client: TestClient, attribution_settings: Settings
    ) -> None:
        seed_state(client, AREA_A, settings=attribution_settings)
        confirm_segment(client, AREA_A, "J1", "SEG-21")
        attribution = client.get("/api/state/network").json()["attribution"]

        assert attribution["used_by_route_costing"] is False
        assert "route costs are unchanged" in attribution["note"]

    def test_asking_for_attribution_writes_nothing(
        self, client: TestClient, attribution_settings: Settings
    ) -> None:
        seed_state(client, AREA_A, settings=attribution_settings)
        confirm_segment(client, AREA_A, "J1", "SEG-21")
        before = client.get("/api/sites/confirmed").json()
        for _ in range(3):
            assert client.get("/api/state/network").status_code == 200
        assert client.get("/api/sites/confirmed").json() == before


# ---------------------------------------------------------------------------
# Routing regression
# ---------------------------------------------------------------------------
class TestRoutingIsUnchanged:
    def _ranking(self, result: dict[str, Any]) -> list[tuple[list[str], Any]]:
        routes = result["routes"]
        assert isinstance(routes, list)
        return [
            (
                [leg["segment_id"] for leg in route["candidate"]["legs"]],
                route["candidate"]["cost"],
            )
            for route in routes
        ]

    def test_attribution_does_not_change_route_costs_or_ranking(
        self, client: TestClient, attribution_settings: Settings
    ) -> None:
        """The property this whole task is built to preserve."""
        request = {
            "origin_junction_id": "J1",
            "destination_junction_id": "J3",
            "mode": "normal",
        }
        seed_state(client, AREA_A, settings=attribution_settings)
        before = client.post("/api/routes/optimize", json=request)
        assert before.status_code == 200

        confirm_segment(client, AREA_A, "J1", "SEG-21")
        attribution = client.get("/api/state/network").json()["attribution"]
        assert attribution["attributed_segment_ids"] == ["SEG-21"]

        after = client.post("/api/routes/optimize", json=request)
        assert after.status_code == 200
        assert self._ranking(after.json()) == self._ranking(before.json())

    def test_the_optimizer_still_reports_the_area_as_unmapped(
        self, client: TestClient, attribution_settings: Settings
    ) -> None:
        """Attribution knows the segment; the optimizer deliberately does not."""
        seed_state(client, AREA_A, settings=attribution_settings)
        confirm_segment(client, AREA_A, "J1", "SEG-21")

        attribution = client.get("/api/state/network").json()["attribution"]
        assert attribution["attributed_area_ids"] == [AREA_A]

        result = client.post(
            "/api/routes/optimize",
            json={
                "origin_junction_id": "J1",
                "destination_junction_id": "J3",
                "mode": "normal",
            },
        ).json()
        assert AREA_A in result["unmapped_observation_area_ids"]
        assert result["mapping_id"] is None
        assert "effective_mapping_hash" not in result
