"""Composing the mapping file and the confirmed bindings into one mapping.

Two sources can bind the same camera to different roads, and this is where that
is resolved. The tests fall into three groups.

**Precedence** — a confirmation supersedes a file binding, an unopposed source
stands, agreement is recorded as agreement, and no area ever ends up bound
twice.

**Identity** — the content hash is deterministic and order-independent, and it
excludes wall-clock time, because an identity that changed on every request
could not answer the replay question it exists for.

**Restraint** — composing changes nothing. Route costs and rankings must be
identical before and after, the composition must not claim to be in force, and a
stale confirmation must not quietly become an effective binding.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from network_fixtures import square_network

from app.core.config import Settings
from app.domain.network import GeoPoint, RoadNetwork
from app.domain.provenance import Provenance, ValueSource
from app.domain.site import (
    BindingConfirmation,
    BindingConfirmationMethod,
    ConfirmedSiteBinding,
    SegmentDirection,
)
from app.main import create_app
from app.routing.effective import (
    BindingResolution,
    BindingSource,
    EffectiveObservationMapping,
    ExclusionReason,
    compose_effective_mapping,
    confirmed_bindings_fingerprint,
)
from app.routing.mapping import MappingSource, ObservationBinding, ObservationMapping
from app.sites.bindings import binding_for

T0 = dt.datetime(2026, 9, 22, 9, 0, tzinfo=dt.UTC)
AREA_A = "observation-area:camera-A"
AREA_B = "observation-area:camera-B"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def provenance(source: ValueSource = ValueSource.OBSERVED) -> Provenance:
    return Provenance(
        source=source,
        producer="tests.effective",
        producer_version="1.0",
        observed_at=T0,
        computed_at=T0,
    )


def file_mapping(
    *bindings: ObservationBinding,
    mapping_id: str = "survey-1",
    version: str = "1.0",
    network_id: str = "test-net",
) -> ObservationMapping:
    """A loaded mapping file."""
    return ObservationMapping(
        mapping_id=mapping_id,
        version=version,
        network_id=network_id,
        provenance=provenance(ValueSource.STATIC),
        bindings=bindings,
    )


def file_source(mapping: ObservationMapping, digest: str = "a" * 64) -> MappingSource:
    return MappingSource(
        relative_path="survey.json",
        sha256=digest,
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
    direction: SegmentDirection | None = SegmentDirection.INCOMING,
    network_id: str = "test-net",
    network_version: str | None = "1.0",
    confirmed_by: str = "surveyor.a",
) -> ConfirmedSiteBinding:
    """A confirmed site binding, optionally refined to a segment."""
    base: dict[str, Any] = {
        "observation_area_id": area_id,
        "junction_id": junction_id,
        "network_id": network_id,
        "network_version": network_version,
        "site_location": GeoPoint(latitude=16.505, longitude=80.600),
        "confirmation": BindingConfirmation(
            confirmed_by=confirmed_by,
            confirmed_at=T0,
            method=BindingConfirmationMethod.ACCEPTED_NEAREST,
            candidate_distance_m=15.4,
        ),
        "provenance": provenance(),
    }
    if segment_id is not None:
        base |= {
            "segment_id": segment_id,
            "segment_direction": direction,
            "segment_confirmation": BindingConfirmation(
                confirmed_by=confirmed_by,
                confirmed_at=T0,
                method=BindingConfirmationMethod.ALIGNED_WITH_BEARING,
            ),
        }
    return ConfirmedSiteBinding.model_validate(base)


def compose(
    *,
    network: RoadNetwork | None = None,
    mapping: ObservationMapping | None = None,
    source: MappingSource | None = None,
    confirmations: tuple[ConfirmedSiteBinding, ...] = (),
    at: dt.datetime = T0,
) -> EffectiveObservationMapping:
    """Compose with the real composer and the real binding renderer."""
    resolved = network or square_network()
    return compose_effective_mapping(
        network=resolved,
        mapping=mapping,
        mapping_source=source if source is not None else (
            file_source(mapping) if mapping else None
        ),
        confirmed=confirmations,
        binding_for=binding_for,
        composed_at=at,
    )


# ---------------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------------
class TestPrecedence:
    def test_a_file_binding_alone_stands(self) -> None:
        result = compose(
            mapping=file_mapping(
                ObservationBinding(observation_area_id=AREA_A, segment_id="SEG-14")
            )
        )
        assert len(result.bindings) == 1
        entry = result.bindings[0]
        assert entry.binding.segment_id == "SEG-14"
        assert entry.source is BindingSource.MAPPING_FILE
        assert entry.resolution is BindingResolution.FILE_ONLY
        assert entry.confirmed is None
        assert result.overrides == ()

    def test_a_confirmed_binding_alone_stands(self) -> None:
        result = compose(confirmations=(confirmed(segment_id="SEG-21"),))
        entry = result.bindings[0]
        assert entry.binding.segment_id == "SEG-21"
        assert entry.source is BindingSource.CONFIRMED_BINDING
        assert entry.resolution is BindingResolution.CONFIRMED_ONLY
        assert entry.confirmed is not None
        assert result.overrides == ()

    def test_a_confirmation_supersedes_the_file(self) -> None:
        """The worked example from the specification."""
        result = compose(
            mapping=file_mapping(
                ObservationBinding(observation_area_id=AREA_A, segment_id="SEG-41")
            ),
            confirmations=(confirmed(segment_id="SEG-21"),),
        )
        entry = result.bindings[0]
        assert entry.binding.segment_id == "SEG-21"
        assert entry.source is BindingSource.CONFIRMED_BINDING
        assert entry.resolution is BindingResolution.CONFIRMED_BINDING_PRECEDENCE
        assert entry.superseded_segment_id == "SEG-41"

        assert len(result.overrides) == 1
        override = result.overrides[0]
        assert override.file_binding == "SEG-41"
        assert override.confirmed_binding == "SEG-21"
        assert override.effective_binding == "SEG-21"
        assert override.resolution is BindingResolution.CONFIRMED_BINDING_PRECEDENCE
        assert "supersedes the mapping file" in override.note

    def test_agreeing_sources_are_recorded_as_agreeing(self) -> None:
        """Corroboration is the only cross-check this system gets; keep it."""
        result = compose(
            mapping=file_mapping(
                ObservationBinding(observation_area_id=AREA_A, segment_id="SEG-21")
            ),
            confirmations=(confirmed(segment_id="SEG-21"),),
        )
        entry = result.bindings[0]
        assert entry.resolution is BindingResolution.SOURCES_AGREE
        assert entry.source is BindingSource.CONFIRMED_BINDING
        assert entry.confirmed is not None
        assert entry.superseded_segment_id is None
        assert result.overrides[0].resolution is BindingResolution.SOURCES_AGREE
        assert "same target" in result.overrides[0].note

    def test_agreement_ignores_a_junction_the_file_omitted(self) -> None:
        """Extra detail is not a contradiction.

        A mapping file binds a camera to a segment and usually records no
        junction. A confirmation records both. Comparing field by field would
        report a disagreement every time a confirmation merely filled that in.
        """
        result = compose(
            mapping=file_mapping(
                ObservationBinding(observation_area_id=AREA_A, segment_id="SEG-21")
            ),
            confirmations=(confirmed(segment_id="SEG-21", junction_id="J1"),),
        )
        assert result.bindings[0].resolution is BindingResolution.SOURCES_AGREE

    def test_a_junction_only_confirmation_never_counts_as_agreement(self) -> None:
        """A vaguer claim must not pass as corroboration of a specific one."""
        result = compose(
            mapping=file_mapping(
                ObservationBinding(observation_area_id=AREA_A, segment_id="SEG-21")
            ),
            confirmations=(confirmed(segment_id=None, direction=None),),
        )
        entry = result.bindings[0]
        assert entry.resolution is BindingResolution.CONFIRMED_BINDING_PRECEDENCE
        assert entry.binding.segment_id is None
        assert entry.binding.junction_id == "J1"
        assert "segment attribution is withdrawn" in result.overrides[0].note
        assert any("restore road-level attribution" in w for w in result.warnings)

    def test_multiple_areas_resolve_independently(self) -> None:
        result = compose(
            mapping=file_mapping(
                ObservationBinding(observation_area_id=AREA_A, segment_id="SEG-41"),
                ObservationBinding(observation_area_id=AREA_B, segment_id="SEG-12"),
            ),
            confirmations=(confirmed(area_id=AREA_A, segment_id="SEG-21"),),
        )
        by_area = {item.observation_area_id: item for item in result.bindings}
        assert by_area[AREA_A].resolution is BindingResolution.CONFIRMED_BINDING_PRECEDENCE
        assert by_area[AREA_A].binding.segment_id == "SEG-21"
        assert by_area[AREA_B].resolution is BindingResolution.FILE_ONLY
        assert by_area[AREA_B].binding.segment_id == "SEG-12"
        assert result.override_count == 1

    def test_no_area_is_bound_twice(self) -> None:
        """Enforced by the existing ObservationMapping validator, not a new one."""
        result = compose(
            mapping=file_mapping(
                ObservationBinding(observation_area_id=AREA_A, segment_id="SEG-41"),
                ObservationBinding(observation_area_id=AREA_B, segment_id="SEG-12"),
            ),
            confirmations=(
                confirmed(area_id=AREA_A, segment_id="SEG-21"),
                confirmed(area_id=AREA_B, segment_id="SEG-41"),
            ),
        )
        area_ids = [item.observation_area_id for item in result.bindings]
        assert sorted(area_ids) == sorted(set(area_ids))
        mapped = [item.observation_area_id for item in result.mapping.bindings]
        assert sorted(mapped) == sorted(set(mapped))
        assert len(result.mapping.area_ids) == 2

    def test_the_composed_mapping_is_an_ordinary_observation_mapping(self) -> None:
        result = compose(confirmations=(confirmed(segment_id="SEG-21"),))
        assert isinstance(result.mapping, ObservationMapping)
        assert result.mapping.network_id == "test-net"
        assert result.mapping.segment_for(AREA_A) == "SEG-21"
        assert result.mapping.provenance.source is ValueSource.DERIVED

    def test_an_empty_composition_is_a_working_state(self) -> None:
        result = compose()
        assert result.bindings == ()
        assert result.mapping.bindings == ()
        assert "No observation area is bound" in result.note


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
class TestProvenance:
    def test_the_whole_confirmation_record_is_preserved(self) -> None:
        """Nothing is dropped to fit the older mapping shape."""
        result = compose(confirmations=(confirmed(segment_id="SEG-21"),))
        record = result.bindings[0].confirmed
        assert record is not None
        assert record.junction_id == "J1"
        assert record.segment_id == "SEG-21"
        assert record.segment_direction is SegmentDirection.INCOMING
        assert record.network_id == "test-net"
        assert record.network_version == "1.0"
        assert record.site_location.latitude == 16.505
        assert record.confirmation.confirmed_by == "surveyor.a"
        assert record.segment_confirmation is not None
        assert record.provenance.source is ValueSource.OBSERVED

    def test_the_rendered_binding_carries_the_confirmation(self) -> None:
        result = compose(confirmations=(confirmed(segment_id="SEG-21"),))
        binding = result.bindings[0].binding
        assert binding.confirmation is not None
        assert binding.confirmation.confirmed_by == "surveyor.a"
        assert binding.coverage_note is not None

    def test_a_file_binding_keeps_its_own_provenance(self) -> None:
        """A file binding is not re-labelled as confirmed."""
        result = compose(
            mapping=file_mapping(
                ObservationBinding(
                    observation_area_id=AREA_A,
                    segment_id="SEG-14",
                    coverage_note="surveyed from the mast",
                )
            )
        )
        binding = result.bindings[0].binding
        assert binding.confirmation is None
        assert binding.coverage_note == "surveyed from the mast"

    def test_the_composition_is_derived_and_names_both_inputs(self) -> None:
        mapping = file_mapping(
            ObservationBinding(observation_area_id=AREA_B, segment_id="SEG-12")
        )
        result = compose(mapping=mapping, confirmations=(confirmed(),))
        assert result.provenance.source is ValueSource.DERIVED
        refs = result.provenance.input_refs
        assert any(ref.startswith("mapping:survey-1@1.0") for ref in refs)
        assert any(ref.startswith("confirmed_bindings:") for ref in refs)
        assert any(ref.startswith("network:test-net@") for ref in refs)

    def test_both_source_identities_are_reported(self) -> None:
        mapping = file_mapping(
            ObservationBinding(observation_area_id=AREA_A, segment_id="SEG-14")
        )
        result = compose(
            mapping=mapping,
            source=file_source(mapping, digest="b" * 64),
            confirmations=(confirmed(area_id=AREA_B),),
        )
        assert result.mapping_id == "survey-1"
        assert result.mapping_version == "1.0"
        assert result.mapping_sha256 == "b" * 64
        assert result.confirmed_binding_count == 1
        assert len(result.confirmed_bindings_hash) == 64

    def test_no_file_means_no_file_identity_rather_than_a_placeholder(self) -> None:
        result = compose(confirmations=(confirmed(),))
        assert result.mapping_id is None
        assert result.mapping_version is None
        assert result.mapping_sha256 is None
        assert result.mapping_source is None


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
class TestIdentity:
    def test_the_hash_is_deterministic(self) -> None:
        first = compose(confirmations=(confirmed(),))
        second = compose(confirmations=(confirmed(),))
        assert first.effective_mapping_hash == second.effective_mapping_hash

    def test_wall_clock_time_does_not_enter_the_hash(self) -> None:
        """An identity that moved with the clock could answer nothing."""
        later = T0 + dt.timedelta(days=30)
        assert (
            compose(confirmations=(confirmed(),), at=T0).effective_mapping_hash
            == compose(confirmations=(confirmed(),), at=later).effective_mapping_hash
        )

    def test_confirmation_ordering_does_not_change_the_hash(self) -> None:
        one = confirmed(area_id=AREA_A, segment_id="SEG-21")
        two = confirmed(area_id=AREA_B, segment_id="SEG-12", junction_id="J2")
        assert (
            compose(confirmations=(one, two)).effective_mapping_hash
            == compose(confirmations=(two, one)).effective_mapping_hash
        )

    def test_file_binding_ordering_does_not_change_the_hash(self) -> None:
        a = ObservationBinding(observation_area_id=AREA_A, segment_id="SEG-14")
        b = ObservationBinding(observation_area_id=AREA_B, segment_id="SEG-12")
        assert (
            compose(mapping=file_mapping(a, b)).effective_mapping_hash
            == compose(mapping=file_mapping(b, a)).effective_mapping_hash
        )

    def test_dictionary_ordering_does_not_change_the_confirmed_hash(self) -> None:
        """Canonical JSON sorts keys, so construction order cannot leak in."""
        one = confirmed(area_id=AREA_A)
        two = confirmed(area_id=AREA_B, junction_id="J2")
        assert confirmed_bindings_fingerprint((one, two)) == (
            confirmed_bindings_fingerprint((two, one))
        )

    def test_a_different_target_changes_the_hash(self) -> None:
        assert (
            compose(confirmations=(confirmed(segment_id="SEG-21"),)).effective_mapping_hash
            != compose(
                confirmations=(confirmed(segment_id="SEG-41"),)
            ).effective_mapping_hash
        )

    def test_a_different_precedence_decision_changes_the_hash(self) -> None:
        """Same roads for different reasons is not the same mapping."""
        agreeing = compose(
            mapping=file_mapping(
                ObservationBinding(observation_area_id=AREA_A, segment_id="SEG-21")
            ),
            confirmations=(confirmed(segment_id="SEG-21"),),
        )
        confirmed_only = compose(confirmations=(confirmed(segment_id="SEG-21"),))
        assert agreeing.bindings[0].binding.segment_id == (
            confirmed_only.bindings[0].binding.segment_id
        )
        assert agreeing.effective_mapping_hash != confirmed_only.effective_mapping_hash

    def test_the_file_content_digest_is_part_of_the_identity(self) -> None:
        mapping = file_mapping(
            ObservationBinding(observation_area_id=AREA_A, segment_id="SEG-14")
        )
        assert (
            compose(mapping=mapping, source=file_source(mapping, "c" * 64))
            .effective_mapping_hash
            != compose(mapping=mapping, source=file_source(mapping, "d" * 64))
            .effective_mapping_hash
        )

    def test_the_version_is_the_hash_prefix(self) -> None:
        result = compose(confirmations=(confirmed(),))
        assert result.effective_mapping_version == result.effective_mapping_hash[:16]
        assert result.mapping.version == result.effective_mapping_version
        assert result.mapping.mapping_id.startswith("effective-")

    def test_the_hash_is_a_sha256_hex_digest(self) -> None:
        result = compose(confirmations=(confirmed(),))
        assert len(result.effective_mapping_hash) == 64
        assert set(result.effective_mapping_hash) <= set("0123456789abcdef")


# ---------------------------------------------------------------------------
# Exclusions
# ---------------------------------------------------------------------------
class TestExclusions:
    def test_a_stale_network_version_is_excluded_not_migrated(self) -> None:
        result = compose(confirmations=(confirmed(network_version="0.9"),))
        assert result.bindings == ()
        assert len(result.excluded) == 1
        excluded = result.excluded[0]
        assert excluded.reason is ExclusionReason.STALE_NETWORK_VERSION
        assert "Re-confirm" in excluded.detail
        assert any("require reconfirmation" in w for w in result.warnings)

    def test_a_binding_for_another_network_is_excluded(self) -> None:
        result = compose(confirmations=(confirmed(network_id="other-net"),))
        assert result.bindings == ()
        assert result.excluded[0].reason is ExclusionReason.DIFFERENT_NETWORK

    def test_a_binding_naming_an_unknown_element_is_excluded(self) -> None:
        result = compose(confirmations=(confirmed(segment_id="SEG-NOPE"),))
        assert result.bindings == ()
        assert result.excluded[0].reason is ExclusionReason.UNKNOWN_NETWORK_ELEMENT
        assert "SEG-NOPE" in result.excluded[0].detail

    def test_an_unknown_junction_is_excluded(self) -> None:
        result = compose(confirmations=(confirmed(junction_id="J-NOPE", segment_id=None),))
        assert result.excluded[0].reason is ExclusionReason.UNKNOWN_NETWORK_ELEMENT

    def test_an_excluded_confirmation_does_not_supersede_the_file(self) -> None:
        """The most important exclusion property: the file binding survives."""
        result = compose(
            mapping=file_mapping(
                ObservationBinding(observation_area_id=AREA_A, segment_id="SEG-14")
            ),
            confirmations=(confirmed(segment_id="SEG-21", network_version="0.9"),),
        )
        assert result.bindings[0].binding.segment_id == "SEG-14"
        assert result.bindings[0].resolution is BindingResolution.FILE_ONLY
        assert result.overrides == ()
        assert len(result.excluded) == 1

    def test_a_binding_with_no_pinned_version_is_admitted(self) -> None:
        result = compose(confirmations=(confirmed(network_version=None),))
        assert len(result.bindings) == 1
        assert result.excluded == ()

    def test_exclusions_are_part_of_the_identity(self) -> None:
        with_exclusion = compose(confirmations=(confirmed(network_version="0.9"),))
        without = compose()
        assert with_exclusion.effective_mapping_hash != without.effective_mapping_hash


# ---------------------------------------------------------------------------
# Restraint
# ---------------------------------------------------------------------------
class TestItCannotClaimToBeInForce:
    def test_the_composition_is_not_used_for_route_costing(self) -> None:
        result = compose(confirmations=(confirmed(),))
        assert result.used_by_route_costing is False

    def test_it_cannot_be_marked_in_force(self) -> None:
        result = compose(confirmations=(confirmed(),))
        with pytest.raises(Exception, match="used_by_route_costing cannot be true"):
            result.model_validate(
                result.model_dump() | {"used_by_route_costing": True}
            )

    def test_the_note_says_route_costing_is_unaffected(self) -> None:
        result = compose(confirmations=(confirmed(),))
        assert "not used for route costing" in result.note


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@pytest.fixture
def effective_settings(tmp_path: Path) -> Settings:
    return Settings(
        app_env="local",
        log_level="ERROR",
        media_dir=tmp_path / "media",
        sites_file=tmp_path / "sites.json",
        confirmed_bindings_file=tmp_path / "bindings.json",
        network_file="example-grid.json",
    )


@pytest.fixture
def client(effective_settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(effective_settings), raise_server_exceptions=False) as c:
        yield c


def seed_confirmation(
    client: TestClient, area_id: str, junction_id: str, segment_id: str
) -> None:
    """Record a site and confirm a junction then a segment, through the API."""
    assert (
        client.post(
            "/api/sites",
            json={
                "observation_area_id": area_id,
                "latitude": 16.505,
                "longitude": 80.600,
                "bearing_degrees": 90,
            },
        ).status_code
        in (200, 201)
    )
    assert (
        client.post(
            f"/api/sites/{area_id}/binding",
            json={"junction_id": junction_id, "confirm": True},
        ).status_code
        in (200, 201)
    )
    assert (
        client.post(
            f"/api/sites/{area_id}/binding/segment",
            json={"junction_id": junction_id, "segment_id": segment_id, "confirm": True},
        ).status_code
        in (200, 201)
    )


class TestEffectiveMappingApi:
    def test_it_answers_with_no_mapping_and_no_confirmations(
        self, client: TestClient
    ) -> None:
        response = client.get("/api/sites/effective-mapping")
        assert response.status_code == 200
        body = response.json()["effective"]
        assert body["bindings"] == []
        assert body["network_id"] == "example-grid"
        assert len(body["effective_mapping_hash"]) == 64

    def test_no_network_loaded_is_refused_with_its_own_code(
        self, tmp_path: Path
    ) -> None:
        """The composition is network-scoped; without one there is no question."""
        settings = Settings(
            app_env="local",
            log_level="ERROR",
            media_dir=tmp_path / "media",
            sites_file=tmp_path / "sites.json",
            confirmed_bindings_file=tmp_path / "bindings.json",
        )
        with TestClient(create_app(settings), raise_server_exceptions=False) as c:
            response = c.get("/api/sites/effective-mapping")
            assert response.status_code == 404
            assert response.json()["error"]["code"] == "network_not_loaded"

    def test_an_unreadable_binding_registry_is_a_502(self, tmp_path: Path) -> None:
        """Never an empty composition: that would hide bindings still on file."""
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
            response = c.get("/api/sites/effective-mapping")
            assert response.status_code == 502
            assert response.json()["error"]["code"] == "data_source_error"

    def test_it_reports_the_file_identity(self, client: TestClient) -> None:
        client.post(
            "/api/routes/mapping/load",
            json={"relative_path": "example-grid-mapping.json"},
        )
        body = client.get("/api/sites/effective-mapping").json()
        assert body["effective"]["mapping_id"] == "example-grid-mapping"
        assert body["effective"]["mapping_version"] == "1.0.0"
        assert len(body["effective"]["mapping_sha256"]) == 64
        assert body["optimizer_mapping_id"] == "example-grid-mapping"

    def test_it_reports_a_confirmation_overriding_the_file(
        self, client: TestClient
    ) -> None:
        client.post(
            "/api/routes/mapping/load",
            json={"relative_path": "example-grid-mapping.json"},
        )
        area = client.get("/api/sites/effective-mapping").json()["effective"][
            "bindings"
        ][0]["observation_area_id"]
        # SEG-21 runs J2 -> J1, so it is incident to J1 and a valid confirmation.
        seed_confirmation(client, area, "J1", "SEG-21")

        body = client.get("/api/sites/effective-mapping").json()["effective"]
        entry = next(
            item for item in body["bindings"] if item["observation_area_id"] == area
        )
        assert entry["binding"]["segment_id"] == "SEG-21"
        assert entry["source"] == "confirmed_binding"
        assert entry["resolution"] == "confirmed_binding_precedence"
        assert entry["confirmed"] is not None

        override = next(
            item for item in body["overrides"] if item["observation_area_id"] == area
        )
        assert override["effective_binding"] == "SEG-21"
        assert override["resolution"] == "confirmed_binding_precedence"

    def test_the_response_states_that_costing_is_unaffected(
        self, client: TestClient
    ) -> None:
        body = client.get("/api/sites/effective-mapping").json()
        assert body["effective_mapping_in_use"] is False
        assert body["effective"]["used_by_route_costing"] is False
        assert "not used for route costing" in body["effective"]["note"]

    def test_the_endpoint_is_not_shadowed_by_the_site_catch_all(
        self, client: TestClient
    ) -> None:
        """`/effective-mapping` must not be read as an observation area id."""
        assert client.get("/api/sites/effective-mapping").status_code == 200
        assert client.get("/api/sites/confirmed").status_code == 200
        seed_confirmation(client, AREA_A, "J1", "SEG-21")
        assert client.get(f"/api/sites/{AREA_A}").status_code == 200

    def test_asking_for_it_writes_nothing(self, client: TestClient) -> None:
        seed_confirmation(client, AREA_A, "J1", "SEG-21")
        before = client.get("/api/sites/confirmed").json()
        for _ in range(3):
            assert client.get("/api/sites/effective-mapping").status_code == 200
        assert client.get("/api/sites/confirmed").json() == before

    def test_the_hash_is_stable_across_repeated_requests(
        self, client: TestClient
    ) -> None:
        seed_confirmation(client, AREA_A, "J1", "SEG-21")
        hashes = {
            client.get("/api/sites/effective-mapping").json()["effective"][
                "effective_mapping_hash"
            ]
            for _ in range(3)
        }
        assert len(hashes) == 1

    def test_the_hash_survives_a_restart(self, effective_settings: Settings) -> None:
        """Derived from persisted inputs, so it is reproducible, not remembered."""
        with TestClient(create_app(effective_settings)) as first:
            seed_confirmation(first, AREA_A, "J1", "SEG-21")
            before = first.get("/api/sites/effective-mapping").json()["effective"]

        with TestClient(create_app(effective_settings)) as second:
            after = second.get("/api/sites/effective-mapping").json()["effective"]

        assert after["effective_mapping_hash"] == before["effective_mapping_hash"]
        assert after["confirmed_bindings_hash"] == before["confirmed_bindings_hash"]
        assert len(after["bindings"]) == 1

    def test_nothing_is_persisted_for_the_effective_mapping(
        self, effective_settings: Settings, client: TestClient
    ) -> None:
        seed_confirmation(client, AREA_A, "J1", "SEG-21")
        client.get("/api/sites/effective-mapping")
        written = {
            path.name for path in effective_settings.sites_file.parent.iterdir()
        }
        assert written <= {
            effective_settings.sites_file.name,
            effective_settings.confirmed_bindings_file.name,
        }


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

    def test_composing_does_not_change_route_costs_or_ranking(
        self, client: TestClient
    ) -> None:
        """The property this whole step is built to preserve."""
        request = {
            "origin_junction_id": "J1",
            "destination_junction_id": "J3",
            "mode": "normal",
        }
        before = client.post("/api/routes/optimize", json=request)
        assert before.status_code == 200

        seed_confirmation(client, AREA_A, "J1", "SEG-21")
        composed = client.get("/api/sites/effective-mapping")
        assert composed.status_code == 200
        assert composed.json()["effective"]["bindings"]

        after = client.post("/api/routes/optimize", json=request)
        assert after.status_code == 200
        assert self._ranking(after.json()) == self._ranking(before.json())
        assert (
            after.json()["unmapped_observation_area_ids"]
            == before.json()["unmapped_observation_area_ids"]
        )

    def test_the_optimizer_still_pins_the_file_mapping_only(
        self, client: TestClient
    ) -> None:
        """A route result must not carry an identity it did not compute against."""
        client.post(
            "/api/routes/mapping/load",
            json={"relative_path": "example-grid-mapping.json"},
        )
        seed_confirmation(client, AREA_A, "J1", "SEG-21")
        effective = client.get("/api/sites/effective-mapping").json()["effective"]

        result = client.post(
            "/api/routes/optimize",
            json={
                "origin_junction_id": "J1",
                "destination_junction_id": "J3",
                "mode": "normal",
            },
        ).json()

        assert result["mapping_id"] == "example-grid-mapping"
        assert result["mapping_version"] == "1.0.0"
        assert result["mapping_version"] != effective["effective_mapping_version"]
        assert "effective_mapping_hash" not in result

    def test_the_loaded_mapping_endpoint_is_unchanged(
        self, client: TestClient
    ) -> None:
        """Existing clients keep working."""
        client.post(
            "/api/routes/mapping/load",
            json={"relative_path": "example-grid-mapping.json"},
        )
        body = client.get("/api/routes/mapping").json()
        assert body["loaded"] is True
        assert body["mapping"]["mapping_id"] == "example-grid-mapping"
        assert "bound_segment_ids" in body
        assert "confirmed_binding_count" in body


# ---------------------------------------------------------------------------
# Canonical form
# ---------------------------------------------------------------------------
class TestCanonicalForm:
    def test_the_canonical_rendering_is_json_serialisable(self) -> None:
        """A hash over something that will not serialise is not reproducible."""
        result = compose(
            mapping=file_mapping(
                ObservationBinding(observation_area_id=AREA_B, segment_id="SEG-12")
            ),
            confirmations=(confirmed(),),
        )
        assert json.loads(result.mapping.model_dump_json())

    def test_an_empty_confirmed_set_still_has_an_identity(self) -> None:
        digest = confirmed_bindings_fingerprint(())
        assert len(digest) == 64
        assert digest == confirmed_bindings_fingerprint(())
