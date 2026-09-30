"""Observation site tests: the location of the junction behind an observation area.

Three properties carry the weight here:

* A site is **keyed by an existing identifier** (``observation_area_id``) and
  adds no new one, so a location joins straight onto the verdicts that already
  use that key.
* A site is **persistent and independent of media**. It survives a restart and
  it survives deleting the footage that prompted it, because the junction did
  not move when the file was deleted.
* An **unreadable registry is never reported as an empty one**, and is never
  overwritten. Both would destroy survey data while looking like success.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.errors import DataSourceError, ResourceNotFoundError
from app.domain.network import GeoPoint
from app.domain.provenance import Provenance, ValueSource
from app.domain.site import (
    MAX_ACCURACY_M,
    ObservationSite,
    observation_area_id_for_media,
)
from app.ingestion.storage import MediaStore
from app.main import create_app
from app.sites.registry import (
    SITES_FILE_SCHEMA_VERSION,
    ObservationSiteRegistry,
    utc_now,
)

pytest.importorskip("PIL", reason="Pillow is required for the upload tests")

AREA = "observation-area:junction-a"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sites_file(tmp_path: Path) -> Path:
    """Registry file for one test. Deliberately not under the media root."""
    return tmp_path / "sites" / "observation-sites.json"


@pytest.fixture
def media_dir(tmp_path: Path) -> Path:
    """Isolated media root for one test."""
    directory = tmp_path / "media"
    directory.mkdir()
    return directory


@pytest.fixture
def site_settings(media_dir: Path, sites_file: Path) -> Settings:
    """Settings pointing both stores at this test's temporary directories."""
    return Settings(
        app_env="local", log_level="ERROR", media_dir=media_dir, sites_file=sites_file
    )


@pytest.fixture
def site_client(site_settings: Settings) -> Iterator[TestClient]:
    """API client whose site registry and media store are temporary."""
    with TestClient(create_app(site_settings), raise_server_exceptions=False) as client:
        yield client


def image_bytes(*, width: int = 40, height: int = 30, fmt: str = "PNG") -> bytes:
    """Generate a small image as raw bytes."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, fmt)
    return buffer.getvalue()


def a_site(
    area_id: str = AREA, *, latitude: float = 16.5, longitude: float = 80.6
) -> ObservationSite:
    """A valid site, built directly rather than through the API."""
    now = utc_now()
    return ObservationSite(
        observation_area_id=area_id,
        location=GeoPoint(latitude=latitude, longitude=longitude),
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


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------
class TestObservationSiteModel:
    def test_a_location_is_required(self) -> None:
        """A site whose purpose is to say where something is cannot omit where."""
        now = utc_now()
        with pytest.raises(Exception, match="location"):
            ObservationSite(  # type: ignore[call-arg]
                observation_area_id=AREA,
                provenance=Provenance(
                    source=ValueSource.OBSERVED,
                    producer="t",
                    producer_version="1",
                    observed_at=now,
                    computed_at=now,
                ),
                recorded_at=now,
                updated_at=now,
            )

    @pytest.mark.parametrize(
        ("latitude", "longitude"),
        [(91.0, 0.0), (-90.5, 0.0), (0.0, 180.5), (0.0, -181.0)],
    )
    def test_coordinates_outside_wgs84_are_rejected(
        self, latitude: float, longitude: float
    ) -> None:
        with pytest.raises(Exception, match=r"less than or equal|greater than or equal"):
            GeoPoint(latitude=latitude, longitude=longitude)

    def test_bearing_of_360_is_rejected_because_it_is_zero(self) -> None:
        """Admitting both 0 and 360 would make two equal bearings compare unequal."""
        site = a_site()
        with pytest.raises(Exception, match="less than 360"):
            site.model_validate(site.model_dump() | {"bearing_degrees": 360.0})

    def test_a_zero_metre_accuracy_is_rejected(self) -> None:
        """Zero accuracy claims a perfection no fix provides."""
        site = a_site()
        with pytest.raises(Exception, match="greater than 0"):
            site.model_validate(site.model_dump() | {"accuracy_m": 0.0})

    def test_an_implausible_accuracy_is_rejected(self) -> None:
        site = a_site()
        with pytest.raises(Exception, match="less than or equal"):
            site.model_validate(site.model_dump() | {"accuracy_m": MAX_ACCURACY_M + 1})

    def test_provenance_is_required(self) -> None:
        """A coordinate with no source cannot be told apart from a guess."""
        now = utc_now()
        with pytest.raises(Exception, match="provenance"):
            ObservationSite(  # type: ignore[call-arg]
                observation_area_id=AREA,
                location=GeoPoint(latitude=0.0, longitude=0.0),
                recorded_at=now,
                updated_at=now,
            )

    def test_the_observation_area_id_convention_is_defined_once(self) -> None:
        """The id a site is keyed by is the one analytics already produces."""
        assert observation_area_id_for_media("abc123") == "observation-area:abc123"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class TestRegistry:
    def test_an_absent_file_is_an_empty_registry_not_an_error(
        self, sites_file: Path
    ) -> None:
        registry = ObservationSiteRegistry(sites_file)
        assert registry.is_degraded is False
        assert registry.list() == ()
        assert not sites_file.exists()

    def test_recording_writes_the_file(self, sites_file: Path) -> None:
        registry = ObservationSiteRegistry(sites_file)
        stored, replaced = registry.record(a_site())
        assert replaced is False
        assert stored.observation_area_id == AREA
        assert sites_file.exists()
        document = json.loads(sites_file.read_text(encoding="utf-8"))
        assert document["schema_version"] == SITES_FILE_SCHEMA_VERSION
        assert len(document["sites"]) == 1

    def test_a_site_survives_a_new_registry_over_the_same_file(
        self, sites_file: Path
    ) -> None:
        """The restart case: a fresh process reads back what the last one wrote."""
        ObservationSiteRegistry(sites_file).record(a_site())
        reopened = ObservationSiteRegistry(sites_file)
        assert reopened.get(AREA).location.latitude == 16.5

    def test_re_recording_replaces_and_preserves_the_original_record_time(
        self, sites_file: Path
    ) -> None:
        registry = ObservationSiteRegistry(sites_file)
        first, _ = registry.record(a_site())
        corrected, replaced = registry.record(a_site(latitude=17.0, longitude=81.0))

        assert replaced is True
        assert len(registry) == 1
        assert corrected.location.latitude == 17.0
        assert corrected.recorded_at == first.recorded_at
        assert corrected.was_updated is True

    def test_an_unknown_area_is_a_404(self, sites_file: Path) -> None:
        with pytest.raises(ResourceNotFoundError):
            ObservationSiteRegistry(sites_file).get("observation-area:nobody")

    def test_an_unreadable_file_is_not_reported_as_an_empty_registry(
        self, sites_file: Path
    ) -> None:
        sites_file.parent.mkdir(parents=True, exist_ok=True)
        sites_file.write_text("{ this is not json", encoding="utf-8")

        registry = ObservationSiteRegistry(sites_file)
        assert registry.is_degraded is True
        with pytest.raises(DataSourceError, match="could not be read"):
            registry.list()
        with pytest.raises(DataSourceError):
            registry.get(AREA)

    def test_an_unreadable_file_is_never_overwritten(self, sites_file: Path) -> None:
        """Writing over it would destroy the sites it still contains."""
        sites_file.parent.mkdir(parents=True, exist_ok=True)
        sites_file.write_text("{ this is not json", encoding="utf-8")

        registry = ObservationSiteRegistry(sites_file)
        with pytest.raises(DataSourceError):
            registry.record(a_site())
        assert sites_file.read_text(encoding="utf-8") == "{ this is not json"

    def test_a_file_from_an_unknown_schema_version_is_refused(
        self, sites_file: Path
    ) -> None:
        sites_file.parent.mkdir(parents=True, exist_ok=True)
        sites_file.write_text(
            json.dumps({"schema_version": "999", "sites": []}), encoding="utf-8"
        )
        registry = ObservationSiteRegistry(sites_file)
        assert registry.is_degraded is True
        assert "schema version" in (registry.load_error or "")

    def test_listing_is_most_recently_updated_first(self, sites_file: Path) -> None:
        registry = ObservationSiteRegistry(sites_file)
        registry.record(a_site("observation-area:one"))
        registry.record(a_site("observation-area:two"))
        assert registry.list()[0].observation_area_id == "observation-area:two"

    def test_the_registry_leaves_no_temporary_files_behind(
        self, sites_file: Path
    ) -> None:
        registry = ObservationSiteRegistry(sites_file)
        registry.record(a_site())
        assert sorted(p.name for p in sites_file.parent.iterdir()) == [sites_file.name]


# ---------------------------------------------------------------------------
# Independence from the media store
# ---------------------------------------------------------------------------
class TestIndependenceFromMedia:
    def test_the_registry_file_is_outside_the_media_directory(
        self, site_settings: Settings
    ) -> None:
        """A survey must not be reachable by anything that clears media."""
        media_root = site_settings.media_dir.resolve()
        assert media_root not in site_settings.sites_file.resolve().parents

    def test_clearing_the_media_store_leaves_sites_intact(
        self, site_settings: Settings
    ) -> None:
        registry = ObservationSiteRegistry(site_settings.sites_file)
        registry.record(a_site())

        store = MediaStore(site_settings.media_dir)
        store.clear()

        assert len(ObservationSiteRegistry(site_settings.sites_file)) == 1


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
class TestSitesApi:
    def test_an_empty_registry_answers_200_with_an_empty_list(
        self, site_client: TestClient
    ) -> None:
        """Nobody has surveyed a camera yet is a true statement, not an error."""
        response = site_client.get("/api/sites")
        assert response.status_code == 200
        body = response.json()
        assert body["sites"] == []
        assert body["count"] == 0
        assert body["note"]

    def test_a_valid_site_is_created(self, site_client: TestClient) -> None:
        response = site_client.post(
            "/api/sites",
            json={
                "observation_area_id": AREA,
                "latitude": 16.5062,
                "longitude": 80.6480,
                "bearing_degrees": 135.0,
                "accuracy_m": 8.0,
                "label": "Benz Circle",
            },
        )
        assert response.status_code == 201
        body = response.json()
        assert body["replaced_existing"] is False
        site = body["site"]
        assert site["location"] == {"latitude": 16.5062, "longitude": 80.6480}
        assert site["bearing_degrees"] == 135.0
        assert site["provenance"]["source"] == "observed"
        assert site["recorded_at"] == site["updated_at"]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("latitude", 91.0),
            ("latitude", -90.1),
            ("longitude", 180.1),
            ("longitude", -181.0),
            ("bearing_degrees", 360.0),
            ("bearing_degrees", -1.0),
            ("accuracy_m", 0.0),
            ("accuracy_m", -5.0),
            ("accuracy_m", MAX_ACCURACY_M + 1),
        ],
    )
    def test_invalid_coordinates_are_refused(
        self, site_client: TestClient, field: str, value: float
    ) -> None:
        payload = {
            "observation_area_id": AREA,
            "latitude": 16.5,
            "longitude": 80.6,
            field: value,
        }
        response = site_client.post("/api/sites", json=payload)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_failed"

    def test_a_predicted_location_cannot_be_claimed(
        self, site_client: TestClient
    ) -> None:
        """A predicted junction position is a guess wearing a measurement's clothes."""
        response = site_client.post(
            "/api/sites",
            json={
                "observation_area_id": AREA,
                "latitude": 16.5,
                "longitude": 80.6,
                "source": "predicted",
            },
        )
        assert response.status_code == 422
        assert "cannot be claimed" in response.json()["error"]["message"]

    def test_an_assumed_location_requires_a_justification(
        self, site_client: TestClient
    ) -> None:
        response = site_client.post(
            "/api/sites",
            json={
                "observation_area_id": AREA,
                "latitude": 16.5,
                "longitude": 80.6,
                "source": "assumed",
            },
        )
        assert response.status_code == 422
        assert "notes" in response.json()["error"]["message"]

    def test_a_site_is_retrieved_by_its_observation_area_id(
        self, site_client: TestClient
    ) -> None:
        site_client.post(
            "/api/sites",
            json={"observation_area_id": AREA, "latitude": 16.5, "longitude": 80.6},
        )
        response = site_client.get(f"/api/sites/{AREA}")
        assert response.status_code == 200
        assert response.json()["observation_area_id"] == AREA

    def test_an_unrecorded_area_is_a_404_not_an_empty_body(
        self, site_client: TestClient
    ) -> None:
        response = site_client.get("/api/sites/observation-area:never-surveyed")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"

    def test_sites_are_listed(self, site_client: TestClient) -> None:
        for index in range(3):
            site_client.post(
                "/api/sites",
                json={
                    "observation_area_id": f"observation-area:j{index}",
                    "latitude": 16.0 + index,
                    "longitude": 80.0 + index,
                },
            )
        body = site_client.get("/api/sites").json()
        assert body["count"] == 3
        assert len(body["sites"]) == 3

    def test_recording_the_same_area_twice_updates_rather_than_duplicating(
        self, site_client: TestClient
    ) -> None:
        first = site_client.post(
            "/api/sites",
            json={"observation_area_id": AREA, "latitude": 16.5, "longitude": 80.6},
        )
        second = site_client.post(
            "/api/sites",
            json={
                "observation_area_id": AREA,
                "latitude": 17.9,
                "longitude": 81.1,
                "label": "corrected",
            },
        )

        assert first.status_code == 201
        assert second.status_code == 200
        assert second.json()["replaced_existing"] is True

        listed = site_client.get("/api/sites").json()
        assert listed["count"] == 1
        stored = listed["sites"][0]
        assert stored["location"]["latitude"] == 17.9
        assert stored["label"] == "corrected"
        assert stored["recorded_at"] == first.json()["site"]["recorded_at"]

    def test_an_unreadable_registry_is_a_502_not_an_empty_success(
        self, site_settings: Settings
    ) -> None:
        site_settings.sites_file.parent.mkdir(parents=True, exist_ok=True)
        site_settings.sites_file.write_text("not json at all", encoding="utf-8")

        with TestClient(create_app(site_settings), raise_server_exceptions=False) as c:
            response = c.get("/api/sites")
            assert response.status_code == 502
            assert response.json()["error"]["code"] == "data_source_error"


class TestPersistenceAcrossAppLifecycle:
    def test_a_site_survives_a_restart_of_the_application(
        self, site_settings: Settings
    ) -> None:
        with TestClient(create_app(site_settings)) as first:
            first.post(
                "/api/sites",
                json={"observation_area_id": AREA, "latitude": 16.5, "longitude": 80.6},
            )

        with TestClient(create_app(site_settings)) as second:
            response = second.get(f"/api/sites/{AREA}")
            assert response.status_code == 200
            assert response.json()["location"]["latitude"] == 16.5

    def test_deleting_the_media_does_not_delete_the_site(
        self, site_client: TestClient
    ) -> None:
        """The junction did not move because the footage of it was deleted."""
        upload = site_client.post(
            "/api/media/images",
            files={"file": ("junction.png", image_bytes(), "image/png")},
            data={"latitude": "16.5", "longitude": "80.6"},
        )
        media_id = upload.json()["media_id"]
        area_id = observation_area_id_for_media(media_id)

        assert site_client.delete(f"/api/media/{media_id}").status_code == 204
        assert site_client.get(f"/api/media/{media_id}").status_code == 404
        assert site_client.get(f"/api/sites/{area_id}").status_code == 200


# ---------------------------------------------------------------------------
# Upload with and without a location
# ---------------------------------------------------------------------------
class TestUploadLocation:
    def test_an_upload_without_a_location_behaves_exactly_as_before(
        self, site_client: TestClient
    ) -> None:
        response = site_client.post(
            "/api/media/images",
            files={"file": ("junction.png", image_bytes(), "image/png")},
        )
        assert response.status_code == 201
        asset = response.json()
        assert asset["kind"] == "image"
        assert asset["image"]["width"] == 40
        # No geographic field has appeared on the asset.
        assert "latitude" not in asset
        assert "location" not in asset
        # And no site was invented for it.
        assert site_client.get("/api/sites").json()["count"] == 0

    def test_an_upload_with_a_location_creates_the_site(
        self, site_client: TestClient
    ) -> None:
        response = site_client.post(
            "/api/media/images",
            files={"file": ("junction.png", image_bytes(), "image/png")},
            data={
                "latitude": "16.5062",
                "longitude": "80.6480",
                "bearing_degrees": "270",
                "accuracy_m": "12",
                "site_label": "Benz Circle",
            },
        )
        assert response.status_code == 201
        asset = response.json()
        assert "latitude" not in asset

        site = site_client.get(
            f"/api/sites/{observation_area_id_for_media(asset['media_id'])}"
        ).json()
        assert site["location"]["latitude"] == 16.5062
        assert site["bearing_degrees"] == 270.0
        assert site["accuracy_m"] == 12.0
        assert site["label"] == "Benz Circle"
        assert site["provenance"]["producer"] == "operator:media-upload"

    def test_a_video_upload_takes_the_same_location(
        self, site_client: TestClient, tmp_path: Path
    ) -> None:
        pytest.importorskip("cv2", reason="OpenCV is required for video ingestion")
        import cv2
        import numpy as np

        path = tmp_path / "clip.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
        )
        for index in range(10):
            writer.write(np.full((48, 64, 3), (index * 15) % 255, dtype=np.uint8))
        writer.release()

        response = site_client.post(
            "/api/media/videos",
            files={"file": ("clip.mp4", path.read_bytes(), "video/mp4")},
            data={"latitude": "16.5", "longitude": "80.6"},
        )
        assert response.status_code == 201
        area_id = observation_area_id_for_media(response.json()["media_id"])
        assert site_client.get(f"/api/sites/{area_id}").status_code == 200

    def test_half_a_coordinate_is_refused_and_stores_nothing(
        self, site_client: TestClient
    ) -> None:
        response = site_client.post(
            "/api/media/images",
            files={"file": ("junction.png", image_bytes(), "image/png")},
            data={"latitude": "16.5"},
        )
        assert response.status_code == 422
        assert "both latitude and longitude" in response.json()["error"]["message"]
        assert site_client.get("/api/media").json()["count"] == 0
        assert site_client.get("/api/sites").json()["count"] == 0

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("latitude", "100"),
            ("longitude", "-200"),
            ("bearing_degrees", "360"),
            ("accuracy_m", "0"),
        ],
    )
    def test_an_invalid_location_refuses_the_upload_before_storing_it(
        self, site_client: TestClient, field: str, value: str
    ) -> None:
        """The file is never written, so a rejected upload leaves nothing behind."""
        payload = {"latitude": "16.5", "longitude": "80.6", field: value}
        response = site_client.post(
            "/api/media/images",
            files={"file": ("junction.png", image_bytes(), "image/png")},
            data=payload,
        )
        assert response.status_code == 422
        assert site_client.get("/api/media").json()["count"] == 0
        assert site_client.get("/api/sites").json()["count"] == 0

    def test_re_uploading_the_same_junction_updates_its_site(
        self, site_client: TestClient
    ) -> None:
        """Two uploads are two areas; posting the id again is what corrects one."""
        first = site_client.post(
            "/api/media/images",
            files={"file": ("a.png", image_bytes(), "image/png")},
            data={"latitude": "16.5", "longitude": "80.6"},
        )
        area_id = observation_area_id_for_media(first.json()["media_id"])

        corrected = site_client.post(
            "/api/sites",
            json={
                "observation_area_id": area_id,
                "latitude": 16.9,
                "longitude": 80.9,
                "label": "re-surveyed",
            },
        )
        assert corrected.status_code == 200
        assert corrected.json()["replaced_existing"] is True
        assert site_client.get("/api/sites").json()["count"] == 1


# ---------------------------------------------------------------------------
# Regression: ingestion is unchanged where no location is involved
# ---------------------------------------------------------------------------
class TestMediaIngestionRegression:
    def test_formats_listing_still_works(self, site_client: TestClient) -> None:
        body = site_client.get("/api/media/formats").json()
        assert body["formats"]
        assert body["max_image_upload_bytes"] > 0

    def test_an_upload_is_still_listed_measured_and_served(
        self, site_client: TestClient
    ) -> None:
        upload = site_client.post(
            "/api/media/images",
            files={"file": ("junction.png", image_bytes(), "image/png")},
        )
        media_id = upload.json()["media_id"]

        assert site_client.get("/api/media").json()["count"] == 1
        assert site_client.get(f"/api/media/{media_id}").status_code == 200
        assert site_client.get(f"/api/media/{media_id}/content").status_code == 200
        assert site_client.delete(f"/api/media/{media_id}").status_code == 204

    def test_content_still_decides_the_format(self, site_client: TestClient) -> None:
        """A .png name over non-image bytes is still refused, location or not."""
        response = site_client.post(
            "/api/media/images",
            files={"file": ("evil.png", b"MZ\x90\x00 not an image", "image/png")},
            data={"latitude": "16.5", "longitude": "80.6"},
        )
        assert response.status_code == 422
        assert site_client.get("/api/sites").json()["count"] == 0

    def test_live_ingestion_is_still_refused_with_its_phase(
        self, site_client: TestClient
    ) -> None:
        response = site_client.post("/api/media/live/connect")
        assert response.status_code == 501
        assert response.json()["error"]["details"]["phase"] == 14
