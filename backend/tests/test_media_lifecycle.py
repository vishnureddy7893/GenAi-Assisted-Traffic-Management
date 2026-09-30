"""Media asset lifecycle across a process restart.

The regression these pin
------------------------
``MediaStore`` documents that "the in-memory index is rebuilt from disk on
construction, so a restart does not orphan previously ingested files". It did
not do that. The bytes were written to ``<media_id><ext>`` and the index was
left empty, so after any restart -- including a dev-server auto-reload -- every
previously uploaded asset answered:

    No media asset with id '...'.

with the file sitting on disk the whole time. Two things went wrong at once:
lookups by id failed, and the store's own managed copies leaked into the
*library* listing, which exists to show source material a user dropped in rather
than what this system already ingested.

What is deliberately not recovered
----------------------------------
The client's original filename and the original ingest instant were never
written anywhere. The rebuild reconstructs them from the stored name and the
file's modification time rather than inventing plausible values, and these tests
pin that distinction: everything *measurable* survives exactly, and the two
display fields are honestly derived.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.ingestion.storage import MediaStore
from app.main import create_app

pytest.importorskip("PIL", reason="Pillow is required for image ingestion")


def image_bytes(*, width: int = 64, height: int = 48, fmt: str = "PNG") -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (30, 60, 90)).save(buffer, fmt)
    return buffer.getvalue()


@pytest.fixture
def media_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "media"
    directory.mkdir()
    return directory


@pytest.fixture
def lifecycle_settings(media_dir: Path, tmp_path: Path) -> Settings:
    return Settings(
        app_env="local",
        log_level="ERROR",
        media_dir=media_dir,
        sites_file=tmp_path / "sites.json",
        confirmed_bindings_file=tmp_path / "bindings.json",
    )


@pytest.fixture
def client(lifecycle_settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(lifecycle_settings), raise_server_exceptions=False) as c:
        yield c


def upload(client: TestClient, name: str = "junction.png") -> str:
    response = client.post(
        "/api/media/images", files={"file": (name, image_bytes(), "image/png")}
    )
    assert response.status_code == 201, response.text
    return str(response.json()["media_id"])


# ---------------------------------------------------------------------------
# The reported failure
# ---------------------------------------------------------------------------
class TestRestartRecovery:
    def test_an_uploaded_asset_survives_a_restart(
        self, lifecycle_settings: Settings
    ) -> None:
        """The exact reported failure: upload, restart, look the asset up."""
        with TestClient(create_app(lifecycle_settings)) as first:
            media_id = upload(first)

        with TestClient(create_app(lifecycle_settings)) as second:
            response = second.get(f"/api/media/{media_id}")

        assert response.status_code == 200
        assert response.json()["media_id"] == media_id

    def test_run_detection_works_after_a_restart(
        self, lifecycle_settings: Settings
    ) -> None:
        """The end of the reported flow: Run detection must reach the pipeline."""
        with TestClient(create_app(lifecycle_settings)) as first:
            media_id = upload(first)

        with TestClient(create_app(lifecycle_settings), raise_server_exceptions=False) as second:
            response = second.post(f"/api/vision/images/{media_id}/analyze")

        assert response.status_code == 200, response.text
        assert response.json()["media_id"] == media_id

    def test_the_no_such_asset_error_cannot_occur_after_a_restart(
        self, lifecycle_settings: Settings
    ) -> None:
        """The literal message from the bug report, on every media endpoint."""
        with TestClient(create_app(lifecycle_settings)) as first:
            media_id = upload(first)

        with TestClient(create_app(lifecycle_settings), raise_server_exceptions=False) as second:
            for path in (
                f"/api/media/{media_id}",
                f"/api/media/{media_id}/content",
            ):
                response = second.get(path)
                assert response.status_code == 200, path
                assert "No media asset with id" not in response.text

    def test_the_asset_is_listed_again_after_a_restart(
        self, lifecycle_settings: Settings
    ) -> None:
        with TestClient(create_app(lifecycle_settings)) as first:
            media_id = upload(first)

        with TestClient(create_app(lifecycle_settings)) as second:
            listed = second.get("/api/media").json()

        assert listed["count"] == 1
        assert listed["assets"][0]["media_id"] == media_id

    def test_a_managed_asset_does_not_leak_into_the_library(
        self, lifecycle_settings: Settings
    ) -> None:
        """The second symptom: the library shows source material, not our copies."""
        with TestClient(create_app(lifecycle_settings)) as first:
            upload(first)

        with TestClient(create_app(lifecycle_settings)) as second:
            library = second.get("/api/media/library").json()

        assert library["count"] == 0
        assert library["entries"] == []

    def test_measured_properties_survive_the_rebuild_exactly(
        self, lifecycle_settings: Settings
    ) -> None:
        """Everything measurable is re-measured from the bytes, not guessed."""
        with TestClient(create_app(lifecycle_settings)) as first:
            media_id = upload(first)
            before = first.get(f"/api/media/{media_id}").json()

        with TestClient(create_app(lifecycle_settings)) as second:
            after = second.get(f"/api/media/{media_id}").json()

        assert after["kind"] == before["kind"]
        assert after["format"] == before["format"]
        assert after["media_type"] == before["media_type"]
        assert after["size_bytes"] == before["size_bytes"]
        assert after["image"] == before["image"]

    def test_the_original_filename_is_the_stored_name_not_an_invention(
        self, lifecycle_settings: Settings
    ) -> None:
        """The client's name was never written down, and is not fabricated."""
        with TestClient(create_app(lifecycle_settings)) as first:
            media_id = upload(first, name="operator-chose-this.png")
            assert (
                first.get(f"/api/media/{media_id}").json()["original_filename"]
                == "operator-chose-this.png"
            )

        with TestClient(create_app(lifecycle_settings)) as second:
            recovered = second.get(f"/api/media/{media_id}").json()

        assert recovered["original_filename"] == f"{media_id}.png"
        assert "operator-chose-this" not in recovered["original_filename"]


# ---------------------------------------------------------------------------
# The store itself
# ---------------------------------------------------------------------------
class TestRebuild:
    def test_an_empty_directory_rebuilds_to_nothing(self, media_dir: Path) -> None:
        assert MediaStore(media_dir).list() == ()

    def test_a_missing_directory_is_not_an_error(self, tmp_path: Path) -> None:
        store = MediaStore(tmp_path / "absent")
        assert store.list() == ()

    def test_only_managed_stems_are_adopted(self, media_dir: Path) -> None:
        """A file a user dropped in is source material and stays selectable."""
        (media_dir / "my-own-footage.png").write_bytes(image_bytes())
        store = MediaStore(media_dir)

        assert store.list() == ()
        assert len(store.library_files(frozenset({".png"}))) == 1

    def test_a_managed_stem_is_adopted(self, media_dir: Path) -> None:
        media_id = "0" * 32
        (media_dir / f"{media_id}.png").write_bytes(image_bytes())
        store = MediaStore(media_dir)

        assert store.has(media_id)
        assert store.get(media_id).kind.value == "image"

    def test_a_uuid_like_but_invalid_stem_is_not_adopted(
        self, media_dir: Path
    ) -> None:
        """31 hex characters, or non-hex, is not a stem this store generated."""
        for stem in ("0" * 31, "0" * 33, "g" * 32, "ABCDEF" + "0" * 26):
            (media_dir / f"{stem}.png").write_bytes(image_bytes())
        assert MediaStore(media_dir).list() == ()

    def test_content_decides_the_format_not_the_extension(
        self, media_dir: Path
    ) -> None:
        """A renamed file must not change what the store believes it holds."""
        media_id = "1" * 32
        (media_dir / f"{media_id}.mp4").write_bytes(image_bytes(fmt="PNG"))
        store = MediaStore(media_dir)

        assert store.get(media_id).format == "png"
        assert store.get(media_id).kind.value == "image"

    def test_an_unreadable_file_is_skipped_rather_than_crashing(
        self, media_dir: Path
    ) -> None:
        """One corrupt file must not stop a process from starting."""
        (media_dir / f"{'2' * 32}.png").write_bytes(b"this is not an image at all")
        (media_dir / f"{'3' * 32}.png").write_bytes(image_bytes())

        store = MediaStore(media_dir)
        assert len(store.list()) == 1
        assert store.has("3" * 32)
        assert not store.has("2" * 32)

    def test_rebuilding_is_idempotent(self, media_dir: Path) -> None:
        (media_dir / f"{'4' * 32}.png").write_bytes(image_bytes())
        store = MediaStore(media_dir)
        assert store.rebuild_from_disk() == 0
        assert len(store.list()) == 1

    def test_a_rebuild_never_overwrites_a_live_record(
        self, media_dir: Path
    ) -> None:
        """A live record holds the client's real filename.

        A rebuild must not replace it with a reconstruction: that would lose
        information every time the store was asked to re-scan.
        """
        from app.ingestion.assets import MediaAsset
        from app.ingestion.media import MediaKind, formats_for_kind
        from app.ingestion.probe import probe_image
        from app.ingestion.storage import utc_now

        png = next(
            fmt for fmt in formats_for_kind(MediaKind.IMAGE) if fmt.name == "png"
        )
        store = MediaStore(media_dir)
        media_id, destination = store.allocate(png)
        destination.write_bytes(image_bytes())
        store.register(
            MediaAsset(
                media_id=media_id,
                kind=MediaKind.IMAGE,
                format=png.name,
                media_type=png.media_type,
                original_filename="operator-chose-this.png",
                size_bytes=destination.stat().st_size,
                ingested_at=utc_now(),
                image=probe_image(destination),
                video=None,
            )
        )

        assert store.rebuild_from_disk() == 0
        assert store.get(media_id).original_filename == "operator-chose-this.png"

    def test_clear_does_not_silently_re_adopt(self, media_dir: Path) -> None:
        """A test asking for an empty index must get one."""
        (media_dir / f"{'5' * 32}.png").write_bytes(image_bytes())
        store = MediaStore(media_dir)
        assert len(store.list()) == 1

        store.clear()
        assert store.list() == ()

    def test_rebuild_can_be_disabled(self, media_dir: Path) -> None:
        (media_dir / f"{'6' * 32}.png").write_bytes(image_bytes())
        assert MediaStore(media_dir, rebuild=False).list() == ()


# ---------------------------------------------------------------------------
# Assets that are genuinely gone
# ---------------------------------------------------------------------------
class TestGenuinelyMissingAssets:
    def test_an_unknown_id_is_still_a_clean_404(self, client: TestClient) -> None:
        """The error must still happen when the asset really does not exist."""
        response = client.get("/api/media/f9cadc38a2444d1f98cce6eda4c1767b")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"
        assert "No media asset with id" in response.json()["error"]["message"]

    def test_running_detection_on_an_unknown_id_is_a_clean_404(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/api/vision/images/f9cadc38a2444d1f98cce6eda4c1767b/analyze"
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"

    def test_a_deleted_asset_stays_deleted_across_a_restart(
        self, lifecycle_settings: Settings
    ) -> None:
        """The rebuild reads the disk, so removing the file removes the asset."""
        with TestClient(create_app(lifecycle_settings)) as first:
            media_id = upload(first)
            assert first.delete(f"/api/media/{media_id}").status_code == 204

        with TestClient(create_app(lifecycle_settings), raise_server_exceptions=False) as second:
            assert second.get(f"/api/media/{media_id}").status_code == 404
            assert second.get("/api/media").json()["count"] == 0

    def test_a_file_removed_outside_the_application_removes_the_asset(
        self, lifecycle_settings: Settings, media_dir: Path
    ) -> None:
        with TestClient(create_app(lifecycle_settings)) as first:
            media_id = upload(first)

        for path in media_dir.glob(f"{media_id}.*"):
            path.unlink()

        with TestClient(create_app(lifecycle_settings), raise_server_exceptions=False) as second:
            response = second.post(f"/api/vision/images/{media_id}/analyze")
            assert response.status_code == 404
            assert "No media asset with id" in response.json()["error"]["message"]


# ---------------------------------------------------------------------------
# The normal flow
# ---------------------------------------------------------------------------
class TestNormalFlow:
    def test_upload_then_run_detection_succeeds(self, client: TestClient) -> None:
        """Upload -> id -> Run detection, with no restart in between."""
        media_id = upload(client)

        assert client.get(f"/api/media/{media_id}").status_code == 200
        response = client.post(f"/api/vision/images/{media_id}/analyze")
        assert response.status_code == 200, response.text
        assert response.json()["media_id"] == media_id

    def test_the_uploaded_id_is_the_one_that_resolves(
        self, client: TestClient
    ) -> None:
        """No id is minted anywhere but the upload response."""
        media_id = upload(client)
        listed = client.get("/api/media").json()["assets"]
        assert [item["media_id"] for item in listed] == [media_id]

    def test_the_no_such_asset_error_cannot_occur_in_the_normal_flow(
        self, client: TestClient
    ) -> None:
        media_id = upload(client)
        for response in (
            client.get(f"/api/media/{media_id}"),
            client.get(f"/api/media/{media_id}/content"),
            client.post(f"/api/vision/images/{media_id}/analyze"),
            client.get(f"/api/vision/images/{media_id}/analysis"),
        ):
            assert "No media asset with id" not in response.text

    def test_several_uploads_each_resolve(self, client: TestClient) -> None:
        ids = [upload(client, name=f"frame-{index}.png") for index in range(3)]
        assert len(set(ids)) == 3
        for media_id in ids:
            assert client.get(f"/api/media/{media_id}").status_code == 200
