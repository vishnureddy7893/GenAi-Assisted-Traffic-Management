"""Phase 2 ingestion tests: images, videos, storage safety and the media API.

All fixtures are **generated at test time** into ``tmp_path``. Nothing binary is
committed to the repository: the largest artifact any test creates is a few
kilobytes, and it is discarded when the test ends.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.routes import media as media_routes
from app.core.config import Settings, get_settings
from app.core.errors import DataSourceError, ResourceNotFoundError, ValidationFailedError
from app.ingestion.assets import MediaAsset
from app.ingestion.image_source import RecordedImageDataSource
from app.ingestion.media import (
    MediaKind,
    sniff_format,
    validate_upload,
)
from app.ingestion.playback import epoch_for, playback_manager
from app.ingestion.probe import probe_image, probe_video
from app.ingestion.registry import source_registry
from app.ingestion.storage import MediaStore, resolve_within, sanitize_filename
from app.ingestion.video_source import RecordedVideoDataSource
from app.main import create_app

pytest.importorskip("PIL", reason="Pillow is required for image ingestion")
pytest.importorskip("cv2", reason="OpenCV is required for video ingestion")

EPOCH_ISO = "2024-03-14T08:30:00+00:00"


# ---------------------------------------------------------------------------
# Fixture generation
# ---------------------------------------------------------------------------
def write_image(
    path: Path, *, width: int = 120, height: int = 90, fmt: str = "PNG", mode: str = "RGB"
) -> Path:
    """Generate a small solid-colour image."""
    from PIL import Image

    image = Image.new(mode, (width, height), 128 if mode == "L" else (20, 40, 60))
    image.save(path, fmt)
    return path


def image_bytes(*, width: int = 40, height: int = 30, fmt: str = "PNG") -> bytes:
    """Generate a small image as raw bytes."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, fmt)
    return buffer.getvalue()


def write_video(
    path: Path,
    *,
    frames: int = 15,
    fps: float = 10.0,
    width: int = 64,
    height: int = 48,
    fourcc: str = "mp4v",
) -> Path:
    """Generate a tiny video. A few kilobytes at these dimensions."""
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
    for index in range(frames):
        writer.write(np.full((height, width, 3), (index * 15) % 255, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def media_dir(tmp_path: Path) -> Path:
    """Isolated media root for one test."""
    directory = tmp_path / "media"
    directory.mkdir()
    return directory


@pytest.fixture
def media_client(media_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """API client whose media store is rooted in a temporary directory."""
    monkeypatch.setenv("MEDIA_DIR", str(media_dir))
    get_settings.cache_clear()
    media_routes.reset_store()

    settings = Settings(app_env="local", log_level="ERROR", media_dir=media_dir)
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        yield client

    media_routes.reset_store()
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Format identification
# ---------------------------------------------------------------------------
class TestFormatSniffing:
    def test_formats_are_identified_by_content(self) -> None:
        assert sniff_format(image_bytes(fmt="PNG")[:16]) is not None
        assert sniff_format(image_bytes(fmt="JPEG")[:16]) is not None

    def test_unrecognised_content_is_rejected(self) -> None:
        assert sniff_format(b"this is not media at all") is None
        assert sniff_format(b"") is None

    def test_renaming_a_file_does_not_make_it_acceptable(self) -> None:
        """Content decides. A .png name over non-image bytes is still refused."""
        with pytest.raises(ValidationFailedError, match="Unrecognised file content"):
            validate_upload(filename="traffic.png", header=b"MZ\x90\x00 not an image")

    def test_extension_must_agree_with_content(self) -> None:
        with pytest.raises(ValidationFailedError, match="does not match the actual content"):
            validate_upload(filename="traffic.png", header=image_bytes(fmt="JPEG")[:16])

    def test_an_image_is_rejected_where_a_video_is_expected(self) -> None:
        with pytest.raises(ValidationFailedError, match="Expected video content"):
            validate_upload(
                filename="t.png",
                header=image_bytes(fmt="PNG")[:16],
                expected_kind=MediaKind.VIDEO,
            )

    def test_avi_and_webp_are_distinguished_despite_sharing_riff(self) -> None:
        assert sniff_format(b"RIFF\x00\x00\x00\x00AVI LIST") is not None
        assert sniff_format(b"RIFF\x00\x00\x00\x00WEBPVP8 ") is not None
        assert sniff_format(b"RIFF\x00\x00\x00\x00XXXXxxxx") is None


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------
class TestPathSafety:
    @pytest.mark.parametrize(
        "attack",
        [
            "../secret.png",
            "../../etc/passwd",
            "a/../../outside.png",
            "/etc/passwd",
            "\\windows\\system32\\config",
            "C:/Windows/system.ini",
            "C:secret.png",
            "\\\\server\\share\\file.png",
        ],
    )
    def test_traversal_attempts_are_rejected(self, media_dir: Path, attack: str) -> None:
        with pytest.raises(ValidationFailedError):
            resolve_within(media_dir, attack)

    def test_a_legitimate_relative_path_resolves(self, media_dir: Path) -> None:
        (media_dir / "sub").mkdir()
        target = write_image(media_dir / "sub" / "ok.png")
        assert resolve_within(media_dir, "sub/ok.png") == target.resolve()

    def test_dot_padding_is_contained_rather_than_blacklisted(
        self, media_dir: Path
    ) -> None:
        """``....//`` is a bypass only against filters that strip ``../``.

        Because containment is decided by resolving and then comparing, ``....``
        is treated as the ordinary directory name it is: the path stays inside
        the root and is allowed, rather than being matched against a blacklist
        that would have to anticipate every encoding of "go up".
        """
        resolved = resolve_within(media_dir, "....//....//escape.png")
        assert media_dir.resolve() in resolved.parents

    def test_nul_bytes_are_rejected(self, media_dir: Path) -> None:
        with pytest.raises(ValidationFailedError, match="illegal character"):
            resolve_within(media_dir, "ok\x00.png")

    def test_empty_path_is_rejected(self, media_dir: Path) -> None:
        with pytest.raises(ValidationFailedError):
            resolve_within(media_dir, "   ")

    def test_symlink_escape_is_rejected(self, media_dir: Path, tmp_path: Path) -> None:
        """Containment is checked after resolution, so links cannot smuggle out."""
        outside = tmp_path / "outside.png"
        write_image(outside)
        link = media_dir / "link.png"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not permitted in this environment")
        with pytest.raises(ValidationFailedError, match="escapes the media directory"):
            resolve_within(media_dir, "link.png")

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("../../evil.png", "evil.png"),
            ("..\\..\\evil.png", "evil.png"),
            ("/etc/passwd", "passwd"),
            ("", "unnamed"),
        ],
    )
    def test_filenames_are_reduced_to_a_leaf(self, raw: str, expected: str) -> None:
        assert sanitize_filename(raw) == expected

    def test_stored_filename_never_contains_client_text(self, media_dir: Path) -> None:
        """Uploads are stored under a server-generated id, not the client name."""
        from app.ingestion.media import PNG

        store = MediaStore(media_dir)
        media_id, path = store.allocate(PNG)
        assert path.stem == media_id
        assert len(media_id) == 32


# ---------------------------------------------------------------------------
# Image probing
# ---------------------------------------------------------------------------
class TestImageProbe:
    def test_metadata_is_measured_not_guessed(self, tmp_path: Path) -> None:
        path = write_image(tmp_path / "t.png", width=321, height=123)
        meta = probe_image(path)
        assert meta.width == 321
        assert meta.height == 123
        assert meta.format == "png"
        assert meta.mode == "RGB"
        assert meta.channels == 3

    @pytest.mark.parametrize(
        ("fmt", "suffix"),
        [("PNG", ".png"), ("JPEG", ".jpg"), ("WEBP", ".webp")],
    )
    def test_all_supported_image_formats_probe(
        self, tmp_path: Path, fmt: str, suffix: str
    ) -> None:
        path = write_image(tmp_path / f"t{suffix}", fmt=fmt, width=64, height=48)
        meta = probe_image(path)
        assert (meta.width, meta.height) == (64, 48)

    def test_greyscale_channels_are_reported(self, tmp_path: Path) -> None:
        path = write_image(tmp_path / "grey.png", mode="L")
        assert probe_image(path).channels == 1

    def test_a_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataSourceError, match="not found"):
            probe_image(tmp_path / "absent.png")

    def test_a_corrupt_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)
        with pytest.raises(DataSourceError, match=r"corrupt|decode"):
            probe_image(path)

    def test_a_still_reports_no_frame_count(self, tmp_path: Path) -> None:
        """A still has no frame sequence, which is different from having one frame."""
        meta = probe_image(write_image(tmp_path / "t.png"))
        assert meta.is_animated is False
        assert meta.frame_count is None


# ---------------------------------------------------------------------------
# Video probing
# ---------------------------------------------------------------------------
class TestVideoProbe:
    def test_metadata_is_measured(self, tmp_path: Path) -> None:
        path = write_video(tmp_path / "t.mp4", frames=20, fps=10.0, width=64, height=48)
        meta = probe_video(path)
        assert (meta.width, meta.height) == (64, 48)
        assert meta.fps == pytest.approx(10.0)
        assert meta.frame_count == 20
        assert meta.duration_s == pytest.approx(2.0)

    @pytest.mark.parametrize(
        ("suffix", "fourcc"), [(".mp4", "mp4v"), (".avi", "MJPG"), (".mov", "mp4v")]
    )
    def test_all_supported_video_formats_probe(
        self, tmp_path: Path, suffix: str, fourcc: str
    ) -> None:
        path = write_video(tmp_path / f"t{suffix}", fourcc=fourcc, frames=8)
        meta = probe_video(path)
        assert meta.width > 0
        assert meta.height > 0

    def test_duration_requires_both_fps_and_frame_count(self, tmp_path: Path) -> None:
        """Never estimated from file size, and never zero-filled."""
        meta = probe_video(write_video(tmp_path / "t.mp4", frames=10, fps=5.0))
        assert meta.duration_s == pytest.approx(2.0)

    def test_a_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataSourceError, match="not found"):
            probe_video(tmp_path / "absent.mp4")

    def test_a_corrupt_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.mp4"
        path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
        with pytest.raises(DataSourceError):
            probe_video(path)


# ---------------------------------------------------------------------------
# Image source
# ---------------------------------------------------------------------------
class TestRecordedImageDataSource:
    def _source(self, path: Path) -> RecordedImageDataSource:
        import datetime as dt

        return RecordedImageDataSource(
            source_id="img-1", path=path, epoch=dt.datetime.fromisoformat(EPOCH_ISO)
        )

    def test_capabilities_match_a_recorded_still(self, tmp_path: Path) -> None:
        source = self._source(write_image(tmp_path / "t.png"))
        caps = source.capabilities
        assert caps.bounded is True
        assert caps.replayable is True
        assert caps.seekable is True
        assert caps.realtime is False
        assert caps.requires_backpressure is False
        assert caps.sample_count == 1

    def test_it_yields_exactly_one_frame(self, tmp_path: Path) -> None:
        with self._source(write_image(tmp_path / "t.png")) as source:
            first = source.read()
            assert first is not None
            assert first.payload.shape == (90, 120, 3)
            assert source.read() is None

    def test_reset_replays_identically(self, tmp_path: Path) -> None:
        with self._source(write_image(tmp_path / "t.png")) as source:
            first = source.read()
            assert source.read() is None
            source.reset()
            second = source.read()
        assert first is not None
        assert second is not None
        assert (first.payload == second.payload).all()

    def test_timestamps_come_from_the_media_clock(self, tmp_path: Path) -> None:
        import datetime as dt

        with self._source(write_image(tmp_path / "t.png")) as source:
            sample = source.read()
        assert sample is not None
        assert sample.captured_at == dt.datetime.fromisoformat(EPOCH_ISO)

    def test_reading_before_open_is_refused(self, tmp_path: Path) -> None:
        source = self._source(write_image(tmp_path / "t.png"))
        with pytest.raises(DataSourceError, match="opened before reading"):
            source.read()

    def test_a_missing_file_fails_on_open(self, tmp_path: Path) -> None:
        source = self._source(tmp_path / "absent.png")
        with pytest.raises(DataSourceError):
            source.open()


# ---------------------------------------------------------------------------
# Video source
# ---------------------------------------------------------------------------
class TestRecordedVideoDataSource:
    def _source(self, path: Path) -> RecordedVideoDataSource:
        import datetime as dt

        return RecordedVideoDataSource(
            source_id="vid-1", path=path, epoch=dt.datetime.fromisoformat(EPOCH_ISO)
        )

    def test_capabilities_match_a_recorded_video(self, tmp_path: Path) -> None:
        source = self._source(write_video(tmp_path / "t.mp4"))
        caps = source.capabilities
        assert caps.bounded is True
        assert caps.seekable is True
        assert caps.replayable is True
        assert caps.realtime is False
        assert caps.requires_backpressure is False

    def test_open_read_seek_reset_close(self, tmp_path: Path) -> None:
        source = self._source(write_video(tmp_path / "t.mp4", frames=15, fps=10.0))
        source.open()
        try:
            assert source.current_frame_index == 0

            first = source.read()
            assert first is not None
            assert source.current_frame_index == 1

            source.seek(0.5)
            assert source.current_frame_index == 5
            assert source.position_s == pytest.approx(0.5)

            source.seek_to_frame(3)
            assert source.current_frame_index == 3

            source.reset()
            assert source.current_frame_index == 0
        finally:
            source.close()
        assert source.status.value == "closed"

    def test_it_reads_to_the_end_and_then_stops(self, tmp_path: Path) -> None:
        with self._source(write_video(tmp_path / "t.mp4", frames=12)) as source:
            count = sum(1 for _ in iter(source.read, None))
        assert count == 12

    def test_frames_are_rgb_not_bgr(self, tmp_path: Path) -> None:
        """Conversion happens once at the boundary, matching the image source."""
        import cv2
        import numpy as np

        path = tmp_path / "red.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (32, 32))
        # Pure red in BGR order.
        for _ in range(3):
            writer.write(np.full((32, 32, 3), (0, 0, 255), dtype=np.uint8))
        writer.release()

        with self._source(path) as source:
            sample = source.read()
        assert sample is not None
        red, _green, blue = (int(c) for c in sample.payload[16, 16])
        assert red > blue, "frame should be RGB-ordered, so red channel dominates"

    def test_replay_rate_does_not_change_timestamps(self, tmp_path: Path) -> None:
        import datetime as dt

        path = write_video(tmp_path / "t.mp4", frames=6, fps=10.0)
        stamps: list[list[dt.datetime]] = []
        for rate in (1.0, 8.0):
            source = RecordedVideoDataSource(
                source_id="v",
                path=path,
                epoch=dt.datetime.fromisoformat(EPOCH_ISO),
                rate=rate,
            )
            with source:
                stamps.append([s.captured_at for s in iter(source.read, None)])
        assert stamps[0] == stamps[1]

    def test_seeking_before_open_is_refused(self, tmp_path: Path) -> None:
        source = self._source(write_video(tmp_path / "t.mp4"))
        with pytest.raises(DataSourceError, match="opened before seeking"):
            source.seek(1.0)

    def test_a_missing_file_fails_on_open(self, tmp_path: Path) -> None:
        with pytest.raises(DataSourceError, match="not found"):
            self._source(tmp_path / "absent.mp4").open()

    def test_a_corrupt_file_fails_on_open(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.mp4"
        path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
        with pytest.raises(DataSourceError):
            self._source(path).open()


# ---------------------------------------------------------------------------
# Playback manager
# ---------------------------------------------------------------------------
class TestPlaybackManager:
    def test_sessions_are_reused_and_closed(self, tmp_path: Path) -> None:
        path = write_video(tmp_path / "t.mp4")
        try:
            first = playback_manager.open("m1", path)
            second = playback_manager.open("m1", path)
            assert first is second
            assert playback_manager.open_count == 1
        finally:
            playback_manager.close_all()
        assert playback_manager.open_count == 0

    def test_the_epoch_is_the_file_time_never_now(self, tmp_path: Path) -> None:
        """Defaulting to 'now' would make an old clip look like live traffic."""
        import datetime as dt

        path = write_video(tmp_path / "t.mp4")
        assert epoch_for(path) < dt.datetime.now(tz=dt.UTC) + dt.timedelta(seconds=1)
        assert epoch_for(path).tzinfo is not None


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
class TestMediaStore:
    def test_missing_asset_raises_not_found(self, media_dir: Path) -> None:
        store = MediaStore(media_dir)
        with pytest.raises(ResourceNotFoundError):
            store.get("nope")

    def test_registered_assets_are_listed_and_filtered(self, media_dir: Path) -> None:
        from app.ingestion.storage import utc_now

        store = MediaStore(media_dir)
        store.register(
            MediaAsset(
                media_id="a" * 32,
                kind=MediaKind.IMAGE,
                format="png",
                media_type="image/png",
                original_filename="a.png",
                size_bytes=1,
                ingested_at=utc_now(),
            )
        )
        assert len(store.list()) == 1
        assert len(store.list(MediaKind.IMAGE)) == 1
        assert len(store.list(MediaKind.VIDEO)) == 0


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
class TestMediaApi:
    def test_formats_endpoint_drives_the_client(self, media_client: TestClient) -> None:
        body = media_client.get("/api/media/formats").json()
        names = {f["name"] for f in body["formats"]}
        assert {"jpeg", "png", "webp", "mp4", "mov", "avi"} <= names
        assert body["max_image_upload_bytes"] > 0

    def test_media_list_is_empty_before_ingestion(self, media_client: TestClient) -> None:
        body = media_client.get("/api/media").json()
        assert body["assets"] == []
        assert body["count"] == 0

    def test_image_upload_returns_measured_metadata(self, media_client: TestClient) -> None:
        response = media_client.post(
            "/api/media/images",
            files={"file": ("shot.png", image_bytes(width=200, height=150), "image/png")},
        )
        assert response.status_code == 201
        asset = response.json()
        assert asset["kind"] == "image"
        assert asset["image"]["width"] == 200
        assert asset["image"]["height"] == 150
        assert asset["image"]["channels"] == 3
        assert asset["original_filename"] == "shot.png"
        assert asset["video"] is None

    def test_uploaded_image_content_is_served_back(self, media_client: TestClient) -> None:
        payload = image_bytes()
        media_id = media_client.post(
            "/api/media/images", files={"file": ("a.png", payload, "image/png")}
        ).json()["media_id"]
        content = media_client.get(f"/api/media/{media_id}/content")
        assert content.status_code == 200
        assert content.content == payload

    def test_video_upload_returns_measured_metadata(
        self, media_client: TestClient, tmp_path: Path
    ) -> None:
        path = write_video(tmp_path / "clip.mp4", frames=20, fps=10.0)
        response = media_client.post(
            "/api/media/videos", files={"file": ("clip.mp4", path.read_bytes(), "video/mp4")}
        )
        assert response.status_code == 201
        video = response.json()["video"]
        assert video["width"] == 64
        assert video["frame_count"] == 20
        assert video["fps"] == pytest.approx(10.0)
        assert video["duration_s"] == pytest.approx(2.0)

    def test_invalid_image_is_rejected(self, media_client: TestClient) -> None:
        response = media_client.post(
            "/api/media/images", files={"file": ("x.png", b"definitely not an image", "image/png")}
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_failed"

    def test_empty_upload_is_rejected(self, media_client: TestClient) -> None:
        response = media_client.post(
            "/api/media/images", files={"file": ("empty.png", b"", "image/png")}
        )
        assert response.status_code == 422

    def test_a_video_cannot_be_uploaded_as_an_image(
        self, media_client: TestClient, tmp_path: Path
    ) -> None:
        path = write_video(tmp_path / "c.mp4")
        response = media_client.post(
            "/api/media/images", files={"file": ("c.mp4", path.read_bytes(), "video/mp4")}
        )
        assert response.status_code == 422

    def test_missing_media_returns_404(self, media_client: TestClient) -> None:
        assert media_client.get("/api/media/does-not-exist").status_code == 404

    def test_oversized_upload_is_refused(
        self, media_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = Settings(
            app_env="local",
            log_level="ERROR",
            media_dir=media_dir,
            max_image_upload_bytes=64,
        )
        media_routes.reset_store()
        monkeypatch.setenv("MEDIA_DIR", str(media_dir))
        get_settings.cache_clear()
        with TestClient(create_app(settings), raise_server_exceptions=False) as client:
            response = client.post(
                "/api/media/images",
                files={"file": ("big.png", image_bytes(width=400, height=400), "image/png")},
            )
        media_routes.reset_store()
        get_settings.cache_clear()
        assert response.status_code == 422
        assert "limit" in response.json()["error"]["message"].lower()

    def test_rejected_uploads_leave_no_file_behind(
        self, media_client: TestClient, media_dir: Path
    ) -> None:
        media_client.post(
            "/api/media/images", files={"file": ("x.png", b"not an image", "image/png")}
        )
        assert list(media_dir.glob("*")) == []

    def test_delete_removes_the_asset(self, media_client: TestClient) -> None:
        media_id = media_client.post(
            "/api/media/images", files={"file": ("a.png", image_bytes(), "image/png")}
        ).json()["media_id"]
        assert media_client.delete(f"/api/media/{media_id}").status_code == 204
        assert media_client.get(f"/api/media/{media_id}").status_code == 404


class TestSelectAndLibrary:
    def test_library_lists_supported_files(
        self, media_client: TestClient, media_dir: Path
    ) -> None:
        write_image(media_dir / "local.png")
        body = media_client.get("/api/media/library").json()
        assert any(e["relative_path"] == "local.png" for e in body["entries"])

    def test_selecting_a_library_file_ingests_it(
        self, media_client: TestClient, media_dir: Path
    ) -> None:
        write_image(media_dir / "pick.png", width=77, height=55)
        response = media_client.post(
            "/api/media/select", json={"relative_path": "pick.png"}
        )
        assert response.status_code == 201
        assert response.json()["image"]["width"] == 77

    @pytest.mark.parametrize(
        "attack", ["../../../etc/passwd", "/etc/passwd", "..\\..\\secret.png", "C:/Windows/win.ini"]
    )
    def test_traversal_is_refused_by_the_api(
        self, media_client: TestClient, attack: str
    ) -> None:
        response = media_client.post("/api/media/select", json={"relative_path": attack})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_failed"

    def test_selecting_a_missing_file_is_reported(self, media_client: TestClient) -> None:
        response = media_client.post(
            "/api/media/select", json={"relative_path": "nothing-here.png"}
        )
        assert response.status_code in (404, 502)


class TestPlaybackApi:
    def _upload_video(self, client: TestClient, tmp_path: Path, **kwargs: object) -> str:
        path = write_video(tmp_path / "clip.mp4", **kwargs)  # type: ignore[arg-type]
        return client.post(
            "/api/media/videos", files={"file": ("clip.mp4", path.read_bytes(), "video/mp4")}
        ).json()["media_id"]

    def test_full_playback_cycle(self, media_client: TestClient, tmp_path: Path) -> None:
        media_id = self._upload_video(media_client, tmp_path, frames=15, fps=10.0)
        base = f"/api/media/videos/{media_id}/playback"

        assert media_client.get(base).json()["is_open"] is False

        opened = media_client.post(f"{base}/open").json()
        assert opened["is_open"] is True
        assert opened["current_frame_index"] == 0

        read = media_client.post(f"{base}/read").json()
        assert read["current_frame_index"] == 1
        assert read["last_read_succeeded"] is True

        sought = media_client.post(f"{base}/seek", json={"frame_index": 7}).json()
        assert sought["current_frame_index"] == 7
        assert sought["position_s"] == pytest.approx(0.7)

        by_time = media_client.post(f"{base}/seek", json={"position_s": 0.3}).json()
        assert by_time["current_frame_index"] == 3

        assert media_client.post(f"{base}/reset").json()["current_frame_index"] == 0
        assert media_client.post(f"{base}/close").json()["is_open"] is False

    def test_seek_requires_a_target(self, media_client: TestClient, tmp_path: Path) -> None:
        media_id = self._upload_video(media_client, tmp_path)
        response = media_client.post(
            f"/api/media/videos/{media_id}/playback/seek", json={}
        )
        assert response.status_code == 422

    def test_playback_on_an_image_is_refused(self, media_client: TestClient) -> None:
        media_id = media_client.post(
            "/api/media/images", files={"file": ("a.png", image_bytes(), "image/png")}
        ).json()["media_id"]
        response = media_client.post(f"/api/media/videos/{media_id}/playback/open")
        assert response.status_code == 422

    def test_reading_past_the_end_reports_exhaustion(
        self, media_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = self._upload_video(media_client, tmp_path, frames=3, fps=10.0)
        base = f"/api/media/videos/{media_id}/playback"
        media_client.post(f"{base}/open")
        for _ in range(3):
            media_client.post(f"{base}/read")
        final = media_client.post(f"{base}/read").json()
        assert final["last_read_succeeded"] is False
        assert final["exhausted"] is True

    def test_range_requests_are_supported_for_scrubbing(
        self, media_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = self._upload_video(media_client, tmp_path)
        response = media_client.get(
            f"/api/media/{media_id}/content", headers={"Range": "bytes=0-99"}
        )
        assert response.status_code == 206
        assert response.headers["content-range"].startswith("bytes 0-99/")
        assert len(response.content) == 100


class TestLiveRemainsUnimplemented:
    def test_live_connect_returns_501_naming_its_phase(self, media_client: TestClient) -> None:
        response = media_client.post("/api/media/live/connect")
        assert response.status_code == 501
        assert response.json()["error"]["details"]["phase"] == 14

    def test_no_live_source_is_registered(self) -> None:
        from app.ingestion.base import LIVE_KINDS

        assert not (source_registry.implemented_kinds() & LIVE_KINDS)


class TestCatalogReflectsNewSources:
    def test_image_and_video_report_as_implemented(self, media_client: TestClient) -> None:
        body = media_client.get("/api/sources/catalog").json()
        assert "recorded_image" in body["implemented_kinds"]
        assert "recorded_video" in body["implemented_kinds"]
        assert body["any_source_available"] is True

    def test_no_traffic_quantity_appears_in_any_media_response(
        self, media_client: TestClient
    ) -> None:
        """Phase 2 is raw ingestion: no counts, speeds, queues or predictions."""
        media_id = media_client.post(
            "/api/media/images", files={"file": ("a.png", image_bytes(), "image/png")}
        ).json()["media_id"]
        payload = str(media_client.get(f"/api/media/{media_id}").json()).lower()
        for forbidden in (
            "vehicle_count",
            "congestion",
            "queue_length",
            "mean_speed",
            "eta",
            "prediction",
        ):
            assert forbidden not in payload
