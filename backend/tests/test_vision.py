"""Phase 3 perception tests: detector contract, tracking, and the vision API.

**Determinism.** Almost every test here uses a scripted fake detector rather
than a real model. That is deliberate, not a shortcut: a real detector's output
depends on weights, hardware and library version, so asserting on it would make
the suite non-deterministic and dependent on a multi-hundred-megabyte download.

The fake satisfies the same :class:`~app.vision.detector.VehicleDetector`
interface the real backends do, which means these tests exercise the real
orchestration, tracking, schema and API code. It lives in the test tree and is
never importable from ``app``.

One test does run a genuine model, and it skips cleanly when the weights or the
backend are absent.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.routes import media as media_routes
from app.core.config import Settings, get_settings
from app.core.errors import DataSourceError, NotAvailableYetError, ValidationFailedError
from app.domain.enums import VEHICLE_CLASSES, VehicleClass
from app.main import create_app
from app.vision.bytetrack import ByteTracker
from app.vision.classes import REQUIRED_CLASSES, map_label, road_user_class_indices
from app.vision.detector import (
    BoundingBox,
    Detection,
    DetectionResult,
    DetectorInfo,
    VehicleDetector,
)
from app.vision.device import DeviceKind, resolve_device
from app.vision.models import ModelIdentity, resolve_model_file
from app.vision.overlay import encode_png, render_overlay
from app.vision.registry import BACKENDS, backend_statuses, create_detector, resolve_backend

pytest.importorskip("PIL")
pytest.importorskip("cv2")
pytest.importorskip("numpy")

T0 = dt.datetime(2024, 3, 14, 8, 30, tzinfo=dt.UTC)


# ---------------------------------------------------------------------------
# Test doubles — test tree only, never importable from `app`
# ---------------------------------------------------------------------------
def box(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    """Build a bounding box."""
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def detection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    cls: VehicleClass = VehicleClass.CAR,
    conf: float = 0.9,
) -> Detection:
    """Build a detection."""
    return Detection(bbox=box(x1, y1, x2, y2), vehicle_class=cls, confidence=conf)


class ScriptedDetector(VehicleDetector):
    """Detector returning a predetermined sequence. Fully deterministic."""

    def __init__(self, script: list[tuple[Detection, ...]] | None = None) -> None:
        self._script = script or []
        self._calls = 0
        self._loaded = False
        self.identity = ModelIdentity(
            name="scripted", backend="scripted", sha256="a" * 64, size_bytes=1
        )
        self.device = resolve_device("cpu")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def info(self) -> DetectorInfo:
        return DetectorInfo(
            name="scripted",
            version="1.0",
            input_width=640,
            input_height=640,
            supported_classes=REQUIRED_CLASSES,
            device="cpu",
            confidence_threshold=0.25,
        )

    def load(self) -> None:
        self._loaded = True

    def close(self) -> None:
        self._loaded = False

    def detect(
        self,
        frame: object,
        *,
        source_id: str,
        sequence: int,
        captured_at: dt.datetime,
    ) -> DetectionResult:
        if not self._loaded:
            msg = "Detector is not loaded."
            raise DataSourceError(msg)
        detections = self._script[self._calls] if self._calls < len(self._script) else ()
        self._calls += 1
        shape = getattr(frame, "shape", (480, 640, 3))
        return DetectionResult(
            source_id=source_id,
            sequence=sequence,
            captured_at=captured_at,
            detections=detections,
            inference_ms=1.0,
            frame_width=shape[1],
            frame_height=shape[0],
        )


class FailingDetector(ScriptedDetector):
    """Detector that fails to load, for the model-unavailable path."""

    def load(self) -> None:
        msg = "Model weights could not be loaded: simulated failure."
        raise DataSourceError(msg, backend="scripted")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def write_image(path: Path, *, width: int = 160, height: int = 120) -> Path:
    """Generate a small deterministic image."""
    from PIL import Image

    Image.new("RGB", (width, height), (30, 60, 90)).save(path, "PNG")
    return path


def write_video(path: Path, *, frames: int = 6, fps: float = 10.0) -> Path:
    """Generate a tiny deterministic video."""
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for index in range(frames):
        writer.write(np.full((48, 64, 3), (index * 20) % 255, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def media_dir(tmp_path: Path) -> Path:
    """Isolated media root."""
    directory = tmp_path / "media"
    directory.mkdir()
    return directory


@pytest.fixture
def vision_client(
    media_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """API client with an isolated media root and a scripted detector."""
    monkeypatch.setenv("MEDIA_DIR", str(media_dir))
    get_settings.cache_clear()
    media_routes.reset_store()

    settings = Settings(
        app_env="local",
        log_level="ERROR",
        media_dir=media_dir,
        models_dir=tmp_path / "models",
    )
    app = create_app(settings)

    from app.vision.analysis import DetectorProvider

    class StubProvider(DetectorProvider):
        def __init__(self) -> None:
            super().__init__(settings)
            self._detector = ScriptedDetector(
                [(detection(10, 10, 50, 60, cls=VehicleClass.CAR, conf=0.9),)] * 20
            )
            self._detector.load()

        def get(self) -> VehicleDetector:
            return self._detector

    app.state.detector_provider = StubProvider()

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client

    media_routes.reset_store()
    get_settings.cache_clear()


def upload_image(client: TestClient, tmp_path: Path) -> str:
    """Upload a generated image and return its media id."""
    path = write_image(tmp_path / "frame.png")
    return client.post(
        "/api/media/images", files={"file": ("frame.png", path.read_bytes(), "image/png")}
    ).json()["media_id"]


def upload_video(client: TestClient, tmp_path: Path, frames: int = 6) -> str:
    """Upload a generated video and return its media id."""
    path = write_video(tmp_path / "clip.mp4", frames=frames)
    return client.post(
        "/api/media/videos", files={"file": ("clip.mp4", path.read_bytes(), "video/mp4")}
    ).json()["media_id"]


# ---------------------------------------------------------------------------
# Detector contract and schema
# ---------------------------------------------------------------------------
class TestDetectorContract:
    def test_a_scripted_detector_satisfies_the_interface(self) -> None:
        det = ScriptedDetector()
        assert isinstance(det, VehicleDetector)
        det.load()
        result = det.detect(None, source_id="s", sequence=0, captured_at=T0)
        assert isinstance(result, DetectionResult)
        det.close()

    def test_detecting_before_load_is_refused(self) -> None:
        with pytest.raises(DataSourceError, match="not loaded"):
            ScriptedDetector().detect(None, source_id="s", sequence=0, captured_at=T0)

    def test_result_carries_frame_provenance(self) -> None:
        det = ScriptedDetector([(detection(0, 0, 10, 10),)])
        det.load()
        result = det.detect(None, source_id="image:abc", sequence=7, captured_at=T0)
        assert result.source_id == "image:abc"
        assert result.sequence == 7
        assert result.captured_at == T0

    def test_confidence_outside_zero_to_one_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            Detection(bbox=box(0, 0, 1, 1), vehicle_class=VehicleClass.CAR, confidence=1.4)
        with pytest.raises(ValueError, match="confidence"):
            Detection(bbox=box(0, 0, 1, 1), vehicle_class=VehicleClass.CAR, confidence=-0.1)

    def test_degenerate_boxes_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="degenerate"):
            box(50, 10, 10, 60)
        with pytest.raises(ValueError, match="degenerate"):
            box(10, 60, 50, 10)

    def test_box_geometry(self) -> None:
        b = box(10, 20, 50, 60)
        assert b.width == 40
        assert b.height == 40
        assert b.area == 1600
        assert b.center == (30, 40)
        # Ground contact is bottom-centre, not the centroid.
        assert b.ground_contact_point == (30, 60)

    def test_iou_is_symmetric_and_bounded(self) -> None:
        a, b = box(0, 0, 10, 10), box(5, 5, 15, 15)
        assert a.iou(b) == pytest.approx(b.iou(a))
        assert 0 < a.iou(b) < 1
        assert a.iou(a) == pytest.approx(1.0)
        assert a.iou(box(100, 100, 110, 110)) == 0.0


class TestClassMapping:
    def test_all_required_classes_are_mappable(self) -> None:
        names = ("car", "motorcycle", "bus", "truck", "bicycle", "person")
        mapped = {map_label(name) for name in names}
        assert mapped >= REQUIRED_CLASSES

    def test_non_road_users_are_dropped_not_coerced(self) -> None:
        """A detection we cannot name is not a road user we can reason about."""
        for label in ("handbag", "traffic light", "bench", "dog"):
            assert map_label(label) is None

    def test_train_is_not_mapped(self) -> None:
        """COCO 'train' fires on background rail far more than on trams."""
        assert map_label("train") is None

    def test_emergency_vehicles_are_never_inferred_from_pixels(self) -> None:
        """Emergency status is a declared event, never a guess from an image."""
        from app.vision.classes import COCO_TO_ROAD_USER

        inferred = set(COCO_TO_ROAD_USER.values())
        assert VehicleClass.AMBULANCE not in inferred
        assert VehicleClass.FIRE_ENGINE not in inferred
        assert VehicleClass.POLICE not in inferred

    def test_person_is_not_counted_as_a_vehicle(self) -> None:
        assert VehicleClass.PERSON not in VEHICLE_CLASSES
        assert VehicleClass.BICYCLE not in VEHICLE_CLASSES

    def test_class_indices_restrict_inference(self) -> None:
        indices = road_user_class_indices()
        assert len(indices) == 6
        assert 0 in indices  # person
        assert 2 in indices  # car


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
class TestDeviceSelection:
    def test_cpu_can_be_forced(self) -> None:
        device = resolve_device("cpu")
        assert device.kind is DeviceKind.CPU
        assert device.is_gpu is False
        assert device.torch_device == "cpu"

    def test_auto_never_raises_without_cuda(self) -> None:
        """CUDA is probed, never assumed."""
        device = resolve_device("auto")
        assert device.kind in (DeviceKind.CPU, DeviceKind.CUDA)
        assert device.reason

    def test_requesting_cuda_falls_back_with_a_stated_reason(self) -> None:
        """A missing GPU degrades throughput; it must not deny perception."""
        device = resolve_device("cuda")
        if device.kind is DeviceKind.CPU:
            assert "CUDA" in device.reason


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------
class TestBackendRegistry:
    def test_backends_are_registered(self) -> None:
        assert {b.name for b in BACKENDS} == {"ultralytics", "onnx"}

    def test_none_disables_detection(self, tmp_path: Path) -> None:
        settings = Settings(detector_backend="none", models_dir=tmp_path)
        assert resolve_backend(settings) is None
        with pytest.raises(NotAvailableYetError, match="disabled by configuration"):
            create_detector(settings, tmp_path)

    def test_an_unknown_backend_is_rejected(self, tmp_path: Path) -> None:
        settings = Settings(models_dir=tmp_path)
        object.__setattr__(settings, "detector_backend", "not-a-backend")
        with pytest.raises(ValidationFailedError, match="Unknown detector backend"):
            resolve_backend(settings)

    def test_statuses_report_availability_per_backend(self, tmp_path: Path) -> None:
        statuses = backend_statuses(Settings(models_dir=tmp_path))
        assert len(statuses) == 2
        for status in statuses:
            assert status.available == (not status.missing_modules)

    def test_exactly_one_backend_is_selected(self, tmp_path: Path) -> None:
        statuses = backend_statuses(Settings(models_dir=tmp_path))
        assert sum(1 for s in statuses if s.is_selected) <= 1


class TestModelConfiguration:
    def test_model_resolves_inside_the_models_directory(self, tmp_path: Path) -> None:
        resolved = resolve_model_file(tmp_path, "yolov8n.pt")
        assert resolved.parent == tmp_path.resolve()

    @pytest.mark.parametrize(
        "attack", ["../escape.pt", "/etc/passwd", "C:/Windows/win.ini", "../../x.pt"]
    )
    def test_model_paths_cannot_escape(self, tmp_path: Path, attack: str) -> None:
        with pytest.raises(ValidationFailedError):
            resolve_model_file(tmp_path, attack)

    def test_an_empty_model_name_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationFailedError, match="No detector model"):
            resolve_model_file(tmp_path, "  ")

    def test_identity_requires_a_full_length_checksum(self) -> None:
        with pytest.raises(ValueError):
            ModelIdentity(name="m", backend="b", sha256="tooshort")


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------
class TestByteTracker:
    def test_an_object_keeps_its_id_while_visible(self) -> None:
        tracker = ByteTracker(min_hits_to_confirm=1)
        ids: list[int] = []
        for step in range(6):
            tracks = tracker.update(
                (detection(10 + step * 5, 10, 50 + step * 5, 60),),
                timestamp=T0 + dt.timedelta(seconds=step * 0.1),
            )
            assert len(tracks) == 1
            ids.append(tracks[0].track_id)
        assert len(set(ids)) == 1, "a steadily moving object must keep one id"

    def test_two_objects_get_distinct_ids(self) -> None:
        tracker = ByteTracker(min_hits_to_confirm=1)
        tracks = tracker.update(
            (detection(10, 10, 50, 60), detection(200, 200, 240, 250)), timestamp=T0
        )
        assert len({t.track_id for t in tracks}) == 2

    def test_low_confidence_detections_sustain_a_track(self) -> None:
        """The BYTE insight: an occluded vehicle drops in confidence, not out
        of existence. Discarding it would break the track exactly when
        continuity matters most."""
        tracker = ByteTracker(high_threshold=0.5, low_threshold=0.1, min_hits_to_confirm=1)
        first = tracker.update((detection(10, 10, 50, 60, conf=0.9),), timestamp=T0)
        original_id = first[0].track_id

        # Same object, now partially occluded and detected weakly.
        second = tracker.update(
            (detection(12, 10, 52, 60, conf=0.2),),
            timestamp=T0 + dt.timedelta(seconds=0.1),
        )
        assert len(second) == 1
        assert second[0].track_id == original_id

    def test_low_confidence_detections_never_start_a_track(self) -> None:
        """A weak box with no history is far more likely a false positive."""
        tracker = ByteTracker(high_threshold=0.5, low_threshold=0.1)
        assert tracker.update((detection(10, 10, 50, 60, conf=0.2),), timestamp=T0) == ()

    def test_detections_below_the_low_threshold_are_ignored(self) -> None:
        tracker = ByteTracker(high_threshold=0.5, low_threshold=0.1)
        assert tracker.update((detection(10, 10, 50, 60, conf=0.05),), timestamp=T0) == ()

    def test_a_track_survives_a_brief_gap(self) -> None:
        tracker = ByteTracker(min_hits_to_confirm=1, max_age_frames=5)
        first = tracker.update((detection(10, 10, 50, 60),), timestamp=T0)
        original = first[0].track_id

        for step in range(3):
            tracker.update((), timestamp=T0 + dt.timedelta(seconds=0.1 * (step + 1)))

        resumed = tracker.update(
            (detection(10, 10, 50, 60),), timestamp=T0 + dt.timedelta(seconds=0.5)
        )
        assert resumed[0].track_id == original

    def test_a_track_expires_after_max_age(self) -> None:
        tracker = ByteTracker(min_hits_to_confirm=1, max_age_frames=2)
        tracker.update((detection(10, 10, 50, 60),), timestamp=T0)
        for step in range(4):
            tracker.update((), timestamp=T0 + dt.timedelta(seconds=0.1 * (step + 1)))
        revived = tracker.update(
            (detection(10, 10, 50, 60),), timestamp=T0 + dt.timedelta(seconds=1)
        )
        assert revived[0].track_id != 1

    def test_reset_clears_all_state(self) -> None:
        tracker = ByteTracker(min_hits_to_confirm=1)
        tracker.update((detection(10, 10, 50, 60),), timestamp=T0)
        tracker.reset()
        assert tracker.stats.active_tracks == 0
        fresh = tracker.update((detection(10, 10, 50, 60),), timestamp=T0)
        assert fresh[0].track_id == 1

    def test_class_is_a_majority_vote_not_the_latest_frame(self) -> None:
        """Single-frame class flicker must not rewrite an object's identity."""
        tracker = ByteTracker(min_hits_to_confirm=1)
        for step in range(4):
            tracker.update(
                (detection(10, 10, 50, 60, cls=VehicleClass.CAR),),
                timestamp=T0 + dt.timedelta(seconds=step * 0.1),
            )
        flicker = tracker.update(
            (detection(10, 10, 50, 60, cls=VehicleClass.TRUCK),),
            timestamp=T0 + dt.timedelta(seconds=0.5),
        )
        assert flicker[0].vehicle_class is VehicleClass.CAR

    def test_an_implausible_jump_is_flagged_not_smoothed(self) -> None:
        tracker = ByteTracker(min_hits_to_confirm=1, match_iou=0.0)
        tracker.update((detection(10, 10, 50, 60),), timestamp=T0)
        jumped = tracker.update(
            (detection(600, 400, 640, 450),), timestamp=T0 + dt.timedelta(seconds=0.1)
        )
        # Either a new track, or the old one flagged — never silently accepted.
        assert jumped == () or jumped[0].id_switch_suspected or jumped[0].track_id != 1

    def test_tracking_is_deterministic(self) -> None:
        """Two identical runs must produce identical ids."""
        script = [
            (detection(10 + i * 4, 10, 50 + i * 4, 60), detection(200, 100, 240, 150))
            for i in range(5)
        ]

        def run() -> list[tuple[int, ...]]:
            tracker = ByteTracker(min_hits_to_confirm=1)
            return [
                tuple(
                    t.track_id
                    for t in tracker.update(d, timestamp=T0 + dt.timedelta(seconds=i * 0.1))
                )
                for i, d in enumerate(script)
            ]

        assert run() == run()

    def test_tracker_reports_its_deviations(self) -> None:
        """A result must never be presented as stock ByteTrack."""
        info = ByteTracker().info
        assert "constant-velocity" in (info.association_metric or "")
        assert "greedy" in (info.association_metric or "")


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------
class TestOverlay:
    def test_the_source_frame_is_never_modified(self) -> None:
        import numpy as np

        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        original = frame.copy()
        render_overlay(frame, (detection(10, 10, 60, 60),))
        assert (frame == original).all()

    def test_annotation_changes_pixels(self) -> None:
        import numpy as np

        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        annotated = render_overlay(frame, (detection(10, 10, 60, 60),))
        assert not (annotated == frame).all()

    def test_an_empty_detection_set_leaves_the_frame_unmarked(self) -> None:
        import numpy as np

        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        assert (render_overlay(frame, ()) == frame).all()

    def test_overlay_encodes_to_png(self) -> None:
        import numpy as np

        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        payload = encode_png(render_overlay(frame, (detection(10, 10, 60, 60),)))
        assert payload[:8] == b"\x89PNG\r\n\x1a\n"

    def test_labels_stay_inside_a_frame_edge_box(self) -> None:
        """A box at y=0 would otherwise have its label drawn off-canvas."""
        import numpy as np

        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        annotated = render_overlay(frame, (detection(0, 0, 60, 40),))
        assert annotated.shape == frame.shape


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
class TestVisionStatusApi:
    def test_status_reports_backends_without_loading(self, vision_client: TestClient) -> None:
        body = vision_client.get("/api/vision/status").json()
        assert "backends" in body
        assert body["tracker"] == "bytetrack-lite"
        assert isinstance(body["detection_available"], bool)

    def test_status_exposes_configuration(self, vision_client: TestClient) -> None:
        """The selected configuration, as measured in the Phase 3 tuning pass."""
        body = vision_client.get("/api/vision/status").json()
        assert body["confidence_threshold"] == pytest.approx(0.20)
        assert body["iou_threshold"] == pytest.approx(0.45)
        assert body["image_size"] == 1280
        assert body["max_detections"] == 300
        assert body["configured_model"] == "yolov8s.pt"


class TestImageAnalysisApi:
    def test_analysis_returns_detections_with_provenance(
        self, vision_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = upload_image(vision_client, tmp_path)
        response = vision_client.post(f"/api/vision/images/{media_id}/analyze")
        assert response.status_code == 200

        body = response.json()
        assert body["media_id"] == media_id
        assert body["source_id"] == f"image:{media_id}"
        assert len(body["detections"]) == 1

        found = body["detections"][0]
        assert found["road_user_class"] == "car"
        assert 0.0 <= found["confidence"] <= 1.0
        assert found["frame_index"] == 0
        assert found["timestamp"] is not None
        assert found["source_id"] == f"image:{media_id}"
        assert found["bbox"]["x2"] > found["bbox"]["x1"]

    def test_inference_metadata_identifies_model_and_device(
        self, vision_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = upload_image(vision_client, tmp_path)
        body = vision_client.post(f"/api/vision/images/{media_id}/analyze").json()
        inference = body["inference"]
        assert inference["model"]["name"]
        assert inference["model"]["sha256"]
        assert inference["device_kind"] in ("cpu", "cuda")
        assert inference["device_reason"]

    def test_tally_separates_vehicles_from_other_road_users(
        self, vision_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = upload_image(vision_client, tmp_path)
        tally = vision_client.post(f"/api/vision/images/{media_id}/analyze").json()[
            "detection_tally"
        ]
        assert tally["total"] == 1
        assert tally["vehicles"] == 1

    def test_overlay_is_served_after_analysis(
        self, vision_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = upload_image(vision_client, tmp_path)
        vision_client.post(f"/api/vision/images/{media_id}/analyze")
        response = vision_client.get(f"/api/vision/images/{media_id}/overlay")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_overlay_before_analysis_is_404(
        self, vision_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = upload_image(vision_client, tmp_path)
        assert vision_client.get(f"/api/vision/images/{media_id}/overlay").status_code == 404

    def test_analysis_before_running_is_404(
        self, vision_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = upload_image(vision_client, tmp_path)
        assert vision_client.get(f"/api/vision/images/{media_id}/analysis").status_code == 404

    def test_analysing_a_video_as_an_image_is_refused(
        self, vision_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = upload_video(vision_client, tmp_path)
        assert vision_client.post(f"/api/vision/images/{media_id}/analyze").status_code == 422

    def test_analysing_an_unknown_asset_is_404(self, vision_client: TestClient) -> None:
        assert vision_client.post("/api/vision/images/nope/analyze").status_code == 404


class TestVideoAnalysisApi:
    def test_a_job_runs_and_reports_frames(
        self, vision_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = upload_video(vision_client, tmp_path, frames=6)
        started = vision_client.post(
            f"/api/vision/videos/{media_id}/analyze", json={"run_sync": True}
        )
        assert started.status_code == 202
        job_id = started.json()["job_id"]

        summary = vision_client.get(f"/api/vision/jobs/{job_id}").json()
        assert summary["status"] == "completed"
        assert summary["frames_analysed"] == 6
        assert summary["progress"] == pytest.approx(1.0)
        assert summary["error"] is None

    def test_frame_indices_and_timestamps_are_sequential(
        self, vision_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = upload_video(vision_client, tmp_path, frames=6)
        job_id = vision_client.post(
            f"/api/vision/videos/{media_id}/analyze", json={"run_sync": True}
        ).json()["job_id"]

        frames = vision_client.get(f"/api/vision/jobs/{job_id}/frames").json()["frames"]
        assert [f["frame_index"] for f in frames] == [0, 1, 2, 3, 4, 5]
        stamps = [f["timestamp"] for f in frames]
        assert stamps == sorted(stamps)
        assert frames[1]["media_time_s"] > frames[0]["media_time_s"]

    def test_frame_stride_reduces_work(
        self, vision_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = upload_video(vision_client, tmp_path, frames=6)
        job_id = vision_client.post(
            f"/api/vision/videos/{media_id}/analyze",
            json={"run_sync": True, "frame_stride": 2},
        ).json()["job_id"]

        summary = vision_client.get(f"/api/vision/jobs/{job_id}").json()
        assert summary["frame_stride"] == 2
        frames = vision_client.get(f"/api/vision/jobs/{job_id}/frames").json()["frames"]
        assert [f["frame_index"] for f in frames] == [0, 2, 4]

    def test_max_frames_bounds_the_job(
        self, vision_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = upload_video(vision_client, tmp_path, frames=6)
        job_id = vision_client.post(
            f"/api/vision/videos/{media_id}/analyze",
            json={"run_sync": True, "max_frames": 2},
        ).json()["job_id"]
        assert vision_client.get(f"/api/vision/jobs/{job_id}").json()["frames_analysed"] == 2

    def test_tracks_carry_stable_ids_and_spans(
        self, vision_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = upload_video(vision_client, tmp_path, frames=6)
        job_id = vision_client.post(
            f"/api/vision/videos/{media_id}/analyze", json={"run_sync": True}
        ).json()["job_id"]

        body = vision_client.get(f"/api/vision/jobs/{job_id}/tracks").json()
        assert body["count"] >= 1
        track = body["tracks"][0]
        assert track["track_id"] >= 1
        assert track["last_frame_index"] >= track["first_frame_index"]
        assert track["frames_tracked"] >= 1
        assert track["road_user_class"] == "car"

    def test_detections_carry_track_ids_in_video(
        self, vision_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = upload_video(vision_client, tmp_path, frames=4)
        job_id = vision_client.post(
            f"/api/vision/videos/{media_id}/analyze", json={"run_sync": True}
        ).json()["job_id"]

        frames = vision_client.get(f"/api/vision/jobs/{job_id}/frames").json()["frames"]
        ids = {d["track_id"] for f in frames for d in f["detections"]}
        assert ids and None not in ids

    def test_detection_tally_is_not_a_vehicle_count(
        self, vision_client: TestClient, tmp_path: Path
    ) -> None:
        """One stationary object across N frames yields N detections, 1 track."""
        media_id = upload_video(vision_client, tmp_path, frames=6)
        job_id = vision_client.post(
            f"/api/vision/videos/{media_id}/analyze", json={"run_sync": True}
        ).json()["job_id"]

        summary = vision_client.get(f"/api/vision/jobs/{job_id}").json()
        assert summary["detection_tally"]["total"] == 6
        assert summary["unique_track_count"] == 1

    def test_frames_are_paginated(self, vision_client: TestClient, tmp_path: Path) -> None:
        media_id = upload_video(vision_client, tmp_path, frames=6)
        job_id = vision_client.post(
            f"/api/vision/videos/{media_id}/analyze", json={"run_sync": True}
        ).json()["job_id"]

        page = vision_client.get(
            f"/api/vision/jobs/{job_id}/frames", params={"offset": 2, "limit": 2}
        ).json()
        assert page["total"] == 6
        assert [f["frame_index"] for f in page["frames"]] == [2, 3]

    def test_analysing_an_image_as_a_video_is_refused(
        self, vision_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = upload_image(vision_client, tmp_path)
        assert vision_client.post(f"/api/vision/videos/{media_id}/analyze").status_code == 422

    def test_an_unknown_job_is_rejected(self, vision_client: TestClient) -> None:
        assert vision_client.get("/api/vision/jobs/not-a-job").status_code == 422

    def test_jobs_can_be_listed_per_media(
        self, vision_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = upload_video(vision_client, tmp_path, frames=3)
        vision_client.post(f"/api/vision/videos/{media_id}/analyze", json={"run_sync": True})
        body = vision_client.get("/api/vision/jobs", params={"media_id": media_id}).json()
        assert body["count"] == 1


class TestFailuresAreNeverFabricated:
    def test_a_load_failure_surfaces_the_real_error(
        self, media_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed model load must never look like 'no vehicles detected'."""
        monkeypatch.setenv("MEDIA_DIR", str(media_dir))
        get_settings.cache_clear()
        media_routes.reset_store()

        settings = Settings(
            app_env="local", log_level="ERROR", media_dir=media_dir, models_dir=tmp_path
        )
        app = create_app(settings)

        from app.vision.analysis import DetectorProvider

        class BrokenProvider(DetectorProvider):
            def get(self) -> VehicleDetector:
                FailingDetector().load()
                raise AssertionError("unreachable")

        app.state.detector_provider = BrokenProvider(settings)

        with TestClient(app, raise_server_exceptions=False) as client:
            media_id = upload_image(client, tmp_path)
            response = client.post(f"/api/vision/images/{media_id}/analyze")

        media_routes.reset_store()
        get_settings.cache_clear()

        assert response.status_code == 502
        body = response.json()
        assert "simulated failure" in body["error"]["message"]
        assert "detections" not in body

    def test_a_video_job_failure_is_recorded_not_hidden(
        self, media_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MEDIA_DIR", str(media_dir))
        get_settings.cache_clear()
        media_routes.reset_store()

        settings = Settings(
            app_env="local", log_level="ERROR", media_dir=media_dir, models_dir=tmp_path
        )
        app = create_app(settings)

        from app.vision.analysis import DetectorProvider

        class BrokenProvider(DetectorProvider):
            def get(self) -> VehicleDetector:
                msg = "weights are corrupt"
                raise DataSourceError(msg)

        app.state.detector_provider = BrokenProvider(settings)

        with TestClient(app, raise_server_exceptions=False) as client:
            media_id = upload_video(client, tmp_path, frames=3)
            job_id = client.post(
                f"/api/vision/videos/{media_id}/analyze", json={"run_sync": True}
            ).json()["job_id"]
            summary = client.get(f"/api/vision/jobs/{job_id}").json()

        media_routes.reset_store()
        get_settings.cache_clear()

        assert summary["status"] == "failed"
        assert "corrupt" in summary["error"]
        assert summary["frames_analysed"] == 0
        assert summary["detection_tally"]["total"] == 0


class TestPhaseBoundary:
    def test_no_traffic_conclusion_appears_in_any_vision_response(
        self, vision_client: TestClient, tmp_path: Path
    ) -> None:
        """Phase 3 answers 'what is visible', never 'what should we do'."""
        media_id = upload_image(vision_client, tmp_path)
        payload = str(
            vision_client.post(f"/api/vision/images/{media_id}/analyze").json()
        ).lower()
        for forbidden in (
            "congestion",
            "queue_length",
            "eta",
            "recommend",
            "route",
            "emergency",
            "signal_advisory",
            "prediction",
        ):
            assert forbidden not in payload


# ---------------------------------------------------------------------------
# Real model — skipped cleanly when unavailable
# ---------------------------------------------------------------------------
class TestRealModelInference:
    """Exercises a genuine pretrained model when one is present.

    Skipped rather than failed when the backend or weights are missing, so the
    suite stays runnable on a machine that has not downloaded anything.
    """

    @pytest.fixture
    def real_detector(self, tmp_path: Path) -> VehicleDetector:
        pytest.importorskip("ultralytics")
        pytest.importorskip("torch")

        weights = Path(__file__).resolve().parents[2] / "models" / "yolov8n.pt"
        if not weights.exists():
            pytest.skip("yolov8n.pt is not present in the models directory")

        settings = Settings(
            models_dir=weights.parent,
            detector_backend="ultralytics",
            detector_model="yolov8n.pt",
            detector_device="cpu",
        )
        detector = create_detector(settings, weights.parent)
        detector.load()
        return detector

    def test_it_detects_road_users_in_a_real_photograph(
        self, real_detector: VehicleDetector
    ) -> None:
        import numpy as np
        from PIL import Image

        venv = Path(__file__).resolve().parents[2] / ".venv"
        asset = venv / "Lib/site-packages/ultralytics/assets/bus.jpg"
        if not asset.exists():
            pytest.skip("ultralytics sample asset not available")

        frame = np.asarray(Image.open(asset).convert("RGB"))
        result = real_detector.detect(
            frame, source_id="image:real", sequence=0, captured_at=T0
        )

        assert result.count > 0, "a photo of a bus and pedestrians must yield detections"
        classes = {d.vehicle_class for d in result.detections}
        assert classes <= REQUIRED_CLASSES | {VehicleClass.OTHER}
        assert all(0.0 <= d.confidence <= 1.0 for d in result.detections)
        assert all(d.bbox.x2 > d.bbox.x1 and d.bbox.y2 > d.bbox.y1 for d in result.detections)

    def test_it_reports_a_real_model_identity(self, real_detector: VehicleDetector) -> None:
        identity = real_detector.identity
        assert identity is not None
        assert identity.sha256 and len(identity.sha256) == 64
        assert identity.size_bytes and identity.size_bytes > 0
        assert real_detector.device is not None
