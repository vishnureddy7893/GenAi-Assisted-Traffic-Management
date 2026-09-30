"""Regression tests for the Phase 3 detector configuration.

Three separate claims are locked down here, because they fail differently:

* **The frame reaches the model in the right channel order.** The ingestion
  layer yields RGB; the ultralytics Python API documents raw numpy input as BGR
  and flips it itself. Feeding it RGB delivered a channel-swapped image to the
  network -- which degraded recall rather than failing, and so went unnoticed.
* **The selected configuration is the deployed one.** Model, confidence, IoU,
  input size and detection ceiling were chosen against measured counts on a real
  junction frame; a silent revert to the previous defaults would cost most of
  the recall back.
* **The four knobs are reachable from the API.** A threshold that can only be
  changed by editing an environment file and restarting cannot be tuned against
  the camera it is wrong for.

The measured evidence behind the numbers is recorded in
``docs/adr/0012-detector-configuration.md``. The reference-image test at the end
re-measures it, and skips cleanly when the weights or the image are absent.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from fastapi.testclient import TestClient

from app.api.routes import media as media_routes
from app.core.config import Settings, get_settings
from app.domain.enums import VEHICLE_CLASSES
from app.main import create_app
from app.vision.analysis import DetectorOverrides, DetectorProvider
from app.vision.backends.ultralytics_detector import (
    UltralyticsYoloDetector,
    _as_ultralytics_input,
)
from app.vision.registry import backend_by_name, create_detector

pytest.importorskip("PIL")
pytest.importorskip("numpy")

import numpy as np
from PIL import Image

T0 = dt.datetime(2024, 3, 14, 8, 30, tzinfo=dt.UTC)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The junction frame the configuration was selected against: an oblique view
#: of a busy crossing with vehicles at every scale, from a lorry filling a
#: hundred pixels to parked cars far up the avenue. Not committed -- it is an
#: ingested media asset -- so every test using it skips when it is absent.
REFERENCE_IMAGE = (
    REPO_ROOT / "datasets" / "media" / "50e429061f274f138ad8d34a75fd32ec.webp"
)

#: The configuration selected in the Phase 3 tuning pass.
SELECTED_MODEL = "yolov8s.pt"
SELECTED_CONFIDENCE = 0.20
SELECTED_IOU = 0.45
SELECTED_IMAGE_SIZE = 1280
SELECTED_MAX_DETECTIONS = 300

#: What the previous configuration produced on the reference frame, and what the
#: selected one produced. Asserted as a floor rather than an equality: a newer
#: ultralytics release may shift a box or two, and a test that fails on that
#: would be measuring the library rather than this project's configuration.
PREVIOUS_VEHICLE_COUNT = 8
SELECTED_VEHICLE_FLOOR = 35


# ---------------------------------------------------------------------------
# Channel order
# ---------------------------------------------------------------------------
class TestUltralyticsChannelOrder:
    """The ingestion layer speaks RGB; this backend must hand over BGR."""

    def test_channels_are_reversed_for_an_image_array(self) -> None:
        frame = np.zeros((4, 5, 3), dtype=np.uint8)
        frame[..., 0] = 10  # R
        frame[..., 1] = 20  # G
        frame[..., 2] = 30  # B

        converted = _as_ultralytics_input(frame)

        assert converted[..., 0].max() == 30, "blue must arrive in the first channel"
        assert converted[..., 1].max() == 20
        assert converted[..., 2].max() == 10

    def test_the_conversion_is_contiguous(self) -> None:
        """A reversed slice is a negative-stride view, which torch rejects."""
        converted = _as_ultralytics_input(np.zeros((4, 5, 3), dtype=np.uint8))
        assert converted.flags["C_CONTIGUOUS"]

    def test_the_source_frame_is_not_modified(self) -> None:
        frame = np.zeros((4, 5, 3), dtype=np.uint8)
        frame[..., 0] = 10
        _as_ultralytics_input(frame)
        assert frame[..., 0].max() == 10, "the caller's frame is not ours to swap"

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            "path/to/image.jpg",
            pytest.param(np.zeros((4, 5), dtype=np.uint8), id="greyscale"),
            pytest.param(np.zeros((2, 4, 5, 3), dtype=np.uint8), id="batched"),
        ],
    )
    def test_non_image_payloads_pass_through_untouched(self, payload: Any) -> None:
        """A PIL image or a path is already handled correctly by the library."""
        assert _as_ultralytics_input(payload) is payload

    def test_the_detector_hands_bgr_to_the_model(self) -> None:
        """End to end: what reaches ``predict`` is the reverse of what came in."""
        captured: dict[str, Any] = {}

        class FakeBoxes:
            def __iter__(self) -> Iterator[Any]:
                return iter(())

        class FakeResult:
            boxes = FakeBoxes()

        class FakeModel:
            names: ClassVar[dict[int, str]] = {2: "car"}

            def predict(self, frame: Any, **kwargs: Any) -> list[FakeResult]:
                captured["frame"] = frame
                captured["kwargs"] = kwargs
                return [FakeResult()]

        detector = UltralyticsYoloDetector(models_dir=Path("."), model_name="fake.pt")
        detector._model = FakeModel()

        frame = np.zeros((6, 7, 3), dtype=np.uint8)
        frame[..., 0] = 200  # a red frame in RGB
        detector.detect(frame, source_id="s", sequence=0, captured_at=T0)

        handed = captured["frame"]
        assert handed[..., 2].max() == 200, "red must arrive in the last channel"
        assert handed[..., 0].max() == 0


# ---------------------------------------------------------------------------
# The selected configuration
# ---------------------------------------------------------------------------
class TestSelectedConfiguration:
    def test_the_defaults_are_the_selected_configuration(self) -> None:
        settings = Settings(app_env="local")
        assert settings.detector_model == SELECTED_MODEL
        assert settings.detector_confidence_threshold == pytest.approx(
            SELECTED_CONFIDENCE
        )
        assert settings.detector_iou_threshold == pytest.approx(SELECTED_IOU)
        assert settings.detector_image_size == SELECTED_IMAGE_SIZE
        assert settings.detector_max_detections == SELECTED_MAX_DETECTIONS

    def test_the_input_size_is_large_enough_for_distant_vehicles(self) -> None:
        """640 was the measured failure: it loses the far half of a junction."""
        assert Settings(app_env="local").detector_image_size >= 1280

    def test_the_registry_default_matches_the_configured_model(self) -> None:
        """Status shows the backend default; disagreeing with config misleads."""
        spec = backend_by_name("ultralytics")
        assert spec is not None
        assert spec.default_model == SELECTED_MODEL

    def test_every_knob_reaches_the_constructed_detector(self, tmp_path: Path) -> None:
        pytest.importorskip("ultralytics")
        pytest.importorskip("torch")

        settings = Settings(
            app_env="local",
            models_dir=tmp_path,
            detector_backend="ultralytics",
            detector_confidence_threshold=0.33,
            detector_iou_threshold=0.55,
            detector_image_size=960,
            detector_max_detections=77,
        )
        detector = create_detector(settings, tmp_path)
        assert isinstance(detector, UltralyticsYoloDetector)
        assert detector._confidence == pytest.approx(0.33)
        assert detector._iou == pytest.approx(0.55)
        assert detector._image_size == 960
        assert detector._max_detections == 77

    def test_the_thresholds_are_passed_to_inference(self) -> None:
        """Configured once, applied at the model rather than filtered after."""
        captured: dict[str, Any] = {}

        class FakeModel:
            names: ClassVar[dict[int, str]] = {2: "car"}

            def predict(self, frame: Any, **kwargs: Any) -> list[Any]:
                captured.update(kwargs)
                return []

        detector = UltralyticsYoloDetector(
            models_dir=Path("."),
            model_name="fake.pt",
            confidence_threshold=0.2,
            iou_threshold=0.45,
            image_size=1280,
            max_detections=300,
        )
        detector._model = FakeModel()
        detector.detect(
            np.zeros((4, 4, 3), dtype=np.uint8),
            source_id="s",
            sequence=0,
            captured_at=T0,
        )

        assert captured["conf"] == pytest.approx(0.2)
        assert captured["iou"] == pytest.approx(0.45)
        assert captured["imgsz"] == 1280
        assert captured["max_det"] == 300


# ---------------------------------------------------------------------------
# Per-request overrides
# ---------------------------------------------------------------------------
class TestDetectorOverrides:
    def test_an_empty_override_leaves_settings_identical(self) -> None:
        settings = Settings(app_env="local")
        assert DetectorOverrides().is_empty
        assert DetectorOverrides().apply_to(settings) is settings

    def test_an_override_copies_rather_than_mutates(self) -> None:
        """One caller's experiment must not become another's configuration."""
        settings = Settings(app_env="local")
        applied = DetectorOverrides(confidence_threshold=0.15, image_size=640).apply_to(
            settings
        )

        assert applied.detector_confidence_threshold == pytest.approx(0.15)
        assert applied.detector_image_size == 640
        assert settings.detector_confidence_threshold == pytest.approx(
            SELECTED_CONFIDENCE
        )
        assert settings.detector_image_size == SELECTED_IMAGE_SIZE

    def test_a_partial_override_leaves_the_rest_configured(self) -> None:
        settings = Settings(app_env="local")
        applied = DetectorOverrides(confidence_threshold=0.15).apply_to(settings)
        assert applied.detector_image_size == SELECTED_IMAGE_SIZE
        assert applied.detector_iou_threshold == pytest.approx(SELECTED_IOU)
        assert applied.detector_max_detections == SELECTED_MAX_DETECTIONS

    def test_every_documented_knob_is_overridable(self) -> None:
        settings = Settings(app_env="local")
        applied = DetectorOverrides(
            confidence_threshold=0.15,
            iou_threshold=0.6,
            image_size=640,
            max_detections=50,
        ).apply_to(settings)

        assert applied.detector_confidence_threshold == pytest.approx(0.15)
        assert applied.detector_iou_threshold == pytest.approx(0.6)
        assert applied.detector_image_size == 640
        assert applied.detector_max_detections == 50

    def test_no_override_uses_the_cached_detector(self) -> None:
        """The common path must not pay a model load."""
        settings = Settings(app_env="local")
        provider = DetectorProvider(settings)
        sentinel = object()
        provider._detector = sentinel  # type: ignore[assignment]

        with provider.with_overrides(None) as (detector, effective):
            assert detector is sentinel
            assert effective is settings

        with provider.with_overrides(DetectorOverrides()) as (detector, effective):
            assert detector is sentinel
            assert effective is settings


# ---------------------------------------------------------------------------
# The API surface
# ---------------------------------------------------------------------------
@pytest.fixture
def tuning_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A client whose provider records the overrides it is asked for."""
    media_dir = tmp_path / "media"
    media_dir.mkdir()
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

    from test_vision import ScriptedDetector, detection

    from app.domain.enums import VehicleClass

    class RecordingProvider(DetectorProvider):
        """Yields a scripted detector while recording the effective settings."""

        def __init__(self) -> None:
            super().__init__(settings)
            self.seen: list[Settings] = []
            self._detector = ScriptedDetector(
                [(detection(10, 10, 50, 60, cls=VehicleClass.CAR, conf=0.9),)] * 20
            )
            self._detector.load()

        def get(self) -> Any:
            return self._detector

        def with_overrides(self, overrides: DetectorOverrides | None) -> Any:
            from contextlib import contextmanager

            @contextmanager
            def _run() -> Iterator[tuple[Any, Settings]]:
                effective = (
                    overrides.apply_to(settings) if overrides is not None else settings
                )
                self.seen.append(effective)
                yield self._detector, effective

            return _run()

    provider = RecordingProvider()
    app.state.detector_provider = provider

    with TestClient(app, raise_server_exceptions=False) as client:
        client.provider = provider  # type: ignore[attr-defined]
        yield client

    media_routes.reset_store()
    get_settings.cache_clear()


def _upload(client: TestClient, tmp_path: Path) -> str:
    """Upload a small generated image and return its media id."""
    path = tmp_path / "frame.png"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(path)
    return client.post(
        "/api/media/images",
        files={"file": ("frame.png", path.read_bytes(), "image/png")},
    ).json()["media_id"]


class TestAnalyzeImageOverrides:
    def test_posting_no_body_uses_the_configured_settings(
        self, tuning_client: TestClient, tmp_path: Path
    ) -> None:
        """The pre-existing call shape must keep working unchanged."""
        media_id = _upload(tuning_client, tmp_path)
        response = tuning_client.post(f"/api/vision/images/{media_id}/analyze")

        assert response.status_code == 200
        inference = response.json()["inference"]
        assert inference["confidence_threshold"] == pytest.approx(SELECTED_CONFIDENCE)
        assert inference["iou_threshold"] == pytest.approx(SELECTED_IOU)

        effective = tuning_client.provider.seen[-1]  # type: ignore[attr-defined]
        assert effective.detector_image_size == SELECTED_IMAGE_SIZE
        assert effective.detector_max_detections == SELECTED_MAX_DETECTIONS

    def test_overrides_are_applied_and_reported_back(
        self, tuning_client: TestClient, tmp_path: Path
    ) -> None:
        """A count is never separable from the settings that produced it."""
        media_id = _upload(tuning_client, tmp_path)
        response = tuning_client.post(
            f"/api/vision/images/{media_id}/analyze",
            json={"confidence_threshold": 0.15, "image_size": 640},
        )

        assert response.status_code == 200
        inference = response.json()["inference"]
        assert inference["confidence_threshold"] == pytest.approx(0.15)

        # The scripted detector reports its own input size, so the settings the
        # request actually ran under are the thing to assert on. A real detector
        # derives info.input_width from exactly these settings.
        effective = tuning_client.provider.seen[-1]  # type: ignore[attr-defined]
        assert effective.detector_confidence_threshold == pytest.approx(0.15)
        assert effective.detector_image_size == 640

    def test_an_override_does_not_leak_into_the_next_request(
        self, tuning_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = _upload(tuning_client, tmp_path)
        tuning_client.post(
            f"/api/vision/images/{media_id}/analyze",
            json={"confidence_threshold": 0.15},
        )
        body = tuning_client.post(f"/api/vision/images/{media_id}/analyze").json()
        assert body["inference"]["confidence_threshold"] == pytest.approx(
            SELECTED_CONFIDENCE
        )

    @pytest.mark.parametrize(
        "payload",
        [
            {"confidence_threshold": 1.5},
            {"confidence_threshold": -0.1},
            {"iou_threshold": 2.0},
            {"image_size": 32},
            {"image_size": 99999},
            {"max_detections": 0},
        ],
    )
    def test_an_out_of_range_override_is_refused(
        self, tuning_client: TestClient, tmp_path: Path, payload: dict
    ) -> None:
        media_id = _upload(tuning_client, tmp_path)
        response = tuning_client.post(
            f"/api/vision/images/{media_id}/analyze", json=payload
        )
        assert response.status_code == 422

    def test_status_publishes_every_tunable_knob(
        self, tuning_client: TestClient
    ) -> None:
        """All four must be readable, or a caller cannot know what to change."""
        body = tuning_client.get("/api/vision/status").json()
        for field in (
            "confidence_threshold",
            "iou_threshold",
            "image_size",
            "max_detections",
        ):
            assert field in body, f"{field} is not published by /api/vision/status"


# ---------------------------------------------------------------------------
# The measurement the configuration rests on
# ---------------------------------------------------------------------------
class TestReferenceImageRecall:
    """Re-measures the selection on the frame it was made against.

    Skipped when the weights or the image are absent, so the suite stays
    runnable on a machine that has downloaded neither.
    """

    @pytest.fixture
    def frame(self) -> Any:
        if not REFERENCE_IMAGE.exists():
            pytest.skip(f"reference image {REFERENCE_IMAGE.name} is not present")
        return np.asarray(Image.open(REFERENCE_IMAGE).convert("RGB"))

    def _detector(self, **kwargs: Any) -> UltralyticsYoloDetector:
        pytest.importorskip("ultralytics")
        pytest.importorskip("torch")
        weights = REPO_ROOT / "models" / kwargs.get("model_name", SELECTED_MODEL)
        if not weights.exists():
            pytest.skip(f"{weights.name} is not present in the models directory")

        detector = UltralyticsYoloDetector(
            models_dir=weights.parent, device_preference="cpu", **kwargs
        )
        detector.load()
        return detector

    @staticmethod
    def _vehicles(result: Any) -> int:
        return sum(1 for d in result.detections if d.vehicle_class in VEHICLE_CLASSES)

    def test_the_selected_configuration_finds_the_traffic(self, frame: Any) -> None:
        """The whole point of the tuning pass, as one number."""
        detector = self._detector(
            model_name=SELECTED_MODEL,
            confidence_threshold=SELECTED_CONFIDENCE,
            image_size=SELECTED_IMAGE_SIZE,
        )
        try:
            result = detector.detect(
                frame, source_id="ref", sequence=0, captured_at=T0
            )
        finally:
            detector.close()

        vehicles = self._vehicles(result)
        assert vehicles >= SELECTED_VEHICLE_FLOOR, (
            f"the selected configuration found {vehicles} vehicles on the reference "
            f"frame, below the {SELECTED_VEHICLE_FLOOR} measured when it was chosen"
        )
        assert vehicles > PREVIOUS_VEHICLE_COUNT * 3

    def test_a_larger_input_size_finds_more_than_640(self, frame: Any) -> None:
        """The setting that mattered most, held against a regression to 640."""
        small = self._detector(
            model_name=SELECTED_MODEL,
            confidence_threshold=SELECTED_CONFIDENCE,
            image_size=640,
        )
        try:
            at_640 = self._vehicles(
                small.detect(frame, source_id="ref", sequence=0, captured_at=T0)
            )
        finally:
            small.close()

        large = self._detector(
            model_name=SELECTED_MODEL,
            confidence_threshold=SELECTED_CONFIDENCE,
            image_size=SELECTED_IMAGE_SIZE,
        )
        try:
            at_1280 = self._vehicles(
                large.detect(frame, source_id="ref", sequence=0, captured_at=T0)
            )
        finally:
            large.close()

        assert at_1280 > at_640, (
            f"1280 px found {at_1280} vehicles against {at_640} at 640 px; if this "
            "reverses, the input-size trade-off needs re-measuring"
        )

    def test_correcting_the_channel_order_does_not_lose_detections(
        self, frame: Any
    ) -> None:
        """The fix must be a fix: the corrected order is the library's own.

        Handing ultralytics a PIL image is the path it converts correctly. A
        numpy frame through our backend must agree with it -- that equality is
        what makes the channel swap a defect rather than a preference.
        """
        detector = self._detector(
            model_name=SELECTED_MODEL,
            confidence_threshold=SELECTED_CONFIDENCE,
            image_size=SELECTED_IMAGE_SIZE,
        )
        try:
            through_backend = detector.detect(
                frame, source_id="ref", sequence=0, captured_at=T0
            )
            reference = detector._model.predict(
                Image.fromarray(frame),
                conf=SELECTED_CONFIDENCE,
                iou=SELECTED_IOU,
                imgsz=SELECTED_IMAGE_SIZE,
                max_det=SELECTED_MAX_DETECTIONS,
                classes=list(detector._class_indices) or None,
                device="cpu",
                verbose=False,
            )
        finally:
            detector.close()

        assert through_backend.count == len(reference[0].boxes), (
            "the backend disagrees with the library's own PIL path, which means "
            "the frame is reaching the model in the wrong channel order"
        )
