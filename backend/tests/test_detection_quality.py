"""Regression tests for detection quality filtering.

What these lock down, in order of how expensive the failure would be:

* **The downstream pipeline counts confirmed detections only.** A duplicate box
  or a building detected as a bus reaching the analytics layer becomes traffic
  that is not there, and every figure above it -- vehicle count, density,
  congestion band, route cost -- inherits the error silently.
* **Nothing is deleted.** A filtered-away box is invisible to an operator trying
  to work out why a count is wrong, so every detection is still reported with
  its verdict and the reason.
* **No rule is arbitrary.** In particular there is no "small boxes are not
  vehicles" rule, and there are no image coordinates anywhere: a filter tuned to
  one photograph is not a filter.

The reference-image tests at the end re-measure the behaviour on the frame the
policy was chosen against, and skip when the weights or the image are absent.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.routes import media as media_routes
from app.core.config import Settings, get_settings
from app.domain.enums import VEHICLE_CLASSES, VehicleClass
from app.main import create_app
from app.vision.detector import BoundingBox, Detection
from app.vision.quality import (
    AssessedDetection,
    DetectionQuality,
    QualityFlag,
    assess,
    confirmed,
    policy_from,
)

pytest.importorskip("PIL")
pytest.importorskip("numpy")

import numpy as np
from PIL import Image

T0 = dt.datetime(2024, 3, 14, 8, 30, tzinfo=dt.UTC)
REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_IMAGE = (
    REPO_ROOT / "datasets" / "media" / "50e429061f274f138ad8d34a75fd32ec.webp"
)

#: A frame large enough that ordinary test boxes sit well under the area
#: ceiling, so a test about confidence is not silently a test about area.
FRAME_W, FRAME_H = 800, 600


def box(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def det(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    cls: VehicleClass = VehicleClass.CAR,
    conf: float = 0.9,
) -> Detection:
    return Detection(bbox=box(x1, y1, x2, y2), vehicle_class=cls, confidence=conf)


def run(
    detections: tuple[Detection, ...],
    *,
    width: int = FRAME_W,
    height: int = FRAME_H,
    **overrides: Any,
) -> tuple[AssessedDetection, ...]:
    """Assess detections under settings-derived policy, with overrides."""
    settings = Settings(app_env="local", **overrides)
    return assess(
        detections,
        frame_width=width,
        frame_height=height,
        policy=policy_from(settings),
    )


# ---------------------------------------------------------------------------
# Class-specific confirmation floors
# ---------------------------------------------------------------------------
class TestClassConfidenceFloors:
    def test_a_car_below_its_floor_is_provisional_not_deleted(self) -> None:
        assessed = run((det(10, 10, 50, 40, conf=0.37),))
        assert len(assessed) == 1, "the detection is still reported"
        assert assessed[0].quality is DetectionQuality.PROVISIONAL
        assert QualityFlag.BELOW_CLASS_CONFIDENCE in assessed[0].flags
        assert "0.40 confirmation floor" in (assessed[0].reason or "")

    def test_a_car_at_its_floor_is_confirmed(self) -> None:
        """The floor is inclusive; a value exactly on it passes."""
        assessed = run((det(10, 10, 50, 40, conf=0.40),))
        assert assessed[0].quality is DetectionQuality.CONFIRMED
        assert assessed[0].flags == ()
        assert assessed[0].reason is None

    def test_floors_are_independent_per_class(self) -> None:
        """One global number cannot express unequal class reliability."""
        assessed = run(
            (
                det(10, 10, 50, 40, cls=VehicleClass.CAR, conf=0.38),
                det(100, 10, 140, 40, cls=VehicleClass.TRUCK, conf=0.38),
                det(200, 10, 240, 40, cls=VehicleClass.BUS, conf=0.38),
            )
        )
        assert assessed[0].quality is DetectionQuality.PROVISIONAL, "car floor 0.40"
        assert assessed[1].quality is DetectionQuality.CONFIRMED, "truck floor 0.35"
        assert assessed[2].quality is DetectionQuality.PROVISIONAL, "bus floor 0.40"

    def test_a_class_without_evidence_falls_back_to_the_detection_threshold(
        self,
    ) -> None:
        """No floor is invented for a class nothing was measured on."""
        settings = Settings(app_env="local")
        policy = policy_from(settings)
        assert policy.floor_for(VehicleClass.MOTORCYCLE) == pytest.approx(
            settings.detector_confidence_threshold
        )
        floors = {f.road_user_class: f for f in policy.floors()}
        assert floors[VehicleClass.MOTORCYCLE].is_class_specific is False
        assert floors[VehicleClass.CAR].is_class_specific is True

    def test_the_floor_is_configurable(self) -> None:
        """An operator preferring recall lowers one number."""
        detection = (det(10, 10, 50, 40, conf=0.30),)
        assert run(detection)[0].quality is DetectionQuality.PROVISIONAL
        assert (
            run(detection, detection_confirm_confidence_car=0.25)[0].quality
            is DetectionQuality.CONFIRMED
        )


# ---------------------------------------------------------------------------
# One object, one detection
# ---------------------------------------------------------------------------
class TestDuplicateSuppression:
    def test_an_overlapping_pair_keeps_the_more_confident_box(self) -> None:
        assessed = run(
            (
                det(100, 100, 200, 180, conf=0.50),
                det(104, 104, 204, 184, conf=0.90),
            )
        )
        assert assessed[1].quality is DetectionQuality.CONFIRMED
        assert assessed[0].quality is DetectionQuality.PROVISIONAL
        assert QualityFlag.DUPLICATE in assessed[0].flags
        assert assessed[0].duplicate_of_index == 1

    def test_a_cross_class_duplicate_is_caught(self) -> None:
        """Per-class NMS inside the model cannot see this pair at all."""
        assessed = run(
            (
                det(100, 100, 200, 180, cls=VehicleClass.CAR, conf=0.80),
                det(101, 101, 201, 181, cls=VehicleClass.TRUCK, conf=0.60),
            )
        )
        assert assessed[0].quality is DetectionQuality.CONFIRMED
        assert assessed[1].quality is DetectionQuality.PROVISIONAL
        assert QualityFlag.DUPLICATE in assessed[1].flags
        assert "car at 0.80" in (assessed[1].reason or "")

    def test_a_nested_box_is_caught_though_its_iou_is_low(self) -> None:
        """A small box inside a large one is still a second count."""
        outer = det(100, 100, 220, 200, conf=0.80)
        inner = det(140, 130, 180, 170, conf=0.60)
        assert outer.bbox.iou(inner.bbox) < 0.55, "IoU alone would miss this"

        assessed = run((outer, inner))
        assert assessed[0].quality is DetectionQuality.CONFIRMED
        assert assessed[1].quality is DetectionQuality.PROVISIONAL
        assert QualityFlag.DUPLICATE in assessed[1].flags

    def test_neighbouring_vehicles_are_not_duplicates(self) -> None:
        """Traffic is dense; adjacent boxes must survive."""
        assessed = run(
            (
                det(100, 100, 160, 150, conf=0.80),
                det(165, 100, 225, 150, conf=0.75),
            )
        )
        assert all(a.quality is DetectionQuality.CONFIRMED for a in assessed)

    def test_suppression_is_deterministic_under_reordering(self) -> None:
        """The survivor must not depend on detector output order."""
        high = det(100, 100, 200, 180, conf=0.90)
        low = det(104, 104, 204, 184, conf=0.50)

        forward = run((high, low))
        backward = run((low, high))
        assert forward[0].quality is DetectionQuality.CONFIRMED
        assert forward[1].quality is DetectionQuality.PROVISIONAL
        assert backward[1].quality is DetectionQuality.CONFIRMED
        assert backward[0].quality is DetectionQuality.PROVISIONAL

    def test_equal_confidence_ties_break_on_input_order(self) -> None:
        assessed = run(
            (
                det(100, 100, 200, 180, conf=0.70),
                det(104, 104, 204, 184, conf=0.70),
            )
        )
        assert assessed[0].quality is DetectionQuality.CONFIRMED
        assert assessed[1].quality is DetectionQuality.PROVISIONAL

    def test_suppression_can_be_disabled(self) -> None:
        assessed = run(
            (
                det(100, 100, 200, 180, conf=0.90),
                det(104, 104, 204, 184, conf=0.85),
            ),
            detection_duplicate_iou=0.0,
            detection_duplicate_containment=0.0,
        )
        assert all(a.quality is DetectionQuality.CONFIRMED for a in assessed)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
class TestGeometryPlausibility:
    def test_a_box_covering_too_much_of_the_frame_is_rejected(self) -> None:
        """A building detected as a bus is not a marginal call."""
        assessed = run((det(0, 0, 400, 400, cls=VehicleClass.BUS, conf=0.90),))
        assert assessed[0].quality is DetectionQuality.REJECTED
        assert QualityFlag.IMPLAUSIBLE_AREA in assessed[0].flags
        assert "% of the frame" in (assessed[0].reason or "")

    def test_a_high_confidence_score_does_not_rescue_an_impossible_box(self) -> None:
        """Geometry beats confidence: physics is not a matter of degree."""
        assessed = run((det(0, 0, 400, 400, conf=0.99),))
        assert assessed[0].quality is DetectionQuality.REJECTED

    def test_the_area_ceiling_is_configurable_and_disablable(self) -> None:
        """A close-range camera legitimately sees a bus fill the view."""
        big = (det(0, 0, 400, 400, conf=0.90),)
        assert run(big, detection_max_area_fraction=0.0)[0].quality is (
            DetectionQuality.CONFIRMED
        )
        assert run(big, detection_max_area_fraction=0.9)[0].quality is (
            DetectionQuality.CONFIRMED
        )

    @pytest.mark.parametrize(
        ("x2", "y2", "label"),
        [(400, 105, "sliver"), (105, 400, "column")],
    )
    def test_degenerate_proportions_are_rejected(
        self, x2: float, y2: float, label: str
    ) -> None:
        assessed = run((det(100, 100, x2, y2, conf=0.90),))
        assert assessed[0].quality is DetectionQuality.REJECTED, label
        assert QualityFlag.IMPLAUSIBLE_ASPECT in assessed[0].flags

    def test_the_default_aspect_band_excludes_nothing_realistic(self) -> None:
        """A guard, not a contributor. Real vehicles span roughly 0.6 to 1.9."""
        assessed = run(
            (
                det(100, 100, 160, 130, conf=0.90),  # 2.00 wide
                det(200, 100, 230, 160, conf=0.90),  # 0.50 tall
            )
        )
        assert all(a.quality is DetectionQuality.CONFIRMED for a in assessed)

    def test_geometry_is_reported_whatever_the_verdict(self) -> None:
        assessed = run((det(100, 100, 200, 180, conf=0.90),))
        assert assessed[0].area_fraction == pytest.approx(
            (100 * 80) / (FRAME_W * FRAME_H), abs=1e-6
        )
        assert assessed[0].aspect_ratio == pytest.approx(1.25)

    def test_a_frame_without_dimensions_skips_the_scene_scale_rules(self) -> None:
        """No frame size, no area rule -- rather than one against a guess."""
        assessed = assess(
            (det(0, 0, 400, 400, conf=0.90),),
            frame_width=0,
            frame_height=0,
            policy=policy_from(Settings(app_env="local")),
        )
        assert assessed[0].quality is DetectionQuality.CONFIRMED
        assert assessed[0].area_fraction == 0.0


class TestNoArbitrarySizeRule:
    """The rule that was deliberately not written."""

    def test_a_tiny_confident_box_is_confirmed(self) -> None:
        """Distant vehicles are small. Removing small boxes removes traffic."""
        assessed = run((det(400, 300, 412, 309, conf=0.85),))
        assert assessed[0].quality is DetectionQuality.CONFIRMED
        assert assessed[0].flags == ()

    def test_a_tiny_box_is_judged_only_on_confidence(self) -> None:
        small_low = run((det(400, 300, 408, 307, conf=0.25),))[0]
        assert small_low.quality is DetectionQuality.PROVISIONAL
        assert small_low.flags == (QualityFlag.BELOW_CLASS_CONFIDENCE,), (
            "size must not appear among the reasons"
        )

    def test_the_module_contains_no_image_coordinates(self) -> None:
        """A filter tuned to one photograph is not a filter."""
        source = (
            Path(__file__).resolve().parents[1] / "app" / "vision" / "quality.py"
        ).read_text(encoding="utf-8")
        for forbidden in ("261", "302", "362", "392", "region_of_interest", "exclude_zone"):
            assert forbidden not in source, (
                f"{forbidden!r} looks like a coordinate or a per-image exception"
            )


# ---------------------------------------------------------------------------
# Frame boundary
# ---------------------------------------------------------------------------
class TestFrameEdgeBehaviour:
    def test_a_truncated_box_needs_more_confidence(self) -> None:
        assessed = run(
            (det(0, 200, 30, 240, conf=0.42),), detection_confirm_confidence_car=0.40
        )
        assert assessed[0].touches_frame_edge is True
        assert assessed[0].quality is DetectionQuality.PROVISIONAL
        assert QualityFlag.EDGE_TRUNCATED_LOW_CONFIDENCE in assessed[0].flags

    def test_a_confident_vehicle_entering_the_frame_is_confirmed(self) -> None:
        """A real vehicle at the edge must not be discarded for being there."""
        assessed = run((det(0, 200, 60, 250, conf=0.85),))
        assert assessed[0].touches_frame_edge is True
        assert assessed[0].quality is DetectionQuality.CONFIRMED

    def test_an_interior_box_is_not_flagged(self) -> None:
        assessed = run((det(100, 100, 160, 140, conf=0.42),))
        assert assessed[0].touches_frame_edge is False
        assert assessed[0].quality is DetectionQuality.CONFIRMED

    @pytest.mark.parametrize(
        "coords",
        [(0, 200, 40, 240), (760, 200, 800, 240), (100, 0, 140, 40), (100, 560, 140, 600)],
    )
    def test_every_border_is_detected(self, coords: tuple[int, ...]) -> None:
        assessed = run((det(*coords, conf=0.85),))
        assert assessed[0].touches_frame_edge is True

    def test_the_edge_margin_can_be_disabled(self) -> None:
        assessed = run(
            (det(0, 200, 30, 240, conf=0.42),), detection_edge_confidence_margin=0.0
        )
        assert assessed[0].quality is DetectionQuality.CONFIRMED


# ---------------------------------------------------------------------------
# The layer as a whole
# ---------------------------------------------------------------------------
class TestPolicyBehaviour:
    def test_disabling_the_layer_confirms_everything(self) -> None:
        """So its effect can be measured by turning it off."""
        detections = (
            det(0, 0, 400, 400, conf=0.90),
            det(10, 10, 50, 40, conf=0.21),
        )
        assessed = run(detections, detection_quality_enabled=False)
        assert all(a.quality is DetectionQuality.CONFIRMED for a in assessed)
        assert all(a.flags == () for a in assessed)

    def test_geometry_is_still_reported_when_disabled(self) -> None:
        assessed = run((det(100, 100, 200, 180, conf=0.9),), detection_quality_enabled=False)
        assert assessed[0].aspect_ratio == pytest.approx(1.25)
        assert assessed[0].area_fraction > 0

    def test_the_verdict_is_stable_across_runs(self) -> None:
        detections = (
            det(100, 100, 200, 180, conf=0.80),
            det(104, 104, 204, 184, conf=0.60),
            det(0, 0, 400, 400, conf=0.90),
            det(300, 300, 340, 330, conf=0.22),
        )
        first = [(a.quality, a.flags) for a in run(detections)]
        second = [(a.quality, a.flags) for a in run(detections)]
        assert first == second

    def test_order_is_preserved(self) -> None:
        """Verdicts are returned alongside their inputs, not reordered."""
        detections = (
            det(10, 10, 50, 40, conf=0.21),
            det(100, 100, 160, 140, conf=0.90),
        )
        assessed = run(detections)
        assert [a.detection for a in assessed] == list(detections)

    def test_confirmed_returns_only_countable_detections(self) -> None:
        assessed = run(
            (
                det(100, 100, 160, 140, conf=0.90),
                det(10, 10, 50, 40, conf=0.21),
                det(0, 0, 400, 400, conf=0.95),
            )
        )
        countable = confirmed(assessed)
        assert len(countable) == 1
        assert countable[0].confidence == pytest.approx(0.90)

    def test_an_empty_frame_assesses_to_nothing(self) -> None:
        assert assess((), frame_width=800, frame_height=600, policy=policy_from(
            Settings(app_env="local")
        )) == ()

    def test_every_unconfirmed_detection_states_a_reason(self) -> None:
        """An unexplained exclusion is indistinguishable from a bug."""
        assessed = run(
            (
                det(0, 0, 400, 400, conf=0.95),
                det(10, 10, 50, 40, conf=0.21),
                det(100, 100, 200, 180, conf=0.80),
                det(104, 104, 204, 184, conf=0.60),
            )
        )
        for item in assessed:
            if item.quality is not DetectionQuality.CONFIRMED:
                assert item.reason, f"{item.quality} carries no reason"
                assert item.flags


# ---------------------------------------------------------------------------
# The downstream contract
# ---------------------------------------------------------------------------
@pytest.fixture
def quality_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A client whose detector emits one good box, one weak box and a duplicate."""
    from test_vision import ScriptedDetector

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

    from app.vision.analysis import DetectorProvider

    script = [
        (
            det(100, 100, 160, 150, cls=VehicleClass.CAR, conf=0.90),
            det(300, 300, 350, 340, cls=VehicleClass.CAR, conf=0.24),
            det(102, 102, 162, 152, cls=VehicleClass.TRUCK, conf=0.70),
        )
    ] * 20

    class StubProvider(DetectorProvider):
        def __init__(self) -> None:
            super().__init__(settings)
            self._detector = ScriptedDetector(script)
            self._detector.load()

        def get(self) -> Any:
            return self._detector

    app.state.detector_provider = StubProvider()

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client

    media_routes.reset_store()
    get_settings.cache_clear()


def _upload(client: TestClient, tmp_path: Path) -> str:
    path = tmp_path / "frame.png"
    Image.new("RGB", (FRAME_W, FRAME_H), (30, 60, 90)).save(path)
    return client.post(
        "/api/media/images",
        files={"file": ("frame.png", path.read_bytes(), "image/png")},
    ).json()["media_id"]


class TestQualityReachesTheApi:
    def test_every_detection_is_reported_with_its_verdict(
        self, quality_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = _upload(quality_client, tmp_path)
        body = quality_client.post(f"/api/vision/images/{media_id}/analyze").json()

        assert len(body["detections"]) == 3, "nothing is deleted"
        verdicts = [d["quality"] for d in body["detections"]]
        assert verdicts == ["confirmed", "provisional", "provisional"]

    def test_the_two_tallies_differ_and_both_are_published(
        self, quality_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = _upload(quality_client, tmp_path)
        body = quality_client.post(f"/api/vision/images/{media_id}/analyze").json()

        assert body["detection_tally"]["total"] == 3
        assert body["confirmed_tally"]["total"] == 1
        assert body["confirmed_tally"]["vehicles"] == 1

    def test_an_excluded_detection_says_why(
        self, quality_client: TestClient, tmp_path: Path
    ) -> None:
        media_id = _upload(quality_client, tmp_path)
        body = quality_client.post(f"/api/vision/images/{media_id}/analyze").json()

        weak = body["detections"][1]
        assert weak["quality_flags"] == ["below_class_confidence"]
        assert "confirmation floor" in weak["quality_reason"]

        duplicate = body["detections"][2]
        assert "duplicate" in duplicate["quality_flags"]
        assert duplicate["duplicate_of_index"] == 0

    def test_the_policy_is_published_with_the_result(
        self, quality_client: TestClient, tmp_path: Path
    ) -> None:
        """A count is not interpretable without the rules that produced it."""
        media_id = _upload(quality_client, tmp_path)
        body = quality_client.post(f"/api/vision/images/{media_id}/analyze").json()
        policy = body["quality_policy"]
        assert policy["enabled"] is True
        assert policy["confirm_confidence"]["car"] == pytest.approx(0.40)
        assert policy["max_area_fraction"] == pytest.approx(0.05)

    def test_status_publishes_the_policy(self, quality_client: TestClient) -> None:
        body = quality_client.get("/api/vision/status").json()
        assert body["quality_policy"]["enabled"] is True
        assert body["quality_policy"]["duplicate_iou"] == pytest.approx(0.55)

    def test_analytics_counts_only_confirmed_detections(
        self, quality_client: TestClient, tmp_path: Path
    ) -> None:
        """The contract that matters: a phantom vehicle must not become traffic."""
        media_id = _upload(quality_client, tmp_path)
        body = quality_client.post(f"/api/analytics/images/{media_id}").json()

        assert body["detection_tally"]["total"] == 1, (
            "analytics must see one vehicle, not three boxes"
        )
        assert body["detection_tally"]["vehicles"] == 1
        assert len(body["object_positions"]) == 1


# ---------------------------------------------------------------------------
# The frame the policy was chosen against
# ---------------------------------------------------------------------------
class TestReferenceImage:
    """Re-measures the findings on the real junction frame.

    The verdicts asserted here rest on a visual inspection of 46 crops -- my
    judgement, on one image, with no second opinion and no labelled ground
    truth. They are asserted because a regression in them would be a real
    regression, not because they are certified correct.
    """

    @pytest.fixture
    def assessed(self) -> tuple[AssessedDetection, ...]:
        pytest.importorskip("ultralytics")
        pytest.importorskip("torch")
        if not REFERENCE_IMAGE.exists():
            pytest.skip(f"reference image {REFERENCE_IMAGE.name} is not present")
        weights = REPO_ROOT / "models" / "yolov8s.pt"
        if not weights.exists():
            pytest.skip("yolov8s.pt is not present in the models directory")

        from app.vision.backends.ultralytics_detector import UltralyticsYoloDetector

        frame = np.asarray(Image.open(REFERENCE_IMAGE).convert("RGB"))
        detector = UltralyticsYoloDetector(
            models_dir=weights.parent,
            model_name="yolov8s.pt",
            device_preference="cpu",
            confidence_threshold=0.20,
            iou_threshold=0.45,
            image_size=1280,
            max_detections=300,
        )
        detector.load()
        try:
            result = detector.detect(
                frame, source_id="ref", sequence=0, captured_at=T0
            )
        finally:
            detector.close()

        return assess(
            result.detections,
            frame_width=result.frame_width or 0,
            frame_height=result.frame_height or 0,
            policy=policy_from(Settings(app_env="local")),
        )

    @staticmethod
    def _find(
        assessed: tuple[AssessedDetection, ...],
        *,
        cls: VehicleClass,
        conf: float,
        tolerance: float = 0.02,
    ) -> AssessedDetection | None:
        """Locate a detection by class and score rather than by coordinates."""
        matches = [
            a
            for a in assessed
            if a.detection.vehicle_class is cls
            and abs(a.detection.confidence - conf) <= tolerance
        ]
        return matches[0] if len(matches) == 1 else None

    def test_the_barrier_false_positive_is_not_confirmed(
        self, assessed: tuple[AssessedDetection, ...]
    ) -> None:
        """The detection this work started from: a traffic island, at 0.37."""
        vehicles = [
            a for a in assessed if a.detection.vehicle_class in VEHICLE_CLASSES
        ]
        near_037 = [
            a
            for a in vehicles
            if a.detection.vehicle_class is VehicleClass.CAR
            and 0.35 <= a.detection.confidence <= 0.39
        ]
        assert near_037, "the reference detection is missing; the frame changed"
        assert all(a.quality is not DetectionQuality.CONFIRMED for a in near_037), (
            "a box on road furniture at 0.37 must not be counted as a vehicle"
        )

    def test_the_building_detected_as_a_bus_is_rejected(
        self, assessed: tuple[AssessedDetection, ...]
    ) -> None:
        buses = [a for a in assessed if a.detection.vehicle_class is VehicleClass.BUS]
        assert buses, "the reference bus detection is missing"
        assert all(a.quality is DetectionQuality.REJECTED for a in buses)
        assert all(QualityFlag.IMPLAUSIBLE_AREA in a.flags for a in buses)

    def test_duplicates_are_found_without_touching_confidence(
        self, assessed: tuple[AssessedDetection, ...]
    ) -> None:
        """Four of six false positives fall to rules that cost no recall."""
        flagged = [a for a in assessed if QualityFlag.DUPLICATE in a.flags]
        assert flagged, "the reference frame contains overlapping boxes"

    def test_confident_vehicles_survive(
        self, assessed: tuple[AssessedDetection, ...]
    ) -> None:
        """Precision work must not quietly become a recall failure."""
        strong = [
            a
            for a in assessed
            if a.detection.vehicle_class in VEHICLE_CLASSES
            and a.detection.confidence >= 0.60
        ]
        assert len(strong) >= 15
        assert all(a.quality is DetectionQuality.CONFIRMED for a in strong)

    def test_the_confirmed_count_is_lower_than_the_reported_count(
        self, assessed: tuple[AssessedDetection, ...]
    ) -> None:
        vehicles = [
            a for a in assessed if a.detection.vehicle_class in VEHICLE_CLASSES
        ]
        countable = [a for a in vehicles if a.is_confirmed]
        assert 25 <= len(countable) < len(vehicles), (
            f"{len(countable)} confirmed of {len(vehicles)} reported: outside the "
            "range measured when the policy was chosen"
        )

    def test_no_confirmed_vehicle_carries_a_flag(
        self, assessed: tuple[AssessedDetection, ...]
    ) -> None:
        for item in assessed:
            if item.is_confirmed:
                assert item.flags == ()
                assert item.reason is None
