"""Mapping a pretrained model's label space onto our road-user vocabulary.

Pretrained detectors speak COCO, which was not built for traffic. The mapping
below is deliberately conservative, and the omissions matter more than the
entries:

* **``truck`` covers a wide range.** COCO lumps a panel van and an articulated
  lorry together. They differ in length, turning envelope and passenger-car
  equivalence, so a later phase that needs that distinction must get it from a
  fine-tuned model, not from a heuristic on box size applied here.
* **``train`` is not mapped.** A tram sharing the carriageway would be traffic,
  but COCO's ``train`` fires far more often on rail in the background. Mapping
  it would inject phantom road users.
* **Emergency vehicles are not inferable.** COCO has no ambulance or fire-engine
  class, and an ambulance detects as ``truck`` or ``car``. Nothing here may
  produce :attr:`~app.domain.enums.VehicleClass.AMBULANCE`: emergency status is
  a *declared* event, never a guess from pixels.

Anything unmapped is dropped rather than coerced to ``OTHER``. A detection we
cannot name is not a road user we can reason about, and keeping it would put
handbags and traffic lights into the road-user stream.
"""

from __future__ import annotations

from typing import Final

from app.domain.enums import VehicleClass

#: COCO label -> our road-user class. Keys are lower-cased COCO names, which is
#: what both ultralytics and standard ONNX exports report.
COCO_TO_ROAD_USER: Final[dict[str, VehicleClass]] = {
    "person": VehicleClass.PERSON,
    "bicycle": VehicleClass.BICYCLE,
    "car": VehicleClass.CAR,
    "motorcycle": VehicleClass.MOTORCYCLE,
    "motorbike": VehicleClass.MOTORCYCLE,
    "bus": VehicleClass.BUS,
    "truck": VehicleClass.TRUCK,
}

#: The classes Phase 3 is required to detect.
REQUIRED_CLASSES: Final[frozenset[VehicleClass]] = frozenset(
    {
        VehicleClass.CAR,
        VehicleClass.MOTORCYCLE,
        VehicleClass.BUS,
        VehicleClass.TRUCK,
        VehicleClass.BICYCLE,
        VehicleClass.PERSON,
    }
)

#: The 80 COCO class names, in the canonical order every YOLO export uses.
#: Needed by the ONNX backend, whose raw output carries indices with no labels.
COCO_CLASS_NAMES: Final[tuple[str, ...]] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


def map_label(label: str) -> VehicleClass | None:
    """Map a model label to a road-user class, or ``None`` to drop it."""
    return COCO_TO_ROAD_USER.get(label.strip().lower())


def road_user_class_indices(class_names: dict[int, str] | None = None) -> tuple[int, ...]:
    """Indices of mappable classes, for restricting inference at the model.

    Filtering inside the model is materially cheaper than detecting all 80
    classes and discarding 74 of them afterwards, and it keeps non-road-user
    boxes from ever competing in non-maximum suppression.
    """
    names = class_names or dict(enumerate(COCO_CLASS_NAMES))
    return tuple(
        index for index, label in names.items() if map_label(str(label)) is not None
    )
