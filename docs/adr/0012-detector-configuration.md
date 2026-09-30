# ADR 0012 — Detector configuration, selected by measurement

**Status:** accepted (Phase 3 revision)

## Context

Phase 3 shipped YOLOv8n at 640 px with a 0.25 confidence threshold. On a real
junction photograph — an oblique view of a busy crossing, 800×533, with vehicles
at every scale from a lorry filling a hundred pixels to parked cars far up the
avenue — it found **8 vehicles**. Well over forty are visible. The overlay made
the failure mode obvious: everything past the near corner of the junction was
simply not detected.

Those defaults were chosen when the point of the phase was to wire a real
perception pipeline, with accuracy tuning explicitly deferred. This is that
deferred work, done against the footage rather than against COCO mAP.

The constraint throughout: **no training and no fine-tuning.** Stock pretrained
COCO weights only.

## Measurements

All figures are vehicle detections (car, motorcycle, bus, truck) on the same
reference frame, through this project's own detector class, on CPU.

| model | channels | imgsz | conf | vehicles | car | bus | truck | person | ms |
|---|---|---|---|---|---|---|---|---|---|
| yolov8n | RGB *(as shipped)* | 640 | 0.25 | **8** | 7 | 0 | 1 | 0 | 247 |
| yolov8n | RGB | 640 | 0.20 | 10 | 7 | 2 | 1 | 1 | 456 |
| yolov8n | RGB | 640 | 0.15 | 16 | 12 | 2 | 2 | 2 | 833 |
| yolov8n | RGB | 1280 | 0.25 | 24 | 17 | 6 | 1 | 6 | 892 |
| yolov8n | BGR | 1280 | 0.25 | 32 | 26 | 6 | 0 | 13 | 796 |
| yolov8n | BGR | 1280 | 0.20 | 40 | 31 | 8 | 1 | 13 | 831 |
| yolov8n | BGR | 1280 | 0.15 | 49 | 39 | 9 | 1 | 15 | 628 |
| yolov8s | BGR | 640 | 0.25 | 28 | 28 | 0 | 0 | 0 | 379 |
| yolov8s | BGR | 1280 | 0.25 | 42 | 39 | 1 | 2 | 7 | 977 |
| **yolov8s** | **BGR** | **1280** | **0.20** | **46** | **43** | **1** | **2** | **10** | **978** |
| yolov8s | BGR | 1280 | 0.15 | 50 | 47 | 1 | 2 | 13 | 935 |

Duplicate boxes (IoU ≥ 0.55, or a smaller box ≥ 70 % contained in a larger one)
and implausible boxes (a vehicle covering ≥ 4 % of the frame, which no road
vehicle does at this camera scale):

| config | vehicles | duplicates | implausible | net |
|---|---|---|---|---|
| yolov8s 1280 conf 0.25 | 42 | 1 | 1 | 40 |
| yolov8s 1280 conf 0.20 | 46 | 2 | 1 | 43 |
| yolov8s 1280 conf 0.15 | 50 | 3 | 1 | 46 |

## Decision

**YOLOv8s, 1280 px, confidence 0.20, IoU 0.45, max 300 detections** — and a
channel-order fix in the ultralytics backend.

### The channel-order defect came first

The ingestion layer yields **RGB** arrays. That is the right contract: it is
what PIL produces and what the ONNX backend consumes directly. But the
ultralytics Python API documents raw numpy input as **BGR** and flips the
channels itself before inference, because it grew up around OpenCV. Handing it
RGB delivered a channel-swapped image to the network.

This was verified rather than assumed. Passing the PIL image — the path the
library converts correctly — gives byte-identical output to passing a BGR array,
and different output from the RGB array the application was passing:

```
PIL image        -> 56 boxes, top scores 0.8748 0.8668 0.8617 …
numpy BGR        -> 56 boxes, top scores 0.8748 0.8668 0.8617 …   (identical)
numpy RGB (ours) -> 49 boxes, top scores 0.8714 0.8700 0.8626 …
```

That equality is what makes this a fix rather than a tuning knob. It is also why
it went unnoticed for a phase: a channel-swapped image still detects *something*,
so the pipeline looked like it worked. Alone, correcting it took yolov8n at
1280 px from 24 vehicles to 32.

### Input size mattered most

A traffic camera looks *down* a street, so most vehicles are small in frame. At
640 px they fall below the detector's effective resolution and are not found at
all: yolov8s found 28 vehicles at 640 and 42 at 1280, and the missing fourteen
were the queue up the far avenue — exactly the traffic a junction model exists
to reason about. This is the single most consequential setting, and its cost is
roughly quadratic: ~1 s per frame on CPU against ~0.3 s at 640.

### YOLOv8s over YOLOv8n, on class accuracy rather than count

Nano found 32 vehicles at the selected size and threshold against small's 42,
which alone might not justify tripling the weights. The deciding evidence was
the labels. Nano called **six** things `bus`: a black saloon at the centre of
the junction, a white minivan, a small red delivery van, and a lorry. Small
called one thing a bus and got those same vehicles right.

That distinction is not cosmetic here. Anything weighting traffic by vehicle
type — passenger-car equivalence, an emergency-vehicle admissibility check, a
capacity estimate — is wrong in a way no threshold setting repairs. A count that
is 25 % low is a known limitation; a saloon car recorded as a bus is a false
statement about the road.

### 0.20 as the balance point

Each step down from 0.25 recovered about three genuine vehicles and added one
duplicate box. 0.20 takes the first step; 0.15 takes a second and starts
producing nested boxes on vehicles already found. A duplicate is not a harmless
extra box — it is a second vehicle in every count derived from the frame, and
Phase 4 has no way to tell it from a real one.

## Consequences

**One configuration, not two.** The same settings apply to stills and to video,
so "what produced this number?" has one answer. The cost is real: ~1 s per frame
means a 900-frame job takes about fifteen minutes on CPU. `video_frame_stride`
is the intended lever, and it is already configurable.

**The four knobs are now reachable from the API.** `POST
/api/vision/images/{id}/analyze` accepts optional `confidence_threshold`,
`iou_threshold`, `image_size` and `max_detections`. Posting no body behaves
exactly as before. An override builds a detector for that one request and
releases it, so no request can mutate the deployment's configuration, and
whatever was used is reported back in `inference` — a count is never separable
from the settings that produced it. `max_detections` is now published by
`/api/vision/status` alongside the other three.

**Weights are larger.** `yolov8s.pt` is ~22 MB against nano's ~6.5 MB, fetched
on first use into `models/` and identified by checksum in every result.

**Known limitations, unchanged by this pass.** The one implausible box — a
building on the right edge detected as a `bus` at 0.37 — survives at every
threshold tested, because it is a classification error rather than a marginal
score. Motorcycles were not detected at all on this frame; there appear to be
none in it, so nothing can be concluded about motorcycle recall. And all of this
is one frame from one camera: it is enough to reject the previous defaults and
to choose between candidates, and it is not a calibration. A measured evaluation
across the corpus belongs to Phase 13.
