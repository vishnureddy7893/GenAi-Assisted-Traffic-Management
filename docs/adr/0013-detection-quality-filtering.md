# ADR 0013 — Detection quality: reporting and counting are different questions

**Status:** accepted (Phase 3 revision)

## Context

A box on a traffic island — cones and painted lane markings, no vehicle — was
detected as `car` at **0.37** on the reference junction frame. It is a real
false positive, and it was being counted as a vehicle.

The obvious fix is to raise the confidence threshold. It does not work, and
measuring why is the whole basis of this decision.

Cropping and inspecting all 46 vehicle detections on that frame gives six that
are not distinct real vehicles:

| # | class | conf | what it actually is |
|---|---|---|---|
| 37 | bus | 0.37 | the office building on the right edge |
| 38 | car | 0.37 | traffic island: cones and road markings |
| 44 | truck | 0.29 | the same black vehicle already found as `car` at 0.34 |
| 50 | car | 0.23 | a box nested inside the red van already found at 0.77 |
| 53 | car | 0.22 | a blue tarpaulin beside the shrine |
| 55 | car | 0.20 | an 8×8 speck at the left frame border |

And the real vehicles run from **0.26 to 0.87**. The two populations overlap:
the barrier at 0.37 outscores eight vehicles that are genuinely there.

Sweeping the global threshold shows the cost directly:

| threshold | kept | real | suspicious | real lost |
|---|---|---|---|---|
| 0.20 | 46 | 40 | 6 | 0 |
| 0.25 | 42 | 39 | 3 | 1 |
| 0.30 | 39 | 37 | 2 | 3 |
| 0.35 | 35 | 33 | 2 | 7 |
| 0.40 | 30 | 30 | 0 | 10 |

The only threshold that removes every false positive also removes a quarter of
the real vehicles. Threshold tuning alone is the wrong instrument.

## Decision

Separate **what is reported** from **what may be counted**, and put the
difference in an explicit, configurable quality layer
(`app/vision/quality.py`).

The detector keeps its 0.20 threshold and reports everything. Each detection is
then assessed into one of three tiers, with the reason attached:

* `CONFIRMED` — may be counted as a road user.
* `PROVISIONAL` — detected and reported, not countable.
* `REJECTED` — geometrically impossible for a road vehicle in this view.

**Nothing is deleted.** Every detection is still returned by the API, still
drawn on the overlay (dashed, and labelled with its tier), and still carries its
own confidence. What changes is that the analytics layer consumes the confirmed
tier only.

### The rules, and what each one is worth

Measured on the reference frame, with the car floor left at the detection
threshold so each rule's own contribution is visible:

| rules enabled | confirmed | real | suspicious | precision | recall |
|---|---|---|---|---|---|
| none (raw detector) | 46 | 40 | 6 | 0.870 | 1.000 |
| area ceiling | 45 | 40 | 5 | 0.889 | 1.000 |
| duplicate suppression | 44 | 40 | 4 | 0.909 | 1.000 |
| edge margin | 45 | 40 | 5 | 0.889 | 1.000 |
| **all three together** | **42** | **40** | **2** | **0.952** | **1.000** |

Four of six false positives fall to rules that cost **no recall at all**. That
is the important result: precision improved from 0.870 to 0.952 without
discarding a single real vehicle.

* **Cross-class duplicate suppression.** Non-maximum suppression runs *per
  class* inside the model, so one vehicle labelled `car` by one head and `truck`
  by another survives it and is counted twice. Containment is checked as well as
  IoU, because a small box nested inside a large one scores a low IoU while
  still being a second count.
* **Area ceiling.** A single road vehicle cannot cover a large fraction of a
  wide-area junction view. The largest real vehicle here covers 0.77%; the
  building covers 7.07%. This is a claim about the camera, not a size
  preference, and a close-range camera must raise or disable it.
* **Edge margin.** A box clipped by the frame border shows part of an object, so
  the same score is weaker evidence. It requires the class floor plus a margin
  rather than being rejected — a confident vehicle entering the frame is real.
* **Class-specific floors.** COCO's classes are not equally reliable in an
  oblique traffic view. `bus` fires on large pale structures; `truck` and `car`
  compete for the same vans.

### The one that needs confidence, and its price

After all of that, two false positives remain, and only a car floor removes
them: the tarpaulin at 0.22 and — the detection this work started from — the
traffic island at **0.37**.

| car floor | confirmed | real | suspicious | precision | recall |
|---|---|---|---|---|---|
| 0.20 | 41 | 39 | 2 | 0.951 | 0.975 |
| 0.25 | 39 | 38 | 1 | 0.974 | 0.950 |
| 0.30 | 37 | 36 | 1 | 0.973 | 0.900 |
| 0.35 | 34 | 33 | 1 | 0.971 | 0.825 |
| **0.40** | **30** | **30** | **0** | **1.000** | **0.750** |

Nothing between 0.25 and 0.40 helps: the barrier scores 0.37, so the floor is
either above it or below it.

**The default is 0.40**, because the requirement was explicit that this
detection must not be treated as a real vehicle. The cost is stated rather than
hidden: ten real vehicles on this frame become provisional. They are still
detected, still reported, still drawn, and still visible in the raw tally — the
confirmed count is simply conservative.

An operator who prefers recall sets `DETECTION_CONFIRM_CONFIDENCE_CAR=0.25` and
gets 38 real vehicles with one false positive. Both positions are defensible;
what is not defensible is choosing between them without the table above.

## What was deliberately not done

**No minimum-size rule.** The smallest *confirmed* vehicle on this frame is
15×10 px — a real car far up the avenue. A size cut sharp enough to remove the
8×8 false positive removes it too. Size enters only through the maximum area
check.

**No coordinates, regions of interest, or per-image exceptions.** Every rule is
a property of a box, a class, or a pair of boxes. A filter tuned to one
photograph is not a filter, and a test asserts the module contains no such
constants.

**No narrow aspect band.** Real vehicles here span 0.66 to 1.81 width-over-
height. The configured band (0.15–6.0) excluded nothing on this frame and is a
guard against degenerate boxes, not a contributor. Narrowing it toward the
observed range would start discarding real vehicles at unusual angles.

**No training.** Stock pretrained COCO weights throughout.

## Consequences

**The measurement pipeline counts confirmed detections only.** Both analytics
entry points — image and video — filter on the tier before building
`ObservedFrame`. A phantom vehicle can no longer become a vehicle count, a
density, a congestion band or a route cost.

**Two tallies are published, and the gap is explained.** `detection_tally` is
every box; `confirmed_tally` is what a measurement may use. The dashboard leads
with the countable figure and states how many were excluded and why.

**The policy travels with the result.** `quality_policy` is embedded in every
image analysis and published by `/api/vision/status`, because a count produced
under one policy and read against another is unreproducible.

**Everything is configurable.** Eleven settings, each with a documented default
and a rationale. Setting `DETECTION_QUALITY_ENABLED=false` makes the layer inert,
so its effect can be measured rather than argued about.

## What this single image cannot establish

The verdicts above are **my visual judgement of 46 boxes on one photograph**.
Nobody labelled this frame; there is no second opinion; a box called real could
be wrong. "Precision 1.000" means *no box I judged suspicious was confirmed* —
it is not a measured accuracy figure and must not be quoted as one.

Specifically unestablished:

* **Whether these thresholds generalise.** One camera, one moment, one lighting
  condition. A night frame, rain, or a different mounting height could move
  every number here.
* **Motorcycle behaviour.** None appear in this frame, so no floor is claimed
  for that class and it falls back to the detection threshold.
* **The true false-negative rate.** Vehicles the detector missed entirely are
  invisible to this analysis — the recall figures are against detections that
  *were* made, not against the traffic actually present.
* **Whether the area ceiling is right for other cameras.** It encodes a
  wide-area junction view. On a close-range approach camera it will reject real
  vehicles, visibly and with a reason, but it will reject them.

A calibration across the corpus, against labels rather than judgement, belongs
to the measured evaluation in Phase 13.

## Note for the video path

A static road feature and a moving vehicle are trivially separable *over time*,
and the tracker already exists. Nothing in this ADR uses that: the assessment is
per-frame, because the evidence here is a still image. Requiring persistence or
motion before confirming a detection in video is the obvious next improvement
and is not attempted here.
