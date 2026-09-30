"""Typed application configuration.

Design notes
------------
* All configuration arrives through the environment (12-factor).  There are no
  hard-coded secrets, hosts or paths anywhere in the codebase.
* Secrets use :class:`~pydantic.SecretStr` so they cannot be printed by accident:
  ``repr`` and ``str`` render ``**********``, and the redaction is verified by a
  test in ``tests/test_config.py``.
* Settings are validated at import time of the app, not at first use, so a
  misconfigured deployment fails fast and loudly instead of at 3 a.m. mid-request.
* ``get_settings`` is cached; tests clear the cache via ``reset_settings_cache``.
"""

from __future__ import annotations

import functools
import math
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root = <repo>/backend/app/core/config.py -> parents[3]
REPO_ROOT: Path = Path(__file__).resolve().parents[3]


class AppEnv(StrEnum):
    """Deployment environment."""

    LOCAL = "local"
    CI = "ci"
    STAGING = "staging"
    PRODUCTION = "production"


class DropPolicy(StrEnum):
    """Backpressure policy for unbounded (live) sources.

    A live camera produces frames whether or not we are ready for them.  The
    only three honest options are to discard the oldest, discard the newest, or
    block the producer.  Blocking a real-time source accumulates latency without
    bound, so it is never the default for live traffic.
    """

    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    BLOCK = "block"


class Settings(BaseSettings):
    """Root configuration object. Instantiate via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
    )

    # --- Application ------------------------------------------------------
    app_env: AppEnv = AppEnv.LOCAL
    app_debug: bool = False
    app_name: str = "GenAI Traffic Management"

    # --- API server -------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: Annotated[int, Field(ge=1, le=65535)] = 8000
    api_prefix: str = "/api"
    api_cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )

    # --- Logging ----------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "console"] = "console"

    # --- Data locations ---------------------------------------------------
    datasets_dir: Path = REPO_ROOT / "datasets"
    models_dir: Path = REPO_ROOT / "models"

    media_dir: Path = REPO_ROOT / "datasets" / "media"
    """Root for ingested images and videos.

    Everything the media API can reach lives under this directory, and every
    path derived from client input is resolved against it and rejected if it
    escapes. Treating this as the only reachable root is what turns path
    traversal from a validation problem into a structural impossibility.
    """

    sites_file: Path = REPO_ROOT / "datasets" / "sites" / "observation-sites.json"
    """Where surveyed observation site locations are persisted.

    Deliberately outside ``media_dir``. A site records where a junction is; a
    media asset records a file of it. Storing the survey under the media root
    would tie the two lifetimes together, so clearing ingested media would
    discard locations that are expensive to re-establish and easy to guess
    wrongly.

    Never reachable from client input: unlike the media library and the network
    directory, no endpoint accepts a path for this file. It is configuration
    only, so there is no traversal surface to guard.
    """

    confirmed_bindings_file: Path = (
        REPO_ROOT / "datasets" / "sites" / "confirmed-bindings.json"
    )
    """Where human-confirmed site-to-junction bindings are persisted.

    Kept beside the sites file and apart from it. A site says where a camera is;
    a confirmed binding says which junction an operator decided it watches. The
    second is a decision about the first, so re-surveying a coordinate must not
    silently rewrite the decision, and one file holding both would make that
    impossible to keep straight.

    Never reachable from client input: no endpoint accepts a path for it.
    """

    site_binding_max_distance_m: Annotated[float, Field(gt=0.0)] = 250.0
    """Range within which proximity to a junction counts as evidence for a
    binding.

    Not a threshold for *making* a binding -- nothing is ever bound without a
    human -- but for how a nearest-junction result is presented. Inside it, the
    junction is offered as a candidate; outside it, the distance is reported and
    no candidate is offered, because at that range the likeliest explanations
    are a mistyped coordinate or a junction that is not on this map.

    250 m is chosen against the geometry of the thing being measured: a camera
    overlooking an intersection is typically mounted within 100-150 m of it,
    while urban junctions in the example network are roughly 550 m apart. A
    range above about half the junction spacing would routinely admit the
    neighbouring junction as a plausible candidate, which is precisely the
    confusion the candidate is supposed to surface rather than absorb.
    """

    # --- Upload limits ----------------------------------------------------
    max_image_upload_bytes: Annotated[int, Field(gt=0)] = 32 * 1024 * 1024
    """32 MiB. A traffic still far exceeding this is more likely a mistake or an
    attack than a camera frame."""

    max_video_upload_bytes: Annotated[int, Field(gt=0)] = 512 * 1024 * 1024
    """512 MiB. Recorded intersection clips are minutes, not hours: the whole
    point of Mode A is a bounded, replayable segment."""

    # --- Traffic state assembly (safety-relevant) -------------------------
    max_observation_staleness_s: Annotated[float, Field(gt=0)] = 5.0
    """Beyond this age an observation still contributes but the junction is
    flagged STALE and any dependent decision is downgraded."""

    drop_observation_after_s: Annotated[float, Field(gt=0)] = 30.0
    """Beyond this age the observation is discarded; the junction becomes UNKNOWN.
    Routing over an UNKNOWN junction is a soft penalty, never a silent zero."""

    # --- Computer vision (phase 3) ---------------------------------------
    detector_backend: Literal["auto", "ultralytics", "onnx", "none"] = "auto"
    """Which detector implementation to use.

    ``auto`` selects the first backend whose dependencies are importable.
    ``none`` disables detection entirely, which is a legitimate deployment for
    an install that only ingests tabular data. Never hard-coded at a call site:
    every consumer resolves the detector through the registry.
    """

    detector_model: str = "yolov8s.pt"
    """Model identifier or filename, resolved inside ``models_dir``.

    The small YOLOv8 weights (~22 MB), selected over the nano ones against
    measured counts on a real junction frame rather than against COCO mAP.

    On the reference frame at 1280 px, nano found 32 vehicles and labelled six
    of them ``bus`` -- a black saloon, a minivan and a small delivery van among
    them. Small found 42, with one bus, and got those same vehicles right. The
    class confusion is the deciding factor rather than the count: a passenger-car
    equivalence weighted by vehicle type is wrong in a way no threshold fixes.

    Still stock pretrained COCO weights. Nothing here is trained or fine-tuned.
    """

    detector_device: Literal["auto", "cpu", "cuda"] = "auto"
    """``auto`` uses CUDA when it is genuinely available and CPU otherwise.
    CUDA is never assumed: the resolved device is probed, then reported in
    every inference result."""

    detector_confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.20
    """Minimum score for a detection to be reported.

    Measured on the reference junction frame with YOLOv8s at 1280 px: 0.25 gave
    42 vehicles, 0.20 gave 46, 0.15 gave 50. Each step down recovered about
    three genuine vehicles -- parked cars far up the avenue -- and added one
    duplicate box on a vehicle already found.

    0.20 is the balance point. Going to 0.15 keeps buying recall, but a duplicate
    is not a harmless extra box: it is a second vehicle in every count derived
    from this frame, and Phase 4 has no way to tell it from a real one.
    """

    detector_iou_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.45
    """Non-maximum-suppression overlap threshold."""

    detector_image_size: Annotated[int, Field(ge=64, le=4096)] = 1280
    """Inference input size, and the single most consequential setting here.

    A traffic camera looks down a street, so most vehicles are small and distant
    in the frame. At 640 px they fall below the detector's effective resolution
    and are simply not found: on the reference junction frame YOLOv8s found 28
    vehicles at 640 and 42 at 1280, and the missing fourteen were the ones
    queued up the far avenue -- exactly the traffic a junction model needs.

    The cost is roughly quadratic in this number, and it is real: about 1 s per
    frame on CPU against 0.3 s at 640. That is affordable for a still and heavy
    for a long video, which is what ``video_frame_stride`` exists to trade
    against.
    """

    detector_max_detections: Annotated[int, Field(gt=0)] = 300
    """Ceiling on detections returned per frame.

    300 is not a tuning choice so much as a guard: a busy junction with a
    crowded crossing produces well over a hundred road users, and a ceiling that
    binds would silently truncate the frame. The reference image peaks at 63.
    """

    # --- Detection quality (phase 3 revision) ----------------------------
    # The detector's threshold decides what is *reported*. These decide what may
    # be *counted*. They are separate because measurement on a real junction
    # frame showed the two populations overlap: false positives ran from 0.20 to
    # 0.37 while real vehicles ran from 0.26 to 0.87, so no single threshold
    # separates them. See docs/adr/0013-detection-quality-filtering.md.
    detection_quality_enabled: bool = True
    """When false, every detection is confirmed and this layer is inert. Left
    configurable so its effect can be measured by turning it off, rather than
    argued about."""

    detection_confirm_confidence_car: Annotated[float, Field(ge=0.0, le=1.0)] | None = 0.40
    """Confirmation floor for cars.

    0.40 rather than the 0.20 detection threshold because of one specific
    finding: a box on a traffic island -- cones and painted markings, no vehicle
    -- scored 0.37, above eight real cars. Nothing in the box's geometry
    separates it from a real car, so confidence is the only lever, and it has to
    be set above 0.37 to exclude it.

    The cost is measured and real: ten real vehicles on that frame score below
    0.40 and become provisional. They are still detected and still reported --
    what changes is that a count does not rest on them. An operator who prefers
    recall to precision sets this to 0.25, which on the same frame confirmed 38
    real vehicles and one false positive.
    """

    detection_confirm_confidence_motorcycle: (
        Annotated[float, Field(ge=0.0, le=1.0)] | None
    ) = None
    """No floor claimed: no motorcycle appeared on the reference frame, so there
    is nothing to set one from. Falls back to the detection threshold."""

    detection_confirm_confidence_bus: Annotated[float, Field(ge=0.0, le=1.0)] | None = 0.40
    """COCO's ``bus`` fires readily on large pale structures in an oblique view.
    The only bus on the reference frame was a building, at 0.37."""

    detection_confirm_confidence_truck: Annotated[float, Field(ge=0.0, le=1.0)] | None = 0.35
    """``truck`` and ``car`` compete for the same vans and SUVs. On the
    reference frame a black vehicle was detected as both -- car at 0.34 and
    truck at 0.29 -- and the truck box was the spurious one."""

    detection_confirm_confidence_bicycle: (
        Annotated[float, Field(ge=0.0, le=1.0)] | None
    ) = None
    detection_confirm_confidence_person: (
        Annotated[float, Field(ge=0.0, le=1.0)] | None
    ) = None
    """Neither is a vehicle and neither was assessed on the reference frame, so
    no floor is claimed for them."""

    detection_max_area_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 0.05
    """Largest share of the frame one road vehicle may cover. ``0`` disables.

    A physical claim about a camera watching a whole junction, not a size
    preference: the largest real vehicle on the reference frame covered 0.77%
    and the building detected as a bus covered 7.07%. A close-range approach
    camera, where a bus can legitimately fill a third of the view, must raise
    this or switch it off.
    """

    detection_min_aspect_ratio: Annotated[float, Field(ge=0.0)] = 0.15
    detection_max_aspect_ratio: Annotated[float, Field(ge=0.0)] = 6.0
    """Width-over-height band outside which a box is a sliver or a column
    rather than a vehicle. ``0`` on either side disables it.

    Deliberately far wider than anything observed -- real vehicles on the
    reference frame spanned 0.66 to 1.81, and the band excluded nothing. It is a
    guard against degenerate boxes, not a contributor, and narrowing it toward
    the observed range would start discarding real vehicles at unusual angles.
    """

    detection_duplicate_iou: Annotated[float, Field(ge=0.0, le=1.0)] = 0.55
    detection_duplicate_containment: Annotated[float, Field(ge=0.0, le=1.0)] = 0.70
    """When two boxes are one object. Non-maximum suppression runs per class
    inside the model, so a vehicle labelled ``car`` by one head and ``truck`` by
    another survives it and is counted twice.

    Containment is separate from IoU because IoU misses the nested case: a small
    box inside a large one scores a low IoU while still being a second count.
    ``0`` on either disables that measure.
    """

    detection_edge_margin_px: Annotated[int, Field(ge=0)] = 1
    """How close to the border counts as touching it."""

    detection_edge_confidence_margin: Annotated[float, Field(ge=0.0, le=1.0)] = 0.10
    """Extra confidence required of a box clipped by the frame edge.

    A truncated box shows part of an object, so the same score is weaker
    evidence. ``0`` disables, which treats a half-visible object exactly like a
    fully visible one.
    """

    video_frame_stride: Annotated[int, Field(ge=1)] = 1
    """Process every Nth frame. Traffic state changes over seconds, so a stride
    above 1 is often free accuracy-wise and linearly cheaper -- but it widens
    the gap tracking must associate across, so it is a deliberate setting."""

    video_max_frames: Annotated[int, Field(gt=0)] = 900
    """Ceiling on frames analysed per job. Bounds worst-case runtime and memory
    on an unexpectedly long upload."""

    # --- Traffic analytics (phase 4) -------------------------------------
    analytics_stationary_speed_frac: Annotated[float, Field(gt=0, le=1)] = 0.015
    """Image-space speed below which an object counts as stationary, expressed
    in **frame diagonals per second**.

    Normalised by frame diagonal rather than given in pixels so one threshold
    behaves the same on a 640x480 clip and a 4K one. A pixel threshold would
    silently mean something different on every camera.
    """

    analytics_min_stationary_frames: Annotated[int, Field(ge=1)] = 3
    """Consecutive stationary frames before an object is queue evidence.

    Guards against a single mis-measured frame: detection jitter routinely makes
    a moving vehicle look motionless for one frame, and a queue built from that
    would appear and vanish frame to frame.
    """

    analytics_queue_proximity_factor: Annotated[float, Field(gt=0)] = 1.8
    """Cluster gap for grouping stationary vehicles, as a multiple of their mean
    box diagonal. Scales with apparent object size, so vehicles far from the
    camera are not judged by the same pixel gap as near ones."""

    analytics_flow_window_s: Annotated[float, Field(gt=0)] = 5.0
    """Bucket width for arrival/presence rates over time."""

    analytics_min_track_frames: Annotated[int, Field(ge=2)] = 2
    """Track points required before direction or speed is computed. Two is the
    arithmetic minimum; below it the measurement is reported unavailable rather
    than defaulted to zero."""

    # --- Network state and congestion interpretation (phase 5) -----------
    # Band cut points on the congestion index, which is a weighted blend of the
    # available signals in [0, 1]. Each is a *lower* bound: a band holds from its
    # cut point up to the next one. Deliberately configurable, because these are
    # policy, not physics: what an operator calls "heavy" differs by city, road
    # class and time of day, and burying the choice in code would make it
    # unarguable.
    congestion_index_light: Annotated[float, Field(ge=0.0, le=1.0)] = 0.20
    congestion_index_moderate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.40
    congestion_index_heavy: Annotated[float, Field(ge=0.0, le=1.0)] = 0.60
    congestion_index_severe: Annotated[float, Field(ge=0.0, le=1.0)] = 0.78
    congestion_index_gridlock: Annotated[float, Field(ge=0.0, le=1.0)] = 0.92

    # Reference values that normalise each raw signal onto [0, 1]. Every one is
    # a site-specific quantity presented as a default, never as a calibration:
    # they must be validated against the footage before a verdict is trusted.
    congestion_occupancy_reference: Annotated[float, Field(gt=0.0, le=1.0)] = 0.35
    """Image occupancy ratio treated as fully congested.

    Well below 1.0 because a road surface is only part of a frame and vehicles
    never tile it completely; an occupancy of 0.35 is already a scene dominated
    by traffic."""

    congestion_stationary_reference: Annotated[float, Field(gt=0.0, le=1.0)] = 0.80
    """Fraction of observed vehicles that must be stationary to score as fully
    congested. Not 1.0: a junction on red routinely stops every visible vehicle
    for a few seconds without being congested, so the top of the scale is set
    where stopping has become the normal state rather than a phase."""

    congestion_image_speed_reference: Annotated[float, Field(gt=0.0)] = 0.08
    """Image speed, in frame diagonals per second, treated as free-flowing."""

    congestion_density_reference_veh_per_km_per_lane: Annotated[float, Field(gt=0.0)] = 90.0
    """Jam density. A common textbook range is 80-140 veh/km/lane depending on
    vehicle mix and lane width; the low end is used because mixed traffic with
    two-wheelers packs more vehicles into the same space."""

    congestion_free_flow_speed_kph: Annotated[float, Field(gt=0.0)] = 50.0
    """Speed treated as fully free-flowing. Urban arterial default."""

    # Signal weights. Renormalised over whichever signals are actually
    # available, so a missing signal reduces confidence rather than silently
    # scoring zero and dragging the verdict toward free-flow.
    congestion_weight_density: Annotated[float, Field(ge=0.0)] = 0.40
    congestion_weight_speed: Annotated[float, Field(ge=0.0)] = 0.35
    congestion_weight_occupancy: Annotated[float, Field(ge=0.0)] = 0.40
    congestion_weight_stationary: Annotated[float, Field(ge=0.0)] = 0.35

    @model_validator(mode="after")
    def _congestion_bands_increase(self) -> Settings:
        """Reject overlapping congestion bands at construction.

        A misordered set would not raise anywhere else: classification would
        simply return a band that no reader could predict, and the error would
        surface as an argument about traffic rather than as a configuration bug.
        """
        bands = (
            ("congestion_index_light", self.congestion_index_light),
            ("congestion_index_moderate", self.congestion_index_moderate),
            ("congestion_index_heavy", self.congestion_index_heavy),
            ("congestion_index_severe", self.congestion_index_severe),
            ("congestion_index_gridlock", self.congestion_index_gridlock),
        )
        for (lower_name, lower), (upper_name, upper) in pairwise(bands):
            if upper <= lower:
                msg = (
                    f"congestion band cut points must strictly increase: "
                    f"{upper_name}={upper} does not exceed {lower_name}={lower}"
                )
                raise ValueError(msg)
        return self

    # --- Road network and emergency management (phase 6) -----------------
    networks_dir: Path = REPO_ROOT / "datasets" / "networks"
    """Library root for loadable road network files. A load request names a path
    *relative to this directory*, resolved with the same traversal guard the
    media library uses: an API that accepted absolute paths would be a file-read
    primitive pointed at the whole filesystem."""

    network_file: str | None = None
    """Network to load at startup, relative to ``networks_dir``. ``None`` loads
    nothing, and ``/api/network`` then answers 404 rather than serving a map
    nobody chose. Auto-loading a bundled example by default would mean a
    deployment could route against synthetic geography without anyone deciding
    to."""

    exclusion_zone_max_radius_m: Annotated[float, Field(gt=0)] = 2000.0
    """Upper bound on an incident exclusion radius. A mistyped radius is the
    cheapest way to disconnect the whole network, and a hard constraint cannot
    be outbid once applied -- so an implausible value is rejected at declaration
    rather than discovered when no route can be found."""

    # --- Route optimization (phase 7) ------------------------------------
    # Every weight is in **equivalent seconds**, so the total cost is a
    # risk-adjusted arrival time rather than an uninterpretable score. A weight
    # here is policy: it states what the operator is willing to trade, and it is
    # meant to be argued with, which is why none of these live as constants in
    # the optimizer.
    route_max_candidates: Annotated[int, Field(ge=1, le=20)] = 4
    """How many ranked routes to return, best first. Alternatives are loopless
    paths from Yen's algorithm; each additional one costs another Dijkstra."""

    route_max_settled_nodes: Annotated[int, Field(ge=100, le=10_000_000)] = 200_000
    """Ceiling on nodes settled per search. Bounds worst-case time on a large or
    pathological network instead of letting one request run unbounded."""

    # Congestion, priced by the band Phase 5 interpreted. UNKNOWN is deliberately
    # absent: an unclassified segment is priced by the uncertainty weights below,
    # never as free-flowing, because treating "we did not see" as "it is clear"
    # is what makes an optimizer prefer unmonitored roads.
    route_congestion_free_flow_s: Annotated[float, Field(ge=0)] = 0.0
    route_congestion_light_s: Annotated[float, Field(ge=0)] = 10.0
    route_congestion_moderate_s: Annotated[float, Field(ge=0)] = 40.0
    route_congestion_heavy_s: Annotated[float, Field(ge=0)] = 110.0
    route_congestion_severe_s: Annotated[float, Field(ge=0)] = 240.0
    route_congestion_gridlock_s: Annotated[float, Field(ge=0)] = 600.0

    # Evidence quality. These are what stop an unobserved road from winning by
    # default, and they are separate from congestion on purpose: "busy" and
    # "unknown" are different claims and deserve different weights.
    route_uncertainty_unmapped_s: Annotated[float, Field(ge=0)] = 30.0
    """Segment with no observation mapped to it at all."""

    route_uncertainty_unclassified_s: Annotated[float, Field(ge=0)] = 25.0
    """Mapped, but the traffic state could not be classified."""

    route_uncertainty_stale_s: Annotated[float, Field(ge=0)] = 20.0
    """Mapped and classified, but the evidence is past the staleness budget."""

    # Incidents that degrade without blocking. A blockage is a hard constraint
    # and is never priced; this is the soft tier only.
    route_incident_degraded_s: Annotated[float, Field(ge=0)] = 90.0

    # Turns. A heavy vehicle turning across traffic is not free, and a U-turn is
    # worse than a turn.
    route_turn_penalty_s: Annotated[float, Field(ge=0)] = 8.0
    route_u_turn_penalty_s: Annotated[float, Field(ge=0)] = 30.0

    route_distance_penalty_s_per_100m: Annotated[float, Field(ge=0)] = 0.0
    """Distance priced as a stated preference. Zero by default because distance
    already enters through travel time; a non-zero value double-counts it
    deliberately. Never a way to infer time for a segment of unknown speed."""

    # Emergency profile. Defaults are identical to the normal profile, so any
    # divergence is an explicit operator decision rather than a number invented
    # here. What differs structurally rather than numerically: emergency routing
    # applies vehicle-type physical exclusions as hard constraints, and prefers
    # an emergency-specific travel time where one exists.
    route_emergency_congestion_free_flow_s: Annotated[float, Field(ge=0)] = 0.0
    route_emergency_congestion_light_s: Annotated[float, Field(ge=0)] = 10.0
    route_emergency_congestion_moderate_s: Annotated[float, Field(ge=0)] = 40.0
    route_emergency_congestion_heavy_s: Annotated[float, Field(ge=0)] = 110.0
    route_emergency_congestion_severe_s: Annotated[float, Field(ge=0)] = 240.0
    route_emergency_congestion_gridlock_s: Annotated[float, Field(ge=0)] = 600.0
    route_emergency_uncertainty_unmapped_s: Annotated[float, Field(ge=0)] = 30.0
    route_emergency_uncertainty_unclassified_s: Annotated[float, Field(ge=0)] = 25.0
    route_emergency_uncertainty_stale_s: Annotated[float, Field(ge=0)] = 20.0
    route_emergency_incident_degraded_s: Annotated[float, Field(ge=0)] = 90.0
    route_emergency_turn_penalty_s: Annotated[float, Field(ge=0)] = 8.0
    route_emergency_u_turn_penalty_s: Annotated[float, Field(ge=0)] = 30.0
    route_emergency_distance_penalty_s_per_100m: Annotated[float, Field(ge=0)] = 0.0

    # --- Traffic simulation (phase 8) ------------------------------------
    # Every parameter of the simulation lives here or arrives on the request.
    # None of them is a constant inside the model, because a simulated figure is
    # only arguable if the assumptions that produced it are visible and can be
    # changed without editing code.
    #
    # Note what is *absent*: there is no default demand, no default capacity and
    # no default free-flow speed. Those are properties of a road, not of a
    # simulator, and a default for any of them would be an invented traffic
    # measurement dressed as configuration. A segment without them is reported
    # as unmodelled, with the reason.
    sim_horizon_s: Annotated[float, Field(gt=0)] = 900.0
    """Simulated duration. Fifteen minutes: long enough for a queue to form and
    discharge at urban capacities, short enough that the constant-demand
    assumption is not absurd."""

    sim_step_s: Annotated[float, Field(gt=0)] = 5.0
    """Time step. The model is a point queue, so the step only bounds how finely
    a queue is resolved; it does not affect stability the way a cell-transmission
    step would."""

    sim_max_steps: Annotated[int, Field(ge=1, le=100_000)] = 20_000
    """Ceiling on steps per run, so a request naming a long horizon and a tiny
    step cannot make one HTTP call run unbounded."""

    sim_departure_offset_s: Annotated[float, Field(ge=0)] = 0.0
    """When the evaluated vehicle departs, measured into the horizon. Non-zero
    lets a route be evaluated against queues that have already built up."""

    sim_emergency_queue_bypass_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    """Fraction of exit-queue delay an emergency vehicle avoids by passing a
    stationary queue.

    Zero by default, and deliberately so. Any non-zero value is a claim about
    operational behaviour -- that opposing traffic yields, that the shoulder is
    usable -- which nothing in this system has measured. Set it only as a stated
    operator assumption; the run records it as one and names it in the result."""

    sim_use_observed_demand: bool = True
    """Whether to derive link demand from observed arrival rates where an
    observation area is bound to the segment. Turning it off leaves demand to
    stated assumptions only, which is the right setting when the bindings are
    not trusted."""

    sim_max_stored_runs: Annotated[int, Field(ge=1, le=1000)] = 50
    """How many completed runs are retained for retrieval. Oldest first out."""

    sim_default_baseline: Literal["shortest_distance", "runner_up", "explicit"] = (
        "shortest_distance"
    )
    """Which route the recommended one is compared against by default. The
    shortest-distance route is the one an operator would take from a map alone,
    which is the comparison that answers "was the recommendation worth it?"."""

    # --- Versioned policy pins -------------------------------------------
    safety_rules_version: str = "0.1.0"
    route_weights_version: str = "0.1.0"
    sim_parameters_version: str = "0.1.0"
    """Pinned version of the simulation parameter set. Two runs under different
    parameters are not comparable, and without this pin that is invisible."""

    # --- GenAI (Phase 12; unused until then) ------------------------------
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str | None = None
    openrouter_timeout_s: Annotated[float, Field(gt=0)] = 30.0
    openrouter_max_tokens: Annotated[int, Field(gt=0)] = 2048

    # --- Live ingestion (Phase 14; unused until then) ---------------------
    live_source_uri: SecretStr | None = None
    """Typed as a secret because RTSP URIs routinely embed credentials."""

    live_source_target_fps: Annotated[float, Field(gt=0)] = 10.0
    live_source_drop_policy: DropPolicy = DropPolicy.DROP_OLDEST
    live_source_max_queue_frames: Annotated[int, Field(gt=0)] = 60

    # --- Validators -------------------------------------------------------
    @field_validator("api_cors_origins", mode="before")
    @classmethod
    def _split_csv_origins(cls, value: object) -> object:
        """Accept a comma-separated string from the environment."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("datasets_dir", "models_dir")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        """Resolve relative data paths against the repository root."""
        return value if value.is_absolute() else (REPO_ROOT / value).resolve()

    @field_validator("drop_observation_after_s")
    @classmethod
    def _drop_after_exceeds_staleness(cls, value: float) -> float:
        # Cross-field validation happens in the model validator below; this
        # guard keeps the standalone bound meaningful.
        return value

    def model_post_init(self, __context: object) -> None:
        if self.drop_observation_after_s < self.max_observation_staleness_s:
            msg = (
                "drop_observation_after_s "
                f"({self.drop_observation_after_s}) must be >= "
                f"max_observation_staleness_s ({self.max_observation_staleness_s}); "
                "an observation cannot be dropped before it is even considered stale."
            )
            raise ValueError(msg)

        if self.sim_step_s > self.sim_horizon_s:
            msg = (
                f"sim_step_s ({self.sim_step_s}) exceeds sim_horizon_s "
                f"({self.sim_horizon_s}); a simulation of less than one step "
                "produces no state to report."
            )
            raise ValueError(msg)

        steps = math.ceil(self.sim_horizon_s / self.sim_step_s)
        if steps > self.sim_max_steps:
            msg = (
                f"sim_horizon_s / sim_step_s is {steps} steps, above the "
                f"sim_max_steps ceiling of {self.sim_max_steps}. Raise the ceiling "
                "deliberately or coarsen the step."
            )
            raise ValueError(msg)

    # --- Derived ----------------------------------------------------------
    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        """True when running in an environment where debug affordances must be off."""
        return self.app_env is AppEnv.PRODUCTION

    @computed_field  # type: ignore[prop-decorator]
    @property
    def genai_configured(self) -> bool:
        """Whether GenAI credentials are present. ``False`` until Phase 12."""
        return self.openrouter_api_key is not None and bool(self.openrouter_model)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the settings cache (tests only)."""
    get_settings.cache_clear()
