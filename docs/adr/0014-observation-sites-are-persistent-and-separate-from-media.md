# ADR 0014 — An observation site is persistent, and separate from the media

**Status:** accepted (Phase 8+, step 1 of real-world routing)

## Context

An `observation_area_id` identifies one camera's field of view. It is what
analytics keys its measurements by, what Phase 5 keys its congestion verdicts
by, and what `ObservationMapping` binds to a road segment. It says nothing about
geography, so nothing in the system can place a junction on a map.

Real-world routing will need that position. Before any of it, the position has
to exist somewhere and be trustworthy.

Two obvious shortcuts were available and both were rejected.

## Decision

A new `ObservationSite` records the WGS-84 position of the junction behind an
observation area, keyed by the existing `observation_area_id` and carrying
`Provenance` like every other asserted value in the system. Optional
`bearing_degrees`, `accuracy_m` and `label` are recorded when surveyed and left
`None` when not.

It is held in `ObservationSiteRegistry`, which **persists to a JSON file**
(`SITES_FILE`, default `datasets/sites/observation-sites.json`) outside the
media directory, and is **not** part of `MediaStore`. `MediaAsset` gains no
geographic field.

An upload may carry `latitude`/`longitude`; when it does, the site for
`observation-area:<media_id>` is created or corrected. When it does not, the
upload behaves exactly as before.

## Rationale

**Why not a field on `MediaAsset`.** An asset describes a file; a site describes
a place. One junction outlives many uploads of it. Folding the coordinates onto
the asset would mean deleting footage deleted the survey, and two files of the
same junction could hold two different answers to where it is, with no rule for
which wins.

**Why persistent, when everything else here is in-memory.** Incidents, analytics
results and playback sessions are per-process on purpose: each is cheap to
reproduce and means nothing after a restart. A surveyed location is the
opposite — somebody had to establish it, and a fact that is expensive to
re-establish is one that gets guessed instead. Guessing is the exact failure
`ObservationMapping` already exists to prevent (ADR-0008); the same reasoning
applies to a coordinate.

**Why a JSON file rather than a database.** It is the shape the road network and
the observation mapping already use — a validated document under a configured
directory — so it introduces no new storage dependency, stays reviewable by eye,
and does not commit the project to a schema before the routing work that will
consume it has settled.

**Why re-recording is an update, not a conflict.** A camera gets re-surveyed and
a first coordinate gets mistyped. A correction has to be able to win, so `POST`
to an existing id replaces the location and answers 200 rather than 201, keeping
the original `recorded_at` so the correction is visible as one.

## Consequences

Writes replace the whole file atomically (temporary file in the same directory,
then `os.replace`), so a crash mid-write leaves the previous file intact rather
than truncating away every site to save one.

A registry file that exists but will not parse puts the registry into a degraded
state: reads answer **502**, writes are refused, and the file is never
overwritten. Answering 404 or an empty 200 would present missing knowledge as
established fact, and the first successful write would destroy whatever the file
still contained. This is ADR-0010's rule applied to storage.

A latitude without a longitude is refused (422) rather than half-recorded, and
an upload carrying an invalid location is rejected before any byte is stored, so
a refused upload leaves nothing behind.

Nothing here geocodes, routes or contacts a mapping provider. A site is a
recorded coordinate and its provenance; that is the whole scope.
