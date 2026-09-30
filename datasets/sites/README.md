# Observation sites

Two runtime-written registries live here.

`observation-sites.json` holds the surveyed location of the junction behind each
observation area: the answer to "where was this footage filmed?".

It is written by the service itself — through `POST /api/sites`, or by supplying
`latitude` and `longitude` alongside a media upload — rather than hand-authored
like the files in `datasets/networks/`. That is the one thing that makes it
different from every other data file in this repository.

## Why it is not in the media directory

A media asset describes a *file*; a site describes a *place*. One junction
outlives many uploads of it, so tying the two lifetimes together would mean
deleting footage deleted the survey, and two files of the same junction could
disagree about where it is. Clearing `datasets/media/` leaves this file alone.

## Why the file is not committed

The registry is keyed by `observation_area_id`, and in practice that is
`observation-area:<media_id>` — a transient id that means nothing in another
checkout. Committing it would pin coordinates to media that no longer exists,
which is the same mistake as committing a generated observation mapping. The
file is git-ignored; this README is not.

A survey you *do* want to keep belongs in version control as a deliberate
artifact, the way `datasets/networks/example-grid-mapping.json` is, with
provenance notes saying who recorded it.

## Format

```json
{
  "schema_version": "1",
  "sites": [
    {
      "observation_area_id": "observation-area:<media_id>",
      "location": { "latitude": 16.5062, "longitude": 80.648 },
      "bearing_degrees": 135.0,
      "accuracy_m": 8.0,
      "label": "Benz Circle",
      "provenance": { "source": "observed", "...": "..." },
      "recorded_at": "2026-09-17T04:25:51Z",
      "updated_at": "2026-09-17T04:25:51Z"
    }
  ]
}
```

Writes replace the whole file atomically. If the file exists but will not parse,
the service refuses to read *or* write it and says so with a 502 — an unreadable
registry is never reported as an empty one, because the first successful write
would then destroy whatever it still contains.


---

## `confirmed-bindings.json`

Human-confirmed site-to-junction bindings: which intersection an operator
decided each camera watches, on which network, with the distance they were shown
and whether they took the nearest junction or overrode it.

A record may additionally carry a **segment refinement** — `segment_id`,
`segment_direction` and a second `segment_confirmation` — recording which
directed road at that junction the camera monitors. The junction confirmation is
kept alongside it rather than overwritten: they are two acts, potentially by
different people, and each is worth reviewing on its own.

A separate file from the sites registry on purpose. A site says *where a camera
is*; a binding is a decision *about* that site. Correcting a coordinate must not
silently rewrite the decision made against the old one, and one file holding
both would make that impossible to keep straight.

Nothing writes to it except an explicit confirmation through
`POST /api/sites/{id}/binding` or `POST /api/sites/{id}/binding/segment`.
Computing the nearest junction (`GET /api/sites/{id}/candidate`) or listing the
segments at it (`GET /api/sites/{id}/segments`) never touches it — see
[ADR-0015](../../docs/adr/0015-nearest-junction-proposes-a-human-binds.md) and
[ADR-0016](../../docs/adr/0016-a-segment-refines-a-junction-binding.md).

Neither kind of binding changes any route cost. `GET /api/sites/confirmed`
reports `affects_route_cost: false` with the reason.

```json
{
  "schema_version": "1",
  "bindings": [
    {
      "observation_area_id": "observation-area:<media_id>",
      "junction_id": "J1",
      "network_id": "example-grid",
      "network_version": "1.0.0",
      "site_location": { "latitude": 16.5051, "longitude": 80.6001 },
      "confirmation": {
        "confirmed_by": "operator.dashboard",
        "method": "accepted_nearest",
        "candidate_distance_m": 15.4,
        "was_nearest_candidate": true
      },
      "segment_id": "SEG-21",
      "segment_direction": "incoming",
      "approach_id": "J1-A-SEG-21",
      "segment_confirmation": {
        "confirmed_by": "operator.dashboard",
        "method": "aligned_with_bearing"
      },
      "provenance": { "source": "observed", "...": "..." }
    }
  ]
}
```

Git-ignored for the same reason as the sites file: the keys are transient media
ids that mean nothing in another checkout.


---

## How these files reach routing

Neither file is read by the route optimizer. They are one of the two inputs to
the **effective observation mapping**, composed on request from:

1. the loaded mapping file in `datasets/networks/` — a versioned survey; and
2. `confirmed-bindings.json` — operator confirmations.

A confirmed binding supersedes a file binding for the same observation area; an
unopposed binding stands; agreement is recorded as agreement; and a confirmation
made against a different network or a stale network version is excluded and
reported rather than migrated.

The composition is derived on every request and never stored, so it cannot drift
from these files. `GET /api/sites/effective-mapping` returns it along with a
deterministic `effective_mapping_hash` that identifies exactly which bindings
were in force.

It does **not** change route costs. The optimizer still reads the mapping file
alone; see
[ADR-0018](../../docs/adr/0018-effective-observation-mapping.md) for why the
identity was built before the connection.
