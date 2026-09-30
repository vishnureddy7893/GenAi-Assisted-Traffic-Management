# Datasets

Real-world captured traffic data. **Nothing in `raw/`, `interim/` or
`processed/` is tracked by Git** — corpora are expected to exceed 5 GB, and
binary video does not belong in version control.

```
raw/         as downloaded, never modified
interim/     validated and preprocessed
processed/   feature-engineered, ready for training
manifests/   TRACKED: checksums and provenance descriptors
```

## Manifests are the tracked artifact

A dataset is reproducible if its *identity* is recorded, not its bytes. Each
manifest in `manifests/` will record source URL and licence, retrieval date,
file checksums, row/frame counts, and the temporal range covered — enough for
another person to obtain the identical corpus and verify they have it.

Point `DATASETS_DIR` in `.env` at wherever the bytes actually live; it does not
have to be this directory.

## Phase 2 comes before feature design

Dataset inspection is deliberately the phase immediately after the foundation,
ahead of computer vision, feature engineering and model selection. What gets
inspected, before anything is built on it:

- structure, columns and units
- what the labels and ground truth actually are
- temporal properties: sampling interval, gaps, seasonality
- missingness, and whether it is random or systematic
- quality: sensor faults, duplicated rows, impossible values

Designing features first and discovering afterwards that the dataset lacks the
ground truth they assume is the most common way a project of this shape loses a
month.

## `networks/` — surveyed road graphs

Road network files loaded by `POST /api/network/load`, named relative to this
directory. Unlike the bulk corpora above, these are small JSON documents and are
tracked in Git.

`example-grid.json` is a **synthetic** four-junction grid, hand-authored for
testing and demonstration. Its provenance records `ASSUMED` and says so in
`notes`: every coordinate, length and lane count is invented, and no route
planned over it describes real geography. Nothing loads it automatically —
`NETWORK_FILE` is unset by default, so a deployment cannot end up routing over
synthetic geography without someone having chosen it.

### Observation mappings

`example-grid-mapping.json` demonstrates the observation-to-network binding format
that Phase 7 requires before any traffic evidence can affect a route cost. It is
**synthetic**: its `observation_area_id` values are placeholders that match no
analysed media, and its provenance records `ASSUMED` and says so.

A real binding uses the `observation_area_id` Phase 5 actually reports — for
ingested media that is `observation-area:<media_id>` — and is a survey fact:
somebody looked at the camera and recorded which road it watches. Nothing infers
it, and an unbound observation affects no road, by design.
