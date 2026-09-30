# GenAI-Assisted Traffic Management System

Decision support for multi-junction traffic management and emergency vehicle
routing, driven by **real-world captured traffic data** replayed in a controlled
environment.

> **Advisory only.** This system analyses recorded and simulated traffic and
> produces recommendations for human operators. It has no control path to any
> real traffic signal, vehicle or piece of infrastructure, and none may be added.

**Current status: Phase 8 (Traffic Simulation) complete.** Recorded images
and videos are ingested and analysed with a **pretrained** YOLO detector and a
ByteTrack-style tracker, producing detections and stable track identities. Those
tracks are turned into measurements of one observation area: counts, spatial
distribution, trajectories and directions, image-space speeds, flow over time,
arrival rate and queue evidence. Metrics in real-world units — density in
veh/km/lane, speed in km/h, queue length in metres — require a camera
calibration and report as unavailable with a reason when there is none.

Detections are not counted merely because they were made. A quality layer sorts
every box into *confirmed*, *provisional* or *rejected* against configurable
rules — cross-class duplicates, a scene-scale area ceiling, frame-edge
truncation, per-class confidence floors — and only the confirmed tier reaches
the measurement pipeline. Nothing is deleted: an excluded box is still returned,
still drawn (dashed, and labelled), and still carries the reason it was
excluded, because a filter nobody can see is a filter nobody can correct.

Those measurements are then **interpreted** into a congestion level per
observation area, using configurable thresholds, and fused into a snapshot under
an explicit staleness budget. Every verdict carries the signals, scores and
threshold set behind it, so it can be recomputed by hand. Where the evidence does
not support a band, the answer is `unknown` with a reason — never a default.

A surveyed **road network** can now be loaded and validated: referential
integrity is checked at load with the offending id named, and the file's SHA-256
is recorded beside the version it declares. **Incidents** and **declared
emergency responses** are tracked against it, each with an explicit lifecycle
whose terminal states are terminal. An incident's exclusion zone is resolved once
onto graph elements — measured to each segment, not merely to its endpoints, so
the road running straight past a fire is excluded too — and anything that could
not be tested is reported rather than skipped.

Routes are then **optimized** over that network: hard constraints prune the graph
before any path is considered, a movement-aware Dijkstra ranks what remains by a
risk-adjusted arrival time in equivalent seconds, and Yen's algorithm supplies
the alternatives so "why not the other way?" has an answer with costs attached.
Traffic evidence reaches a road **only** through an explicit
observation-to-network mapping; unbound evidence is named in the result and
priced at nothing.

Those routes can then be **simulated**. A deterministic point-queue model
evolves each segment's exit queue over a configured horizon and drives two
routes — a baseline and the recommended one — through the *same* conditions, so
a difference between them belongs to the routes rather than to the weather. It
reports travel time, delay, queue accumulation, throughput, route completion
time and the improvement between the alternatives. Every one of those figures
carries `SIMULATED` provenance and can never be stated as a measurement of a
real road. Demand and capacity are surveyed, observed through an explicit
binding, or stated as a justified assumption — never defaulted — and a segment
with none of the three is reported *unmodelled* with the reason. The model has
no random component and no seed: identical inputs produce an identical run,
addressed by the SHA-256 fingerprint of its own scenario.

No model has been trained by this project, and no route is approved or executed
— the optimizer recommends, the simulator evaluates, and both stop there. Where
every path is cut by a hard constraint the answer is `no_admissible_route` with
its causes, never a downgraded unsafe route. Every capability belonging to a
later phase returns HTTP **501** naming that phase, rather than an empty success
— see [Honesty by construction](#honesty-by-construction).

---

## Quick start

Two terminals. Backend first.

### Backend

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python.exe -m pip install -e "backend[dev,vision]"
```

The `vision` extra adds the pretrained YOLO backend. Omit it for an
ingestion-only deployment: the application probes for detector backends at
runtime and reports detection as unavailable rather than failing to start.

```bash
cd backend && ..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

API docs at <http://127.0.0.1:8000/docs>.

### Frontend

```bash
cd frontend && npm install
```

```bash
npm run dev
```

Dashboard at <http://localhost:5173>. The dev server proxies `/api` to the
backend, so CORS is not involved in local development.

### Configuration

```bash
copy .env.example .env
```

Every setting is read through `backend/app/core/config.py`. Secrets are typed as
`SecretStr` and are redacted from logs, reprs and API responses. **No secret is
ever hard-coded**, and `.env` is git-ignored.

---

## Verification

```bash
.venv\Scripts\python.exe -m pytest backend -q
```

```bash
.venv\Scripts\python.exe -m ruff check backend
```

```bash
cd frontend && npm run typecheck
```

```bash
cd frontend && npm test
```

---

## Architecture

```
        RECORDED REAL-WORLD DATA              LIVE TRAFFIC
        (video / datasets / counts)        (RTSP / HTTP / webcam)
                    │                              │
                    │   Mode A                     │   Mode B
                    │   bounded, replayable        │   unbounded, lossy
                    │   VirtualClock               │   WallClock
                    └──────────────┬───────────────┘
                                   │
                    Detection → Tracking → Extraction
                                   │
                        JunctionObservation          ← one source, one instant
                                   │
                    fusion under a staleness budget
                                   │
                          NetworkState                ← whole network, one instant
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
        RoadNetwork          EmergencyState         Incidents
        (static graph)       (declared event)       (+ exclusion zones)
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   │
                      HARD safety constraints
                      prune the search graph
                                   │
                       Route search over the
                       admissible subgraph
                                   │
                      SOFT constraints price
                      routes in equivalent seconds
                                   │
                          RouteComparison
                                   │
                        Controlled simulation
                        (baseline vs recommended)
                                   │
                    GenAI narrative over a fact table
                    (every number validated mechanically)
                                   │
                          DecisionRecord              ← fully reproducible
```

Recorded and live inputs **converge** at `JunctionObservation`; everything
downstream is agnostic to which produced it.

Full design rationale: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
Decision records: [`docs/adr/`](docs/adr/) — including
[the detector configuration](docs/adr/0012-detector-configuration.md) and
[detection quality filtering](docs/adr/0013-detection-quality-filtering.md), both
selected against measured counts on a real junction frame.

---

## Honesty by construction

The requirement "do not fabricate values" is enforced by the type system and the
API contract, not by discipline.

| Mechanism | Where | What it prevents |
|---|---|---|
| `Measurement` requires `Provenance` | `domain/provenance.py` | A traffic number entering the system without stating its origin |
| No `MOCK`/`DEFAULT` in `ValueSource` | `domain/provenance.py` | A placeholder masquerading as a measurement |
| Absent is `None`, never `0` | every domain model | An unmonitored gridlocked road reading as empty |
| Metric fields rejected without calibration | `domain/observation.py` | Pixels/second presented as km/h |
| Emergency ETA independent of nominal | `domain/state.py` | The constant-ratio shortcut, wrong exactly where it matters |
| Emergency ETA requires stated assumptions | `domain/state.py` | An unconditional ETA overstating certainty |
| Hard constraints carry no penalty | `domain/safety.py` | A fire exclusion zone being "outbid" by a faster route |
| Blockage requires a causing incident | `domain/state.py` | An unexplainable, unclearable closure |
| 501 instead of empty 200 | `api/routes/` | A blank dashboard reading as "no traffic" |
| Unclassifiable is `UNKNOWN`, never a band | `state/classifier.py` | A missing signal defaulting toward "free flow" |
| Unavailable signals carry zero weight | `state/schemas.py` | An absent measurement scoring as 0 and diluting the verdict |
| Referential integrity checked at load | `domain/network.py` | A dangling segment id failing inside a live emergency request |
| Zone distance measured to the segment | `emergency/zones.py` | The road past a fire staying admissible because its endpoints are outside |
| Untestable zone elements are named | `emergency/zones.py` | A hard constraint that looks resolved and covers less than it claims |
| Terminal lifecycle states are terminal | `emergency/incidents.py` | One incident id holding two answers to "was this blocked at 14:05?" |
| Pruning happens before search | `routing/pruning.py` | A blocked segment appearing in a candidate at all |
| Unbound evidence changes no cost | `routing/mapping.py` | A camera's verdict attributed to a road nobody surveyed it against |
| No time basis means no ETA | `routing/costing.py` | A partial sum, or an invented km/h, presented as an arrival time |
| Simulated outputs carry `SIMULATED` | `simulation/runner.py` | A modelled outcome read as a measurement |
| No default demand, capacity or speed | `simulation/scenario.py` | A confident simulation of data nobody has |
| Unmodelled links are named, not zeroed | `simulation/schemas.py` | An unsurveyed road reading as an empty one |
| Past the horizon means no travel time | `simulation/runner.py` | A truncated journey presented as an arrival |
| An unavailable arm yields no improvement | `simulation/runner.py` | A one-sided figure sold as a benefit |
| Run id is the input fingerprint | `simulation/scenario.py` | An unreproducible comparison |
| Detection settings travel with the count | `vision/analysis.py` | A tally read without the threshold that produced it |
| Only confirmed detections are counted | `vision/quality.py` | A building detected as a bus becoming traffic |
| An excluded detection is reported, not deleted | `vision/quality.py` | A wrong verdict nobody can see to dispute |
| Uncountable boxes are drawn dashed | `vision/overlay.py` | A picture that argues for a count it is not in |
| Fact table + numeric scan | `domain/decision.py` | An LLM inventing an ETA in prose |

`backend/tests/test_architecture.py` enforces several of these as executable
guardrails.

---

## Layout

```
backend/          FastAPI decision-support service
  app/core/       config, structured logging, clocks, errors, versions, phases
  app/domain/     schemas: provenance, network, observation, site, state,
                  incident, emergency, safety, routing, decision
  app/ingestion/  TrafficDataSource abstraction, Mode A / Mode B contracts
  app/vision/     detector/tracker interfaces, pretrained YOLO backends,
                  ByteTrack-style tracker, detection quality assessment,
                  overlay rendering, analysis jobs
  app/analytics/  calibration, geometry, analytics engine, state extractor
  app/state/      congestion classifier, interpreted state, snapshot assembly
  app/network/    network loading, the loaded-network store, state assembly
  app/emergency/  exclusion zones, incident and emergency lifecycles
  app/routing/    observation mapping, effective-mapping composition and
                  content hashing, evidence attribution, pruning, costing,
                  search, optimizer
  app/simulation/ point-queue model, scenario assembly, runner, run store
  app/sites/      observation site registry, nearest-junction and incident-
                  segment candidates, human-confirmed junction and directed-
                  segment bindings
  app/advisor/    AI Traffic Advisor: fact contract, LLM provider behind an
                  interface, strict prompt, post-generation validation and a
                  deterministic fallback
  app/api/        HTTP layer
  tests/          1268 tests
frontend/         React + Vite + TypeScript operator dashboard
training/         preprocessing, feature engineering, training, evaluation
models/           model weights (git-ignored; identity reported by checksum)
simulation/       scenario notes (the model itself lives in backend/app)
datasets/         raw / interim / processed (git-ignored) + manifests, networks,
                  sites (surveyed junction locations, written at runtime)
docs/             architecture and decision records
```

---

## Development phases

Two ordering decisions are deliberate. **Dataset inspection is part of Phase 2**
— before any computer vision, feature design or model selection, because every
one of those decisions is contingent on what the data actually contains. And
**traffic analytics (Phase 4) precedes traffic state (Phase 5)**, because state
interprets exactly what analytics produces; building the interpretation layer
first would mean interpreting a value type that did not exist yet. The surveyed
road graph arrives in Phase 6 for the same reason: it is the first phase that
cannot work without one, since an incident names the segments it blocks and a
dispatch names the junction it starts from.

| | Phase | |
|---|---|---|
| 1 | Foundation | ✅ complete |
| 2 | Data Ingestion | ✅ complete |
| 3 | Computer Vision | ✅ complete |
| 4 | Traffic Analytics | ✅ complete |
| 5 | Traffic State | ✅ complete |
| 6 | Road Network + Emergency Management | ✅ complete |
| 7 | Route Optimization | ✅ complete |
| 8 | Traffic Simulation | ✅ complete |
| 9 | Decision Engine | |
| 10 | ML Models where justified | |
| 11 | Safety & Validation | |
| 12 | GenAI Explanation | |
| 13 | Evaluation & Integration | |
| 14 | Live Traffic Extension | |

Live status is served from `/api/meta/phases` and rendered on the dashboard, so
the documentation and the running system cannot drift apart.

---

## Data and models are not in Git

Raw traffic corpora are expected to exceed 5 GB and model artifacts are binary.
Neither belongs in version control. `datasets/manifests/` tracks checksums and
provenance descriptors; the bytes live on local disk, pointed at by
`DATASETS_DIR` and `MODELS_DIR`.
