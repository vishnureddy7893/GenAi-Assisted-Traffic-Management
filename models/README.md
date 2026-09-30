# Models

Trained artifacts. **Not tracked by Git** — binaries do not diff, and a model
file in version control quickly becomes indistinguishable from every other
version of itself.

## Phase 3: pretrained detection weights

`yolov8s.pt` (~22 MB) is downloaded here on first use by
`app/vision/models.py:ensure_weights`. It is **pretrained COCO weights, not
trained by this project** — no training of any kind happens in this repository.

It replaced `yolov8n.pt` after a measured comparison on a real junction frame:
nano labelled six vehicles `bus` that were a saloon, a minivan, a delivery van
and a lorry, while small got them right. The reasoning and the numbers are in
[`docs/adr/0012-detector-configuration.md`](../docs/adr/0012-detector-configuration.md).
Both remain selectable through `DETECTOR_MODEL`.

Downloads are written here and nowhere else. Ultralytics would otherwise write
to the process working directory, scattering weight files through the source
tree and eventually into a commit. `.gitignore` excludes `*.pt` and `models/**`.

Every inference result reports the **SHA-256 of the weights actually loaded**,
not just the configured name: two runs of "yolov8s" can use different bytes, and
only a checksum makes that visible.

No *trained-by-us* model exists, so `/api/meta/versions` still reports
`model_versions: {}` — the honest value, not a placeholder version string.

## Registry

Each artifact will be recorded with: role (`travel_time`, `congestion`,
`queue_growth`), version, training dataset manifest id, feature set version,
evaluation metrics with the split they were measured on, and a checksum.

`RunManifest.model_versions` pins the exact artifacts that contributed to a
decision, so a change in routing behaviour can be attributed to a model change
rather than confused with a rule change or a data change.

## Model selection is a Phase 10 decision

No algorithm is chosen here. XGBoost, random forests, gradient boosting and
temporal models are all candidates, and the choice depends on properties of the
data that Phase 2 established — sample size, feature/label structure, temporal
autocorrelation, missingness.

Reinforcement learning is a candidate only for signal control policy, only if a
validated simulator exists to train against, and only if a measured baseline
shows the problem genuinely warrants it. RL against an unvalidated simulator
optimises the simulator's flaws.

Splits must be **temporal**, never random: traffic is autocorrelated, so a random
split leaks the future into training and produces an evaluation score that will
not survive deployment.
