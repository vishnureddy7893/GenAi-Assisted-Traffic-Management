# Training

ML pipeline. **Phase 10.** Empty until then — no model has been trained, and
none will be before the dataset is inspected.

```
preprocessing/        validation and cleaning of raw data
feature_engineering/  feature construction from TrafficState records
train/                model training
evaluation/           honest evaluation against held-out data
```

## Pipeline

```
Raw dataset → Validation → Preprocessing → Feature extraction
  → TrafficState dataset → Feature engineering
  → Temporal train/validation/test split → Training → Evaluation
  → Saved artifact → Application integration
```

## Rules fixed in advance

**Splits are temporal, never random.** Traffic is strongly autocorrelated in
time. A random split places a 08:31 sample in training and 08:32 in test, which
leaks the answer and produces a score that will not survive deployment.

**No target leakage.** A travel-time model must not receive a feature computed
from the arrival it is predicting. This is easy to do accidentally when features
are derived from the same aggregation window as the label.

**A baseline is mandatory.** Historical mean by time-of-day and day-of-week is
the benchmark. A gradient-boosted model that does not beat it is not a result.

**Predictions carry `PREDICTED` provenance and a confidence.** They are never
rendered identically to measurements, and the decision pipeline must remain
correct when no model is loaded — in which case predicted values are absent, not
defaulted.

**Model choice follows the data.** Algorithm selection happens against what
Phase 2 established about the dataset, not ahead of it.
