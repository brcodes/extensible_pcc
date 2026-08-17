# Model KPI Guide

This file describes the training KPIs printed at the end of a run. These are general model-training diagnostics, not tied to one architecture or dataset.

## Printed KPIs

- `Start`: epoch-0 `Jr`, `Jc`, and `Accuracy`.
- `Final`: final-epoch `Jr`, `Jc`, and `Accuracy`.
- `Best Accuracy`: highest recorded accuracy and the epoch where it occurred.
- `Accuracy delta`: final accuracy minus start accuracy, in percentage points and correct-count change when sample count is known.
- `Best Accuracy delta`: best recorded accuracy minus start accuracy, in percentage points and correct-count change when sample count is known.
- `Final Jr gain`: percent reduction of `Jr` from epoch 0 to the final epoch.
- `Final Jc gain`: percent reduction of `Jc` from epoch 0 to the final epoch.
- `Best Jr`: minimum recorded `Jr`, the epoch where it occurred, and its gain relative to epoch 0.
- `Best Jc`: minimum recorded `Jc`, the epoch where it occurred, and its gain relative to epoch 0.
- `Jc rebound from best to final`: how much `Jc` worsened after its best epoch, normalized by `Jc0`.
- `Jc max blowup vs start`: worst `Jc` increase over the run, normalized by `Jc0`.
- `Late-window slopes`: average per-epoch change in `Jr`, `Jc`, and `Accuracy` over the final window.

## How To Read Them

### Start vs Final

- Use `Start` and `Final` first to see the actual scale of change.
- Do not infer training quality from normalized gains alone; always compare them against absolute final values.

### Final Jr gain and Final Jc gain

- Positive gain means the loss decreased from epoch 0.
- Near-zero gain means the metric barely moved.
- Negative gain means the metric ended worse than it started.
- Large `Jr` gain with poor `Jc` gain often means the model learned representation structure without preserving classification quality.

### Best Jr and Best Jc

- These tell you the best point reached anywhere in training, not only where the run ended.
- If `Best Jc` occurs much earlier than the final epoch, the run may want early stopping or a schedule change rather than longer training.
- If `Best Jr` and `Best Jc` happen in very different epochs, the optimization objectives may be competing.

### Accuracy metrics

- Use raw final accuracy and best accuracy more than derived percent changes.
- Accuracy is discrete and often low-resolution on small datasets, so correct-count changes are more interpretable than normalized ratios.
- If `Best Accuracy` improves but `Final Accuracy` falls back, the model is unstable late in training.

### Jc rebound from best to final

- Small rebound means the classifier stayed near its best point.
- Large rebound means classification performance deteriorated after initially improving.
- This is often a stronger early-stopping signal than final `Jc` alone.

### Jc max blowup vs start

- This captures the worst classification instability during training.
- A very large blowup suggests the run entered a bad region even if it later partly recovered.
- Use this to reject unstable runs that look acceptable only at the final epoch.

### Late-window slopes

- Negative `Jr` slope means `Jr` is still decreasing late.
- Negative `Jc` slope means `Jc` is still decreasing late.
- Positive `Jc` slope means `Jc` is worsening late and may already be reversing.
- Near-zero slopes mean the metric has mostly plateaued.
- Use slopes together with rebound: a large rebound plus positive late `Jc` slope is a strong sign of instability.

## Good General Patterns

- `Final Jr gain > 0` and `Final Jc gain > 0`.
- `Best Jc` close to the final epoch.
- Small `Jc rebound from best to final`.
- `Late-window Jc slope <= 0` or very close to zero.
- `Best Accuracy` and `Final Accuracy` at or above baseline.

## Warning Patterns

- `Final Jc gain < 0`.
- `Jc max blowup vs start` is large.
- `Best Jc` occurs early and `Jc rebound from best to final` is large.
- `Late-window Jc slope > 0`.
- `Jr` improves a lot while `Accuracy` stays flat or drops.

## What These KPIs Do Not Tell You

- They do not explain why the training dynamics behave that way.
- They do not replace activation, calibration, or representation-quality diagnostics.
- They do not remove the need to compare multiple seeds when selecting a setting.
