# AVA stage diagnostics

[Dataset navigation](README.md) · [setup and video-aligned preparation](BENCHMARK_GUIDE.md)

## Scope and fixed protocol

Actual AVA ActiveSpeaker validation video `053oq2xB3oU`, original seconds `[900,910)`,
source frames 22500–22749, 480x360 at 25 FPS. Native runs have 250 sampled frames and
1,020 labeled face instances; 10 FPS runs have 100 frames and 408 labeled instances.
The registered six conditions use original/640x480/320x240 at native/10 FPS.

Measurements use the unchanged MediaPipe + aligned SFace + DeepSORT baseline, configured
confidence 0.65, `n_init=3`, memory three seconds and evaluation/initialization IoU 0.5.
The measured implementation revision is `6723bf7`; each run records the full source,
video, annotation/model hashes and environment. Predictions in all six conditions were
compared with the previously recorded baseline and are identical apart from timing.
No thresholds were trained, tuned or selected on this interval.

## What each stage means

Normal inference never receives evaluation GT. The optional backend captures only
detector boxes/scores, successful descriptor boxes, observations and timing. Scoring
joins that telemetry to annotations afterward.

| Stage | Meaning |
| --- | --- |
| Raw detector probe | Separate MediaPipe call requesting a 0.01 score floor, after its internal postprocessing/NMS. This is not pre-NMS or truly unthresholded neural-network output. |
| Baseline detector output | Actual boxes returned by the unchanged configured-0.65 detector invocation and passed to SFace. |
| Confidence accepted | Explicit returned-score >= 0.65 subset, for diagnosis only. It is not substituted into the baseline. |
| Descriptor success | Actual SFace-alignable/encoded boxes that reach DeepSORT. |
| Tentative tracks | Unconfirmed DeepSORT observations. |
| Confirmed tracks | All confirmed tracks, including predicted-only boxes; count only, detection recall unavailable. |
| Confirmed observed | Confirmed tracks with current detections; the original benchmark's usable-output policy. |
| Spatial matches | One-to-one IoU-gated matches of a specified stage to provided GT. |

Stage recall uses provided visible GT instances as denominator, never all detected
faces. Unknown annotation frames are excluded. Precision/FP counts and full identity
metrics remain unavailable because exhaustive coverage has not been established.

## Original-resolution native-FPS counts

| Stage | Output box instances | GT matches at IoU >= 0.5 | Recall / 1,020 GT instances |
| --- | ---: | ---: | ---: |
| Raw returned probe | 753 | 215 | 21.08% |
| Actual baseline detector output | 753 | 215 | 21.08% |
| Explicit confidence-eligible diagnostic subset | 701 | 198 | 19.41% |
| Successful descriptors | 745 | 215 | 21.08% |
| Tentative observations | 13 | 8 | 0.78% |
| Confirmed tracks, including predictions | 995 | unavailable | unavailable |
| Confirmed current observations | 732 | 207 | 20.29% |

Of the 813 final missed labels, 805 already lack a baseline box meeting IoU 0.5;
eight have observed boxes that remain unconfirmed. Eight boxes are dropped during
appearance extraction, but none of those supplied an additional IoU-0.5 GT match in
this condition. Association/confirmation removes eight of the 215 qualifying matches.
These are operational spatial-stage diagnostics; per-stage one-to-one assignments can
differ, so their differences are not a general causal partition for arbitrary videos.

### Confidence behavior must be measured

In this installed pinned stack, both detector calls returned the same 753 candidates,
with minimum reported score about 0.50068. **52 baseline-returned candidates have scores
below the configured 0.65 value.** Thus the configured API argument must not be treated
as a verified hard floor on the returned scores. This report does not assert an
unverified explanation inside MediaPipe's graph/postprocessing.

An explicit score filter is therefore reported separately. Post-filtering the probe
at 0.1/0.3/0.5/0.65/0.8 yields 215/215/215/198/143 GT matches at IoU 0.5. This is a
diagnostic sensitivity analysis, not separate operational detector runs or automatic
optimization. No filter was inserted into the fixed baseline and no new default was chosen.

### Spatial extent explains much of the strict-gate loss

With the same baseline boxes, detector-only GT matches are 752 at IoU >= 0.3, versus
215 at >= 0.5 and 21 at >= 0.7. The reported benchmark gate remains 0.5.

For the 752 one-to-one pairs at the diagnostic 0.3 gate, median predicted/GT area is
0.46582, median width ratio 0.74372 and median height ratio 0.61479. Median center
error divided by sqrt(GT area) is 0.16639. Example overlays show a difference in box
extent/placement between MediaPipe and the provided labels. Lower overlap at 0.5
cannot simply be described as no detected face, nor as an SFace or ID-association failure.

The evidence points primarily to detector-to-annotation spatial coverage on this
interval. It does not prove all missing labels are annotation errors or that MediaPipe
classification alone is responsible; the probe cannot expose all internal network candidates.

## Reproducible initialization failure

The one registered attempt occurs at **source frame 22502**, timestamp **900.08 s**.
The designated GT box is approximately `[161,26,344,240]`. A confirmed observed ID 1
exists, with box approximately `[195.844,96.100,321.207,221.460]`. Original precision is
retained in the JSON, not rounded for scoring.

Its best baseline/descriptor overlap is **0.4012948**, below the fixed **0.5** gate.
The confirmed observation has the same overlap. The failure category is
`raw_detection_below_initialization_iou_gate`: a face box exists and is confirmed,
but it does not qualify spatially. It is not a video-opening failure, missing-descriptor
failure or tentative-track failure. No future frame was used to retry designation.

`initialization_failure.json` retains the exact frame, target box, candidate observations,
stage detections, maximum overlaps and configuration. `initialization_frame.png` is
generated afterward. Other supported categories cover no raw overlap, probe-only
below-confidence candidates, appearance loss, still-unconfirmed tracks, no matching
prediction and no eligible annotation frame for the policy. Multiple plausible tracks
are recorded, but no nonexistent ambiguity-rejection rule is claimed. Input alignment
errors abort the run and produce `alignment_failure.json` when raised by the decoder checks.

## Identical-interval comparisons

| Condition | GT instances | Baseline box matches | Descriptor matches | Confirmed observed matches |
| --- | ---: | ---: | ---: | ---: |
| Original / native | 1,020 | 215 | 215 | 207 |
| Original / 10 FPS | 408 | 87 | 87 | 79 |
| 640x480 / native | 1,020 | 210 | 210 | 202 |
| 640x480 / 10 FPS | 408 | 86 | 86 | 78 |
| 320x240 / native | 1,020 | 226 | 225 | 217 |
| 320x240 / 10 FPS | 408 | 89 | 89 | 80 |

All six initialization attempts fail the spatial gate; best target overlaps range
approximately 0.388–0.402. These comparisons do not select an optimal resolution or
threshold, and do not generalize ten seconds into full AVA accuracy.

## Repeat and inspect

After the benchmark guide's AVA preparation, from PowerShell at the repository root:

```powershell
.\.venv\Scripts\python.exe -m c3_tracker.evaluate --dataset ava --data-root datasets/data --output results/ava_diagnostics_repeat --matrix --stage-diagnostics
```

Use a fresh output directory. Each condition adds `diagnostic_stages.csv`,
`stage_detections.csv`, `stage_snapshots.jsonl`, `stage_metrics.json`,
`initialization_diagnostics.json`, optional `initialization_failure.json`, a diagnostic
report and post-inference example PNGs. Existing prediction/metric outputs remain available.
Baseline detector boxes in `stage_detections.csv` provide detector-only output on the
exact same frames as full tracking. No synthetic track identities are assigned to them.

Instrumented latency/FPS includes the extra probe. Snapshot fields separately record
baseline detector+tracking time and probe time; do not present instrumented throughput
as the uninstrumented baseline's speed. No descriptors/embedding vectors are exported.

For original/native on the recorded Windows/Python 3.11 CPU setup, instrumented
throughput was 8.92 FPS; mean/p95 selected-frame latency was 111.97/151.73 ms.
Detector+tracking time averaged 108.92 ms and the separate probe 2.74 ms. Exact
hardware, package versions and definitions are retained in `performance.json`.

Local visual review checked `initialization_frame.png` and
`example_track_confirmation_loss_22510_3.png` under
`results/ava_stage_diagnostics_final/original_nativefps/`. They show the smaller
confirmed face box at initialization and a separate confirmation-loss example.

Actual outputs are local under ignored `results/ava_stage_diagnostics_final/`. The
geometry/continuity analysis is recorded in `results/benchmark_improvements_validation/`.
MobiFace's distinct selected-ID continuity result is documented in [its guide](mobiface/README.md).
