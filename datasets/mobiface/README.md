# MobiFace80 native benchmark

[Dataset navigation](../README.md) · [provided Drive folder](https://drive.google.com/drive/folders/1LrVrx_InOo3og9RfCynQkMkfaVydn7ra)
· [source inventory](source.json) · [official project](https://mobiface.github.io/)
· [annotation protocol](https://arxiv.org/pdf/1805.09749)

The native adapter reuses existing benchmark inference, exports, scoring and overlays.
Data/media are not distributed in Git. The supplied toolkit was inspected, not executed;
its train/test folders provide annotation CSVs, not source videos.

## Verified layout and conventions

```text
datasets/data/raw/mobiface/
  mobiface80.meta.csv
  mobiface-toolkit.zip
  mobiface80/
    train.meta.csv
    test.meta.csv
    train/<sequence>.annot.csv
    test/<sequence>.annot.csv
  videos/<matching-media-file>
  manifest.json
```

Metadata columns: `name,youtube_video_id,width,height,frame_rate,first_frame,last_frame`.
Annotation columns: `frame,left,top,width,height`.

The actual release contains **64 train / 16 test** sequences and 94,910 annotation rows:
75,874 visible boxes and 19,036 absent targets. Split files are cross-checked against
combined metadata; paper statistics are not substituted for the supplied release.

Annotation `frame` is zero-based **clip-relative**, checked across all 80 files and the
toolkit's zero-based JPEG enumeration. Metadata first/last frames describe the inclusive
original source interval. `3GHuGGv5tmE_3` ends two frames early and `SZl1J2AZ7Ew_0`
ends 729 early; these 731 frames remain unknown.

Visible xywh becomes `(left,top,left+width,top+height)` in pixels. Exactly four zeros
mean absence. Missing rows remain unknown. Duplicates, missing/nonfinite fields,
invalid boxes, out-of-range clip indices and metadata inconsistencies fail explicitly.
Boxes are never silently clipped, and missing annotations are never filled.

## Acquire and audit

Download permitted metadata/annotations from the provided folder, preserving the layout.
Obtain matching media through an authorized method and retain acquisition, resolution,
frame-clock and offset provenance. From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m c3_tracker.evaluate audit-mobiface --data-root datasets/data/raw/mobiface --output results/mobiface_annotation_audit.json
```

This audits annotations only. It requires all metadata-listed files, loads no models
and makes no video-accuracy claim. Missing files are errors, not silent skips.

## Explicit video alignment

Create `datasets/data/raw/mobiface/manifest.json`:

```json
{
  "dataset": "mobiface",
  "split": "test",
  "data_root": ".",
  "videos": [{
    "video_id": "7I5t6BAHSGQ_0",
    "video_path": "videos/7I5t6BAHSGQ_prefix60.webm",
    "video_layout": "full_source",
    "cfr": true,
    "alignment_provenance": "Record actual acquisition and multiple metadata-offset overlay checks here"
  }]
}
```

| Layout | Required media and index mapping |
| --- | --- |
| `clip` | Frame zero is metadata `first_frame`; decoded count equals `last_frame-first_frame+1`. Annotation index maps directly to local video frame. |
| `full_source` | Media starts at original frame zero and includes the annotated range; a sufficient prefix is allowed. Local index = metadata `first_frame` + annotation index. |

Dimensions/FPS must match native metadata. `cfr: true` and `alignment_provenance` are
explicit assertions; positive decoder timestamps are also checked. Changed media, VFR
conversions or guessed trims need audited remapping. Preparation may use annotations
to check alignment; future GT is never supplied to the tracker.

```powershell
.\.venv\Scripts\python.exe -m c3_tracker.evaluate prepare-mobiface --manifest datasets/data/raw/mobiface/manifest.json --output datasets/data/processed/mobiface
.\.venv\Scripts\python.exe -m c3_tracker.evaluate --dataset mobiface --data-root datasets/data --output results/mobiface_repeat --annotate-video --stage-diagnostics --target-iou-thresholds 0.3,0.5,0.7
```

Use fresh outputs. Preparation writes the common schema, provenance/hashes and official
split. Coverage remains partial because bystanders are not exhaustively labeled.

## Target policy and metrics

After two processed warmup frames, the first labeled-visible target frame is the one
designation opportunity. It must match a current confirmed prediction at the fixed IoU
gate. Failure is recorded without retry; subsequent GT never selects a replacement ID.
This differs from official SOT first-frame-GT-box initialization: these are **not official
MobiFace leaderboard scores**. Default `n_init=3` and tracking thresholds are preserved.

| Measurement | Definition |
| --- | --- |
| Mean best-observed IoU | Best overlap among observed confirmed boxes per labeled-visible sample, zero for no match. Coverage diagnostic, not a GT-assisted output trajectory. |
| Target-box recall | Fraction with qualifying box coverage at each configured IoU threshold. |
| Consecutive coverage / interruptions | Runs, longest run and covered-to-uncovered transitions within contiguous labeled-visible sampled frames; gaps/absence censor and reset continuity. |
| Visible selected-target retention | Fixed selected ID matches at the gate / labeled-visible samples after initialization, excluding the initialization frame. |
| Selected-track interruptions | Covered-to-uncovered transitions for that selected ID, with no replacement selection. |
| Latency/FPS | Existing benchmark timing; diagnostic mode includes an extra detector probe, with baseline detector+tracking time retained separately. |

Fields are under `tracking_metrics.json -> per_video -> <sequence> -> target_localization`.
Detection precision/FP counts, IDF1/HOTA, exhaustive bystander rejection and independently
verified physical-person reacquisition remain unavailable.

## Actual validation on 2026-10-30

All 80 native files passed parsing/geometry/range checks. Test sequence `7I5t6BAHSGQ_0`
was evaluated without threshold tuning. A public format-243 source prefix was stream-
copied from original frame zero through 60 seconds, without resize, seek or re-encoding:
202x360, 30 FPS, 1,800 decoded frames. Annotation frames 0–1091 map to source frames
**391–1482**. Multiple overlays, including later frames, support this declared metadata
mapping; the offset was not fitted to tracker results.

Of 1,092 labeled-visible samples, 982 had observed-confirmed matches at IoU >= 0.5
(**89.93% coverage**). Mean best-observed IoU including misses was **0.59532**.
Initialization succeeded at source frame 393 (native 2), but the selected ID matched
only **88/1,089** later visible samples (**8.08% retention**). Any-track target-box
coverage had 28 interruptions. High localization coverage does not establish stable
selected-target tracking.

Post-inference spatial matches comprise 89 samples on ID 1 (including initialization)
and 893 on ID 2. Selected ID 1 was last observed at source frame 481 and remained as
a predicted track through frame 571 before expiry. This exposes ID continuity loss
while target-box coverage continues. These outputs do not isolate which association
gate caused the transition and are not independently verified physical re-ID scores.

This clip has no annotated absence or missing-label intervals. Those semantics were
audited in the full annotation release and tested with synthetic inference fixtures;
real-video disappearance/return performance was not established by this run.

On the recorded Windows/Python 3.11 CPU setup, instrumented throughput was 20.83 FPS;
mean/p95 selected-frame latency was 47.97/63.74 ms. The separate probe averaged 2.98 ms,
and detector+tracking time averaged 44.76 ms. These measurements include diagnostics
and are not an uninstrumented speed claim; exact hardware/environment are in the output.

Local outputs are under ignored `results/mobiface_native_test_final/`, including stage
reports, initialization evidence and an annotated video. This is one test sequence,
not full-split accuracy or a tuning set; the other 79 videos were not evaluated.
