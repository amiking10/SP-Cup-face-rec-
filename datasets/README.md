# C3 benchmark datasets

Start here for dataset availability, preparation and measured evidence. Code and source
provenance are distributed in Git; datasets, recordings, weights and results stay local.

## Current support and local validation

| Dataset | Local inputs / experiments | Supported scope and limits |
| --- | --- | --- |
| [AVA ActiveSpeaker](https://sites.research.google/gr/ava/download/) | Validation labels and matching `053oq2xB3oU`; seconds 900–910 at six operating conditions with stage diagnostics | Annotation-relative box IoU/recall. Track-local IDs and uncertified exhaustive coverage keep physical reacquisition and full identity/FP metrics unavailable. [Findings](AVA_DIAGNOSTICS.md). |
| [MobiFace80](mobiface/README.md) | Provided Drive metadata, 80 native annotation CSVs, toolkit, one matching public test-video prefix; native audit and sequence `7I5t6BAHSGQ_0` evaluated | Target-box coverage and fixed selected-ID retention. Official split preserved. One labeled target does not certify bystander identities or detection precision. |
| [ChokePoint](https://arma.sourceforge.net/chokepoint/) | Official eye-point XML and P1E_S1 JPEGs, three cameras; short resolution/FPS matrix and complete available recording at native/10 FPS | Eye-containment and custom ID-change/reuse diagnostics, not face-box IoU/IDF1/HOTA. [Protocol/results](ADDITIONAL_DATASET_VALIDATION.md). |
| [VPCD](https://www.robots.ox.ac.uk/~vgg/data/Video_Person_Clustering/) | Source availability inspected; matching raw footage absent locally; no tracker run | No native adapter or accuracy claim. [Required mapping](ADDITIONAL_DATASET_VALIDATION.md#vpcd-matching-source-footage-missing). |

These are bounded local experiments, not full-dataset accuracy. MobiFace videos can
contain multiple faces while only one target is labeled. AVA entity IDs are local tracks,
not independently verified physical identities after disappearance.

## Guides

- [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md): environment, AVA acquisition/preparation,
  common schemas, baseline/oracle modes, operating conditions and outputs.
- [AVA_DIAGNOSTICS.md](AVA_DIAGNOSTICS.md): stage counts, initialization failure,
  fixed-condition comparisons and failure illustrations.
- [mobiface/README.md](mobiface/README.md): source, native conventions, splits, explicit
  frame mapping, audit/prepare/run commands, metrics and actual validation scope.
- [ADDITIONAL_DATASET_VALIDATION.md](ADDITIONAL_DATASET_VALIDATION.md): ChokePoint
  protocols/results and VPCD limitations.
- [mobiface/source.json](mobiface/source.json): verified Drive source IDs and provenance.

## Canonical local layout

Paths are relative to the repository root:

```text
datasets/data/
  raw/
    ava_active_speaker/{annotations,videos}/
    mobiface/
      mobiface80.meta.csv
      mobiface80/{train.meta.csv,test.meta.csv,train/,test/}
      videos/
    chokepoint/
    vpcd/
  processed/
    ava_active_speaker/{manifest.json,<video_id>/annotations.csv}
    mobiface/{manifest.json,<sequence>/annotations.csv}
results/                         # predictions, metrics, reports and optional media
```

`datasets/data/.gitignore` excludes local data; `results/` and `.models/` are also
ignored. Obtain matching media under its source terms and retain acquisition provenance.

## Evaluation rules

Inference receives raw frames. Only one recorded initialization frame may use GT to
designate a target; subsequent GT is used for scoring and illustrations afterward.
Unknown labels are not empty scenes; unlabeled bystanders are not automatic false
positives. Keep oracle boxes separate and freeze settings before held-out evaluation.

The older `scripts/prepare_ava_annotations.py` inspects metadata only. Use
`python -m c3_tracker.evaluate prepare` for video-aligned AVA preparation. The benchmark
guide documents separate output directories for the two utilities.
