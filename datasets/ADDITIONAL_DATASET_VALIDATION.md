# Additional dataset testing

This follow-up covers ChokePoint, VPCD and MobiFace. Dataset access and annotation
semantics determine which experiments can run. No model or tracking threshold was
tuned using these recordings, and dataset material remains local and ignored.

## ChokePoint: real eye-point tests

The [official project](https://arma.sourceforge.net/chokepoint/) links the
[Zenodo release 815657](https://zenodo.org/records/815657). The actual downloaded
`groundtruth.tar.xz` contains XML with this structure:

```xml
<dataset name="P1E_S1_C1">
  <frame number="00000233">
    <person id="0003">
      <leftEye x="..." y="..."/>
      <rightEye x="..." y="..."/>
    </person>
  </frame>
</dataset>
```

The coordinates above are placeholders illustrating the verified fields. These are
eye points, **not face bounding boxes**. The source uses 800x600 JPEGs at 30 FPS,
named by zero-based original frame index. Frame numbering has gaps: they must not
be compressed into a new timeline or filled with invented input frames. Some images
contain visible bystanders without eye annotations. Empty/missing eye labels do not
certify physical absence or exhaustive face coverage.

The dedicated `c3_tracker.benchmark.chokepoint` command runs the existing tracker on
raw decoded JPEGs. It parses the XML only **after inference**. It designates no target,
supplies no GT-derived boxes, and offers no oracle mode. It exports the common
prediction schema plus `frame_metrics.csv`, `tracking_metrics.json`, `performance.json`,
`experiment_config.json`, `report.md`, and optional post-inference overlays. Source
JPEG/annotation hashes, original frame indices/timestamps, code/environment information,
and missing sampled-frame counts are recorded. Failed runs are marked as failed.

### Download and repeat on Windows

```powershell
.\.venv\Scripts\python.exe datasets/scripts/download_c3_datasets.py chokepoint --files groundtruth.tar.xz P1E_S1.tar.xz README.txt
$chokeRoot = "datasets/data/raw/chokepoint"
tar -xJf "$chokeRoot/groundtruth.tar.xz" -C "$chokeRoot"
tar -xJf "$chokeRoot/P1E_S1.tar.xz" -C "$chokeRoot"
foreach ($cameraNumber in 1,2,3) {
    $sequenceName = "P1E_S1_C$cameraNumber"
    tar -xJf "$chokeRoot/$sequenceName.tar.xz" -C "$chokeRoot"
    .\.venv\Scripts\python.exe -m c3_tracker.benchmark.chokepoint --images "$chokeRoot/$sequenceName" --annotations "$chokeRoot/groundtruth/$sequenceName.xml" --output "results/chokepoint_repeat/$sequenceName" --start-frame 0 --end-frame 900 --matrix --annotate-video
    if ($LASTEXITCODE -ne 0) { throw "ChokePoint test failed: $sequenceName" }
}
```

Use fresh output directories. The downloader verifies the release's MD5 values:
`groundtruth.tar.xz`: `5c9d4d38a1c614905fe48da48ecdf06c`;
`P1E_S1.tar.xz`: `195191772856c1a62650a1d47e897b2a`.
The outer sequence archive is 385,315,324 bytes and contains three camera archives.
Extraction above restores all available source JPEGs; the command evaluates only
the registered interval `[0,900)`. Local validation extracted that interval only,
with the same original source bytes and indices.

All three cameras use the same first 30 source seconds, four annotated subjects,
and the unchanged SFace baseline. Six conditions per camera cover original/640x480/
320x240 at native and 10 FPS. There are 599 available JPEGs per native condition and
201 at 10 FPS; missing original indices are preserved as time gaps. These views are
correlated recordings, not three independent datasets. Overlays contain only available
sampled frames, so playback compresses gaps; displayed source timestamps remain exact.

The 18 local runs completed, processing 7,200 samples across the repeated operating
conditions (1,797 distinct source JPEGs across the three cameras). Eye-containment
recall percentages were:

| Condition | Camera 1 | Camera 2 | Camera 3 |
| --- | ---: | ---: | ---: |
| Original, native 30 FPS | 46.85 | 65.48 | 62.35 |
| Original, 10 FPS | 43.02 | 56.06 | 39.29 |
| 640x480, native 30 FPS | 45.67 | 64.47 | 54.12 |
| 640x480, 10 FPS | 43.02 | 60.61 | 28.57 |
| 320x240, native 30 FPS | 46.06 | 67.51 | 61.18 |
| 320x240, 10 FPS | 40.70 | 54.55 | 39.29 |

The native-FPS denominators are 254/197/85 annotated eye pairs for cameras 1/2/3;
at 10 FPS they are 86/66/28. Original/native matches were 119/129/53 respectively.
All conditions recorded zero custom within-run ID changes and zero cross-person
prediction-ID reassignment events **among matched samples**. This does not establish
reliable tracking of missed faces, target retention after disappearance, or standard
MOT performance. Raw outputs and videos are under ignored
`results/chokepoint_eye_baseline/`; repeat commands above produce a fresh result set.

### Metric definitions and limits

| Diagnostic | Definition |
| --- | --- |
| Eye-containment recall | Unambiguous matches containing both annotated eye points / annotated eye-pair samples. Only observed AND confirmed predictions are eligible. |
| Ambiguous eye pairs | Both eyes fit multiple predicted boxes, or one predicted box contains eyes from multiple annotated people. These count as unmatched. |
| Observed ID changes | Different predicted IDs on successive matched samples of a continuously annotated person. Missing source samples and annotation gaps reset continuity. This is not CLEAR MOT IDSW. |
| Cross-person prediction-ID reassignment events | A predicted ID matches another annotated person after its last eye-based match. Not a physical-reacquisition accuracy rate. |
| Per-person coverage | Annotated/matched samples and the set of matched predicted IDs, including people with no matches. |
| Processing FPS/latency | JPEG read/hash/decode/resize and inference; excludes model startup, XML parsing, scoring, exports and rendering. |

An oversized box can pass eye containment. **Do not compare its recall directly with
AVA IoU-gated recall.** Face-box IoU, detector precision/FP counts, standard IDSW,
IDF1, HOTA and target reacquisition remain unavailable. Missing detections are counted
in the eye-pair denominator; zero ID changes among successful matches cannot establish
reliable retention when many annotated samples were missed. See the saved per-person
denominators and overlays before interpreting identity diagnostics.

## VPCD: matching source footage missing

The [official project](https://www.robots.ox.ac.uk/~vgg/data/Video_Person_Clustering/)
links the [authors' repository](https://github.com/Andrew-Brown1/Video_Person_Clustering),
whose README gives the [metadata archive](https://www.robots.ox.ac.uk/~abrown/VPCD.tar.gz)
and a separate video-resolution/frame-rate document. A live header check found an
archive size of 5,690,329,235 bytes, last modified 2021-05-11. Streaming inspection
found `VPCD_Release1/Buffy.pk` (1,028,199,619 uncompressed bytes) as the first data file.

The archive was not downloaded in full, and no untrusted pickle was executed. No
matching source footage is present locally. Features or an already annotated preview
video cannot substitute for raw-video inference. Therefore **no VPCD tracking score or
verified native loader is claimed**. To proceed, provide matching raw clips plus the
release's metadata/README and verified frame-rate, resolution and timestamp mapping.
Character identity semantics and annotation coverage must be checked before MOT/C3 scoring.

## MobiFace: approved dataset not available locally

The [official access page](https://mobiface.github.io/) accepts requests from research
or academic institutions. No approved recordings/annotations are present locally;
no request was submitted on the user's behalf and no restricted mirror was used.
The linked evaluation download returned an unavailable response during inspection.

The [authors' annotation protocol](https://arxiv.org/pdf/1805.09749) labels one target
per video, including its absence state. Videos can contain multiple faces: 64 videos
have the Multiple Faces attribute. This corrects the overly broad assumption that
MobiFace videos contain only one face. Unlabeled bystanders prevent exhaustive MOT
precision/IDF1/HOTA scoring without additional ground truth. The paper also distinguishes
released training annotations from held-out evaluation labels; verify the actual release.

To run real tests, provide the approved local dataset folder with annotations and
matching frames/videos. Verify the actual box format, frame origin, sampling clock,
zero/absent encoding and available split. **No MobiFace tracking score or verified
native adapter is claimed until those files can be inspected.**
