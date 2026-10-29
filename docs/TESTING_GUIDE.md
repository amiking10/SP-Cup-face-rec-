# C3 experiment and annotation protocol

## Desktop controls and quick regression check

Launch from the repository root with `.\.venv\Scripts\python.exe -m c3_tracker --source webcam`.
The normal display shows only confirmed tracks with a current face detection. Lost
tracks remain in the CSV and selected-target LOST state, but their prediction boxes
are hidden. Use `--debug-overlay` when investigating detections and stale tracks.
The default detection threshold is now 0.65; use the same threshold when comparing runs.

Click **Save [S/Space]**, or focus the tracking window and press S or Space. Look for
the **Saved:** message at the bottom, then click **Open folder [O]** to find the PNG
inside `results/<run>/`. The terminal also prints its absolute path. Space and the
mouse button avoid keyboard-layout differences. Test each input once and open each PNG.
An error message should keep the tracker running if a screenshot cannot be written.

Move sideways, turn your head, and briefly cover your face for a few frames. Verify
manually that the same person's ID persists where recovery is unambiguous, that missing
faces lose their visible box, and that no second box lingers over an old position.
Repeat with two different people; record any identity swaps, failures, or new IDs.
Recovery after short gaps uses appearance plus nearby geometry; ambiguous returns and
returns after track expiry can still need manual reselection. Passing automated
regressions does not establish accuracy during live motion or crossing people.
The default memory now lasts three seconds, with recovery throughout that interval;
use `--track-memory-seconds 5` for five seconds. Select A, hide only A for one second
while B remains visible, and verify A returns under A's original ID while B keeps B's.
Repeat at different FPS. A should be LOST while hidden; never count predictions as visibility.

The default backend now uses aligned SFace face descriptors. Setup once with
`python -m c3_tracker.models --download`. When selecting A, its clicked-frame anchor
and a small bank of closely consistent recent views are frozen.
After loss, automatic return needs three consecutive compatible observations and a
margin over the other face. Some uncertain returns will stay LOST and need manual selection.
Add a deliberate wrong-person test: while A is selected, A leaves and B moves into
A's former position. The selected ID must remain LOST, with no green box on B.
Then A returns before memory expires: verify that only A turns green after confirmation.
Record any failure; similarity checks do not guarantee identity correctness.
Also turn and move closer/farther while staying visible. Verify that a rejected
selected match does not generate a new tentative ID on every frame. After a false
loss, a compatible return should recover the original selected ID before any
duplicate track can consume the face. Manually reselect and verify that stale
IDs for that same face do not permanently block the new selection.

## Preparation

Obtain consent from both participants. Assign real-person labels A and B on a paper
or private session note, distinct from track IDs. Wear distinguishable clothing and
keep a reviewer able to identify the selected person throughout crossings. Do not use
tracker IDs to decide who a person is. Keep lighting, camera position and distance fixed.
Record scene/camera, software versions, settings and actual resolution from metadata.
Keep the full output directory; record participant/session labels separately from code.
Use `--record` and write down frame ranges for each trial, selection, reset and failure.
The annotated recording includes frame-aligned IDs; CSV frame indices are zero-based.
Use a video editor that supports frame stepping. Keep raw recordings separately if collected.

Before measuring, wait until both tracks are confirmed and click A. Verify manually that
the green box encloses A. Clicking another face deliberately changes the designated target;
do not do this mid-trial. If selection expires, record the loss before explicitly starting
a new trial with a new selection. Do not discard failures or reset just to improve metrics.

## Experiment 1: two stationary people

1. Run `python -m c3_tracker --source webcam --width 640 --height 480 --record`.
2. A and B stand side by side, faces frontal and fully visible, similar distance from camera.
3. Wait for distinct confirmed IDs, select A, note start frame/time, and verify green encloses A.
4. Observe for 30 seconds after selection. Keep both faces visible and positions fixed.
5. Quit with Q. Review all 30 seconds frame by frame (or annotate all measurement frames).
6. Record whether both faces were detected, whether IDs remained distinct, which real person
   the selected ID enclosed, loss episodes, incorrect target assignments and FPS/latency.

Expected behavior is stable distinct IDs, A highlighted, and no B assignment. These are
hypotheses to check, not assertions established by unit tests. A smooth green box can still
follow the wrong person. Mark an unjudgeable association as unannotated, never a success.

## Experiment 2: crossing people

1. Start a fresh run with the same settings, both subjects visible. Select A and verify.
2. Have A and B exchange left/right positions three times at normal walking speed.
3. At least one crossing must involve partial face overlap; avoid unsafe physical contact.
4. Note frame ranges for each crossing, including approach, overlap and separation.
5. Hold each final position briefly. Include an additional optional occlusion/exit trial:
   A leaves the frame for one second, then returns while B stays visible. Repeat with
   an absence longer than the configured memory (three seconds by default, or `max_age`
   processed frames if that override is supplied). Do not automatically reselect on return.
6. Review the actual person inside every highlighted box through the crossings. Record ID
   changes, loss spans, wrong assignments and whether a return reused a still-alive ID or
   created a new one. New IDs do not establish identity recovery.

Expected: correct associations when possible and visible LOST state on missing detections.
A swap to B under A's old ID is an algorithm failure even though the UI still says VISIBLE.
Report that failure explicitly; do not label the crossing passed from ID continuity alone.

## Resolution and performance comparison

Repeat both experiments at `--width 1280 --height 720`, 640×480 and 320×240, preferably
three repeats each. Check metadata: webcams may reject requested resolutions. For video
comparison, prepare equivalent CFR input clips at each resolution outside this repository;
width/height flags do not resize video. Keep confidence, age, confirmation and appearance
thresholds fixed. Record actual resolution, face pixel size, lighting, CPU/GPU, source FPS,
measured end-to-end processing FPS and mean/p95 latency. Never equate source FPS with throughput.

## Independent annotations

For a single trial with a fixed intended real target (e.g. A), make a UTF-8 CSV:

```csv
frame_index,target_visible,target_person_id,selected_person_id
15,1,A,A
16,1,A,B
17,1,A,
18,0,A,
```

For each annotated frame:

- `target_visible`: independently judged visibility of A (0/1), not the program's flag.
- `target_person_id`: fixed intended person label for this trial.
- `selected_person_id`: the real person in the currently highlighted observed target box,
  blank when the application has no visible target. Determine this from the image, not the
  ID. If you cannot judge, omit the frame and report coverage. Do not invent a person label.
- Frame indices must be unique and present in the log. Annotate every trial frame for temporal
  analysis; sparse annotations cannot establish continuous identity stability.

Run:

```powershell
python -m c3_tracker.evaluation results\RUN\tracking_log.csv --annotations annotations.csv
```

Target recall is correct highlighted assignments / independently target-visible annotated
frames (includes visible frames with no selection as misses). False assignment rate is wrong
highlighted assignments / all highlighted assignments in annotated frames. Selected-person
switches count changes in the manually labeled person across highlighted frames, ignoring
loss gaps. These are target-focused measures, not standard MOT IDSW/MOTA/IDF1. Zero-denominator
rates are null. No annotations means runtime statistics only. Separate trials before evaluation;
use the same target_person_id within a trial. Evaluation rejects incomplete or contradictory rows.

For later target-loss duration/recovery analysis, inspect `video_timestamp_s`, `target_status`,
`selected_target_id` and the independent annotations. Mark loss start/end and whether the return
was correct; report unrecovered intervals as censored at run end. Automated runtime `target_lost_frames`
and `target_loss_events` count tracker-visible state only, never independently verified losses.
These temporal identity metrics are not yet automated. General multi-person MOT evaluation will
need independent person bounding boxes for all faces, IDs, occlusion flags and a matching policy;
the current annotated-target CSV does not provide that ground truth.

## Automated and headless checks

For a short live failure recording, use:

```powershell
.\.venv\Scripts\python.exe -m c3_tracker --source webcam --record --record-raw --diagnostics
```

Select one face, reproduce a head turn or brief disappearance, then quit with Q.
Keep the entire run folder. `source_video.mp4` has no overlays; it shares frame indices
with `annotated_video.mp4`, the CSV and `association_diagnostics.jsonl`. Source video is
still compressed; the JSONL records the actual live matching distances and decisions.
It distinguishes absent detection, failed alignment, appearance-gallery/reference/anchor
rejection, competing-face ambiguity, motion rejection, pending verification and expiry.
It contains no embedding vectors. Do not estimate live identity thresholds from an
annotated-video replay alone: boxes, text and compression can alter model outputs.

```powershell
python -m pytest -q
python -m c3_tracker --help
python -m c3_tracker --source video --input "C:\videos\consented-test.mp4" --headless --record --max-frames 100
python -m c3_tracker.evaluation results\RUN\tracking_log.csv
```

Tests use mock tracks/detections for deterministic logic, not tracking quality. For headless
runs, no target is selected. Confirm the expected frame count, nonempty detections/confirmed
track rows when faces exist, current-frame confidence only on observations, summary with no
error, and a decodable annotated MP4. A PID, successful import or empty log is insufficient.

Cloud checks cannot establish Windows camera permissions, desktop interactions, sustained FPS,
or person-to-track correctness. Complete both live protocols locally before making C3 claims.

Opt-in tests load the actual face models and generate temporary MP4s from sample
photographs. A distinct-face replacement regression uses a second public OpenCV sample:

```powershell
python -m c3_tracker.models --download --test-fixtures
$env:C3_RUN_MODEL_TESTS = "1"
python -m pytest -q
Remove-Item Env:\C3_RUN_MODEL_TESTS
```

These verify detection, confirmation, predicted-only loss, same-ID return while alive,
expiry without cross-ID reassignment, rejection of a second face occupying the selection's
old position, three-frame selected return, end-to-end CSV/JSON output and MP4 decoding. They
are still synthetic pipeline smoke checks, not evidence of real-person tracking accuracy.
