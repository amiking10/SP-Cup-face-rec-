# C3 baseline limitations

- MediaPipe detects faces, not whole people. Tiny/profile/poorly lit/occluded faces may be
  missed. A detection score is not a probability of correct identity association.
- The default SFace backend compares aligned face descriptors within a run. It needs
  a compatible YuNet landmark detection; faces that cannot be aligned count as missing.
  Its similarity limits are heuristic and can produce both false matches and false rejections.
  The optional MobileNetV2 backend uses ImageNet features, not trained face recognition.
- The application never substitutes a **different raw track ID** automatically. It cannot
  guarantee that a different person will never receive the same ID. All confirmed
  matches now pass appearance checks before assignment, including the upstream IoU fallback.
  Selection freezes a clicked-face anchor and up to eight closely consistent recent views,
  and requires a margin over competing faces observed at the same time. Stale duplicate
  IDs are not automatically competitors. After a loss, three
  consecutive compatible observations are required before visibility resumes. These
  reduce specific swaps but do not establish C3 success; review actual person associations.
  Head pose, face size, motion blur, lighting and failed alignment can still cause false
  rejection and expiry while a real person is present. A frozen bank reduces sensitivity
  to one selected frame; it does not guarantee recognition of every later view.
- Missing detection means LOST, even if the Kalman filter predicts a plausible location.
  Prediction boxes are hidden by default and labeled with `--debug-overlay`; they
  remain in logs and are never eligible observed ROIs. No rPPG or
  physiological sampling is performed. A future sampler must explicitly check confirmation,
  current detection, target selection, image quality, and independent association reliability.
- `REACQUIRED` is only a one-frame label for same-ID observation after LOST while the track
  survives. After the configured memory (three seconds by default, or `max_age`
  unmatched frames when explicitly supplied), a deleted ID has no automatic recovery path.
  Returning people need manual reselection if assigned a new ID. No cross-ID matching is claimed.
- Confirmation and the optional max_age override are frame-based. Default confirmed-track
  memory is time-based: monotonic elapsed time for webcam, source timestamps for video.
  Tentative tracks can vanish immediately on a miss; wait for confirmation before selecting.
- Position-only gating and appearance/geometry recovery throughout the live memory window reduce
  fragmentation after brief missed detections. Recovery rejects close competing candidates
  and never merges or revives deleted IDs. It is still heuristic and may misassociate
  similar faces; a raw ID is not verified identity. The default detector threshold is
  0.65; changes to it affect missed detections as well as background false positives.
- Outputs log all tracks plus empty frames and keep raw IDs. Predicted box confidence is blank.
  No tracker association confidence is fabricated. Selection can be manually changed; split
  trials on deliberate reselection/reset when evaluating a fixed designated real person.
- Video timestamps assume constant frame rate. Webcam timestamps reflect processing-loop
  arrival, not sensor exposure time; dropped camera frames and I/O stalls are not reconstructed.
  OpenCV cannot reliably distinguish every truncated-video read failure from normal EOF.
- CPU is supported; throughput depends on hardware, faces, capture, encoding and display.
  The smoke benchmark is not a real-world performance promise. GPU and Windows remain local checks.
- Pinning older dependencies preserves this baseline API, with upstream deprecation warnings.
  DeepSORT uses upstream commit `3341f823ee0ad20aa58c906079a37f3d43b42bbe`, which
  fixes repeated insertion of stale appearance features on missed frames; its package
  version remains 1.3.2. Dependencies pin the commit rather than relying on that version alone.
  Replacing MediaPipe with Tasks or updating torch/DeepSORT requires compatibility tests.
  The two upstream OpenCV dependency names overlap; use matching versions in a fresh venv.
- Face recordings are personal data. Collect consent, control access and retention, and never
  commit experimental recordings/weights/results. No cloud upload or database is implemented.
  Face embeddings and frozen references stay in process memory; reset or reselection clears
  the selected reference, and stopping the process clears the track gallery. Only pretrained
  model weights, screenshots, video, and existing CSV/JSON runtime metadata are written.
- This is a development baseline, not competition, clinical, security or medical validation.
  A separate literature review and independently annotated experiments should guide replacement
  of the backend. No HR, SpO2 or blood-pressure algorithms are included.
