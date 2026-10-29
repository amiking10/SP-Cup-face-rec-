# C3 baseline limitations

- MediaPipe detects faces, not whole people. Tiny/profile/poorly lit/occluded faces may be
  missed. A detection score is not a probability of correct identity association.
- DeepSORT's bundled MobileNetV2 uses ImageNet appearance features, not a trained face
  recognition model. Similar faces, crossings and overlap can cause ID switches. No
  biometric verification or independently calibrated association uncertainty is available.
- The application never substitutes a **different raw track ID** automatically. It cannot
  prevent DeepSORT from assigning a different person to the same ID. A persistent green
  box is insufficient evidence of C3 success; review actual person associations.
- Missing detection means LOST, even if the Kalman filter predicts a plausible location.
  Prediction boxes remain labeled, but are never eligible observed ROIs. No rPPG or
  physiological sampling is performed. A future sampler must explicitly check confirmation,
  current detection, target selection, image quality, and independent association reliability.
- `REACQUIRED` is only a one-frame label for same-ID observation after LOST while the track
  survives. After `max_age` unmatched frames a deleted ID has no automatic recovery path.
  Returning people need manual reselection if assigned a new ID. No cross-ID matching is claimed.
- Confirmation and max_age are frame-based. Low FPS changes their wall-clock behavior.
  Tentative tracks can vanish immediately on a miss; wait for confirmation before selecting.
- Outputs log all tracks plus empty frames and keep raw IDs. Predicted box confidence is blank.
  No tracker association confidence is fabricated. Selection can be manually changed; split
  trials on deliberate reselection/reset when evaluating a fixed designated real person.
- Video timestamps assume constant frame rate. Webcam timestamps reflect processing-loop
  arrival, not sensor exposure time; dropped camera frames and I/O stalls are not reconstructed.
  OpenCV cannot reliably distinguish every truncated-video read failure from normal EOF.
- CPU is supported; throughput depends on hardware, faces, capture, encoding and display.
  The smoke benchmark is not a real-world performance promise. GPU and Windows remain local checks.
- Pinning older dependencies preserves this baseline API, with upstream deprecation warnings.
  Replacing MediaPipe with Tasks or updating torch/DeepSORT requires compatibility tests.
  The two upstream OpenCV dependency names overlap; use matching versions in a fresh venv.
- Face recordings are personal data. Collect consent, control access and retention, and never
  commit experimental recordings/weights/results. No cloud upload or database is implemented.
- This is a development baseline, not competition, clinical, security or medical validation.
  A separate literature review and independently annotated experiments should guide replacement
  of the backend. No HR, SpO2 or blood-pressure algorithms are included.
