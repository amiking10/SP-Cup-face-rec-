# Tracking limits

- Raw track IDs are associations within a run, not verified person identities.
- A missing face can retain its ID during the configured memory period, but this
  does not guarantee reacquisition. Deleted-ID recovery is not implemented.
- Selection remains lost when confirmation/appearance checks fail. Predicted boxes
  are not valid observed face samples; never reuse old pixels as fresh pulse data.
- FaceVitals emits centered 1.5x boxes and NaN on missing frames. It processes valid
  spans without interpolating across loss. BPM accuracy and output coverage must
  be reported together, on aligned source windows.
- Existing benchmark clocks do not yet form a valid paired comparison. No timing
  correction or acceptance threshold change is authorized by this cleanup.
- The prior one-target checks are descriptive only. Full cohort acceptance,
  source/third-party licence decisions and competition packaging remain open.

Historical measurements and full research procedures remain on the archive
branches linked in [README](../README.md). Passing core tests is not biometric,
physiological or competition validation.
