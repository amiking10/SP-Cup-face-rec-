"""OpenCV overlay; predictions are visibly marked and never valid sample ROIs."""
import cv2


def annotate(frame, detections, observations, target, fps):
    image = frame.copy()
    for d in detections:
        l, t, r, b = map(int, d.bbox)
        cv2.rectangle(image, (l, t), (r, b), (160, 160, 160), 1)
    for o in observations:
        color = (0, 220, 0) if o.track_id == target.target_id and o.observed and o.confirmed else (0, 160, 255)
        l, t, r, b = map(int, o.bbox)
        cv2.rectangle(image, (l, t), (r, b), color, 2)
        label = f"ID {o.track_id} {'confirmed' if o.confirmed else 'tentative'}"
        if not o.observed:
            label += " PREDICTED - NO SAMPLE"
        cv2.putText(image, label, (l, max(18, t - 6)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
    text = f"Target: {target.target_id or '-'} {target.status} | processing {fps:.1f} FPS"
    cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55,
                (0, 0, 255) if target.status == "LOST" else (255, 255, 255), 2)
    cv2.putText(image, "Click confirmed face | R reset | S screenshot | Q quit", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    return image
