"""Clean live overlay; an optional diagnostic overlay includes predictions."""
import cv2


def annotate(frame, detections, observations, target, fps, debug=False, capture_roi=None):
    image = frame.copy()
    for d in detections if debug else []:
        l, t, r, b = map(int, d.bbox)
        cv2.rectangle(image, (l, t), (r, b), (160, 160, 160), 1)
    for o in observations:
        if not debug and not (o.observed and o.confirmed):
            continue
        selected = o.track_id == target.target_id and o.observed and o.confirmed and target.visible
        color = (0, 220, 0) if selected else (0, 160, 255)
        if selected:
            if capture_roi is None:
                continue  # The adjusted region can fall entirely outside the image.
            l, t, r, b = capture_roi
            r, b = r - 1, b - 1  # Capture bounds are exclusive; rectangle endpoints are inclusive.
        else:
            l, t, r, b = map(int, o.bbox)
        label = f"ID {o.track_id} {'confirmed' if o.confirmed else 'tentative'}"
        if not o.observed:
            label += " PREDICTED - NO SAMPLE"
        (text_width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .45, 1)
        x = max(0, min(l, image.shape[1] - text_width - 4))
        y = t - 6
        if y < 98:
            y = min(image.shape[0] - 5, b + 18)
        cv2.rectangle(image, (x, y - 15), (x + text_width + 4, y + 3), (25, 25, 25), -1)
        cv2.putText(image, label, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
        cv2.rectangle(image, (l, t), (r, b), color, 2)
    cv2.rectangle(image, (0, 0), (image.shape[1], 85), (25, 25, 25), -1)
    text = f"Target: {target.target_id or '-'} {target.status} | processing {fps:.1f} FPS"
    cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55,
                (0, 0, 255) if target.status == "LOST" else (255, 255, 255), 2)
    cv2.putText(image, "Click face to select | R reset | Q quit | Focus window for keys", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    return image
