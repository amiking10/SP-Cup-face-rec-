"""CLI and capture lifecycle; heavyweight models are loaded only after input validation."""
import argparse
import math
from pathlib import Path
import time
import warnings
import cv2
import numpy as np
from .detection import FaceDetector
from .tracking import FaceTracker
from .target_manager import TargetManager
from .visualization import annotate
from .logging_utils import RunLogger

WINDOW = "C3 face tracking"


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def probability(value):
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return number


def parser():
    p = argparse.ArgumentParser(description="MediaPipe + DeepSORT C3 baseline")
    p.add_argument("--source", choices=("webcam", "video"), required=True)
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--input", type=Path)
    p.add_argument("--width", type=positive, default=640)
    p.add_argument("--height", type=positive, default=480)
    p.add_argument("--confidence", type=probability, default=0.5)
    p.add_argument("--max-age", type=positive, default=30)
    p.add_argument("--n-init", type=positive, default=3)
    p.add_argument("--max-cosine-distance", type=probability, default=0.2)
    p.add_argument("--model-selection", type=int, choices=(0, 1), default=0)
    p.add_argument("--gpu", action="store_true", help="use a CUDA-compatible torch installation")
    p.add_argument("--record", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--headless", action="store_true", help="no display or mouse selection")
    p.add_argument("--max-frames", type=positive)
    return p


def open_capture(args):
    if args.source == "video":
        if args.input is None or not args.input.is_file():
            raise ValueError("Video input must be an existing file: use --input path/to/video.mp4")
        cap = cv2.VideoCapture(str(args.input))
    elif args.source == "webcam":
        if args.camera_index < 0:
            raise ValueError("Camera index must be nonnegative")
        cap = cv2.VideoCapture(args.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    else:
        raise ValueError(f"Unsupported source: {args.source}")
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Cannot open input. Check video codec or camera index, permissions, "
                           "and whether another application is using the camera.")
    return cap


def run(args):
    if args.headless and args.source == "webcam" and args.max_frames is None:
        raise ValueError("Headless webcam runs require --max-frames")
    cap = open_capture(args)
    detector = logger = writer = None
    latencies = []
    target = TargetManager()
    observations = []
    frames = visible = lost = loss_events = 0
    previous_lost = False
    error = None
    wall_start = None
    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Input opened but yielded no decodable frames")
        reported_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(reported_fps) or reported_fps <= 0:
            reported_fps = None
        if args.source == "video" and reported_fps is None:
            raise RuntimeError("Video has no usable FPS; cannot assign defensible video timestamps")
        detector = FaceDetector(args.confidence, args.model_selection)
        tracker = FaceTracker(args.max_age, args.n_init, args.max_cosine_distance, args.gpu)
        height, width = frame.shape[:2]
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        logger = RunLogger(args.output_dir, {
            "configuration": config, "resolution": [width, height],
            "reported_source_fps": reported_fps,
            "timestamp_basis": "frame_index / reported_source_fps" if args.source == "video" else "monotonic capture elapsed",
            "latency_definition": "capture/read + detection + tracking + overlay; excludes logging, encoding and GUI wait",
            "reacquisition_definition": "current detection assigned to the SAME still-alive raw ID; not verified identity",
            "recording_fps": (reported_fps or 30.0) if args.record else None})
        print(f"Results: {logger.directory}")
        if args.record:
            if reported_fps is None:
                warnings.warn("Camera did not report FPS; recording uses 30 FPS, not measured throughput")
            writer = cv2.VideoWriter(str(logger.directory / "annotated_video.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), reported_fps or 30.0, (width, height))
            if not writer.isOpened():
                raise RuntimeError("Cannot initialize MP4 writer; check OpenCV codecs and output permissions")
        if not args.headless:
            cv2.namedWindow(WINDOW)

            def mouse(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    target.select(x, y, observations)
            cv2.setMouseCallback(WINDOW, mouse)
        wall_start = time.perf_counter()
        read_ms = 0.0  # first frame was decoded before model initialization
        while ok:
            started = time.perf_counter()
            timestamp = frames / reported_fps if args.source == "video" else started - wall_start
            detections = detector.detect(frame)
            observations = tracker.update(detections, frame)
            target.update(observations)
            elapsed = time.perf_counter() - wall_start
            image = annotate(frame, detections, observations, target,
                             frames / elapsed if elapsed > 0 else 0)
            latency = (time.perf_counter() - started) * 1000 + read_ms
            latencies.append(latency)
            logger.log_frame(frames, timestamp, observations, target, latency)
            visible += int(target.visible)
            is_lost = target.status == "LOST"
            lost += int(is_lost)
            loss_events += int(is_lost and not previous_lost)
            previous_lost = is_lost
            if writer is not None:
                if image.shape[:2] != (height, width):
                    raise RuntimeError("Input resolution changed during recording")
                writer.write(image)
            frames += 1
            if not args.headless:
                cv2.imshow(WINDOW, image)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord("r"), ord("R")):
                    target.reset()
                if key in (ord("s"), ord("S")):
                    path = logger.directory / f"screenshot_{frames - 1:06d}.png"
                    if not cv2.imwrite(str(path), image):
                        raise RuntimeError(f"Screenshot write failed: {path}")
            if args.max_frames and frames >= args.max_frames:
                break
            read_start = time.perf_counter()
            ok, frame = cap.read()
            read_ms = (time.perf_counter() - read_start) * 1000
            if not ok and args.source == "webcam":
                raise RuntimeError("Camera stopped yielding frames")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed = time.perf_counter() - wall_start if wall_start is not None else 0
        cap.release()
        if detector is not None:
            detector.close()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if logger is not None:
            logger.close({"frames_processed": frames, "elapsed_s": elapsed,
                          "end_to_end_processing_fps": frames / elapsed if elapsed else 0,
                          "latency_mean_ms": float(np.mean(latencies)) if latencies else None,
                          "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
                          "target_visible_frames": visible, "target_lost_frames": lost,
                          "target_loss_events": loss_events, "error": error,
                          "identity_metrics": "Not calculated: requires independent person annotations"})


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        return 130
    except (ValueError, RuntimeError, OSError, ImportError, cv2.error) as exc:
        print(f"C3 tracker error: {exc}", file=__import__("sys").stderr)
        return 1
    return 0
