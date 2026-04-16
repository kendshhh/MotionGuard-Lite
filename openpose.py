from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from contextlib import contextmanager
from pathlib import Path

try:
    import cv2 as cv
except ModuleNotFoundError as exc:
    if exc.name == "cv2":
        raise ModuleNotFoundError(
            "OpenCV (cv2) is not installed for the active Python interpreter. "
            "Run install_requirements.bat, or use .\\.venv\\Scripts\\python.exe to start this project."
        ) from exc
    raise


BODY_PARTS = {
    "Nose": 0,
    "Neck": 1,
    "RShoulder": 2,
    "RElbow": 3,
    "RWrist": 4,
    "LShoulder": 5,
    "LElbow": 6,
    "LWrist": 7,
    "RHip": 8,
    "RKnee": 9,
    "RAnkle": 10,
    "LHip": 11,
    "LKnee": 12,
    "LAnkle": 13,
    "REye": 14,
    "LEye": 15,
    "REar": 16,
    "LEar": 17,
    "Background": 18,
}

POSE_PAIRS = [
    ["Neck", "RShoulder"],
    ["Neck", "LShoulder"],
    ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"],
    ["LShoulder", "LElbow"],
    ["LElbow", "LWrist"],
    ["Neck", "RHip"],
    ["RHip", "RKnee"],
    ["RKnee", "RAnkle"],
    ["Neck", "LHip"],
    ["LHip", "LKnee"],
    ["LKnee", "LAnkle"],
    ["Neck", "Nose"],
    ["Nose", "REye"],
    ["REye", "REar"],
    ["Nose", "LEye"],
    ["LEye", "LEar"],
]

IMAGE_EXTENSIONS = {".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".webp", ".pbm", ".pgm", ".ppm", ".pxm", ".pnm", ".sr", ".ras", ".tif", ".tiff"}
TECHNIQUE_NAMES = ("Punch", "Block", "Escape")
CAMERA_PROBE_MAX_INDEX = 5
WINDOWS_CAMERA_BACKENDS = (
    ("DirectShow", cv.CAP_DSHOW),
    ("Media Foundation", cv.CAP_MSMF),
    ("Auto", cv.CAP_ANY),
)
WINDOW_STATES: dict[str, bool] = {}
COLOR_SUCCESS = (0, 180, 0)
COLOR_WARNING = (0, 185, 255)
COLOR_ERROR = (0, 0, 255)
COLOR_INFO = (255, 255, 255)
COLOR_MUTED = (210, 210, 210)
COLOR_PANEL = (18, 18, 18)
COLOR_PANEL_ALT = (15, 15, 15)
COLOR_LABEL = (255, 220, 120)
COLOR_BAR_BG = (70, 70, 70)
COLOR_STATUS_UNKNOWN = (0, 210, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Human pose estimation in OpenCV using a MobileNet OpenPose TensorFlow graph."
    )
    parser.add_argument("--input", help="Path to an image or video file. Omit to use webcam 0.")
    parser.add_argument(
        "--camera-index",
        default=0,
        type=int,
        help="Webcam index to open when --input is omitted. Use 1 or higher to try a different camera.",
    )
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parent / "models" / "graph_opt.pb"),
        help="Path to the TensorFlow graph (.pb) file.",
    )
    parser.add_argument("--thr", default=0.2, type=float, help="Confidence threshold for pose parts.")
    parser.add_argument("--width", default=368, type=int, help="Network input width.")
    parser.add_argument("--height", default=368, type=int, help="Network input height.")
    parser.add_argument("--output", help="Optional path for an annotated image or video.")
    parser.add_argument("--no-display", action="store_true", help="Run without opening an OpenCV window.")
    parser.add_argument(
        "--target-technique",
        type=lambda value: value.title(),
        choices=TECHNIQUE_NAMES,
        help="Optional named technique to validate against: Punch, Block, or Escape.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional limit for processed video/webcam frames, useful for testing.",
    )
    parser.add_argument(
        "--confirm-frames",
        default=5,
        type=int,
        help="Number of recent frames to use when stabilizing technique recognition.",
    )
    parser.add_argument(
        "--json-output",
        help="Optional path to write a JSON summary of the final result.",
    )
    parser.add_argument(
        "--list-cameras-json",
        help="Optional path to write detected camera indices as JSON and exit.",
    )
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"Error: {message}", file=sys.stderr)
    return 1


def score_to_color(score: float) -> tuple[int, int, int]:
    if score >= 0.75:
        return COLOR_SUCCESS
    if score >= 0.4:
        return COLOR_WARNING
    return COLOR_ERROR


def wrap_text_lines(text: str, max_width: int, font_scale: float, thickness: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current_line = words[0]
    for word in words[1:]:
        candidate = f"{current_line} {word}"
        candidate_width = cv.getTextSize(candidate, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
        if candidate_width <= max_width:
            current_line = candidate
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines


def get_overlay_layout(frame_width: int) -> dict:
    margin = 12
    gap = 12
    available_width = max(frame_width - (margin * 2) - gap, 300)
    left_width = min(520, max(360, int(available_width * 0.58)))
    right_width = min(420, max(300, available_width - left_width))
    if left_width + right_width > available_width:
        right_width = max(280, available_width - left_width)
    return {
        "margin": margin,
        "gap": gap,
        "left": margin,
        "left_width": left_width,
        "right_left": margin + left_width + gap,
        "right_width": right_width,
    }


def describe_input_source(input_source) -> str:
    if isinstance(input_source, int):
        return f"webcam {input_source}"
    return f"'{input_source}'"


@contextmanager
def suppress_native_stderr():
    try:
        saved_stderr = os.dup(2)
    except OSError:
        yield
        return

    null_stream = open(os.devnull, "w")
    try:
        os.dup2(null_stream.fileno(), 2)
        yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        null_stream.close()


def open_camera_with_windows_backends(camera_index: int):
    attempted_backends = []
    for backend_name, backend_flag in WINDOWS_CAMERA_BACKENDS:
        with suppress_native_stderr():
            capture = cv.VideoCapture(camera_index, backend_flag)
        attempted_backends.append(backend_name)
        if capture.isOpened():
            return capture, backend_name, attempted_backends
        capture.release()
    return None, None, attempted_backends


def discover_working_cameras(max_index: int = CAMERA_PROBE_MAX_INDEX, exclude_indices: set[int] | None = None) -> list[dict[str, object]]:
    if max_index < 0:
        return []

    excluded = exclude_indices or set()
    discovered: list[dict[str, object]] = []
    for camera_index in range(max_index + 1):
        if camera_index in excluded:
            continue
        if sys.platform.startswith("win"):
            capture, backend_name, _ = open_camera_with_windows_backends(camera_index)
        else:
            with suppress_native_stderr():
                capture = cv.VideoCapture(camera_index)
            backend_name = "Auto" if capture.isOpened() else None
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            continue
        capture.release()
        discovered.append({"index": camera_index, "backend": backend_name or "Auto"})
    return discovered


def format_detected_cameras(cameras: list[dict[str, object]]) -> str:
    return ", ".join(f"{camera['index']} ({camera['backend']})" for camera in cameras)


def open_capture(input_source):
    if isinstance(input_source, int) and sys.platform.startswith("win"):
        return open_camera_with_windows_backends(input_source)

    capture = cv.VideoCapture(input_source)
    if capture.isOpened():
        return capture, None, []
    capture.release()
    return None, None, []


def build_capture_error(input_source, attempted_backends: list[str]) -> str:
    source_description = describe_input_source(input_source)
    if attempted_backends:
        backend_summary = ", ".join(attempted_backends)
        available_cameras = discover_working_cameras(exclude_indices={input_source} if isinstance(input_source, int) else None)
        if available_cameras:
            available_summary = format_detected_cameras(available_cameras)
            return (
                f"Could not open {source_description}. Tried Windows camera backends: {backend_summary}. "
                f"Detected working camera indices: {available_summary}. Use one of those indices, or close other apps that may be using the selected camera, then retry."
            )
        return (
            f"Could not open {source_description}. Tried Windows camera backends: {backend_summary}. "
            f"No working camera indices were detected from 0 to {CAMERA_PROBE_MAX_INDEX}. "
            "Close other apps that may be using the camera, reconnect the webcam, then retry."
        )
    return f"Could not open {source_description}."


def write_json_summary(output_path: str | None, payload: dict) -> None:
    if not output_path:
        return
    path = Path(output_path)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_net(model_path: Path) -> cv.dnn.Net:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. Download graph_opt.pb and place it there or pass --model."
        )
    return cv.dnn.readNetFromTensorflow(str(model_path))


def detect_points(frame, net: cv.dnn.Net, in_width: int, in_height: int, threshold: float):
    frame_height, frame_width = frame.shape[:2]
    blob = cv.dnn.blobFromImage(
        frame,
        1.0,
        (in_width, in_height),
        (127.5, 127.5, 127.5),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)
    out = net.forward()
    out = out[:, :19, :, :]

    if out.shape[1] != len(BODY_PARTS):
        raise RuntimeError(
            f"Unexpected network output shape {out.shape}; expected {len(BODY_PARTS)} body-part maps."
        )

    points = []
    for index in range(len(BODY_PARTS)):
        heat_map = out[0, index, :, :]
        _, confidence, _, point = cv.minMaxLoc(heat_map)
        x = (frame_width * point[0]) / out.shape[3]
        y = (frame_height * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if confidence > threshold else None)

    return points


def get_point(points, part_name: str):
    return points[BODY_PARTS[part_name]]


def point_distance(point_a, point_b) -> float:
    if point_a is None or point_b is None:
        return 0.0
    delta_x = point_a[0] - point_b[0]
    delta_y = point_a[1] - point_b[1]
    return (delta_x * delta_x + delta_y * delta_y) ** 0.5


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def count_visible_points(points, part_names: tuple[str, ...]) -> int:
    return sum(1 for part_name in part_names if get_point(points, part_name) is not None)


def analyze_frame_visibility(frame) -> dict:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())
    brightness_std = float(gray.std())
    return {
        "mean_brightness": mean_brightness,
        "brightness_std": brightness_std,
        "is_dark": mean_brightness < 28,
        "is_low_detail": brightness_std < 10,
    }


def classify_technique(points, frame_shape) -> dict:
    frame_height, frame_width = frame_shape[:2]
    neck = get_point(points, "Neck")
    left_shoulder = get_point(points, "LShoulder")
    right_shoulder = get_point(points, "RShoulder")
    left_elbow = get_point(points, "LElbow")
    right_elbow = get_point(points, "RElbow")
    left_wrist = get_point(points, "LWrist")
    right_wrist = get_point(points, "RWrist")
    left_hip = get_point(points, "LHip")
    right_hip = get_point(points, "RHip")
    left_ankle = get_point(points, "LAnkle")
    right_ankle = get_point(points, "RAnkle")
    upper_body_parts = ("Nose", "Neck", "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist")
    lower_body_parts = ("LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle")
    upper_body_visible = count_visible_points(points, upper_body_parts)
    lower_body_visible = count_visible_points(points, lower_body_parts)
    upper_body_only_mode = upper_body_visible >= 5 and lower_body_visible <= 2

    if neck is None or left_shoulder is None or right_shoulder is None or upper_body_visible < 4:
        return {
            "label": "Unknown",
            "confidence": 0.0,
            "scores": {"Punch": 0.0, "Block": 0.0, "Escape": 0.0},
            "reason": "missing_upper_body",
        }

    shoulder_span = point_distance(left_shoulder, right_shoulder)
    if shoulder_span < 15:
        return {
            "label": "Unknown",
            "confidence": 0.0,
            "scores": {"Punch": 0.0, "Block": 0.0, "Escape": 0.0},
            "reason": "degenerate_pose",
        }

    hip_center = None
    if left_hip is not None and right_hip is not None:
        hip_center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
    shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
    base_size = shoulder_span
    if hip_center is not None:
        base_size = max(base_size, abs(hip_center[1] - shoulder_center[1]))

    def arm_state(side: str) -> dict:
        if side == "left":
            shoulder, elbow, wrist = left_shoulder, left_elbow, left_wrist
            direction = -1
        else:
            shoulder, elbow, wrist = right_shoulder, right_elbow, right_wrist
            direction = 1

        if shoulder is None or elbow is None or wrist is None:
            return {
                "visible": False,
                "extension": 0.0,
                "reach": 0.0,
                "horizontal": False,
                "guard": False,
                "raised": False,
                "lowered": False,
                "bent": False,
                "outward": False,
                "near_head": False,
                "cross_guard": False,
            }

        extension = abs(wrist[0] - shoulder[0]) / max(base_size, 1.0)
        reach = point_distance(wrist, shoulder) / max(base_size, 1.0)
        horizontal = abs(wrist[1] - shoulder[1]) <= 0.35 * base_size
        guard = wrist[1] <= neck[1] + 0.35 * base_size and abs(wrist[0] - neck[0]) <= 1.0 * base_size
        raised = wrist[1] <= shoulder[1] + 0.1 * base_size
        lowered = wrist[1] >= shoulder[1] + 0.35 * base_size
        bent = reach <= 1.35
        outward = (wrist[0] - shoulder[0]) * direction >= 0.45 * base_size
        near_head = wrist[1] <= neck[1] + 0.2 * base_size and abs(wrist[0] - neck[0]) <= 0.8 * base_size
        cross_guard = abs(wrist[0] - neck[0]) <= 0.45 * base_size and wrist[1] <= neck[1] + 0.45 * base_size
        return {
            "visible": True,
            "extension": extension,
            "reach": reach,
            "horizontal": horizontal,
            "guard": guard,
            "raised": raised,
            "lowered": lowered,
            "bent": bent,
            "outward": outward,
            "near_head": near_head,
            "cross_guard": cross_guard,
        }

    left_arm = arm_state("left")
    right_arm = arm_state("right")

    left_punch = 0.0
    right_punch = 0.0
    if left_arm["visible"]:
        left_punch = (
            0.35 * clamp_score((left_arm["extension"] - 0.55) / 0.85)
            + 0.2 * clamp_score((left_arm["reach"] - 0.85) / 0.55)
            + 0.15 * float(left_arm["horizontal"])
            + 0.15 * float(left_arm["outward"])
            + 0.15 * float(right_arm["guard"] or right_arm["cross_guard"])
        )
    if right_arm["visible"]:
        right_punch = (
            0.35 * clamp_score((right_arm["extension"] - 0.55) / 0.85)
            + 0.2 * clamp_score((right_arm["reach"] - 0.85) / 0.55)
            + 0.15 * float(right_arm["horizontal"])
            + 0.15 * float(right_arm["outward"])
            + 0.15 * float(left_arm["guard"] or left_arm["cross_guard"])
        )
    if upper_body_only_mode:
        left_punch = clamp_score(left_punch + 0.05 * float(left_arm["visible"]))
        right_punch = clamp_score(right_punch + 0.05 * float(right_arm["visible"]))
    punch_score = max(left_punch, right_punch)

    left_block = 0.0
    right_block = 0.0
    if left_arm["visible"]:
        left_block = (
            0.25 * float(left_arm["raised"])
            + 0.2 * float(left_arm["bent"])
            + 0.25 * float(left_arm["near_head"] or left_arm["cross_guard"])
            + 0.15 * float(not left_arm["outward"])
        )
    if right_arm["visible"]:
        right_block = (
            0.25 * float(right_arm["raised"])
            + 0.2 * float(right_arm["bent"])
            + 0.25 * float(right_arm["near_head"] or right_arm["cross_guard"])
            + 0.15 * float(not right_arm["outward"])
        )

    if left_arm["visible"] and right_arm["visible"]:
        block_score = 0.5 * left_block + 0.5 * right_block
        wrist_symmetry = abs(left_wrist[1] - right_wrist[1]) <= 0.35 * base_size
        if wrist_symmetry:
            block_score += 0.1
        if left_arm["near_head"] and right_arm["near_head"]:
            block_score += 0.1
    else:
        block_score = max(left_block, right_block)
        if upper_body_only_mode:
            block_score += 0.1
    block_score = clamp_score(block_score)

    escape_score = 0.0
    body_offset = abs(shoulder_center[0] - (frame_width / 2)) / max(frame_width, 1)
    ankles_visible = left_ankle is not None and right_ankle is not None
    ankle_span = point_distance(left_ankle, right_ankle) / max(base_size, 1.0) if ankles_visible else 0.0
    hips_visible = left_hip is not None and right_hip is not None
    shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1]) / max(base_size, 1.0)
    hip_shift = 0.0
    if hips_visible:
        hip_center_x = (left_hip[0] + right_hip[0]) / 2
        hip_shift = abs(hip_center_x - shoulder_center[0]) / max(base_size, 1.0)
    lowered_arms = float(left_arm["lowered"]) + float(right_arm["lowered"])
    escape_score = (
        0.3 * clamp_score((body_offset - 0.06) / 0.18)
        + 0.2 * clamp_score((ankle_span - 0.8) / 0.8)
        + 0.25 * clamp_score(lowered_arms / 2.0)
        + 0.15 * clamp_score((shoulder_tilt - 0.08) / 0.2)
        + 0.1 * clamp_score((hip_shift - 0.08) / 0.2)
    )
    if left_arm["guard"] or right_arm["guard"]:
        escape_score *= 0.7
    if upper_body_only_mode:
        escape_score = clamp_score(escape_score + 0.08 * clamp_score((body_offset - 0.05) / 0.15))
    escape_score = clamp_score(escape_score)

    scores = {
        "Punch": round(punch_score, 3),
        "Block": round(block_score, 3),
        "Escape": round(escape_score, 3),
    }
    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]
    if best_score < 0.55:
        return {"label": "Unknown", "confidence": best_score, "scores": scores, "reason": "low_confidence"}
    return {"label": best_label, "confidence": best_score, "scores": scores, "reason": "classified"}


def smooth_technique_result(history: deque) -> dict:
    if not history:
        return {"label": "Unknown", "confidence": 0.0, "scores": {name: 0.0 for name in TECHNIQUE_NAMES}}

    count_map = {name: 0 for name in TECHNIQUE_NAMES}
    confidence_map = {name: 0.0 for name in TECHNIQUE_NAMES}
    last_scores = {name: 0.0 for name in TECHNIQUE_NAMES}

    for result in history:
        for name in TECHNIQUE_NAMES:
            last_scores[name] = max(last_scores[name], result.get("scores", {}).get(name, 0.0))
        label = result.get("label", "Unknown")
        if label in count_map:
            count_map[label] += 1
            confidence_map[label] += result.get("confidence", 0.0)

    best_label = max(TECHNIQUE_NAMES, key=lambda name: (count_map[name], confidence_map[name]))
    required_votes = max(1, len(history) // 2 + 1)
    average_confidence = confidence_map[best_label] / max(count_map[best_label], 1)

    if count_map[best_label] < required_votes or average_confidence < 0.55:
        return {"label": "Unknown", "confidence": average_confidence, "scores": last_scores, "reason": "not_stable"}

    return {"label": best_label, "confidence": average_confidence, "scores": last_scores, "reason": "stable"}


def build_pose_feedback(points, frame_shape, frame_visibility: dict, target_technique: str | None = None) -> tuple[str, list[str], tuple[int, int, int]]:
    frame_height, frame_width = frame_shape[:2]
    feedback_lines = []
    detected_parts = sum(point is not None for point in points)
    detection_rate = detected_parts / len(points)
    tolerance_y = max(int(frame_height * 0.08), 20)
    tolerance_x = max(int(frame_width * 0.12), 30)

    nose = get_point(points, "Nose")
    neck = get_point(points, "Neck")
    left_shoulder = get_point(points, "LShoulder")
    right_shoulder = get_point(points, "RShoulder")
    left_wrist = get_point(points, "LWrist")
    right_wrist = get_point(points, "RWrist")
    left_elbow = get_point(points, "LElbow")
    right_elbow = get_point(points, "RElbow")
    left_hip = get_point(points, "LHip")
    right_hip = get_point(points, "RHip")
    left_ankle = get_point(points, "LAnkle")
    right_ankle = get_point(points, "RAnkle")
    upper_body_visible = count_visible_points(points, ("Nose", "Neck", "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist"))
    lower_body_visible = count_visible_points(points, ("LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle"))

    def build_arm_state(shoulder, elbow, wrist, direction: int) -> dict:
        if neck is None or shoulder is None or elbow is None or wrist is None:
            return {
                "visible": False,
                "horizontal": False,
                "guard": False,
                "raised": False,
                "lowered": False,
                "bent": False,
                "outward": False,
            }

        reach = point_distance(wrist, shoulder) / max(point_distance(left_shoulder, right_shoulder), 1.0)
        return {
            "visible": True,
            "horizontal": abs(wrist[1] - shoulder[1]) <= 0.35 * max(point_distance(left_shoulder, right_shoulder), 1.0),
            "guard": wrist[1] <= neck[1] + 0.35 * max(point_distance(left_shoulder, right_shoulder), 1.0) and abs(wrist[0] - neck[0]) <= 1.0 * max(point_distance(left_shoulder, right_shoulder), 1.0),
            "raised": wrist[1] <= shoulder[1] + 0.1 * max(point_distance(left_shoulder, right_shoulder), 1.0),
            "lowered": wrist[1] >= shoulder[1] + 0.35 * max(point_distance(left_shoulder, right_shoulder), 1.0),
            "bent": reach <= 1.35,
            "outward": (wrist[0] - shoulder[0]) * direction >= 0.45 * max(point_distance(left_shoulder, right_shoulder), 1.0),
        }

    left_arm = build_arm_state(left_shoulder, left_elbow, left_wrist, -1)
    right_arm = build_arm_state(right_shoulder, right_elbow, right_wrist, 1)

    if frame_visibility["is_dark"] and frame_visibility["is_low_detail"]:
        return (
            "Camera image too dark",
            ["Check lighting, camera shutter, or try --camera-index 1"],
            COLOR_ERROR,
        )

    if detected_parts < 4 or detection_rate < 0.22 or neck is None:
        return "Pose not clear", ["Move back and show your shoulders and arms"], COLOR_ERROR

    if target_technique == "Punch":
        if upper_body_visible < 6:
            feedback_lines.append("Keep both shoulders, elbows, and wrists visible")
        if not (left_arm["outward"] or right_arm["outward"]):
            feedback_lines.append("Extend one arm forward for the punch")
        if not (left_arm["horizontal"] or right_arm["horizontal"]):
            feedback_lines.append("Keep the punching arm closer to shoulder height")
        if not (left_arm["guard"] or right_arm["guard"]):
            feedback_lines.append("Keep the other hand near your face as a guard")

        if not feedback_lines:
            return "Punch form correct", ["Hold the punch and keep your guard up"], COLOR_SUCCESS
        return "Adjust punch", feedback_lines[:3], COLOR_WARNING

    if target_technique == "Block":
        if upper_body_visible < 6:
            feedback_lines.append("Keep both shoulders, elbows, and wrists visible")
        if not (left_arm["raised"] and right_arm["raised"]):
            feedback_lines.append("Raise both arms higher to protect your head")
        if not (left_arm["bent"] and right_arm["bent"]):
            feedback_lines.append("Keep both arms bent for a tighter block")
        if not (left_arm["guard"] and right_arm["guard"]):
            feedback_lines.append("Keep both hands closer to your face")

        if not feedback_lines:
            return "Block form correct", ["Hold your guard position"], COLOR_SUCCESS
        return "Adjust block", feedback_lines[:3], COLOR_WARNING

    if target_technique == "Escape":
        if lower_body_visible <= 1:
            feedback_lines.append("Step back so your hips and legs are visible")
        if not (left_arm["lowered"] or right_arm["lowered"]):
            feedback_lines.append("Lower your arms slightly as you move away")
        if abs(neck[0] - (frame_width // 2)) < tolerance_x:
            feedback_lines.append("Shift your body off the center line")

        if not feedback_lines:
            return "Escape form correct", ["Keep moving away from the center"], COLOR_SUCCESS
        return "Adjust escape", feedback_lines[:3], COLOR_WARNING

    if upper_body_visible < 6:
        feedback_lines.append("Keep both shoulders, elbows, and wrists in frame")

    if nose is None:
        feedback_lines.append("Lift camera or keep your head visible")

    if lower_body_visible <= 1 and upper_body_visible >= 6:
        feedback_lines.append("Upper body framing is OK for punch and block")
    elif (left_hip is None or right_hip is None) and (left_ankle is None or right_ankle is None):
        feedback_lines.append("Step back if you want better escape detection")

    if neck is not None:
        center_offset = neck[0] - (frame_width // 2)
        if center_offset > tolerance_x:
            feedback_lines.append("Move slightly to your left")
        elif center_offset < -tolerance_x:
            feedback_lines.append("Move slightly to your right")

    arm_messages = []
    for label, shoulder, elbow, wrist in [
        ("Left", left_shoulder, left_elbow, left_wrist),
        ("Right", right_shoulder, right_elbow, right_wrist),
    ]:
        if shoulder is None or elbow is None or wrist is None:
            arm_messages.append(f"Keep your {label.lower()} arm visible")
            continue

        delta_y = wrist[1] - shoulder[1]
        if delta_y > tolerance_y:
            arm_messages.append(f"Raise {label.lower()} arm higher")
        elif delta_y < -tolerance_y:
            arm_messages.append(f"Lower {label.lower()} arm slightly")
        else:
            arm_messages.append(f"{label} arm correct!")

    if all(message.endswith("correct!") for message in arm_messages[:2]):
        feedback_lines.append("Arm position correct!")
    else:
        for message in arm_messages:
            if not message.endswith("correct!"):
                feedback_lines.append(message)

    if not feedback_lines:
        return "Correct!", ["Pose detected well"], COLOR_SUCCESS

    if len(feedback_lines) == 1 and feedback_lines[0] == "Arm position correct!":
        return "Correct!", feedback_lines, COLOR_SUCCESS

    return "Adjust pose", feedback_lines[:3], COLOR_WARNING


def draw_technique_info(frame, technique_result: dict, target_technique: str | None = None) -> None:
    layout = get_overlay_layout(frame.shape[1])
    label = technique_result.get("label", "Unknown")
    scores = technique_result.get("scores", {})
    confidence = technique_result.get("confidence", 0.0) * 100
    panel_width = layout["right_width"]
    panel_height = 194 if target_technique else 148
    panel_left = layout["right_left"]
    panel_top = 12

    overlay = frame.copy()
    cv.rectangle(overlay, (panel_left, panel_top), (panel_left + panel_width, panel_top + panel_height), COLOR_PANEL, cv.FILLED)
    cv.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    if target_technique:
        selected_score = scores.get(target_technique, 0.0)
        label_color = score_to_color(selected_score)
        headline = f"Selected: {target_technique}"
    elif label == "Unknown":
        label_color = COLOR_STATUS_UNKNOWN
        headline = f"Detected: {label}"
    else:
        label_color = COLOR_SUCCESS
        headline = f"Detected: {label} ({confidence:.0f}%)"

    cv.putText(frame, "PHASE 2", (panel_left + 12, panel_top + 18), cv.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_LABEL, 1, cv.LINE_AA)
    cv.putText(frame, "Technique Feedback", (panel_left + 12, panel_top + 42), cv.FONT_HERSHEY_SIMPLEX, 0.64, COLOR_INFO, 2, cv.LINE_AA)
    cv.putText(frame, headline, (panel_left + 12, panel_top + 70), cv.FONT_HERSHEY_SIMPLEX, 0.62, label_color, 2, cv.LINE_AA)

    score_items = [(target_technique, scores.get(target_technique, 0.0))] if target_technique else [(name, scores.get(name, 0.0)) for name in TECHNIQUE_NAMES]
    for index, (name, score) in enumerate(score_items, start=1):
        score_percent = int(round(score * 100))
        bar_left = panel_left + 112
        bar_top = panel_top + 78 + index * 28
        bar_width = max(90, panel_width - 220)
        fill_width = int(bar_width * max(0.0, min(score, 1.0)))
        fill_color = score_to_color(score)
        cv.putText(frame, f"{name}:", (panel_left + 12, bar_top + 5), cv.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_INFO, 1, cv.LINE_AA)
        cv.rectangle(frame, (bar_left, bar_top - 8), (bar_left + bar_width, bar_top + 4), COLOR_BAR_BG, 1)
        if fill_width > 0:
            cv.rectangle(frame, (bar_left + 1, bar_top - 7), (bar_left + fill_width, bar_top + 3), fill_color, cv.FILLED)
        cv.putText(frame, f"{score_percent}%", (bar_left + bar_width + 10, bar_top + 5), cv.FONT_HERSHEY_SIMPLEX, 0.47, COLOR_MUTED, 1, cv.LINE_AA)

    if target_technique:
        matched = label == target_technique and technique_result.get("confidence", 0.0) >= 0.55
        result_text = "Status: Matched" if matched else "Status: Not Matched Yet"
        follow_up = "Good form. Hold it steady." if matched else "Keep adjusting until the selected move matches"
        result_color = COLOR_SUCCESS if matched else COLOR_WARNING
        follow_up_lines = wrap_text_lines(follow_up, panel_width - 24, 0.47, 1)
        status_y = panel_top + panel_height - 44
        cv.putText(frame, result_text, (panel_left + 12, status_y), cv.FONT_HERSHEY_SIMPLEX, 0.54, result_color, 2, cv.LINE_AA)
        for line_index, follow_line in enumerate(follow_up_lines[:2], start=1):
            cv.putText(frame, follow_line, (panel_left + 12, status_y + 22 * line_index), cv.FONT_HERSHEY_SIMPLEX, 0.47, COLOR_MUTED, 1, cv.LINE_AA)
    else:
        cv.putText(frame, "Technique guide: Punch / Block / Escape", (panel_left + 12, panel_top + panel_height - 16), cv.FONT_HERSHEY_SIMPLEX, 0.47, COLOR_MUTED, 1, cv.LINE_AA)


def annotate_frame(
    frame,
    points,
    inference_ms: float,
    feedback_status: str,
    feedback_lines: list[str],
    feedback_color,
    technique_result: dict,
    detected_parts: int,
    frame_visibility: dict,
    target_technique: str | None = None,
):
    annotated = frame.copy()
    layout = get_overlay_layout(annotated.shape[1])
    for part_from, part_to in POSE_PAIRS:
        index_from = BODY_PARTS[part_from]
        index_to = BODY_PARTS[part_to]
        if points[index_from] and points[index_to]:
            cv.line(annotated, points[index_from], points[index_to], (0, 255, 0), 3)
            cv.ellipse(annotated, points[index_from], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(annotated, points[index_to], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    panel_width = layout["left_width"]
    panel_height = min(194, annotated.shape[0] - 24)
    overlay = annotated.copy()
    cv.rectangle(overlay, (layout["left"], 12), (layout["left"] + panel_width, 12 + panel_height), COLOR_PANEL_ALT, cv.FILLED)
    cv.addWeighted(overlay, 0.82, annotated, 0.18, 0, annotated)

    cv.putText(
        annotated,
        "PHASE 2",
        (layout["left"] + 12, 28),
        cv.FONT_HERSHEY_SIMPLEX,
        0.42,
        COLOR_LABEL,
        1,
        cv.LINE_AA,
    )

    cv.putText(
        annotated,
        "Live Pose Feedback",
        (layout["left"] + 12, 52),
        cv.FONT_HERSHEY_SIMPLEX,
        0.72,
        COLOR_INFO,
        2,
        cv.LINE_AA,
    )

    cv.putText(
        annotated,
        f"Inference: {inference_ms:.2f}ms",
        (layout["left"] + 12, 78),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_MUTED,
        1,
        cv.LINE_AA,
    )

    if target_technique:
        cv.putText(
            annotated,
            f"Selected move: {target_technique}",
            (layout["left"] + 12, 102),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_LABEL,
            1,
            cv.LINE_AA,
        )

    cv.putText(
        annotated,
        feedback_status,
        (layout["left"] + 12, 128),
        cv.FONT_HERSHEY_SIMPLEX,
        0.85,
        feedback_color,
        2,
        cv.LINE_AA,
    )

    wrapped_feedback_lines: list[str] = []
    text_width = panel_width - 30
    for line in feedback_lines:
        wrapped_feedback_lines.extend(wrap_text_lines(f"- {line}", text_width, 0.55, 1))

    for index, line in enumerate(wrapped_feedback_lines[:4], start=1):
        cv.putText(
            annotated,
            line,
            (layout["left"] + 12, 128 + index * 22),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_INFO,
            1,
            cv.LINE_AA,
        )

    cv.putText(
        annotated,
        "Press F for fullscreen, Q or Esc to quit",
        (layout["left"] + 12, min(128 + (len(wrapped_feedback_lines) + 1) * 22, 12 + panel_height - 12)),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_MUTED,
        1,
        cv.LINE_AA,
    )

    draw_technique_info(annotated, technique_result, target_technique)

    if frame_visibility["is_dark"] and frame_visibility["is_low_detail"]:
        overlay = annotated.copy()
        center_box_width = min(annotated.shape[1] - 40, 620)
        center_box_height = min(170, annotated.shape[0] - 60)
        left = max((annotated.shape[1] - center_box_width) // 2, 20)
        top = max((annotated.shape[0] - center_box_height) // 2, 30)
        cv.rectangle(overlay, (left, top), (left + center_box_width, top + center_box_height), (0, 0, 110), cv.FILLED)
        cv.addWeighted(overlay, 0.62, annotated, 0.38, 0, annotated)
        cv.putText(annotated, "STATUS: CAMERA IMAGE TOO DARK", (left + 24, top + 56), cv.FONT_HERSHEY_SIMPLEX, 0.92, COLOR_INFO, 3, cv.LINE_AA)
        cv.putText(annotated, "Check room lighting, privacy shutter, or camera index", (left + 24, top + 98), cv.FONT_HERSHEY_SIMPLEX, 0.68, COLOR_INFO, 2, cv.LINE_AA)
        cv.putText(annotated, f"Brightness: {frame_visibility['mean_brightness']:.1f}", (left + 24, top + 136), cv.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_MUTED, 2, cv.LINE_AA)
    elif detected_parts < 4:
        overlay = annotated.copy()
        center_box_width = min(annotated.shape[1] - 40, 560)
        center_box_height = min(150, annotated.shape[0] - 60)
        left = max((annotated.shape[1] - center_box_width) // 2, 20)
        top = max((annotated.shape[0] - center_box_height) // 2, 30)
        cv.rectangle(overlay, (left, top), (left + center_box_width, top + center_box_height), (0, 0, 90), cv.FILLED)
        cv.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)
        cv.putText(annotated, "STATUS: POSE NOT CLEAR", (left + 28, top + 52), cv.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_INFO, 3, cv.LINE_AA)
        cv.putText(annotated, "Move back and keep both arms visible", (left + 28, top + 92), cv.FONT_HERSHEY_SIMPLEX, 0.76, COLOR_INFO, 2, cv.LINE_AA)
        cv.putText(annotated, f"Detected body parts: {detected_parts}/{len(BODY_PARTS)}", (left + 28, top + 124), cv.FONT_HERSHEY_SIMPLEX, 0.62, COLOR_MUTED, 2, cv.LINE_AA)

    return annotated


def summarize_points(points) -> dict:
    detected_parts = sum(1 for point in points if point is not None)
    missing_parts = len(points) - detected_parts
    connected_pairs = 0

    for part_from, part_to in POSE_PAIRS:
        index_from = BODY_PARTS[part_from]
        index_to = BODY_PARTS[part_to]
        if points[index_from] and points[index_to]:
            connected_pairs += 1

    detection_rate = detected_parts / len(points) if points else 0.0
    return {
        "detected_parts": detected_parts,
        "missing_parts": missing_parts,
        "connected_pairs": connected_pairs,
        "detection_rate": detection_rate,
        "has_pose": detected_parts > 0,
    }


def infer_frame(frame, net: cv.dnn.Net, in_width: int, in_height: int, threshold: float, target_technique: str | None = None):
    points = detect_points(frame, net, in_width, in_height, threshold)
    ticks, _ = net.getPerfProfile()
    inference_ms = ticks / (cv.getTickFrequency() / 1000)
    summary = summarize_points(points)
    frame_visibility = analyze_frame_visibility(frame)
    feedback_status, feedback_lines, feedback_color = build_pose_feedback(points, frame.shape, frame_visibility, target_technique)
    technique_result = classify_technique(points, frame.shape)
    summary["inference_ms"] = inference_ms
    summary["feedback_status"] = feedback_status
    summary["feedback_lines"] = feedback_lines
    summary["technique_result"] = technique_result
    summary["frame_visibility"] = frame_visibility
    return (
        annotate_frame(
            frame,
            points,
            inference_ms,
            feedback_status,
            feedback_lines,
            feedback_color,
            technique_result,
            summary["detected_parts"],
            frame_visibility,
            target_technique,
        ),
        summary,
    )


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def write_image(output_path: Path, frame) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(output_path), frame):
        raise RuntimeError(f"Failed to write image output to '{output_path}'.")


def build_writer(output_path: Path, fps: float, frame_size: tuple[int, int]) -> cv.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer for '{output_path}'.")
    return writer


def show_frame(window_name: str, frame) -> bool:
    if window_name not in WINDOW_STATES:
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        WINDOW_STATES[window_name] = False
    cv.imshow(window_name, frame)
    key = cv.waitKey(1) & 0xFF
    if key in (ord("f"), ord("F")):
        is_fullscreen = WINDOW_STATES.get(window_name, False)
        cv.setWindowProperty(
            window_name,
            cv.WND_PROP_FULLSCREEN,
            cv.WINDOW_NORMAL if is_fullscreen else cv.WINDOW_FULLSCREEN,
        )
        WINDOW_STATES[window_name] = not is_fullscreen
        return False
    return key in (27, ord("q"))


def process_image(args: argparse.Namespace, net: cv.dnn.Net, input_path: Path) -> int:
    frame = cv.imread(str(input_path))
    if frame is None:
        return fail(f"Could not read image file '{input_path}'.")

    annotated, summary = infer_frame(frame, net, args.width, args.height, args.thr, args.target_technique)

    if args.output:
        write_image(Path(args.output), annotated)

    if not args.no_display:
        cv.imshow("OpenPose using OpenCV", annotated)
        cv.waitKey(0)

    print(
        f"Detected technique: {summary['technique_result']['label']} "
        f"({summary['technique_result']['confidence'] * 100:.0f}%)"
    )

    write_json_summary(
        args.json_output,
        {
            "mode": "image",
            "input_source": str(input_path),
            "output_path": args.output or "",
            "processed_frames": 1,
            "detected_parts": summary["detected_parts"],
            "missing_parts": summary["missing_parts"],
            "connected_pairs": summary["connected_pairs"],
            "detection_rate": summary["detection_rate"],
            "has_pose": summary["has_pose"],
            "inference_ms": summary["inference_ms"],
            "feedback_status": summary["feedback_status"],
            "feedback_lines": summary["feedback_lines"],
            "recognized_technique": summary["technique_result"]["label"],
            "recognized_confidence": summary["technique_result"]["confidence"],
            "target_technique": args.target_technique or "",
            "technique_match": bool(args.target_technique and summary["technique_result"]["label"] == args.target_technique and summary["technique_result"]["confidence"] >= 0.55),
            "technique_match_ratio": 1.0 if args.target_technique and summary["technique_result"]["label"] == args.target_technique and summary["technique_result"]["confidence"] >= 0.55 else 0.0,
        },
    )

    return 0


def process_stream(args: argparse.Namespace, net: cv.dnn.Net, input_source) -> int:
    capture, backend_name, attempted_backends = open_capture(input_source)
    if capture is None:
        return fail(build_capture_error(input_source, attempted_backends))

    writer = None
    frame_count = 0
    output_path = Path(args.output) if args.output else None
    window_name = "OpenPose using OpenCV"
    last_summary = None
    technique_history = deque(maxlen=max(1, args.confirm_frames))
    matched_frames = 0
    matched_repetitions = 0
    in_matched_streak = False

    try:
        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break

            annotated, summary = infer_frame(frame, net, args.width, args.height, args.thr, args.target_technique)
            technique_history.append(summary["technique_result"])
            stable_technique = smooth_technique_result(technique_history)
            summary["stable_technique"] = stable_technique
            last_summary = summary
            frame_count += 1

            if (
                args.target_technique
                and stable_technique["label"] == args.target_technique
                and stable_technique["confidence"] >= 0.55
            ):
                matched_frames += 1
                if not in_matched_streak:
                    matched_repetitions += 1
                    in_matched_streak = True
            else:
                in_matched_streak = False

            if output_path and writer is None:
                fps = capture.get(cv.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 25.0
                height, width = annotated.shape[:2]
                writer = build_writer(output_path, fps, (width, height))

            if writer is not None:
                writer.write(annotated)

            should_stop = False
            if not args.no_display:
                should_stop = show_frame(window_name, annotated)

            if should_stop:
                break

            if args.max_frames and frame_count >= args.max_frames:
                break

    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if frame_count == 0:
        if backend_name:
            return fail(
                f"No frames were processed from {describe_input_source(input_source)} using the {backend_name} backend. "
                "Close other camera apps, reconnect the webcam, or try a different USB camera."
            )
        return fail("No frames were processed from the selected input source.")

    if last_summary and last_summary["has_pose"]:
        stable = last_summary.get("stable_technique", last_summary.get("technique_result", {}))
        print(
            "Processed"
            f" {frame_count} frame(s); detected {last_summary['detected_parts']} body parts"
            f" with {last_summary['connected_pairs']} skeletal connections in the final frame."
        )
        print(f"Feedback: {last_summary['feedback_status']} - {'; '.join(last_summary['feedback_lines'])}")
        print(f"Technique: {stable.get('label', 'Unknown')} ({stable.get('confidence', 0.0) * 100:.0f}%)")
        if args.target_technique:
            print(f"Target match ratio: {(matched_frames / frame_count) * 100:.0f}%")
            print(f"Matched repetitions: {matched_repetitions}")

    stable = last_summary.get("stable_technique", last_summary.get("technique_result", {})) if last_summary else {}
    write_json_summary(
        args.json_output,
        {
            "mode": "webcam" if isinstance(input_source, int) else "video",
            "input_source": describe_input_source(input_source),
            "output_path": str(output_path) if output_path else "",
            "processed_frames": frame_count,
            "detected_parts": last_summary["detected_parts"] if last_summary else 0,
            "missing_parts": last_summary["missing_parts"] if last_summary else len(BODY_PARTS),
            "connected_pairs": last_summary["connected_pairs"] if last_summary else 0,
            "detection_rate": last_summary["detection_rate"] if last_summary else 0.0,
            "has_pose": bool(last_summary and last_summary["has_pose"]),
            "inference_ms": last_summary["inference_ms"] if last_summary else 0.0,
            "feedback_status": last_summary["feedback_status"] if last_summary else "Pose not clear",
            "feedback_lines": last_summary["feedback_lines"] if last_summary else ["Move into frame and face the camera"],
            "recognized_technique": stable.get("label", "Unknown"),
            "recognized_confidence": stable.get("confidence", 0.0),
            "target_technique": args.target_technique or "",
            "technique_match": bool(args.target_technique and matched_frames > 0),
            "technique_match_ratio": (matched_frames / frame_count) if args.target_technique and frame_count else 0.0,
            "matched_repetitions": matched_repetitions,
        },
    )

    return 0


def analyze_image_file(
    input_path: Path,
    model_path: Path,
    threshold: float = 0.2,
    width: int = 368,
    height: int = 368,
    output_path: Path | None = None,
    display: bool = False,
    target_technique: str | None = None,
) -> dict:
    net = load_net(model_path)
    frame = cv.imread(str(input_path))
    if frame is None:
        raise ValueError(f"Could not read image file '{input_path}'.")

    annotated, summary = infer_frame(frame, net, width, height, threshold, target_technique)

    if output_path is not None:
        write_image(output_path, annotated)

    if display:
        cv.imshow("OpenPose using OpenCV", annotated)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return {
        "mode": "image",
        "input_source": str(input_path),
        "output_path": str(output_path) if output_path else "",
        "processed_frames": 1,
        "recognized_technique": summary["technique_result"]["label"],
        "recognized_confidence": summary["technique_result"]["confidence"],
        "target_technique": target_technique or "",
        "technique_match": bool(target_technique and summary["technique_result"]["label"] == target_technique and summary["technique_result"]["confidence"] >= 0.55),
        "technique_match_ratio": 1.0 if target_technique and summary["technique_result"]["label"] == target_technique and summary["technique_result"]["confidence"] >= 0.55 else 0.0,
        **summary,
    }


def analyze_stream_source(
    input_source,
    model_path: Path,
    threshold: float = 0.2,
    width: int = 368,
    height: int = 368,
    output_path: Path | None = None,
    display: bool = False,
    max_frames: int | None = None,
    target_technique: str | None = None,
    confirm_frames: int = 5,
) -> dict:
    net = load_net(model_path)
    capture, backend_name, attempted_backends = open_capture(input_source)
    if capture is None:
        raise ValueError(build_capture_error(input_source, attempted_backends))

    writer = None
    frame_count = 0
    total_detected_parts = 0
    total_connected_pairs = 0
    pose_frames = 0
    last_summary = None
    resolved_output = output_path if output_path else None
    technique_history = deque(maxlen=max(1, confirm_frames))
    matched_frames = 0
    matched_repetitions = 0
    in_matched_streak = False
    recognized_counts = {name: 0 for name in TECHNIQUE_NAMES}

    try:
        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break

            annotated, summary = infer_frame(frame, net, width, height, threshold, target_technique)
            technique_history.append(summary["technique_result"])
            stable_technique = smooth_technique_result(technique_history)
            frame_count += 1
            last_summary = summary
            total_detected_parts += summary["detected_parts"]
            total_connected_pairs += summary["connected_pairs"]
            if summary["has_pose"]:
                pose_frames += 1
            if stable_technique["label"] in recognized_counts:
                recognized_counts[stable_technique["label"]] += 1
            if (
                target_technique
                and stable_technique["label"] == target_technique
                and stable_technique["confidence"] >= 0.55
            ):
                matched_frames += 1
                if not in_matched_streak:
                    matched_repetitions += 1
                    in_matched_streak = True
            else:
                in_matched_streak = False
            summary["stable_technique"] = stable_technique

            if resolved_output and writer is None:
                fps = capture.get(cv.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 25.0
                frame_height, frame_width = annotated.shape[:2]
                writer = build_writer(resolved_output, fps, (frame_width, frame_height))

            if writer is not None:
                writer.write(annotated)

            if display and show_frame("OpenPose using OpenCV", annotated):
                break

            if max_frames and frame_count >= max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if display:
            cv.destroyAllWindows()

    if frame_count == 0:
        if backend_name:
            raise ValueError(
                f"No frames were processed from {describe_input_source(input_source)} using the {backend_name} backend. "
                "Close other camera apps, reconnect the webcam, or try a different USB camera."
            )
        raise ValueError("No frames were processed from the selected input source.")

    average_detected_parts = total_detected_parts / frame_count
    average_connected_pairs = total_connected_pairs / frame_count
    pose_frame_ratio = pose_frames / frame_count
    stable = last_summary.get("stable_technique", last_summary.get("technique_result", {})) if last_summary else {}
    technique_match_ratio = matched_frames / frame_count if target_technique else 0.0
    technique_match = bool(target_technique and technique_match_ratio > 0.0)

    return {
        "mode": "webcam" if isinstance(input_source, int) else "video",
        "input_source": describe_input_source(input_source),
        "output_path": str(resolved_output) if resolved_output else "",
        "processed_frames": frame_count,
        "detected_parts": last_summary["detected_parts"] if last_summary else 0,
        "missing_parts": last_summary["missing_parts"] if last_summary else len(BODY_PARTS),
        "connected_pairs": last_summary["connected_pairs"] if last_summary else 0,
        "detection_rate": last_summary["detection_rate"] if last_summary else 0.0,
        "has_pose": bool(last_summary and last_summary["has_pose"]),
        "inference_ms": last_summary["inference_ms"] if last_summary else 0.0,
        "feedback_status": last_summary["feedback_status"] if last_summary else "Pose not clear",
        "feedback_lines": last_summary["feedback_lines"] if last_summary else ["Move into frame and face the camera"],
        "recognized_technique": stable.get("label", "Unknown"),
        "recognized_confidence": stable.get("confidence", 0.0),
        "target_technique": target_technique or "",
        "technique_match": technique_match,
        "technique_match_ratio": technique_match_ratio,
        "matched_repetitions": matched_repetitions,
        "recognized_counts": recognized_counts,
        "average_detected_parts": average_detected_parts,
        "average_connected_pairs": average_connected_pairs,
        "pose_frame_ratio": pose_frame_ratio,
    }


def main() -> int:
    args = parse_args()

    if args.width <= 0 or args.height <= 0:
        return fail("--width and --height must be positive integers.")
    if not 0 <= args.thr <= 1:
        return fail("--thr must be between 0 and 1.")
    if args.confirm_frames <= 0:
        return fail("--confirm-frames must be a positive integer.")
    if args.camera_index < 0:
        return fail("--camera-index must be zero or a positive integer.")
    if args.list_cameras_json:
        write_json_summary(args.list_cameras_json, {"cameras": discover_working_cameras()})
        return 0

    model_path = Path(args.model).expanduser().resolve()

    try:
        net = load_net(model_path)
    except Exception as exc:
        return fail(str(exc))

    try:
        if args.input:
            input_path = Path(args.input).expanduser().resolve()
            if not input_path.exists():
                return fail(f"Input path '{input_path}' does not exist.")
            if is_image_path(input_path):
                return process_image(args, net, input_path)
            return process_stream(args, net, str(input_path))
        return process_stream(args, net, args.camera_index)
    finally:
        cv.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())