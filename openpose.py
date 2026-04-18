from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import numpy as np

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
DEFAULT_PRESET_NAME = "fast"
WEBCAM_PRESETS = {
    "fast": {"name": "fast", "width": 288, "height": 288, "confirm_frames": 3},
    "balanced": {"name": "balanced", "width": 320, "height": 320, "confirm_frames": 4},
    "strict": {"name": "strict", "width": 368, "height": 368, "confirm_frames": 6},
}
PRESET_MATCH_BEHAVIOR = {
    "fast": {"allow_near_match": True},
    "balanced": {"allow_near_match": False},
    "strict": {"allow_near_match": False},
}
TECHNIQUE_PROFILES = {
    "Punch": {
        "min_confidence": 0.34,
        "match_threshold": 0.41,
        "tracking_threshold": 0.3,
        "motion_frames": 4,
    },
    "Block": {
        "min_confidence": 0.5,
        "match_threshold": 0.54,
        "tracking_threshold": 0.44,
        "motion_frames": 4,
    },
    "Escape": {
        "min_confidence": 0.3,
        "match_threshold": 0.37,
        "tracking_threshold": 0.27,
        "motion_frames": 5,
    },
}
BODY_PART_LABELS = {
    "Nose": "Nose",
    "Neck": "Neck",
    "RShoulder": "Right shoulder",
    "RElbow": "Right elbow",
    "RWrist": "Right wrist",
    "LShoulder": "Left shoulder",
    "LElbow": "Left elbow",
    "LWrist": "Left wrist",
    "RHip": "Right hip",
    "RKnee": "Right knee",
    "RAnkle": "Right ankle",
    "LHip": "Left hip",
    "LKnee": "Left knee",
    "LAnkle": "Left ankle",
    "REye": "Right eye",
    "LEye": "Left eye",
    "REar": "Right ear",
    "LEar": "Left ear",
}
TARGET_DIAGNOSTIC_PARTS = {
    "Punch": ("Neck", "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist", "Nose"),
    "Block": ("Neck", "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist", "Nose"),
    "Escape": ("Neck", "LShoulder", "RShoulder", "LHip", "RHip", "LAnkle", "RAnkle", "LWrist", "RWrist"),
}
CAMERA_PROBE_MAX_INDEX = 5
WINDOWS_CAMERA_BACKENDS = (
    ("DirectShow", cv.CAP_DSHOW),
    ("Media Foundation", cv.CAP_MSMF),
    ("Auto", cv.CAP_ANY),
)
WINDOW_STATES: dict[str, bool] = {}
COLOR_SUCCESS = (88, 196, 102)
COLOR_WARNING = (0, 191, 255)
COLOR_ERROR = (82, 96, 255)
COLOR_INFO = (245, 247, 250)
COLOR_MUTED = (176, 186, 198)
COLOR_PANEL = (22, 26, 32)
COLOR_PANEL_ALT = (16, 20, 26)
COLOR_LABEL = (255, 214, 120)
COLOR_BAR_BG = (56, 66, 78)
COLOR_STATUS_UNKNOWN = (255, 214, 120)
COLOR_PANEL_EDGE = (74, 86, 101)
COLOR_PANEL_GLOW = (255, 181, 76)
COLOR_META_FILL = (38, 45, 56)
COLOR_META_FILL_ALT = (29, 35, 44)
COLOR_TEXT_SUBTLE = (131, 145, 161)
COLOR_SKELETON = (90, 215, 154)
COLOR_JOINT = (255, 214, 120)
OVERLAY_MARGIN = 16
OVERLAY_GAP = 12
PANEL_TOP = 20
PANEL_BOTTOM_MARGIN = 20
PANEL_PADDING = 16
CARD_ALPHA = 0.82
ALERT_ALPHA = 0.72
HEADER_TITLE_OFFSET = 20
HEADER_BLOCK_HEIGHT = 34
SECTION_GAP = 10
CHIP_ROW_GAP = 10
TEXT_ROW_GAP = 16
BAR_TOP_OFFSET = 12
BAR_BLOCK_HEIGHT = 28
META_WRAP_GAP = 6
META_ROW_HEIGHT = 24
META_CHIP_HEIGHT = 22
FOOTER_BOTTOM_MARGIN = 30
FLOATING_OVERLAY_MARGIN = 8
ALERT_LINE_GAP = 24
ALERT_TOP_SHIFT = 36
ALERT_MAX_WIDTH = 520
ALERT_MAX_HEIGHT = 148
ALERT_HEADER_FONT = 0.46
ALERT_BODY_FONT = 0.44
ALERT_META_FONT = 0.38
MATCH_HEADLINE_GAP = 10
MATCH_BAR_START_GAP = 40
MATCH_STATUS_GAP = 14
MATCH_FOLLOWUP_GAP = 16
MATCH_STACK_GAP = 10
MATCH_FOLLOWUP_TOP_OFFSET = 14
MATCH_STATUS_TOP_OFFSET = 88
MATCH_CARD_MIN_HEIGHT = 200
MATCH_CARD_BOTTOM_PADDING = 16
FEEDBACK_CARD_BOTTOM_PADDING = 8
REQUIRED_PANEL_BOTTOM_OFFSET = 72
REQUIRED_PANEL_MIN_HEIGHT = 120
REQUIRED_PANEL_ROW_GAP = 16
REQUIRED_PANEL_WIDTH_RATIO = 0.84
REQUIRED_PANEL_MIN_WIDTH = 210
REQUIRED_SUMMARY_BOTTOM_GAP = 12
FEEDBACK_STATUS_GAP = 14
FEEDBACK_STATUS_TOP_OFFSET = 96
TIPS_LINE_GAP = 16
CARD_WIDTH_RATIO = 0.27
CARD_MAX_WIDTH = 300
CARD_MIN_WIDTH = 210
CARD_MIN_HEIGHT = 120
CARD_MAX_HEIGHT = 228


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
        "--preset",
        default=DEFAULT_PRESET_NAME,
        choices=tuple(WEBCAM_PRESETS),
        help="Webcam preset that balances speed and stability.",
    )
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parent / "models" / "graph_opt.pb"),
        help="Path to the TensorFlow graph (.pb) file.",
    )
    parser.add_argument("--thr", default=0.2, type=float, help="Confidence threshold for pose parts.")
    parser.add_argument("--width", type=int, help="Optional network input width override.")
    parser.add_argument("--height", type=int, help="Optional network input height override.")
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
        type=int,
        help="Optional number of recent frames to use when stabilizing technique recognition.",
    )
    parser.add_argument(
        "--json-output",
        help="Optional path to write a JSON summary of the final result.",
    )
    parser.add_argument(
        "--list-cameras-json",
        help="Optional path to write detected camera indices as JSON and exit.",
    )
    parser.add_argument(
        "--reference-image",
        help="Optional path to a real-photo pose reference image to show beside the live webcam feed.",
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


def get_webcam_preset(name: str | None) -> dict:
    preset_name = (name or DEFAULT_PRESET_NAME).strip().lower()
    return WEBCAM_PRESETS.get(preset_name, WEBCAM_PRESETS[DEFAULT_PRESET_NAME])


def resolve_runtime_settings(
    preset: str | None = None,
    width: int | None = None,
    height: int | None = None,
    confirm_frames: int | None = None,
) -> dict:
    preset_config = get_webcam_preset(preset)
    return {
        "preset": preset_config["name"],
        "width": width if width is not None else preset_config["width"],
        "height": height if height is not None else preset_config["height"],
        "confirm_frames": confirm_frames if confirm_frames is not None else preset_config["confirm_frames"],
    }


def get_target_match_thresholds(target_technique: str | None, preset: str | None = None) -> dict:
    preset_name = get_webcam_preset(preset)["name"]
    profile = TECHNIQUE_PROFILES.get(target_technique or "")
    allow_near_match = bool(profile and PRESET_MATCH_BEHAVIOR.get(preset_name, {}).get("allow_near_match"))
    match_threshold = profile["match_threshold"] if profile else 0.55
    near_match_threshold = profile["min_confidence"] if profile and allow_near_match else match_threshold
    return {
        "preset": preset_name,
        "allow_near_match": allow_near_match,
        "match_threshold": match_threshold,
        "near_match_threshold": near_match_threshold,
    }


def evaluate_target_match(technique_result: dict, target_technique: str | None, preset: str | None = None) -> dict:
    thresholds = get_target_match_thresholds(target_technique, preset)
    if target_technique not in TECHNIQUE_PROFILES:
        return {
            **thresholds,
            "state": "none",
            "full_match": False,
            "near_match": False,
            "qualifying_match": False,
        }

    label = technique_result.get("label", "Unknown")
    confidence = float(technique_result.get("confidence", 0.0))
    full_match = label == target_technique and confidence >= thresholds["match_threshold"]
    near_match = (
        label == target_technique
        and thresholds["allow_near_match"]
        and confidence >= thresholds["near_match_threshold"]
        and not full_match
    )
    state = "full" if full_match else "near" if near_match else "none"
    return {
        **thresholds,
        "state": state,
        "full_match": full_match,
        "near_match": near_match,
        "qualifying_match": full_match or near_match,
    }


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


def get_overlay_layout(frame_width: int, frame_height: int, include_required_panel: bool = False) -> dict:
    margin = OVERLAY_MARGIN
    gap = OVERLAY_GAP
    card_count = 3 if include_required_panel else 2
    available_width = max(frame_width - (margin * 2), CARD_MIN_WIDTH)
    card_width = min(CARD_MAX_WIDTH, max(CARD_MIN_WIDTH, int(available_width * CARD_WIDTH_RATIO)))
    usable_height = max(frame_height - PANEL_TOP - PANEL_BOTTOM_MARGIN, card_count)
    card_height = min(CARD_MAX_HEIGHT, max(CARD_MIN_HEIGHT, (usable_height - (gap * (card_count - 1))) // card_count))
    min_gap = 6
    total_stack_height = (card_height * card_count) + (gap * (card_count - 1))
    if total_stack_height > usable_height and card_count > 1:
        gap = max(min_gap, (usable_height - (card_height * card_count)) // (card_count - 1))
        total_stack_height = (card_height * card_count) + (gap * (card_count - 1))
    if total_stack_height > usable_height:
        gap = min_gap
        card_height = max(96, (usable_height - (gap * (card_count - 1))) // card_count)
    feedback_top = PANEL_TOP
    match_top = feedback_top + card_height + gap
    required_top = match_top + card_height + gap if include_required_panel else None
    return {
        "margin": margin,
        "gap": gap,
        "card_left": margin,
        "card_width": card_width,
        "card_height": card_height,
        "feedback_top": feedback_top,
        "match_top": match_top,
        "required_top": required_top,
        "panel_top": PANEL_TOP,
    }


def clamp_score(value: float) -> float:
    return max(0.0, min(value, 1.0))


def fit_text_to_width(text: str, max_width: int, font_scale: float, thickness: int) -> str:
    if cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0] <= max_width:
        return text

    ellipsis = "..."
    truncated = text
    while truncated:
        truncated = truncated[:-1]
        candidate = f"{truncated}{ellipsis}"
        if cv.getTextSize(candidate, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0] <= max_width:
            return candidate
    return ellipsis


def wrap_text_lines_limited(text: str, max_width: int, font_scale: float, thickness: int, max_lines: int) -> list[str]:
    lines = wrap_text_lines(text, max_width, font_scale, thickness)
    if len(lines) <= max_lines:
        return lines
    trimmed = lines[:max_lines]
    trimmed[-1] = fit_text_to_width(trimmed[-1], max_width, font_scale, thickness)
    return trimmed


def compact_overlay_status(text: str) -> str:
    status_map = {
        "Camera image too dark": "Low light",
        "Pose not clear": "Pose unclear",
        "Adjust pose": "Adjust pose",
        "Adjust punch": "Adjust punch",
        "Adjust block": "Adjust block",
        "Adjust escape": "Adjust escape",
        "Punch form correct": "Punch ready",
        "Block form correct": "Block ready",
        "Escape form correct": "Escape ready",
        "Correct!": "Good form",
    }
    return status_map.get(text, text)


def compact_overlay_line(text: str) -> str:
    line_map = {
        "Guide: keep shoulders, elbows, and wrists visible": "Guide: keep shoulders and arms visible.",
        "Guide: keep one hand guarding your face": "Guide: keep one hand up as guard.",
        "Guide: extend one hand forward at shoulder height": "Guide: extend one hand at shoulder height.",
        "Guide: aim the punch through the center line": "Guide: aim through the center line.",
        "Guide: hold the punch briefly before returning": "Guide: hold the punch briefly.",
        "Guide: keep one hand protecting your head": "Guide: keep one hand protecting your head.",
        "Guide: move backward or sideways off the center line": "Guide: move off the center line.",
        "Guide: keep moving to create distance": "Guide: keep moving to create distance.",
        "Guide: upper-body escape is visible; step back for a stronger match": "Guide: upper-body escape looks good. Step back for a stronger match.",
        "Move back and show your shoulders and arms": "Step back. Show shoulders and arms.",
        "Keep both shoulders, elbows, and wrists visible": "Keep shoulders and arms visible.",
        "Extend one arm forward for the punch": "Extend one arm forward.",
        "Keep the punching arm closer to shoulder height": "Keep the punch near shoulder height.",
        "Keep the other hand near your face as a guard": "Keep the other hand up as guard.",
        "Raise both arms higher to protect your head": "Raise both arms higher.",
        "Keep both arms bent for a tighter block": "Keep both arms bent.",
        "Keep both hands closer to your face": "Bring both hands closer to your face.",
        "Step back so your hips and legs are visible": "Step back so hips and legs show.",
        "Lower your arms slightly as you move away": "Lower your arms slightly.",
        "Shift your body off the center line": "Shift off the center line.",
        "Keep both shoulders, elbows, and wrists in frame": "Keep shoulders and arms in frame.",
        "Lift camera or keep your head visible": "Keep your head visible.",
        "Upper body framing is OK for punch and block": "Upper body framing looks fine.",
        "Step back if you want better escape detection": "Step back for better escape detection.",
        "Move slightly to your left": "Move a little left.",
        "Move slightly to your right": "Move a little right.",
        "Raise left arm higher": "Raise left arm.",
        "Raise right arm higher": "Raise right arm.",
        "Lower left arm slightly": "Lower left arm slightly.",
        "Lower right arm slightly": "Lower right arm slightly.",
        "Keep your left arm visible": "Keep left arm visible.",
        "Keep your right arm visible": "Keep right arm visible.",
        "Arm position correct!": "Arm position looks good.",
        "Pose detected well": "Pose looks good.",
    }
    return line_map.get(text, text)


def draw_panel(
    frame,
    left: int,
    top: int,
    width: int,
    height: int,
    *,
    fill_color,
    alpha: float,
    border_color=COLOR_PANEL_EDGE,
    accent_color=None,
    draw_border: bool = True,
) -> None:
    if alpha > 0:
        overlay = frame.copy()
        cv.rectangle(overlay, (left, top), (left + width, top + height), fill_color, cv.FILLED)
        cv.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    if draw_border:
        cv.rectangle(frame, (left, top), (left + width, top + height), border_color, 1)
    if draw_border and accent_color is not None:
        cv.line(frame, (left + 1, top + 1), (left + width - 1, top + 1), accent_color, 2, cv.LINE_AA)


def draw_section_header(frame, eyebrow: str, title: str, left: int, top: int) -> int:
    cv.putText(frame, eyebrow, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_LABEL, 1, cv.LINE_AA)
    cv.putText(frame, title, (left, top + HEADER_TITLE_OFFSET), cv.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_INFO, 1, cv.LINE_AA)
    return top + HEADER_BLOCK_HEIGHT


def draw_chip(
    frame,
    text: str,
    left: int,
    top: int,
    *,
    fill_color,
    text_color=COLOR_INFO,
    border_color=None,
    font_scale: float = 0.38,
    thickness: int = 1,
    max_width: int | None = None,
    fill_alpha: float = 0.9,
    draw_border: bool = True,
) -> int:
    if max_width is not None:
        usable_width = max(32, max_width - 14)
        text = fit_text_to_width(text, usable_width, font_scale, thickness)
    (text_width, text_height), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    width = text_width + 14
    height = text_height + baseline + 10
    if fill_alpha > 0:
        overlay = frame.copy()
        cv.rectangle(overlay, (left, top), (left + width, top + height), fill_color, cv.FILLED)
        cv.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, frame)
    if draw_border:
        cv.rectangle(frame, (left, top), (left + width, top + height), border_color or fill_color, 1)
    cv.putText(
        frame,
        text,
        (left + 7, top + height - baseline - 5),
        cv.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        thickness,
        cv.LINE_AA,
    )
    return width


def get_text_height(font_scale: float, thickness: int) -> int:
    return cv.getTextSize("Ag", cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][1]


def get_chip_height(font_scale: float, thickness: int) -> int:
    (_, text_height), baseline = cv.getTextSize("Ag", cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    return text_height + baseline + 10


def get_chip_width(text: str, font_scale: float, thickness: int, max_width: int | None = None) -> int:
    if max_width is not None:
        usable_width = max(32, max_width - 14)
        text = fit_text_to_width(text, usable_width, font_scale, thickness)
    (text_width, _), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    return text_width + 14


def measure_meta_row_height(items: list[tuple[str, str]], max_width: int) -> int:
    cursor_x = 0
    rows = 1
    for label, value in items:
        text = f"{label}: {value}"
        (text_width, _), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        chip_width = text_width + 14
        if cursor_x and cursor_x + chip_width > max_width:
            cursor_x = 0
            rows += 1
        cursor_x += chip_width + 6
    return (rows * META_CHIP_HEIGHT) + ((rows - 1) * META_WRAP_GAP)


def get_right_overlay_bounds(frame_width: int, min_left: int = 0) -> tuple[int, int]:
    safe_left = max(0, min_left)
    safe_right = max(safe_left, frame_width - FLOATING_OVERLAY_MARGIN)
    return safe_left, safe_right


def build_screen_controls_layout(frame_width: int, frame_height: int, min_left: int = 0) -> dict | None:
    footer_font_scale = 0.34
    footer_gap = 8
    chip_height = get_chip_height(footer_font_scale, 1)
    safe_left, safe_right = get_right_overlay_bounds(frame_width, min_left)
    available_width = safe_right - safe_left
    if available_width < 24:
        return None

    layout_options = [
        ("F Full", "Q Exit", False),
        ("F", "Q", False),
        ("F Full", "Q Exit", True),
        ("F", "Q", True),
    ]

    for full_label, exit_label, stacked in layout_options:
        full_width = get_chip_width(full_label, footer_font_scale, 1)
        exit_width = get_chip_width(exit_label, footer_font_scale, 1)
        if stacked:
            layout_width = max(full_width, exit_width)
            layout_height = (chip_height * 2) + footer_gap
            if layout_width > available_width:
                continue
            top = max(FLOATING_OVERLAY_MARGIN, frame_height - layout_height - FLOATING_OVERLAY_MARGIN)
            left = safe_right - layout_width
            return {
                "font_scale": footer_font_scale,
                "top": top,
                "stacked": True,
                "items": [
                    {"text": full_label, "left": left, "top": top},
                    {"text": exit_label, "left": safe_right - exit_width, "top": top + chip_height + footer_gap},
                ],
            }

        layout_width = full_width + footer_gap + exit_width
        if layout_width > available_width:
            continue
        top = max(FLOATING_OVERLAY_MARGIN, frame_height - chip_height - FLOATING_OVERLAY_MARGIN)
        exit_left = safe_right - exit_width
        full_left = exit_left - footer_gap - full_width
        return {
            "font_scale": footer_font_scale,
            "top": top,
            "stacked": False,
            "items": [
                {"text": full_label, "left": full_left, "top": top},
                {"text": exit_label, "left": exit_left, "top": top},
            ],
        }

    top = max(FLOATING_OVERLAY_MARGIN, frame_height - chip_height - FLOATING_OVERLAY_MARGIN)
    q_width = get_chip_width("Q", footer_font_scale, 1)
    if q_width <= available_width:
        return {
            "font_scale": footer_font_scale,
            "top": top,
            "stacked": False,
            "items": [
                {"text": "Q", "left": safe_right - q_width, "top": top},
            ],
        }

    return None


def draw_guidance_overlay(
    frame,
    feedback_lines: list[str],
    feedback_color,
    required_joints_status: dict | None = None,
    *,
    min_left: int = 0,
    max_bottom: int | None = None,
) -> int | None:
    top = FLOATING_OVERLAY_MARGIN
    safe_left, safe_right = get_right_overlay_bounds(frame.shape[1], min_left)
    available_width = safe_right - safe_left - FLOATING_OVERLAY_MARGIN
    if available_width < 120:
        return None

    panel_width = min(240, max(120, int(frame.shape[1] * 0.2), min(180, available_width)))
    panel_width = min(panel_width, available_width)
    left = safe_right - panel_width
    bottom_limit = max_bottom if max_bottom is not None else frame.shape[0] - FLOATING_OVERLAY_MARGIN
    available_height = bottom_limit - top
    if available_height < 52:
        return None

    content_left = left + PANEL_PADDING
    content_width = panel_width - (PANEL_PADDING * 2)
    section_gap = 18
    tips_title_scale = 0.3
    tips_body_scale = 0.3 if required_joints_status else 0.32
    tips_line_gap = 13 if required_joints_status else 14
    required_title_scale = 0.3
    required_row_scale = 0.3
    required_row_gap = 14
    required_summary_scale = 0.3

    wrapped_tip_lines: list[str] = []
    for line in feedback_lines:
        wrapped_tip_lines.extend(wrap_text_lines(compact_overlay_line(line), content_width - 16, tips_body_scale, 1))
    if not wrapped_tip_lines:
        wrapped_tip_lines = ["Pose looks good."]

    required_lines: list[dict] = []
    required_summary_text = None
    required_accent = feedback_color
    if required_joints_status and required_joints_status.get("joint_statuses"):
        visible_required = required_joints_status.get("visible_required", 0)
        required_total = max(1, len(required_joints_status["joint_statuses"]))
        required_summary_text = f"Visible {visible_required}/{required_total}"
        required_accent = COLOR_SUCCESS if required_joints_status.get("all_required_visible") else COLOR_WARNING
        for joint_status in required_joints_status["joint_statuses"]:
            required_lines.append(
                {
                    "label": fit_text_to_width(joint_status["label"], max(70, content_width - 54), required_row_scale, 1),
                    "detected": joint_status["detected"],
                }
            )

    required_header_height = 0
    required_summary_height = 0
    required_rows_height = 0
    required_top = top + 16
    if required_summary_text is not None:
        required_header_height = HEADER_BLOCK_HEIGHT
        required_summary_height = get_chip_height(required_summary_scale, 1) + REQUIRED_SUMMARY_BOTTOM_GAP
        required_rows_height = max(1, len(required_lines)) * required_row_gap

    tips_header_height = HEADER_BLOCK_HEIGHT
    tips_lines_height = max(1, len(wrapped_tip_lines)) * tips_line_gap
    panel_height = 16 + required_header_height + required_summary_height + required_rows_height
    if required_summary_text is not None:
        panel_height += section_gap
    panel_height += tips_header_height + tips_lines_height + 12
    panel_height = min(panel_height, available_height)

    draw_panel(
        frame,
        left,
        top,
        panel_width,
        panel_height,
        fill_color=COLOR_PANEL_ALT,
        alpha=CARD_ALPHA,
        accent_color=required_accent if required_summary_text is not None else feedback_color,
    )

    cursor_y = top + 16
    if required_summary_text is not None:
        header_baseline = draw_section_header(frame, "REQUIRED", "Joints", content_left, cursor_y + 6)
        draw_chip(
            frame,
            required_summary_text,
            content_left,
            header_baseline + 2,
            fill_color=COLOR_META_FILL_ALT,
            text_color=required_accent,
            border_color=required_accent,
            font_scale=required_summary_scale,
            max_width=content_width,
        )
        row_y = header_baseline + 2 + get_chip_height(required_summary_scale, 1) + REQUIRED_SUMMARY_BOTTOM_GAP
        max_required_bottom = top + panel_height - (tips_header_height + max(1, len(wrapped_tip_lines)) * tips_line_gap + 12 + section_gap)
        available_required_rows = max(1, (max_required_bottom - row_y) // required_row_gap)
        for joint_status in required_lines[:available_required_rows]:
            row_color = COLOR_SUCCESS if joint_status["detected"] else COLOR_ERROR
            status_text = "OK" if joint_status["detected"] else "MISS"
            cv.circle(frame, (content_left + 5, row_y - 4), 4, row_color, cv.FILLED, cv.LINE_AA)
            cv.putText(frame, joint_status["label"], (content_left + 16, row_y), cv.FONT_HERSHEY_SIMPLEX, required_row_scale, COLOR_INFO, 1, cv.LINE_AA)
            cv.putText(frame, status_text, (left + panel_width - PANEL_PADDING - 28, row_y), cv.FONT_HERSHEY_SIMPLEX, required_row_scale, row_color, 1, cv.LINE_AA)
            row_y += required_row_gap
        cursor_y = row_y + 4
        cursor_y = min(cursor_y + section_gap, top + panel_height - (tips_header_height + tips_line_gap + 12))

    tips_header_baseline = draw_section_header(frame, "TIPS", "Guide", content_left, cursor_y + 6)
    tip_y = tips_header_baseline
    max_tip_lines = max(1, (top + panel_height - 12 - tip_y) // tips_line_gap)
    for line in wrapped_tip_lines[:max_tip_lines]:
        cv.circle(frame, (content_left + 5, tip_y - 4), 3, feedback_color, cv.FILLED, cv.LINE_AA)
        cv.putText(
            frame,
            fit_text_to_width(line, content_width - 16, tips_body_scale, 1),
            (content_left + 16, tip_y),
            cv.FONT_HERSHEY_SIMPLEX,
            tips_body_scale,
            COLOR_INFO,
            1,
            cv.LINE_AA,
        )
        tip_y += tips_line_gap

    return top + panel_height


def draw_screen_controls(frame, min_left: int = 0) -> int | None:
    layout = build_screen_controls_layout(frame.shape[1], frame.shape[0], min_left)
    if layout is None:
        return None

    for item in layout["items"]:
        draw_chip(
            frame,
            item["text"],
            item["left"],
            item["top"],
            fill_color=COLOR_META_FILL,
            text_color=COLOR_MUTED,
            border_color=COLOR_PANEL_EDGE,
            font_scale=layout["font_scale"],
        )

    return layout["top"]


def draw_meta_row(frame, items: list[tuple[str, str]], left: int, top: int, max_width: int) -> int:
    cursor_x = left
    cursor_y = top
    row_height = META_ROW_HEIGHT
    chip_height = META_CHIP_HEIGHT
    for label, value in items:
        text = f"{label}: {value}"
        (text_width, _), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        chip_width = text_width + 14
        if cursor_x + chip_width > left + max_width:
            cursor_x = left
            cursor_y += row_height + META_WRAP_GAP
        overlay = frame.copy()
        cv.rectangle(overlay, (cursor_x, cursor_y), (cursor_x + chip_width, cursor_y + chip_height), COLOR_META_FILL, cv.FILLED)
        cv.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv.rectangle(frame, (cursor_x, cursor_y), (cursor_x + chip_width, cursor_y + chip_height), COLOR_PANEL_EDGE, 1)
        cv.putText(
            frame,
            text,
            (cursor_x + 7, cursor_y + chip_height - baseline - 4),
            cv.FONT_HERSHEY_SIMPLEX,
            0.38,
            COLOR_MUTED,
            1,
            cv.LINE_AA,
        )
        cursor_x += chip_width + 6
    return cursor_y + chip_height


def draw_progress_bar(frame, label: str, score: float, left: int, top: int, width: int, *, emphasized: bool = False) -> int:
    score = clamp_score(score)
    label_color = COLOR_INFO if emphasized else COLOR_MUTED
    score_color = score_to_color(score)
    bar_left = left
    bar_top = top + BAR_TOP_OFFSET
    percent_text = f"{int(round(score * 100)):02d}%"
    percent_width = cv.getTextSize(percent_text, cv.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0][0]
    bar_width = max(68, width - percent_width - 14)
    bar_height = 8
    fill_width = int((bar_width - 2) * score)

    cv.putText(frame, label.upper(), (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.34, label_color, 1, cv.LINE_AA)
    cv.putText(frame, percent_text, (left + width - percent_width, top), cv.FONT_HERSHEY_SIMPLEX, 0.36, score_color, 1, cv.LINE_AA)

    overlay = frame.copy()
    cv.rectangle(overlay, (bar_left, bar_top), (bar_left + bar_width, bar_top + bar_height), COLOR_BAR_BG, cv.FILLED)
    if fill_width > 0:
        cv.rectangle(overlay, (bar_left + 1, bar_top + 1), (bar_left + 1 + fill_width, bar_top + bar_height - 1), score_color, cv.FILLED)
    cv.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
    cv.rectangle(frame, (bar_left, bar_top), (bar_left + bar_width, bar_top + bar_height), COLOR_PANEL_EDGE, 1)
    return BAR_BLOCK_HEIGHT


def draw_alert_card(frame, title: str, detail_lines: list[str], meta_text: str, accent_color) -> None:
    card_width = min(frame.shape[1] - 72, ALERT_MAX_WIDTH)
    card_height = min(frame.shape[0] - 96, ALERT_MAX_HEIGHT)
    left = max((frame.shape[1] - card_width) // 2, 24)
    top = min(max((frame.shape[0] - card_height) // 2 + ALERT_TOP_SHIFT, 36), frame.shape[0] - card_height - 24)
    draw_panel(
        frame,
        left,
        top,
        card_width,
        card_height,
        fill_color=COLOR_PANEL,
        alpha=ALERT_ALPHA,
        border_color=accent_color,
        accent_color=accent_color,
    )
    content_left = left + 22
    cv.putText(frame, "ALERT", (content_left, top + 22), cv.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_LABEL, 1, cv.LINE_AA)
    cv.putText(frame, compact_overlay_status(title), (content_left, top + 44), cv.FONT_HERSHEY_SIMPLEX, ALERT_HEADER_FONT, COLOR_INFO, 2, cv.LINE_AA)
    baseline_y = top + 60
    for index, line in enumerate(detail_lines[:2]):
        cv.putText(
            frame,
            compact_overlay_line(line),
            (content_left, baseline_y + 6 + (index * ALERT_LINE_GAP)),
            cv.FONT_HERSHEY_SIMPLEX,
            ALERT_BODY_FONT,
            COLOR_INFO,
            2 if index == 0 else 1,
            cv.LINE_AA,
        )
    draw_chip(
        frame,
        meta_text,
        content_left,
        top + card_height - 44,
        fill_color=COLOR_META_FILL_ALT,
        text_color=COLOR_MUTED,
        border_color=accent_color,
        font_scale=ALERT_META_FONT,
    )


def draw_pose_skeleton(frame, points) -> None:
    for point in points:
        if point is not None:
            cv.ellipse(frame, point, (4, 4), 0, 0, 360, COLOR_JOINT, cv.FILLED)


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


def get_missing_joint_names(points, target_technique: str | None = None, max_items: int = 4) -> list[str]:
    priority_parts = TARGET_DIAGNOSTIC_PARTS.get(target_technique or "", tuple(BODY_PARTS.keys())[:-1])
    missing_labels = [BODY_PART_LABELS.get(part_name, part_name) for part_name in priority_parts if get_point(points, part_name) is None]
    if len(missing_labels) < max_items:
        for part_name in BODY_PARTS:
            if part_name == "Background" or part_name in priority_parts:
                continue
            if get_point(points, part_name) is None:
                missing_labels.append(BODY_PART_LABELS.get(part_name, part_name))
            if len(missing_labels) >= max_items:
                break
    return missing_labels[:max_items]


def format_missing_joint_message(missing_joint_names: list[str]) -> str:
    if not missing_joint_names:
        return ""
    return f"Missing joints: {', '.join(missing_joint_names)}"


def get_required_joint_names(target_technique: str | None = None) -> tuple[str, ...]:
    if target_technique in TARGET_DIAGNOSTIC_PARTS:
        return TARGET_DIAGNOSTIC_PARTS[target_technique]
    return tuple(part_name for part_name in BODY_PARTS if part_name != "Background")


def extract_required_joints_status(points, target_technique: str | None = None) -> dict:
    required_joints = get_required_joint_names(target_technique)
    joint_statuses = []
    visible_required = 0
    missing_required = []

    for part_name in required_joints:
        detected = get_point(points, part_name) is not None
        label = BODY_PART_LABELS.get(part_name, part_name)
        if detected:
            visible_required += 1
        else:
            missing_required.append(label)
        joint_statuses.append(
            {
                "name": part_name,
                "label": label,
                "detected": detected,
                "status_text": "OK" if detected else "Missing",
            }
        )

    required_count = max(1, len(required_joints))
    return {
        "technique": target_technique or "",
        "required_joints": list(required_joints),
        "joint_statuses": joint_statuses,
        "visible_required": visible_required,
        "missing_required": missing_required,
        "required_visible_ratio": visible_required / required_count,
        "all_required_visible": visible_required == len(required_joints),
    }


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


def extract_pose_features(points, frame_shape) -> dict:
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

    features = {
        "frame_width": frame_width,
        "frame_height": frame_height,
        "neck": neck,
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "left_elbow": left_elbow,
        "right_elbow": right_elbow,
        "left_wrist": left_wrist,
        "right_wrist": right_wrist,
        "left_hip": left_hip,
        "right_hip": right_hip,
        "left_ankle": left_ankle,
        "right_ankle": right_ankle,
        "upper_body_visible": upper_body_visible,
        "lower_body_visible": lower_body_visible,
        "upper_body_only_mode": upper_body_only_mode,
        "motion_snapshot": {"center_x": None, "hip_center_x": None, "body_offset": 0.0, "avg_wrist_height": None},
    }

    if neck is None or left_shoulder is None or right_shoulder is None or upper_body_visible < 4:
        features["invalid_reason"] = "missing_upper_body"
        return features

    shoulder_span = point_distance(left_shoulder, right_shoulder)
    if shoulder_span < 15:
        features["invalid_reason"] = "degenerate_pose"
        return features

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
                "straightness": 0.0,
                "centerline_score": 0.0,
                "guard_quality": 0.0,
                "protective": False,
                "strike_ready": False,
            }

        extension = abs(wrist[0] - shoulder[0]) / max(base_size, 1.0)
        reach = point_distance(wrist, shoulder) / max(base_size, 1.0)
        arm_path = point_distance(shoulder, elbow) + point_distance(elbow, wrist)
        straightness = point_distance(wrist, shoulder) / max(arm_path, 1.0)
        horizontal = abs(wrist[1] - shoulder[1]) <= 0.35 * base_size
        guard_width = clamp_score((0.75 * base_size - abs(wrist[0] - neck[0])) / max(0.75 * base_size, 1.0))
        guard_height = clamp_score(((neck[1] + 0.4 * base_size) - wrist[1]) / max(0.55 * base_size, 1.0))
        guard_quality = clamp_score((0.55 * guard_width) + (0.45 * guard_height))
        guard = wrist[1] <= neck[1] + 0.35 * base_size and abs(wrist[0] - neck[0]) <= 1.0 * base_size
        raised = wrist[1] <= shoulder[1] + 0.1 * base_size
        lowered = wrist[1] >= shoulder[1] + 0.35 * base_size
        bent = reach <= 1.35
        outward = (wrist[0] - shoulder[0]) * direction >= 0.45 * base_size
        near_head = wrist[1] <= neck[1] + 0.2 * base_size and abs(wrist[0] - neck[0]) <= 0.8 * base_size
        cross_guard = abs(wrist[0] - neck[0]) <= 0.45 * base_size and wrist[1] <= neck[1] + 0.45 * base_size
        centerline_score = clamp_score((0.8 * base_size - abs(wrist[0] - neck[0])) / max(0.8 * base_size, 1.0))
        protective = near_head or cross_guard or guard_quality >= 0.55
        strike_ready = reach >= 0.78 and straightness >= 0.82 and horizontal and centerline_score >= 0.42
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
            "straightness": straightness,
            "centerline_score": centerline_score,
            "guard_quality": guard_quality,
            "protective": protective,
            "strike_ready": strike_ready,
        }

    left_arm = arm_state("left")
    right_arm = arm_state("right")

    body_offset = abs(shoulder_center[0] - (frame_width / 2)) / max(frame_width, 1)
    ankles_visible = left_ankle is not None and right_ankle is not None
    ankle_span = point_distance(left_ankle, right_ankle) / max(base_size, 1.0) if ankles_visible else 0.0
    hips_visible = left_hip is not None and right_hip is not None
    shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1]) / max(base_size, 1.0)
    hip_shift = 0.0
    if hips_visible:
        hip_center_x = (left_hip[0] + right_hip[0]) / 2
        hip_shift = abs(hip_center_x - shoulder_center[0]) / max(base_size, 1.0)

    visible_wrist_heights = [wrist[1] / max(frame_height, 1) for wrist in (left_wrist, right_wrist) if wrist is not None]
    features.update(
        {
            "hip_center": hip_center,
            "shoulder_center": shoulder_center,
            "shoulder_span": shoulder_span,
            "base_size": base_size,
            "left_arm": left_arm,
            "right_arm": right_arm,
            "body_offset": body_offset,
            "ankle_span": ankle_span,
            "shoulder_tilt": shoulder_tilt,
            "hip_shift": hip_shift,
            "lowered_arms": float(left_arm["lowered"]) + float(right_arm["lowered"]),
            "guard_count": float(left_arm["guard"] or left_arm["cross_guard"]) + float(right_arm["guard"] or right_arm["cross_guard"]),
            "motion_snapshot": {
                "center_x": shoulder_center[0] / max(frame_width, 1),
                "hip_center_x": (hip_center[0] / max(frame_width, 1)) if hip_center is not None else None,
                "body_offset": body_offset,
                "avg_wrist_height": (sum(visible_wrist_heights) / len(visible_wrist_heights)) if visible_wrist_heights else None,
                "base_size": base_size / max(frame_height, 1),
            },
        }
    )
    return features


def compute_motion_metrics(current_snapshot: dict, motion_history: deque | None) -> dict:
    snapshots = list(motion_history or []) + [current_snapshot]

    def metric_values(key: str) -> list[float]:
        return [float(snapshot[key]) for snapshot in snapshots if snapshot.get(key) is not None]

    center_values = metric_values("center_x")
    hip_values = metric_values("hip_center_x")
    offset_values = metric_values("body_offset")
    wrist_values = metric_values("avg_wrist_height")
    size_values = metric_values("base_size")

    center_delta = (center_values[-1] - center_values[0]) if len(center_values) > 1 else 0.0
    hip_delta = (hip_values[-1] - hip_values[0]) if len(hip_values) > 1 else 0.0
    size_delta = (size_values[-1] - size_values[0]) if len(size_values) > 1 else 0.0
    center_range = (max(center_values) - min(center_values)) if len(center_values) > 1 else 0.0
    hip_range = (max(hip_values) - min(hip_values)) if len(hip_values) > 1 else 0.0
    offset_range = (max(offset_values) - min(offset_values)) if len(offset_values) > 1 else 0.0

    return {
        "center_range": center_range,
        "hip_range": hip_range,
        "offset_range": offset_range,
        "wrist_drop": max(0.0, wrist_values[-1] - min(wrist_values)) if wrist_values else 0.0,
        "center_delta": center_delta,
        "hip_delta": hip_delta,
        "lateral_shift": abs(center_delta),
        "hip_lateral_shift": abs(hip_delta),
        "direction_consistency": clamp_score(abs(center_delta) / max(center_range, 0.001)) if center_range else 0.0,
        "hip_direction_consistency": clamp_score(abs(hip_delta) / max(hip_range, 0.001)) if hip_range else 0.0,
        "retreat_shift": clamp_score(max(0.0, -size_delta) / max(size_values[0], 0.001)) if len(size_values) > 1 else 0.0,
    }


def classify_technique(features: dict, motion_history: deque | None = None, target_technique: str | None = None) -> dict:
    if features.get("invalid_reason"):
        return {
            "label": "Unknown",
            "confidence": 0.0,
            "scores": {"Punch": 0.0, "Block": 0.0, "Escape": 0.0},
            "reason": features["invalid_reason"],
        }

    left_arm = features["left_arm"]
    right_arm = features["right_arm"]
    upper_body_only_mode = features["upper_body_only_mode"]
    motion_metrics = compute_motion_metrics(features["motion_snapshot"], motion_history)

    def punch_arm_score(arm: dict, opposite_arm: dict, wrist_key: str, shoulder_key: str) -> float:
        if not arm["visible"]:
            return 0.0

        level_delta = abs(features[wrist_key][1] - features[shoulder_key][1]) / max(features["base_size"], 1.0)
        level_score = clamp_score((0.42 - min(level_delta, 0.42)) / 0.42)
        guide_extension = float(
            arm["reach"] >= 0.68
            and arm["straightness"] >= 0.76
            and arm["horizontal"]
            and arm["centerline_score"] >= 0.34
        )
        score = (
            0.2 * clamp_score((arm["reach"] - 0.66) / 0.72)
            + 0.16 * clamp_score((arm["extension"] - 0.18) / 0.6)
            + 0.16 * clamp_score((arm["straightness"] - 0.74) / 0.22)
            + 0.13 * arm["centerline_score"]
            + 0.12 * level_score
            + 0.14 * opposite_arm["guard_quality"]
            + 0.09 * guide_extension
        )
        if arm["strike_ready"]:
            score += 0.08
        if opposite_arm["protective"]:
            score += 0.06
        if opposite_arm["strike_ready"]:
            score -= 0.14
        elif opposite_arm["guard_quality"] < 0.4:
            score -= 0.08
        return clamp_score(score)

    left_punch = punch_arm_score(left_arm, right_arm, "left_wrist", "left_shoulder")
    right_punch = punch_arm_score(right_arm, left_arm, "right_wrist", "right_shoulder")
    left_guide_punch = left_arm["reach"] >= 0.68 and left_arm["horizontal"] and left_arm["centerline_score"] >= 0.34
    right_guide_punch = right_arm["reach"] >= 0.68 and right_arm["horizontal"] and right_arm["centerline_score"] >= 0.34
    left_slow_punch = left_arm["reach"] >= 0.6 and left_arm["horizontal"] and left_arm["centerline_score"] >= 0.26
    right_slow_punch = right_arm["reach"] >= 0.6 and right_arm["horizontal"] and right_arm["centerline_score"] >= 0.26
    single_guarded_punch = (
        (left_guide_punch and not right_guide_punch and right_arm["protective"])
        or (right_guide_punch and not left_guide_punch and left_arm["protective"])
    )
    slow_guarded_punch = (
        (left_slow_punch and not right_slow_punch and right_arm["protective"])
        or (right_slow_punch and not left_slow_punch and left_arm["protective"])
    )
    if upper_body_only_mode:
        left_punch = clamp_score(left_punch + 0.05 * float(left_arm["strike_ready"]))
        right_punch = clamp_score(right_punch + 0.05 * float(right_arm["strike_ready"]))
    punch_score = max(left_punch, right_punch)
    if single_guarded_punch:
        punch_score = clamp_score(punch_score + 0.14)
    elif slow_guarded_punch:
        punch_score = clamp_score(punch_score + 0.1)
    if (left_arm["strike_ready"] ^ right_arm["strike_ready"]) and (left_arm["protective"] or right_arm["protective"]):
        punch_score = clamp_score(punch_score + 0.1)
    if left_arm["strike_ready"] and right_arm["strike_ready"]:
        punch_score = clamp_score(punch_score - 0.16)
    elif left_guide_punch and right_guide_punch:
        punch_score = clamp_score(punch_score - 0.08)
    if not (left_arm["protective"] or right_arm["protective"]):
        punch_score = clamp_score(punch_score - 0.12)
    if left_arm["raised"] and right_arm["raised"]:
        punch_score = clamp_score(punch_score - 0.03)

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

    block_head_coverage = float(left_arm["near_head"] or left_arm["cross_guard"]) + float(right_arm["near_head"] or right_arm["cross_guard"])
    block_protective_coverage = float(left_arm["protective"]) + float(right_arm["protective"])
    block_bent_coverage = float(left_arm["bent"]) + float(right_arm["bent"])
    block_raised_coverage = float(left_arm["raised"]) + float(right_arm["raised"])
    block_guard_pose = bool(
        block_raised_coverage >= 2
        and block_bent_coverage >= 2
        and block_protective_coverage >= 2
        and block_head_coverage >= 1
    )

    if left_arm["visible"] and right_arm["visible"]:
        block_score = 0.5 * left_block + 0.5 * right_block
        wrist_symmetry = abs(features["left_wrist"][1] - features["right_wrist"][1]) <= 0.35 * features["base_size"]
        if wrist_symmetry:
            block_score += 0.1
        if left_arm["near_head"] and right_arm["near_head"]:
            block_score += 0.1
        if block_protective_coverage >= 2:
            block_score += 0.08
        if (left_arm["cross_guard"] or left_arm["near_head"]) and (right_arm["cross_guard"] or right_arm["near_head"]):
            block_score += 0.08
    else:
        block_score = max(left_block, right_block)
        if upper_body_only_mode:
            block_score += 0.1
    if left_arm["outward"] or right_arm["outward"]:
        block_score -= 0.06 * float(left_arm["outward"] or right_arm["outward"])
    if left_arm["strike_ready"] or right_arm["strike_ready"]:
        if block_guard_pose:
            block_score -= 0.04 if (left_arm["strike_ready"] ^ right_arm["strike_ready"]) else 0.02
        else:
            block_score -= 0.12 if (left_arm["strike_ready"] ^ right_arm["strike_ready"]) else 0.08
    if single_guarded_punch and not block_guard_pose:
        block_score = clamp_score(block_score - 0.06)
    block_score = clamp_score(block_score)

    protective_arms = float(left_arm["protective"]) + float(right_arm["protective"])
    release_score = clamp_score(max(features["lowered_arms"] / 2.0, float(left_arm["cross_guard"] or right_arm["cross_guard"])))
    movement_score = max(
        clamp_score((motion_metrics["lateral_shift"] - 0.035) / 0.16),
        clamp_score((motion_metrics["retreat_shift"] - 0.03) / 0.14),
    )
    distance_score = max(
        clamp_score((features["body_offset"] - 0.05) / 0.18),
        clamp_score((motion_metrics["offset_range"] - 0.03) / 0.18),
    )
    direction_score = max(
        motion_metrics["direction_consistency"],
        motion_metrics["hip_direction_consistency"],
    )
    hip_movement_score = max(
        clamp_score((motion_metrics["hip_lateral_shift"] - 0.02) / 0.12),
        clamp_score((motion_metrics["hip_range"] - 0.02) / 0.14),
    )
    balance_score = clamp_score((0.18 - min(features["shoulder_tilt"], 0.18)) / 0.18)
    protective_score = clamp_score(max(protective_arms / 2.0, features["guard_count"] / 2.0))
    off_center_score = max(
        clamp_score((features["body_offset"] - 0.04) / 0.16),
        clamp_score((features["hip_shift"] - 0.04) / 0.22),
        distance_score,
    )
    guided_escape_pose = bool(protective_score >= 0.28 and (off_center_score >= 0.32 or movement_score >= 0.24 or hip_movement_score >= 0.2))
    upper_body_escape_pose = bool(
        upper_body_only_mode
        and protective_score >= 0.3
        and max(off_center_score, movement_score, distance_score) >= 0.26
    )
    escape_target_pose = bool(
        guided_escape_pose
        or upper_body_escape_pose
        or (protective_score >= 0.3 and max(off_center_score, movement_score, distance_score, hip_movement_score) >= 0.24)
    )

    escape_score = (
        0.22 * movement_score
        + 0.2 * distance_score
        + 0.14 * direction_score
        + 0.12 * hip_movement_score
        + 0.18 * protective_score
        + 0.06 * release_score
        + 0.05 * balance_score
        + 0.07 * off_center_score
        + 0.03 * clamp_score((features["ankle_span"] - 0.8) / 0.8)
    )
    if movement_score < 0.2:
        escape_score = clamp_score(escape_score - 0.06)
    if protective_score < 0.28 and release_score < 0.3:
        escape_score = clamp_score(escape_score - 0.06)
    if upper_body_only_mode:
        escape_score = clamp_score(
            escape_score
            + 0.08 * movement_score
            + 0.08 * distance_score
            + 0.06 * protective_score
        )
    if guided_escape_pose:
        escape_score = clamp_score(escape_score + 0.12)
    if upper_body_escape_pose:
        escape_score = clamp_score(escape_score + 0.14)
    escape_score = clamp_score(escape_score)

    if target_technique in TECHNIQUE_PROFILES:
        target_bonus = 0.03
        if target_technique == "Punch" and (single_guarded_punch or slow_guarded_punch or ((left_arm["strike_ready"] ^ right_arm["strike_ready"]) and (left_arm["protective"] or right_arm["protective"]))):
            punch_score = clamp_score(punch_score + target_bonus)
        elif target_technique == "Block":
            if block_guard_pose:
                block_score = clamp_score(block_score + 0.05)
            elif block_raised_coverage >= 2 and block_protective_coverage >= 1:
                block_score = clamp_score(block_score + target_bonus)
        elif target_technique == "Escape":
            if escape_target_pose:
                escape_score = clamp_score(escape_score + 0.05)
            elif protective_score >= 0.28 and max(off_center_score, movement_score, distance_score) >= 0.18:
                escape_score = clamp_score(escape_score + target_bonus)

    scores = {
        "Punch": round(punch_score, 3),
        "Block": round(block_score, 3),
        "Escape": round(escape_score, 3),
    }

    if target_technique in TECHNIQUE_PROFILES:
        target_score = scores[target_technique]
        target_profile = TECHNIQUE_PROFILES[target_technique]
        competing_score = max(score for label, score in scores.items() if label != target_technique)
        if target_technique == "Block":
            block_priority_score = max(target_profile["min_confidence"], target_profile["match_threshold"] - 0.06)
            if block_guard_pose and target_score >= block_priority_score and (competing_score - target_score) <= 0.1:
                return {"label": target_technique, "confidence": target_score, "scores": scores, "reason": "target_priority"}
        elif target_technique == "Escape":
            escape_priority_score = max(target_profile["min_confidence"], target_profile["match_threshold"] - 0.06)
            if escape_target_pose and target_score >= escape_priority_score and (competing_score - target_score) <= 0.24:
                return {"label": target_technique, "confidence": target_score, "scores": scores, "reason": "target_priority"}
        elif target_score >= target_profile["match_threshold"] and (competing_score - target_score) <= 0.14:
            return {"label": target_technique, "confidence": target_score, "scores": scores, "reason": "target_priority"}

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]
    second_best = max(score for label, score in scores.items() if label != best_label)
    best_profile = TECHNIQUE_PROFILES[best_label]
    if best_score < best_profile["min_confidence"]:
        return {"label": "Unknown", "confidence": best_score, "scores": scores, "reason": "low_confidence"}
    if (best_score - second_best) < 0.035 and best_score < best_profile["match_threshold"]:
        return {"label": "Unknown", "confidence": best_score, "scores": scores, "reason": "ambiguous_pose"}
    return {"label": best_label, "confidence": best_score, "scores": scores, "reason": "classified"}


def smooth_technique_result(history: deque, target_technique: str | None = None, confirm_frames: int | None = None) -> dict:
    if not history:
        return {"label": "Unknown", "confidence": 0.0, "scores": {name: 0.0 for name in TECHNIQUE_NAMES}}

    weights = list(range(1, len(history) + 1))
    total_weight = sum(weights)
    weighted_scores = {name: 0.0 for name in TECHNIQUE_NAMES}
    recent_scores = {name: [] for name in TECHNIQUE_NAMES}

    for weight, result in zip(weights, history):
        for name in TECHNIQUE_NAMES:
            score = result.get("scores", {}).get(name, 0.0)
            weighted_scores[name] += score * weight
            recent_scores[name].append(score)

    averaged_scores = {name: round(weighted_scores[name] / max(total_weight, 1), 3) for name in TECHNIQUE_NAMES}
    window = max(2, min(confirm_frames or len(history), len(history)))

    if target_technique in TECHNIQUE_PROFILES:
        target_profile = TECHNIQUE_PROFILES[target_technique]
        target_recent_scores = recent_scores[target_technique][-window:]
        target_hits = sum(score >= target_profile["tracking_threshold"] for score in target_recent_scores)
        target_peak = max(target_recent_scores, default=0.0)
        target_average = averaged_scores[target_technique]
        required_hits = max(2, window - 1)
        required_confidence = target_profile["min_confidence"]
        if target_technique == "Punch":
            required_hits = max(1, window - 2)
            required_confidence = max(target_profile["tracking_threshold"], target_profile["min_confidence"] - 0.03)
        elif target_technique == "Block":
            required_hits = max(1, window - 2)
            required_confidence = max(target_profile["tracking_threshold"], target_profile["min_confidence"] - 0.04)
        elif target_technique == "Escape":
            required_hits = max(1, window - 2)
            required_confidence = max(target_profile["tracking_threshold"], target_profile["min_confidence"] - 0.04)
        if target_hits >= required_hits and max(target_average, target_peak) >= required_confidence:
            return {
                "label": target_technique,
                "confidence": round(max(target_average, target_peak), 3),
                "scores": averaged_scores,
                "reason": "stable_target",
            }

    best_label = max(TECHNIQUE_NAMES, key=averaged_scores.get)
    best_score = averaged_scores[best_label]
    second_best = max(score for label, score in averaged_scores.items() if label != best_label)
    best_profile = TECHNIQUE_PROFILES[best_label]
    recent_best_hits = sum(score >= best_profile["tracking_threshold"] for score in recent_scores[best_label][-window:])

    if best_score < best_profile["min_confidence"] or recent_best_hits < max(2, window - 1):
        return {"label": "Unknown", "confidence": best_score, "scores": averaged_scores, "reason": "not_stable"}
    if (best_score - second_best) < 0.03 and best_score < best_profile["match_threshold"]:
        return {"label": "Unknown", "confidence": best_score, "scores": averaged_scores, "reason": "ambiguous_stable"}

    return {"label": best_label, "confidence": best_score, "scores": averaged_scores, "reason": "stable"}


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
    missing_joint_names = get_missing_joint_names(points, target_technique)
    upper_body_visible = count_visible_points(points, ("Nose", "Neck", "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist"))
    lower_body_visible = count_visible_points(points, ("LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle"))
    shoulder_span = point_distance(left_shoulder, right_shoulder) if left_shoulder is not None and right_shoulder is not None else 0.0
    base_size = max(shoulder_span, 1.0)

    def build_arm_state(shoulder, elbow, wrist, direction: int) -> dict:
        if neck is None or shoulder is None or elbow is None or wrist is None:
            return {
                "visible": False,
                "reach": 0.0,
                "horizontal": False,
                "guard": False,
                "raised": False,
                "lowered": False,
                "bent": False,
                "outward": False,
                "near_head": False,
                "cross_guard": False,
                "centerline_score": 0.0,
                "guard_quality": 0.0,
                "protective": False,
                "strike_ready": False,
            }

        reach = point_distance(wrist, shoulder) / base_size
        horizontal = abs(wrist[1] - shoulder[1]) <= 0.35 * base_size
        guard = wrist[1] <= neck[1] + 0.35 * base_size and abs(wrist[0] - neck[0]) <= 1.0 * base_size
        raised = wrist[1] <= shoulder[1] + 0.1 * base_size
        lowered = wrist[1] >= shoulder[1] + 0.35 * base_size
        bent = reach <= 1.35
        outward = (wrist[0] - shoulder[0]) * direction >= 0.45 * base_size
        near_head = wrist[1] <= neck[1] + 0.2 * base_size and abs(wrist[0] - neck[0]) <= 0.8 * base_size
        cross_guard = abs(wrist[0] - neck[0]) <= 0.45 * base_size and wrist[1] <= neck[1] + 0.45 * base_size
        centerline_score = clamp_score((0.8 * base_size - abs(wrist[0] - neck[0])) / max(0.8 * base_size, 1.0))
        guard_width = clamp_score((0.75 * base_size - abs(wrist[0] - neck[0])) / max(0.75 * base_size, 1.0))
        guard_height = clamp_score(((neck[1] + 0.4 * base_size) - wrist[1]) / max(0.55 * base_size, 1.0))
        guard_quality = clamp_score((0.55 * guard_width) + (0.45 * guard_height))
        protective = near_head or cross_guard or guard_quality >= 0.55
        arm_path = point_distance(shoulder, elbow) + point_distance(elbow, wrist)
        straightness = point_distance(wrist, shoulder) / max(arm_path, 1.0)
        strike_ready = reach >= 0.78 and straightness >= 0.82 and horizontal and centerline_score >= 0.42

        return {
            "visible": True,
            "reach": reach,
            "horizontal": horizontal,
            "guard": guard,
            "raised": raised,
            "lowered": lowered,
            "bent": bent,
            "outward": outward,
            "near_head": near_head,
            "cross_guard": cross_guard,
            "centerline_score": centerline_score,
            "guard_quality": guard_quality,
            "protective": protective,
            "strike_ready": strike_ready,
        }

    left_arm = build_arm_state(left_shoulder, left_elbow, left_wrist, -1)
    right_arm = build_arm_state(right_shoulder, right_elbow, right_wrist, 1)
    punch_ready_left = left_arm["reach"] >= 0.64 and left_arm["horizontal"] and left_arm["centerline_score"] >= 0.3
    punch_ready_right = right_arm["reach"] >= 0.64 and right_arm["horizontal"] and right_arm["centerline_score"] >= 0.3
    single_guarded_punch = (
        (punch_ready_left and not punch_ready_right and right_arm["protective"])
        or (punch_ready_right and not punch_ready_left and left_arm["protective"])
    )
    off_center_now = abs(neck[0] - (frame_width // 2)) >= tolerance_x if neck is not None else False
    upper_body_escape_ready = (left_arm["protective"] or right_arm["protective"]) and off_center_now

    if frame_visibility["is_dark"] and frame_visibility["is_low_detail"]:
        return (
            "Camera image too dark",
            [format_missing_joint_message(missing_joint_names) or "Check lighting, camera shutter, or try --camera-index 1"],
            COLOR_ERROR,
        )

    if detected_parts < 4 or detection_rate < 0.22 or neck is None:
        diagnostic_lines = []
        missing_message = format_missing_joint_message(missing_joint_names)
        if missing_message:
            diagnostic_lines.append(missing_message)
        diagnostic_lines.append("Move back and show your shoulders and arms")
        return "Pose not clear", diagnostic_lines[:3], COLOR_ERROR

    if target_technique == "Punch":
        if upper_body_visible < 6:
            feedback_lines.append("Guide: keep shoulders, elbows, and wrists visible")
        if not (left_arm["protective"] or right_arm["protective"]):
            feedback_lines.append("Guide: keep one hand guarding your face")
        if not (punch_ready_left or punch_ready_right):
            feedback_lines.append("Guide: extend one hand forward at shoulder height")
        if not (left_arm["centerline_score"] >= 0.3 or right_arm["centerline_score"] >= 0.3):
            feedback_lines.append("Guide: aim the punch through the center line")
        if (punch_ready_left or punch_ready_right) and not single_guarded_punch:
            feedback_lines.append("Guide: hold the punch briefly before returning")
        if not (left_arm["horizontal"] or right_arm["horizontal"]):
            feedback_lines.append("Keep the punching arm closer to shoulder height")
        if not (left_arm["protective"] or right_arm["protective"]):
            feedback_lines.append("Keep the other hand near your face as a guard")
        missing_message = format_missing_joint_message(missing_joint_names)
        if upper_body_visible < 7 and missing_message:
            feedback_lines.append(missing_message)

        if not feedback_lines:
            return "Punch form correct", ["Hold the punch briefly and keep your guard up"], COLOR_SUCCESS
        return "Adjust punch", feedback_lines[:3], COLOR_WARNING

    if target_technique == "Block":
        if upper_body_visible < 6:
            feedback_lines.append("Keep both shoulders, elbows, and wrists visible")
        if not (left_arm["raised"] and right_arm["raised"]):
            feedback_lines.append("Raise both arms higher to protect your head")
        if not (left_arm["bent"] and right_arm["bent"]):
            feedback_lines.append("Keep both arms bent for a tighter block")
        if not ((left_arm["near_head"] or left_arm["cross_guard"]) and (right_arm["near_head"] or right_arm["cross_guard"])):
            feedback_lines.append("Keep both hands closer to your head and center line")
        if (left_arm["strike_ready"] ^ right_arm["strike_ready"]) and not ((left_arm["near_head"] or left_arm["cross_guard"]) and (right_arm["near_head"] or right_arm["cross_guard"])):
            feedback_lines.append("Do not extend one arm forward like a punch")

        if not feedback_lines:
            return "Block form correct", ["Hold your guard position"], COLOR_SUCCESS
        return "Adjust block", feedback_lines[:3], COLOR_WARNING

    if target_technique == "Escape":
        if not (left_arm["protective"] or right_arm["protective"]):
            feedback_lines.append("Guide: keep one hand protecting your head")
        if not upper_body_escape_ready:
            feedback_lines.append("Guide: move backward or sideways off the center line")
        if lower_body_visible <= 1 and upper_body_escape_ready:
            feedback_lines.append("Guide: upper-body escape is visible; step back for a stronger match")
        elif lower_body_visible <= 1:
            feedback_lines.append("Guide: keep moving to create distance")
        if not off_center_now:
            feedback_lines.append("Move farther backward or sideways off the center line")
        missing_message = format_missing_joint_message(missing_joint_names)
        if missing_message:
            feedback_lines.append(missing_message)

        if not feedback_lines or (upper_body_escape_ready and lower_body_visible <= 1 and len(feedback_lines) == 1):
            return "Escape form correct", ["Keep moving away while protecting your head"], COLOR_SUCCESS
        return "Adjust escape", feedback_lines[:3], COLOR_WARNING

    if upper_body_visible < 6:
        feedback_lines.append("Keep both shoulders, elbows, and wrists in frame")

    if nose is None:
        feedback_lines.append("Lift camera or keep your head visible")

    if lower_body_visible <= 1 and upper_body_visible >= 6:
        feedback_lines.append("Upper body framing is OK for punch and block")
    elif (left_hip is None or right_hip is None) and (left_ankle is None or right_ankle is None):
        feedback_lines.append("Step back if you want better escape detection")
    missing_message = format_missing_joint_message(missing_joint_names)
    if missing_message:
        feedback_lines.append(missing_message)

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


def draw_technique_info(
    frame,
    technique_result: dict,
    panel_left: int,
    panel_top: int,
    panel_width: int,
    panel_height: int,
    target_technique: str | None = None,
    preset: str = DEFAULT_PRESET_NAME,
) -> int:
    label = technique_result.get("label", "Unknown")
    scores = technique_result.get("scores", {})
    confidence = technique_result.get("confidence", 0.0) * 100

    content_left = panel_left + PANEL_PADDING
    content_width = panel_width - (PANEL_PADDING * 2)
    cursor_y = draw_section_header(frame, "MATCH", "Technique", content_left, panel_top + 22)

    if target_technique:
        selected_score = scores.get(target_technique, 0.0)
        headline = f"Target: {target_technique}"
    elif label == "Unknown":
        headline = "Detecting..."
    else:
        headline = f"Detected: {label}"

    headline = fit_text_to_width(headline, content_width, 0.4, 1)
    cv.putText(frame, headline, (content_left, cursor_y), cv.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_INFO, 1, cv.LINE_AA)
    headline_height = get_text_height(0.4, 1)
    cursor_y += headline_height + MATCH_STACK_GAP

    secondary_label = "Camera live"
    secondary_top = cursor_y
    draw_chip(
        frame,
        secondary_label,
        content_left,
        secondary_top,
        fill_color=COLOR_META_FILL,
        text_color=COLOR_MUTED,
        border_color=COLOR_PANEL_EDGE,
        font_scale=0.32,
        max_width=content_width,
    )
    cursor_y = secondary_top + get_chip_height(0.32, 1) + MATCH_BAR_START_GAP

    score_items = [(target_technique, scores.get(target_technique, 0.0))] if target_technique else [(name, scores.get(name, 0.0)) for name in TECHNIQUE_NAMES]
    max_bars = 1 if target_technique else 3
    for index, (name, score) in enumerate(score_items[:max_bars]):
        draw_progress_bar(
            frame,
            name,
            score,
            content_left,
            cursor_y,
            content_width,
            emphasized=bool(target_technique and name == target_technique),
        )
        cursor_y += BAR_BLOCK_HEIGHT + MATCH_STACK_GAP

    if target_technique:
        match_state = evaluate_target_match(technique_result, target_technique, preset)
        if match_state["full_match"]:
            result_text = "Match"
            follow_up = "Hold steady."
            result_color = COLOR_SUCCESS
        elif match_state["near_match"]:
            result_text = "Near Match"
            follow_up = "Close enough. Hold steady."
            result_color = COLOR_LABEL
        else:
            result_text = "Adjusting"
            follow_up = "Keep adjusting."
            result_color = COLOR_WARNING
        status_chip_height = get_chip_height(0.34, 1)
        follow_up_height = MATCH_FOLLOWUP_TOP_OFFSET + get_text_height(0.34, 1)
        status_block_height = status_chip_height + follow_up_height
        min_match_height = (cursor_y - panel_top) + MATCH_STATUS_GAP + status_block_height + MATCH_CARD_BOTTOM_PADDING
        panel_height = max(panel_height, min_match_height)
        preferred_status_top = panel_top + MATCH_STATUS_TOP_OFFSET
        content_status_top = cursor_y + MATCH_STATUS_GAP
        status_top = max(content_status_top, preferred_status_top)
        status_top = min(status_top, panel_top + panel_height - status_block_height - MATCH_CARD_BOTTOM_PADDING)
        draw_chip(
            frame,
            result_text,
            content_left,
            status_top,
            fill_color=COLOR_META_FILL_ALT,
            text_color=result_color,
            border_color=result_color,
            font_scale=0.34,
            max_width=content_width,
        )
        line_space = max(0, (panel_top + panel_height - MATCH_CARD_BOTTOM_PADDING) - (status_top + status_chip_height + MATCH_FOLLOWUP_TOP_OFFSET))
        max_lines = max(0, min(1, line_space // MATCH_FOLLOWUP_GAP))
        if max_lines:
            follow_up_lines = wrap_text_lines_limited(follow_up, content_width, 0.34, 1, max_lines)
            for line_index, follow_line in enumerate(follow_up_lines, start=1):
                cv.putText(frame, follow_line, (content_left, status_top + status_chip_height + MATCH_FOLLOWUP_TOP_OFFSET + ((line_index - 1) * MATCH_FOLLOWUP_GAP)), cv.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_MUTED, 1, cv.LINE_AA)
        return status_top + status_block_height + MATCH_CARD_BOTTOM_PADDING
    else:
        footer_text = fit_text_to_width("Punch / Block / Escape", content_width, 0.32, 1)
        cv.putText(frame, footer_text, (content_left, panel_top + panel_height - 16), cv.FONT_HERSHEY_SIMPLEX, 0.32, COLOR_MUTED, 1, cv.LINE_AA)
        return panel_top + panel_height


def draw_required_joints_panel(frame, required_joints_status: dict, left: int, top: int, width: int, height: int) -> None:
    if not required_joints_status.get("joint_statuses"):
        return

    joint_statuses = required_joints_status["joint_statuses"]
    visible_required = required_joints_status.get("visible_required", 0)
    required_total = max(1, len(joint_statuses))
    panel_height = max(height, REQUIRED_PANEL_MIN_HEIGHT)

    content_left = left + PANEL_PADDING
    content_width = width - (PANEL_PADDING * 2)
    cursor_y = draw_section_header(frame, "REQUIRED", "Joints", content_left, top + 22)
    summary_text = f"Visible {visible_required}/{required_total}"
    draw_chip(
        frame,
        summary_text,
        content_left,
        cursor_y + 2,
        fill_color=COLOR_META_FILL_ALT,
        text_color=COLOR_SUCCESS if required_joints_status.get("all_required_visible") else COLOR_WARNING,
        border_color=COLOR_SUCCESS if required_joints_status.get("all_required_visible") else COLOR_WARNING,
        font_scale=0.32,
        max_width=content_width,
    )

    summary_bottom = cursor_y + 2 + get_chip_height(0.32, 1)
    row_y = summary_bottom + REQUIRED_SUMMARY_BOTTOM_GAP
    max_rows = min(len(joint_statuses), max(1, (panel_height - 52) // REQUIRED_PANEL_ROW_GAP))
    for joint_status in joint_statuses[:max_rows]:
        detected = joint_status["detected"]
        row_color = COLOR_SUCCESS if detected else COLOR_ERROR
        status_text = "OK" if detected else "MISS"
        label = fit_text_to_width(joint_status["label"], max(70, content_width - 54), 0.34, 1)
        cv.circle(frame, (content_left + 5, row_y - 5), 5, row_color, cv.FILLED, cv.LINE_AA)
        cv.putText(frame, label, (content_left + 18, row_y), cv.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_INFO, 1, cv.LINE_AA)
        cv.putText(frame, status_text, (left + width - PANEL_PADDING - 30, row_y), cv.FONT_HERSHEY_SIMPLEX, 0.34, row_color, 1, cv.LINE_AA)
        row_y += REQUIRED_PANEL_ROW_GAP


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
    missing_joint_names: list[str],
    required_joints_status: dict | None,
    target_technique: str | None = None,
    preset: str = DEFAULT_PRESET_NAME,
):
    annotated = frame.copy()
    show_required_panel = bool(target_technique and required_joints_status)
    layout = get_overlay_layout(annotated.shape[1], annotated.shape[0], False)
    draw_pose_skeleton(annotated, points)

    panel_left = layout["card_left"]
    panel_top = layout["feedback_top"]
    panel_width = layout["card_width"]
    base_card_height = layout["card_height"]
    panel_height = base_card_height

    content_left = panel_left + PANEL_PADDING
    content_width = panel_width - (PANEL_PADDING * 2)
    cursor_y = draw_section_header(annotated, "FEEDBACK", "Form", content_left, panel_top + 22)
    right_overlay_min_left = panel_left + panel_width + OVERLAY_GAP

    meta_items = [("Inference", f"{inference_ms:.0f} ms"), ("Visible", f"{detected_parts}/{len(BODY_PARTS)}")]
    status_chip_height = get_chip_height(0.36, 1)
    min_feedback_height = FEEDBACK_STATUS_TOP_OFFSET + status_chip_height + FOOTER_BOTTOM_MARGIN + FEEDBACK_CARD_BOTTOM_PADDING
    if panel_height < min_feedback_height:
        panel_height = min_feedback_height
        draw_panel(
            annotated,
            panel_left,
            panel_top,
            panel_width,
            panel_height,
            fill_color=COLOR_PANEL,
            alpha=0.0,
            border_color=COLOR_PANEL_EDGE,
            accent_color=None,
            draw_border=False,
        )
        cursor_y = draw_section_header(annotated, "FEEDBACK", "Form", content_left, panel_top + 22)
    meta_top = cursor_y + META_WRAP_GAP
    preferred_status_top = panel_top + FEEDBACK_STATUS_TOP_OFFSET
    available_meta_height = max(0, preferred_status_top - FEEDBACK_STATUS_GAP - meta_top)
    while len(meta_items) > 1 and measure_meta_row_height(meta_items, content_width) > available_meta_height:
        meta_items.pop()
    meta_bottom = draw_meta_row(annotated, meta_items, content_left, meta_top, content_width)

    status_top = max(meta_bottom + FEEDBACK_STATUS_GAP, preferred_status_top)
    draw_chip(
        annotated,
        compact_overlay_status(feedback_status),
        content_left,
        status_top,
        fill_color=COLOR_META_FILL_ALT,
        text_color=feedback_color,
        border_color=feedback_color,
        font_scale=0.36,
        thickness=1,
        max_width=content_width,
        fill_alpha=0.9,
        draw_border=True,
    )

    controls_top = draw_screen_controls(annotated, right_overlay_min_left)
    guidance_limit_bottom = controls_top - OVERLAY_GAP if controls_top is not None else None
    draw_guidance_overlay(
        annotated,
        feedback_lines,
        feedback_color,
        required_joints_status if show_required_panel else None,
        min_left=right_overlay_min_left,
        max_bottom=guidance_limit_bottom,
    )

    feedback_bottom = status_top + status_chip_height + FEEDBACK_CARD_BOTTOM_PADDING
    match_top = feedback_bottom + layout["gap"]

    match_bottom = draw_technique_info(
        annotated,
        technique_result,
        panel_left,
        match_top,
        panel_width,
        base_card_height,
        target_technique,
        preset,
    )
    return annotated


def summarize_points(points) -> dict:
    detected_parts = sum(1 for point in points if point is not None)
    missing_parts = len(points) - detected_parts
    missing_joint_names = get_missing_joint_names(points)
    required_joints_status = extract_required_joints_status(points)
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
        "missing_joint_names": missing_joint_names,
        "required_joints_status": required_joints_status,
        "connected_pairs": connected_pairs,
        "detection_rate": detection_rate,
        "has_pose": detected_parts > 0,
    }


def infer_frame(
    frame,
    net: cv.dnn.Net,
    in_width: int,
    in_height: int,
    threshold: float,
    target_technique: str | None = None,
    motion_history: deque | None = None,
    preset: str = DEFAULT_PRESET_NAME,
):
    points = detect_points(frame, net, in_width, in_height, threshold)
    ticks, _ = net.getPerfProfile()
    inference_ms = ticks / (cv.getTickFrequency() / 1000)
    summary = summarize_points(points)
    summary["missing_joint_names"] = get_missing_joint_names(points, target_technique)
    summary["required_joints_status"] = extract_required_joints_status(points, target_technique)
    frame_visibility = analyze_frame_visibility(frame)
    feedback_status, feedback_lines, feedback_color = build_pose_feedback(points, frame.shape, frame_visibility, target_technique)
    pose_features = extract_pose_features(points, frame.shape)
    technique_result = classify_technique(pose_features, motion_history=motion_history, target_technique=target_technique)
    if motion_history is not None:
        motion_history.append(pose_features.get("motion_snapshot", {}))
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
            summary["missing_joint_names"],
            summary["required_joints_status"],
            target_technique,
            preset,
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


def load_reference_image(reference_image_path: str | Path | None):
    if not reference_image_path:
        return None

    path = Path(reference_image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Reference image was not found at '{path}'.")

    image = cv.imread(str(path))
    if image is None:
        raise ValueError(
            f"Reference image could not be loaded from '{path}'. Use a PNG, JPG, JPEG, WEBP, BMP, or PPM file."
        )
    return image


def resize_and_crop_to_fill(image, target_width: int, target_height: int):
    image_height, image_width = image.shape[:2]
    if image_height <= 0 or image_width <= 0 or target_width <= 0 or target_height <= 0:
        raise ValueError("Reference image dimensions must be positive.")

    scale = min(target_width / image_width, target_height / image_height)
    resized_width = max(1, int(round(image_width * scale)))
    resized_height = max(1, int(round(image_height * scale)))
    if scale < 1:
        interpolation = cv.INTER_AREA
    elif max(image_width, image_height) <= 96:
        interpolation = cv.INTER_NEAREST
    else:
        interpolation = cv.INTER_CUBIC
    resized = cv.resize(image, (resized_width, resized_height), interpolation=interpolation)

    panel = np.full((target_height, target_width, 3), COLOR_PANEL_ALT, dtype=np.uint8)
    x_offset = max(0, (target_width - resized_width) // 2)
    y_offset = max(0, (target_height - resized_height) // 2)
    panel[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized
    cv.rectangle(panel, (0, 0), (target_width - 1, target_height - 1), COLOR_PANEL_EDGE, 2)
    return panel


def compose_reference_display(frame, reference_image):
    if reference_image is None:
        return frame

    frame_height, frame_width = frame.shape[:2]
    reference_width = min(max(260, int(frame_width * 0.36)), 420)
    reference_panel = resize_and_crop_to_fill(reference_image, reference_width, frame_height)
    separator = np.full((frame_height, 8, 3), COLOR_PANEL_EDGE, dtype=np.uint8)
    return cv.hconcat([frame, separator, reference_panel])


def show_frame(window_name: str, frame) -> bool:
    try:
        if window_name not in WINDOW_STATES:
            cv.namedWindow(window_name, cv.WINDOW_NORMAL)
            WINDOW_STATES[window_name] = False
        cv.imshow(window_name, frame)
        key_code = cv.waitKey(1)
    except cv.error as exc:
        WINDOW_STATES.pop(window_name, None)
        raise RuntimeError(
            "OpenCV could not open the webcam window. "
            "Check Windows camera permissions, close other camera apps, and retry."
        ) from exc

    key = key_code & 0xFF
    if key in (ord("f"), ord("F")):
        is_fullscreen = WINDOW_STATES.get(window_name, False)
        try:
            cv.setWindowProperty(
                window_name,
                cv.WND_PROP_FULLSCREEN,
                cv.WINDOW_NORMAL if is_fullscreen else cv.WINDOW_FULLSCREEN,
            )
        except cv.error as exc:
            raise RuntimeError(
                "OpenCV could not switch the webcam window to fullscreen. "
                "Retry without fullscreen or restart the app."
            ) from exc
        WINDOW_STATES[window_name] = not is_fullscreen
        return False
    return key in (27, ord("q"))


def display_single_frame(window_name: str, frame) -> None:
    try:
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.imshow(window_name, frame)
        cv.waitKey(0)
    except cv.error as exc:
        raise RuntimeError(
            "OpenCV could not open the display window. "
            "Check that a desktop display session is available, then retry."
        ) from exc
    finally:
        cv.destroyAllWindows()
        WINDOW_STATES.pop(window_name, None)


def process_image(args: argparse.Namespace, net: cv.dnn.Net, input_path: Path) -> int:
    frame = cv.imread(str(input_path))
    if frame is None:
        return fail(f"Could not read image file '{input_path}'.")
    reference_image = load_reference_image(args.reference_image)

    annotated, summary = infer_frame(frame, net, args.width, args.height, args.thr, args.target_technique, preset=args.preset)
    match_state = evaluate_target_match(summary["technique_result"], args.target_technique, args.preset)

    if args.output:
        write_image(Path(args.output), annotated)

    if not args.no_display:
        display_single_frame("OpenPose using OpenCV", compose_reference_display(annotated, reference_image))

    print(
        f"Detected technique: {summary['technique_result']['label']} "
        f"({summary['technique_result']['confidence'] * 100:.0f}%)"
    )
    if summary["missing_joint_names"]:
        print(f"Missing joints: {', '.join(summary['missing_joint_names'])}")

    write_json_summary(
        args.json_output,
        {
            "mode": "image",
            "input_source": str(input_path),
            "output_path": args.output or "",
            "processed_frames": 1,
            "detected_parts": summary["detected_parts"],
            "missing_parts": summary["missing_parts"],
            "missing_joint_names": summary["missing_joint_names"],
            "required_joints": summary["required_joints_status"]["required_joints"],
            "visible_required_joints": summary["required_joints_status"]["visible_required"],
            "missing_required_joints": summary["required_joints_status"]["missing_required"],
            "required_joints_visible_ratio": summary["required_joints_status"]["required_visible_ratio"],
            "required_joint_statuses": summary["required_joints_status"]["joint_statuses"],
            "connected_pairs": summary["connected_pairs"],
            "detection_rate": summary["detection_rate"],
            "has_pose": summary["has_pose"],
            "inference_ms": summary["inference_ms"],
            "feedback_status": summary["feedback_status"],
            "feedback_lines": summary["feedback_lines"],
            "recognized_technique": summary["technique_result"]["label"],
            "recognized_confidence": summary["technique_result"]["confidence"],
            "target_technique": args.target_technique or "",
            "preset": args.preset,
            "match_threshold": match_state["match_threshold"],
            "near_match_threshold": match_state["near_match_threshold"],
            "full_match": match_state["full_match"],
            "near_match": match_state["near_match"],
            "match_state": match_state["state"],
            "technique_match": match_state["qualifying_match"],
            "technique_match_ratio": 1.0 if match_state["qualifying_match"] else 0.0,
            "full_match_ratio": 1.0 if match_state["full_match"] else 0.0,
            "near_match_ratio": 1.0 if match_state["near_match"] else 0.0,
            "matched_repetitions": 1 if match_state["qualifying_match"] else 0,
            "full_matched_repetitions": 1 if match_state["full_match"] else 0,
            "near_matched_repetitions": 1 if match_state["near_match"] else 0,
        },
    )

    return 0


def process_stream(args: argparse.Namespace, net: cv.dnn.Net, input_source) -> int:
    capture, backend_name, attempted_backends = open_capture(input_source)
    if capture is None:
        return fail(build_capture_error(input_source, attempted_backends))
    reference_image = load_reference_image(args.reference_image)

    writer = None
    frame_count = 0
    output_path = Path(args.output) if args.output else None
    window_name = "OpenPose using OpenCV"
    last_summary = None
    technique_history = deque(maxlen=max(1, args.confirm_frames))
    motion_history = deque(maxlen=max(2, args.confirm_frames + 2))
    matched_frames = 0
    full_matched_frames = 0
    near_matched_frames = 0
    matched_repetitions = 0
    full_matched_repetitions = 0
    near_matched_repetitions = 0
    in_matched_streak = False
    matched_streak_state = "none"
    match_thresholds = get_target_match_thresholds(args.target_technique, args.preset)

    try:
        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break

            annotated, summary = infer_frame(
                frame,
                net,
                args.width,
                args.height,
                args.thr,
                args.target_technique,
                motion_history=motion_history,
                preset=args.preset,
            )
            technique_history.append(summary["technique_result"])
            stable_technique = smooth_technique_result(
                technique_history,
                target_technique=args.target_technique,
                confirm_frames=args.confirm_frames,
            )
            summary["stable_technique"] = stable_technique
            match_state = evaluate_target_match(stable_technique, args.target_technique, args.preset)
            summary["match_state"] = match_state
            last_summary = summary
            frame_count += 1

            if match_state["full_match"]:
                full_matched_frames += 1
            elif match_state["near_match"]:
                near_matched_frames += 1

            if match_state["qualifying_match"]:
                matched_frames += 1
                if not in_matched_streak:
                    matched_repetitions += 1
                    if match_state["full_match"]:
                        full_matched_repetitions += 1
                        matched_streak_state = "full"
                    else:
                        near_matched_repetitions += 1
                        matched_streak_state = "near"
                    in_matched_streak = True
                elif match_state["full_match"] and matched_streak_state == "near":
                    near_matched_repetitions = max(0, near_matched_repetitions - 1)
                    full_matched_repetitions += 1
                    matched_streak_state = "full"
            else:
                in_matched_streak = False
                matched_streak_state = "none"

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
                should_stop = show_frame(window_name, compose_reference_display(annotated, reference_image))

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
        if last_summary.get("missing_joint_names"):
            print(f"Missing joints: {', '.join(last_summary['missing_joint_names'])}")
        print(f"Feedback: {last_summary['feedback_status']} - {'; '.join(last_summary['feedback_lines'])}")
        print(f"Technique: {stable.get('label', 'Unknown')} ({stable.get('confidence', 0.0) * 100:.0f}%)")
        if args.target_technique:
            print(f"Target match ratio: {(matched_frames / frame_count) * 100:.0f}%")
            print(f"Matched repetitions: {matched_repetitions}")
            if near_matched_repetitions:
                print(f"Near-match repetitions: {near_matched_repetitions}")

    stable = last_summary.get("stable_technique", last_summary.get("technique_result", {})) if last_summary else {}
    overall_match_state = "full" if full_matched_frames else "near" if near_matched_frames else "none"
    write_json_summary(
        args.json_output,
        {
            "mode": "webcam" if isinstance(input_source, int) else "video",
            "input_source": describe_input_source(input_source),
            "output_path": str(output_path) if output_path else "",
            "processed_frames": frame_count,
            "detected_parts": last_summary["detected_parts"] if last_summary else 0,
            "missing_parts": last_summary["missing_parts"] if last_summary else len(BODY_PARTS),
            "missing_joint_names": last_summary["missing_joint_names"] if last_summary else [],
            "required_joints": last_summary["required_joints_status"]["required_joints"] if last_summary else [],
            "visible_required_joints": last_summary["required_joints_status"]["visible_required"] if last_summary else 0,
            "missing_required_joints": last_summary["required_joints_status"]["missing_required"] if last_summary else [],
            "required_joints_visible_ratio": last_summary["required_joints_status"]["required_visible_ratio"] if last_summary else 0.0,
            "required_joint_statuses": last_summary["required_joints_status"]["joint_statuses"] if last_summary else [],
            "connected_pairs": last_summary["connected_pairs"] if last_summary else 0,
            "detection_rate": last_summary["detection_rate"] if last_summary else 0.0,
            "has_pose": bool(last_summary and last_summary["has_pose"]),
            "inference_ms": last_summary["inference_ms"] if last_summary else 0.0,
            "feedback_status": last_summary["feedback_status"] if last_summary else "Pose not clear",
            "feedback_lines": last_summary["feedback_lines"] if last_summary else ["Move into frame and face the camera"],
            "recognized_technique": stable.get("label", "Unknown"),
            "recognized_confidence": stable.get("confidence", 0.0),
            "target_technique": args.target_technique or "",
            "preset": args.preset,
            "match_threshold": match_thresholds["match_threshold"],
            "near_match_threshold": match_thresholds["near_match_threshold"],
            "full_match": bool(args.target_technique and full_matched_frames > 0),
            "near_match": bool(args.target_technique and near_matched_frames > 0),
            "match_state": overall_match_state,
            "technique_match": bool(args.target_technique and matched_frames > 0),
            "technique_match_ratio": (matched_frames / frame_count) if args.target_technique and frame_count else 0.0,
            "full_match_ratio": (full_matched_frames / frame_count) if args.target_technique and frame_count else 0.0,
            "near_match_ratio": (near_matched_frames / frame_count) if args.target_technique and frame_count else 0.0,
            "matched_repetitions": matched_repetitions,
            "full_matched_repetitions": full_matched_repetitions,
            "near_matched_repetitions": near_matched_repetitions,
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
    preset: str = DEFAULT_PRESET_NAME,
    reference_image_path: Path | None = None,
) -> dict:
    net = load_net(model_path)
    frame = cv.imread(str(input_path))
    if frame is None:
        raise ValueError(f"Could not read image file '{input_path}'.")
    reference_image = load_reference_image(reference_image_path)

    annotated, summary = infer_frame(frame, net, width, height, threshold, target_technique, preset=preset)
    match_state = evaluate_target_match(summary["technique_result"], target_technique, preset)

    if output_path is not None:
        write_image(output_path, annotated)

    if display:
        display_single_frame("OpenPose using OpenCV", compose_reference_display(annotated, reference_image))

    return {
        "mode": "image",
        "input_source": str(input_path),
        "output_path": str(output_path) if output_path else "",
        "processed_frames": 1,
        "recognized_technique": summary["technique_result"]["label"],
        "recognized_confidence": summary["technique_result"]["confidence"],
        "target_technique": target_technique or "",
        "preset": preset,
        "missing_joint_names": summary["missing_joint_names"],
        "required_joints": summary["required_joints_status"]["required_joints"],
        "visible_required_joints": summary["required_joints_status"]["visible_required"],
        "missing_required_joints": summary["required_joints_status"]["missing_required"],
        "required_joints_visible_ratio": summary["required_joints_status"]["required_visible_ratio"],
        "required_joint_statuses": summary["required_joints_status"]["joint_statuses"],
        "match_threshold": match_state["match_threshold"],
        "near_match_threshold": match_state["near_match_threshold"],
        "full_match": match_state["full_match"],
        "near_match": match_state["near_match"],
        "match_state": match_state["state"],
        "technique_match": match_state["qualifying_match"],
        "technique_match_ratio": 1.0 if match_state["qualifying_match"] else 0.0,
        "full_match_ratio": 1.0 if match_state["full_match"] else 0.0,
        "near_match_ratio": 1.0 if match_state["near_match"] else 0.0,
        "matched_repetitions": 1 if match_state["qualifying_match"] else 0,
        "full_matched_repetitions": 1 if match_state["full_match"] else 0,
        "near_matched_repetitions": 1 if match_state["near_match"] else 0,
        **summary,
    }


def analyze_stream_source(
    input_source,
    model_path: Path,
    threshold: float = 0.2,
    width: int | None = None,
    height: int | None = None,
    output_path: Path | None = None,
    display: bool = False,
    max_frames: int | None = None,
    target_technique: str | None = None,
    confirm_frames: int | None = None,
    preset: str = DEFAULT_PRESET_NAME,
    reference_image_path: Path | None = None,
) -> dict:
    runtime_settings = resolve_runtime_settings(preset, width, height, confirm_frames)
    net = load_net(model_path)
    capture, backend_name, attempted_backends = open_capture(input_source)
    if capture is None:
        raise ValueError(build_capture_error(input_source, attempted_backends))
    reference_image = load_reference_image(reference_image_path)

    writer = None
    frame_count = 0
    total_detected_parts = 0
    total_connected_pairs = 0
    pose_frames = 0
    last_summary = None
    resolved_output = output_path if output_path else None
    technique_history = deque(maxlen=max(1, runtime_settings["confirm_frames"]))
    motion_history = deque(maxlen=max(2, runtime_settings["confirm_frames"] + 2))
    matched_frames = 0
    full_matched_frames = 0
    near_matched_frames = 0
    matched_repetitions = 0
    full_matched_repetitions = 0
    near_matched_repetitions = 0
    in_matched_streak = False
    matched_streak_state = "none"
    recognized_counts = {name: 0 for name in TECHNIQUE_NAMES}
    match_thresholds = get_target_match_thresholds(target_technique, runtime_settings["preset"])

    try:
        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break

            annotated, summary = infer_frame(
                frame,
                net,
                runtime_settings["width"],
                runtime_settings["height"],
                threshold,
                target_technique,
                motion_history=motion_history,
                preset=runtime_settings["preset"],
            )
            technique_history.append(summary["technique_result"])
            stable_technique = smooth_technique_result(
                technique_history,
                target_technique=target_technique,
                confirm_frames=runtime_settings["confirm_frames"],
            )
            match_state = evaluate_target_match(stable_technique, target_technique, runtime_settings["preset"])
            frame_count += 1
            last_summary = summary
            total_detected_parts += summary["detected_parts"]
            total_connected_pairs += summary["connected_pairs"]
            if summary["has_pose"]:
                pose_frames += 1
            if stable_technique["label"] in recognized_counts:
                recognized_counts[stable_technique["label"]] += 1
            if match_state["full_match"]:
                full_matched_frames += 1
            elif match_state["near_match"]:
                near_matched_frames += 1
            if match_state["qualifying_match"]:
                matched_frames += 1
                if not in_matched_streak:
                    matched_repetitions += 1
                    if match_state["full_match"]:
                        full_matched_repetitions += 1
                        matched_streak_state = "full"
                    else:
                        near_matched_repetitions += 1
                        matched_streak_state = "near"
                    in_matched_streak = True
                elif match_state["full_match"] and matched_streak_state == "near":
                    near_matched_repetitions = max(0, near_matched_repetitions - 1)
                    full_matched_repetitions += 1
                    matched_streak_state = "full"
            else:
                in_matched_streak = False
                matched_streak_state = "none"
            summary["stable_technique"] = stable_technique
            summary["match_state"] = match_state

            if resolved_output and writer is None:
                fps = capture.get(cv.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 25.0
                frame_height, frame_width = annotated.shape[:2]
                writer = build_writer(resolved_output, fps, (frame_width, frame_height))

            if writer is not None:
                writer.write(annotated)

            if display and show_frame("OpenPose using OpenCV", compose_reference_display(annotated, reference_image)):
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
    overall_match_state = "full" if full_matched_frames else "near" if near_matched_frames else "none"

    return {
        "mode": "webcam" if isinstance(input_source, int) else "video",
        "input_source": describe_input_source(input_source),
        "output_path": str(resolved_output) if resolved_output else "",
        "processed_frames": frame_count,
        "detected_parts": last_summary["detected_parts"] if last_summary else 0,
        "missing_parts": last_summary["missing_parts"] if last_summary else len(BODY_PARTS),
        "missing_joint_names": last_summary["missing_joint_names"] if last_summary else [],
        "required_joints": last_summary["required_joints_status"]["required_joints"] if last_summary else [],
        "visible_required_joints": last_summary["required_joints_status"]["visible_required"] if last_summary else 0,
        "missing_required_joints": last_summary["required_joints_status"]["missing_required"] if last_summary else [],
        "required_joints_visible_ratio": last_summary["required_joints_status"]["required_visible_ratio"] if last_summary else 0.0,
        "required_joint_statuses": last_summary["required_joints_status"]["joint_statuses"] if last_summary else [],
        "connected_pairs": last_summary["connected_pairs"] if last_summary else 0,
        "detection_rate": last_summary["detection_rate"] if last_summary else 0.0,
        "has_pose": bool(last_summary and last_summary["has_pose"]),
        "inference_ms": last_summary["inference_ms"] if last_summary else 0.0,
        "feedback_status": last_summary["feedback_status"] if last_summary else "Pose not clear",
        "feedback_lines": last_summary["feedback_lines"] if last_summary else ["Move into frame and face the camera"],
        "recognized_technique": stable.get("label", "Unknown"),
        "recognized_confidence": stable.get("confidence", 0.0),
        "target_technique": target_technique or "",
        "preset": runtime_settings["preset"],
        "match_threshold": match_thresholds["match_threshold"],
        "near_match_threshold": match_thresholds["near_match_threshold"],
        "full_match": bool(target_technique and full_matched_frames > 0),
        "near_match": bool(target_technique and near_matched_frames > 0),
        "match_state": overall_match_state,
        "technique_match": technique_match,
        "technique_match_ratio": technique_match_ratio,
        "full_match_ratio": (full_matched_frames / frame_count) if target_technique else 0.0,
        "near_match_ratio": (near_matched_frames / frame_count) if target_technique else 0.0,
        "matched_repetitions": matched_repetitions,
        "full_matched_repetitions": full_matched_repetitions,
        "near_matched_repetitions": near_matched_repetitions,
        "recognized_counts": recognized_counts,
        "average_detected_parts": average_detected_parts,
        "average_connected_pairs": average_connected_pairs,
        "pose_frame_ratio": pose_frame_ratio,
    }


def main() -> int:
    args = parse_args()

    runtime_settings = resolve_runtime_settings(args.preset, args.width, args.height, args.confirm_frames)
    args.preset = runtime_settings["preset"]
    args.width = runtime_settings["width"]
    args.height = runtime_settings["height"]
    args.confirm_frames = runtime_settings["confirm_frames"]

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