import cv2
import gc
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import gradio as gr
import numpy as np
import torch
from PIL import Image

from backend import DEFAULT_MODELS, SurveillanceAI

# --- Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# --- Configuration ---


@dataclass
class Config:
    """Centralised configuration for the surveillance system."""

    buffer_seconds: int = 4
    sample_frames: int = 8
    check_interval: float = 2.0
    resize_height: int = 480
    display_max_width: int = 1024
    display_frame_interval: int = 4
    alert_linger_extra: float = 1.5
    max_log_entries: int = 100
    max_debug_entries: int = 200
    evidence_dir: str = field(default_factory=lambda: os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "evidence_snapshots")
    ))


CONFIG = Config()
os.makedirs(CONFIG.evidence_dir, exist_ok=True)

DEFAULT_PROMPT = (
    "You are a surveillance system monitoring a computer lab. "
    "You are shown a sequence of consecutive frames captured seconds apart from a surveillance camera.\n\n"
    "Look carefully at EVERY person visible in these frames, including those partially visible, "
    "far away, in the background, or in groups.\n\n"
    "Determine if ANY of the following PROHIBITED activities are occurring:\n"
    "1. Running (person moving fast across frames)\n"
    "2. Fighting or aggressive physical contact between people\n"
    "3. Eating or Drinking (food, beverages, bottles, cups near mouth)\n"
    "4. Tampering with or damaging equipment\n"
    "5. Sleeping (head down on desk, eyes closed)\n"
    "6. Using a phone or mobile device (holding phone, looking at phone)\n\n"
    "IMPORTANT: If you see ANY prohibited activity by ANY person, you MUST output an ALERT.\n"
    "Even if only one person out of many is doing something prohibited, output ALERT.\n\n"
    "OUTPUT FORMAT (respond with EXACTLY one line):\n"
    "- If ALL activities are safe: Status: Safe\n"
    "- If ANY prohibited activity is detected: ALERT: [describe what activity and who]\n\n"
    "Examples:\n"
    "- Status: Safe\n"
    "- ALERT: Person at desk is eating food\n"
    "- ALERT: Person in back row using phone\n"
    "- ALERT: Two people fighting near entrance"
)

STRICT_MODE_SUFFIX = (
    "\n\nSTRICT MODE (video file):\n"
    "If ANY prohibited activity appears in ANY frame, output ALERT. "
    "Do not default to SAFE if uncertain. "
    "If motion suggests running or aggressive contact, alert. "
    "If a person appears to eat/drink, tamper with equipment, phone usage, or sleep, alert. "
    "Return exactly one line in the required format."
)

SYSTEM_PROMPT = (
    "You are a strict computer lab surveillance AI. "
    "Your ONLY job is to detect prohibited activities and output exactly one line: "
    "either 'Status: Safe' or 'ALERT: [description]'. "
    "Never describe the scene. Never explain your reasoning. "
    "Just classify and respond in the required format."
)

BBOX_LOCALIZATION_PROMPT = (
    "This surveillance image contains a person performing a prohibited activity: {activity}. "
    "Look carefully at the ENTIRE image. Locate the person performing that activity. "
    "Return ONLY the bounding box of that person as [xmin, ymin, xmax, ymax] on a 0-1000 scale "
    "where (0,0) is top-left and (1000,1000) is bottom-right. "
    "Return nothing else."
)


# --- System State ---


class SystemState:
    """Thread-safe container for shared surveillance state."""

    def __init__(self) -> None:
        # Control flags
        self.running: bool = False
        self.source_type: str = "Camera"
        self.use_video_mode: bool = False
        self.enable_bbox: bool = False
        self.debug_enabled: bool = True
        self.debug_save_clips: bool = False
        self.sample_strategy: str = "Uniform"

        # Tunable parameters (may be overridden per source type)
        self.sample_frames: int = CONFIG.sample_frames
        self.buffer_seconds: int = CONFIG.buffer_seconds
        self.prompt: str = DEFAULT_PROMPT

        # Frame buffer (protected by buffer_lock)
        self.frame_buffer: deque = deque(
            maxlen=int(CONFIG.buffer_seconds * 30))
        self.buffer_lock = threading.Lock()

        # AI trigger
        self.trigger_ai_event = threading.Event()

        # AI results (protected by state_lock)
        self.ai_text: str = "Initializing..."
        self.ai_boxes: list[tuple[float, float, float, float]] = []
        self.last_alert_time: float = 0.0
        self.last_safe_text: str = "Status: Safe"
        self.state_lock = threading.Lock()

        # Logs (protected by log_lock)
        self.log_history: list[str] = []
        self.debug_history: list[str] = []
        self.log_lock = threading.Lock()


state = SystemState()
ai_engine = SurveillanceAI()

# --- File Utilities ---


def safe_copy_video(src: str, dst: str, retries: int = 5, delay: float = 0.5) -> bool:
    """Copy a file with retries to handle Windows file-lock contention."""
    for attempt in range(retries):
        try:
            time.sleep(delay)
            shutil.copy(src, dst)
            return True
        except PermissionError:
            logger.warning(
                "File locked, retrying copy (%d/%d)...", attempt + 1, retries)
            time.sleep(1.0)
        except OSError as exc:
            logger.error("Copy failed: %s", exc)
            return False
    return False


def open_video_capture(
    upload_path: str, local_path: str
) -> tuple[Optional[cv2.VideoCapture], Optional[str]]:
    """Open a video file, attempting a safe local copy first."""
    if not upload_path or not os.path.exists(upload_path):
        return None, None

    used_path = local_path if safe_copy_video(
        upload_path, local_path) else upload_path
    cap = cv2.VideoCapture(used_path)
    if not cap.isOpened():
        cap.release()
        return None, used_path

    return cap, used_path


def resolve_upload_path(uploaded) -> Optional[str]:
    """Normalise Gradio's various upload return types into a file path."""
    if uploaded is None:
        return None
    if isinstance(uploaded, str):
        return uploaded
    if isinstance(uploaded, dict):
        return uploaded.get("path") or uploaded.get("name")
    return getattr(uploaded, "name", None) or getattr(uploaded, "path", None)


# --- Image / Frame Utilities ---


def extract_boxes(
    text_response: str, img_width: int, img_height: int
) -> list[tuple[int, int, int, int]]:
    """Parse bounding boxes in [x1, y1, x2, y2] (0-1000 scale) from AI text."""
    boxes: list[tuple[int, int, int, int]] = []
    for match in re.findall(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', text_response):
        try:
            x1n, y1n, x2n, y2n = map(int, match)
            boxes.append((
                int(x1n / 1000 * img_width),
                int(y1n / 1000 * img_height),
                int(x2n / 1000 * img_width),
                int(y2n / 1000 * img_height),
            ))
        except (ValueError, IndexError):
            continue
    return boxes


def _extract_alert_description(alert_text: str) -> str:
    """Pull the short activity description from 'ALERT: <description> [bbox]'."""
    desc = alert_text
    # Strip leading "ALERT:" prefix
    if ":" in desc:
        desc = desc.split(":", 1)[1].strip()
    # Remove bounding-box bracket suffix
    bracket_pos = desc.find("[")
    if bracket_pos != -1:
        desc = desc[:bracket_pos].strip()
    # Trim to a reasonable length
    return desc[:80] if desc else "ALERT"


def draw_alert_overlay(
    frame: np.ndarray,
    norm_boxes: list[tuple[float, float, float, float]],
    alert_text: str = "",
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 4,
) -> None:
    """Draw bounding boxes and alert description onto a BGR frame (in-place)."""
    h, w = frame.shape[:2]
    label = _extract_alert_description(alert_text) if alert_text else "ALERT"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    font_thickness = 2

    for nx1, ny1, nx2, ny2 in norm_boxes:
        x1, y1 = int(nx1 * w), int(ny1 * h)
        x2, y2 = int(nx2 * w), int(ny2 * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Draw label with background above the box
        (tw, th), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness,
        )
        label_y = max(y1 - 8, th + 4)
        cv2.rectangle(
            frame, (x1, label_y - th - 4), (x1 + tw + 6, label_y + 4),
            color, -1,
        )
        cv2.putText(
            frame, label, (x1 + 3, label_y),
            font, font_scale, (255, 255, 255), font_thickness,
        )


def get_sample_indices(
    total_frames: int, target_count: int, strategy: str,
    buffer_snapshot: Optional[list] = None,
) -> list[int]:
    """Return frame indices to sample from a buffer, per the chosen strategy."""
    if total_frames <= 0:
        return []
    if total_frames <= target_count:
        return list(range(total_frames))

    if strategy == "Recent Focus":
        recent_window = max(target_count, int(total_frames * 0.6))
        start = max(0, total_frames - recent_window)
        return np.linspace(start, total_frames - 1, target_count, dtype=int).tolist()

    if strategy == "Motion Focus" and buffer_snapshot is not None:
        return _motion_focus_indices(buffer_snapshot, total_frames, target_count)

    # Default: Uniform
    return np.linspace(0, total_frames - 1, target_count, dtype=int).tolist()


def _motion_focus_indices(
    buffer_snapshot: list, total_frames: int, target_count: int,
) -> list[int]:
    """Pick frames with the highest inter-frame motion."""
    scores: list[tuple[int, float]] = []
    prev = None
    for idx, frame in enumerate(buffer_snapshot):
        if prev is None:
            prev = frame
            continue
        curr_small = cv2.resize(frame, (64, 36))
        prev_small = cv2.resize(prev, (64, 36))
        score = float(np.sum(cv2.absdiff(curr_small, prev_small)))
        scores.append((idx, score))
        prev = frame

    scores.sort(key=lambda x: x[1], reverse=True)
    picked = sorted(
        {idx for idx, _ in scores[: max(
            1, target_count - 1)]} | {total_frames - 1}
    )

    # Back-fill with uniform indices if needed
    if len(picked) < target_count:
        uniform = np.linspace(0, total_frames - 1,
                              target_count, dtype=int).tolist()
        for i in uniform:
            if len(picked) >= target_count:
                break
            if i not in picked:
                picked.append(i)
        picked = sorted(picked)

    return picked


def sample_frames_from_buffer(
    buffer_snapshot: list, target_count: int = CONFIG.sample_frames,
    strategy: str = "Uniform", indices: Optional[list[int]] = None,
) -> list[Image.Image]:
    """Sample and resize frames from a buffer snapshot, returning PIL images."""
    total = len(buffer_snapshot)
    if indices is None:
        indices = get_sample_indices(
            total, target_count, strategy, buffer_snapshot)

    pil_frames: list[Image.Image] = []
    for i in indices:
        frame = buffer_snapshot[i]
        h, w = frame.shape[:2]
        scale = CONFIG.resize_height / h
        resized = cv2.resize(frame, (int(w * scale), CONFIG.resize_height))
        pil_frames.append(Image.fromarray(
            cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))

    return pil_frames


# --- Logging Helpers ---


def log_debug(message: str) -> None:
    """Append a timestamped message to the in-memory debug log."""
    if not state.debug_enabled:
        return
    entry = f"[{datetime.now():%H:%M:%S}] {message}"
    with state.log_lock:
        state.debug_history.insert(0, entry)
        if len(state.debug_history) > CONFIG.max_debug_entries:
            state.debug_history.pop()


def _append_event_log(entry: str) -> None:
    """Thread-safe append to the event log."""
    with state.log_lock:
        state.log_history.insert(0, entry)
        if len(state.log_history) > CONFIG.max_log_entries:
            state.log_history.pop()


def build_prompt(base_prompt: str, strict: bool = False) -> str:
    """Optionally append strict-mode instructions to the base prompt."""
    return base_prompt + STRICT_MODE_SUFFIX if strict else base_prompt


# --- Keywords that indicate prohibited activity even in freeform text ---
_ACTIVITY_KEYWORDS = [
    "eating", "drinking", "food", "beverage", "bottle", "cup",
    "fighting", "aggressive", "punch", "hit", "attack",
    "running", "sprint",
    "sleeping", "asleep", "head down",
    "phone", "mobile", "cellphone", "smartphone", "device",
    "tamper", "damage", "vandal",
]


def _normalize_ai_response(raw_text: str) -> str:
    """Normalise AI output so downstream 'ALERT' checks work reliably.

    Handles:
    - Case variations  ('Alert:', 'alert:', 'ALERT:')
    - Error passthrough ('Error: ...')
    - Ambiguous freeform text that describes a prohibited activity but
      doesn't use the ALERT keyword
    """
    if not raw_text:
        return "Status: Safe"

    # Pass through errors untouched
    if raw_text.startswith("Error:"):
        return raw_text

    text_lower = raw_text.lower().strip()

    # Already correctly formatted
    if text_lower.startswith("alert"):
        # Capitalise to canonical form: "ALERT: ..."
        colon_pos = raw_text.find(":")
        if colon_pos != -1:
            return "ALERT:" + raw_text[colon_pos + 1:]
        return "ALERT: " + raw_text[6:].strip()  # len('alert') + possible space

    if "status" in text_lower and "safe" in text_lower:
        return "Status: Safe"

    # --- Fallback: check for prohibited-activity keywords in freeform text ---
    for kw in _ACTIVITY_KEYWORDS:
        if kw in text_lower:
            # The model described a prohibited activity without using ALERT format
            description = raw_text.strip().split("\n")[0][:120]
            log_debug(f"Normalised freeform response to ALERT (matched '{kw}'): {description}")
            return f"ALERT: {description}"

    # Nothing suspicious found
    return "Status: Safe"


def _frame_hash(pil_img: Image.Image, size: int = 16) -> str:
    """Compute a coarse perceptual hash for deduplication / debugging."""
    gray = pil_img.convert("L").resize((size, size))
    arr = np.array(gray, dtype=np.uint8)
    bits = (arr > arr.mean()).flatten()
    nibbles = []
    for i in range(0, len(bits), 4):
        val = sum(int(bits[i + j]) << (3 - j) for j in range(4))
        nibbles.append(f"{val:x}")
    return "".join(nibbles)


def _save_contact_sheet(
    frames: list[Image.Image], out_path: str, cols: int = 4, thumb_h: int = 180,
) -> bool:
    """Stitch sampled frames into a contact-sheet JPEG for debugging."""
    if not frames:
        return False

    thumbs = []
    for img in frames:
        w, h = img.size
        thumbs.append(img.resize((int(w * thumb_h / h), thumb_h)))

    rows = (len(thumbs) + cols - 1) // cols
    row_widths = [0] * rows
    row_heights = [0] * rows
    for idx, t in enumerate(thumbs):
        r = idx // cols
        row_widths[r] += t.size[0]
        row_heights[r] = max(row_heights[r], t.size[1])

    sheet = Image.new("RGB", (max(row_widths), sum(row_heights)), (10, 10, 10))

    y = 0
    idx = 0
    for r in range(rows):
        x = 0
        for _ in range(cols):
            if idx >= len(thumbs):
                break
            sheet.paste(thumbs[idx], (x, y))
            x += thumbs[idx].size[0]
            idx += 1
        y += row_heights[r]

    sheet.save(out_path)
    return True


# --- AI Worker ---


def _make_localization_frame(raw_frame: np.ndarray) -> Image.Image:
    """Convert a raw BGR frame to a PIL image for bbox localization.

    Uses a higher resolution than the clip sampling path so the
    model can locate people more precisely.
    """
    h, w = raw_frame.shape[:2]
    # Use up to 720p for localization (higher than the 480p clip frames)
    target_h = min(h, 720)
    scale = target_h / h
    resized = cv2.resize(raw_frame, (int(w * scale), target_h))
    return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))


def _localize_bbox(
    raw_frame: np.ndarray, activity_desc: str,
) -> list[tuple[float, float, float, float]]:
    """Dedicated single-frame pass: locate the person on the actual last frame.

    Returns normalised [0, 1] bounding boxes.
    """
    loc_image = _make_localization_frame(raw_frame)
    loc_w, loc_h = loc_image.size
    prompt = BBOX_LOCALIZATION_PROMPT.format(activity=activity_desc)

    try:
        loc_text = ai_engine.analyze_single_image(loc_image, prompt)
        log_debug(f"Bbox localization output: {loc_text[:200]}")
        boxes_px = extract_boxes(loc_text, loc_w, loc_h)
        if boxes_px:
            return [
                (bx1 / loc_w, by1 / loc_h, bx2 / loc_w, by2 / loc_h)
                for bx1, by1, bx2, by2 in boxes_px
            ]
    except Exception as exc:
        log_debug(f"Bbox localization error: {exc}")

    return []


def _run_ai_analysis(
    snapshot: list,
) -> tuple[str, list[tuple[float, float, float, float]]]:
    """Run AI analysis on a buffer snapshot.

    Pass 1 — Activity detection on sampled clip (multi-frame, lower res).
    Pass 2 — If ALERT, precise bbox localization on the actual last frame
             at higher resolution (single-image pass).

    Returns (result_text, normalised_boxes).
    """
    total_frames = len(snapshot)
    target_count = state.sample_frames
    sampled_indices = get_sample_indices(
        total_frames, target_count, state.sample_strategy, snapshot,
    )

    clip_images = sample_frames_from_buffer(
        snapshot, target_count=target_count,
        strategy=state.sample_strategy, indices=sampled_indices,
    )

    # Debug logging
    hashes = [_frame_hash(img) for img in clip_images]
    log_debug(
        f"AI trigger: buffer={total_frames}, frames={target_count}, "
        f"strategy={state.sample_strategy}, indices={sampled_indices}"
    )
    log_debug(f"AI sample hashes: {hashes}")

    if state.debug_save_clips:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(CONFIG.evidence_dir, f"debug_clip_{ts}.jpg")
        _save_contact_sheet(clip_images, out_path)
        log_debug(f"Saved debug clip sheet: {out_path}")

    # --- Pass 1: Activity detection (multi-frame) ---
    if state.source_type == "Video File" and state.use_video_mode:
        result_text = ai_engine.analyze_video_clip_as_video(
            clip_images, state.prompt, fps=2.0,
            system_prompt=SYSTEM_PROMPT,
        )
    else:
        result_text = ai_engine.analyze_video_clip(
            clip_images, state.prompt,
            system_prompt=SYSTEM_PROMPT,
        )

    log_debug(f"AI raw output: {(result_text or '<empty>')[:200]}")

    # Normalise so all downstream 'ALERT' checks work regardless of model
    # casing / freeform responses
    result_text = _normalize_ai_response(result_text)
    log_debug(f"AI normalised: {result_text[:200]}")

    # --- Pass 2: Bbox localization on the actual last frame ---
    if "ALERT" in result_text and state.enable_bbox:
        activity_desc = _extract_alert_description(result_text)
        last_raw_frame = snapshot[-1]  # full-resolution original frame
        log_debug(
            f"Running bbox localization on last frame "
            f"({last_raw_frame.shape[1]}x{last_raw_frame.shape[0]}), "
            f"activity='{activity_desc}'"
        )
        norm_boxes = _localize_bbox(last_raw_frame, activity_desc)
        if norm_boxes:
            log_debug(f"Localized {len(norm_boxes)} box(es): {norm_boxes}")
        else:
            log_debug("Bbox localization returned no boxes")
    else:
        norm_boxes = []

    return result_text, norm_boxes


def _save_alert_evidence(
    snapshot: list, norm_boxes: list[tuple[float, float, float, float]],
    alert_text: str = "",
) -> None:
    """Save the last frame with alert boxes burned in as evidence."""
    key_frame = snapshot[-1].copy()
    draw_alert_overlay(key_frame, norm_boxes, alert_text=alert_text)

    # Build a filename that includes the activity description
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    desc = _extract_alert_description(alert_text)
    # Sanitise for filesystem: keep only alphanumeric, spaces, hyphens
    safe_desc = re.sub(r'[^\w\s-]', '', desc).strip().replace(' ', '_')[:50]
    tag = f"_{safe_desc}" if safe_desc else ""
    filename = os.path.join(CONFIG.evidence_dir, f"alert_{ts}{tag}.jpg")

    if cv2.imwrite(filename, key_frame):
        log_debug(f"Saved alert image: {filename}")
    else:
        log_debug(f"Failed to save alert image: {filename}")


def ai_worker_loop() -> None:
    """Background thread: waits for trigger events, then runs AI analysis."""
    logger.info("AI Worker Thread started (Sliding Window Mode)")

    while True:
        triggered = state.trigger_ai_event.wait(timeout=1.0)

        if not state.running:
            time.sleep(1)
            continue

        if not triggered:
            continue

        state.trigger_ai_event.clear()

        if not ai_engine.ready:
            with state.state_lock:
                state.ai_text = "⚠️ AI Model not loaded."
            log_debug("AI trigger ignored: model not loaded")
            continue

        with state.buffer_lock:
            buffer_size = len(state.frame_buffer)
            if buffer_size < 4:
                log_debug(
                    f"AI trigger: buffer too small ({buffer_size}), waiting for more frames...")
                with state.state_lock:
                    state.ai_text = f"Buffering frames... ({buffer_size} frames)"
                continue
            snapshot = list(state.frame_buffer)

        try:
            result_text, norm_boxes = _run_ai_analysis(snapshot)
            timestamp = datetime.now().strftime("%H:%M:%S")

            with state.state_lock:
                state.ai_text = result_text
                state.ai_boxes = norm_boxes
                if "ALERT" in result_text:
                    state.last_alert_time = time.time()
                else:
                    state.last_safe_text = result_text

            if "ALERT" in result_text:
                _save_alert_evidence(snapshot, norm_boxes,
                                     alert_text=result_text)
                _append_event_log(f"[{timestamp}] 🚨 {result_text}")
            else:
                _append_event_log(f"[{timestamp}] 👁️ {result_text[:50]}...")

        except Exception as exc:
            logger.exception("AI Analysis Error")
            log_debug(f"AI error: {exc}")
            # Attempt to recover GPU memory after errors (especially OOM)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


threading.Thread(target=ai_worker_loop, name="AI_Worker", daemon=True).start()


# --- Monitoring Pipeline ---


def apply_model_settings(model_choice: str, use_4bit: bool) -> str:
    """Load the selected AI model."""
    if not model_choice:
        return "⚠️ Please select a model."
    return ai_engine.load_model(model_choice, use_4bit)


def _configure_run_params(
    source_type: str, prompt_text: str, debug_enabled: bool,
    debug_save: bool, sample_strategy: str, use_video_mode: bool,
    buffer_seconds: int = CONFIG.buffer_seconds,
    sample_frames: int = CONFIG.sample_frames,
    enable_bbox: bool = False,
) -> None:
    """Set SystemState parameters for a new monitoring run."""
    strict = source_type == "Video File"
    state.prompt = build_prompt(prompt_text, strict=strict)
    state.debug_enabled = bool(debug_enabled)
    state.debug_save_clips = bool(debug_save)
    state.sample_strategy = sample_strategy or "Uniform"
    state.source_type = source_type
    state.use_video_mode = bool(use_video_mode)
    state.buffer_seconds = int(buffer_seconds)
    state.sample_frames = int(sample_frames)
    state.enable_bbox = bool(enable_bbox)
    log_debug(
        f"Prompt strict mode: {strict}, buffer={state.buffer_seconds}s, "
        f"sample_frames={state.sample_frames}"
    )


def _open_capture(
    source_type: str, cam_id: str, video_file,
) -> tuple[Optional[cv2.VideoCapture], bool, Optional[str]]:
    """Open a VideoCapture from either a camera or an uploaded file.

    Returns (cap, is_file, error_message).
    """
    if source_type == "Video File":
        upload_path = resolve_upload_path(video_file)
        if not upload_path:
            return None, True, "❌ Error: Upload video."
        local_path = os.path.join(
            tempfile.gettempdir(), "sentinel_processing.mp4")
        cap, _ = open_video_capture(upload_path, local_path)
        if cap is None:
            return None, True, f"❌ Error: Could not open video. Path: {upload_path}"
        return cap, True, None

    src = int(cam_id) if str(cam_id).strip().isdigit() else cam_id
    cap = cv2.VideoCapture(src)
    return cap, False, None


def _render_display_frame(
    frame: np.ndarray, ai_text: str, alert_active: bool, last_safe: str,
) -> np.ndarray:
    """Prepare a frame for Gradio display: resize, overlay status bar."""
    h, w = frame.shape[:2]
    if w > CONFIG.display_max_width:
        scale = CONFIG.display_max_width / w
        frame = cv2.resize(frame, (CONFIG.display_max_width, int(h * scale)))
        h, w = frame.shape[:2]

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.rectangle(frame_rgb, (0, 0), (w, 50), (0, 0, 0), -1)
    display_text = ai_text if alert_active else (last_safe or ai_text)
    color = (255, 50, 50) if alert_active else (50, 255, 50)
    cv2.putText(
        frame_rgb, display_text[:90], (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
    )
    return frame_rgb


# --- Video-File Segment Pipeline ---


def _analyze_and_yield_segment(
    segment_frames: list, seg_idx: int, total_segments: int,
):
    """Run AI analysis on one video segment and yield UI updates."""
    num_frames = len(segment_frames)
    progress_msg = (
        f"Analyzing segment {seg_idx}/{total_segments} ({num_frames} frames)..."
    )
    log_debug(
        f"Segment {seg_idx}/{total_segments}: {num_frames} frames"
    )

    # Show a preview frame while analysis runs
    mid_frame = segment_frames[len(segment_frames) // 2]
    display = _render_display_frame(mid_frame, progress_msg, False, "")
    with state.log_lock:
        debug_text = "\n".join(state.debug_history)
        log_text = "\n".join(state.log_history)
    yield display, progress_msg, log_text, debug_text

    if not ai_engine.ready:
        log_debug("Skipping segment: model not ready")
        return

    try:
        result_text, norm_boxes = _run_ai_analysis(segment_frames)
        timestamp = datetime.now().strftime("%H:%M:%S")

        with state.state_lock:
            state.ai_text = result_text
            state.ai_boxes = norm_boxes
            if "ALERT" in result_text:
                state.last_alert_time = time.time()
            else:
                state.last_safe_text = result_text

        alert_active = "ALERT" in result_text
        if alert_active:
            _save_alert_evidence(segment_frames, norm_boxes,
                                 alert_text=result_text)
            _append_event_log(
                f"[{timestamp}] \U0001f6a8 Seg {seg_idx}: {result_text}"
            )
            last_frame = segment_frames[-1].copy()
            draw_alert_overlay(last_frame, norm_boxes, alert_text=result_text)
            display = _render_display_frame(
                last_frame, result_text, True, ""
            )
        else:
            _append_event_log(
                f"[{timestamp}] \U0001f441\ufe0f Seg {seg_idx}: {result_text[:50]}..."
            )
            display = _render_display_frame(
                segment_frames[-1], result_text, False, result_text
            )

        with state.log_lock:
            debug_text = "\n".join(state.debug_history)
            log_text = "\n".join(state.log_history)
        yield display, result_text, log_text, debug_text

    except Exception as exc:
        logger.exception("AI Analysis Error on segment %d", seg_idx)
        log_debug(f"AI error on segment {seg_idx}: {exc}")


def _process_video_file(
    cap: cv2.VideoCapture,
):
    """Process a video file segment-by-segment with AI analysis.

    Reads frames at full speed (not real-time), groups them into segments
    of buffer_seconds length, and runs AI analysis on each segment
    sequentially.  This ensures every part of the video is checked.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not (0 < fps <= 120):
        fps = 30.0

    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frame_count / fps if fps > 0 else 0
    segment_size = max(4, int(state.buffer_seconds * fps))
    num_segments = max(
        1, (total_frame_count + segment_size - 1) // segment_size
    )

    log_debug(
        f"Video file: {total_frame_count} frames, "
        f"{video_duration:.1f}s @ {fps:.1f}fps"
    )
    log_debug(
        f"Segment size: {segment_size} frames ({state.buffer_seconds}s), "
        f"~{num_segments} segments"
    )

    segment_frames: list = []
    seg_idx = 0

    try:
        while state.running:
            ret, frame = cap.read()
            if not ret:
                break
            segment_frames.append(frame)

            if len(segment_frames) >= segment_size:
                seg_idx += 1
                yield from _analyze_and_yield_segment(
                    segment_frames, seg_idx, num_segments,
                )
                segment_frames = []

        # Analyse leftover frames
        if segment_frames and len(segment_frames) >= 4 and state.running:
            seg_idx += 1
            yield from _analyze_and_yield_segment(
                segment_frames, seg_idx, num_segments,
            )

    finally:
        cap.release()
        state.running = False

    # Final status
    with state.log_lock:
        debug_text = "\n".join(state.debug_history)
        log_text = "\n".join(state.log_history)
    yield (
        None,
        f"\u2705 Video analysis complete ({seg_idx} segments analyzed).",
        log_text,
        debug_text,
    )


def monitor_stream(
    source_type, cam_id, video_file, interval_setting,
    prompt_text, debug_enabled, debug_save, sample_strategy, use_video_mode,
    buffer_seconds, sample_frames, enable_bbox,
):
    """Main Gradio generator: streams annotated frames with AI analysis."""
    _configure_run_params(
        source_type, prompt_text, debug_enabled,
        debug_save, sample_strategy, use_video_mode,
        buffer_seconds=buffer_seconds, sample_frames=sample_frames,
        enable_bbox=enable_bbox,
    )

    if not ai_engine.ready:
        yield None, "❌ ERROR: No AI Model Loaded.", "", ""
        return

    with state.buffer_lock:
        state.frame_buffer.clear()
    with state.log_lock:
        state.debug_history.clear()

    # Open source
    if source_type == "Video File":
        yield None, "⏳ Processing file...", "", ""

    cap, is_file, error = _open_capture(source_type, cam_id, video_file)
    if error:
        yield None, error, "", ""
        return
    if not cap or not cap.isOpened():
        yield None, "Error: Cannot open source", "", ""
        return

    # Video files: fast segment-based analysis (not real-time)
    if is_file:
        state.running = True
        with state.state_lock:
            state.ai_text = "Reading video file..."
        yield from _process_video_file(cap)
        return

    # --- Camera / live-stream real-time path ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not (0 < fps <= 120):
        fps = 30.0

    with state.buffer_lock:
        state.frame_buffer = deque(maxlen=int(state.buffer_seconds * fps))

    state.running = True
    with state.state_lock:
        state.ai_text = "Monitoring active, building buffer..."
    # Start the timer early so first check happens quickly
    last_check_time = time.time() - interval_setting + \
        2.0  # First check in ~2 seconds
    frame_counter = 0

    try:
        while state.running:
            loop_start = time.time()
            ret, frame = cap.read()

            if not ret:
                if is_file:
                    state.ai_text = "End of Video."
                    break
                time.sleep(1)
                continue

            with state.buffer_lock:
                state.frame_buffer.append(frame)
                current_buffer_size = len(state.frame_buffer)

            now = time.time()
            if (now - last_check_time) > interval_setting and current_buffer_size >= 4:
                state.trigger_ai_event.set()
                last_check_time = now
                log_debug(
                    f"AI trigger set at t={now:.2f}s, buffer={current_buffer_size}")

            # Read AI state once per frame
            with state.state_lock:
                ai_text = state.ai_text
                ai_norm_boxes = list(state.ai_boxes)
                last_safe = state.last_safe_text
                last_alert_t = state.last_alert_time

            alert_active = (
                "ALERT" in ai_text
                and (now - last_alert_t) <= (interval_setting + CONFIG.alert_linger_extra)
            )

            if alert_active:
                draw_alert_overlay(frame, ai_norm_boxes, alert_text=ai_text)

            frame_counter += 1
            if frame_counter % CONFIG.display_frame_interval == 0:
                frame_rgb = _render_display_frame(
                    frame, ai_text, alert_active, last_safe)
                with state.log_lock:
                    debug_text = "\n".join(state.debug_history)
                    log_text = "\n".join(state.log_history)
                yield frame_rgb, ai_text, log_text, debug_text

            if is_file:
                elapsed = time.time() - loop_start
                time.sleep(max(0.001, (1.0 / fps) - elapsed))

    finally:
        state.running = False
        cap.release()


def stop_monitoring() -> str:
    """Signal the monitoring loop to stop."""
    state.running = False
    return "Stopping..."


# --- Manual Analysis ---


def analyze_uploaded_file(file, prompt_override):
    """One-shot analysis of a user-uploaded image or video."""
    if not ai_engine.ready:
        return "Model not loaded.", None
    upload_path = resolve_upload_path(file)
    if not upload_path:
        return "No file uploaded.", None

    prompt = prompt_override if len(
        prompt_override.strip()) > 5 else DEFAULT_PROMPT

    try:
        if upload_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            return _analyze_uploaded_video(upload_path, prompt)
        return _analyze_uploaded_image(upload_path, prompt)
    except Exception as exc:
        return f"Error: {exc}", None


def _analyze_uploaded_video(upload_path: str, prompt: str):
    """Analyse a video file by sampling frames and running AI."""
    temp_path = os.path.join(tempfile.gettempdir(),
                             "sentinel_manual_analysis.mp4")
    if not safe_copy_video(upload_path, temp_path):
        return "Error: File locked by system. Try again.", None

    cap = cv2.VideoCapture(temp_path)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = set(
            np.linspace(0, max(total_frames - 2, 0),
                        CONFIG.sample_frames, dtype=int)
        )

        frames: list[Image.Image] = []
        current = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if current in indices:
                frames.append(Image.fromarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            current += 1
    finally:
        cap.release()

    result = ai_engine.analyze_video_clip(frames, prompt, system_prompt=SYSTEM_PROMPT)
    return _normalize_ai_response(result), None


def _analyze_uploaded_image(upload_path: str, prompt: str):
    """Analyse a single uploaded image."""
    img = Image.open(upload_path)
    result = ai_engine.analyze_single_image(img, prompt)
    return _normalize_ai_response(result), img


def update_prompt_state(new_prompt: str) -> None:
    """Update the live monitoring prompt."""
    state.prompt = new_prompt

# --- Gradio UI ---


with gr.Blocks(title="Sentinel AI - Video Context") as demo:
    gr.Markdown("# 🛡️ Qwen Sentinel - Sliding Window Video Analytics")

    # Model Loading Panel
    with gr.Row(variant="panel"):
        with gr.Column(scale=2):
            model_selector = gr.Dropdown(
                choices=list(DEFAULT_MODELS.keys()),
                label="Select AI Model",
                value="Qwen2.5-VL-3B-Instruct",
            )
        with gr.Column(scale=1):
            quant_check = gr.Checkbox(
                label="Use 4-bit Quantization", value=True)
        with gr.Column(scale=1):
            load_btn = gr.Button("⬇️ Load Model", variant="secondary")
        with gr.Column(scale=2):
            load_status = gr.Textbox(
                label="Status", value="Please load model.", interactive=False,
            )

    load_btn.click(
        apply_model_settings,
        inputs=[model_selector, quant_check],
        outputs=[load_status],
    )

    with gr.Tabs():
        # --- Live Monitor Tab ---
        with gr.Tab("Live Context Monitor"):
            with gr.Row():
                with gr.Column(scale=3):
                    live_display = gr.Image(
                        label="Live Surveillance Feed", streaming=True)

                with gr.Column(scale=1):
                    source_type = gr.Radio(
                        ["Camera", "Video File"], label="Source", value="Video File",
                    )
                    cam_id = gr.Textbox(value="0", label="Camera ID")
                    vid_file = gr.File(
                        label="Test Video File",
                        file_types=[".mp4", ".avi", ".mov", ".mkv"],
                        type="filepath",
                    )
                    interval = gr.Slider(
                        1.0, 10.0, 3.0, step=0.5, label="AI Check Interval (Sec)",
                    )
                    sample_strategy = gr.Dropdown(
                        ["Uniform", "Recent Focus", "Motion Focus"],
                        value="Motion Focus",
                        label="Sampling Strategy",
                    )

                    with gr.Accordion("⚙️ Buffer & Segment", open=False):
                        buffer_seconds = gr.Slider(
                            2, 30, value=8, step=1,
                            label="Buffer / Segment Length (sec)",
                            info="Seconds of video per AI analysis segment",
                        )
                        sample_frames_slider = gr.Slider(
                            4, 16, value=8, step=1,
                            label="Frames Sampled per Segment",
                            info="Number of frames sent to the AI model per segment",
                        )

                    use_video_mode = gr.Checkbox(
                        label="Use video mode (experimental)", value=False,
                    )
                    enable_bbox_check = gr.Checkbox(
                        label="Enable bbox localization (extra VRAM)",
                        value=False,
                    )
                    debug_toggle = gr.Checkbox(
                        label="Debug AI sampling", value=False)
                    debug_save = gr.Checkbox(
                        label="Save sampled frames", value=False)

                    with gr.Accordion("📝 Prompt", open=False):
                        prompt_box = gr.TextArea(value=DEFAULT_PROMPT, lines=5)
                        upd_prompt = gr.Button("Update Prompt")

                    start_btn = gr.Button("▶ START", variant="primary")
                    stop_btn = gr.Button("⏹ STOP", variant="stop")
                    status_box = gr.Textbox(label="Last Detection")
                    log_box = gr.TextArea(label="Event Log", lines=8)
                    debug_box = gr.TextArea(
                        label="Debug Log (Sampling)", lines=8, interactive=False,
                    )

            upd_prompt.click(update_prompt_state, prompt_box, None)
            start_btn.click(
                monitor_stream,
                inputs=[
                    source_type, cam_id, vid_file, interval, prompt_box,
                    debug_toggle, debug_save, sample_strategy, use_video_mode,
                    buffer_seconds, sample_frames_slider, enable_bbox_check,
                ],
                outputs=[live_display, status_box, log_box, debug_box],
            )
            stop_btn.click(stop_monitoring, outputs=[status_box])

        # --- Manual Analysis Tab ---
        with gr.Tab("Manual Analysis"):
            u_file = gr.File(label="Upload Image or Video")
            u_prompt = gr.TextArea(label="Prompt", value=DEFAULT_PROMPT)
            u_btn = gr.Button("Analyze")
            u_out_txt = gr.Markdown()
            u_out_img = gr.Image()

            u_btn.click(
                analyze_uploaded_file, [u_file, u_prompt], [
                    u_out_txt, u_out_img],
            )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", share=False)
