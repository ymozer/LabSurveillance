import gradio as gr
import cv2
import time
import re
import os
import threading
import shutil
import numpy as np
from collections import deque
from PIL import Image, ImageDraw
from backend import SurveillanceAI, DEFAULT_MODELS
from datetime import datetime

# --- Configuration ---
BUFFER_SECONDS = 4       # How many seconds of history to keep
SAMPLE_FRAMES = 8        # How many frames to send to AI
CHECK_INTERVAL = 2.0     # Run AI every X seconds
RESIZE_HEIGHT = 480      # Resize frames before sending to AI

ai_engine = SurveillanceAI()

DEFAULT_PROMPT = (
    "Analyze this video clip from a lab surveillance camera. "
    "Check for PROHIBITED activities that require motion context:\n"
    "1. Running (moving fast across frames)\n"
    "2. Fighting or Aggressive Physical Contact\n"
    "3. Eating/Drinking\n"
    "4. Tampering with equipment\n"
    "5. Sleeping\n\n"
    "OUTPUT FORMAT:\n"
    "If SAFE: 'Status: Safe'\n"
    "If DANGEROUS: 'ALERT: [Description of activity]'\n"
    "If an alert is found, provide the bounding box of the person involved in the LAST frame: [xmin, ymin, xmax, ymax] (0-1000 scale)."
)


class SystemState:
    def __init__(self):
        self.running = False
        self.frame_buffer = deque(maxlen=int(BUFFER_SECONDS * 30))
        self.buffer_lock = threading.Lock()
        self.trigger_ai_event = threading.Event()
        self.ai_text = "System Standby."
        self.ai_boxes = []
        self.log_history = []
        self.debug_history = []
        self.debug_enabled = True
        self.debug_save_clips = False
        self.sample_strategy = "Uniform"
        self.source_type = "Camera"
        self.use_video_mode = False
        self.sample_frames = SAMPLE_FRAMES
        self.buffer_seconds = BUFFER_SECONDS
        self.prompt = DEFAULT_PROMPT
        self.state_lock = threading.Lock()
        self.last_alert_time = 0.0
        self.last_safe_text = "Status: Safe"


state = SystemState()

EVIDENCE_DIR = os.path.join(os.path.dirname(
    __file__), "..", "evidence_snapshots")
EVIDENCE_DIR = os.path.abspath(EVIDENCE_DIR)
if not os.path.exists(EVIDENCE_DIR):
    os.makedirs(EVIDENCE_DIR, exist_ok=True)

# --- Helper Functions ---


def safe_copy_video(src, dst, retries=5, delay=0.5):
    """
    Safely copies a file on Windows, waiting if it's currently locked 
    by the upload process.
    """
    for i in range(retries):
        try:
            # Short sleep to allow file handle release
            time.sleep(delay)
            shutil.copy(src, dst)
            return True
        except PermissionError:
            print(f"⚠️ File locked. Retrying copy ({i+1}/{retries})...")
            time.sleep(1.0)
        except Exception as e:
            print(f"❌ Copy failed: {e}")
            return False
    return False


def open_video_capture(upload_path, local_video_path):
    """
    Tries to open a video in a resilient way:
    1) If a safe copy succeeds, open the local copy.
    2) Otherwise, try opening the original upload path directly.
    Returns (cap, used_path).
    """
    used_path = None

    if upload_path and os.path.exists(upload_path):
        if safe_copy_video(upload_path, local_video_path):
            used_path = local_video_path
        else:
            used_path = upload_path

    if not used_path:
        return None, None

    cap = cv2.VideoCapture(used_path)
    if not cap or not cap.isOpened():
        if cap:
            cap.release()
        return None, used_path

    return cap, used_path


def resolve_upload_path(uploaded):
    if uploaded is None:
        return None
    if isinstance(uploaded, str):
        return uploaded
    if isinstance(uploaded, dict):
        return uploaded.get("path") or uploaded.get("name")
    return getattr(uploaded, "name", None) or getattr(uploaded, "path", None)


def extract_boxes(text_response, img_width, img_height):
    found_boxes = []
    matches = re.findall(
        r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', text_response)
    for box in matches:
        try:
            x1_n, y1_n, x2_n, y2_n = map(int, box)
            x1 = int(x1_n / 1000 * img_width)
            y1 = int(y1_n / 1000 * img_height)
            x2 = int(x2_n / 1000 * img_width)
            y2 = int(y2_n / 1000 * img_height)
            found_boxes.append((x1, y1, x2, y2))
        except:
            continue
    return found_boxes


def detect_person_boxes(frame_bgr):
    """Fallback person detector using OpenCV HOG."""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]
    max_w = 640
    scale = 1.0
    if w > max_w:
        scale = max_w / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    rects, _ = hog.detectMultiScale(
        frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
    boxes = []
    for (x, y, bw, bh) in rects:
        x1 = int(x / scale)
        y1 = int(y / scale)
        x2 = int((x + bw) / scale)
        y2 = int((y + bh) / scale)
        boxes.append((x1, y1, x2, y2))
    return boxes


def get_sample_indices(total_frames, target_count, strategy, buffer_snapshot=None):
    if total_frames <= 0:
        return []
    if total_frames < target_count:
        return list(range(total_frames))

    if strategy == "Recent Focus":
        recent_window = max(target_count, int(total_frames * 0.6))
        start = max(0, total_frames - recent_window)
        return np.linspace(start, total_frames - 1, target_count, dtype=int).tolist()

    if strategy == "Motion Focus" and buffer_snapshot is not None:
        scores = []
        prev = None
        for idx, frame in enumerate(buffer_snapshot):
            if prev is None:
                prev = frame
                continue
            curr_small = cv2.resize(frame, (64, 36))
            prev_small = cv2.resize(prev, (64, 36))
            diff = cv2.absdiff(curr_small, prev_small)
            score = float(np.sum(diff))
            scores.append((idx, score))
            prev = frame

        scores.sort(key=lambda x: x[1], reverse=True)
        pick_count = max(1, target_count - 1)
        picked = [idx for idx, _ in scores[:pick_count]]
        picked.append(total_frames - 1)
        picked = sorted(set(picked))

        if len(picked) < target_count:
            needed = target_count - len(picked)
            fallback = np.linspace(0, total_frames - 1,
                                   target_count, dtype=int).tolist()
            for i in fallback:
                if i not in picked:
                    picked.append(i)
                    needed -= 1
                    if needed == 0:
                        break
            picked = sorted(picked)

        return picked

    return np.linspace(0, total_frames - 1, target_count, dtype=int).tolist()


def sample_frames_from_buffer(buffer_snapshot, target_count=SAMPLE_FRAMES, strategy="Uniform", indices=None):
    total_frames = len(buffer_snapshot)
    if indices is None:
        indices = get_sample_indices(
            total_frames, target_count, strategy, buffer_snapshot)

    pil_frames = []
    for i in indices:
        frame = buffer_snapshot[i]
        h, w, _ = frame.shape
        scale = RESIZE_HEIGHT / h
        new_w = int(w * scale)
        frame_resized = cv2.resize(frame, (new_w, RESIZE_HEIGHT))
        pil_img = Image.fromarray(cv2.cvtColor(
            frame_resized, cv2.COLOR_BGR2RGB))
        pil_frames.append(pil_img)

    return pil_frames


def log_debug(message):
    if not state.debug_enabled:
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    state.debug_history.insert(0, f"[{timestamp}] {message}")
    if len(state.debug_history) > 200:
        state.debug_history.pop()


def build_prompt(base_prompt, strict=False):
    if not strict:
        return base_prompt
    return (
        base_prompt
        + "\n\nSTRICT MODE (video file):\n"
        + "If ANY prohibited activity appears in ANY frame, output ALERT. "
        + "Do not default to SAFE if uncertain. "
        + "If motion suggests running or aggressive contact, alert. "
        + "If a person appears to eat/drink, tamper with equipment, or sleep, alert. "
        + "Return exactly one line in the required format. "
        + "For ALERT, you MUST include a bounding box for the person in the LAST frame: [xmin, ymin, xmax, ymax] (0-1000 scale)."
    )


def _frame_hash(pil_img, size=16):
    gray = pil_img.convert("L").resize((size, size))
    arr = np.array(gray, dtype=np.uint8)
    avg = arr.mean()
    bits = arr > avg
    flat = bits.flatten()
    out = []
    for i in range(0, len(flat), 4):
        val = 0
        for j in range(4):
            val = (val << 1) | int(flat[i + j])
        out.append(f"{val:x}")
    return "".join(out)


def _save_contact_sheet(frames, out_path, cols=4, thumb_h=180):
    if not frames:
        return False
    thumbs = []
    for img in frames:
        w, h = img.size
        scale = thumb_h / h
        tw = int(w * scale)
        thumbs.append(img.resize((tw, thumb_h)))

    rows = (len(thumbs) + cols - 1) // cols
    row_widths = [0] * rows
    row_heights = [0] * rows
    for idx, t in enumerate(thumbs):
        r = idx // cols
        row_widths[r] += t.size[0]
        row_heights[r] = max(row_heights[r], t.size[1])

    sheet_w = max(row_widths)
    sheet_h = sum(row_heights)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (10, 10, 10))

    y = 0
    idx = 0
    for r in range(rows):
        x = 0
        for _ in range(cols):
            if idx >= len(thumbs):
                break
            t = thumbs[idx]
            sheet.paste(t, (x, y))
            x += t.size[0]
            idx += 1
        y += row_heights[r]

    sheet.save(out_path)
    return True

# --- AI Worker ---


def ai_worker_loop():
    print("AI Worker Thread: Started (Sliding Window Mode)")
    while True:
        is_set = state.trigger_ai_event.wait(timeout=1.0)

        if not state.running:
            time.sleep(1)
            continue

        if is_set:
            state.trigger_ai_event.clear()

            if not ai_engine.ready:
                with state.state_lock:
                    state.ai_text = "⚠️ AI Model not loaded."
                log_debug("AI trigger ignored: model not loaded")
                continue

            with state.buffer_lock:
                if len(state.frame_buffer) < 4:
                    log_debug(
                        f"AI trigger ignored: buffer too small ({len(state.frame_buffer)})")
                    continue
                snapshot = list(state.frame_buffer)

            try:
                # Debug: which indices are sampled from the sliding window
                total_frames = len(snapshot)
                target_count = state.sample_frames
                sampled_indices = get_sample_indices(
                    total_frames, target_count, state.sample_strategy, snapshot)

                clip_images = sample_frames_from_buffer(
                    snapshot,
                    target_count=target_count,
                    strategy=state.sample_strategy,
                    indices=sampled_indices)
                hashes = [_frame_hash(img) for img in clip_images]
                log_debug(
                    f"AI trigger: buffer={total_frames}, frames={target_count}, strategy={state.sample_strategy}, indices={sampled_indices}")
                log_debug(f"AI sample hashes: {hashes}")

                if state.debug_save_clips:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = f"evidence_snapshots/debug_clip_{ts}.jpg"
                    _save_contact_sheet(clip_images, out_path)
                    log_debug(f"Saved debug clip sheet: {out_path}")
                if state.source_type == "Video File" and state.use_video_mode:
                    result_text = ai_engine.analyze_video_clip_as_video(
                        clip_images, state.prompt, fps=2.0)
                else:
                    result_text = ai_engine.analyze_video_clip(
                        clip_images, state.prompt)
                if not result_text:
                    log_debug("AI output: <empty>")
                else:
                    log_debug(f"AI output: {result_text[:200]}")

                # Boxes logic
                last_frame_w, last_frame_h = clip_images[-1].size
                boxes_on_resized = extract_boxes(
                    result_text, last_frame_w, last_frame_h)
                if "ALERT" in result_text and not boxes_on_resized:
                    followup_prompt = (
                        "Return ONLY the bounding box for the person involved in the LAST frame, "
                        "format: [xmin, ymin, xmax, ymax] on 0-1000 scale."
                    )
                    try:
                        followup_text = ai_engine.analyze_video_clip(
                            clip_images, followup_prompt)
                        boxes_on_resized = extract_boxes(
                            followup_text, last_frame_w, last_frame_h)
                        log_debug(f"AI bbox followup: {followup_text[:200]}")
                    except Exception as e:
                        log_debug(f"AI bbox followup error: {e}")
                timestamp = datetime.now().strftime("%H:%M:%S")

                # Note: if no boxes were returned, UI will show alert without boxes.

                with state.state_lock:
                    state.ai_text = result_text
                    state.ai_boxes = []
                    for (rx1, ry1, rx2, ry2) in boxes_on_resized:
                        n_box = (rx1/last_frame_w, ry1/last_frame_h,
                                 rx2/last_frame_w, ry2/last_frame_h)
                        state.ai_boxes.append(n_box)

                    if "ALERT" in result_text:
                        state.last_alert_time = time.time()
                    else:
                        state.last_safe_text = result_text

                if "ALERT" in result_text:
                    log_entry = f"[{timestamp}] 🚨 {result_text}"
                    key_frame = snapshot[-1].copy()
                    kh, kw, _ = key_frame.shape
                    for (nx1, ny1, nx2, ny2) in state.ai_boxes:
                        cv2.rectangle(key_frame, (int(nx1*kw), int(ny1*kh)),
                                      (int(nx2*kw), int(ny2*kh)), (0, 0, 255), 3)

                    filename = os.path.join(
                        EVIDENCE_DIR,
                        f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    )
                    saved = cv2.imwrite(filename, key_frame)
                    if saved:
                        log_debug(f"Saved alert image: {filename}")
                    else:
                        log_debug(f"Failed to save alert image: {filename}")
                else:
                    log_entry = f"[{timestamp}] 👁️ {result_text[:50]}..."

                state.log_history.insert(0, log_entry)
                if len(state.log_history) > 100:
                    state.log_history.pop()

            except Exception as e:
                print(f"AI Analysis Error: {e}")
                log_debug(f"AI error: {e}")


threading.Thread(target=ai_worker_loop, name="AI_Worker", daemon=True).start()

# --- Main Logic ---


def apply_model_settings(model_choice, use_4bit):
    if not model_choice:
        return "⚠️ Please select a model."
    return ai_engine.load_model(model_choice, use_4bit)


def monitor_stream(source_type, cam_id, video_file, interval_setting, prompt_text, debug_enabled, debug_save, sample_strategy, use_video_mode):
    strict_mode = (source_type == "Video File")
    state.prompt = build_prompt(prompt_text, strict=strict_mode)
    state.debug_enabled = bool(debug_enabled)
    state.debug_save_clips = bool(debug_save)
    state.sample_strategy = sample_strategy or "Uniform"
    state.source_type = source_type
    state.use_video_mode = bool(use_video_mode)
    log_debug(f"Prompt strict mode: {strict_mode}")
    if source_type == "Video File":
        state.sample_frames = max(SAMPLE_FRAMES, 16)
        state.buffer_seconds = max(BUFFER_SECONDS, 8)
    else:
        state.sample_frames = SAMPLE_FRAMES
        state.buffer_seconds = BUFFER_SECONDS

    if not ai_engine.ready:
        yield None, "❌ ERROR: No AI Model Loaded.", "", ""
        return

    with state.buffer_lock:
        state.frame_buffer.clear()
    state.debug_history.clear()

    # --- Source Setup ---
    cap = None
    local_video_path = "temp_processing_video.mp4"

    if source_type == "Video File":
        upload_path = resolve_upload_path(video_file)
        if not upload_path:
            yield None, "❌ Error: Upload video.", "", ""
            return

        # Use Safe Copy to avoid PermissionError, fallback to direct open
        yield None, "⏳ Processing file...", "", ""
        cap, used_path = open_video_capture(upload_path, local_video_path)
        if not cap:
            yield None, f"❌ Error: Could not open video. Path: {upload_path}", "", ""
            return

        is_file = True
    else:
        src = int(cam_id) if str(cam_id).strip().isdigit() else cam_id
        cap = cv2.VideoCapture(src)
        is_file = False

    if not cap.isOpened():
        yield None, "Error: Cannot open source", "", ""
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0

    # Recreate buffer based on actual FPS for better sampling window
    with state.buffer_lock:
        state.frame_buffer = deque(maxlen=int(state.buffer_seconds * fps))

    state.running = True
    last_check_time = time.time()
    display_frame_skip = 0

    try:
        while state.running:
            loop_start = time.time()
            ret, frame = cap.read()

            if not ret:
                if is_file:
                    state.ai_text = "End of Video."
                    break
                else:
                    time.sleep(1)
                    continue

            with state.buffer_lock:
                state.frame_buffer.append(frame)

            now = time.time()
            if (now - last_check_time) > interval_setting:
                state.trigger_ai_event.set()
                last_check_time = now
                log_debug(f"AI trigger set at t={now:.2f}s")

            with state.state_lock:
                ai_text = state.ai_text
                ai_norm_boxes = list(state.ai_boxes)
                last_safe = state.last_safe_text
                last_alert_t = state.last_alert_time

            alert_active = ("ALERT" in ai_text and (
                now - last_alert_t) <= (interval_setting + 1.5))

            if alert_active:
                h, w, _ = frame.shape
                for (nx1, ny1, nx2, ny2) in ai_norm_boxes:
                    x1, y1 = int(nx1*w), int(ny1*h)
                    x2, y2 = int(nx2*w), int(ny2*h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    cv2.putText(frame, "ALERT", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            display_frame_skip += 1
            if display_frame_skip % 4 == 0:
                h, w = frame.shape[:2]
                if w > 1024:
                    scale = 1024 / w
                    frame = cv2.resize(frame, (1024, int(h*scale)))
                    h, w = frame.shape[:2]

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.rectangle(frame_rgb, (0, 0), (w, 50), (0, 0, 0), -1)
                display_text = ai_text if alert_active else (
                    last_safe or ai_text)
                color = (255, 50, 50) if alert_active else (50, 255, 50)
                cv2.putText(
                    frame_rgb, display_text[:90], (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                debug_text = "\n".join(state.debug_history)
                yield frame_rgb, ai_text, "\n".join(state.log_history), debug_text

            if is_file:
                elapsed = time.time() - loop_start
                delay = max(0.001, (1.0/fps) - elapsed)
                time.sleep(delay)

    finally:
        state.running = False
        if cap:
            cap.release()


def stop_monitoring():
    state.running = False
    return "Stopping..."


def analyze_uploaded_file(file, prompt_override):
    if not ai_engine.ready:
        return "Model not loaded.", None
    upload_path = resolve_upload_path(file)
    if not upload_path:
        return "No file uploaded.", None

    prompt = prompt_override if len(
        prompt_override.strip()) > 5 else DEFAULT_PROMPT

    try:
        if upload_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            temp_path = "temp_manual_analysis.mp4"
            if not safe_copy_video(upload_path, temp_path):
                return "Error: File locked by system. Try again.", None

            cap = cv2.VideoCapture(temp_path)
            frames = []
            try:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    indices = np.linspace(
                        0, total_frames-2, SAMPLE_FRAMES, dtype=int)
                else:
                    indices = [0, 5, 10]  # Fallback

                current = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if current in indices:
                        f_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(f_rgb))
                    current += 1
            finally:
                cap.release()  # Release IMMEDIATELY after reading

            result = ai_engine.analyze_video_clip(frames, prompt)
            return result, None
        else:
            img = Image.open(upload_path)
            result = ai_engine.analyze_single_image(img, prompt)
            return result, img

    except Exception as e:
        return f"Error: {e}", None


def update_prompt_state(new_prompt):
    state.prompt = new_prompt
    return "✅ Prompt updated."

# --- Gradio UI ---


with gr.Blocks(title="Sentinel AI - Video Context") as demo:
    gr.Markdown("# 🛡️ Qwen Sentinel - Sliding Window Video Analytics")

    with gr.Row(variant="panel"):
        with gr.Column(scale=2):
            model_selector = gr.Dropdown(
                choices=list(DEFAULT_MODELS.keys()),
                label="Select AI Model",
                value="Qwen2.5-VL-3B-Instruct"
            )
        with gr.Column(scale=1):
            quant_check = gr.Checkbox(
                label="Use 4-bit Quantization", value=True)
        with gr.Column(scale=1):
            load_btn = gr.Button("⬇️ Load Model", variant="secondary")
        with gr.Column(scale=2):
            load_status = gr.Textbox(
                label="Status", value="Please load model.", interactive=False)

    load_btn.click(apply_model_settings, inputs=[
                   model_selector, quant_check], outputs=[load_status])

    with gr.Tabs():
        with gr.Tab("Live Context Monitor"):
            with gr.Row():
                with gr.Column(scale=3):
                    live_display = gr.Image(
                        label="Live Surveillance Feed", streaming=True)

                with gr.Column(scale=1):
                    source_type = gr.Radio(
                        ["Camera", "Video File"], label="Source", value="Video File")
                    cam_id = gr.Textbox(value="0", label="Camera ID")
                    vid_file = gr.File(
                        label="Test Video File",
                        file_types=[".mp4", ".avi", ".mov", ".mkv"],
                        type="filepath"
                    )
                    interval = gr.Slider(
                        1.0, 10.0, 3.0, step=0.5, label="AI Check Interval (Sec)")
                    sample_strategy = gr.Dropdown(
                        ["Uniform", "Recent Focus", "Motion Focus"],
                        value="Motion Focus",
                        label="Sampling Strategy")
                    use_video_mode = gr.Checkbox(
                        label="Use video mode (experimental)", value=False)
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
                        label="Debug Log (Sampling)", lines=8, interactive=False)

            upd_prompt.click(update_prompt_state, prompt_box, None)
            start_btn.click(monitor_stream,
                            inputs=[source_type, cam_id,
                                    vid_file, interval, prompt_box, debug_toggle, debug_save, sample_strategy, use_video_mode],
                            outputs=[live_display, status_box, log_box, debug_box])
            stop_btn.click(stop_monitoring, outputs=[status_box])

        with gr.Tab("Manual Analysis"):
            u_file = gr.File(label="Upload Image or Video")
            u_prompt = gr.TextArea(label="Prompt", value=DEFAULT_PROMPT)
            u_btn = gr.Button("Analyze")
            u_out_txt = gr.Markdown()
            u_out_img = gr.Image()

            u_btn.click(analyze_uploaded_file, [
                        u_file, u_prompt], [u_out_txt, u_out_img])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", share=False)
