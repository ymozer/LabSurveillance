import gradio as gr
import cv2
import torch
import os
import time
import json
import tempfile
import logging
import threading
import queue
import numpy as np
from datetime import datetime
from PIL import Image
from collections import deque, defaultdict
from ultralytics import YOLO
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    AutoModelForImageTextToText,
)
from qwen_vl_utils import process_vision_info

# --- Configuration ---
LOG_DIR = "security_events"
CACHE_DIR = os.path.join(os.getcwd(), ".hf_cache")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"
os.environ["TORCH_FORCE_SDPA"] = "1"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# COCO Class Indices relevant to Lab
# 0: Person, 39: Bottle, 41: Cup, 42: Fork, 43: Knife, 44: Spoon, 45: Bowl
# 62: TV (Monitor), 63: Laptop, 64: Mouse, 65: Remote, 66: Keyboard, 67: Cell Phone
RELEVANT_CLASSES = [0, 39, 41, 62, 63, 64, 66, 67]
EQUIPMENT_CLASSES = [62, 63, 64, 66]  # TV, Laptop, Mouse, Keyboard
FOOD_CLASSES = [39, 41, 42, 43, 44, 45]

MODEL_OPTIONS = {
    "Qwen2.5-VL-3B (Fastest)": "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen2.5-VL-7B (Balanced)": "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen3-VL-8B (Smartest)": "Qwen/Qwen3-VL-8B-Instruct",
}

# --- Shared State for Async Processing ---


class SharedState:
    # Stores last 30 raw frames (1-2 seconds of video)
    frame_queue = deque(maxlen=30)
    event_queue = queue.Queue()     # Events waiting for Qwen analysis
    current_alerts = []             # Active alerts to display on UI
    yolo_model = YOLO(None)  # Placeholder, will be loaded on demand
    qwen_model = None
    processor = None
    is_analyzing = False            # Mutex flag


state = SharedState()

# --- Model Loading ---


def load_models(qwen_model_name):
    # 1. Load YOLO (Fast Tracker)
    if state.yolo_model is None:
        logger.info("Loading YOLOv11...")
        state.yolo_model = YOLO("yolo11n.pt")  # Nano model for speed

    # 2. Load Qwen (The "Brain")
    target_model = MODEL_OPTIONS[qwen_model_name]
    if state.qwen_model is None or state.qwen_model.name_or_path != target_model:
        logger.info(f"Loading Qwen: {target_model}...")

        # Free memory
        if state.qwen_model is not None:
            del state.qwen_model
            del state.processor
            torch.cuda.empty_cache()

        try:
            if "Qwen2.5" in target_model:
                cls = Qwen2_5_VLForConditionalGeneration
            elif "Qwen2" in target_model:
                cls = Qwen2VLForConditionalGeneration
            else:
                cls = AutoModelForImageTextToText

            state.qwen_model = cls.from_pretrained(
                target_model,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                cache_dir=CACHE_DIR
            )
            state.processor = AutoProcessor.from_pretrained(
                target_model, cache_dir=CACHE_DIR)
            logger.info("Qwen Loaded.")
        except Exception as e:
            logger.error(f"Failed to load Qwen: {e}")

# --- Background Analyzer Worker ---


def analysis_worker():
    """
    Runs in a separate thread. Pops events, crops images, asks Qwen.
    """
    while True:
        try:
            # Wait for an event (Blocking)
            event = state.event_queue.get(timeout=1)

            # Event Structure: {'type': 'STEALING', 'crop_images': [PIL_Images], 'track_id': 5}

            if state.qwen_model is None:
                state.event_queue.task_done()
                continue

            state.is_analyzing = True
            logger.info(
                f"Analyzing Event: {event['type']} on ID {event['track_id']}")

            # 1. Construct Prompt based on Event Type
            prompt = ""
            if event['type'] == "CARRYING_CHECK":
                prompt = (
                    "You are a security officer. Analyze this walking person. "
                    "Are they holding or carrying any computer equipment (keyboard, mouse, monitor) "
                    "or are they holding a mobile phone? "
                    "If they are holding a phone, report it immediately. "
                    "Output JSON: {'detected': true/false, 'details': 'Carrying Keyboard/Mouse/Monitor OR Holding Phone'}"
                )
            elif event['type'] == "PHONE_CHECK":
                prompt = (
                    "You are a security officer. Analyze this person. "
                    "Are they using a mobile phone? "
                    "Output JSON: {'detected': true/false, 'details': 'Phone usage details'}"
                )
            elif event['type'] == "POSSIBLE_THEFT":
                prompt = (
                    "You are a security officer. Analyze this desk area. "
                    "A keyboard or mouse seems to have disappeared. "
                    "Is the person DISCONNECTING or HIDING the equipment? "
                    "Output JSON: {'detected': true/false, 'details': 'reasoning'}"
                )
            elif event['type'] == "EATING_DRINKING":
                prompt = (
                    "You are a security officer. Analyze this person. "
                    "Are they holding food, a bottle, or a cup to their mouth? "
                    "Output JSON: {'detected': true/false, 'details': 'what are they eating/drinking'}"
                )
            elif event['type'] == "SLEEPING":
                prompt = (
                    "You are a security officer. This person has been static for a long time. "
                    "Are they SLEEPING? Look for head on desk, closed eyes, or slumped posture. "
                    "Output JSON: {'detected': true/false, 'details': 'posture description'}"
                )

            # 2. Prepare Inputs
            # Save temp files for Qwen-VL-Utils
            temp_paths = []
            with tempfile.TemporaryDirectory() as temp_dir:
                for i, img in enumerate(event['crop_images']):
                    path = os.path.join(temp_dir, f"frame_{i}.jpg")
                    img.save(path)
                    temp_paths.append(path)

                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "video", "video": temp_paths, "fps": 1.0},
                        {"type": "text", "text": prompt},
                    ],
                }]

                # 3. Inference
                text = state.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = state.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(state.qwen_model.device)

                generated_ids = state.qwen_model.generate(
                    **inputs, max_new_tokens=128)
                output_text = state.processor.batch_decode(
                    generated_ids, skip_special_tokens=True)[0]

            # 4. Parse Result
            try:
                json_part = output_text.split("assistant")[-1].strip()
                start, end = json_part.find('{'), json_part.rfind('}')
                if start != -1:
                    result = json.loads(json_part[start:end+1])
                    if result.get("detected"):
                        alert_msg = f"🚨 {event['type']} (ID {event['track_id']}): {result.get('details')}"
                        state.current_alerts.insert(0, alert_msg)
                        logger.info(alert_msg)

                        # Save Evidence
                        ts = datetime.now().strftime("%H%M%S")
                        event['crop_images'][-1].save(
                            f"{LOG_DIR}/alert_{event['type']}_{ts}.jpg")
            except Exception as e:
                logger.error(f"Parsing error: {e}")

            state.is_analyzing = False
            state.event_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Worker Error: {e}")
            state.is_analyzing = False


# Start the worker thread
thread = threading.Thread(target=analysis_worker, daemon=True)
thread.start()

# --- Main Logic Class ---


class LabMonitor:
    def __init__(self):
        self.track_history = defaultdict(
            lambda: deque(maxlen=30))  # Store positions
        self.equipment_registry = {}  # {track_id: (last_seen_time, last_box)}
        self.cooldowns = {}  # {track_id_event_type: last_trigger_time}

    def is_occluded(self, box, image_shape):
        return False

    def get_context_crop(self, frame, person_box):
        """
        Instead of the whole frame, crop the person + immediate surroundings.
        This allows Qwen to see 'small' things like mice/food.
        """
        h, w, _ = frame.shape
        x1, y1, x2, y2 = person_box

        # Expand box by 50% to include desk/carrying context
        cw = x2 - x1
        ch = y2 - y1
        nx1 = max(0, x1 - cw * 0.5)
        ny1 = max(0, y1 - ch * 0.2)
        nx2 = min(w, x2 + cw * 0.5)
        ny2 = min(h, y2 + ch * 0.5)

        crop = frame[int(ny1):int(ny2), int(nx1):int(nx2)]
        return Image.fromarray(crop)

    def process(self, frame, model_name):
        load_models(model_name)

        # 1. YOLO Tracking
        results = state.yolo_model.track(
            frame, persist=True, classes=RELEVANT_CLASSES, verbose=False)

        annotated_frame = results[0].plot()
        boxes = results[0].boxes

        if boxes.id is None:
            return annotated_frame, state.current_alerts

        # Convert to easy format
        detections = []
        for box, cls, track_id in zip(boxes.xyxy, boxes.cls, boxes.id):
            detections.append({
                "box": box.cpu().numpy(),
                "cls": int(cls),
                "id": int(track_id)
            })

        current_time = time.time()

        # Organize by type
        persons = [d for d in detections if d['cls'] == 0]
        equipment = [d for d in detections if d['cls'] in EQUIPMENT_CLASSES]
        food_items = [d for d in detections if d['cls'] in FOOD_CLASSES]
        phones = [d for d in detections if d['cls'] == 67]

        # Store raw frame for buffering
        state.frame_queue.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # --- Heuristic Checks ---

        for p in persons:
            pid = p['id']
            px1, py1, px2, py2 = p['box']
            p_center = ((px1+px2)/2, (py1+py2)/2)

            self.track_history[pid].append(p_center)

            # Calculate Movement (Velocity) for Stealing/Carrying check
            is_moving = False
            if len(self.track_history[pid]) >= 10:
                recent_path = list(self.track_history[pid])
                dist = np.linalg.norm(
                    np.array(recent_path[-1]) - np.array(recent_path[-10]))

                # If moved > 50 pixels in last 10 frames (approx 0.3s) -> Moving
                if dist > 50:
                    is_moving = True
                    trigger_key = f"{pid}_carrying"
                    if current_time - self.cooldowns.get(trigger_key, 0) > 5.0:
                        # TRIGGER QWEN: "Is this moving person carrying a keyboard/mouse/monitor?"
                        # This catches stealing even if YOLO misses the object due to occlusion
                        crops = [self.get_context_crop(
                            f, p['box']) for f in list(state.frame_queue)[::3]]
                        if len(crops) > 0:
                            state.event_queue.put({
                                "type": "CARRYING_CHECK",
                                "track_id": pid,
                                "crop_images": crops
                            })
                            self.cooldowns[trigger_key] = current_time

            # --- D. PHONE DETECTION (Explicit YOLO) ---
            # If YOLO sees a phone overlapping a person
            for ph in phones:
                phx1, phy1, phx2, phy2 = ph['box']
                # Simple intersection check
                overlap = max(0, min(px2, phx2) - max(px1, phx1)) * \
                    max(0, min(py2, phy2) - max(py1, phy1))
                if overlap > 0:
                    trigger_key = f"{pid}_phone_detected"
                    if current_time - self.cooldowns.get(trigger_key, 0) > 5.0:
                        crops = [self.get_context_crop(
                            f, p['box']) for f in list(state.frame_queue)[::3]]
                        state.event_queue.put({
                            "type": "PHONE_CHECK",
                            "track_id": pid,
                            "crop_images": crops
                        })
                        self.cooldowns[trigger_key] = current_time

            # --- A. SLEEPING DETECTION (Static Head) ---
            if len(self.track_history[pid]) >= 20 and not is_moving:
                recent_path = list(self.track_history[pid])
                dist_static = np.linalg.norm(
                    np.array(recent_path[-1]) - np.array(recent_path[0]))

                if dist_static < 10:
                    trigger_key = f"{pid}_sleeping"
                    if current_time - self.cooldowns.get(trigger_key, 0) > 10.0:
                        crops = [self.get_context_crop(
                            f, p['box']) for f in list(state.frame_queue)[::5]]
                        if len(crops) > 0:
                            state.event_queue.put({
                                "type": "SLEEPING",
                                "track_id": pid,
                                "crop_images": crops
                            })
                            self.cooldowns[trigger_key] = current_time

            # --- B. EATING/DRINKING DETECTION ---
            head_region_y = py1 + (py2 - py1) * 0.3
            for f in food_items:
                fx1, fy1, fx2, fy2 = f['box']
                f_center_y = (fy1 + fy2) / 2

                if (fx1 < px2 and fx2 > px1) and (f_center_y < head_region_y + 50):
                    trigger_key = f"{pid}_eating"
                    if current_time - self.cooldowns.get(trigger_key, 0) > 10.0:
                        crops = [self.get_context_crop(
                            f, p['box']) for f in list(state.frame_queue)[::3]]
                        state.event_queue.put({
                            "type": "EATING_DRINKING",
                            "track_id": pid,
                            "crop_images": crops
                        })
                        self.cooldowns[trigger_key] = current_time

            # --- C. STEALING DETECTION (Theft of Equipment - Disappearance) ---
            # Update registry
            for eq in equipment:
                self.equipment_registry[eq['id']] = (current_time, eq['box'])

            for eq in equipment:
                ex1, ey1, ex2, ey2 = eq['box']
                overlap = max(0, min(px2, ex2) - max(px1, ex1)) * \
                    max(0, min(py2, ey2) - max(py1, ey1))
                if overlap > 0:
                    trigger_key = f"{pid}_theft_check"
                    if current_time - self.cooldowns.get(trigger_key, 0) > 8.0:
                        crops = [self.get_context_crop(
                            f, p['box']) for f in list(state.frame_queue)[::3]]
                        state.event_queue.put({
                            "type": "POSSIBLE_THEFT",
                            "track_id": pid,
                            "crop_images": crops
                        })
                        self.cooldowns[trigger_key] = current_time

        return annotated_frame, state.current_alerts


# --- UI Functions ---
monitor = LabMonitor()


def run_pipeline(image, model_key):
    if image is None:
        return None, []
    # OpenCV expects BGR
    frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Process
    annotated_bgr, alerts = monitor.process(frame_bgr, model_key)

    # Return RGB for Gradio
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), "\n".join(alerts[:5])


# --- Gradio Interface ---
with gr.Blocks(title="Modern Lab Guardian", theme=gr.themes.Glass()) as demo:
    gr.Markdown(
        """
        # 🛡️ Modern Lab Guardian (Async Event-Driven)
        **Architecture:** YOLOv11 (Tracker) ➡️ Heuristic State Machine ➡️ Async Qwen-VL (Verification)
        
        *Detects: Sleeping, Eating/Drinking, Theft attempts (Carrying equipment), Phone Usage.*
        *Handles: Occlusions by focusing on available 'Upper Body' context.*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                choices=list(MODEL_OPTIONS.keys()),
                value="Qwen2.5-VL-3B (Fastest)",
                label="Verification Model"
            )
            log_box = gr.Textbox(label="Security Alerts (Real-time)", lines=10)

        with gr.Column(scale=3):
            webcam = gr.Image(
                sources=["webcam"], streaming=True, type="numpy", label="Lab Feed")
            overlay = gr.Image(label="System HUD")

    webcam.stream(
        fn=run_pipeline,
        inputs=[webcam, model_selector],
        outputs=[overlay, log_box],
        show_progress="hidden"
    )

if __name__ == "__main__":
    demo.launch()
