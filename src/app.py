import gradio as gr
import cv2
import time
import re
import os
import threading
from PIL import Image
from backend import SurveillanceAI, DEFAULT_MODELS
from datetime import datetime

ai_engine = SurveillanceAI()

class SystemState:
    def __init__(self):
        self.running = False
        self.frame_for_ai = None  
        self.frame_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.ai_text = "System Standby. Load a model to begin."
        self.ai_boxes = [] 
        self.check_interval = 5 
        self.log_history = []

        self.alert_display_seconds = 3.0
        self.last_alert_time = 0.0
        self.last_safe_text = "Status: Safe"

state = SystemState()

if not os.path.exists("evidence_snapshots"):
    os.makedirs("evidence_snapshots")

def get_surveillance_prompt():
    return (
        "Analyze the entire classroom scene. Identify all individuals present. "
        "Definitions: "
        "- Teacher: Person standing at the front, near the whiteboard, or walking between rows supervising. "
        "- Student: People sitting at desks or working at computers. "
        "Task: Scan everyone for prohibited behaviors: "
        "1. Eating/Drinking 2. Fighting 3. Sleeping 4. Unauthorized phone usage. "
        "INSTRUCTIONS:\n"
        "1. If SAFE, respond only: 'Status: Safe'.\n"
        "2. If VIOLATION, respond: 'ALERT: [Role] [Issue]'. "
        "AND provide the bounding box of the person involved using the format: "
        "[xmin, ymin, xmax, ymax] based on a 1000x1000 grid.\n"
        "Example: ALERT: Student sleeping [450, 200, 600, 500]"
    )

def extract_boxes(text_response, img_width, img_height):
    found_boxes = []
    matches = re.findall(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', text_response)
    for box in matches:
        try:
            x1_n, y1_n, x2_n, y2_n = map(int, box)
            x1 = int(x1_n / 1000 * img_width)
            y1 = int(y1_n / 1000 * img_height)
            x2 = int(x2_n / 1000 * img_width)
            y2 = int(y2_n / 1000 * img_height)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_width, x2), min(img_height, y2)
            found_boxes.append((x1, y1, x2, y2))
        except:
            continue
    return found_boxes

def ai_worker_loop():
    print("AI Worker Thread: Started")
    last_check_time = 0
    
    while state.running:
        current_time = time.time()
        
        if (current_time - last_check_time > state.check_interval) and \
           (state.frame_for_ai is not None) and \
           ai_engine.ready:
            
            try:
                with state.frame_lock:
                    frame_rgb = cv2.cvtColor(state.frame_for_ai, cv2.COLOR_BGR2RGB)
                    height, width, _ = state.frame_for_ai.shape

                pil_image = Image.fromarray(frame_rgb)
                
                result_text = ai_engine.analyze_frame_pil(pil_image, get_surveillance_prompt())
                
                new_boxes = extract_boxes(result_text, width, height)
                timestamp = datetime.now().strftime("%H:%M:%S")
                with state.state_lock:
                    state.ai_text = f"{result_text} (Last: {timestamp})"
                    state.ai_boxes = new_boxes
                    if "ALERT" in result_text:
                        state.last_alert_time = time.time()
                    else:
                        state.last_safe_text = state.ai_text
                
                if "ALERT" in result_text:
                    log_entry = f"[{timestamp}] 🚨 {result_text}"
                    evidence_frame = state.frame_for_ai.copy()
                    for (x1, y1, x2, y2) in new_boxes:
                        cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    filename = f"evidence_snapshots/alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, evidence_frame)
                else:
                    log_entry = f"[{timestamp}] ✅ {result_text}"
                
                state.log_history.insert(0, log_entry)
                if len(state.log_history) > 50: state.log_history.pop()

                last_check_time = time.time()
                
            except Exception as e:
                print(f"AI Worker Error: {e}")
                with state.state_lock:
                    state.ai_text = f"Error: {str(e)}"
        
        elif not ai_engine.ready and state.running:
             with state.state_lock:
                 state.ai_text = "⚠️ AI Model not loaded. Please load a model in settings."

        time.sleep(0.1)
    print("AI Worker Thread: Stopped")

def apply_model_settings(model_choice, use_4bit):
    if not model_choice:
        return "⚠️ Please select a model first."
    
    was_running = state.running
    state.running = False
    time.sleep(1)
    
    status = ai_engine.load_model(model_choice, use_4bit)
    
    if was_running:
        state.running = True
        if threading.active_count() < 2: 
             t = threading.Thread(target=ai_worker_loop, daemon=True)
             t.start()
             
    return status

def monitor_stream(cam_id, interval_setting):
    state.check_interval = float(interval_setting)
    
    if not ai_engine.ready:
        yield None, "❌ ERROR: No AI Model Loaded. Please use 'System Setup' above.", ""
        return

    cam_source = int(cam_id) if str(cam_id).strip().isdigit() else cam_id
    cap = cv2.VideoCapture(cam_source)
    
    if not cap.isOpened():
        yield None, "Error: Cannot open camera", ""
        return

    state.running = True
    state.frame_for_ai = None
    state.log_history = []
    
    worker_thread = threading.Thread(target=ai_worker_loop, daemon=True)
    worker_thread.start()

    try:
        while state.running:
            ret, frame = cap.read()
            if not ret: break
            
            with state.frame_lock:
                state.frame_for_ai = frame.copy()
            
            with state.state_lock:
                ai_text = state.ai_text
                ai_boxes = list(state.ai_boxes)
                last_safe_text = state.last_safe_text
                last_alert_time = state.last_alert_time
                alert_ttl = float(state.alert_display_seconds)

            alert_active = (
                "ALERT" in ai_text
                and last_alert_time > 0
                and (time.time() - last_alert_time) <= alert_ttl
            )

            if alert_active:
                for (x1, y1, x2, y2) in ai_boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "ALERT", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            cv2.rectangle(frame_rgb, (0, 0), (900, 40), (0,0,0), -1)
            display_text = ai_text if alert_active else (last_safe_text or ai_text)
            color = (255, 50, 50) if alert_active else (50, 255, 50)
            cv2.putText(frame_rgb, display_text[:90], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            yield frame_rgb, ai_text, "\n".join(state.log_history)
            time.sleep(0.01)
            
    finally:
        state.running = False
        cap.release()

def stop_monitoring():
    state.running = False
    return "Stopping..."

def analyze_file(file, prompt):
    if not ai_engine.ready: return "Error: Model not loaded."
    if file is None: return "Please upload a file."
    media_type = "video" if file.endswith(('.mp4', '.avi', '.mov')) else "image"
    return ai_engine.analyze_media(file, prompt, media_type)

with gr.Blocks(title="Sentinel AI - Multi-Model") as demo:
    gr.Markdown("# 🛡️ Qwen Sentinel - Multi-Model Lab")
    
    with gr.Row(variant="panel"):
        gr.Markdown("### ⚙️ System Setup")
        with gr.Column(scale=2):
            model_selector = gr.Dropdown(
                choices=list(DEFAULT_MODELS.keys()), 
                label="Select AI Model", 
                value="Qwen3-VL-2B-Instruct"
            )
        with gr.Column(scale=1):
            quant_check = gr.Checkbox(label="Use 4-bit Quantization", value=True, info="Recommended for <24GB VRAM")
        with gr.Column(scale=1):
            load_btn = gr.Button("⬇️ Load Model", variant="secondary")
        with gr.Column(scale=2):
            load_status = gr.Textbox(label="Status", value="Please load a model.", interactive=False)

    load_btn.click(apply_model_settings, inputs=[model_selector, quant_check], outputs=[load_status])

    with gr.Tabs():
        with gr.Tab("Live Surveillance"):
            with gr.Row():
                with gr.Column(scale=3):
                    live_display = gr.Image(label="Live Feed", streaming=True)
                
                with gr.Column(scale=1):
                    cam_input = gr.Textbox(value="0", label="Camera Source (0 or RTSP)")
                    interval_slider = gr.Slider(1, 30, 5, step=1, label="AI Check Interval (s)")
                    
                    with gr.Row():
                        start_btn = gr.Button("▶ Start Monitor", variant="primary")
                        stop_btn = gr.Button("⏹ Stop", variant="stop")
                    
                    current_status = gr.Textbox(label="Last AI Result")
                    log_box = gr.TextArea(label="Activity Log", lines=10)

            start_btn.click(monitor_stream, inputs=[cam_input, interval_slider], outputs=[live_display, current_status, log_box])
            stop_btn.click(stop_monitoring, outputs=[current_status])

        with gr.Tab("Investigate Footage"):
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(label="Upload Evidence")
                    query_input = gr.Textbox(label="Question", value="Describe the scene.")
                    analyze_btn = gr.Button("Analyze", variant="primary")
                with gr.Column():
                    ai_output = gr.Markdown(label="Analysis Result")
            
            analyze_btn.click(analyze_file, inputs=[file_input, query_input], outputs=[ai_output])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", share=False)