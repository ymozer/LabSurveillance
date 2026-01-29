import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig
from qwen_vl_utils import process_vision_info
import cv2
from PIL import Image
import threading
import time
import gc
import os
import json
import io
import base64
from huggingface_hub import hf_hub_download


def _configure_hf_downloads(project_root: str) -> str:
    """Best-effort Hugging Face Hub config for Windows reliability.

    Notes:
    - Large files can appear "stuck" near 99% while the hub finalizes the
      download (hash check, flush to disk, atomic rename). On Windows this step
      is frequently slowed by antivirus / filesystem drivers.
    - We set environment variables only if the user hasn't already set them.
    """

    cache_dir = os.environ.get("HF_HOME")
    if not cache_dir:
        cache_dir = os.path.join(project_root, ".hf_cache")
        os.environ.setdefault("HF_HOME", cache_dir)

    # Keep caches co-located (older libs may still consult these).
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_dir, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_dir, "transformers"))

    # Faster downloader when available (falls back silently if not installed).
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # Reduce spurious "hangs" from slow etag requests / final range fetch.
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "1800")

    # Windows users often don't have symlink privileges; avoid noisy warnings.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    os.makedirs(cache_dir, exist_ok=True)

    # huggingface_hub reads these at import time; update them at runtime too.
    try:
        from huggingface_hub import constants as hub_constants
        from huggingface_hub.utils._http import close_session

        hub_constants.HF_HUB_ETAG_TIMEOUT = int(os.environ.get("HF_HUB_ETAG_TIMEOUT", "60"))
        hub_constants.HF_HUB_DOWNLOAD_TIMEOUT = int(os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", "1800"))
        close_session()
    except Exception:
        pass

    return cache_dir

# Try importing llama-cpp for GGUF support
try:
    from llama_cpp import Llama, LlamaChatFormat
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False

DEFAULT_MODELS = {
    "Qwen3-VL-2B-Instruct":         { "type": "hf", "id": "Qwen/Qwen3-VL-2B-Instruct"},
    "Qwen3-VL-2B-Instruct-FP8":     { "type": "hf", "id": "Qwen/Qwen3-VL-2B-Instruct-FP8"},
    "Qwen3-VL-2B-Instruct-GGUF":    { "type": "gguf", "id": "Qwen/Qwen3-VL-2B-Instruct-GGUF"},
    "Qwen3-VL-4B-Instruct":         { "type": "hf", "id": "Qwen/Qwen3-VL-4B-Instruct"}, # 8 GB VRAM minimum
    "Qwen3-VL-4B-Instruct-FP8":     { "type": "hf", "id": "Qwen/Qwen3-VL-4B-Instruct-FP8"},
    "Qwen3-VL-4B-Instruct-GGUF":    { "type": "gguf", "id": "Qwen/Qwen3-VL-4B-Instruct-GGUF"},
    "Qwen3-VL-8B-Instruct":         { "type": "hf", "id": "Qwen/Qwen3-VL-8B-Instruct"}, # RECOMMENDED for ACC, 16 GB VRAM minimum
    "Qwen3-VL-8B-Instruct-FP8":     { "type": "hf", "id": "Qwen/Qwen3-VL-8B-Instruct-FP8"},
    "Qwen3-VL-8B-Instruct-GGUF":    { "type": "gguf", "id": "Qwen/Qwen3-VL-8B-Instruct-GGUF"},
}

def pil_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
class SurveillanceAI:
    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model_type = None # 'hf' or 'gguf'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ready = False 
        self.lock = threading.RLock()

    def load_model(self, model_key, use_4bit=False):
        """
        Dynamically loads a model (HF or GGUF), unloading the previous one.
        """
        if model_key not in DEFAULT_MODELS:
            return f"Error: Model {model_key} not found."

        config = DEFAULT_MODELS[model_key]
        model_type = config.get("type", "hf")
        model_id = config["id"]

        requested_4bit = use_4bit

        if "FP8" in model_key or "Int8" in model_id or "FP8" in model_id:
            if use_4bit:
                print(f"⚠️ Warning: {model_key} is already quantized. Disabling 4-bit loading to prevent conflicts.")
                use_4bit = False

        with self.lock:
            print(f"🔄 Switching to {model_id} ({model_type})...")
            self.ready = False

            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            cache_dir = _configure_hf_downloads(project_root)
            print(f"📦 HF cache: {cache_dir}")
            
            # 1. Cleanup Memory
            if self.model is not None:
                del self.model
                if self.processor: del self.processor
                self.model = None
                self.processor = None
                gc.collect()
                torch.cuda.empty_cache()
                print("🧹 VRAM cleared.")

            try:
                # --- GGUF LOADING ---
                if model_type == "gguf":
                    if not GGUF_AVAILABLE:
                        return "Error: `llama-cpp-python` not installed. Run: pip install llama-cpp-python"
                    
                    filename = config.get("filename")
                    if not filename:
                        return "Error: GGUF config missing 'filename'."

                    # Download or retrieve path
                    print(f"📥 Verifying/Downloading {filename}...")
                    model_path = hf_hub_download(
                        repo_id=model_id,
                        filename=filename,
                        cache_dir=os.environ.get("HF_HUB_CACHE") or cache_dir,
                        resume_download=True,
                    )

                    # Load Llama
                    # n_gpu_layers=-1 attempts to offload ALL layers to GPU
                    self.model = Llama(
                        model_path=model_path,
                        n_ctx=4096,            # Vision models need large context
                        n_gpu_layers=-1,       # Max GPU usage
                        verbose=False
                    )
                    self.current_model_type = "gguf"

                # --- HUGGING FACE LOADING ---
                else:
                    bnb_config = None
                    if use_4bit:
                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16
                        )

                    # transformers 5.0.1.dev0 workaround:
                    # Some configs define `quantization_config` but don't populate it from config.json,
                    # and the quantizer detection code crashes on None.
                    config_obj = AutoConfig.from_pretrained(
                        model_id,
                        trust_remote_code=True,
                        cache_dir=cache_dir,
                    )
                    if hasattr(config_obj, "quantization_config") and getattr(config_obj, "quantization_config", None) is None:
                        try:
                            cfg_path = hf_hub_download(
                                repo_id=model_id,
                                filename="config.json",
                                cache_dir=os.environ.get("HF_HUB_CACHE") or cache_dir,
                                resume_download=True,
                            )
                            with open(cfg_path, "r", encoding="utf-8") as f:
                                raw_cfg = json.load(f)
                            raw_qc = raw_cfg.get("quantization_config")
                            if isinstance(raw_qc, dict) and raw_qc:
                                config_obj.quantization_config = raw_qc
                        except Exception:
                            # If we can't hydrate it, proceed; transformers may still handle it.
                            pass

                    self.processor = AutoProcessor.from_pretrained(
                        model_id,
                        trust_remote_code=True,
                        cache_dir=cache_dir,
                    )
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        model_id,
                        trust_remote_code=True,
                        config=config_obj,
                        device_map="auto",
                        quantization_config=bnb_config if use_4bit else None,
                        torch_dtype="auto",
                        cache_dir=cache_dir,
                    )
                    self.current_model_type = "hf"
                
                self.ready = True
                print(f"✅ {model_key} Loaded Successfully!")
                return f"Loaded {model_key}"
            
            except Exception as e:
                import traceback
                print(f"❌ Error loading model: {e}")
                traceback.print_exc()
                self.ready = False

                # Friendly fallback: FP8 checkpoints commonly require Triton and/or newer GPUs.
                # If that environment isn't available, try the non-FP8 variant automatically.
                err_str = str(e)
                if "FP8" in model_key and (
                    "No module named 'triton'" in err_str
                    or "compute capability" in err_str
                    or "FineGrainedFP8" in err_str
                ):
                    fallback_key = model_key.replace("-FP8", "")
                    if fallback_key in DEFAULT_MODELS and fallback_key != model_key:
                        print(f"↩️ Falling back to {fallback_key} (requested 4-bit={requested_4bit})")
                        return self.load_model(fallback_key, use_4bit=requested_4bit)

                return f"Error: {str(e)}"

    def analyze_media(self, media_path, prompt, media_type="image"):
        if not self.ready: return "Error: Model not loaded."

        # Prepare messages based on media type logic
        # Note: GGUF implementation here primarily supports Images for now.
        if media_type == "video" and self.current_model_type == "gguf":
            return "Warning: Video analysis is not fully optimized for GGUF in this version. Try 'image'."

        if media_type == "image":
             pil_image = Image.open(media_path)
             return self.analyze_frame_pil(pil_image, prompt)
        
        # Fallback for HF Video
        messages = [{
            "role": "user",
            "content": [
                {"type": media_type, media_type: media_path, **({"fps": 1.0} if media_type == "video" else {})},
                {"type": "text", "text": prompt},
            ],
        }]
        return self._run_inference(messages)

    def analyze_frame_pil(self, pil_image, prompt):
        if not self.ready: return "Waiting..."

        if self.current_model_type == "gguf":
            # GGUF expects Base64 for images in OpenAI-compatible format
            base64_img = pil_to_base64(pil_image)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]
            }]
            return self._run_inference_gguf(messages)
        else:
            # HF Standard Format
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }]
            return self._run_inference(messages, max_tokens=128)

    def _run_inference_gguf(self, messages, max_tokens=256):
        try:
            with self.lock:
                response = self.model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.2 # Low temp for factual surveillance
                )
                return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"GGUF Inference Error: {e}")
            return f"Error: {e}"

    def _run_inference(self, messages, max_tokens=256):
        """HF Inference"""
        try:
            with self.lock:
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.model.device)

                generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                return self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        except Exception as e:
            print(f"HF Inference Error: {e}")
            return f"Error: {e}"