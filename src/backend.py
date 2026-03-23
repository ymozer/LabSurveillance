import os
import gc
import re
import tempfile
import threading

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# Reduce CUDA memory fragmentation on GPUs with limited VRAM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _configure_hf_downloads(project_root: str) -> str:
    cache_dir = os.environ.get("HF_HOME")
    if not cache_dir:
        cache_dir = os.path.join(project_root, ".hf_cache")
        os.environ.setdefault("HF_HOME", cache_dir)

    os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_dir, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE",
                          os.path.join(cache_dir, "transformers"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


# --- Model Definitions ---
DEFAULT_MODELS = {
    "Qwen2.5-VL-3B-Instruct":       {"type": "hf", "id": "Qwen/Qwen2.5-VL-3B-Instruct"},
    "Qwen2.5-VL-7B-Instruct":       {"type": "hf", "id": "Qwen/Qwen2.5-VL-7B-Instruct"},
    "Qwen2-VL-2B-Instruct":         {"type": "hf", "id": "Qwen/Qwen2-VL-2B-Instruct"},
    "Qwen2-VL-7B-Instruct":         {"type": "hf", "id": "Qwen/Qwen2-VL-7B-Instruct"},
    "Qwen3-VL-2B-Instruct":         {"type": "hf", "id": "Qwen/Qwen3-VL-2B-Instruct"},
    "Qwen3-VL-2B-Instruct-FP8":     {"type": "hf", "id": "Qwen/Qwen3-VL-2B-Instruct-FP8"},
    "Qwen3-VL-4B-Instruct":         {"type": "hf", "id": "Qwen/Qwen3-VL-4B-Instruct"},
    "Qwen3-VL-4B-Instruct-FP8":     {"type": "hf", "id": "Qwen/Qwen3-VL-4B-Instruct-FP8"},
    "Qwen3-VL-8B-Instruct":         {"type": "hf", "id": "Qwen/Qwen3-VL-8B-Instruct"},
    "Qwen3-VL-8B-Instruct-FP8":     {"type": "hf", "id": "Qwen/Qwen3-VL-8B-Instruct-FP8"},
}


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from Qwen3 model output."""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # If stripping removed everything, return original
    return cleaned if cleaned else text


class SurveillanceAI:
    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model_type = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ready = False
        self.lock = threading.RLock()

    def load_model(self, model_key, use_4bit=False):
        if model_key not in DEFAULT_MODELS:
            return f"Error: Model {model_key} not found."

        config = DEFAULT_MODELS[model_key]
        model_id = config["id"]

        with self.lock:
            print(f"🔄 Loading {model_id}...")
            self.ready = False

            # Cleanup
            if self.model is not None:
                del self.model
                if self.processor:
                    del self.processor
                gc.collect()
                torch.cuda.empty_cache()

            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), ".."))
            cache_dir = _configure_hf_downloads(project_root)

            try:
                # 4-bit Config
                bnb_config = None
                if use_4bit:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )

                # Load Processor – limit per-image pixels to control VRAM
                self.processor = AutoProcessor.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    min_pixels=128 * 28 * 28,
                    max_pixels=512 * 28 * 28,
                )

                # Load Model
                # Note: Qwen2-VL and Qwen2.5-VL usually rely on flash_attn if available
                dtype = torch.bfloat16 if use_4bit else torch.float16
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    device_map="auto",
                    quantization_config=bnb_config,
                    cache_dir=cache_dir,
                    low_cpu_mem_usage=True,
                )

                self.current_model_type = "hf"
                self.ready = True
                print(f"✅ {model_key} Loaded Successfully on {self.device}")
                return f"Loaded {model_key}"

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.ready = False
                return f"Error: {str(e)}"

    def analyze_video_clip(self, frame_list_pil, prompt, system_prompt=None):
        """
        Analyzes a sequence of frames. 
        Note: We send frames as a sequence of 'image' types to bypass 
        qwen_vl_utils limitations with in-memory PIL lists for 'video' tags.
        """
        if not self.ready:
            return "Error: Model not loaded."

        if not frame_list_pil or len(frame_list_pil) == 0:
            return "Error: No frames in clip."

        # Construct Message Payload
        # We convert the list of frames into a sequence of image blocks.
        content_blocks = []
        for frame in frame_list_pil:
            content_blocks.append({"type": "image", "image": frame})

        # Add the text prompt at the end
        content_blocks.append({"type": "text", "text": prompt})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": content_blocks,
            }
        )

        return self._run_inference(messages)

    def analyze_video_clip_as_video(self, frame_list_pil, prompt, fps=2.0, system_prompt=None):
        """
        Analyzes a sequence of frames using a 'video' content block.
        Saves frames to temp files to preserve temporal encoding.
        """
        if not self.ready:
            return "Error: Model not loaded."

        if not frame_list_pil or len(frame_list_pil) == 0:
            return "Error: No frames in clip."

        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = []
            for i, frame in enumerate(frame_list_pil):
                path = os.path.join(tmp_dir, f"frame_{i}.jpg")
                frame.save(path)
                paths.append(path)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": paths, "fps": float(fps)},
                        {"type": "text", "text": prompt},
                    ],
                }
            )

            result = self._run_inference(messages)
            if isinstance(result, str) and result.startswith("Error:"):
                # Fallback to image-sequence mode if video processing fails
                return self.analyze_video_clip(frame_list_pil, prompt, system_prompt=system_prompt)
            return result

    def analyze_single_image(self, pil_image, prompt):
        """Legacy support for single image analysis"""
        if not self.ready:
            return "Error: Model not loaded."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._run_inference(messages)

    def _run_inference(self, messages, max_tokens=256):
        inputs = None
        generated_ids = None
        try:
            with self.lock, torch.inference_mode():
                # 1. Prepare inputs using qwen_vl_utils
                try:
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    messages, True, True)

                # 2. Processor
                processor_kwargs = {
                    "text": [text],
                    "images": image_inputs,
                    "padding": True,
                    "return_tensors": "pt",
                }
                valid_video = False
                if video_inputs is not None:
                    try:
                        if len(video_inputs) > 0:
                            valid_video = True
                    except Exception:
                        valid_video = False

                    if valid_video and isinstance(video_inputs, (list, tuple)):
                        if len(video_inputs) == 0:
                            valid_video = False
                        else:
                            first = video_inputs[0]
                            try:
                                if isinstance(first, (list, tuple)) and len(first) == 0:
                                    valid_video = False
                                elif all(isinstance(v, (list, tuple)) for v in video_inputs):
                                    if any(len(v) == 0 for v in video_inputs):
                                        valid_video = False
                                elif hasattr(first, "size") and getattr(first, "size") == 0:
                                    valid_video = False
                            except Exception:
                                pass

                    if valid_video and hasattr(video_inputs, "size") and getattr(video_inputs, "size") == 0:
                        valid_video = False

                if valid_video:
                    processor_kwargs["videos"] = video_inputs
                    if video_kwargs:
                        processor_kwargs.update(video_kwargs)

                try:
                    inputs = self.processor(
                        **processor_kwargs).to(self.model.device)
                except IndexError:
                    if "videos" in processor_kwargs:
                        processor_kwargs.pop("videos", None)
                    inputs = self.processor(
                        **processor_kwargs).to(self.model.device)

                # 3. Generate
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=max_tokens)

                # 4. Decode
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

                output_text = _strip_thinking(output_text)
                return output_text

        except torch.cuda.OutOfMemoryError:
            print("CUDA OOM – freeing cache and returning error")
            return "Error: GPU out of memory. Reduce frame count or use a smaller model."
        except Exception as e:
            print(f"Inference Error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
        finally:
            # Always release GPU tensors to prevent VRAM leaks
            try:
                del inputs, generated_ids
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
