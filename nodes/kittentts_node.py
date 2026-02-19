"""KittenTTS - Single node for text-to-speech synthesis."""

import logging
import re
from typing import Dict, Any, Tuple

try:
    from comfy.utils import ProgressBar
    COMFYUI_PROGRESS_AVAILABLE = True
except ImportError:
    COMFYUI_PROGRESS_AVAILABLE = False

try:
    import comfy.model_management as mm
    INTERRUPTION_SUPPORT = True
except ImportError:
    INTERRUPTION_SUPPORT = False

from .utils import (
    format_audio_for_comfyui, 
    get_cached_model, 
    unload_model,
    AVAILABLE_VOICES, 
    AVAILABLE_MODELS, 
    MODEL_MAP,
    check_onnxruntime,
    has_cuda,
)

logger = logging.getLogger("KittenTTS")


class KittenTTS:
    """KittenTTS - Ultra-lightweight text-to-speech synthesis."""
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        _, has_gpu = check_onnxruntime()
        cuda_available = has_cuda()
        
        if has_gpu:
            device_default = "auto"
            device_options = ["auto", "cuda", "cpu"]
        elif cuda_available:
            device_options = ["auto", "cpu"]
            device_default = "auto"
        else:
            device_options = ["cpu"]
            device_default = "cpu"
        
        return {
            "required": {
                "model_name": (AVAILABLE_MODELS, {
                    "default": "KittenML/kitten-tts-mini-0.8 (80M)",
                    "tooltip": "Mini=80MB best quality, Micro=40MB balanced, Nano=15M lightweight, Int8=19MB smallest",
                }),
                "device": (device_options, {
                    "default": device_default,
                    "tooltip": "Auto: GPU if available. CUDA: force GPU. CPU: force CPU.",
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello, this is a test of KittenTTS text to speech synthesis.",
                    "tooltip": "Text to convert to speech. Long text auto-split into sentences.",
                }),
                "voice": (AVAILABLE_VOICES, {
                    "default": "Jasper",
                    "tooltip": "Jasper/Bruno/Hugo/Leo=male. Bella/Luna/Rosie/Kiki=female.",
                }),
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "1.0=normal, <1.0=slower, >1.0=faster. Recommended: 0.8-1.2",
                }),
                "keep_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model in memory for faster re-use. Disable to free RAM/VRAM.",
                }),
                "output_stereo": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Output stereo (2ch) instead of mono. Some pipelines need stereo.",
                }),
                "clean_text": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Normalize text: numbers→words, expand abbreviations. Keep ON.",
                }),
            },
            "optional": {
                "custom_model": ("STRING", {
                    "default": "",
                    "tooltip": "Custom HuggingFace model ID (e.g., 'user/my-model'). Overrides dropdown.",
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio"
    DESCRIPTION = "Text-to-speech. Deterministic (no seed needed). Models cached in ComfyUI/models/kittentts/"
    
    def generate(
        self,
        model_name: str,
        device: str,
        text: str,
        voice: str,
        speed: float,
        keep_loaded: bool,
        output_stereo: bool,
        clean_text: bool,
        custom_model: str = "",
    ) -> Tuple[Dict[str, Any]]:
        """Generate speech from text."""
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        self._check_interrupt()
        
        actual_model = custom_model.strip() if custom_model.strip() else MODEL_MAP.get(model_name, model_name)
        
        try:
            tts_model, actual_device, is_gpu = get_cached_model(actual_model, device)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise Exception(f"Failed to load KittenTTS model '{actual_model}': {str(e)}")
        
        gpu_str = "GPU" if is_gpu else "CPU"
        logger.info(f"Generating on {gpu_str}")
        
        try:
            import numpy as np
            
            audio = tts_model.model.generate(
                text=text,
                voice=voice,
                speed=speed,
                clean_text=clean_text,
            )
            
            audio_output = format_audio_for_comfyui(audio, sample_rate=24000, stereo=output_stereo)
            
            duration_sec = len(audio) / 24000
            channels = "stereo" if output_stereo else "mono"
            logger.info(f"Generated {duration_sec:.2f}s {channels} audio")
            
            if not keep_loaded:
                unload_model()
            
            return (audio_output,)
            
        except Exception as e:
            if INTERRUPTION_SUPPORT:
                from comfy.model_management import InterruptProcessingException
                if isinstance(e, InterruptProcessingException):
                    raise
            
            logger.error(f"Generation failed: {e}")
            raise Exception(f"Speech generation failed: {str(e)}")
    
    def _check_interrupt(self):
        """Check if user requested cancellation."""
        if INTERRUPTION_SUPPORT:
            try:
                mm.throw_exception_if_processing_interrupted()
            except Exception:
                raise
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash(str(kwargs))
