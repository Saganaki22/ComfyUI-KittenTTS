"""Utility functions for KittenTTS nodes."""

import logging
import os
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger("KittenTTS")


def has_cuda():
    """Check if CUDA is available via PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except ImportError:
        pass
    return False


def check_onnxruntime():
    """Check if onnxruntime is installed (GPU or CPU).
    
    Returns:
        Tuple of (is_installed, has_gpu_support)
    """
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        has_gpu = 'CUDAExecutionProvider' in providers or 'ROCMExecutionProvider' in providers
        return True, has_gpu
    except ImportError:
        return False, False


def get_models_dir() -> Path:
    """Get the ComfyUI models directory for KittenTTS.
    
    Uses folder_paths from ComfyUI if available, otherwise falls back to
    the models directory relative to ComfyUI root.
    """
    try:
        from folder_paths import models_dir
        models_path = Path(models_dir)
    except ImportError:
        current_file = Path(__file__).resolve()
        for parent in current_file.parents:
            if (parent / "custom_nodes").exists():
                models_path = parent / "models"
                break
        else:
            models_path = Path(__file__).parent.parent / "models"
    
    kittentts_dir = models_path / "kittentts"
    kittentts_dir.mkdir(parents=True, exist_ok=True)
    return kittentts_dir


def get_model_cache_dir(model_name: str) -> Path:
    """Get the cache directory for a specific model.
    
    Args:
        model_name: HuggingFace model name (e.g., "KittenML/kitten-tts-nano-0.2")
        
    Returns:
        Path to the model's cache directory
    """
    models_dir = get_models_dir()
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    model_dir = models_dir / safe_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def get_onnx_providers(device: str = "auto") -> Tuple[list, str, bool]:
    """Get appropriate ONNX providers based on device selection.
    
    Args:
        device: "auto", "cuda", or "cpu"
        
    Returns:
        Tuple of (providers_list, device_name, is_gpu)
    """
    available_providers = []
    
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
    except ImportError:
        return ['CPUExecutionProvider'], "cpu", False
    
    has_cuda = 'CUDAExecutionProvider' in available_providers
    has_rocm = 'ROCMExecutionProvider' in available_providers
    has_gpu = has_cuda or has_rocm
    
    if device == "cpu":
        return ['CPUExecutionProvider'], "cpu", False
    
    if device == "cuda":
        if has_gpu:
            gpu_name = "cuda" if has_cuda else "rocm"
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if has_cuda else ['ROCMExecutionProvider', 'CPUExecutionProvider']
            return providers, gpu_name, True
        else:
            logger.warning("CUDA requested but not available, falling back to CPU")
            return ['CPUExecutionProvider'], "cpu", False
    
    if device == "auto":
        if has_gpu:
            gpu_name = "cuda" if has_cuda else "rocm"
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if has_cuda else ['ROCMExecutionProvider', 'CPUExecutionProvider']
            logger.info(f"Auto-detected GPU ({gpu_name}), using GPU acceleration")
            return providers, gpu_name, True
        else:
            return ['CPUExecutionProvider'], "cpu", False
    
    return ['CPUExecutionProvider'], "cpu", False


def format_audio_for_comfyui(audio, sample_rate: int = 24000, stereo: bool = False):
    """Format audio tensor for ComfyUI.
    
    Args:
        audio: Audio data (numpy array or torch tensor)
        sample_rate: Sample rate
        stereo: If True, output stereo (duplicates mono to 2 channels)
        
    Returns:
        Dict with 'waveform' and 'sample_rate'
    """
    import torch
    import numpy as np
    
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float()
    
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)
    
    if stereo:
        audio = audio.repeat(1, 2, 1)
    
    audio = audio.contiguous().cpu().float()
    
    return {
        "waveform": audio,
        "sample_rate": sample_rate,
    }


_shared_model = None
_shared_model_config = None


def get_cached_model(model_name: str, device: str = "auto"):
    """Get or create cached KittenTTS model.
    
    Args:
        model_name: HuggingFace model name
        device: Device preference
        
    Returns:
        Tuple of (model, actual_device, is_gpu)
    """
    global _shared_model, _shared_model_config
    
    providers, actual_device, is_gpu = get_onnx_providers(device)
    
    cache_dir = get_model_cache_dir(model_name)
    
    config = {
        "model_name": model_name,
        "device": actual_device,
        "cache_dir": str(cache_dir),
    }
    
    if _shared_model is not None and _shared_model_config == config:
        logger.debug("Using cached KittenTTS model")
        return _shared_model, actual_device, is_gpu
    
    if _shared_model is not None:
        del _shared_model
        _shared_model = None
    
    logger.info(f"Loading KittenTTS model: {model_name}")
    logger.info(f"Cache directory: {cache_dir}")
    
    from kittentts import KittenTTS
    
    model = KittenTTS(model_name, cache_dir=str(cache_dir))
    
    if hasattr(model.model, 'session'):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        new_session = ort.InferenceSession(
            model.model.session._model_path,
            sess_options=so,
            providers=providers
        )
        model.model.session = new_session
        
        actual_providers = new_session.get_providers()
        logger.info(f"ONNX session providers: {actual_providers}")
    
    _shared_model = model
    _shared_model_config = config
    
    return model, actual_device, is_gpu


def unload_model():
    """Unload the cached model to free memory."""
    global _shared_model, _shared_model_config
    
    if _shared_model is not None:
        logger.info("Unloading KittenTTS model from memory")
        del _shared_model
        _shared_model = None
        _shared_model_config = None
        
        import gc
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        return True
    return False


AVAILABLE_VOICES = [
    'Jasper',
    'Bella', 
    'Luna', 
    'Bruno', 
    'Rosie', 
    'Hugo', 
    'Kiki', 
    'Leo'
]

AVAILABLE_MODELS = [
    "KittenML/kitten-tts-mini-0.8 (80M)",
    "KittenML/kitten-tts-micro-0.8 (40M)",
    "KittenML/kitten-tts-nano-0.8 (15M)",
    "KittenML/kitten-tts-nano-0.8-int8 (15M quantized)",
]

MODEL_MAP = {
    "KittenML/kitten-tts-mini-0.8 (80M)": "KittenML/kitten-tts-mini-0.8",
    "KittenML/kitten-tts-micro-0.8 (40M)": "KittenML/kitten-tts-micro-0.8",
    "KittenML/kitten-tts-nano-0.8 (15M)": "KittenML/kitten-tts-nano-0.8",
    "KittenML/kitten-tts-nano-0.8-int8 (15M quantized)": "KittenML/kitten-tts-nano-0.8-int8",
}
