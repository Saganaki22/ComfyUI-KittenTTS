"""ComfyUI nodes for KittenTTS - Ultra-lightweight text-to-speech.

This package provides ComfyUI integration for KittenTTS.
"""

__version__ = "1.0.1"
__author__ = "Saganaki22"

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger("KittenTTS")
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[KittenTTS] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def has_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except ImportError:
        pass
    return False


def check_onnxruntime():
    """Check if onnxruntime is installed (GPU or CPU)."""
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        has_gpu = 'CUDAExecutionProvider' in providers or 'ROCMExecutionProvider' in providers
        return True, has_gpu
    except ImportError:
        return False, False


def install_dependencies():
    """Install required dependencies including the bundled wheel."""
    current_dir = Path(__file__).parent
    
    missing_packages = []
    
    try:
        import onnxruntime
    except ImportError:
        missing_packages.append("onnxruntime")
    
    try:
        import phonemizer
    except ImportError:
        missing_packages.append("phonemizer")
    
    try:
        import huggingface_hub
    except ImportError:
        missing_packages.append("huggingface_hub")
    
    try:
        import soundfile
    except ImportError:
        missing_packages.append("soundfile")
    
    try:
        from misaki import en
    except ImportError:
        missing_packages.append("misaki[en]")
    
    try:
        import espeakng_loader
    except ImportError:
        missing_packages.append("espeakng_loader")
    
    if missing_packages:
        logger.info(f"Installing missing dependencies: {', '.join(missing_packages)}")
        
        install_cmd = [sys.executable, "-m", "pip", "install"]
        
        for pkg in missing_packages:
            install_cmd.append(pkg)
        
        try:
            result = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode != 0:
                logger.error(f"Failed to install dependencies: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error installing dependencies: {e}")
            return False
    
    try:
        import kittentts
    except ImportError:
        whl_dir = current_dir / "whl"
        wheel_file = whl_dir / "kittentts-0.8.0-py3-none-any.whl"
        if wheel_file.exists():
            logger.info("Installing bundled KittenTTS wheel...")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", str(wheel_file)],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.returncode != 0:
                    logger.error(f"Failed to install wheel: {result.stderr}")
                    return False
                logger.info("KittenTTS wheel installed successfully!")
            except Exception as e:
                logger.error(f"Error installing wheel: {e}")
                return False
        else:
            logger.error(f"Bundled wheel not found: {wheel_file}")
            return False
    
    return True


def install_onnxruntime_gpu():
    """Install onnxruntime-gpu if CUDA is available and user wants GPU."""
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime"],
            capture_output=True,
            check=False
        )
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "onnxruntime-gpu"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            logger.info("onnxruntime-gpu installed successfully!")
            return True
        else:
            logger.warning(f"Failed to install onnxruntime-gpu: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error installing onnxruntime-gpu: {e}")
        return False


NODE_CLASS_MAPPINGS: Dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}

if install_dependencies():
    try:
        from .nodes import KittenTTS
        
        NODE_CLASS_MAPPINGS["KittenTTS"] = KittenTTS
        NODE_DISPLAY_NAME_MAPPINGS["KittenTTS"] = "😻 Kitten-TTS"
        
        _, has_gpu = check_onnxruntime()
        gpu_status = "GPU" if has_gpu else "CPU"
        logger.info(f"Nodes registered successfully (v{__version__}) - ONNX Runtime: {gpu_status}")
        
    except Exception as e:
        logger.error(f"Failed to register nodes: {e}")
        import traceback
        traceback.print_exc()
else:
    logger.warning("Nodes unavailable - missing dependencies")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', '__version__', 'install_onnxruntime_gpu', 'has_cuda', 'check_onnxruntime']
