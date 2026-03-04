# 😻 Kitten-TTS for ComfyUI

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-blue)](https://github.com/comfyanonymous/ComfyUI)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange)](LICENSE)

A simple ComfyUI custom node for [KittenTTS](https://github.com/KittenML/KittenTTS) - an ultra-lightweight text-to-speech model. Works on **CUDA** and **CPU**.

<img width="1496" height="1015" alt="Screenshot 2026-02-19 194339" src="https://github.com/user-attachments/assets/6af8a500-0a48-47da-adff-e020a5437e88" />


## Demo

https://github.com/user-attachments/assets/d80120f2-c751-407e-a166-068dd1dd9e8d

## Features

- 🚀 **Ultra-lightweight**: Models from 19MB to 80MB
- 💻 **CPU & CUDA**: Runs on any device, GPU optional
- 🎯 **Single node**: All settings in one place
- 📦 **Auto-download**: Models cached in `ComfyUI/models/kittentts/`
- ⚡ **Fast inference**: Real-time speech synthesis
- 🎤 **8 voices**: 4 male, 4 female

## Installation

### Prerequisites (Required)
Before installing the custom node, you **must** install `espeak-ng` and set up its environment variable:

1. **Install espeak-ng**: Download and install it from [https://github.com/espeak-ng/espeak-ng/](https://github.com/espeak-ng/espeak-ng/).
2. **Set Environment Variable**: Add a new system environment variable named `PHONEMIZER_ESPEAK_LIBRARY` pointing to the `espeak-ng` installation path (e.g., `C:\Program Files\eSpeak NG\libespeak-ng.dll` on Windows).
3. **Reboot**: You must completely restart the ComfyUI CLI/console for the updated environment variables to be securely loaded.

### Method 1: ComfyUI Manager (Recommended)
Search for "KittenTTS" in ComfyUI Manager and install.

### Method 2: Manual Install
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/ComfyUI-KittenTTS.git
```

### Dependencies

The node auto-installs most dependencies on first run. The correct version of `onnxruntime` (CPU or GPU) is dynamically chosen based on whether your PyTorch environment supports CUDA.

If you need to install or fix it manually:

**Standard Python environment:**
```bash
# For CPU
pip install onnxruntime

# For CUDA GPU
pip install onnxruntime-gpu
```

**For Windows portable ComfyUI, use the embedded Python:**
```bash
# From ComfyUI_windows_portable directory

# For CPU
.\python_embeded\python.exe -m pip install onnxruntime

# For CUDA GPU
.\python_embeded\python.exe -m pip install onnxruntime-gpu
```

## Models

| Model | Params | Size | Quality | Link |
|-------|--------|------|---------|------|
| kitten-tts-mini | 80M | 80MB | Best | [🤗 Download](https://huggingface.co/KittenML/kitten-tts-mini-0.8) |
| kitten-tts-micro | 40M | 41MB | Good | [🤗 Download](https://huggingface.co/KittenML/kitten-tts-micro-0.8) |
| kitten-tts-nano | 15M | 56MB | Lightweight | [🤗 Download](https://huggingface.co/KittenML/kitten-tts-nano-0.8) |
| kitten-tts-nano-int8 | 15M | 19MB | Smallest | [🤗 Download](https://huggingface.co/KittenML/kitten-tts-nano-0.8-int8) |

> ⚠️ **Note**: Some users report issues with the int8 quantized model. We recommend the mini or micro versions for best results.

## Usage

The node is simple - just one node with all settings:

### Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | Dropdown | mini-0.8 (80M) | Model size/quality |
| `device` | Dropdown | auto | auto/cuda/cpu |
| `text` | String | - | Text to synthesize |
| `voice` | Dropdown | Jasper | Voice selection |
| `speed` | Float | 1.0 | Speech speed (0.5-2.0) |
| `keep_loaded` | Boolean | True | Keep model in memory |
| `output_stereo` | Boolean | False | Stereo output |
| `clean_text` | Boolean | True | Normalize text |
| `custom_model` | String | "" | Custom HF model ID |

### Voices

**Male:** Jasper, Bruno, Hugo, Leo  
**Female:** Bella, Luna, Rosie, Kiki

### Example

Just add the **🐱 KittenTTS** node, type your text, select a voice, and connect the audio output to your pipeline.

```
┌─────────────────────────┐
│  🐱 KittenTTS           │ → Connect to audio nodes
├─────────────────────────┤
│ model: mini-0.8 (80M)   │
│ device: auto            │
│ text: "Hello world!"    │
│ voice: Jasper           │
│ speed: 1.0              │
│ keep_loaded: True       │
│ output_stereo: False    │
│ clean_text: True        │
│ custom_model_path ...   │
├─────────────────────────┤
└─────────────────────────┘
```

## Model Storage

Models are downloaded to: `ComfyUI/models/kittentts/<model_name>/`

You can create symlinks to this folder if you want to share models between ComfyUI installations.

## Troubleshooting

### "CUDA out of memory"
- Switch to a smaller model (nano or micro)
- Set `keep_loaded` to False
- Use CPU device

### "onnxruntime not found"
**Standard Python environment:**
```bash
# For CPU
pip install onnxruntime

# For CUDA
pip install onnxruntime-gpu
```

**For Windows portable ComfyUI, use the embedded Python:**
```bash
# From ComfyUI_windows_portable directory

# For CPU
.\python_embeded\python.exe -m pip install onnxruntime

# For CUDA
.\python_embeded\python.exe -m pip install onnxruntime-gpu
```

### "No module named 'kittentts'"
The bundled wheel should auto-install. If not:

**Standard Python environment:**
```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.8/kittentts-0.8.0-py3-none-any.whl
```

**For Windows portable ComfyUI:**
```bash
.\python_embeded\python.exe -m pip install https://github.com/KittenML/KittenTTS/releases/download/0.8/kittentts-0.8.0-py3-none-any.whl
```

## Links

- **Original KittenTTS Repo**: [https://github.com/KittenML/KittenTTS](https://github.com/KittenML/KittenTTS)
- **HuggingFace Models**: [https://huggingface.co/KittenML](https://huggingface.co/KittenML)
- **Discord**: [Join KittenML Discord](https://discord.com/invite/VJ86W4SURW)
- **Demo Video**: [Watch on GitHub](https://github.com/KittenML/KittenTTS)

## Credits

- KittenTTS by [KittenML](https://github.com/KittenML)
- ComfyUI node by [Saganaki22](https://github.com/Saganaki22)

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
