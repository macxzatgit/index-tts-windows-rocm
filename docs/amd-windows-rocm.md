# IndexTTS on native Windows with an AMD GPU (ROCm)

This guide records a tested setup for IndexTTS 2.5 on native Windows with an AMD Radeon RX 9060 XT 16 GB. It keeps inference on the AMD GPU through ROCm PyTorch. It is a personal Windows/ROCm compatibility recipe; the upstream project does not currently declare this exact configuration as a supported install path.

## Tested configuration

- Windows 11, Python 3.12.10
- AMD Radeon RX 9060 XT 16 GB eGPU, with an integrated GPU also present
- AMD Adrenalin 26.11.2
- `torch 2.9.1+rocm7.2.1`, HIP `7.2.53211-158bd99533`
- `transformers==4.52.1`, `tokenizers==0.21.0`
- IndexTTS 2.5, WebUI launched with `--fp16`

Verification succeeded with `torch.cuda.is_available() == True`, device name `AMD Radeon RX 9060 XT`, and an end-to-end generated 22,050 Hz PCM WAV. ROCm PyTorch intentionally exposes AMD GPU operations through PyTorch's `torch.cuda` API.

## Why the normal project sync is unsuitable

The upstream `pyproject.toml` declares Python `<3.12`, pins PyTorch 2.8, and directs `uv` to the NVIDIA CUDA 12.8 wheel index. AMD's Windows PyTorch wheels require Python 3.12. Do not run `uv sync` or install the CUDA/DeepSpeed/FlashAttention extras in this environment; that can replace the working ROCm PyTorch packages or bring in NVIDIA-only components.

AMD's Windows ROCm release is a PyTorch-focused stack, and its documentation notes that `torch.distributed` is not supported. Check AMD's current [Windows PyTorch install guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html) and [GPU support matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html) before changing ROCm or driver versions. The RX 9060 XT appears in the Windows ROCm 7.2.1 support matrix.

## Fresh environment

Use the AMD wheel instructions for the ROCm release that matches the installed driver. For the tested ROCm 7.2.1 release, AMD specifies Python 3.12 and Radeon Software 26.2.2 or newer. Example layout used here: virtual environment at `C:\AI\index-tts\.venv`, clone at `C:\AI\index-tts\source`. Create the venv and keep its Python path in `$py`; calling it directly also avoids PowerShell activation-policy issues:

```powershell
cd C:\AI\index-tts
py -3.12 -m venv .venv
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $py -m pip install --upgrade pip
```

Install the four SDK wheels first, then the PyTorch wheels. These PowerShell commands are from AMD's [Windows install guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/windows/install-pytorch.html):

```powershell
& $py -m pip install --no-cache-dir `
  'https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl' `
  'https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl' `
  'https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl' `
  'https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz'

& $py -m pip install --no-cache-dir `
  'https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl' `
  'https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl' `
  'https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl'
```

From the cloned repository, install the non-Torch dependencies:

```powershell
cd C:\AI\index-tts\source
& $py -m pip install -r .\requirements-amd-windows.txt
```

`requirements-amd-windows.txt` deliberately excludes Torch packages and optional CUDA acceleration packages. It pins Transformers 4.52.1 and Tokenizers 0.21.0, which match this IndexTTS checkout. Transformers 4.56.0 produced additional missing-API errors in IndexTTS's copied generation code, including a missing `GenerationConfig.forced_decoder_ids` during synthesis.

Verify the GPU before starting IndexTTS:

```powershell
& $py -c "import torch; print('Torch:', torch.__version__); print('HIP:', torch.version.hip); print('GPU:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Expected on the tested machine: `2.9.1+rocm7.2.1`, HIP `7.2.53211-158bd99533`, GPU `True`, device `AMD Radeon RX 9060 XT`. On a multi-GPU/eGPU system, confirm device 0 is the RX 9060 XT before launching. Connecting the display to the eGPU made it the detected compute device on the tested machine.
## Source compatibility changes in this fork

These small changes are included in this branch:

1. `indextts/__init__.py` provides the missing distributed enum/tensor names only when absent, and marks distributed support unavailable. This is for single-process inference; it does not add DDP or training support.
2. `indextts/gpt/transformers_gpt2.py` imports `SequenceSummary` from IndexTTS's local Transformers copy. Importing it from upstream Transformers pulled in the Windows ROCm build's unavailable distributed tensor extension.
3. `indextts/utils/common.py` falls back to the already-installed SoundFile writer when Torchaudio 2.9 reports that TorchCodec is missing. It still writes PCM-16 WAV output.
4. `webui.py` makes QwenEmotion opt-in with `--qwen_emo`, avoiding an extra model load unless emotion-from-text control is needed. The regular voice reference, emotion reference audio, and emotion vector modes remain available.

Keep the project-pinned Transformers version with these changes. Do not use the experimental Transformers 4.56 compatibility edits from earlier troubleshooting attempts; they were removed after the working 4.52.1 configuration was verified.

## Launch and memory notes

Start one WebUI process from the source directory:

```powershell
cd C:\AI\index-tts\source
& ..\.venv\Scripts\python.exe .\webui.py --version 2.5 --fp16
```

Then open `http://127.0.0.1:7860`. In IndexTTS 2.5, `--fp16` selects BF16 when the GPU reports support. Add `--qwen_emo` only if emotion-from-text control is needed. Stop a running server with Ctrl+C before starting it again; two WebUI processes each load their own full model and can fill 16 GB of VRAM. Successful inference can still produce MIOpen workspace/timing warnings on this Windows ROCm build; the verified warnings did not prevent WAV generation.

## Smoke check

A short English request was sent through the running WebUI using the bundled `examples/voice_03.wav` prompt. The response was a valid PCM-16 WAV at 22,050 Hz, and GPU verification reported the RX 9060 XT. Model checkpoints are downloaded separately into `checkpoints/`; do not commit the model weights or local generated audio to the fork.
