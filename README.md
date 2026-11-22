# IndexTTS on Windows with AMD ROCm

A Windows-focused fork with a tested AMD ROCm PyTorch setup for IndexTTS 2.5. Requires a ROCm-supported AMD GPU with at least 12 GB of VRAM. The steps below were verified on Windows 11 with an AMD Radeon RX 9060 XT 16 GB and Python 3.12.

## Installation

### 1. Create the environment and clone this fork

Use PowerShell. Calling the virtual environment's Python directly avoids PowerShell script-activation policy changes.

```powershell
Set-Location $env:USERPROFILE
git clone https://github.com/macxzatgit/index-tts-windows-rocm.git
cd .\index-tts-windows-rocm
py -3.12 -m venv .venv
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $py -m pip install --upgrade pip
```

### 2. Install AMD ROCm PyTorch

Use AMD's Windows PyTorch wheels for ROCm 7.2.1. AMD lists Windows 11, Python 3.12, and Radeon Software 26.2.2 or newer for this release. Install the ROCm SDK wheels first, followed by PyTorch, TorchAudio, and TorchVision:

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

### 3. Install IndexTTS dependencies

The requirements file keeps the ROCm PyTorch packages intact and pins the Transformers versions tested with this fork.

```powershell
& $py -m pip install -r .\requirements-amd-windows.txt
```

Do not run `uv sync` for this setup: the upstream project configuration selects NVIDIA CUDA PyTorch and declares Python below 3.12.

### 4. Verify GPU detection

```powershell
& $py -c "import torch; print('Torch:', torch.__version__); print('HIP:', torch.version.hip); print('GPU available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Confirm the output shows `GPU available: True` and your Radeon GPU. On a system with an integrated GPU and an eGPU, check that device 0 is the intended Radeon card before launching.

AMD's [Windows PyTorch installation guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/windows/install-pytorch.html) and [GPU support matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html) list the current driver and hardware requirements.

## How to use it

From the repository folder, start the IndexTTS 2.5 WebUI:

```powershell
& .\.venv\Scripts\python.exe .\webui.py --version 2.5 --fp16
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). On first launch, IndexTTS downloads its model checkpoints and example audio files; wait for startup to finish. In the interface, choose a reference voice, enter text, select the language, and generate speech.

For IndexTTS 2.5, `--fp16` enables BF16 when the GPU supports it. To enable emotion-from-text control, add `--qwen_emo`:

```powershell
& .\.venv\Scripts\python.exe .\webui.py --version 2.5 --fp16 --qwen_emo
```

Only run one WebUI instance at a time; each instance loads a separate copy of the models. Stop the server with **Ctrl+C** before restarting it.
