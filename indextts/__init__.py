# AMD Windows PyTorch builds may omit distributed support. IndexTTS inference
# is single-process; provide enum names required by optional DDP helpers.
import sys as _sys
import types as _types
import torch.distributed as _torch_dist
if not hasattr(_torch_dist, "ReduceOp"):
    # Mark DDP unavailable and stub its optional tensor API so Transformers can import.
    _torch_dist.is_available = lambda: False
    _dtensor = _types.ModuleType("torch.distributed.tensor")
    _dtensor.DTensor = type("DTensor", (), {})
    _sys.modules["torch.distributed.tensor"] = _dtensor
    _torch_dist.tensor = _dtensor
    from enum import IntEnum as _IntEnum
    class _InferenceReduceOp(_IntEnum):
        SUM = 0
        AVG = 1
    _torch_dist.ReduceOp = _InferenceReduceOp
