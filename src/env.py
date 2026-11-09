"""Environment checks and version logging (Colab T4)."""
import json
import platform
from pathlib import Path


def check_environment(expect_gpu="T4", require_gpu=True, save_to=None):
    import torch
    import transformers

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 1)
        if expect_gpu and expect_gpu not in props.name:
            print(f"WARNING: expected a {expect_gpu}, found {props.name}. "
                  "Results stay valid, but runtime/memory assumptions may differ.")
    elif require_gpu:
        raise RuntimeError("No GPU found. In Colab: Runtime > Change runtime type > T4 GPU.")

    print(json.dumps(info, indent=2))
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        Path(save_to).write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
