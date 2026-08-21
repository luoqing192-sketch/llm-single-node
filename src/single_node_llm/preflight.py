from __future__ import annotations

import json
import os
import platform

import torch


def main() -> None:
    memory = None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        memory = round(pages * page_size / 1024**3, 1)
    except (AttributeError, ValueError):
        pass
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory_gib": memory,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "recommendation": (
            "cuda-1.5b.yaml"
            if torch.cuda.is_available()
            else "cpu-smoke.yaml first, then cpu-local.yaml"
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

