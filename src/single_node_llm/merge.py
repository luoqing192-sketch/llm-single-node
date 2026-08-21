from __future__ import annotations

import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM

from .common import (
    load_config,
    load_tokenizer,
    project_path,
    resolve_device,
    resolve_dtype,
    stage_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    device = resolve_device(config)
    dtype = resolve_dtype(config.get("dtype", "auto"), device)
    adapter = project_path(args.adapter) if args.adapter else stage_dir(config, "grpo")
    output = (
        project_path(args.output)
        if args.output
        else project_path(config["output_root"]) / "merged"
    )
    base = AutoModelForCausalLM.from_pretrained(
        config["model_name"], dtype=dtype, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base, adapter)
    model = model.merge_and_unload()
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    load_tokenizer(config["model_name"]).save_pretrained(output)
    print(f"merged={output}")


if __name__ == "__main__":
    main()
