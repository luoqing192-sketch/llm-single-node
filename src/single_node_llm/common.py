from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(config_path)
    return config


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with project_path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = config.get("device", "auto")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但 PyTorch 未检测到 CUDA GPU")
    return torch.device(requested)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"不支持的 dtype: {name}")
    if device.type == "cpu" and name == "float16":
        raise ValueError("CPU 训练不能使用 float16，请使用 float32")
    return mapping[name]


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(
    config: dict[str, Any],
    adapter_path: str | Path | None = None,
    trainable: bool = True,
):
    device = resolve_device(config)
    dtype = resolve_dtype(config.get("dtype", "auto"), device)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        dtype=dtype,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    if adapter_path:
        model = PeftModel.from_pretrained(
            model,
            str(project_path(adapter_path)),
            is_trainable=trainable,
        )
    elif trainable:
        lora = config["lora"]
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=int(lora["rank"]),
                lora_alpha=int(lora["alpha"]),
                lora_dropout=float(lora["dropout"]),
                target_modules=list(lora["target_modules"]),
            ),
        )
    if config.get("gradient_checkpointing") and trainable:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model, device


def save_adapter(model, tokenizer, output_dir: str | Path) -> Path:
    target = project_path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    return target


def stage_dir(config: dict[str, Any], stage: str) -> Path:
    return project_path(config["output_root"]) / stage


def trainable_parameters(model) -> list[torch.nn.Parameter]:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise RuntimeError("模型没有可训练参数")
    return params


def pad_batch(sequences: list[list[int]], pad_id: int, device: torch.device):
    max_len = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for index, sequence in enumerate(sequences):
        input_ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[index, : len(sequence)] = 1
    return input_ids.to(device), attention_mask.to(device)


def completion_logps(model, input_ids, attention_mask, prompt_lengths):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
    labels = input_ids[:, 1:]
    token_logps = torch.log_softmax(logits.float(), dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    positions = torch.arange(labels.shape[1], device=input_ids.device).unsqueeze(0)
    prompt_starts = torch.tensor(prompt_lengths, device=input_ids.device).unsqueeze(1) - 1
    mask = (positions >= prompt_starts) & attention_mask[:, 1:].bool()
    return (token_logps * mask).sum(-1), mask.sum(-1).clamp_min(1)
