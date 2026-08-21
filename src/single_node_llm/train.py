from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import (
    completion_logps,
    load_config,
    load_model,
    load_tokenizer,
    pad_batch,
    project_path,
    read_jsonl,
    save_adapter,
    set_seed,
    stage_dir,
    trainable_parameters,
)


def causal_examples(config, tokenizer, stage: str) -> list[list[int]]:
    max_length = int(config["max_length"])
    if stage == "pt":
        text = project_path("data/pretrain.txt").read_text(encoding="utf-8")
        texts = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        records = read_jsonl("data/sft.jsonl")
        texts = [
            tokenizer.apply_chat_template(
                record["messages"], tokenize=False, add_generation_prompt=False
            )
            for record in records
        ]
    return [
        tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=True)[
            "input_ids"
        ]
        for text in texts
    ]


def train_causal(config, stage: str, input_adapter: str | None) -> Path:
    tokenizer = load_tokenizer(config["model_name"])
    model, device = load_model(config, input_adapter, trainable=True)
    model.train()
    examples = causal_examples(config, tokenizer, stage)
    train_cfg = config["train"]
    optimizer = torch.optim.AdamW(
        trainable_parameters(model), lr=float(train_cfg["learning_rate"])
    )
    grad_accum = int(train_cfg["gradient_accumulation_steps"])
    max_steps = int(train_cfg["max_steps"])
    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(max_steps * grad_accum):
        ids = examples[micro_step % len(examples)]
        input_ids, attention_mask = pad_batch([ids], tokenizer.pad_token_id, device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        loss = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).loss
        (loss / grad_accum).backward()
        if (micro_step + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step = (micro_step + 1) // grad_accum
            if step % int(train_cfg["log_every"]) == 0:
                print(f"stage={stage} step={step}/{max_steps} loss={loss.item():.6f}")
    return save_adapter(model, tokenizer, stage_dir(config, stage))


def preference_tokens(tokenizer, prompt: str, answer: str, max_length: int):
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(
        prompt_text, add_special_tokens=False, truncation=True, max_length=max_length
    )["input_ids"]
    full_ids = tokenizer(
        prompt_text + answer + tokenizer.eos_token,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    return full_ids, min(len(prompt_ids), len(full_ids) - 1)


def train_dpo(config, input_adapter: str) -> Path:
    tokenizer = load_tokenizer(config["model_name"])
    policy, device = load_model(config, input_adapter, trainable=True)
    reference, _ = load_model(config, input_adapter, trainable=False)
    policy.train()
    reference.eval()
    records = read_jsonl("data/preferences.jsonl")
    optimizer = torch.optim.AdamW(
        trainable_parameters(policy), lr=float(config["train"]["learning_rate"])
    )
    max_steps = int(config["train"]["max_steps"])
    beta = float(config["dpo"]["beta"])
    for step in range(1, max_steps + 1):
        record = records[(step - 1) % len(records)]
        chosen, chosen_prompt = preference_tokens(
            tokenizer, record["prompt"], record["chosen"], int(config["max_length"])
        )
        rejected, rejected_prompt = preference_tokens(
            tokenizer, record["prompt"], record["rejected"], int(config["max_length"])
        )
        input_ids, attention_mask = pad_batch(
            [chosen, rejected], tokenizer.pad_token_id, device
        )
        prompt_lengths = [chosen_prompt, rejected_prompt]
        policy_logps, _ = completion_logps(
            policy, input_ids, attention_mask, prompt_lengths
        )
        with torch.no_grad():
            reference_logps, _ = completion_logps(
                reference, input_ids, attention_mask, prompt_lengths
            )
        policy_ratio = policy_logps[0] - policy_logps[1]
        reference_ratio = reference_logps[0] - reference_logps[1]
        loss = -F.logsigmoid(beta * (policy_ratio - reference_ratio))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % int(config["train"]["log_every"]) == 0:
            accuracy = float((policy_ratio > reference_ratio).item())
            print(
                f"stage=dpo step={step}/{max_steps} loss={loss.item():.6f} "
                f"preference_accuracy={accuracy:.0f}"
            )
    del reference
    return save_adapter(policy, tokenizer, stage_dir(config, "dpo"))


def reward_completion(text: str, record: dict) -> float:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    reward = sum(
        1.0 for term in record.get("required_terms", []) if term.lower() in cleaned.lower()
    )
    if record.get("require_json"):
        try:
            json.loads(cleaned)
            reward += 2.0
        except json.JSONDecodeError:
            reward -= 1.0
    return reward


def train_grpo(config, input_adapter: str) -> Path:
    tokenizer = load_tokenizer(config["model_name"])
    policy, device = load_model(config, input_adapter, trainable=True)
    reference, _ = load_model(config, input_adapter, trainable=False)
    policy.train()
    reference.eval()
    records = read_jsonl("data/grpo.jsonl")
    optimizer = torch.optim.AdamW(
        trainable_parameters(policy), lr=float(config["train"]["learning_rate"])
    )
    grpo = config["grpo"]
    group_size = int(grpo["num_generations"])
    max_steps = int(config["train"]["max_steps"])
    for step in range(1, max_steps + 1):
        record = records[(step - 1) % len(records)]
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": record["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=int(config["max_length"]) - int(grpo["max_new_tokens"]),
        )["input_ids"]
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        prompt_mask = torch.ones_like(prompt_tensor)
        policy.eval()
        with torch.no_grad():
            generated = policy.generate(
                input_ids=prompt_tensor,
                attention_mask=prompt_mask,
                do_sample=True,
                temperature=float(grpo["temperature"]),
                top_p=0.95,
                num_return_sequences=group_size,
                max_new_tokens=int(grpo["max_new_tokens"]),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completions = generated[:, len(prompt_ids) :]
        texts = tokenizer.batch_decode(completions, skip_special_tokens=True)
        rewards = torch.tensor(
            [reward_completion(text, record) for text in texts],
            dtype=torch.float32,
            device=device,
        )
        advantages = rewards - rewards.mean()
        std = rewards.std(unbiased=False)
        if std > 1e-6:
            advantages = advantages / std
        attention_mask = (generated != tokenizer.pad_token_id).long()
        prompt_lengths = [len(prompt_ids)] * group_size
        policy.train()
        policy_logps, token_counts = completion_logps(
            policy, generated, attention_mask, prompt_lengths
        )
        with torch.no_grad():
            reference_logps, _ = completion_logps(
                reference, generated, attention_mask, prompt_lengths
            )
        policy_mean = policy_logps / token_counts
        reference_mean = reference_logps / token_counts
        log_ratio = reference_mean - policy_mean
        kl = torch.exp(log_ratio) - log_ratio - 1.0
        loss = -(advantages * policy_mean - float(grpo["beta"]) * kl).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % int(config["train"]["log_every"]) == 0:
            print(
                f"stage=grpo step={step}/{max_steps} loss={loss.item():.6f} "
                f"rewards={rewards.tolist()} samples={texts}"
            )
    del reference
    return save_adapter(policy, tokenizer, stage_dir(config, "grpo"))


def default_input_adapter(config, stage: str) -> str | None:
    previous = {"pt": None, "sft": "pt", "dpo": "sft", "grpo": "dpo"}[stage]
    return str(stage_dir(config, previous)) if previous else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=["pt", "sft", "dpo", "grpo"])
    parser.add_argument("--input-adapter")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.max_steps is not None:
        config["train"]["max_steps"] = args.max_steps
    set_seed(int(config["seed"]))
    input_adapter = args.input_adapter or default_input_adapter(config, args.stage)
    if input_adapter and not Path(input_adapter).exists():
        raise FileNotFoundError(
            f"前一阶段适配器不存在: {input_adapter}。请先运行前一阶段。"
        )
    if args.stage in {"pt", "sft"}:
        output = train_causal(config, args.stage, input_adapter)
    elif args.stage == "dpo":
        output = train_dpo(config, input_adapter)
    else:
        output = train_grpo(config, input_adapter)
    print(f"saved={output}")


if __name__ == "__main__":
    main()

