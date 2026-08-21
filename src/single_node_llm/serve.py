from __future__ import annotations

import argparse
import asyncio
import json
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = Field(default=256, ge=1, le=4096)


STATE: dict[str, Any] = {}
MODEL_LOCK = threading.Lock()


def parse_tool_calls(text: str):
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    calls = []
    for block in blocks:
        try:
            value = json.loads(block)
        except json.JSONDecodeError:
            continue
        name = value.get("name")
        if not name:
            continue
        arguments = value.get("arguments", {})
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )
    content = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()
    return (content or None), calls


def generate(request: ChatRequest) -> tuple[str | None, list[dict[str, Any]], int, int]:
    tokenizer = STATE["tokenizer"]
    model = STATE["model"]
    device = STATE["device"]
    template_args = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if request.tools:
        template_args["tools"] = request.tools
    try:
        prompt = tokenizer.apply_chat_template(request.messages, **template_args)
    except (TypeError, ValueError):
        prompt = tokenizer.apply_chat_template(
            request.messages, tokenize=False, add_generation_prompt=True
        )
        if request.tools:
            prompt += "\nAvailable tools:\n" + json.dumps(
                request.tools, ensure_ascii=False
            )
    context_window = int(
        getattr(model.config, "max_position_embeddings", STATE["context_window"])
    )
    generation_tokens = min(request.max_tokens, context_window - 1)
    max_prompt_tokens = max(1, context_window - generation_tokens)
    tokenizer.truncation_side = "left"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
    ).to(device)
    do_sample = request.temperature > 0
    with MODEL_LOCK, torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=generation_tokens,
            do_sample=do_sample,
            temperature=request.temperature if do_sample else None,
            top_p=request.top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion_ids = output[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    content, tool_calls = parse_tool_calls(text)
    return content, tool_calls, int(inputs["input_ids"].numel()), int(completion_ids.numel())


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(title="Single Node LLM OpenAI API", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": STATE.get("served_model"),
        "device": str(STATE.get("device")),
    }


@app.get("/v1/models")
def models():
    model_id = STATE["served_model"]
    return {"object": "list", "data": [{"id": model_id, "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    if request.model != STATE["served_model"]:
        raise HTTPException(status_code=404, detail=f"Unknown model: {request.model}")
    content, tool_calls, prompt_tokens, completion_tokens = await asyncio.to_thread(
        generate, request
    )
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    finish_reason = "tool_calls" if tool_calls else "stop"
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if not request.stream:
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def events():
        delta = dict(message)
        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        final = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="local-warehouse-llm")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但 PyTorch 未检测到 CUDA GPU")
    device = torch.device(device_name)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model_path = str(Path(args.model).resolve())
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    STATE.update(
        model=model,
        tokenizer=tokenizer,
        device=device,
        served_model=args.served_model_name,
        context_window=int(getattr(model.config, "max_position_embeddings", 8192)),
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
