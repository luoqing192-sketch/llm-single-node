"""最小 Agent 循环：网关聊天 + 沙箱工具，直到模型给出最终回答。"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import time
import uuid
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": "在隔离沙箱中执行 Python 代码，返回 stdout/stderr。不要访问网络或本地主机文件。",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "完整 Python 源码"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算只含数字和加减乘除括号的表达式。",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "你是单机实验室里的工具 Agent。"
    "需要计算或执行代码时调用工具；得到结果后用中文给出简洁最终答案。"
    "不要假装已经执行过代码。"
)

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval_ast(node):
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_ast(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    value = _eval_ast(tree)
    return str(value)


app = FastAPI(title="Single Node LLM Agent")
STATE: dict[str, Any] = {}


class AgentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    model: str | None = None
    max_steps: int = Field(default=6, ge=1, le=12)


def parse_args_json(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return value if isinstance(value, dict) else {"value": value}


async def call_sandbox(code: str) -> str:
    client: httpx.AsyncClient = STATE["client"]
    response = await client.post(
        f"{STATE['sandbox_url']}/exec",
        json={"code": code, "timeout_sec": 5},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    return json.dumps(data, ensure_ascii=False)


async def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    if name == "calculator":
        try:
            return calculator(str(arguments.get("expression", "")))
        except Exception as exc:  # noqa: BLE001
            return f"calculator error: {exc}"
    if name == "python_exec":
        code = str(arguments.get("code", ""))
        if not code.strip():
            return "python_exec error: empty code"
        try:
            return await call_sandbox(code)
        except Exception as exc:  # noqa: BLE001
            return f"sandbox error: {exc}"
    return f"unknown tool: {name}"


async def chat_once(model: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    client: httpx.AsyncClient = STATE["client"]
    response = await client.post(
        f"{STATE['gateway_url']}/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "tools": TOOLS,
            "temperature": 0.2,
            "max_tokens": 512,
        },
        timeout=120,
    )
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        raise HTTPException(status_code=502, detail="empty model choices")
    return choices[0].get("message") or {}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gateway": STATE["gateway_url"],
        "sandbox": STATE["sandbox_url"],
        "default_model": STATE["default_model"],
    }


@app.post("/v1/agent/run")
async def run_agent(request: AgentRequest):
    model = request.model or STATE["default_model"]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": request.message},
    ]
    trace: list[dict[str, Any]] = []
    final_text = ""
    for step in range(1, request.max_steps + 1):
        message = await chat_once(model, messages)
        messages.append(message)
        tool_calls = message.get("tool_calls") or []
        content = message.get("content")
        trace.append({"step": step, "assistant": message})
        if not tool_calls:
            final_text = content or ""
            break
        for call in tool_calls:
            function = call.get("function") or {}
            name = function.get("name") or ""
            arguments = parse_args_json(function.get("arguments") or "")
            result = await execute_tool(name, arguments)
            tool_msg = {
                "role": "tool",
                "tool_call_id": call.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                "name": name,
                "content": result,
            }
            messages.append(tool_msg)
            trace.append({"step": step, "tool": name, "arguments": arguments, "result": result})
    else:
        final_text = final_text or "达到最大步数，未得到最终回答"
    return {
        "id": f"agent-{uuid.uuid4().hex}",
        "model": model,
        "output": final_text,
        "trace": trace,
        "created": int(time.time()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--gateway-url", default=os.getenv("GATEWAY_URL", "http://127.0.0.1:8080/v1"))
    parser.add_argument("--sandbox-url", default=os.getenv("SANDBOX_URL", "http://127.0.0.1:8090"))
    parser.add_argument("--default-model", default=os.getenv("AGENT_MODEL", "agent-default"))
    args = parser.parse_args()
    STATE["gateway_url"] = args.gateway_url.rstrip("/")
    STATE["sandbox_url"] = args.sandbox_url.rstrip("/")
    STATE["default_model"] = args.default_model
    STATE["client"] = httpx.AsyncClient()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
