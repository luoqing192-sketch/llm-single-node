"""多路由 OpenAI 兼容网关：按 model 把请求转到不同上游。"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .common import project_path


def load_gateway_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def index_routes(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    routes = {}
    for item in config.get("routes", []):
        routes[item["id"]] = item
    if not routes:
        raise RuntimeError("gateway.yaml 里至少需要一条 route")
    return routes


app = FastAPI(title="Single Node LLM Gateway")
STATE: dict[str, Any] = {}


def get_route(model: str) -> dict[str, Any]:
    routes: dict[str, dict[str, Any]] = STATE["routes"]
    if model in routes:
        return routes[model]
    raise HTTPException(status_code=404, detail=f"Unknown model route: {model}")


def echo_completion(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages") or []
    last = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            last = str(message.get("content") or "")
            break
    tools = payload.get("tools") or []
    content = f"[echo-demo] received: {last[:500]}"
    if tools:
        content += f" (tools={len(tools)})"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model", "echo-demo"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def forward_openai(route: dict[str, Any], path: str, payload: dict[str, Any], request: Request):
    upstream = route["upstream"].rstrip("/")
    body = dict(payload)
    if route.get("rewrite_model"):
        body["model"] = route["rewrite_model"]
    headers = {"Content-Type": "application/json"}
    auth = request.headers.get("authorization")
    key_env = route.get("api_key_env")
    if key_env and os.getenv(key_env):
        headers["Authorization"] = f"Bearer {os.getenv(key_env)}"
    elif auth:
        headers["Authorization"] = auth
    timeout = float(STATE["config"].get("request_timeout_sec", 120))
    client: httpx.AsyncClient = STATE["client"]
    url = f"{upstream}{path}"
    if body.get("stream"):
        async def stream():
            async with client.stream("POST", url, json=body, headers=headers, timeout=timeout) as upstream_resp:
                if upstream_resp.status_code >= 400:
                    text = await upstream_resp.aread()
                    raise HTTPException(status_code=upstream_resp.status_code, detail=text.decode("utf-8", "replace"))
                async for chunk in upstream_resp.aiter_bytes():
                    yield chunk
        return StreamingResponse(stream(), media_type="text/event-stream")
    response = await client.post(url, json=body, headers=headers, timeout=timeout)
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return JSONResponse(response.json())


@app.get("/health")
def health():
    return {
        "status": "ok",
        "routes": list(STATE["routes"].keys()),
    }


@app.get("/v1/models")
def models():
    data = [{"id": route_id, "object": "model", "owned_by": "gateway"} for route_id in STATE["routes"]]
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    payload = await request.json()
    model = payload.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="missing model")
    route = get_route(str(model))
    if route.get("type") == "echo":
        result = echo_completion(payload)
        if payload.get("stream"):
            async def events():
                chunk = {
                    "id": result["id"],
                    "object": "chat.completion.chunk",
                    "created": result["created"],
                    "model": model,
                    "choices": [{"index": 0, "delta": result["choices"][0]["message"], "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")
        return result
    if route.get("type") == "openai":
        return await forward_openai(route, "/chat/completions", payload, request)
    raise HTTPException(status_code=500, detail=f"unsupported route type: {route.get('type')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gateway.yaml")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    config = load_gateway_config(project_path(args.config))
    STATE["config"] = config
    STATE["routes"] = index_routes(config)
    STATE["client"] = httpx.AsyncClient()
    host = args.host or config.get("listen_host", "0.0.0.0")
    port = args.port or int(config.get("listen_port", 8080))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
