"""OpenAI 兼容的本地推理服务：Chat Completions + 伪流式 + 工具调用解析。"""  # 模块说明

from __future__ import annotations  # 允许延后解析类型注解

import argparse  # 解析服务启动参数
import asyncio  # 把阻塞 generate 丢到线程
import json  # 解析工具调用和 SSE JSON
import re  # 用正则抽 <tool_call>
import threading  # 模型推理互斥锁
import time  # 生成 created 时间戳
import uuid  # 生成 completion/tool_call id
from contextlib import asynccontextmanager  # FastAPI lifespan
from pathlib import Path  # 解析模型目录
from typing import Any  # 宽松 JSON 字段

import torch  # 推理张量
import uvicorn  # ASGI 服务器
from fastapi import FastAPI, HTTPException  # HTTP 接口
from fastapi.responses import StreamingResponse  # SSE 流式
from pydantic import BaseModel, Field  # 请求体校验
from transformers import AutoModelForCausalLM, AutoTokenizer  # 加载合并后的模型


class ChatRequest(BaseModel):  # OpenAI Chat Completions 请求
    """对齐 OpenAI Chat Completions 的请求字段。"""  # 类说明

    model: str  # 模型 ID，必须和服务端一致
    messages: list[dict[str, Any]]  # 对话消息
    tools: list[dict[str, Any]] | None = None  # 可选工具 schema
    tool_choice: Any | None = None  # 预留字段，当前不使用
    stream: bool = False  # 是否走伪流式
    temperature: float = 0.7  # 采样温度
    top_p: float = 0.95  # nucleus 采样
    max_tokens: int = Field(default=256, ge=1, le=4096)  # 生成上限


STATE: dict[str, Any] = {}  # 进程内保存模型/分词器/设备
MODEL_LOCK = threading.Lock()  # 一次只允许一个 generate


def parse_tool_calls(text: str):  # 解析 Qwen 风格工具调用
    """从 Qwen 风格 <tool_call> JSON 块解析出 OpenAI tool_calls。"""  # 函数说明
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)  # 抽出 JSON 块
    calls = []  # 收集 tool_calls
    for block in blocks:  # 逐块解析
        try:  # JSON 可能不合法
            value = json.loads(block)  # 解析成 dict
        except json.JSONDecodeError:  # 解析失败
            continue  # 跳过这块
        name = value.get("name")  # 函数名
        if not name:  # 没有名字就不是合法调用
            continue  # 跳过
        arguments = value.get("arguments", {})  # 参数对象
        calls.append(  # 转成 OpenAI 结构
            {  # 一条 tool_call
                "id": f"call_{uuid.uuid4().hex[:24]}",  # 随机 id
                "type": "function",  # 固定类型
                "function": {  # 函数体
                    "name": name,  # 函数名
                    "arguments": json.dumps(arguments, ensure_ascii=False),  # 参数必须是字符串
                },  # function 结束
            }  # dict 结束
        )  # append 结束
    content = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()  # 去掉工具块剩正文
    return (content or None), calls  # 空正文改成 None


def generate(request: ChatRequest) -> tuple[str | None, list[dict[str, Any]], int, int]:  # 同步生成
    """同步生成一条回复。超长输入从左侧截断，给输出预留空间。"""  # 函数说明
    tokenizer = STATE["tokenizer"]  # 取分词器
    model = STATE["model"]  # 取模型
    device = STATE["device"]  # 取设备
    template_args = {  # chat template 参数
        "tokenize": False,  # 先得到字符串
        "add_generation_prompt": True,  # 追加 assistant 前缀
    }  # template_args 结束
    if request.tools:  # 请求带了工具
        template_args["tools"] = request.tools  # 交给 tokenizer 渲染
    try:  # 有的模板不接受 tools
        prompt = tokenizer.apply_chat_template(request.messages, **template_args)  # 套模板
    except (TypeError, ValueError):  # 不支持 tools 参数
        prompt = tokenizer.apply_chat_template(  # 退回普通模板
            request.messages, tokenize=False, add_generation_prompt=True  # 不含 tools
        )  # apply_chat_template 结束
        if request.tools:  # 手工把工具说明拼到末尾
            prompt += "\nAvailable tools:\n" + json.dumps(  # 追加 JSON
                request.tools, ensure_ascii=False  # 保留中文
            )  # dumps 结束
    context_window = int(  # 上下文窗口
        getattr(model.config, "max_position_embeddings", STATE["context_window"])  # 优先读模型配置
    )  # int 结束
    generation_tokens = min(request.max_tokens, context_window - 1)  # 生成长度不能超过窗口
    max_prompt_tokens = max(1, context_window - generation_tokens)  # 给输出留位置
    tokenizer.truncation_side = "left"  # 超长时丢掉更早的历史
    inputs = tokenizer(  # 编码 prompt
        prompt,  # 模板文本
        return_tensors="pt",  # 返回 PyTorch 张量
        truncation=True,  # 允许截断
        max_length=max_prompt_tokens,  # 输入上限
    ).to(device)  # 搬到设备
    do_sample = request.temperature > 0  # 温度为 0 则贪心
    with MODEL_LOCK, torch.inference_mode():  # 串行 + 无梯度
        output = model.generate(  # 生成
            **inputs,  # input_ids 和 attention_mask
            max_new_tokens=generation_tokens,  # 新 token 上限
            do_sample=do_sample,  # 是否采样
            temperature=request.temperature if do_sample else None,  # 贪心时不要温度
            top_p=request.top_p if do_sample else None,  # 贪心时不要 top_p
            pad_token_id=tokenizer.pad_token_id,  # pad
            eos_token_id=tokenizer.eos_token_id,  # eos
        )  # generate 结束
    completion_ids = output[0, inputs["input_ids"].shape[1] :]  # 切出新生成 token
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)  # 解码文本
    content, tool_calls = parse_tool_calls(text)  # 拆正文和工具调用
    return content, tool_calls, int(inputs["input_ids"].numel()), int(completion_ids.numel())  # 再带上 token 统计


@asynccontextmanager  # FastAPI 生命周期
async def lifespan(_: FastAPI):  # 目前没有额外启停逻辑
    yield  # 保持应用运行


app = FastAPI(title="Single Node LLM OpenAI API", lifespan=lifespan)  # 创建应用


@app.get("/health")  # 健康检查
def health():  # 返回服务状态
    return {  # 简单 JSON
        "status": "ok",  # 存活标记
        "model": STATE.get("served_model"),  # 对外模型名
        "device": str(STATE.get("device")),  # 设备字符串
    }  # return 结束


@app.get("/v1/models")  # 模型列表
def models():  # OpenAI /v1/models
    model_id = STATE["served_model"]  # 当前对外 ID
    return {"object": "list", "data": [{"id": model_id, "object": "model"}]}  # 只暴露一个模型


@app.post("/v1/chat/completions")  # 对话补全
async def chat(request: ChatRequest):  # 主接口
    """阻塞推理放到线程池，避免卡住事件循环。stream 是整段算完再按 SSE 吐出。"""  # 函数说明
    if request.model != STATE["served_model"]:  # 模型名对不上
        raise HTTPException(status_code=404, detail=f"Unknown model: {request.model}")  # 404
    content, tool_calls, prompt_tokens, completion_tokens = await asyncio.to_thread(  # 线程里跑 generate
        generate, request  # 传入请求
    )  # to_thread 结束
    response_id = f"chatcmpl-{uuid.uuid4().hex}"  # 补全 id
    created = int(time.time())  # unix 时间
    finish_reason = "tool_calls" if tool_calls else "stop"  # 有工具调用就改 finish_reason
    message = {"role": "assistant", "content": content}  # 助手消息
    if tool_calls:  # 带了工具
        message["tool_calls"] = tool_calls  # 挂到 message 上
    if not request.stream:  # 非流式直接返回完整对象
        return {  # OpenAI 非流式结构
            "id": response_id,  # id
            "object": "chat.completion",  # 对象类型
            "created": created,  # 时间
            "model": request.model,  # 模型名
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],  # 单 choice
            "usage": {  # token 统计
                "prompt_tokens": prompt_tokens,  # 输入
                "completion_tokens": completion_tokens,  # 输出
                "total_tokens": prompt_tokens + completion_tokens,  # 合计
            },  # usage 结束
        }  # return 结束

    async def events():  # SSE 生成器
        delta = dict(message)  # 第一包把整段 message 一次性给出
        chunk = {  # 第一个 chunk
            "id": response_id,  # 同一 id
            "object": "chat.completion.chunk",  # 流式对象
            "created": created,  # 时间
            "model": request.model,  # 模型
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],  # 还没结束
        }  # chunk 结束
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"  # 发第一包
        final = {  # 结束包
            "id": response_id,  # id
            "object": "chat.completion.chunk",  # 类型
            "created": created,  # 时间
            "model": request.model,  # 模型
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],  # 带结束原因
        }  # final 结束
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"  # 发结束包
        yield "data: [DONE]\n\n"  # OpenAI 流结束标记

    return StreamingResponse(events(), media_type="text/event-stream")  # 返回 SSE


def main() -> None:  # 启动服务
    parser = argparse.ArgumentParser()  # 参数解析器
    parser.add_argument("--model", required=True)  # 合并后的模型目录
    parser.add_argument("--served-model-name", default="local-warehouse-llm")  # 对外模型 ID
    parser.add_argument("--host", default="0.0.0.0")  # 监听地址
    parser.add_argument("--port", type=int, default=8000)  # 端口
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")  # 设备
    args = parser.parse_args()  # 解析
    device_name = (  # 先按 auto+CUDA 选择
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device  # auto 且有卡则 cuda
    )  # device_name 结束
    if device_name == "auto":  # auto 但没卡
        device_name = "cpu"  # 回落到 CPU
    if device_name == "cuda" and not torch.cuda.is_available():  # 强制 CUDA 却没卡
        raise RuntimeError("指定了 CUDA，但 PyTorch 未检测到 CUDA GPU")  # 报错
    device = torch.device(device_name)  # 创建设备
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32  # GPU 用 bf16
    model_path = str(Path(args.model).resolve())  # 模型绝对路径
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # 加载分词器
    if tokenizer.pad_token_id is None:  # 没有 pad
        tokenizer.pad_token = tokenizer.eos_token  # 用 eos 顶上
    model = AutoModelForCausalLM.from_pretrained(  # 加载完整模型
        model_path, dtype=dtype, low_cpu_mem_usage=True  # 精度和省内存
    ).to(device)  # 放到设备
    model.eval()  # 推理模式
    STATE.update(  # 填入全局状态
        model=model,  # 模型
        tokenizer=tokenizer,  # 分词器
        device=device,  # 设备
        served_model=args.served_model_name,  # 对外名
        context_window=int(getattr(model.config, "max_position_embeddings", 8192)),  # 窗口
    )  # update 结束
    uvicorn.run(app, host=args.host, port=args.port)  # 阻塞启动 HTTP 服务


if __name__ == "__main__":  # 直接运行本文件
    main()  # 启动
