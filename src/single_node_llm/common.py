"""训练与推理共用的配置、模型加载和张量工具。"""  # 模块说明

from __future__ import annotations  # 允许延后解析类型注解

import json  # 解析 JSONL
import random  # 设置 Python 随机种子
from pathlib import Path  # 路径处理
from typing import Any  # 配置字典的宽松类型

import torch  # 张量和设备
import yaml  # 读取 YAML 配置
from peft import LoraConfig, PeftModel, get_peft_model  # LoRA 配置与包装
from transformers import AutoModelForCausalLM, AutoTokenizer  # HF 模型与分词器


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 仓库根目录：向上两级


def load_config(path: str | Path) -> dict[str, Any]:  # 读取训练配置
    """读取 YAML 配置，并记下原始路径便于排查。"""  # 函数说明
    config_path = Path(path).resolve()  # 转成绝对路径
    with config_path.open("r", encoding="utf-8") as handle:  # 以 UTF-8 打开
        config = yaml.safe_load(handle)  # 解析 YAML 成 dict
    config["_config_path"] = str(config_path)  # 把路径塞进配置方便日志
    return config  # 返回配置


def project_path(value: str | Path) -> Path:  # 相对路径按仓库根解析
    """相对路径按仓库根目录解析，绝对路径原样返回。"""  # 函数说明
    path = Path(value)  # 先包成 Path
    return path if path.is_absolute() else PROJECT_ROOT / path  # 相对则拼到仓库根


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:  # 读 JSONL 数据集
    """逐行读取 JSONL，跳过空行。"""  # 函数说明
    with project_path(path).open("r", encoding="utf-8") as handle:  # 打开数据文件
        return [json.loads(line) for line in handle if line.strip()]  # 跳过空行并解析


def set_seed(seed: int) -> None:  # 固定随机性
    """固定 Python 与 PyTorch 随机种子，方便复现。"""  # 函数说明
    random.seed(seed)  # Python 随机库
    torch.manual_seed(seed)  # CPU/默认生成器
    if torch.cuda.is_available():  # 有 GPU 时
        torch.cuda.manual_seed_all(seed)  # 所有 CUDA 设备


def resolve_device(config: dict[str, Any]) -> torch.device:  # 解析运行设备
    """把配置里的 device 解析成实际设备。auto 表示有 CUDA 就用 GPU。"""  # 函数说明
    requested = config.get("device", "auto")  # 没写就当 auto
    if requested == "auto":  # 自动选择
        requested = "cuda" if torch.cuda.is_available() else "cpu"  # 有卡用卡
    if requested == "cuda" and not torch.cuda.is_available():  # 配了 CUDA 却没卡
        raise RuntimeError("配置要求 CUDA，但 PyTorch 未检测到 CUDA GPU")  # 直接报错
    return torch.device(requested)  # 返回 device 对象


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:  # 解析精度
    """解析训练/推理精度。CPU 上默认 float32，GPU 上默认 bfloat16。"""  # 函数说明
    if name == "auto":  # 自动精度
        return torch.bfloat16 if device.type == "cuda" else torch.float32  # GPU 用 bf16
    mapping = {  # 名字到 dtype
        "float32": torch.float32,  # 全精度
        "float16": torch.float16,  # 半精度
        "bfloat16": torch.bfloat16,  # bf16
    }  # mapping 结束
    if name not in mapping:  # 未知名字
        raise ValueError(f"不支持的 dtype: {name}")  # 报错
    if device.type == "cpu" and name == "float16":  # CPU 不适合 fp16 训练
        raise ValueError("CPU 训练不能使用 float16，请使用 float32")  # 报错
    return mapping[name]  # 返回对应 dtype


def load_tokenizer(model_name: str):  # 加载分词器
    """加载分词器，并补齐 pad_token，训练时从右侧 padding。"""  # 函数说明
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)  # 从 HF 拉分词器
    if tokenizer.pad_token_id is None:  # 有的模型没有 pad
        tokenizer.pad_token = tokenizer.eos_token  # 用 eos 充当 pad
    tokenizer.padding_side = "right"  # 训练时右侧补齐
    return tokenizer  # 返回分词器


def load_model(  # 加载底座并按需挂 LoRA
    config: dict[str, Any],  # YAML 配置
    adapter_path: str | Path | None = None,  # 上一阶段 adapter，可空
    trainable: bool = True,  # True 表示要训练 LoRA
):  # 函数签名结束
    """加载底座模型，并按需挂上已有 LoRA 或新建 LoRA。

    - 有 adapter_path：从上一阶段继续训练或冻结推理。
    - 无 adapter_path 且 trainable：新建 LoRA。
    - 无 adapter_path 且不可训练：返回裸底座。
    """  # 函数说明结束
    device = resolve_device(config)  # 解析设备
    dtype = resolve_dtype(config.get("dtype", "auto"), device)  # 解析精度
    model = AutoModelForCausalLM.from_pretrained(  # 加载因果语言模型
        config["model_name"],  # HF 仓库名或本地路径
        dtype=dtype,  # 权重精度
        trust_remote_code=False,  # 不执行远程自定义代码
        low_cpu_mem_usage=True,  # 降低加载时 CPU 内存峰值
    )  # from_pretrained 结束
    model.to(device)  # 搬到目标设备
    if adapter_path:  # 有现成 adapter
        model = PeftModel.from_pretrained(  # 挂上已有 LoRA
            model,  # 底座
            str(project_path(adapter_path)),  # adapter 目录
            is_trainable=trainable,  # 是否继续训练
        )  # from_pretrained 结束
    elif trainable:  # 没有 adapter 但要训练
        lora = config["lora"]  # 读取 LoRA 超参
        model = get_peft_model(  # 注入新 LoRA
            model,  # 底座
            LoraConfig(  # LoRA 配置
                task_type="CAUSAL_LM",  # 因果语言模型任务
                r=int(lora["rank"]),  # 秩
                lora_alpha=int(lora["alpha"]),  # 缩放系数
                lora_dropout=float(lora["dropout"]),  # dropout
                target_modules=list(lora["target_modules"]),  # 插入哪些线性层
            ),  # LoraConfig 结束
        )  # get_peft_model 结束
    if config.get("gradient_checkpointing") and trainable:  # 开了检查点且要训练
        model.enable_input_require_grads()  # 让输入也需要梯度，配合 checkpoint
        model.gradient_checkpointing_enable()  # 用重算激活换显存
        model.config.use_cache = False  # 关掉 KV cache，避免冲突
    return model, device  # 返回模型和设备


def save_adapter(model, tokenizer, output_dir: str | Path) -> Path:  # 保存 adapter
    """把 LoRA adapter 和 tokenizer 一起落到指定目录。"""  # 函数说明
    target = project_path(output_dir)  # 解析输出目录
    target.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    model.save_pretrained(target, safe_serialization=True)  # 保存 LoRA 权重
    tokenizer.save_pretrained(target)  # 保存分词器文件
    return target  # 返回目录


def stage_dir(config: dict[str, Any], stage: str) -> Path:  # 阶段产物路径
    """某训练阶段 adapter 的默认输出目录。"""  # 函数说明
    return project_path(config["output_root"]) / stage  # output_root/stage


def trainable_parameters(model) -> list[torch.nn.Parameter]:  # 收集可训练参数
    """收集 requires_grad=True 的参数，供优化器使用。"""  # 函数说明
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]  # 只要可训练的
    if not params:  # 一个都没有说明 LoRA 没挂上
        raise RuntimeError("模型没有可训练参数")  # 报错
    return params  # 返回参数列表


def pad_batch(sequences: list[list[int]], pad_id: int, device: torch.device):  # 把变长序列 pad 成 batch
    """把不等长 token 序列 pad 成 batch，并构造 attention_mask。"""  # 函数说明
    max_len = max(len(sequence) for sequence in sequences)  # 本 batch 最长长度
    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)  # 先填 pad
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)  # 掩码先全 0
    for index, sequence in enumerate(sequences):  # 逐条写入真实 token
        input_ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)  # 写入 ids
        attention_mask[index, : len(sequence)] = 1  # 有效位置标 1
    return input_ids.to(device), attention_mask.to(device)  # 搬到设备


def completion_logps(model, input_ids, attention_mask, prompt_lengths):  # 算 completion 对数概率
    """计算 prompt 之后生成 token 的对数概率和。

    DPO/GRPO 只关心 completion 部分：prompt 位置被掩掉。
    返回 (每条序列的总 logp, 有效 token 数)。
    """  # 函数说明结束
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]  # 下一步预测，去掉最后一位
    labels = input_ids[:, 1:]  # 标签是右移一位的 token
    token_logps = torch.log_softmax(logits.float(), dim=-1).gather(  # 取标签 token 的 log 概率
        -1, labels.unsqueeze(-1)  # 在词表维 gather
    ).squeeze(-1)  # 去掉多余维
    positions = torch.arange(labels.shape[1], device=input_ids.device).unsqueeze(0)  # token 位置 0..L-1
    prompt_starts = torch.tensor(prompt_lengths, device=input_ids.device).unsqueeze(1) - 1  # prompt 结束位置（对齐 labels）
    mask = (positions >= prompt_starts) & attention_mask[:, 1:].bool()  # 只保留 completion 且非 pad
    return (token_logps * mask).sum(-1), mask.sum(-1).clamp_min(1)  # 总 logp 和有效长度（至少 1）
