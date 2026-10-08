"""把最后一阶段 LoRA adapter 合并回底座权重。"""  # 模块说明

from __future__ import annotations  # 允许延后解析类型注解

import argparse  # 解析命令行参数

from peft import PeftModel  # 加载 LoRA adapter
from transformers import AutoModelForCausalLM  # 加载底座因果语言模型

from .common import (  # 复用公共工具
    load_config,  # 读 YAML 配置
    load_tokenizer,  # 加载分词器
    project_path,  # 相对路径转仓库路径
    resolve_device,  # 解析 cpu/cuda
    resolve_dtype,  # 解析精度
    stage_dir,  # 阶段输出目录
)  # import 结束


def main() -> None:  # 命令行入口
    """默认合并 GRPO adapter，输出到 outputs/<cfg>/merged。"""  # 函数说明
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument("--config", required=True)  # 必填：配置文件
    parser.add_argument("--adapter")  # 可选：指定 adapter 目录
    parser.add_argument("--output")  # 可选：指定合并输出目录
    args = parser.parse_args()  # 解析参数
    config = load_config(args.config)  # 读取 YAML
    device = resolve_device(config)  # 解析设备（合并时主要用来选 dtype）
    dtype = resolve_dtype(config.get("dtype", "auto"), device)  # 解析权重精度
    adapter = project_path(args.adapter) if args.adapter else stage_dir(config, "grpo")  # 默认用 GRPO adapter
    output = (  # 计算输出目录
        project_path(args.output)  # 用户指定了就用指定路径
        if args.output  # 判断是否传了 --output
        else project_path(config["output_root"]) / "merged"  # 否则落到 merged
    )  # output 赋值结束
    base = AutoModelForCausalLM.from_pretrained(  # 加载底座
        config["model_name"], dtype=dtype, low_cpu_mem_usage=True  # 按配置名和精度加载
    )  # from_pretrained 结束
    model = PeftModel.from_pretrained(base, adapter)  # 把 LoRA 挂到底座上
    model = model.merge_and_unload()  # 把 LoRA 增量写回底座并卸掉 adapter
    output.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在
    model.save_pretrained(output, safe_serialization=True)  # 保存完整模型
    load_tokenizer(config["model_name"]).save_pretrained(output)  # 同时保存分词器
    print(f"merged={output}")  # 打印产物路径


if __name__ == "__main__":  # 直接运行本文件时
    main()  # 执行合并
