"""手写训练循环：pt / sft / dpo / grpo 四个阶段。"""  # 模块说明

from __future__ import annotations  # 允许延后解析类型注解

import argparse  # 解析 --config/--stage 等参数
import json  # GRPO 奖励里检查 JSON 是否合法
from pathlib import Path  # 检查上一阶段 adapter 是否存在

import torch  # 张量、优化器、设备
import torch.nn.functional as F  # DPO 的 logsigmoid

from .common import (  # 公共工具
    completion_logps,  # 计算 completion 对数概率
    load_config,  # 读 YAML
    load_model,  # 加载模型+LoRA
    load_tokenizer,  # 加载分词器
    pad_batch,  # 变长序列 pad
    project_path,  # 路径解析
    read_jsonl,  # 读 JSONL
    save_adapter,  # 保存 LoRA
    set_seed,  # 固定种子
    stage_dir,  # 阶段目录
    trainable_parameters,  # 可训练参数
)  # import 结束


def causal_examples(config, tokenizer, stage: str) -> list[list[int]]:  # 准备 pt/sft 的 token 序列
    """把 pt 文本或 sft 对话转成截断后的 input_ids。"""  # 函数说明
    max_length = int(config["max_length"])  # 截断长度
    if stage == "pt":  # 继续预训练读纯文本
        text = project_path("data/pretrain.txt").read_text(encoding="utf-8")  # 读领域文本
        texts = [line.strip() for line in text.splitlines() if line.strip()]  # 每行一段
    else:  # sft 读对话 JSONL
        records = read_jsonl("data/sft.jsonl")  # 读指令数据
        texts = [  # 用 chat template 拼成模型看到的对话文本
            tokenizer.apply_chat_template(  # 套对话模板
                record["messages"], tokenize=False, add_generation_prompt=False  # 不追加 assistant 前缀
            )  # apply_chat_template 结束
            for record in records  # 每条样本
        ]  # texts 结束
    return [  # tokenize 成 ids
        tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=True)[  # 截断并加特殊符号
            "input_ids"  # 只要 token id
        ]  # 下标结束
        for text in texts  # 每段文本
    ]  # 返回列表


def train_causal(config, stage: str, input_adapter: str | None) -> Path:  # pt/sft 训练
    """标准因果语言模型训练，用于继续预训练和 SFT。"""  # 函数说明
    tokenizer = load_tokenizer(config["model_name"])  # 加载分词器
    model, device = load_model(config, input_adapter, trainable=True)  # 加载可训练 LoRA
    model.train()  # 进入训练模式
    examples = causal_examples(config, tokenizer, stage)  # 准备样本
    train_cfg = config["train"]  # 训练超参
    optimizer = torch.optim.AdamW(  # AdamW 优化器
        trainable_parameters(model), lr=float(train_cfg["learning_rate"])  # 只更新 LoRA
    )  # optimizer 结束
    grad_accum = int(train_cfg["gradient_accumulation_steps"])  # 梯度累积步数
    max_steps = int(train_cfg["max_steps"])  # 优化器更新次数
    optimizer.zero_grad(set_to_none=True)  # 清空梯度
    for micro_step in range(max_steps * grad_accum):  # 微步循环
        ids = examples[micro_step % len(examples)]  # 样例少就循环复用
        input_ids, attention_mask = pad_batch([ids], tokenizer.pad_token_id, device)  # 组成 batch=1
        labels = input_ids.clone()  # 标签先复制 ids
        labels[attention_mask == 0] = -100  # pad 位置不参与损失
        loss = model(  # 前向：标准 causal LM loss
            input_ids=input_ids, attention_mask=attention_mask, labels=labels  # 传入 ids/mask/labels
        ).loss  # 取出标量损失
        (loss / grad_accum).backward()  # 按累积步数缩放后再反传
        if (micro_step + 1) % grad_accum == 0:  # 凑够一次优化器步进
            optimizer.step()  # 更新参数
            optimizer.zero_grad(set_to_none=True)  # 清空梯度
            step = (micro_step + 1) // grad_accum  # 当前优化步
            if step % int(train_cfg["log_every"]) == 0:  # 到了日志间隔
                print(f"stage={stage} step={step}/{max_steps} loss={loss.item():.6f}")  # 打印 loss
    return save_adapter(model, tokenizer, stage_dir(config, stage))  # 保存本阶段 adapter


def preference_tokens(tokenizer, prompt: str, answer: str, max_length: int):  # DPO 样本编码
    """构造 DPO 用的 prompt+answer token，并返回 prompt 长度。"""  # 函数说明
    prompt_text = tokenizer.apply_chat_template(  # 把 user prompt 套模板
        [{"role": "user", "content": prompt}],  # 单轮 user
        tokenize=False,  # 先拿字符串
        add_generation_prompt=True,  # 追加 assistant 前缀
    )  # apply_chat_template 结束
    prompt_ids = tokenizer(  # 只编码 prompt
        prompt_text, add_special_tokens=False, truncation=True, max_length=max_length  # 不再重复加 bos
    )["input_ids"]  # 取出 ids
    full_ids = tokenizer(  # 编码 prompt+答案+eos
        prompt_text + answer + tokenizer.eos_token,  # 拼接完整序列
        add_special_tokens=False,  # 不重复特殊符号
        truncation=True,  # 超长截断
        max_length=max_length,  # 上限
    )["input_ids"]  # 取出 ids
    return full_ids, min(len(prompt_ids), len(full_ids) - 1)  # 保证 prompt 长度至少留 1 个 completion token


def train_dpo(config, input_adapter: str) -> Path:  # DPO 训练
    """直接偏好优化：让 chosen 相对 rejected 的优势大于参考模型。"""  # 函数说明
    tokenizer = load_tokenizer(config["model_name"])  # 分词器
    policy, device = load_model(config, input_adapter, trainable=True)  # 可训练策略模型
    reference, _ = load_model(config, input_adapter, trainable=False)  # 冻结参考模型
    policy.train()  # 策略训练模式
    reference.eval()  # 参考模型评估模式
    records = read_jsonl("data/preferences.jsonl")  # 偏好对
    optimizer = torch.optim.AdamW(  # 优化器
        trainable_parameters(policy), lr=float(config["train"]["learning_rate"])  # 只训策略 LoRA
    )  # optimizer 结束
    max_steps = int(config["train"]["max_steps"])  # 步数
    beta = float(config["dpo"]["beta"])  # DPO 温度系数
    for step in range(1, max_steps + 1):  # 逐步训练
        record = records[(step - 1) % len(records)]  # 循环取偏好样本
        chosen, chosen_prompt = preference_tokens(  # 编码优选答案
            tokenizer, record["prompt"], record["chosen"], int(config["max_length"])  # prompt+chosen
        )  # preference_tokens 结束
        rejected, rejected_prompt = preference_tokens(  # 编码拒绝答案
            tokenizer, record["prompt"], record["rejected"], int(config["max_length"])  # prompt+rejected
        )  # preference_tokens 结束
        input_ids, attention_mask = pad_batch(  # 两条拼成一个 batch
            [chosen, rejected], tokenizer.pad_token_id, device  # chosen 在前 rejected 在后
        )  # pad_batch 结束
        prompt_lengths = [chosen_prompt, rejected_prompt]  # 两条各自的 prompt 长度
        policy_logps, _ = completion_logps(  # 策略模型 completion logp
            policy, input_ids, attention_mask, prompt_lengths  # 传入 batch
        )  # completion_logps 结束
        with torch.no_grad():  # 参考模型不反传
            reference_logps, _ = completion_logps(  # 参考模型 completion logp
                reference, input_ids, attention_mask, prompt_lengths  # 同一批输入
            )  # completion_logps 结束
        policy_ratio = policy_logps[0] - policy_logps[1]  # 策略上 chosen 相对 rejected
        reference_ratio = reference_logps[0] - reference_logps[1]  # 参考模型同样差值
        loss = -F.logsigmoid(beta * (policy_ratio - reference_ratio))  # DPO 损失
        optimizer.zero_grad(set_to_none=True)  # 清梯度
        loss.backward()  # 反传
        optimizer.step()  # 更新
        if step % int(config["train"]["log_every"]) == 0:  # 日志间隔
            accuracy = float((policy_ratio > reference_ratio).item())  # 本步是否排对
            print(  # 打印 DPO 日志
                f"stage=dpo step={step}/{max_steps} loss={loss.item():.6f} "  # loss
                f"preference_accuracy={accuracy:.0f}"  # 0 或 1
            )  # print 结束
    del reference  # 释放参考模型
    return save_adapter(policy, tokenizer, stage_dir(config, "dpo"))  # 保存 DPO adapter


def reward_completion(text: str, record: dict) -> float:  # GRPO 规则奖励
    """规则奖励：关键词命中加分，JSON 合法再加分。"""  # 函数说明
    cleaned = text.strip()  # 去掉首尾空白
    if cleaned.startswith("```"):  # 模型可能包了代码块
        cleaned = cleaned.strip("`")  # 去掉反引号
        if cleaned.startswith("json"):  # ```json 前缀
            cleaned = cleaned[4:].strip()  # 去掉 json 标记
    reward = sum(  # 关键词命中数
        1.0 for term in record.get("required_terms", []) if term.lower() in cleaned.lower()  # 大小写不敏感
    )  # sum 结束
    if record.get("require_json"):  # 要求输出 JSON
        try:  # 尝试解析
            json.loads(cleaned)  # 合法 JSON
            reward += 2.0  # 加分
        except json.JSONDecodeError:  # 解析失败
            reward -= 1.0  # 扣分
    return reward  # 返回标量奖励


def train_grpo(config, input_adapter: str) -> Path:  # GRPO 训练
    """组相对策略优化：同 prompt 多样本，组内标准化奖励并加 KL 约束。"""  # 函数说明
    tokenizer = load_tokenizer(config["model_name"])  # 分词器
    policy, device = load_model(config, input_adapter, trainable=True)  # 策略模型
    reference, _ = load_model(config, input_adapter, trainable=False)  # 冻结参考模型
    policy.train()  # 策略先标成训练
    reference.eval()  # 参考模型评估
    records = read_jsonl("data/grpo.jsonl")  # 带奖励规则的提示
    optimizer = torch.optim.AdamW(  # 优化器
        trainable_parameters(policy), lr=float(config["train"]["learning_rate"])  # 只训策略
    )  # optimizer 结束
    grpo = config["grpo"]  # GRPO 超参
    group_size = int(grpo["num_generations"])  # 每组采样条数 G
    max_steps = int(config["train"]["max_steps"])  # 步数
    for step in range(1, max_steps + 1):  # 逐步训练
        record = records[(step - 1) % len(records)]  # 循环取样本
        prompt_text = tokenizer.apply_chat_template(  # 套对话模板
            [{"role": "user", "content": record["prompt"]}],  # 单轮 user
            tokenize=False,  # 先拿字符串
            add_generation_prompt=True,  # 追加 assistant 前缀
        )  # apply_chat_template 结束
        prompt_ids = tokenizer(  # 编码 prompt
            prompt_text,  # 模板后的文本
            add_special_tokens=False,  # 不重复特殊符号
            truncation=True,  # 截断
            max_length=int(config["max_length"]) - int(grpo["max_new_tokens"]),  # 给生成留位置
        )["input_ids"]  # 取出 ids
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # batch=1 的 prompt
        prompt_mask = torch.ones_like(prompt_tensor)  # prompt 全有效
        policy.eval()  # 采样时关掉 dropout
        with torch.no_grad():  # 采样不需要梯度
            generated = policy.generate(  # 同 prompt 采 G 条
                input_ids=prompt_tensor,  # 输入
                attention_mask=prompt_mask,  # 掩码
                do_sample=True,  # 随机采样
                temperature=float(grpo["temperature"]),  # 温度
                top_p=0.95,  # nucleus
                num_return_sequences=group_size,  # 组大小
                max_new_tokens=int(grpo["max_new_tokens"]),  # 生成上限
                pad_token_id=tokenizer.pad_token_id,  # pad
                eos_token_id=tokenizer.eos_token_id,  # eos
            )  # generate 结束
        completions = generated[:, len(prompt_ids) :]  # 切出新生成部分
        texts = tokenizer.batch_decode(completions, skip_special_tokens=True)  # 解码成字符串
        rewards = torch.tensor(  # 规则打分
            [reward_completion(text, record) for text in texts],  # 每条一个奖励
            dtype=torch.float32,  # 浮点
            device=device,  # 放到同一设备
        )  # tensor 结束
        advantages = rewards - rewards.mean()  # 减去组内均值
        std = rewards.std(unbiased=False)  # 组内标准差
        if std > 1e-6:  # 避免除 0
            advantages = advantages / std  # 标准化成 advantage
        attention_mask = (generated != tokenizer.pad_token_id).long()  # 生成序列的有效位置
        prompt_lengths = [len(prompt_ids)] * group_size  # 每条 prompt 一样长
        policy.train()  # 切回训练模式算梯度
        policy_logps, token_counts = completion_logps(  # 策略对生成内容的 logp
            policy, generated, attention_mask, prompt_lengths  # 完整序列
        )  # completion_logps 结束
        with torch.no_grad():  # 参考模型不反传
            reference_logps, _ = completion_logps(  # 参考模型 logp
                reference, generated, attention_mask, prompt_lengths  # 同一批生成
            )  # completion_logps 结束
        policy_mean = policy_logps / token_counts  # 变成 per-token 平均
        reference_mean = reference_logps / token_counts  # 参考模型同样平均
        log_ratio = reference_mean - policy_mean  # x = log ref - log pi
        kl = torch.exp(log_ratio) - log_ratio - 1.0  # KL 近似 exp(x)-x-1
        loss = -(advantages * policy_mean - float(grpo["beta"]) * kl).mean()  # 强化高奖励并惩罚偏离
        optimizer.zero_grad(set_to_none=True)  # 清梯度
        loss.backward()  # 反传
        optimizer.step()  # 更新
        if step % int(config["train"]["log_every"]) == 0:  # 日志间隔
            print(  # 打印奖励和样例
                f"stage=grpo step={step}/{max_steps} loss={loss.item():.6f} "  # loss
                f"rewards={rewards.tolist()} samples={texts}"  # 奖励和文本
            )  # print 结束
    del reference  # 释放参考模型
    return save_adapter(policy, tokenizer, stage_dir(config, "grpo"))  # 保存 GRPO adapter


def default_input_adapter(config, stage: str) -> str | None:  # 默认上一阶段路径
    """默认从前一阶段目录读 adapter：pt -> sft -> dpo -> grpo。"""  # 函数说明
    previous = {"pt": None, "sft": "pt", "dpo": "sft", "grpo": "dpo"}[stage]  # 阶段依赖
    return str(stage_dir(config, previous)) if previous else None  # pt 没有上一阶段


def main() -> None:  # 命令行入口
    parser = argparse.ArgumentParser()  # 参数解析器
    parser.add_argument("--config", required=True)  # 配置文件
    parser.add_argument("--stage", required=True, choices=["pt", "sft", "dpo", "grpo"])  # 训练阶段
    parser.add_argument("--input-adapter")  # 可手动指定上一阶段
    parser.add_argument("--max-steps", type=int)  # 临时覆盖步数
    args = parser.parse_args()  # 解析
    config = load_config(args.config)  # 读配置
    if args.max_steps is not None:  # 命令行覆盖
        config["train"]["max_steps"] = args.max_steps  # 写入配置
    set_seed(int(config["seed"]))  # 固定种子
    input_adapter = args.input_adapter or default_input_adapter(config, args.stage)  # 确定起点 adapter
    if input_adapter and not Path(input_adapter).exists():  # 缺上一阶段产物
        raise FileNotFoundError(  # 提示先跑前一阶段
            f"前一阶段适配器不存在: {input_adapter}。请先运行前一阶段。"  # 中文错误
        )  # FileNotFoundError 结束
    if args.stage in {"pt", "sft"}:  # 因果 LM 两个阶段
        output = train_causal(config, args.stage, input_adapter)  # pt/sft
    elif args.stage == "dpo":  # 偏好对齐
        output = train_dpo(config, input_adapter)  # dpo
    else:  # 剩下就是 grpo
        output = train_grpo(config, input_adapter)  # grpo
    print(f"saved={output}")  # 打印保存路径


if __name__ == "__main__":  # 直接运行本文件
    main()  # 进入入口
