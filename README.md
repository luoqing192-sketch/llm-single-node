# 单机大模型训练实验室

这套工程把一个开源模型依次经过领域继续预训练、SFT、DPO、GRPO、LoRA 合并，再以 OpenAI 兼容接口提供服务。最后可由 DeepSeek Harness 连接。

它用于学习和验证完整训练链路，不声称在消费级单机上从随机参数训练出通用大模型。这里的 `pt` 是在已有模型上做领域继续预训练（DAPT）；真正从零预训练还需要训练 tokenizer、海量语料和多机 GPU 集群。

## 当前机器结论

环境探测结果：Windows + WSL2 Ubuntu 22.04、8 vCPU、约 24 GiB 内存、无 NVIDIA GPU。推荐顺序：

1. 用 `cpu-smoke.yaml` 和 135M 模型验证完整链路。
2. 用 `cpu-local.yaml` 和 Qwen2.5 0.5B 做小规模实验。
3. 增加 NVIDIA 显卡后使用 `cuda-1.5b.yaml` 和 DeepSeek-R1-Distill-Qwen-1.5B。

CPU 上的 0.5B 配置可以运行，但训练和生成会明显较慢。7B 训练不适合当前硬件。

## 一键运行

以下命令在 Windows PowerShell 中执行。

安装独立 Python 环境：

```powershell
wsl -d Ubuntu -- bash /mnt/d/llm_learning/llm-single-node/scripts/bootstrap.sh
```

跑完整的 135M 冒烟流水线：

```powershell
wsl -d Ubuntu -- bash /mnt/d/llm_learning/llm-single-node/scripts/run_pipeline.sh configs/cpu-smoke.yaml
```

第一次运行需要下载约数百 MB 的模型。结果写入：

```text
outputs/cpu-smoke/pt       继续预训练 adapter
outputs/cpu-smoke/sft      SFT adapter
outputs/cpu-smoke/dpo      DPO adapter
outputs/cpu-smoke/grpo     GRPO adapter
outputs/cpu-smoke/merged   合并后的完整模型
```

脚本默认通过当前网络可访问的 `https://hf-mirror.com` 下载公开模型。其他网络环境可在命令前设置 `HF_ENDPOINT=https://huggingface.co` 覆盖。

启动本地接口：

```powershell
wsl -d Ubuntu -- bash /mnt/d/llm_learning/llm-single-node/scripts/serve.sh outputs/cpu-smoke/merged
```

另开一个 PowerShell 验证：

```powershell
wsl -d Ubuntu -- bash /mnt/d/llm_learning/llm-single-node/scripts/test_api.sh
```

接口地址为 `http://127.0.0.1:8000/v1`，模型 ID 为 `local-warehouse-llm`。

轻量服务会读取模型的 `max_position_embeddings`，为输出预留空间，并在输入超限时从左侧截断，优先保留最近消息。Harness 标准模式的系统提示和工具 schema 很长；135M 冒烟模型会发生截断，只适合验证链路。要获得可用的 Agent 行为，应使用长上下文的 1.5B/7B 模型并只挂载实际需要的工具。

## 运行 0.5B 本地实验

先把 `data/` 中的示例替换为真实的脱敏数据，再执行：

```powershell
wsl -d Ubuntu -- bash /mnt/d/llm_learning/llm-single-node/scripts/run_pipeline.sh configs/cpu-local.yaml
wsl -d Ubuntu -- bash /mnt/d/llm_learning/llm-single-node/scripts/serve.sh outputs/qwen-0.5b/merged
```

建议先将 `cpu-local.yaml` 中 `max_steps` 保持在 20，确认损失、内存和耗时后再逐步增加。样例数据太少，只能验证代码，不能得到有实际质量的模型。

## 单独运行某个阶段

```bash
cd /mnt/d/llm_learning/llm-single-node
~/.venvs/llm-single-node/bin/python -m single_node_llm.train --config configs/cpu-local.yaml --stage pt
~/.venvs/llm-single-node/bin/python -m single_node_llm.train --config configs/cpu-local.yaml --stage sft
~/.venvs/llm-single-node/bin/python -m single_node_llm.train --config configs/cpu-local.yaml --stage dpo
~/.venvs/llm-single-node/bin/python -m single_node_llm.train --config configs/cpu-local.yaml --stage grpo
~/.venvs/llm-single-node/bin/python -m single_node_llm.merge --config configs/cpu-local.yaml
```

默认阶段依赖关系是 `pt -> sft -> dpo -> grpo`。也可以使用 `--input-adapter PATH` 显式指定上一阶段权重，使用 `--max-steps N` 临时覆盖配置。

## 数据格式

`data/pretrain.txt` 每行是一段领域文本。正式使用时应该去重、脱敏、去除乱码，并控制不同来源的数据比例。

`data/sft.jsonl` 每行格式：

```json
{"messages":[{"role":"user","content":"问题"},{"role":"assistant","content":"答案"}]}
```

`data/preferences.jsonl` 每行包含同一提示的优选和拒绝答案：

```json
{"prompt":"问题","chosen":"安全、正确的回答","rejected":"较差的回答"}
```

`data/grpo.jsonl` 使用可验证奖励。当前示例奖励检查 JSON 合法性和必需关键词：

```json
{"prompt":"只输出 JSON...","required_terms":["status","ok"],"require_json":true}
```

生产训练应在 `reward_completion()` 中接入更可靠的验证器，例如单元测试、编译器、JSON Schema、SQL 只读沙箱或业务模拟器。不要把生产数据库写操作直接作为奖励工具。

## 连接 DeepSeek Harness

先启动模型服务，再另开一个 PowerShell 一键运行 Harness：

```powershell
powershell -ExecutionPolicy Bypass -File D:\llm_learning\llm-single-node\scripts\start-harness.ps1
```

脚本使用工程内隔离的 `.dsh-home`，首次启动会自动复制本地 Provider 配置。打开 `http://127.0.0.1:3080`，从模型选择器选 `Local Warehouse LLM`。若需要手工配置，进入 `Settings -> Models -> Add a custom provider`，填写：

```text
Provider ID: local-lab
Base URL: http://127.0.0.1:8000/v1
API protocol: openai-completions
API key: local
Model ID: local-warehouse-llm
```

兼容参数模板位于 `harness/settings.yaml`。可以将对应片段合并到 `$DSH_HOME/settings.yaml`，不要直接覆盖已有设置。当前轻量服务支持普通和流式 Chat Completions，也会解析 Qwen 风格的 `<tool_call>`；135M/0.5B 模型的工具调用能力有限，这是模型能力限制而非 Harness 限制。

## 换成 NVIDIA 单卡

安装支持 WSL2 的 Windows NVIDIA 驱动后，先确认下面命令能显示显卡：

```powershell
wsl -d Ubuntu -- nvidia-smi
```

建立单独的 CUDA 环境，避免污染 CPU 环境：

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/d/llm_learning/llm-single-node && USE_CUDA=1 VENV_DIR=~/.venvs/llm-single-node-cuda bash scripts/bootstrap.sh"
wsl -d Ubuntu -- bash -lc "cd /mnt/d/llm_learning/llm-single-node && VENV_DIR=~/.venvs/llm-single-node-cuda bash scripts/run_pipeline.sh configs/cuda-1.5b.yaml"
```

24 GB 显存适合 1.5B/7B QLoRA 类实验，但本工程当前使用 BF16 LoRA，没有接入 4-bit bitsandbytes。若升级至 7B，应再加入 4-bit 量化、显存监控和更成熟的分布式训练框架。

## 算法边界

- PT/SFT 使用标准 causal language modeling loss。
- DPO 使用冻结参考模型与偏好对的直接偏好损失。
- GRPO 使用同一提示的多次采样、组内标准化奖励以及参考模型 KL 约束。
- 实现优先保证算法透明和单机可运行，不替代 verl、TRL、LLaMA-Factory 或 DeepSpeed 的生产级吞吐、分布式容错和数据管线。
