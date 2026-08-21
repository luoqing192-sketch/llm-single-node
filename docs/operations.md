# 操作手册 — llm-single-node

> 本文档是「怎么跑起来」的实操指南。理论内容见 [architecture.md](./architecture.md)。
>
> 所有命令默认在 **Windows PowerShell** 中执行;训练脚本通过 `wsl -d Ubuntu -- bash ...` 进入 WSL 里跑(因为依赖在本机 WSL Ubuntu 里装)。

---

## 1. 环境要求

| 项目 | 当前机器实测 | 要求 |
|---|---|---|
| 系统 | Windows 11 Pro | 任意支持 WSL2 的版本 |
| WSL | Ubuntu 22.04 | 已安装,WSL2 |
| CPU / 内存 | 8 vCPU / 约 24 GiB | 至少能跑 135M~0.5B |
| GPU | 无 NVIDIA 显卡 | 可后加,见第 9 节 |

> ⚠️ 7B 模型训练不适合当前硬件,`cuda-1.5b.yaml` 也只是 1.5B。

---

## 2. 一键环境搭建

在 PowerShell 中执行(首次会自动建 venv、装依赖、跑环境自检,需几分钟):

```powershell
wsl -d Ubuntu -- bash /mnt/d/llm_learning/llm-single-node/scripts/bootstrap.sh
```

脚本做的事:

1. 在 `~/.venvs/llm-single-node` 建独立 Python venv;
2. 装 `torch`(默认 CPU 版;`USE_CUDA=1` 时装 CUDA 版)+ `pip install -e .`(装本项目及依赖);
3. 跑 `python -m single_node_llm.preflight` 输出环境信息与推荐配置。

**自检输出长这样:**

```json
{
  "platform": "Linux-5.15.153.1-microsoft-standard-WSL2-x86_64",
  "cpu_count": 8,
  "memory_gib": 23.5,
  "cuda_available": false,
  "recommendation": "cpu-smoke.yaml first, then cpu-local.yaml"
}
```

> 网络受限时,脚本内依赖 `HF_ENDPOINT`(默认 `https://hf-mirror.com`)下载模型,可用 `export HF_ENDPOINT=https://huggingface.co` 覆盖。

---

## 3. 快速体验(5 分钟冒烟)

用 135M 最小模型验证「全链路能通」:

```powershell
wsl -d Ubuntu -- bash /mnt/d/llm_learning/llm-single-node/scripts/run_pipeline.sh configs/cpu-smoke.yaml
```

预期输出:依次出现 `stage=pt/sft/dpo/grpo` 的 loss 日志,最后一行:

```text
Pipeline complete for configs/cpu-smoke.yaml
```

产物写入:

```text
outputs/cpu-smoke/pt/     pt adapter
outputs/cpu-smoke/sft/    sft adapter
outputs/cpu-smoke/dpo/    dpo adapter
outputs/cpu-smoke/grpo/   grpo adapter
outputs/cpu-smoke/merged/ 合并后的完整模型
```

**启动服务并测试:**

```powershell
# 窗口 A:启动服务
wsl -d Ubuntu -- bash /mnt/d/llm_learning/llm-single-node/scripts/serve.sh outputs/cpu-smoke/merged

# 窗口 B:冒烟测试
wsl -d Ubuntu -- bash /mnt/d/llm_learning/llm-single-node/scripts/test_api.sh
```

`test_api.sh` 会依次打 `/health`、普通 chat、流式 chat 三个请求。看到 200 即链路 OK。

---

## 4. 配置选择与调整

### 4.1 怎么选配置

| 场景 | 用哪个 | 说明 |
|---|---|---|
| 首次跑通 | `configs/cpu-smoke.yaml` | 1 step,几分钟,只验证链路 |
| CPU 小实验 | `configs/cpu-local.yaml` | Qwen 0.5B,20 steps |
| 有 GPU 认真训 | `configs/cuda-1.5b.yaml` | DeepSeek-R1-Distill 1.5B,200 steps |

### 4.2 常用调整项(改 YAML 即可)

| 想做的事 | 改哪里 |
|---|---|
| 训练多久 | `train.max_steps`(优化器更新次数,不是 epoch) |
| 显存/内存不够 | 调小 `max_length`、`train.gradient_accumulation_steps` 调大 |
| 学不动/震荡 | `train.learning_rate`(0.5B 级别通常 1e-4 附近) |
| 想更快试 | 减小 `grpo.num_generations`(每组采样数) |
| 生成太长 | 调小 `grpo.max_new_tokens` |
| 偏好/RL 强度 | `dpo.beta`、`grpo.beta`(越大越强约束) |
| 换底座模型 | 改 `model_name`(HuggingFace 仓库名) |

### 4.3 分阶段跑(推荐先这样练手)

一次跑全流程前,建议先单阶段跑,看每步的日志格式:

```bash
cd /mnt/d/llm_learning/llm-single-node
~/.venvs/llm-single-node/bin/python -m single_node_llm.train --config configs/cpu-local.yaml --stage pt
~/.venvs/llm-single-node/bin/python -m single_node_llm.train --config configs/cpu-local.yaml --stage sft
~/.venvs/llm-single-node/bin/python -m single_node_llm.train --config configs/cpu-local.yaml --stage dpo
~/.venvs/llm-single-node/bin/python -m single_node_llm.train --config configs/cpu-local.yaml --stage grpo
~/.venvs/llm-single-node/bin/python -m single_node_llm.merge --config configs/cpu-local.yaml
```

> 阶段依赖 `pt → sft → dpo → grpo`。跳过前一阶段会报「前一阶段适配器不存在」;也可用 `--input-adapter PATH` 指定任意 adapter 作为起点,`--max-steps N` 临时覆盖步数。

---

## 5. 数据准备(把样例换成真实数据)

> ⚠️ `data/` 里目前全是**样例**,只够验证代码,不足以得到有质量的模型。正式训练前务必替换为**真实、脱敏、去重**的数据。

### 5.1 继续预训练 `data/pretrain.txt`

```text
每行一段领域文本,行与行之间不需要空行
例如你的 WES/ESS 业务文档、代码片段、产品手册
```

要点:去重、脱敏、清理乱码,控制不同来源语料比例。

### 5.2 SFT `data/sft.jsonl`

```json
{"messages":[{"role":"user","content":"问题"},{"role":"assistant","content":"答案"}]}
```

要点:答案要完整规范;`train.py` 会按模型的 chat template 自动拼成对话。

### 5.3 DPO `data/preferences.jsonl`

```json
{"prompt":"问题","chosen":"正确/安全的回答","rejected":"较差/有害的回答"}
```

要点:chosen 与 rejected 对应**同一个 prompt**,差异要明显且真实。

### 5.4 GRPO `data/grpo.jsonl`

```json
{"prompt":"只输出 JSON 的任务","required_terms":["status","ok"],"require_json":true}
```

要点:奖励规则在 `train.py:142 reward_completion()`。**正式场景必须替换为可靠验证器**(单元测试 / 编译器 / JSON Schema / SQL 只读沙箱 / 业务模拟器),不要把生产库写操作直接当奖励工具。

---

## 6. 训练运行

### 6.1 全流程

```powershell
wsl -d Ubuntu -- bash /mnt/d/llm_learning/llm-single-node/scripts/run_pipeline.sh configs/cpu-local.yaml
```

### 6.2 看懂训练日志

```text
stage=pt step=20/20 loss=1.952731        # causal 交叉熵,往下掉就对了
stage=dpo step=10/10 loss=0.470190 preference_accuracy=1   # 1 表示这次偏好排序正确
stage=grpo step=5/5 loss=-0.853918 rewards=[2.0, 1.0, ...] samples=[...]  # 奖励越高越好
```

### 6.3 常见判断

| 现象 | 含义 | 处理 |
|---|---|---|
| pt/sft loss 稳定下降 | 训练正常 | 继续 |
| loss 震荡不降 | 学习率太大 / 数据太杂 | 调小 lr,清洗数据 |
| dpo accuracy 长期为 0 | 偏好对没学出来 | 换更有区分度的 chosen/rejected |
| grpo rewards 全一样 | 奖励规则太简单 | 增强 `reward_completion` |

---

## 7. 推理服务

### 7.1 启动

```powershell
wsl -d Ubuntu -- bash /mnt/d/llm_learning/llm-single-node/scripts/serve.sh outputs/cpu-local/merged
```

参数可用环境变量覆盖:`MODEL_NAME=xxx PORT=8000`。服务地址:`http://127.0.0.1:8000/v1`,模型 ID 默认 `local-warehouse-llm`。

### 7.2 直接调接口(不依赖脚本)

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-warehouse-llm","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
```

### 7.3 已知限制

- **单路串行**:一次只处理一个请求,并发请求会排队;
- **伪流式**:stream 也是整体算完一次性吐回;
- 135M / 0.5B 模型的工具调用和 Agent 能力很弱,这是模型能力限制;
- 长上下文(system prompt + 工具 schema 很长)时 135M 会被截断,只适合验证链路;要可用 Agent 行为请用 1.5B+ 并只挂必要工具。

---

## 8. 连接 DeepSeek Harness

1. 先按第 7 节**启动推理服务**;
2. 另开 PowerShell 启动 Harness:

```powershell
powershell -ExecutionPolicy Bypass -File D:\llm_learning\llm-single-node\scripts\start-harness.ps1
```

3. 打开 `http://127.0.0.1:3080`,模型选择器里选 **Local Warehouse LLM**;
4. 首次启动脚本会把 `harness/settings.yaml` 复制到隔离的 `.dsh-home`,并设置 `DSH_HOME` 与 `LOCAL_LLM_API_KEY=local`。

> 若需手工配置:Settings → Models → Add a custom provider,填 `Provider ID=local-lab`、`Base URL=http://127.0.0.1:8000/v1`、`API protocol=openai-completions`、`API key=local`、`Model ID=local-warehouse-llm`。

---

## 9. 上 NVIDIA GPU

### 9.1 前提

安装支持 WSL2 的 Windows NVIDIA 驱动,并确认:

```powershell
wsl -d Ubuntu -- nvidia-smi   # 能看到显卡
```

### 9.2 建独立 CUDA 环境(避免污染 CPU 环境)

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/d/llm_learning/llm-single-node && USE_CUDA=1 VENV_DIR=~/.venvs/llm-single-node-cuda bash scripts/bootstrap.sh"
wsl -d Ubuntu -- bash -lc "cd /mnt/d/llm_learning/llm-single-node && VENV_DIR=~/.venvs/llm-single-node-cuda bash scripts/run_pipeline.sh configs/cuda-1.5b.yaml"
```

### 9.3 显存规划参考

- 24GB 显存适合 **1.5B / 7B 的 QLoRA 类实验**;
- 本工程当前是 **BF16 LoRA,没接 4-bit bitsandbytes**;要升 7B 请先补 4-bit 量化、显存监控和更成熟的分布式框架(README 建议)。

---

## 10. 常见问题排查

| 症状 | 原因 | 处理 |
|---|---|---|
| `配置要求 CUDA,但 PyTorch 未检测到 CUDA GPU` | 用了 cuda 配置但没装 CUDA torch / 没显卡 | 装 `USE_CUDA=1` 的 venv,或改用 cpu 配置 |
| `CPU 训练不能使用 float16` | CPU 上 dtype 配了 float16 | 改成 float32 |
| `前一阶段适配器不存在: ...` | 跳过阶段直接跑后续 | 先跑前一阶段,或用 `--input-adapter` |
| 下载模型很慢/失败 | 网络到 huggingface.co 不通 | 检查 `HF_ENDPOINT` 是否指向 hf-mirror.com |
| 训练时内存/显存溢出 | `max_length` 太大 / 模型太大 | 调小 `max_length`、开 `gradient_checkpointing`、减小 batch |
| `Unknown model: xxx` | 请求里的 model 与服务不一致 | 用 `local-warehouse-llm`,或用 `/v1/models` 查实际 ID |
| WSL 里 `docker`/资源报错 | 与 llm 无关的环境问题 | 见 `local-env` / `gpc-env-local` skill 对应排查 |
| IDEA 里 import 报红 | 工程没挂 Python 解释器 | 在 `.idea` 配置 SDK 指向 `.venv-idea\Scripts\python.exe`(见仓库 README) |

---

## 11. 常用命令速查

```powershell
# 环境自检
wsl -d Ubuntu -- ~/.venvs/llm-single-node/bin/python -m single_node_llm.preflight

# 跑全流程
wsl -d Ubuntu -- bash /mnt/d/llm_learning/llm-single-node/scripts/run_pipeline.sh configs/cpu-smoke.yaml

# 单阶段
wsl -d Ubuntu -- ~/.venvs/llm-single-node/bin/python -m single_node_llm.train --config configs/cpu-local.yaml --stage dpo

# 合并 LoRA
wsl -d Ubuntu -- ~/.venvs/llm-single-node/bin/python -m single_node_llm.merge --config configs/cpu-local.yaml

# 起服务 / 冒烟
wsl -d Ubuntu -- bash /mnt/d/llm_learning/llm-single-node/scripts/serve.sh outputs/cpu-local/merged
wsl -d Ubuntu -- bash /mnt/d/llm_learning/llm-single-node/scripts/test_api.sh

# 起 Harness
powershell -ExecutionPolicy Bypass -File D:\llm_learning\llm-single-node\scripts\start-harness.ps1
```

---

## 12. 提交到 git(可选)

```powershell
cd D:\llm_learning\llm-single-node
git add -A
git commit -m "docs: add architecture and operations docs"
git push origin main
```

> `.gitignore` 已排除 `.venv/ .venv-idea/ outputs/ .dsh-home/` 等,不会误提交大文件。
