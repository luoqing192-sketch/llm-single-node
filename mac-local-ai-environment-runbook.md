# Mac 本地 AI 环境部署与运行手册

本文记录 2026-08-21 至 2026-08-22 在 MacBook Air 上完成的 Docker 环境、Cortex Agent 和 `llm-single-node` 训练流程部署。文档不包含 SSH 密码、API Key 或 JWT Secret。

## 1. 机器与当前服务

| 项目 | 当前值 |
| --- | --- |
| 主机名 | `qingdeMacBook-Air.local` |
| 系统 | macOS 15.7.7 |
| 芯片 | Apple M1，8 核 CPU、8 核 GPU |
| 内存 | 16 GB 统一内存 |
| 当前局域网 IP | `192.168.2.144` |
| SSH 用户 | `qing` |

当前 IP 由 DHCP 分配，曾从 `192.168.2.145` 变为 `192.168.2.144`。建议在路由器中按 Mac 地址绑定固定 IP。

| 服务 | 地址 | 状态 |
| --- | --- | --- |
| Cortex | `http://192.168.2.144:8000` | 健康检查、首页和登录已验证 |
| 本地模型 API | `http://192.168.2.144:18000/v1` | health、普通 Chat、SSE 已验证 |
| Docker Engine | Colima VM | `hello-world`、Compose 已验证 |

## 2. Docker、Colima 与 Compose

### 2.1 最终版本和资源

- Homebrew `6.0.18`
- Docker CLI `29.7.2`
- Docker Engine `29.5.2`
- Docker Compose `5.5.0`
- Colima `0.10.3`
- Lima `2.2.0`
- Docker context：`colima`
- VM：Apple Virtualization Framework（VZ）、`aarch64`、`virtiofs`、`overlayfs`
- VM 资源：4 CPU、6 GiB 内存、60 GiB 磁盘

### 2.2 安装命令

Homebrew 官方 API 在当时的网络环境中较慢，因此只对本次安装临时使用清华镜像，没有修改全局 Homebrew 配置：

```bash
HOMEBREW_NO_AUTO_UPDATE=1 \
HOMEBREW_API_DOMAIN=https://mirrors.tuna.tsinghua.edu.cn/homebrew-bottles/api \
HOMEBREW_BOTTLE_DOMAIN=https://mirrors.tuna.tsinghua.edu.cn/homebrew-bottles \
brew install docker colima docker-compose
```

Compose 作为 Docker CLI 插件使用。`~/.docker/config.json` 中包含以下关键配置：

```json
{
  "currentContext": "colima",
  "cliPluginsExtraDirs": [
    "/opt/homebrew/lib/docker/cli-plugins"
  ]
}
```

首次启动参数：

```bash
colima start \
  --cpu 4 \
  --memory 6 \
  --disk 60 \
  --arch aarch64 \
  --runtime docker
```

持久资源配置位于 `~/.colima/default/colima.yaml`。当前关键值为：

```yaml
cpu: 4
memory: 6
disk: 60
arch: aarch64
runtime: docker
vmType: vz
```

### 2.3 首次 VM 镜像下载问题

Colima 使用的 ARM64 VM 镜像为：

```text
https://github.com/abiosoft/colima-core/releases/download/v0.10.4/ubuntu-24.04-minimal-cloudimg-arm64-docker.raw.gz
```

- 文件大小：`332354401` bytes
- SHA-512：

```text
32242674b046b5057e60c4aba334b51e3665f05412cda89ed081cc2de153ae5c41f6b105b5c442cbe48d78e2cc21e9ba1950e406b6fb4fc2fd1dd2259240abbd
```

Mac 直连 GitHub 下载速度很慢。Windows 下载虽然较快，但企业 DLP 会给落盘文件添加 `%TSD-Header-###%` 包装，导致文件大小和 SHA 不匹配，不能直接使用。最后使用二进制管道把 HTTP 响应直接传给 Mac，并在 Mac 端校验 SHA，然后通过本地镜像启动：

```bash
colima start \
  --cpu 4 \
  --memory 6 \
  --disk 60 \
  --arch aarch64 \
  --runtime docker \
  --disk-image /path/to/verified-image.raw.gz
```

现有 Colima VM 已完成解压，不再依赖原始压缩文件。`~/Library/Caches/colima/caches/` 下仍可能存在约 317 MiB 的 `.downloading` 缓存；本次未删除用户缓存。确认不再需要重建 VM 后可手工清理。

### 2.4 DNS 与 UniClash 代理修复

Colima Core v0.10.4 中，初始 `/etc/resolv.conf` 是悬空链接：

```text
../run/systemd/resolve/stub-resolv.conf
```

镜像中没有对应的 `/run/systemd/resolve` 和 `systemd-resolved.service`，因此 Docker DNS 回退到 `[::1]:53` 并报 `connection refused`。公网 DNS 或 UniClash DNS 返回的 fake-IP 又无法被 VM 直接路由。

宿主 UniClash 代理监听 `127.0.0.1:7993`，在 VM 中可通过宿主网关 `192.168.5.2:7993` 访问。最终采用以下持久修复。

VM 内 `/etc/resolv.conf` 已替换为普通文件：

```text
nameserver 192.168.5.2
options timeout:2 attempts:2
```

Docker daemon 不能只依赖登录 shell 的 `HTTP_PROXY`。已创建：

```text
/etc/systemd/system/docker.service.d/http-proxy.conf
```

内容如下：

```ini
[Service]
Environment="HTTP_PROXY=http://192.168.5.2:7993"
Environment="HTTPS_PROXY=http://192.168.5.2:7993"
Environment="NO_PROXY=localhost,127.0.0.1,::1,host.lima.internal,host.docker.internal,192.168.5.0/24"
```

应用并验证：

```bash
colima ssh -- sudo systemctl daemon-reload
colima ssh -- sudo systemctl restart docker
colima ssh -- systemctl is-active docker
colima ssh -- systemctl show docker -p Environment
docker info
```

注意事项：

- 拉取新镜像依赖 Mac 上的 UniClash `7993` 代理可用。
- Colima 登录自启可能早于 UniClash。代理启动后重试 `docker pull` 即可。
- 删除并重建 Colima VM 后，VM 内的 `resolv.conf` 和 systemd drop-in 需要重新配置。

### 2.5 Docker 与 Compose 验证

Docker Hub ARM64 镜像拉取和容器运行成功：

```bash
docker run --rm hello-world
```

输出包含：

```text
Hello from Docker!
```

拉取到的镜像 digest：

```text
sha256:5dd0d3e6e255913fc30f90b9f2b1d359cc2cbdb48090cc4b65f1676e203243cc
```

Compose 已实测以下完整流程，退出码为 0，测试容器和网络已清理：

```bash
docker compose config
docker compose up --abort-on-container-exit --exit-code-from hello
docker compose down --remove-orphans
```

### 2.6 Colima 登录自启动

已通过 Homebrew LaunchAgent 配置登录自启动：

```bash
colima stop
brew services start colima
```

LaunchAgent：

```text
~/Library/LaunchAgents/homebrew.mxcl.colima.plist
```

实际参数是 `/opt/homebrew/opt/colima/bin/colima start -f`，已验证停止后约 17 秒恢复。重启后 DNS 文件、Docker daemon 代理和 4/6/60 资源配置均保留，重新拉取和运行 `hello-world` 成功。

常用运维命令：

```bash
brew services info colima
brew services restart colima
brew services stop colima
colima status
docker info
docker ps -a
```

## 3. Cortex Agent

### 3.1 当前状态

- 项目：`/Users/qing/Documents/llm_proj/cortex`
- 地址：`http://192.168.2.144:8000`
- 当前进程：PID `89239`
- 启动命令：`.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000`
- Python：`3.12.14`
- Git 工作树：clean
- 健康检查、首页、登录和 `/api/auth/me`：已通过

登录凭据和模型 API Key 不记录在本手册中。首次登录后应修改默认密码。

### 3.2 关键路径

| 内容 | 路径 |
| --- | --- |
| Python 虚拟环境 | `backend/.venv` |
| 环境文件 | `backend/.env`，权限 `600` |
| SQLite | `backend/cortex.db` |
| 前端构建 | `frontend/dist` |
| 主启动日志 | `~/Library/Logs/cortex.log` |
| 应用日志 | `backend/app.log` |

不要输出或提交 `backend/.env` 中的 `JWT_SECRET`。

### 3.3 首次安装与构建

```bash
cd /Users/qing/Documents/llm_proj/cortex

export PATH="$HOME/.local/bin:$PATH"
export NVM_DIR="$HOME/.nvm"
source "$NVM_DIR/nvm.sh"

uv venv --python 3.12 backend/.venv
uv pip install --python backend/.venv -r backend/requirements.txt

(cd frontend && npm install && npm run build)

cp backend/.env.example backend/.env
chmod 600 backend/.env
```

需要把 `backend/.env` 的 `JWT_SECRET` 替换为至少 32 字节的本机随机值，且不能提交到 Git。`bash start.sh 8000` 也会自动创建缺失的虚拟环境、安装依赖并构建前端。

### 3.4 后台启动、停止与检查

后台启动：

```bash
cd /Users/qing/Documents/llm_proj/cortex

export PATH="$HOME/.local/bin:$PATH"
export NVM_DIR="$HOME/.nvm"
source "$NVM_DIR/nvm.sh"

mkdir -p "$HOME/Library/Logs"
nohup bash start.sh 8000 \
  >> "$HOME/Library/Logs/cortex.log" 2>&1 < /dev/null &
```

停止：

```bash
pid=$(lsof -tiTCP:8000 -sTCP:LISTEN)
[ -n "$pid" ] && kill "$pid"
```

检查：

```bash
curl -fsS http://127.0.0.1:8000/api/health
lsof -nP -iTCP:8000 -sTCP:LISTEN
tail -f "$HOME/Library/Logs/cortex.log"
```

Cortex 当前使用 `nohup`，没有配置 LaunchAgent，Mac 重启后需要手工执行后台启动命令。

### 3.5 模型配置

当前默认模型名为 `deepseek-chat`，但 `llm_api_key` 尚未配置，因此 Agent 编排可以运行，真实模型请求会返回 401。

在 Web 管理后台的“系统设置”中配置：

- `llm_base_url`
- `llm_api_key`
- `llm_model`

配置存储在 `backend/cortex.db`，通常动态读取，不需要重启。接口需要兼容 OpenAI Chat Completions。Embedding Key 目前也未配置；不配置时知识库使用确定性 hash 向量，检索质量较低。

## 4. llm-single-node 训练流程

### 4.1 当前状态与结果

- 项目：`/Users/qing/Documents/llm_proj/llm-single-node`
- 流程：PT -> SFT -> DPO -> GRPO -> LoRA merge
- 基础模型：`HuggingFaceTB/SmolLM2-135M-Instruct`
- 配置：`configs/cpu-smoke.yaml`
- 总耗时：`26.82s`
- Git 工作树：clean

训练指标：

| 阶段 | 结果 |
| --- | --- |
| PT | loss `4.383376` |
| SFT | loss `2.875782` |
| DPO | loss `0.692433`，preference accuracy `1` |
| GRPO | loss 约 `0`，rewards `[1, 1]` |
| Merge | 成功 |

这是 135M 模型、样例数据、每阶段 1 step 的完整链路冒烟测试，只能证明环境与代码可运行，不代表模型已具有实际业务质量。

### 4.2 Python 环境

- uv `0.12.5`
- Python `3.12.14`
- PyTorch `2.13.0` Apple arm64
- Transformers `4.57.6`
- PEFT `0.20.0`
- 虚拟环境：`~/.venvs/llm-single-node`

MPS 检测为可用，但仓库当前仅实现 `cpu/cuda`，所以本次使用 CPU。

首次环境创建：

```bash
cd /Users/qing/Documents/llm_proj/llm-single-node

curl --proto '=https' --tlsv1.2 -LsSf \
  https://astral.sh/uv/install.sh | sh

~/.local/bin/uv venv --python 3.12 ~/.venvs/llm-single-node
~/.local/bin/uv pip install \
  --python ~/.venvs/llm-single-node/bin/python torch
~/.local/bin/uv pip install \
  --python ~/.venvs/llm-single-node/bin/python -e .
```

### 4.3 模型离线缓存

默认 `hf-mirror.com` 的 HEAD 元数据与当前 `huggingface_hub` 不兼容，直连 Hugging Face 又超时。最终从 hf-mirror 下载权重、从 ModelScope 下载 config/tokenizer，构建了完整 Hugging Face 本地快照。

- commit：`12fd25f77366fa6b3b4b768ec3050bf629380bac`
- 缓存约：275 MiB
- 路径：

```text
~/.cache/huggingface/hub/
  models--HuggingFaceTB--SmolLM2-135M-Instruct/
```

完整缓存已经存在，后续无需重新下载。

### 4.4 运行完整 CPU smoke 流水线

```bash
cd /Users/qing/Documents/llm_proj/llm-single-node
mkdir -p outputs/mac-run

nohup env \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  /usr/bin/time -p \
  bash scripts/run_pipeline.sh configs/cpu-smoke.yaml \
  > outputs/mac-run/pipeline.log 2>&1 < /dev/null &
```

训练产物：

```text
outputs/cpu-smoke            约 574 MiB
outputs/cpu-smoke/merged     约 518 MiB
```

日志：

| 日志 | 用途 |
| --- | --- |
| `outputs/mac-run/pipeline.log` | 成功流水线 |
| `outputs/mac-run/serve.log` | 模型服务 |
| `outputs/mac-run/api-test.log` | API 验证 |
| `outputs/mac-run/deps.log` | 依赖安装 |
| `outputs/mac-run/model-download.log` | 模型下载 |
| `outputs/mac-run/pipeline-download-failure.log` | 首次镜像失败留档 |

### 4.5 启动与验证模型 API

后台启动：

```bash
cd /Users/qing/Documents/llm_proj/llm-single-node

nohup env \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PORT=18000 \
  bash scripts/serve.sh outputs/cpu-smoke/merged \
  > outputs/mac-run/serve.log 2>&1 < /dev/null &
```

当前进程 PID 为 `90723`，RSS 约 655 MB，监听 `0.0.0.0:18000`。已验证：

- `GET /health`
- `GET /v1/models`
- 普通 Chat Completions
- SSE 流式 `chat.completion.chunk`、`finish_reason=stop`、`[DONE]`

仓库测试脚本默认端口是 8000，本次替换为 18000：

```bash
sed 's/127.0.0.1:8000/127.0.0.1:18000/g' \
  scripts/test_api.sh | bash
```

进程检查、停止和重启：

```bash
lsof -nP -iTCP:18000 -sTCP:LISTEN
kill $(lsof -tiTCP:18000 -sTCP:LISTEN)

# 停止后重新执行上面的 nohup serve.sh 命令即可重启
```

当前模型 API 没有配置认证并监听全部网络接口，只适合可信局域网。若需要跨网访问，应增加鉴权和反向代理，不能直接映射到公网。模型服务使用 `nohup`，Mac 重启后需要手工启动。

## 5. 日常检查清单

SSH 登录：

```bash
ssh qing@192.168.2.144
```

Docker：

```bash
brew services info colima
colima status
docker info
docker run --rm hello-world
```

Cortex：

```bash
curl -fsS http://127.0.0.1:8000/api/health
tail -n 100 "$HOME/Library/Logs/cortex.log"
```

本地模型：

```bash
curl -fsS http://127.0.0.1:18000/health
tail -n 100 \
  /Users/qing/Documents/llm_proj/llm-single-node/outputs/mac-run/serve.log
```

机器重启后：

1. 等待 UniClash 和 Colima 启动。
2. 用 `docker info` 和 `docker run --rm hello-world` 检查 Docker。
3. 手工启动 Cortex。
4. 手工启动本地模型 API。
5. 检查当前 DHCP 地址；地址变化时同步更新访问 URL，或在路由器中设置 DHCP 固定租约。

## 6. 安全与维护建议

- 修改 Cortex 默认管理员密码。
- 不要在 Git、日志或命令历史中保存 SSH 密码、JWT Secret 和模型 API Key。
- Cortex 的 `.env` 保持 `600` 权限。
- 本地模型 API 当前无认证，不要暴露到公网。
- 如需长期运行，为 Cortex 和模型服务增加 LaunchAgent，而不是长期依赖 `nohup`。
- 16 GB M1 适合 135M 小模型训练验证、7B 左右量化推理和轻量 LoRA；不适合从零预训练大模型。
- 删除/重建 Colima VM 前备份所需容器卷，并记录 VM 内 DNS 与代理修复。
