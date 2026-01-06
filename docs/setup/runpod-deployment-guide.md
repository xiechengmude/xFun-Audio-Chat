# Fun-Audio-Chat RunPod 部署指南

> **版本**: v1.2-20260106160000
> **适用环境**: RunPod GPU Cloud (A40/4090/A100/H100)
> **最低显存**: 24GB (推理)

---

## 目录

- [概述](#概述)
- [环境要求](#环境要求)
- [RunPod 资源管理](#runpod-资源管理)
- [部署步骤](#部署步骤)
- [启动服务](#启动服务)
- [API 接口](#api-接口)
- [常见问题](#常见问题)
- [参考链接](#参考链接)

---

## 概述

本文档记录 Fun-Audio-Chat 在 RunPod 云平台上的完整部署流程，包括：
- RunPod API 自动化管理
- Template 和 Pod 创建
- 依赖安装与问题解决
- 服务启动与验证

### 架构能力

Fun-Audio-Chat 是阿里云的大型音频语言模型，提供：

| 能力 | 说明 |
|------|------|
| **语音问答 (S2T)** | 语音输入 → 文本输出 |
| **语音对话 (S2S)** | 语音输入 → 语音输出（流式） |
| **音频理解** | 音频内容分析、转录 |
| **语音函数调用** | 通过语音触发工具调用 |
| **语音情感共鸣** | 情感感知的语音响应 |

### 核心技术

- **双分辨率语音表征**: 5Hz共享骨干 + 25Hz精细化头部
- **模型规模**: ~8B 参数
- **TTS引擎**: CosyVoice3-0.5B

---

## 环境要求

```yaml
硬件:
  GPU: NVIDIA A40/4090/A100/H100 (>= 24GB VRAM)
  推理显存占用: ~22GB
  推荐: A40 (48GB) - 性价比最优

软件:
  Python: 3.11+ (推荐 3.12)
  PyTorch: 2.8.0 + CUDA 12.8
  ffmpeg: 必需
```

### GPU 选择建议

| GPU | VRAM | 价格 | 适用场景 |
|-----|------|------|----------|
| **A40** | 48GB | $0.40/hr | 推荐 - 显存充足，单卡部署 |
| RTX 4090 | 24GB | $0.50-0.70/hr | 刚好够用，无冗余 |
| A100 | 80GB | $1.89/hr | 多模型/训练场景 |

---

## RunPod 资源管理

### 配置 API Key

```bash
# 在项目根目录创建 .env 文件
echo "RUNPOD_KEY=your_api_key_here" > .env
```

### 管理脚本使用

项目提供 `scripts/runpod_manager.py` 用于自动化管理 RunPod 资源：

#### 列出可用 GPU

```bash
# 列出 A40 GPU
python3 scripts/runpod_manager.py --action list-gpus --gpu A40

# 列出 4090 GPU
python3 scripts/runpod_manager.py --action list-gpus --gpu 4090
```

#### 列出当前 Pods

```bash
python3 scripts/runpod_manager.py --action list-pods
```

输出示例：
```
Name                      ID                   GPU             Status          Public IP
------------------------------------------------------------------------------------------
fun-audio-chat-a40        r3jj7s3xtj89gu       1               RUNNING         69.30.85.139
  Port mappings: {'22': 22198, '8000': 22199, '8001': 22197, '8002': 22196, ...}
```

#### 创建 Template

```bash
python3 scripts/runpod_manager.py --action create-template \
    --name "Fun-Audio-Chat" \
    --disk 100 \
    --volume 100
```

**已创建的 Template**:
- **ID**: `f4ertqge9p`
- **Ports**: 22/tcp, 8000-8010/tcp, 8080/http, 8888/http
- **Container Disk**: 100GB
- **Volume**: 100GB

#### 创建 Pod

```bash
# 创建 A40 Pod
python3 scripts/runpod_manager.py --action create-pod \
    --name "fun-audio-chat-a40" \
    --gpu A40 \
    --disk 100 \
    --volume 100

# 创建 4090 Pod
python3 scripts/runpod_manager.py --action create-pod \
    --name "fun-audio-chat-4090" \
    --gpu 4090 \
    --disk 100 \
    --volume 100
```

#### Pod 生命周期管理

```bash
# 停止 Pod (保留数据，停止计费)
python3 scripts/runpod_manager.py --action stop --pod-id <pod_id>

# 启动已停止的 Pod
python3 scripts/runpod_manager.py --action start --pod-id <pod_id>

# 删除 Pod (数据丢失)
python3 scripts/runpod_manager.py --action terminate --pod-id <pod_id>
```

### RunPod API 经验总结

#### REST API 要点

| 参数 | 说明 | 注意事项 |
|------|------|----------|
| `gpuTypeIds` | GPU类型ID数组 | 必须是数组，如 `["NVIDIA A40"]` |
| `cloudType` | 云类型 | 只能是 `SECURE` 或 `COMMUNITY` |
| `ports` | 端口配置 | 必须是数组，如 `["22/tcp", "8000/tcp"]` |
| `containerDiskInGb` | 容器磁盘 | 整数，单位GB |
| `volumeInGb` | 持久化卷 | 整数，单位GB |

#### GraphQL API 要点

```graphql
# 创建 Template
mutation {
  saveTemplate(input: {
    name: "Fun-Audio-Chat",
    imageName: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    dockerArgs: "",  # 必填字段
    containerDiskInGb: 100,
    volumeInGb: 100,
    ports: "22/tcp,8000/tcp,8001/tcp,...",
    isServerless: false
  }) {
    id
    name
  }
}
```

### 端口映射查询

端口映射是动态分配的，通过 API 查询：

```bash
python3 scripts/runpod_manager.py --action list-pods
```

或直接调用 API：

```bash
curl -s -X GET "https://rest.runpod.io/v1/pods" \
  -H "Authorization: Bearer $RUNPOD_KEY" \
  -H "Content-Type: application/json" | python3 -m json.tool
```

---

## 部署步骤

### 1. SSH 连接到 Pod

```bash
# 使用 list-pods 获取的端口
ssh root@<public_ip> -p <ssh_port> -i ~/.ssh/id_ed25519
```

### 2. 克隆项目

```bash
cd /workspace
git clone --recurse-submodules https://github.com/FunAudioLLM/Fun-Audio-Chat
cd Fun-Audio-Chat
```

### 3. 安装系统依赖

```bash
apt update && apt install -y ffmpeg
```

### 4. 安装 Python 依赖

```bash
# PyTorch + CUDA (必须匹配版本)
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Web Demo 依赖
pip install sphn aiohttp

# 项目依赖
pip install -r requirements.txt
```

### 5. 修复依赖版本冲突

**关键**: ruamel.yaml >= 0.18 与 hyperpyyaml 不兼容

```bash
pip install 'ruamel.yaml<0.18' --force-reinstall
```

### 6. 下载预训练模型

```bash
# 使用 HuggingFace
pip install huggingface-hub
hf download FunAudioLLM/Fun-Audio-Chat-8B --local-dir ./pretrained_models/Fun-Audio-Chat-8B
hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir ./pretrained_models/Fun-CosyVoice3-0.5B-2512
```

或使用 ModelScope（国内更快）：

```bash
modelscope download --model FunAudioLLM/Fun-Audio-Chat-8B --local_dir pretrained_models/Fun-Audio-Chat-8B
modelscope download --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local_dir pretrained_models/Fun-CosyVoice3-0.5B-2512
```

---

## 启动服务

### 单 GPU 配置 (A40/4090)

```bash
cd /workspace/Fun-Audio-Chat
export PYTHONPATH=$(pwd)

# 后台启动 (单GPU使用 --tts-gpu 0)
nohup python3 -m web_demo.server.server \
    --model-path pretrained_models/Fun-Audio-Chat-8B \
    --port 8002 \
    --tts-gpu 0 \
    --host 0.0.0.0 \
    > server.log 2>&1 &
```

### 双 GPU 配置 (推荐)

```bash
# GPU 0: S2S模型, GPU 1: TTS模型
python3 -m web_demo.server.server \
    --model-path pretrained_models/Fun-Audio-Chat-8B \
    --port 8002 \
    --tts-gpu 1 \
    --host 0.0.0.0
```

### 验证启动

```bash
# 检查端口监听
ss -tlnp | grep 8002

# 检查日志
tail -f server.log

# 关键成功日志:
# [INFO] s2s model loaded (: cuda)
# [INFO] cosyvoice loaded
```

### 资源占用

| 组件 | 显存 |
|------|------|
| S2S 模型 (8B) | ~18GB |
| TTS 模型 (CosyVoice3) | ~4GB |
| **总计** | **~22GB** |

---

## API 接口

### WebSocket 端点

```
ws://<public_ip>:<mapped_port>/api/chat
```

例如：`ws://69.30.85.139:22196/api/chat`

### 协议说明

Server 使用二进制 WebSocket 协议：

| 消息类型 | 格式 | 说明 |
|----------|------|------|
| audio | `\x01` + opus_bytes | Opus 编码音频 |
| text | `\x02` + utf8_bytes | 文本消息 |
| control | JSON | 控制信号 (start/pause/endTurn) |
| metadata | JSON | 会话元数据 (system_prompt) |

### 客户端交互流程

```
Client                          Server
  |                               |
  |-------- WebSocket Connect --->|
  |<------- Handshake ------------|
  |                               |
  |-------- control: start ------>|
  |-------- audio (opus) -------->|
  |-------- audio (opus) -------->|
  |-------- control: pause ------>|
  |                               |
  |<------- text (processing) ----|
  |<------- audio (opus) ---------|
  |<------- audio (opus) ---------|
  |<------- text (final) ---------|
```

---

## 常见问题

### Q1: ruamel.yaml AttributeError

**错误**: `'Loader' object has no attribute 'max_depth'`

**原因**: ruamel.yaml 0.18+ 与 hyperpyyaml 不兼容

**解决**:
```bash
pip install 'ruamel.yaml<0.18' --force-reinstall
```

### Q2: torchvision 版本不兼容

**错误**: PyTorch 与 torchvision 版本不匹配

**解决**:
```bash
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

### Q3: ONNX Runtime CUDA 警告

**警告**: `Failed to create CUDAExecutionProvider`

**原因**: 缺少 libcudnn.so.8

**影响**: TTS 部分组件回退到 CPU，但不影响主要功能

### Q4: 显存不足 (OOM)

**症状**: CUDA out of memory

**解决**:
- 使用 >= 24GB 显存的 GPU (推荐 A40)
- 双 GPU 配置分离 S2S 和 TTS
- 减少 batch size

### Q5: RunPod API 端口格式错误

**错误**: `ports/type: got string, want array`

**解决**: 端口必须是数组格式
```python
# 错误
"ports": "22/tcp,8000/tcp"

# 正确
"ports": ["22/tcp", "8000/tcp"]
```

### Q6: RunPod cloudType 错误

**错误**: `value must be one of 'SECURE', 'COMMUNITY'`

**解决**: cloudType 只能是 `SECURE` 或 `COMMUNITY`，不支持 `ALL`

---

## 监控命令

```bash
# GPU 状态
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv

# 服务进程
ps aux | grep web_demo.server

# 实时日志
tail -f /workspace/Fun-Audio-Chat/server.log
```

---

## 停止服务

```bash
pkill -f 'web_demo.server'
```

---

## 一键部署脚本

可以创建一键部署脚本 `scripts/setup.sh`：

```bash
#!/bin/bash
set -e

echo "=== Fun-Audio-Chat Setup ==="

# 安装系统依赖
apt update && apt install -y ffmpeg

# 安装 Python 依赖
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install sphn aiohttp
pip install -r requirements.txt
pip install 'ruamel.yaml<0.18' --force-reinstall

# 下载模型 (如果不存在)
if [ ! -d "pretrained_models/Fun-Audio-Chat-8B" ]; then
    pip install huggingface-hub
    hf download FunAudioLLM/Fun-Audio-Chat-8B --local-dir ./pretrained_models/Fun-Audio-Chat-8B
    hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir ./pretrained_models/Fun-CosyVoice3-0.5B-2512
fi

echo "=== Setup Complete ==="
echo "Run: python3 -m web_demo.server.server --model-path pretrained_models/Fun-Audio-Chat-8B --port 8002 --tts-gpu 0 --host 0.0.0.0"
```

---

## 自动化部署 (CLAUDE SKILL)

项目提供完整的自动化部署工具，支持一键完成 Pod 创建、环境配置、服务启动和验证。

### 自动化脚本

| 脚本 | 功能 |
|------|------|
| `scripts/runpod_manager.py` | RunPod API 管理（创建/停止/删除 Pod） |
| `scripts/auto_deploy.py` | 全自动部署（创建→配置→启动→验证） |
| `scripts/test_deployment.py` | 部署验证测试 |

### 一键部署

```bash
# 使用 A40 GPU 部署（推荐）
python3 scripts/auto_deploy.py --gpu A40

# 使用 4090 GPU 部署
python3 scripts/auto_deploy.py --gpu 4090

# 自定义配置
python3 scripts/auto_deploy.py --gpu A40 --disk 150 --volume 150

# 部署后运行测试
python3 scripts/auto_deploy.py --gpu A40 --test
```

### 查看 Pod 状态

```bash
python3 scripts/auto_deploy.py --status
```

### 部署流程

自动化脚本执行以下 6 个阶段：

1. **Phase 1**: 创建 Pod（选择 GPU、配置磁盘和卷）
2. **Phase 2**: 等待 Pod 就绪（RUNNING 状态 + 公网 IP）
3. **Phase 3**: 建立 SSH 连接
4. **Phase 4**: 环境配置（系统依赖 + Python 包 + 模型下载）
5. **Phase 5**: 启动服务（后台运行 server.py）
6. **Phase 6**: 验证部署（检查端口监听 + 模型加载）

### 部署验证

```bash
# 基础连接测试
python3 scripts/test_deployment.py --host <ip> --port <port>

# 完整 S2S 功能测试
python3 scripts/test_deployment.py --host <ip> --port <port> --full
```

### CLAUDE SKILL 集成

项目包含 Claude Code Skill 定义文件：

```
.claude/commands/runpod-deploy.md
```

在 Claude Code 中使用：

```bash
/runpod-deploy --gpu A40
/runpod-deploy --status
```

---

## 参考链接

- [Fun-Audio-Chat GitHub](https://github.com/FunAudioLLM/Fun-Audio-Chat)
- [arXiv Paper](https://arxiv.org/pdf/2512.20156)
- [HuggingFace Models](https://huggingface.co/FunAudioLLM/Fun-Audio-Chat-8B)
- [Demo Page](https://funaudiollm.github.io/funaudiochat)
- [RunPod Docs - Manage Pods](https://docs.runpod.io/pods/manage-pods)
- [RunPod GraphQL API](https://docs.runpod.io/sdks/graphql/manage-pod-templates)
