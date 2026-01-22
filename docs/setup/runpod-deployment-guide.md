# Fun-Audio-Chat RunPod 部署指南

> **版本**: v1.4-20260108
> **适用环境**: RunPod GPU Cloud (A40/4090/A100/H100)
> **最低显存**: 24GB (推理) / 32GB (三服务完整部署)

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

Fun-Audio-Chat 是阿里云的大型音频语言模型，提供完整的语音服务栈：

| 能力 | 说明 | 模型 |
|------|------|------|
| **语音识别 (ASR)** | 语音 → 文本（31种语言） | Fun-ASR-Nano-2512 |
| **语音问答 (S2T)** | 语音输入 → 文本输出 | Fun-Audio-Chat-8B |
| **语音对话 (S2S)** | 语音输入 → 语音输出（流式） | Fun-Audio-Chat-8B |
| **语音合成 (TTS)** | 文本 → 语音 | Fun-CosyVoice3-0.5B |
| **语音函数调用** | 通过语音触发工具调用 | Fun-Audio-Chat-8B |
| **语音情感共鸣** | 情感感知的语音响应 | Fun-Audio-Chat-8B |

### 核心技术

**语音对话 (Fun-Audio-Chat-8B)**
- 双分辨率语音表征: 5Hz共享骨干 + 25Hz精细化头部
- 模型规模: ~8B 参数

**语音识别 (Fun-ASR-Nano-2512)**
- 端到端语音识别大模型
- 千万小时级真实语音数据训练
- 支持31种语言 + 7大方言
- 低延迟实时转录
- 模型规模: ~0.8B 参数

**语音合成 (Fun-CosyVoice3-0.5B)**
- 高质量流式语音合成
- 模型规模: ~0.5B 参数

---

## 环境要求

```yaml
硬件:
  GPU: NVIDIA A40/4090/A100/H100 (>= 24GB VRAM)
  基础服务显存: ~22GB (S2S + TTS)
  完整服务显存: ~25GB (S2S + TTS + ASR)
  推荐: A40 (48GB) - 显存充足，支持完整服务栈

软件:
  Python: 3.11+ (推荐 3.12)
  PyTorch: 2.8.0 + CUDA 12.8
  ffmpeg: 必需
  funasr: ASR 服务依赖
```

### GPU 选择建议

| GPU | VRAM | 价格 | 适用场景 |
|-----|------|------|----------|
| **A40** | 48GB | $0.40/hr | **推荐** - 显存充足，完整服务栈 |
| RTX 4090 | 24GB | $0.50-0.70/hr | 基础服务，ASR 需独立部署 |
| A100 | 80GB | $1.89/hr | 多模型/训练场景 |

### 显存分配参考

| 组件 | 显存占用 | 说明 |
|------|----------|------|
| S2S 模型 (Fun-Audio-Chat-8B) | ~18GB | 语音对话核心 |
| TTS 模型 (CosyVoice3-0.5B) | ~4GB | 语音合成 |
| ASR 模型 (Fun-ASR-Nano-2512) | ~3GB | 语音识别 |
| **总计** | **~25GB** | 完整语音服务 |

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

# 语音对话模型 (S2S)
hf download FunAudioLLM/Fun-Audio-Chat-8B --local-dir ./pretrained_models/Fun-Audio-Chat-8B

# 语音合成模型 (TTS)
hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir ./pretrained_models/Fun-CosyVoice3-0.5B-2512

# 语音识别模型 (ASR)
hf download FunAudioLLM/Fun-ASR-Nano-2512 --local-dir ./pretrained_models/Fun-ASR-Nano-2512
```

或使用 ModelScope（国内更快）：

```bash
modelscope download --model FunAudioLLM/Fun-Audio-Chat-8B --local_dir pretrained_models/Fun-Audio-Chat-8B
modelscope download --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local_dir pretrained_models/Fun-CosyVoice3-0.5B-2512
modelscope download --model FunAudioLLM/Fun-ASR-Nano-2512 --local_dir pretrained_models/Fun-ASR-Nano-2512
```

### 7. 安装 ASR 依赖

```bash
# Fun-ASR 依赖
pip install funasr
```

---

## 启动服务

### 单 GPU 配置 (A40/4090) - 基础服务

```bash
cd /workspace/Fun-Audio-Chat
export PYTHONPATH=$(pwd)

# 后台启动 S2S + TTS (单GPU使用 --tts-gpu 0)
nohup python3 -m web_demo.server.server \
    --model-path pretrained_models/Fun-Audio-Chat-8B \
    --port 8002 \
    --tts-gpu 0 \
    --host 0.0.0.0 \
    > server.log 2>&1 &
```

### 完整服务配置 (含 ASR)

#### 启动 ASR 服务

```bash
cd /workspace/Fun-Audio-Chat
export PYTHONPATH=$(pwd)

# 后台启动 ASR 服务 (端口 8003)
nohup python3 -m web_demo.server.asr_server \
    --model-path pretrained_models/Fun-ASR-Nano-2512 \
    --port 8003 \
    --host 0.0.0.0 \
    > asr_server.log 2>&1 &
```

#### ASR 服务代码示例

在 `web_demo/server/` 目录下创建 `asr_server.py`：

```python
"""
ASR Server for Fun-ASR-Nano-2512
提供独立的语音识别 API 服务
"""
import argparse
import asyncio
from aiohttp import web
import torch
from funasr import AutoModel

def log(level, message):
    print(f"[{level.upper()}] {message}")

class ASRServer:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.device = device
        log("info", f"Loading ASR model from {model_path}...")

        self.model = AutoModel(
            model=model_path,
            trust_remote_code=True,
            device=device,
        )
        log("info", "ASR model loaded successfully")

    async def handle_transcribe(self, request):
        """POST /api/transcribe - 语音转文本"""
        try:
            data = await request.post()
            audio_file = data.get('audio')
            language = data.get('language', '中文')
            hotwords = data.get('hotwords', '')

            if audio_file is None:
                return web.json_response(
                    {"error": "No audio file provided"},
                    status=400
                )

            # 保存临时文件
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
                f.write(audio_file.file.read())
                temp_path = f.name

            try:
                # ASR 推理
                hotwords_list = [h.strip() for h in hotwords.split(',') if h.strip()]
                result = self.model.generate(
                    input=[temp_path],
                    cache={},
                    batch_size=1,
                    hotwords=hotwords_list if hotwords_list else None,
                    language=language,
                    itn=True,
                )
                text = result[0]["text"] if result else ""

                return web.json_response({
                    "text": text,
                    "language": language,
                    "success": True
                })
            finally:
                os.unlink(temp_path)

        except Exception as e:
            log("error", f"Transcription failed: {e}")
            return web.json_response(
                {"error": str(e), "success": False},
                status=500
            )

    async def handle_health(self, request):
        """GET /health - 健康检查"""
        return web.json_response({
            "status": "healthy",
            "model": "Fun-ASR-Nano-2512",
            "device": self.device
        })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8003, type=int)
    parser.add_argument("--model-path", type=str,
                        default="pretrained_models/Fun-ASR-Nano-2512")
    parser.add_argument("--device", default="cuda:0", type=str)
    args = parser.parse_args()

    server = ASRServer(args.model_path, args.device)

    app = web.Application()
    app.router.add_post("/api/transcribe", server.handle_transcribe)
    app.router.add_get("/health", server.handle_health)

    log("info", f"ASR Server starting at http://{args.host}:{args.port}")
    web.run_app(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
```

### 双 GPU 配置 (推荐)

```bash
# GPU 0: S2S 模型
# GPU 1: TTS + ASR 模型
python3 -m web_demo.server.server \
    --model-path pretrained_models/Fun-Audio-Chat-8B \
    --port 8002 \
    --tts-gpu 1 \
    --host 0.0.0.0

# 另一个终端启动 ASR
python3 -m web_demo.server.asr_server \
    --model-path pretrained_models/Fun-ASR-Nano-2512 \
    --port 8003 \
    --device cuda:1 \
    --host 0.0.0.0
```

### 验证启动

```bash
# 检查 S2S/TTS 服务端口
ss -tlnp | grep 8002

# 检查 ASR 服务端口
ss -tlnp | grep 8003

# 检查 S2S 日志
tail -f server.log

# 检查 ASR 日志
tail -f asr_server.log

# 关键成功日志:
# [INFO] s2s model loaded (: cuda)
# [INFO] cosyvoice loaded
# [INFO] ASR model loaded successfully
```

### 测试 ASR 服务

```bash
# 健康检查
curl http://localhost:8003/health

# 语音转文本测试
curl -X POST http://localhost:8003/api/transcribe \
    -F "audio=@test.wav" \
    -F "language=中文"
```

### 资源占用

| 组件 | 显存 | 端口 |
|------|------|------|
| S2S 模型 (Fun-Audio-Chat-8B) | ~18GB | 8002 |
| TTS 模型 (CosyVoice3-0.5B) | ~4GB | 8002 |
| ASR 模型 (Fun-ASR-Nano-2512) | ~3GB | 8003 |
| **总计** | **~25GB** | - |

---

## API 接口

### 服务端点概览

| 服务 | 端点 | 协议 | 说明 |
|------|------|------|------|
| S2S 对话 | `/api/chat` | WebSocket | 语音输入→语音输出 |
| ASR 转写 | `/api/transcribe` | HTTP POST | 语音→文本 |
| ASR 健康检查 | `/health` | HTTP GET | ASR 服务状态 |

### S2S WebSocket 端点

```
ws://<public_ip>:<mapped_port>/api/chat
```

例如：`ws://69.30.85.139:22196/api/chat`

### ASR HTTP 端点

```
POST http://<public_ip>:<asr_port>/api/transcribe
GET  http://<public_ip>:<asr_port>/health
```

例如：`http://69.30.85.139:22197/api/transcribe`

### ASR API 详细说明

#### POST /api/transcribe

**请求参数 (multipart/form-data)**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| audio | file | 是 | 音频文件 (wav/mp3/flac) |
| language | string | 否 | 识别语言，默认"中文" |
| hotwords | string | 否 | 热词，逗号分隔 |

**支持语言**
- 中文、英文、日文（单语言模型）
- 31种语言（多语言 MLT 版本）

**响应示例**

```json
{
    "text": "识别出的文本内容",
    "language": "中文",
    "success": true
}
```

**错误响应**

```json
{
    "error": "错误信息",
    "success": false
}
```

#### GET /health

**响应示例**

```json
{
    "status": "healthy",
    "model": "Fun-ASR-Nano-2512",
    "device": "cuda:0"
}
```

### S2S 协议说明

Server 使用二进制 WebSocket 协议：

| 消息类型 | 格式 | 说明 |
|----------|------|------|
| audio | `\x01` + opus_bytes | Opus 编码音频 |
| text | `\x02` + utf8_bytes | 文本消息 |
| control | JSON | 控制信号 (start/pause/endTurn) |
| metadata | JSON | 会话元数据 (system_prompt) |

### 客户端交互流程

**S2S 对话流程**
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

**ASR 转写流程**
```
Client                          Server
  |                               |
  |-------- POST audio file ----->|
  |<------- JSON response --------|
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

echo "=== Fun-Audio-Chat Complete Setup ==="

# 安装系统依赖
apt update && apt install -y ffmpeg

# 安装 Python 依赖
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install sphn aiohttp
pip install -r requirements.txt
pip install 'ruamel.yaml<0.18' --force-reinstall

# 安装 ASR 依赖
pip install funasr

# 下载模型 (如果不存在)
pip install huggingface-hub

# S2S 模型
if [ ! -d "pretrained_models/Fun-Audio-Chat-8B" ]; then
    echo "Downloading Fun-Audio-Chat-8B..."
    huggingface-cli download FunAudioLLM/Fun-Audio-Chat-8B --local-dir ./pretrained_models/Fun-Audio-Chat-8B
fi

# TTS 模型
if [ ! -d "pretrained_models/Fun-CosyVoice3-0.5B-2512" ]; then
    echo "Downloading Fun-CosyVoice3-0.5B-2512..."
    huggingface-cli download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir ./pretrained_models/Fun-CosyVoice3-0.5B-2512
fi

# ASR 模型
if [ ! -d "pretrained_models/Fun-ASR-Nano-2512" ]; then
    echo "Downloading Fun-ASR-Nano-2512..."
    huggingface-cli download FunAudioLLM/Fun-ASR-Nano-2512 --local-dir ./pretrained_models/Fun-ASR-Nano-2512
fi

echo "=== Setup Complete ==="
echo ""
echo "启动 S2S + TTS 服务:"
echo "  python3 -m web_demo.server.server --model-path pretrained_models/Fun-Audio-Chat-8B --port 8002 --tts-gpu 0 --host 0.0.0.0"
echo ""
echo "启动 ASR 服务:"
echo "  python3 -m web_demo.server.asr_server --model-path pretrained_models/Fun-ASR-Nano-2512 --port 8003 --host 0.0.0.0"
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

### Fun-Audio-Chat (S2S)
- [Fun-Audio-Chat GitHub](https://github.com/FunAudioLLM/Fun-Audio-Chat)
- [arXiv Paper](https://arxiv.org/pdf/2512.20156)
- [HuggingFace - Fun-Audio-Chat-8B](https://huggingface.co/FunAudioLLM/Fun-Audio-Chat-8B)
- [Demo Page](https://funaudiollm.github.io/funaudiochat)

### Fun-ASR (ASR)
- [Fun-ASR GitHub](https://github.com/FunAudioLLM/Fun-ASR)
- [HuggingFace - Fun-ASR-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)
- [HuggingFace - Fun-ASR-MLT-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) (多语言版)
- [Fun-ASR Demo Space](https://huggingface.co/spaces/FunAudioLLM/Fun-ASR-Nano)

### Fun-CosyVoice (TTS)
- [HuggingFace - Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)

### RunPod
- [RunPod Docs - Manage Pods](https://docs.runpod.io/pods/manage-pods)
- [RunPod GraphQL API](https://docs.runpod.io/sdks/graphql/manage-pod-templates)

---

## 三服务完整部署 (S2S + TTS + ASR)

> **验证日期**: 2026-01-08
> **验证环境**: RunPod A40 48GB

### 架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Fun-Audio-Chat 三服务架构                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  S2S 对话服务        │  │  TTS 合成服务    │  │  ASR 识别服务    │ │
│  │  Port: 8002         │  │  Port: 8004     │  │  Port: 8003     │ │
│  │  协议: WebSocket    │  │  协议: HTTP     │  │  协议: HTTP     │ │
│  │                     │  │                 │  │                 │ │
│  │  Fun-Audio-Chat-8B  │  │  CosyVoice3     │  │  Fun-ASR-Nano   │ │
│  │  + 内置 TTS         │  │  (vLLM 可选)    │  │  (funasr)       │ │
│  │  (~22GB)            │  │  (~4GB)         │  │  (~3GB)         │ │
│  └─────────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                     │
│  总显存: ~29GB (推荐 A40 48GB)                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 服务端口分配

| 服务 | 内部端口 | 功能 | 协议 |
|------|----------|------|------|
| S2S | 8002 | 语音对话 (输入语音→输出语音+文本) | WebSocket |
| ASR | 8003 | 语音识别 (语音→文本) | HTTP |
| TTS | 8004 | 语音合成 (文本→语音) | HTTP |

### 启动三服务

```bash
cd /workspace/Fun-Audio-Chat
export PYTHONPATH=$(pwd)

# 1. 启动 S2S + 内置TTS 服务 (端口 8002)
nohup python3 -m web_demo.server.server \
    --model-path pretrained_models/Fun-Audio-Chat-8B \
    --port 8002 \
    --tts-gpu 0 \
    --host 0.0.0.0 \
    > server.log 2>&1 &

# 等待 S2S 模型加载完成 (~2-3分钟)
sleep 180

# 2. 启动独立 TTS 服务 (端口 8004, 支持 vLLM 加速)
nohup python3 -m web_demo.server.tts_server \
    --model-path pretrained_models/Fun-CosyVoice3-0.5B-2512 \
    --port 8004 \
    --device cuda:0 \
    --use-vllm \
    --host 0.0.0.0 \
    > tts_server.log 2>&1 &

# 3. 启动 ASR 服务 (端口 8003)
nohup python3 -m web_demo.server.asr_server \
    --model-path pretrained_models/Fun-ASR-Nano-2512 \
    --port 8003 \
    --device cuda:0 \
    --host 0.0.0.0 \
    > asr_server.log 2>&1 &
```

### 验证部署

```bash
# 检查所有服务端口
ss -tlnp | grep -E '8002|8003|8004'

# ASR 健康检查
curl http://localhost:8003/health

# TTS 健康检查
curl http://localhost:8004/health

# 检查日志
tail -f server.log tts_server.log asr_server.log
```

### API 测试

**ASR 测试 (语音转文本)**:
```bash
curl -X POST http://<host>:<asr_port>/api/transcribe \
    -F "audio=@test.wav" \
    -F "language=中文"
```

**TTS 测试 (文本转语音)**:
```bash
curl -X POST http://<host>:<tts_port>/api/synthesize \
    -H "Content-Type: application/json" \
    -d '{"text": "你好，这是测试", "speaker_id": "中文女"}'
```

### RunPod 端口映射示例

当前部署实例 (2026-01-08):
- **Public IP**: 69.30.85.123
- **SSH**: 22198
- **S2S (8002)**: 22196
- **ASR (8003)**: 22126
- **TTS (8004)**: 22125

```bash
# 外部访问示例
curl http://69.30.85.123:22126/health   # ASR
curl http://69.30.85.123:22125/health   # TTS
```

### 一键启动脚本

创建 `scripts/start_all_services.sh`:

```bash
#!/bin/bash
set -e

cd /workspace/Fun-Audio-Chat
export PYTHONPATH=$(pwd)

echo "=== 启动 Fun-Audio-Chat 三服务 ==="

# 停止现有服务
pkill -f 'web_demo.server' 2>/dev/null || true
sleep 2

# 1. S2S 服务
echo "[1/3] 启动 S2S 服务..."
nohup python3 -m web_demo.server.server \
    --model-path pretrained_models/Fun-Audio-Chat-8B \
    --port 8002 --tts-gpu 0 --host 0.0.0.0 \
    > server.log 2>&1 &
S2S_PID=$!
echo "S2S PID: $S2S_PID"

# 等待模型加载
echo "等待 S2S 模型加载 (约 3 分钟)..."
sleep 180

# 2. TTS 服务
echo "[2/3] 启动 TTS 服务..."
nohup python3 -m web_demo.server.tts_server \
    --model-path pretrained_models/Fun-CosyVoice3-0.5B-2512 \
    --port 8004 --device cuda:0 --use-vllm --host 0.0.0.0 \
    > tts_server.log 2>&1 &
TTS_PID=$!
echo "TTS PID: $TTS_PID"

# 3. ASR 服务
echo "[3/3] 启动 ASR 服务..."
nohup python3 -m web_demo.server.asr_server \
    --model-path pretrained_models/Fun-ASR-Nano-2512 \
    --port 8003 --device cuda:0 --host 0.0.0.0 \
    > asr_server.log 2>&1 &
ASR_PID=$!
echo "ASR PID: $ASR_PID"

# 等待服务启动
sleep 30

# 验证
echo ""
echo "=== 验证服务 ==="
echo "ASR: $(curl -s http://localhost:8003/health | grep -o '"status":"[^"]*"')"
echo "TTS: $(curl -s http://localhost:8004/health | grep -o '"status":"[^"]*"')"

echo ""
echo "=== 部署完成 ==="
echo "S2S: ws://0.0.0.0:8002/api/chat"
echo "ASR: http://0.0.0.0:8003/api/transcribe"
echo "TTS: http://0.0.0.0:8004/api/synthesize"
```

### 停止所有服务

```bash
pkill -f 'web_demo.server'
```

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.4 | 2026-01-08 | 添加三服务完整部署章节，含验证过的端口映射 |
| v1.3 | 2026-01-08 | 添加 ASR 独立服务，三服务架构文档 |
| v1.2 | 2026-01-07 | 添加自动化部署脚本 |
| v1.1 | 2026-01-06 | 添加 GPU 选择建议，常见问题 |
| v1.0 | 2026-01-05 | 初始版本 |
