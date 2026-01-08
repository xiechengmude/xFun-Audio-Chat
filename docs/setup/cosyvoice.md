# CosyVoice 部署说明 (Fun-Audio-Chat 集成版)

> **版本**: v1.0-20260108
> **用途**: Fun-Audio-Chat 项目中的 TTS 模块

---

## 概述

Fun-Audio-Chat 使用 **Fun-CosyVoice3-0.5B-2512** 作为 TTS (Text-to-Speech) 模块，该模型已针对流式语音合成进行优化。

**注意**: CosyVoice 已作为 Git 子模块集成到项目中，**不需要单独克隆** CosyVoice 仓库。

---

## 集成方式

### 1. 子模块结构

```
Fun-Audio-Chat/
├── third_party/
│   └── CosyVoice/           # Git 子模块 (自动克隆)
│       └── third_party/
│           └── Matcha-TTS/  # 嵌套子模块
├── utils/
│   └── cosyvoice_detokenizer.py  # TTS 封装调用
└── pretrained_models/
    └── Fun-CosyVoice3-0.5B-2512/  # 预训练模型 (需下载)
```

### 2. 初始化子模块

```bash
# 克隆项目时一并初始化子模块
git clone --recurse-submodules https://github.com/FunAudioLLM/Fun-Audio-Chat

# 或后续补充初始化
cd Fun-Audio-Chat
git submodule update --init --recursive
```

---

## 模型下载

### HuggingFace (推荐海外)

```bash
pip install huggingface-hub

# 下载 Fun-CosyVoice3-0.5B-2512
huggingface-cli download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    --local-dir ./pretrained_models/Fun-CosyVoice3-0.5B-2512
```

### ModelScope (推荐国内)

```bash
pip install modelscope

modelscope download --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    --local_dir pretrained_models/Fun-CosyVoice3-0.5B-2512
```

### 可选: TTS 文本规范化增强包

```bash
# 下载 ttsfrd 资源 (可选，默认使用 wetext)
huggingface-cli download FunAudioLLM/CosyVoice-ttsfrd \
    --local-dir ./pretrained_models/CosyVoice-ttsfrd

# 安装增强包
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```

---

## 依赖安装

CosyVoice 相关依赖已包含在项目 `requirements.txt` 中：

```bash
# 主要依赖
pip install -r requirements.txt

# 关键组件
# - onnxruntime-gpu  # 推理加速
# - conformer        # 模型架构
# - diffusers        # 扩散模型
# - HyperPyYAML      # 配置解析
```

### 依赖冲突修复

```bash
# ruamel.yaml 版本冲突 (必需)
pip install 'ruamel.yaml<0.18' --force-reinstall
```

---

## 显存占用

| 组件 | 显存 | 说明 |
|------|------|------|
| Fun-CosyVoice3-0.5B | ~4GB | 语音合成模型 |

在 Fun-Audio-Chat 服务中，TTS 模块可配置到独立 GPU:

```bash
python3 -m web_demo.server.server \
    --model-path pretrained_models/Fun-Audio-Chat-8B \
    --tts-gpu 1  # TTS 使用 GPU 1
```

---

## 代码调用示例

Fun-Audio-Chat 已封装 CosyVoice 调用，无需直接使用:

```python
# utils/cosyvoice_detokenizer.py 中的封装

from utils.cosyvoice_detokenizer import get_audio_detokenizer, tts_infer_streaming

# 加载 TTS 模型
tts_model = get_audio_detokenizer()

# 流式推理
speech = tts_infer_streaming(
    tts_model,
    speaker_embedding,
    audio_tokens,
    offset,
    session_uuid,
    finalize=False
)
```

---

## 与独立 CosyVoice 的区别

| 特性 | Fun-Audio-Chat 集成 | 独立 CosyVoice |
|------|---------------------|----------------|
| 安装方式 | Git 子模块 (自动) | 独立克隆 |
| 预训练模型 | Fun-CosyVoice3-0.5B-2512 | CosyVoice/CosyVoice2 |
| 调用方式 | 封装在 detokenizer | 直接 CLI/API |
| 流式支持 | 针对 S2S 优化 | 通用流式 |

---

## 常见问题

### Q1: 子模块未初始化

**症状**: `ImportError: No module named 'cosyvoice'`

**解决**:
```bash
git submodule update --init --recursive
```

### Q2: 模型路径错误

**症状**: `FileNotFoundError: pretrained_models/Fun-CosyVoice3-0.5B-2512`

**解决**: 确保模型已下载到正确路径

### Q3: ONNX Runtime CUDA 警告

**症状**: `Failed to create CUDAExecutionProvider`

**影响**: TTS 部分组件回退 CPU，但主功能正常

---

## 参考链接

- [CosyVoice GitHub](https://github.com/FunAudioLLM/CosyVoice)
- [Fun-CosyVoice3-0.5B-2512 (HuggingFace)](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- [CosyVoice3 官方文档](https://funaudiollm.github.io/cosyvoice3/)
