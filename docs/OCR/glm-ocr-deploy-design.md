# GLM-OCR 高性能 API 服务 - 设计规范文档

> 版本: v1.0 | 2026-02-06

## 1. 架构概览

```
Client Request
     │
     ▼
FastAPI (port 8007, async, uvicorn)
├── 简单识别路由 (直接 async httpx → vLLM)
│   ├── POST /api/ocr/recognize     单图识别(文本/公式/表格)
│   ├── POST /api/ocr/extract       信息提取(JSON schema)
│   └── POST /api/ocr/batch         批量图片识别
├── 文档解析路由 (PP-DocLayout + vLLM)
│   ├── POST /api/ocr/document      完整文档解析(布局分析+并行OCR)
│   └── POST /api/ocr/document/stream  SSE流式文档解析
└── 管理路由
    ├── GET  /health                 健康检查
    └── GET  /api/info               服务信息
           │
           ▼
    ┌──────────────────────┐
    │  vLLM Server (8000)  │
    │  zai-org/GLM-OCR     │
    │  + MTP 推测解码      │
    │  ~4GB VRAM           │
    └──────────────────────┘
    ┌──────────────────────┐
    │  PP-DocLayout-V3     │
    │  (paddlex, CPU/GPU)  │
    │  布局检测 + NMS      │
    └──────────────────────┘
```

## 2. 核心组件

### 2.1 vLLM Server (GLM-OCR)

- **模型**: `zai-org/GLM-OCR` (0.9B 参数)
- **推测解码**: MTP (Multi-Token Prediction), `num_speculative_tokens=1`
- **端口**: 8000 (内部)
- **VRAM**: ~4GB 模型 + ~2GB overhead
- **并发**: `max-num-seqs=64`

**启动命令**:
```bash
vllm serve zai-org/GLM-OCR \
    --port 8000 \
    --allowed-local-media-path / \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
    --served-model-name glm-ocr \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64
```

### 2.2 FastAPI OCR Server

- **端口**: 8007 (对外)
- **框架**: FastAPI + uvicorn
- **HTTP 客户端**: httpx.AsyncClient (连接池)
- **并发控制**: asyncio.Semaphore (默认 max=16)
- **CPU 密集任务**: ThreadPoolExecutor (PDF 渲染 + 布局检测)

### 2.3 PP-DocLayout-V3

- **框架**: PaddleX
- **功能**: 文档布局分析（检测文本区域、公式、表格等）
- **执行**: CPU 线程池（不占用 GPU）
- **输出**: 区域列表 `[{label, bbox, score}]`

## 3. API 规范

### 3.1 POST /api/ocr/recognize - 单图识别

**请求**: `multipart/form-data`

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| image | UploadFile | (必填) | 图片文件 |
| task | string | "text" | text / formula / table |
| max_tokens | int | 8192 | 最大输出 token |

**响应**:
```json
{
    "text": "识别出的文本内容...",
    "task": "text",
    "time": 1.23
}
```

### 3.2 POST /api/ocr/extract - 信息提取

**请求**: `multipart/form-data`

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| image | UploadFile | (必填) | 图片文件 |
| schema | string | (必填) | JSON schema 字符串 |
| max_tokens | int | 8192 | 最大输出 token |

**响应**:
```json
{
    "data": {"name": "张三", "id_number": "110101..."},
    "raw": "{\"name\": \"张三\", ...}",
    "time": 2.15
}
```

### 3.3 POST /api/ocr/batch - 批量识别

**请求**: `multipart/form-data`

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| images | List[UploadFile] | (必填) | 多张图片 |
| task | string | "text" | text / formula / table |
| max_tokens | int | 8192 | 最大输出 token |

**响应**:
```json
{
    "results": [
        {"index": 0, "text": "..."},
        {"index": 1, "text": "..."}
    ],
    "total_time": 3.45,
    "throughput": "2.9 imgs/s"
}
```

### 3.4 POST /api/ocr/document - 文档解析

**请求**: `multipart/form-data`

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| file | UploadFile | (必填) | PDF 或图片 |
| pages | string | "all" | 页码规范 (all / 1-5 / 1,3,5) |
| enable_layout | bool | true | 是否启用布局检测 |
| dpi | int | 200 | PDF 渲染 DPI |
| task | string | "text" | 默认 OCR 任务 |
| max_tokens | int | 8192 | 最大输出 token |

**响应**:
```json
{
    "pages": [
        {
            "page_number": 1,
            "text": "全部文本...",
            "regions": [
                {
                    "label": "text",
                    "bbox": [100, 50, 500, 200],
                    "score": 0.95,
                    "task": "text",
                    "text": "区域文本..."
                }
            ],
            "region_count": 5,
            "width": 1654,
            "height": 2339,
            "processing_time": 2.3
        }
    ],
    "total_pages": 10,
    "parsed_pages": 5,
    "total_time": 11.5,
    "throughput": "0.43 pgs/s"
}
```

### 3.5 POST /api/ocr/document/stream - SSE 流式

响应: `text/event-stream`

```
data: {"type": "start", "total_pages": 10, "pages_to_parse": 5}

data: {"type": "page", "page_number": 1, "text": "...", "progress": "1/5"}

data: {"type": "page", "page_number": 2, "text": "...", "progress": "2/5"}

data: {"type": "complete", "total_time": 11.5, "throughput": "0.43 pgs/s"}
```

### 3.6 GET /health - 健康检查

```json
{
    "status": "healthy",
    "vllm": true,
    "layout_model": true,
    "model": "glm-ocr"
}
```

### 3.7 GET /api/info - 服务信息

```json
{
    "model": "glm-ocr",
    "version": "0.9B",
    "vllm_endpoint": "http://localhost:8000",
    "layout_detection": true,
    "features": ["Text Recognition", "Formula Recognition", ...],
    "endpoints": {...}
}
```

## 4. 高性能并发设计

### 4.1 连接池

```python
httpx.AsyncClient(
    timeout=Timeout(120.0, connect=10.0),
    limits=Limits(max_connections=64, max_keepalive_connections=32),
)
```

- 复用 TCP 连接，避免频繁握手
- 64 最大连接数匹配 vLLM `max-num-seqs=64`

### 4.2 并发控制

```python
self.ocr_semaphore = asyncio.Semaphore(16)
```

- 防止过多并发请求导致 vLLM OOM
- 默认 16，可通过 `--max-concurrent-ocr` 调节

### 4.3 CPU 密集任务隔离

```python
self.cpu_executor = ThreadPoolExecutor(max_workers=4)

# PDF 渲染
await loop.run_in_executor(self.cpu_executor, self.render_page_sync, ...)

# 布局检测
await loop.run_in_executor(self.cpu_executor, self._detect_layout_sync, ...)
```

### 4.4 并行文档处理

```python
# 文档内多区域并行 OCR
region_results = await asyncio.gather(*ocr_tasks, return_exceptions=True)

# 多页并行处理
page_results = await asyncio.gather(*page_tasks, return_exceptions=True)
```

### 4.5 流式读取

- `UploadFile` 流式读取，避免大文件内存爆炸
- SSE 流式响应，逐页返回结果

## 5. GLM-OCR Prompt 设计

```python
TASK_PROMPTS = {
    "text": "Text Recognition:",
    "formula": "Formula Recognition:",
    "table": "Table Recognition:",
}
```

- **文本识别**: 直接输出文本
- **公式识别**: 输出 LaTeX
- **表格识别**: 输出 HTML/Markdown 格式
- **信息提取**: 用户传入 JSON schema 作为 prompt

## 6. 部署流程

### 6.1 自动部署 (RunPod)

```bash
python3 scripts/ocr_deploy.py --gpu A40 --benchmark
```

**9 个 Phase**:
1. Create Pod → RunPod API 创建 GPU Pod
2. Wait for Pod Ready → 等待 RUNNING 状态
3. SSH Connect → 建立 SSH 连接
4. Environment Setup → 安装依赖 (transformers source + paddlex)
5. Start vLLM → 启动 vLLM (GLM-OCR + MTP)
6. Wait for vLLM Ready → 等待模型加载
7. Start OCR API → 启动 FastAPI 服务
8. Verify Services → 健康检查验证
9. Benchmark → 性能基准测试

### 6.2 Claude Skill

```bash
/ocr-deploy --gpu A40              # 部署
/ocr-deploy --gpu A40 --benchmark  # 部署 + 基准测试
/ocr-deploy --status               # 查看状态
/ocr-deploy --plan --gpu A40       # 预览部署计划
```

### 6.3 依赖安装

```bash
# 系统依赖
apt update && apt install -y poppler-utils bc

# Python 依赖 (注意顺序)
pip install git+https://github.com/huggingface/transformers.git  # 源码安装
pip install vllm>=0.11.1
pip install paddlepaddle paddlex
pip install fastapi uvicorn httpx pypdfium2 pillow python-multipart
```

## 7. 性能基准

### 7.1 预估指标

| GPU | 单图延迟 | 文档吞吐 | 日容量 | VRAM 使用 |
|-----|---------|---------|--------|----------|
| A40 | ~0.5s | ~1.86 pgs/s | ~160K pages | ~6GB |
| A100 | ~0.4s | ~2.5 pgs/s | ~216K pages | ~6GB |
| H100 | ~0.3s | ~3.5 pgs/s | ~302K pages | ~6GB |

### 7.2 并发性能

| 并发数 | A40 吞吐 | A100 吞吐 |
|--------|---------|----------|
| 1 | ~2 imgs/s | ~2.5 imgs/s |
| 5 | ~5 imgs/s | ~7 imgs/s |
| 10 | ~8 imgs/s | ~12 imgs/s |
| 16 | ~10 imgs/s | ~15 imgs/s |

*注: 实际吞吐取决于图像复杂度和文本长度*

## 8. 错误处理

| 错误场景 | 处理方式 |
|---------|---------|
| vLLM 不可用 | health 返回 degraded，OCR 请求返回 500 |
| 图片格式错误 | 400 + 错误消息 |
| PDF 格式错误 | 400 + "Invalid PDF format" |
| 页码超范围 | 自动忽略无效页码 |
| 单区域 OCR 失败 | gather + return_exceptions，其他区域正常 |
| Layout 模型不可用 | 降级为全页 OCR（无区域分割） |
| vLLM 超时 | httpx 120s timeout |

## 9. 端口分配

| 服务 | 端口 | 说明 |
|------|------|------|
| vLLM (GLM-OCR) | 8000 | 内部，OpenAI-compatible API |
| OCR API Server | 8007 | 对外，FastAPI |

## 10. 文件结构

```
web_demo/server/ocr_server.py    # FastAPI OCR API 服务
scripts/ocr_deploy.py            # RunPod 自动部署脚本
scripts/runpod_manager.py        # RunPod API 管理 (复用)
.claude/commands/ocr-deploy.md   # Claude Skill 定义
docs/OCR/glm-ocr-deploy-design.md  # 本文档
```
