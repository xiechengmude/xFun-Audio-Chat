# GLM-OCR 高性能 OCR API Server 架构总结

> 版本: v1.5 | 2026-02-06 | 基于 RunPod A40 实际部署验证

## 1. 系统架构总览

```
                        ┌─────────────────────────────────────────┐
                        │            Client (Browser/API)          │
                        │  web_demo/ocr/index.html (Tailwind UI)  │
                        └──────────────────┬──────────────────────┘
                                           │ HTTP / SSE
                                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    FastAPI OCR Server (port 8007)                        │
│                    web_demo/server/ocr_server.py                         │
│                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  /recognize  │  │  /extract   │  │   /batch     │  │  /document   │  │
│  │  单图识别    │  │  信息提取   │  │  批量识别    │  │  文档解析    │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                │                │                  │           │
│         ▼                ▼                ▼                  ▼           │
│  ┌────────────────────────────────┐  ┌──────────────────────────┐       │
│  │  httpx.AsyncClient 连接池      │  │  ThreadPoolExecutor (4)  │       │
│  │  max_conn=64, keepalive=32    │  │  CPU: PDF渲染 + 布局检测 │       │
│  └──────────────┬────────────────┘  └────────────┬─────────────┘       │
│                 │                                 │                      │
│  ┌──────────────▼────────────────┐               │                      │
│  │  asyncio.Semaphore (max=16)   │               │                      │
│  │  GPU 并发控制，防止 OOM       │               │                      │
│  └──────────────┬────────────────┘               │                      │
└─────────────────┼────────────────────────────────┼──────────────────────┘
                  │                                │
                  ▼                                ▼
   ┌──────────────────────────┐     ┌──────────────────────────┐
   │   vLLM Server (port 8000) │     │   PP-DocLayout-V3        │
   │   zai-org/GLM-OCR (0.9B)  │     │   PaddleX 布局检测       │
   │   vLLM nightly 必须       │     │   CPU 线程池执行         │
   │   OpenAI-compatible API   │     │   文本/公式/表格区域分割 │
   │   VRAM: 38.7GB / 46GB    │     └──────────────────────────┘
   │   max-num-seqs: 64        │
   └──────────────────────────┘
              GPU (A40 48GB)
```

## 2. 核心技术栈

| 层级 | 技术 | 作用 |
|------|------|------|
| **推理引擎** | vLLM nightly + GLM-OCR 0.9B | 视觉语言 OCR 模型推理 |
| **API 框架** | FastAPI + uvicorn | 异步高并发 HTTP 服务 |
| **HTTP 通信** | httpx.AsyncClient | 连接池复用，高效调用 vLLM |
| **布局检测** | PaddleX PP-DocLayout-V3 | 文档区域分割（文本/公式/表格） |
| **PDF 渲染** | pypdfium2 | PDF 页面转图片 |
| **并发控制** | asyncio.Semaphore + ThreadPoolExecutor | GPU/CPU 任务隔离 |
| **前端** | Tailwind CSS 单页应用 | 拖拽上传 + SSE 流式进度 |
| **部署** | RunPod GPU Cloud + SSH | 一键自动化部署脚本 |

## 3. API 端点

| 端点 | 方法 | 功能 | 输入 | 输出 |
|------|------|------|------|------|
| `/api/ocr/recognize` | POST | 单图识别 | image + task(text/formula/table) | `{text, task, time}` |
| `/api/ocr/extract` | POST | 信息提取 | image + JSON schema | `{data, raw, time}` |
| `/api/ocr/batch` | POST | 批量识别 | images[] + task | `{results[], throughput}` |
| `/api/ocr/document` | POST | 文档解析 | PDF/图片 + pages + dpi | `{pages[], throughput}` |
| `/api/ocr/document/stream` | POST | 流式解析 | 同上 | SSE 逐页推送 |
| `/health` | GET | 健康检查 | - | `{status, vllm, layout_model}` |
| `/api/info` | GET | 服务信息 | - | `{model, version, features}` |

## 4. 高性能并发设计

```
请求进入
  │
  ├── 单图/批量 ──────────────────► httpx.AsyncClient ──► vLLM
  │                                  (连接池 64)         (Semaphore 16 控制)
  │
  └── 文档解析
       │
       ├── 1. PDF 渲染 ──────────► ThreadPoolExecutor ──► pypdfium2 (CPU)
       │
       ├── 2. 布局检测 ──────────► ThreadPoolExecutor ──► PaddleX (CPU)
       │
       └── 3. 区域并行 OCR ──────► asyncio.gather ──────► vLLM (GPU)
            (多区域同时识别)          (Semaphore 限流)
```

**关键设计决策**:
- **连接池 64** 匹配 vLLM `max-num-seqs=64`，TCP 连接复用
- **Semaphore 16** 防止 GPU OOM，可通过 `--max-concurrent-ocr` 调节
- **ThreadPoolExecutor 4** 隔离 CPU 密集任务（PDF渲染/布局检测），不阻塞事件循环
- **asyncio.gather** 文档内多区域并行 OCR，充分利用 GPU 批处理能力

## 5. GLM-OCR 识别能力

| Prompt | 功能 | 输出格式 |
|--------|------|---------|
| `Text Recognition:` | 文本识别 | 纯文本 |
| `Formula Recognition:` | 公式识别 | LaTeX |
| `Table Recognition:` | 表格识别 | HTML/Markdown |
| 用户 JSON schema | 信息提取 | 结构化 JSON |

## 6. 实测性能（A40 48GB）

| 测试项 | 结果 | 备注 |
|--------|------|------|
| 单图 OCR（冷启动） | 6.9s | 首次请求含模型预热 |
| 单图 OCR（热状态） | **1.16s** | 稳定延迟 |
| 5 并发 OCR | **10.0 imgs/s** | 0.50s 总耗时 |
| 信息提取 | 0.86s | JSON schema 输出 |
| 22页中文 PDF 文档 | **19.74s (1.11 pgs/s)** | 房屋租赁备案合同 |
| 2页英文学术论文 | 6.48s (0.31 pgs/s) | 含 LaTeX 数学公式 |
| VRAM 占用 | 38.7GB / 46GB | gpu-memory-utilization=0.85 |

**识别质量验证**:
- 中文合同：条款、checkbox(☑/□)、身份证号、电话、地址均准确
- 英文论文：LaTeX 数学符号（$X_1, \ldots, X_p$, $\perp$）、特殊字符（ñ, ü, ö）正确还原
- 已知弱点：手写体形近字（邓→双）、手写数字+字母混淆（12D→522）

## 7. 依赖安装顺序（关键）

```
pip install vLLM nightly          ← 第一（会 pin transformers<5）
    ↓
pip install paddlepaddle paddlex  ← 第二（布局检测）
    ↓
pip install fastapi uvicorn ...   ← 第三（服务栈）
    ↓
pip install transformers 源码      ← 最后（覆盖 4.x → 5.x.dev，支持 glm_ocr）
```

**为什么顺序重要**: vLLM nightly 依赖 `transformers<5`，pip 会自动降级。transformers 源码必须最后安装以覆盖回 5.x.dev0，GLM-OCR 架构才能被识别。

## 8. 部署踩坑与修复

| 问题 | 根因 | 修复 |
|------|------|------|
| `does not recognize architecture: glm_ocr` | pip transformers 4.x 无 glm_ocr | transformers 源码安装 |
| `Glm46VImageProcessorFast` 报错 | vLLM 稳定版回退到不兼容的 Transformer 后端 | 使用 vLLM **nightly** |
| `OCRModelManager` 未初始化 | uvicorn 以字符串导入模块时不执行 `main()` | `lifespan` startup 从环境变量初始化 |
| PaddleX import 耗时 30s+ | 网络连通性检查 | `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` |
| SSH nohup 返回 exit 255 | 进程未脱离终端会话 | `setsid nohup cmd &` |
| MTP 推测解码推理错误 | vLLM nightly 尚不稳定 | 暂不启用，待修复 |

## 9. 项目文件结构

```
xFun-Audio-Chat/
├── web_demo/
│   ├── server/
│   │   └── ocr_server.py          # FastAPI OCR API 服务（核心）
│   └── ocr/
│       └── index.html              # OCR 前端页面（Tailwind）
├── scripts/
│   ├── ocr_deploy.py               # RunPod 一键部署脚本
│   └── runpod_manager.py           # RunPod API 管理（复用）
├── .claude/commands/
│   └── ocr-deploy.md               # Claude Skill（/ocr-deploy）
└── docs/OCR/
    ├── glm-ocr-deploy-design.md    # 完整设计规范文档
    └── glm-ocr-高性能OCR-API-Server架构.md  # 本文档
```

## 10. 当前部署实例

```
Pod:      glm-ocr-a40 (ID: 5h5kvymeo14fo4)
GPU:      NVIDIA A40 48GB
IP:       194.68.245.30
SSH:      ssh root@194.68.245.30 -p 22159 -i ~/.ssh/id_ed25519
vLLM:     http://194.68.245.30:22160  (内部 8000)
OCR API:  http://194.68.245.30:22167  (内部 8007)
VRAM:     38.7GB / 46GB
```

**快速验证**:
```bash
# 健康检查
curl http://194.68.245.30:22167/health

# 单图 OCR
curl -X POST http://194.68.245.30:22167/api/ocr/recognize \
    -F "image=@test.png" -F "task=text"

# 文档解析
curl -X POST http://194.68.245.30:22167/api/ocr/document \
    -F "file=@document.pdf" -F "pages=1-3"
```
