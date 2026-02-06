# Handoff: GLM-OCR vs LightOnOCR 双模型对比测试

> 创建: 2026-02-06 | 状态: 中断，待继续

## 目标

在同一 A40 Pod 上同时运行两个 OCR 模型，用相同文档做对比测试。

## 当前进度

### 已完成
- [x] GLM-OCR vLLM 降低 GPU 占用率至 0.40（18.1GB / 46GB）启动成功
- [x] LightOnOCR 模型下载完成（`lightonai/LightOnOCR-2-1B`）
- [x] LightOnOCR vLLM 启动命令调试（去掉 `--limit-mm-per-prompt` 参数，nightly 格式变了）
- [x] OCR server（GLM）在 port 8007 运行中

### 未完成
- [ ] LightOnOCR vLLM 加载完成确认（启动了但 Pod 连接断了）
- [ ] 启动 pdf_server.py 对接 LightOnOCR（port 8006）
- [ ] 双模型同文档对比测试
- [ ] 整理对比结果

## Pod 信息

```
Pod: glm-ocr-a40 (ID: 5h5kvymeo14fo4)
IP: 194.68.245.30
SSH Port: 22159
端口映射:
  22160 → 8000 (vLLM GLM-OCR)
  22161 → 8001 (vLLM LightOnOCR)
  22167 → 8007 (OCR API - GLM)
  22166 → 8006 (PDF API - LightOn)
```

## 双模型 vLLM 配置

### GLM-OCR (port 8000)
```bash
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
vllm serve zai-org/GLM-OCR \
    --port 8000 \
    --served-model-name glm-ocr \
    --gpu-memory-utilization 0.40 \
    --max-num-seqs 16
```
- VRAM: ~18.1GB
- 状态: 运行中（已验证 health OK）

### LightOnOCR (port 8001)
```bash
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
vllm serve lightonai/LightOnOCR-2-1B \
    --port 8001 \
    --served-model-name lighton-ocr \
    --gpu-memory-utilization 0.55 \
    --max-num-seqs 16
```
- VRAM: 预计 ~25GB
- 状态: 已启动（PID 5215），加载中断（Pod 连接丢失）

### GPU 分配
| 模型 | utilization | VRAM 预计 | 并发 |
|------|------------|----------|------|
| GLM-OCR (0.9B) | 0.40 | ~18GB | max_seqs=16 |
| LightOnOCR (2.1B) | 0.55 | ~25GB | max_seqs=16 |
| **合计** | 0.95 | ~43GB / 46GB | - |

## 恢复步骤

### 1. 检查 Pod 状态
```bash
# 检查 Pod 是否在运行
python3 scripts/runpod_manager.py --action status

# 如果停了，重启
python3 scripts/runpod_manager.py --action start --pod-id 5h5kvymeo14fo4
```

### 2. SSH 连接
```bash
ssh -i ~/.ssh/id_ed25519 root@194.68.245.30 -p 22159
```

### 3. 检查/重启服务
```bash
# 检查进程
ps aux | grep vllm | grep -v grep
nvidia-smi

# 如果需要重启 GLM-OCR
cd /workspace/Fun-Audio-Chat
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
nohup vllm serve zai-org/GLM-OCR --port 8000 --served-model-name glm-ocr \
    --gpu-memory-utilization 0.40 --max-num-seqs 16 > vllm_glm.log 2>&1 &

# 等 GLM 加载完后启动 LightOnOCR
nohup vllm serve lightonai/LightOnOCR-2-1B --port 8001 --served-model-name lighton-ocr \
    --gpu-memory-utilization 0.55 --max-num-seqs 16 > vllm_lighton.log 2>&1 &

# 等两个都 ready 后启动 API servers
# GLM OCR server
export OCR_VLLM_ENDPOINT=http://localhost:8000 OCR_MODEL_NAME=glm-ocr OCR_ENABLE_LAYOUT=false
setsid nohup python3 -m web_demo.server.ocr_server --port 8007 \
    --vllm-endpoint http://localhost:8000 --no-layout --host 0.0.0.0 > ocr_server.log 2>&1 &

# LightOn PDF server
setsid nohup python3 -m web_demo.server.pdf_server --port 8006 \
    --vllm-url http://localhost:8001 --host 0.0.0.0 > pdf_server.log 2>&1 &
```

### 4. 对比测试
```bash
# 测试文件
TEST_PDF="/Users/gump_m2/Documents/Agent-RL/xFun-Audio-Chat/data/test/房屋租赁备案（住宅）(1).pdf"
TEST_ARXIV="/Users/gump_m2/Documents/Agent-RL/xFun-Audio-Chat/docs/data/arxiv_finance_deep_learning.pdf"

# GLM-OCR 测试
curl -s -X POST 'http://194.68.245.30:22167/api/ocr/document' \
    -F "file=@$TEST_PDF" -F "pages=1-3" | python3 -m json.tool

# LightOnOCR 测试
curl -s -X POST 'http://194.68.245.30:22166/api/parse' \
    -F "file=@$TEST_PDF" -F "pages=1-3" | python3 -m json.tool
```

### 5. 对比维度
| 维度 | 说明 |
|------|------|
| 中文识别准确率 | 合同条款、姓名、地址 |
| 英文/LaTeX | 学术论文、数学公式 |
| 表格结构 | 房屋交付确认书表格 |
| 身份证/证件 | 数字、特殊字符 |
| 单图延迟 | 冷启动 + 热状态 |
| 并发吞吐 | 5 并发 imgs/s |
| 文档解析速度 | pages/s |

## 已有 GLM-OCR 基准数据

| 测试项 | 结果 |
|--------|------|
| 单图热状态 | 1.16s |
| 5 并发 | 10.0 imgs/s |
| 22页中文 PDF | 19.74s (1.11 pgs/s) |
| 2页英文论文 | 6.48s (0.31 pgs/s) |
| 已知错误 | 邓→双（手写），12D→522（手写） |

## 踩坑记录

1. **GPU 显存残留**: `kill -9` vLLM 后显存不释放，需要 `kill -9` 所有子进程（包括 `EngineCore`），可用 `ls /proc/*/maps | xargs grep -l nvidia` 找到隐藏进程
2. **`--limit-mm-per-prompt` 格式变更**: vLLM nightly 中此参数需要 JSON 格式，不再支持 `image=10`
3. **PaddleX 检查**: 每次启动都要 `export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`
4. **gpu-memory-utilization 0.45 失败**: GLM-OCR 需要至少 ~18GB（0.40 刚好够），低于此值 KV cache 分配失败
