#!/bin/bash
# Fun-Audio-Chat 多服务一键启动脚本
# 版本: v1.1 (2026-01-21)
# 验证: RunPod A40/H100
#
# 支持服务:
#   - S2S: Speech-to-Speech (WebSocket)
#   - ASR: Automatic Speech Recognition (HTTP)
#   - TTS: Text-to-Speech (HTTP)
#   - PDF: PDF-AI Parsing (HTTP, 需要 vLLM 后端)

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 默认配置
S2S_PORT=${S2S_PORT:-8002}
ASR_PORT=${ASR_PORT:-8003}
TTS_PORT=${TTS_PORT:-8004}
PDF_PORT=${PDF_PORT:-8006}
VLLM_PORT=${VLLM_PORT:-8000}
TTS_GPU=${TTS_GPU:-0}
ASR_DEVICE=${ASR_DEVICE:-cuda:0}
TTS_DEVICE=${TTS_DEVICE:-cuda:0}
USE_VLLM=${USE_VLLM:-true}
ENABLE_PDF=${ENABLE_PDF:-false}  # PDF 服务默认禁用（需要额外 vLLM）
WAIT_S2S=${WAIT_S2S:-180}  # S2S 模型加载等待时间(秒)
WAIT_VLLM=${WAIT_VLLM:-180}  # vLLM 模型加载等待时间(秒)

# 工作目录
WORK_DIR=${WORK_DIR:-/workspace/Fun-Audio-Chat}
cd "$WORK_DIR"
export PYTHONPATH=$(pwd)

echo "=============================================="
echo "   Fun-Audio-Chat 多服务启动脚本"
echo "=============================================="
echo ""
log_info "工作目录: $WORK_DIR"
log_info "S2S 端口: $S2S_PORT"
log_info "ASR 端口: $ASR_PORT"
log_info "TTS 端口: $TTS_PORT"
log_info "TTS GPU: $TTS_GPU"
log_info "vLLM 加速: $USE_VLLM"
if [ "$ENABLE_PDF" = "true" ]; then
    log_info "PDF 服务: 启用 (端口 $PDF_PORT)"
    log_info "vLLM OCR 端口: $VLLM_PORT"
fi
echo ""

# 停止现有服务
log_warn "停止现有服务..."
pkill -f 'web_demo.server' 2>/dev/null || true
pkill -f 'vllm serve' 2>/dev/null || true
sleep 2

# 计算总服务数
TOTAL_SERVICES=3
if [ "$ENABLE_PDF" = "true" ]; then
    TOTAL_SERVICES=5  # +vLLM +PDF
fi

# 1. 启动 S2S 服务
log_info "[1/$TOTAL_SERVICES] 启动 S2S 服务..."
nohup python3 -m web_demo.server.server \
    --model-path pretrained_models/Fun-Audio-Chat-8B \
    --port $S2S_PORT \
    --tts-gpu $TTS_GPU \
    --host 0.0.0.0 \
    > server.log 2>&1 &
S2S_PID=$!
log_info "S2S 服务 PID: $S2S_PID"

# 等待 S2S 模型加载
log_info "等待 S2S 模型加载 (约 ${WAIT_S2S} 秒)..."
sleep $WAIT_S2S

# 检查 S2S 服务是否启动
if ! ps -p $S2S_PID > /dev/null 2>&1; then
    log_error "S2S 服务启动失败，请检查 server.log"
    tail -20 server.log
    exit 1
fi
log_info "S2S 服务已启动"

# 2. 启动 TTS 服务
log_info "[2/$TOTAL_SERVICES] 启动 TTS 服务..."
TTS_ARGS="--model-path pretrained_models/Fun-CosyVoice3-0.5B-2512 --port $TTS_PORT --device $TTS_DEVICE --host 0.0.0.0"
if [ "$USE_VLLM" = "true" ]; then
    TTS_ARGS="$TTS_ARGS --use-vllm"
fi

nohup python3 -m web_demo.server.tts_server $TTS_ARGS > tts_server.log 2>&1 &
TTS_PID=$!
log_info "TTS 服务 PID: $TTS_PID"

# 3. 启动 ASR 服务
log_info "[3/$TOTAL_SERVICES] 启动 ASR 服务..."
nohup python3 -m web_demo.server.asr_server \
    --model-path pretrained_models/Fun-ASR-Nano-2512 \
    --port $ASR_PORT \
    --device $ASR_DEVICE \
    --host 0.0.0.0 \
    > asr_server.log 2>&1 &
ASR_PID=$!
log_info "ASR 服务 PID: $ASR_PID"

# 4-5. 启动 PDF 服务 (可选)
if [ "$ENABLE_PDF" = "true" ]; then
    # 4. 启动 vLLM OCR 服务
    log_info "[4/$TOTAL_SERVICES] 启动 vLLM OCR 服务..."
    nohup vllm serve lightonai/LightOnOCR-2-1B \
        --port $VLLM_PORT \
        --limit-mm-per-prompt '{"image": 1}' \
        --mm-processor-cache-gb 0 \
        --no-enable-prefix-caching \
        > vllm.log 2>&1 &
    VLLM_PID=$!
    log_info "vLLM 服务 PID: $VLLM_PID"

    # 等待 vLLM 加载模型
    log_info "等待 vLLM 加载模型 (约 ${WAIT_VLLM} 秒)..."
    sleep $WAIT_VLLM

    # 检查 vLLM 是否就绪
    VLLM_READY=false
    for i in {1..10}; do
        if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
            VLLM_READY=true
            break
        fi
        sleep 10
    done

    if [ "$VLLM_READY" = "true" ]; then
        log_info "vLLM 服务已就绪"

        # 5. 启动 PDF API 服务
        log_info "[5/$TOTAL_SERVICES] 启动 PDF API 服务..."
        nohup python3 -m web_demo.server.pdf_server \
            --port $PDF_PORT \
            --vllm-endpoint http://localhost:$VLLM_PORT \
            --host 0.0.0.0 \
            > pdf_server.log 2>&1 &
        PDF_PID=$!
        log_info "PDF API 服务 PID: $PDF_PID"
    else
        log_error "vLLM 服务启动失败，跳过 PDF API 服务"
        log_warn "请检查 vllm.log"
    fi
fi

# 等待服务启动
log_info "等待服务就绪..."
sleep 30

# 验证服务
echo ""
echo "=============================================="
echo "   服务验证"
echo "=============================================="

# ASR 健康检查
ASR_STATUS=$(curl -s http://localhost:$ASR_PORT/health 2>/dev/null | grep -o '"status":"[^"]*"' || echo "FAILED")
if [[ "$ASR_STATUS" == *"healthy"* ]]; then
    log_info "ASR 服务: ${GREEN}健康${NC}"
else
    log_warn "ASR 服务: 未就绪 (可能仍在加载)"
fi

# TTS 健康检查
TTS_STATUS=$(curl -s http://localhost:$TTS_PORT/health 2>/dev/null | grep -o '"status":"[^"]*"' || echo "FAILED")
if [[ "$TTS_STATUS" == *"healthy"* ]]; then
    log_info "TTS 服务: ${GREEN}健康${NC}"
else
    log_warn "TTS 服务: 未就绪 (可能仍在加载)"
fi

# PDF 健康检查 (如果启用)
if [ "$ENABLE_PDF" = "true" ]; then
    PDF_STATUS=$(curl -s http://localhost:$PDF_PORT/health 2>/dev/null | grep -o '"status":"[^"]*"' || echo "FAILED")
    if [[ "$PDF_STATUS" == *"healthy"* ]]; then
        log_info "PDF 服务: ${GREEN}健康${NC}"
    else
        log_warn "PDF 服务: 未就绪 (可能仍在加载)"
    fi
fi

# 端口检查
echo ""
log_info "端口监听状态:"
PORT_PATTERN="$S2S_PORT|$ASR_PORT|$TTS_PORT"
if [ "$ENABLE_PDF" = "true" ]; then
    PORT_PATTERN="$PORT_PATTERN|$VLLM_PORT|$PDF_PORT"
fi
ss -tlnp 2>/dev/null | grep -E "$PORT_PATTERN" || netstat -tlnp 2>/dev/null | grep -E "$PORT_PATTERN"

echo ""
echo "=============================================="
echo "   部署完成"
echo "=============================================="
echo ""
echo "服务端点:"
echo "  S2S: ws://0.0.0.0:$S2S_PORT/api/chat"
echo "  ASR: http://0.0.0.0:$ASR_PORT/api/transcribe"
echo "  TTS: http://0.0.0.0:$TTS_PORT/api/synthesize"
if [ "$ENABLE_PDF" = "true" ]; then
    echo "  PDF: http://0.0.0.0:$PDF_PORT/api/parse"
fi
echo ""
echo "日志文件:"
echo "  S2S: $WORK_DIR/server.log"
echo "  ASR: $WORK_DIR/asr_server.log"
echo "  TTS: $WORK_DIR/tts_server.log"
if [ "$ENABLE_PDF" = "true" ]; then
    echo "  vLLM: $WORK_DIR/vllm.log"
    echo "  PDF: $WORK_DIR/pdf_server.log"
fi
echo ""
echo "停止服务: pkill -f 'web_demo.server' && pkill -f 'vllm serve'"
echo ""
echo "启用 PDF 服务: ENABLE_PDF=true ./scripts/start_all_services.sh"
echo ""
