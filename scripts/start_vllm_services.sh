#!/bin/bash
# =============================================================================
# Fun-Audio-Chat: 全 vLLM 部署启动脚本
#
# 架构: S2S (原生) + TTS (vLLM) + ASR (vLLM Whisper)
#
# 注意: ASR 使用 Whisper-large-v3 替代 Fun-ASR-Nano-2512
#       因为 Fun-ASR 不支持 vLLM
# =============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 默认配置
S2S_PORT=${S2S_PORT:-8002}
TTS_PORT=${TTS_PORT:-8004}
ASR_PORT=${ASR_PORT:-8005}
TTS_GPU=${TTS_GPU:-0}
ASR_GPU=${ASR_GPU:-0}

# 项目路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=============================================="
echo "  Fun-Audio-Chat 全 vLLM 部署"
echo "=============================================="
echo ""
echo "服务配置:"
echo "  - S2S (原生):     Port $S2S_PORT"
echo "  - TTS (vLLM):     Port $TTS_PORT, GPU $TTS_GPU"
echo "  - ASR (vLLM):     Port $ASR_PORT, GPU $ASR_GPU"
echo ""

# 检查 vLLM 安装
check_vllm() {
    log_info "检查 vLLM 安装..."
    if python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')" 2>/dev/null; then
        log_info "vLLM 已安装"
    else
        log_error "vLLM 未安装，请运行: pip install 'vllm[audio]>=0.11.0'"
        exit 1
    fi
}

# 下载 Whisper 模型 (如果需要)
download_whisper() {
    log_info "检查 Whisper 模型..."
    if [ -d "pretrained_models/whisper-large-v3" ]; then
        log_info "Whisper 模型已存在"
    else
        log_info "下载 Whisper-large-v3..."
        huggingface-cli download openai/whisper-large-v3 \
            --local-dir ./pretrained_models/whisper-large-v3
    fi
}

# 启动 S2S 服务 (原生推理)
start_s2s() {
    log_info "启动 S2S 服务 (Port $S2S_PORT)..."

    nohup python3 -m web_demo.server.server \
        --model-path pretrained_models/Fun-Audio-Chat-8B \
        --port $S2S_PORT \
        --tts-gpu $TTS_GPU \
        --host 0.0.0.0 \
        > logs/s2s_server.log 2>&1 &

    echo $! > logs/s2s_server.pid
    log_info "S2S 服务 PID: $(cat logs/s2s_server.pid)"
}

# 启动 TTS 服务 (vLLM 加速)
start_tts_vllm() {
    log_info "启动 TTS 服务 (vLLM, Port $TTS_PORT)..."

    nohup python3 -m web_demo.server.tts_server \
        --model-path pretrained_models/Fun-CosyVoice3-0.5B-2512 \
        --port $TTS_PORT \
        --device cuda:$TTS_GPU \
        --use-vllm \
        --host 0.0.0.0 \
        > logs/tts_server.log 2>&1 &

    echo $! > logs/tts_server.pid
    log_info "TTS 服务 PID: $(cat logs/tts_server.pid)"
}

# 启动 ASR 服务 (vLLM Whisper)
start_asr_vllm() {
    log_info "启动 ASR 服务 (vLLM Whisper, Port $ASR_PORT)..."

    # 使用 vLLM 原生 OpenAI 兼容服务
    nohup python3 -m vllm.entrypoints.openai.api_server \
        --model openai/whisper-large-v3 \
        --port $ASR_PORT \
        --host 0.0.0.0 \
        --task transcription \
        --gpu-memory-utilization 0.2 \
        --device cuda:$ASR_GPU \
        > logs/asr_server.log 2>&1 &

    echo $! > logs/asr_server.pid
    log_info "ASR 服务 PID: $(cat logs/asr_server.pid)"
}

# 等待服务就绪
wait_for_service() {
    local name=$1
    local port=$2
    local max_wait=${3:-120}
    local wait_time=0

    log_info "等待 $name 服务就绪 (Port $port)..."

    while [ $wait_time -lt $max_wait ]; do
        if nc -z localhost $port 2>/dev/null; then
            log_info "$name 服务已就绪"
            return 0
        fi
        sleep 5
        wait_time=$((wait_time + 5))
        echo -n "."
    done

    log_error "$name 服务启动超时"
    return 1
}

# 主函数
main() {
    # 创建日志目录
    mkdir -p logs

    # 检查环境
    check_vllm

    # 可选: 下载 Whisper
    # download_whisper

    # 启动服务
    log_info "开始启动服务..."

    start_s2s
    sleep 5

    start_tts_vllm
    sleep 5

    start_asr_vllm

    echo ""
    log_info "所有服务已启动"
    echo ""
    echo "=============================================="
    echo "  服务端点"
    echo "=============================================="
    echo ""
    echo "  S2S 对话:  ws://0.0.0.0:$S2S_PORT/api/chat"
    echo "  TTS 合成:  http://0.0.0.0:$TTS_PORT/api/synthesize"
    echo "  ASR 识别:  http://0.0.0.0:$ASR_PORT/v1/audio/transcriptions"
    echo ""
    echo "  ASR 调用示例:"
    echo "    curl -X POST http://localhost:$ASR_PORT/v1/audio/transcriptions \\"
    echo "      -F 'file=@audio.wav' \\"
    echo "      -F 'model=openai/whisper-large-v3' \\"
    echo "      -F 'language=zh'"
    echo ""
    echo "  日志文件:"
    echo "    - logs/s2s_server.log"
    echo "    - logs/tts_server.log"
    echo "    - logs/asr_server.log"
    echo ""
    echo "  停止服务: ./scripts/stop_services.sh"
    echo "=============================================="
}

# 运行主函数
main "$@"
