#!/bin/bash
# =============================================================================
# Fun-Audio-Chat: 三服务启动脚本 (混合部署)
#
# 架构:
#   - S2S: Fun-Audio-Chat-8B (原生推理)
#   - TTS: CosyVoice3 (vLLM 加速，可选)
#   - ASR: Fun-ASR-Nano-2512 (funasr 框架)
# =============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 默认配置
S2S_PORT=${S2S_PORT:-8002}
ASR_PORT=${ASR_PORT:-8003}
TTS_PORT=${TTS_PORT:-8004}
TTS_GPU=${TTS_GPU:-0}
ASR_GPU=${ASR_GPU:-0}
USE_VLLM=${USE_VLLM:-false}
ENABLE_TTS=${ENABLE_TTS:-false}  # 独立 TTS 服务默认关闭

# 项目路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 取消代理 (重要!)
unset HTTP_PROXY
unset HTTPS_PROXY
unset http_proxy
unset https_proxy

echo ""
echo -e "${BLUE}=============================================="
echo "  Fun-Audio-Chat 三服务部署"
echo "==============================================${NC}"
echo ""
echo "服务配置:"
echo "  - S2S (原生):        Port $S2S_PORT"
echo "  - ASR (funasr):      Port $ASR_PORT, GPU $ASR_GPU"
if [ "$ENABLE_TTS" = "true" ]; then
    echo "  - TTS (独立服务):    Port $TTS_PORT, GPU $TTS_GPU, vLLM=$USE_VLLM"
else
    echo "  - TTS (内置):        随 S2S 服务启动"
fi
echo ""

# 创建日志目录
mkdir -p logs

# 启动 S2S 服务 (包含内置 TTS)
start_s2s() {
    log_info "启动 S2S + 内置TTS 服务 (Port $S2S_PORT)..."

    nohup python3 -m web_demo.server.server \
        --model-path pretrained_models/Fun-Audio-Chat-8B \
        --port $S2S_PORT \
        --tts-gpu $TTS_GPU \
        --host 0.0.0.0 \
        > logs/s2s_server.log 2>&1 &

    echo $! > logs/s2s_server.pid
    log_info "S2S 服务已启动 (PID: $(cat logs/s2s_server.pid))"
}

# 启动 ASR 服务 (Fun-ASR-Nano-2512)
start_asr() {
    log_info "启动 ASR 服务 (Port $ASR_PORT)..."

    nohup python3 -m web_demo.server.asr_server \
        --model-path pretrained_models/Fun-ASR-Nano-2512 \
        --port $ASR_PORT \
        --device cuda:$ASR_GPU \
        --host 0.0.0.0 \
        > logs/asr_server.log 2>&1 &

    echo $! > logs/asr_server.pid
    log_info "ASR 服务已启动 (PID: $(cat logs/asr_server.pid))"
}

# 启动独立 TTS 服务 (可选)
start_tts() {
    if [ "$ENABLE_TTS" != "true" ]; then
        log_info "独立 TTS 服务已跳过 (使用内置 TTS)"
        return
    fi

    log_info "启动独立 TTS 服务 (Port $TTS_PORT)..."

    local vllm_flag=""
    if [ "$USE_VLLM" = "true" ]; then
        vllm_flag="--use-vllm"
    fi

    nohup python3 -m web_demo.server.tts_server \
        --model-path pretrained_models/Fun-CosyVoice3-0.5B-2512 \
        --port $TTS_PORT \
        --device cuda:$TTS_GPU \
        $vllm_flag \
        --host 0.0.0.0 \
        > logs/tts_server.log 2>&1 &

    echo $! > logs/tts_server.pid
    log_info "TTS 服务已启动 (PID: $(cat logs/tts_server.pid))"
}

# 等待服务就绪
wait_for_service() {
    local name=$1
    local port=$2
    local max_wait=${3:-180}
    local wait_time=0

    echo -n "等待 $name 就绪"

    while [ $wait_time -lt $max_wait ]; do
        if nc -z localhost $port 2>/dev/null; then
            echo ""
            log_info "$name 已就绪 (Port $port)"
            return 0
        fi
        echo -n "."
        sleep 5
        wait_time=$((wait_time + 5))
    done

    echo ""
    log_error "$name 启动超时 (Port $port)"
    return 1
}

# 显示服务状态
show_status() {
    echo ""
    echo -e "${BLUE}=============================================="
    echo "  服务端点"
    echo "==============================================${NC}"
    echo ""
    echo "  S2S 对话:  ws://0.0.0.0:$S2S_PORT/api/chat"
    echo "  ASR 识别:  http://0.0.0.0:$ASR_PORT/api/transcribe"
    if [ "$ENABLE_TTS" = "true" ]; then
        echo "  TTS 合成:  http://0.0.0.0:$TTS_PORT/api/synthesize"
    fi
    echo ""
    echo "  ASR 调用示例:"
    echo "    curl -X POST http://localhost:$ASR_PORT/api/transcribe \\"
    echo "      -F 'audio=@audio.wav' \\"
    echo "      -F 'language=中文'"
    echo ""
    echo "  日志文件:"
    echo "    - logs/s2s_server.log"
    echo "    - logs/asr_server.log"
    if [ "$ENABLE_TTS" = "true" ]; then
        echo "    - logs/tts_server.log"
    fi
    echo ""
    echo "  停止服务: ./scripts/stop_services.sh"
    echo -e "${BLUE}==============================================${NC}"
}

# 主函数
main() {
    log_info "开始启动服务..."
    echo ""

    # 启动 S2S (最耗时，先启动)
    start_s2s

    # 启动 ASR
    start_asr

    # 启动 TTS (可选)
    start_tts

    echo ""
    log_info "等待模型加载..."
    echo ""

    # 等待服务就绪
    wait_for_service "ASR" $ASR_PORT 120 || true

    echo ""
    log_info "所有服务已启动"

    show_status
}

# 显示帮助
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help       显示帮助"
    echo "  --with-tts       启用独立 TTS 服务"
    echo "  --use-vllm       TTS 使用 vLLM 加速"
    echo ""
    echo "环境变量:"
    echo "  S2S_PORT=8002    S2S 服务端口"
    echo "  ASR_PORT=8003    ASR 服务端口"
    echo "  TTS_PORT=8004    TTS 服务端口"
    echo "  TTS_GPU=0        TTS 使用的 GPU"
    echo "  ASR_GPU=0        ASR 使用的 GPU"
    echo ""
    echo "示例:"
    echo "  ./scripts/start_services.sh"
    echo "  ./scripts/start_services.sh --with-tts --use-vllm"
    echo "  ASR_GPU=1 ./scripts/start_services.sh"
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --with-tts)
            ENABLE_TTS=true
            shift
            ;;
        --use-vllm)
            USE_VLLM=true
            shift
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 运行主函数
main
