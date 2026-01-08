#!/bin/bash
# =============================================================================
# Fun-Audio-Chat: 停止所有服务
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOGS_DIR="$PROJECT_ROOT/logs"

echo "停止所有 Fun-Audio-Chat 服务..."

# 停止通过 PID 文件记录的服务
stop_by_pid() {
    local name=$1
    local pid_file="$LOGS_DIR/${name}.pid"

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "停止 $name (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
            rm -f "$pid_file"
        else
            echo "$name 已停止"
            rm -f "$pid_file"
        fi
    fi
}

# 停止各服务
stop_by_pid "s2s_server"
stop_by_pid "tts_server"
stop_by_pid "asr_server"

# 清理可能残留的进程
echo "清理残留进程..."
pkill -f 'web_demo.server.server' 2>/dev/null || true
pkill -f 'web_demo.server.tts_server' 2>/dev/null || true
pkill -f 'web_demo.server.asr_server' 2>/dev/null || true
pkill -f 'vllm.entrypoints.openai.api_server.*whisper' 2>/dev/null || true

echo "所有服务已停止"
