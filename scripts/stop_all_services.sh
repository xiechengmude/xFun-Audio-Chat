#!/bin/bash
# Fun-Audio-Chat 服务停止脚本
# 版本: v1.0 (2026-01-08)

echo "停止所有 Fun-Audio-Chat 服务..."

# 停止服务
pkill -f 'web_demo.server.server' 2>/dev/null && echo "S2S 服务已停止" || echo "S2S 服务未运行"
pkill -f 'web_demo.server.tts_server' 2>/dev/null && echo "TTS 服务已停止" || echo "TTS 服务未运行"
pkill -f 'web_demo.server.asr_server' 2>/dev/null && echo "ASR 服务已停止" || echo "ASR 服务未运行"

# 确认停止
sleep 2

# 检查是否还有残留进程
REMAINING=$(pgrep -f 'web_demo.server' 2>/dev/null)
if [ -n "$REMAINING" ]; then
    echo "警告: 仍有服务进程运行: $REMAINING"
    echo "使用 'pkill -9 -f web_demo.server' 强制停止"
else
    echo "所有服务已停止"
fi
