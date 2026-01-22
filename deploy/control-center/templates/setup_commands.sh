#!/bin/bash
# PDF-AI Service Setup Script Template
# This script is executed on the RunPod pod to set up the environment

set -e

REPO_URL="${REPO_URL:-https://github.com/xiechengmude/xFun-Audio-Chat}"
VLLM_PORT="${VLLM_PORT:-8000}"
API_PORT="${API_PORT:-8006}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

echo "=============================================="
echo "PDF-AI Service Setup"
echo "=============================================="
echo "Repository: $REPO_URL"
echo "vLLM Port: $VLLM_PORT"
echo "API Port: $API_PORT"
echo "GPU Memory Util: $GPU_MEMORY_UTIL"
echo "Max Sequences: $MAX_NUM_SEQS"
echo "=============================================="

# Phase 1: System Dependencies
echo ""
echo "=== Phase 1: System Dependencies ==="
apt update
apt install -y poppler-utils bc

# Phase 2: Clone/Update Repository
echo ""
echo "=== Phase 2: Repository Setup ==="
cd /workspace

if [ -d "Fun-Audio-Chat" ]; then
    echo "Repository exists, pulling latest..."
    cd Fun-Audio-Chat
    git pull origin main || true
else
    echo "Cloning repository..."
    git clone --recurse-submodules "$REPO_URL" Fun-Audio-Chat
    cd Fun-Audio-Chat
fi

# Phase 3: Python Dependencies
echo ""
echo "=== Phase 3: Python Dependencies ==="
pip install --upgrade pip
pip install vllm>=0.11.1 || echo "vLLM installation skipped (may already exist)"
pip install pypdfium2>=4.0.0 pillow>=10.0.0 aiohttp

# Phase 4: Verify Required Files
echo ""
echo "=== Phase 4: Verification ==="
if [ ! -f "web_demo/server/pdf_server.py" ]; then
    echo "ERROR: pdf_server.py not found!"
    echo "This usually means the wrong repository was cloned."
    echo "Expected: $REPO_URL"
    exit 1
fi
echo "pdf_server.py found"

# Phase 5: Pre-download Model (optional, speeds up first inference)
echo ""
echo "=== Phase 5: Model Download ==="
pip install huggingface-hub
python3 -c "
from huggingface_hub import snapshot_download
try:
    snapshot_download('lightonai/LightOnOCR-2-1B')
    print('Model downloaded successfully')
except Exception as e:
    print(f'Model download skipped: {e}')
"

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
