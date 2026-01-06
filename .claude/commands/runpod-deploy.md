**Purpose**: Automated Fun-Audio-Chat deployment on RunPod GPU Cloud

---

## Command Execution
Execute: immediate. --plan→show plan first
Purpose: "Deploy Fun-Audio-Chat to RunPod with $ARGUMENTS"

Fully automated deployment of Fun-Audio-Chat to RunPod GPU Cloud, including:
- GPU Pod provisioning (A40/4090)
- Environment setup and dependency installation
- Model download and server startup
- Post-deployment verification tests

## Usage Examples

```bash
# Full deployment with A40 GPU (recommended)
/runpod-deploy --gpu A40

# Deploy with 4090 GPU
/runpod-deploy --gpu 4090

# Deploy with custom disk size
/runpod-deploy --gpu A40 --disk 150 --volume 150

# Plan mode - show deployment plan without executing
/runpod-deploy --plan --gpu A40

# Deploy and run comprehensive tests
/runpod-deploy --gpu A40 --test-full

# Quick status check of existing pods
/runpod-deploy --status

# Stop/Start existing pod
/runpod-deploy --stop --pod-id <id>
/runpod-deploy --start --pod-id <id>
```

## Command Flags

**--gpu:** GPU type selection
- A40: NVIDIA A40 48GB (recommended, $0.40/hr)
- 4090: RTX 4090 24GB ($0.50-0.70/hr)
- A100: NVIDIA A100 80GB ($1.89/hr)

**--disk:** Container disk size in GB (default: 100)
**--volume:** Persistent volume size in GB (default: 100)
**--test:** Run basic connectivity test after deployment
**--test-full:** Run comprehensive S2S functionality tests
**--status:** Show status of all pods
**--stop:** Stop a running pod
**--start:** Start a stopped pod
**--pod-id:** Specify pod ID for stop/start operations
**--plan:** Show deployment plan without executing

## Deployment Workflow

### Phase 1: Infrastructure Provisioning
1. Check RunPod API credentials (.env file)
2. Query available GPU inventory
3. Create Pod with specified GPU type
4. Wait for Pod to reach RUNNING state
5. Retrieve SSH connection details and port mappings

### Phase 2: Environment Setup
1. SSH connect to Pod
2. Clone Fun-Audio-Chat repository
3. Install system dependencies (ffmpeg)
4. Install Python dependencies:
   - PyTorch 2.8.0 + CUDA 12.8
   - sphn, aiohttp for web demo
   - Project requirements.txt
5. Fix known compatibility issues:
   - ruamel.yaml <0.18 for hyperpyyaml
   - torchvision version matching

### Phase 3: Model Deployment
1. Check if models exist in /workspace
2. Download models if needed:
   - Fun-Audio-Chat-8B (~16GB)
   - Fun-CosyVoice3-0.5B-2512 (~1GB)
3. Verify model integrity

### Phase 4: Server Startup
1. Set PYTHONPATH
2. Start server with appropriate GPU allocation:
   - Single GPU: --tts-gpu 0
   - Dual GPU: --tts-gpu 1
3. Wait for server initialization
4. Verify port listening

### Phase 5: Verification
1. Basic connectivity test (HTTP health check)
2. WebSocket connection test
3. Full S2S test (optional, with --test-full):
   - Send test audio
   - Verify audio response
   - Check response latency

## Prerequisites

### API Credentials
```bash
# Create .env file in project root
RUNPOD_KEY=your_api_key_here
```

### SSH Key
```bash
# Ensure SSH key exists
~/.ssh/id_ed25519
```

## Output Information

After successful deployment, the skill outputs:

```
=== Fun-Audio-Chat Deployment Complete ===
Pod ID: <pod_id>
Pod Name: <pod_name>
GPU: <gpu_type>
Public IP: <ip_address>
SSH: ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519
API Endpoint: ws://<ip>:<port>/api/chat
Server Status: RUNNING
Memory Usage: ~22GB / 48GB
```

## Automation Scripts

This skill uses the following scripts:
- `scripts/runpod_manager.py` - RunPod API management
- `scripts/auto_deploy.py` - Full deployment automation
- `scripts/test_deployment.py` - Deployment verification tests

## Resource Requirements

| Component | VRAM | Notes |
|-----------|------|-------|
| S2S Model (8B) | ~18GB | Main audio-language model |
| TTS (CosyVoice3) | ~4GB | Text-to-speech engine |
| **Total** | **~22GB** | A40 recommended for headroom |

## Error Handling

The skill handles common errors:
- GPU inventory unavailable → Retry with alternative GPU
- SSH connection timeout → Retry with exponential backoff
- Model download failure → Resume from checkpoint
- Server startup failure → Check logs, report diagnostics
- Port mapping unavailable → Query and update mappings

## Rollback

If deployment fails:
1. Server logs saved to `server.log`
2. Pod not terminated (preserves debugging capability)
3. Use `--stop` to pause billing while debugging
4. Use `python3 scripts/runpod_manager.py --action terminate --pod-id <id>` to cleanup

## Template Reference

Pre-configured Template ID: `f4ertqge9p`
- Image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
- Ports: 22/tcp, 8000-8010/tcp, 8080/http, 8888/http
- Volume: /workspace (persistent)
