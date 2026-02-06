**Purpose**: Automated GLM-OCR Service deployment on RunPod GPU Cloud

---

## Command Execution
Execute: immediate. --plan→show plan first
Purpose: "Deploy GLM-OCR service to RunPod with $ARGUMENTS"

Fully automated deployment of GLM-OCR service using vLLM + GLM-OCR (0.9B) + PP-DocLayout-V3, including:
- GPU Pod provisioning (H100/A100/A40)
- vLLM server with MTP speculative decoding
- FastAPI OCR API server startup
- PP-DocLayout-V3 layout detection
- Post-deployment verification and benchmark

## Usage Examples

```bash
# Full deployment with A40 GPU (recommended, ~4GB VRAM sufficient)
/ocr-deploy --gpu A40

# Deploy with H100 for max throughput
/ocr-deploy --gpu H100

# Deploy and run benchmark test
/ocr-deploy --gpu A40 --benchmark

# Plan mode - show deployment plan without executing
/ocr-deploy --plan --gpu A40

# Quick status check of existing pods
/ocr-deploy --status

# Stop/Start existing pod
/ocr-deploy --stop --pod-id <id>
/ocr-deploy --start --pod-id <id>
```

## Command Flags

**--gpu:** GPU type selection
- H100: NVIDIA H100 80GB (best performance, ~3.5 pgs/s)
- A100: NVIDIA A100 80GB (~2.5 pgs/s)
- A40: NVIDIA A40 48GB (~1.86 pgs/s, recommended cost/perf)

**--vllm-port:** vLLM internal port (default: 8000)
**--api-port:** OCR API external port (default: 8007)
**--disk:** Container disk size in GB (default: 100)
**--volume:** Persistent volume size in GB (default: 100)
**--benchmark:** Run benchmark test after deployment
**--status:** Show status of all pods
**--stop:** Stop a running pod
**--start:** Start a stopped pod
**--pod-id:** Specify pod ID for stop/start operations
**--plan:** Show deployment plan without executing

## Deployment Workflow

### Phase 1: Infrastructure Provisioning
1. Check RunPod API credentials (.env file)
2. Query available GPU inventory
3. Create Pod with vLLM-optimized image
4. Wait for Pod to reach RUNNING state
5. Retrieve SSH connection details and port mappings

### Phase 2: Environment Setup
1. SSH connect to Pod
2. Install system dependencies
3. Install Python dependencies:
   - transformers (from source, required by GLM-OCR)
   - vllm>=0.11.1
   - paddlepaddle + paddlex (PP-DocLayout-V3)
   - fastapi, uvicorn, httpx, pypdfium2, pillow, python-multipart
4. Clone xFun-Audio-Chat repository
5. Pre-download GLM-OCR model

### Phase 3: vLLM Server Startup
1. Start vLLM with MTP speculative decoding:
   ```bash
   vllm serve zai-org/GLM-OCR \
       --port 8000 \
       --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
       --served-model-name glm-ocr \
       --gpu-memory-utilization 0.85 --max-num-seqs 64
   ```
2. Wait for model loading (~2-5 minutes)
3. Verify vLLM health endpoint

### Phase 4: OCR API Server Startup
1. Start FastAPI OCR server:
   ```bash
   python3 -m web_demo.server.ocr_server \
       --port 8007 --vllm-endpoint http://localhost:8000 \
       --enable-layout --host 0.0.0.0
   ```
2. Wait for server initialization
3. Verify API endpoints

### Phase 5: Verification
1. Health check (/health endpoint)
2. Single image OCR test
3. Optional benchmark (--benchmark flag):
   - Single image OCR latency
   - 5x concurrent OCR throughput

## Prerequisites

### API Credentials
```bash
# Create .env file in project root
RUNPOD_KEY=your_api_key_here
```

### SSH Key
```bash
~/.ssh/id_ed25519
```

## Output Information

After successful deployment:

```
=== GLM-OCR Deployment Complete ===
Pod ID: <pod_id>
GPU: <gpu_type>
Public IP: <ip_address>
SSH: ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519

Services:
  vLLM (GLM-OCR): http://<ip>:8000
  OCR API:        http://<ip>:8007

API Endpoints:
  POST http://<ip>:8007/api/ocr/recognize        - Single image OCR
  POST http://<ip>:8007/api/ocr/extract          - Information extraction
  POST http://<ip>:8007/api/ocr/batch            - Batch image OCR
  POST http://<ip>:8007/api/ocr/document         - Full document parsing
  POST http://<ip>:8007/api/ocr/document/stream   - SSE streaming
  GET  http://<ip>:8007/health                   - Health check
  GET  http://<ip>:8007/api/info                 - Service info
```

## Automation Scripts

- `scripts/runpod_manager.py` - RunPod API management
- `scripts/ocr_deploy.py` - OCR service deployment automation
- `web_demo/server/ocr_server.py` - FastAPI OCR API server

## Resource Requirements

| Component | VRAM | Notes |
|-----------|------|-------|
| GLM-OCR (0.9B) | ~4GB | Vision-language OCR model |
| vLLM + MTP overhead | ~2GB | KV cache, speculative buffers |
| **Total** | **~6GB** | A40 has 48GB, plenty of headroom |

## Performance Metrics

| GPU | Throughput | Daily Capacity | Cost/1K Pages |
|-----|------------|----------------|---------------|
| H100 | ~3.5 pgs/s | ~302K | <$0.02 |
| A100 | ~2.5 pgs/s | ~216K | <$0.02 |
| A40 | ~1.86 pgs/s | ~160K | <$0.015 |

## Error Handling

- GPU inventory unavailable -> Retry with alternative GPU
- SSH connection timeout -> Retry with exponential backoff
- vLLM startup failure -> Check logs, increase timeout
- transformers incompatibility -> Ensure source install
- PaddleX load failure -> Layout disabled, OCR still works
- Port mapping unavailable -> Query and update mappings

## Rollback

If deployment fails:
1. vLLM logs: `/workspace/Fun-Audio-Chat/vllm.log`
2. OCR server logs: `/workspace/Fun-Audio-Chat/ocr_server.log`
3. Pod not terminated (preserves debugging)
4. Use `--stop` to pause billing while debugging
5. Use `python3 scripts/runpod_manager.py --action terminate --pod-id <id>` to cleanup

## Port Configuration

| Service | Port | Protocol | Notes |
|---------|------|----------|-------|
| vLLM | 8000 | HTTP | Internal only |
| OCR API | 8007 | HTTP | External endpoint |
| SSH | 22 | TCP | Management |

## Test Commands

```bash
# Health check
curl http://<ip>:8007/health

# Single image OCR
curl -X POST http://<ip>:8007/api/ocr/recognize \
    -F "image=@test.png" -F "task=text"

# Formula recognition
curl -X POST http://<ip>:8007/api/ocr/recognize \
    -F "image=@formula.png" -F "task=formula"

# Information extraction
curl -X POST http://<ip>:8007/api/ocr/extract \
    -F "image=@id_card.png" \
    -F 'schema={"name":"","id_number":""}'

# Full document parsing
curl -X POST http://<ip>:8007/api/ocr/document \
    -F "file=@document.pdf" -F "pages=1-5" -F "enable_layout=true" | jq

# Batch OCR
curl -X POST http://<ip>:8007/api/ocr/batch \
    -F "images=@img1.png" -F "images=@img2.png" -F "task=text" | jq

# SSE streaming
curl -X POST http://<ip>:8007/api/ocr/document/stream \
    -F "file=@document.pdf" \
    -H "Accept: text/event-stream"
```
