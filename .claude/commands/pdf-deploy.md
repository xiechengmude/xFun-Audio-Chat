**Purpose**: Automated PDF-AI Service deployment on RunPod GPU Cloud

---

## Command Execution
Execute: immediate. --plan→show plan first
Purpose: "Deploy PDF-AI parsing service to RunPod with $ARGUMENTS"

Fully automated deployment of PDF-AI parsing service using vLLM + LightOnOCR-2-1B, including:
- GPU Pod provisioning (H100/A100/A40)
- vLLM server setup with vision model
- PDF API server startup
- Post-deployment verification and benchmark

## Usage Examples

```bash
# Full deployment with H100 GPU (recommended for best performance)
/pdf-deploy --gpu H100

# Deploy with A100 GPU
/pdf-deploy --gpu A100

# Deploy with A40 GPU (budget option)
/pdf-deploy --gpu A40

# Deploy and run benchmark test
/pdf-deploy --gpu H100 --benchmark

# Plan mode - show deployment plan without executing
/pdf-deploy --plan --gpu H100

# Quick status check of existing pods
/pdf-deploy --status

# Stop/Start existing pod
/pdf-deploy --stop --pod-id <id>
/pdf-deploy --start --pod-id <id>
```

## Command Flags

**--gpu:** GPU type selection
- H100: NVIDIA H100 80GB (best performance, ~5.71 pages/s)
- A100: NVIDIA A100 80GB (~4.0 pages/s)
- A40: NVIDIA A40 48GB (~3.0 pages/s)

**--vllm-port:** vLLM internal port (default: 8000)
**--api-port:** PDF API external port (default: 8006)
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
2. Query available GPU inventory (prefer H100 for best throughput)
3. Create Pod with vLLM-optimized image
4. Wait for Pod to reach RUNNING state
5. Retrieve SSH connection details and port mappings

### Phase 2: Environment Setup
1. SSH connect to Pod
2. Install system dependencies
3. Install Python dependencies:
   - vllm>=0.11.1
   - pypdfium2>=4.0.0
   - pillow>=10.0.0
   - aiohttp
4. Clone Fun-Audio-Chat repository (for pdf_server.py)

### Phase 3: vLLM Server Startup
1. Download LightOnOCR-2-1B model
2. Start vLLM server with vision model configuration:
   ```bash
   vllm serve lightonai/LightOnOCR-2-1B \
       --port 8000 \
       --limit-mm-per-prompt '{"image": 1}' \
       --mm-processor-cache-gb 0 \
       --no-enable-prefix-caching
   ```
3. Wait for model loading (~2-3 minutes)
4. Verify vLLM health endpoint

### Phase 4: PDF API Server Startup
1. Set PYTHONPATH
2. Start PDF API server:
   ```bash
   python3 -m web_demo.server.pdf_server \
       --port 8006 \
       --vllm-endpoint http://localhost:8000
   ```
3. Wait for server initialization
4. Verify API endpoints

### Phase 5: Verification
1. Health check (/health endpoint)
2. Basic PDF parse test
3. Optional benchmark (--benchmark flag):
   - Parse test PDF
   - Measure throughput
   - Report pages/second

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
=== PDF-AI Deployment Complete ===
Pod ID: <pod_id>
Pod Name: pdf-ai-<gpu>
GPU: <gpu_type>
Public IP: <ip_address>
SSH: ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519

Services:
  vLLM:    http://<ip>:8000 (internal)
  PDF API: http://<ip>:8006

API Endpoints:
  POST http://<ip>:8006/api/parse         - Parse single PDF
  POST http://<ip>:8006/api/parse/batch   - Parse multiple PDFs
  POST http://<ip>:8006/api/parse/stream  - SSE streaming
  GET  http://<ip>:8006/health            - Health check
  GET  http://<ip>:8006/api/info          - Service info

Benchmark (if --benchmark):
  Throughput: 5.71 pages/s
  Daily capacity: ~493K pages
```

## Automation Scripts

This skill uses the following scripts:
- `scripts/runpod_manager.py` - RunPod API management
- `scripts/pdf_deploy.py` - PDF service deployment automation
- `web_demo/server/pdf_server.py` - PDF parsing API server

## Resource Requirements

| Component | VRAM | Notes |
|-----------|------|-------|
| LightOnOCR-2-1B | ~6GB | Vision-language OCR model |
| vLLM overhead | ~2GB | KV cache and buffers |
| **Total** | **~8GB** | A40 sufficient, H100 for speed |

## Performance Metrics

| GPU | Throughput | Daily Capacity | Cost/1K Pages |
|-----|------------|----------------|---------------|
| H100 | 5.71 pages/s | ~493K | <$0.01 |
| A100 | ~4.0 pages/s | ~345K | <$0.02 |
| A40 | ~3.0 pages/s | ~260K | <$0.015 |

## Error Handling

The skill handles common errors:
- GPU inventory unavailable → Retry with alternative GPU
- SSH connection timeout → Retry with exponential backoff
- vLLM startup failure → Check logs, increase timeout
- PDF server startup failure → Verify vLLM is healthy first
- Port mapping unavailable → Query and update mappings

## Rollback

If deployment fails:
1. vLLM logs: `nohup.out` or systemd journal
2. PDF server logs: `pdf_server.log`
3. Pod not terminated (preserves debugging capability)
4. Use `--stop` to pause billing while debugging
5. Use `python3 scripts/runpod_manager.py --action terminate --pod-id <id>` to cleanup

## Port Configuration

| Service | Port | Protocol | Notes |
|---------|------|----------|-------|
| vLLM | 8000 | HTTP | Internal only |
| PDF API | 8006 | HTTP | External endpoint |
| SSH | 22 | TCP | Management |

## Test Commands

```bash
# Health check
curl http://<ip>:8006/health

# Parse single PDF
curl -X POST http://<ip>:8006/api/parse \
    -F "file=@document.pdf" \
    -F "pages=1-5" | jq

# Batch parse
curl -X POST http://<ip>:8006/api/parse/batch \
    -F "file1=@doc1.pdf" \
    -F "file2=@doc2.pdf" | jq

# Stream parse
curl -X POST http://<ip>:8006/api/parse/stream \
    -F "file=@document.pdf" \
    -H "Accept: text/event-stream"
```
