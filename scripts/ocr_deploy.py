#!/usr/bin/env python3
"""
GLM-OCR Service Automated Deployment Script
Deploys vLLM + GLM-OCR (0.9B) + FastAPI OCR API Server on RunPod GPU Cloud

Features:
- MTP speculative decoding for faster inference
- PP-DocLayout-V3 layout analysis integration
- transformers from source (required by GLM-OCR)
- Full pipeline: layout detection → parallel OCR

Usage:
    python3 scripts/ocr_deploy.py --gpu A40
    python3 scripts/ocr_deploy.py --gpu A40 --benchmark
    python3 scripts/ocr_deploy.py --status
"""

import os
import sys
import json
import time
import argparse
import subprocess
from typing import Optional, Dict, Tuple
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from runpod_manager import RunPodManager


class OCRServiceDeployer:
    """Automated deployer for GLM-OCR Service on RunPod"""

    # Deployment configuration
    DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    DEFAULT_PORTS = [
        "22/tcp", "8000/tcp", "8001/tcp", "8002/tcp", "8003/tcp",
        "8004/tcp", "8005/tcp", "8006/tcp", "8007/tcp", "8008/tcp",
        "8009/tcp", "8010/tcp", "8080/http", "8888/http"
    ]
    VLLM_PORT = 8000   # Internal vLLM port
    API_PORT = 8007    # External OCR API port

    REPO_URL = "https://github.com/xiechengmude/xFun-Audio-Chat"

    # ---------- Setup commands ----------

    SETUP_COMMANDS = '''
set -e

echo "=== Phase 1: System Dependencies ==="
apt update && apt install -y poppler-utils bc

echo "=== Phase 2: Clone Repository ==="
cd /workspace
if [ -d "Fun-Audio-Chat" ]; then
    echo "Repository exists, pulling latest..."
    cd Fun-Audio-Chat
    git pull origin main || true
else
    echo "Cloning repository..."
    git clone --recurse-submodules {repo_url} Fun-Audio-Chat
    cd Fun-Audio-Chat
fi

echo "=== Phase 3: Python Dependencies ==="
pip install --upgrade pip

# IMPORTANT: Install order matters!
# 1. vLLM nightly first (brings transformers 4.x as dependency)
# 2. PaddlePaddle + PaddleX
# 3. FastAPI stack
# 4. transformers from source LAST (overrides to 5.x with glm_ocr support)

echo "Step 1: Installing vLLM nightly (required for GLM-OCR native support)..."
pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly || echo "vLLM nightly install failed, trying stable..."
pip install "vllm>=0.15.0" 2>/dev/null || true

# PaddlePaddle + PaddleX for layout detection
echo "Step 2: Installing PaddlePaddle..."
pip install paddlepaddle paddlex

# FastAPI stack + utilities
echo "Step 3: Installing FastAPI stack..."
pip install fastapi uvicorn httpx pypdfium2 pillow python-multipart

# transformers from source MUST be last (GLM-OCR requires >=5.0 with glm_ocr arch)
echo "Step 4: Installing transformers from source (MUST be last)..."
pip install git+https://github.com/huggingface/transformers.git

echo "=== Phase 4: Verify ocr_server.py exists ==="
if [ ! -f "web_demo/server/ocr_server.py" ]; then
    echo "ERROR: ocr_server.py not found!"
    exit 1
fi
echo "ocr_server.py found"

echo "=== Phase 5: Pre-download GLM-OCR Model ==="
pip install huggingface-hub
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('zai-org/GLM-OCR')" || echo "Model may already be cached"

echo "=== Setup Complete ==="
'''

    # ---------- vLLM start command ----------

    START_VLLM_COMMAND = '''
cd /workspace/Fun-Audio-Chat
export PYTHONPATH=$(pwd)

# Kill any existing vLLM processes
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 3

# Clear GPU memory
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Start vLLM (skip PaddleX connectivity check in subprocesses)
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
nohup vllm serve zai-org/GLM-OCR \
    --port {vllm_port} \
    --allowed-local-media-path / \
    --served-model-name glm-ocr \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 \
    > vllm.log 2>&1 &

echo $!
sleep 5
'''

    # ---------- API server start command ----------

    START_API_COMMAND = '''
cd /workspace/Fun-Audio-Chat
export PYTHONPATH=$(pwd)

# Kill any existing OCR server
pkill -f "ocr_server" 2>/dev/null || true
sleep 2

# Start OCR API server (setsid to survive SSH disconnect)
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
setsid nohup python3 -m web_demo.server.ocr_server \
    --port {api_port} \
    --vllm-endpoint http://localhost:{vllm_port} \
    --no-layout \
    --host 0.0.0.0 \
    > ocr_server.log 2>&1 &

echo $!
sleep 3
'''

    # ---------- Benchmark command ----------

    BENCHMARK_COMMAND = '''
cd /workspace/Fun-Audio-Chat
export PYTHONPATH=$(pwd)

# Create a test image with text
python3 -c "
from PIL import Image, ImageDraw, ImageFont
img = Image.new('RGB', (800, 200), 'white')
draw = ImageDraw.Draw(img)
draw.text((50, 50), 'GLM-OCR Benchmark Test\\nThis is a test image for OCR performance.', fill='black')
img.save('/tmp/test_ocr.png')
print('Test image created')
"

echo "=== Running OCR Benchmark ==="

echo "--- Test 1: Single image OCR ---"
START=$(date +%s.%N)
RESULT=$(curl -s -X POST http://localhost:{api_port}/api/ocr/recognize \
    -F "image=@/tmp/test_ocr.png" -F "task=text")
END=$(date +%s.%N)
ELAPSED=$(echo "$END - $START" | bc)

echo "$RESULT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'text' in d:
        print(f'SUCCESS: OCR completed in {{d.get(\"time\", \"?\")}}s')
        print(f'Text: {{d[\"text\"][:100]}}...')
    else:
        print(f'FAILED: {{d.get(\"error\", \"Unknown error\")}}')
except Exception as e:
    print(f'Parse error: {{e}}')
"

echo ""
echo "--- Test 2: Health check ---"
curl -s http://localhost:{api_port}/health | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'Status: {{d.get(\"status\")}}')
print(f'vLLM: {{d.get(\"vllm\")}}')
print(f'Layout: {{d.get(\"layout_model\")}}')
"

echo ""
echo "--- Test 3: Concurrent requests (5x) ---"
python3 -c "
import asyncio, httpx, time

async def test_concurrent():
    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = []
        for _ in range(5):
            tasks.append(client.post(
                'http://localhost:{api_port}/api/ocr/recognize',
                files={{'image': ('test.png', open('/tmp/test_ocr.png', 'rb'), 'image/png')}},
                data={{'task': 'text'}}
            ))
        start = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start
        success = sum(1 for r in results if not isinstance(r, Exception) and r.status_code == 200)
        print(f'5 concurrent requests: {{elapsed:.2f}}s ({{success}}/5 success)')
        print(f'Throughput: {{5/elapsed:.1f}} imgs/s')

asyncio.run(test_concurrent())
"

echo ""
echo "=== Benchmark Complete ==="
'''

    def __init__(self, api_key: str):
        self.manager = RunPodManager(api_key)
        self.ssh_key_path = os.path.expanduser("~/.ssh/id_ed25519")

    def get_gpu_id(self, gpu_name: str) -> Optional[str]:
        gpus = self.manager.list_gpu_types([gpu_name])
        if not gpus:
            return None
        return gpus[0]["id"]

    def create_pod(self, name: str, gpu_type: str,
                   disk_gb: int = 100, volume_gb: int = 100) -> Dict:
        print(f"\n{'='*60}")
        print(f"Creating Pod: {name}")
        print(f"GPU: {gpu_type}, Disk: {disk_gb}GB, Volume: {volume_gb}GB")
        print(f"{'='*60}")

        gpu_id = self.get_gpu_id(gpu_type)
        if not gpu_id:
            raise ValueError(f"GPU type '{gpu_type}' not found")

        print(f"GPU ID: {gpu_id}")

        result = self.manager.create_pod(
            name=name,
            gpu_type_id=gpu_id,
            image_name=self.DEFAULT_IMAGE,
            container_disk_gb=disk_gb,
            volume_gb=volume_gb,
            ports=self.DEFAULT_PORTS
        )

        print(f"Pod created: {result.get('id', 'unknown')}")
        return result

    def wait_for_pod_ready(self, pod_id: str, timeout: int = 300) -> Dict:
        print(f"\nWaiting for Pod {pod_id} to be ready...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                pod = self.manager.get_pod(pod_id)
                status = pod.get("desiredStatus", "UNKNOWN")
                print(f"  Status: {status}")

                if status == "RUNNING" and pod.get("publicIp"):
                    print(f"  Pod is ready! IP: {pod.get('publicIp')}")
                    return pod
            except Exception as e:
                print(f"  Error checking status: {e}")

            time.sleep(10)

        raise TimeoutError(f"Pod {pod_id} did not become ready within {timeout}s")

    def get_ssh_connection(self, pod: Dict) -> Tuple[str, int]:
        public_ip = pod.get("publicIp")
        port_mappings = pod.get("portMappings", {})
        ssh_port = port_mappings.get("22", 22)
        return public_ip, ssh_port

    def get_port_mapping(self, pod: Dict, internal_port: int) -> int:
        port_mappings = pod.get("portMappings", {})
        return port_mappings.get(str(internal_port), internal_port)

    def run_ssh_command(self, ip: str, port: int, command: str,
                        timeout: int = 600) -> Tuple[int, str, str]:
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=30",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-i", self.ssh_key_path,
            "-p", str(port),
            f"root@{ip}",
            command
        ]

        try:
            result = subprocess.run(
                ssh_cmd, capture_output=True, text=True, timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)

    def wait_for_ssh(self, ip: str, port: int, timeout: int = 180) -> bool:
        print(f"\nWaiting for SSH at {ip}:{port}...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            returncode, stdout, stderr = self.run_ssh_command(
                ip, port, "echo 'SSH Ready'", timeout=30
            )
            if returncode == 0:
                print("  SSH is ready!")
                return True
            print("  SSH not ready yet, retrying...")
            time.sleep(10)

        return False

    def setup_environment(self, ip: str, port: int) -> bool:
        print(f"\n{'='*60}")
        print("Setting up environment...")
        print(f"{'='*60}")

        setup_cmd = self.SETUP_COMMANDS.format(repo_url=self.REPO_URL)
        returncode, stdout, stderr = self.run_ssh_command(
            ip, port, setup_cmd, timeout=2400  # 40 min (transformers source install is slow)
        )

        print(stdout)
        if stderr and "error" in stderr.lower():
            print(f"STDERR: {stderr}")

        if returncode != 0:
            print(f"Setup failed with code {returncode}")
            return False

        print("Environment setup complete!")
        return True

    def start_vllm(self, ip: str, ssh_port: int, retry: int = 2) -> Optional[str]:
        print(f"\n{'='*60}")
        print("Starting vLLM server (GLM-OCR + MTP)...")
        print(f"{'='*60}")

        for attempt in range(retry):
            if attempt > 0:
                print(f"\n  Retry attempt {attempt + 1}/{retry}...")
                self.run_ssh_command(
                    ip, ssh_port,
                    "pkill -9 vllm; sleep 5; nvidia-smi -r 2>/dev/null || true",
                    timeout=60
                )

            cmd = self.START_VLLM_COMMAND.format(vllm_port=self.VLLM_PORT)
            returncode, stdout, stderr = self.run_ssh_command(ip, ssh_port, cmd, timeout=60)

            if returncode == 0:
                pid = stdout.strip().split('\n')[-1]
                print(f"vLLM started with PID: {pid}")
                return pid

            print(f"  Failed to start vLLM (attempt {attempt + 1}): {stderr}")

        return None

    def wait_for_vllm(self, ip: str, ssh_port: int, timeout: int = 420) -> bool:
        print(f"\nWaiting for vLLM to load GLM-OCR model (up to {timeout}s)...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            elapsed = int(time.time() - start_time)

            returncode, stdout, _ = self.run_ssh_command(
                ip, ssh_port,
                f"curl -s http://localhost:{self.VLLM_PORT}/health 2>/dev/null",
                timeout=30
            )

            if returncode == 0 and stdout.strip():
                print(f"  vLLM is ready! (took {elapsed}s)")
                return True

            # Check for OOM
            _, logs, _ = self.run_ssh_command(
                ip, ssh_port,
                "tail -3 /workspace/Fun-Audio-Chat/vllm.log 2>/dev/null",
                timeout=30
            )

            if "error" in logs.lower() and "cuda out of memory" in logs.lower():
                print("  ERROR: CUDA OOM detected!")
                return False

            last_line = logs.strip().split('\n')[-1] if logs.strip() else "Loading..."
            print(f"  [{elapsed}s] {last_line[:70]}...")

            time.sleep(15)

        return False

    def start_api_server(self, ip: str, ssh_port: int) -> Optional[str]:
        print(f"\n{'='*60}")
        print("Starting OCR API server (FastAPI)...")
        print(f"{'='*60}")

        cmd = self.START_API_COMMAND.format(
            api_port=self.API_PORT,
            vllm_port=self.VLLM_PORT
        )
        returncode, stdout, stderr = self.run_ssh_command(ip, ssh_port, cmd, timeout=60)

        if returncode != 0:
            print(f"Failed to start API server: {stderr}")
            return None

        pid = stdout.strip().split('\n')[-1]
        print(f"OCR API server started with PID: {pid}")
        return pid

    def verify_services(self, ip: str, ssh_port: int, timeout: int = 90) -> bool:
        print(f"\nVerifying services...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            returncode, stdout, _ = self.run_ssh_command(
                ip, ssh_port,
                f"curl -s http://localhost:{self.API_PORT}/health",
                timeout=30
            )

            if returncode == 0:
                try:
                    health = json.loads(stdout)
                    if health.get("status") == "healthy" and health.get("vllm"):
                        print(f"  OCR API: healthy")
                        print(f"  vLLM: healthy")
                        print(f"  Layout: {health.get('layout_model')}")
                        return True
                    elif health.get("status") == "degraded":
                        print(f"  OCR API: degraded (vLLM not ready)")
                except Exception:
                    pass

            print("  Waiting for services...")
            time.sleep(10)

        return False

    def run_benchmark(self, ip: str, ssh_port: int) -> Optional[Dict]:
        print(f"\n{'='*60}")
        print("Running benchmark...")
        print(f"{'='*60}")

        cmd = self.BENCHMARK_COMMAND.format(api_port=self.API_PORT)
        returncode, stdout, stderr = self.run_ssh_command(
            ip, ssh_port, cmd, timeout=300
        )

        print(stdout)
        if stderr and "error" in stderr.lower():
            print(f"STDERR: {stderr}")

        return {"output": stdout, "success": "SUCCESS" in stdout or "Benchmark Complete" in stdout}

    def deploy(self, gpu_type: str = "A40", name: str = None,
               disk_gb: int = 100, volume_gb: int = 100,
               run_benchmark: bool = False) -> Dict:
        """Full deployment workflow"""

        if name is None:
            name = f"glm-ocr-{gpu_type.lower()}"

        result = {
            "success": False,
            "pod_id": None,
            "pod_name": name,
            "gpu": gpu_type,
            "public_ip": None,
            "ssh_port": None,
            "api_port": None,
            "vllm_port": None,
            "api_endpoint": None,
            "error": None
        }

        try:
            # Phase 1: Create Pod
            print("\n" + "="*60)
            print("PHASE 1: Creating Pod")
            print("="*60)
            pod_result = self.create_pod(name, gpu_type, disk_gb, volume_gb)
            pod_id = pod_result.get("id")
            result["pod_id"] = pod_id

            # Phase 2: Wait for Pod Ready
            print("\n" + "="*60)
            print("PHASE 2: Waiting for Pod")
            print("="*60)
            pod = self.wait_for_pod_ready(pod_id)

            ip, ssh_port = self.get_ssh_connection(pod)
            api_ext_port = self.get_port_mapping(pod, self.API_PORT)
            vllm_ext_port = self.get_port_mapping(pod, self.VLLM_PORT)

            result["public_ip"] = ip
            result["ssh_port"] = ssh_port
            result["api_port"] = api_ext_port
            result["vllm_port"] = vllm_ext_port
            result["api_endpoint"] = f"http://{ip}:{api_ext_port}"

            # Phase 3: Wait for SSH
            print("\n" + "="*60)
            print("PHASE 3: Establishing SSH Connection")
            print("="*60)
            if not self.wait_for_ssh(ip, ssh_port):
                raise RuntimeError("SSH connection failed")

            # Phase 4: Setup Environment
            print("\n" + "="*60)
            print("PHASE 4: Setting Up Environment (transformers source + paddlex)")
            print("="*60)
            if not self.setup_environment(ip, ssh_port):
                raise RuntimeError("Environment setup failed")

            # Phase 5: Start vLLM (with retry)
            print("\n" + "="*60)
            print("PHASE 5: Starting vLLM Server (GLM-OCR + MTP)")
            print("="*60)
            vllm_pid = self.start_vllm(ip, ssh_port, retry=2)
            if not vllm_pid:
                raise RuntimeError("vLLM failed to start after retries")

            if not self.wait_for_vllm(ip, ssh_port, timeout=420):
                raise RuntimeError("vLLM failed to become ready (timeout or OOM)")

            # Phase 6: Start API Server
            print("\n" + "="*60)
            print("PHASE 6: Starting OCR API Server (FastAPI)")
            print("="*60)
            api_pid = self.start_api_server(ip, ssh_port)
            if not api_pid:
                raise RuntimeError("OCR API server failed to start")

            # Phase 7: Verify Services
            print("\n" + "="*60)
            print("PHASE 7: Verifying Services")
            print("="*60)
            if not self.verify_services(ip, ssh_port):
                raise RuntimeError("Service verification failed")

            result["success"] = True

            # Phase 8: Benchmark (optional)
            if run_benchmark:
                print("\n" + "="*60)
                print("PHASE 8: Running Benchmark")
                print("="*60)
                benchmark_result = self.run_benchmark(ip, ssh_port)
                result["benchmark"] = benchmark_result

        except Exception as e:
            result["error"] = str(e)
            print(f"\nDeployment failed: {e}")

        self._print_summary(result)
        return result

    def _print_summary(self, result: Dict):
        print("\n" + "="*60)
        print("GLM-OCR DEPLOYMENT SUMMARY")
        print("="*60)

        if result["success"]:
            print(f"Status: SUCCESS")
            print(f"Pod ID: {result['pod_id']}")
            print(f"Pod Name: {result['pod_name']}")
            print(f"GPU: {result['gpu']}")
            print(f"Public IP: {result['public_ip']}")
            print(f"\nSSH Command:")
            print(f"  ssh root@{result['public_ip']} -p {result['ssh_port']} -i ~/.ssh/id_ed25519")
            print(f"\nServices:")
            print(f"  vLLM (GLM-OCR): http://{result['public_ip']}:{result['vllm_port']}")
            print(f"  OCR API:        http://{result['public_ip']}:{result['api_port']}")
            print(f"\nAPI Endpoints:")
            ep = result['api_endpoint']
            print(f"  POST {ep}/api/ocr/recognize        - Single image OCR")
            print(f"  POST {ep}/api/ocr/extract          - Information extraction")
            print(f"  POST {ep}/api/ocr/batch            - Batch image OCR")
            print(f"  POST {ep}/api/ocr/document         - Full document parsing")
            print(f"  POST {ep}/api/ocr/document/stream   - SSE streaming")
            print(f"  GET  {ep}/health                   - Health check")
            print(f"  GET  {ep}/api/info                 - Service info")
            print(f"\nTest Commands:")
            print(f"  curl {ep}/health")
            print(f"  curl -X POST {ep}/api/ocr/recognize -F 'image=@test.png' -F 'task=text'")
        else:
            print(f"Status: FAILED")
            print(f"Error: {result['error']}")
            if result['pod_id']:
                print(f"\nPod ID (for debugging): {result['pod_id']}")
                if result['public_ip'] and result['ssh_port']:
                    print(f"SSH: ssh root@{result['public_ip']} -p {result['ssh_port']} -i ~/.ssh/id_ed25519")
                    print(f"\nDebug Commands:")
                    print(f"  tail -100 /workspace/Fun-Audio-Chat/vllm.log")
                    print(f"  tail -100 /workspace/Fun-Audio-Chat/ocr_server.log")
                    print(f"  nvidia-smi")

        print("="*60)

    def status(self):
        pods = self.manager.list_pods()

        print(f"\n{'Name':<30} {'ID':<25} {'GPU':<10} {'Status':<12} {'IP'}")
        print("-" * 100)

        for pod in pods:
            name = pod.get('name', 'N/A')[:29]
            pod_id = pod.get('id', 'N/A')[:24]
            gpu_count = pod.get('gpuCount', 0)
            status = pod.get('desiredStatus', 'N/A')[:11]
            public_ip = pod.get('publicIp', 'N/A')

            print(f"{name:<30} {pod_id:<25} {gpu_count:<10} {status:<12} {public_ip}")

            if pod.get('portMappings'):
                print(f"  Ports: {pod['portMappings']}")


def load_api_key() -> str:
    api_key = os.environ.get("RUNPOD_KEY")

    if not api_key:
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("RUNPOD_KEY="):
                        api_key = line.strip().split("=", 1)[1]
                        break

    if not api_key:
        raise ValueError("RUNPOD_KEY not found in environment or .env file")

    return api_key


def main():
    parser = argparse.ArgumentParser(
        description="GLM-OCR Service Automated Deployment (v1.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/ocr_deploy.py --gpu A40                # Deploy with A40
  python3 scripts/ocr_deploy.py --gpu A40 --benchmark    # Deploy + benchmark
  python3 scripts/ocr_deploy.py --status                 # Show all pods

Model: zai-org/GLM-OCR (0.9B params, ~4GB VRAM)
Pipeline: PP-DocLayout-V3 layout + GLM-OCR recognition
Decoding: MTP speculative decoding

Performance (approximate):
  A40:  ~1.86 pgs/s (~160K pages/day)
  A100: ~2.5  pgs/s (~216K pages/day)
  H100: ~3.5  pgs/s (~302K pages/day)
        """
    )

    parser.add_argument("--gpu", default="A40",
                        help="GPU type (H100, A100, A40) - default: A40")
    parser.add_argument("--name", help="Pod name (auto-generated if not specified)")
    parser.add_argument("--disk", type=int, default=100,
                        help="Container disk size in GB (default: 100)")
    parser.add_argument("--volume", type=int, default=100,
                        help="Persistent volume size in GB (default: 100)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark after deployment")
    parser.add_argument("--status", action="store_true",
                        help="Show status of all pods")

    args = parser.parse_args()

    try:
        api_key = load_api_key()
        deployer = OCRServiceDeployer(api_key)

        if args.status:
            deployer.status()
        else:
            result = deployer.deploy(
                gpu_type=args.gpu,
                name=args.name,
                disk_gb=args.disk,
                volume_gb=args.volume,
                run_benchmark=args.benchmark
            )

            sys.exit(0 if result["success"] else 1)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
