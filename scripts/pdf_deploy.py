#!/usr/bin/env python3
"""
PDF-AI Service Automated Deployment Script
Deploys vLLM + LightOnOCR-2-1B + PDF API Server on RunPod GPU Cloud

Based on real deployment experience (2026-01-21):
- Fixed vLLM OOM issue with memory optimization params
- Uses user's fork repo with pdf_server.py
- Improved error handling and retry logic
- Extended timeouts for model loading

Usage:
    python3 scripts/pdf_deploy.py --gpu H100
    python3 scripts/pdf_deploy.py --gpu A40 --benchmark
    python3 scripts/pdf_deploy.py --status
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


class PDFServiceDeployer:
    """Automated deployer for PDF-AI Service on RunPod"""

    # Deployment configuration
    DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    DEFAULT_PORTS = [
        "22/tcp", "8000/tcp", "8001/tcp", "8002/tcp", "8003/tcp",
        "8004/tcp", "8005/tcp", "8006/tcp", "8007/tcp", "8008/tcp",
        "8009/tcp", "8010/tcp", "8080/http", "8888/http"
    ]
    VLLM_PORT = 8000  # Internal vLLM port
    API_PORT = 8006   # External PDF API port

    # Use user's fork which contains pdf_server.py
    REPO_URL = "https://github.com/xiechengmude/xFun-Audio-Chat"

    # Setup commands - updated based on real deployment
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
# vLLM and dependencies (already installed in RunPod image usually)
pip install --upgrade pip
pip install vllm>=0.11.1 || echo "vLLM already installed"
pip install pypdfium2>=4.0.0 pillow>=10.0.0 aiohttp

echo "=== Phase 4: Verify pdf_server.py exists ==="
if [ ! -f "web_demo/server/pdf_server.py" ]; then
    echo "ERROR: pdf_server.py not found!"
    exit 1
fi
echo "pdf_server.py found"

echo "=== Phase 5: Pre-download Model ==="
pip install huggingface-hub
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('lightonai/LightOnOCR-2-1B')" || echo "Model may already be cached"

echo "=== Setup Complete ==="
'''

    # vLLM start command with memory optimization (learned from OOM issue)
    START_VLLM_COMMAND = '''
cd /workspace/Fun-Audio-Chat
export PYTHONPATH=$(pwd)

# Kill any existing vLLM processes
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 3

# Clear GPU memory
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Start vLLM server with memory optimization
# --gpu-memory-utilization 0.85: Prevent OOM by limiting GPU memory usage
# --max-num-seqs 64: Reduce concurrent sequences to save memory
nohup vllm serve lightonai/LightOnOCR-2-1B \\
    --port {vllm_port} \\
    --limit-mm-per-prompt '{{"image": 1}}' \\
    --mm-processor-cache-gb 0 \\
    --no-enable-prefix-caching \\
    --gpu-memory-utilization 0.85 \\
    --max-num-seqs 64 \\
    > vllm.log 2>&1 &

echo $!
sleep 5
'''

    START_API_COMMAND = '''
cd /workspace/Fun-Audio-Chat
export PYTHONPATH=$(pwd)

# Kill any existing PDF server
pkill -f "pdf_server" 2>/dev/null || true
sleep 2

# Start PDF API server
nohup python3 -m web_demo.server.pdf_server \\
    --port {api_port} \\
    --vllm-endpoint http://localhost:{vllm_port} \\
    --host 0.0.0.0 \\
    > pdf_server.log 2>&1 &

echo $!
sleep 3
'''

    BENCHMARK_COMMAND = '''
cd /workspace/Fun-Audio-Chat
export PYTHONPATH=$(pwd)

# Download a real test PDF from arxiv
curl -s -L -o /tmp/test.pdf "https://arxiv.org/pdf/1807.03090" 2>/dev/null || {{
    # Fallback: create minimal PDF
    echo '%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >> endobj
4 0 obj << /Length 44 >> stream
BT /F1 12 Tf 100 700 Td (PDF-AI Test) Tj ET
endstream endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000206 00000 n
trailer << /Size 5 /Root 1 0 R >>
startxref
300
%%EOF' > /tmp/test.pdf
}}

echo "=== Running Benchmark ==="
echo "Testing single page parse..."

START=$(date +%s.%N)
RESULT=$(curl -s -X POST http://localhost:{api_port}/api/parse -F "file=@/tmp/test.pdf" -F "pages=1")
END=$(date +%s.%N)

echo "$RESULT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if d.get('success'):
        print(f'SUCCESS: Parsed 1 page in {{d[\"total_time\"]}}s')
        print(f'Throughput: {{d[\"throughput\"]}} pages/s')
    else:
        print(f'FAILED: {{d.get(\"error\", \"Unknown error\")}}')
except Exception as e:
    print(f'Parse error: {{e}}')
"
'''

    def __init__(self, api_key: str):
        self.manager = RunPodManager(api_key)
        self.ssh_key_path = os.path.expanduser("~/.ssh/id_ed25519")

    def get_gpu_id(self, gpu_name: str) -> Optional[str]:
        """Get GPU type ID from name"""
        gpus = self.manager.list_gpu_types([gpu_name])
        if not gpus:
            return None
        return gpus[0]["id"]

    def create_pod(self, name: str, gpu_type: str,
                   disk_gb: int = 100, volume_gb: int = 100) -> Dict:
        """Create a new RunPod with specified GPU"""
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
        """Wait for pod to be in RUNNING state"""
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
        """Extract SSH connection details from pod info"""
        public_ip = pod.get("publicIp")
        port_mappings = pod.get("portMappings", {})
        ssh_port = port_mappings.get("22", 22)
        return public_ip, ssh_port

    def get_port_mapping(self, pod: Dict, internal_port: int) -> int:
        """Get external port mapping for internal port"""
        port_mappings = pod.get("portMappings", {})
        return port_mappings.get(str(internal_port), internal_port)

    def run_ssh_command(self, ip: str, port: int, command: str,
                        timeout: int = 600) -> Tuple[int, str, str]:
        """Run command via SSH"""
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
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)

    def wait_for_ssh(self, ip: str, port: int, timeout: int = 180) -> bool:
        """Wait for SSH to become available"""
        print(f"\nWaiting for SSH at {ip}:{port}...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            returncode, stdout, stderr = self.run_ssh_command(
                ip, port, "echo 'SSH Ready'", timeout=30
            )
            if returncode == 0:
                print("  SSH is ready!")
                return True
            print(f"  SSH not ready yet, retrying...")
            time.sleep(10)

        return False

    def setup_environment(self, ip: str, port: int) -> bool:
        """Run environment setup on the pod"""
        print(f"\n{'='*60}")
        print("Setting up environment...")
        print(f"{'='*60}")

        setup_cmd = self.SETUP_COMMANDS.format(repo_url=self.REPO_URL)
        returncode, stdout, stderr = self.run_ssh_command(
            ip, port, setup_cmd, timeout=1800  # 30 min timeout
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
        """Start vLLM server with retry logic"""
        print(f"\n{'='*60}")
        print("Starting vLLM server...")
        print(f"{'='*60}")

        for attempt in range(retry):
            if attempt > 0:
                print(f"\n  Retry attempt {attempt + 1}/{retry}...")
                # Clear GPU memory before retry
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
        """Wait for vLLM to be ready (extended timeout: 7 min)"""
        print(f"\nWaiting for vLLM to load model (up to {timeout}s)...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            elapsed = int(time.time() - start_time)

            # Check if vLLM is listening
            returncode, stdout, _ = self.run_ssh_command(
                ip, ssh_port,
                f"curl -s http://localhost:{self.VLLM_PORT}/health 2>/dev/null",
                timeout=30
            )

            if returncode == 0 and stdout.strip():
                print(f"  vLLM is ready! (took {elapsed}s)")
                return True

            # Check for errors in logs
            _, logs, _ = self.run_ssh_command(
                ip, ssh_port,
                "tail -3 /workspace/Fun-Audio-Chat/vllm.log 2>/dev/null",
                timeout=30
            )

            if "error" in logs.lower() and "cuda out of memory" in logs.lower():
                print("  ERROR: CUDA OOM detected!")
                return False

            # Show progress
            last_line = logs.strip().split('\n')[-1] if logs.strip() else "Loading..."
            print(f"  [{elapsed}s] {last_line[:70]}...")

            time.sleep(15)

        return False

    def start_api_server(self, ip: str, ssh_port: int) -> Optional[str]:
        """Start PDF API server"""
        print(f"\n{'='*60}")
        print("Starting PDF API server...")
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
        print(f"PDF API server started with PID: {pid}")
        return pid

    def verify_services(self, ip: str, ssh_port: int,
                        timeout: int = 90) -> bool:
        """Verify both services are running"""
        print(f"\nVerifying services...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check PDF API health
            returncode, stdout, _ = self.run_ssh_command(
                ip, ssh_port,
                f"curl -s http://localhost:{self.API_PORT}/health",
                timeout=30
            )

            if returncode == 0:
                try:
                    health = json.loads(stdout)
                    if health.get("status") == "healthy" and health.get("vllm_healthy"):
                        print(f"  PDF API: healthy")
                        print(f"  vLLM: healthy")
                        return True
                    elif health.get("status") == "degraded":
                        print(f"  PDF API: degraded (vLLM not ready)")
                except:
                    pass

            print("  Waiting for services...")
            time.sleep(10)

        return False

    def run_benchmark(self, ip: str, ssh_port: int) -> Optional[Dict]:
        """Run benchmark test"""
        print(f"\n{'='*60}")
        print("Running benchmark...")
        print(f"{'='*60}")

        cmd = self.BENCHMARK_COMMAND.format(api_port=self.API_PORT)
        returncode, stdout, stderr = self.run_ssh_command(
            ip, ssh_port, cmd, timeout=180
        )

        print(stdout)
        if stderr and "error" in stderr.lower():
            print(f"STDERR: {stderr}")

        return {"output": stdout, "success": "SUCCESS" in stdout}

    def deploy(self, gpu_type: str = "A40", name: str = None,
               disk_gb: int = 100, volume_gb: int = 100,
               run_benchmark: bool = False) -> Dict:
        """Full deployment workflow"""

        if name is None:
            name = f"pdf-ai-{gpu_type.lower()}"

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
            result["api_endpoint"] = f"http://{ip}:{api_ext_port}/api/parse"

            # Phase 3: Wait for SSH
            print("\n" + "="*60)
            print("PHASE 3: Establishing SSH Connection")
            print("="*60)
            if not self.wait_for_ssh(ip, ssh_port):
                raise RuntimeError("SSH connection failed")

            # Phase 4: Setup Environment
            print("\n" + "="*60)
            print("PHASE 4: Setting Up Environment")
            print("="*60)
            if not self.setup_environment(ip, ssh_port):
                raise RuntimeError("Environment setup failed")

            # Phase 5: Start vLLM (with retry)
            print("\n" + "="*60)
            print("PHASE 5: Starting vLLM Server")
            print("="*60)
            vllm_pid = self.start_vllm(ip, ssh_port, retry=2)
            if not vllm_pid:
                raise RuntimeError("vLLM failed to start after retries")

            # Wait for vLLM to load model (extended timeout)
            if not self.wait_for_vllm(ip, ssh_port, timeout=420):
                raise RuntimeError("vLLM failed to become ready (timeout or OOM)")

            # Phase 6: Start API Server
            print("\n" + "="*60)
            print("PHASE 6: Starting PDF API Server")
            print("="*60)
            api_pid = self.start_api_server(ip, ssh_port)
            if not api_pid:
                raise RuntimeError("PDF API server failed to start")

            # Phase 7: Verify Services
            print("\n" + "="*60)
            print("PHASE 7: Verifying Services")
            print("="*60)
            if not self.verify_services(ip, ssh_port):
                raise RuntimeError("Service verification failed")

            result["success"] = True

            # Optional: Run benchmark
            if run_benchmark:
                print("\n" + "="*60)
                print("PHASE 8: Running Benchmark")
                print("="*60)
                benchmark_result = self.run_benchmark(ip, ssh_port)
                result["benchmark"] = benchmark_result

        except Exception as e:
            result["error"] = str(e)
            print(f"\nDeployment failed: {e}")

        # Print summary
        self._print_summary(result)
        return result

    def _print_summary(self, result: Dict):
        """Print deployment summary"""
        print("\n" + "="*60)
        print("PDF-AI DEPLOYMENT SUMMARY")
        print("="*60)

        if result["success"]:
            print(f"Status: SUCCESS ✓")
            print(f"Pod ID: {result['pod_id']}")
            print(f"Pod Name: {result['pod_name']}")
            print(f"GPU: {result['gpu']}")
            print(f"Public IP: {result['public_ip']}")
            print(f"\nSSH Command:")
            print(f"  ssh root@{result['public_ip']} -p {result['ssh_port']} -i ~/.ssh/id_ed25519")
            print(f"\nServices:")
            print(f"  vLLM:    http://{result['public_ip']}:{result['vllm_port']}")
            print(f"  PDF API: http://{result['public_ip']}:{result['api_port']}")
            print(f"\nAPI Endpoints:")
            print(f"  POST {result['api_endpoint']}")
            print(f"  POST http://{result['public_ip']}:{result['api_port']}/api/parse/batch")
            print(f"  POST http://{result['public_ip']}:{result['api_port']}/api/parse/stream")
            print(f"  GET  http://{result['public_ip']}:{result['api_port']}/health")
            print(f"\nTest Command:")
            print(f"  curl -X POST {result['api_endpoint']} -F 'file=@test.pdf'")
        else:
            print(f"Status: FAILED ✗")
            print(f"Error: {result['error']}")
            if result['pod_id']:
                print(f"\nPod ID (for debugging): {result['pod_id']}")
                if result['public_ip'] and result['ssh_port']:
                    print(f"SSH: ssh root@{result['public_ip']} -p {result['ssh_port']} -i ~/.ssh/id_ed25519")
                    print(f"\nDebug Commands:")
                    print(f"  tail -100 /workspace/Fun-Audio-Chat/vllm.log")
                    print(f"  tail -100 /workspace/Fun-Audio-Chat/pdf_server.log")
                    print(f"  nvidia-smi")

        print("="*60)

    def status(self):
        """Show status of all pods"""
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
    """Load RunPod API key from environment or .env file"""
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
        description="PDF-AI Service Automated Deployment (v2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 pdf_deploy.py --gpu A40               # Deploy with A40 GPU (tested)
  python3 pdf_deploy.py --gpu H100 --benchmark  # Deploy with H100 and benchmark
  python3 pdf_deploy.py --status                # Show all pods status
  python3 pdf_deploy.py --gpu A40 --disk 150    # Custom disk size

Performance (approximate):
  H100: ~5.71 pages/s
  A100: ~4.0 pages/s
  A40:  ~3.0 pages/s
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
        deployer = PDFServiceDeployer(api_key)

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

            # Exit with appropriate code
            sys.exit(0 if result["success"] else 1)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
