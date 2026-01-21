#!/usr/bin/env python3
"""
PDF-AI Service Automated Deployment Script
Deploys vLLM + LightOnOCR-2-1B + PDF API Server on RunPod GPU Cloud

Usage:
    python3 scripts/pdf_deploy.py --gpu H100
    python3 scripts/pdf_deploy.py --gpu A100 --benchmark
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

    # Setup commands
    SETUP_COMMANDS = '''
set -e

echo "=== Phase 1: System Dependencies ==="
apt update && apt install -y poppler-utils

echo "=== Phase 2: Clone Repository ==="
cd /workspace
if [ ! -d "Fun-Audio-Chat" ]; then
    git clone --recurse-submodules https://github.com/FunAudioLLM/Fun-Audio-Chat
fi
cd Fun-Audio-Chat

echo "=== Phase 3: Python Dependencies ==="
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install vllm>=0.11.1
pip install pypdfium2>=4.0.0 pillow>=10.0.0 aiohttp

echo "=== Phase 4: Download Model ==="
pip install huggingface-hub

# Pre-download the model to cache
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('lightonai/LightOnOCR-2-1B')"

echo "=== Setup Complete ==="
'''

    START_VLLM_COMMAND = '''
cd /workspace/Fun-Audio-Chat
export PYTHONPATH=$(pwd)

# Start vLLM server
nohup vllm serve lightonai/LightOnOCR-2-1B \\
    --port {vllm_port} \\
    --limit-mm-per-prompt '{{"image": 1}}' \\
    --mm-processor-cache-gb 0 \\
    --no-enable-prefix-caching \\
    > vllm.log 2>&1 &
echo $!
'''

    START_API_COMMAND = '''
cd /workspace/Fun-Audio-Chat
export PYTHONPATH=$(pwd)

# Start PDF API server
nohup python3 -m web_demo.server.pdf_server \\
    --port {api_port} \\
    --vllm-endpoint http://localhost:{vllm_port} \\
    --host 0.0.0.0 \\
    > pdf_server.log 2>&1 &
echo $!
'''

    BENCHMARK_COMMAND = '''
cd /workspace/Fun-Audio-Chat
export PYTHONPATH=$(pwd)

# Create test PDF
python3 -c "
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# Create simple test PDF
buffer = io.BytesIO()
c = canvas.Canvas(buffer, pagesize=letter)
c.drawString(100, 750, 'PDF-AI Benchmark Test')
c.drawString(100, 730, 'This is a test document for measuring throughput.')
c.drawString(100, 710, 'The quick brown fox jumps over the lazy dog.')
c.save()

with open('/tmp/test.pdf', 'wb') as f:
    f.write(buffer.getvalue())
print('Test PDF created')
" 2>/dev/null || echo "reportlab not available, using alternative method"

# If reportlab failed, create a simple text-based PDF
if [ ! -f /tmp/test.pdf ]; then
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
fi

# Run benchmark
echo "Running benchmark..."
START=$(date +%s.%N)

curl -s -X POST http://localhost:{api_port}/api/parse \\
    -F "file=@/tmp/test.pdf" \\
    -o /tmp/result.json

END=$(date +%s.%N)
ELAPSED=$(echo "$END - $START" | bc)

echo "Benchmark complete in $ELAPSED seconds"
cat /tmp/result.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Throughput: {{d.get(\"throughput\", \"N/A\")}} pages/s')"
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

        returncode, stdout, stderr = self.run_ssh_command(
            ip, port, self.SETUP_COMMANDS, timeout=1800
        )

        print(stdout)
        if stderr:
            print(f"STDERR: {stderr}")

        if returncode != 0:
            print(f"Setup failed with code {returncode}")
            return False

        print("Environment setup complete!")
        return True

    def start_vllm(self, ip: str, ssh_port: int) -> Optional[str]:
        """Start vLLM server"""
        print(f"\n{'='*60}")
        print("Starting vLLM server...")
        print(f"{'='*60}")

        cmd = self.START_VLLM_COMMAND.format(vllm_port=self.VLLM_PORT)
        returncode, stdout, stderr = self.run_ssh_command(ip, ssh_port, cmd, timeout=60)

        if returncode != 0:
            print(f"Failed to start vLLM: {stderr}")
            return None

        pid = stdout.strip().split('\n')[-1]
        print(f"vLLM started with PID: {pid}")
        return pid

    def wait_for_vllm(self, ip: str, ssh_port: int, timeout: int = 300) -> bool:
        """Wait for vLLM to be ready"""
        print(f"\nWaiting for vLLM to load model...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if vLLM is listening
            returncode, stdout, _ = self.run_ssh_command(
                ip, ssh_port,
                f"curl -s http://localhost:{self.VLLM_PORT}/health",
                timeout=30
            )

            if returncode == 0 and stdout.strip():
                print("  vLLM is ready!")
                return True

            # Check logs for progress
            _, logs, _ = self.run_ssh_command(
                ip, ssh_port,
                "tail -5 /workspace/Fun-Audio-Chat/vllm.log 2>/dev/null || echo 'Loading...'",
                timeout=30
            )
            print(f"  {logs.strip().split(chr(10))[-1][:60]}...")

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

    def verify_services(self, ip: str, ssh_port: int, api_ext_port: int,
                        timeout: int = 60) -> bool:
        """Verify both services are running"""
        print(f"\nVerifying services...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check PDF API
            returncode, stdout, _ = self.run_ssh_command(
                ip, ssh_port,
                f"curl -s http://localhost:{self.API_PORT}/health",
                timeout=30
            )

            if returncode == 0 and "healthy" in stdout.lower():
                print(f"  PDF API server is healthy!")
                return True

            print("  Waiting for PDF API server...")
            time.sleep(10)

        return False

    def run_benchmark(self, ip: str, ssh_port: int) -> Optional[Dict]:
        """Run benchmark test"""
        print(f"\n{'='*60}")
        print("Running benchmark...")
        print(f"{'='*60}")

        cmd = self.BENCHMARK_COMMAND.format(api_port=self.API_PORT)
        returncode, stdout, stderr = self.run_ssh_command(
            ip, ssh_port, cmd, timeout=120
        )

        print(stdout)
        if stderr:
            print(f"STDERR: {stderr}")

        return {"output": stdout}

    def deploy(self, gpu_type: str = "H100", name: str = None,
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

            # Phase 5: Start vLLM
            print("\n" + "="*60)
            print("PHASE 5: Starting vLLM Server")
            print("="*60)
            vllm_pid = self.start_vllm(ip, ssh_port)
            if not vllm_pid:
                raise RuntimeError("vLLM failed to start")

            # Wait for vLLM to load model
            if not self.wait_for_vllm(ip, ssh_port, timeout=300):
                raise RuntimeError("vLLM failed to become ready")

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
            if not self.verify_services(ip, ssh_port, api_ext_port):
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
            print(f"Status: SUCCESS")
            print(f"Pod ID: {result['pod_id']}")
            print(f"Pod Name: {result['pod_name']}")
            print(f"GPU: {result['gpu']}")
            print(f"Public IP: {result['public_ip']}")
            print(f"SSH: ssh root@{result['public_ip']} -p {result['ssh_port']} -i ~/.ssh/id_ed25519")
            print("")
            print("Services:")
            print(f"  vLLM:    http://{result['public_ip']}:{result['vllm_port']} (internal)")
            print(f"  PDF API: http://{result['public_ip']}:{result['api_port']}")
            print("")
            print("API Endpoints:")
            print(f"  POST {result['api_endpoint']}")
            print(f"  POST http://{result['public_ip']}:{result['api_port']}/api/parse/batch")
            print(f"  POST http://{result['public_ip']}:{result['api_port']}/api/parse/stream")
            print(f"  GET  http://{result['public_ip']}:{result['api_port']}/health")
        else:
            print(f"Status: FAILED")
            print(f"Error: {result['error']}")
            if result['pod_id']:
                print(f"Pod ID (for debugging): {result['pod_id']}")

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
        description="PDF-AI Service Automated Deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 pdf_deploy.py --gpu H100              # Deploy with H100 GPU (best)
  python3 pdf_deploy.py --gpu A100 --benchmark  # Deploy and benchmark
  python3 pdf_deploy.py --status                # Show all pods status
  python3 pdf_deploy.py --gpu A40 --disk 150    # Custom disk size
        """
    )

    parser.add_argument("--gpu", default="H100",
                        help="GPU type (H100, A100, A40)")
    parser.add_argument("--name", help="Pod name (auto-generated if not specified)")
    parser.add_argument("--disk", type=int, default=100,
                        help="Container disk size in GB")
    parser.add_argument("--volume", type=int, default=100,
                        help="Persistent volume size in GB")
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
