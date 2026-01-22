#!/usr/bin/env python3
"""
Fun-Audio-Chat Automated Deployment Script
Handles end-to-end deployment on RunPod GPU Cloud

Usage:
    python3 scripts/auto_deploy.py --gpu A40
    python3 scripts/auto_deploy.py --gpu A40 --test
    python3 scripts/auto_deploy.py --status
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


class FunAudioChatDeployer:
    """Automated deployer for Fun-Audio-Chat on RunPod"""

    # Deployment configuration
    DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    DEFAULT_PORTS = [
        "22/tcp", "8000/tcp", "8001/tcp", "8002/tcp", "8003/tcp",
        "8004/tcp", "8005/tcp", "8006/tcp", "8007/tcp", "8008/tcp",
        "8009/tcp", "8010/tcp", "8080/http", "8888/http"
    ]
    SERVER_PORT = 8002  # Internal port for the server

    # Setup commands to run on the pod
    SETUP_COMMANDS = '''
set -e

echo "=== Phase 1: System Dependencies ==="
apt update && apt install -y ffmpeg

echo "=== Phase 2: Clone Repository ==="
cd /workspace
if [ ! -d "Fun-Audio-Chat" ]; then
    git clone --recurse-submodules https://github.com/FunAudioLLM/Fun-Audio-Chat
fi
cd Fun-Audio-Chat

echo "=== Phase 3: Python Dependencies ==="
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install sphn aiohttp
pip install -r requirements.txt
pip install 'ruamel.yaml<0.18' --force-reinstall

echo "=== Phase 3.5: ASR Dependencies ==="
pip install funasr

echo "=== Phase 4: Download Models ==="
pip install huggingface-hub

# S2S Model
if [ ! -d "pretrained_models/Fun-Audio-Chat-8B" ]; then
    echo "Downloading Fun-Audio-Chat-8B..."
    huggingface-cli download FunAudioLLM/Fun-Audio-Chat-8B --local-dir ./pretrained_models/Fun-Audio-Chat-8B
fi

# TTS Model
if [ ! -d "pretrained_models/Fun-CosyVoice3-0.5B-2512" ]; then
    echo "Downloading Fun-CosyVoice3-0.5B-2512..."
    huggingface-cli download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir ./pretrained_models/Fun-CosyVoice3-0.5B-2512
fi

# ASR Model
if [ ! -d "pretrained_models/Fun-ASR-Nano-2512" ]; then
    echo "Downloading Fun-ASR-Nano-2512..."
    huggingface-cli download FunAudioLLM/Fun-ASR-Nano-2512 --local-dir ./pretrained_models/Fun-ASR-Nano-2512
fi

echo "=== Setup Complete ==="
'''

    START_SERVER_COMMAND = '''
cd /workspace/Fun-Audio-Chat
export PYTHONPATH=$(pwd)
nohup python3 -m web_demo.server.server \
    --model-path pretrained_models/Fun-Audio-Chat-8B \
    --port {port} \
    --tts-gpu 0 \
    --host 0.0.0.0 \
    > server.log 2>&1 &
echo $!
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

        # Find SSH port (internal 22)
        ssh_port = port_mappings.get("22", 22)

        return public_ip, ssh_port

    def get_server_port(self, pod: Dict) -> int:
        """Get the external port mapping for the server"""
        port_mappings = pod.get("portMappings", {})
        return port_mappings.get(str(self.SERVER_PORT), self.SERVER_PORT)

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

        # Run setup commands
        returncode, stdout, stderr = self.run_ssh_command(
            ip, port, self.SETUP_COMMANDS, timeout=1800  # 30 min timeout
        )

        print(stdout)
        if stderr:
            print(f"STDERR: {stderr}")

        if returncode != 0:
            print(f"Setup failed with code {returncode}")
            return False

        print("Environment setup complete!")
        return True

    def start_server(self, ip: str, port: int) -> Optional[str]:
        """Start the Fun-Audio-Chat server"""
        print(f"\n{'='*60}")
        print("Starting server...")
        print(f"{'='*60}")

        cmd = self.START_SERVER_COMMAND.format(port=self.SERVER_PORT)
        returncode, stdout, stderr = self.run_ssh_command(ip, port, cmd, timeout=60)

        if returncode != 0:
            print(f"Failed to start server: {stderr}")
            return None

        pid = stdout.strip().split('\n')[-1]
        print(f"Server started with PID: {pid}")
        return pid

    def verify_server(self, ip: str, ssh_port: int, server_port: int,
                      timeout: int = 120) -> bool:
        """Verify the server is running and responding"""
        print(f"\nVerifying server at {ip}:{server_port}...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if process is running
            returncode, stdout, _ = self.run_ssh_command(
                ip, ssh_port,
                f"ss -tlnp | grep :{self.SERVER_PORT}",
                timeout=30
            )

            if returncode == 0 and stdout.strip():
                print(f"  Server is listening on port {self.SERVER_PORT}")

                # Check logs for successful model loading
                _, logs, _ = self.run_ssh_command(
                    ip, ssh_port,
                    "tail -20 /workspace/Fun-Audio-Chat/server.log",
                    timeout=30
                )

                if "s2s model loaded" in logs.lower() or "cosyvoice loaded" in logs.lower():
                    print("  Models loaded successfully!")
                    return True
                else:
                    print("  Waiting for models to load...")
            else:
                print("  Server not yet listening...")

            time.sleep(10)

        return False

    def deploy(self, gpu_type: str = "A40", name: str = None,
               disk_gb: int = 100, volume_gb: int = 100,
               run_test: bool = False) -> Dict:
        """Full deployment workflow"""

        if name is None:
            name = f"fun-audio-chat-{gpu_type.lower()}"

        result = {
            "success": False,
            "pod_id": None,
            "pod_name": name,
            "gpu": gpu_type,
            "public_ip": None,
            "ssh_port": None,
            "server_port": None,
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
            server_port = self.get_server_port(pod)

            result["public_ip"] = ip
            result["ssh_port"] = ssh_port
            result["server_port"] = server_port
            result["api_endpoint"] = f"ws://{ip}:{server_port}/api/chat"

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

            # Phase 5: Start Server
            print("\n" + "="*60)
            print("PHASE 5: Starting Server")
            print("="*60)
            pid = self.start_server(ip, ssh_port)
            if not pid:
                raise RuntimeError("Server failed to start")

            # Phase 6: Verify
            print("\n" + "="*60)
            print("PHASE 6: Verifying Deployment")
            print("="*60)
            if not self.verify_server(ip, ssh_port, server_port):
                raise RuntimeError("Server verification failed")

            result["success"] = True

            # Optional: Run tests
            if run_test:
                print("\n" + "="*60)
                print("PHASE 7: Running Tests")
                print("="*60)
                # Import and run test script
                try:
                    from test_deployment import test_deployment
                    test_result = test_deployment(ip, server_port)
                    result["test_result"] = test_result
                except ImportError:
                    print("Test script not found, skipping tests")

        except Exception as e:
            result["error"] = str(e)
            print(f"\nDeployment failed: {e}")

        # Print summary
        self._print_summary(result)
        return result

    def _print_summary(self, result: Dict):
        """Print deployment summary"""
        print("\n" + "="*60)
        print("DEPLOYMENT SUMMARY")
        print("="*60)

        if result["success"]:
            print(f"Status: SUCCESS")
            print(f"Pod ID: {result['pod_id']}")
            print(f"Pod Name: {result['pod_name']}")
            print(f"GPU: {result['gpu']}")
            print(f"Public IP: {result['public_ip']}")
            print(f"SSH: ssh root@{result['public_ip']} -p {result['ssh_port']} -i ~/.ssh/id_ed25519")
            print(f"API Endpoint: {result['api_endpoint']}")
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
        description="Fun-Audio-Chat Automated Deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 auto_deploy.py --gpu A40                    # Deploy with A40 GPU
  python3 auto_deploy.py --gpu 4090 --test            # Deploy with 4090 and run tests
  python3 auto_deploy.py --status                     # Show all pods status
  python3 auto_deploy.py --gpu A40 --disk 150         # Custom disk size
        """
    )

    parser.add_argument("--gpu", default="A40",
                        help="GPU type (A40, 4090, A100)")
    parser.add_argument("--name", help="Pod name (auto-generated if not specified)")
    parser.add_argument("--disk", type=int, default=100,
                        help="Container disk size in GB")
    parser.add_argument("--volume", type=int, default=100,
                        help="Persistent volume size in GB")
    parser.add_argument("--test", action="store_true",
                        help="Run tests after deployment")
    parser.add_argument("--status", action="store_true",
                        help="Show status of all pods")

    args = parser.parse_args()

    try:
        api_key = load_api_key()
        deployer = FunAudioChatDeployer(api_key)

        if args.status:
            deployer.status()
        else:
            result = deployer.deploy(
                gpu_type=args.gpu,
                name=args.name,
                disk_gb=args.disk,
                volume_gb=args.volume,
                run_test=args.test
            )

            # Exit with appropriate code
            sys.exit(0 if result["success"] else 1)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
