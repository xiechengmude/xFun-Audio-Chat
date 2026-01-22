"""Deploy Manager - Core deployment orchestration for PDF-AI service."""

import asyncio
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from utils.runpod_client import RunPodClient, PodInfo, GPU_CONFIGS
from utils.ssh_client import SSHClient
from utils.test_runner import TestRunner, TestResult, BenchmarkResult

logger = logging.getLogger(__name__)


class DeploymentPhase(str, Enum):
    """Deployment phases."""
    IDLE = "idle"
    CREATING_POD = "creating_pod"
    WAITING_POD = "waiting_pod"
    SSH_CONNECT = "ssh_connect"
    SETUP_ENV = "setup_env"
    START_VLLM = "start_vllm"
    WAIT_VLLM = "wait_vllm"
    START_API = "start_api"
    WAIT_API = "wait_api"
    TESTING = "testing"
    BENCHMARK = "benchmark"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DeploymentState:
    """Current state of deployment."""
    phase: DeploymentPhase = DeploymentPhase.IDLE
    pod_id: Optional[str] = None
    pod_info: Optional[dict] = None
    gpu_type: str = "A40"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    test_results: list = field(default_factory=list)
    benchmark_result: Optional[dict] = None
    logs: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "phase": self.phase.value,
            "pod_id": self.pod_id,
            "pod_info": self.pod_info,
            "gpu_type": self.gpu_type,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "test_results": self.test_results,
            "benchmark_result": self.benchmark_result,
            "logs": self.logs[-50:],  # Keep last 50 logs
        }

    def add_log(self, message: str):
        """Add log message with timestamp."""
        timestamp = datetime.now().isoformat()
        self.logs.append(f"[{timestamp}] {message}")
        logger.info(message)


class DeployManager:
    """Manages the full deployment lifecycle of PDF-AI service."""

    # Repository with pdf_server.py
    REPO_URL = "https://github.com/xiechengmude/xFun-Audio-Chat"

    # vLLM and API ports
    VLLM_PORT = 8000
    API_PORT = 8006

    # Timeouts
    POD_READY_TIMEOUT = 300
    SSH_CONNECT_TIMEOUT = 120
    VLLM_READY_TIMEOUT = 420
    API_READY_TIMEOUT = 60

    def __init__(
        self,
        runpod_api_key: str,
        state_dir: str = "/data/state",
        ssh_key_path: Optional[str] = None,
    ):
        """Initialize deploy manager."""
        self.runpod = RunPodClient(runpod_api_key)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.ssh_key_path = ssh_key_path
        self.state = DeploymentState()
        self._load_state()

    def _load_state(self):
        """Load state from disk."""
        state_file = self.state_dir / "deployment_state.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                self.state.phase = DeploymentPhase(data.get("phase", "idle"))
                self.state.pod_id = data.get("pod_id")
                self.state.pod_info = data.get("pod_info")
                self.state.gpu_type = data.get("gpu_type", "A40")
                self.state.started_at = data.get("started_at")
                self.state.completed_at = data.get("completed_at")
                self.state.error = data.get("error")
                self.state.test_results = data.get("test_results", [])
                self.state.benchmark_result = data.get("benchmark_result")
                logger.info(f"Loaded state: phase={self.state.phase}, pod_id={self.state.pod_id}")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

    def _save_state(self):
        """Save state to disk."""
        state_file = self.state_dir / "deployment_state.json"
        state_file.write_text(json.dumps(self.state.to_dict(), indent=2))

    def _get_setup_commands(self) -> str:
        """Get environment setup commands."""
        return f'''
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
    git clone --recurse-submodules {self.REPO_URL} Fun-Audio-Chat
    cd Fun-Audio-Chat
fi

echo "=== Phase 3: Python Dependencies ==="
pip install --upgrade pip
pip install vllm>=0.11.1 || echo "vLLM already installed"
pip install pypdfium2>=4.0.0 pillow>=10.0.0 aiohttp

echo "=== Phase 4: Verify pdf_server.py ==="
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

    def _get_vllm_start_command(self, gpu_config) -> str:
        """Get vLLM start command with GPU-specific optimization."""
        return f'''
cd /workspace/Fun-Audio-Chat
export PYTHONPATH=$(pwd)

# Kill any existing vLLM processes
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 3

# Clear GPU memory
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Start vLLM server with memory optimization
nohup vllm serve lightonai/LightOnOCR-2-1B \\
    --port {self.VLLM_PORT} \\
    --limit-mm-per-prompt '{{"image": 1}}' \\
    --mm-processor-cache-gb 0 \\
    --no-enable-prefix-caching \\
    --gpu-memory-utilization {gpu_config.vllm_memory_util} \\
    --max-num-seqs {gpu_config.max_num_seqs} \\
    > vllm.log 2>&1 &

echo "vLLM server starting..."
sleep 5
'''

    def _get_api_start_command(self) -> str:
        """Get PDF API server start command."""
        return f'''
cd /workspace/Fun-Audio-Chat
export PYTHONPATH=$(pwd)

# Kill any existing PDF server
pkill -9 -f "pdf_server" 2>/dev/null || true
sleep 2

# Start PDF API server
nohup python3 -m web_demo.server.pdf_server \\
    --port {self.API_PORT} \\
    --vllm-endpoint http://localhost:{self.VLLM_PORT} \\
    > pdf_api.log 2>&1 &

echo "PDF API server starting..."
sleep 3
'''

    async def deploy(
        self,
        gpu_type: str = "A40",
        run_benchmark: bool = False,
        pod_name: Optional[str] = None,
    ) -> DeploymentState:
        """Execute full deployment workflow."""
        self.state = DeploymentState()
        self.state.gpu_type = gpu_type
        self.state.started_at = datetime.now().isoformat()
        self.state.add_log(f"Starting deployment with GPU: {gpu_type}")

        gpu_config = self.runpod.get_gpu_config(gpu_type)
        ssh_client = None

        try:
            # Phase 1: Create Pod
            self.state.phase = DeploymentPhase.CREATING_POD
            self._save_state()

            pod_name = pod_name or f"pdf-ai-{gpu_type.lower()}"
            self.state.add_log(f"Creating pod: {pod_name}")

            pod_info = self.runpod.create_pod(
                name=pod_name,
                gpu_type=gpu_type,
                volume_size=100,
                disk_size=100,
            )
            self.state.pod_id = pod_info.pod_id
            self.state.add_log(f"Pod created: {pod_info.pod_id}")
            self._save_state()

            # Phase 2: Wait for Pod
            self.state.phase = DeploymentPhase.WAITING_POD
            self._save_state()

            pod_info = await self.runpod.wait_for_pod_ready(
                pod_info.pod_id,
                timeout=self.POD_READY_TIMEOUT,
            )
            self.state.pod_info = {
                "ip": pod_info.ip,
                "ssh_port": pod_info.ssh_port,
                "status": pod_info.status,
            }
            self.state.add_log(f"Pod ready: {pod_info.ip}:{pod_info.ssh_port}")
            self._save_state()

            # Phase 3: SSH Connect
            self.state.phase = DeploymentPhase.SSH_CONNECT
            self._save_state()

            ssh_client = SSHClient(
                host=pod_info.ip,
                port=pod_info.ssh_port,
                key_path=self.ssh_key_path,
            )
            await ssh_client.connect(timeout=self.SSH_CONNECT_TIMEOUT)
            self.state.add_log("SSH connection established")
            self._save_state()

            # Phase 4: Setup Environment
            self.state.phase = DeploymentPhase.SETUP_ENV
            self._save_state()

            self.state.add_log("Setting up environment...")
            setup_commands = self._get_setup_commands()
            stdout, stderr, _ = await ssh_client.run_script(setup_commands, timeout=600)
            self.state.add_log("Environment setup complete")
            self._save_state()

            # Phase 5: Start vLLM
            self.state.phase = DeploymentPhase.START_VLLM
            self._save_state()

            self.state.add_log("Starting vLLM server...")
            vllm_command = self._get_vllm_start_command(gpu_config)
            await ssh_client.run_script(vllm_command, timeout=60)
            self._save_state()

            # Phase 6: Wait for vLLM
            self.state.phase = DeploymentPhase.WAIT_VLLM
            self._save_state()

            self.state.add_log(f"Waiting for vLLM (timeout: {self.VLLM_READY_TIMEOUT}s)...")
            await self._wait_for_vllm(ssh_client, pod_info.ip)
            self.state.add_log("vLLM server is ready")
            self._save_state()

            # Phase 7: Start PDF API
            self.state.phase = DeploymentPhase.START_API
            self._save_state()

            self.state.add_log("Starting PDF API server...")
            api_command = self._get_api_start_command()
            await ssh_client.run_script(api_command, timeout=60)
            self._save_state()

            # Phase 8: Wait for API
            self.state.phase = DeploymentPhase.WAIT_API
            self._save_state()

            self.state.add_log(f"Waiting for PDF API (timeout: {self.API_READY_TIMEOUT}s)...")
            await self._wait_for_api(pod_info.ip)
            self.state.add_log("PDF API server is ready")
            self._save_state()

            # Phase 9: Run Tests
            self.state.phase = DeploymentPhase.TESTING
            self._save_state()

            self.state.add_log("Running E2E tests...")
            test_runner = TestRunner(
                api_host=pod_info.ip,
                api_port=self.API_PORT,
                vllm_port=self.VLLM_PORT,
            )
            test_results = await test_runner.run_all_tests()
            self.state.test_results = [
                {"name": r.name, "passed": r.passed, "duration": r.duration, "message": r.message}
                for r in test_results
            ]
            self._save_state()

            # Check test results
            failed_tests = [r for r in test_results if not r.passed]
            if failed_tests:
                self.state.phase = DeploymentPhase.FAILED
                self.state.error = f"Tests failed: {[r.name for r in failed_tests]}"
                self.state.add_log(f"Deployment failed: {self.state.error}")
                self._save_state()
                return self.state

            self.state.add_log(f"All tests passed ({len(test_results)} tests)")

            # Phase 10: Benchmark (optional)
            if run_benchmark:
                self.state.phase = DeploymentPhase.BENCHMARK
                self._save_state()

                self.state.add_log("Running benchmark...")
                # TODO: Use actual PDF files for benchmark
                benchmark = await test_runner.run_benchmark(
                    pdf_paths=[],  # Will use generated test PDFs
                    num_iterations=3,
                )
                self.state.benchmark_result = {
                    "total_pages": benchmark.total_pages,
                    "total_time": benchmark.total_time,
                    "throughput": benchmark.throughput,
                    "avg_latency": benchmark.avg_latency,
                    "errors": benchmark.errors,
                }
                self.state.add_log(
                    f"Benchmark: {benchmark.throughput:.2f} pages/s, "
                    f"avg latency: {benchmark.avg_latency:.2f}s"
                )
                self._save_state()

            # Completed
            self.state.phase = DeploymentPhase.COMPLETED
            self.state.completed_at = datetime.now().isoformat()
            self.state.add_log("Deployment completed successfully!")
            self._save_state()

            return self.state

        except Exception as e:
            self.state.phase = DeploymentPhase.FAILED
            self.state.error = str(e)
            self.state.add_log(f"Deployment failed: {e}")
            self._save_state()
            logger.exception("Deployment failed")
            return self.state

        finally:
            if ssh_client:
                await ssh_client.disconnect()

    async def _wait_for_vllm(self, ssh_client: SSHClient, host: str):
        """Wait for vLLM to be ready with OOM detection."""
        import httpx

        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > self.VLLM_READY_TIMEOUT:
                # Check logs for OOM
                logs = await ssh_client.tail_log("/workspace/Fun-Audio-Chat/vllm.log", 100)
                if "CUDA out of memory" in logs or "OutOfMemoryError" in logs:
                    raise RuntimeError("vLLM failed with CUDA OOM error. Try a larger GPU.")
                raise TimeoutError(f"vLLM not ready after {self.VLLM_READY_TIMEOUT}s")

            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(f"http://{host}:{self.VLLM_PORT}/health")
                    if resp.status_code == 200:
                        return
            except Exception:
                pass

            # Check for OOM in logs
            if int(elapsed) % 30 == 0:
                logs = await ssh_client.tail_log("/workspace/Fun-Audio-Chat/vllm.log", 20)
                if "CUDA out of memory" in logs or "OutOfMemoryError" in logs:
                    raise RuntimeError("vLLM failed with CUDA OOM error. Try a larger GPU.")

            await asyncio.sleep(5)

    async def _wait_for_api(self, host: str):
        """Wait for PDF API to be ready."""
        import httpx

        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > self.API_READY_TIMEOUT:
                raise TimeoutError(f"PDF API not ready after {self.API_READY_TIMEOUT}s")

            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(f"http://{host}:{self.API_PORT}/health")
                    if resp.status_code == 200:
                        return
            except Exception:
                pass

            await asyncio.sleep(2)

    async def terminate(self, pod_id: Optional[str] = None) -> bool:
        """Terminate a deployment."""
        target_pod = pod_id or self.state.pod_id
        if not target_pod:
            logger.warning("No pod to terminate")
            return False

        self.state.add_log(f"Terminating pod: {target_pod}")
        success = self.runpod.terminate_pod(target_pod)

        if success:
            self.state.phase = DeploymentPhase.IDLE
            self.state.pod_id = None
            self.state.pod_info = None
            self._save_state()

        return success

    def get_status(self) -> DeploymentState:
        """Get current deployment status."""
        return self.state

    def list_pods(self) -> list[PodInfo]:
        """List all pods."""
        return self.runpod.list_pods()
