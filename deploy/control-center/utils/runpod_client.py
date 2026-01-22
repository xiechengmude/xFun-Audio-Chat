"""RunPod API Client wrapper for PDF-AI Control Center."""

import asyncio
import logging
from typing import Optional
from dataclasses import dataclass, field

import runpod

logger = logging.getLogger(__name__)


@dataclass
class PodInfo:
    """Information about a RunPod pod."""
    pod_id: str
    name: str
    status: str
    gpu_type: str
    ip: Optional[str] = None
    ssh_port: Optional[int] = None
    gpu_count: int = 1
    volume_size: int = 100
    disk_size: int = 100

    @property
    def ssh_host(self) -> Optional[str]:
        """Get SSH connection string."""
        if self.ip and self.ssh_port:
            return f"{self.ip}:{self.ssh_port}"
        return None


@dataclass
class GPUConfig:
    """GPU configuration for different types."""
    gpu_id: str
    display_name: str
    memory_gb: int
    vllm_memory_util: float = 0.85
    max_num_seqs: int = 64


# GPU configurations based on real deployment experience
GPU_CONFIGS = {
    "H100": GPUConfig("NVIDIA H100 80GB HBM3", "H100 80GB", 80, 0.90, 128),
    "A100": GPUConfig("NVIDIA A100 80GB PCIe", "A100 80GB", 80, 0.90, 128),
    "A100-40": GPUConfig("NVIDIA A100-PCIE-40GB", "A100 40GB", 40, 0.85, 64),
    "A40": GPUConfig("NVIDIA A40", "A40 48GB", 48, 0.85, 64),
    "RTX4090": GPUConfig("NVIDIA GeForce RTX 4090", "RTX 4090", 24, 0.80, 32),
    "RTX3090": GPUConfig("NVIDIA GeForce RTX 3090", "RTX 3090", 24, 0.80, 32),
}


class RunPodClient:
    """Client for interacting with RunPod API."""

    # Default RunPod image with CUDA and Python
    DEFAULT_IMAGE = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"

    # Template for vLLM optimized image
    VLLM_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

    def __init__(self, api_key: str):
        """Initialize RunPod client with API key."""
        self.api_key = api_key
        runpod.api_key = api_key

    def get_gpu_config(self, gpu_type: str) -> GPUConfig:
        """Get GPU configuration by type."""
        gpu_type_upper = gpu_type.upper()
        if gpu_type_upper not in GPU_CONFIGS:
            available = ", ".join(GPU_CONFIGS.keys())
            raise ValueError(f"Unknown GPU type: {gpu_type}. Available: {available}")
        return GPU_CONFIGS[gpu_type_upper]

    def create_pod(
        self,
        name: str,
        gpu_type: str = "A40",
        gpu_count: int = 1,
        volume_size: int = 100,
        disk_size: int = 100,
        image: Optional[str] = None,
    ) -> PodInfo:
        """Create a new RunPod pod."""
        gpu_config = self.get_gpu_config(gpu_type)

        logger.info(f"Creating pod: {name}")
        logger.info(f"GPU: {gpu_config.display_name} x{gpu_count}")
        logger.info(f"Volume: {volume_size}GB, Disk: {disk_size}GB")

        pod = runpod.create_pod(
            name=name,
            image_name=image or self.VLLM_IMAGE,
            gpu_type_id=gpu_config.gpu_id,
            gpu_count=gpu_count,
            volume_in_gb=volume_size,
            container_disk_in_gb=disk_size,
            ports="8000/http,8006/http,22/tcp",
            volume_mount_path="/workspace",
        )

        pod_id = pod.get("id")
        logger.info(f"Pod created: {pod_id}")

        return PodInfo(
            pod_id=pod_id,
            name=name,
            status="CREATED",
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            volume_size=volume_size,
            disk_size=disk_size,
        )

    def get_pod(self, pod_id: str) -> Optional[PodInfo]:
        """Get pod information by ID."""
        pod = runpod.get_pod(pod_id)
        if not pod:
            return None

        # Extract SSH connection info
        ip = None
        ssh_port = None

        runtime = pod.get("runtime")
        if runtime:
            ports = runtime.get("ports", [])
            for port in ports:
                if port.get("privatePort") == 22:
                    ip = port.get("ip")
                    ssh_port = port.get("publicPort")
                    break

        return PodInfo(
            pod_id=pod_id,
            name=pod.get("name", ""),
            status=pod.get("desiredStatus", "UNKNOWN"),
            gpu_type=pod.get("gpuTypeId", "UNKNOWN"),
            ip=ip,
            ssh_port=ssh_port,
        )

    def list_pods(self) -> list[PodInfo]:
        """List all pods."""
        pods = runpod.get_pods()
        result = []

        for pod in pods:
            pod_id = pod.get("id")
            info = self.get_pod(pod_id)
            if info:
                result.append(info)

        return result

    def terminate_pod(self, pod_id: str) -> bool:
        """Terminate a pod."""
        logger.info(f"Terminating pod: {pod_id}")
        try:
            runpod.terminate_pod(pod_id)
            logger.info(f"Pod {pod_id} terminated")
            return True
        except Exception as e:
            logger.error(f"Failed to terminate pod {pod_id}: {e}")
            return False

    async def wait_for_pod_ready(
        self,
        pod_id: str,
        timeout: int = 300,
        poll_interval: int = 5,
    ) -> PodInfo:
        """Wait for pod to be ready with SSH accessible."""
        logger.info(f"Waiting for pod {pod_id} to be ready...")

        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Pod {pod_id} not ready after {timeout}s")

            pod_info = self.get_pod(pod_id)
            if not pod_info:
                raise ValueError(f"Pod {pod_id} not found")

            logger.info(f"  Status: {pod_info.status}")

            if pod_info.status == "RUNNING" and pod_info.ip and pod_info.ssh_port:
                logger.info(f"Pod ready! IP: {pod_info.ip}, SSH Port: {pod_info.ssh_port}")
                return pod_info

            await asyncio.sleep(poll_interval)

    def get_gpu_availability(self) -> dict[str, bool]:
        """Check GPU availability across regions."""
        result = {}
        for gpu_name, config in GPU_CONFIGS.items():
            try:
                gpus = runpod.get_gpus()
                available = any(
                    g.get("id") == config.gpu_id and g.get("available", False)
                    for g in gpus
                )
                result[gpu_name] = available
            except Exception:
                result[gpu_name] = False
        return result
