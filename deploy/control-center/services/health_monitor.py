"""Health Monitor - Background service for monitoring deployed pods."""

import asyncio
import logging
from datetime import datetime
from typing import Optional

import httpx

from utils.runpod_client import RunPodClient, PodInfo
from utils.test_runner import TestRunner

logger = logging.getLogger(__name__)


class HealthMonitor:
    """Background health monitoring for PDF-AI deployments."""

    def __init__(
        self,
        runpod_client: RunPodClient,
        check_interval: int = 60,
        vllm_port: int = 8000,
        api_port: int = 8006,
    ):
        """Initialize health monitor."""
        self.runpod = runpod_client
        self.check_interval = check_interval
        self.vllm_port = vllm_port
        self.api_port = api_port
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._health_status: dict = {}

    async def start(self):
        """Start the health monitoring loop."""
        if self._running:
            logger.warning("Health monitor already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitor started")

    async def stop(self):
        """Stop the health monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitor stopped")

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_all_pods()
            except Exception as e:
                logger.error(f"Health check error: {e}")

            await asyncio.sleep(self.check_interval)

    async def _check_all_pods(self):
        """Check health of all active pods."""
        pods = self.runpod.list_pods()

        for pod in pods:
            if pod.status != "RUNNING" or not pod.ip:
                self._health_status[pod.pod_id] = {
                    "status": "not_running",
                    "pod_status": pod.status,
                    "checked_at": datetime.now().isoformat(),
                }
                continue

            health = await self._check_pod_health(pod)
            self._health_status[pod.pod_id] = health

    async def _check_pod_health(self, pod: PodInfo) -> dict:
        """Check health of a single pod."""
        result = {
            "pod_id": pod.pod_id,
            "ip": pod.ip,
            "checked_at": datetime.now().isoformat(),
            "vllm_healthy": False,
            "api_healthy": False,
            "status": "unhealthy",
        }

        # Check vLLM health
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"http://{pod.ip}:{self.vllm_port}/health")
                result["vllm_healthy"] = resp.status_code == 200
        except Exception as e:
            logger.debug(f"vLLM health check failed for {pod.pod_id}: {e}")

        # Check API health
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"http://{pod.ip}:{self.api_port}/health")
                result["api_healthy"] = resp.status_code == 200
        except Exception as e:
            logger.debug(f"API health check failed for {pod.pod_id}: {e}")

        # Overall status
        if result["vllm_healthy"] and result["api_healthy"]:
            result["status"] = "healthy"
        elif result["vllm_healthy"] or result["api_healthy"]:
            result["status"] = "degraded"
        else:
            result["status"] = "unhealthy"

        return result

    def get_health_status(self) -> dict:
        """Get current health status of all pods."""
        return self._health_status

    def get_pod_health(self, pod_id: str) -> Optional[dict]:
        """Get health status of a specific pod."""
        return self._health_status.get(pod_id)

    async def check_pod_now(self, pod_id: str) -> Optional[dict]:
        """Immediately check health of a specific pod."""
        pod = self.runpod.get_pod(pod_id)
        if not pod:
            return None

        health = await self._check_pod_health(pod)
        self._health_status[pod_id] = health
        return health
