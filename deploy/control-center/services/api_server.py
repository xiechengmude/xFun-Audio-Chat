"""API Server - FastAPI gateway for PDF-AI Control Center."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from services.deploy_manager import DeployManager, DeploymentPhase
from services.health_monitor import HealthMonitor
from utils.runpod_client import RunPodClient, GPU_CONFIGS
from utils.test_runner import TestRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings from environment variables."""
    runpod_api_key: str = Field(default="", env="RUNPOD_API_KEY")
    default_gpu: str = Field(default="A40", env="DEFAULT_GPU")
    vllm_port: int = Field(default=8000, env="VLLM_PORT")
    api_port: int = Field(default=8006, env="API_PORT")
    ssh_key_path: Optional[str] = Field(default=None, env="SSH_KEY_PATH")
    state_dir: str = Field(default="/data/state", env="STATE_DIR")
    health_check_interval: int = Field(default=60, env="HEALTH_CHECK_INTERVAL")

    class Config:
        env_file = ".env"


settings = Settings()


# Global instances
deploy_manager: Optional[DeployManager] = None
health_monitor: Optional[HealthMonitor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global deploy_manager, health_monitor

    if not settings.runpod_api_key:
        logger.warning("RUNPOD_API_KEY not set - deployment features disabled")
    else:
        # Initialize services
        deploy_manager = DeployManager(
            runpod_api_key=settings.runpod_api_key,
            state_dir=settings.state_dir,
            ssh_key_path=settings.ssh_key_path,
        )

        runpod_client = RunPodClient(settings.runpod_api_key)
        health_monitor = HealthMonitor(
            runpod_client=runpod_client,
            check_interval=settings.health_check_interval,
            vllm_port=settings.vllm_port,
            api_port=settings.api_port,
        )

        # Start health monitor
        await health_monitor.start()
        logger.info("Control center started")

    yield

    # Cleanup
    if health_monitor:
        await health_monitor.stop()
    logger.info("Control center stopped")


app = FastAPI(
    title="PDF-AI Control Center",
    description="Deployment and management API for PDF-AI service on RunPod",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class DeployRequest(BaseModel):
    """Deployment request parameters."""
    gpu_type: str = Field(default="A40", description="GPU type (H100, A100, A40, RTX4090)")
    run_benchmark: bool = Field(default=False, description="Run performance benchmark")
    pod_name: Optional[str] = Field(default=None, description="Custom pod name")


class DeployResponse(BaseModel):
    """Deployment response."""
    success: bool
    message: str
    deployment_id: Optional[str] = None
    status: Optional[dict] = None


class StatusResponse(BaseModel):
    """Status response."""
    phase: str
    pod_id: Optional[str]
    pod_info: Optional[dict]
    gpu_type: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error: Optional[str]
    test_results: list
    benchmark_result: Optional[dict]


class PodResponse(BaseModel):
    """Pod information response."""
    pod_id: str
    name: str
    status: str
    gpu_type: str
    ip: Optional[str]
    ssh_port: Optional[int]


class TestRequest(BaseModel):
    """Test request parameters."""
    pod_id: Optional[str] = Field(default=None, description="Target pod ID")


class TestResponse(BaseModel):
    """Test response."""
    success: bool
    results: list
    summary: dict


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "pdf-ai-control-center",
        "runpod_configured": bool(settings.runpod_api_key),
    }


@app.get("/api/info")
async def get_info():
    """Get control center information."""
    return {
        "service": "PDF-AI Control Center",
        "version": "1.0.0",
        "default_gpu": settings.default_gpu,
        "available_gpus": list(GPU_CONFIGS.keys()),
        "vllm_port": settings.vllm_port,
        "api_port": settings.api_port,
    }


@app.post("/api/deploy", response_model=DeployResponse)
async def deploy(request: DeployRequest, background_tasks: BackgroundTasks):
    """Trigger a new deployment."""
    if not deploy_manager:
        raise HTTPException(status_code=503, detail="RunPod not configured")

    # Check if deployment already in progress
    current_state = deploy_manager.get_status()
    if current_state.phase not in [DeploymentPhase.IDLE, DeploymentPhase.COMPLETED, DeploymentPhase.FAILED]:
        return DeployResponse(
            success=False,
            message=f"Deployment already in progress (phase: {current_state.phase.value})",
            status=current_state.to_dict(),
        )

    # Validate GPU type
    if request.gpu_type.upper() not in GPU_CONFIGS:
        available = ", ".join(GPU_CONFIGS.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Invalid GPU type: {request.gpu_type}. Available: {available}",
        )

    # Start deployment in background
    async def run_deployment():
        await deploy_manager.deploy(
            gpu_type=request.gpu_type,
            run_benchmark=request.run_benchmark,
            pod_name=request.pod_name,
        )

    background_tasks.add_task(asyncio.create_task, run_deployment())

    return DeployResponse(
        success=True,
        message=f"Deployment started with GPU: {request.gpu_type}",
        deployment_id=None,  # Will be set once pod is created
    )


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get current deployment status."""
    if not deploy_manager:
        raise HTTPException(status_code=503, detail="RunPod not configured")

    state = deploy_manager.get_status()
    return StatusResponse(
        phase=state.phase.value,
        pod_id=state.pod_id,
        pod_info=state.pod_info,
        gpu_type=state.gpu_type,
        started_at=state.started_at,
        completed_at=state.completed_at,
        error=state.error,
        test_results=state.test_results,
        benchmark_result=state.benchmark_result,
    )


@app.get("/api/pods")
async def list_pods():
    """List all RunPod pods."""
    if not deploy_manager:
        raise HTTPException(status_code=503, detail="RunPod not configured")

    pods = deploy_manager.list_pods()
    return {
        "pods": [
            PodResponse(
                pod_id=p.pod_id,
                name=p.name,
                status=p.status,
                gpu_type=p.gpu_type,
                ip=p.ip,
                ssh_port=p.ssh_port,
            ).model_dump()
            for p in pods
        ]
    }


@app.delete("/api/pods/{pod_id}")
async def terminate_pod(pod_id: str):
    """Terminate a specific pod."""
    if not deploy_manager:
        raise HTTPException(status_code=503, detail="RunPod not configured")

    success = await deploy_manager.terminate(pod_id)
    if success:
        return {"success": True, "message": f"Pod {pod_id} terminated"}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to terminate pod {pod_id}")


@app.post("/api/test", response_model=TestResponse)
async def run_tests(request: TestRequest):
    """Run E2E tests against deployed service."""
    if not deploy_manager:
        raise HTTPException(status_code=503, detail="RunPod not configured")

    # Get pod info
    state = deploy_manager.get_status()
    pod_id = request.pod_id or state.pod_id

    if not pod_id:
        raise HTTPException(status_code=400, detail="No pod ID specified and no active deployment")

    pod_info = state.pod_info
    if not pod_info or not pod_info.get("ip"):
        raise HTTPException(status_code=400, detail="Pod info not available")

    # Run tests
    test_runner = TestRunner(
        api_host=pod_info["ip"],
        api_port=settings.api_port,
        vllm_port=settings.vllm_port,
    )

    results = await test_runner.run_all_tests()

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    return TestResponse(
        success=failed == 0,
        results=[
            {"name": r.name, "passed": r.passed, "duration": r.duration, "message": r.message}
            for r in results
        ],
        summary={"total": len(results), "passed": passed, "failed": failed},
    )


@app.get("/api/health-status")
async def get_health_status():
    """Get health status of all monitored pods."""
    if not health_monitor:
        raise HTTPException(status_code=503, detail="Health monitor not configured")

    return {"health_status": health_monitor.get_health_status()}


@app.post("/api/health-check/{pod_id}")
async def check_pod_health(pod_id: str):
    """Immediately check health of a specific pod."""
    if not health_monitor:
        raise HTTPException(status_code=503, detail="Health monitor not configured")

    health = await health_monitor.check_pod_now(pod_id)
    if health is None:
        raise HTTPException(status_code=404, detail=f"Pod {pod_id} not found")

    return health


@app.get("/api/logs")
async def get_deployment_logs(limit: int = 50):
    """Get deployment logs."""
    if not deploy_manager:
        raise HTTPException(status_code=503, detail="RunPod not configured")

    state = deploy_manager.get_status()
    return {"logs": state.logs[-limit:]}


def main():
    """Run the API server."""
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(
        "services.api_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
