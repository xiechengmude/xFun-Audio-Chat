"""CLI Tool for PDF-AI Control Center."""

import asyncio
import os
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.deploy_manager import DeployManager, DeploymentPhase
from utils.runpod_client import RunPodClient, GPU_CONFIGS
from utils.test_runner import TestRunner

app = typer.Typer(
    name="pdf-ai",
    help="PDF-AI Control Center CLI",
    add_completion=False,
)
console = Console()


def get_api_key() -> str:
    """Get RunPod API key from environment."""
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        console.print("[red]Error: RUNPOD_API_KEY not set[/red]")
        raise typer.Exit(1)
    return api_key


@app.command()
def deploy(
    gpu: str = typer.Option("A40", "--gpu", "-g", help="GPU type (H100, A100, A40, RTX4090)"),
    benchmark: bool = typer.Option(False, "--benchmark", "-b", help="Run performance benchmark"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Custom pod name"),
):
    """Deploy PDF-AI service to RunPod."""
    api_key = get_api_key()

    # Validate GPU type
    if gpu.upper() not in GPU_CONFIGS:
        available = ", ".join(GPU_CONFIGS.keys())
        console.print(f"[red]Invalid GPU type: {gpu}. Available: {available}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold blue]Deploying PDF-AI service with {gpu}...[/bold blue]")

    async def run_deploy():
        manager = DeployManager(
            runpod_api_key=api_key,
            state_dir="/data/state" if os.path.exists("/data") else "./state",
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Starting deployment...", total=None)

            # Start deployment
            state = await manager.deploy(
                gpu_type=gpu,
                run_benchmark=benchmark,
                pod_name=name,
            )

            progress.update(task, description=f"Deployment {state.phase.value}")

        # Show results
        if state.phase == DeploymentPhase.COMPLETED:
            console.print("\n[bold green]Deployment successful![/bold green]")

            if state.pod_info:
                table = Table(title="Pod Information")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="green")
                table.add_row("Pod ID", state.pod_id or "N/A")
                table.add_row("IP", state.pod_info.get("ip", "N/A"))
                table.add_row("SSH Port", str(state.pod_info.get("ssh_port", "N/A")))
                table.add_row("API URL", f"http://{state.pod_info.get('ip')}:8006")
                console.print(table)

            if state.test_results:
                console.print("\n[bold]Test Results:[/bold]")
                for result in state.test_results:
                    status = "[green]PASS[/green]" if result["passed"] else "[red]FAIL[/red]"
                    console.print(f"  {status} {result['name']}: {result['message']}")

            if state.benchmark_result:
                console.print(f"\n[bold]Benchmark:[/bold]")
                console.print(f"  Throughput: {state.benchmark_result['throughput']:.2f} pages/s")
                console.print(f"  Avg Latency: {state.benchmark_result['avg_latency']:.2f}s/page")

        else:
            console.print(f"\n[bold red]Deployment failed: {state.error}[/bold red]")
            raise typer.Exit(1)

    asyncio.run(run_deploy())


@app.command()
def status():
    """Show current deployment status."""
    api_key = get_api_key()

    manager = DeployManager(
        runpod_api_key=api_key,
        state_dir="/data/state" if os.path.exists("/data") else "./state",
    )

    state = manager.get_status()

    table = Table(title="Deployment Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    phase_color = {
        "completed": "green",
        "failed": "red",
        "idle": "dim",
    }.get(state.phase.value, "yellow")

    table.add_row("Phase", f"[{phase_color}]{state.phase.value}[/{phase_color}]")
    table.add_row("GPU Type", state.gpu_type)
    table.add_row("Pod ID", state.pod_id or "N/A")

    if state.pod_info:
        table.add_row("IP", state.pod_info.get("ip", "N/A"))
        table.add_row("SSH Port", str(state.pod_info.get("ssh_port", "N/A")))

    table.add_row("Started", state.started_at or "N/A")
    table.add_row("Completed", state.completed_at or "N/A")

    if state.error:
        table.add_row("Error", f"[red]{state.error}[/red]")

    console.print(table)


@app.command()
def pods():
    """List all RunPod pods."""
    api_key = get_api_key()
    client = RunPodClient(api_key)

    pod_list = client.list_pods()

    if not pod_list:
        console.print("[dim]No pods found[/dim]")
        return

    table = Table(title="RunPod Pods")
    table.add_column("Pod ID", style="cyan")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("GPU")
    table.add_column("IP")
    table.add_column("SSH Port")

    for pod in pod_list:
        status_color = "green" if pod.status == "RUNNING" else "yellow"
        table.add_row(
            pod.pod_id,
            pod.name,
            f"[{status_color}]{pod.status}[/{status_color}]",
            pod.gpu_type,
            pod.ip or "N/A",
            str(pod.ssh_port) if pod.ssh_port else "N/A",
        )

    console.print(table)


@app.command()
def terminate(
    pod_id: str = typer.Argument(..., help="Pod ID to terminate"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Terminate a RunPod pod."""
    api_key = get_api_key()

    if not force:
        confirm = typer.confirm(f"Terminate pod {pod_id}?")
        if not confirm:
            console.print("[dim]Cancelled[/dim]")
            raise typer.Exit(0)

    client = RunPodClient(api_key)

    if client.terminate_pod(pod_id):
        console.print(f"[green]Pod {pod_id} terminated[/green]")
    else:
        console.print(f"[red]Failed to terminate pod {pod_id}[/red]")
        raise typer.Exit(1)


@app.command()
def test(
    host: str = typer.Option(..., "--host", "-h", help="Target host IP"),
    api_port: int = typer.Option(8006, "--api-port", help="PDF API port"),
    vllm_port: int = typer.Option(8000, "--vllm-port", help="vLLM port"),
):
    """Run E2E tests against a deployed service."""

    async def run_tests():
        runner = TestRunner(
            api_host=host,
            api_port=api_port,
            vllm_port=vllm_port,
        )

        console.print(f"[bold]Running tests against {host}...[/bold]\n")

        results = await runner.run_all_tests()

        for result in results:
            status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
            console.print(f"{status} {result.name}")
            console.print(f"     Duration: {result.duration:.2f}s")
            console.print(f"     Message: {result.message}\n")

        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        if failed == 0:
            console.print(f"[bold green]All {passed} tests passed![/bold green]")
        else:
            console.print(f"[bold red]{failed} tests failed, {passed} passed[/bold red]")
            raise typer.Exit(1)

    asyncio.run(run_tests())


@app.command()
def gpus():
    """Show available GPU configurations."""
    table = Table(title="Available GPUs")
    table.add_column("Type", style="cyan")
    table.add_column("Display Name")
    table.add_column("Memory")
    table.add_column("vLLM Memory Util")
    table.add_column("Max Sequences")

    for name, config in GPU_CONFIGS.items():
        table.add_row(
            name,
            config.display_name,
            f"{config.memory_gb} GB",
            f"{config.vllm_memory_util:.0%}",
            str(config.max_num_seqs),
        )

    console.print(table)


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
