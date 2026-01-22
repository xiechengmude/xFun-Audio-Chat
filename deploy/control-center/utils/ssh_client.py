"""SSH Client for remote command execution on RunPod pods."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import asyncssh

logger = logging.getLogger(__name__)


class SSHClient:
    """Async SSH client for RunPod pod management."""

    def __init__(
        self,
        host: str,
        port: int,
        username: str = "root",
        key_path: Optional[str] = None,
    ):
        """Initialize SSH client."""
        self.host = host
        self.port = port
        self.username = username
        self.key_path = key_path
        self._conn: Optional[asyncssh.SSHClientConnection] = None

    async def connect(self, timeout: int = 120, retry_interval: int = 5) -> None:
        """Establish SSH connection with retry logic."""
        logger.info(f"Connecting to {self.host}:{self.port}...")

        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"SSH connection timeout after {timeout}s")

            try:
                options = {
                    "host": self.host,
                    "port": self.port,
                    "username": self.username,
                    "known_hosts": None,  # Disable host key checking for RunPod
                    "connect_timeout": 30,
                }

                if self.key_path and Path(self.key_path).exists():
                    options["client_keys"] = [self.key_path]

                self._conn = await asyncssh.connect(**options)
                logger.info(f"SSH connected to {self.host}:{self.port}")
                return

            except (OSError, asyncssh.Error) as e:
                logger.debug(f"SSH connection attempt failed: {e}")
                await asyncio.sleep(retry_interval)

    async def disconnect(self) -> None:
        """Close SSH connection."""
        if self._conn:
            self._conn.close()
            await self._conn.wait_closed()
            self._conn = None
            logger.info("SSH disconnected")

    async def run_command(
        self,
        command: str,
        timeout: int = 300,
        check: bool = True,
    ) -> tuple[str, str, int]:
        """Execute command and return stdout, stderr, exit code."""
        if not self._conn:
            raise RuntimeError("Not connected. Call connect() first.")

        logger.debug(f"Running command: {command[:100]}...")

        try:
            result = await asyncio.wait_for(
                self._conn.run(command, check=False),
                timeout=timeout,
            )

            stdout = result.stdout or ""
            stderr = result.stderr or ""
            exit_code = result.exit_status or 0

            if check and exit_code != 0:
                logger.error(f"Command failed (exit {exit_code}): {stderr}")
                raise RuntimeError(f"Command failed with exit code {exit_code}: {stderr}")

            return stdout, stderr, exit_code

        except asyncio.TimeoutError:
            logger.error(f"Command timed out after {timeout}s")
            raise

    async def run_script(
        self,
        script: str,
        timeout: int = 600,
        check: bool = True,
    ) -> tuple[str, str, int]:
        """Execute a multi-line bash script."""
        # Escape the script for heredoc
        escaped_script = script.replace("'", "'\\''")
        command = f"bash -c '{escaped_script}'"
        return await self.run_command(command, timeout=timeout, check=check)

    async def upload_file(
        self,
        local_path: str,
        remote_path: str,
    ) -> None:
        """Upload a file via SFTP."""
        if not self._conn:
            raise RuntimeError("Not connected. Call connect() first.")

        logger.info(f"Uploading {local_path} -> {remote_path}")

        async with self._conn.start_sftp_client() as sftp:
            await sftp.put(local_path, remote_path)

        logger.info(f"Upload complete: {remote_path}")

    async def download_file(
        self,
        remote_path: str,
        local_path: str,
    ) -> None:
        """Download a file via SFTP."""
        if not self._conn:
            raise RuntimeError("Not connected. Call connect() first.")

        logger.info(f"Downloading {remote_path} -> {local_path}")

        async with self._conn.start_sftp_client() as sftp:
            await sftp.get(remote_path, local_path)

        logger.info(f"Download complete: {local_path}")

    async def file_exists(self, remote_path: str) -> bool:
        """Check if a remote file exists."""
        if not self._conn:
            raise RuntimeError("Not connected. Call connect() first.")

        try:
            async with self._conn.start_sftp_client() as sftp:
                await sftp.stat(remote_path)
                return True
        except asyncssh.SFTPNoSuchFile:
            return False

    async def read_file(self, remote_path: str) -> str:
        """Read a remote file content."""
        stdout, _, _ = await self.run_command(f"cat {remote_path}")
        return stdout

    async def tail_log(
        self,
        log_path: str,
        lines: int = 50,
    ) -> str:
        """Tail a log file."""
        stdout, _, _ = await self.run_command(
            f"tail -n {lines} {log_path} 2>/dev/null || echo 'Log file not found'",
            check=False,
        )
        return stdout

    async def check_process(self, process_name: str) -> bool:
        """Check if a process is running."""
        stdout, _, exit_code = await self.run_command(
            f"pgrep -f '{process_name}' > /dev/null && echo 'running' || echo 'stopped'",
            check=False,
        )
        return "running" in stdout

    async def get_gpu_memory(self) -> dict:
        """Get GPU memory usage."""
        stdout, _, _ = await self.run_command(
            "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits",
            check=False,
        )
        lines = stdout.strip().split("\n")
        if not lines or not lines[0]:
            return {"used": 0, "total": 0}

        parts = lines[0].split(",")
        if len(parts) >= 2:
            return {
                "used": int(parts[0].strip()),
                "total": int(parts[1].strip()),
            }
        return {"used": 0, "total": 0}

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
