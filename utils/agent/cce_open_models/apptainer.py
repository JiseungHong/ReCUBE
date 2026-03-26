#!/usr/bin/env python3
"""
Apptainer environment class compatible with minisweagent API.

This provides the same interface as minisweagent's DockerEnvironment
but uses Apptainer instead of Docker.
"""

import logging
import os
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ApptainerEnvironmentConfig:
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the instance."""
    forward_env: list[str] = field(default_factory=list)
    """Environment variables to forward to the instance.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 30
    """Timeout for executing commands in the instance."""
    executable: str = os.getenv("MSWEA_APPTAINER_EXECUTABLE", "apptainer")
    """Path to the apptainer executable."""
    run_args: list[str] = field(default_factory=lambda: ["--writable-tmpfs"])
    """Additional arguments to pass to apptainer instance start.
    Default is ["--writable-tmpfs"], which allows pip installs.
    """
    instance_timeout: str = "2h"
    """Max duration to keep instance running. Uses the same format as the sleep command."""
    pull_timeout: int = 300
    """Timeout in seconds for pulling images."""


class ApptainerEnvironment:
    def __init__(
        self,
        *,
        config_class: type = ApptainerEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        """This class executes bash commands in an Apptainer instance using apptainer commands.
        See `ApptainerEnvironmentConfig` for keyword arguments.
        """
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.instance_name: str | None = None
        self.config = config_class(**kwargs)
        self._start_instance()

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    def _start_instance(self):
        """Start the Apptainer instance and return the instance name."""
        self.instance_name = f"minisweagent-{uuid.uuid4().hex[:8]}"

        # Pull image to SIF if needed
        sif_path = self._ensure_sif_image()

        cmd = [
            self.config.executable,
            "instance",
            "start",
            *self.config.run_args,
            str(sif_path),
            self.instance_name,
        ]

        self.logger.debug(f"Starting instance with command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.config.pull_timeout,
            check=True,
        )
        self.logger.info(f"Started Apptainer instance {self.instance_name}")

    def _ensure_sif_image(self) -> str:
        """Ensure SIF image exists, pulling if necessary."""
        from pathlib import Path

        # Check if image is already a SIF file
        if self.config.image.endswith('.sif') and Path(self.config.image).exists():
            return self.config.image

        # Create cache directory for SIF files
        cache_dir = Path.home() / ".cache" / "apptainer_sif"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate SIF filename from image URI
        image_uri = self.config.image
        if not image_uri.startswith("docker://"):
            image_uri = f"docker://{image_uri}"

        sif_name = image_uri.replace("docker://", "").replace(":", "_").replace("/", "_") + ".sif"
        sif_path = cache_dir / sif_name

        # Pull image if not already cached
        if not sif_path.exists():
            self.logger.info(f"Pulling {image_uri}...")
            subprocess.run(
                [self.config.executable, "pull", str(sif_path), image_uri],
                check=True,
                capture_output=True,
                text=True,
                timeout=self.config.pull_timeout
            )
            self.logger.info(f"Image pulled: {sif_path.name}")

        return str(sif_path)

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the Apptainer instance and return the result as a dict."""
        cwd = cwd or self.config.cwd
        assert self.instance_name, "Instance not started"

        cmd = [self.config.executable, "exec", "--pwd", cwd]

        # Add forwarded environment variables
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["--env", f"{key}={value}"])

        # Add explicit environment variables
        for key, value in self.config.env.items():
            cmd.extend(["--env", f"{key}={value}"])

        cmd.extend([f"instance://{self.instance_name}", "bash", "-lc", command])

        result = subprocess.run(
            cmd,
            text=True,
            timeout=timeout or self.config.timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": result.stdout, "returncode": result.returncode}

    def cleanup(self):
        """Stop the Apptainer instance."""
        if getattr(self, "instance_name", None) is not None:
            cmd = f"(timeout 60 {self.config.executable} instance stop {self.instance_name}) >/dev/null 2>&1 &"
            subprocess.Popen(cmd, shell=True)

    def __del__(self):
        """Cleanup instance when object is destroyed."""
        self.cleanup()
