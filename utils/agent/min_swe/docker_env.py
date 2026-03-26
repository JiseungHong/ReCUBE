#!/usr/bin/env python3
"""
Docker environment management for mini-swe-agent inference.
"""

import docker
import json
import os
import tempfile
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Optional
import io


class DockerEnvironmentManager:
    """Manages Docker containers for mini-swe-agent inference."""

    def __init__(self, base_image: str = "python:3.12"):
        """
        Initialize Docker environment manager.

        Args:
            base_image: Base Docker image to use
        """
        self.client = docker.from_env()
        self.base_image = base_image
        self.containers: Dict[str, docker.models.containers.Container] = {}
        self.workspace_paths: Dict[str, Path] = {}
        self.original_files: Dict[str, Dict[str, str]] = {}  # repo_id -> {file_path: content}

    def create_environment(
        self,
        repo_id: str,
        dependencies: List[str],
        files: Dict[str, str],
        target_files: List[str],
        readme: str = ""
    ) -> str:
        """
        Create a Docker container with dependencies installed and ALL files in original state.
        Target files are NOT modified during creation - they will be modified one at a time
        when processing each file.

        Args:
            repo_id: Repository ID
            dependencies: List of pip package specifications
            files: Dictionary of file paths to contents
            target_files: List of target files (for reference, not modified here)
            readme: README content to write as README.md

        Returns:
            Container name
        """
        print(f"Creating Docker environment for repo {repo_id}...")

        # Create a temporary directory for the workspace
        workspace_path = Path(f"/tmp/mini_swe_workspace_{repo_id}")
        workspace_path.mkdir(parents=True, exist_ok=True)
        self.workspace_paths[repo_id] = workspace_path

        # Store original files for later restoration
        self.original_files[repo_id] = files.copy()

        # Write README.md if provided
        if readme:
            readme_file = workspace_path / "README.md"
            readme_file.write_text(readme)

        # Write all files to workspace in ORIGINAL state (no modification)
        for file_path, content in files.items():
            full_path = workspace_path / file_path

            # Create parent directories
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file AS-IS (no modification)
            full_path.write_text(content)

        # Create requirements.txt if dependencies exist
        if dependencies:
            req_file = workspace_path / "requirements.txt"
            req_file.write_text("\n".join(dependencies))

        # Pull base image if not exists
        try:
            self.client.images.get(self.base_image)
        except docker.errors.ImageNotFound:
            print(f"Pulling {self.base_image}...")
            self.client.images.pull(self.base_image)

        # Create and start container with timestamp to avoid conflicts with dead containers
        timestamp = int(time.time())
        container_name = f"mini_swe_{repo_id}_{timestamp}"

        # Remove existing container if exists (skip if it's corrupted/dead)
        try:
            old_container = self.client.containers.get(container_name)
            old_container.remove(force=True)
        except (docker.errors.NotFound, docker.errors.APIError):
            pass

        # Create container with workspace mounted
        container = self.client.containers.create(
            self.base_image,
            command="/bin/bash",
            name=container_name,
            detach=True,
            tty=True,
            stdin_open=True,
            working_dir="/workspace",
            volumes={
                str(workspace_path.absolute()): {
                    'bind': '/workspace',
                    'mode': 'rw'
                }
            },
            # Increase shared memory size for some ML packages
            shm_size='2G'
        )

        container.start()

        # Install dependencies
        if dependencies:
            print(f"Installing {len(dependencies)} dependencies...")
            self._install_dependencies(container, dependencies)

        self.containers[repo_id] = container

        print(f"✓ Docker environment created: {container.id[:12]}")
        print(f"  All files are in ORIGINAL state (not modified)")
        return container_name  # Return container name, not ID

    def prepare_target_file(self, repo_id: str, target_file: str):
        """
        Prepare a single target file for reconstruction by:
        1. Modifying it (remove imports, NotImplementedError for Python files)
        2. Writing it to the workspace

        This should be called BEFORE running the agent on each target file.

        Args:
            repo_id: Repository ID
            target_file: Target file to prepare
        """
        from utils.parsers import modify_file_content

        if repo_id not in self.original_files:
            raise ValueError(f"No original files found for repo {repo_id}")

        if target_file not in self.original_files[repo_id]:
            raise ValueError(f"File {target_file} not found in original files")

        workspace_path = self.workspace_paths[repo_id]
        original_content = self.original_files[repo_id][target_file]

        # Modify the content
        try:
            modified_content = modify_file_content(original_content, target_file)
        except Exception as e:
            print(f"Warning: Could not modify {target_file}: {e}")
            modified_content = original_content

        # Write modified content to workspace
        full_path = workspace_path / target_file
        full_path.write_text(modified_content)

        print(f"✓ Prepared {target_file} for reconstruction (modified in workspace)")

    def restore_original_file(self, repo_id: str, target_file: str):
        """
        Restore a file to its original state.
        This should be called AFTER processing each target file to reset the environment.

        Args:
            repo_id: Repository ID
            target_file: Target file to restore
        """
        if repo_id not in self.original_files:
            raise ValueError(f"No original files found for repo {repo_id}")

        if target_file not in self.original_files[repo_id]:
            raise ValueError(f"File {target_file} not found in original files")

        workspace_path = self.workspace_paths[repo_id]
        original_content = self.original_files[repo_id][target_file]

        # Write original content back to workspace
        full_path = workspace_path / target_file
        full_path.write_text(original_content)

        print(f"✓ Restored {target_file} to original state")

    def _install_dependencies(
        self,
        container: docker.models.containers.Container,
        dependencies: List[str]
    ):
        """
        Install dependencies in the container.

        Args:
            container: Docker container
            dependencies: List of pip package specifications
        """
        # Upgrade pip first
        exit_code, output = container.exec_run(
            "pip install --upgrade pip",
            workdir="/workspace"
        )

        if exit_code != 0:
            print(f"Warning: pip upgrade failed: {output.decode()}")

        # Install each dependency
        # Install all at once for faster installation
        deps_str = " ".join(f'"{dep}"' for dep in dependencies)
        cmd = f"pip install {deps_str}"

        exit_code, output = container.exec_run(
            cmd,
            workdir="/workspace"
        )

        if exit_code != 0:
            print(f"Warning: Some dependencies failed to install:")
            print(output.decode())
            # Try installing one by one for failed packages
            print("Attempting individual installation...")
            for dep in dependencies:
                exit_code, output = container.exec_run(
                    f'pip install "{dep}"',
                    workdir="/workspace"
                )
                if exit_code != 0:
                    print(f"  ✗ Failed to install {dep}")
                else:
                    print(f"  ✓ Installed {dep}")
        else:
            print("✓ All dependencies installed successfully")

    def execute_command(
        self,
        repo_id: str,
        command: str,
        workdir: str = "/workspace"
    ) -> tuple[int, str]:
        """
        Execute a command in the container.

        Args:
            repo_id: Repository ID
            command: Command to execute
            workdir: Working directory

        Returns:
            Tuple of (exit_code, output)
        """
        if repo_id not in self.containers:
            raise ValueError(f"No container found for repo {repo_id}")

        container = self.containers[repo_id]

        exit_code, output = container.exec_run(
            command,
            workdir=workdir
        )

        return exit_code, output.decode()

    def read_file(self, repo_id: str, file_path: str) -> Optional[str]:
        """
        Read a file from the container.

        Args:
            repo_id: Repository ID
            file_path: Path to file in container

        Returns:
            File contents or None if file doesn't exist
        """
        if repo_id not in self.containers:
            raise ValueError(f"No container found for repo {repo_id}")

        container = self.containers[repo_id]

        try:
            # Use cat to read file
            exit_code, output = container.exec_run(
                f"cat {file_path}",
                workdir="/workspace"
            )

            if exit_code == 0:
                return output.decode()
            else:
                return None
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    def cleanup(self, repo_id: str):
        """
        Clean up container for a repository.

        Args:
            repo_id: Repository ID
        """
        if repo_id in self.containers:
            container = self.containers[repo_id]
            try:
                container.stop()
                container.remove()
                print(f"✓ Cleaned up container for repo {repo_id}")
            except Exception as e:
                print(f"Warning: Error cleaning up container: {e}")

            del self.containers[repo_id]

        # Clean up workspace directory
        if repo_id in self.workspace_paths:
            workspace_path = self.workspace_paths[repo_id]
            if workspace_path.exists():
                import shutil
                shutil.rmtree(workspace_path, ignore_errors=True)
            del self.workspace_paths[repo_id]

        # Clean up original files
        if repo_id in self.original_files:
            del self.original_files[repo_id]

    def cleanup_all(self):
        """Clean up all containers."""
        for repo_id in list(self.containers.keys()):
            self.cleanup(repo_id)

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup_all()
        except:
            pass
