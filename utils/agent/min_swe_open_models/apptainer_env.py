#!/usr/bin/env python3
"""
Apptainer environment management for mini-swe-agent inference.
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shlex


class ApptainerEnvironmentManager:
    """Manages Apptainer instances for mini-swe-agent inference."""

    def __init__(self, base_image: str = "docker://python:3.12"):
        """
        Initialize Apptainer environment manager.

        Args:
            base_image: Base image URI (e.g., docker://python:3.12)
        """
        self.base_image = base_image
        self.instances: Dict[str, str] = {}  # repo_id -> instance_name
        self.workspace_paths: Dict[str, Path] = {}
        self.original_files: Dict[str, Dict[str, str]] = {}  # repo_id -> {file_path: content}
        self.sif_images: Dict[str, Path] = {}  # image_uri -> sif_path

        # Ensure apptainer is available
        try:
            subprocess.run(["apptainer", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Apptainer is not installed or not in PATH")

    def _pull_image(self, image_uri: str) -> Path:
        """
        Pull a container image and convert to SIF format.

        Args:
            image_uri: Image URI (e.g., docker://python:3.12)

        Returns:
            Path to SIF file
        """
        if image_uri in self.sif_images:
            return self.sif_images[image_uri]

        # Create a cache directory for SIF files
        cache_dir = Path.home() / ".cache" / "apptainer_sif"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate SIF filename from image URI
        sif_name = image_uri.replace("docker://", "").replace(":", "_").replace("/", "_") + ".sif"
        sif_path = cache_dir / sif_name

        # Pull image if not already cached
        if not sif_path.exists():
            print(f"Pulling {image_uri}...")
            try:
                subprocess.run(
                    ["apptainer", "pull", str(sif_path), image_uri],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"✓ Image pulled: {sif_path.name}")
            except subprocess.CalledProcessError as e:
                print(f"Error pulling image: {e.stderr}")
                raise
        else:
            print(f"✓ Using cached image: {sif_path.name}")

        self.sif_images[image_uri] = sif_path
        return sif_path

    def create_environment(
        self,
        repo_id: str,
        dependencies: List[str],
        files: Dict[str, str],
        target_files: List[str],
        readme: str = ""
    ) -> str:
        """
        Create an Apptainer instance with dependencies installed and ALL files in original state.
        Target files are NOT modified during creation - they will be modified one at a time
        when processing each file.

        Args:
            repo_id: Repository ID
            dependencies: List of pip package specifications
            files: Dictionary of file paths to contents
            target_files: List of target files (for reference, not modified here)
            readme: README content to write as README.md

        Returns:
            Instance name
        """
        print(f"Creating Apptainer environment for repo {repo_id}...")

        # Pull/get SIF image
        sif_path = self._pull_image(self.base_image)

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

        # Create and start instance with timestamp to avoid conflicts
        timestamp = int(time.time())
        instance_name = f"mini_swe_{repo_id}_{timestamp}"

        # Stop existing instance if exists
        try:
            subprocess.run(
                ["apptainer", "instance", "stop", instance_name],
                capture_output=True,
                timeout=10
            )
        except:
            pass

        # Start instance with writable tmpfs and workspace bind
        print(f"Starting Apptainer instance {instance_name}...")
        try:
            subprocess.run(
                [
                    "apptainer", "instance", "start",
                    "--writable-tmpfs",  # Allow pip installs
                    "--bind", f"{workspace_path.absolute()}:/workspace",
                    str(sif_path),
                    instance_name
                ],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error starting instance: {e.stderr}")
            raise

        self.instances[repo_id] = instance_name

        # Install dependencies
        if dependencies:
            print(f"Installing {len(dependencies)} dependencies...")
            self._install_dependencies(instance_name, dependencies)

        print(f"✓ Apptainer environment created: {instance_name}")
        print(f"  All files are in ORIGINAL state (not modified)")
        return instance_name

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
        instance_name: str,
        dependencies: List[str]
    ):
        """
        Install dependencies in the instance.

        Args:
            instance_name: Apptainer instance name
            dependencies: List of pip package specifications
        """
        # Upgrade pip first
        result = subprocess.run(
            ["apptainer", "exec", f"instance://{instance_name}",
             "pip", "install", "--upgrade", "pip"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Warning: pip upgrade failed: {result.stderr}")

        # Install all dependencies at once for faster installation
        deps_list = ["pip", "install"] + dependencies
        result = subprocess.run(
            ["apptainer", "exec", f"instance://{instance_name}"] + deps_list,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Warning: Some dependencies failed to install:")
            print(result.stderr)
            # Try installing one by one for failed packages
            print("Attempting individual installation...")
            for dep in dependencies:
                result = subprocess.run(
                    ["apptainer", "exec", f"instance://{instance_name}",
                     "pip", "install", dep],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
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
    ) -> Tuple[int, str]:
        """
        Execute a command in the instance.

        Args:
            repo_id: Repository ID
            command: Command to execute
            workdir: Working directory

        Returns:
            Tuple of (exit_code, output)
        """
        if repo_id not in self.instances:
            raise ValueError(f"No instance found for repo {repo_id}")

        instance_name = self.instances[repo_id]

        result = subprocess.run(
            ["apptainer", "exec", "--pwd", workdir,
             f"instance://{instance_name}", "bash", "-c", command],
            capture_output=True,
            text=True
        )

        return result.returncode, result.stdout

    def read_file(self, repo_id: str, file_path: str) -> Optional[str]:
        """
        Read a file from the instance.

        Args:
            repo_id: Repository ID
            file_path: Path to file in instance

        Returns:
            File contents or None if file doesn't exist
        """
        if repo_id not in self.instances:
            raise ValueError(f"No instance found for repo {repo_id}")

        instance_name = self.instances[repo_id]

        try:
            result = subprocess.run(
                ["apptainer", "exec", f"instance://{instance_name}",
                 "cat", file_path],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                return result.stdout
            else:
                return None
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    def cleanup(self, repo_id: str):
        """
        Clean up instance for a repository.

        Args:
            repo_id: Repository ID
        """
        if repo_id in self.instances:
            instance_name = self.instances[repo_id]
            try:
                subprocess.run(
                    ["apptainer", "instance", "stop", instance_name],
                    capture_output=True,
                    timeout=10
                )
                print(f"✓ Cleaned up instance for repo {repo_id}")
            except Exception as e:
                print(f"Warning: Error cleaning up instance: {e}")

            del self.instances[repo_id]

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
        """Clean up all instances."""
        for repo_id in list(self.instances.keys()):
            self.cleanup(repo_id)

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup_all()
        except:
            pass
