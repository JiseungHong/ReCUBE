#!/usr/bin/env python3
"""
Docker environment management for graph-based mini-swe-agent inference.

Extends the base docker environment to include dependency graph tools.
"""

import docker
import json
import os
import pickle
import shutil
import tempfile
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Optional
import io


class DockerEnvironmentManager:
    """Manages Docker containers for graph-based mini-swe-agent inference."""

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
        readme: str = "",
        graph_path: Optional[Path] = None
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
            graph_path: Path to the dependency graph .pkl file

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

        # Create tools directory for graph tools
        tools_path = workspace_path / "tools"
        tools_path.mkdir(parents=True, exist_ok=True)

        # Copy graph tools and helper modules
        modules_to_copy = [
            ("graph_tools.py", "graph_tools.py"),
            ("graph_loader.py", "graph_loader.py"),
            ("graph_helper.py", "graph_helper.py"),
            ("similar_files.py", "similar_files.py"),
        ]

        for src_name, dst_name in modules_to_copy:
            src_path = Path(__file__).parent / src_name
            if src_path.exists():
                shutil.copy(src_path, tools_path / dst_name)
                print(f"  ✓ Copied {dst_name} to container")
            else:
                print(f"  ✗ Warning: {src_name} not found at {src_path}")

        # Copy dependency graph if provided
        if graph_path and graph_path.exists():
            graphs_dir = tools_path / "graphs"
            graphs_dir.mkdir(parents=True, exist_ok=True)
            graph_dest = graphs_dir / f"{repo_id}.pkl"
            shutil.copy(graph_path, graph_dest)
            print(f"  ✓ Copied dependency graph to container: {graph_dest.name}")
        else:
            print(f"  ✗ Warning: Graph file not found at {graph_path}")

        # Update graph_loader.py to use /workspace/tools/graphs/ path
        graph_loader_container = tools_path / "graph_loader.py"
        if graph_loader_container.exists():
            content = graph_loader_container.read_text()
            # Replace the path to look in /workspace/tools/graphs/ instead
            content = content.replace(
                'base_dir / f"data/Code_GitHub/graphs/{repo_id}.pkl"',
                'Path(f"/workspace/tools/graphs/{repo_id}.pkl")'
            )
            graph_loader_container.write_text(content)
            print(f"  ✓ Updated graph_loader.py paths for container")

        # Pull base image if not exists
        try:
            self.client.images.get(self.base_image)
        except docker.errors.ImageNotFound:
            print(f"Pulling {self.base_image}...")
            self.client.images.pull(self.base_image)

        # Create and start container with timestamp to avoid conflicts
        timestamp = int(time.time())
        container_name = f"mini_swe_graph_{repo_id}_{timestamp}"

        # Remove existing container if exists
        try:
            old_container = self.client.containers.get(container_name)
            old_container.remove(force=True)
        except docker.errors.NotFound:
            pass

        print(f"Starting container {container_name}...")

        # Start container
        container = self.client.containers.run(
            self.base_image,
            command="sleep infinity",
            name=container_name,
            detach=True,
            working_dir="/workspace",
            volumes={
                str(workspace_path.absolute()): {
                    'bind': '/workspace',
                    'mode': 'rw'
                }
            }
        )

        self.containers[repo_id] = container

        # Install dependencies if needed
        if dependencies:
            print("Installing dependencies...")
            exit_code, output = container.exec_run(
                "pip install -q -r requirements.txt",
                workdir="/workspace"
            )
            if exit_code != 0:
                print(f"Warning: Dependency installation had issues:")
                print(output.decode())
            else:
                print(f"  ✓ Dependencies installed")

        # Install networkx (required for graph tools)
        print("Installing networkx for graph tools...")
        exit_code, output = container.exec_run("pip install -q networkx")
        if exit_code != 0:
            error_msg = f"CRITICAL: networkx installation failed:\n{output.decode()}"
            print(error_msg)
            container.remove(force=True)
            raise RuntimeError(error_msg)
        else:
            print(f"  ✓ networkx installed")

        print(f"✓ Container {container_name} ready")
        print(f"  All files are in ORIGINAL state (not modified)")

        return container_name

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

    def cleanup(self, repo_id: str):
        """Clean up Docker container for a repository."""
        if repo_id in self.containers:
            try:
                container = self.containers[repo_id]
                container.stop(timeout=5)
                container.remove()
                print(f"✓ Cleaned up container for repo {repo_id}")
            except Exception as e:
                print(f"Warning: Failed to cleanup container for repo {repo_id}: {e}")
            finally:
                del self.containers[repo_id]

        # Clean up workspace directory
        if repo_id in self.workspace_paths:
            workspace_path = self.workspace_paths[repo_id]
            if workspace_path.exists():
                try:
                    shutil.rmtree(workspace_path)
                except Exception as e:
                    print(f"Warning: Failed to cleanup workspace for repo {repo_id}: {e}")
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
