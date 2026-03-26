#!/usr/bin/env python3
"""
Apptainer environment management for graph-based mini-swe-agent inference.

Extends the base apptainer environment to include dependency graph tools.
"""

import json
import os
import subprocess
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ApptainerEnvironmentManager:
    """Manages Apptainer instances for graph-based mini-swe-agent inference."""

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
        readme: str = "",
        graph_path: Optional[Path] = None
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
            graph_path: Path to the dependency graph .pkl file

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
                print(f"  ✓ Copied {dst_name} to workspace")
            else:
                print(f"  ✗ Warning: {src_name} not found at {src_path}")

        # Copy dependency graph if provided
        if graph_path and graph_path.exists():
            graphs_dir = tools_path / "graphs"
            graphs_dir.mkdir(parents=True, exist_ok=True)
            graph_dest = graphs_dir / f"{repo_id}.pkl"
            shutil.copy(graph_path, graph_dest)
            print(f"  ✓ Copied dependency graph to workspace: {graph_dest.name}")
        else:
            print(f"  ✗ Warning: Graph file not found at {graph_path}")

        # Update graph_loader.py to use /workspace/tools/graphs/ path
        graph_loader_file = tools_path / "graph_loader.py"
        if graph_loader_file.exists():
            content = graph_loader_file.read_text()
            # Replace the path to look in /workspace/tools/graphs/ instead
            content = content.replace(
                'base_dir / f"data/Code_GitHub/graphs/{repo_id}.pkl"',
                'Path(f"/workspace/tools/graphs/{repo_id}.pkl")'
            )
            graph_loader_file.write_text(content)
            print(f"  ✓ Updated graph_loader.py paths for instance")

        # Create and start instance with timestamp to avoid conflicts
        timestamp = int(time.time())
        instance_name = f"mini_swe_graph_{repo_id}_{timestamp}"

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

        # Install dependencies if needed
        if dependencies:
            print("Installing dependencies...")
            result = subprocess.run(
                ["apptainer", "exec", f"instance://{instance_name}",
                 "pip", "install", "-q", "-r", "/workspace/requirements.txt"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"Warning: Dependency installation had issues:")
                print(result.stderr)
            else:
                print(f"  ✓ Dependencies installed")

        # Install networkx (required for graph tools)
        print("Installing networkx for graph tools...")
        result = subprocess.run(
            ["apptainer", "exec", f"instance://{instance_name}",
             "pip", "install", "-q", "networkx"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            error_msg = f"CRITICAL: networkx installation failed:\n{result.stderr}"
            print(error_msg)
            subprocess.run(
                ["apptainer", "instance", "stop", instance_name],
                capture_output=True
            )
            raise RuntimeError(error_msg)
        else:
            print(f"  ✓ networkx installed")

        print(f"✓ Instance {instance_name} ready")
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

    def cleanup(self, repo_id: str):
        """Clean up Apptainer instance for a repository."""
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
                print(f"Warning: Failed to cleanup instance for repo {repo_id}: {e}")
            finally:
                del self.instances[repo_id]

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
        """Clean up all instances."""
        for repo_id in list(self.instances.keys()):
            self.cleanup(repo_id)

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup_all()
        except:
            pass
