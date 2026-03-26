#!/usr/bin/env python3
"""
Wrapper for real mini-swe-agent library with token tracking and custom configuration.
Apptainer version - uses Apptainer instead of Docker.
"""

import json
import os
import subprocess
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import openai


# Import TokenTracker and TokenTrackingModel from the original agent_wrapper
# We can reuse these classes since they don't depend on Docker
from utils.agent.min_swe_open_models.agent_wrapper import (
    TokenTracker,
    TokenTrackingModel
)


class MiniSWEAgentRunner:
    """
    Runner for mini-swe-agent with custom configuration for code reconstruction.
    Uses Apptainer instead of Docker.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        config_path: Path,
        max_turns: int = 75,
        api_base_url: str = None,
        use_custom_endpoint: bool = False
    ):
        """
        Initialize mini-swe-agent runner.

        Args:
            model_name: Model name
            api_key: API key (can be dummy for custom endpoints)
            config_path: Path to YAML config file
            max_turns: Maximum number of turns
            api_base_url: Custom API base URL (e.g., "http://localhost:8000/v1")
            use_custom_endpoint: Whether using a custom OpenAI-compatible endpoint
        """
        self.model_name = model_name
        self.api_key = api_key
        self.config_path = config_path
        self.max_turns = max_turns
        self.api_base_url = api_base_url
        self.use_custom_endpoint = use_custom_endpoint
        self.token_tracker = TokenTracker(model_name)

    def create_task_instance(self, target_file: str, readme: str) -> str:
        """
        Create task instance by rendering the YAML template.

        Args:
            target_file: Target file to implement
            readme: README content

        Returns:
            Rendered task prompt
        """
        # Load config
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Get instance template
        instance_template = config['agent']['instance_template']

        # Render with Jinja2
        from jinja2 import Template
        template = Template(instance_template)
        rendered = template.render(target_file=target_file, readme=readme)

        return rendered

    def run(
        self,
        target_file: str,
        readme: str,
        container_name: str  # This is actually instance_name for Apptainer
    ) -> Dict[str, Any]:
        """
        Run mini-swe-agent on a task.

        Args:
            target_file: Target file to implement
            readme: README content
            container_name: Apptainer instance name (from apptainer_manager)

        Returns:
            Result dictionary
        """
        try:
            # Import mini-swe-agent components
            from minisweagent.agents.default import DefaultAgent
            from minisweagent.models.litellm_model import LitellmModel

            # Import our custom Apptainer environment
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from apptainer import ApptainerEnvironment

            print(f"\nInitializing mini-swe-agent for {target_file}...")

            # Load config
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Override step limit
            config['agent']['step_limit'] = self.max_turns

            # Set up environment - use config from YAML
            env_config = config.get('environment', {})
            # Create ApptainerEnvironment with config from YAML
            env = ApptainerEnvironment(
                image="python:3.12",  # Will use docker://python:3.12
                cwd=env_config.get('cwd', '/workspace'),
                env=env_config.get('env', {}),
                timeout=env_config.get('timeout', 60)
            )

            # Copy files from apptainer_manager's instance to mini-swe-agent's instance
            print(f"Copying project files from {container_name} to mini-swe-agent instance...")
            source_instance = container_name
            target_instance = env.instance_name

            # Use tar to copy files between instances
            # 1. Export files from source instance to tar
            tar_file = Path(f"/tmp/workspace_transfer_{os.getpid()}.tar")
            try:
                # Create tar from source instance
                with open(tar_file, 'wb') as f:
                    result = subprocess.run(
                        ["apptainer", "exec", f"instance://{source_instance}",
                         "tar", "-C", "/", "-cf", "-", "workspace"],
                        stdout=f,
                        check=True
                    )

                # Extract tar into target instance
                with open(tar_file, 'rb') as f:
                    result = subprocess.run(
                        ["apptainer", "exec", f"instance://{target_instance}",
                         "tar", "-C", "/", "-xf", "-"],
                        stdin=f,
                        check=True
                    )

                print(f"✓ Files copied to instance {target_instance}")
            finally:
                # Cleanup tar file
                if tar_file.exists():
                    tar_file.unlink()

            # Install dependencies in mini-swe-agent instance
            # Check if requirements.txt exists
            result = subprocess.run(
                ["apptainer", "exec", f"instance://{target_instance}",
                 "test", "-f", "/workspace/requirements.txt"],
                capture_output=True
            )
            if result.returncode == 0:
                print("Installing project dependencies from requirements.txt...")
                result = subprocess.run(
                    ["apptainer", "exec", f"instance://{target_instance}",
                     "pip", "install", "-q", "-r", "/workspace/requirements.txt"],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print(f"Warning: Dependency installation had issues:\n{result.stderr}")
                else:
                    print(f"  ✓ Dependencies installed")

            # Set up model configuration
            if self.use_custom_endpoint:
                # For custom endpoints like Qwen/Qwen3-Coder-30B-A3B-Instruct server
                # Use openai/ prefix to indicate OpenAI-compatible endpoint
                model_name_for_litellm = f"openai/{self.model_name}"

                # Set environment variables for custom endpoint
                if self.api_base_url:
                    os.environ['OPENAI_API_BASE'] = self.api_base_url
                    os.environ['OPENAI_BASE_URL'] = self.api_base_url  # Alternative env var

                # API key might be optional for local servers
                os.environ['OPENAI_API_KEY'] = self.api_key or "dummy-key"
            else:
                # Standard OpenAI models
                os.environ['OPENAI_API_KEY'] = self.api_key
                model_name_for_litellm = self.model_name

            # Get model_kwargs from config - these are passed to litellm.completion(), not to LitellmModel
            model_kwargs_for_completion = config.get('model', {}).get('model_kwargs', {})

            # Handle custom endpoint configuration
            if self.use_custom_endpoint:
                # Add base_url for custom endpoint
                if self.api_base_url:
                    model_kwargs_for_completion['api_base'] = self.api_base_url

                # Convert OpenAI-specific parameters to standard ones for custom models
                if 'max_completion_tokens' in model_kwargs_for_completion:
                    # Most open models use max_tokens instead of max_completion_tokens
                    model_kwargs_for_completion['max_tokens'] = model_kwargs_for_completion.pop('max_completion_tokens')

                # Keep drop_params=true to handle any remaining incompatible parameters
                model_kwargs_for_completion['drop_params'] = True

                print(f"Using custom endpoint with model_kwargs: {model_kwargs_for_completion}")

            base_model = LitellmModel(
                model_name=model_name_for_litellm,
                model_kwargs=model_kwargs_for_completion  # These get passed to litellm.completion()
            )

            # Wrap model with token tracking
            model = TokenTrackingModel(base_model, self.token_tracker)

            # Import the exception class we need
            from minisweagent.agents.default import TerminatingException

            # Create custom exception for early stopping
            class EarlyStopEmptyResponses(TerminatingException):
                """Raised when agent produces 3 consecutive empty responses."""
                pass

            # Create agent with config from YAML
            # Pass agent config as kwargs
            agent_config = config.get('agent', {})
            agent = DefaultAgent(model, env, **agent_config)

            # Monkey-patch the agent's query method to:
            # 1. Fix the add_message unpacking bug
            # 2. Detect consecutive empty responses for early stopping
            consecutive_empty_count = [0]  # Use list to allow mutation in nested function

            def query_with_fixes():
                # Query the model
                if 0 < agent.config.step_limit <= agent.model.n_calls or 0 < agent.config.cost_limit <= agent.model.cost:
                    from minisweagent.agents.default import LimitsExceeded
                    raise LimitsExceeded()

                response = agent.model.query(agent.messages)

                # Fix: Extract content and extra before calling add_message
                # mini-swe-agent returns {'content': '...', 'extra': {...}}
                # but add_message expects content as positional arg
                content = response.get('content', '')
                extra = response.get('extra', {})
                agent.add_message("assistant", content, **extra)

                # Check for empty responses
                if not content.strip():
                    consecutive_empty_count[0] += 1
                    print(f"⚠️  Empty response detected ({consecutive_empty_count[0]}/3)")

                    if consecutive_empty_count[0] >= 3:
                        print("\n⚠️  3 consecutive empty responses - stopping early!")
                        raise EarlyStopEmptyResponses('Stopped due to 3 consecutive empty responses')
                else:
                    consecutive_empty_count[0] = 0  # Reset counter

                return response

            agent.query = query_with_fixes

            print(f"Starting agent execution (max {self.max_turns} turns)...\n")

            # Run agent - pass template variables as kwargs
            # The instance_template in YAML uses {{ target_file }} and {{ readme }}
            exit_status, message = agent.run(
                task="",  # task is not used in our instance_template
                target_file=target_file,
                readme=readme
            )

            # Get turn count from agent history
            # Each turn = user message + assistant message, so divide by 2
            # Subtract 1 for initial system/user messages
            turns = max(0, (len(agent.messages) - 2) // 2)

            # Read the implemented file from the instance BEFORE it's removed
            implemented_content = None
            try:
                result = subprocess.run(
                    ["apptainer", "exec", f"instance://{env.instance_name}",
                     "cat", f"/workspace/{target_file}"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                implemented_content = result.stdout
            except Exception as e:
                print(f"Error reading file from instance: {e}")

            # Extract trajectory (full message history)
            trajectory = []
            for msg in agent.messages:
                trajectory.append({
                    'role': msg.get('role', ''),
                    'content': msg.get('content', '')
                })

            # Check multiple completion indicators
            is_completed = (
                'CompletedTask' in exit_status or
                'completed' in exit_status.lower() or
                'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' in message or
                turns < self.max_turns  # If agent finished early, likely completed
            )

            return {
                'status': 'completed' if is_completed else 'failed',
                'turns': turns,
                'output': message,
                'token_usage': self.token_tracker.get_summary(),
                'exit_status': exit_status,
                'implemented_content': implemented_content,
                'trajectory': trajectory
            }

        except Exception as e:
            print(f"Error running mini-swe-agent: {e}")
            import traceback
            traceback.print_exc()

            return {
                'status': 'error',
                'error': str(e),
                'turns': 0,
                'token_usage': self.token_tracker.get_summary()
            }
