#!/usr/bin/env python3
"""
Wrapper for mini-swe-agent with graph tool support.
Apptainer version - uses Apptainer instead of Docker.

Extends the base agent_wrapper_apptainer to pass repo_id to templates.
"""

import sys
from pathlib import Path
import yaml
from typing import Dict, Any

from utils.agent.min_swe_open_models.agent_wrapper_apptainer import (
    MiniSWEAgentRunner as BaseRunner,
    TokenTracker
)


class GraphMiniSWEAgentRunner(BaseRunner):
    """Extended MiniSWEAgentRunner that supports graph tools with repo_id."""

    def create_task_instance(self, target_file: str, readme: str, repo_id: str = "0") -> str:
        """
        Create task instance by rendering the YAML template with repo_id.

        Args:
            target_file: Target file to implement
            readme: README content
            repo_id: Repository ID for graph tools

        Returns:
            Rendered task prompt
        """
        # Load config
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Get instance template
        instance_template = config['agent']['instance_template']

        # Render with Jinja2, including repo_id
        from jinja2 import Template
        template = Template(instance_template)
        rendered = template.render(target_file=target_file, readme=readme, repo_id=repo_id)

        return rendered

    def run(
        self,
        target_file: str,
        readme: str,
        container_name: str,  # This is actually instance_name for Apptainer
        repo_id: str = "0"
    ) -> Dict[str, Any]:
        """
        Run mini-swe-agent on a task with graph tool support.

        Args:
            target_file: Target file to implement
            readme: README content
            container_name: Apptainer instance name (from apptainer_manager)
            repo_id: Repository ID for graph tools

        Returns:
            Result dictionary
        """
        try:
            # Import mini-swe-agent components
            from minisweagent.agents.default import DefaultAgent
            from minisweagent.models.litellm_model import LitellmModel
            import os
            import subprocess

            # Import our custom Apptainer environment
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from apptainer import ApptainerEnvironment

            print(f"\nInitializing graph-based mini-swe-agent for {target_file}...")
            print(f"Repository ID: {repo_id}")

            # Load config
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Override step limit
            config['agent']['step_limit'] = self.max_turns

            # Set up environment - use config from YAML
            env_config = config.get('environment', {})
            env = ApptainerEnvironment(
                image="python:3.12",
                cwd=env_config.get('cwd', '/workspace'),
                env=env_config.get('env', {}),
                timeout=env_config.get('timeout', 60)
            )

            # Copy files from apptainer_manager's instance to mini-swe-agent's instance
            print(f"Copying project files from {container_name} to mini-swe-agent instance...")
            source_instance = container_name
            target_instance = env.instance_name

            # Use tar to copy files between instances (includes /workspace/tools with graph files)
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

            # Install networkx for graph tools
            print("Installing networkx for graph tools...")
            result = subprocess.run(
                ["apptainer", "exec", f"instance://{target_instance}",
                 "pip", "install", "-q", "networkx"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"Warning: networkx installation had issues:\n{result.stderr}")
            else:
                print(f"  ✓ networkx installed")

            # Set up model configuration
            if self.use_custom_endpoint:
                # For custom endpoints
                model_name_for_litellm = f"openai/{self.model_name}"

                # Set environment variables for custom endpoint
                if self.api_base_url:
                    os.environ['OPENAI_API_BASE'] = self.api_base_url
                    os.environ['OPENAI_BASE_URL'] = self.api_base_url

                os.environ['OPENAI_API_KEY'] = self.api_key or "dummy-key"
            else:
                # Standard OpenAI models
                os.environ['OPENAI_API_KEY'] = self.api_key
                model_name_for_litellm = self.model_name

            # Get model_kwargs from config
            model_kwargs_for_completion = config.get('model', {}).get('model_kwargs', {})

            # Handle custom endpoint configuration
            if self.use_custom_endpoint:
                if self.api_base_url:
                    model_kwargs_for_completion['api_base'] = self.api_base_url

                if 'max_completion_tokens' in model_kwargs_for_completion:
                    model_kwargs_for_completion['max_tokens'] = model_kwargs_for_completion.pop('max_completion_tokens')

                model_kwargs_for_completion['drop_params'] = True

                print(f"Using custom endpoint with model_kwargs: {model_kwargs_for_completion}")

            base_model = LitellmModel(
                model_name=model_name_for_litellm,
                model_kwargs=model_kwargs_for_completion
            )

            # Wrap model with token tracking
            from utils.agent.min_swe_open_models.agent_wrapper import TokenTrackingModel
            model = TokenTrackingModel(base_model, self.token_tracker)

            # Import the exception class we need
            from minisweagent.agents.default import TerminatingException

            # Create custom exception for early stopping
            class EarlyStopEmptyResponses(TerminatingException):
                """Raised when agent produces 3 consecutive empty responses."""
                pass

            # Create agent with config from YAML
            agent_config = config.get('agent', {})
            agent = DefaultAgent(model, env, **agent_config)

            # Monkey-patch the agent's query method
            consecutive_empty_count = [0]

            def query_with_fixes():
                if 0 < agent.config.step_limit <= agent.model.n_calls or 0 < agent.config.cost_limit <= agent.model.cost:
                    from minisweagent.agents.default import LimitsExceeded
                    raise LimitsExceeded()

                response = agent.model.query(agent.messages)

                content = response.get('content', '')
                extra = response.get('extra', {})
                agent.add_message("assistant", content, **extra)

                if not content.strip():
                    consecutive_empty_count[0] += 1
                    print(f"⚠️  Empty response detected ({consecutive_empty_count[0]}/3)")

                    if consecutive_empty_count[0] >= 3:
                        print("\n⚠️  3 consecutive empty responses - stopping early!")
                        raise EarlyStopEmptyResponses('Stopped due to 3 consecutive empty responses')
                else:
                    consecutive_empty_count[0] = 0

                return response

            agent.query = query_with_fixes

            print(f"Starting agent execution (max {self.max_turns} turns)...\n")

            # Run agent - pass template variables including repo_id
            exit_status, message = agent.run(
                task="",
                target_file=target_file,
                readme=readme,
                repo_id=repo_id  # Pass repo_id for graph tools
            )

            # Get turn count from agent history
            turns = max(0, (len(agent.messages) - 2) // 2)

            # Read the implemented file from the instance
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

            # Extract trajectory
            trajectory = []
            for msg in agent.messages:
                trajectory.append({
                    'role': msg.get('role', ''),
                    'content': msg.get('content', '')
                })

            # Check completion indicators
            is_completed = (
                'CompletedTask' in exit_status or
                'completed' in exit_status.lower() or
                'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' in message or
                turns < self.max_turns
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
