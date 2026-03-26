#!/usr/bin/env python3
"""
Wrapper for mini-swe-agent with graph tool support.

Extends the base agent_wrapper to pass repo_id to templates.
"""

import sys
from pathlib import Path
import yaml
from typing import Dict, Any

from utils.agent.min_swe.agent_wrapper import MiniSWEAgentRunner as BaseRunner, TokenTracker


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
        container_name: str,
        repo_id: str = "0"
    ) -> Dict[str, Any]:
        """
        Run mini-swe-agent on a task with graph tool support.

        Args:
            target_file: Target file to implement
            readme: README content
            container_name: Docker container name (from docker_manager)
            repo_id: Repository ID for graph tools

        Returns:
            Result dictionary
        """
        try:
            # Import mini-swe-agent components
            from minisweagent.agents.default import DefaultAgent
            from minisweagent.environments.docker import DockerEnvironment
            from minisweagent.models.litellm_model import LitellmModel
            import docker
            import os

            print(f"\nInitializing graph-based mini-swe-agent for {target_file}...")
            print(f"Repository ID: {repo_id}")

            # Load config
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Override step limit
            config['agent']['step_limit'] = self.max_turns

            # Set up environment - use config from YAML
            env_config = config.get('environment', {})
            env = DockerEnvironment(
                image="python:3.12",
                cwd=env_config.get('cwd', '/workspace'),
                env=env_config.get('env', {}),
                timeout=env_config.get('timeout', 60)
            )

            # Copy files from docker_manager's container to mini-swe-agent's container
            print(f"Copying project files from {container_name} to mini-swe-agent container...")
            client = docker.from_env()
            source_container = client.containers.get(container_name)
            target_container = client.containers.get(env.container_id)

            # Copy entire workspace directory (includes /workspace/tools with graph files)
            import tarfile
            import io
            bits, stat = source_container.get_archive('/workspace')
            target_container.put_archive('/', bits)
            print(f"✓ Files copied to container {env.container_id[:12]}")
            print(f"  (includes /workspace/tools with dependency graph and tools)")

            # Install dependencies in mini-swe-agent container
            # Check if requirements.txt exists
            exit_code, _ = target_container.exec_run("test -f /workspace/requirements.txt")
            if exit_code == 0:
                print("Installing project dependencies from requirements.txt...")
                exit_code, output = target_container.exec_run("pip install -q -r /workspace/requirements.txt")
                if exit_code != 0:
                    print(f"Warning: Dependency installation had issues:\n{output.decode()}")
                else:
                    print(f"  ✓ Dependencies installed")

            # Install networkx (required for graph tools)
            print("Installing networkx for graph tools...")
            exit_code, output = target_container.exec_run("pip install -q networkx")
            if exit_code != 0:
                print(f"Warning: networkx installation failed:\n{output.decode()}")
            else:
                print(f"  ✓ networkx installed")

            # Set up model with OpenAI API key
            os.environ['OPENAI_API_KEY'] = self.api_key

            # Get model_kwargs from config
            model_kwargs_for_completion = config.get('model', {}).get('model_kwargs', {})

            # Use OpenAI models directly
            model_name_for_litellm = self.model_name

            base_model = LitellmModel(
                model_name=model_name_for_litellm,
                model_kwargs=model_kwargs_for_completion
            )

            # Wrap model with token tracking
            from utils.agent.basic.agent_wrapper import TokenTrackingModel
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

            # Monkey-patch similar to base class
            consecutive_empty_count = [0]

            def query_with_fixes():
                from minisweagent.agents.default import LimitsExceeded
                if 0 < agent.config.step_limit <= agent.model.n_calls or 0 < agent.config.cost_limit <= agent.model.cost:
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

            # Create task instance with repo_id
            print(f"Creating task instance...")
            task_instance = self.create_task_instance(target_file, readme, repo_id)

            # Run the agent
            print(f"Running agent (max {self.max_turns} turns)...")
            print("="*80)

            try:
                agent.run(
                    task="",  # Not used in our template
                    target_file=target_file,
                    readme=task_instance,
                    repo_id=repo_id  # Pass repo_id for any template rendering
                )
                exit_status = "completed"
            except Exception as e:
                print(f"\nAgent stopped: {e}")
                exit_status = str(e)

            # Extract trajectory (full message history)
            trajectory = []
            for msg in agent.messages:
                trajectory.append({
                    'role': msg.get('role', ''),
                    'content': msg.get('content', '')
                })

            # Get implemented file from container
            implemented_content = None
            try:
                container = client.containers.get(env.container_id)
                exit_code, output = container.exec_run(f"cat /workspace/{target_file}")
                if exit_code == 0:
                    implemented_content = output.decode('utf-8')
                else:
                    print(f"Failed to read {target_file}: {output.decode('utf-8')}")
            except Exception as e:
                print(f"Error reading implemented file: {e}")

            # Get token usage
            token_usage = {
                'total_input_tokens': self.token_tracker.total_input_tokens,
                'total_output_tokens': self.token_tracker.total_output_tokens,
                'total_tokens': self.token_tracker.total_input_tokens + self.token_tracker.total_output_tokens,
                'total_cost': self.token_tracker.get_cost()
            }

            return {
                'status': 'success' if implemented_content else 'failed',
                'exit_status': exit_status,
                'turns': agent.model.n_calls,
                'trajectory': trajectory,
                'implemented_content': implemented_content,
                'token_usage': token_usage
            }

        except Exception as e:
            print(f"Error during agent execution: {e}")
            import traceback
            traceback.print_exc()

            return {
                'status': 'error',
                'error': str(e),
                'turns': 0,
                'trajectory': [],
                'implemented_content': None,
                'token_usage': {
                    'total_input_tokens': 0,
                    'total_output_tokens': 0,
                    'total_tokens': 0,
                    'total_cost': 0.0
                }
            }
