#!/usr/bin/env python3
"""
Wrapper for real mini-swe-agent library with token tracking and custom configuration.
"""

import json
import os
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import openai


class TokenTrackingModel:
    """
    Wrapper around mini-swe-agent LitellmModel to track token usage.
    """

    def __init__(self, base_model, token_tracker):
        """
        Initialize token tracking model.

        Args:
            base_model: The underlying LitellmModel instance
            token_tracker: TokenTracker instance
        """
        self.base_model = base_model
        self.token_tracker = token_tracker
        self.config = base_model.config

    def query(self, messages):
        """Intercept query to track token usage."""
        import json
        import tiktoken

        # Call the base model's query
        try:
            response = self.base_model.query(messages)
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR: Exception during LLM API call")
            print(f"{'='*80}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"{'='*80}\n")
            raise

        # DEBUG: Print the full response structure (COMMENTED OUT - too verbose)
        # print(f"\n{'='*80}")
        # print(f"DEBUG: LLM Response Structure")
        # print(f"{'='*80}")
        # print(f"Response type: {type(response)}")

        # # Print response content for debugging
        # if isinstance(response, dict):
        #     print(f"Response keys: {response.keys()}")
        #     print(f"Full response (first 500 chars): {json.dumps(response, indent=2, default=str)[:500]}")
        #     if 'usage' in response:
        #         print(f"Usage info: {json.dumps(response['usage'], indent=2)}")
        #     if 'choices' in response and len(response['choices']) > 0:
        #         choice = response['choices'][0]
        #         message = choice.get('message', {})
        #         content = message.get('content', '')
        #         print(f"Message content length: {len(content)} chars")
        #         print(f"Message content (first 200 chars): {content[:200]}")
        #         if not content:
        #             print("WARNING: Empty content in response!")
        # else:
        #     print(f"Response attributes: {dir(response)}")
        #     if hasattr(response, 'usage'):
        #         print(f"Usage: {response.usage}")
        #     if hasattr(response, 'choices'):
        #         print(f"Choices: {response.choices}")
        # print(f"{'='*80}\n")

        # Extract token usage from response
        input_tokens = 0
        output_tokens = 0

        enc = tiktoken.get_encoding("cl100k_base")

        # Extract from dict response
        if isinstance(response, dict):
            # Check if usage is nested in extra.response (mini-swe-agent format)
            usage = None
            if 'extra' in response and 'response' in response['extra']:
                nested_response = response['extra']['response']
                if 'usage' in nested_response:
                    usage = nested_response['usage']
            # Otherwise check top-level
            elif 'usage' in response:
                usage = response['usage']

            if usage:
                # Get tokens, trying all possible key names
                input_tokens = usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0) or usage.get('output_tokens', 0) or usage.get('generated_tokens', 0)

            # Always try to get actual content for output token estimation
            # Check both top-level and nested locations
            response_for_content = None
            if 'choices' in response and len(response['choices']) > 0:
                response_for_content = response
            elif 'extra' in response and 'response' in response['extra']:
                nested_response = response['extra']['response']
                if 'choices' in nested_response and len(nested_response['choices']) > 0:
                    # Use nested response for content extraction (but don't modify original response)
                    response_for_content = nested_response

            if response_for_content and 'choices' in response_for_content and len(response_for_content['choices']) > 0:
                try:
                    content = response_for_content['choices'][0].get('message', {}).get('content', '')
                    if content:
                        # Calculate actual output tokens from content
                        actual_output_tokens = len(enc.encode(content))

                        # If API didn't provide output tokens, or provided 0, use our calculation
                        if not output_tokens:
                            output_tokens = actual_output_tokens
                    else:
                        print(f"WARNING: Empty content in API response!")
                        if output_tokens:
                            print(f"WARNING: API reported {output_tokens} output tokens but content is empty!")
                except Exception as e:
                    print(f"ERROR: Failed to extract content: {e}")

        elif hasattr(response, 'usage'):
            # Some APIs return object with .usage attribute
            usage = response.usage
            input_tokens = getattr(usage, 'prompt_tokens', 0) or getattr(usage, 'input_tokens', 0)
            output_tokens = getattr(usage, 'completion_tokens', 0) or getattr(usage, 'output_tokens', 0)

            # Try to get content
            if hasattr(response, 'choices') and len(response.choices) > 0:
                content = response.choices[0].message.content or ""
                if content:
                    actual_output_tokens = len(enc.encode(content))
                    if not output_tokens:
                        output_tokens = actual_output_tokens

        # Fallback: estimate tokens if not provided by API
        if not input_tokens:
            try:
                # Estimate input tokens from messages
                input_text = " ".join([str(m.get('content', '')) if isinstance(m, dict) else str(getattr(m, 'content', '')) for m in messages])
                input_tokens = len(enc.encode(input_text))
                print(f"INFO: Estimated input tokens: {input_tokens}")
            except Exception as e:
                print(f"ERROR: Failed to estimate input tokens: {e}")
                # Rough estimate: 1 token ~ 4 chars
                input_text = " ".join([str(m.get('content', '')) if isinstance(m, dict) else str(getattr(m, 'content', '')) for m in messages])
                input_tokens = len(input_text) // 4

        # Final summary
        print(f"\nToken Summary for this turn:")
        print(f"  Input tokens:  {input_tokens:,}")
        print(f"  Output tokens: {output_tokens:,}")
        print(f"  Total tokens:  {input_tokens + output_tokens:,}\n")

        if input_tokens or output_tokens:
            self.token_tracker.add_turn(input_tokens, output_tokens)

        return response

    def __getattr__(self, name):
        """Forward all other attribute access to base model."""
        return getattr(self.base_model, name)


class TokenTracker:
    """Track token usage and costs."""

    # Model pricing (per million tokens)
    PRICING = {
        'o1-mini-2024-09-12': {'input': 1.21, 'output': 4.84},
        'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
        'gpt-4o': {'input': 2.50, 'output': 10.00},
        'gpt-5-mini': {'input': 0.28, 'output': 2.20},
        'claude-3-haiku-20240307': {'input': 0.25, 'output': 1.25},
        'claude-sonnet-4-20250514-v1:0': {'input': 3.00, 'output': 15.00},
        'gpt-5': {'input': 1.25, 'output': 10.00},
        'gemini-1.5-pro-002': {'input': 3.50, 'output': 10.50},
        'gemini-2.5-pro': {'input': 1.25, 'output': 10.00},
        'gemini-2.5-flash': {'input': 0.30, 'output': 2.50}
    }

    def __init__(self, model: str):
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_turns = 0

        if model in self.PRICING:
            self.pricing = self.PRICING[model]
        else:
            print(f"Warning: Unknown model {model}, using gpt-5-mini pricing")
            self.pricing = self.PRICING['gpt-5-mini']

    def add_turn(self, input_tokens: int, output_tokens: int):
        """Add token usage for a turn."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_turns += 1

    def get_cost(self) -> float:
        """Calculate total cost in USD."""
        input_cost = (self.total_input_tokens / 1_000_000) * self.pricing['input']
        output_cost = (self.total_output_tokens / 1_000_000) * self.pricing['output']
        return input_cost + output_cost

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of token usage and costs."""
        return {
            'model': self.model,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'total_turns': self.total_turns,
            'input_cost': (self.total_input_tokens / 1_000_000) * self.pricing['input'],
            'output_cost': (self.total_output_tokens / 1_000_000) * self.pricing['output'],
            'total_cost': self.get_cost()
        }

    def print_summary(self):
        """Print cost summary."""
        print("\n" + "="*60)
        print(f"TOKEN USAGE SUMMARY ({self.model})")
        print("="*60)
        print(f"Total turns:        {self.total_turns}")
        print(f"Input tokens:       {self.total_input_tokens:,}")
        print(f"Output tokens:      {self.total_output_tokens:,}")
        print(f"Total tokens:       {self.total_input_tokens + self.total_output_tokens:,}")
        print("-"*60)
        print(f"Input cost:         ${(self.total_input_tokens / 1_000_000) * self.pricing['input']:.4f}")
        print(f"Output cost:        ${(self.total_output_tokens / 1_000_000) * self.pricing['output']:.4f}")
        print(f"TOTAL COST:         ${self.get_cost():.4f}")
        print("="*60 + "\n")


class MiniSWEAgentRunner:
    """
    Runner for mini-swe-agent with custom configuration for code reconstruction.
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
        container_name: str
    ) -> Dict[str, Any]:
        """
        Run mini-swe-agent on a task.

        Args:
            target_file: Target file to implement
            readme: README content
            container_name: Docker container name (from docker_manager)

        Returns:
            Result dictionary
        """
        try:
            # Import mini-swe-agent components
            from minisweagent.agents.default import DefaultAgent
            from minisweagent.environments.docker import DockerEnvironment
            from minisweagent.models.litellm_model import LitellmModel
            import docker

            print(f"\nInitializing mini-swe-agent for {target_file}...")

            # Load config
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Override step limit
            config['agent']['step_limit'] = self.max_turns

            # Set up environment - use config from YAML
            env_config = config.get('environment', {})
            # Create DockerEnvironment with config from YAML
            env = DockerEnvironment(
                image="python:3.12",  # Let it create its own container
                cwd=env_config.get('cwd', '/workspace'),
                env=env_config.get('env', {}),
                timeout=env_config.get('timeout', 60)
            )

            # Copy files from docker_manager's container to mini-swe-agent's container
            print(f"Copying project files from {container_name} to mini-swe-agent container...")
            client = docker.from_env()
            source_container = client.containers.get(container_name)
            target_container = client.containers.get(env.container_id)

            # Copy entire workspace directory
            import tarfile
            import io
            bits, stat = source_container.get_archive('/workspace')
            target_container.put_archive('/', bits)
            print(f"✓ Files copied to container {env.container_id[:12]}")

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

            # Read the implemented file from the container BEFORE it's removed
            # Mini-swe-agent containers have --rm flag so they're auto-removed after agent.run()
            implemented_content = None
            try:
                import docker
                client = docker.from_env()
                container = client.containers.get(env.container_id)
                exit_code, output = container.exec_run(f"cat /workspace/{target_file}")
                if exit_code == 0:
                    implemented_content = output.decode('utf-8')
                else:
                    print(f"Error reading file from container: exit code {exit_code}")
            except Exception as e:
                print(f"Error reading file from container: {e}")

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

    def _parse_trajectory(self, result: Dict):
        """
        Parse trajectory to extract token usage.

        Args:
            result: Agent result dictionary
        """
        # Try to extract token info from result
        if 'trajectory' in result and 'steps' in result['trajectory']:
            for step in result['trajectory']['steps']:
                if 'usage' in step:
                    usage = step['usage']
                    self.token_tracker.add_turn(
                        usage.get('prompt_tokens', 0),
                        usage.get('completion_tokens', 0)
                    )
        elif 'usage' in result:
            # Fallback: use total usage
            usage = result['usage']
            self.token_tracker.add_turn(
                usage.get('prompt_tokens', 0),
                usage.get('completion_tokens', 0)
            )
