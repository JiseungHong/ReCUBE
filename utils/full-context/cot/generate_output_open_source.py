#!/usr/bin/env python3
"""
Inference script for LLM code reconstruction task with Chain-of-Thought (Target files only).

This script generates LLM outputs by:
1. Reading target.json to determine which files to process
2. Replacing function bodies with NotImplementedError
3. Removing all imports
4. Prompting the LLM with CoT to reconstruct the file based on context
5. Saving both the reconstructed outputs and thoughts separately

Only processes files specified in target.json.
"""

import argparse
import ast
import json
import os
import re
import time
from pathlib import Path
from typing import Dict
import openai


def replace_function_bodies(code: str) -> str:
    """
    Replace all function bodies with 'raise NotImplementedError'.
    Keep docstrings and function signatures.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    class FunctionBodyReplacer(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            # Keep the original decorators, name, args, and returns
            # Replace body with just docstring (if exists) + NotImplementedError
            new_body = []

            # Keep docstring if it exists
            if (node.body and
                isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str)):
                new_body.append(node.body[0])

            # Add raise NotImplementedError
            new_body.append(
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id='NotImplementedError', ctx=ast.Load()),
                        args=[],
                        keywords=[]
                    ),
                    cause=None
                )
            )

            node.body = new_body
            # Continue visiting nested functions/classes
            self.generic_visit(node)
            return node

        def visit_AsyncFunctionDef(self, node):
            return self.visit_FunctionDef(node)

    replacer = FunctionBodyReplacer()
    new_tree = replacer.visit(tree)
    ast.fix_missing_locations(new_tree)

    return ast.unparse(new_tree)


def remove_imports(code: str) -> str:
    """Remove all import statements from the code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    class ImportRemover(ast.NodeTransformer):
        def visit_Import(self, node):
            return None

        def visit_ImportFrom(self, node):
            return None

    remover = ImportRemover()
    new_tree = remover.visit(tree)
    ast.fix_missing_locations(new_tree)

    return ast.unparse(new_tree)


def modify_file_in_instruction(instruction_content: str, target_file: str) -> str:
    """
    Modify a specific file in the instruction by:
    1. Removing all imports from that file
    2. Replacing all function bodies with NotImplementedError

    Args:
        instruction_content: Full content of the instruction file
        target_file: File path to modify (e.g., "app/bedrock.py")

    Returns:
        Modified instruction content
    """
    # Find the implementations section
    impl_match = re.search(r'<implementations>(.*)</implementations>', instruction_content, re.DOTALL)
    if not impl_match:
        raise ValueError("Could not find <implementations> section")

    impl_section = impl_match.group(1)

    # Pattern to match the target file section
    # Match: ## app/path/file.py followed by ```python code ```
    file_pattern = re.escape(f"## {target_file}") + r'\s*\n\s*```python\s*\n(.*?)```'

    file_match = re.search(file_pattern, impl_section, re.DOTALL)
    if not file_match:
        raise ValueError(f"Could not find file {target_file} in implementations")

    original_code = file_match.group(1)

    # Step 1: Remove imports
    code_no_imports = remove_imports(original_code)

    # Step 2: Replace function bodies
    modified_code = replace_function_bodies(code_no_imports)

    # Replace in the original content
    new_file_section = f"## {target_file}\n\n```python\n{modified_code}\n```"
    old_file_section = file_match.group(0)

    new_impl_section = impl_section.replace(old_file_section, new_file_section)
    new_instruction = instruction_content.replace(impl_match.group(0), f"<implementations>{new_impl_section}</implementations>")

    return new_instruction


def create_prompt(modified_instruction: str, target_file: str) -> str:
    """
    Create the prompt for the LLM to reconstruct the file using Chain-of-Thought.

    Args:
        modified_instruction: Instruction content with modified target file
        target_file: The file that needs to be reconstructed

    Returns:
        Complete prompt for LLM with CoT format
    """
    task_guideline = f"""Your task is to reconstruct the complete implementation of the file `{target_file}`.

The file has been provided with:
- All function signatures and docstrings intact
- All function bodies replaced with `raise NotImplementedError`
- All import statements removed

You need to:
1. Add all necessary import statements at the beginning of the file
2. Implement the complete function bodies based on:
   - The function signatures and docstrings
   - The README documentation
   - The dependencies and their descriptions
   - All other files in the codebase that are fully implemented
   - The overall project structure and design patterns

INSTRUCTIONS:
- Keep your THOUGHT section concise but informative - focus on the most relevant details that directly impact implementation decisions
- In <readme></readme>: Extract 2-3 key points about project purpose and architecture
- In <dependencies></dependencies>: List only dependencies directly needed for `{target_file}`
- In <implementations></implementations>: Mention only the files/classes/functions that `{target_file}` directly uses or interacts with
- In <file_context></file_context>: Briefly describe `{target_file}`'s role and main implementation approach
- The FINAL_OUTPUT code block must contain the complete, runnable Python file with all imports and implementations

IMPORTANT OUTPUT FORMAT:
<format_example>
THOUGHT:
<readme>Agent framework for autonomous task execution. Architecture: modular agent system with pluggable tools, sandbox execution environment for safety, LLM-based reasoning with ReAct pattern.</readme>

<dependencies>openai for LLM API calls, docker for sandbox isolation, pydantic for schema validation in app/schema.py.</dependencies>

<implementations>Inherits from BaseAgent in app/agent/base.py which provides core agent loop. Uses Tool base class from app/tool/base.py for tool registry. Interacts with Sandbox from app/sandbox/core/sandbox.py for code execution. Imports Message and AgentConfig schemas from app/schema.py.</implementations>

<file_context>app/agent/react.py implements ReAct agent - combines reasoning and acting in iterative loops. Needs to: parse LLM responses for thought/action/observation pattern, manage tool execution through base class methods, handle error cases when actions fail, format conversation history for context window.</file_context>

FINAL_OUTPUT:
```python
# Complete reconstructed file for app/agent/react.py
# Include ALL necessary imports
# Include ALL complete function implementations (no pass statements or placeholder comments)

from typing import List, Optional, Dict, Any
import re
from app.agent.base import BaseAgent
from app.schema import Message, AgentConfig
from app.tool.base import Tool
from app.llm import LLMClient

class ReactAgent(BaseAgent):
    # ReAct agent that combines reasoning and acting

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.llm_client = LLMClient(config.model_name)
        self.max_iterations = config.max_iterations or 10

    def execute(self, task: str) -> str:
        # Execute task using ReAct pattern with thought/action/observation loop
        history = []
        for i in range(self.max_iterations):
            response = self.llm_client.chat(self._format_prompt(task, history))
            thought, action, action_input = self._parse_response(response)
            history.append({{"thought": thought, "action": action}})

            if action == "finish":
                return action_input

            observation = self._execute_tool(action, action_input)
            history.append({{"observation": observation}})

        return "Max iterations reached"

    def _parse_response(self, response: str) -> tuple[str, str, str]:
        # Parse LLM response into thought, action, and input using regex
        thought_match = re.search(r"Thought: (.*?)\\n", response)
        action_match = re.search(r"Action: (.*?)\\n", response)
        input_match = re.search(r"Action Input: (.*?)\\n", response)

        thought = thought_match.group(1) if thought_match else ""
        action = action_match.group(1) if action_match else ""
        action_input = input_match.group(1) if input_match else ""

        return thought, action, action_input

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        # Execute tool and return observation with error handling
        tool = self.tool_registry.get(tool_name)
        if not tool:
            return f"Error: Tool {{tool_name}} not found"

        try:
            return tool.run(tool_input)
        except Exception as e:
            return f"Error executing tool: {{str(e)}}"

    def _format_prompt(self, task: str, history: List[Dict]) -> str:
        # Format the prompt with task and conversation history
        prompt = f"Task: {{task}}\\n\\n"
        for entry in history:
            if "thought" in entry:
                prompt += f"Thought: {{entry['thought']}}\\n"
                prompt += f"Action: {{entry['action']}}\\n"
            if "observation" in entry:
                prompt += f"Observation: {{entry['observation']}}\\n"
        return prompt
```
</format_example>

Here is the project context:

"""

    return task_guideline + modified_instruction


def extract_code_and_thought_from_response(response: str) -> tuple[str, str, bool]:
    """
    Extract the Python code and thought process from LLM response.
    Uses THOUGHT and FINAL_OUTPUT markers, but is generous in parsing.

    Returns:
        tuple[str, str, bool]: (extracted_code, thought, has_correct_format)
    """
    # Handle None response
    if response is None:
        return "", "", False

    thought = ""
    code = ""
    has_correct_format = False

    # Try to find THOUGHT section
    thought_match = re.search(r'THOUGHT:\s*(.*?)(?=FINAL_OUTPUT:|$)', response, re.DOTALL | re.IGNORECASE)
    if thought_match:
        thought = thought_match.group(1).strip()
        has_correct_format = True

    # Try to find FINAL_OUTPUT section with code block
    final_output_match = re.search(r'FINAL_OUTPUT:\s*```python\s*\n(.*?)```', response, re.DOTALL | re.IGNORECASE)
    if final_output_match:
        code = final_output_match.group(1)
        has_correct_format = has_correct_format and True
    else:
        # Be generous - look for any python code block
        code_match = re.search(r'```python\s*\n(.*?)```', response, re.DOTALL)
        if code_match:
            code = code_match.group(1)
            # If we found a code block but no FINAL_OUTPUT marker, consider the rest as thought
            if not thought:
                # Everything before the code block is thought
                code_start = response.find('```python')
                if code_start > 0:
                    thought = response[:code_start].strip()
                    # Remove THOUGHT: marker if present
                    thought = re.sub(r'^THOUGHT:\s*', '', thought, flags=re.IGNORECASE).strip()
        else:
            # No code block found at all - try to extract any code-like content
            cleaned_response = response.strip()
            if cleaned_response.startswith('```python'):
                cleaned_response = cleaned_response[9:].lstrip('\n')
            elif cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:].lstrip('\n')
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3].rstrip()
            code = cleaned_response

    return code, thought, has_correct_format


class CostTracker:
    """Track LLM API costs."""

    # Model pricing (per million tokens)
    PRICING = {
        'o1-mini-2024-09-12': {
            'input': 1.21,
            'output': 4.84
        },
        'gpt-4o-mini': {
            'input': 0.15,
            'output': 0.60
        },
        'gpt-4o': {
            'input': 2.50,
            'output': 10.00
        },
        'gpt-5-mini': {
            'input': 0.28,
            'output': 2.20
        },
        'claude-3-haiku-20240307': {
            'input': 0.25,
            'output': 1.25
        },
        'claude-sonnet-4-20250514-v1:0': {
            'input': 3.00,
            'output': 15.00
        },
        'gpt-5': {
            'input': 1.25,
            'output': 10.00
        },
        'gemini-1.5-pro-002': {
            'input': 3.50,
            'output': 10.50
        },
        'gemini-2.5-pro': {
            'input': 1.25,
            'output': 10.00
        },
        'gemini-2.5-flash': {
            'input': 0.30,
            'output': 2.50
        },
        # Open-source models (free when self-hosted)
        'mistralai/Devstral-Small-2507': {
            'input': 0.0,
            'output': 0.0
        },
        'Qwen/Qwen3-Coder-30B-A3B-Instruct': {
            'input': 0.0,
            'output': 0.0
        }
    }

    def __init__(self, model: str = 'gpt-5-mini'):
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Set pricing based on model
        if model in self.PRICING:
            self.pricing = self.PRICING[model]
        else:
            # Default to gpt-5-mini pricing
            self.pricing = self.PRICING['gpt-5-mini']

    def add_usage(self, usage_dict: dict):
        """Add token usage from API response."""
        self.total_input_tokens += usage_dict.get('prompt_tokens', 0)
        self.total_output_tokens += usage_dict.get('completion_tokens', 0)

    def get_cost(self) -> float:
        """Calculate total cost in USD."""
        input_cost = (self.total_input_tokens / 1_000_000) * self.pricing['input']
        output_cost = (self.total_output_tokens / 1_000_000) * self.pricing['output']
        return input_cost + output_cost

    def print_summary(self):
        """Print cost summary."""
        print("\n" + "="*60)
        print(f"LLM API COST SUMMARY ({self.model})")
        print("="*60)
        print(f"Input tokens:       {self.total_input_tokens:,}")
        print(f"Output tokens:      {self.total_output_tokens:,}")
        print(f"Total tokens:       {self.total_input_tokens + self.total_output_tokens:,}")
        print("-"*60)
        print(f"Input cost:         ${(self.total_input_tokens / 1_000_000) * self.pricing['input']:.4f}")
        print(f"Output cost:        ${(self.total_output_tokens / 1_000_000) * self.pricing['output']:.4f}")
        print(f"TOTAL COST:         ${self.get_cost():.4f}")
        print("="*60 + "\n")

    def save_summary(self, output_file: Path):
        """Save cost summary to JSON file."""
        summary = {
            'model': self.model,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'input_cost': (self.total_input_tokens / 1_000_000) * self.pricing['input'],
            'output_cost': (self.total_output_tokens / 1_000_000) * self.pricing['output'],
            'total_cost': self.get_cost()
        }
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)


def run_llm_inference(
    prompt: str,
    api_key: str,
    model: str,
    cost_tracker: CostTracker,
    max_retries: int = 3,
    api_base_url: str = None,
    use_custom_endpoint: bool = False
) -> str:
    """
    Run LLM inference with the given prompt, with retry logic.

    Args:
        prompt: The prompt to send to the LLM
        api_key: OpenAI API key (or dummy key for custom endpoints)
        model: Model name
        cost_tracker: Cost tracker instance
        max_retries: Maximum number of retries on failure
        api_base_url: Custom API base URL (for vLLM/local models)
        use_custom_endpoint: Whether to use custom endpoint

    Returns:
        LLM response text
    """
    # Use custom endpoint or OpenAI's default API endpoint
    if use_custom_endpoint and api_base_url:
        client = openai.OpenAI(api_key=api_key, base_url=api_base_url)
    else:
        client = openai.OpenAI(api_key=api_key)

    for attempt in range(max_retries):
        try:
            # Prepare request parameters
            request_params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "timeout": 300.0  # 5 minute timeout
            }

            # Add max_tokens for custom endpoints (vLLM)
            if use_custom_endpoint:
                request_params["max_tokens"] = 16384

            response = client.chat.completions.create(**request_params)

            # Track cost
            if response.usage:
                cost_tracker.add_usage(response.usage.model_dump())

            return response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)

            # Check if it's a WAF/403 error
            if "403" in error_msg or "Forbidden" in error_msg or "blocked" in error_msg.lower():
                print(f"  WAF blocking detected (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5  # 5, 10, 20 seconds
                    print(f"  Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"WAF blocking persists after {max_retries} attempts: {error_msg}")
            else:
                # For other errors, raise immediately
                raise

    raise Exception(f"Failed after {max_retries} attempts")


def main():
    """Main inference function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run LLM inference for code reconstruction with CoT (target files only)')
    parser.add_argument('--model', type=str, default='claude-3-haiku-20240307',
                        help='Model name to use for inference (default: claude-3-haiku-20240307)')
    parser.add_argument('--repo-ids', type=str, default=None,
                        help='Comma-separated list of repo IDs to process (default: all)')
    parser.add_argument('--mode', type=str, default='original', choices=['original', 'large', 'all'],
                        help='Dataset mode: original (repo_id < 55, default), large (repo_id >= 55), or all')
    parser.add_argument('--api-base-url', type=str, default=None,
                        help='Custom API base URL for OpenAI-compatible endpoints (e.g., http://localhost:8000/v1)')
    parser.add_argument('--use-custom-endpoint', action='store_true',
                        help='Use custom OpenAI-compatible endpoint instead of OpenAI API')
    args = parser.parse_args()

    # Configuration - using relative paths from project root
    INSTRUCTIONS_DIR = Path("data/prompts")
    TARGET_FILE = Path("data/target_verified.json")

    # Model configuration
    MODEL = args.model

    # Output directory structure: outputs/full-context_cot/{model}/{repo_id}/
    OUTPUTS_BASE_DIR = Path("outputs/full-context_cot") / MODEL

    # Handle API key - optional for custom endpoints
    if args.use_custom_endpoint:
        # For custom endpoints, API key might be optional
        api_key = os.environ.get("OPENAI_API_KEY", "dummy-key")
        print(f"Using custom endpoint: {args.api_base_url}")
    else:
        # For OpenAI models, API key is required
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

    # Load target data
    with open(TARGET_FILE, 'r') as f:
        target_data = json.load(f)

    # Filter by mode
    if args.mode == 'original':
        target_data = [entry for entry in target_data if entry['repo_id'] < 55]
    elif args.mode == 'large':
        target_data = [entry for entry in target_data if entry['repo_id'] >= 55]
    # else: 'all' - use all repos

    # Filter by repo IDs if specified (after mode filtering)
    if args.repo_ids:
        repo_ids = [int(rid.strip()) for rid in args.repo_ids.split(',')]
        target_data = [entry for entry in target_data if entry['repo_id'] in repo_ids]
        print(f"Processing {len(target_data)} repositories (mode: {args.mode}): {repo_ids}\n")
    else:
        print(f"Processing {len(target_data)} repositories (mode: {args.mode})\n")

    # Process each target entry
    for target_entry in target_data:
        repo_id = str(target_entry['repo_id'])
        selected_files = target_entry['selected_files']

        # Get instruction file
        instr_file = INSTRUCTIONS_DIR / f"{repo_id}.txt"
        if not instr_file.exists():
            print(f"Instruction file not found for repo {repo_id}: {instr_file}")
            continue

        # Create output directory for this instruction: outputs/Code_cot/{model}/{repo_id}/
        OUTPUTS_DIR = OUTPUTS_BASE_DIR / repo_id
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Running CoT inference for repo {repo_id}")
        print(f"Target files: {len(selected_files)}")
        print(f"{'='*80}\n")

        # Load instruction file
        with open(instr_file, 'r') as f:
            instruction_content = f.read()

        # Initialize cost tracker
        cost_tracker = CostTracker(MODEL)

        # Load or initialize inference status
        inference_status_file = OUTPUTS_DIR / "inference_status.json"
        if inference_status_file.exists():
            with open(inference_status_file, 'r') as f:
                inference_status = json.load(f)
            print(f"Loading existing inference status...")
            print(f"  Already completed: {len(inference_status['completed_files'])}")
            print(f"  Previously failed: {len(inference_status['failed_files'])}")
        else:
            # Initialize new inference status
            inference_status = {
                'instruction_id': repo_id,
                'total_files': len(selected_files),
                'completed_files': [],
                'failed_files': [],
                'status': 'in_progress',
                'format_stats': {
                    'correct_format': 0,
                    'incorrect_format': 0,
                    'incorrect_format_files': []
                }
            }

        # Get set of completed files for fast lookup
        completed_files_set = set(inference_status['completed_files'])

        # Get list of failed files to retry
        failed_files_to_retry = [f['file'] for f in inference_status['failed_files']]

        # Clear failed_files list to re-populate with any new failures
        inference_status['failed_files'] = []

        # Load or initialize thoughts tracking
        thoughts_file = OUTPUTS_DIR / "thoughts.json"
        if thoughts_file.exists():
            with open(thoughts_file, 'r') as f:
                thoughts_data = json.load(f)
        else:
            thoughts_data = {}

        # Process each target file
        for i, target_file in enumerate(selected_files, 1):
            output_file = OUTPUTS_DIR / target_file.replace('/', '_')

            # Skip if already completed (check both output file and status)
            if target_file in completed_files_set and output_file.exists():
                print(f"[{i}/{len(selected_files)}] {target_file} - already completed, skipping")
                continue

            # If file was previously failed, indicate retry
            if target_file in failed_files_to_retry:
                print(f"\n[{i}/{len(selected_files)}] Processing {target_file} (retry)...")
            else:
                print(f"\n[{i}/{len(selected_files)}] Processing {target_file}...")

            # Modify instruction to remove imports and function bodies from target file
            try:
                modified_instruction = modify_file_in_instruction(instruction_content, target_file)
            except Exception as e:
                print(f"Error modifying instruction for {target_file}: {e}")
                inference_status['failed_files'].append({'file': target_file, 'error': str(e)})
                continue

            # Create prompt
            prompt = create_prompt(modified_instruction, target_file)

            # Run LLM inference with retry on None response
            llm_response = None
            for retry in range(3):
                try:
                    llm_response = run_llm_inference(
                        prompt, api_key, MODEL, cost_tracker,
                        api_base_url=args.api_base_url,
                        use_custom_endpoint=args.use_custom_endpoint
                    )
                    if llm_response is not None:
                        break
                    print(f"  Got None response (attempt {retry + 1}/3), retrying...")
                    time.sleep(2)
                except Exception as e:
                    print(f"  Error during LLM inference (attempt {retry + 1}/3): {e}")
                    if retry == 2:  # Last attempt
                        inference_status['failed_files'].append({'file': target_file, 'error': str(e)})
                        break
                    time.sleep(2)

            if llm_response is None:
                print(f"Error: Got None response after 3 attempts for {target_file}")
                if not any(f['file'] == target_file for f in inference_status['failed_files']):
                    inference_status['failed_files'].append({'file': target_file, 'error': 'Got None response after 3 attempts'})
                continue

            # Extract code and thought from response
            reconstructed_code, thought, has_correct_format = extract_code_and_thought_from_response(llm_response)

            # Track format statistics
            if has_correct_format:
                inference_status['format_stats']['correct_format'] += 1
            else:
                inference_status['format_stats']['incorrect_format'] += 1
                inference_status['format_stats']['incorrect_format_files'].append(target_file)

            # Save output code
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(reconstructed_code)

            # Save thought
            thoughts_data[target_file] = thought

            # Save thoughts immediately
            with open(thoughts_file, 'w') as f:
                json.dump(thoughts_data, f, indent=2)

            inference_status['completed_files'].append(target_file)
            format_indicator = "" if has_correct_format else "�"
            print(f"  {format_indicator} Saved code to {output_file.name}")
            print(f"  {format_indicator} Saved thought to thoughts.json")

            # Save intermediate status
            with open(inference_status_file, 'w') as f:
                json.dump(inference_status, f, indent=2)

            # Add delay between requests to avoid rate limiting
            if i < len(selected_files):
                time.sleep(2)  # 2 second delay between files

        # Mark as completed only if there are no failed files
        if len(inference_status['failed_files']) == 0:
            inference_status['status'] = 'completed'
        else:
            inference_status['status'] = 'partial'  # Some files failed
        with open(inference_status_file, 'w') as f:
            json.dump(inference_status, f, indent=2)

        # Save cost summary
        cost_summary_file = OUTPUTS_DIR / "cost_summary.json"
        cost_tracker.save_summary(cost_summary_file)

        # Print cost summary
        cost_tracker.print_summary()

        print(f"\n CoT Inference completed for repo {repo_id}")
        print(f"  Completed files: {len(inference_status['completed_files'])}/{len(selected_files)}")
        print(f"  Failed files: {len(inference_status['failed_files'])}")
        print(f"  Format statistics:")
        print(f"    - Correct format (THOUGHT/FINAL_OUTPUT): {inference_status['format_stats']['correct_format']}")
        print(f"    - Incorrect format: {inference_status['format_stats']['incorrect_format']}")
        correct_pct = 100 * inference_status['format_stats']['correct_format'] / len(inference_status['completed_files']) if inference_status['completed_files'] else 0
        print(f"    - Correct format rate: {correct_pct:.1f}%")
        print(f"  Output directory: {OUTPUTS_DIR}")
        print(f"  Cost summary saved to: {cost_summary_file}")


if __name__ == "__main__":
    main()
