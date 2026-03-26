#!/usr/bin/env python3
"""
Graph-based Mini-SWE-Agent Inference for Code Reconstruction.

This script uses dependency graph tools to guide code reconstruction.
Based on the LocAgent approach with graph-guided context selection.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

from utils.parsers import extract_dependencies, extract_implementations, extract_readme
from utils.agent.cce.docker_env_graph import DockerEnvironmentManager
from utils.agent.cce.agent_wrapper_graph import GraphMiniSWEAgentRunner


def run_inference_for_repo(
    repo_id: str,
    target_files: List[str],
    instruction_file: Path,
    model: str,
    api_key: str,
    output_dir: Path,
    config_path: Path,
    graphs_dir: Path,
    max_turns: int = 125,
    force: bool = False
) -> Dict:
    """
    Run graph-based mini-swe-agent inference for a single repository.

    Args:
        repo_id: Repository ID
        target_files: List of target files to implement
        instruction_file: Path to instruction file
        model: Model name
        api_key: API key
        output_dir: Output directory
        config_path: Path to YAML config
        graphs_dir: Path to graphs directory
        max_turns: Maximum turns per file
        force: Force re-run

    Returns:
        Summary dictionary
    """
    print(f"\n{'='*80}")
    print(f"Running graph-based inference for repo {repo_id}")
    print(f"Target files: {len(target_files)}")
    print(f"Model: {model}")
    print(f"Max turns per file: {max_turns}")
    print(f"Config: {config_path.name}")
    print(f"{'='*80}\n")

    # Load instruction file
    with open(instruction_file, 'r') as f:
        instruction_content = f.read()

    # Parse instruction file
    print("Parsing instruction file...")
    dependencies = extract_dependencies(instruction_content)
    implementations = extract_implementations(instruction_content)
    readme = extract_readme(instruction_content)

    print(f"  Dependencies: {len(dependencies)}")
    print(f"  Files: {len(implementations)}")
    print(f"  README length: {len(readme)} chars")

    # Check if dependency graph exists
    graph_file = graphs_dir / f"{repo_id}.pkl"
    if not graph_file.exists():
        print(f"Error: Dependency graph not found: {graph_file}")
        print(f"Please run build_graphs.py first to generate dependency graphs.")
        return {
            'repo_id': repo_id,
            'status': 'error',
            'error': f'Dependency graph not found: {graph_file}',
            'completed_files': [],
            'failed_files': target_files
        }

    print(f"✓ Found dependency graph: {graph_file.name}")

    # Load or initialize summary BEFORE creating container
    summary_file = output_dir / "inference_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        print(f"\n✓ Loaded existing summary:")
        print(f"  Already completed: {len(summary['completed_files'])}")
        print(f"  Previously failed: {len(summary['failed_files'])}")
    else:
        summary = {
            'repo_id': repo_id,
            'model': model,
            'total_files': len(target_files),
            'completed_files': [],
            'failed_files': [],
            'file_results': {},
            'total_cost': 0.0,
            'total_tokens': 0,
            'total_turns': 0,
            'graph_enabled': True
        }

    # Get set of already completed files
    completed_files_set = set(summary['completed_files'])

    # Check if there are any files to process
    files_to_process = []
    for target_file in target_files:
        output_file = output_dir / target_file.replace('/', '_')
        if not force and target_file in completed_files_set and output_file.exists():
            continue
        files_to_process.append(target_file)

    if not files_to_process:
        print(f"\n✓ All {len(target_files)} files already completed for repo {repo_id}")
        print(f"  Use --force to re-run completed files")
        return summary

    print(f"\nFiles to process: {len(files_to_process)}/{len(target_files)}")
    print(f"  Skipping {len(target_files) - len(files_to_process)} already completed files")

    # Initialize Docker environment manager
    docker_manager = DockerEnvironmentManager(base_image="python:3.12")

    # Create Docker environment with graph tools
    try:
        container_name = docker_manager.create_environment(
            repo_id=repo_id,
            dependencies=dependencies,
            files=implementations,
            target_files=target_files,
            readme=readme,
            graph_path=graph_file
        )
    except Exception as e:
        print(f"Error creating Docker environment: {e}")
        return {
            'repo_id': repo_id,
            'status': 'error',
            'error': str(e),
            'completed_files': [],
            'failed_files': target_files
        }

    # Process each target file
    for i, target_file in enumerate(target_files, 1):
        output_file = output_dir / target_file.replace('/', '_')

        # Skip if already completed (unless force flag is set)
        if not force and target_file in completed_files_set and output_file.exists():
            print(f"\n[{i}/{len(target_files)}] {target_file} - already completed, skipping")
            continue

        print(f"\n{'='*80}")
        print(f"[{i}/{len(target_files)}] Processing {target_file}")
        print(f"{'='*80}\n")

        # CRITICAL: Prepare this specific target file for reconstruction
        # This modifies ONLY this file, leaving all others in original state
        docker_manager.prepare_target_file(repo_id, target_file)

        # Initialize graph-based mini-swe-agent runner
        runner = GraphMiniSWEAgentRunner(
            model_name=model,
            api_key=api_key,
            config_path=config_path,
            max_turns=max_turns
        )

        # Run agent with repo_id for graph tools
        try:
            result = runner.run(
                target_file=target_file,
                readme=readme,
                container_name=container_name,
                repo_id=repo_id  # Pass repo_id for graph tools
            )

            # Get implemented file content from result
            implemented_content = result.get('implemented_content')

            if implemented_content:
                # Save output
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_text(implemented_content)
                print(f"\n✓ Saved to {output_file.name}")

                # Add to completed_files (avoid duplicates)
                if target_file not in summary['completed_files']:
                    summary['completed_files'].append(target_file)
                # Remove from failed_files if it was previously failed
                if target_file in summary['failed_files']:
                    summary['failed_files'].remove(target_file)
            else:
                print(f"\n✗ Could not read implemented file")
                # Add to failed_files (avoid duplicates)
                if target_file not in summary['failed_files']:
                    summary['failed_files'].append(target_file)
                # Remove from completed_files if it was previously completed
                if target_file in summary['completed_files']:
                    summary['completed_files'].remove(target_file)

            # Save trajectory to separate file
            if 'trajectory' in result:
                trajectory_file = output_dir / f"{output_file.stem}_trajectory.json"
                with open(trajectory_file, 'w') as f:
                    json.dump({
                        'target_file': target_file,
                        'status': result['status'],
                        'exit_status': result.get('exit_status', ''),
                        'turns': result['turns'],
                        'trajectory': result['trajectory']
                    }, f, indent=2)
                print(f"✓ Trajectory saved to {trajectory_file.name}")

            # Add to summary
            summary['file_results'][target_file] = {
                'status': result['status'],
                'turns': result['turns'],
                'token_usage': result['token_usage']
            }

            # Recalculate totals from all file_results (handles re-runs correctly)
            summary['total_cost'] = sum(
                r.get('token_usage', {}).get('total_cost', 0)
                for r in summary['file_results'].values()
            )
            summary['total_tokens'] = sum(
                r.get('token_usage', {}).get('total_tokens', 0)
                for r in summary['file_results'].values()
            )
            summary['total_turns'] = sum(
                r.get('turns', 0)
                for r in summary['file_results'].values()
            )

            # Print token summary
            runner.token_tracker.print_summary()

        except Exception as e:
            print(f"\nError processing {target_file}: {e}")
            import traceback
            traceback.print_exc()

            # Add to failed_files (avoid duplicates)
            if target_file not in summary['failed_files']:
                summary['failed_files'].append(target_file)
            # Remove from completed_files if it was previously completed
            if target_file in summary['completed_files']:
                summary['completed_files'].remove(target_file)

            summary['file_results'][target_file] = {
                'status': 'error',
                'error': str(e)
            }

            # Recalculate totals from all file_results
            summary['total_cost'] = sum(
                r.get('token_usage', {}).get('total_cost', 0)
                for r in summary['file_results'].values()
            )
            summary['total_tokens'] = sum(
                r.get('token_usage', {}).get('total_tokens', 0)
                for r in summary['file_results'].values()
            )
            summary['total_turns'] = sum(
                r.get('turns', 0)
                for r in summary['file_results'].values()
            )

        # Save intermediate progress after each file
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Progress saved to {summary_file.name}")

        # CRITICAL: Restore this file to original state for next iteration
        # This ensures each file is reconstructed independently
        docker_manager.restore_original_file(repo_id, target_file)

        # Small delay between files
        time.sleep(2)

    # Cleanup Docker environment
    print("\nCleaning up Docker environment...")
    docker_manager.cleanup(repo_id)

    # Save final summary
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print(f"\n{'='*80}")
    print(f"INFERENCE COMPLETED FOR REPO {repo_id}")
    print(f"{'='*80}")
    print(f"Completed files: {len(summary['completed_files'])}/{len(target_files)}")
    print(f"Failed files: {len(summary['failed_files'])}")
    print(f"Total turns: {summary['total_turns']}")
    print(f"Total tokens: {summary['total_tokens']:,}")
    print(f"Total cost: ${summary['total_cost']:.4f}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*80}\n")

    return summary


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description='Run graph-based mini-swe-agent inference for code reconstruction'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-5-mini',
        help='Model name to use for inference'
    )
    parser.add_argument(
        '--max-turns',
        type=int,
        default=125,
        help='Maximum number of turns per file'
    )
    parser.add_argument(
        '--repo-ids',
        type=str,
        default=None,
        help='Comma-separated list of repo IDs to process (default: all)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-run even if files are already completed'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='original',
        choices=['original', 'large', 'all'],
        help='Dataset mode: original (repo_id < 55, default), large (repo_id >= 55), or all'
    )
    args = parser.parse_args()

    # Configuration - using relative paths from project root
    INSTRUCTIONS_DIR = Path("data/prompts")
    TARGET_FILE = Path("data/target_dev.json")
    GRAPHS_DIR = Path("data/graphs")
    CONFIG_PATH = Path("config/agent_cce.yaml")

    if not CONFIG_PATH.exists():
        print(f"Error: Config file not found: {CONFIG_PATH}")
        sys.exit(1)

    if not GRAPHS_DIR.exists():
        print(f"Error: Graphs directory not found: {GRAPHS_DIR}")
        print(f"Please run build_graphs.py first to generate dependency graphs.")
        sys.exit(1)

    # Model configuration
    MODEL = args.model
    MAX_TURNS = args.max_turns

    # Output directory: outputs/agent_cce/{model}/
    OUTPUTS_BASE_DIR = Path("outputs/agent_cce") / MODEL
    OUTPUTS_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Get API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    print("\n" + "="*80)
    print("GRAPH-BASED MINI-SWE-AGENT INFERENCE")
    print("="*80)
    print(f"Model: {MODEL}")
    print(f"Max turns: {MAX_TURNS}")
    print(f"Config: {CONFIG_PATH.name}")
    print(f"Mode: {args.mode}")
    print(f"Graphs directory: {GRAPHS_DIR}")
    print(f"API: OpenAI")
    print("="*80 + "\n")

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

    # Global summary
    global_summary = {
        'model': MODEL,
        'max_turns_per_file': MAX_TURNS,
        'config': str(CONFIG_PATH),
        'mode': args.mode,
        'version': 'graph-based',
        'repositories': {},
        'total_files': 0,
        'total_completed': 0,
        'total_failed': 0,
        'total_cost': 0.0,
        'total_tokens': 0
    }

    # Process each repository
    for entry in target_data:
        repo_id = str(entry['repo_id'])
        target_files = entry['selected_files']

        # Get instruction file
        instruction_file = INSTRUCTIONS_DIR / f"{repo_id}.txt"
        if not instruction_file.exists():
            print(f"Warning: Instruction file not found for repo {repo_id}: {instruction_file}")
            continue

        # Create output directory
        output_dir = OUTPUTS_BASE_DIR / repo_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run inference
        try:
            repo_summary = run_inference_for_repo(
                repo_id=repo_id,
                target_files=target_files,
                instruction_file=instruction_file,
                model=MODEL,
                api_key=api_key,
                output_dir=output_dir,
                config_path=CONFIG_PATH,
                graphs_dir=GRAPHS_DIR,
                max_turns=MAX_TURNS,
                force=args.force
            )

            # Update global summary
            global_summary['repositories'][repo_id] = repo_summary
            global_summary['total_files'] += repo_summary.get('total_files', len(target_files))
            global_summary['total_completed'] += len(repo_summary.get('completed_files', []))
            global_summary['total_failed'] += len(repo_summary.get('failed_files', []))
            global_summary['total_cost'] += repo_summary.get('total_cost', 0)
            global_summary['total_tokens'] += repo_summary.get('total_tokens', 0)

        except Exception as e:
            print(f"Error processing repo {repo_id}: {e}")
            import traceback
            traceback.print_exc()

    # Save global summary
    global_summary_file = OUTPUTS_BASE_DIR / "global_summary.json"
    with open(global_summary_file, 'w') as f:
        json.dump(global_summary, f, indent=2)

    # Print global summary
    print(f"\n{'='*80}")
    print("GLOBAL SUMMARY")
    print(f"{'='*80}")
    print(f"Repositories processed: {len(global_summary['repositories'])}")
    print(f"Total files: {global_summary['total_files']}")
    print(f"Completed: {global_summary['total_completed']}")
    print(f"Failed: {global_summary['total_failed']}")
    print(f"Total tokens: {global_summary['total_tokens']:,}")
    print(f"Total cost: ${global_summary['total_cost']:.4f}")
    print(f"Global summary saved to: {global_summary_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
