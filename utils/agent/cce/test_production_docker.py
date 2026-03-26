#!/usr/bin/env python3
"""
Test the ACTUAL production code path used by generate_output.py
This tests DockerEnvironmentManager.create_environment() with networkx installation
"""

import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.agent.cce.docker_env_graph import DockerEnvironmentManager
from utils.parsers import extract_dependencies, extract_implementations, extract_readme

def test_production_docker_setup():
    """Test the actual production Docker setup that generate_output.py uses."""

    print("="*70)
    print("TESTING PRODUCTION DOCKER ENVIRONMENT SETUP")
    print("="*70)

    # Use repo 0 like in the actual outputs
    repo_id = "0"
    instruction_file = PROJECT_ROOT / "data" / "instructions" / "0.jsonl"
    graphs_dir = PROJECT_ROOT / "data" / "graphs"
    graph_file = graphs_dir / f"{repo_id}.pkl"

    if not instruction_file.exists():
        print(f"ERROR: Instruction file not found: {instruction_file}")
        return False

    if not graph_file.exists():
        print(f"ERROR: Graph file not found: {graph_file}")
        return False

    print(f"\n✓ Using instruction file: {instruction_file.name}")
    print(f"✓ Using graph file: {graph_file.name}")

    # Parse instruction file (same as generate_output.py)
    with open(instruction_file) as f:
        data = json.load(f)

    readme = extract_readme(data)
    dependencies = extract_dependencies(data)
    implementations = extract_implementations(data)
    target_files = list(implementations.keys())

    print(f"\n✓ Loaded {len(target_files)} target files")
    print(f"✓ Found {len(dependencies)} dependencies")

    # Initialize Docker manager (same as generate_output.py line 131)
    docker_manager = DockerEnvironmentManager(base_image="python:3.12")

    try:
        print(f"\n{'='*70}")
        print("CREATING DOCKER ENVIRONMENT (this is the actual production code)")
        print("="*70)

        # This is the ACTUAL production code path from generate_output.py:134-141
        container_name = docker_manager.create_environment(
            repo_id=repo_id,
            dependencies=dependencies,
            files=implementations,
            target_files=target_files,
            readme=readme,
            graph_path=graph_file
        )

        print(f"\n✓ Container created: {container_name}")

        # Now test that graph tools work
        print(f"\n{'='*70}")
        print("TESTING GRAPH TOOLS IN PRODUCTION ENVIRONMENT")
        print("="*70)

        container = docker_manager.client.containers.get(container_name)

        # Test 1: show_implementation_context (requires networkx)
        print("\n[TEST 1] show_implementation_context")
        exit_code, output = container.exec_run(
            "python /workspace/tools/graph_tools.py show_implementation_context --target app/schema.py",
            environment={"TARGET_FILE": "app/schema.py"}
        )
        result = output.decode()

        if exit_code != 0:
            print(f"✗ FAILED with exit code {exit_code}")
            print(result)
            return False

        if "IMPLEMENTATION CONTEXT FOR:" in result:
            print("✓ PASSED - Tool works correctly")
        else:
            print("✗ FAILED - Unexpected output")
            print(result[:500])
            return False

        # Test 2: validate_code (should work without graph)
        print("\n[TEST 2] validate_code")
        exit_code, output = container.exec_run(
            "python /workspace/tools/graph_tools.py validate_code --file /workspace/app/schema.py"
        )
        result = output.decode()

        if "VALIDATION REPORT:" in result:
            print("✓ PASSED - Validation tool works")
        else:
            print("✗ FAILED - Validation failed")
            print(result[:500])
            return False

        print(f"\n{'='*70}")
        print("ALL PRODUCTION TESTS PASSED! ✓")
        print("="*70)
        print("\nThe production Docker setup is working correctly:")
        print("  ✓ Container created successfully")
        print("  ✓ networkx installed automatically")
        print("  ✓ Graph tools copied correctly")
        print("  ✓ All tools functional")
        print("\nYou can now re-run generate_output.py and it will work!")

        # Cleanup
        container.remove(force=True)
        print(f"\n✓ Cleaned up test container")

        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_production_docker_setup()
    sys.exit(0 if success else 1)
