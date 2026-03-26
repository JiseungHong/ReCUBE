#!/usr/bin/env python3
"""
Check setup and prerequisites for mini-swe-agent evaluation.
"""

import os
import sys
from pathlib import Path


def check_setup():
    """Check all prerequisites."""

    print("="*80)
    print("Mini-SWE-Agent Evaluation - Setup Check")
    print("="*80)
    print()

    all_good = True

    # Check Python version
    print("1. Checking Python version...")
    python_version = sys.version_info
    if python_version >= (3, 9):
        print(f"   ✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"   ✗ Python {python_version.major}.{python_version.minor}.{python_version.micro} (requires 3.9+)")
        all_good = False

    # Check Docker
    print("\n2. Checking Docker...")
    try:
        import docker
        client = docker.from_env()
        client.ping()
        print("   ✓ Docker is running")
        # Get Docker version
        version = client.version()
        print(f"   ✓ Docker version: {version['Version']}")
    except ImportError:
        print("   ✗ docker package not installed (run: pip install docker)")
        all_good = False
    except Exception as e:
        print(f"   ✗ Docker is not running or accessible: {e}")
        all_good = False

    # Check OpenAI package
    print("\n3. Checking OpenAI package...")
    try:
        import openai
        print(f"   ✓ openai package installed (version: {openai.__version__})")
    except ImportError:
        print("   ✗ openai package not installed (run: pip install openai)")
        all_good = False

    # Check API key
    print("\n4. Checking API key...")
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        print(f"   ✓ OPENAI_API_KEY is set ({api_key[:10]}...)")
    else:
        print("   ✗ OPENAI_API_KEY environment variable not set")
        print("      Set it with: export OPENAI_API_KEY='your-key-here'")
        all_good = False

    # Check data files
    print("\n5. Checking data files...")

    target_file = Path("data/target.json")
    if target_file.exists():
        print(f"   ✓ target.json found")
        # Count repos
        import json
        with open(target_file) as f:
            target_data = json.load(f)
        print(f"   ✓ {len(target_data)} repositories defined")
    else:
        print(f"   ✗ target.json not found at {target_file}")
        all_good = False

    instructions_dir = Path("data/prompts")
    if instructions_dir.exists():
        instruction_files = list(instructions_dir.glob("*.txt"))
        print(f"   ✓ Prompts directory found with {len(instruction_files)} files")
    else:
        print(f"   ✗ Prompts directory not found at {instructions_dir}")
        all_good = False

    # Check output directory
    print("\n6. Checking output directory...")
    output_dir = Path("outputs/agent_min_swe")
    if not output_dir.exists():
        print(f"   ! Output directory will be created at {output_dir}")
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"   ✓ Created output directory")
        except Exception as e:
            print(f"   ✗ Could not create output directory: {e}")
            all_good = False
    else:
        print(f"   ✓ Output directory exists at {output_dir}")

    # Check disk space
    print("\n7. Checking disk space...")
    import shutil
    stat = shutil.disk_usage(".")
    free_gb = stat.free / (1024**3)
    if free_gb > 10:
        print(f"   ✓ {free_gb:.1f} GB free disk space")
    else:
        print(f"   ⚠ Only {free_gb:.1f} GB free disk space (Docker needs space)")

    # Summary
    print("\n" + "="*80)
    if all_good:
        print("✓ All checks passed! You're ready to run inference.")
        print("\nTo run inference:")
        print("  python utils/agent/min_swe/generate_output.py --model claude-3-haiku-20240307")
    else:
        print("✗ Some checks failed. Please fix the issues above before running.")
    print("="*80)
    print()

    return all_good


if __name__ == "__main__":
    sys.exit(0 if check_setup() else 1)
