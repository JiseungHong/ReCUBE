#!/usr/bin/env python3
"""
Unified evaluation script for all settings.

Takes model and setting as arguments and evaluates generated outputs against test suites.
Supports external/internal test classification and large-scale dataset evaluation.

Usage:
    python utils/evaluate.py --setting full-context_basic --model gpt-5 --timeout 600
    python utils/evaluate.py --setting agent_cce --model gpt-5-mini --timeout 600
    python utils/evaluate.py --setting full-context_cot --model claude-3-haiku-20240307
    python utils/evaluate.py --setting agent_min_swe --model gemini-2.5-flash --repo-ids 0,1,2
    python utils/evaluate.py --setting full-context_basic --model gpt-5 --mode large
    python utils/evaluate.py --setting agent_min_swe --model gpt-5-mini --mode all

Input Paths:
    - Generated outputs: outputs/{setting}/{model}/{repo_id}/*.py
    - Test metadata: data/tests/{repo_id}/test_metadata.json
    - Test classifications: data/test_classifications/{repo_id}.json
    - Target data: data/target.json
    - Docker images: wlqmfl0990/recube:{repo_id}

Output Paths:
    - Results directory: results/{setting}/{model}/
    - Per-repo results: results/{setting}/{model}/{repo_id}.json
    - Overall statistics: results/{setting}/{model}/overall_statistics.json
    - Overall statistics (large): results/{setting}/{model}/overall_statistics_large.json
    - Overall statistics (all): results/{setting}/{model}/overall_statistics_all.json

Mode:
    - original (default): Evaluate only original repos (repo_id < 55)
    - large: Evaluate only large-scale repos (repo_id >= 55)
    - all: Evaluate all repos

Skip Logic:
    - By default, the script automatically skips repos that have already been fully evaluated
    - A repo is considered complete if its {repo_id}.json file exists with valid results
      for all expected files (from target data)
    - Use --force to override skip logic and re-evaluate all repos

WARNING:
    With --force flag, this script OVERWRITES all existing result files.
    Without --force, only incomplete or missing evaluations are re-run.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Tuple

import docker


def load_test_metadata(repo_id: str) -> Dict:
    """Load test metadata for a repository."""
    metadata_file = Path("data/tests") / repo_id / "test_metadata.json"
    if not metadata_file.exists():
        return {}
    with open(metadata_file) as f:
        return json.load(f)


def load_test_classifications(repo_id: str) -> Dict:
    """
    Load test classifications (external vs internal) for a repo.

    These pre-classified tests ensure all models are evaluated with
    identical denominators for fair comparison.

    Returns:
        Dict with classification data, or empty dict if not found
    """
    classification_file = Path("data/test_classifications") / f"{repo_id}.json"

    if not classification_file.exists():
        return {}

    with open(classification_file, 'r') as f:
        return json.load(f)


def is_repo_evaluation_complete(
    repo_id: str,
    results_dir: Path,
    target_data: list,
    source_repo_ids: Dict[str, list] = None
) -> bool:
    """
    Check if a repository has already been fully evaluated.

    Args:
        repo_id: Repository ID to check
        results_dir: Directory containing evaluation results
        target_data: Target data list (from target.json or target_test.json)
        source_repo_ids: For large-scale repos, mapping of original repo_id -> list of files

    Returns:
        True if evaluation is complete and valid, False otherwise
    """
    result_file = results_dir / f"{repo_id}.json"

    # If result file doesn't exist, evaluation is incomplete
    if not result_file.exists():
        return False

    try:
        with open(result_file, 'r') as f:
            result = json.load(f)
    except (json.JSONDecodeError, IOError):
        # Corrupted or unreadable file
        return False

    # Check if status is valid
    status = result.get("status")
    if status not in ["completed", "container_failed", "collection_failed", "no_metadata"]:
        return False

    # Get expected files from target data
    expected_files = set()
    repo_id_int = int(repo_id)

    # Find the repo entry in target data
    repo_entry = None
    for entry in target_data:
        if entry.get("repo_id") == repo_id_int:
            repo_entry = entry
            break

    if not repo_entry:
        # Repo not in target data, can't validate
        return False

    # For large-scale repos, expected files come from source_repo_ids
    if source_repo_ids:
        for files in source_repo_ids.values():
            expected_files.update(files)
    else:
        # For original repos, get from selected_files
        expected_files = set(repo_entry.get("selected_files", []))

    # Check if all expected files have results in per_file_results
    per_file_results = result.get("per_file_results", {})

    if len(per_file_results) == 0:
        # No file results, incomplete
        return False

    # Check if all expected files are present
    for expected_file in expected_files:
        if expected_file not in per_file_results:
            return False

        # Verify each file result has required fields
        file_result = per_file_results[expected_file]
        required_fields = ["status", "passed", "failed", "total"]

        for field in required_fields:
            if field not in file_result:
                return False

        # Verify the status is valid
        if file_result.get("status") not in ["completed", "collection_failed", "container_failed", "no_tests"]:
            return False

    # All checks passed, evaluation is complete
    return True


def parse_pytest_output_by_test(output: str) -> Dict[str, str]:
    """
    Parse pytest output to extract per-test results.
    Returns dict of test_name -> status (PASSED/FAILED)
    """
    results = {}

    # Pattern: tests/file.py::TestClass::test_name PASSED [xx%]
    # or:      tests/file.py::test_name FAILED [xx%]
    pattern = r'tests/[^:]+::(?:[^:]+::)?(\w+)\s+(PASSED|FAILED)'

    for match in re.finditer(pattern, output):
        test_name = match.group(1)
        status = match.group(2)
        results[test_name] = status

    return results


def run_tests_in_docker(
    repo_id: str,
    docker_image: str,
    source_file: str,
    test_file: str,
    generated_file_path: Path,
    timeout: int = 600,
    expected_test_count: int = 0,
    test_classifications: Dict[str, str] = None,
    docker_client: docker.DockerClient = None,
    pulled_images: set = None
) -> Dict:
    """
    Run tests in Docker container for a specific source file.

    Returns:
        Dict with test results
    """
    if docker_client is None:
        docker_client = docker.from_env()

    if pulled_images is None:
        pulled_images = set()

    try:
        # OPTIMIZATION: Check if image exists, pull only if not already pulled in this session
        if docker_image not in pulled_images:
            try:
                docker_client.images.get(docker_image)
                pulled_images.add(docker_image)
            except docker.errors.ImageNotFound:
                print(f"    Pulling {docker_image} from Docker Hub...")
                try:
                    docker_client.images.pull(docker_image)
                    print(f"    ✓ Image pulled successfully")
                    pulled_images.add(docker_image)
                except Exception as pull_error:
                    return "error", f"Failed to pull Docker image {docker_image}: {pull_error}", -1

        # FIX 3: Create container with pytest command directly, don't start it yet
        container = docker_client.containers.create(
            docker_image,
            command=["pytest", f"tests/{test_file}", "-v", "--tb=short"],
            detach=True
        )

        # Copy generated file into container
        import tarfile
        import io
        import tempfile

        # FIX 1: Create tar archive with proper directory structure using arcname
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            # Create file with full path structure
            full_path = workspace / source_file
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy generated file content
            with open(generated_file_path, 'rb') as f:
                full_path.write_bytes(f.read())

            # Create tar archive with full path preserved
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                # FIX 1: Use arcname to preserve full path (e.g., "app/agent/base.py")
                tar.add(full_path, arcname=source_file)

            tar_stream.seek(0)
            # FIX 2: Put archive in /testbed/ not /workspace/
            container.put_archive('/testbed/', tar_stream)

        # Start the container
        container.start()

        # Poll logs to detect test completion
        import time
        start_time = time.time()
        output = ""
        test_completed = False

        while time.time() - start_time < timeout:
            # Get current logs
            output = container.logs().decode('utf-8')

            # Check if tests completed (look for pytest final summary line)
            if re.search(r'=+\s+.*\s+in\s+[\d.]+s.*=+', output):
                # Tests completed - give it 2 more seconds for cleanup
                time.sleep(2)
                output = container.logs().decode('utf-8')
                test_completed = True
                break

            time.sleep(1)

        # Kill container
        try:
            container.kill()
        except:
            pass

        if test_completed:
            exit_code = 0 if ' error' not in output.split('\n')[-50:] else 2
        else:
            exit_code = 124  # Timeout

        # Cleanup container
        try:
            container.remove(force=True)
        except:
            pass

        # Parse test results from output
        lines = output.split('\n')

        # Look for summary line
        summary_line = None
        for line in reversed(lines):
            if 'passed' in line or 'failed' in line:
                summary_line = line
                break

        # Parse results
        passed = failed = 0
        if summary_line:
            passed_match = re.search(r'(\d+) passed', summary_line)
            failed_match = re.search(r'(\d+) failed', summary_line)

            if passed_match:
                passed = int(passed_match.group(1))
            if failed_match:
                failed = int(failed_match.group(1))

        # Check for collection errors or test skipping
        has_collection_error = ('error during collection' in output.lower() or
                              'interrupted:' in output.lower() or
                              re.search(r'collected \d+ items? / (\d+) errors?', output) is not None)

        # Check if tests were skipped (import errors cause pytest to skip entire test file)
        has_skip = re.search(r'collected 0 items / \d+ skipped', output) is not None

        # Handle cases where no tests were counted but we know the expected count
        # This includes: timeouts, collection failures, syntax errors, import errors, skips, etc.
        # All these indicate the model-generated code is wrong
        if (passed + failed) == 0 and expected_test_count > 0:
            if exit_code == 124 or has_collection_error or has_skip:
                # Timeout, collection failure, or skip - count all tests as failed
                failed = expected_test_count
                passed = 0
                # Also ensure external/internal counts are set from pre-classified data
                # This will be handled in the classification section below

        total_tests = passed + failed

        # Determine status
        if has_collection_error:
            status = "collection_failed"
        elif exit_code == 0:
            status = "completed"
        else:
            status = "failed"

        # Calculate external/internal metrics using pre-classified test counts
        # This ensures consistent denominators across all models
        external_passed = external_failed = internal_passed = internal_failed = 0

        # Parse per-test results from output
        test_results = parse_pytest_output_by_test(output)

        # Use pre-classified test data when available
        if test_classifications:
            # Iterate through ALL pre-classified tests to maintain consistent denominators
            for test_name, classification in test_classifications.items():
                if test_name in test_results:
                    # Test ran - use actual result
                    test_status = test_results[test_name]
                    if classification == 'external':
                        if test_status == 'PASSED':
                            external_passed += 1
                        else:
                            external_failed += 1
                    elif classification == 'internal':
                        if test_status == 'PASSED':
                            internal_passed += 1
                        else:
                            internal_failed += 1
                else:
                    # Test didn't run (collection error, timeout, etc.) - count as FAILED
                    # This ensures all pre-classified tests contribute to metrics
                    if classification == 'external':
                        external_failed += 1
                    elif classification == 'internal':
                        internal_failed += 1
        else:
            # Fallback when no classifications exist (for backward compatibility)
            for test_name, test_status in test_results.items():
                # Use heuristics only as last resort
                if any(pattern in test_name.lower() for pattern in [
                    '_internal', '_private', '_helper', '__init__',
                    'test_init', 'test_setup', 'test_teardown', 'test_cleanup'
                ]):
                    classification = 'internal'
                else:
                    classification = 'external'

                if classification == 'external':
                    if test_status == 'PASSED':
                        external_passed += 1
                    else:
                        external_failed += 1
                else:  # internal
                    if test_status == 'PASSED':
                        internal_passed += 1
                    else:
                        internal_failed += 1

        # Calculate totals for verification
        total_classified = external_passed + external_failed + internal_passed + internal_failed

        # Ensure external_total and internal_total reflect the pre-classified counts
        # This is crucial for maintaining consistent denominators across all models
        external_total = external_passed + external_failed
        internal_total = internal_passed + internal_failed

        # For overall test counts, we may need to reconcile with pytest output
        # But for external/internal, we ALWAYS use pre-classified totals
        return {
            "status": status,
            "source_file": source_file,
            "test_file": test_file,
            "tests_passed": passed,
            "tests_failed": failed,
            "total_tests": total_tests,
            "pass_rate": passed / (passed + failed) if (passed + failed) > 0 else 0,
            "exit_code": exit_code,
            "collection_error": has_collection_error,
            "output": output,
            "external_passed": external_passed,
            "external_failed": external_failed,
            "external_total": external_total,
            "internal_passed": internal_passed,
            "internal_failed": internal_failed,
            "internal_total": internal_total,
            "total_classified": total_classified,
            "unclassified": max(0, total_tests - total_classified)
        }

    except Exception as e:
        print(f"    ✗ Error running test for {source_file}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "source_file": source_file,
            "test_file": test_file,
            "error": str(e),
            "external_passed": 0,
            "external_failed": 0,
            "internal_passed": 0,
            "internal_failed": 0
        }


def evaluate_repo(
    repo_id: str,
    outputs_dir: Path,
    timeout: int = 600,
    source_repo_ids: Dict[str, list] = None
) -> Dict:
    """
    Evaluate all generated files for a repository.

    Args:
        repo_id: Repository ID
        outputs_dir: Directory containing generated outputs for this repo
        timeout: Timeout for each test file
        source_repo_ids: For large-scale repos, mapping of original repo_id -> list of files

    Returns:
        Dict with evaluation results
    """
    # For large-scale repos, we'll determine docker_image per file
    # For original repos, use single docker image
    default_docker_image = f"wlqmfl0990/recube:{repo_id}"

    # Create reverse mapping: file -> original repo_id for large-scale repos
    file_to_repo_id = {}
    if source_repo_ids:
        for orig_repo_id, files in source_repo_ids.items():
            for file in files:
                file_to_repo_id[file] = orig_repo_id

    # Load metadata - for large-scale repos, load from original repos
    if source_repo_ids:
        # Large-scale repo: aggregate metadata from all source repos
        test_metadata = {"instances": []}
        test_classifications_all = {"files": {}}

        for orig_repo_id in source_repo_ids.keys():
            # Load test metadata from original repo
            orig_test_metadata = load_test_metadata(orig_repo_id)
            orig_test_classifications = load_test_classifications(orig_repo_id)

            if orig_test_metadata and "instances" in orig_test_metadata:
                # Filter instances to only those files in source_repo_ids[orig_repo_id]
                files_for_this_repo = set(source_repo_ids[orig_repo_id])
                for instance in orig_test_metadata["instances"]:
                    if instance.get("file") in files_for_this_repo:
                        test_metadata["instances"].append(instance)

            if orig_test_classifications and "files" in orig_test_classifications:
                # Merge test classifications
                for file_path, file_data in orig_test_classifications["files"].items():
                    if file_path in source_repo_ids[orig_repo_id]:
                        test_classifications_all["files"][file_path] = file_data
    else:
        # Original repo: load normally
        test_metadata = load_test_metadata(repo_id)
        test_classifications_all = load_test_classifications(repo_id)

    if not test_metadata or not test_metadata.get("instances"):
        return {
            "repo_id": repo_id,
            "status": "no_metadata",
            "tests_passed": 0,
            "tests_failed": 0,
            "total_tests": 0,
            "per_file_results": {}
        }

    print(f"\n{'='*80}")
    print(f"Testing repo_id {repo_id}")
    print(f"{'='*80}\n")
    if source_repo_ids:
        print(f"Large-scale repo: using multiple Docker images based on source_repo_ids\n")
    else:
        print(f"Using Docker image: {default_docker_image}\n")

    # Extract file mappings and test counts from instances array
    file_mappings = {}
    test_counts = {}
    for instance in test_metadata.get("instances", []):
        source_file = instance.get("file")
        test_file = instance.get("test_file")
        total_tests = instance.get("total_tests", 0)
        if source_file and test_file:
            file_mappings[source_file] = test_file
            test_counts[source_file] = total_tests

    print(f"File mappings from test_metadata.json: {len(file_mappings)} files")
    for src, test in list(file_mappings.items())[:10]:
        print(f"  {src} → {test}")

    # OPTIMIZATION: Create shared Docker client and track pulled images
    docker_client = docker.from_env()
    pulled_images = set()

    # Evaluate each file
    per_file_results = {}
    total_passed = 0
    total_failed = 0

    for source_file, test_file in file_mappings.items():
        # Determine docker image for this file
        if file_to_repo_id and source_file in file_to_repo_id:
            # Large-scale repo: use original repo's docker image
            orig_repo_id = file_to_repo_id[source_file]
            docker_image = f"wlqmfl0990/recube:{orig_repo_id}"
        else:
            # Original repo: use default docker image
            docker_image = default_docker_image

        # Find generated file
        generated_filename = source_file.replace('/', '_').replace('.py', '.py')
        generated_file = outputs_dir / generated_filename
        expected_tests = test_counts.get(source_file, 0)

        # Get test classifications for this file
        file_classifications = test_classifications_all.get('files', {}).get(source_file, {}).get('test_classifications', {})

        if not generated_file.exists():
            print(f"\n  Testing {source_file}...")
            print(f"    ✗ Missing model output - counting {expected_tests} tests as failed")

            # Use pre-classified test counts for consistent evaluation
            # This ensures all models have the same denominators
            external_failed = 0
            internal_failed = 0

            # Use pre-classified counts from test_classifications data
            file_metadata = test_classifications_all.get('files', {}).get(source_file, {})
            external_failed = file_metadata.get('external_tests', 0)
            internal_failed = file_metadata.get('internal_tests', 0)

            # Verify the counts match expected total
            if external_failed + internal_failed != expected_tests and expected_tests > 0:
                # Log discrepancy but use the pre-classified counts
                print(f"    Note: Classification mismatch - expected {expected_tests}, got {external_failed + internal_failed}")

            per_file_results[source_file] = {
                "status": "missing_output",
                "passed": 0,
                "failed": expected_tests,
                "total": expected_tests,
                "external_passed": 0,
                "external_failed": external_failed,
                "external_total": external_failed,
                "internal_passed": 0,
                "internal_failed": internal_failed,
                "internal_total": internal_failed,
                "total_classified": external_failed + internal_failed,
                "unclassified": max(0, expected_tests - (external_failed + internal_failed))
            }
            total_failed += expected_tests
            continue

        print(f"\n  Testing {source_file}...")

        # Run tests with optimized Docker handling
        result = run_tests_in_docker(
            repo_id,
            docker_image,
            source_file,
            test_file,
            generated_file,
            timeout,
            expected_test_count=expected_tests,
            test_classifications=file_classifications,
            docker_client=docker_client,
            pulled_images=pulled_images
        )

        # Print brief result
        if result.get("status") in ["completed", "failed", "collection_failed"]:
            print(f"    ✓ {result['tests_passed']}/{result['total_tests']} tests passed")
        else:
            print(f"    ✗ Error: {result.get('status')}")

        total_passed += result.get("tests_passed", 0)
        total_failed += result.get("tests_failed", 0)

        per_file_results[source_file] = {
            "status": result.get("status"),
            "passed": result.get("tests_passed", 0),
            "failed": result.get("tests_failed", 0),
            "total": result.get("total_tests", 0),
            "external_passed": result.get("external_passed", 0),
            "external_failed": result.get("external_failed", 0),
            "external_total": result.get("external_total", 0),
            "internal_passed": result.get("internal_passed", 0),
            "internal_failed": result.get("internal_failed", 0),
            "internal_total": result.get("internal_total", 0),
            "total_classified": result.get("total_classified", 0),
            "unclassified": result.get("unclassified", 0)
        }

    print(f"\n{'='*80}")
    print(f"RESULTS FOR REPO {repo_id}")
    print(f"{'='*80}")
    print(f"Status: completed")
    print(f"Tests passed: {total_passed}/{total_passed + total_failed}")
    if total_passed + total_failed > 0:
        print(f"Pass rate: {total_passed / (total_passed + total_failed) * 100:.1f}%")
    print(f"{'='*80}\n")

    return {
        "repo_id": repo_id,
        "status": "completed",
        "tests_passed": total_passed,
        "tests_failed": total_failed,
        "total_tests": total_passed + total_failed,
        "per_file_results": per_file_results
    }


def calculate_summary_from_results(results: list, model: str, setting: str, mode: str) -> dict:
    """Calculate summary statistics from a list of results."""
    # Calculate instances with tests
    instances_with_tests = []
    for result in results:
        if result.get("status") in ["completed", "container_failed", "collection_failed"]:
            total_tests = result.get("tests_passed", 0) + result.get("tests_failed", 0)
            if total_tests > 0:
                instances_with_tests.append({
                    "repo_id": result.get("repo_id"),
                    "passed": result.get("tests_passed", 0),
                    "total": total_tests,
                    "pass_rate": result.get("tests_passed", 0) / total_tests
                })

    # Calculate average/strict pass rates
    average_pass_rate = None
    strict_pass_rate = None
    if len(instances_with_tests) > 0:
        average_pass_rate = (sum(inst["pass_rate"] for inst in instances_with_tests) / len(instances_with_tests)) * 100
        fully_resolved = sum(1 for inst in instances_with_tests if inst["passed"] == inst["total"])
        strict_pass_rate = (fully_resolved / len(instances_with_tests)) * 100

    # Calculate external/internal metrics
    external_instances = []
    internal_instances = []

    for result in results:
        if result.get("status") in ["completed", "container_failed", "collection_failed"]:
            per_file_results = result.get("per_file_results", {})

            for source_file, file_result in per_file_results.items():
                # External instances
                ext_passed = file_result.get("external_passed", 0)
                ext_failed = file_result.get("external_failed", 0)
                ext_total = ext_passed + ext_failed

                if ext_total > 0:
                    external_instances.append({
                        "repo_id": result.get("repo_id"),
                        "source_file": source_file,
                        "passed": ext_passed,
                        "total": ext_total,
                        "pass_rate": ext_passed / ext_total
                    })

                # Internal instances
                int_passed = file_result.get("internal_passed", 0)
                int_failed = file_result.get("internal_failed", 0)
                int_total = int_passed + int_failed

                if int_total > 0:
                    internal_instances.append({
                        "repo_id": result.get("repo_id"),
                        "source_file": source_file,
                        "passed": int_passed,
                        "total": int_total,
                        "pass_rate": int_passed / int_total
                    })

    # Calculate external/internal metrics
    external_average_pass_rate = None
    external_strict_pass_rate = None
    if len(external_instances) > 0:
        external_average_pass_rate = (sum(inst["pass_rate"] for inst in external_instances) / len(external_instances)) * 100
        external_fully_resolved = sum(1 for inst in external_instances if inst["passed"] == inst["total"])
        external_strict_pass_rate = (external_fully_resolved / len(external_instances)) * 100

    internal_average_pass_rate = None
    internal_strict_pass_rate = None
    if len(internal_instances) > 0:
        internal_average_pass_rate = (sum(inst["pass_rate"] for inst in internal_instances) / len(internal_instances)) * 100
        internal_fully_resolved = sum(1 for inst in internal_instances if inst["passed"] == inst["total"])
        internal_strict_pass_rate = (internal_fully_resolved / len(internal_instances)) * 100

    # Determine total_repos based on mode
    if mode == "dev":
        total_repos_count = 5  # Validation set
    elif mode == "test":
        total_repos_count = len(set(int(r['repo_id']) for r in results))  # Dynamic count
    else:
        total_repos_count = len(set(int(r['repo_id']) for r in results))

    return {
        "model": model,
        "setting": setting,
        "mode": mode,
        "total_repos": total_repos_count,
        "repos_tested": sum(1 for r in results if r.get("status") == "completed"),
        "total_instances": len(instances_with_tests),
        "total_tests_passed": sum(r.get("tests_passed", 0) for r in results),
        "total_tests_failed": sum(r.get("tests_failed", 0) for r in results),
        "total_tests": sum(r.get("total_tests", 0) for r in results),
        "average_pass_rate": average_pass_rate,
        "strict_pass_rate": strict_pass_rate,
        "instances_with_external_tests": len(external_instances),
        "instances_with_internal_tests": len(internal_instances),
        "total_external_passed": sum(inst["passed"] for inst in external_instances),
        "total_external_failed": sum(inst["total"] - inst["passed"] for inst in external_instances),
        "total_external": sum(inst["total"] for inst in external_instances),
        "external_average_pass_rate": external_average_pass_rate,
        "external_strict_pass_rate": external_strict_pass_rate,
        "total_internal_passed": sum(inst["passed"] for inst in internal_instances),
        "total_internal_failed": sum(inst["total"] - inst["passed"] for inst in internal_instances),
        "total_internal": sum(inst["total"] for inst in internal_instances),
        "internal_average_pass_rate": internal_average_pass_rate,
        "internal_strict_pass_rate": internal_strict_pass_rate,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation script for all settings")
    parser.add_argument("--setting", type=str, required=True,
                       help="Setting name (e.g., full-context_basic, agent_cce)")
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (e.g., gpt-5, claude-3-haiku-20240307)")
    parser.add_argument("--timeout", type=int, default=600,
                       help="Timeout per test file in seconds")
    parser.add_argument("--repo-ids", type=str, default=None,
                       help="Comma-separated list of repo IDs to evaluate (default: all)")
    parser.add_argument("--mode", type=str, default="original", choices=["original", "large", "all"],
                       help="Dataset mode: 'original' (repo_id < 55), 'large' (repo_id >= 55), 'all' (default: original)")
    parser.add_argument("--force", action="store_true",
                       help="Force re-evaluation of all repos, even if already evaluated")

    args = parser.parse_args()

    # Construct outputs path using relative paths from project root
    outputs_path = Path("outputs") / args.setting / args.model

    if not outputs_path.exists():
        print(f"Error: Outputs directory not found: {outputs_path}")
        sys.exit(1)

    print(f"Setting: {args.setting}")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Outputs directory: {outputs_path}")
    if args.force:
        print("Force mode: ON (will re-evaluate all repos)")

    # Load target data
    target_file = Path("data/target.json")
    print(f"Using target file: {target_file}")

    with open(target_file, 'r') as f:
        target_data = json.load(f)

    # Create mapping: repo_id -> entry for quick lookup
    repo_metadata = {entry['repo_id']: entry for entry in target_data}

    # Determine results directory
    results_dir = Path("results") / args.setting / args.model
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}\n")

    # Find all repo directories in outputs
    if args.repo_ids:
        repo_ids = [rid.strip() for rid in args.repo_ids.split(',')]
    else:
        repo_dirs = [d for d in outputs_path.iterdir() if d.is_dir() and d.name.isdigit()]
        repo_ids = sorted([d.name for d in repo_dirs], key=lambda x: int(x))

    # Filter repo_ids based on mode and verify they exist in target data
    if args.mode == "original":
        # Only original repos (repo_id < 55) that exist in outputs
        repo_ids = [rid for rid in repo_ids if int(rid) < 55]
    elif args.mode == "large":
        # Only large-scale repos (repo_id >= 55) that exist in outputs
        repo_ids = [rid for rid in repo_ids if int(rid) >= 55]
    else:  # "all"
        # For "all" mode, we need to check target data for all repo_ids (both original and large)
        # Get all repo_ids from target data
        all_target_repo_ids = set(str(entry['repo_id']) for entry in target_data)
        # Intersect with available output directories
        available_repo_ids = set(repo_ids)
        repo_ids = sorted(list(available_repo_ids & all_target_repo_ids), key=lambda x: int(x))

    print(f"Found {len(repo_ids)} repo(s) to evaluate ({args.mode} mode): {repo_ids}\n")

    # Check which repos are already evaluated and can be skipped
    repos_to_evaluate = []
    repos_to_skip = []

    for repo_id in repo_ids:
        repo_id_int = int(repo_id)
        source_repo_ids = None

        # Get source_repo_ids for large-scale repos
        if repo_id_int in repo_metadata:
            entry = repo_metadata[repo_id_int]
            if entry.get('is_large_scale', False):
                # Convert keys to strings for consistency
                source_repo_ids = {str(k): v for k, v in entry.get('source_repo_ids', {}).items()}

        # Check if evaluation is already complete (unless --force is used)
        if not args.force and is_repo_evaluation_complete(repo_id, results_dir, target_data, source_repo_ids):
            repos_to_skip.append(repo_id)
        else:
            repos_to_evaluate.append(repo_id)

    if repos_to_skip:
        print(f"Skipping {len(repos_to_skip)} already-evaluated repo(s): {repos_to_skip}")
    if repos_to_evaluate:
        print(f"Evaluating {len(repos_to_evaluate)} repo(s): {repos_to_evaluate}\n")
    else:
        print("All repos are already evaluated. Loading existing results...\n")

    # Evaluate repos that need evaluation
    all_results = []

    for repo_id in repos_to_evaluate:
        repo_outputs_dir = outputs_path / repo_id
        if not repo_outputs_dir.exists():
            print(f"Warning: Outputs directory not found for repo {repo_id}: {repo_outputs_dir}")
            continue

        # Get source_repo_ids for large-scale repos
        repo_id_int = int(repo_id)
        source_repo_ids = None
        if repo_id_int in repo_metadata:
            entry = repo_metadata[repo_id_int]
            if entry.get('is_large_scale', False):
                # Convert keys to strings for consistency
                source_repo_ids = {str(k): v for k, v in entry.get('source_repo_ids', {}).items()}

        result = evaluate_repo(repo_id, repo_outputs_dir, args.timeout, source_repo_ids)
        all_results.append(result)

        # Save individual repo result
        repo_result_file = results_dir / f"{repo_id}.json"
        with open(repo_result_file, 'w') as f:
            json.dump(result, f, indent=2)

    # Load results for skipped repos
    for repo_id in repos_to_skip:
        repo_result_file = results_dir / f"{repo_id}.json"
        try:
            with open(repo_result_file, 'r') as f:
                result = json.load(f)
                all_results.append(result)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load existing result for repo {repo_id}: {e}")
            continue

    # Calculate overall statistics
    # Build instance list (files with tests)
    all_instances = []
    for result in all_results:
        if result.get("status") in ["completed", "container_failed", "collection_failed"]:
            per_file_results = result.get("per_file_results", {})

            for source_file, file_result in per_file_results.items():
                instance = {
                    "repo_id": result.get("repo_id"),
                    "source_file": source_file,
                    "passed": file_result.get("passed", 0),
                    "total": file_result.get("total", 0)
                }
                all_instances.append(instance)

    # Filter out instances with 0 tests
    instances_with_tests = [inst for inst in all_instances if inst["total"] > 0]

    # Calculate average/strict pass rates
    average_pass_rate = None
    strict_pass_rate = None

    if len(instances_with_tests) > 0:
        # Average Pass Rate: Average of (passed/total) across all instances
        apr_sum = sum(inst["passed"] / inst["total"] for inst in instances_with_tests)
        average_pass_rate = (apr_sum / len(instances_with_tests)) * 100

        # Strict Pass Rate: Percentage of instances with 100% pass rate
        fully_resolved = sum(1 for inst in instances_with_tests
                            if inst["passed"] == inst["total"])
        strict_pass_rate = (fully_resolved / len(instances_with_tests)) * 100

    # Calculate external/internal metrics
    external_instances = []
    internal_instances = []

    for result in all_results:
        if result.get("status") in ["completed", "container_failed", "collection_failed"]:
            per_file_results = result.get("per_file_results", {})

            for source_file, file_result in per_file_results.items():
                # External instances
                ext_passed = file_result.get("external_passed", 0)
                ext_failed = file_result.get("external_failed", 0)
                ext_total = ext_passed + ext_failed

                if ext_total > 0:
                    external_instances.append({
                        "repo_id": result.get("repo_id"),
                        "source_file": source_file,
                        "passed": ext_passed,
                        "total": ext_total,
                        "pass_rate": ext_passed / ext_total
                    })

                # Internal instances
                int_passed = file_result.get("internal_passed", 0)
                int_failed = file_result.get("internal_failed", 0)
                int_total = int_passed + int_failed

                if int_total > 0:
                    internal_instances.append({
                        "repo_id": result.get("repo_id"),
                        "source_file": source_file,
                        "passed": int_passed,
                        "total": int_total,
                        "pass_rate": int_passed / int_total
                    })

    # Calculate external/internal metrics
    external_average_pass_rate = None
    external_strict_pass_rate = None
    if len(external_instances) > 0:
        external_average_pass_rate = (sum(inst["pass_rate"] for inst in external_instances) / len(external_instances)) * 100
        external_fully_resolved = sum(1 for inst in external_instances if inst["passed"] == inst["total"])
        external_strict_pass_rate = (external_fully_resolved / len(external_instances)) * 100

    internal_average_pass_rate = None
    internal_strict_pass_rate = None
    if len(internal_instances) > 0:
        internal_average_pass_rate = (sum(inst["pass_rate"] for inst in internal_instances) / len(internal_instances)) * 100
        internal_fully_resolved = sum(1 for inst in internal_instances if inst["passed"] == inst["total"])
        internal_strict_pass_rate = (internal_fully_resolved / len(internal_instances)) * 100

    # Calculate correct total_repos based on mode
    if args.mode == "original":
        # Count repos in target data with repo_id < 55
        total_repos_count = len([e for e in target_data if int(e['repo_id']) < 55])
    elif args.mode == "large":
        # Count repos in target data with repo_id >= 55
        total_repos_count = len([e for e in target_data if int(e['repo_id']) >= 55])
    else:  # all
        # Count all repos in target data
        total_repos_count = len(target_data)

    # Save overall statistics
    summary = {
        "model": args.model,
        "setting": args.setting,
        "mode": args.mode,
        "total_repos": total_repos_count,
        "repos_tested": sum(1 for r in all_results if r.get("status") == "completed"),
        "total_instances": len(instances_with_tests),
        "total_tests_passed": sum(r.get("tests_passed", 0) for r in all_results),
        "total_tests_failed": sum(r.get("tests_failed", 0) for r in all_results),
        "total_tests": sum(r.get("total_tests", 0) for r in all_results),
        "average_pass_rate": average_pass_rate,
        "strict_pass_rate": strict_pass_rate,
        "instances_with_external_tests": len(external_instances),
        "instances_with_internal_tests": len(internal_instances),
        "total_external_passed": sum(inst["passed"] for inst in external_instances),
        "total_external_failed": sum(inst["total"] - inst["passed"] for inst in external_instances),
        "total_external": sum(inst["total"] for inst in external_instances),
        "external_average_pass_rate": external_average_pass_rate,
        "external_strict_pass_rate": external_strict_pass_rate,
        "total_internal_passed": sum(inst["passed"] for inst in internal_instances),
        "total_internal_failed": sum(inst["total"] - inst["passed"] for inst in internal_instances),
        "total_internal": sum(inst["total"] for inst in internal_instances),
        "internal_average_pass_rate": internal_average_pass_rate,
        "internal_strict_pass_rate": internal_strict_pass_rate,
        "results": all_results
    }

    # Determine summary filename based on mode
    if args.mode == "large":
        summary_filename = "overall_statistics_large.json"
    elif args.mode == "all":
        summary_filename = "overall_statistics_all.json"
    else:  # original
        summary_filename = "overall_statistics.json"

    # Save main summary file
    summary_file = results_dir / summary_filename
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY FOR {args.model} ({args.setting}) - Mode: {args.mode}")
    print(f"{'='*80}")
    print(f"Total repos: {summary['total_repos']}")
    print(f"Repos tested: {summary['repos_tested']}")
    print(f"Total instances (files with tests): {summary['total_instances']}")
    print(f"Total tests passed: {summary['total_tests_passed']}/{summary['total_tests']}")
    if average_pass_rate is not None:
        print(f"Average Pass Rate: {average_pass_rate:.2f}%")
    if strict_pass_rate is not None:
        print(f"Strict Pass Rate: {strict_pass_rate:.2f}%")

    print(f"\nExternal/Internal Test Breakdown:")
    print(f"  Instances with external tests: {summary['instances_with_external_tests']}")
    print(f"  External tests: {summary['total_external_passed']}/{summary['total_external']}")
    if external_average_pass_rate is not None:
        print(f"  External average pass rate: {external_average_pass_rate:.2f}%")
        print(f"  External strict pass rate: {external_strict_pass_rate:.2f}%")

    print(f"  Instances with internal tests: {summary['instances_with_internal_tests']}")
    print(f"  Internal tests: {summary['total_internal_passed']}/{summary['total_internal']}")
    if internal_average_pass_rate is not None:
        print(f"  Internal average pass rate: {internal_average_pass_rate:.2f}%")
        print(f"  Internal strict pass rate: {internal_strict_pass_rate:.2f}%")

    print("\nResults saved to:", results_dir)
    print("Overall statistics saved to:", summary_file)
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
