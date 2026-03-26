#!/bin/bash
# Run evaluation for code reconstruction outputs
# Usage: ./scripts/run_evaluation.sh <setting> <model> [additional_args...]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to print usage
usage() {
    cat << EOF
Usage: $0 <setting> <model> [additional_args...]

Arguments:
    setting         The experimental setting to evaluate:
                    - full-context_basic    : Basic prompt-based reconstruction
                    - full-context_cot      : Chain-of-Thought prompt-based reconstruction
                    - agent_min_swe : Agent with min_swe config
                    - agent_cce : Agent with dependency graph

    model           The model to evaluate, e.g.:
                    - claude-3-haiku-20240307
                    - gpt-5-mini
                    - gpt-5
                    - gemini-2.5-flash
                    - gemini-2.5-pro

    additional_args Optional additional arguments:
                    --timeout SECONDS    : Timeout per test file (default: 600)
                    --repo-ids IDS       : Comma-separated repo IDs to evaluate
                    --mode MODE          : Dataset mode: original (default), large, or all

Examples:
    $0 full-context_basic claude-3-haiku-20240307
    $0 agent_cce gpt-5 --timeout 300
    $0 full-context_cot gpt-5-mini --repo-ids 0,1,2
    $0 full-context_basic gpt-5 --mode large

Input Requirements:
    - Generated outputs: outputs/{setting}/{model}/{repo_id}/*.py
    - Test metadata: data/tests/{repo_id}/test_metadata.json
    - Test classifications: data/test_classifications/{repo_id}.json
    - Docker images: wlqmfl0990/recube:{repo_id}

Output:
    - Results directory: results/{setting}/{model}/
    - Per-repo results: results/{setting}/{model}/{repo_id}.json
    - Overall statistics: results/{setting}/{model}/overall_statistics.json
    - Metrics: Average Pass Rate (APR) and Strict Pass Rate (SPR)

WARNING:
    This script OVERWRITES existing evaluation results for the same setting and model!

EOF
}

# Check if correct number of arguments
if [ $# -lt 2 ]; then
    print_error "Missing required arguments"
    echo ""
    usage
    exit 1
fi

# Parse arguments
SETTING=$1
MODEL=$2
shift 2  # Remove first two arguments

# Filter out -y/--yes flags and parse --mode
ADDITIONAL_ARGS=""
SKIP_CONFIRMATION=false
MODE="original"  # default mode
PARSING_MODE=false
for arg in "$@"; do
    if [ "$arg" = "-y" ] || [ "$arg" = "--yes" ]; then
        SKIP_CONFIRMATION=true
    elif [ "$arg" = "--mode" ]; then
        PARSING_MODE=true
        ADDITIONAL_ARGS="$ADDITIONAL_ARGS $arg"
    elif [ "$PARSING_MODE" = true ]; then
        MODE="$arg"
        PARSING_MODE=false
        ADDITIONAL_ARGS="$ADDITIONAL_ARGS $arg"
    else
        ADDITIONAL_ARGS="$ADDITIONAL_ARGS $arg"
    fi
done
ADDITIONAL_ARGS=$(echo "$ADDITIONAL_ARGS" | xargs)  # Trim whitespace

# Get project root (script is in scripts/, so go up one level)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

print_info "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Validate setting
case $SETTING in
    full-context_basic)
        print_info "Setting: Basic Prompt-based Reconstruction"
        ;;

    full-context_cot)
        print_info "Setting: Chain-of-Thought Prompt-based Reconstruction"
        ;;

    agent_min_swe)
        print_info "Setting: Agent (Min-SWE Config)"
        ;;

    agent_cce)
        print_info "Setting: Agent (CCE-based)"
        ;;

    *)
        print_error "Unknown setting: $SETTING"
        echo ""
        usage
        exit 1
        ;;
esac

# Check if outputs directory exists
OUTPUTS_DIR="outputs/$SETTING/$MODEL"
if [ ! -d "$OUTPUTS_DIR" ]; then
    print_error "Outputs directory not found: $OUTPUTS_DIR"
    print_error "Please run generation first using ./scripts/generate_model_output.sh"
    exit 1
fi

print_success "Found outputs directory: $OUTPUTS_DIR"

# Check for missing outputs by comparing with target.json
print_info "Checking for missing outputs..."

if [ ! -f "data/target.json" ]; then
    print_warning "Cannot verify completeness: data/target.json not found"
else
    # Use Python to check missing files
    MISSING_CHECK=$(python3 - "$OUTPUTS_DIR" "$MODE" << 'EOF'
import json
from pathlib import Path
import sys

try:
    # Load target file
    with open("data/target.json") as f:
        target_data = json.load(f)

    # Get outputs directory and mode from command line arguments
    if len(sys.argv) < 3:
        print("ERROR:Missing outputs directory or mode argument", file=sys.stderr)
        sys.exit(1)

    outputs_dir = Path(sys.argv[1])
    mode = sys.argv[2]

    missing_repos = []
    missing_files = []
    total_expected = 0
    total_found = 0

    for entry in target_data:
        repo_id = str(entry["repo_id"])
        repo_id_int = int(repo_id)

        # Filter based on mode
        if mode == "original" and repo_id_int >= 55:
            continue
        elif mode == "large" and repo_id_int < 55:
            continue
        # mode == "all" includes everything

        selected_files = entry["selected_files"]
        total_expected += len(selected_files)

        repo_outputs_dir = outputs_dir / repo_id

        if not repo_outputs_dir.exists():
            missing_repos.append(repo_id)
            missing_files.extend([f"{repo_id}:{f}" for f in selected_files])
            continue

        # Check each expected file
        repo_missing = []
        for source_file in selected_files:
            output_filename = source_file.replace('/', '_')
            output_file = repo_outputs_dir / output_filename

            if output_file.exists():
                total_found += 1
            else:
                repo_missing.append(source_file)
                missing_files.append(f"{repo_id}:{source_file}")

        if repo_missing:
            missing_repos.append(repo_id)

    # Print results
    if missing_files:
        print(f"MISSING:{len(missing_repos)}:{total_found}:{total_expected}")
        for item in missing_files[:20]:  # Show first 20 missing files
            print(f"FILE:{item}")
        if len(missing_files) > 20:
            print(f"... and {len(missing_files) - 20} more missing files")
    else:
        print(f"COMPLETE:{total_found}:{total_expected}")

except Exception as e:
    print(f"ERROR:{e}", file=sys.stderr)
    sys.exit(1)
EOF
)

    if echo "$MISSING_CHECK" | grep -q "^ERROR:"; then
        ERROR_MSG=$(echo "$MISSING_CHECK" | grep "^ERROR:" | cut -d: -f2-)
        print_error "Failed to check outputs: $ERROR_MSG"
        exit 1
    elif echo "$MISSING_CHECK" | grep -q "^MISSING:"; then
        STATS=$(echo "$MISSING_CHECK" | grep "^MISSING:" | cut -d: -f2-)
        MISSING_REPOS=$(echo "$STATS" | cut -d: -f1)
        FOUND=$(echo "$STATS" | cut -d: -f2)
        EXPECTED=$(echo "$STATS" | cut -d: -f3)

        print_warning "Incomplete outputs detected!"
        echo "  Found: $FOUND/$EXPECTED files"
        echo "  Missing repos/files in: $MISSING_REPOS repos"
        echo ""
        echo "Missing files (will be counted as test failures):"
        echo "$MISSING_CHECK" | grep "^FILE:" | cut -d: -f2- | while read line; do
            echo "  - $line"
        done
        echo ""
        print_warning "Evaluation will proceed, missing files will have all tests marked as failed"
    else
        STATS=$(echo "$MISSING_CHECK" | grep "^COMPLETE:" | cut -d: -f2-)
        FOUND=$(echo "$STATS" | cut -d: -f1)
        EXPECTED=$(echo "$STATS" | cut -d: -f2)
        print_success "All outputs present: $FOUND/$EXPECTED files"
    fi
fi

# Check if test metadata directory exists
if [ ! -d "data/tests" ]; then
    print_error "Test metadata directory not found: data/tests"
    print_error "Test metadata is required for evaluation"
    exit 1
fi

# Check if test classifications directory exists
if [ ! -d "data/test_classifications" ]; then
    print_warning "Test classifications directory not found: data/test_classifications"
    print_warning "External/internal test metrics will not be available"
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running or not accessible"
    print_error "Docker is required to run tests in containers"
    print_error "Please start Docker and try again"
    exit 1
fi

print_success "Docker is running"

# Print configuration
echo ""
print_info "Configuration:"
echo "  Setting: $SETTING"
echo "  Model: $MODEL"
echo "  Outputs: $OUTPUTS_DIR"
echo "  Evaluation script: utils/evaluate.py"
if [ -n "$ADDITIONAL_ARGS" ]; then
    echo "  Additional args: $ADDITIONAL_ARGS"
fi
echo ""

# Confirm before running (can skip with -y flag)
if [ "$SKIP_CONFIRMATION" = false ]; then
    read -p "Do you want to proceed with evaluation? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Aborted by user"
        exit 0
    fi
fi

# Run the evaluation script
print_info "Starting evaluation..."
print_warning "This may take a while depending on the number of test files..."
echo "=================================="
echo ""

python3 utils/evaluate.py --setting "$SETTING" --model "$MODEL" $ADDITIONAL_ARGS

EXIT_CODE=$?

echo ""
echo "=================================="

if [ $EXIT_CODE -eq 0 ]; then
    print_success "Evaluation completed successfully!"
    print_info "Results saved to: results/$SETTING/$MODEL/"
    print_info "View overall statistics: results/$SETTING/$MODEL/overall_statistics.json"
else
    print_error "Evaluation failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
