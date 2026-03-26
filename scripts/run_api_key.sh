#!/bin/bash
# Generate model outputs for code reconstruction
# Usage: ./scripts/generate_model_output.sh <setting> <model> [additional_args...]

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
    setting         The experimental setting to use:
                    - full-context_basic    : Basic prompt-based reconstruction
                    - full-context_cot      : Chain-of-Thought prompt-based reconstruction
                    - agent_min_swe : Agent with min_swe config
                    - agent_cce : Agent with dependency graph

    model           The model to use for generation, e.g.:
                    - claude-3-haiku-20240307
                    - gpt-5-mini
                    - gpt-5
                    - gemini-2.5-flash
                    - gemini-2.5-pro

    additional_args Optional additional arguments to pass to the script

Examples:
    $0 full-context_basic claude-3-haiku-20240307
    $0 agent_cce gpt-5 --max-turns 100
    $0 full-context_cot gpt-5-mini --repo-ids 0,1,2

Available Settings:
    1. full-context_basic
       - Uses basic prompt with all context provided
       - Script: utils/full-context/basic/generate_output.py
       - Output: outputs/full-context_basic/{model}/

    2. full-context_cot
       - Uses Chain-of-Thought prompting for better reasoning
       - Script: utils/full-context/cot/generate_output.py
       - Output: outputs/full-context_cot/{model}/

    3. agent_min_swe
       - Uses mini-swe-agent with bash commands only
       - Script: utils/agent/min_swe/generate_output.py
       - Config: config/agent_min_swe.yaml
       - Output: outputs/agent_min_swe/{model}/

    4. agent_cce
       - Uses mini-swe-agent with dependency graph tools
       - Script: utils/agent/cce/generate_output.py
       - Config: config/agent_cce.yaml
       - Output: outputs/agent_cce/{model}/
       - Note: Requires dependency graphs in data/graphs/

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
ADDITIONAL_ARGS="$@"

# Get project root (script is in scripts/, so go up one level)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

print_info "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Validate setting and determine which script to run
case $SETTING in
    full-context_basic)
        SCRIPT_PATH="utils/full-context/basic/generate_output.py"
        print_info "Setting: Basic Prompt-based Reconstruction"
        ;;

    full-context_cot)
        SCRIPT_PATH="utils/full-context/cot/generate_output.py"
        print_info "Setting: Chain-of-Thought Prompt-based Reconstruction"
        ;;

    agent_min_swe)
        SCRIPT_PATH="utils/agent/min_swe/generate_output.py"
        print_info "Setting: Agent (Min-SWE Config)"
        print_warning "This requires Docker to be running"
        ;;

    agent_cce)
        SCRIPT_PATH="utils/agent/cce/generate_output.py"
        print_info "Setting: Agent (CCE-based)"
        print_warning "This requires Docker to be running"
        print_warning "This requires dependency graphs in data/graphs/"
        ;;

    *)
        print_error "Unknown setting: $SETTING"
        echo ""
        usage
        exit 1
        ;;
esac

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    print_error "Script not found: $SCRIPT_PATH"
    exit 1
fi

print_success "Found script: $SCRIPT_PATH"

# Check if required data directories exist
if [ ! -d "data/prompts" ]; then
    print_error "Data directory not found: data/prompts"
    exit 1
fi

if [ ! -f "data/target.json" ]; then
    print_error "Target file not found: data/target.json"
    exit 1
fi

# For cce-based setting, check if graphs directory exists
if [ "$SETTING" = "agent_cce" ]; then
    if [ ! -d "data/graphs" ]; then
        print_error "Graphs directory not found: data/graphs"
        print_error "Please run the graph generation script first"
        exit 1
    fi
fi

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    print_error "OPENAI_API_KEY environment variable not set"
    print_error "Please set it with: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

# Print configuration
echo ""
print_info "Configuration:"
echo "  Setting: $SETTING"
echo "  Model: $MODEL"
echo "  Script: $SCRIPT_PATH"
if [ -n "$ADDITIONAL_ARGS" ]; then
    echo "  Additional args: $ADDITIONAL_ARGS"
fi
echo ""

# Confirm before running (can skip with -y flag)
if [[ ! " $ADDITIONAL_ARGS " =~ " -y " ]] && [[ ! " $ADDITIONAL_ARGS " =~ " --yes " ]]; then
    read -p "Do you want to proceed? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Aborted by user"
        exit 0
    fi
fi

# Run the script
print_info "Starting generation..."
echo "=================================="
echo ""

export PYTHONPATH="$PROJECT_ROOT"
python3 "$SCRIPT_PATH" --model "$MODEL" $ADDITIONAL_ARGS

EXIT_CODE=$?

echo ""
echo "=================================="

if [ $EXIT_CODE -eq 0 ]; then
    print_success "Generation completed successfully!"
    print_info "Check outputs in: outputs/$SETTING/$MODEL/"
else
    print_error "Generation failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
