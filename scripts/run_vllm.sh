#!/bin/bash
# Run inference with vLLM server for open-source models
# Usage: ./scripts/run_vllm.sh <setting> <model> [additional_args...]

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
                    - agent_min_swe_open_models : Agent with min_swe config (open models)
                    - agent_cce_open_models : Agent with dependency graph (open models)

    model           The model to use for generation (HuggingFace model name), e.g.:
                    - Qwen/Qwen3-Coder-30B-A3B-Instruct
                    - deepseek-ai/DeepSeek-V3
                    - meta-llama/Llama-3.3-70B-Instruct

    additional_args Optional additional arguments to pass to the script

Examples:
    $0 agent_cce_open_models Qwen/Qwen3-Coder-30B-A3B-Instruct
    $0 agent_cce_open_models Qwen/Qwen3-Coder-30B-A3B-Instruct --max-turns 100
    $0 full-context_cot Qwen/Qwen3-Coder-30B-A3B-Instruct --repo-ids 0,1,2

Available Settings:
    1. full-context_basic
       - Uses basic prompt with all context provided
       - Script: utils/full-context/basic/generate_output.py
       - Output: outputs/full-context_basic/{model}/

    2. full-context_cot
       - Uses Chain-of-Thought prompting for better reasoning
       - Script: utils/full-context/cot/generate_output_open_source.py
       - Output: outputs/full-context_cot/{model}/

    3. agent_min_swe_open_models
       - Uses agent with bash commands only
       - Script: utils/agent/min_swe_open_models/generate_output.py
       - Output: outputs/agent_min_swe/{model}/

    4. agent_cce_open_models
       - Uses agent with dependency graph tools
       - Script: utils/agent/cce_open_models/generate_output.py
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
        SCRIPT_PATH="utils/full-context/cot/generate_output_open_source.py"
        print_info "Setting: Chain-of-Thought Prompt-based Reconstruction (Open Source)"
        ;;

    agent_min_swe_open_models)
        SCRIPT_PATH="utils/agent/min_swe_open_models/generate_output.py"
        print_info "Setting: Agent (Min-SWE Config - Open Models)"
        print_warning "This requires Docker to be running"
        ;;

    agent_cce_open_models)
        SCRIPT_PATH="utils/agent/cce_open_models/generate_output.py"
        print_info "Setting: Agent (CCE-based - Open Models)"
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
    print_warning "Data directory not found: data/prompts"
    print_info "Downloading data from Hugging Face..."
    pip install -q huggingface_hub
    huggingface-cli download wlqmfl1999/recube-data --repo-type=dataset --local-dir data/
fi

if [ ! -f "data/target.json" ]; then
    print_error "Target file not found: data/target.json"
    print_info "Please download data first using: huggingface-cli download wlqmfl1999/recube-data --repo-type=dataset --local-dir data/"
    exit 1
fi

# For cce-based setting, check if graphs directory exists
if [[ "$SETTING" == *"cce"* ]]; then
    if [ ! -d "data/graphs" ]; then
        print_error "Graphs directory not found: data/graphs"
        print_error "Please download data using: huggingface-cli download wlqmfl1999/recube-data --repo-type=dataset --local-dir data/"
        exit 1
    fi
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

# Start vLLM server if not already running
VLLM_PORT=8000
VLLM_HOST="127.0.0.1"
API_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"

if ! curl -s "${API_BASE_URL}/models" > /dev/null 2>&1; then
    print_info "Starting vLLM server for model: $MODEL"
    print_warning "This will take a few minutes..."

    # Kill any existing vLLM processes
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 2

    # Start vLLM server in background
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --host "$VLLM_HOST" \
        --port "$VLLM_PORT" &

    VLLM_PID=$!

    # Cleanup on exit
    trap "kill $VLLM_PID 2>/dev/null" EXIT

    # Wait for server to be ready
    print_info "Waiting for vLLM server to be ready..."
    for i in {1..100}; do
        if curl -s "${API_BASE_URL}/models" > /dev/null 2>&1; then
            print_success "vLLM server ready!"
            break
        fi
        if [ $i -eq 100 ]; then
            print_error "vLLM server failed to start after 200 seconds"
            exit 1
        fi
        sleep 2
    done
else
    print_success "vLLM server already running at ${API_BASE_URL}"
fi

# Run the script
print_info "Starting generation..."
echo "=================================="
echo ""

export PYTHONPATH="$PROJECT_ROOT"
export OPENAI_API_KEY="EMPTY"  # vLLM doesn't need real API key
export OPENAI_BASE_URL="$API_BASE_URL"

python3 "$SCRIPT_PATH" --model "$MODEL" $ADDITIONAL_ARGS

EXIT_CODE=$?

echo ""
echo "=================================="

if [ $EXIT_CODE -eq 0 ]; then
    print_success "Generation completed successfully!"
    # Extract output directory from setting name
    OUTPUT_SETTING=$(echo "$SETTING" | sed 's/_open_models$//')
    print_info "Check outputs in: outputs/${OUTPUT_SETTING}/${MODEL}/"
else
    print_error "Generation failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
