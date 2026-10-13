<p align="center">
  <img src="assets/benchmark.png" alt="Benchmark Construction" width="600"/>
</p>

<h1 align="center">ReCUBE: Repository-Level Code Reconstruction Benchmark</h1>

<p align="center">
  <a href="https://huggingface.co/datasets/recube-anon-2026/recube-data">
    <img src="https://img.shields.io/badge/🤗_dataset-recube--data-yellow" alt="HuggingFace Dataset"/>
  </a>
  <a href="https://hub.docker.com/r/recubeanon2026/recube">
    <img src="https://img.shields.io/badge/🐳_docker-recube-blue" alt="Docker Images"/>
  </a>
</p>

A benchmark for evaluating LLMs on repository-level context utilization.

## Introduction

[This is an anonymized repository for peer review.]
ReCUBE evaluates LLMs' ability to reconstruct complete, functional code files within real-world repositories:

- **Repository-level understanding:** Full codebase context with multiple files and dependencies
- **35 functional subsets** from 18 repositories, spanning agent frameworks, LLM services, tool & MCP integration, RAG, code & vision, and voice & speech
- **297 target files** with usage-aware test suites (8,984 tests; external and internal)
- **ReCUBE-large**: 106 instances over 6 merged repositories with contexts up to 338K tokens
- **Docker-based evaluation** with reproducible, isolated test execution
- **External/internal test classification** for detailed capability analysis

## Quick Start

### Prerequisites

- Python 3.9+
- Docker (running)
- OpenAI API key (for API-based models) or vLLM (for open-source models)

### Installation

```bash
git clone THIS REPO
cd ReCUBE
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download benchmark data
pip install huggingface_hub
huggingface-cli download recube-anon-2026/recube-data --repo-type=dataset --local-dir data/

# Set API key (for API-based models)
export OPENAI_API_KEY="your-api-key-here"
```

### Generate & Evaluate

```bash
# 1. Generate outputs (choose one approach)
./scripts/run_api_key.sh agent_cce gpt-5              # Graph-guided agent
./scripts/run_api_key.sh agent_min_swe gpt-5          # Basic agent
./scripts/run_api_key.sh full-context_basic gpt-5     # Prompt-based

# 2. Run evaluation
./scripts/evaluate.sh agent_cce gpt-5

# 3. View results
cat results/agent_cce/gpt-5/overall_statistics.json
```

## Experimental Settings

| Setting | Description | Script |
|---------|-------------|--------|
| **agent_cce** | Graph-guided agent with dependency navigation | `utils/agent/cce/generate_output.py` |
| **agent_min_swe** | Agent with bash tools only | `utils/agent/min_swe/generate_output.py` |
| **full-context_basic** | Basic prompt with full repository context | `utils/full-context/basic/generate_output.py` |
| **full-context_cot** | Chain-of-Thought prompting | `utils/full-context/cot/generate_output.py` |

**For open-source models (vLLM):**
- Use `run_vllm.sh` with settings: `agent_cce_open_models`, `agent_min_swe_open_models`

## File Structure

```
ReCUBE/
├── config/                    # Agent configuration files
├── data/                      # Benchmark data (download from HuggingFace)
│   ├── graphs/               # Dependency graphs for CCE agent
│   ├── prompts/              # Repository context files
│   ├── tests/                # Test suites and metadata
│   ├── test_classifications/ # External/internal test labels
│   └── target.json           # 297 target files (35 functional subsets) + 106 large-scale instances
├── outputs/                   # Generated code outputs
│   └── {setting}/{model}/{repo_id}/
├── results/                   # Evaluation results
│   └── {setting}/{model}/
├── scripts/                   # Execution scripts
│   ├── run_api_key.sh        # Run with API-based models
│   ├── run_vllm.sh           # Run with vLLM (open-source models)
│   └── evaluate.sh           # Run evaluation
└── utils/                     # Implementation
    ├── evaluate.py           # Evaluation script
    ├── agent/                # Agent implementations
    │   ├── cce/             # CCE agent (graph-guided)
    │   └── min_swe/         # Min-SWE agent (basic)
    └── full-context/         # Prompt-based approaches
```

## Evaluation Metrics

Results are saved to `results/{setting}/{model}/overall_statistics.json`:

- **Average Pass Rate (APR):** Average pass rate across all instances
- **Strict Pass Rate (SPR):** Percentage of fully resolved instances (100% tests passed)
- **External/internal breakdown:** Tests classified by usage patterns (API contract vs implementation)
