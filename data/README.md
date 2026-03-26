---
license: mit
task_categories:
- text-generation
- question-answering
language:
- code
tags:
- code
- benchmark
- repository-level
- code-reconstruction
size_categories:
- n<1K
configs:
- config_name: target
  data_files: "target.json"
- config_name: graphs
  data_files: "graphs/*.pkl"
- config_name: prompts
  data_files: "prompts/*.txt"
---

# Data

This directory contains all benchmark data for the Re2Code repository-level code reconstruction benchmark.

## Download

All data files are hosted on Hugging Face and can be downloaded using:

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Download the entire dataset
huggingface-cli download wlqmfl1999/recube-data --repo-type=dataset --local-dir data/

# Or download in Python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="wlqmfl1999/recube-data", repo_type="dataset", local_dir="data/")
```

Alternatively, you can clone the repository:
```bash
git clone https://huggingface.co/datasets/wlqmfl1999/recube-data data/
```

## Graphs (`graphs/`)

Dependency graphs in NetworkX format (`.pkl` files) for each functional subset.

**Usage**: Used by the graph-guided setting (`bash_only_graph_setting`) to provide efficient context navigation tools.

Format: NetworkX MultiDiGraph serialized with pickle

- Node Types:
  - Directory nodes (e.g., /, app/, app/agent/)
  - File nodes with full source code (e.g., app/agent/base.py)
  - Class definitions with code snippet (e.g., app/agent/base.py:BaseAgent)
  - Function/method definitions with code snippet (e.g., app/agent/base.py:initialize_agent)

- Node Attributes:
  - Node type (directory/file/class/function)
  - Full source code (for files) or code snippet (for classes/functions)
  - Line number range (for classes/functions only)

- Edge Types:
  - contains: Hierarchical containment (directory→file, file→class, class→method, file→function)
  - imports: Import dependencies (file→imported entity)
  - invokes: Function call dependencies (function→called function/class)
  - inherits: Class inheritance (child class→base class)

- Node ID Format:
  - Directories: Path string (e.g., app/agent)
  - Files: File path (e.g., app/agent/base.py)
  - Classes: {file_path}:{class_name} (e.g., app/agent/base.py:BaseAgent)
  - Methods: {file_path}:{class_name}.{method_name} (e.g., app/agent/base.py:BaseAgent.run)
  - Functions: {file_path}:{function_name} (e.g., app/utils.py:setup_logger)

**Files**: One graph per functional subset (e.g., `0.pkl`, `1.pkl`, ..., `54.pkl`)

## Prompts (`prompts/`)

Repository context files in text format (`.txt`) for each functional subset. Formatted repository context including:
- README documentation
- Package dependencies
- All implemented files with complete code

**Usage**: Provided as input context to `prompt` experimental settings

**Format**: Structured text with XML-like tags:
```
<readme>
... repository documentation ...
</readme>

<dependencies>
... pip package list with versions ...
</dependencies>

<implementations>
## path/to/file.py
```python
... complete implementation ...
```
</implementations>
```

**Files**: One prompt file per functional subset (e.g., `0.txt`, `1.txt`, ..., `54.txt`)

## Testcases (`tests/`)

Test files for evaluating generated code. These files are already embedded in Docker images.

**Content**: pytest test files for each target file
- Unit tests for individual functions/classes
- Integration tests for multi-component interactions
- Real usage patterns extracted from original repositories

**Structure**:
```
tests/
├── 0/                          # Functional subset 0
│   ├── app_agent_base_test.py
│   ├── app_agent_manus_test.py
|   ├── ...
│   └── test_metadata.json
├── 1/                          # Functional subset 1
│   └── ...
```

**Note**: These files are copied into Docker images (`wlqmfl0990/recube:{repo_id}`) and do not need to be manually managed during evaluation.

### `test_metadata.json`
```json
{
  "repo_id": 0,
  "repo_url": "https://github.com/...",
  "tests_generated": "2026-01-05",
  "instances": [
    {
      "file": "app/agent/base.py",
      "test_file": "app_agent_base_test.py",
      "total_tests": 47,
      "functions_tested": 8,
      "validation_status": "pending",
      "real_usage_sources": { ... },
      "test_categories": {
        "unit_tests": 44,
        "integration_tests": 3
      },
      "difficulty_factors": [ ... ]
    }
  ],
  "validation": {
    "status": "passed",
    "tests_passed": 282,
    "tests_failed": 0
  }
}
```

**Usage**: Loaded during evaluation to map source files to test files and count expected tests.

## Test Classifications (`test_classifications/`)

External vs internal test classifications for analyzing model capabilities.

**Structure**: One JSON file per functional subset (e.g., `0.json`, `1.json`)

**Content**:
```json
{
  "repo_id": 0,
  "files": {
    "app/agent/base.py": {
      "test_file": "app_agent_base_test.py",
      "test_classifications": {
        "test_initialize_agent": "internal",
        "test_state_context": "external",
        "test_update_memory": "external",
        ...
      }
    }
  }
}
```

**Test Types**:
- **External**: Tests that import and use the target file (API contract tests)
- **Internal**: Tests that are within the target file itself (implementation tests)

**Usage**: Used during evaluation to calculate external/internal pass rates separately.

## Target Files

### `target.json`

366 verified target files used for official evaluation (i.e., 40 functional subset, 366 instances).

**Format**: Same as `target.json`

**Usage**: Used by generation and evaluation scripts to determine which files to process.

## Functional Subset Breakdown

Detailed breakdown of the 40 functional subsets in our benchmark. Each entry represents a specific functional subset of a larger repository, selected to target distinct development capabilities.

| ID | Repository | Functionality | Key Components |
|----|------------|---------------|----------------|
| 0 | OpenManus | Agent orchestration with sandbox execution | Base agent, ReAct pattern, tool execution |
| 1 | nanochat | GPT training infrastructure | SFT/RL training, data loading, checkpointing |
| 2 | DeepSeek-OCR | Vision-based OCR system | CLIP/SAM encoders, image/PDF processing |
| 3 | deer-flow | Workflow-based agent with RAG | Graph orchestration, web crawling, retrieval |
| 4 | openai-agents | Core runtime with guardrails | Agent execution, function schemas, prompts |
| 5 | openai-agents | Computer use & file editing | Editor interface, computer control |
| 6 | openai-agents | Multi-agent handoffs & visualization | Handoff filtering, workflow visualization |
| 7 | openai-agents | MCP server with persistent memory | MCP integration, SQLite sessions |
| 8 | openai-agents | Streaming chat completions | Stream handling, chat conversion |
| 9 | openai-agents | Tracing & observability | Span data, trace processors, providers |
| 10 | openai-agents | Error handling & computer control | Error tracing, guardrails |
| 11 | openai-agents | Real-time voice agent | Realtime API, audio formats, voice handoffs |
| 12 | openai-agents | Voice pipeline (STT/TTS) | Speech-to-text, text-to-speech workflow |
| 13 | serena | Semantic code editing | Symbol-level operations, code editor |
| 14 | serena | LSP server implementation | Protocol handler, IDE integration |
| 15 | blender-mcp | Blender MCP server | 3D modeling control, telemetry |
| 16 | VibeVoice | Streaming voice generation | Diffusion models, DPM solver |
| 17 | deepwiki-open | Multi-provider LLM API | OpenAI/DashScope/Ollama, embeddings |
| 18 | context-engineering | Multi-agent system with RAG | Research agent, document ingestion |
| 19 | context-engineering | Standalone RAG agent | CLI, embedding pipeline, DB utilities |
| 20 | DeepCode | Multi-interface coding assistant | CLI & Streamlit UI, workflows |
| 21 | DeepCode | Code implementation agent | MCP tools, Git ops, code indexing |
| 22 | fastapi_mcp | FastAPI MCP server | HTTP/SSE transports, Auth0 |
| 23 | RAG-Anything | Document processing RAG | Parser, processor, query system |
| 24 | memvid | Video chat with retrieval | Indexing, encoding, ffmpeg processing |
| 25 | trae-agent | Software engineering agent | Docker execution, bash/edit tools, CKG |
| 26 | trae-agent | Trae with MCP integration | MCP tools, code knowledge graph |
| 27 | langextract | Structured extraction with plugins | Multi-provider, Gemini/Ollama |
| 28 | strix | Core agent with LLM management | Memory compression, request queuing |
| 29 | strix | Tool runtime with Docker | File editing, web search, thinking tools |
| 30 | strix | Interactive tool managers | Python/terminal/proxy/graph sessions |
| 31 | strix | Browser automation & CLI | Browser control, tab management |
| 32 | strix | Tool rendering system | UI component renderers registry |
| 33 | strix | Terminal user interface | Text-based UI, utilities |
| 34 | Wan2.1 | Video generation models | Text2video, image2video, VAE |
| 35 | Second-Me | Personalized LLM data generation | L0/L1 generators, DPO training |
| 36 | Second-Me | GGUF model utilities | Format conversion, quantization |
| 37 | Second-Me | Personalized AI API services | Chat/knowledge/role management |
| 38 | Second-Me | Complete AI application backend | Document processing, vector storage, MCP |
| 39 | Spark-TTS | Neural TTS with tokenization | BiCodec, FSQ, Triton runtime |

**Diversity**:
- 20 unique base repositories & 40 functional subsets
- Domains: AI agents, ML training, OCR, TTS, RAG, code editing, MCP servers