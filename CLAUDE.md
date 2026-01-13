# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LlamaIndex integration package for DigitalOcean Gradient AI, providing LLM capabilities with full support for function/tool calling. The package wraps the official `gradient` SDK and implements the LlamaIndex LLM interface.

**Package name**: `llama-index-llms-digitalocean-gradientai`
**Main class**: `GradientAI` (extends `FunctionCallingLLM` from LlamaIndex)

## Development Commands

### Installation
```bash
# Install package in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Testing
```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_gradient_llm.py

# Run specific test function
pytest tests/test_gradient_llm.py::test_complete_and_chat_sync

# Run tests without coverage reports
pytest -v --no-cov
```

**Important**: Integration tests require `MODEL_ACCESS_KEY` environment variable (Gradient API key). Tests are skipped automatically if credentials are not found. Optionally set `GRADIENT_MODEL` to override the default model (`openai-gpt-oss-120b`).

### Linting and Formatting
```bash
# Format code with black (line length: 100)
black .

# Lint with ruff
ruff check .

# Auto-fix ruff issues
ruff check . --fix

# Type check with mypy
mypy llama_index/llms/digitalocean/gradientai

# Run all pre-commit hooks
pre-commit run --all-files
```

### Building and Publishing
```bash
# Build distribution packages
python -m build

# Publish to PyPI (requires credentials)
python -m twine upload dist/*
```

## Code Architecture

### Package Structure
```
llama_index/llms/digitalocean/gradientai/
├── __init__.py     # Exports GradientAI and backward-compat alias
└── base.py         # Main GradientAI implementation
```

### GradientAI Class (base.py)

The `GradientAI` class in `llama_index/llms/digitalocean/gradientai/base.py` is the core implementation. Key architectural points:

**Inheritance**: Extends `FunctionCallingLLM` from LlamaIndex core, which provides the interface for LLMs with tool/function calling capabilities.

**Client Management**:
- `_client` property: Returns synchronous `Gradient` client
- `_async_client` property: Returns asynchronous `AsyncGradient` client
- Clients are created on-demand with configured API key, base URL, and timeout

**Message Format Conversion**:
- `_format_messages()`: Converts LlamaIndex `ChatMessage` objects to OpenAI-compatible format expected by Gradient API
- Handles both simple text messages and complex messages with tool calls
- Extracts content from `TextBlock` and tool calls from `ToolCallBlock` within message blocks

**Response Conversion**:
- `_to_chat_message()`: Converts Gradient SDK responses back to LlamaIndex `ChatMessage` format
- Handles both dict and object response formats from the SDK
- Builds message blocks from response content and tool calls

**Tool/Function Calling**:
- `_prepare_chat_with_tools()`: Converts LlamaIndex tools to OpenAI-compatible tool specifications
- `get_tool_calls_from_response()`: Extracts tool calls from responses (supports both modern block-based and legacy format)
- `_resolve_tool_choice()`: Resolves tool choice strings ("auto", "required", "none") or specific tool names to API format

**Core Methods**:
- Implements both sync and async versions of: `complete`, `chat`, `stream_complete`, `stream_chat`
- All methods use `_get_request_payload()` to build API requests with proper formatting
- Streaming methods use `_extract_delta()` to parse incremental response chunks

### Key Design Patterns

**Dual Format Support**: The code handles both object-based responses (from newer SDK versions) and dict-based responses (for backward compatibility) by checking with `hasattr()` and `isinstance()` before accessing fields.

**Tool Call Extraction**: Supports extracting tool calls from two locations:
1. Modern path: From `ToolCallBlock` objects in `message.blocks`
2. Legacy path: From `message.additional_kwargs["tool_calls"]`

**Streaming Delta Extraction**: The `_extract_delta()` method gracefully handles various response formats, trying `delta.content` first (streaming format), then falling back to `message.content` (non-streaming format).

## Configuration

**pyproject.toml** defines:
- Line length: 100 characters (Black, Ruff)
- Python support: 3.8-3.12
- Test coverage excludes: tests, abstract methods, type checking blocks
- Pytest configured with: coverage reports (term, HTML, XML), async mode auto, strict markers

**pre-commit hooks** run:
- Black formatting (line-length=100)
- Ruff linting with auto-fix
- Mypy type checking
- Standard pre-commit hooks (trailing whitespace, YAML/JSON/TOML validation, etc.)

## Backward Compatibility

The package maintains a backward-compatible alias: `DigitalOceanGradientAILLM = GradientAI` in `__init__.py` for users who may have been using the old class name.
