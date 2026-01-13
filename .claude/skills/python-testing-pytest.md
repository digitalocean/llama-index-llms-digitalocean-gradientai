---
skill: python-testing-pytest
category: testing
tags: [pytest, testing, async, mocking]
updated: 2026-01-13
---
# Python Testing with Pytest

## Overview
Comprehensive guide to writing and maintaining pytest tests for the LlamaIndex Gradient AI integration project.

## When to Use This Skill
Reference when writing new tests, debugging test failures, or setting up test fixtures and mocks.

## Test Structure

### Test File Organization
- Tests live in `tests/` directory
- Test files follow `test_*.py` naming convention
- Test classes follow `Test*` naming convention
- Test functions follow `test_*` naming convention

### Configuration
Pytest configuration is in `pyproject.toml` under `[tool.pytest.ini_options]`. Key settings:
- `--asyncio-mode=auto`: Automatic async test detection
- `--cov`: Coverage enabled by default
- Custom markers: `@pytest.mark.integration`, `@pytest.mark.slow`

## Test Patterns

### Environment Variable Setup
```python
import os
from dotenv import load_dotenv

load_dotenv()
REQUIRED_ENV = "MODEL_ACCESS_KEY"

def _skip_if_no_creds():
    if not os.getenv(REQUIRED_ENV):
        pytest.skip(
            f"Live Gradient credentials required (set {REQUIRED_ENV}); skipping live integration test",
            allow_module_level=False,
        )
```

### Test Fixtures
```python
def _make_llm():
    _skip_if_no_creds()
    model = os.getenv("GRADIENT_MODEL", "openai-gpt-oss-120b")
    api_key = os.environ[REQUIRED_ENV]
    return GradientAI(
        model=model,
        model_access_key=api_key,
        timeout=30,
    )
```

## Testing Sync Methods

### Basic Completion Test
```python
def test_complete_and_chat_sync():
    llm = _make_llm()
    
    completion = llm.complete("Say 'hello' in one word.")
    assert completion.text
    
    chat = llm.chat([ChatMessage(role="user", content="Say 'ping' once.")])
    assert chat.message.content
```

### Streaming Tests
```python
def test_stream_complete_and_chat_sync():
    llm = _make_llm()
    
    chunks = list(llm.stream_complete("Say 'stream' in two short parts."))
    assert chunks, "stream_complete returned no chunks"
    assert chunks[-1].text
    
    messages = [ChatMessage(role="user", content="Answer with one short word, streamed.")]
    responses = list(llm.stream_chat(messages))
    assert responses, "stream_chat returned no chunks"
    assert responses[-1].message.content
```

## Testing Async Methods

### Async Test Decorator
```python
@pytest.mark.asyncio
async def test_complete_and_chat_async():
    llm = _make_llm()
    
    completion = await llm.acomplete("Say 'async' in one word.")
    assert completion.text
    
    chat = await llm.achat([ChatMessage(role="user", content="Answer with 'pong'.")])
    assert chat.message.content
```

### Async Streaming Tests
```python
@pytest.mark.asyncio
async def test_stream_complete_and_chat_async():
    llm = _make_llm()
    
    chunks = []
    async for delta in llm.astream_complete("Stream two short pieces."):
        chunks.append(delta.delta)
    assert chunks, "astream_complete returned no chunks"
    assert "".join(chunks)
```

## Testing Function/Tool Calling

### Chat with Tools Test
```python
def test_chat_with_tools():
    llm = _make_llm()
    tools = [FunctionTool.from_defaults(fn=add), FunctionTool.from_defaults(fn=multiply)]
    
    response = llm.chat_with_tools(
        tools=tools,
        user_msg="What is 5 multiplied by 8?",
    )
    
    assert response is not None
    assert response.message is not None
```

### Tool Call Extraction Test
```python
def test_get_tool_calls_from_response():
    llm = _make_llm()
    tools = [FunctionTool.from_defaults(fn=add)]
    
    response = llm.chat_with_tools(
        tools=tools,
        user_msg="Calculate 10 plus 15 using the add function.",
        tool_required=True,
    )
    
    tool_calls = llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)
    assert isinstance(tool_calls, list)
```

### Predict and Call Test (with Error Handling)
```python
def test_predict_and_call():
    llm = _make_llm()
    tools = [FunctionTool.from_defaults(fn=add)]
    
    try:
        response = llm.predict_and_call(
            tools=tools,
            user_msg="Use the add function to calculate 10 plus 15.",
            tool_required=True,
        )
        assert response is not None
    except ValueError as e:
        if "Expected at least one tool call" in str(e):
            pytest.skip("Model chose not to use tools for this query")
        raise
```

## Mocking for Unit Tests

### Mocking the Gradient API
For fast unit tests without hitting the real API:

```python
from unittest.mock import Mock, patch, MagicMock

def test_complete_with_mock():
    """Test completion without hitting real API."""
    with patch('gradient.Gradient') as mock_gradient:
        # Setup mock client
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Mocked response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create.return_value = mock_response
        mock_gradient.return_value = mock_client

        # Test
        llm = GradientAI(model="test-model", model_access_key="fake-key")
        response = llm.complete("test prompt")

        assert response.text == "Mocked response"
        mock_client.chat.completions.create.assert_called_once()
```

### Mocking Streaming Responses
```python
def test_stream_complete_with_mock():
    """Test streaming without real API."""
    with patch('gradient.Gradient') as mock_gradient:
        mock_client = Mock()

        # Create mock stream chunks
        chunks = []
        for text in ["Hello", " ", "world"]:
            chunk = Mock()
            chunk.choices = [Mock(delta=Mock(content=text))]
            chunks.append(chunk)

        mock_client.chat.completions.create.return_value = iter(chunks)
        mock_gradient.return_value = mock_client

        llm = GradientAI(model="test", model_access_key="fake")
        results = list(llm.stream_complete("test"))

        assert len(results) == 3
        assert results[-1].text == "Hello world"
```

### Mocking Tool Calls
```python
def test_tool_calling_with_mock():
    """Test tool calling without real API."""
    with patch('gradient.Gradient') as mock_gradient:
        mock_client = Mock()
        mock_response = Mock()

        # Mock tool call response
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_tool_call.function = Mock(
            name="add",
            arguments='{"a": 5, "b": 3}'
        )

        mock_message = Mock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        mock_message.role = "assistant"

        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        mock_gradient.return_value = mock_client

        from llama_index.core.tools import FunctionTool

        def add(a: int, b: int) -> int:
            return a + b

        llm = GradientAI(model="test", model_access_key="fake")
        response = llm.chat_with_tools(
            tools=[FunctionTool.from_defaults(fn=add)],
            user_msg="What is 5 + 3?"
        )

        tool_calls = llm.get_tool_calls_from_response(response)
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "add"
        assert tool_calls[0].tool_kwargs == {"a": 5, "b": 3}
```

## Testing Error Cases

### API Errors
```python
def test_api_error_handling():
    """Test handling of API errors."""
    with patch('gradient.Gradient') as mock_gradient:
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_gradient.return_value = mock_client

        llm = GradientAI(model="test", model_access_key="fake")

        with pytest.raises(Exception, match="API Error"):
            llm.complete("test")
```

### No Tool Calls
```python
def test_no_tool_calls():
    """Test when model doesn't call any tools."""
    with patch('gradient.Gradient') as mock_gradient:
        mock_client = Mock()
        mock_message = Mock(content="I don't need tools", tool_calls=None, role="assistant")
        mock_response = Mock(choices=[Mock(message=mock_message)])
        mock_client.chat.completions.create.return_value = mock_response
        mock_gradient.return_value = mock_client

        llm = GradientAI(model="test", model_access_key="fake")

        from llama_index.core.tools import FunctionTool
        def add(a: int, b: int) -> int:
            return a + b

        response = llm.chat_with_tools(
            tools=[FunctionTool.from_defaults(fn=add)],
            user_msg="Hello"
        )

        # Should not error when error_on_no_tool_call=False
        tool_calls = llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)
        assert tool_calls == []
```

## Best Practices

1. **Skip Integration Tests When Credentials Missing**: Use `_skip_if_no_creds()` pattern
2. **Mock for Unit Tests**: Use mocks for fast tests, real API for integration tests
3. **Test Both Sync and Async**: Ensure parity between implementations
4. **Handle Model Variability**: Models may not always use tools - test gracefully
5. **Test Error Cases**: Mock API errors, missing tool calls, malformed responses
6. **Use Descriptive Assertions**: Include helpful error messages

## Coverage Configuration

Coverage configuration is in `pyproject.toml` under `[tool.coverage.run]` and `[tool.coverage.report]`. Reports are generated in HTML, XML, and terminal formats.

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run specific test file
pytest tests/test_gradient_llm.py

# Run specific test
pytest tests/test_gradient_llm.py::test_complete_and_chat_sync

# Run async tests only
pytest -m asyncio

# Run with verbose output
pytest -v
```

