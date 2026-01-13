---
skill: python-async-patterns
category: patterns
tags: [python, async, asyncio, patterns]
updated: 2026-01-13
---
# Python Async Patterns

## Overview
This project implements both sync and async versions of all LLM methods. The key pattern is maintaining separate `_client` and `_async_client` properties that create clients on-demand.

## When to Use This Skill
Reference when implementing async methods, debugging async issues, or ensuring sync/async parity.

## Key Pattern: Dual Client Properties

The sync/async split happens at the client level, not the method level. This allows code reuse in helpers like `_get_request_payload()` and `_extract_delta()`.

```python
@property
def _client(self) -> Gradient:
    """Synchronous Gradient client."""
    return Gradient(
        model_access_key=self.model_access_key,
        base_url=self.base_url,
        timeout=self.timeout,
    )

@property
def _async_client(self) -> AsyncGradient:
    """Asynchronous Gradient client."""
    return AsyncGradient(
        model_access_key=self.model_access_key,
        base_url=self.base_url,
        timeout=self.timeout,
    )
```

**Key insight**: Helper methods like `_get_request_payload()`, `_format_messages()`, `_to_chat_message()`, and `_extract_delta()` are all synchronous and shared between sync and async methods. Only the actual API calls differ.

## Method Pairs
Always implement both sync and async versions:

**Sync:**
```python
def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
    response = self._client.chat.completions.create(**payload)
    return CompletionResponse(text=text)
```

**Async:**
```python
async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
    response = await self._async_client.chat.completions.create(**payload)
    return CompletionResponse(text=text)
```

## Streaming Patterns

### Sync Streaming
```python
def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
    payload = self._get_request_payload(prompt, **kwargs)
    payload["stream"] = True
    stream = self._client.chat.completions.create(**payload)
    text = ""
    for completion in stream:
        delta = self._extract_delta(completion)
        if delta:
            text += delta
            yield CompletionResponse(text=text, delta=delta)
```

### Async Streaming
```python
async def astream_complete(
    self, prompt: str, **kwargs: Any
) -> CompletionResponseAsyncGen:
    payload = self._get_request_payload(prompt, **kwargs)
    payload["stream"] = True
    stream = await self._async_client.chat.completions.create(**payload)
    text = ""
    async for completion in stream:
        delta = self._extract_delta(completion)
        if delta:
            text += delta
            yield CompletionResponse(text=text, delta=delta)
```

## Chat Streaming

### Sync Chat Streaming
```python
def stream_chat(
    self, messages: Sequence[ChatMessage], **kwargs: Any
) -> ChatResponseGen:
    payload = self._get_request_payload("", messages=messages, **kwargs)
    payload["stream"] = True
    stream = self._client.chat.completions.create(**payload)
    text = ""
    for completion in stream:
        delta = self._extract_delta(completion)
        if delta:
            text += delta
            yield ChatResponse(
                message=ChatMessage(role="assistant", content=text),
                delta=delta
            )
```

### Async Chat Streaming
```python
async def astream_chat(
    self, messages: Sequence[ChatMessage], **kwargs: Any
) -> ChatResponseAsyncGen:
    payload = self._get_request_payload("", messages=messages, **kwargs)
    payload["stream"] = True
    stream = await self._async_client.chat.completions.create(**payload)
    text = ""
    async for completion in stream:
        delta = self._extract_delta(completion)
        if delta:
            text += delta
            yield ChatResponse(
                message=ChatMessage(role="assistant", content=text),
                delta=delta
            )
```

## Function Calling with Async

### Async Chat with Tools
```python
async def achat_with_tools(
    self,
    tools: Sequence["BaseTool"],
    user_msg: Optional[Union[str, ChatMessage]] = None,
    chat_history: Optional[List[ChatMessage]] = None,
    verbose: bool = False,
    allow_parallel_tool_calls: bool = False,
    tool_required: bool = False,
    tool_choice: Optional[Union[str, dict]] = None,
    **kwargs: Any,
) -> ChatResponse:
    # Prepare request
    chat_kwargs = self._prepare_chat_with_tools(
        tools=tools,
        user_msg=user_msg,
        chat_history=chat_history,
        verbose=verbose,
        allow_parallel_tool_calls=allow_parallel_tool_calls,
        tool_required=tool_required,
        tool_choice=tool_choice,
        **kwargs,
    )
    
    # Make async call
    response = await self.achat(**chat_kwargs)
    
    # Validate response
    return self._validate_chat_with_tools_response(
        response, tools, allow_parallel_tool_calls, **kwargs
    )
```

### Async Predict and Call
```python
async def apredict_and_call(
    self,
    tools: Sequence["BaseTool"],
    user_msg: Optional[Union[str, ChatMessage]] = None,
    chat_history: Optional[List[ChatMessage]] = None,
    verbose: bool = False,
    allow_parallel_tool_calls: bool = False,
    tool_required: bool = False,
    tool_choice: Optional[Union[str, dict]] = None,
    **kwargs: Any,
) -> Any:
    # Chat with tools
    response = await self.achat_with_tools(
        tools=tools,
        user_msg=user_msg,
        chat_history=chat_history,
        verbose=verbose,
        allow_parallel_tool_calls=allow_parallel_tool_calls,
        tool_required=tool_required,
        tool_choice=tool_choice,
        **kwargs,
    )
    
    # Extract and execute tool calls
    tool_calls = self.get_tool_calls_from_response(
        response, error_on_no_tool_call=tool_required
    )
    
    # Execute tools and return results
    # ...
```

## Testing Async Code

### Pytest Async Tests
```python
import pytest

@pytest.mark.asyncio
async def test_complete_and_chat_async():
    llm = _make_llm()
    
    completion = await llm.acomplete("Say 'async' in one word.")
    assert completion.text
    
    chat = await llm.achat([ChatMessage(role="user", content="Answer with 'pong'.")])
    assert chat.message.content
```

### Testing Async Generators
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

### Pytest Configuration
Pytest is configured with `--asyncio-mode=auto` in `pyproject.toml` for automatic async test detection.

### Testing Async Streaming Specifically
```python
@pytest.mark.asyncio
async def test_async_stream_accumulation():
    """Test that async streaming properly accumulates text."""
    llm = _make_llm()

    full_text = ""
    chunk_count = 0

    async for response in llm.astream_complete("Count to 5"):
        chunk_count += 1
        # Each response should have accumulated text
        assert len(response.text) >= len(full_text)
        full_text = response.text
        # Each response should have a delta
        assert response.delta

    assert chunk_count > 0
    assert len(full_text) > 0

@pytest.mark.asyncio
async def test_async_stream_chat_blocks():
    """Test async streaming with message blocks."""
    llm = _make_llm()

    messages = [ChatMessage(role="user", content="Hello")]

    last_response = None
    async for response in llm.astream_chat(messages):
        last_response = response
        # Should always have a message
        assert response.message
        assert response.message.role == "assistant"

    assert last_response is not None
```

## Error Handling

### Async Error Handling
```python
async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
    try:
        payload = self._get_request_payload(prompt, **kwargs)
        response = await self._async_client.chat.completions.create(**payload)
        text = response.choices[0].message.content
        return CompletionResponse(text=text)
    except Exception as e:
        # Handle async-specific errors
        raise
```

## Best Practices

1. **Always Provide Both Versions**: Sync and async methods should have feature parity
2. **Use Async Generators for Streaming**: `async def` with `yield` for streaming responses
3. **Proper Await Usage**: Always `await` async operations
4. **Async Context Managers**: Use `async with` for resource management
5. **Error Propagation**: Let async errors propagate naturally
6. **Type Hints**: Use proper async type hints (`AsyncGenerator`, `Coroutine`, etc.)
7. **Testing**: Use `@pytest.mark.asyncio` for async tests
8. **Client Separation**: Maintain separate sync and async client instances

## Common Patterns

### Running Async Code
```python
import asyncio

async def main():
    llm = GradientAI(...)
    response = await llm.acomplete("Hello")
    print(response.text)

asyncio.run(main())
```

### Collecting Async Generator Results
```python
async def collect_stream():
    llm = GradientAI(...)
    chunks = []
    async for delta in llm.astream_complete("Hello"):
        chunks.append(delta.delta)
    return "".join(chunks)
```

### Parallel Async Operations
```python
import asyncio

async def parallel_calls():
    llm = GradientAI(...)
    tasks = [
        llm.acomplete("Question 1"),
        llm.acomplete("Question 2"),
        llm.acomplete("Question 3"),
    ]
    results = await asyncio.gather(*tasks)
    return results
```

## Type Hints for Async

```python
from typing import AsyncGenerator, Coroutine
from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseAsyncGen,
    ChatResponse,
    ChatResponseAsyncGen,
)

# Async function
async def acomplete(...) -> CompletionResponse:
    ...

# Async generator
async def astream_complete(...) -> CompletionResponseAsyncGen:
    ...

# Coroutine type
def get_coroutine() -> Coroutine[Any, Any, CompletionResponse]:
    return self.acomplete("test")
```

