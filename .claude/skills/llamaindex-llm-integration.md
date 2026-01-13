---
skill: llamaindex-llm-integration
category: llamaindex
tags: [llamaindex, llm, integration, function-calling]
updated: 2026-01-13
---
# LlamaIndex LLM Integration

## Overview
Guide to building custom LLM integrations for LlamaIndex, specifically for the DigitalOcean Gradient AI integration.

## When to Use This Skill
Reference this when implementing or modifying LLM methods, handling tool calling, or debugging message formatting issues.

## Base Classes

### FunctionCallingLLM
The primary base class for LLMs that support function/tool calling:
```python
from llama_index.core.llms.function_calling import FunctionCallingLLM

class GradientAI(FunctionCallingLLM):
    """DigitalOcean Gradient AI LLM."""
```

### Key Properties
- `metadata: LLMMetadata` - Must return LLMMetadata with model information
- `is_function_calling_model: bool` - Indicates tool calling support

## Required Methods

### Sync Methods

#### `complete(prompt: str, **kwargs) -> CompletionResponse`
Basic text completion:
```python
@llm_completion_callback()
def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
    payload = self._get_request_payload(prompt, **kwargs)
    response = self._client.chat.completions.create(**payload)
    text = response.choices[0].message.content
    return CompletionResponse(text=text)
```

#### `chat(messages: Sequence[ChatMessage], **kwargs) -> ChatResponse`
Chat interface with message history:
```python
@llm_chat_callback()
def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
    payload = self._get_request_payload("", messages=messages, **kwargs)
    response = self._client.chat.completions.create(**payload)
    message = response.choices[0].message
    return ChatResponse(message=self._to_chat_message(message))
```

#### `stream_complete(prompt: str, **kwargs) -> CompletionResponseGen`
Streaming completion:
```python
@llm_completion_callback()
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

#### `stream_chat(messages: Sequence[ChatMessage], **kwargs) -> ChatResponseGen`
Streaming chat:
```python
@llm_chat_callback()
def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
    payload = self._get_request_payload("", messages=messages, **kwargs)
    payload["stream"] = True
    stream = self._client.chat.completions.create(**payload)
    text = ""
    for completion in stream:
        delta = self._extract_delta(completion)
        if delta:
            text += delta
            yield ChatResponse(message=ChatMessage(role="assistant", content=text), delta=delta)
```

### Async Methods

#### `acomplete(prompt: str, **kwargs) -> CompletionResponse`
Async completion:
```python
@llm_completion_callback()
async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
    payload = self._get_request_payload(prompt, **kwargs)
    response = await self._async_client.chat.completions.create(**payload)
    text = response.choices[0].message.content
    return CompletionResponse(text=text)
```

#### `achat(messages: Sequence[ChatMessage], **kwargs) -> ChatResponse`
Async chat:
```python
@llm_chat_callback()
async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
    payload = self._get_request_payload("", messages=messages, **kwargs)
    response = await self._async_client.chat.completions.create(**payload)
    message = response.choices[0].message
    return ChatResponse(message=self._to_chat_message(message))
```

#### `astream_complete()` and `astream_chat()`
Similar to sync versions but using async generators.

## Function/Tool Calling Methods

### `chat_with_tools()`
Inherited from `FunctionCallingLLM`, handles tool calling:
```python
response = llm.chat_with_tools(
    tools=[add_tool, multiply_tool],
    user_msg="What is 5 multiplied by 8?",
)
```

### `get_tool_calls_from_response()`
Extract tool calls from response:
```python
tool_calls = llm.get_tool_calls_from_response(
    response, 
    error_on_no_tool_call=False
)
```

### `predict_and_call()`
Automatically execute tools and return results:
```python
response = llm.predict_and_call(
    tools=[add_tool],
    user_msg="What is 10 plus 15?",
)
```

### Internal Helper Methods

#### `_prepare_chat_with_tools()`
Prepares request payload with tools:
```python
def _prepare_chat_with_tools(
    self,
    tools: Sequence["BaseTool"],
    user_msg: Optional[Union[str, ChatMessage]] = None,
    chat_history: Optional[List[ChatMessage]] = None,
    verbose: bool = False,
    allow_parallel_tool_calls: bool = False,
    tool_required: bool = False,
    tool_choice: Optional[Union[str, dict]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    # Convert tools to OpenAI-compatible format
    tool_specs = [
        tool.metadata.to_openai_tool(skip_length_check=True) for tool in tools
    ] if tools else []
    # ... build messages and return payload
```

#### `_validate_chat_with_tools_response()`
Validates and optionally limits tool calls:
```python
def _validate_chat_with_tools_response(
    self,
    response: ChatResponse,
    tools: Sequence["BaseTool"],
    allow_parallel_tool_calls: bool = False,
    **kwargs: Any,
) -> ChatResponse:
    # Validate and limit tool calls if needed
```

## Message Formatting

### ChatMessage Structure
```python
from llama_index.core.base.llms.types import ChatMessage, TextBlock, ToolCallBlock

# Simple message
message = ChatMessage(role="user", content="Hello")

# Message with blocks (for tool calls)
message = ChatMessage(
    role="assistant",
    blocks=[
        TextBlock(text="I'll calculate that for you."),
        ToolCallBlock(
            tool_call_id="call_123",
            tool_name="add",
            tool_kwargs={"a": 5, "b": 3}
        )
    ]
)
```

### Formatting Messages for API
```python
def _format_messages(self, messages: Sequence[ChatMessage]) -> List[Dict[str, Any]]:
    """Format messages for Gradient API (OpenAI-compatible format)."""
    formatted = []
    for msg in messages:
        role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        message_dict: Dict[str, Any] = {"role": role}
        
        # Extract content and tool calls from blocks
        if hasattr(msg, "blocks") and msg.blocks:
            text_parts = [
                block.text for block in msg.blocks if isinstance(block, TextBlock)
            ]
            content = "".join(text_parts) if text_parts else None
            
            tool_call_blocks = [
                block for block in msg.blocks if isinstance(block, ToolCallBlock)
            ]
            # ... format tool calls
```

### Converting API Response to ChatMessage
```python
def _to_chat_message(self, message: Any) -> ChatMessage:
    """Convert a Gradient SDK response message to ChatMessage."""
    # Extract role, content, tool_calls
    # Build blocks list with TextBlock and ToolCallBlock
    # Return ChatMessage with blocks
```

## Response Types

### CompletionResponse
```python
from llama_index.core.base.llms.types import CompletionResponse

response = CompletionResponse(text="Generated text")
```

### ChatResponse
```python
from llama_index.core.base.llms.types import ChatResponse

response = ChatResponse(message=ChatMessage(role="assistant", content="Response"))
```

### Streaming Responses
- `CompletionResponseGen` - Generator of CompletionResponse
- `ChatResponseGen` - Generator of ChatResponse
- `CompletionResponseAsyncGen` - Async generator
- `ChatResponseAsyncGen` - Async generator

## LLMMetadata

```python
from llama_index.core.base.llms.types import LLMMetadata

@property
def metadata(self) -> LLMMetadata:
    return LLMMetadata(
        context_window=self.context_window,
        num_output=self.num_output,
        model_name=self.model,
        is_function_calling_model=self.is_function_calling_model,
    )
```

## Callback Decorators

Always use these decorators on LLM methods:
```python
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback
)

@llm_completion_callback()
def complete(self, ...):
    ...

@llm_chat_callback()
def chat(self, ...):
    ...
```

## Common Gotchas

### Message Formatting
- **Always check for both `blocks` and `content` attributes in ChatMessage**: Some messages use the older `content` field, newer ones use `blocks` list
- **Tool arguments from API are strings, not dicts**: Must parse with `_parse_tool_arguments()` - don't assume they're already dicts
- **SDK responses can be objects or dicts**: Use `hasattr()` checks before accessing attributes to support both formats
  ```python
  # Good - handles both formats
  if hasattr(message, "content"):
      content = message.content
  elif isinstance(message, dict):
      content = message.get("content")

  # Bad - assumes object format
  content = message.content  # Will fail on dict responses
  ```

### Tool Calling
- **Some models ignore `tool_choice`**: Always set `error_on_no_tool_call=False` for graceful handling when model doesn't call tools
- **ToolCallBlock is modern format**: Support legacy `additional_kwargs` for backward compatibility with older LlamaIndex versions
- **Tool call IDs are required**: Even if empty string, must be present for multi-turn conversations
- **Arguments can be partial during streaming**: Use `parse_partial_json()` not `json.loads()` for streaming tool calls

### Sync vs Async
- **Use async when possible**: Async methods allow concurrent requests and better resource utilization
- **Don't mix sync/async clients**: Keep `_client` and `_async_client` separate - never call sync methods from async context
- **Streaming generators are different types**: `CompletionResponseGen` vs `CompletionResponseAsyncGen` - use correct type hints

### Version Compatibility
- **LlamaIndex >= 0.10.0 required**: Earlier versions don't have `ToolCallBlock`
- **Gradient SDK >= 3.8.0 required**: Earlier versions may not support tool calling

## Best Practices

1. **Implement Both Sync and Async**: Always provide both versions with feature parity
2. **Use Callback Decorators**: Apply `@llm_chat_callback()` and `@llm_completion_callback()` for observability
3. **Handle Streaming Properly**: Accumulate deltas and yield complete responses with both `text` and `delta`
4. **Support Tool Calling**: Implement all function calling methods if model supports it
5. **Type Safety**: Use proper type hints throughout

## Common Patterns

### Request Payload Building
```python
def _get_request_payload(
    self, prompt: str, messages: Optional[Sequence[ChatMessage]] = None, **kwargs: Any
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": self.model,
        "temperature": kwargs.get("temperature", self.temperature),
        "top_p": kwargs.get("top_p", self.top_p),
    }
    if messages:
        payload["messages"] = self._format_messages(messages)
    else:
        payload["messages"] = [{"role": "user", "content": prompt}]
    
    if "tools" in kwargs and kwargs["tools"]:
        payload["tools"] = kwargs["tools"]
    if "tool_choice" in kwargs and kwargs["tool_choice"]:
        payload["tool_choice"] = kwargs["tool_choice"]
    
    return payload
```

### Delta Extraction for Streaming
```python
def _extract_delta(self, completion: Any) -> str:
    """Extract delta content from a streaming completion chunk."""
    try:
        if not hasattr(completion, "choices") or not completion.choices:
            return ""
        
        choice = completion.choices[0]
        delta = getattr(choice, "delta", None)
        if delta is not None:
            if hasattr(delta, "content"):
                return delta.content or ""
        return ""
    except Exception:
        return ""
```

