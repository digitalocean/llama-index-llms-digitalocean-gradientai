---
skill: openai-tool-calling-spec
category: specifications
tags: [openai, tool-calling, function-calling, api]
updated: 2026-01-13
---
# OpenAI Tool Calling Specification

## Overview
Guide to implementing OpenAI-compatible function/tool calling in LLM integrations.

## When to Use This Skill
Reference when implementing tool calling features, debugging tool call issues, or understanding the OpenAI tool calling format.

## Tool Calling Format

### Tool Definition Schema
Tools are defined using OpenAI's function calling format:
```python
tool_spec = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two numbers together.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "integer",
                    "description": "First number"
                },
                "b": {
                    "type": "integer",
                    "description": "Second number"
                }
            },
            "required": ["a", "b"]
        }
    }
}
```

### Converting LlamaIndex Tools to OpenAI Format
```python
from llama_index.core.tools import FunctionTool

def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

# Convert to OpenAI format
tool_spec = add_tool.metadata.to_openai_tool(skip_length_check=True)
```

## Tool Choice Parameter

### Tool Choice Values
```python
# Auto - model decides whether to call tools
tool_choice = "auto"

# Required - model must call a tool
tool_choice = "required"

# None - no tools allowed
tool_choice = "none"

# Specific tool - force a specific tool
tool_choice = {
    "type": "function",
    "function": {"name": "add"}
}
```

### Resolving Tool Choice
```python
def _resolve_tool_choice(
    tool_choice: Optional[Union[str, dict]], tool_required: bool = False
) -> Union[str, dict]:
    """Resolve tool choice to OpenAI-compatible format."""
    if tool_choice is None:
        return "required" if tool_required else "auto"
    if isinstance(tool_choice, dict):
        return tool_choice
    if tool_choice not in ("none", "auto", "required"):
        return {"type": "function", "function": {"name": tool_choice}}
    return tool_choice
```

## Request Payload

### Chat Request with Tools
```python
payload = {
    "model": "openai-gpt-oss-120b",
    "messages": [
        {"role": "user", "content": "What is 5 plus 3?"}
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"}
                    },
                    "required": ["a", "b"]
                }
            }
        }
    ],
    "tool_choice": "auto"  # or "required", "none", or specific tool
}
```

## Response Format

### Tool Call in Response
```python
{
    "id": "call_abc123",
    "type": "function",
    "function": {
        "name": "add",
        "arguments": '{"a": 5, "b": 3}'
    }
}
```

### Parsing Tool Arguments
```python
def _parse_tool_arguments(arguments: Any) -> dict:
    """Parse tool arguments from string or dict format."""
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            return parse_partial_json(arguments)
        except (ValueError, TypeError, json.JSONDecodeError):
            return {}
    return {}
```

## Message Format with Tool Calls

### Assistant Message with Tool Call
```python
message = {
    "role": "assistant",
    "content": None,  # May be None when tool calls are present
    "tool_calls": [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "add",
                "arguments": '{"a": 5, "b": 3}'
            }
        }
    ]
}
```

### Tool Result Message
After executing a tool, send the result back:
```python
message = {
    "role": "tool",
    "content": "8",  # Result of the tool execution
    "tool_call_id": "call_abc123"  # Must match the tool call ID
}
```

## Multi-Turn Tool Conversations

### Conversation Flow
```python
# Turn 1: User asks question
messages = [
    {"role": "user", "content": "What is 5 plus 3?"}
]

# Turn 2: Assistant calls tool
response = llm.chat(messages, tools=[add_tool])
# Response contains tool_call

# Turn 3: Execute tool and send result
tool_call = extract_tool_call(response)
result = execute_tool(tool_call)
messages.append({
    "role": "assistant",
    "tool_calls": [tool_call]
})
messages.append({
    "role": "tool",
    "content": str(result),
    "tool_call_id": tool_call["id"]
})

# Turn 4: Get final answer
final_response = llm.chat(messages, tools=[add_tool])
```

## Implementation in LlamaIndex

### Preparing Chat with Tools
```python
def _prepare_chat_with_tools(
    self,
    tools: Sequence["BaseTool"],
    user_msg: Optional[Union[str, ChatMessage]] = None,
    chat_history: Optional[List[ChatMessage]] = None,
    tool_required: bool = False,
    tool_choice: Optional[Union[str, dict]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    # Convert tools to OpenAI-compatible format
    tool_specs = [
        tool.metadata.to_openai_tool(skip_length_check=True) for tool in tools
    ] if tools else []
    
    # Build messages list
    messages: List[ChatMessage] = list(chat_history) if chat_history else []
    
    if user_msg is not None:
        if isinstance(user_msg, str):
            messages.append(ChatMessage(role=MessageRole.USER, content=user_msg))
        else:
            messages.append(user_msg)
    
    result: Dict[str, Any] = {"messages": messages, **kwargs}
    
    if tool_specs:
        result["tools"] = tool_specs
        result["tool_choice"] = _resolve_tool_choice(tool_choice, tool_required)
    
    return result
```

### Extracting Tool Calls from Response
```python
def get_tool_calls_from_response(
    self,
    response: ChatResponse,
    error_on_no_tool_call: bool = True,
    **kwargs: Any,
) -> List[ToolSelection]:
    """Extract tool calls from the LLM response."""
    # Primary path: extract from ToolCallBlock in message blocks
    tool_call_blocks = [
        block for block in response.message.blocks
        if isinstance(block, ToolCallBlock)
    ]
    
    if tool_call_blocks:
        return [
            ToolSelection(
                tool_call_id=block.tool_call_id or "",
                tool_name=block.tool_name,
                tool_kwargs=_parse_tool_arguments(block.tool_kwargs),
            )
            for block in tool_call_blocks
        ]
    
    # Fallback: check additional_kwargs (backward compatibility)
    # ...
```

## Tool Call Blocks in LlamaIndex

### Creating ToolCallBlock
```python
from llama_index.core.base.llms.types import ToolCallBlock

tool_call_block = ToolCallBlock(
    tool_call_id="call_abc123",
    tool_name="add",
    tool_kwargs={"a": 5, "b": 3}
)
```

### Converting API Response to ToolCallBlock
```python
def _to_chat_message(self, message: Any) -> ChatMessage:
    """Convert API response to ChatMessage with ToolCallBlocks."""
    blocks = []
    
    # Extract tool calls
    tool_calls = message.get("tool_calls") or []
    for tool_call in tool_calls:
        tool_id = tool_call.get("id", "")
        func = tool_call.get("function", {})
        tool_name = func.get("name", "")
        tool_args = func.get("arguments", "{}")
        
        blocks.append(
            ToolCallBlock(
                tool_call_id=tool_id,
                tool_name=tool_name,
                tool_kwargs=_parse_tool_arguments(tool_args),
            )
        )
    
    return ChatMessage(role=role, blocks=blocks)
```

## Parallel Tool Calls

### Allowing Multiple Tool Calls
```python
response = llm.chat_with_tools(
    tools=[add_tool, multiply_tool],
    user_msg="Add 5 and 3, then multiply 4 and 2",
    allow_parallel_tool_calls=True,  # Allow multiple tool calls
)
```

### Validating Parallel Tool Calls
```python
def _validate_chat_with_tools_response(
    self,
    response: ChatResponse,
    tools: Sequence["BaseTool"],
    allow_parallel_tool_calls: bool = False,
    **kwargs: Any,
) -> ChatResponse:
    """Validate and optionally limit tool calls in response."""
    if allow_parallel_tool_calls:
        return response
    
    tool_call_blocks = [
        block for block in response.message.blocks 
        if isinstance(block, ToolCallBlock)
    ]
    
    if len(tool_call_blocks) > 1:
        # Keep only the first tool call
        non_tool_blocks = [
            block for block in response.message.blocks 
            if not isinstance(block, ToolCallBlock)
        ]
        response.message.blocks = non_tool_blocks + [tool_call_blocks[0]]
    
    return response
```

## Debugging Tool Calls

### Inspecting Tool Call Blocks
```python
# Print what the model is trying to call
for block in response.message.blocks:
    if isinstance(block, ToolCallBlock):
        print(f"Tool: {block.tool_name}")
        print(f"Args: {block.tool_kwargs}")
        print(f"ID: {block.tool_call_id}")
        print(f"Args type: {type(block.tool_kwargs)}")
```

### Debugging Tool Call Flow
```python
from llama_index.core.tools import FunctionTool

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

llm = GradientAI(model="...", model_access_key="...")

# Step 1: Check tool spec generation
tool = FunctionTool.from_defaults(fn=add)
tool_spec = tool.metadata.to_openai_tool(skip_length_check=True)
print("Tool spec:", tool_spec)

# Step 2: Make request and inspect response
response = llm.chat_with_tools(
    tools=[tool],
    user_msg="Add 5 and 3",
    tool_required=True
)

print("Response message:", response.message)
print("Message blocks:", response.message.blocks)
print("Additional kwargs:", response.message.additional_kwargs)

# Step 3: Extract tool calls
tool_calls = llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)
print(f"Found {len(tool_calls)} tool calls")
for tc in tool_calls:
    print(f"  - {tc.tool_name}({tc.tool_kwargs})")
```

### Common Failure Patterns

**Empty arguments**: Model returns `"arguments": "{}"` or `"arguments": ""`
```python
def _parse_tool_arguments(arguments: Any) -> dict:
    """Parse tool arguments with fallback to empty dict."""
    if arguments is None or arguments == "" or arguments == "{}":
        return {}  # Return empty dict, not None
    # ... rest of parsing
```

**Partial JSON during streaming**: Model streams tool calls incrementally
```python
from llama_index.core.llms.utils import parse_partial_json

# Good - handles incomplete JSON
args = parse_partial_json('{"a": 5, "b":')  # Won't crash

# Bad - will crash on partial JSON
args = json.loads('{"a": 5, "b":')  # JSONDecodeError
```

**Wrong tool name**: Model hallucinates tool names not in the list
```python
def validate_tool_call(tool_call: ToolSelection, available_tools: List[BaseTool]) -> bool:
    """Validate tool call against available tools."""
    tool_names = {tool.metadata.name for tool in available_tools}
    if tool_call.tool_name not in tool_names:
        print(f"Warning: Model called unknown tool '{tool_call.tool_name}'")
        print(f"Available tools: {tool_names}")
        return False
    return True
```

**Arguments as string when expecting dict**: SDK inconsistency
```python
# Always check type before using
if isinstance(tool_call.function.arguments, str):
    args = json.loads(tool_call.function.arguments)
elif isinstance(tool_call.function.arguments, dict):
    args = tool_call.function.arguments
else:
    args = {}
```

### Debugging Multi-Turn Conversations
```python
def debug_tool_conversation():
    """Debug a full tool calling conversation."""
    llm = GradientAI(model="...", model_access_key="...")

    def add(a: int, b: int) -> int:
        return a + b

    tool = FunctionTool.from_defaults(fn=add)
    messages = []

    # Turn 1: User question
    user_msg = ChatMessage(role="user", content="What is 5 + 3?")
    messages.append(user_msg)
    print(f"\n[User] {user_msg.content}")

    # Turn 2: Model calls tool
    response = llm.chat(messages, tools=[tool.metadata.to_openai_tool()])
    messages.append(response.message)
    print(f"[Assistant] {response.message}")

    # Extract tool call
    tool_calls = llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)
    if tool_calls:
        for tc in tool_calls:
            print(f"  Tool call: {tc.tool_name}({tc.tool_kwargs})")

            # Turn 3: Execute and return result
            result = add(**tc.tool_kwargs)
            tool_result_msg = ChatMessage(
                role="tool",
                content=str(result),
                additional_kwargs={"tool_call_id": tc.tool_id}
            )
            messages.append(tool_result_msg)
            print(f"[Tool] {result}")

    # Turn 4: Final answer
    final_response = llm.chat(messages)
    print(f"[Assistant] {final_response.message.content}")
```

## Best Practices

1. **Always Parse Arguments Safely**: Tool arguments come as JSON strings - use `parse_partial_json()` for robustness
2. **Handle Missing Tool Calls**: Models may not always call tools even when `tool_required=True`
3. **Validate Tool Names**: Check that called tools exist in your available tools list
4. **Preserve Tool Call IDs**: Required for multi-turn conversations with tool results
5. **Support Parallel Calls**: Handle multiple tool calls in one response when `allow_parallel_tool_calls=True`
6. **Backward Compatibility**: Support both ToolCallBlock and legacy `additional_kwargs` formats
7. **Debug with Logging**: Add debug prints to understand what the model is actually calling

## Common Issues

### Tool Arguments as String
Tool arguments are often JSON strings, not dicts:
```python
# API returns: '{"a": 5, "b": 3}'
# Need to parse: {"a": 5, "b": 3}
arguments = json.loads(tool_call.function.arguments)
```

### Missing Tool Calls
Models may choose not to use tools:
```python
tool_calls = llm.get_tool_calls_from_response(
    response, 
    error_on_no_tool_call=False  # Don't error if no tools called
)
```

### Tool Choice Not Respected
Some models may ignore `tool_choice`:
```python
# Use tool_required=True for stronger hint
response = llm.chat_with_tools(
    tools=[add_tool],
    user_msg="Calculate 5 + 3",
    tool_required=True,  # Stronger hint to use tools
)
```

