# llama-index-llms-digitalocean-gradientai

LlamaIndex integration for DigitalOcean Gradient AI with full support for function/tool calling.

## Installation

```bash
pip install llama-index-llms-digitalocean-gradientai
```

This package uses the official [gradient](https://github.com/digitalocean/gradient-python) SDK (PyPI package: `gradient`) under the hood; it is installed automatically as a dependency.

## Usage

### Basic Usage

```python
from llama_index.llms.digitalocean.gradientai import GradientAI

llm = GradientAI(
    model="openai-gpt-oss-120b",
    model_access_key="your-api-key",
)

response = llm.complete("What is DigitalOcean Gradient AI Platform?")
print(response.text)
```

### Chat Interface

```python
from llama_index.core.llms import ChatMessage
from llama_index.llms.digitalocean.gradientai import GradientAI

llm = GradientAI(
    model="openai-gpt-oss-120b",
    model_access_key="your-api-key",
)

messages = [
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(role="user", content="What is the capital of France?")
]

response = llm.chat(messages)
print(response.message.content)
```

### Streaming

```python
from llama_index.llms.digitalocean.gradientai import GradientAI

llm = GradientAI(
    model="openai-gpt-oss-120b",
    model_access_key="your-api-key",
)

# Streaming completion
for chunk in llm.stream_complete("Tell me a story about AI:"):
    print(chunk.delta, end="", flush=True)
```

### Async Usage

```python
import asyncio
from llama_index.llms.digitalocean.gradientai import GradientAI

async def main():
    llm = GradientAI(
        model="openai-gpt-oss-120b",
        model_access_key="your-api-key",
    )
    response = await llm.acomplete("What is Gradient?")
    print(response.text)

asyncio.run(main())
```

### Function/Tool Calling

This integration supports OpenAI-compatible function calling, enabling the LLM to invoke tools based on user queries.

#### Using `chat_with_tools`

```python
from llama_index.llms.digitalocean.gradientai import GradientAI
from llama_index.core.tools import FunctionTool

# Define tools
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

# Create tool instances
add_tool = FunctionTool.from_defaults(fn=add)
multiply_tool = FunctionTool.from_defaults(fn=multiply)
tools = [add_tool, multiply_tool]

# Initialize LLM
llm = GradientAI(
    model="openai-gpt-oss-120b",
    model_access_key="your-api-key",
)

# Chat with tools
response = llm.chat_with_tools(
    tools=tools,
    user_msg="What is 5 multiplied by 8?",
)
print(response.message)

# Extract tool calls from response
tool_calls = llm.get_tool_calls_from_response(
    response, 
    error_on_no_tool_call=False
)
for tool_call in tool_calls:
    print(f"Tool: {tool_call.tool_name}, Args: {tool_call.tool_kwargs}")
```

#### Using `predict_and_call`

For automatic tool execution and result handling:

```python
from llama_index.llms.digitalocean.gradientai import GradientAI
from llama_index.core.tools import FunctionTool

def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

llm = GradientAI(
    model="openai-gpt-oss-120b",
    model_access_key="your-api-key",
)

# Automatically calls the tool and returns the result
response = llm.predict_and_call(
    tools=[add_tool],
    user_msg="What is 10 plus 15?",
)
print(response)  # Output: 25
```

#### Async Function Calling

```python
import asyncio
from llama_index.llms.digitalocean.gradientai import GradientAI
from llama_index.core.tools import FunctionTool

def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

async def main():
    llm = GradientAI(
        model="openai-gpt-oss-120b",
        model_access_key="your-api-key",
    )
    
    response = await llm.achat_with_tools(
        tools=[multiply_tool],
        user_msg="What is 7 times 9?",
    )
    print(response.message)

asyncio.run(main())
```

## Package Structure

```
llama-index-llms-digitalocean-gradientai/
├── llama_index/
│   └── llms/
│       └── digitalocean/
│           └── gradientai/
│               ├── __init__.py
│               └── base.py
├── tests/
│   └── test_gradient_llm.py
├── pyproject.toml
├── README.md
└── LICENSE
```

## License

MIT
