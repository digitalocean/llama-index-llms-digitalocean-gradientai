# llama-index-llms-digitalocean-gradientai

LlamaIndex integration for DigitalOcean Gradient AI LLM.

## Installation

```bash
pip install llama-index-llms-digitalocean-gradientai
```

This package uses the official [gradient](https://github.com/digitalocean/gradient-python) SDK (PyPI package: `gradient`) under the hood; it is installed automatically as a dependency.

Or install from source:

```bash
git clone https://github.com/yourusername/llama-index-llms-digitalocean-gradientai
cd llama-index-llms-digitalocean-gradientai
pip install -e .
```

## Usage

### Basic Usage

```python
from llama_index.llms.gradient import DigitalOceanGradientAILLM

llm = DigitalOceanGradientAILLM(
    model="meta-llama-3-70b-instruct",
    api_key="your-api-key",
    workspace_id="your-workspace-id"
)

response = llm.complete("What is DigitalOcean Gradient?")
print(response)
```

### Using Environment Variables

```python
import os
from llama_index.llms.gradient import DigitalOceanGradientAILLM

os.environ["GRADIENT_API_KEY"] = "your-api-key"
os.environ["GRADIENT_WORKSPACE_ID"] = "your-workspace-id"

llm = DigitalOceanGradientAILLM(model="meta-llama-3-70b-instruct")
```

You can also use `GRADIENT_MODEL_ACCESS_KEY` (recommended) in place of `GRADIENT_API_KEY`.

### Chat Interface

```python
from llama_index.core.llms import ChatMessage
from llama_index.llms.gradient import DigitalOceanGradientAILLM

llm = DigitalOceanGradientAILLM(
    model="meta-llama-3-70b-instruct",
    api_key="your-api-key",
    workspace_id="your-workspace-id"
)

messages = [
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(role="user", content="What is Gradient?")
]

response = llm.chat(messages)
print(response.message.content)
```

### Streaming

```python
from llama_index.llms.gradient import DigitalOceanGradientAILLM

llm = DigitalOceanGradientAILLM(
    model="meta-llama-3-70b-instruct",
    api_key="your-api-key",
    workspace_id="your-workspace-id"
)

response_gen = llm.stream_complete("Tell me a story about AI:")
for delta in response_gen:
    print(delta.delta, end="", flush=True)
```

### Async Usage

```python
import asyncio
from llama_index.llms.gradient import DigitalOceanGradientAILLM

async def main():
llm = DigitalOceanGradientAILLM(
        model="meta-llama-3-70b-instruct",
        api_key="your-api-key",
        workspace_id="your-workspace-id"
    )
    response = await llm.acomplete("What is Gradient?")
    print(response)

asyncio.run(main())
```

### With RAG Pipeline

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.gradient import GradientLLM

llm = DigitalOceanGradientAILLM(
    model="meta-llama-3-70b-instruct",
    api_key="your-api-key",
    workspace_id="your-workspace-id"
)

documents = [Document(text="DigitalOcean Gradient is a managed LLM API service...")]
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("What is Gradient?")
print(response)
```

## Package Structure

```
llama-index-llms-digitalocean-gradientai/
├── llama_index/
│   └── llms/
│       └── gradient/
│           ├── __init__.py
│           └── base.py
├── setup.py
├── pyproject.toml
├── README.md
└── requirements.txt
```

## License

MIT
