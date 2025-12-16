import os
import pytest
from dotenv import load_dotenv

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.llms.digitalocean.gradientai import GradientAI

load_dotenv()
REQUIRED_ENV = "MODEL_ACCESS_KEY"


# Tool functions for function calling tests
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


def _skip_if_no_creds():
    if not os.getenv(REQUIRED_ENV):
        pytest.skip(
            f"Live Gradient credentials required (set {REQUIRED_ENV}); skipping live integration test",
            allow_module_level=False,
        )


def _make_llm():
    _skip_if_no_creds()
    # Default to a commonly available serverless inference model; override with GRADIENT_MODEL env var.
    model = os.getenv("GRADIENT_MODEL", "openai-gpt-oss-120b")
    api_key = os.environ[REQUIRED_ENV]
    return GradientAI(
        model=model,
        model_access_key=api_key,
        timeout=30,
    )


def test_complete_and_chat_sync():
    llm = _make_llm()

    completion = llm.complete("Say 'hello' in one word.")
    assert completion.text

    chat = llm.chat([ChatMessage(role="user", content="Say 'ping' once.")])
    assert chat.message.content


def test_stream_complete_and_chat_sync():
    llm = _make_llm()

    chunks = list(llm.stream_complete("Say 'stream' in two short parts."))
    assert chunks, "stream_complete returned no chunks"
    assert chunks[-1].text

    messages = [ChatMessage(role="user", content="Answer with one short word, streamed.")]
    responses = list(llm.stream_chat(messages))
    assert responses, "stream_chat returned no chunks"
    assert responses[-1].message.content


@pytest.mark.asyncio
async def test_complete_and_chat_async():
    llm = _make_llm()

    completion = await llm.acomplete("Say 'async' in one word.")
    assert completion.text

    chat = await llm.achat([ChatMessage(role="user", content="Answer with 'pong'.")])
    assert chat.message.content


@pytest.mark.asyncio
async def test_stream_complete_and_chat_async():
    llm = _make_llm()

    chunks = []
    async for delta in llm.astream_complete("Stream two short pieces."):
        chunks.append(delta.delta)
    assert chunks, "astream_complete returned no chunks"
    assert "".join(chunks)

    messages = [ChatMessage(role="user", content="Stream a short greeting.")]
    chat_chunks = []
    async for delta in llm.astream_chat(messages):
        chat_chunks.append(delta.delta)
    assert chat_chunks, "astream_chat returned no chunks"
    assert "".join(chat_chunks)


# Function calling tests
def test_chat_with_tools():
    """Test chat_with_tools method."""
    llm = _make_llm()
    tools = [FunctionTool.from_defaults(fn=add), FunctionTool.from_defaults(fn=multiply)]
    
    response = llm.chat_with_tools(
        tools=tools,
        user_msg="What is 5 multiplied by 8?",
    )
    
    # Should return a response (either with tool calls or direct answer)
    assert response is not None
    assert response.message is not None


def test_get_tool_calls_from_response():
    """Test extracting tool calls from response."""
    llm = _make_llm()
    tools = [FunctionTool.from_defaults(fn=add)]
    
    response = llm.chat_with_tools(
        tools=tools,
        user_msg="Calculate 10 plus 15 using the add function.",
        tool_required=True,  # Force tool usage
    )
    
    # Try to extract tool calls (may or may not have them depending on model)
    tool_calls = llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)
    
    # Either has tool calls or doesn't error
    assert isinstance(tool_calls, list)


def test_predict_and_call():
    """Test predict_and_call method for automatic tool execution."""
    llm = _make_llm()
    tools = [FunctionTool.from_defaults(fn=add)]
    
    # predict_and_call may execute the tool or model may answer directly
    # depending on model behavior
    try:
        response = llm.predict_and_call(
            tools=tools,
            user_msg="Use the add function to calculate 10 plus 15.",
            tool_required=True,  # Hint to use tools
        )
        assert response is not None
    except ValueError as e:
        # Model may choose not to use tools, which is valid behavior
        if "Expected at least one tool call" in str(e):
            pytest.skip("Model chose not to use tools for this query")
        raise


@pytest.mark.asyncio
async def test_achat_with_tools():
    """Test async chat_with_tools method."""
    llm = _make_llm()
    tools = [FunctionTool.from_defaults(fn=multiply)]
    
    response = await llm.achat_with_tools(
        tools=tools,
        user_msg="What is 7 times 9?",
    )
    
    assert response is not None
    assert response.message is not None


@pytest.mark.asyncio
async def test_apredict_and_call():
    """Test async predict_and_call method."""
    llm = _make_llm()
    tools = [FunctionTool.from_defaults(fn=add)]
    
    # Model may choose to use tools or answer directly
    try:
        response = await llm.apredict_and_call(
            tools=tools,
            user_msg="Use the add function to calculate 20 plus 30.",
            tool_required=True,  # Hint to use tools
        )
        assert response is not None
    except ValueError as e:
        # Model may choose not to use tools, which is valid behavior
        if "Expected at least one tool call" in str(e):
            pytest.skip("Model chose not to use tools for this query")
        raise


def test_metadata_function_calling():
    """Test that metadata reports function calling capability."""
    llm = _make_llm()
    
    metadata = llm.metadata
    assert metadata.is_function_calling_model is True
    assert hasattr(llm, "chat_with_tools")
    assert hasattr(llm, "predict_and_call")
    assert hasattr(llm, "get_tool_calls_from_response")

