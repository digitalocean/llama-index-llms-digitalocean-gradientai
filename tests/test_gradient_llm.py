import os
import pytest
from dotenv import load_dotenv

from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.digitalocean.gradientai import DigitalOceanGradientAILLM

load_dotenv()
REQUIRED_ENV = "MODEL_ACCESS_KEY"


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
    workspace_id = os.getenv("GRADIENT_WORKSPACE_ID")
    return DigitalOceanGradientAILLM(
        model=model,
        model_access_key=api_key,
        workspace_id=workspace_id,
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

