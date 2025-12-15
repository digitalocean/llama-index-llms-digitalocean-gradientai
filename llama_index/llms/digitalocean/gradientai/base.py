"""Base DigitalOcean Gradient AI LLM implementation."""

import json
import os
from typing import Any, Dict, Optional, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM

try:
    from gradient import AsyncGradient, Gradient
except ImportError as exc:  # pragma: no cover - surfaced at runtime for users
    raise ImportError(
        "gradient is required for GradientLLM. Install with: pip install gradient"
    ) from exc


class DigitalOceanGradientAILLM(CustomLLM):
    """DigitalOcean Gradient AI LLM wrapper built on the official SDK."""

    model: str
    model_access_key: str
    workspace_id: Optional[str]
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    context_window: int = 4096
    num_output: int = 256
    timeout: float = 60.0

    def __init__(
        self,
        model: str,
        model_access_key: Optional[str] = None,
        workspace_id: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        context_window: int = 4096,
        num_output: int = 256,
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> None:
        model_access_key = (
            model_access_key
            or os.getenv("MODEL_ACCESS_KEY")
            or os.getenv("GRADIENT_MODEL_ACCESS_KEY")
            or os.getenv("GRADIENT_API_KEY")
        )
        workspace_id = workspace_id or os.getenv("GRADIENT_WORKSPACE_ID")

        if not model_access_key:
            raise ValueError(
                "Model access key required. Set MODEL_ACCESS_KEY (preferred), "
                "GRADIENT_MODEL_ACCESS_KEY, GRADIENT_API_KEY, or pass model_access_key."
            )

        super().__init__(
            model=model,
            model_access_key=model_access_key,
            workspace_id=workspace_id,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            context_window=context_window,
            num_output=num_output,
            timeout=timeout,
            **kwargs,
        )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model,
        )

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

    def _format_messages(self, messages: Sequence[ChatMessage]) -> list:
        formatted = []
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            formatted.append({"role": role, "content": msg.content})
        return formatted

    def _get_request_payload(
        self, prompt: str, messages: Optional[Sequence[ChatMessage]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if kwargs.get("max_tokens"):
            payload["max_tokens"] = kwargs["max_tokens"]

        if messages:
            payload["messages"] = self._format_messages(messages)
        else:
            payload["messages"] = [{"role": "user", "content": prompt}]
        return payload

    def _to_chat_message(self, message: Any) -> ChatMessage:
        role = getattr(message, "role", None) or message.get("role", "assistant")
        content = getattr(message, "content", None) or message.get("content", "")
        return ChatMessage(role=role, content=content)

    def _extract_delta(self, completion: Any) -> str:
        """Extract delta content from a Gradient SDK completion object."""
        try:
            if hasattr(completion, "choices") and completion.choices:
                choice = completion.choices[0]
                if hasattr(choice, "delta") and choice.delta:
                    if hasattr(choice.delta, "content"):
                        return choice.delta.content or ""
                    elif isinstance(choice.delta, dict):
                        return choice.delta.get("content", "")
                elif hasattr(choice, "message") and choice.message:
                    if hasattr(choice.message, "content"):
                        return choice.message.content or ""
                    elif isinstance(choice.message, dict):
                        return choice.message.get("content", "")
            return ""
        except Exception:
            return ""

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        payload = self._get_request_payload(prompt, **kwargs)
        response = self._client.chat.completions.create(**payload)
        text = response.choices[0].message.content
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        payload = self._get_request_payload(prompt, **kwargs)
        payload["stream"] = True
        stream = self._client.chat.completions.create(**payload)
        text = ""
        for completion in stream:
            delta = self._extract_delta(completion)
            if delta:
                text += delta
                yield CompletionResponse(text=text, delta=delta)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        payload = self._get_request_payload("", messages=messages, **kwargs)
        response = self._client.chat.completions.create(**payload)
        message = response.choices[0].message
        return ChatResponse(message=self._to_chat_message(message))

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

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        payload = self._get_request_payload(prompt, **kwargs)
        response = await self._async_client.chat.completions.create(**payload)
        text = response.choices[0].message.content
        return CompletionResponse(text=text)

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
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

    @llm_chat_callback()
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        payload = self._get_request_payload("", messages=messages, **kwargs)
        response = await self._async_client.chat.completions.create(**payload)
        message = response.choices[0].message
        return ChatResponse(message=self._to_chat_message(message))

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
                yield ChatResponse(message=ChatMessage(role="assistant", content=text), delta=delta)

