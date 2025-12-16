"""Base DigitalOcean Gradient AI LLM implementation."""

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
    TextBlock,
    ToolCallBlock,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.llms.utils import parse_partial_json

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool

try:
    from gradient import AsyncGradient, Gradient
except ImportError as exc:  # pragma: no cover - surfaced at runtime for users
    raise ImportError(
        "gradient is required for GradientLLM. Install with: pip install gradient"
    ) from exc


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


class GradientAI(FunctionCallingLLM):
    """DigitalOcean Gradient AI LLM.
    
    Supports function/tool calling similar to OpenAI's implementation.
    
    Example:
        >>> from llama_index.llms.digitalocean.gradientai import GradientAI
        >>> llm = GradientAI(model="openai-gpt-oss-120b", model_access_key="...")
        >>> response = llm.complete("Hello!")
    """

    model: str
    model_access_key: str
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    context_window: int = 4096
    num_output: int = 256
    timeout: float = 60.0
    is_function_calling_model: bool = True

    def __init__(
        self,
        model: str,
        model_access_key: str,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        context_window: int = 4096,
        num_output: int = 256,
        timeout: float = 60.0,
        is_function_calling_model: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize DigitalOcean Gradient AI LLM.
        
        Args:
            model: Model name/identifier.
            model_access_key: API key for authentication (required).
            base_url: Optional custom API base URL.
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling parameter.
            context_window: Maximum context window size.
            num_output: Default number of output tokens.
            timeout: Request timeout in seconds.
            is_function_calling_model: Whether the model supports function calling.
        """
        if not model_access_key:
            raise ValueError("model_access_key is required and must be provided explicitly.")

        super().__init__(
            model=model,
            model_access_key=model_access_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            context_window=context_window,
            num_output=num_output,
            timeout=timeout,
            is_function_calling_model=is_function_calling_model,
            **kwargs,
        )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model,
            is_function_calling_model=self.is_function_calling_model,
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
                
                if tool_call_blocks:
                    message_dict["tool_calls"] = [
                        {
                            "id": block.tool_call_id or "",
                            "type": "function",
                            "function": {
                                "name": block.tool_name,
                                "arguments": json.dumps(block.tool_kwargs) 
                                    if isinstance(block.tool_kwargs, dict) 
                                    else str(block.tool_kwargs or "{}"),
                            }
                        }
                        for block in tool_call_blocks
                    ]
                
                if content:
                    message_dict["content"] = content
                elif not tool_call_blocks:
                    message_dict["content"] = ""
            else:
                message_dict["content"] = msg.content or ""
            
            formatted.append(message_dict)
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
        
        # Add tools if provided
        if "tools" in kwargs and kwargs["tools"]:
            payload["tools"] = kwargs["tools"]
        if "tool_choice" in kwargs and kwargs["tool_choice"]:
            payload["tool_choice"] = kwargs["tool_choice"]
        
        return payload

    def _to_chat_message(self, message: Any) -> ChatMessage:
        """Convert a Gradient SDK response message to ChatMessage."""
        # Extract role (handle both object attributes and dict access)
        if hasattr(message, "role"):
            role = message.role
        elif isinstance(message, dict):
            role = message.get("role", "assistant")
        else:
            role = "assistant"
        
        # Extract content
        if hasattr(message, "content"):
            content = message.content
        elif isinstance(message, dict):
            content = message.get("content")
        else:
            content = None
        
        # Build blocks list
        blocks = []
        if content:
            blocks.append(TextBlock(text=content))
        
        # Extract tool calls (handle both object attributes and dict access)
        if hasattr(message, "tool_calls"):
            tool_calls = message.tool_calls or []
        elif isinstance(message, dict):
            tool_calls = message.get("tool_calls") or []
        else:
            tool_calls = []
        
        for tool_call in tool_calls:
            # Extract tool call fields
            if hasattr(tool_call, "id"):
                tool_id = tool_call.id or ""
                func = tool_call.function
                tool_name = func.name if hasattr(func, "name") else ""
                tool_args = func.arguments if hasattr(func, "arguments") else "{}"
            elif isinstance(tool_call, dict):
                tool_id = tool_call.get("id", "")
                func = tool_call.get("function", {})
                tool_name = func.get("name", "")
                tool_args = func.get("arguments", "{}")
            else:
                continue
            
            blocks.append(
                ToolCallBlock(
                    tool_call_id=tool_id,
                    tool_name=tool_name,
                    tool_kwargs=_parse_tool_arguments(tool_args),
                )
            )
        
        if blocks:
            return ChatMessage(role=role, blocks=blocks)
        return ChatMessage(role=role, content=content or "")

    def _extract_delta(self, completion: Any) -> str:
        """Extract delta content from a streaming completion chunk."""
        try:
            if not hasattr(completion, "choices") or not completion.choices:
                return ""
            
            choice = completion.choices[0]
            
            # Try delta first (streaming format)
            delta = getattr(choice, "delta", None)
            if delta is not None:
                if hasattr(delta, "content"):
                    return delta.content or ""
                if isinstance(delta, dict):
                    return delta.get("content", "")
            
            # Fallback to message (non-streaming format)
            message = getattr(choice, "message", None)
            if message is not None:
                if hasattr(message, "content"):
                    return message.content or ""
                if isinstance(message, dict):
                    return message.get("content", "")
            
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
        """Prepare chat request with tools for function calling.
        
        Args:
            tools: Sequence of tools available for the LLM to call.
            user_msg: User message (string or ChatMessage).
            chat_history: Previous chat messages.
            verbose: Whether to print verbose output.
            allow_parallel_tool_calls: Allow multiple tool calls in one response.
            tool_required: If True, LLM must call a tool.
            tool_choice: Specific tool choice ("auto", "required", "none", or tool name).
            **kwargs: Additional arguments passed to chat.
            
        Returns:
            Dict of arguments ready for the chat method.
        """
        # Convert tools to OpenAI-compatible format
        tool_specs = [
            tool.metadata.to_openai_tool(skip_length_check=True) for tool in tools
        ] if tools else []
        
        # Build messages list (copy to avoid mutation)
        messages: List[ChatMessage] = list(chat_history) if chat_history else []
        
        # Add user message
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
    
    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> List[ToolSelection]:
        """Extract tool calls from the LLM response.
        
        Args:
            response: The chat response from the LLM.
            error_on_no_tool_call: Raise error if no tool calls found.
            **kwargs: Additional arguments (unused).
            
        Returns:
            List of ToolSelection objects representing the tool calls.
            
        Raises:
            ValueError: If error_on_no_tool_call is True and no tools were called.
        """
        # Primary path: extract from ToolCallBlock in message blocks
        tool_call_blocks = [
            block for block in response.message.blocks
            if isinstance(block, ToolCallBlock)
        ]
        
        if tool_call_blocks:
            return [
                ToolSelection(
                    tool_id=block.tool_call_id or "",
                    tool_name=block.tool_name,
                    tool_kwargs=_parse_tool_arguments(block.tool_kwargs),
                )
                for block in tool_call_blocks
            ]
        
        # Fallback: check additional_kwargs (backward compatibility)
        legacy_tool_calls = response.message.additional_kwargs.get("tool_calls") or []
        
        if not legacy_tool_calls:
            if error_on_no_tool_call:
                raise ValueError("Expected at least one tool call, but got 0 tool calls.")
            return []
        
        tool_selections = []
        for tc in legacy_tool_calls:
            if isinstance(tc, dict):
                tool_id = tc.get("id", "")
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                arguments = func.get("arguments", "{}")
            else:
                tool_id = getattr(tc, "id", "") or ""
                func = getattr(tc, "function", None)
                tool_name = getattr(func, "name", "") if func else ""
                arguments = getattr(func, "arguments", "{}") if func else "{}"
            
            tool_selections.append(
                ToolSelection(
                    tool_id=tool_id,
                    tool_name=tool_name,
                    tool_kwargs=_parse_tool_arguments(arguments),
                )
            )
        
        return tool_selections

