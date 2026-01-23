"""Tests for user agent information being passed correctly to Gradient SDK clients.

This module verifies that when GradientAI instances are created, the user agent
parameters (package name and version) are correctly passed to both sync and async
Gradient clients.
"""

from unittest.mock import MagicMock, patch

import pytest

from llama_index.llms.digitalocean.gradientai import GradientAI
from llama_index.llms.digitalocean.gradientai.base import PACKAGE_NAME, PACKAGE_VERSION


class TestUserAgentConfiguration:
    """Tests verifying user agent information is correctly configured."""

    def test_package_name_constant(self):
        """Test that PACKAGE_NAME constant is correctly set."""
        assert PACKAGE_NAME == "llama-index-llms-digitalocean-gradientai"

    def test_package_version_is_set(self):
        """Test that PACKAGE_VERSION is a valid version string."""
        # PACKAGE_VERSION should be a string like "0.1.1" or "0.0.0" in dev
        assert isinstance(PACKAGE_VERSION, str)
        assert len(PACKAGE_VERSION) > 0
        # Version should contain at least one dot (e.g., "0.1.1" or "0.0.0")
        assert "." in PACKAGE_VERSION

    def test_package_version_format(self):
        """Test that PACKAGE_VERSION follows semantic versioning format."""
        parts = PACKAGE_VERSION.split(".")
        # Should have at least major.minor format
        assert len(parts) >= 2
        # Each part should be numeric
        for part in parts:
            # Handle pre-release versions like "0.1.1a1"
            numeric_part = "".join(c for c in part if c.isdigit())
            assert len(numeric_part) > 0, f"Version part '{part}' should contain digits"


class TestSyncClientUserAgent:
    """Tests for synchronous Gradient client user agent configuration."""

    @patch("llama_index.llms.digitalocean.gradientai.base.Gradient")
    def test_sync_client_receives_user_agent_params(self, mock_gradient_class):
        """Test that sync Gradient client is created with user agent parameters."""
        # Arrange
        mock_client = MagicMock()
        mock_gradient_class.return_value = mock_client

        llm = GradientAI(
            model="test-model",
            model_access_key="test-key",
        )

        # Act - access the _client property to trigger client creation
        _ = llm._client

        # Assert
        mock_gradient_class.assert_called_once()
        call_kwargs = mock_gradient_class.call_args.kwargs

        assert "user_agent_package" in call_kwargs
        assert call_kwargs["user_agent_package"] == PACKAGE_NAME

        assert "user_agent_version" in call_kwargs
        assert call_kwargs["user_agent_version"] == PACKAGE_VERSION

    @patch("llama_index.llms.digitalocean.gradientai.base.Gradient")
    def test_sync_client_user_agent_with_custom_config(self, mock_gradient_class):
        """Test that user agent is passed even with custom base_url and timeout."""
        # Arrange
        mock_client = MagicMock()
        mock_gradient_class.return_value = mock_client

        llm = GradientAI(
            model="test-model",
            model_access_key="test-key",
            base_url="https://custom.api.com",
            timeout=120.0,
        )

        # Act
        _ = llm._client

        # Assert
        call_kwargs = mock_gradient_class.call_args.kwargs

        # Verify all expected parameters are passed
        assert call_kwargs["model_access_key"] == "test-key"
        assert call_kwargs["base_url"] == "https://custom.api.com"
        assert call_kwargs["timeout"] == 120.0
        assert call_kwargs["user_agent_package"] == PACKAGE_NAME
        assert call_kwargs["user_agent_version"] == PACKAGE_VERSION

    @patch("llama_index.llms.digitalocean.gradientai.base.Gradient")
    def test_sync_client_creates_new_instance_each_access(self, mock_gradient_class):
        """Test that _client property creates a new client instance each time."""
        # Arrange
        mock_client1 = MagicMock()
        mock_client2 = MagicMock()
        mock_gradient_class.side_effect = [mock_client1, mock_client2]

        llm = GradientAI(
            model="test-model",
            model_access_key="test-key",
        )

        # Act - access _client twice
        _ = llm._client
        _ = llm._client

        # Assert - Gradient() should be called twice
        assert mock_gradient_class.call_count == 2

        # Both calls should have user agent params
        for call in mock_gradient_class.call_args_list:
            call_kwargs = call.kwargs
            assert call_kwargs["user_agent_package"] == PACKAGE_NAME
            assert call_kwargs["user_agent_version"] == PACKAGE_VERSION


class TestAsyncClientUserAgent:
    """Tests for asynchronous Gradient client user agent configuration."""

    @patch("llama_index.llms.digitalocean.gradientai.base.AsyncGradient")
    def test_async_client_receives_user_agent_params(self, mock_async_gradient_class):
        """Test that async Gradient client is created with user agent parameters."""
        # Arrange
        mock_client = MagicMock()
        mock_async_gradient_class.return_value = mock_client

        llm = GradientAI(
            model="test-model",
            model_access_key="test-key",
        )

        # Act - access the _async_client property
        _ = llm._async_client

        # Assert
        mock_async_gradient_class.assert_called_once()
        call_kwargs = mock_async_gradient_class.call_args.kwargs

        assert "user_agent_package" in call_kwargs
        assert call_kwargs["user_agent_package"] == PACKAGE_NAME

        assert "user_agent_version" in call_kwargs
        assert call_kwargs["user_agent_version"] == PACKAGE_VERSION

    @patch("llama_index.llms.digitalocean.gradientai.base.AsyncGradient")
    def test_async_client_user_agent_with_custom_config(self, mock_async_gradient_class):
        """Test that async client passes user agent with custom configuration."""
        # Arrange
        mock_client = MagicMock()
        mock_async_gradient_class.return_value = mock_client

        llm = GradientAI(
            model="test-model",
            model_access_key="test-key",
            base_url="https://async.custom.api.com",
            timeout=90.0,
        )

        # Act
        _ = llm._async_client

        # Assert
        call_kwargs = mock_async_gradient_class.call_args.kwargs

        # Verify all expected parameters are passed
        assert call_kwargs["model_access_key"] == "test-key"
        assert call_kwargs["base_url"] == "https://async.custom.api.com"
        assert call_kwargs["timeout"] == 90.0
        assert call_kwargs["user_agent_package"] == PACKAGE_NAME
        assert call_kwargs["user_agent_version"] == PACKAGE_VERSION

    @patch("llama_index.llms.digitalocean.gradientai.base.AsyncGradient")
    def test_async_client_creates_new_instance_each_access(self, mock_async_gradient_class):
        """Test that _async_client property creates a new client instance each time."""
        # Arrange
        mock_client1 = MagicMock()
        mock_client2 = MagicMock()
        mock_async_gradient_class.side_effect = [mock_client1, mock_client2]

        llm = GradientAI(
            model="test-model",
            model_access_key="test-key",
        )

        # Act - access _async_client twice
        _ = llm._async_client
        _ = llm._async_client

        # Assert - AsyncGradient() should be called twice
        assert mock_async_gradient_class.call_count == 2

        # Both calls should have user agent params
        for call in mock_async_gradient_class.call_args_list:
            call_kwargs = call.kwargs
            assert call_kwargs["user_agent_package"] == PACKAGE_NAME
            assert call_kwargs["user_agent_version"] == PACKAGE_VERSION


class TestUserAgentInApiCalls:
    """Tests verifying user agent is passed when making actual API calls."""

    @patch("llama_index.llms.digitalocean.gradientai.base.Gradient")
    def test_user_agent_passed_during_complete(self, mock_gradient_class):
        """Test that user agent is configured when complete() is called."""
        # Arrange
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_gradient_class.return_value = mock_client

        llm = GradientAI(
            model="test-model",
            model_access_key="test-key",
        )

        # Act
        llm.complete("Test prompt")

        # Assert - Gradient client was created with user agent params
        call_kwargs = mock_gradient_class.call_args.kwargs
        assert call_kwargs["user_agent_package"] == PACKAGE_NAME
        assert call_kwargs["user_agent_version"] == PACKAGE_VERSION

    @patch("llama_index.llms.digitalocean.gradientai.base.Gradient")
    def test_user_agent_passed_during_chat(self, mock_gradient_class):
        """Test that user agent is configured when chat() is called."""
        from llama_index.core.base.llms.types import ChatMessage

        # Arrange
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.tool_calls = None

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_gradient_class.return_value = mock_client

        llm = GradientAI(
            model="test-model",
            model_access_key="test-key",
        )

        # Act
        llm.chat([ChatMessage(role="user", content="Hello")])

        # Assert - Gradient client was created with user agent params
        call_kwargs = mock_gradient_class.call_args.kwargs
        assert call_kwargs["user_agent_package"] == PACKAGE_NAME
        assert call_kwargs["user_agent_version"] == PACKAGE_VERSION

    @patch("llama_index.llms.digitalocean.gradientai.base.AsyncGradient")
    @pytest.mark.asyncio
    async def test_user_agent_passed_during_acomplete(self, mock_async_gradient_class):
        """Test that user agent is configured when acomplete() is called."""
        # Arrange
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test async response"

        mock_client = MagicMock()
        # Make the create method return an awaitable
        mock_client.chat.completions.create = MagicMock(return_value=mock_response)

        # We need to mock the async call properly
        async def mock_create(**kwargs):
            return mock_response

        mock_client.chat.completions.create = mock_create
        mock_async_gradient_class.return_value = mock_client

        llm = GradientAI(
            model="test-model",
            model_access_key="test-key",
        )

        # Act
        await llm.acomplete("Test async prompt")

        # Assert - AsyncGradient client was created with user agent params
        call_kwargs = mock_async_gradient_class.call_args.kwargs
        assert call_kwargs["user_agent_package"] == PACKAGE_NAME
        assert call_kwargs["user_agent_version"] == PACKAGE_VERSION


class TestUserAgentConsistency:
    """Tests verifying consistent user agent between sync and async clients."""

    @patch("llama_index.llms.digitalocean.gradientai.base.AsyncGradient")
    @patch("llama_index.llms.digitalocean.gradientai.base.Gradient")
    def test_sync_and_async_clients_have_same_user_agent(
        self, mock_gradient_class, mock_async_gradient_class
    ):
        """Test that both sync and async clients receive identical user agent info."""
        # Arrange
        mock_gradient_class.return_value = MagicMock()
        mock_async_gradient_class.return_value = MagicMock()

        llm = GradientAI(
            model="test-model",
            model_access_key="test-key",
        )

        # Act - access both clients
        _ = llm._client
        _ = llm._async_client

        # Assert
        sync_kwargs = mock_gradient_class.call_args.kwargs
        async_kwargs = mock_async_gradient_class.call_args.kwargs

        # Both should have the same user agent package
        assert sync_kwargs["user_agent_package"] == async_kwargs["user_agent_package"]
        assert sync_kwargs["user_agent_package"] == PACKAGE_NAME

        # Both should have the same user agent version
        assert sync_kwargs["user_agent_version"] == async_kwargs["user_agent_version"]
        assert sync_kwargs["user_agent_version"] == PACKAGE_VERSION

    def test_user_agent_constants_are_exported(self):
        """Test that user agent constants can be imported from the base module."""
        from llama_index.llms.digitalocean.gradientai.base import (
            PACKAGE_NAME,
            PACKAGE_VERSION,
        )

        assert PACKAGE_NAME is not None
        assert PACKAGE_VERSION is not None
        assert isinstance(PACKAGE_NAME, str)
        assert isinstance(PACKAGE_VERSION, str)
