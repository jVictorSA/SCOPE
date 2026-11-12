"""
Tests for model adapters.

Run with: pytest tests/test_models.py -v
"""
import pytest
from typing import List

from scope.models import (
    DeepSeekAdapter,
    Message,
    ModelResponse,
    ModelProtocol,
    BaseModelAdapter,
    CallableModelAdapter,
    SyncModelAdapter,
    create_deepseek_model,
)


class TestMessage:
    """Tests for Message class."""
    
    def test_message_creation(self):
        """Test creating a message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
    
    def test_message_user_helper(self):
        """Test Message.user() helper."""
        msg = Message.user("Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
    
    def test_message_assistant_helper(self):
        """Test Message.assistant() helper."""
        msg = Message.assistant("Hi there")
        assert msg.role == "assistant"
        assert msg.content == "Hi there"
    
    def test_message_system_helper(self):
        """Test Message.system() helper."""
        msg = Message.system("You are helpful")
        assert msg.role == "system"
        assert msg.content == "You are helpful"


class TestModelResponse:
    """Tests for ModelResponse class."""
    
    def test_response_creation(self):
        """Test creating a response."""
        response = ModelResponse(content="Hello")
        assert response.content == "Hello"
    
    def test_response_with_raw(self):
        """Test response with raw_response."""
        response = ModelResponse(
            content="Hello",
            raw_response={"id": "123"}
        )
        assert response.content == "Hello"
        assert response.raw_response == {"id": "123"}


class TestBaseModelAdapter:
    """Tests for BaseModelAdapter."""
    
    def test_abstract_class(self):
        """Test that BaseModelAdapter is abstract."""
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            BaseModelAdapter()
    
    @pytest.mark.asyncio
    async def test_custom_adapter(self):
        """Test creating a custom adapter."""
        class CustomAdapter(BaseModelAdapter):
            async def generate(self, messages: List[Message]) -> ModelResponse:
                return ModelResponse(content="Custom response")
        
        adapter = CustomAdapter()
        response = await adapter.generate([Message.user("Test")])
        assert response.content == "Custom response"


class TestCallableModelAdapter:
    """Tests for CallableModelAdapter."""
    
    @pytest.mark.asyncio
    async def test_callable_adapter_async(self):
        """Test using an async callable as adapter."""
        async def my_model(messages):
            return ModelResponse(content="From async callable")
        
        adapter = CallableModelAdapter(my_model)
        response = await adapter.generate([Message.user("Test")])
        assert response.content == "From async callable"
    
    @pytest.mark.asyncio
    async def test_callable_adapter_sync(self):
        """Test using a sync callable as adapter."""
        def my_sync_model(messages):
            return ModelResponse(content="From sync callable")
        
        adapter = CallableModelAdapter(my_sync_model)
        response = await adapter.generate([Message.user("Test")])
        assert response.content == "From sync callable"
    
    @pytest.mark.asyncio
    async def test_callable_with_string_return(self):
        """Test callable that returns a string."""
        def my_model(messages):
            return "Simple string response"
        
        adapter = CallableModelAdapter(my_model)
        response = await adapter.generate([Message.user("Test")])
        assert response.content == "Simple string response"
    
    @pytest.mark.asyncio
    async def test_callable_receives_messages(self):
        """Test that callable receives messages correctly."""
        received_messages = []
        
        async def capture_model(messages):
            received_messages.extend(messages)
            return ModelResponse(content="OK")
        
        adapter = CallableModelAdapter(capture_model)
        test_messages = [
            Message.system("System"),
            Message.user("User"),
        ]
        await adapter.generate(test_messages)
        
        assert len(received_messages) == 2
        assert received_messages[0].role == "system"
        assert received_messages[1].role == "user"


class TestSyncModelAdapter:
    """Tests for SyncModelAdapter."""
    
    @pytest.mark.asyncio
    async def test_sync_adapter(self):
        """Test creating a sync adapter."""
        class MySyncAdapter(SyncModelAdapter):
            def generate_sync(self, messages: List[Message]) -> ModelResponse:
                return ModelResponse(content="From sync adapter")
        
        adapter = MySyncAdapter()
        response = await adapter.generate([Message.user("Test")])
        assert response.content == "From sync adapter"
    
    @pytest.mark.asyncio
    async def test_sync_adapter_receives_messages(self):
        """Test that sync adapter receives messages correctly."""
        received_messages = []
        
        class CaptureSyncAdapter(SyncModelAdapter):
            def generate_sync(self, messages: List[Message]) -> ModelResponse:
                received_messages.extend(messages)
                return ModelResponse(content="OK")
        
        adapter = CaptureSyncAdapter()
        test_messages = [
            Message.system("System"),
            Message.user("User"),
        ]
        await adapter.generate(test_messages)
        
        assert len(received_messages) == 2
        assert received_messages[0].role == "system"
        assert received_messages[1].role == "user"


class TestDeepSeekAdapter:
    """Tests for the DeepSeek adapter."""

    def test_factory_uses_environment(self, monkeypatch):
        """Test factory resolves the API key and base URL from env vars."""
        pytest.importorskip("openai")
        captured = {}

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
        monkeypatch.setenv("DEEPSEEK_API_BASE", "https://example.test")
        monkeypatch.setattr("openai.AsyncOpenAI", FakeAsyncOpenAI)

        adapter = create_deepseek_model()

        assert isinstance(adapter, DeepSeekAdapter)
        assert adapter.model == "deepseek-chat"
        assert captured["api_key"] == "env-key"
        assert captured["base_url"] == "https://example.test"

    def test_factory_default_base_url(self, monkeypatch):
        """Test the default DeepSeek base URL is used when unset."""
        pytest.importorskip("openai")
        captured = {}

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
        monkeypatch.delenv("DEEPSEEK_API_BASE", raising=False)
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
        monkeypatch.setattr("openai.AsyncOpenAI", FakeAsyncOpenAI)

        create_deepseek_model()

        assert captured["base_url"] == "https://api.deepseek.com"

    def test_factory_alternate_base_url_env(self, monkeypatch):
        """Test DEEPSEEK_BASE_URL is also supported."""
        pytest.importorskip("openai")
        captured = {}

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
        monkeypatch.delenv("DEEPSEEK_API_BASE", raising=False)
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://base-url.test")
        monkeypatch.setattr("openai.AsyncOpenAI", FakeAsyncOpenAI)

        create_deepseek_model()

        assert captured["base_url"] == "https://base-url.test"

    def test_factory_explicit_args_override_env(self, monkeypatch):
        """Test explicit arguments take precedence over environment variables."""
        pytest.importorskip("openai")
        captured = {}

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
        monkeypatch.setenv("DEEPSEEK_API_BASE", "https://env.test")
        monkeypatch.setattr("openai.AsyncOpenAI", FakeAsyncOpenAI)

        adapter = create_deepseek_model(
            model="deepseek-reasoner",
            api_key="explicit-key",
            base_url="https://explicit.test",
        )

        assert adapter.model == "deepseek-reasoner"
        assert captured["api_key"] == "explicit-key"
        assert captured["base_url"] == "https://explicit.test"

    def test_factory_requires_api_key(self, monkeypatch):
        """Test a clear error is raised when no API key is available."""
        pytest.importorskip("openai")
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            create_deepseek_model()

    @pytest.mark.asyncio
    async def test_adapter_generate(self):
        """Test generating a response through an OpenAI-compatible client."""
        from types import SimpleNamespace

        captured = {}

        class FakeCompletions:
            async def create(self, **params):
                captured.update(params)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="DeepSeek reply")
                        )
                    ]
                )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )
        adapter = DeepSeekAdapter(client)

        response = await adapter.generate([Message.user("Hello")])

        assert response.content == "DeepSeek reply"
        assert captured["model"] == "deepseek-chat"
        assert captured["messages"] == [{"role": "user", "content": "Hello"}]
        assert "temperature" not in captured
        assert "max_tokens" not in captured

    def test_deepseek_exports(self):
        """Test DeepSeek helpers are exported from the package root."""
        import scope
        import scope.models

        assert scope.DeepSeekAdapter is scope.models.DeepSeekAdapter
        assert scope.create_deepseek_model is create_deepseek_model


class TestModelProtocol:
    """Tests for ModelProtocol compliance."""
    
    def test_protocol_compliance(self):
        """Test that adapters comply with ModelProtocol."""
        class MyAdapter(BaseModelAdapter):
            async def generate(self, messages: List[Message]) -> ModelResponse:
                return ModelResponse(content="Test")
        
        adapter = MyAdapter()
        
        # Should have generate method
        assert hasattr(adapter, 'generate')
        assert callable(adapter.generate)
