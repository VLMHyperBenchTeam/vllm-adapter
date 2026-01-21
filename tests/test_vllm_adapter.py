import pytest
from unittest.mock import MagicMock, patch
from vllm_adapter.adapter import vLLMAdapter
from api_wrapper.schemas.chat import ChatCompletionRequest, ChatMessage

def test_vllm_adapter_init():
    params = {"tensor_parallel_size": 1}
    adapter = vLLMAdapter(engine_params=params)
    assert adapter.engine_params == params
    assert adapter.engine is None

def test_vllm_adapter_format_prompt():
    adapter = vLLMAdapter(engine_params={})
    messages = [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi"),
        ChatMessage(role="user", content="How are you?"),
    ]
    prompt = adapter._format_prompt(messages)
    assert "user: Hello" in prompt
    assert "assistant: Hi" in prompt
    assert "user: How are you?" in prompt
    assert prompt.endswith("assistant: ")

@pytest.mark.asyncio
async def test_vllm_adapter_generate_no_vllm():
    adapter = vLLMAdapter(engine_params={})
    request = ChatCompletionRequest(
        model="test",
        messages=[ChatMessage(role="user", content="test")]
    )
    
    with patch("vllm_adapter.adapter.VLLM_AVAILABLE", False):
        with pytest.raises(ImportError):
            await adapter.generate(request)