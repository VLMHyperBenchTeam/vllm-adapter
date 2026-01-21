import time
from typing import AsyncGenerator, Union, List, Dict, Any
from api_wrapper.backends.base import AbstractBackend
from api_wrapper.schemas.chat import (
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ChatCompletionResponseChoice,
    ChatCompletionUsage,
    ChatMessage
)

try:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.utils import random_uuid
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

class vLLMAdapter(AbstractBackend):
    """Adapter for vLLM AsyncLLMEngine."""
    
    def __init__(self, engine_params: Dict[str, Any]):
        self.engine_params = engine_params
        self.engine = None
        if VLLM_AVAILABLE:
            # In a real scenario, we would initialize AsyncLLMEngine here
            # self.engine = AsyncLLMEngine.from_engine_args(...)
            pass

    async def generate(self, request: ChatCompletionRequest) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionResponse, None]]:
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed or available.")
            
        if self.engine is None:
            # For demo/test purposes, if engine is not init, we raise or dummy
            raise ValueError("vLLM engine is not initialized.")

        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or 1024,
            **request.model_params
        )
        
        request_id = random_uuid()
        prompt = self._format_prompt(request.messages)
        
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        if request.stream:
            return self._stream_results(results_generator, request)
        else:
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            
            return self._create_response(final_output, request)

    def _format_prompt(self, messages: List[ChatMessage]) -> str:
        prompt = ""
        for msg in messages:
            prompt += f"{msg.role}: {msg.content}\n"
        prompt += "assistant: "
        return prompt

    async def _stream_results(self, generator, request: ChatCompletionRequest):
        async for request_output in generator:
            yield self._create_response(request_output, request)

    def _create_response(self, vllm_output, request: ChatCompletionRequest) -> ChatCompletionResponse:
        choice = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=vllm_output.outputs[0].text),
            finish_reason=vllm_output.outputs[0].finish_reason
        )
        
        usage = ChatCompletionUsage(
            prompt_tokens=len(vllm_output.prompt_token_ids),
            completion_tokens=len(vllm_output.outputs[0].token_ids),
            total_tokens=len(vllm_output.prompt_token_ids) + len(vllm_output.outputs[0].token_ids)
        )
        
        return ChatCompletionResponse(
            id=vllm_output.request_id,
            created=int(time.time()),
            model=request.model,
            choices=[choice],
            usage=usage
        )