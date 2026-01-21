# vLLM Adapter for VLMHyperBench

[Русский](README.ru.md)

Standard adapter for integrating the `vLLM` inference engine into the VLMHyperBench platform.

## Features

- Seamless integration with `vllm.AsyncLLMEngine`.
- Support for OpenAI-style chat completions.
- Optimized for high-throughput inference.

## Installation

Requires `api-wrapper` to be installed.

```bash
pip install -e .
```

## Usage

This package is intended to be loaded dynamically by the `api-wrapper` core.

**Discovery Config:**
```yaml
framework: "vllm"
backend_class: "vllm_adapter.adapter.vLLMAdapter"