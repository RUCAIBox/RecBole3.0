#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat server for LLM4RS (transformers backend, no vLLM)."""

from __future__ import annotations

import argparse
import threading
import time
import uuid
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_SYSTEM_PROMPT = (
    "You are a recommender. Reply with ONLY the ranking letters requested in the user message "
    "(for example A B C D E), separated by spaces. No explanation or other text."
)


def _prepare_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if any(message.get("role") == "system" for message in messages):
        return messages
    return [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, *messages]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 10
    temperature: float = 0.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: str | list[str] | None = None


class GenerationBackend:
    def __init__(self, model_path: str, device: str, dtype: str) -> None:
        torch_dtype = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self._lock = threading.Lock()

    def complete(self, messages: list[dict[str, str]], *, max_tokens: int, temperature: float, stop: Any) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Qwen3.5: skip long chain-of-thought for short ranking answers
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        stop_strings: list[str] | None
        if stop is None:
            stop_strings = None
        elif isinstance(stop, str):
            stop_strings = [stop] if stop else None
        else:
            stop_strings = [str(item) for item in stop if str(item)]

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_tokens),
            "do_sample": float(temperature) > 0.0,
            "temperature": max(float(temperature), 1e-5),
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if stop_strings:
            generate_kwargs["stop_strings"] = stop_strings
            generate_kwargs["tokenizer"] = self.tokenizer

        with self._lock:
            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, **generate_kwargs)
        prompt_len = int(inputs["input_ids"].shape[-1])
        return self.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()


def build_app(backend: GenerationBackend, served_model_name: str) -> FastAPI:
    app = FastAPI(title="LLM4RS Transformers Server")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        messages = _prepare_messages([message.model_dump() for message in request.messages])
        content = backend.complete(
            messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop,
        )
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model or served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI-compatible server for LLM4RS.")
    parser.add_argument("--model-path", default="/home/cce/wzc/Qwen3.5-4B")
    parser.add_argument("--served-model-name", default="Qwen3.5-4B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    args = parser.parse_args()

    print(f"[llm4rs-server] loading {args.model_path} on {args.device} ...")
    backend = GenerationBackend(args.model_path, args.device, args.dtype)
    print(f"[llm4rs-server] ready at http://{args.host}:{args.port}/v1/chat/completions")
    uvicorn.run(build_app(backend, args.served_model_name), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
