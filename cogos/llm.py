from __future__ import annotations

import json
from typing import List, Literal, Optional

from .pyd_compat import BaseModel
from .util import extract_first_json_object


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatModel:
    def generate_text(self, messages: List[ChatMessage], *, temperature: float = 0.2, max_tokens: int = 800) -> str:
        raise NotImplementedError

    def generate_json(
        self, messages: List[ChatMessage], schema: type[BaseModel], *, temperature: float = 0.2, max_tokens: int = 1200
    ) -> BaseModel:
        txt = self.generate_text(messages, temperature=temperature, max_tokens=max_tokens)
        data = extract_first_json_object(txt)
        return schema(**data)


class StubChatModel(ChatModel):
    def generate_text(self, messages: List[ChatMessage], *, temperature: float = 0.0, max_tokens: int = 256) -> str:
        # Always abstain; forces grounded/tool-only behavior.
        return json.dumps({"claims": [], "draft": "I don't know.", "proactive": []})


class LlamaCppChatModel(ChatModel):
    def __init__(
        self, model_path: str, *, n_ctx: int = 4096, n_threads: Optional[int] = None, n_gpu_layers: int = 0
    ):
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:
            raise ImportError("llama-cpp-python not installed. pip install llama-cpp-python") from e
        self._llm = Llama(
            model_path=model_path,
            n_ctx=int(n_ctx),
            n_threads=n_threads,
            n_gpu_layers=int(n_gpu_layers),
        )

    def _fmt(self, messages: List[ChatMessage]) -> str:
        # Generic formatting; for best results use a model-specific chat template.
        parts: List[str] = []
        for m in messages:
            parts.append(f"{m.role.upper()}: {m.content}")
        parts.append("ASSISTANT:")
        return "\n".join(parts)

    def generate_text(self, messages: List[ChatMessage], *, temperature: float = 0.2, max_tokens: int = 800) -> str:
        prompt = self._fmt(messages)
        out = self._llm(
            prompt,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            stop=["USER:", "SYSTEM:"],
        )
        return str(out["choices"][0]["text"]).strip()

