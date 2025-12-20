from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Dict, List, Literal, Optional

from .pyd_compat import BaseModel, _model_json_schema
from .util import extract_first_json_object, short


_JSON_OBJECT_GBNF = r"""
root ::= object

value ::= object | array | string | number | ("true" | "false" | "null") ws

object ::= "{" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}" ws
array ::= "[" ws ( value ("," ws value)* )? "]" ws

string ::= "\"" ( [^"\\] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F]{4} ) )* "\"" ws
number ::= "-"? ("0" | [1-9] [0-9]* ) ( "." [0-9]+ )? ( [eE] [+-]? [0-9]+ )? ws

ws ::= [ \t\n\r]*
"""

_PLAN_GBNF = r"""
root ::= plan

plan ::= "{" ws "\"steps\"" ws ":" ws steps ws "}" ws
steps ::= "[" ws (headsteps ws "," ws)? respondstep ws "]" ws

headsteps ::= headstep (ws "," ws headstep)*
headstep ::= memorysearchstep | toolcallstep | writenotestep | createtaskstep

respondstep ::= "{" ws "\"type\"" ws ":" ws "\"respond\"" ws "," ws "\"style\"" ws ":" ws string "}" ws

memorysearchstep ::= "{" ws "\"type\"" ws ":" ws "\"memory_search\"" ws "," ws "\"query\"" ws ":" ws string ws "," ws "\"k\"" ws ":" ws int "}" ws
toolcallstep ::= "{" ws "\"type\"" ws ":" ws "\"tool_call\"" ws "," ws "\"tool\"" ws ":" ws string ws "," ws "\"arguments\"" ws ":" ws object "}" ws
writenotestep ::= "{" ws "\"type\"" ws ":" ws "\"write_note\"" ws "," ws "\"title\"" ws ":" ws string ws "," ws "\"content\"" ws ":" ws string ws "," ws "\"tags\"" ws ":" ws stringarray ws "," ws "\"confidence\"" ws ":" ws number "}" ws
createtaskstep ::= "{" ws "\"type\"" ws ":" ws "\"create_task\"" ws "," ws "\"title\"" ws ":" ws string ws "," ws "\"description\"" ws ":" ws string ws "," ws "\"priority\"" ws ":" ws int ws "," ws "\"payload\"" ws ":" ws object "}" ws

value ::= object | array | string | number | ("true" | "false" | "null") ws

object ::= "{" ws ( member (ws "," ws member)* )? "}" ws
member ::= string ":" ws value

array ::= "[" ws ( value (ws "," ws value)* )? "]" ws
stringarray ::= "[" ws ( string (ws "," ws string)* )? "]" ws

string ::= "\"" ( [^"\\] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F]{4} ) )* "\"" ws
number ::= "-"? ("0" | [1-9] [0-9]* ) ( "." [0-9]+ )? ( [eE] [+-]? [0-9]+ )? ws
int ::= "-"? ("0" | [1-9] [0-9]* ) ws

ws ::= [ \t\n\r]*
"""

_REASONER_GBNF = r"""
root ::= answer

answer ::= "{" ws "\"claims\"" ws ":" ws claims ws "," ws "\"draft\"" ws ":" ws string ws "," ws "\"proactive\"" ws ":" ws objectarray ws "}" ws

claims ::= "[" ws "]" ws
       | "[" ws claim ws "]" ws
       | "[" ws claim ws "," ws claim ws "]" ws
       | "[" ws claim ws "," ws claim ws "," ws claim ws "]" ws

claim ::= "{" ws "\"text\"" ws ":" ws string ws "," ws "\"evidence_ids\"" ws ":" ws evidlist ws "," ws "\"support_spans\"" ws ":" ws spanlist ws "," ws "\"kind\"" ws ":" ws kind "}" ws

evidlist ::= "[" ws string ws "]" ws
spanlist ::= "[" ws string ws "]" ws
         | "[" ws string ws "," ws string ws "]" ws

kind ::= "\"fact\"" ws | "\"math\"" ws | "\"inference\"" ws

value ::= object | array | string | number | ("true" | "false" | "null") ws

object ::= "{" ws ( member (ws "," ws member)* )? "}" ws
member ::= string ":" ws value

array ::= "[" ws ( value (ws "," ws value)* )? "]" ws
objectarray ::= "[" ws "]" ws
            | "[" ws object ws "]" ws
            | "[" ws object ws "," ws object ws "]" ws
stringarray ::= "[" ws ( string (ws "," ws string)* )? "]" ws

string ::= "\"" ( [^"\\] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F]{4} ) )* "\"" ws
number ::= "-"? ("0" | [1-9] [0-9]* ) ( "." [0-9]+ )? ( [eE] [+-]? [0-9]+ )? ws

ws ::= [ \t\n\r]*
"""


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
        try:
            from llama_cpp import LlamaGrammar  # type: ignore
        except Exception:
            LlamaGrammar = None  # type: ignore[assignment]
        self._llm = Llama(
            model_path=model_path,
            n_ctx=int(n_ctx),
            n_threads=n_threads,
            n_gpu_layers=int(n_gpu_layers),
        )
        self._grammar_cls = LlamaGrammar

    @staticmethod
    def _to_chat_messages(messages: List[ChatMessage]) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    @staticmethod
    def _extract_completion_text(resp: Dict[str, Any]) -> str:
        choices = resp.get("choices") or []
        if not choices:
            raise ValueError(f"llama_cpp returned no choices: {list(resp.keys())}")
        c0 = choices[0] or {}
        if isinstance(c0, dict):
            msg = c0.get("message")
            if isinstance(msg, dict) and "content" in msg:
                content = msg.get("content")
                if isinstance(content, (dict, list)):
                    return json.dumps(content, ensure_ascii=False)
                return str(content or "").strip()
            if "text" in c0:
                text = c0.get("text")
                if isinstance(text, (dict, list)):
                    return json.dumps(text, ensure_ascii=False)
                return str(text or "").strip()
        return str(c0).strip()

    def _chat_complete(self, messages: List[ChatMessage], *, temperature: float, max_tokens: int, extra: Dict[str, Any]) -> str:
        if not hasattr(self._llm, "create_chat_completion"):
            raise AttributeError("llama_cpp.Llama.create_chat_completion is not available in this install")
        fn: Callable[..., Any] = getattr(self._llm, "create_chat_completion")
        sig = inspect.signature(fn)

        kwargs: Dict[str, Any] = {"messages": self._to_chat_messages(messages)}
        if "temperature" in sig.parameters:
            kwargs["temperature"] = float(temperature)
        if "max_tokens" in sig.parameters:
            kwargs["max_tokens"] = int(max_tokens)

        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if accepts_kwargs:
            kwargs.update(extra)
        else:
            unknown = [k for k in extra.keys() if k not in sig.parameters]
            if unknown:
                raise TypeError(f"create_chat_completion does not accept: {unknown}")
            kwargs.update(extra)

        resp = fn(**kwargs)
        if not isinstance(resp, dict):
            raise TypeError(f"llama_cpp chat completion returned {type(resp).__name__}, expected dict")
        return self._extract_completion_text(resp)

    def _build_grammar_for_schema(self, schema: type[BaseModel]) -> Any:
        if self._grammar_cls is None:
            return None
        if schema.__module__ == "cogos.ir" and schema.__name__ == "Plan" and hasattr(self._grammar_cls, "from_string"):
            return self._grammar_cls.from_string(_PLAN_GBNF)  # type: ignore[no-any-return]
        if (
            schema.__module__ == "cogos.reasoner"
            and schema.__qualname__.endswith("LLMReasoner._Schema")
            and hasattr(self._grammar_cls, "from_string")
        ):
            return self._grammar_cls.from_string(_REASONER_GBNF)  # type: ignore[no-any-return]
        if hasattr(self._grammar_cls, "from_json_schema"):
            try:
                schema_json = json.dumps(_model_json_schema(schema))
                return self._grammar_cls.from_json_schema(schema_json)  # type: ignore[no-any-return]
            except Exception:
                pass
        if hasattr(self._grammar_cls, "from_string"):
            return self._grammar_cls.from_string(_JSON_OBJECT_GBNF)  # type: ignore[no-any-return]
        return None

    def generate_text(self, messages: List[ChatMessage], *, temperature: float = 0.2, max_tokens: int = 800) -> str:
        if hasattr(self._llm, "create_chat_completion"):
            return self._chat_complete(messages, temperature=temperature, max_tokens=max_tokens, extra={})

        # Fallback for older llama-cpp-python installs: raw completion with a basic transcript.
        parts: List[str] = []
        for m in messages:
            parts.append(f"{m.role.upper()}: {m.content}")
        parts.append("ASSISTANT:")
        prompt = "\n".join(parts)
        out = self._llm(prompt, max_tokens=int(max_tokens), temperature=float(temperature), stop=["USER:", "SYSTEM:"])
        if not isinstance(out, dict):
            raise TypeError(f"llama_cpp completion returned {type(out).__name__}, expected dict")
        return self._extract_completion_text(out)

    def generate_json(
        self, messages: List[ChatMessage], schema: type[BaseModel], *, temperature: float = 0.2, max_tokens: int = 1200
    ) -> BaseModel:
        grammar = self._build_grammar_for_schema(schema)

        txt: str
        if hasattr(self._llm, "create_chat_completion"):
            # Prefer schema-constrained decoding (via grammar) when available.
            if grammar is not None:
                try:
                    txt = self._chat_complete(
                        messages, temperature=temperature, max_tokens=max_tokens, extra={"grammar": grammar}
                    )
                except TypeError:
                    # Some llama-cpp-python versions don't plumb grammar through chat completions; fall back explicitly.
                    parts: List[str] = []
                    for m in messages:
                        parts.append(f"{m.role.upper()}: {m.content}")
                    parts.append("ASSISTANT:")
                    prompt = "\n".join(parts)
                    out = self._llm(
                        prompt,
                        max_tokens=int(max_tokens),
                        temperature=float(temperature),
                        grammar=grammar,
                        stop=["USER:", "SYSTEM:"],
                    )
                    if not isinstance(out, dict):
                        raise TypeError(f"llama_cpp completion returned {type(out).__name__}, expected dict")
                    txt = self._extract_completion_text(out)
            else:
                # If grammar isn't available, require explicit JSON support.
                fn = getattr(self._llm, "create_chat_completion")
                sig = inspect.signature(fn)
                accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                if ("response_format" not in sig.parameters) and (not accepts_kwargs):
                    raise RuntimeError(
                        "llama-cpp-python does not expose JSON constrained decoding (missing LlamaGrammar and response_format). "
                        "Upgrade llama-cpp-python to use --planner llm/--reasoner llm safely."
                    )
                txt = self._chat_complete(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra={"response_format": {"type": "json_object"}},
                )
        else:
            if grammar is None:
                raise RuntimeError(
                    "llama-cpp-python install is missing create_chat_completion and grammar support; cannot safely request JSON."
                )
            parts: List[str] = []
            for m in messages:
                parts.append(f"{m.role.upper()}: {m.content}")
            parts.append("ASSISTANT:")
            prompt = "\n".join(parts)
            out = self._llm(prompt, max_tokens=int(max_tokens), temperature=float(temperature), grammar=grammar, stop=["USER:", "SYSTEM:"])
            if not isinstance(out, dict):
                raise TypeError(f"llama_cpp completion returned {type(out).__name__}, expected dict")
            txt = self._extract_completion_text(out)

        try:
            data = extract_first_json_object(txt)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON from model output: {short(txt, 600)}") from e
        return schema(**data)
