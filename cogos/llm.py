from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, TypeVar, cast

from .pyd_compat import BaseModel, _model_json_schema
from .util import extract_first_json_object, short


_TModel = TypeVar("_TModel", bound=BaseModel)


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

_PLAN_GBNF_DYNAMIC_TOOLS = r"""
root ::= plan

plan ::= "{" ws "\"steps\"" ws ":" ws steps ws "}" ws
steps ::= "[" ws (headsteps ws "," ws)? respondstep ws "]" ws

headsteps ::= headstep (ws "," ws headstep)*
headstep ::= memorysearchstep | toolcallstep | writenotestep | createtaskstep

respondstep ::= "{" ws "\"type\"" ws ":" ws "\"respond\"" ws "," ws "\"style\"" ws ":" ws string "}" ws

memorysearchstep ::= "{" ws "\"type\"" ws ":" ws "\"memory_search\"" ws "," ws "\"query\"" ws ":" ws string ws "," ws "\"k\"" ws ":" ws int "}" ws
toolcallstep ::= "{" ws "\"type\"" ws ":" ws "\"tool_call\"" ws "," ws "\"tool\"" ws ":" ws toolname ws "," ws "\"arguments\"" ws ":" ws object "}" ws
writenotestep ::= "{" ws "\"type\"" ws ":" ws "\"write_note\"" ws "," ws "\"title\"" ws ":" ws string ws "," ws "\"content\"" ws ":" ws string ws "," ws "\"tags\"" ws ":" ws stringarray ws "," ws "\"confidence\"" ws ":" ws number "}" ws
createtaskstep ::= "{" ws "\"type\"" ws ":" ws "\"create_task\"" ws "," ws "\"title\"" ws ":" ws string ws "," ws "\"description\"" ws ":" ws string ws "," ws "\"priority\"" ws ":" ws int ws "," ws "\"payload\"" ws ":" ws object "}" ws

__TOOLNAME_RULE__

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


def _gbnf_literal_for_json_string(s: str) -> str:
    """
    Return a GBNF literal that matches the exact JSON string encoding of `s`.

    Example: s="calc" -> "\"calc\""
    """
    js = json.dumps(str(s), ensure_ascii=False)
    escaped = js.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_plan_gbnf(tool_names: Sequence[str]) -> str:
    """
    Build a plan grammar that constrains tool names to the provided set.
    Falls back to the static grammar if tool_names is empty.
    """
    names = sorted({str(n).strip() for n in tool_names if str(n).strip()})
    if not names:
        return _PLAN_GBNF
    alts = " | ".join(_gbnf_literal_for_json_string(n) for n in names)
    tool_rule = f"toolname ::= ( {alts} ) ws"
    return _PLAN_GBNF_DYNAMIC_TOOLS.replace("__TOOLNAME_RULE__", tool_rule)


_REASONER_GBNF = r"""
root ::= answer

answer ::= "{" ws "\"claims\"" ws ":" ws claims ws "," ws "\"draft\"" ws ":" ws string ws "," ws "\"proactive\"" ws ":" ws "[" ws "]" ws "}" ws

claims ::= "[" ws "]" ws | "[" ws claim ws "]" ws | "[" ws claim ws "," ws claim ws "]" ws | "[" ws claim ws "," ws claim ws "," ws claim ws "]" ws

claim ::= "{" ws "\"text\"" ws ":" ws string ws "," ws "\"evidence_ids\"" ws ":" ws evidlist ws "," ws "\"support_span_ids\"" ws ":" ws spanidlist ws "," ws "\"kind\"" ws ":" ws kind "}" ws

evidlist ::= "[" ws string ws "]" ws
spanidlist ::= "[" ws int ws "]" ws | "[" ws int ws "," ws int ws "]" ws

kind ::= "\"fact\"" ws | "\"math\"" ws | "\"inference\"" ws

value ::= object | array | string | number | ("true" | "false" | "null") ws

object ::= "{" ws ( member (ws "," ws member)* )? "}" ws
member ::= string ":" ws value

array ::= "[" ws ( value (ws "," ws value)* )? "]" ws
objectarray ::= "[" ws "]" ws | "[" ws object ws "]" ws | "[" ws object ws "," ws object ws "]" ws
stringarray ::= "[" ws ( string (ws "," ws string)* )? "]" ws

string ::= "\"" ( [^"\\] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F]{4} ) )* "\"" ws
number ::= "-"? ("0" | [1-9] [0-9]* ) ( "." [0-9]+ )? ( [eE] [+-]? [0-9]+ )? ws
int ::= "-"? ("0" | [1-9] [0-9]* ) ws

ws ::= [ \t\n\r]*
"""


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatModel:
    name: str = "chat_model"

    def generate_text(self, messages: List[ChatMessage], *, temperature: float = 0.2, max_tokens: int = 800) -> str:
        raise NotImplementedError

    def generate_json(
        self,
        messages: List[ChatMessage],
        schema: type[_TModel],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> _TModel:
        txt = self.generate_text(messages, temperature=temperature, max_tokens=max_tokens)
        data = extract_first_json_object(txt)
        return cast(_TModel, schema(**data))


class StubChatModel(ChatModel):
    name = "stub"

    def generate_text(self, messages: List[ChatMessage], *, temperature: float = 0.0, max_tokens: int = 256) -> str:
        # Always abstain; forces grounded/tool-only behavior.
        return json.dumps({"claims": [], "draft": "I don't know.", "proactive": []})


class LlamaCppChatModel(ChatModel):
    name = "llama_cpp"

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ):
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:
            raise ImportError("llama-cpp-python not installed. pip install llama-cpp-python") from e
        try:
            from llama_cpp import LlamaGrammar  # type: ignore
        except Exception:
            LlamaGrammar = None  # type: ignore[assignment]
        # Construct kwargs defensively to support multiple llama-cpp-python versions.
        init_sig = inspect.signature(getattr(Llama, "__init__"))
        llm_kwargs: Dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": int(n_ctx),
            "n_threads": n_threads,
            "n_gpu_layers": int(n_gpu_layers),
        }
        if "verbose" in init_sig.parameters:
            llm_kwargs["verbose"] = bool(verbose)
        self._llm = Llama(**llm_kwargs)
        self._grammar_cls = LlamaGrammar
        self._plan_tool_names: Optional[List[str]] = None

    def set_plan_tool_names(self, tool_names: Sequence[str]) -> None:
        """
        Configure the set of tool names that the Plan grammar will allow.

        If unset/empty, the planner grammar allows any string tool name.
        """
        names = sorted({str(n).strip() for n in tool_names if str(n).strip()})
        self._plan_tool_names = names if names else None

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
            gbnf = build_plan_gbnf(self._plan_tool_names or [])
            try:
                return self._grammar_cls.from_string(gbnf)  # type: ignore[no-any-return]
            except Exception:
                return None
        if (
            schema.__module__ == "cogos.reasoner"
            and schema.__qualname__.endswith("LLMReasoner._Schema")
            and hasattr(self._grammar_cls, "from_string")
        ):
            try:
                return self._grammar_cls.from_string(_REASONER_GBNF)  # type: ignore[no-any-return]
            except Exception:
                return None
        if hasattr(self._grammar_cls, "from_json_schema"):
            try:
                schema_json = json.dumps(_model_json_schema(schema))
                return self._grammar_cls.from_json_schema(schema_json)  # type: ignore[no-any-return]
            except Exception:
                pass
        if hasattr(self._grammar_cls, "from_string"):
            try:
                return self._grammar_cls.from_string(_JSON_OBJECT_GBNF)  # type: ignore[no-any-return]
            except Exception:
                return None
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
        self,
        messages: List[ChatMessage],
        schema: type[_TModel],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> _TModel:
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
                    fallback_parts: List[str] = []
                    for m in messages:
                        fallback_parts.append(f"{m.role.upper()}: {m.content}")
                    fallback_parts.append("ASSISTANT:")
                    prompt = "\n".join(fallback_parts)
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
        return cast(_TModel, schema(**data))


class OllamaChatModel(ChatModel):
    """
    Minimal Ollama backend (no extra deps).

    Uses the local Ollama HTTP API (default host: http://localhost:11434).
    """

    name = "ollama"

    def __init__(self, *, host: str = "http://localhost:11434", model: str = ""):
        self.host = (host or "").rstrip("/") or "http://localhost:11434"
        self.model = model.strip()
        if not self.model:
            self.model = self._pick_default_model()

    def _url(self, path: str) -> str:
        p = path if path.startswith("/") else ("/" + path)
        return self.host + p

    def _pick_default_model(self) -> str:
        """
        Pick the first installed Ollama model (via /api/tags).
        """
        import urllib.request

        try:
            with urllib.request.urlopen(self._url("/api/tags"), timeout=1.5) as r:  # noqa: S310
                raw = r.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Ollama is not reachable. Start Ollama and ensure it listens on "
                f"{self.host} (or pass --ollama-host)."
            ) from e

        try:
            data = json.loads(raw)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Failed to parse Ollama /api/tags response: {short(raw, 400)}") from e

        models = data.get("models") or []
        for m in models:
            if isinstance(m, dict) and isinstance(m.get("name"), str) and m["name"].strip():
                return m["name"].strip()
        raise RuntimeError(
            "Ollama is running, but no models are installed. Run `ollama pull <model>` "
            "or pass --ollama-model."
        )

    def _chat(self, messages: List[ChatMessage], *, temperature: float, max_tokens: int, want_json: bool) -> str:
        import urllib.request
        import urllib.error

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }

        # Best-effort JSON constraint: some Ollama versions support format="json".
        if want_json:
            payload["format"] = "json"

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self._url("/api/chat"),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        def _do(req_: "urllib.request.Request") -> str:
            with urllib.request.urlopen(req_, timeout=60.0) as r:  # noqa: S310
                raw = r.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(raw)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"Ollama returned non-JSON: {short(raw, 600)}") from e
            msg = (data.get("message") or {}) if isinstance(data, dict) else {}
            content = msg.get("content") if isinstance(msg, dict) else None
            return str(content or "").strip()

        try:
            return _do(req)
        except urllib.error.HTTPError as e:
            # Retry without format if server doesn't accept it.
            try:
                raw = e.read().decode("utf-8", errors="replace")
            except Exception:
                raw = ""
            if want_json and ("format" in raw.lower() or "unknown field" in raw.lower()):
                payload.pop("format", None)
                body2 = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                req2 = urllib.request.Request(
                    self._url("/api/chat"),
                    data=body2,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                return _do(req2)
            raise RuntimeError(f"Ollama HTTP error: {e.code} {short(raw, 600)}") from e

    def generate_text(self, messages: List[ChatMessage], *, temperature: float = 0.2, max_tokens: int = 800) -> str:
        return self._chat(messages, temperature=temperature, max_tokens=max_tokens, want_json=False)

    def generate_json(
        self,
        messages: List[ChatMessage],
        schema: type[_TModel],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> _TModel:
        txt = self._chat(messages, temperature=temperature, max_tokens=max_tokens, want_json=True)
        try:
            data = extract_first_json_object(txt)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Failed to parse JSON from Ollama output: {short(txt, 600)}") from e
        return cast(_TModel, schema(**data))
