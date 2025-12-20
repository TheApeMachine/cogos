from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional


def utc_ts() -> float:
    return time.time()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def jdump(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True)


def jload(s: Optional[str]) -> Any:
    if not s:
        return None
    return json.loads(s)


def short(s: str, n: int = 200) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else (s[: n - 1] + "…")


def toks(s: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", (s or "").lower())


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def extract_first_json_object(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from arbitrary text by scanning balanced braces.

    This is intentionally conservative. If it can't parse safely, it raises ValueError.
    """
    s = text.strip()
    # Fast path: direct parse
    try:
        if s.startswith("{") and s.endswith("}"):
            return json.loads(s)
    except Exception:
        pass

    start = s.find("{")
    if start < 0:
        raise ValueError("No JSON object found.")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = s[start : i + 1]
                    return json.loads(chunk)
    raise ValueError("Unbalanced JSON braces.")

