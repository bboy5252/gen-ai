from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Optional, Type, TypeVar, get_args, get_origin

import httpx
from openai import OpenAI
from pydantic import TypeAdapter

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except Exception:
    pass


T = TypeVar("T")
_RATE_LOCK = threading.Lock()
_LAST_CALL_TS = 0.0
_HARMONY_RE = re.compile(r"<\|[^|>]*\|>")


def llm_available() -> bool:
    if os.environ.get("OFFLINE", "").lower() in {"1", "true", "yes"}:
        return False
    if os.environ.get("LLM_BASE_URL"):
        return bool(os.environ.get("LLM_AUTH_TOKEN") or os.environ.get("OPENAI_API_KEY"))
    return bool(os.environ.get("OPENAI_API_KEY"))


def get_model() -> str:
    return os.environ.get("LLM_MODEL", "gpt-4.1-mini")


def _throttle() -> None:
    interval = float(os.environ.get("LLM_CALL_INTERVAL_SEC", "0") or "0")
    if interval <= 0:
        return
    global _LAST_CALL_TS
    with _RATE_LOCK:
        now = time.monotonic()
        wait = _LAST_CALL_TS + interval - now
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL_TS = time.monotonic()


def make_raw_client() -> OpenAI:
    base = os.environ.get("LLM_BASE_URL")
    if base:
        key = os.environ.get("LLM_AUTH_TOKEN") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("LLM_BASE_URL задан, но LLM_AUTH_TOKEN/OPENAI_API_KEY пуст.")
        timeout = float(os.environ.get("LLM_TIMEOUT", "120") or "120")
        return OpenAI(api_key=key, base_url=base, http_client=httpx.Client(verify=False, timeout=timeout))
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Нужен OPENAI_API_KEY или LLM_BASE_URL + LLM_AUTH_TOKEN.")
    return OpenAI(api_key=key)


def _clean(text: str) -> str:
    text = _HARMONY_RE.sub("", text or "").strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _extract_first_json(text: str) -> Any:
    cleaned = _clean(text)
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(cleaned):
        if ch not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned, idx)
            return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("В ответе не найден валидный JSON: %r" % text[:300])


def _thinking_off_payload() -> dict[str, Any]:
    if os.environ.get("LLM_THINKING", "off").lower() in {"on", "1", "true", "yes"}:
        return {}
    return {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "reasoning_effort": "none",
    }


class _Completions:
    def __init__(self, client: OpenAI):
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_model: Type[T],
        max_retries: int = 2,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> T:
        wrap_list = get_origin(response_model) is list
        if wrap_list:
            item_type = get_args(response_model)[0]
            adapter = TypeAdapter(list[item_type])
            item_schema = TypeAdapter(item_type).json_schema()
            schema = {"type": "object", "properties": {"items": {"type": "array", "items": item_schema}}, "required": ["items"]}
        else:
            adapter = TypeAdapter(response_model)
            schema = adapter.json_schema()

        schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
        addendum = (
            "\n\nВерни один валидный JSON-объект по схеме ниже. "
            "Не добавляй markdown, комментарии или текст вокруг JSON.\n"
            f"{schema_text}"
        )
        if wrap_list:
            addendum += "\nМассив верни в поле items."

        msgs = [dict(message) for message in messages]
        sys_idx = next((i for i, m in enumerate(msgs) if m.get("role") == "system"), None)
        if sys_idx is None:
            msgs.insert(0, {"role": "system", "content": addendum.strip()})
        else:
            msgs[sys_idx]["content"] += addendum

        extra = _thinking_off_payload()
        last_error: Optional[Exception] = None
        raw = ""
        for _ in range(max_retries + 1):
            try:
                _throttle()
                try:
                    resp = self._client.chat.completions.create(
                        model=model,
                        messages=msgs,
                        response_format={"type": "json_object"},
                        temperature=temperature,
                        **extra,
                        **kwargs,
                    )
                except Exception as exc:
                    if extra and any(key in str(exc) for key in ("reasoning_effort", "chat_template_kwargs", "enable_thinking")):
                        extra = {}
                        _throttle()
                        resp = self._client.chat.completions.create(
                            model=model,
                            messages=msgs,
                            response_format={"type": "json_object"},
                            temperature=temperature,
                            **kwargs,
                        )
                    else:
                        raise
                raw = resp.choices[0].message.content or ""
                obj = _extract_first_json(raw)
                if wrap_list and isinstance(obj, dict) and "items" in obj:
                    obj = obj["items"]
                return adapter.validate_python(obj)
            except Exception as exc:
                last_error = exc
                msgs.append({"role": "assistant", "content": raw})
                msgs.append({"role": "user", "content": f"Ответ не прошел валидацию: {exc}. Верни только корректный JSON."})
        assert last_error is not None
        raise last_error


class _Chat:
    def __init__(self, client: OpenAI):
        self.completions = _Completions(client)


class JsonClient:
    def __init__(self, client: OpenAI):
        self.chat = _Chat(client)


def make_client() -> JsonClient:
    return JsonClient(make_raw_client())
