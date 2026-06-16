from __future__ import annotations

import json
import os
import re
import warnings
from typing import Any, Type, TypeVar, get_args, get_origin

import httpx
from openai import OpenAI
from pydantic import TypeAdapter

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

T = TypeVar("T")

_HARMONY_RE = re.compile(r"<\|[^|>]*\|>")


def get_model() -> str:
    return os.environ.get("LLM_MODEL", "gpt-4.1-mini")


def get_max_tokens() -> int:
    return int(os.environ.get("LLM_MAX_TOKENS", "700"))


def _make_openai_client() -> OpenAI:
    base = os.environ.get("LLM_BASE_URL")

    if base:
        key = os.environ.get("LLM_AUTH_TOKEN") or os.environ.get("OPENAI_API_KEY")

        if not key:
            raise RuntimeError(
                "LLM_AUTH_TOKEN не задан. Либо экспортируй токен, либо положи LLM_AUTH_TOKEN=... в .env."
            )

        timeout = float(os.environ.get("LLM_TIMEOUT", "200"))
        http = httpx.Client(verify=False, timeout=timeout)

        return OpenAI(
            api_key=key,
            base_url=base,
            http_client=http,
        )

    key = os.environ.get("OPENAI_API_KEY")

    if not key:
        raise RuntimeError(
            "Ни LLM_BASE_URL, ни OPENAI_API_KEY не заданы. Сконфигурируй стенд через .env."
        )

    return OpenAI(api_key=key)


def _thinking_off_payload() -> dict:
    if os.environ.get("LLM_THINKING", "off").lower() in ("on", "1", "true", "yes"):
        return {}

    return {
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": False,
            }
        },
        "reasoning_effort": "none",
    }


def _clean(text: str) -> str:
    text = _HARMONY_RE.sub("", text).strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)

    return text.strip()


def _extract_first_json(text: str):
    t = _clean(text)
    decoder = json.JSONDecoder()

    for i, ch in enumerate(t):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(t, i)
                return obj
            except json.JSONDecodeError:
                continue

    raise ValueError(f"В ответе не найдено валидного JSON: {text[:300]!r}")


def _prepare_request_kwargs(extra: dict) -> dict:
    result = dict(extra)

    result.pop("max_completion_tokens", None)

    if "max_tokens" not in result:
        result["max_tokens"] = get_max_tokens()

    return result


class _Completions:
    def __init__(self, client: OpenAI):
        self._c = client

    def create(
        self,
        *,
        model: str,
        messages: list[dict],
        response_model: Type[T],
        max_retries: int = 1,
        temperature: float = 0.0,
        with_completion: bool = False,
        **kw: Any,
    ) -> T:
        wrap_list = get_origin(response_model) is list

        if wrap_list:
            item_type = get_args(response_model)[0]
            adapter = TypeAdapter(list[item_type])
            item_schema = TypeAdapter(item_type).json_schema()
            schema = {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": item_schema,
                    }
                },
                "required": ["items"],
            }
        else:
            adapter = TypeAdapter(response_model)
            schema = adapter.json_schema()

        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)

        addendum = (
            "\n\nВерни только один JSON-объект с данными заявки. "
            "Не копируй JSON-схему. "
            "Не возвращай поля $defs, properties, required, title, type или description. "
            "Не добавляй markdown, пояснения, комментарии или второй JSON-объект. "
            f"JSON должен соответствовать этой схеме:\n{schema_str}\n"
        )

        if wrap_list:
            addendum += " Массив верни в поле `items`."

        msgs = [dict(m) for m in messages]
        sys_i = next((i for i, m in enumerate(msgs) if m["role"] == "system"), None)

        if sys_i is not None:
            msgs[sys_i]["content"] = msgs[sys_i]["content"] + addendum
        else:
            msgs.insert(
                0,
                {
                    "role": "system",
                    "content": addendum.lstrip(),
                },
            )

        thinking_kw = _thinking_off_payload()

        def _call(extra: dict):
            request_kwargs = _prepare_request_kwargs({**kw, **extra})

            try:
                return self._c.chat.completions.create(
                    model=model,
                    messages=msgs,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    **request_kwargs,
                )
            except TypeError:
                safe = {
                    k: v
                    for k, v in request_kwargs.items()
                    if k not in {"reasoning_effort"}
                }

                return self._c.chat.completions.create(
                    model=model,
                    messages=msgs,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    **safe,
                )

        last_err: Exception | None = None
        raw = ""

        for _ in range(max_retries + 1):
            try:
                try:
                    resp = _call(thinking_kw)
                except Exception as sdk_err:
                    msg = str(sdk_err)

                    bad = (
                        "reasoning_effort" in msg
                        or "chat_template_kwargs" in msg
                        or "enable_thinking" in msg
                    )

                    if bad and thinking_kw:
                        thinking_kw = {}
                        resp = _call(thinking_kw)
                    else:
                        raise

                raw = resp.choices[0].message.content or ""
                obj = _extract_first_json(raw)

                if wrap_list and isinstance(obj, dict) and "items" in obj:
                    obj = obj["items"]

                result = adapter.validate_python(obj)

                if with_completion:
                    return result, resp

                return result

            except Exception as e:
                last_err = e

                msgs.append(
                    {
                        "role": "assistant",
                        "content": raw,
                    }
                )

                msgs.append(
                    {
                        "role": "user",
                        "content": f"Невалидный ответ: {e}. Верни ТОЛЬКО один корректный JSON по схеме.",
                    }
                )

        assert last_err is not None
        raise last_err


class _Chat:
    def __init__(self, client: OpenAI):
        self.completions = _Completions(client)


class JsonClient:
    def __init__(self, openai_client: OpenAI):
        self._c = openai_client
        self.chat = _Chat(openai_client)


def make_client() -> JsonClient:
    return JsonClient(_make_openai_client())


class _RawCompletions:
    def __init__(self, inner):
        self._inner = inner

    def create(self, **kw: Any):
        thinking = _thinking_off_payload()

        def _call(extra: dict):
            request_kwargs = _prepare_request_kwargs({**kw, **extra})

            try:
                return self._inner.create(**request_kwargs)
            except TypeError:
                safe = {
                    k: v
                    for k, v in request_kwargs.items()
                    if k not in {"reasoning_effort"}
                }

                return self._inner.create(**safe)

        try:
            return _call(thinking)
        except Exception as e:
            msg = str(e)

            bad = (
                "reasoning_effort" in msg
                or "chat_template_kwargs" in msg
                or "enable_thinking" in msg
            )

            if bad and thinking:
                return _call({})

            raise


class _RawChat:
    def __init__(self, inner):
        self.completions = _RawCompletions(inner.completions)


class RawClient:
    def __init__(self, openai_client: OpenAI):
        self._c = openai_client
        self.chat = _RawChat(openai_client.chat)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._c, name)


def make_raw_client() -> RawClient:
    return RawClient(_make_openai_client())