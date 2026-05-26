from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIError, OpenAI, RateLimitError

from judgebench.judge.parser import extract_json_object


class JudgeClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatRequest:
    system: str
    user: str
    temperature: float = 0.2
    max_tokens: int = 800


class OpenAICompatibleClient:
    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def chat_json(self, request: ChatRequest, *, retries: int = 3) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    messages=[
                        {"role": "system", "content": request.system},
                        {"role": "user", "content": request.user},
                    ],
                )
                text = (response.choices[0].message.content or "").strip()
                obj = extract_json_object(text)
                if not isinstance(obj, dict):
                    raise JudgeClientError("Expected a JSON object.")
                return obj
            except (APIError, APIConnectionError, RateLimitError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(0.5 * (2 ** (attempt - 1)))
                    continue
                raise JudgeClientError(f"LLM request failed after {retries} attempts: {exc}") from exc
        raise JudgeClientError(f"LLM request failed: {last_error}")

