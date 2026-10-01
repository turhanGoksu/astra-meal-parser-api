"""LLM tie-breaker (Step 8): is one of the retrieved candidates the same food?

Embeddings measure "how related", not "same food": dried grapes and grapes
are close in meaning but about 4x apart in calories. The judge sees the item
name and a few candidate foods (names and kcal per 100 g) and must answer
with one of the candidate ids or "none". Its answer is validated like any
other model output; on any failure the item stays unmatched.

Privacy: only the item name is sent to the provider, never the meal text.
"""

import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Protocol

import httpx
from pydantic import BaseModel, ValidationError

from astra_nutrition.matcher import FoodDetails, JudgeVerdict

logger = logging.getLogger(__name__)

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

SYSTEM_PROMPT = (
    "You are a strict judge for a nutrition app. Decide whether the food item "
    "a user ate is the SAME food as one of the candidate foods from a nutrition "
    "table, so that the candidate's nutrition per 100 g can be used for it.\n"
    "Same food: another language (Turkish or English), a spelling mistake, a "
    "synonym, a plural, or a plain preparation that barely changes nutrition "
    "(sliced, grilled or steamed without added fat).\n"
    "Not the same food: a dish made with the candidate; a dried, fried, "
    "breaded, sweetened, flavored or diet version; a different part or "
    "product; anything whose calories per 100 g would clearly differ from the "
    "candidate's.\n"
    "If no candidate is the same food, answer none. Never invent an id. "
    'Return only JSON: {"food_id": "<candidate id>"} or {"food_id": "none"}.'
)


class JudgeAnswer(BaseModel):
    """Expected shape of the judge's JSON answer."""

    food_id: str


class LlmProvider(Protocol):
    """One chat call that returns JSON text."""

    name: str
    model: str

    def complete_json(self, system: str, user: str) -> str: ...


class GroqProvider:
    """Groq's OpenAI-compatible chat completions endpoint."""

    name = "groq"

    def __init__(self, client: httpx.Client, api_key: str, model: str) -> None:
        self._client = client
        self._api_key = api_key
        self.model = model

    def complete_json(self, system: str, user: str) -> str:
        response = self._client.post(
            GROQ_CHAT_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class GeminiProvider:
    """Gemini's REST generateContent endpoint (key in a header, not the URL)."""

    name = "gemini"

    def __init__(self, client: httpx.Client, api_key: str, model: str) -> None:
        self._client = client
        self._api_key = api_key
        self.model = model

    def complete_json(self, system: str, user: str) -> str:
        response = self._client.post(
            f"{GEMINI_BASE_URL}/models/{self.model}:generateContent",
            headers={"x-goog-api-key": self._api_key},
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "temperature": 0,
                    "responseMimeType": "application/json",
                },
            },
        )
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]


class RateLimiter:
    """Spaces calls to stay under a requests-per-minute limit (thread-safe)."""

    def __init__(
        self,
        rpm: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._interval = 60.0 / rpm
        self._clock = clock
        self._sleep = sleep
        self._next_at = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            delay = self._next_at - self._clock()
            if delay > 0:
                self._sleep(delay)
            self._next_at = max(self._clock(), self._next_at) + self._interval


def build_prompt(item_name: str, candidates: list[FoodDetails]) -> str:
    """User message: the item (JSON-quoted, so it reads as data) and candidates."""
    lines = [f"Item: {json.dumps(item_name, ensure_ascii=False)}", "Candidates:"]
    lines += [
        f"- id: {c.food_id} | {c.name_en} / {c.name_tr} | "
        f"{c.kcal_100g:g} kcal per 100 g"
        for c in candidates
    ]
    return "\n".join(lines)


def parse_verdict(raw: str, allowed: set[str]) -> JudgeVerdict:
    """Validate the judge's answer: a candidate id, "none", or an error."""
    try:
        answer = JudgeAnswer.model_validate_json(raw)
    except ValidationError as exc:
        return JudgeVerdict(None, f"invalid judge output: {exc.errors()[0]['msg']}")
    food_id = answer.food_id.strip()
    if food_id.lower() == "none":
        return JudgeVerdict(None)
    if food_id not in allowed:
        return JudgeVerdict(None, f"unknown food id {food_id!r}")
    return JudgeVerdict(food_id)


class FoodJudge:
    """Asks an LLM provider to pick a candidate or none, with retries."""

    def __init__(
        self,
        provider: LlmProvider,
        rate_limiter: RateLimiter | None = None,
        max_retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._provider = provider
        self._limiter = rate_limiter
        self._max_retries = max_retries
        self._sleep = sleep

    def choose(self, item_name: str, candidates: list[FoodDetails]) -> JudgeVerdict:
        if not candidates:
            return JudgeVerdict(None)
        try:
            raw = self._call(build_prompt(item_name, candidates))
        except Exception as exc:  # provider down or rate limited: stay unmatched
            logger.warning("LLM judge failed: %s", type(exc).__name__)
            return JudgeVerdict(None, f"{type(exc).__name__}: {exc}")
        return parse_verdict(raw, {c.food_id for c in candidates})

    def _call(self, user: str) -> str:
        for attempt in range(self._max_retries + 1):
            if self._limiter is not None:
                self._limiter.wait()
            try:
                return self._provider.complete_json(SYSTEM_PROMPT, user)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status not in RETRYABLE_STATUS or attempt == self._max_retries:
                    raise
                delay = _retry_after(exc.response) or 2.0**attempt
            except httpx.TransportError:
                if attempt == self._max_retries:
                    raise
                delay = 2.0**attempt
            self._sleep(delay)
        raise AssertionError("unreachable")


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers["retry-after"])
    except (KeyError, ValueError):
        return None


PROVIDERS: dict[str, type[GroqProvider] | type[GeminiProvider]] = {
    "groq": GroqProvider,
    "gemini": GeminiProvider,
}


def build_judge(
    provider: str,
    api_key: str,
    model: str,
    rpm: int,
    timeout_seconds: float = 30.0,
    client: httpx.Client | None = None,
) -> FoodJudge:
    """A judge for "groq" or "gemini", rate-limited to ``rpm`` requests/minute.

    Model names and free-tier limits change over time, so the caller passes
    them in (the service reads them from .env).
    """
    return FoodJudge(
        build_provider(provider, api_key, model, timeout_seconds, client),
        RateLimiter(rpm),
    )


def build_provider(
    provider: str,
    api_key: str,
    model: str,
    timeout_seconds: float = 30.0,
    client: httpx.Client | None = None,
) -> LlmProvider:
    """A raw provider (no rate limiting or retries)."""
    if provider not in PROVIDERS:
        raise ValueError(
            f"unknown provider {provider!r}; use one of {sorted(PROVIDERS)}"
        )
    client = client or httpx.Client(timeout=timeout_seconds)
    return PROVIDERS[provider](client, api_key, model)
