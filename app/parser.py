"""Meal parser: GGUF model inference with validated JSON output.

Defense in depth:
- optionally, a grammar (built from ``MealParseSchema``) guarantees the SHAPE;
- Pydantic validation checks the MEANING we can check here (e.g. blank names);
- every call ends in an explicit ``ParseStatus``, never a silent guess.

Grammar-constrained decoding is OFF by default (provisional decision, small
sample): on 4 real meals it produced identical output at ~50% higher latency,
and on a non-meal input it turned a loud ``invalid_output`` into a silent
``success`` with an invented food. Re-evaluate with a larger parser eval set.
"""

import json
import logging
import threading
import time
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from llama_cpp import Llama
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class ParseStatus(StrEnum):
    """Outcome of one parse call."""

    SUCCESS = "success"  # all items valid
    PARTIAL = "partial"  # some items valid, some rejected
    EMPTY = "empty"  # valid output with no items
    INVALID_OUTPUT = "invalid_output"  # not JSON, wrong shape, or no valid item
    ERROR = "error"  # inference raised an exception


class ParsedItem(BaseModel):
    """One food item extracted by the model."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str
    amount: str  # may be empty: amount normalization flags it, never guesses

    # A field_validator (unlike Field(min_length=1)) is NOT part of the JSON
    # schema, so the grammar does not force the model to invent a name.
    @field_validator("name")
    @classmethod
    def name_not_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("name must not be blank")
        return value


class MealParseSchema(BaseModel):
    """Expected output shape. Its JSON schema becomes the decoding grammar."""

    items: list[ParsedItem]


class RejectedItem(BaseModel):
    """A raw item from the model output that failed validation."""

    raw: Any
    reason: str


class ParseResult(BaseModel):
    """Result of parsing one meal description."""

    status: ParseStatus
    items: list[ParsedItem] = Field(default_factory=list)
    rejected_items: list[RejectedItem] = Field(default_factory=list)
    raw_output: str | None = None
    error: str | None = None
    latency_ms: float | None = None


class ChatModel(Protocol):
    """The subset of ``llama_cpp.Llama`` we use (lets tests inject a fake)."""

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]: ...


def validate_output(raw_output: str) -> ParseResult:
    """Validate raw model text into a ParseResult, item by item."""
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        return ParseResult(
            status=ParseStatus.INVALID_OUTPUT,
            raw_output=raw_output,
            error=f"not valid JSON: {exc}",
        )

    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        return ParseResult(
            status=ParseStatus.INVALID_OUTPUT,
            raw_output=raw_output,
            error="expected an object with an 'items' list",
        )

    items: list[ParsedItem] = []
    rejected: list[RejectedItem] = []
    for raw_item in data["items"]:
        try:
            items.append(ParsedItem.model_validate(raw_item))
        except ValidationError as exc:
            reason = "; ".join(err["msg"] for err in exc.errors())
            rejected.append(RejectedItem(raw=raw_item, reason=reason))

    if not items and not rejected:
        status = ParseStatus.EMPTY
    elif not items:
        status = ParseStatus.INVALID_OUTPUT
    elif rejected:
        status = ParseStatus.PARTIAL
    else:
        status = ParseStatus.SUCCESS

    return ParseResult(
        status=status, items=items, rejected_items=rejected, raw_output=raw_output
    )


class MealParser:
    """Thread-safe wrapper around the GGUF meal parser. Load once, reuse."""

    def __init__(
        self, llm: ChatModel, use_grammar: bool = False, max_tokens: int = 512
    ) -> None:
        self._llm = llm
        # llama.cpp keeps one KV cache per model instance: never run two
        # generations on it at the same time.
        self._lock = threading.Lock()
        self._max_tokens = max_tokens
        self._response_format = (
            {"type": "json_object", "schema": MealParseSchema.model_json_schema()}
            if use_grammar
            else None
        )

    @classmethod
    def from_path(
        cls,
        model_path: Path,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        use_grammar: bool = False,
    ) -> "MealParser":
        """Load the GGUF model from disk (slow: call once at startup)."""
        llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,  # CPU only, same as inside Docker
            chat_format="chatml",
            verbose=False,
        )
        return cls(llm, use_grammar=use_grammar)

    def parse(self, meal_text: str) -> ParseResult:
        """Parse one meal description. Never raises: failures become a status."""
        start = time.perf_counter()
        try:
            with self._lock:
                response = self._llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": meal_text},
                    ],
                    temperature=0,
                    stop=["<|im_end|>"],
                    max_tokens=self._max_tokens,
                    response_format=self._response_format,
                )
            choice = response["choices"][0]
            raw_output = choice["message"]["content"] or ""
        except Exception as exc:  # isolate inference failures from the caller
            logger.exception("Parser inference failed")
            return ParseResult(
                status=ParseStatus.ERROR,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=_elapsed_ms(start),
            )

        if choice.get("finish_reason") == "length":
            result = ParseResult(
                status=ParseStatus.INVALID_OUTPUT,
                raw_output=raw_output,
                error=f"output truncated at max_tokens={self._max_tokens}",
            )
        else:
            result = validate_output(raw_output)
        result.latency_ms = _elapsed_ms(start)
        return result


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
