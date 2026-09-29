"""Unit tests for the meal parser wrapper (no real model needed)."""

from typing import Any

import pytest

from app.parser import MealParser, ParseStatus, validate_output
from app.prompts import SYSTEM_PROMPT


class FakeLlm:
    """Returns a canned completion (or raises) instead of running a model."""

    def __init__(
        self,
        content: str = "",
        finish_reason: str = "stop",
        error: Exception | None = None,
    ) -> None:
        self.content = content
        self.finish_reason = finish_reason
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {
            "choices": [
                {
                    "message": {"content": self.content},
                    "finish_reason": self.finish_reason,
                }
            ]
        }


@pytest.mark.parametrize(
    ("raw_output", "expected"),
    [
        ('{"items": [{"name": "Muz", "amount": "1"}]}', ParseStatus.SUCCESS),
        ('{"items": []}', ParseStatus.EMPTY),
        (
            '{"items": [{"name": "Muz", "amount": "1"},'
            ' {"name": "", "amount": "2 adet"}]}',
            ParseStatus.PARTIAL,
        ),
        ('{"items": [{"name": "  ", "amount": "2 adet"}]}', ParseStatus.INVALID_OUTPUT),
        ('{"items": [{"name": "Muz"}]}', ParseStatus.INVALID_OUTPUT),
        ("Tabii, işte haftalık diyet listeniz", ParseStatus.INVALID_OUTPUT),
        ('{"foods": []}', ParseStatus.INVALID_OUTPUT),
        ('[{"name": "Muz", "amount": "1"}]', ParseStatus.INVALID_OUTPUT),
    ],
)
def test_validate_output_status(raw_output: str, expected: ParseStatus) -> None:
    assert validate_output(raw_output).status == expected


def test_blank_name_is_rejected_explicitly_not_dropped() -> None:
    result = validate_output(
        '{"items": [{"name": "Muz", "amount": "1"}, {"name": "", "amount": "2 adet"}]}'
    )
    assert [item.name for item in result.items] == ["Muz"]
    assert result.rejected_items[0].raw == {"name": "", "amount": "2 adet"}
    assert "blank" in result.rejected_items[0].reason


def test_empty_amount_is_kept_for_the_normalizer_to_flag() -> None:
    result = validate_output('{"items": [{"name": "Pilav", "amount": ""}]}')
    assert result.status == ParseStatus.SUCCESS
    assert result.items[0].amount == ""


def test_whitespace_is_stripped() -> None:
    result = validate_output('{"items": [{"name": "  Muz ", "amount": " 1 "}]}')
    assert (result.items[0].name, result.items[0].amount) == ("Muz", "1")


def test_parse_uses_model_card_settings_without_grammar_by_default() -> None:
    fake = FakeLlm('{"items": []}')
    MealParser(fake).parse("2 yumurta")

    call = fake.calls[0]
    assert call["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert call["messages"][1] == {"role": "user", "content": "2 yumurta"}
    assert call["temperature"] == 0
    assert call["stop"] == ["<|im_end|>"]
    assert call["response_format"] is None


def test_parse_with_grammar_sends_json_schema() -> None:
    fake = FakeLlm('{"items": []}')
    MealParser(fake, use_grammar=True).parse("2 yumurta")

    response_format = fake.calls[0]["response_format"]
    assert response_format["type"] == "json_object"
    assert "items" in response_format["schema"]["properties"]


def test_truncated_output_is_invalid_even_if_it_looks_like_json() -> None:
    fake = FakeLlm('{"items": [{"name": "Muz", "amount": "1"}]}', "length")
    result = MealParser(fake, max_tokens=20).parse("1 muz")
    assert result.status == ParseStatus.INVALID_OUTPUT
    assert "truncated" in (result.error or "")


def test_inference_exception_becomes_error_status() -> None:
    fake = FakeLlm(error=RuntimeError("llama_decode failed"))
    result = MealParser(fake).parse("1 muz")
    assert result.status == ParseStatus.ERROR
    assert "RuntimeError" in (result.error or "")
    assert result.latency_ms is not None
