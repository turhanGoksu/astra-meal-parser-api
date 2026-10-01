"""Tests for building library objects from service settings."""

import pytest

from app.config import Settings
from app.factory import judge_from_settings
from astra_nutrition.judge import FoodJudge


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LLM_JUDGE_PROVIDER", "GROQ_API_KEY", "GROQ_MODEL", "GROQ_RPM"):
        monkeypatch.delenv(name, raising=False)


def test_judge_is_off_by_default() -> None:
    assert judge_from_settings(Settings(_env_file=None)) is None


def test_missing_settings_are_named() -> None:
    settings = Settings(_env_file=None, llm_judge_provider="groq")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY, GROQ_MODEL, GROQ_RPM"):
        judge_from_settings(settings)


def test_full_settings_build_a_judge() -> None:
    settings = Settings(
        _env_file=None,
        llm_judge_provider="gemini",
        gemini_api_key="k",
        gemini_model="m",
        gemini_rpm=10,
    )
    assert isinstance(judge_from_settings(settings), FoodJudge)
