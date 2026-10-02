"""Builds library objects from the service settings (.env)."""

from app.config import Settings
from astra_nutrition.judge import FoodJudge, LlmProvider, build_judge, build_provider


def _judge_settings(settings: Settings, provider: str) -> tuple[str, str, int]:
    """API key, model and rpm for a provider; names missing .env variables."""
    key, model, rpm = {
        "groq": (settings.groq_api_key, settings.groq_model, settings.groq_rpm),
        "gemini": (settings.gemini_api_key, settings.gemini_model, settings.gemini_rpm),
    }[provider]
    prefix = provider.upper()
    missing = [
        f"{prefix}_{field}"
        for field, value in (("API_KEY", key), ("MODEL", model), ("RPM", rpm))
        if not value
    ]
    if missing:
        raise RuntimeError(f"{provider} judge needs {', '.join(missing)} in .env")
    assert key is not None and model is not None and rpm is not None
    return key.get_secret_value(), model, rpm


def judge_from_settings(settings: Settings) -> FoodJudge | None:
    """The configured judge, or None when LLM_JUDGE_PROVIDER=off."""
    if settings.llm_judge_provider == "off":
        return None
    key, model, rpm = _judge_settings(settings, settings.llm_judge_provider)
    return build_judge(
        settings.llm_judge_provider,
        key,
        model,
        rpm,
        timeout_seconds=settings.llm_timeout_seconds,
    )


def provider_from_settings(
    settings: Settings, provider: str
) -> tuple[LlmProvider, int]:
    """A raw provider and its rpm limit (the evaluation adds its own cache)."""
    key, model, rpm = _judge_settings(settings, provider)
    provider_obj = build_provider(
        provider, key, model, timeout_seconds=settings.llm_timeout_seconds
    )
    return provider_obj, rpm
