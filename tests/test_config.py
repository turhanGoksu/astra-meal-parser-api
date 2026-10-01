"""Unit tests for settings (never reads the real .env)."""

import pytest

from app.config import Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("DATABASE_URL", "POSTGRES_PASSWORD", "POSTGRES_HOST"):
        monkeypatch.delenv(name, raising=False)


def test_database_url_is_built_from_postgres_settings() -> None:
    settings = Settings(_env_file=None, postgres_password="secret123")
    assert settings.database_url == (
        "postgresql://astra:secret123@localhost:5432/astra_meals"
    )


def test_password_special_characters_are_url_encoded() -> None:
    settings = Settings(_env_file=None, postgres_password="p@ss:w/rd#1")
    assert "p%40ss%3Aw%2Frd%231@" in (settings.database_url or "")


def test_explicit_database_url_wins() -> None:
    url = "postgresql://other:pw@remote:5433/x"
    settings = Settings(_env_file=None, postgres_password="ignored", database_url=url)
    assert settings.database_url == url


def test_no_password_means_no_url() -> None:
    assert Settings(_env_file=None).database_url is None


def test_empty_values_in_env_mean_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_RPM", "")
    monkeypatch.setenv("GROQ_MODEL", "")
    settings = Settings(_env_file=None)
    assert settings.groq_rpm is None
    assert settings.groq_model is None
