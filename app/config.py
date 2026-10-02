"""Application settings loaded from environment variables (and a local .env)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration. Each field maps to an upper-case env variable."""

    # env_ignore_empty: "GROQ_RPM=" in .env means "not set", not an invalid int.
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_ignore_empty=True
    )

    model_repo_id: str = "Turhan123/astra-meal-parser-gguf"
    model_filename: str = "astra-meal-parser-1.5b-q4_k_m.gguf"
    model_dir: Path = Path("models")
    parser_use_grammar: bool = False
    parser_resplit_merged: bool = True  # Design R: re-parse "X with Y" items

    # Same POSTGRES_* variables docker compose uses for the db service.
    postgres_user: str = "astra"
    postgres_password: str | None = None
    postgres_db: str = "astra_meals"
    postgres_host: str = "localhost"  # "db" inside docker compose
    postgres_port: int = 5432
    database_url: str | None = None  # optional override; built if not set

    # Model 2 (e5-small) replaced Model 1 (MiniLM) after the dev evaluation:
    # recall@5 of candidates 16/16 vs 9/16 on items exact+fuzzy missed.
    # e5 expects a "query: " prefix on both sides for symmetric matching.
    embedding_model_name: str = "intfloat/multilingual-e5-small"
    embedding_prefix: str = "query: "

    # Matching thresholds are not configured here: the service uses the values
    # chosen on the dev set, kept in astra_nutrition.analyzer.DEFAULT_MATCH_CONFIG.
    # Where the matcher looks foods up: in memory (bundled table, same results)
    # or in PostgreSQL (the ingested table).
    food_index_backend: Literal["memory", "postgres"] = "memory"
    parser_threads: int | None = None  # llama.cpp CPU threads (None: its default)

    # Step 8: optional LLM judge, off by default. As in Project 4, model names
    # and free-tier rate limits live in .env: a deprecated model is fixed by
    # editing .env, not code.
    llm_judge_provider: Literal["off", "groq", "gemini"] = "off"
    groq_api_key: SecretStr | None = None
    groq_model: str | None = None
    groq_rpm: int | None = Field(default=None, gt=0)
    gemini_api_key: SecretStr | None = None
    gemini_model: str | None = None
    gemini_rpm: int | None = Field(default=None, gt=0)
    llm_timeout_seconds: float = 30.0

    @model_validator(mode="after")
    def build_database_url(self) -> "Settings":
        """Derive DATABASE_URL from POSTGRES_* so the password lives in one place.

        Keeping the password in two variables let them drift apart and caused
        "password authentication failed" even though both looked correct.
        """
        if not self.database_url and self.postgres_password:
            password = quote(self.postgres_password, safe="")
            self.database_url = (
                f"postgresql://{self.postgres_user}:{password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )
        return self

    @property
    def model_path(self) -> Path:
        """Full path to the GGUF file inside the model directory."""
        return self.model_dir / self.model_filename


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (env is read only once)."""
    return Settings()
