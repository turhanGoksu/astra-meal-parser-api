"""Application settings loaded from environment variables (and a local .env)."""

from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration. Each field maps to an upper-case env variable."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

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

    embedding_model_name: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # PROVISIONAL placeholders: replaced by values tuned on the dev set (Step 7).
    match_strategy: str = "hybrid"
    match_fuzzy_threshold: float = 0.5
    match_embedding_threshold: float = 0.9

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
