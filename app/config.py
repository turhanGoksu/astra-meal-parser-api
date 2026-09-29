"""Application settings loaded from environment variables (and a local .env)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration. Each field maps to an upper-case env variable."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_repo_id: str = "Turhan123/astra-meal-parser-gguf"
    model_filename: str = "astra-meal-parser-1.5b-q4_k_m.gguf"
    model_dir: Path = Path("models")
    parser_use_grammar: bool = False
    embedding_model_name: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    @property
    def model_path(self) -> Path:
        """Full path to the GGUF file inside the model directory."""
        return self.model_dir / self.model_filename


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (env is read only once)."""
    return Settings()
