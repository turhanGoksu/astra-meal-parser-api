"""The GGUF parser model: found in the Hugging Face cache or downloaded once."""

from pathlib import Path

DEFAULT_REPO_ID = "Turhan123/astra-meal-parser-gguf"
DEFAULT_FILENAME = "astra-meal-parser-1.5b-q4_k_m.gguf"  # ~1 GB, Apache-2.0


def default_model_path(
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_FILENAME,
    cache_dir: str | Path | None = None,
) -> Path:
    """Local path of the GGUF file, downloading it on first use.

    Uses the standard Hugging Face cache (HF_HOME), so the file is shared with
    other tools and downloaded only once. If it is already cached, no network
    request is made.
    """
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    cached = try_to_load_from_cache(repo_id, filename, cache_dir=cache_dir)
    if isinstance(cached, str):
        return Path(cached)
    return Path(hf_hub_download(repo_id, filename, cache_dir=cache_dir))
