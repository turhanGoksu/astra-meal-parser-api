"""Download the GGUF parser model from Hugging Face into MODEL_DIR.

Usage (from the project root):
    python -m scripts.download_model

In Docker, MODEL_DIR points to a mounted volume, so the weights live outside
the image and survive rebuilds.
"""

from pathlib import Path

from huggingface_hub import hf_hub_download

from app.config import get_settings


def download_model() -> Path:
    """Download the configured GGUF file unless it is already there.

    An existing file is used as is, without a network call, so containers
    restart offline once the model is in the volume.
    """
    settings = get_settings()
    if settings.model_path.exists():
        return settings.model_path
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=settings.model_repo_id,
        filename=settings.model_filename,
        local_dir=settings.model_dir,
    )
    return Path(path)


if __name__ == "__main__":
    model_path = download_model()
    size_gb = model_path.stat().st_size / 1e9
    print(f"Model ready: {model_path} ({size_gb:.2f} GB)")
