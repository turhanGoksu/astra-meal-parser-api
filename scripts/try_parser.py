"""Parse a few meals with the GGUF model and report size, latency and memory.

Usage (from the project root, after `python -m scripts.download_model`):
    python -m scripts.try_parser
"""

import json
import resource
import sys
import time

import llama_cpp
from llama_cpp import Llama

from app.config import get_settings
from astra_nutrition.prompts import SYSTEM_PROMPT

SAMPLE_MEALS = [
    "2 yumurta, 100g tavuk göğsü ve 1 muz",
    "For lunch I had a bowl of lentil soup, two slices of bread and a glass of ayran",
    "kahvaltıda 1 dilim beyaz peynir, yarım simit and a cup of black tea",
    "bir tabak mantı, 200 ml ayran ve 3 adet kuru kayısı",
]


def peak_rss_mb() -> float:
    """Peak resident memory of this process in MB.

    ru_maxrss is reported in bytes on macOS but in kilobytes on Linux.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    bytes_per_unit = 1 if sys.platform == "darwin" else 1024
    return peak * bytes_per_unit / 1e6


def print_size_report(llm: Llama, file_bytes: int) -> None:
    """Compare the quantized file with a rough fp16 estimate."""
    n_params = llama_cpp.llama_model_n_params(llm.model)
    tensor_bytes = llama_cpp.llama_model_size(llm.model)
    fp16_bytes = n_params * 2
    print(f"Parameters:           {n_params / 1e9:.3f} B")
    print(f"GGUF file size:       {file_bytes / 1e9:.3f} GB")
    print(f"  of which tensors:   {tensor_bytes / 1e9:.3f} GB")
    print(f"Bits per weight:      {tensor_bytes * 8 / n_params:.2f}")
    print(
        f"fp16 estimate:        {fp16_bytes / 1e9:.3f} GB "
        f"({fp16_bytes / file_bytes:.1f}x larger)"
    )


def main() -> None:
    settings = get_settings()
    model_path = settings.model_path
    if not model_path.exists():
        raise SystemExit(
            f"{model_path} not found. Run: python -m scripts.download_model"
        )

    rss_before_load = peak_rss_mb()
    start = time.perf_counter()
    llm = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        chat_format="chatml",
        n_gpu_layers=0,  # CPU only, same as inside Docker
        verbose=False,
    )
    load_s = time.perf_counter() - start

    print_size_report(llm, model_path.stat().st_size)
    print(f"Load time:            {load_s:.2f} s (threads: {llm.n_threads})\n")

    for meal in SAMPLE_MEALS:
        start = time.perf_counter()
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": meal},
            ],
            temperature=0,
            stop=["<|im_end|>"],
        )
        latency_s = time.perf_counter() - start
        text = response["choices"][0]["message"]["content"]
        usage = response["usage"]

        try:
            json.loads(text)
            valid = "valid JSON"
        except json.JSONDecodeError:
            valid = "INVALID JSON"

        print(f"Meal:   {meal}")
        print(f"Output: {text}")
        print(
            f"        {latency_s:.2f} s | prompt {usage['prompt_tokens']} tok, "
            f"output {usage['completion_tokens']} tok "
            f"({usage['completion_tokens'] / latency_s:.1f} tok/s) | {valid}\n"
        )

    print(f"Peak RSS before load: {rss_before_load:.0f} MB")
    print(f"Peak RSS after runs:  {peak_rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
