"""astra-nutrition: Turkish/English meal text -> foods, grams and nutrition.

Offline by default: a quantized GGUF parser on CPU, deterministic amount
normalization, and matching against a traceable USDA-based food table.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("astra-nutrition")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0"
