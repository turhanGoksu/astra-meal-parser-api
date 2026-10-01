"""astra-nutrition: Turkish/English meal text -> foods, grams and nutrition.

Offline by default: a quantized GGUF parser on CPU, deterministic amount
normalization, and matching against a traceable USDA-based food table.
"""

from importlib.metadata import PackageNotFoundError, version

from astra_nutrition.analyzer import (
    AnalysisResult,
    Analyzer,
    ItemResult,
    ItemStatus,
    Nutrition,
    Totals,
)
from astra_nutrition.foods import Food, FoodTable

__all__ = [
    "AnalysisResult",
    "Analyzer",
    "Food",
    "FoodTable",
    "ItemResult",
    "ItemStatus",
    "Nutrition",
    "Totals",
]

try:
    __version__ = version("astra-nutrition")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0"
