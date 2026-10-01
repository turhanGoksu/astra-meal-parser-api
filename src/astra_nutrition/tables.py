"""Access to the food table files shipped inside the package."""

import csv
import io
from importlib.resources import files


def read_table(filename: str) -> list[dict[str, str]]:
    """Rows of a bundled CSV ("foods.csv" or "food_portions.csv")."""
    text = files("astra_nutrition").joinpath("data", filename).read_text("utf-8")
    return list(csv.DictReader(io.StringIO(text)))
