"""Access to the food table files shipped inside the package."""

import csv
import io
from importlib.resources import files

from astra_nutrition.text import embedding_text, fold


def read_table(filename: str) -> list[dict[str, str]]:
    """Rows of a bundled CSV ("foods.csv" or "food_portions.csv")."""
    text = files("astra_nutrition").joinpath("data", filename).read_text("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def alias_rows(
    foods: list[dict[str, str]],
) -> list[tuple[str, str, str, str]]:
    """One (food_id, alias, alias_folded, kind) row per distinct name of a food.

    Names with the same embedding input ("Kefir" as both EN and TR name) would
    produce identical vectors, so only the first is kept. "Yoğurt" and "yogurt"
    stay separate: they fold to the same key but embed differently.
    """
    rows = []
    for food in foods:
        names = [(food["name_en"], "name_en"), (food["name_tr"], "name_tr")]
        names += [(a, "alias") for a in food["aliases"].split("|") if a]
        seen: set[str] = set()
        for name, kind in names:
            if embedding_text(name) not in seen:
                seen.add(embedding_text(name))
                rows.append((food["id"], name, fold(name), kind))
    return rows
