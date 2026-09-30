"""Build the curated nutrition table from USDA FoodData Central (SR Legacy).

Inputs:
    data/food_selection.csv  hand-curated names, aliases and portion specs
    USDA SR Legacy CSVs      downloaded into data/raw/ if missing (public domain)
Outputs:
    data/foods.csv           one row per food, macros per 100 g
    data/food_portions.csv   one row per (food, unit) with the source of the grams

Every gram value is traceable: a portion either references a USDA household
measure ("usda:<modifier>") or is an explicit assumption (a plain number).

Usage (from the project root):
    python -m scripts.build_food_table
"""

import csv
import re
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

from app.amounts import Unit
from app.text import fold

USDA_URL = (
    "https://fdc.nal.usda.gov/fdc-datasets/"
    "FoodData_Central_sr_legacy_food_csv_2018-04.zip"
)
RAW_DIR = Path("data/raw")
USDA_DIR = RAW_DIR / "sr_legacy" / "FoodData_Central_sr_legacy_food_csv_2018-04"
SELECTION_PATH = Path("data/food_selection.csv")
FOODS_PATH = Path("data/foods.csv")
PORTIONS_PATH = Path("data/food_portions.csv")

NUTRIENTS = {"1008": "kcal", "1003": "protein", "1005": "carbs", "1004": "fat"}
SPEC_UNITS = {
    unit.value: unit
    for unit in Unit
    if unit not in (Unit.GRAM, Unit.MILLILITER, Unit.PORTION)
}
US_CUP_ML = 236.588
FL_OZ_ML = 29.5735
_FL_OZ = re.compile(r"(\d+(?:\.\d+)?)?\s*fl oz")

Portions = list[tuple[str, float]]  # (USDA modifier, grams per one unit)


def ensure_usda() -> None:
    """Download and extract the SR Legacy CSVs if they are not present."""
    if (USDA_DIR / "food.csv").exists():
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "sr_legacy.zip"
    print(f"Downloading {USDA_URL}")
    urllib.request.urlretrieve(USDA_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(RAW_DIR / "sr_legacy")


def load_usda(
    fdc_ids: set[str],
) -> tuple[dict[str, str], dict[str, dict[str, float]], dict[str, Portions]]:
    """Load descriptions, macros per 100 g and household portions."""
    with open(USDA_DIR / "food.csv", encoding="utf-8") as f:
        descriptions = {
            r["fdc_id"]: r["description"]
            for r in csv.DictReader(f)
            if r["fdc_id"] in fdc_ids
        }
    nutrients: dict[str, dict[str, float]] = defaultdict(dict)
    with open(USDA_DIR / "food_nutrient.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["fdc_id"] in fdc_ids and r["nutrient_id"] in NUTRIENTS:
                nutrients[r["fdc_id"]][NUTRIENTS[r["nutrient_id"]]] = float(r["amount"])
    portions: dict[str, Portions] = defaultdict(list)
    with open(USDA_DIR / "food_portion.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["fdc_id"] in fdc_ids:
                grams = float(r["gram_weight"]) / float(r["amount"])
                portions[r["fdc_id"]].append((r["modifier"].strip(), grams))
    return descriptions, nutrients, portions


def find_portion(prefix: str, portions: Portions) -> tuple[str, float] | None:
    """Find a USDA portion by modifier: exact match first, then prefix."""
    for modifier, grams in portions:
        if modifier.lower() == prefix.lower():
            return modifier, grams
    for modifier, grams in portions:
        if modifier.lower().startswith(prefix.lower()):
            return modifier, grams
    return None


def volume_ml(modifier: str) -> float | None:
    """Volume of a USDA household measure, if it is a volume measure."""
    match = _FL_OZ.search(modifier.lower())
    if match:
        return float(match.group(1) or 1) * FL_OZ_ML
    if "cup" in modifier.lower():
        return US_CUP_ML
    return None


def parse_spec(spec: str) -> list[tuple[str, str]]:
    """Split "piece=usda:large;bowl=200" into [("piece", "usda:large"), ...]."""
    items = []
    for part in filter(None, (p.strip() for p in spec.split(";"))):
        key, _, value = part.partition("=")
        items.append((key.strip(), value.strip()))
    return items


def build() -> list[str]:
    """Build both output CSVs. Returns a list of errors (empty on success)."""
    with open(SELECTION_PATH, encoding="utf-8") as f:
        selection = list(csv.DictReader(f))
    ensure_usda()
    descriptions, nutrients, usda_portions = load_usda(
        {row["fdc_id"] for row in selection}
    )

    errors: list[str] = []
    foods: list[dict[str, object]] = []
    portion_rows: list[dict[str, object]] = []
    seen_names: dict[str, str] = {}

    for row in selection:
        food_id, fdc_id = row["id"], row["fdc_id"]
        if fdc_id not in descriptions:
            errors.append(f"{food_id}: fdc_id {fdc_id} not found in USDA data")
            continue
        macros = nutrients.get(fdc_id, {})
        if set(macros) != set(NUTRIENTS.values()):
            errors.append(
                f"{food_id}: missing macros {set(NUTRIENTS.values()) - set(macros)}"
            )
            continue

        aliases = [a.strip() for a in row["aliases"].split("|") if a.strip()]
        for name in [row["name_en"], row["name_tr"], *aliases]:
            owner = seen_names.setdefault(fold(name), food_id)
            if owner != food_id:
                errors.append(f"{food_id}: name '{name}' already used by {owner}")

        density: float | None = None
        for key, value in parse_spec(row["portions"]):
            if value.startswith("usda:"):
                found = find_portion(value.removeprefix("usda:"), usda_portions[fdc_id])
                if found is None:
                    errors.append(f"{food_id}: no USDA portion matching '{value}'")
                    continue
                modifier, grams = found
                source = f"usda: {modifier}"
            else:
                modifier, grams, source = "", float(value), "assumption"

            if key == "density":
                ml = volume_ml(modifier) if modifier else None
                if ml is None:
                    errors.append(
                        f"{food_id}: density needs a volume measure, got '{value}'"
                    )
                    continue
                density = round(grams / ml, 3)
            elif key in SPEC_UNITS:
                portion_rows.append(
                    {
                        "food_id": food_id,
                        "unit": key,
                        "grams": round(grams, 2),
                        "source": source,
                    }
                )
            else:
                errors.append(f"{food_id}: unknown portion unit '{key}'")

        foods.append(
            {
                "id": food_id,
                "fdc_id": fdc_id,
                "usda_description": descriptions[fdc_id],
                "name_en": row["name_en"],
                "name_tr": row["name_tr"],
                "aliases": "|".join(aliases),
                "kcal_100g": macros["kcal"],
                "protein_100g": macros["protein"],
                "carbs_100g": macros["carbs"],
                "fat_100g": macros["fat"],
                "default_grams": float(row["default_grams"]),
                "density_g_per_ml": density,
                "note": row["note"],
            }
        )

    if not errors:
        _write_csv(FOODS_PATH, foods)
        _write_csv(PORTIONS_PATH, portion_rows)
    return errors


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    problems = build()
    if problems:
        print("Build failed:", *problems, sep="\n  - ")
        sys.exit(1)
    print(f"Wrote {FOODS_PATH} and {PORTIONS_PATH}")
