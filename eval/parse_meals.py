"""Run the real parser over the eval meal files to collect eval item names.

Using parser output (not hand-written names) keeps the eval items in the same
form the matcher sees in production ("Yarım Simit", "lentil soup", ...).
The output is committed so the eval set stays fixed even if llama.cpp changes.

Two meal sets are kept apart so results can be reported per set:
- natural:  everyday meals (data/eval/meals.txt)
- variants: deliberately varied names, near misses and typos
            (data/eval/meals_variants.txt)

Usage (from the project root):
    python -m eval.parse_meals
"""

import csv
from pathlib import Path

from app.config import get_settings
from astra_nutrition.parser import MealParser
from astra_nutrition.text import fold

MEAL_SETS = {
    "natural": Path("data/eval/meals.txt"),
    "variants": Path("data/eval/meals_variants.txt"),
}
PARSED_PATH = Path("data/eval/parsed_items.csv")


def main() -> None:
    settings = get_settings()
    parser = MealParser.from_path(
        settings.model_path,
        use_grammar=settings.parser_use_grammar,
        resplit_merged=settings.parser_resplit_merged,
    )

    rows = []
    for set_name, path in MEAL_SETS.items():
        meals = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for meal_id, meal in enumerate(meals, start=1):
            result = parser.parse(meal)
            if not result.items:
                print(f"[{result.status}] {set_name} {meal_id}: {meal!r}")
            split_names = {name for r in result.resplits for name in r.names}
            for item in result.items:
                rows.append(
                    {
                        "set": set_name,
                        "meal_id": meal_id,
                        "meal_text": meal,
                        "status": result.status,
                        "name": item.name,
                        "amount": item.amount,
                        "from_resplit": item.name in split_names,
                    }
                )
            for resplit in result.resplits:
                print(f"resplit: {resplit.original.name!r} -> {resplit.names}")

    with open(PARSED_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for set_name in MEAL_SETS:
        names = {fold(r["name"]) for r in rows if r["set"] == set_name}
        items = sum(r["set"] == set_name for r in rows)
        print(f"{set_name}: {items} items ({len(names)} unique names)")


if __name__ == "__main__":
    main()
