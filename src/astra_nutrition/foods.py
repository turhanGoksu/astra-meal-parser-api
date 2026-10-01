"""The food table: per-100 g nutrition, default portions and household units."""

from collections import defaultdict
from dataclasses import dataclass

from astra_nutrition.amounts import FoodPortions, Unit
from astra_nutrition.tables import read_table


@dataclass(frozen=True)
class Food:
    """One food with nutrition per 100 g (kcal straight from the source data)."""

    id: str
    name_en: str
    name_tr: str
    kcal_100g: float
    protein_100g: float
    carbs_100g: float
    fat_100g: float
    default_grams: float
    density_g_per_ml: float | None
    note: str = ""


class FoodTable:
    """Foods and their portions, built from rows in the foods.csv format."""

    def __init__(
        self, food_rows: list[dict[str, str]], portion_rows: list[dict[str, str]]
    ) -> None:
        self.rows = food_rows  # kept for building indexes over the same foods
        self._foods = {row["id"]: _to_food(row) for row in food_rows}
        units: dict[str, dict[Unit, float]] = defaultdict(dict)
        for row in portion_rows:
            units[row["food_id"]][Unit(row["unit"])] = float(row["grams"])
        self._units = dict(units)

    @classmethod
    def bundled(cls) -> "FoodTable":
        """The table shipped with the package."""
        return cls(read_table("foods.csv"), read_table("food_portions.csv"))

    def __len__(self) -> int:
        return len(self._foods)

    def __contains__(self, food_id: object) -> bool:
        return food_id in self._foods

    def get(self, food_id: str) -> Food:
        return self._foods[food_id]

    def portions(self, food_id: str) -> FoodPortions:
        food = self._foods[food_id]
        return FoodPortions(
            default_grams=food.default_grams,
            unit_grams=self._units.get(food_id, {}),
            density_g_per_ml=food.density_g_per_ml,
        )


def _to_food(row: dict[str, str]) -> Food:
    density = row.get("density_g_per_ml") or ""
    return Food(
        id=row["id"],
        name_en=row["name_en"],
        name_tr=row["name_tr"],
        kcal_100g=float(row["kcal_100g"]),
        protein_100g=float(row["protein_100g"]),
        carbs_100g=float(row["carbs_100g"]),
        fat_100g=float(row["fat_100g"]),
        default_grams=float(row["default_grams"]),
        density_g_per_ml=float(density) if density else None,
        note=row.get("note", ""),
    )
