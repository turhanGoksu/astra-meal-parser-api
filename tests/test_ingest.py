"""Unit tests for the ingest helpers (no database needed)."""

from astra_nutrition.tables import alias_rows
from scripts.ingest_foods import typed_food

FOOD = {
    "id": "chicken_breast",
    "fdc_id": "171477",
    "usda_description": "Chicken, broilers or fryers, breast, meat only, cooked",
    "name_en": "Chicken breast",
    "name_tr": "Tavuk göğsü",
    "aliases": "tavuk fileto|chicken fillet",
    "kcal_100g": "165.0",
    "protein_100g": "31.02",
    "carbs_100g": "0.0",
    "fat_100g": "3.57",
    "default_grams": "120.0",
    "density_g_per_ml": "",
    "note": "",
}


def test_alias_rows_include_both_names_and_all_aliases() -> None:
    assert alias_rows([FOOD]) == [
        ("chicken_breast", "Chicken breast", "chicken breast", "name_en"),
        ("chicken_breast", "Tavuk göğsü", "tavuk gogsu", "name_tr"),
        ("chicken_breast", "tavuk fileto", "tavuk fileto", "alias"),
        ("chicken_breast", "chicken fillet", "chicken fillet", "alias"),
    ]


def test_alias_rows_skip_empty_aliases() -> None:
    rows = alias_rows([{**FOOD, "aliases": ""}])
    assert [kind for *_, kind in rows] == ["name_en", "name_tr"]


def test_typed_food_converts_numbers_and_empty_density() -> None:
    typed = typed_food(FOOD)
    assert typed["fdc_id"] == 171477
    assert typed["kcal_100g"] == 165.0
    assert typed["density_g_per_ml"] is None
    assert (
        typed_food({**FOOD, "density_g_per_ml": "1.031"})["density_g_per_ml"] == 1.031
    )


def test_alias_rows_drop_names_with_identical_embedding_input() -> None:
    food = {**FOOD, "name_en": "Kefir", "name_tr": "Kefir", "aliases": "kefir"}
    assert [alias for _, alias, _, _ in alias_rows([food])] == ["Kefir"]


def test_alias_rows_keep_names_that_only_fold_to_the_same_key() -> None:
    food = {**FOOD, "name_en": "Plain yogurt", "name_tr": "Yoğurt", "aliases": "yogurt"}
    rows = alias_rows([food])
    assert [alias for _, alias, _, _ in rows] == ["Plain yogurt", "Yoğurt", "yogurt"]
    assert rows[1][2] == rows[2][2] == "yogurt"
