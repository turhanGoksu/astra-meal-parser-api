"""Tests for the food table builder helpers and the committed table files.

The data tests read the table files bundled in the package, so they run in CI
without downloading USDA data.
"""

import pytest

from astra_nutrition.amounts import Unit
from astra_nutrition.tables import read_table
from astra_nutrition.text import fold
from scripts.build_food_table import (
    find_portion,
    parse_spec,
    recipe_per_100g,
    resolve_grams,
    volume_ml,
)

PORTIONS = [
    ("slice", 29.0),
    ("slice, large", 30.0),
    ("slice", 25.0),
    ("cup (8 fl oz)", 237.0),
]


def test_find_portion_prefers_first_exact_match() -> None:
    assert find_portion("slice", PORTIONS) == ("slice", 29.0)


def test_find_portion_falls_back_to_prefix() -> None:
    assert find_portion("slice, l", PORTIONS) == ("slice, large", 30.0)
    assert find_portion("wedge", PORTIONS) is None


@pytest.mark.parametrize(
    ("modifier", "expected_ml"),
    [
        ("fl oz", 29.5735),
        ("cup (8 fl oz)", 236.588),
        ("can or bottle (12 fl oz)", 354.882),
        ("serving 1 cup", 236.588),
        ("large", None),
    ],
)
def test_volume_ml(modifier: str, expected_ml: float | None) -> None:
    result = volume_ml(modifier)
    if expected_ml is None:
        assert result is None
    else:
        assert result == pytest.approx(expected_ml, abs=0.01)


def test_parse_spec() -> None:
    assert parse_spec("piece=usda:large; bowl=200;") == [
        ("piece", "usda:large"),
        ("bowl", "200"),
    ]


FOODS = read_table("foods.csv")
FOOD_PORTIONS = read_table("food_portions.csv")


def test_food_ids_are_unique() -> None:
    ids = [food["id"] for food in FOODS]
    assert len(ids) == len(set(ids))


def test_names_and_aliases_are_unambiguous_after_folding() -> None:
    owners: dict[str, str] = {}
    for food in FOODS:
        names = [food["name_en"], food["name_tr"], *food["aliases"].split("|")]
        for name in filter(None, names):
            assert owners.setdefault(fold(name), food["id"]) == food["id"], name


def test_macros_and_defaults_are_valid() -> None:
    for food in FOODS:
        for column in ("kcal_100g", "protein_100g", "carbs_100g", "fat_100g"):
            assert float(food[column]) >= 0, (food["id"], column)
        assert float(food["default_grams"]) > 0, food["id"]


def test_portions_reference_known_foods_and_units() -> None:
    food_ids = {food["id"] for food in FOODS}
    for portion in FOOD_PORTIONS:
        assert portion["food_id"] in food_ids
        assert Unit(portion["unit"]) not in (Unit.GRAM, Unit.MILLILITER)
        assert float(portion["grams"]) > 0
        assert portion["source"] == "assumption" or portion["source"].startswith(
            "usda: "
        )


def test_resolve_grams_from_usda_or_an_explicit_assumption() -> None:
    assert resolve_grams("usda:cup", PORTIONS) == (
        "cup (8 fl oz)",
        237.0,
        "usda: cup (8 fl oz)",
    )
    assert resolve_grams("200", PORTIONS) == ("", 200.0, "assumption")
    assert resolve_grams("usda:1 piece", PORTIONS) is None


SOURCES = ("USDA SR Legacy,", "USDA FNDDS", "Recipe from USDA SR Legacy ingredients")


def test_every_food_names_its_source_dataset() -> None:
    for food in FOODS:
        assert food["source"].startswith(SOURCES), food["id"]


def test_recipe_macros_are_divided_by_the_cooked_weight() -> None:
    lentils = {"kcal": 358.0, "protein": 24.0, "carbs": 63.0, "fat": 2.0}
    water = {"kcal": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
    parts = [(100.0, lentils), (400.0, water)]  # 500 g raw: 358 kcal in the pot
    assert recipe_per_100g(parts, 500.0)["kcal"] == 71.6  # nothing evaporated
    assert recipe_per_100g(parts, 400.0)["kcal"] == 89.5  # 100 g of water gone


@pytest.mark.parametrize("cooked", [0.0, 501.0])
def test_recipe_cooked_weight_must_be_possible(cooked: float) -> None:
    water = {"kcal": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
    with pytest.raises(ValueError, match="raw total 500 g"):
        recipe_per_100g([(500.0, water)], cooked)


def test_recipe_dishes_name_their_cooked_weight_and_assumed_portions() -> None:
    recipes = {f["id"] for f in FOODS if f["source"].startswith("Recipe")}
    assert recipes
    for food in FOODS:
        if food["id"] in recipes:
            assert "cooked weight" in food["source"] and food["fdc_id"] == ""
    assert all(
        p["source"] == "assumption" for p in FOOD_PORTIONS if p["food_id"] in recipes
    )


def test_fndds_dishes_have_no_piece_or_slice_sizes() -> None:
    # FNDDS pieces are US sizes; "adet" or "dilim" must read as unknown (Design R).
    fndds = {food["id"] for food in FOODS if food["source"].startswith("USDA FNDDS")}
    units = {p["unit"] for p in FOOD_PORTIONS if p["food_id"] in fndds}
    assert fndds and units <= {Unit.CUP, Unit.TABLESPOON, Unit.TEASPOON}
