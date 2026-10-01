"""User food CSVs: strict validation and loud merge conflicts."""

from pathlib import Path

import pytest

from astra_nutrition import Analyzer, FoodTable, ItemStatus
from astra_nutrition.foods import FoodTableError, load_user_foods

HEADER = (
    "id,name_en,name_tr,aliases,kcal_100g,protein_100g,carbs_100g,fat_100g,"
    "default_grams,source,grams_per_bowl\n"
)
SOUP = "test_soup,Test soup,Test çorbası,deneme çorbası,50,3,7,1,250,unit test,300\n"


def write(tmp_path: Path, body: str, header: str = HEADER) -> Path:
    path = tmp_path / "foods.csv"
    path.write_text(header + body, encoding="utf-8")
    return path


def test_user_food_is_matched_and_its_portions_used(tmp_path: Path) -> None:
    table = FoodTable.bundled().with_user_foods(write(tmp_path, SOUP))
    item = Analyzer(table=table).analyze_items([("Deneme çorbası", "1 kase")]).items[0]
    assert (item.food_id, item.status, item.grams) == (
        "test_soup",
        ItemStatus.OK,
        300.0,
    )
    assert item.food_source == "unit test"
    assert len(table) == len(FoodTable.bundled()) + 1  # bundled foods kept


def test_bundled_foods_report_their_usda_source() -> None:
    assert FoodTable.bundled().get("banana").source.startswith("USDA SR Legacy")


@pytest.mark.parametrize(
    ("header", "body", "problem"),
    [
        (HEADER.replace("kcal_100g", "kcal_100"), SOUP, "unknown column 'kcal_100'"),
        (
            HEADER.replace(",source", ""),
            SOUP.replace(",unit test", ""),
            "missing column",
        ),
        (HEADER, SOUP.replace(",50,", ",-5,"), "kcal_100g must be >= 0"),
        (HEADER, SOUP.replace(",250,", ",abc,"), "default_grams must be a number"),
        (HEADER, SOUP.replace("unit test", ""), "source is empty"),
        (HEADER, SOUP.replace("test_soup", "Test Soup"), "id must use a-z"),
        (HEADER, SOUP + SOUP, "duplicate id"),
    ],
)
def test_invalid_files_are_rejected_with_the_reason(
    tmp_path: Path, header: str, body: str, problem: str
) -> None:
    with pytest.raises(FoodTableError, match=problem):
        load_user_foods(write(tmp_path, body, header))


def test_alias_owned_by_another_food_is_a_loud_error(tmp_path: Path) -> None:
    kasar = "ev_kasari,Homemade kashkaval,Ev kaşarı,peynir,360,25,2,28,30,own,\n"
    with pytest.raises(
        FoodTableError, match="'peynir' of 'ev_kasari' is already used by 'feta'"
    ):
        FoodTable.bundled().with_user_foods(write(tmp_path, kasar))


def test_existing_id_needs_an_explicit_replace(tmp_path: Path) -> None:
    my_banana = "banana,Banana,Muz,,95,1,23,0.3,120,my lab,\n"
    with pytest.raises(FoodTableError, match="add it to replace_ids"):
        FoodTable.bundled().with_user_foods(write(tmp_path, my_banana))

    table = FoodTable.bundled().with_user_foods(
        write(tmp_path, my_banana), replace_ids=["banana"]
    )
    assert table.get("banana").kcal_100g == 95.0
    assert table.portions("banana").unit_grams == {}  # old USDA portions dropped


def test_replacing_an_unknown_id_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FoodTableError, match="'nope' is not in the table"):
        FoodTable.bundled().with_user_foods(write(tmp_path, SOUP), replace_ids=["nope"])
