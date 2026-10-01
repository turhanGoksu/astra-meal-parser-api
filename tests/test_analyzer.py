"""Tests for the public Analyzer API (bundled table, fake parser, no model)."""

import pytest

import astra_nutrition.analyzer as analyzer_module
from astra_nutrition import Analyzer, FoodTable, ItemStatus
from astra_nutrition.parser import MealParser, ParseStatus
from tests.test_parser import FakeLlm

ANALYZER = Analyzer()


def test_statuses_grams_and_totals() -> None:
    result = ANALYZER.analyze_items(
        [
            ("Yumurta", "2"),
            ("Pilav", "biraz"),
            ("Baklava", "1 dilim"),
            ("Muz", "bir tutam"),
        ]
    )
    rows = [(i.name, i.status, i.grams) for i in result.items]
    assert rows == [
        ("Yumurta", ItemStatus.OK, 100.0),  # 2 x 50 g
        ("Pilav", ItemStatus.ESTIMATED, 150.0),  # vague -> default portion
        ("Baklava", ItemStatus.UNMATCHED, None),  # not in the table
        ("Muz", ItemStatus.AMOUNT_UNKNOWN, None),  # unreadable amount
    ]
    totals = result.totals
    assert totals.kcal == 338.0  # 143 + 195: unmatched and unknown not counted
    assert (totals.counted, totals.estimated, totals.unmatched) == (2, 1, 1)
    assert totals.amount_unknown == 1
    assert totals.includes_estimates is True
    assert totals.complete is False


def test_unmatched_item_keeps_its_best_candidate_but_no_nutrition() -> None:
    item = ANALYZER.analyze_items([("Baklava", "1 dilim")]).items[0]
    assert item.nutrition is None
    assert item.food_id is None
    assert item.best_candidate_food_id is not None  # visible, never counted


def test_complete_meal_without_estimates() -> None:
    totals = ANALYZER.analyze_items([("Tavuk Göğsü", "200 g")]).totals
    assert (totals.complete, totals.includes_estimates) == (True, False)
    assert totals.kcal == 330.0


def test_empty_meal() -> None:
    totals = ANALYZER.analyze_items([]).totals
    assert (totals.kcal, totals.items, totals.counted) == (0.0, 0, 0)


def test_model_added_weight_is_used_only_if_the_user_wrote_it() -> None:
    invented = ANALYZER.analyze_items(
        [("Beyaz Peynir", "1 dilim (30g)")], meal_text="1 dilim beyaz peynir"
    )
    written = ANALYZER.analyze_items(
        [("Beyaz Peynir", "1 dilim (40g)")], meal_text="1 dilim (40 g) beyaz peynir"
    )
    assert invented.items[0].grams == 30.0  # table slice, the note is ignored
    assert written.items[0].grams == 40.0  # grounded note
    assert written.items[0].amount_detail == "mass from grounded note"


def test_totals_are_rounded_once_after_summing() -> None:
    row = {
        "id": "tiny",
        "name_en": "Tiny food",
        "name_tr": "Minik",
        "aliases": "",
        "kcal_100g": "16",
        "protein_100g": "0",
        "carbs_100g": "0",
        "fat_100g": "0",
        "default_grams": "1",
        "density_g_per_ml": "",
        "note": "",
    }
    analyzer = Analyzer(table=FoodTable([row], []))
    result = analyzer.analyze_items([("Minik", "1 g")] * 3)
    assert [i.nutrition.kcal for i in result.items] == [0.2, 0.2, 0.2]  # 0.16 each
    assert result.totals.kcal == 0.5  # round(0.48), not 0.2 + 0.2 + 0.2 = 0.6


def test_analyze_uses_the_parser_and_reports_its_status() -> None:
    fake = FakeLlm(
        '{"items": [{"name": "Muz", "amount": "1"}, {"name": "", "amount": "2"}]}'
    )
    analyzer = Analyzer(parser=MealParser(fake))
    result = analyzer.analyze("1 muz")
    assert result.parse_status == ParseStatus.PARTIAL
    assert len(result.rejected_items) == 1  # the blank name, reported
    assert result.items[0].food_id == "banana"
    assert result.items[0].grams == 118.0  # 1 medium banana


def test_parser_is_loaded_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_download() -> None:
        raise AssertionError("model requested")

    monkeypatch.setattr(analyzer_module, "default_model_path", no_download)
    analyzer = Analyzer()  # must not download anything
    analyzer.analyze_items([("Muz", "1")])  # no parser needed either
    with pytest.raises(AssertionError, match="model requested"):
        _ = analyzer.parser
