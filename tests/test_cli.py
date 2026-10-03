"""Command-line tests with a fake parser (no model download)."""

import json
from pathlib import Path

import pytest

import astra_nutrition.cli as cli
from astra_nutrition import Analyzer
from astra_nutrition.parser import MealParser
from tests.test_parser import FakeLlm

OUTPUT = (
    '{"items": [{"name": "Yumurta", "amount": "2"}, '
    '{"name": "Kokoreç", "amount": "1 dilim"}]}'
)


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def analyzer(**kwargs) -> Analyzer:
        kwargs.pop("model_path")
        return Analyzer(parser=MealParser(FakeLlm(OUTPUT)), **kwargs)

    monkeypatch.setattr(cli, "Analyzer", analyzer)


def test_text_output_shows_items_totals_and_what_was_not_counted(
    fake_model: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["2 yumurta ve 1 dilim kokoreç"]) == 0
    out = capsys.readouterr().out
    assert "Yumurta" in out and "unmatched" in out
    assert "Total: 143 kcal" in out
    assert "Not counted: 1 unmatched" in out


def test_text_output_names_rejected_items(
    fake_model: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["2 yumurta"]) == 0  # Kokoreç is not in the text
    out = capsys.readouterr().out
    assert "1 rejected by the parser checks" in out
    assert "Rejected: 'Kokoreç' (not in the meal text: kokorec)" in out


def test_json_output_is_the_analysis_result(
    fake_model: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["2 yumurta ve 1 dilim kokoreç", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["totals"]["kcal"] == 143.0
    assert data["items"][1]["status"] == "unmatched"


def test_bad_food_file_exits_with_2_and_explains(
    fake_model: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "foods.csv"
    bad.write_text("id,name\nx,y\n", encoding="utf-8")
    assert cli.main(["1 muz", "--foods", str(bad)]) == 2
    assert "missing column 'name_en'" in capsys.readouterr().err


def test_judge_without_settings_exits_with_2_and_names_them(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("JUDGE_BASE_URL", "JUDGE_MODEL", "JUDGE_API_KEY", "JUDGE_RPM"):
        monkeypatch.delenv(name, raising=False)
    assert cli.main(["1 muz", "--judge", "openai-compatible"]) == 2
    assert "JUDGE_MODEL, JUDGE_BASE_URL" in capsys.readouterr().err
