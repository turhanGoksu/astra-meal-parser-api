"""Command line: astra-nutrition "2 yumurta ve 1 muz" [--json] [--foods FILE].

Exit codes: 0 analyzed, 1 the parser could not read the meal, 2 bad input
(for example a user food file with problems or missing judge settings).

The LLM judge is off unless --judge is given. Its settings come from the
environment: GROQ_API_KEY / GROQ_MODEL / GROQ_RPM, GEMINI_API_KEY / ...,
or JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_API_KEY / JUDGE_RPM for any
OpenAI-compatible server (for example a local Ollama).
"""

import argparse
import os
import sys
from pathlib import Path

from astra_nutrition import __version__
from astra_nutrition.analyzer import AnalysisResult, Analyzer, ItemStatus
from astra_nutrition.foods import FoodTable, FoodTableError
from astra_nutrition.parser import ParseStatus

STATUS_MARK = {
    ItemStatus.OK: "ok",
    ItemStatus.ESTIMATED: "estimated*",
    ItemStatus.AMOUNT_UNKNOWN: "amount?",
    ItemStatus.UNMATCHED: "unmatched",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="astra-nutrition",
        description="Turkish/English meal text to foods, grams and nutrition.",
    )
    parser.add_argument("meal", help='meal description, e.g. "2 yumurta ve 1 muz"')
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument(
        "--foods", type=Path, help="CSV with your own foods (added to the table)"
    )
    parser.add_argument(
        "--replace",
        action="append",
        default=[],
        metavar="FOOD_ID",
        help="existing food your CSV deliberately replaces (repeatable)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="local GGUF file (default: downloaded from Hugging Face on first use)",
    )
    parser.add_argument(
        "--judge",
        choices=["groq", "gemini", "openai-compatible"],
        help="send unresolved item names to an LLM judge (needs the [judge] extra)",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def judge_settings(provider: str) -> dict[str, object]:
    """Judge settings from environment variables; raises ValueError if missing."""
    prefix = "JUDGE" if provider == "openai-compatible" else provider.upper()
    env = {name: os.environ.get(f"{prefix}_{name}") for name in ("API_KEY", "MODEL")}
    required = ["MODEL"] + (["API_KEY"] if prefix != "JUDGE" else [])
    if prefix == "JUDGE":
        env["BASE_URL"] = os.environ.get("JUDGE_BASE_URL")
        required.append("BASE_URL")
    missing = [f"{prefix}_{name}" for name in required if not env.get(name)]
    if missing:
        raise ValueError(f"--judge {provider} needs {', '.join(missing)}")
    rpm = os.environ.get(f"{prefix}_RPM")
    return {
        "model": env["MODEL"],
        "api_key": env["API_KEY"],
        "base_url": env.get("BASE_URL"),
        "rpm": int(rpm) if rpm else None,
    }


def render(result: AnalysisResult) -> str:
    """A plain-text table of the items and the totals."""
    lines = []
    for item in result.items:
        food = item.food_name_en or "-"
        grams = f"{item.grams:g} g" if item.grams is not None else "-"
        kcal = f"{item.nutrition.kcal:g} kcal" if item.nutrition else "-"
        lines.append(
            f"{item.name[:24]:24} {item.amount[:14]:14} {STATUS_MARK[item.status]:11}"
            f" {food[:24]:24} {grams:>9} {kcal:>11}"
        )
    t = result.totals
    lines.append(
        f"\nTotal: {t.kcal:g} kcal | protein {t.protein_g:g} g | "
        f"carbs {t.carbs_g:g} g | fat {t.fat_g:g} g"
    )
    if t.includes_estimates:
        lines.append(f"* {t.estimated} item(s) use a default portion (estimate).")
    if not t.complete:
        rejected = f", {t.rejected} rejected by the parser checks" if t.rejected else ""
        lines.append(
            f"Not counted: {t.unmatched} unmatched, "
            f"{t.amount_unknown} with an unreadable amount{rejected}."
        )
    for item in result.rejected_items:
        name = item.raw.get("name") if isinstance(item.raw, dict) else None
        lines.append(f"Rejected: {name or item.raw!r} ({item.reason})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        table = FoodTable.bundled()
        if args.foods:
            table = table.with_user_foods(args.foods, replace_ids=args.replace)
    except (FoodTableError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.judge:
        try:
            settings = judge_settings(args.judge)
            analyzer = Analyzer.with_judge(
                args.judge, table=table, model_path=args.model, **settings
            )
        except (ValueError, ImportError) as exc:
            print(exc, file=sys.stderr)
            return 2
        print(
            f"LLM judge on: unresolved item names are sent to {args.judge} "
            f"({settings['model']}).",
            file=sys.stderr,
        )
    else:
        analyzer = Analyzer(table=table, model_path=args.model)
    result = analyzer.analyze(args.meal)
    print(result.model_dump_json(indent=2) if args.json else render(result))
    if result.parse_status in (ParseStatus.INVALID_OUTPUT, ParseStatus.ERROR):
        print(f"Parser problem: {result.parse_error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
