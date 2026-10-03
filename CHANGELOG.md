# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). While the version is 0.x, minor
releases may change the API.

A released version is never changed: fixes ship as a new version.

## [Unreleased]

### Added

- Browser demo with Gradio (`demo/`), run locally with one Docker command
  and also ready as a Hugging Face Docker Space. It installs the hash-checked
  v0.1.0 wheel and runs offline, without the LLM judge.

### Changed

- CI runs on a pinned `ubuntu-24.04` instead of the moving `ubuntu-latest`.

## [0.1.0] - 2026-10-02

First public release (alpha).

### Added

- **Library** (`astra_nutrition`): `Analyzer` turns Turkish or English meal
  text into foods, grams and nutrition, offline after the first model download.
  - Parsing with the fine-tuned 1.5B meal parser
    ([Turhan123/astra-meal-parser-gguf](https://huggingface.co/Turhan123/astra-meal-parser-gguf),
    Q4_K_M) on CPU via llama.cpp. Output is validated and every parse has an
    explicit status; merged items with a conjunction are parsed again.
  - Deterministic amount normalization: Turkish and English units, number
    words and fractions. Weights the model adds in parentheses are used only
    if the user wrote them; an amount that repeats the item's name
    (`1 muz`) reads as a count.
  - Matching: exact, then fuzzy (trigram similarity identical to PostgreSQL
    pg_trgm). Unmatched items are explicit and keep their closest candidate.
  - Honest totals: item statuses `ok`, `estimated`, `amount_unknown`,
    `unmatched`, and `complete` / `includes_estimates` flags on the totals.
  - Bundled food table: 128 foods, 455 Turkish and English aliases and 190
    portions from USDA SR Legacy (CC0), each food traceable to its FDC id.
  - Your own foods from a strictly validated CSV (`--foods`,
    `FoodTable.with_user_foods`).
  - Optional LLM judge (`[judge]` extra): embedding retrieval, then a Groq,
    Gemini or any OpenAI-compatible model picks one of the offered foods or
    none. Only item names are sent.
  - Optional PostgreSQL + pgvector index (`[postgres]` extra) with the same
    results as the in-memory index.
- **CLI** `astra-nutrition` with table and JSON output.
- **Web service** (FastAPI): `/analyze`, `/foods/search`, `/stats/unmatched`,
  `/health`, and a per-item request log in PostgreSQL with best-effort writes.
- **Docker**: a slim image with the model in a volume; the judge is an opt-in
  build.
- **Evaluation** harness with a leakage-safe dev/test split; thresholds were
  tuned on dev only.
- **CI**: lint, unit tests, PostgreSQL parity tests and a clean wheel install.

### Known limitations

- Many Turkish dishes are not in the table yet; they are reported as
  `unmatched`.
- Parser output can differ between llama.cpp builds and machines, even at
  temperature 0.
- Fuzzy matching can match a modified dish to its base food
  (`Etli Kuru Fasulye`).

[Unreleased]: https://github.com/turhanGoksu/astra-nutrition/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/turhanGoksu/astra-nutrition/releases/tag/v0.1.0
