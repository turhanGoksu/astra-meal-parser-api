-- Service request log: one row per analysis, one row per item (per-row design).
-- Safe to run repeatedly. food_id has no foreign key on purpose: with the
-- in-memory backend (or user foods) the matched food may not be in the DB.

CREATE TABLE IF NOT EXISTS analyses (
    id                 bigserial PRIMARY KEY,
    created_at         timestamptz NOT NULL DEFAULT now(),
    meal_text          text,         -- NULL when LOG_MEAL_TEXT=false (privacy)
    parse_status       text,
    parse_error        text,
    latency_ms         real NOT NULL,
    items              integer NOT NULL,
    counted            integer NOT NULL,
    estimated          integer NOT NULL,
    amount_unknown     integer NOT NULL,
    unmatched          integer NOT NULL,
    complete           boolean NOT NULL,
    includes_estimates boolean NOT NULL,
    kcal               real NOT NULL,
    protein_g          real NOT NULL,
    carbs_g            real NOT NULL,
    fat_g              real NOT NULL,
    app_version        text NOT NULL,
    judge              text          -- "provider:model", NULL when the judge is off
);

CREATE TABLE IF NOT EXISTS analysis_items (
    id                     bigserial PRIMARY KEY,
    analysis_id            bigint NOT NULL REFERENCES analyses (id) ON DELETE CASCADE,
    position               smallint NOT NULL,
    name                   text NOT NULL,
    amount                 text NOT NULL,
    status                 text NOT NULL,
    food_id                text,
    match_method           text,
    match_similarity       real,
    best_candidate_food_id text,     -- what an unmatched item was closest to
    grams                  real,
    amount_status          text,
    kcal                   real,
    protein_g              real,
    carbs_g                real,
    fat_g                  real,
    note                   text,
    UNIQUE (analysis_id, position)
);

-- Logs grow without bound, unlike the food table: the coverage-gap query
-- filters on status, so this index pays off as the table grows.
CREATE INDEX IF NOT EXISTS analysis_items_status_idx ON analysis_items (status);
