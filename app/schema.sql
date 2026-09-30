-- Nutrition table schema. Safe to run repeatedly.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- fuzzy matching (Step 6)

CREATE TABLE IF NOT EXISTS foods (
    id               text PRIMARY KEY,
    fdc_id           integer NOT NULL,
    usda_description text NOT NULL,
    name_en          text NOT NULL,
    name_tr          text NOT NULL,
    kcal_100g        real NOT NULL CHECK (kcal_100g >= 0),
    protein_100g     real NOT NULL CHECK (protein_100g >= 0),
    carbs_100g       real NOT NULL CHECK (carbs_100g >= 0),
    fat_100g         real NOT NULL CHECK (fat_100g >= 0),
    default_grams    real NOT NULL CHECK (default_grams > 0),
    density_g_per_ml real CHECK (density_g_per_ml > 0),
    note             text NOT NULL DEFAULT ''
);

-- One row (and one embedding) per name: Schema 2 from the design discussion.
CREATE TABLE IF NOT EXISTS food_aliases (
    id           serial PRIMARY KEY,
    food_id      text NOT NULL REFERENCES foods (id) ON DELETE CASCADE,
    alias        text NOT NULL,
    alias_folded text NOT NULL,         -- app.text.fold(alias): exact/fuzzy key
    kind         text NOT NULL CHECK (kind IN ('name_en', 'name_tr', 'alias')),
    embedding    vector(384) NOT NULL,  -- of app.text.embedding_text(alias)
    UNIQUE (food_id, alias)
);
-- alias_folded may repeat within one food ("Yoğurt" and "yogurt" both fold to
-- "yogurt", but their embeddings differ). It must never point to two foods;
-- scripts/build_food_table.py and tests/test_food_table.py enforce that.
-- No vector index on purpose: with ~440 rows an exact scan takes microseconds,
-- while HNSW/IVFFlat would add approximate results (recall < 100%).

CREATE TABLE IF NOT EXISTS food_portions (
    food_id text NOT NULL REFERENCES foods (id) ON DELETE CASCADE,
    unit    text NOT NULL,
    grams   real NOT NULL CHECK (grams > 0),
    source  text NOT NULL,
    PRIMARY KEY (food_id, unit)
);

-- Which embedding model produced the stored vectors. Vectors from different
-- models are not comparable, so queries must check this before searching.
CREATE TABLE IF NOT EXISTS ingest_metadata (
    key   text PRIMARY KEY,
    value text NOT NULL
);
