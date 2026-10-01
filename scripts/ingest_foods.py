"""Load the bundled food table (foods.csv, food_portions.csv) into PostgreSQL.

Idempotent: the tables are emptied and reloaded in a single transaction, so a
failed run leaves the previous data untouched.

Usage (from the project root, with the db service running):
    python -m scripts.ingest_foods
"""

from datetime import UTC, datetime

from app.config import get_settings
from astra_nutrition.embeddings import Embedder, SentenceTransformerEmbedder
from astra_nutrition.index.postgres import apply_schema, connect
from astra_nutrition.tables import alias_rows, read_table


def typed_food(row: dict[str, str]) -> dict[str, object]:
    """Convert a foods.csv row to Python types matching the SQL columns."""
    numeric = ("kcal_100g", "protein_100g", "carbs_100g", "fat_100g", "default_grams")
    density = row["density_g_per_ml"]
    return {
        **row,
        **{column: float(row[column]) for column in numeric},
        "fdc_id": int(row["fdc_id"]),
        "density_g_per_ml": float(density) if density else None,
    }


def ingest(embedder: Embedder, database_url: str | None) -> dict[str, int]:
    """Embed every alias and reload all nutrition tables."""
    foods = read_table("foods.csv")
    portions = [
        {**p, "grams": float(p["grams"])} for p in read_table("food_portions.csv")
    ]
    aliases = alias_rows(foods)
    vectors = embedder.embed([alias for _, alias, _, _ in aliases])

    with connect(database_url) as conn, conn.transaction():
        apply_schema(conn)
        conn.execute("TRUNCATE foods, food_aliases, food_portions, ingest_metadata")
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO foods (id, fdc_id, usda_description, name_en, name_tr,
                    kcal_100g, protein_100g, carbs_100g, fat_100g,
                    default_grams, density_g_per_ml, note)
                VALUES (%(id)s, %(fdc_id)s, %(usda_description)s, %(name_en)s,
                    %(name_tr)s, %(kcal_100g)s, %(protein_100g)s, %(carbs_100g)s,
                    %(fat_100g)s, %(default_grams)s, %(density_g_per_ml)s, %(note)s)
                """,
                [typed_food(food) for food in foods],
            )
            cur.executemany(
                """
                INSERT INTO food_aliases (food_id, alias, alias_folded, kind, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                [(*row, vector) for row, vector in zip(aliases, vectors, strict=True)],
            )
            cur.executemany(
                """
                INSERT INTO food_portions (food_id, unit, grams, source)
                VALUES (%(food_id)s, %(unit)s, %(grams)s, %(source)s)
                """,
                portions,
            )
            cur.executemany(
                "INSERT INTO ingest_metadata (key, value) VALUES (%s, %s)",
                [
                    ("embedding_model", embedder.model_name),
                    ("embedding_signature", embedder.signature),
                    ("embedding_dimension", str(embedder.dimension)),
                    ("ingested_at", datetime.now(UTC).isoformat()),
                ],
            )
    return {"foods": len(foods), "aliases": len(aliases), "portions": len(portions)}


if __name__ == "__main__":
    settings = get_settings()
    counts = ingest(
        SentenceTransformerEmbedder(
            settings.embedding_model_name, prefix=settings.embedding_prefix
        ),
        settings.database_url,
    )
    print("Ingested:", ", ".join(f"{v} {k}" for k, v in counts.items()))
