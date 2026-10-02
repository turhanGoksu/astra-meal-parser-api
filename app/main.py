"""FastAPI service: a thin HTTP layer over the astra_nutrition library.

All endpoints are plain ``def``: the work inside is blocking (llama.cpp
inference, sync psycopg, optional sync HTTP to the judge), so FastAPI runs
them in its threadpool and the event loop stays free (for /health above all).

Run (from the project root, db running):
    uvicorn app.main:app
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.config import get_settings
from app.factory import build_analyzer
from astra_nutrition import AnalysisResult, Analyzer, __version__
from astra_nutrition.matcher import MatchMethod

MAX_MEAL_CHARS = 1000  # keeps prompts far below the model's 2048-token context


class AnalyzeRequest(BaseModel):
    meal_text: str = Field(min_length=1, max_length=MAX_MEAL_CHARS)


class CandidateOut(BaseModel):
    food_id: str
    alias: str
    similarity: float
    method: MatchMethod


class HealthOut(BaseModel):
    status: str
    database: str
    version: str


def create_app(analyzer: Analyzer | None = None, pool=None) -> FastAPI:
    """Build the app. Tests inject an analyzer and a pool; production builds
    both once at startup (lifespan) and closes the pool at shutdown."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned_pool = None
        if pool is None:
            from pgvector.psycopg import register_vector
            from psycopg_pool import ConnectionPool

            settings = get_settings()
            owned_pool = ConnectionPool(
                settings.database_url or "",
                min_size=1,
                max_size=5,
                configure=register_vector,
                open=True,
            )
        app.state.pool = pool or owned_pool
        app.state.analyzer = analyzer or build_analyzer(get_settings(), app.state.pool)
        yield
        if owned_pool is not None:
            owned_pool.close()

    app = FastAPI(
        title="Astra Nutrition API",
        version=__version__,
        description="Turkish/English meal text to foods, grams and nutrition.",
        lifespan=lifespan,
    )

    @app.post("/analyze", response_model=AnalysisResult)
    def analyze(body: AnalyzeRequest, request: Request) -> AnalysisResult:
        """Parse a meal and compute its nutrition. Never guesses silently:
        every item has a status and the totals carry honesty flags."""
        return request.app.state.analyzer.analyze(body.meal_text)

    @app.get("/foods/search", response_model=list[CandidateOut])
    def search_foods(
        request: Request,
        q: str = Query(min_length=1, max_length=100),
        k: int = Query(default=5, ge=1, le=20),
    ) -> list[CandidateOut]:
        """Candidates from every matching stage, with scores (for inspection)."""
        return [
            CandidateOut(
                food_id=c.food_id,
                alias=c.alias,
                similarity=c.similarity,
                method=c.method,
            )
            for c in request.app.state.analyzer.candidates(q, k)
        ]

    @app.get("/health", response_model=HealthOut)
    def health(request: Request) -> HealthOut:
        """Database check only: never touches the LLM."""
        try:
            with request.app.state.pool.connection() as conn:
                conn.execute("SELECT 1")
        except Exception as exc:
            raise HTTPException(status_code=503, detail="database unavailable") from exc
        return HealthOut(status="ok", database="ok", version=__version__)

    return app


app = create_app()
