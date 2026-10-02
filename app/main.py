"""FastAPI service: a thin HTTP layer over the astra_nutrition library.

All endpoints are plain ``def``: the work inside is blocking (llama.cpp
inference, sync psycopg, optional sync HTTP to the judge), so FastAPI runs
them in its threadpool and the event loop stays free (for /health above all).

Run (from the project root, db running):
    uvicorn app.main:app
"""

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.analysis_log import AnalysisLogger
from app.config import Settings, get_settings
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
    log_failures: int  # analyses whose log could not be written since startup


class CoverageGap(BaseModel):
    name: str
    times: int
    closest_food: str | None


def _judge_label(settings: Settings) -> str | None:
    provider = settings.llm_judge_provider
    if provider == "off":
        return None
    return f"{provider}:{getattr(settings, f'{provider}_model')}"


def create_app(
    analyzer: Analyzer | None = None,
    pool=None,
    analysis_logger: AnalysisLogger | None = None,
) -> FastAPI:
    """Build the app. Tests inject the analyzer, pool and logger; production
    builds them once at startup (lifespan) and closes the pool at shutdown."""

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
        if analysis_logger is not None:
            app.state.analysis_log = analysis_logger
        else:
            settings = get_settings()
            app.state.analysis_log = AnalysisLogger(
                app.state.pool,
                log_meal_text=settings.log_meal_text,
                judge=_judge_label(settings),
            )
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
    def analyze(
        body: AnalyzeRequest, request: Request, background: BackgroundTasks
    ) -> AnalysisResult:
        """Parse a meal and compute its nutrition. Never guesses silently:
        every item has a status and the totals carry honesty flags.

        The result is logged after the response is sent (best effort)."""
        start = time.perf_counter()
        result = request.app.state.analyzer.analyze(body.meal_text)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        background.add_task(request.app.state.analysis_log.write, result, latency_ms)
        return result

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

    @app.get("/stats/unmatched", response_model=list[CoverageGap])
    def unmatched(
        request: Request, limit: int = Query(default=20, ge=1, le=200)
    ) -> list[CoverageGap]:
        """Most frequent unmatched names: the next foods to add to the table."""
        try:
            gaps = request.app.state.analysis_log.coverage_gaps(limit)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="database unavailable") from exc
        return [CoverageGap(**gap) for gap in gaps]

    @app.get("/health", response_model=HealthOut)
    def health(request: Request) -> HealthOut:
        """Database check only: never touches the LLM."""
        try:
            with request.app.state.pool.connection(timeout=2.0) as conn:
                conn.execute("SELECT 1")
        except Exception as exc:
            raise HTTPException(status_code=503, detail="database unavailable") from exc
        return HealthOut(
            status="ok",
            database="ok",
            version=__version__,
            log_failures=request.app.state.analysis_log.failures,
        )

    return app


app = create_app()
