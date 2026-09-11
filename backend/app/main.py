"""Respira clinical backend - FastAPI application.

Wires auth, doctors, patients, real ML analysis, history, explainability,
reports, and system routes. Loads the REAL Respira checkpoints once at
startup via ModelManager (CUDA when available).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api import analyses, auth, doctors, explainability, history, patients, reports, system
from backend.app.config import settings
from backend.app.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Load real models once (lazy singleton also loads on first use).
    from backend.app.services.model_manager import get_manager

    try:
        get_manager()
    except Exception:
        pass  # status endpoint surfaces the error; API stays up
    yield


app = FastAPI(title=settings.APP_NAME, description=settings.APP_TAGLINE, version=settings.APP_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL, "http://localhost:5173", "http://localhost:3000", "app://."],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def secure_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response


@app.exception_handler(Exception)
async def friendly_errors(request: Request, exc: Exception):
    from fastapi import HTTPException

    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(status_code=500, content={"detail": "Something went wrong. Please try again."})


prefix = settings.API_PREFIX
app.include_router(auth.router, prefix=prefix)
app.include_router(doctors.router, prefix=prefix)
app.include_router(patients.router, prefix=prefix)
app.include_router(analyses.router, prefix=prefix)
app.include_router(history.router, prefix=prefix)
app.include_router(explainability.router, prefix=prefix)
app.include_router(reports.router, prefix=prefix)
app.include_router(system.router, prefix=prefix)


@app.get("/")
def root():
    return {"app": settings.APP_NAME, "tagline": settings.APP_TAGLINE, "api": settings.API_PREFIX}
