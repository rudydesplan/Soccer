"""FastAPI backend for the Salary Benchmark interface."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root and backend dir to path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
_BACKEND_DIR = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import players, benchmark

app = FastAPI(
    title="Soccer Salary Benchmark API",
    version="1.0.0",
    description="Predict expected salary range for football players based on market value, age, position, and league.",
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(players.router, prefix="/api")
app.include_router(benchmark.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}
