from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from demo_app.backend.pipeline import get_pipeline_service


class InferenceRequest(BaseModel):
    mode: str = "sample"
    cash: float = 10_000_000
    holdings: Dict[str, int] = Field(default_factory=dict)
    news_text: str = ""
    crawl_latest: bool = False


app = FastAPI(title="RLFinanceAgent Demo API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/live")
def live():
    return {"status": "alive"}


@app.get("/api/health")
def health():
    try:
        return get_pipeline_service().health()
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/api/infer")
def infer(request: InferenceRequest):
    try:
        return get_pipeline_service().infer(
            mode=request.mode,
            cash=request.cash,
            holdings=request.holdings,
            news_text=request.news_text,
            crawl_latest=request.crawl_latest,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


DIST_DIR = Path(__file__).resolve().parents[1] / "dist"
if DIST_DIR.exists():
    assets = DIST_DIR / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        candidate = DIST_DIR / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(DIST_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "demo_app.backend.app:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
