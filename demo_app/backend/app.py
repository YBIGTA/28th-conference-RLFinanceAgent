from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from demo_app.backend.pipeline import get_pipeline_service, load_latest_cached_news


AUTH_COOKIE = "rlfinance_demo_access"
AUTH_MAX_AGE_SECONDS = 4 * 60 * 60


class AuthRequest(BaseModel):
    password: str


class InferenceRequest(BaseModel):
    mode: str = "sample"
    cash: float = 10_000_000
    holdings: Dict[str, int] = Field(default_factory=dict)
    news_text: str = ""
    crawl_latest: bool = False


def _demo_password() -> str:
    return os.environ.get("DEMO_PASSWORD", "").strip()


def _access_token(password: str) -> str:
    return hmac.new(
        password.encode("utf-8"),
        b"rlfinanceagent-demo-access",
        hashlib.sha256,
    ).hexdigest()


def _has_demo_access(request: Request) -> bool:
    password = _demo_password()
    supplied = request.cookies.get(AUTH_COOKIE, "")
    return bool(password) and hmac.compare_digest(supplied, _access_token(password))


def _require_demo_access(request: Request) -> None:
    if not _demo_password():
        raise HTTPException(status_code=503, detail="데모 비밀번호가 설정되지 않았습니다.")
    if not _has_demo_access(request):
        raise HTTPException(status_code=401, detail="데모 비밀번호 인증이 필요합니다.")


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


@app.get("/api/auth")
def auth_status(request: Request):
    return {
        "authenticated": _has_demo_access(request),
        "configured": bool(_demo_password()),
    }


@app.post("/api/auth")
def authenticate(payload: AuthRequest, response: Response):
    password = _demo_password()
    if not password:
        raise HTTPException(status_code=503, detail="데모 비밀번호가 설정되지 않았습니다.")
    if not hmac.compare_digest(payload.password, password):
        raise HTTPException(status_code=401, detail="비밀번호가 올바르지 않습니다.")
    response.set_cookie(
        key=AUTH_COOKIE,
        value=_access_token(password),
        max_age=AUTH_MAX_AGE_SECONDS,
        httponly=True,
        secure=bool(os.environ.get("K_SERVICE")),
        samesite="strict",
    )
    return {"authenticated": True}


@app.post("/api/infer")
def infer(payload: InferenceRequest, request: Request):
    _require_demo_access(request)
    try:
        return get_pipeline_service().infer(
            mode=payload.mode,
            cash=payload.cash,
            holdings=payload.holdings,
            news_text=payload.news_text,
            crawl_latest=payload.crawl_latest,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/api/news/latest")
def latest_news(request: Request):
    _require_demo_access(request)
    try:
        return load_latest_cached_news()
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


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
