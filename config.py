"""
공통 설정 — 모든 크롤러가 이 파일을 import해서 사용
"""
from datetime import date

# ── 크롤링 기간 ─────────────────────────────
START_DATE = date(2025, 1, 1)
END_DATE   = date(2026, 5, 18)

# ── 네이버 설정 ─────────────────────────────
NAVER_DELAY    = 0.5    # 페이지 이동 간 대기(초)
NAVER_MAX_PAGE = 100    # 종목당 최대 페이지 (네이버 검색 최대 1000건)
NAVER_TIMEOUT  = 15000  # Playwright 타임아웃(ms)

# ── 구글 설정 ─────────────────────────────
GOOGLE_DELAY   = 1.5    # 페이지 이동 간 대기(초)
GOOGLE_TIMEOUT = 15000  # Playwright 대기 타임아웃(ms)

# ── 공통 HTTP 헤더 ─────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9",
}

# ── 출력 컬럼 순서 ─────────────────────────
COLUMNS = ["date", "company", "ticker", "keyword", "title", "press", "source"]
