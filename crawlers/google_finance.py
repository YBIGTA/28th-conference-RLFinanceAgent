"""
구글 파이낸스 종목별 뉴스 크롤러
JavaScript 렌더링 → Playwright (headless Chromium) 사용

URL 패턴: https://www.google.com/finance/quote/{TICKER}:KRX?hl=ko
"""
import re
import time
from datetime import date, datetime, timedelta

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
from config import START_DATE, END_DATE, GOOGLE_DELAY, GOOGLE_TIMEOUT


# ── 한국어 상대 날짜 파싱 ───────────────────────────────────
def _parse_google_date(raw: str) -> date | None:
    """
    구글 파이낸스의 다양한 날짜 표현 → date 객체

    지원 형식
    ---------
    '방금'  '방금 전'                → 오늘
    'N분 전'                        → 오늘
    'N시간 전'                       → 오늘
    '어제'                           → 어제
    'N일 전'                         → 오늘 - N일
    'N주 전'                         → 오늘 - N*7일
    'N개월 전'                       → 오늘 - N*30일 (근사)
    'YYYY년 M월 D일'                 → 절대 날짜
    'M월 D일'                        → 올해
    'YYYY. M. D.'  '2025.03.15' 등  → 절대 날짜
    """
    raw = raw.strip()
    today = datetime.now().date()

    # ── 상대 표현 ──────────────────────────────────────────
    if any(k in raw for k in ("방금", "초 전", "분 전", "시간 전")):
        return today
    if "어제" in raw:
        return today - timedelta(days=1)

    m = re.search(r"(\d+)\s*일\s*전", raw)
    if m:
        return today - timedelta(days=int(m.group(1)))

    m = re.search(r"(\d+)\s*주\s*전", raw)
    if m:
        return today - timedelta(weeks=int(m.group(1)))

    m = re.search(r"(\d+)\s*개월\s*전", raw)
    if m:
        return today - timedelta(days=int(m.group(1)) * 30)

    # ── 절대 날짜: YYYY년 M월 D일 ─────────────────────────
    m = re.search(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", raw)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # ── 절대 날짜: M월 D일 ────────────────────────────────
    m = re.search(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", raw)
    if m:
        return date(today.year, int(m.group(1)), int(m.group(2)))

    # ── 절대 날짜: YYYY.MM.DD / YYYY-MM-DD ───────────────
    m = re.search(r"(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})", raw)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    return None


# ── 종목 1개 크롤링 ─────────────────────────────────────────
def _crawl_one(page, ticker: str, company: str) -> list[dict]:
    """
    이미 열려 있는 Playwright page 객체로 구글 파이낸스 뉴스 수집.
    """
    url = f"https://www.google.com/finance/quote/{ticker}:KRX?hl=ko"
    records: list[dict] = []

    try:
        page.goto(url, timeout=GOOGLE_TIMEOUT)
        page.wait_for_selector(".LIT7If", timeout=GOOGLE_TIMEOUT)
    except PWTimeout:
        print(f"    [구글 타임아웃] {ticker} 뉴스 섹션 미발견 → 건너뜀")
        return records
    except Exception as e:
        print(f"    [구글 오류] {ticker}: {e}")
        return records

    items = page.query_selector_all(".LIT7If")

    for item in items:
        try:
            title_el = item.query_selector("a")
            title = title_el.inner_text().strip() if title_el else ""

            meta_el = item.query_selector(".jDwR9c")
            meta_text = meta_el.inner_text().strip() if meta_el else ""

            if "·" in meta_text:
                parts    = meta_text.split("·", 1)
                press    = parts[0].strip()
                date_raw = parts[1].strip()
            else:
                press    = meta_text
                date_raw = ""

            if not title:
                continue

            art_date = _parse_google_date(date_raw)
            if art_date is None:
                continue
            if art_date > END_DATE or art_date < START_DATE:
                continue

            records.append({
                "date":    art_date.strftime("%Y.%m.%d"),
                "company": company,
                "ticker":  ticker,
                "keyword": "",
                "title":   title,
                "press":   press,
                "source":  "google",
            })

        except Exception:
            continue

    return records


# ── 전체 종목 크롤링 (외부 호출용) ─────────────────────────
def crawl_google(tickers_df) -> list[dict]:
    """
    tickers_df : DataFrame(ticker, company)
    반환        : list[dict] — 전체 기사 레코드
    """
    all_records: list[dict] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        context = browser.new_context(
            locale="ko-KR",
            timezone_id="Asia/Seoul",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        # 이미지·폰트 차단으로 속도 향상
        context.route(
            "**/*.{png,jpg,jpeg,gif,webp,svg,woff,woff2,ttf}",
            lambda route, _: route.abort(),
        )

        page = context.new_page()
        total = len(tickers_df)

        for i, row in tickers_df.iterrows():
            ticker  = row["ticker"]
            company = row["company"]
            rank    = i + 1
            print(f"  [구글 {rank:>3}/{total}] {company} ({ticker}) ...", end=" ")

            records = _crawl_one(page, ticker, company)
            all_records.extend(records)
            print(f"{len(records)}건")

            time.sleep(GOOGLE_DELAY)

        browser.close()

    return all_records
