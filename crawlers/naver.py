"""
네이버 뉴스 검색 크롤러 (Playwright + 월별 분할)

월별로 나눠 검색해 Naver의 1000건 결과 제한을 회피하고
전체 기간(START_DATE ~ END_DATE) 기사를 수집한다.
"""
import re
import time
from datetime import date, datetime, timedelta
from urllib.parse import quote

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
from config import START_DATE, END_DATE, NAVER_DELAY, NAVER_MAX_PAGE, NAVER_TIMEOUT


# ── 월별 날짜 범위 생성 ───────────────────────────────────────
def _month_ranges(start: date, end: date) -> list[tuple[date, date]]:
    """start ~ end 를 월 단위로 쪼갠 (월시작, 월끝) 리스트 반환."""
    ranges, d = [], start
    while d <= end:
        month_end = date(d.year + (d.month == 12), d.month % 12 + 1, 1) - timedelta(days=1)
        ranges.append((d, min(month_end, end)))
        d = month_end + timedelta(days=1)
    return ranges


# ── 날짜 파싱 ────────────────────────────────────────────────
_ABS_DATE_RE = re.compile(r"(\d{4})[.\-](\d{1,2})[.\-](\d{1,2})")


def _parse_naver_date(raw: str) -> date | None:
    raw = raw.strip()
    today = datetime.now().date()

    m = _ABS_DATE_RE.search(raw)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None

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

    return None


# ── 월 1개 크롤링 ────────────────────────────────────────────
def _crawl_month(page, ticker: str, company: str,
                 ds: str, de: str) -> list[dict]:
    records: list[dict] = []
    query = quote(company)

    for page_num in range(NAVER_MAX_PAGE):
        start_n = page_num * 10 + 1
        url = (
            "https://search.naver.com/search.naver"
            f"?where=news&query={query}"
            f"&pd=3&ds={ds}&de={de}&sort=1&start={start_n}"
        )

        try:
            page.goto(url, timeout=NAVER_TIMEOUT)
            page.wait_for_selector(
                'a[data-heatmap-target=".tit"]', timeout=NAVER_TIMEOUT
            )
        except PWTimeout:
            break
        except Exception as e:
            print(f"\n    [오류] {ticker} {ds}~{de} p{page_num+1}: {e}")
            break

        titles   = page.query_selector_all('a[data-heatmap-target=".tit"]')
        profiles = page.query_selector_all('[data-sds-comp="Profile"]')

        if not titles:
            break

        for title_el, profile_el in zip(titles, profiles):
            try:
                title = title_el.inner_text().strip()
                if not title:
                    continue

                lines = [l.strip()
                         for l in profile_el.inner_text().splitlines()
                         if l.strip()]
                press = lines[0] if lines else ""

                art_date = None
                for line in lines[1:]:
                    art_date = _parse_naver_date(line)
                    if art_date:
                        break

                if art_date is None:
                    continue

                records.append({
                    "date":    art_date.strftime("%Y.%m.%d"),
                    "company": company,
                    "ticker":  ticker,
                    "keyword": "",
                    "title":   title,
                    "press":   press,
                    "source":  "naver",
                })
            except Exception:
                continue

        if len(titles) < 10:
            break

        time.sleep(NAVER_DELAY)

    return records


# ── 전체 종목 크롤링 (외부 호출용) ──────────────────────────
def crawl_naver(tickers_df, checkpoint_path: str | None = None) -> list[dict]:
    """
    tickers_df      : DataFrame(ticker, company)
    checkpoint_path : 종목 완료 시마다 결과를 append할 CSV 경로 (None이면 저장 안 함)
    반환             : list[dict] — 전체 기사 레코드
    """
    import os
    all_records: list[dict] = []
    month_ranges = _month_ranges(START_DATE, END_DATE)

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
            viewport={"width": 1280, "height": 720},
        )
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
            print(f"  [네이버 {rank:>3}/{total}] {company} ({ticker}) ...", end=" ", flush=True)

            stock_records: list[dict] = []
            seen_titles: set[str] = set()

            for month_start, month_end in month_ranges:
                ds = month_start.strftime("%Y.%m.%d")
                de = month_end.strftime("%Y.%m.%d")
                monthly = _crawl_month(page, ticker, company, ds, de)

                for rec in monthly:
                    if rec["title"] not in seen_titles:
                        seen_titles.add(rec["title"])
                        stock_records.append(rec)

            all_records.extend(stock_records)
            print(f"{len(stock_records)}건")

            if checkpoint_path and stock_records:
                import pandas as pd
                from config import COLUMNS
                pd.DataFrame(stock_records, columns=COLUMNS).to_csv(
                    checkpoint_path,
                    mode="a",
                    header=not os.path.exists(checkpoint_path),
                    index=False,
                    encoding="utf-8-sig",
                )

        browser.close()

    return all_records
