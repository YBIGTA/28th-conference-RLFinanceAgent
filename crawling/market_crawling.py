"""
시장 전체 기사 크롤링 스크립트 (네이버 뉴스) v4
- 일별로 쪼개서 검색 → 날짜 완전히 균등하게 수집
- 거래일(월~금)만 수집
- 하루 키워드당 최대 3건 제한
- 수집 항목: date, company, ticker, keyword, title, press, source
"""

from anyio import Path
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import re
import random
from collections import defaultdict, deque
from datetime import datetime, timedelta

# ── 설정 ──────────────────────────────────────────────
START_DATE  = "2025.01.01"
END_DATE    = datetime.now().strftime("%Y.%m.%d")   # 오늘 날짜 자동 반영
DELAY_MIN   = 3.0    # 요청 간 최소 대기 (초)
DELAY_MAX   = 6.0    # 요청 간 최대 대기 (초) — 랜덤으로 패턴 감지 방지
BLOCK_PAUSE = 300    # 403 최종 실패 후 대기 (초, 5분)
BLOCK_LIMIT = 5      # 연속 403 횟수 — 초과 시 10분 강제 휴식
MAX_PER_DAY = 3
_DIR = Path(__file__).parent
SAVE_PATH = str(_DIR / "market_news_v4.csv")
CKPT_PATH = str(_DIR / "market_news_v4_checkpoint.txt")
PRINT_EVERY = 10    # 진행바 상세 출력 주기 (작업 수)
SAVE_EVERY  = 10    # CSV 저장 주기
ETA_WINDOW  = 30    # ETA 계산에 쓸 최근 작업 수

KEYWORDS = [
    "코스피", "코스닥", "증시", "국내 증시", "주식시장",
    "외국인 순매수", "기관 순매수", "개인 투자자",
    "금리", "환율", "원달러", "유가", "물가", "인플레이션",
    "경기침체", "수출", "반도체", "미국 증시",
    "FOMC", "연준", "한국은행", "기준금리", "중국 경기"
]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
]

def make_session() -> requests.Session:
    """새 세션 생성 — 네이버 홈 방문으로 실제 쿠키 획득"""
    s = requests.Session()
    s.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    })
    try:
        s.get("https://www.naver.com", timeout=10)
        time.sleep(random.uniform(1.5, 3.0))
    except Exception:
        pass
    return s

SESSION = make_session()

# ── 거래일 목록 생성 (월~금) ───────────────────────────
def generate_trading_days() -> list:
    days = []
    cur = datetime.strptime(START_DATE, "%Y.%m.%d")
    end = datetime.strptime(END_DATE,   "%Y.%m.%d")
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return days

TRADING_DAYS = generate_trading_days()

# ── 날짜 유틸 ──────────────────────────────────────────
def normalize_date(raw: str) -> str:
    m = re.search(r'(\d{4}\.\d{2}\.\d{2})', raw.strip())
    return m.group(1) if m else ""

def in_range(d: str) -> bool:
    return START_DATE <= d <= END_DATE

# ── 검색결과 파싱 ──────────────────────────────────────
def crawl_search_page(keyword: str, date_str: str):
    """성공 시 list, 실패 시 None 반환 — None이면 체크포인트에 기록하지 않음"""
    global SESSION
    params = {
        "where": "news",
        "query": keyword,
        "start": 1,
        "nso":   f"so:dd,p:from{date_str}to{date_str}",
    }
    res = None
    for attempt in range(3):
        try:
            res = SESSION.get(
                "https://search.naver.com/search.naver",
                params=params, timeout=10
            )
            res.raise_for_status()
            break
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status == 403:
                print(f"    [403 차단] 세션 교체 후 재시도 ({attempt+1}/3)...", flush=True)
                SESSION = make_session()   # 새 UA + 새 쿠키
                time.sleep(random.uniform(20, 40))
                if attempt == 2:
                    print(f"    [403 차단] 3회 실패 — 나중에 재시도", flush=True)
                    return None
            else:
                wait = (attempt + 1) * 5
                if attempt < 2:
                    print(f"    [HTTP {status}] {wait}초 후 재시도", flush=True)
                    time.sleep(wait)
                else:
                    print(f"    [HTTP {status}] 3회 실패, 스킵", flush=True)
                    return None
        except requests.RequestException as e:
            wait = (attempt + 1) * 5
            if attempt < 2:
                print(f"    [오류] {e} → {wait}초 후 재시도", flush=True)
                time.sleep(wait)
            else:
                print(f"    [오류] 3회 실패, 스킵", flush=True)
                return None

    soup = BeautifulSoup(res.text, "html.parser")
    results = []

    news_list = soup.select_one("ul.list_news")
    if not news_list:
        return []

    article_blocks = news_list.select("div.c5JFKSsfV1F7jYk8uxWp")

    for block in article_blocks:
        title_tag = block.select_one("span.sds-comps-text-type-headline1")
        if not title_tag:
            continue
        title = title_tag.get_text(strip=True)
        if not title:
            continue

        date_tag  = block.select_one("span.sds-comps-profile-info-subtext")
        art_date  = normalize_date(date_tag.get_text(strip=True)) if date_tag else ""
        # 최근 기사: "4주 전" 등 상대 시간으로 표시 → 검색한 날짜로 대체
        # nso=p:from{date}to{date} 필터가 해당 날짜만 반환하므로 안전
        if not art_date:
            art_date = f"{date_str[:4]}.{date_str[4:6]}.{date_str[6:]}"

        press_tag = block.select_one("span.sds-comps-profile-info-title-text")
        press     = press_tag.get_text(strip=True) if press_tag else ""

        results.append({
            "date":    art_date,
            "company": "시장전체",
            "ticker":  "",
            "keyword": keyword,
            "title":   title,
            "press":   press,
            "source":  "naver",
        })

    return results

# ── 체크포인트 ─────────────────────────────────────────
def load_checkpoint() -> set:
    done = set()
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH, encoding="utf-8") as f:
            for line in f:
                done.add(line.strip())
        print(f"[재시작] 완료된 작업 {len(done)}개 확인, 이어서 시작\n")
    return done

def mark_done(keyword: str, date_str: str):
    with open(CKPT_PATH, "a", encoding="utf-8") as f:
        f.write(f"{keyword}|{date_str}\n")

def save_csv(records: list):
    pd.DataFrame(records).to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")

# ── 시간 포맷 ─────────────────────────────────────────
def fmt(sec: float) -> str:
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}시간 {m}분 {s}초" if h else (f"{m}분 {s}초" if m else f"{s}초")

# ── 진행률 출력 ───────────────────────────────────────
def print_progress(done: int, total: int, elapsed: float, recent_times: deque):
    pct = done / total
    bar = "█" * int(30 * pct) + "░" * (30 - int(30 * pct))
    if recent_times:
        # 최근 ETA_WINDOW개 작업의 평균으로 ETA 계산 (초반 outlier 영향 최소화)
        avg     = sum(recent_times) / len(recent_times)
        eta     = avg * (total - done)
        eta_str  = fmt(eta)
        eta_time = (datetime.now() + timedelta(seconds=eta)).strftime("%H:%M:%S")
    else:
        eta_str = eta_time = "-"
    print(f"\n  ┌─ 진행률: [{bar}] {pct*100:.1f}% ({done:,}/{total:,})")
    print(f"  ├─ 경과:            {fmt(elapsed)}")
    print(f"  ├─ 예상 남은 시간:  {eta_str}")
    print(f"  └─ 예상 완료 시각:  {eta_time}\n", flush=True)

# ── 메인 ──────────────────────────────────────────────
def main():
    global SESSION
    total_start = time.time()
    total_tasks = len(KEYWORDS) * len(TRADING_DAYS)

    print("=== 시장 뉴스 크롤링 시작 v4 ===")
    print(f"기간:       {START_DATE} ~ {END_DATE}")
    print(f"거래일 수:  {len(TRADING_DAYS)}일")
    print(f"키워드:     {len(KEYWORDS)}개 / 하루 최대 {MAX_PER_DAY}건")
    print(f"총 작업 수: {total_tasks:,}개 (키워드 × 거래일)")
    print(f"최대 수집:  약 {len(TRADING_DAYS) * len(KEYWORDS) * MAX_PER_DAY:,}건")
    print(f"저장 파일:  {SAVE_PATH}")
    print(f"시작 시각:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    done_set = load_checkpoint()

    # 재시작 시 이전 세션 결과 복원
    if os.path.exists(SAVE_PATH) and done_set:
        _df = pd.read_csv(SAVE_PATH, encoding="utf-8-sig")
        _df["date"] = pd.to_datetime(_df["date"], errors="coerce").dt.strftime("%Y.%m.%d")
        all_results = _df.to_dict("records")
        print(f"[복원] 기존 CSV에서 {len(all_results):,}건 로드\n")
    else:
        all_results = []

    recent_times = deque(maxlen=ETA_WINDOW)   # 이동 평균용 슬라이딩 윈도우
    done_count   = len(done_set)
    consec_403   = 0   # 연속 403 카운터
    consec_zero  = 0   # 연속 0건 카운터 (소프트 차단 감지용)

    for ki, keyword in enumerate(KEYWORDS):
        seen_titles = set()
        daily_count = defaultdict(int)
        # 키워드 전환마다 세션 교체 — 누적 요청 초기화
        SESSION = make_session()
        print(f"\n[키워드 {ki+1}/{len(KEYWORDS)}] '{keyword}' 시작 — 세션 갱신", flush=True)

        for date_str in TRADING_DAYS:
            task_key = f"{keyword}|{date_str}"
            if task_key in done_set:
                continue

            t0       = time.time()
            articles = crawl_search_page(keyword, date_str)

            # None = 요청 실패(403) → 체크포인트 미기록, 재시도
            if articles is None:
                consec_403 += 1
                if consec_403 >= BLOCK_LIMIT:
                    print(f"    [연속 {consec_403}회 차단] 세션 교체 + 5분 강제 휴식...", flush=True)
                    SESSION = make_session()
                    time.sleep(300)
                    consec_403 = 0
                continue
            consec_403 = 0

            # 소프트 차단 감지: 연속 5건 이상 0건이면 세션 교체
            if len(articles) == 0:
                consec_zero += 1
                if consec_zero >= 5:
                    print(f"    [소프트 차단] 연속 {consec_zero}회 0건 → 세션 교체", flush=True)
                    SESSION = make_session()
                    time.sleep(random.uniform(15, 30))
                    consec_zero = 0
                    # 체크포인트 미기록 — 재시도
                    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
                    elapsed_task = time.time() - t0
                    recent_times.append(elapsed_task)
                    continue
            else:
                consec_zero = 0

            collected = 0
            for a in articles:
                art_date = a["date"]
                if not art_date or not in_range(art_date):
                    continue

                dedup_key = f"{art_date}|{a['title']}"
                if dedup_key in seen_titles:
                    continue
                if daily_count[art_date] >= MAX_PER_DAY:
                    continue

                seen_titles.add(dedup_key)
                daily_count[art_date] += 1
                all_results.append(a)
                collected += 1

            done_count += 1
            mark_done(keyword, date_str)

            # 매 작업마다 한 줄 출력 (인라인 ETA 포함) — sleep 전에 출력
            fmt_date = f"{date_str[:4]}.{date_str[4:6]}.{date_str[6:]}"
            pct      = done_count / total_tasks * 100
            if recent_times:
                eta_sec  = (sum(recent_times) / len(recent_times)) * (total_tasks - done_count)
                eta_str  = fmt(eta_sec)
                eta_time = (datetime.now() + timedelta(seconds=eta_sec)).strftime("%H:%M")
            else:
                eta_str = eta_time = "-"
            print(
                f"  [{ki+1:2d}/{len(KEYWORDS)}] {keyword:<10s} | {fmt_date}"
                f" → {collected}건  (누적 {len(all_results):,}건 / {pct:.1f}%)"
                f"  ETA {eta_str} ({eta_time})",
                flush=True
            )

            # PRINT_EVERY마다 진행바 + ETA 상세 출력 & 저장
            if done_count % PRINT_EVERY == 0:
                save_csv(all_results)
                print(f"  💾 저장 완료 — {SAVE_PATH}")
                print_progress(done_count, total_tasks,
                               time.time() - total_start, recent_times)

            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

            # sleep 포함한 전체 작업 시간으로 ETA 계산 (더 정확)
            elapsed_task = time.time() - t0
            recent_times.append(elapsed_task)

    # 최종 저장
    df = pd.DataFrame(all_results)
    if df.empty:
        print("\n수집된 데이터가 없습니다.")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date", ascending=False)
    df = df.drop_duplicates(subset=["date", "title"])
    df["date"] = df["date"].dt.strftime("%Y.%m.%d")
    df.to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")

    print(f"\n{'='*45}")
    print(f"=== 크롤링 완료 ===")
    print(f"총 소요 시간: {fmt(time.time() - total_start)}")
    print(f"총 수집 건수: {len(df):,}건")
    print(f"저장 위치:    {SAVE_PATH}")
    print(f"완료 시각:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*45}")
    print("\n키워드별 수집 건수:")
    print(df.groupby("keyword").size().sort_values(ascending=False).to_string())
    print("\n월별 수집 건수:")
    df["month"] = df["date"].dt.to_period("M")
    print(df.groupby("month").size().sort_index().to_string())

if __name__ == "__main__":
    main()
