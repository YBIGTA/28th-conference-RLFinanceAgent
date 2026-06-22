"""
메인 실행 파일
 1) KOSPI 100 종목 리스트 수집
 2) 네이버 금융 크롤링
 3) 구글 파이낸스 크롤링
 4) 합쳐서 stock_news.csv 저장
"""
import os
import pandas as pd
from get_tickers import get_kospi100
from crawlers.naver import crawl_naver
from crawlers.google_finance import crawl_google
from config import START_DATE, END_DATE, COLUMNS

NAVER_CHECKPOINT = "naver_checkpoint.csv"


def main():
    print("=" * 55)
    print(" KOSPI 100 뉴스 크롤러 시작")
    print(f" 기간: {START_DATE} ~ {END_DATE}")
    print("=" * 55)

    # ── 1. 종목 리스트 ─────────────────────────────────────
    tickers_df = get_kospi100()

    # ── 2. 네이버 크롤링 (체크포인트 이어서 실행) ──────────
    if os.path.exists(NAVER_CHECKPOINT):
        checkpoint_df = pd.read_csv(NAVER_CHECKPOINT, encoding="utf-8-sig")
        done_tickers = set(checkpoint_df["ticker"].astype(str).unique())
        naver_records = checkpoint_df.to_dict("records")
        remaining_df = tickers_df[~tickers_df["ticker"].astype(str).isin(done_tickers)].reset_index(drop=True)
        print(f"\n[체크포인트] {len(done_tickers)}개 완료, {len(remaining_df)}개 남음 — 이어서 시작")
    else:
        naver_records = []
        remaining_df = tickers_df

    print(f"\n[네이버] 크롤링 시작 — {len(remaining_df)}개 종목")
    new_records = crawl_naver(remaining_df, checkpoint_path=NAVER_CHECKPOINT)
    naver_records.extend(new_records)
    print(f"  → 네이버 총 {len(naver_records):,}건 수집")

    # ── 3. 구글 크롤링 ─────────────────────────────────────
    print(f"\n[구글] 크롤링 시작 — {len(tickers_df)}개 종목")
    google_records = crawl_google(tickers_df)
    print(f"  → 구글 총 {len(google_records):,}건 수집")

    # ── 4. 합치기 + 저장 ───────────────────────────────────
    print("\n[저장] 병합 및 중복 제거 중...")
    stock_df = pd.DataFrame(
        naver_records + google_records,
        columns=COLUMNS,
    )
    stock_df = stock_df.sort_values(["date", "company"]).reset_index(drop=True)

    # 동일 날짜·종목·제목 중복 제거
    before = len(stock_df)
    stock_df = stock_df.drop_duplicates(subset=["date", "ticker", "title"])
    after = len(stock_df)
    if before != after:
        print(f"  중복 제거: {before - after:,}건")

    out_path = "stock_news.csv"
    stock_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    if os.path.exists(NAVER_CHECKPOINT):
        os.remove(NAVER_CHECKPOINT)

    print(f"\n완료! 총 {len(stock_df):,}건 → '{out_path}' 저장됨")
    print(f"  네이버: {len(naver_records):,}건 / 구글: {len(google_records):,}건")
    print("\n[샘플 5건]")
    print(stock_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
