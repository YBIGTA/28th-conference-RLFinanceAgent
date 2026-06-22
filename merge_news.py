"""
market_news.csv (친구 파트) + stock_news.csv (내 파트) → all_news.csv
"""
import pandas as pd
from config import COLUMNS


def merge():
    # ── 로드 ────────────────────────────────────────────────
    try:
        market_df = pd.read_csv("market_news.csv", encoding="utf-8-sig")
        print(f"시장 데이터: {len(market_df):,}건")
    except FileNotFoundError:
        print("[!] market_news.csv 없음 → 내 파트만으로 진행")
        market_df = pd.DataFrame(columns=COLUMNS)

    stock_df = pd.read_csv("stock_news.csv", encoding="utf-8-sig")
    print(f"종목 데이터: {len(stock_df):,}건")

    # ── 컬럼 순서 통일 ──────────────────────────────────────
    market_df = market_df.reindex(columns=COLUMNS)
    stock_df  = stock_df.reindex(columns=COLUMNS)

    # ── 합치기 ──────────────────────────────────────────────
    final = pd.concat([market_df, stock_df], ignore_index=True)
    final = final.sort_values(["date", "company"]).reset_index(drop=True)

    before = len(final)
    final = final.drop_duplicates(subset=["date", "ticker", "title"])
    removed = before - len(final)
    if removed:
        print(f"중복 제거: {removed:,}건")

    final.to_csv("all_news.csv", index=False, encoding="utf-8-sig")
    print(f"\n최종: {len(final):,}건 → all_news.csv 저장 완료")
    print(final.head(3).to_string(index=False))


if __name__ == "__main__":
    merge()
