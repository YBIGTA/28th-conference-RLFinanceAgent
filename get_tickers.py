"""
FinanceDataReader로 KOSPI 시총 상위 100개 종목코드 + 회사명 반환
pykrx 대비 로그인 불필요, 안정적
"""
import FinanceDataReader as fdr


def get_kospi100():
    print("[종목] KOSPI 100 로딩 중 ...")

    df = fdr.StockListing("KOSPI")

    # 컬럼명 정규화 (버전마다 다를 수 있음)
    df.columns = [c.strip() for c in df.columns]
    cap_col  = next(c for c in df.columns if "marcap" in c.lower() or "시가총액" in c.lower())
    code_col = next(c for c in df.columns if c in ("Code", "Symbol", "종목코드"))
    name_col = next(c for c in df.columns if c in ("Name", "종목명", "회사명"))

    df = df.sort_values(cap_col, ascending=False).head(100)
    df = df[[code_col, name_col]].copy()
    df.columns = ["ticker", "company"]
    df = df.reset_index(drop=True)

    print(f"  → {len(df)}개 종목 로드 완료")
    return df