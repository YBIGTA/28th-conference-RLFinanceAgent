"""
data/gdrive_loader.py
────────────────────────────────────────────────────────────────
JSONL 형식 임베딩 로더 (Upstage autoencoder 등)

실제 JSONL 스키마:
  {
    "row_id": 0,
    "source_type": "market",          # "market" | "company"
    "date": "2026.05.21",             # "YYYY.MM.DD"
    "company": "시장전체",
    "ticker": "",                     # 종목코드 or ""
    "keyword": "중국 경기",
    "title": "...",
    "press": "머니투데이방송",
    "source": "naver",
    "embedding_text": "...",
    "embedding_model": "upstage-autoencoder-16d",
    "embedding_dim": 16,
    "embedding": [-0.363, 0.019, ...]   # List[float], 길이 = embedding_dim
  }

출력 DataFrame 컬럼:
  date (YYYY-MM-DD), tic (or "market"), source_type, keyword,
  emb_0 … emb_{D-1}

Google Drive에서 .jsonl 파일을 다운로드하거나,
로컬 파일/디렉토리를 직접 읽을 수 있습니다.
"""

from __future__ import annotations
import io, json
from pathlib import Path
from typing import Optional, List
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════
# 핵심 파싱 유틸
# ════════════════════════════════════════════════════════════════

def _parse_jsonl_to_df(path: Path, emb_dim: Optional[int] = None) -> pd.DataFrame:
    """
    JSONL 파일 한 개를 읽어 DataFrame으로 변환.

    - date 정규화: "YYYY.MM.DD" / "YYYY-MM-DD" / "YYYYMMDD" → "YYYY-MM-DD"
    - ticker가 비어 있으면 source_type=="market" 행은 tic="__market__"
    - embedding 리스트 → emb_0 … emb_{D-1}
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            # ── date 정규화 ──────────────────────────────────────
            raw_date = str(rec.get("date", ""))
            date = _normalise_date(raw_date)
            if date is None:
                continue

            # ── ticker / tic 결정 ────────────────────────────────
            ticker = str(rec.get("ticker", "")).strip()
            src    = str(rec.get("source_type", "")).strip()
            # "stock" 은 "company" 와 동일하게 취급
            if src == "stock":
                src = "company"
            if ticker:
                # 6자리 숫자 코드 → Yahoo Finance 형식 (.KS 추가)
                tic = ticker + ".KS" if ticker.isdigit() else ticker
            elif src == "market":
                tic = "__market__"
            else:
                company = str(rec.get("company", "")).strip()
                tic = company if company else "__unknown__"

            # ── embedding 벡터 ───────────────────────────────────
            emb: List[float] = rec.get("embedding", [])
            if not emb:
                continue
            dim = len(emb)

            row = {
                "date"       : date,
                "tic"        : tic,
                "source_type": src,
                "keyword"    : rec.get("keyword", ""),
                "title"      : rec.get("title", ""),
                "emb_dim"    : dim,
            }
            for i, v in enumerate(emb):
                row[f"emb_{i}"] = float(v)

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # emb_dim 맞추기 (지정된 경우 truncate/pad)
    # NOTE: "emb_dim" 컬럼 자체는 제외하고, emb_0, emb_1, ... 패턴만 선택
    numeric_emb_cols = [c for c in df.columns
                        if c.startswith("emb_") and c != "emb_dim"
                        and c.split("_")[1].isdigit()]
    actual_dim = max(int(c.split("_")[1]) for c in numeric_emb_cols) + 1 if numeric_emb_cols else 0
    target_dim = emb_dim if emb_dim else actual_dim
    if actual_dim > target_dim:
        drop_cols = [f"emb_{i}" for i in range(target_dim, actual_dim)]
        df.drop(columns=drop_cols, inplace=True, errors="ignore")
    elif actual_dim < target_dim:
        for i in range(actual_dim, target_dim):
            df[f"emb_{i}"] = 0.0

    return df


def _normalise_date(raw: str) -> Optional[str]:
    """여러 날짜 포맷을 YYYY-MM-DD 로 통일."""
    raw = raw.strip()
    for fmt in ("%Y.%m.%d", "%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return pd.to_datetime(raw, format=fmt).strftime("%Y-%m-%d")
        except Exception:
            pass
    # pandas 자동 파싱 마지막 시도
    try:
        return pd.to_datetime(raw).strftime("%Y-%m-%d")
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════
# Google Drive JSONL 로더
# ════════════════════════════════════════════════════════════════

class GDriveEmbeddingLoader:
    """
    Google Drive 폴더에서 .jsonl 임베딩 파일을 다운로드해 DataFrame으로 반환.

    Parameters
    ----------
    folder_id : str
        Drive 폴더 ID (URL의 /drive/folders/<id> 부분)
    emb_dim : int
        임베딩 차원 (파일에서 자동 감지하려면 None)
    credentials_path : str, optional
        서비스 계정 JSON 키 파일 경로.
        None이면 ADC(Application Default Credentials) 또는 OAuth2 브라우저 플로우 사용.
    cache_dir : str
        로컬 캐시 디렉토리
    """

    def __init__(
        self,
        folder_id: str,
        emb_dim: Optional[int] = None,
        credentials_path: Optional[str] = None,
        cache_dir: str = ".emb_cache",
    ):
        self.folder_id        = folder_id
        self.emb_dim          = emb_dim
        self.credentials_path = credentials_path
        self.cache_dir        = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._service         = None

    # ─────────────────────────────────────────────────────────────
    def load(self) -> pd.DataFrame:
        """인증 → 파일 목록 → 다운로드 → 파싱 → 반환."""
        self._authenticate()
        files = self._list_jsonl_files()
        if not files:
            raise FileNotFoundError(
                f"Drive 폴더에 .jsonl 파일이 없습니다: {self.folder_id}"
            )
        dfs = []
        for meta in files:
            local = self._download_file(meta)
            df    = _parse_jsonl_to_df(local, self.emb_dim)
            if not df.empty:
                dfs.append(df)
        if not dfs:
            raise ValueError("파싱 가능한 임베딩이 없습니다.")
        result = pd.concat(dfs, ignore_index=True)
        return self._post_process(result)

    # ─────────────────────────────────────────────────────────────
    def _authenticate(self):
        try:
            from googleapiclient.discovery import build
            from google.oauth2 import service_account
            import google.auth
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError as e:
            raise ImportError(
                "Google API 클라이언트 설치 필요:\n"
                "  pip install google-api-python-client google-auth "
                "google-auth-oauthlib google-auth-httplib2"
            ) from e

        SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
        if self.credentials_path and Path(self.credentials_path).exists():
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_file(
                self.credentials_path, scopes=SCOPES
            )
        else:
            try:
                import google.auth
                creds, _ = google.auth.default(scopes=SCOPES)
            except Exception:
                flow  = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
                creds = flow.run_local_server(port=0)
                (self.cache_dir / "token.json").write_text(creds.to_json())

        from googleapiclient.discovery import build
        self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
        print("      Google Drive 인증 완료 ✓")

    def _list_jsonl_files(self) -> list:
        query = (
            f"'{self.folder_id}' in parents "
            "and trashed=false "
            "and name contains '.jsonl'"
        )
        res   = self._service.files().list(
            q=query, fields="files(id, name, size)"
        ).execute()
        files = res.get("files", [])
        print(f"      Drive에서 {len(files)}개 .jsonl 파일 발견")
        return files

    def _download_file(self, meta: dict) -> Path:
        from googleapiclient.http import MediaIoBaseDownload
        local = self.cache_dir / meta["name"]
        if local.exists():
            print(f"      [캐시] {meta['name']}")
            return local
        print(f"      다운로드 중: {meta['name']} ({int(meta.get('size',0))//1024} KB) …")
        req = self._service.files().get_media(fileId=meta["id"])
        buf = io.BytesIO()
        dl  = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
        local.write_bytes(buf.getvalue())
        return local

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        return _post_process_emb_df(df, self.emb_dim)


# ════════════════════════════════════════════════════════════════
# 로컬 JSONL 로더
# ════════════════════════════════════════════════════════════════

class LocalEmbeddingLoader:
    """
    로컬 .jsonl 파일(또는 디렉토리)에서 임베딩을 로드.

    Parameters
    ----------
    path : str
        .jsonl 단일 파일 경로 또는 여러 .jsonl을 담은 디렉토리
    emb_dim : int, optional
        임베딩 차원 (None이면 파일에서 자동 감지)
    """

    def __init__(self, path: str, emb_dim: Optional[int] = None):
        self.path    = Path(path)
        self.emb_dim = emb_dim

    def load(self) -> pd.DataFrame:
        if self.path.is_file():
            files = [self.path]
        elif self.path.is_dir():
            files = sorted(self.path.glob("**/*.jsonl"))
        else:
            raise FileNotFoundError(f"경로를 찾을 수 없습니다: {self.path}")

        if not files:
            raise FileNotFoundError(f".jsonl 파일이 없습니다: {self.path}")

        dfs = []
        for f in files:
            print(f"      로드 중: {f.name}")
            df = _parse_jsonl_to_df(f, self.emb_dim)
            if not df.empty:
                dfs.append(df)

        if not dfs:
            raise ValueError("파싱 가능한 임베딩 행이 없습니다.")

        result = pd.concat(dfs, ignore_index=True)
        return _post_process_emb_df(result, self.emb_dim)


# ════════════════════════════════════════════════════════════════
# 공통 후처리
# ════════════════════════════════════════════════════════════════

def _post_process_emb_df(df: pd.DataFrame, emb_dim: Optional[int]) -> pd.DataFrame:
    """
    중복 제거, 정렬, emb_* 컬럼 채우기.

    source_type=="market" 행은 날짜별로 하나만 유지 (마지막 행 우선).
    source_type=="company" 행은 (date, tic) 기준으로 평균 집계.
    두 집합을 합쳐서 반환 — StateBuilder에서 merge 방식 결정.
    """
    emb_cols = sorted(
        [c for c in df.columns
         if c.startswith("emb_") and c != "emb_dim" and c.split("_")[1].isdigit()],
        key=lambda x: int(x.split("_")[1])
    )
    # emb_dim 은 메타 컬럼이므로 제거
    df.drop(columns=["emb_dim"], inplace=True, errors="ignore")
    if emb_dim:
        # 필요한 dim이 명시된 경우 최종 확인
        current = len(emb_cols)
        if current < emb_dim:
            for i in range(current, emb_dim):
                df[f"emb_{i}"] = 0.0
                emb_cols.append(f"emb_{i}")
        elif current > emb_dim:
            drop = [c for c in emb_cols if int(c.split("_")[1]) >= emb_dim]
            df.drop(columns=drop, inplace=True, errors="ignore")
            emb_cols = emb_cols[:emb_dim]

    df[emb_cols] = df[emb_cols].fillna(0.0)
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)

    actual_dim = len(emb_cols)
    n_dates = df["date"].nunique()
    n_tickers = df["tic"].nunique()
    print(f"      임베딩 로드 완료: {len(df):,}행 | 날짜 {n_dates}개 | "
          f"tic {n_tickers}개 | dim={actual_dim}")
    return df


# ════════════════════════════════════════════════════════════════
# 테스트용 더미 생성
# ════════════════════════════════════════════════════════════════

def make_dummy_embeddings(
    dates: list,
    tickers: list,
    emb_dim: int = 16,
    seed: int = 42,
    include_market: bool = True,
) -> pd.DataFrame:
    """
    실제 JSONL 스키마와 동일한 구조의 더미 DataFrame 생성.
    (단위 테스트 / dry-run 전용)
    """
    rng  = np.random.default_rng(seed)
    rows = []

    for d in dates:
        date_str = pd.to_datetime(str(d)).strftime("%Y-%m-%d")

        # 시장 전체 임베딩 (source_type=market)
        if include_market:
            emb = rng.standard_normal(emb_dim).astype(np.float32)
            row = {"date": date_str, "tic": "__market__",
                   "source_type": "market", "keyword": "dummy_market", "title": ""}
            for i, v in enumerate(emb):
                row[f"emb_{i}"] = float(v)
            rows.append(row)

        # 종목별 임베딩 (source_type=company)
        for tic in tickers:
            emb = rng.standard_normal(emb_dim).astype(np.float32)
            row = {"date": date_str, "tic": tic,
                   "source_type": "company", "keyword": "dummy_company", "title": ""}
            for i, v in enumerate(emb):
                row[f"emb_{i}"] = float(v)
            rows.append(row)

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
# Ticker 유틸
# ════════════════════════════════════════════════════════════════

def get_tickers_from_jsonl(path: str) -> List[str]:
    """
    JSONL 임베딩 파일(또는 디렉토리)에서 종목 ticker 목록을 추출합니다.

    - source_type == "stock" / "company" 행에서 ticker 필드를 읽습니다.
    - 6자리 숫자 코드는 Yahoo Finance 형식으로 변환합니다 (예: "005930" → "005930.KS").
    - 중복 제거 후 정렬해서 반환합니다.
    """
    p = Path(path)
    files = [p] if p.is_file() else sorted(p.glob("**/*.jsonl"))

    tickers: set = set()
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                src = str(rec.get("source_type", "")).strip()
                if src not in ("stock", "company"):
                    continue
                ticker = str(rec.get("ticker", "")).strip()
                if not ticker:
                    continue
                tic = ticker + ".KS" if ticker.isdigit() else ticker
                tickers.add(tic)

    return sorted(tickers)
