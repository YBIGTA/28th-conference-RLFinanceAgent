"""
data/state_builder.py
────────────────────────────────────────────────────────────────
StateBuilder — KOSPI 시장 데이터 + JSONL 텍스트 임베딩 결합

임베딩 스키마(실제 데이터 기준):
  source_type = "market"          → tic = "__market__"  (날짜당 1개 벡터)
  source_type = "company"/"stock" → tic = 종목코드       (종목×날짜 벡터)

결합 전략:
  1. company 임베딩 → market DataFrame 의 (date, tic) 에 left-join
  2. market  임베딩 → 모든 종목에 동일하게 broadcast
  3. 두 임베딩을 concat → 최종 state 에 포함
  4. 빈 날짜는 직전 영업일 값으로 forward-fill

State 벡터 (per step):
  [balance_ratio(1)]
  + [shares_ratio × n_stocks]
  + [price_ratio  × n_stocks]
  + [tech_indicators × n_stocks × n_tech]
  + [company_emb × n_stocks × emb_dim]   (있을 경우)
  + [market_emb  × emb_dim]              (있을 경우)
"""

from __future__ import annotations
from typing import Optional, Tuple, List
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


TECH_INDICATORS = [
    "macd", "boll_ub", "boll_lb",
    "rsi_30", "cci_30", "dx_30",
    "close_30_sma", "close_60_sma",
    "volume_change", "return_1d", "return_5d",
]


class StateBuilder:
    """
    Parameters
    ----------
    df_market : pd.DataFrame
        KOSPIDataLoader 출력 (OHLCV + 기술지표, 컬럼: date, tic, close, …)
    df_embeddings : pd.DataFrame or None
        GDriveEmbeddingLoader / LocalEmbeddingLoader 출력
        컬럼: date, tic, source_type, emb_0 … emb_{D-1}
    emb_dim : int
        임베딩 차원 (자동 감지하려면 None → df_embeddings 에서 읽음)
    train_end : str
        학습/테스트 분리 기준 날짜 'YYYY-MM-DD'
    use_company_emb : bool
        source_type=="company" 임베딩 사용 여부
    use_market_emb  : bool
        source_type=="market"  임베딩 사용 여부
    scale_market : bool
        기술 지표 StandardScaler 적용 여부
    """

    def __init__(
        self,
        df_market: pd.DataFrame,
        df_embeddings: Optional[pd.DataFrame],
        emb_dim: Optional[int] = None,
        train_end: str = "2023-01-01",
        use_company_emb: bool = True,
        use_market_emb: bool = True,
        scale_market: bool = True,
    ):
        self.df_market       = df_market.copy()
        self.df_embeddings   = df_embeddings
        self.train_end       = train_end
        self.use_company_emb = use_company_emb
        self.use_market_emb  = use_market_emb
        self.scale_market    = scale_market

        # 임베딩 차원 자동 감지
        if emb_dim is not None:
            self.emb_dim = emb_dim
        elif df_embeddings is not None:
            emb_cols = [c for c in df_embeddings.columns if c.startswith("emb_")]
            self.emb_dim = len(emb_cols)
        else:
            self.emb_dim = 0

        self._scaler: Optional[StandardScaler] = None
        self._state_dim: Optional[int] = None
        self._tech_cols: List[str] = []
        self.co_emb_dim: int = 0
        self.mk_emb_dim: int = 0

    # ─────────────────────────────────────────────────────────────
    @property
    def state_dim(self) -> int:
        if self._state_dim is None:
            raise RuntimeError("build_train_test() 먼저 호출하세요")
        return self._state_dim

    def build_train_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """전처리 → 스케일링 → 학습/테스트 분리."""
        df = self._merge_embeddings()
        df = self._ensure_tech_cols(df)
        df = self._scale_features(df)
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        self._compute_state_dim(df)

        df_train = df[df["date"] <  self.train_end].copy()
        df_test  = df[df["date"] >= self.train_end].copy()
        return df_train, df_test

    # ─────────────────────────────────────────────────────────────
    # 임베딩 결합
    # ─────────────────────────────────────────────────────────────
    def _merge_embeddings(self) -> pd.DataFrame:
        df  = self.df_market.copy()
        emb = self.df_embeddings

        emb_cols = [f"emb_{i}" for i in range(self.emb_dim)]

        if emb is None or self.emb_dim == 0:
            # 임베딩 없음 → 0으로 채움
            for c in emb_cols:
                df[c] = 0.0
            return df

        # ── source_type 분리 ─────────────────────────────────────
        market_emb  = emb[emb["source_type"] == "market"].copy()  if self.use_market_emb  else pd.DataFrame()
        company_emb = emb[emb["source_type"] == "company"].copy() if self.use_company_emb else pd.DataFrame()

        # ── 1) company 임베딩: (date, tic) join ──────────────────
        if not company_emb.empty:
            # 같은 (date, tic) 에 여러 뉴스 → 평균 집계
            company_agg = (
                company_emb
                .groupby(["date", "tic"])[emb_cols]
                .mean()
                .reset_index()
            )
            co_cols = {c: f"co_{c}" for c in emb_cols}
            company_agg = company_agg.rename(columns=co_cols)
            df = pd.merge(df, company_agg, on=["date", "tic"], how="left")
            co_emb_cols = list(co_cols.values())
            # forward-fill 빈 날짜
            df = df.sort_values(["tic", "date"])
            df[co_emb_cols] = (
                df.groupby("tic")[co_emb_cols]
                .transform(lambda x: x.ffill().fillna(0.0))
            )
        else:
            co_emb_cols = []

        # ── 2) market 임베딩: 날짜별로 broadcast ─────────────────
        if not market_emb.empty:
            # 같은 날짜에 여러 기사 → 평균
            market_agg = (
                market_emb
                .groupby("date")[emb_cols]
                .mean()
                .reset_index()
            )
            mk_cols = {c: f"mk_{c}" for c in emb_cols}
            market_agg = market_agg.rename(columns=mk_cols)
            df = pd.merge(df, market_agg, on="date", how="left")
            mk_emb_cols = list(mk_cols.values())
            df = df.sort_values(["tic", "date"])
            df[mk_emb_cols] = (
                df.groupby("tic")[mk_emb_cols]
                .transform(lambda x: x.ffill().fillna(0.0))
            )
        else:
            mk_emb_cols = []

        # ── 3) co_emb_* / mk_emb_* 를 그대로 유지 ───────────────
        # env가 종목별로 co_emb_* 를 읽고, mk_emb_* 를 broadcast로 읽음
        self.co_emb_dim = len(co_emb_cols) if co_emb_cols else 0
        self.mk_emb_dim = len(mk_emb_cols) if mk_emb_cols else 0
        self.emb_dim    = self.co_emb_dim + self.mk_emb_dim  # 총합 (메트릭용)

        return df

    # ─────────────────────────────────────────────────────────────
    def _ensure_tech_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        self._tech_cols = []
        for col in TECH_INDICATORS:
            if col not in df.columns:
                df[col] = 0.0
            self._tech_cols.append(col)
        return df

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.scale_market:
            return df
        # close는 env가 자체 정규화(price_ratio)하므로 스케일 제외
        targets = [c for c in self._tech_cols + ["volume"] if c in df.columns]
        train_mask = df["date"] < self.train_end
        self._scaler = StandardScaler()
        self._scaler.fit(df.loc[train_mask, targets].fillna(0))
        df[targets] = self._scaler.transform(df[targets].fillna(0))
        return df

    def _compute_state_dim(self, df: pd.DataFrame):
        n_stocks = df["tic"].nunique()
        n_tech   = len(self._tech_cols)
        # state = balance(1) + shares(S) + price(S) + tech(S×T)
        #       + co_emb(S×co_emb_dim) + mk_emb(mk_emb_dim)
        self._state_dim = (
            1
            + n_stocks
            + n_stocks
            + n_stocks * n_tech
            + n_stocks * self.co_emb_dim
            + self.mk_emb_dim
        )

        print(f"\n      ┌─ State 구성 ──────────────────────────────────────┐")
        print(f"      │  balance ratio              :      1")
        print(f"      │  shares        (S={n_stocks:<3})      : {n_stocks:>6}")
        print(f"      │  prices        (S={n_stocks:<3})      : {n_stocks:>6}")
        print(f"      │  tech          (S×{n_tech})     : {n_stocks*n_tech:>6}")
        print(f"      │  co_emb (per-stock S×{self.co_emb_dim:<2})  : {n_stocks*self.co_emb_dim:>6}")
        print(f"      │  mk_emb (broadcast)         : {self.mk_emb_dim:>6}")
        print(f"      │  ─────────────────────────────────────────────────")
        print(f"      │  total state_dim            : {self._state_dim:>6}")
        print(f"      └───────────────────────────────────────────────────")
