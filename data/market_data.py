"""
data/market_data.py
────────────────────────────────────────────────────────────────
KOSPI Market Data Loader
  • Downloads OHLCV from Yahoo Finance (yfinance)
  • Adds standard FinRL technical indicators via stockstats
  • Returns a clean DataFrame compatible with FinRL conventions
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Optional

warnings.filterwarnings("ignore")


# ────────────────────────────────────────────────────────────────
# Default technical indicators used in FinRL
# ────────────────────────────────────────────────────────────────
DEFAULT_TECH_INDICATORS = [
    "macd",          # MACD
    "boll_ub",       # Bollinger Upper Band
    "boll_lb",       # Bollinger Lower Band
    "rsi_30",        # RSI (30-period)
    "cci_30",        # CCI (30-period)
    "dx_30",         # Directional Movement Index
    "close_30_sma",  # 30-day SMA
    "close_60_sma",  # 60-day SMA
    "volume_change", # Volume change rate
    "return_1d",     # 1-day return
    "return_5d",     # 5-day return
]


class KOSPIDataLoader:
    """
    Download and pre-process KOSPI stock data from Yahoo Finance.

    Parameters
    ----------
    tickers : list of str
        Yahoo Finance tickers, e.g. ["005930.KS", "000660.KS"]
    start, end : str
        Date range in 'YYYY-MM-DD' format
    tech_indicators : list of str, optional
        Technical indicator names (stockstats convention)
    """

    def __init__(
        self,
        tickers: List[str],
        start: str,
        end: str,
        tech_indicators: Optional[List[str]] = None,
    ):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.tech_indicators = tech_indicators or DEFAULT_TECH_INDICATORS

    # ─────────────────────────────────────────────────────────────
    # Public
    # ─────────────────────────────────────────────────────────────
    def fetch_and_process(self) -> pd.DataFrame:
        """Download raw OHLCV then compute indicators. Returns a long-format DataFrame."""
        raw = self._download()
        processed = self._add_tech_indicators(raw)
        processed = self._clean(processed)
        return processed

    # ─────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────
    def _download(self) -> pd.DataFrame:
        """Download multi-ticker OHLCV from Yahoo Finance and reshape to long format."""
        print(f"      Downloading {len(self.tickers)} tickers: {self.start} → {self.end}")

        raw = yf.download(
            tickers=self.tickers,
            start=self.start,
            end=self.end,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )

        # ── Reshape wide → long ──────────────────────────────────
        frames = []
        for tic in self.tickers:
            if len(self.tickers) == 1:
                df_tic = raw.copy()
            else:
                df_tic = raw[tic].copy()

            # 최신 yfinance는 MultiIndex 컬럼 반환 → 첫 번째 레벨로 flatten
            if isinstance(df_tic.columns, pd.MultiIndex):
                df_tic.columns = df_tic.columns.get_level_values(0)

            # 컬럼명 대소문자 정규화 후 dropna
            df_tic.columns = [c.capitalize() for c in df_tic.columns]
            df_tic = df_tic.dropna(subset=["Close"])
            df_tic.columns = [c.lower() for c in df_tic.columns]
            df_tic["tic"] = tic
            df_tic.index.name = "date"
            df_tic = df_tic.reset_index()
            df_tic["date"] = pd.to_datetime(df_tic["date"]).dt.date.astype(str)
            frames.append(df_tic)

        df = pd.concat(frames, ignore_index=True)
        # Standardise column names to FinRL convention
        df = df.rename(columns={"open":"open","high":"high","low":"low",
                                  "close":"close","volume":"volume"})
        return df

    def _add_tech_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicators using stockstats.
        Falls back to manual computation if stockstats is unavailable.
        """
        try:
            from stockstats import StockDataFrame
            frames = []
            for tic, g in df.groupby("tic"):
                g = g.sort_values("date").copy()
                sdf = StockDataFrame.retype(g.set_index("date"))
                for ind in self.tech_indicators:
                    # Skip manually computed ones
                    if ind in ("volume_change", "return_1d", "return_5d"):
                        continue
                    try:
                        _ = sdf[ind]  # triggers computation
                    except Exception:
                        pass
                g2 = sdf.reset_index()
                g2["tic"] = tic
                frames.append(g2)
            df = pd.concat(frames, ignore_index=True)
        except ImportError:
            print("      [warn] stockstats not installed; using manual indicators")
            df = self._manual_indicators(df)

        # Manual indicators always computed on top
        df = self._manual_indicators(df)
        return df

    def _manual_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume_change, return_1d, return_5d per ticker."""
        frames = []
        for tic, g in df.groupby("tic"):
            g = g.sort_values("date").copy()
            g["volume_change"] = g["volume"].pct_change().fillna(0)
            g["return_1d"]     = g["close"].pct_change(1).fillna(0)
            g["return_5d"]     = g["close"].pct_change(5).fillna(0)
            frames.append(g)
        return pd.concat(frames, ignore_index=True)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop NaN rows, clip extremes, sort by date then tic."""
        # Replace inf with NaN then forward-fill
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.sort_values(["date", "tic"])
        tech_cols = [c for c in self.tech_indicators if c in df.columns]
        df[tech_cols] = df[tech_cols].groupby(
            df["tic"], group_keys=False
        ).apply(lambda x: x.fillna(method="ffill").fillna(0))
        df = df.dropna(subset=["close"])
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df


# ────────────────────────────────────────────────────────────────
# Quick test
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = KOSPIDataLoader(
        tickers=["005930.KS", "000660.KS"],
        start="2023-01-01",
        end="2023-06-30",
    )
    df = loader.fetch_and_process()
    print(df.head())
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)
