"""
rl/env_kospi.py
────────────────────────────────────────────────────────────────
Custom Gym Environment for KOSPI Multi-Stock Trading
  • Compatible with FinRL's StockTradingEnv interface
  • State = [balance, shares, prices, tech_indicators, embeddings]
  • Action = continuous ∈ [-1, +1]^n_stocks
             negative → sell, positive → buy, |x| ≥ threshold → hold
  • Reward = log portfolio value change (Sharpe-adjusted optional)
  • Korean market specifics: 0.3 % transaction cost, KRW denominated
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


# ────────────────────────────────────────────────────────────────
# Constants — Korean market
# ────────────────────────────────────────────────────────────────
KRW_TRANSACTION_COST = 0.003   # 0.3 % total (commission + tax)
HOLD_THRESHOLD       = 0.10    # |action| < threshold → do nothing
MAX_SHARES_PER_STOCK = 500     # position size cap per stock


class KOSPITradingEnv(gym.Env):
    """
    Multi-stock KOSPI trading environment with NLP embedding state.

    Observation space:
        Box(state_dim,) — normalised float32 vector
        [balance_ratio,                  # 1
         shares_ratio × n_stocks,        # n_stocks
         price_ratio  × n_stocks,        # n_stocks
         tech_indicators × n_stocks,     # n_stocks × n_tech
         embedding_t]                    # emb_dim

    Action space:
        Box(-1, 1, shape=(n_stocks,))
        Discretised to buy / hold / sell with variable lot size.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-processed market + embedding DataFrame (one split).
    tech_indicators : list of str
        Column names of technical indicators.
    emb_dim : int
        Number of embedding dimensions in state.
    initial_capital : float
        Starting portfolio value in KRW.
    max_shares : int
        Maximum shares per action per stock.
    reward_scaling : float
        Multiplier for reward signal.
    sharpe_window : int
        Look-back window for Sharpe-adjusted reward (0 = disable).
    turbulence_threshold : float
        If turbulence index exceeds this, force liquidation.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        tech_indicators: List[str],
        co_emb_dim: int = 0,
        mk_emb_dim: int = 0,
        initial_capital: float = 10_000_000,
        max_shares: int = MAX_SHARES_PER_STOCK,
        reward_scaling: float = 1e-4,
        sharpe_window: int = 20,
        turbulence_threshold: float = 140.0,
        hold_threshold: float = HOLD_THRESHOLD,
    ):
        super().__init__()

        self.df               = df.sort_values(["date", "tic"]).reset_index(drop=True)
        self.tech_indicators  = tech_indicators
        self.co_emb_dim       = co_emb_dim
        self.mk_emb_dim       = mk_emb_dim
        self.initial_capital  = initial_capital
        self.max_shares       = max_shares
        self.reward_scaling   = reward_scaling
        self.sharpe_window    = sharpe_window
        self.hold_threshold   = hold_threshold
        self.turbulence_threshold = turbulence_threshold

        # ── Derived constants ────────────────────────────────────
        self.tickers       = sorted(self.df["tic"].unique().tolist())
        self.n_stocks      = len(self.tickers)
        self.dates         = sorted(self.df["date"].unique().tolist())
        self.n_tech        = len(tech_indicators)
        self.co_emb_cols   = [f"co_emb_{i}" for i in range(co_emb_dim)]
        self.mk_emb_cols   = [f"mk_emb_{i}" for i in range(mk_emb_dim)]

        # state = 1 + S + S + S×n_tech + S×co_emb_dim + mk_emb_dim
        self.state_dim = (
            1
            + self.n_stocks
            + self.n_stocks
            + self.n_stocks * self.n_tech
            + self.n_stocks * co_emb_dim
            + mk_emb_dim
        )

        # ── Gym spaces ───────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1, high=1,
            shape=(self.n_stocks,), dtype=np.float32,
        )

        # ── Episode state ────────────────────────────────────────
        self._day_idx        = 0
        self._balance        = initial_capital
        self._shares_held    = np.zeros(self.n_stocks, dtype=np.float32)
        self._portfolio_vals: List[float] = []
        self._returns_buf   : List[float] = []

        # Pre-build date→row lookup
        self._date_data: Dict[str, pd.DataFrame] = {
            d: g.set_index("tic").reindex(self.tickers)
            for d, g in self.df.groupby("date")
        }

    # ─────────────────────────────────────────────────────────────
    # Gym API
    # ─────────────────────────────────────────────────────────────
    def reset(
        self, *, seed=None, options=None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._day_idx     = 0
        self._balance     = self.initial_capital
        self._shares_held = np.zeros(self.n_stocks, dtype=np.float32)
        self._portfolio_vals = []
        self._returns_buf    = []
        obs = self._get_observation()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, -1, 1)

        # Turbulence check → liquidate if needed
        turb = self._turbulence()
        if turb > self.turbulence_threshold:
            action = np.full(self.n_stocks, -1.0)  # force sell all

        # ── Execute trades ───────────────────────────────────────
        prices = self._current_prices()
        self._execute_actions(action, prices)

        # ── Advance day ──────────────────────────────────────────
        self._day_idx += 1
        done = self._day_idx >= len(self.dates) - 1

        # ── Portfolio value & reward ─────────────────────────────
        new_prices = self._current_prices()
        portfolio  = self._balance + float(
            np.dot(self._shares_held, new_prices)
        )
        self._portfolio_vals.append(portfolio)

        if len(self._portfolio_vals) >= 2:
            prev = self._portfolio_vals[-2]
            ret  = (portfolio - prev) / (prev + 1e-8)
            self._returns_buf.append(ret)
            reward = self._compute_reward(ret)
        else:
            reward = 0.0

        obs  = self._get_observation()
        info = {
            "portfolio_value"   : portfolio,
            "balance"           : self._balance,
            "shares_held"       : self._shares_held.tolist(),
            "day"               : self.dates[self._day_idx] if not done else "done",
            "turbulence"        : turb,
        }
        return obs, reward * self.reward_scaling, done, False, info

    def render(self, mode="human"):
        if len(self._portfolio_vals):
            pv = self._portfolio_vals[-1]
            ret = (pv - self.initial_capital) / self.initial_capital * 100
            print(f"  Day {self._day_idx:4d} | Portfolio: {pv:,.0f} KRW "
                  f"({ret:+.2f}%) | Balance: {self._balance:,.0f}")

    # ─────────────────────────────────────────────────────────────
    # State construction
    # ─────────────────────────────────────────────────────────────
    def _get_observation(self) -> np.ndarray:
        date   = self.dates[self._day_idx]
        data   = self._date_data[date]
        prices = data["close"].values.astype(np.float32)

        # Balance ratio
        portfolio = self._balance + float(np.dot(self._shares_held, prices))
        bal_ratio = np.array([self._balance / (portfolio + 1e-8)], dtype=np.float32)

        # Shares (normalised)
        shares_norm = self._shares_held / (self.max_shares + 1e-8)

        # Price (normalised by initial_capital proxy)
        price_norm = prices / (self.initial_capital / self.n_stocks + 1e-8)

        # Tech indicators (n_stocks × n_tech → flatten)
        tech = np.zeros((self.n_stocks, self.n_tech), dtype=np.float32)
        for j, col in enumerate(self.tech_indicators):
            if col in data.columns:
                tech[:, j] = data[col].fillna(0).values.astype(np.float32)
        tech_flat = tech.flatten()

        # Company embedding: 종목별로 개별 읽기 (n_stocks × co_emb_dim → flatten)
        co_emb = np.zeros((self.n_stocks, self.co_emb_dim), dtype=np.float32)
        for j, col in enumerate(self.co_emb_cols):
            if col in data.columns:
                co_emb[:, j] = data[col].fillna(0).values.astype(np.float32)
        co_emb_flat = co_emb.flatten()

        # Market embedding: 날짜 공통 벡터 (첫 번째 행 값 사용)
        mk_emb = np.zeros(self.mk_emb_dim, dtype=np.float32)
        for j, col in enumerate(self.mk_emb_cols):
            if col in data.columns:
                mk_emb[j] = float(data[col].fillna(0).iloc[0])

        obs = np.concatenate([bal_ratio, shares_norm, price_norm, tech_flat, co_emb_flat, mk_emb])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs.astype(np.float32)

    # ─────────────────────────────────────────────────────────────
    # Action execution
    # ─────────────────────────────────────────────────────────────
    def _execute_actions(
        self, actions: np.ndarray, prices: np.ndarray
    ):
        """Map continuous actions → discrete buy/sell/hold orders."""
        # Sort: sell first to free up capital, then buy
        sell_idx = np.where(actions < -self.hold_threshold)[0]
        buy_idx  = np.where(actions >  self.hold_threshold)[0]

        for i in sell_idx:
            if prices[i] <= 0 or self._shares_held[i] <= 0:
                continue
            n_sell = int(abs(actions[i]) * self.max_shares)
            n_sell = min(n_sell, int(self._shares_held[i]))
            proceeds = n_sell * prices[i] * (1 - KRW_TRANSACTION_COST)
            self._shares_held[i] -= n_sell
            self._balance         += proceeds

        for i in buy_idx:
            if prices[i] <= 0:
                continue
            n_buy = int(abs(actions[i]) * self.max_shares)
            # max_shares 총 보유량 초과 방지
            n_buy = min(n_buy, max(0, self.max_shares - int(self._shares_held[i])))
            cost  = n_buy * prices[i] * (1 + KRW_TRANSACTION_COST)
            if cost > self._balance:
                n_buy = int(self._balance / (prices[i] * (1 + KRW_TRANSACTION_COST)))
                cost  = n_buy * prices[i] * (1 + KRW_TRANSACTION_COST)
            if n_buy > 0:
                self._shares_held[i] += n_buy
                self._balance         -= cost

    # ─────────────────────────────────────────────────────────────
    # Reward
    # ─────────────────────────────────────────────────────────────
    def _compute_reward(self, daily_return: float) -> float:
        """
        Sharpe-adjusted reward when window is active, else raw return.
        """
        if self.sharpe_window > 0 and len(self._returns_buf) >= self.sharpe_window:
            r = np.array(self._returns_buf[-self.sharpe_window:])
            std = r.std() + 1e-8
            sharpe = r.mean() / std
            return float(sharpe)
        return daily_return

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────
    def _current_prices(self) -> np.ndarray:
        date = self.dates[min(self._day_idx, len(self.dates)-1)]
        return self._date_data[date]["close"].fillna(0).values.astype(np.float32)

    def _turbulence(self) -> float:
        """Compute a simple turbulence proxy (rolling covariance trace)."""
        if self._day_idx < 10:
            return 0.0
        start = max(0, self._day_idx - 30)
        slc   = self.dates[start:self._day_idx+1]
        rets  = []
        for d in slc:
            r = self._date_data[d]["return_1d"].fillna(0).values if "return_1d" in self._date_data[d].columns else np.zeros(self.n_stocks)
            rets.append(r.astype(float))
        arr = np.array(rets)
        if arr.shape[0] < 2:
            return 0.0
        cov = np.cov(arr.T)
        return float(np.trace(cov)) if cov.ndim == 2 else float(cov)
