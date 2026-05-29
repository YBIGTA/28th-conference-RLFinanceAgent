"""
rl/backtest.py
────────────────────────────────────────────────────────────────
Traditional strategy backtesting for comparison with RL agent.

  MinVarStrategy  — rolling covariance minimum-variance portfolio
  Backtester      — runs strategies + generates comparison report
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

TRANSACTION_COST = 0.003


class MinVarStrategy:
    """
    Minimum Variance portfolio strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-processed market DataFrame (date, tic, close, ...)
    initial_capital : float
    lookback : int
        Trading days used to estimate the covariance matrix
    rebalance_every : int
        Rebalancing frequency in trading days
    cost : float
        One-way transaction cost rate
    """

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 10_000_000,
        lookback: int = 60,
        rebalance_every: int = 20,
        cost: float = TRANSACTION_COST,
    ):
        self.initial_capital  = initial_capital
        self.lookback         = lookback
        self.rebalance_every  = rebalance_every
        self.cost             = cost

        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        self.tickers = sorted(df["tic"].unique().tolist())
        self.n       = len(self.tickers)
        self.dates   = sorted(df["date"].unique().tolist())

        self._price_pivot = (
            df.pivot(index="date", columns="tic", values="close")
            .reindex(columns=self.tickers)
            .ffill()
            .fillna(0)
        )
        self._ret_pivot = self._price_pivot.pct_change().fillna(0)

    # ─────────────────────────────────────────────────────────────
    def run(self) -> Dict:
        balance = self.initial_capital
        shares  = np.zeros(self.n)

        # Equal-weight initial buy at day 0 (유효 종목만)
        prices_0 = self._prices_at(0)
        valid_0  = prices_0 > 0
        n_valid  = valid_0.sum()
        if n_valid == 0:
            raise ValueError("day 0에 유효한 가격 데이터가 없습니다.")
        w0 = valid_0.astype(float) / n_valid
        shares   = np.where(valid_0, w0 * self.initial_capital / prices_0, 0.0)
        balance  = self.initial_capital - float(np.dot(shares, prices_0)) * (1 + self.cost)

        portfolio_vals: List[float] = []

        for i, date in enumerate(self.dates):
            prices = self._prices_at(i)

            # MinVar rebalance once we have enough history
            if i > 0 and i % self.rebalance_every == 0 and i >= self.lookback:
                ret_window = self._ret_pivot.iloc[i - self.lookback : i].values
                weights    = self._solve_min_var(ret_window)
                balance, shares = self._rebalance(balance, shares, prices, weights)

            portfolio_vals.append(balance + float(np.dot(shares, prices)))

        metrics = _compute_metrics(portfolio_vals)
        metrics["strategy"] = "MinVar"
        _print_results("MinVar", metrics)

        return {"portfolio_vals": portfolio_vals, "dates": self.dates, "metrics": metrics}

    # ─────────────────────────────────────────────────────────────
    def _prices_at(self, idx: int) -> np.ndarray:
        p = self._price_pivot.iloc[idx].values.astype(float)
        return np.where(np.isfinite(p) & (p > 0), p, 0.0)  # 0 = 거래 불가

    def _rebalance(
        self,
        balance: float,
        shares: np.ndarray,
        prices: np.ndarray,
        weights: np.ndarray,
    ):
        valid = prices > 0
        portfolio_val = balance + float(np.dot(shares * valid, prices))

        # 유효 종목에만 비중 배분, 재정규화
        w = weights * valid
        w_sum = w.sum()
        w = w / w_sum if w_sum > 1e-8 else (valid.astype(float) / valid.sum())

        target_shares = np.where(valid, w * portfolio_val / prices, 0.0)
        delta = target_shares - shares

        # 가격이 있는 종목만 거래 비용 계산
        trade_cost = float(np.dot(np.abs(delta) * valid, prices)) * self.cost
        balance -= float(np.dot(delta * valid, prices)) + trade_cost
        return balance, target_shares

    def _solve_min_var(self, returns: np.ndarray) -> np.ndarray:
        """Solve min w^T Σ w  s.t. sum(w)=1, w≥0. 유효 종목만 사용."""
        # 분산이 0인 열(가격 변동 없음) 제외
        valid_cols = returns.std(axis=0) > 1e-8
        n_valid = valid_cols.sum()
        w_full = np.zeros(self.n)
        if n_valid == 0:
            w_full[:] = 1.0 / self.n
            return w_full

        r = returns[:, valid_cols]
        cov = np.cov(r.T) + np.eye(n_valid) * 1e-8
        w0  = np.ones(n_valid) / n_valid

        result = minimize(
            fun=lambda w: w @ cov @ w,
            x0=w0,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_valid,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        w_valid = result.x if result.success else w0
        w_full[valid_cols] = w_valid
        return w_full


# ════════════════════════════════════════════════════════════════
# Backtester
# ════════════════════════════════════════════════════════════════

class Backtester:
    """
    Compare RL agent results against traditional strategies.

    Parameters
    ----------
    df_test : pd.DataFrame
        Test-split market DataFrame (same as used for RL evaluation)
    initial_capital : float
    output_dir : str
    """

    def __init__(
        self,
        df_test: pd.DataFrame,
        initial_capital: float = 10_000_000,
        output_dir: str = "results/",
    ):
        self.df_test         = df_test
        self.initial_capital = initial_capital
        self.output_dir      = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_minvar(
        self,
        lookback: int = 60,
        rebalance_every: int = 20,
    ) -> Dict:
        print("\n[Backtest] MinVar 전략 실행 …")
        strategy = MinVarStrategy(
            df=self.df_test,
            initial_capital=self.initial_capital,
            lookback=lookback,
            rebalance_every=rebalance_every,
        )
        return strategy.run()

    def compare(self, results: Dict[str, Dict]) -> None:
        """
        results: {
            "RL (PPO)": {"portfolio_vals": [...], "dates": [...], "metrics": {...}},
            "MinVar":   {"portfolio_vals": [...], "dates": [...], "metrics": {...}},
        }
        """
        self._print_comparison_table(results)
        self._plot_comparison(results)
        self._save_metrics(results)

    # ─────────────────────────────────────────────────────────────
    def _plot_comparison(self, results: Dict[str, Dict]) -> None:
        colors = ["#0ea5e9", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"]
        fig, axes = plt.subplots(2, 1, figsize=(14, 9))
        ax1, ax2  = axes

        for (name, res), color in zip(results.items(), colors):
            pvs     = np.array(res["portfolio_vals"])
            ret_pct = (pvs - self.initial_capital) / self.initial_capital * 100
            ax1.plot(ret_pct, label=name, color=color, linewidth=1.5)

            running_max = np.maximum.accumulate(pvs)
            dd = (pvs - running_max) / (running_max + 1e-8) * 100
            ax2.plot(dd, label=name, color=color, linewidth=1.2, alpha=0.85)

        ax1.axhline(0, color="#94a3b8", linestyle="--", linewidth=1, label="Break-even")
        ax1.set_title("Portfolio Return Comparison (%)", fontsize=13)
        ax1.set_ylabel("Return (%)")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}%"))
        ax1.legend(); ax1.grid(alpha=0.3)

        ax2.axhline(0, color="#94a3b8", linestyle="--", linewidth=0.8)
        ax2.set_title("Drawdown (%)", fontsize=13)
        ax2.set_ylabel("Drawdown (%)")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}%"))
        ax2.legend(); ax2.grid(alpha=0.3)

        plt.tight_layout()
        path = self.output_dir / "comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"      Comparison plot saved → {path}")

    def _save_metrics(self, results: Dict[str, Dict]) -> None:
        summary = {name: res["metrics"] for name, res in results.items()}
        path = self.output_dir / "comparison_metrics.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"      Metrics saved       → {path}")

    def _print_comparison_table(self, results: Dict[str, Dict]) -> None:
        metric_keys = [
            "total_return", "annualised_return", "sharpe_ratio",
            "max_drawdown", "calmar_ratio", "win_rate",
        ]
        names = list(results.keys())
        col_w = 14
        sep   = "  " + "─" * (24 + col_w * len(names))

        print(f"\n{sep}")
        header = f"  {'Metric':<22}" + "".join(f"{n:>{col_w}}" for n in names)
        print(header)
        print(sep)
        for k in metric_keys:
            row = f"  {k:<22}"
            for name in names:
                v = results[name]["metrics"].get(k, float("nan"))
                row += f"{v:>{col_w}.4f}"
            print(row)
        print(sep)


# ════════════════════════════════════════════════════════════════
# Shared helpers
# ════════════════════════════════════════════════════════════════

def _compute_metrics(portfolio_vals: list) -> dict:
    pvs  = np.array(portfolio_vals, dtype=float)
    rets = np.diff(pvs) / (pvs[:-1] + 1e-8)

    total_ret   = (pvs[-1] - pvs[0]) / pvs[0]
    annualised  = (1 + total_ret) ** (252 / max(len(pvs), 1)) - 1
    sharpe      = (rets.mean() / (rets.std() + 1e-8)) * np.sqrt(252)
    running_max = np.maximum.accumulate(pvs)
    dd          = (pvs - running_max) / (running_max + 1e-8)
    max_dd      = float(dd.min())
    calmar      = annualised / (abs(max_dd) + 1e-8)
    win_rate    = float((rets > 0).mean())

    return {
        "total_return"     : float(total_ret),
        "annualised_return": float(annualised),
        "sharpe_ratio"     : float(sharpe),
        "max_drawdown"     : float(max_dd),
        "calmar_ratio"     : float(calmar),
        "win_rate"         : float(win_rate),
        "final_portfolio"  : float(pvs[-1]),
    }


def _print_results(name: str, metrics: dict) -> None:
    print(f"\n  ┌──────────────────────────────────────────┐")
    print(f"  │  Backtest Results ({name:<22s}│")
    print(f"  ├──────────────────────────────────────────┤")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  │  {k:<24s}: {v:>10.4f}          │")
        else:
            print(f"  │  {k:<24s}: {str(v):>10s}          │")
    print(f"  └──────────────────────────────────────────┘")
