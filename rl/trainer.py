"""
rl/trainer.py
────────────────────────────────────────────────────────────────
RL Trainer
  • Wraps KOSPITradingEnv in a Stable-Baselines3 (SB3) compatible interface
  • Supports PPO, A2C, DDPG, TD3, SAC
  • Saves model checkpoints and evaluation metrics
  • Generates performance report with Sharpe / Max-DD / Calmar
"""

from __future__ import annotations
import os, json
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ────────────────────────────────────────────────────────────────
# Tech indicator list (must match state_builder.py)
# ────────────────────────────────────────────────────────────────
TECH_INDICATORS = [
    "macd", "boll_ub", "boll_lb",
    "rsi_30", "cci_30", "dx_30",
    "close_30_sma", "close_60_sma",
    "volume_change", "return_1d", "return_5d",
]


class RLTrainer:
    """
    Train and evaluate a Stable-Baselines3 RL agent on KOSPITradingEnv.

    Parameters
    ----------
    df_train, df_test : pd.DataFrame
    state_dim : int
    n_stocks : int
    initial_capital : float
    algo : str     — 'ppo' | 'a2c' | 'ddpg' | 'td3' | 'sac'
    timesteps : int
    output_dir : str
    """

    ALGO_MAP = {
        "ppo" : ("stable_baselines3", "PPO"),
        "a2c" : ("stable_baselines3", "A2C"),
        "ddpg": ("stable_baselines3", "DDPG"),
        "td3" : ("stable_baselines3", "TD3"),
        "sac" : ("stable_baselines3", "SAC"),
    }

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        state_dim: int,
        n_stocks: int,
        initial_capital: float = 10_000_000,
        algo: str = "ppo",
        timesteps: int = 100_000,
        output_dir: str = "results/",
        co_emb_dim: int = 0,
        mk_emb_dim: int = 0,
        hold_threshold: float = 0.10,
    ):
        self.df_train        = df_train
        self.df_test         = df_test
        self.state_dim       = state_dim
        self.n_stocks        = n_stocks
        self.initial_capital = initial_capital
        self.algo            = algo.lower()
        self.timesteps       = timesteps
        self.output_dir      = Path(output_dir)
        self.co_emb_dim      = co_emb_dim
        self.mk_emb_dim      = mk_emb_dim
        self.hold_threshold  = hold_threshold
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────
    # Train
    # ─────────────────────────────────────────────────────────────
    def train(self):
        """Build env + model, run training, save checkpoint, return model."""
        from stable_baselines3.common.vec_env import DummyVecEnv

        train_env = DummyVecEnv([self._make_env(self.df_train)])

        model = self._build_model(train_env)

        # ── Callbacks ───────────────────────────────────────────
        from stable_baselines3.common.callbacks import (
            EvalCallback, CheckpointCallback
        )
        eval_env  = DummyVecEnv([self._make_env(self.df_train)])
        callbacks = [
            CheckpointCallback(
                save_freq=max(self.timesteps // 10, 1000),
                save_path=str(self.output_dir / "checkpoints"),
                name_prefix=self.algo,
            ),
            EvalCallback(
                eval_env,
                best_model_save_path=str(self.output_dir / "best_model"),
                log_path=str(self.output_dir / "eval_logs"),
                eval_freq=max(self.timesteps // 20, 500),
                deterministic=True,
                render=False,
                n_eval_episodes=1,
            ),
        ]

        print(f"      Algorithm : {self.algo.upper()}")
        print(f"      Timesteps : {self.timesteps:,}")
        print(f"      State dim : {self.state_dim}")
        print(f"      n_stocks  : {self.n_stocks}")

        model.learn(
            total_timesteps=self.timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        model_path = self.output_dir / f"{self.algo}_final"
        model.save(str(model_path))
        print(f"      Model saved → {model_path}.zip")
        return model

    # ─────────────────────────────────────────────────────────────
    # Evaluate
    # ─────────────────────────────────────────────────────────────
    def evaluate(self, model) -> dict:
        """Run model on test env, compute metrics, save plots."""
        from stable_baselines3.common.vec_env import DummyVecEnv

        test_env = DummyVecEnv([self._make_env(self.df_test)])
        obs = test_env.reset()

        portfolio_vals = [self.initial_capital]  # day-0 baseline (before first trade)
        dates          = []

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = test_env.step(action)
            info = infos[0]
            portfolio_vals.append(info["portfolio_value"])
            dates.append(info["day"])
            if dones[0]:
                break

        # ── Metrics ─────────────────────────────────────────────
        metrics = self._compute_metrics(portfolio_vals)
        metrics["algorithm"]  = self.algo.upper()
        metrics["n_stocks"]   = self.n_stocks
        metrics["co_emb_dim"] = self.co_emb_dim
        metrics["mk_emb_dim"] = self.mk_emb_dim

        print("\n  ┌──────────────────────────────────────────┐")
        print(f"  │  Evaluation Results ({self.algo.upper():5s})                │")
        print("  ├──────────────────────────────────────────┤")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  │  {k:<24s}: {v:>10.4f}          │")
            else:
                print(f"  │  {k:<24s}: {str(v):>10s}          │")
        print("  └──────────────────────────────────────────┘")

        # ── Save ────────────────────────────────────────────────
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        self._plot_portfolio(portfolio_vals, dates)
        return {"portfolio_vals": portfolio_vals, "dates": dates, "metrics": metrics}

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────
    def _make_env(self, df: pd.DataFrame):
        """Return a callable that creates a KOSPITradingEnv."""
        from rl.env_kospi import KOSPITradingEnv
        tech = [c for c in TECH_INDICATORS if c in df.columns]

        def _factory():
            return KOSPITradingEnv(
                df=df,
                tech_indicators=tech,
                co_emb_dim=self.co_emb_dim,
                mk_emb_dim=self.mk_emb_dim,
                initial_capital=self.initial_capital,
                reward_scaling=1e-4,
                sharpe_window=0,
                hold_threshold=self.hold_threshold,
            )
        return _factory

    def _build_model(self, env):
        """Instantiate an SB3 model with sensible defaults per algo."""
        pkg, cls_name = self.ALGO_MAP[self.algo]
        import importlib
        module = importlib.import_module(pkg)
        AlgoCls = getattr(module, cls_name)

        common_kwargs = dict(env=env, verbose=1,
                             tensorboard_log=str(self.output_dir / "tb_logs"))

        if self.algo in ("ppo", "a2c"):
            return AlgoCls(
                "MlpPolicy",
                learning_rate=3e-4,
                n_steps=2048 if self.algo == "ppo" else 5,
                batch_size=64 if self.algo == "ppo" else None,
                policy_kwargs={"net_arch": [256, 256]},
                **common_kwargs,
            )
        elif self.algo == "ddpg":
            from stable_baselines3.common.noise import NormalActionNoise
            n_act = env.action_space.shape[0]
            noise = NormalActionNoise(
                mean=np.zeros(n_act), sigma=0.1 * np.ones(n_act)
            )
            return AlgoCls(
                "MlpPolicy",
                action_noise=noise,
                learning_rate=1e-3,
                policy_kwargs={"net_arch": [256, 256]},
                **common_kwargs,
            )
        elif self.algo in ("td3", "sac"):
            from stable_baselines3.common.noise import NormalActionNoise
            n_act = env.action_space.shape[0]
            noise = NormalActionNoise(
                mean=np.zeros(n_act), sigma=0.1 * np.ones(n_act)
            )
            return AlgoCls(
                "MlpPolicy",
                action_noise=noise if self.algo == "td3" else None,
                learning_rate=3e-4,
                policy_kwargs={"net_arch": [256, 256]},
                **common_kwargs,
            )
        else:
            raise ValueError(f"Unknown algo: {self.algo}")

    @staticmethod
    def _compute_metrics(portfolio_vals: list) -> dict:
        if not portfolio_vals:
            return {}
        pvs   = np.array(portfolio_vals, dtype=float)
        rets  = np.diff(pvs) / (pvs[:-1] + 1e-8)

        base        = pvs[0]  # evaluate()에서 initial_capital이 pvs[0]으로 prepend됨
        total_ret   = (pvs[-1] - base) / base
        n_days      = max(len(pvs) - 1, 1)   # 구간 수 = 포인트 수 - 1
        annualised  = (1 + total_ret) ** (252 / n_days) - 1
        sharpe      = (rets.mean() / (rets.std() + 1e-8)) * np.sqrt(252)
        running_max = np.maximum.accumulate(pvs)
        dd          = (pvs - running_max) / (running_max + 1e-8)
        max_dd      = float(dd.min())
        calmar      = annualised / (abs(max_dd) + 1e-8)
        win_rate    = float((rets > 0).mean())

        return {
            "total_return"    : float(total_ret),
            "annualised_return": float(annualised),
            "sharpe_ratio"    : float(sharpe),
            "max_drawdown"    : float(max_dd),
            "calmar_ratio"    : float(calmar),
            "win_rate"        : float(win_rate),
            "final_portfolio" : float(pvs[-1]),
        }

    def _plot_portfolio(self, pvs: list, dates: list):
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        pvs_arr = np.array(pvs)
        ret_pct = (pvs_arr - self.initial_capital) / self.initial_capital * 100

        # ── 수익률 (%) ───────────────────────────────────────────
        ax1 = axes[0]
        ax1.plot(ret_pct, color="#0ea5e9", linewidth=1.5, label="Portfolio")
        ax1.axhline(0, color="#94a3b8", linestyle="--", linewidth=1, label="Break-even")
        ax1.set_title(f"KOSPI Portfolio Return — {self.algo.upper()}", fontsize=13)
        ax1.set_ylabel("Return (%)")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}%"))
        ax1.legend()
        ax1.grid(alpha=0.3)

        # ── Drawdown ─────────────────────────────────────────────
        ax2 = axes[1]
        running_max = np.maximum.accumulate(pvs_arr)
        dd          = (pvs_arr - running_max) / (running_max + 1e-8) * 100
        ax2.fill_between(range(len(dd)), dd, 0, color="#ef4444", alpha=0.4, label="Drawdown")
        ax2.set_title("Drawdown (%)", fontsize=13)
        ax2.set_ylabel("Drawdown (%)")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}%"))
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        path = self.output_dir / f"portfolio_{self.algo}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"      Plot saved → {path}")
