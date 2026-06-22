from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from openai import OpenAI
from stable_baselines3 import PPO
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "model/results_ppo_2000000_50stocks_seed27_hold010/best_model/best_model.zip"
TICKER_PATH = Path(__file__).resolve().parent / "artifacts/best_model_action_tickers.txt"
ENCODER_PATH = REPO_ROOT / "AutoEncoder/encoder_dim16.pth"
EMBEDDING_PATH = REPO_ROOT / "Embeddings"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
SCALER_PATH = ARTIFACT_DIR / "market_scaler.joblib"
SCALER_META_PATH = ARTIFACT_DIR / "market_scaler.json"

TRAIN_START = "2025-01-01"
TRAIN_END = "2026-01-01"
MODEL_INITIAL_CAPITAL = 10_000_000.0
MAX_SHARES = 500
HOLD_THRESHOLD = 0.10
TRANSACTION_COST = 0.003
KST = timezone(timedelta(hours=9))


def _load_tickers() -> list[str]:
    tickers: list[str] = []
    for line in TICKER_PATH.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        _, ticker = line.split("\t", maxsplit=1)
        tickers.append(ticker.strip())
    if len(tickers) != 50:
        raise RuntimeError(f"Expected 50 action tickers, found {len(tickers)}")
    return tickers


def _normalise_ticker(value: str) -> str:
    ticker = value.strip().upper()
    if ticker.isdigit() and len(ticker) == 6:
        return f"{ticker}.KS"
    return ticker


class NewsEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class PipelineService:
    def __init__(self) -> None:
        load_dotenv(REPO_ROOT / ".env")
        self._lock = threading.Lock()
        self.tickers = _load_tickers()
        self.names = self._load_company_names()
        self.model = PPO.load(MODEL_PATH, device="cpu")
        if self.model.observation_space.shape != (1467,) or self.model.action_space.shape != (50,):
            raise RuntimeError("Saved PPO model dimensions do not match the demo contract")
        self.encoder = NewsEncoder()
        state = torch.load(ENCODER_PATH, map_location="cpu", weights_only=True)
        self.encoder.layers.load_state_dict(state)
        self.encoder.eval()
        self._latest_context: dict[str, Any] | None = None

    def _load_company_names(self) -> dict[str, str]:
        names: dict[str, str] = {}
        wanted = set(self.tickers)
        path = EMBEDDING_PATH / "stock_embeddings_16d.jsonl"
        if not path.exists():
            return {ticker: ticker.removesuffix(".KS") for ticker in wanted}

        with path.open("r", encoding="utf-8") as source:
            for line in source:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ticker = _normalise_ticker(str(record.get("ticker", "")))
                company = str(record.get("company", "")).strip()
                if ticker in wanted and company:
                    names[ticker] = company
                    if len(names) == len(wanted):
                        break
        return {ticker: names.get(ticker, ticker.removesuffix(".KS")) for ticker in self.tickers}

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "model": "PPO (FinRL)",
            "observation_dim": 1467,
            "action_dim": 50,
            "encoder": "4096 → 1024 → 256 → 64 → 16",
            "scaler_artifact": SCALER_PATH.exists(),
            "ticker_mapping": len(self.tickers),
            "latest_context_cached": self._latest_context is not None,
        }

    def infer(
        self,
        *,
        mode: str,
        cash: float,
        holdings: dict[str, int],
        news_text: str,
    ) -> dict[str, Any]:
        if cash < 0:
            raise ValueError("현금은 0 이상이어야 합니다.")
        holdings_array = self._holdings_array(holdings)

        with self._lock:
            if mode == "sample":
                observation, prices, data_date, pipeline = self._sample_observation(cash, holdings_array)
            elif mode == "latest":
                observation, prices, data_date, pipeline = self._latest_observation(
                    cash, holdings_array, news_text
                )
            else:
                raise ValueError("mode는 sample 또는 latest여야 합니다.")

            actions, _ = self.model.predict(observation.reshape(1, -1), deterministic=True)

        actions = np.asarray(actions, dtype=np.float32).reshape(-1)
        return self._format_result(
            mode=mode,
            cash=cash,
            holdings=holdings_array,
            prices=prices,
            actions=actions,
            data_date=data_date,
            pipeline=pipeline,
        )

    def _holdings_array(self, holdings: dict[str, int]) -> np.ndarray:
        normalised = {_normalise_ticker(key): int(value) for key, value in holdings.items()}
        unknown = sorted(set(normalised) - set(self.tickers))
        if unknown:
            raise ValueError(f"모델 대상이 아닌 종목이 포함되어 있습니다: {', '.join(unknown)}")
        values = np.array([normalised.get(ticker, 0) for ticker in self.tickers], dtype=np.float32)
        if np.any(values < 0) or np.any(values > MAX_SHARES):
            raise ValueError(f"보유 수량은 종목별 0~{MAX_SHARES}주 범위여야 합니다.")
        return values

    def _sample_observation(
        self, cash: float, holdings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, str, list[dict[str, str]]]:
        if self.model._last_obs is None:
            raise RuntimeError("Saved model does not contain a reproducible observation")
        observation = np.asarray(self.model._last_obs[0], dtype=np.float32).copy()
        prices = observation[51:101] * (MODEL_INITIAL_CAPITAL / len(self.tickers))
        portfolio_value = cash + float(np.dot(holdings, prices))
        observation[0] = cash / max(portfolio_value, 1e-8)
        observation[1:51] = holdings / MAX_SHARES
        pipeline = [
            {"name": "Yahoo 데이터", "status": "cached", "detail": "저장 관측값"},
            {"name": "Upstage", "status": "cached", "detail": "저장 임베딩"},
            {"name": "Autoencoder", "status": "ready", "detail": "16차원"},
            {"name": "PPO", "status": "ready", "detail": "1,467 → 50"},
        ]
        return observation, prices, "모델 저장 관측값", pipeline

    def _latest_observation(
        self, cash: float, holdings: np.ndarray, news_text: str
    ) -> tuple[np.ndarray, np.ndarray, str, list[dict[str, str]]]:
        now = datetime.now(KST)
        if now.weekday() < 5 and (now.hour, now.minute) < (15, 30):
            raise ValueError("최신 모드는 한국 시장 장 마감(15:30 KST) 이후에 실행할 수 있습니다.")

        context = self._prepare_latest_context()
        frame = context["frame"].copy()
        latest_date = context["date"]
        embedding_mode = "기존 최신 임베딩"

        live_embeddings = self._build_live_embeddings(news_text, latest_date)
        if not live_embeddings.empty:
            from data.gdrive_loader import LocalEmbeddingLoader

            base_embeddings = LocalEmbeddingLoader(path=EMBEDDING_PATH, emb_dim=16).load()
            embeddings = pd.concat([base_embeddings, live_embeddings], ignore_index=True)
            frame, scaler = self._build_state_frame(context["market"], embeddings)
            self._save_scaler(scaler, latest_date)
            embedding_mode = f"신규 뉴스 {len(live_embeddings)}개"

        latest = frame[frame["date"] == latest_date].copy()
        if latest["tic"].nunique() != len(self.tickers):
            raise RuntimeError("최신 날짜의 50종목 시장 데이터를 완성하지 못했습니다.")

        from rl.env_kospi import KOSPITradingEnv
        from rl.trainer import TECH_INDICATORS

        env = KOSPITradingEnv(
            df=latest,
            tech_indicators=TECH_INDICATORS,
            co_emb_dim=16,
            mk_emb_dim=16,
            initial_capital=MODEL_INITIAL_CAPITAL,
            max_shares=MAX_SHARES,
            sharpe_window=0,
            hold_threshold=HOLD_THRESHOLD,
        )
        env._balance = cash
        env._shares_held = holdings.copy()
        observation = env._get_observation()
        prices = env._current_prices()
        pipeline = [
            {"name": "Yahoo 데이터", "status": "ready", "detail": latest_date},
            {"name": "Upstage", "status": "ready", "detail": embedding_mode},
            {"name": "Autoencoder", "status": "ready", "detail": "4096 → 16"},
            {"name": "PPO", "status": "ready", "detail": "1,467 → 50"},
        ]
        return observation, prices, latest_date, pipeline

    def _prepare_latest_context(self) -> dict[str, Any]:
        if self._latest_context is not None:
            return self._latest_context

        from data.gdrive_loader import LocalEmbeddingLoader
        from data.market_data import KOSPIDataLoader

        end = (datetime.now(KST).date() + timedelta(days=1)).isoformat()
        market = KOSPIDataLoader(
            tickers=self.tickers,
            start=TRAIN_START,
            end=end,
        ).fetch_and_process()
        missing = sorted(set(self.tickers) - set(market["tic"].unique()))
        if missing:
            raise RuntimeError(f"Yahoo 데이터가 누락된 종목: {', '.join(missing)}")

        embeddings = LocalEmbeddingLoader(path=EMBEDDING_PATH, emb_dim=16).load()
        frame, scaler = self._build_state_frame(market, embeddings)
        latest_date = str(frame["date"].max())
        self._save_scaler(scaler, latest_date)
        self._latest_context = {"market": market, "frame": frame, "date": latest_date}
        return self._latest_context

    def _build_state_frame(
        self, market: pd.DataFrame, embeddings: pd.DataFrame
    ) -> tuple[pd.DataFrame, Any]:
        from data.state_builder import StateBuilder

        builder = StateBuilder(
            df_market=market,
            df_embeddings=embeddings,
            emb_dim=16,
            train_end=TRAIN_END,
            use_company_emb=True,
            use_market_emb=True,
            scale_market=True,
        )
        train, test = builder.build_train_test()
        frame = pd.concat([train, test], ignore_index=True)
        if builder.state_dim != 1467:
            raise RuntimeError(f"Reconstructed state dimension is {builder.state_dim}, expected 1467")
        return frame, builder._scaler

    def _save_scaler(self, scaler: Any, latest_date: str) -> None:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        metadata = {
            "train_start": TRAIN_START,
            "train_end_exclusive": TRAIN_END,
            "latest_market_date": latest_date,
            "features": list(getattr(scaler, "feature_names_in_", [])),
            "generated_at": datetime.now(KST).isoformat(),
        }
        SCALER_META_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_live_embeddings(self, news_text: str, date: str) -> pd.DataFrame:
        payload = news_text.strip()
        if not payload:
            return pd.DataFrame()

        market_news: list[str] = []
        stock_news: dict[str, list[str]] = {}
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            market_news = [line.strip() for line in payload.splitlines() if line.strip()]
        else:
            if not isinstance(parsed, dict):
                raise ValueError("뉴스 JSON은 객체 형식이어야 합니다.")
            market_news = self._normalise_news_list(parsed.get("market_news", []))
            raw_stock = parsed.get("stock_news", {})
            if not isinstance(raw_stock, dict):
                raise ValueError("stock_news는 ticker별 뉴스 배열 객체여야 합니다.")
            for raw_ticker, items in raw_stock.items():
                ticker = _normalise_ticker(raw_ticker)
                if ticker not in self.tickers:
                    raise ValueError(f"모델 대상이 아닌 뉴스 ticker입니다: {ticker}")
                stock_news[ticker] = self._normalise_news_list(items)

        texts: list[str] = []
        descriptors: list[tuple[str, str]] = []
        for headline in market_news:
            texts.append(f"시장 뉴스 | 제목: {headline}")
            descriptors.append(("market", "__market__"))
        for ticker, headlines in stock_news.items():
            for headline in headlines:
                texts.append(f"회사: {self.names[ticker]} | 제목: {headline}")
                descriptors.append(("company", ticker))
        if not texts:
            return pd.DataFrame()

        api_key = os.environ.get("UPSTAGE_API_KEY", "").strip()
        if not api_key or api_key == "XXXXXXX":
            raise ValueError("최신 뉴스 임베딩을 실행하려면 UPSTAGE_API_KEY가 필요합니다.")
        client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("UPSTAGE_BASE_URL", "https://api.upstage.ai/v1"),
        )
        model_name = os.environ.get("UPSTAGE_EMBEDDING_MODEL", "embedding-passage")
        response = client.embeddings.create(input=texts, model=model_name)
        raw = np.asarray([item.embedding for item in response.data], dtype=np.float32)
        if raw.shape[1] != 4096:
            raise RuntimeError(f"Upstage embedding dimension is {raw.shape[1]}, expected 4096")
        with torch.inference_mode():
            latent = self.encoder(torch.from_numpy(raw)).cpu().numpy()

        rows = []
        for vector, (source_type, ticker) in zip(latent, descriptors):
            row: dict[str, Any] = {"date": date, "tic": ticker, "source_type": source_type}
            row.update({f"emb_{index}": float(value) for index, value in enumerate(vector)})
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _normalise_news_list(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if not isinstance(value, list):
            raise ValueError("뉴스 항목은 문자열 또는 문자열 배열이어야 합니다.")
        result = []
        for item in value:
            if isinstance(item, dict):
                item = item.get("title", "")
            text = str(item).strip()
            if text:
                result.append(text)
        return result

    def _format_result(
        self,
        *,
        mode: str,
        cash: float,
        holdings: np.ndarray,
        prices: np.ndarray,
        actions: np.ndarray,
        data_date: str,
        pipeline: list[dict[str, str]],
    ) -> dict[str, Any]:
        remaining_cash = float(cash)
        estimated_holdings = holdings.astype(int).copy()
        quantities = np.zeros(len(self.tickers), dtype=int)

        for index in np.where(actions < -HOLD_THRESHOLD)[0]:
            quantity = min(int(abs(actions[index]) * MAX_SHARES), int(estimated_holdings[index]))
            if quantity <= 0 or prices[index] <= 0:
                continue
            remaining_cash += quantity * float(prices[index]) * (1 - TRANSACTION_COST)
            estimated_holdings[index] -= quantity
            quantities[index] = -quantity

        for index in np.where(actions > HOLD_THRESHOLD)[0]:
            if prices[index] <= 0:
                continue
            quantity = min(
                int(abs(actions[index]) * MAX_SHARES),
                MAX_SHARES - int(estimated_holdings[index]),
            )
            unit_cost = float(prices[index]) * (1 + TRANSACTION_COST)
            quantity = min(quantity, int(remaining_cash / unit_cost))
            if quantity <= 0:
                continue
            remaining_cash -= quantity * unit_cost
            estimated_holdings[index] += quantity
            quantities[index] = quantity

        rows = []
        for index, ticker in enumerate(self.tickers):
            score = float(actions[index])
            label = "매수" if score > HOLD_THRESHOLD else "매도" if score < -HOLD_THRESHOLD else "보유"
            rows.append(
                {
                    "index": index,
                    "ticker": ticker,
                    "company": self.names[ticker],
                    "score": score,
                    "action": label,
                    "estimated_quantity": int(quantities[index]),
                    "current_shares": int(holdings[index]),
                    "estimated_shares": int(estimated_holdings[index]),
                    "price": float(prices[index]),
                    "estimated_amount": float(abs(quantities[index]) * prices[index]),
                    "freshness": data_date,
                }
            )

        buys = sorted((row for row in rows if row["action"] == "매수"), key=lambda row: row["score"], reverse=True)
        sells = sorted((row for row in rows if row["action"] == "매도"), key=lambda row: row["score"])
        generated_at = datetime.now(KST).isoformat(timespec="seconds")
        return {
            "mode": mode,
            "data_date": data_date,
            "generated_at": generated_at,
            "pipeline": pipeline,
            "summary": {
                "buy_count": len(buys),
                "sell_count": len(sells),
                "hold_count": len(rows) - len(buys) - len(sells),
                "estimated_cash": remaining_cash,
                "starting_cash": cash,
            },
            "top_buys": buys[:5],
            "top_sells": sells[:5],
            "rows": rows,
            "model": {
                "name": "PPO (FinRL)",
                "observation_dim": 1467,
                "action_dim": 50,
                "hold_threshold": HOLD_THRESHOLD,
                "position_cap": MAX_SHARES,
                "transaction_cost": TRANSACTION_COST,
            },
            "disclaimer": "연구용 데모 · 실제 투자 주문 아님 · 수량은 t일 종가 기준 추정치",
        }


_service: PipelineService | None = None
_service_lock = threading.Lock()


def get_pipeline_service() -> PipelineService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = PipelineService()
    return _service
