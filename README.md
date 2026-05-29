# KOSPI FinRL + Text Embedding Pipeline

강화학습 기반 KOSPI 멀티 종목 트레이딩 에이전트.  
Yahoo Finance OHLCV + 기술지표와 뉴스 텍스트 임베딩(JSONL)을 결합해 State를 구성하고,  
Stable-Baselines3(SB3)로 학습한 뒤 MinVar 등 전통 전략과 성능을 비교합니다.

---

## 전체 파이프라인

```
Yahoo Finance ──► KOSPIDataLoader
                  (OHLCV + 11 기술지표)          ┐
                                                 ▼
Embeddings/*.jsonl ──► LocalEmbeddingLoader ──► StateBuilder ──► df_train / df_test
  market_embeddings_16d.jsonl                   (스케일링 +         │
  stock_embeddings_16d.jsonl                     날짜 split)        │
                                                               ┌────┴────┐
                                                               ▼         ▼
                                                       KOSPITradingEnv  MinVarStrategy
                                                          (Gym Env)      (Backtesting)
                                                               │              │
                                                        SB3 Agent        비교 리포트
                                                   (PPO/A2C/DDPG/TD3/SAC)
```

---

## MDP 정의

### State

매 타임스텝(= 1 영업일)마다 에이전트가 관측하는 벡터:

```
[ balance_ratio         ]  1
[ shares_ratio × S      ]  S       보유 주식 수 / max_shares
[ price_ratio  × S      ]  S       현재 종가 / 자본 프록시
[ tech_indicators × S×T ]  S×11    MACD, Bollinger, RSI, CCI, DX, SMA30/60, vol_chg, ret_1d/5d
[ text_embedding        ]  D       당일 뉴스 임베딩 평균 (market + stock concat)

총 state_dim = 1 + 2S + 11S + D
```

### Action

```
Box(-1, 1)^S  — 종목 수만큼의 연속값
  action >  0.10  → 매수  (action × max_shares 주)
  action < -0.10  → 매도  (action × 보유 주식 수)
  |action| ≤ 0.10 → 홀드
```

### Reward

```
기본:  r_t = (P_t - P_{t-1}) / P_{t-1}        일간 수익률

Sharpe 보정 (sharpe_window=20):
  r_t = mean(ret[-20:]) / std(ret[-20:])       변동성 패널티 포함

최종:  reward = r_t × reward_scaling (1e-4)
```

### Transition

환경은 과거 데이터를 날짜 순서대로 재생합니다.  
`P(s_{t+1} | s_t, a_t)`를 별도로 모델링하지 않고, `_day_idx += 1` 후  
다음 날짜의 실제 데이터를 읽어 s_{t+1}을 구성합니다.

---

## 정확한 트레이딩 흐름

```
reset()
  _day_idx = 0
  s_0 = 관측 (dates[0] 종가 + 임베딩)

──── 루프 ────────────────────────────────────────────────────

step(a_t):
  1. prices    = dates[t] 종가            ← 현재 가격 확인
  2. execute   → dates[t] 종가로 체결     ← 매수/매도 실행
  3. day_idx  += 1
  4. portfolio = balance + shares · dates[t+1] 종가   ← 다음날 평가
  5. reward    = f(portfolio 변화율)
  6. s_{t+1}  = 관측 (dates[t+1] 종가 + 임베딩)
  반환: (s_{t+1}, reward, done)
```

> **체결 시점 주의**: 코드는 t일 종가를 보고 즉시 t일 종가로 체결합니다.  
> 현실에서는 t일 종가 확인 후 주문을 내면 t+1일 시가에 체결됩니다.  
> 유동성이 낮은 종목일수록 슬리피지 영향이 커질 수 있습니다.

---

## 텍스트 임베딩

### JSONL 스키마

```json
{
  "source_type": "market",
  "date": "2026.05.21",
  "ticker": "",
  "company": "시장전체",
  "keyword": "반도체",
  "title": "...",
  "embedding_dim": 16,
  "embedding": [-0.363, 0.019, ...]
}
```

`source_type` 값:
- `"market"` → 시장 전체 뉴스, 날짜별 평균 → 모든 종목에 broadcast
- `"stock"` (또는 `"company"`) → 종목별 뉴스, (date, ticker)별 평균 → 해당 종목에만 join

로더에서 `"stock"` → `"company"` 로 자동 정규화합니다.

### State 결합 방식

```
company 임베딩  (date, tic) left-join  → co_emb_*
market  임베딩  date         broadcast  → mk_emb_*
concat → emb_0 … emb_{D-1}
빈 날짜: forward-fill → 0 채움
```

---

## 학습 / 테스트 분리

`StateBuilder.build_train_test()` 가 날짜 기준으로 단순 분리합니다.

```python
df_train = df[df["date"] <  train_end]   # 예: 2020-01-01 ~ 2022-12-31
df_test  = df[df["date"] >= train_end]   # 예: 2023-01-01 ~ 2024-06-30
```

- `StandardScaler`는 train에만 `fit`, 양쪽 모두 `transform`
- Validation set은 별도로 없음 — `EvalCallback`이 train 데이터로 best model 선정

---

## Backtesting — MinVar 비교

학습 완료 후 동일한 `df_test`로 MinVar 전략을 실행해 자동 비교합니다.

### MinVar 전략

| 항목 | 값 |
|------|-----|
| 공분산 추정 윈도우 | 직전 60 거래일 수익률 |
| 리밸런싱 주기 | 20 거래일 (약 1개월) |
| 최적화 | `min w^T Σ w`, s.t. `Σw=1`, `w≥0` (scipy SLSQP) |
| 초기 할당 | 균등비중 (lookback 이전 기간) |
| 거래 비용 | 0.3% |

### 비교 지표

| 지표 | 설명 |
|------|------|
| total_return | 전체 기간 수익률 |
| annualised_return | 연환산 수익률 |
| sharpe_ratio | 연환산 Sharpe (√252 스케일) |
| max_drawdown | 최대 낙폭 |
| calmar_ratio | annualised_return / \|max_drawdown\| |
| win_rate | 수익 발생 거래일 비율 |

### 출력 파일

```
results/
├── ppo_final.zip               ← 최종 모델
├── best_model/                 ← EvalCallback 최고 성능 모델
├── checkpoints/                ← 학습 중간 체크포인트
├── portfolio_ppo.png           ← RL 단독 포트폴리오 + Drawdown
├── comparison.png              ← RL vs MinVar 비교 차트
├── comparison_metrics.json     ← 지표 비교 JSON
└── metrics.json                ← RL 단독 지표
```

---

## 설치

```bash
pip install -r requirements.txt
```

---

## 실행

### 기본 실행 (로컬 임베딩)

```bash
python main.py \
  --tickers 005930.KS 000660.KS 035420.KS 005380.KS 051910.KS \
  --start 2020-01-01 --split 2023-01-01 --end 2024-06-30 \
  --emb-dim 16 --algo ppo --timesteps 200000
```

### 주요 옵션

```
--emb-source   {local|gdrive|none}   임베딩 소스 (기본: local)
--emb-local-path                     로컬 .jsonl 디렉토리 (기본: Embeddings/)
--gdrive-folder-id                   Google Drive 폴더 ID
--emb-dim      INT                   임베딩 차원 (기본: 자동 감지)
--algo         {ppo|a2c|ddpg|td3|sac}
--timesteps    INT                   학습 스텝 수
--split        YYYY-MM-DD            학습/테스트 분리 날짜
--initial-capital  FLOAT             초기 자본 (KRW)
--no-company-emb                     종목 임베딩 제외
--no-market-emb                      시장 임베딩 제외
--output-dir   PATH                  결과 저장 경로 (기본: results/)
```

### Google Drive 인증

```python
# 서비스 계정 (서버 환경 권장)
GDriveEmbeddingLoader(folder_id="...", credentials_path="service_account.json")

# OAuth2 브라우저 플로우 (로컬)
GDriveEmbeddingLoader(folder_id="...")  # 최초 실행 시 브라우저 열림

# Application Default Credentials
# gcloud auth application-default login
```

---

## 알고리즘

| 알고리즘 | 타입 | 특징 |
|----------|------|------|
| PPO | On-policy | 안정적, 권장 시작점 |
| A2C | On-policy | 빠른 학습 |
| DDPG | Off-policy | 연속 행동 공간 |
| TD3 | Off-policy | DDPG 개선, 안정적 |
| SAC | Off-policy | 엔트로피 기반, 탐험 우수 |

---

## 파일 구조

```
RLFinanceAgent/
├── main.py                        CLI 진입점 (6단계 파이프라인)
├── requirements.txt
├── Embeddings/
│   ├── market_embeddings_16d.jsonl  시장 전체 뉴스 임베딩
│   └── stock_embeddings_16d.jsonl   종목별 뉴스 임베딩
├── AutoEncoder/
│   └── encoder_dim16.pth            Upstage autoencoder 가중치
├── data/
│   ├── market_data.py               Yahoo Finance KOSPI 로더
│   ├── gdrive_loader.py             JSONL 파서 / 로컬·Drive 로더
│   └── state_builder.py             시장 데이터 + 임베딩 → RL State
├── rl/
│   ├── env_kospi.py                 Gym 환경 (MDP 구현)
│   ├── trainer.py                   SB3 학습 + 평가
│   └── backtest.py                  MinVar 등 전통 전략 비교
└── notebooks/
    └── kospi_finrl_pipeline.ipynb
```

---

## 한국 시장 특이사항

- 거래 비용: 0.3% (매매수수료 + 증권거래세 합산)
- Turbulence 임계값: 140 (30일 수익률 공분산 trace, 초과 시 전량 매도)
- 초기 자본: 1,000만 KRW (기본값)
- 티커 형식: `{종목코드}.KS` (예: `005930.KS` = 삼성전자)
- 최대 보유 수량: 종목당 500주

---

## 알려진 설계 한계

| 항목 | 현황 | 개선 방향 |
|------|------|-----------|
| 체결 가격 | t일 종가로 즉시 체결 | t+1일 시가 체결로 변경 |
| 임베딩 시점 | t일 뉴스로 t일 결정 | t-1일 임베딩 1일 shift |
| Validation set | 없음 (train으로 best model 선정) | Walk-forward validation 추가 |
| 공매도 | 미지원 (action ≥ 0 for buy) | short position 허용 시 bounds 조정 |
