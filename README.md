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
[ text_embedding        ]  D       당일 뉴스 임베딩 평균 (stock 16d + market 16d = 32d)

총 state_dim = 1 + 2S + 11S + D
예) S=100종목, D=32 → state_dim = 1 + 200 + 1100 + 32 = 1333
```

임베딩 구조:
- `emb_0~15` : stock 임베딩 (종목별 뉴스, date × tic join 후 평균)
- `emb_16~31` : market 임베딩 (시장 전체 뉴스, 날짜별 broadcast)

### Action

```
Box(-1, 1)^S  — 종목 수만큼의 연속값
  action >  threshold  → 매수  (action × max_shares 주)
  action < -threshold  → 매도  (action × 보유 주식 수)
  |action| ≤ threshold → 홀드  (기본 threshold = 0.10)
```

### Reward

```
r_t = (P_t - P_{t-1}) / P_{t-1}   ← 일간 수익률 (기본)
최종 reward = r_t × 1e-4           ← reward_scaling
```

### Transition

환경은 과거 데이터를 날짜 순서대로 재생합니다.  
`_day_idx += 1` 후 다음 날짜의 실제 데이터를 읽어 s_{t+1}을 구성합니다.  
에이전트의 행동이 미래 가격에 영향을 주지 않는다고 가정합니다.

---

## 정확한 트레이딩 흐름

```
reset()
  _day_idx = 0
  s_0 = 관측 (dates[0] 종가 + dates[0] 임베딩)

──── 루프 ────────────────────────────────────────────────────

step(a_t):
  1. prices    = dates[t] 종가
  2. execute   → dates[t] 종가로 체결      ← 매수/매도 실행
  3. day_idx  += 1
  4. portfolio = balance + shares · dates[t+1] 종가
  5. reward    = (portfolio_t - portfolio_{t-1}) / portfolio_{t-1} × 1e-4
  6. s_{t+1}  = (dates[t+1] 종가 + dates[t+1] 임베딩)
  반환: (s_{t+1}, reward, done)
```

> **체결 시점 주의**: t일 종가를 확인하는 동시에 t일 종가로 체결합니다.  
> 현실에서는 t일 종가 확인 후 주문을 내면 t+1일 시가에 체결됩니다.

---

## 텍스트 임베딩

### 데이터 범위

| 파일 | 시작 | 끝 | 행 수 |
|------|------|----|-------|
| `market_embeddings_16d.jsonl` | 2025-01-01 | 2026-05-21 | 18,003 |
| `stock_embeddings_16d.jsonl` | 2025-01-01 | 2026-05-18 | 119,243 |

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
- `"stock"` / `"company"` → 종목별 뉴스, (date, ticker)별 평균 → 해당 종목에만 join

로더에서 `"stock"` → `"company"` 자동 정규화, `"000990"` → `"000990.KS"` 자동 변환합니다.

### State 결합 방식

```
stock  임베딩  (date, tic) left-join  → co_emb_0~15
market 임베딩  date         broadcast  → mk_emb_0~15
np.concatenate → emb_0~31  (32d)
빈 날짜: forward-fill → 0 채움
```

---

## 학습 / 테스트 분리

`StateBuilder.build_train_test()` 가 날짜 기준으로 분리합니다.

```python
df_train = df[df["date"] <  train_end]   # 2025-01-01 ~ 2025-12-31
df_test  = df[df["date"] >= train_end]   # 2026-01-01 ~ 2026-05-18
```

- `StandardScaler`는 train에만 `fit`, 양쪽 모두 `transform`
- Validation set 없음 — `EvalCallback`이 train 데이터로 best model 선정

---

## Backtesting — MinVar 비교

학습 완료 후 동일한 `df_test`로 MinVar 전략을 실행해 자동 비교합니다.

### MinVar 전략

| 항목 | 값 |
|------|-----|
| 공분산 추정 윈도우 | 직전 60 거래일 수익률 |
| 리밸런싱 주기 | 20 거래일 (약 1개월) |
| 최적화 | `min w^T Σ w`, s.t. `Σw=1`, `w≥0` (scipy SLSQP) |
| 초기 할당 | 유효 종목 균등비중 |
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
├── portfolio_ppo.png           ← RL 수익률(%) + Drawdown 차트
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

### 기본 실행

```bash
# 임베딩 파일 기반 종목 자동 감지 (권장)
python main.py --tickers-from-emb --algo ppo --timesteps 1000000

# 종목 수동 지정
python main.py \
  --tickers 005930.KS 000660.KS 035420.KS 005380.KS 051910.KS \
  --algo ppo --timesteps 500000
```

### 전체 옵션

```
--tickers          TICKER [...]   종목 수동 지정 (Yahoo Finance 형식)
--tickers-from-emb               임베딩 파일에서 종목 자동 감지
--start            YYYY-MM-DD    데이터 시작일 (기본: 2025-01-01)
--split            YYYY-MM-DD    학습/테스트 분리일 (기본: 2026-01-01)
--end              YYYY-MM-DD    데이터 종료일 (기본: 2026-05-18)
--emb-source       {local|gdrive|none}   임베딩 소스 (기본: local)
--emb-local-path   PATH          로컬 .jsonl 디렉토리 (기본: Embeddings/)
--gdrive-folder-id ID            Google Drive 폴더 ID
--emb-dim          INT           임베딩 차원 (기본: 자동 감지)
--no-company-emb                 종목 임베딩 제외
--no-market-emb                  시장 임베딩 제외
--algo             {ppo|a2c|ddpg|td3|sac}  (기본: ppo)
--timesteps        INT           학습 스텝 수 (기본: 100,000)
--hold-threshold   FLOAT         홀드 임계값 (기본: 0.10, 낮출수록 매매 증가)
--initial-capital  FLOAT         초기 자본 KRW (기본: 10,000,000)
--output-dir       PATH          결과 저장 경로 (기본: results/)
```

### 학습 팁

| 상황 | 권장 설정 |
|------|-----------|
| 빠른 테스트 | `--timesteps 100000 --tickers 005930.KS 000660.KS 035420.KS` |
| 100종목 학습 | `--timesteps 2000000 --hold-threshold 0.05` |
| 학습률 불안정 시 | trainer.py의 `learning_rate` 를 `3e-4 → 1e-4` 로 조정 |

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
│   ├── market_embeddings_16d.jsonl  시장 전체 뉴스 임베딩 (2025-01 ~ 2026-05)
│   └── stock_embeddings_16d.jsonl   종목별 뉴스 임베딩 (2025-01 ~ 2026-05, 101종목)
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
| 공매도 | 미지원 | short position 허용 시 action bounds 조정 |
| 데이터 기간 | 임베딩 2025~2026 (약 1.5년) | 장기 데이터 확보 시 성능 개선 기대 |
