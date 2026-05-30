"""
KOSPI FinRL Pipeline — JSONL 임베딩 버전
=========================================
Pipeline:
  1. Yahoo Finance  → KOSPI OHLCV + 기술지표
  2. Google Drive   → .jsonl 텍스트 임베딩
       source_type="market"  (시장 전체 뉴스)
       source_type="company" (종목별 뉴스)
  3. StateBuilder   → 시장 + 임베딩 결합 State
  4. PPO/A2C/DDPG/TD3/SAC 학습 & 평가
"""

import os, argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="KOSPI FinRL + JSONL 임베딩 파이프라인")
    p.add_argument("--tickers", nargs="+",
                   default=["005930.KS","000660.KS","035420.KS","005380.KS","051910.KS"])
    p.add_argument("--tickers-from-emb", action="store_true",
                   help="임베딩 파일에 존재하는 종목을 자동으로 티커 목록으로 사용")
    p.add_argument("--start",  default="2025-01-01")
    p.add_argument("--split",  default="2026-01-01", help="학습/테스트 분리 날짜")
    p.add_argument("--end",    default="2026-05-18")

    p.add_argument("--emb-source", choices=["gdrive","local","none"], default="local")
    p.add_argument("--gdrive-folder-id", default=None, help="Drive 폴더 ID")
    p.add_argument("--emb-local-path",   default="Embeddings/", help="로컬 .jsonl 경로")
    p.add_argument("--emb-dim",   type=int, default=None,
                   help="임베딩 차원 (None=자동 감지, 예: 16)")
    p.add_argument("--no-company-emb", action="store_true", help="종목 임베딩 제외")
    p.add_argument("--no-market-emb",  action="store_true", help="시장 임베딩 제외")

    p.add_argument("--algo",      choices=["ppo","a2c","ddpg","td3","sac"], default="ppo")
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--initial-capital", type=float, default=10_000_000)
    p.add_argument("--hold-threshold", type=float, default=0.10,
                   help="이 값 미만의 |action|은 홀드 처리 (기본: 0.10)")
    p.add_argument("--output-dir", default="results/")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*58)
    print("  KOSPI FinRL + JSONL 임베딩 강화학습 파이프라인")
    print("="*58)

    # ── 티커 목록 결정 ────────────────────────────────────────────
    if args.tickers_from_emb:
        from data.gdrive_loader import get_tickers_from_jsonl
        tickers = get_tickers_from_jsonl(args.emb_local_path)
        print(f"\n      임베딩 기반 티커 {len(tickers)}개 자동 감지")
    else:
        tickers = args.tickers

    # ── 1. 시장 데이터 ────────────────────────────────────────────
    print("\n[1/6] Yahoo Finance에서 KOSPI 시장 데이터 로드 …")
    from data.market_data import KOSPIDataLoader
    df_market = KOSPIDataLoader(
        tickers=tickers, start=args.start, end=args.end
    ).fetch_and_process()
    actual_tickers = df_market["tic"].unique().tolist()
    print(f"      {len(df_market):,}행 | {len(actual_tickers)}개 종목 로드됨")

    # ── 2. JSONL 임베딩 ───────────────────────────────────────────
    print(f"\n[2/6] 텍스트 임베딩 로드 ({args.emb_source}) …")
    df_emb = None
    if args.emb_source == "gdrive":
        from data.gdrive_loader import GDriveEmbeddingLoader
        df_emb = GDriveEmbeddingLoader(
            folder_id=args.gdrive_folder_id,
            emb_dim=args.emb_dim,
        ).load()
    elif args.emb_source == "local":
        from data.gdrive_loader import LocalEmbeddingLoader
        df_emb = LocalEmbeddingLoader(
            path=args.emb_local_path,
            emb_dim=args.emb_dim,
        ).load()
    else:
        print("      임베딩 없음 (none 모드)")

    # ── 3. State 구성 ─────────────────────────────────────────────
    print("\n[3/6] 시장 데이터 + 임베딩 → RL State 구성 …")
    from data.state_builder import StateBuilder
    builder = StateBuilder(
        df_market=df_market,
        df_embeddings=df_emb,
        emb_dim=args.emb_dim,
        train_end=args.split,
        use_company_emb=not args.no_company_emb,
        use_market_emb =not args.no_market_emb,
    )
    df_train, df_test = builder.build_train_test()
    print(f"      학습: {len(df_train):,}행 | 테스트: {len(df_test):,}행")

    # ── 4. 학습 ───────────────────────────────────────────────────
    print(f"\n[4/6] {args.algo.upper()} 학습 ({args.timesteps:,} timesteps) …")
    from rl.trainer import RLTrainer
    trainer = RLTrainer(
        df_train=df_train, df_test=df_test,
        state_dim=builder.state_dim, n_stocks=len(actual_tickers),
        initial_capital=args.initial_capital,
        algo=args.algo, timesteps=args.timesteps,
        output_dir=args.output_dir, emb_dim=builder.emb_dim,
        hold_threshold=args.hold_threshold,
    )
    model = trainer.train()

    # ── 5. 평가 ───────────────────────────────────────────────────
    print("\n[5/6] 테스트셋 평가 …")
    rl_result = trainer.evaluate(model)

    # ── 6. Backtesting 비교 ───────────────────────────────────────
    print("\n[6/6] 전통 전략 비교 (MinVar) …")
    from rl.backtest import Backtester
    backtester = Backtester(
        df_test=df_test,
        initial_capital=args.initial_capital,
        output_dir=args.output_dir,
    )
    minvar_result = backtester.run_minvar(lookback=60, rebalance_every=20)

    backtester.compare({
        f"RL ({args.algo.upper()})": rl_result,
        "MinVar"                  : minvar_result,
    })

    print(f"\n✅  완료!  결과 저장 위치: {args.output_dir}\n")


if __name__ == "__main__":
    main()
