# PPO 투자 에이전트 데모

저장된 50종목 PPO 모델을 실제로 로드해 매수·보유·매도 점수와 t일 종가 기준 예상 수량을 보여주는 연구용 대시보드임.

## 실행 모드

- 샘플 모드: 모델 ZIP에 저장된 1,467차원 관측값을 사용해 외부 API 없이 재현 가능함
- 최신 데이터: 장 마감 후 Yahoo Finance OHLCV와 입력 뉴스를 사용함. 기술지표 scaler를 학습 구간에서 재구성하고, 신규 뉴스가 있으면 Upstage 4,096차원 임베딩과 공용 Autoencoder를 거쳐 16차원으로 압축함

실제 증권 주문은 수행하지 않으며 수량과 금액은 t일 종가 기준 추정치임.

## 실행

```bash
pip install -r requirements.txt
cd demo_app
npm install
npm run build
cd ..
python -m demo_app.backend.app
```

`http://127.0.0.1:8000`에서 확인함.

최신 뉴스 임베딩에는 루트 `.env`의 `UPSTAGE_API_KEY`, `UPSTAGE_BASE_URL`, `UPSTAGE_EMBEDDING_MODEL`을 사용함. 뉴스 입력을 비우면 저장된 최신 16차원 임베딩을 forward-fill함.

## 뉴스 JSON

```json
{
  "market_news": ["시장 전체 뉴스 제목"],
  "stock_news": {
    "005930.KS": ["삼성전자 뉴스 제목"]
  }
}
```

일반 텍스트를 입력하면 줄마다 시장 뉴스 한 건으로 처리함. 종목 뉴스 ticker는 모델의 고정 50종목만 허용함.
