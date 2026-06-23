# PPO 투자 에이전트 데모

저장된 50종목 PPO 모델을 실제로 로드해 매수·보유·매도 점수와 t일 종가 기준 예상 수량을 보여주는 연구용 대시보드임.

## 실행 모드

- 샘플 모드: 모델 ZIP에 저장된 1,467차원 관측값을 사용해 외부 API 없이 재현 가능함
- 최신 데이터: 장 마감 후 Yahoo Finance OHLCV와 당일 네이버 시장·종목 뉴스를 사용함. 기술지표 scaler를 학습 구간에서 재구성하고, 신규 뉴스는 Upstage 4,096차원 임베딩과 공용 Autoencoder를 거쳐 16차원으로 압축함

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

최신 뉴스 임베딩에는 루트 `.env`의 `UPSTAGE_API_KEY`, `UPSTAGE_BASE_URL`, `UPSTAGE_EMBEDDING_MODEL`을 사용함. 최신 모드의 `최신 뉴스 자동 수집`은 기본 활성화되며, 고정 50종목과 시장 키워드의 해당 거래일 뉴스만 수집함. 종목명·허용 별칭과 시장 키워드가 제목에 확인되는 기사만 사용함. 수집 CSV와 생성 임베딩은 `demo_app/backend/runtime/`에 캐시하며 Git에는 포함하지 않음.

자동 수집을 끄고 뉴스 입력을 비우면 저장된 최신 16차원 임베딩을 forward-fill함. 직접 입력한 뉴스는 자동 수집 뉴스에 추가함.

네이버가 HTTP 403/429를 반환하거나 해당 날짜 뉴스가 없으면 최신 추론을 중단하지 않고 기존 최신 임베딩으로 fallback함. 이 경우 결과의 Upstage 단계에 `자동 수집 실패`가 표시됨.

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

크롤러만 별도로 실행하는 방법은 [`../news_crawler/README.md`](../news_crawler/README.md)에 정리함.

Docker와 Cloud Run 배포 준비는 [`DEPLOYMENT.md`](DEPLOYMENT.md)에 정리함.
