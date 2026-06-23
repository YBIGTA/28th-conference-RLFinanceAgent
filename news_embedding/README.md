# 뉴스 임베딩 데이터셋

`stock_news.csv`와 `market_news_v4.csv`는 뉴스 1행을 임베딩 1개로 변환한다. 기사 본문이 없고 제목 중심 데이터이므로 `title`을 핵심 텍스트로 사용하되, 강화학습 상태 구성에서 구분 정보가 사라지지 않도록 짧은 메타데이터를 함께 붙인다.

- `stock_news.csv`: `company`, `title`
- `market_news_v4.csv`: `keyword`, `title`
- 제외: `date`와 `ticker`는 시계열 정렬, 가격 데이터 조인, 종목별 aggregation 키로만 쓰고, `press`와 `source`는 수집 메타데이터 성격이 강해 임베딩 텍스트에서는 제외한다. stock의 `keyword`와 market의 `ticker/press`는 현재 비어 있어 제외한다.

이 디렉터리는 학습 데이터 준비용 오프라인 도구임. CSV 뉴스에서 Upstage 4,096차원 원본 임베딩을 생성하며, 배포된 데모의 실시간 추론 코드에는 포함되지 않음. PPO 상태에 사용하는 16차원 벡터를 만들려면 생성 결과에 학습된 Autoencoder를 별도로 적용해야 함.

실행 전 저장소 루트의 `.env`에 `UPSTAGE_API_KEY`를 설정함. `.env`와 생성된 JSONL은 Git에 커밋하지 않음.

```bash
pip install -r requirements.txt
python news_embedding/embed_news.py
```

기본 입력은 `~/Documents/stock_news.csv`, `~/Documents/market_news_v4.csv`임. 다른 위치의 CSV는 실행 옵션으로 지정할 수 있음.

```bash
python news_embedding/embed_news.py \
  --stock-csv /path/to/stock_news.csv \
  --market-csv /path/to/market_news_v4.csv
```

출력은 기본적으로 `news_embedding/stock_news_embeddings.jsonl`, `news_embedding/market_news_embeddings.jsonl`에 저장됨. 각 줄은 원본 행 메타데이터, 실제 임베딩에 보낸 `embedding_text`, 4,096차원 `embedding`을 포함함. 중간에 끊겨도 같은 파일에 이어서 실행하면 이미 처리한 `row_id`는 건너뜀.

## 저장 구조

출력 파일은 JSONL 형식이며, 한 줄에 뉴스 한 행의 JSON 객체 1개가 저장된다. 두 파일 모두 원본 메타데이터, 실제 임베딩에 사용한 `embedding_text`, 모델명, 임베딩 차원, 4096차원 임베딩 배열을 가진다.

`stock_news_embeddings.jsonl`의 한 행 예시는 다음과 같다.

```json
{
  "row_id": 0,
  "source_type": "stock",
  "date": "2025.01.01",
  "company": "DB하이텍",
  "ticker": "000990",
  "keyword": "",
  "title": "저점 찍은 증시 ‘밸류업’ 타고 점프 할까…주주환원 확대 종목 노려라...",
  "press": "파이낸셜뉴스",
  "source": "naver",
  "embedding_text": "회사: DB하이텍 | 제목: 저점 찍은 증시 ‘밸류업’ 타고 점프 할까…주주환원 확대 종목 노려라...",
  "embedding_model": "embedding-passage",
  "embedding_dim": 4096,
  "embedding": [-0.005828857421875, -0.007053375244140625, "..."]
}
```

stock 뉴스에서 실제 임베딩에 들어간 텍스트는 `company + title`이다. `date`, `ticker`, `press`, `source`는 임베딩 입력이 아니라 날짜별/종목별 aggregation, 가격 데이터 join, 추적용 메타데이터로 저장된다.

`market_news_embeddings.jsonl`의 한 행 예시는 다음과 같다.

```json
{
  "row_id": 0,
  "source_type": "market",
  "date": "2026.05.21",
  "company": "시장전체",
  "ticker": "",
  "keyword": "중국 경기",
  "title": "미 국채 금리 상승에 한중일 통화 비상[천조국 리포트]",
  "press": "머니투데이방송",
  "source": "naver",
  "embedding_text": "시장 키워드: 중국 경기 | 제목: 미 국채 금리 상승에 한중일 통화 비상[천조국 리포트]",
  "embedding_model": "embedding-passage",
  "embedding_dim": 4096,
  "embedding": [-0.01171112060546875, -0.01035308837890625, "..."]
}
```

market 뉴스에서 실제 임베딩에 들어간 텍스트는 `keyword + title`이다. `date`, `press`, `source`는 임베딩 입력이 아니라 시점별 시장 심리 벡터를 만들기 위한 메타데이터로 저장된다.

## `embed_news.py` 함수 설명

- `clean(value)`: CSV에서 읽은 문자열의 앞뒤 공백을 제거하고, `None` 값은 빈 문자열로 바꾼다.
- `make_embedding_text(row, source_type)`: CSV 한 행을 Upstage에 보낼 임베딩 입력 텍스트로 변환한다. stock 뉴스는 `company + title`, market 뉴스는 `keyword + title`만 사용한다.
- `read_rows(csv_path, source_type)`: CSV 파일을 읽고, 각 행마다 `row_id`, 원본 행, `embedding_text`를 생성한다. 제목이 비어 있어 임베딩할 수 없는 행은 건너뛴다.
- `batched(items, size)`: API 호출 횟수를 줄이기 위해 입력 데이터를 지정한 batch 크기 단위로 묶는다.
- `load_done_ids(output_path)`: 기존 JSONL 출력 파일에서 이미 저장된 `row_id`를 읽어 온다. 중간에 실행이 끊겼을 때 이어서 처리하기 위해 사용한다.
- `create_embeddings_with_retry(client, texts, model, max_retries)`: 일시적인 연결 오류, timeout, rate limit이 발생하면 exponential backoff 방식으로 API 요청을 재시도한다.
- `embed_csv(client, csv_path, output_path, source_type, model, batch_size, sleep_seconds, log_every, max_retries)`: CSV 하나를 읽어 Upstage Embedding API를 호출하고, 결과를 JSONL 파일로 저장한다. 응답 임베딩 차원이 4096이 아니면 에러를 발생시킨다.
- `parse_args()`: 실행 옵션을 정의한다. 입력 CSV 경로, 출력 폴더, batch 크기, sleep 시간, 로그 출력 주기, 재시도 횟수, stock/market 선택 옵션을 받을 수 있다.
- `main()`: `.env`에서 API 설정을 읽고 OpenAI-compatible Upstage client를 만든 뒤, stock/market 임베딩 작업을 실행한다.
