# 최신 뉴스 크롤러

원격 `feature/kospi-crawler`의 종목 뉴스 흐름과 `crawling`의 시장 키워드 흐름을 현재 데모용으로 통합함.

- 전체 과거 기간을 다시 순회하지 않고 지정한 거래일 한 날짜만 수집함
- PPO action과 동일한 고정 50종목을 사용함
- 시장 뉴스는 키워드별 최대 2건, 종목 뉴스는 종목별 최대 1건을 기본으로 사용함
- `market_news.csv`, `stock_news.csv`를 캐시하여 같은 날짜의 반복 요청을 방지함
- 전체 요청은 기본 0.8초 간격으로 제한함
- HTTP 403/429에서는 우회하지 않고 중단함

```bash
python -m news_crawler --date 2026-06-22
```

요청 간격은 `--request-interval`로 조정할 수 있으며, 차단 위험 때문에 기본값보다 낮추지 않는 것을 권장함.

빠른 확인:

```bash
python -m news_crawler \
  --date 2026-06-22 \
  --keyword-limit 1 \
  --stock-limit 2 \
  --force
```

출력 CSV 스키마는 기존 임베딩 입력과 동일함.

```text
date,company,ticker,keyword,title,press,source
```
