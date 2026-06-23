FROM node:22-alpine AS frontend

WORKDIR /build/demo_app
COPY demo_app/package.json demo_app/package-lock.json demo_app/.npmrc ./
RUN npm ci
COPY demo_app/index.html demo_app/vite.config.mjs ./
COPY demo_app/src ./src
RUN npm run build

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOST=0.0.0.0 \
    PORT=8080

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY demo_app/backend/requirements.deploy.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.10.0 \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY data ./data
COPY rl ./rl
COPY news_crawler ./news_crawler
COPY demo_app ./demo_app
COPY model/results_ppo_2000000_50stocks_seed27_hold010/best_model/best_model.zip \
    ./model/results_ppo_2000000_50stocks_seed27_hold010/best_model/best_model.zip
COPY AutoEncoder/encoder_dim16.pth ./AutoEncoder/encoder_dim16.pth
COPY Embeddings/market_embeddings_16d.jsonl ./Embeddings/market_embeddings_16d.jsonl
COPY Embeddings/stock_embeddings_16d.jsonl ./Embeddings/stock_embeddings_16d.jsonl
COPY --from=frontend /build/demo_app/dist ./demo_app/dist

RUN addgroup --system app \
    && adduser --system --ingroup app app \
    && mkdir -p /app/demo_app/backend/runtime /tmp/matplotlib \
    && chown -R app:app /app/demo_app/backend/runtime /tmp/matplotlib

USER app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health', timeout=8)"

CMD ["python", "-m", "demo_app.backend.app"]
