#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError


DEFAULT_INPUTS = {
    "stock": Path("~/Documents/stock_news.csv").expanduser(),
    "market": Path("~/Documents/market_news_v4.csv").expanduser(),
}

EXPECTED_DIM = 4096


def clean(value):
    return (value or "").strip()


def make_embedding_text(row, source_type):
    title = clean(row.get("title"))
    if not title:
        return ""

    company = clean(row.get("company"))
    keyword = clean(row.get("keyword"))

    parts = []
    if source_type == "stock":
        if company:
            parts.append(f"회사: {company}")
    elif source_type == "market":
        if keyword:
            parts.append(f"시장 키워드: {keyword}")
    parts.append(f"제목: {title}")
    return " | ".join(parts)


def read_rows(csv_path, source_type):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_id, row in enumerate(reader):
            text = make_embedding_text(row, source_type)
            if not text:
                continue
            yield row_id, row, text


def batched(items, size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_done_ids(output_path):
    done = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["row_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def create_embeddings_with_retry(client, texts, model, max_retries):
    for attempt in range(max_retries + 1):
        try:
            return client.embeddings.create(input=list(texts), model=model)
        except (APIConnectionError, APITimeoutError, RateLimitError):
            if attempt >= max_retries:
                raise
            wait_seconds = min(2 ** attempt, 60)
            print(f"API request failed; retrying in {wait_seconds}s ({attempt + 1}/{max_retries})", flush=True)
            time.sleep(wait_seconds)


def embed_csv(client, csv_path, output_path, source_type, model, batch_size, sleep_seconds, log_every, max_retries):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(output_path)
    rows = (item for item in read_rows(csv_path, source_type) if item[0] not in done_ids)

    written = 0
    batch_count = 0
    with output_path.open("a", encoding="utf-8") as out:
        for batch in batched(rows, batch_size):
            batch_count += 1
            row_ids, raw_rows, texts = zip(*batch)
            response = create_embeddings_with_retry(client, texts, model, max_retries)
            if len(response.data) != len(batch):
                raise RuntimeError(f"Embedding count mismatch for {csv_path}: got {len(response.data)}, expected {len(batch)}")

            for row_id, raw_row, text, datum in zip(row_ids, raw_rows, texts, response.data):
                embedding = datum.embedding
                if len(embedding) != EXPECTED_DIM:
                    raise RuntimeError(f"Unexpected embedding dim for row {row_id}: got {len(embedding)}, expected {EXPECTED_DIM}")

                record = {
                    "row_id": row_id,
                    "source_type": source_type,
                    "date": clean(raw_row.get("date")),
                    "company": clean(raw_row.get("company")),
                    "ticker": clean(raw_row.get("ticker")),
                    "keyword": clean(raw_row.get("keyword")),
                    "title": clean(raw_row.get("title")),
                    "press": clean(raw_row.get("press")),
                    "source": clean(raw_row.get("source")),
                    "embedding_text": text,
                    "embedding_model": model,
                    "embedding_dim": len(embedding),
                    "embedding": embedding,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

            out.flush()
            if batch_count % log_every == 0:
                print(f"{source_type}: wrote {written} new embeddings to {output_path}", flush=True)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    if written:
        print(f"{source_type}: completed {written} new embeddings to {output_path}", flush=True)
    else:
        print(f"{source_type}: no new rows to embed for {output_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Embed stock_news.csv and market_news.csv with Upstage Solar embeddings.")
    parser.add_argument("--stock-csv", type=Path, default=DEFAULT_INPUTS["stock"])
    parser.add_argument("--market-csv", type=Path, default=DEFAULT_INPUTS["market"])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--only", choices=["stock", "market", "all"], default="all")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    api_key = os.environ.get("UPSTAGE_API_KEY", "").strip()
    if not api_key or api_key == "XXXXXXX":
        raise RuntimeError("Set UPSTAGE_API_KEY in .env before running embeddings.")

    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("UPSTAGE_BASE_URL", "https://api.upstage.ai/v1"),
    )
    model = os.environ.get("UPSTAGE_EMBEDDING_MODEL", "embedding-passage")

    jobs = []
    if args.only in ("stock", "all"):
        jobs.append(("stock", args.stock_csv, args.output_dir / "stock_news_embeddings.jsonl"))
    if args.only in ("market", "all"):
        jobs.append(("market", args.market_csv, args.output_dir / "market_news_embeddings.jsonl"))

    for source_type, csv_path, output_path in jobs:
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        embed_csv(
            client,
            csv_path,
            output_path,
            source_type,
            model,
            args.batch_size,
            args.sleep_seconds,
            args.log_every,
            args.max_retries,
        )


if __name__ == "__main__":
    main()
