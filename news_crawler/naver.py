from __future__ import annotations

import re
import time
from datetime import date, timedelta
from typing import Any

import requests
from bs4 import BeautifulSoup


NAVER_SEARCH_URL = "https://search.naver.com/search.naver"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.7",
}


def parse_display_date(raw: str, *, reference_date: date) -> date | None:
    text = " ".join(raw.split())
    absolute = re.search(r"(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})", text)
    if absolute:
        try:
            return date(*(int(value) for value in absolute.groups()))
        except ValueError:
            return None

    if any(token in text for token in ("방금", "초 전", "분 전", "시간 전")):
        return reference_date
    if "어제" in text:
        return reference_date - timedelta(days=1)

    relative = re.search(r"(\d+)\s*일\s*전", text)
    if relative:
        return reference_date - timedelta(days=int(relative.group(1)))
    relative = re.search(r"(\d+)\s*주\s*전", text)
    if relative:
        return reference_date - timedelta(weeks=int(relative.group(1)))
    return None


def parse_search_results(
    html: str,
    *,
    target_date: date,
    max_results: int,
    reference_date: date | None = None,
) -> list[dict[str, str]]:
    relative_date_reference = reference_date or date.today()
    soup = BeautifulSoup(html, "html.parser")
    titles = soup.select('a[data-heatmap-target=".tit"]')
    profiles = soup.select('[data-sds-comp="Profile"]')
    records: list[dict[str, str]] = []
    seen: set[str] = set()

    for title_element, profile_element in zip(titles, profiles):
        title = " ".join(title_element.get_text(" ", strip=True).split())
        if not title:
            continue
        dedup_key = title.casefold()
        if dedup_key in seen:
            continue

        profile_lines = [
            " ".join(value.split())
            for value in profile_element.stripped_strings
            if value.strip()
        ]
        press = profile_lines[0] if profile_lines else ""
        parsed_date = next(
            (
                value
                for line in profile_lines[1:]
                if (
                    value := parse_display_date(
                        line,
                        reference_date=relative_date_reference,
                    )
                )
                is not None
            ),
            target_date,
        )
        if parsed_date != target_date:
            continue

        seen.add(dedup_key)
        records.append(
            {
                "date": target_date.strftime("%Y.%m.%d"),
                "title": title,
                "press": press,
                "source": "naver",
            }
        )
        if len(records) >= max_results:
            break
    return records


class NaverNewsClient:
    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        timeout_seconds: float = 15.0,
        retries: int = 2,
    ) -> None:
        self.session = session or requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    def search(
        self,
        query: str,
        *,
        target_date: date,
        max_results: int,
    ) -> list[dict[str, str]]:
        date_token = target_date.strftime("%Y%m%d")
        params = {
            "where": "news",
            "query": query,
            "start": 1,
            "nso": f"so:dd,p:from{date_token}to{date_token}",
        }
        response: Any = None
        for attempt in range(self.retries + 1):
            try:
                response = self.session.get(
                    NAVER_SEARCH_URL,
                    params=params,
                    timeout=self.timeout_seconds,
                )
                if response.status_code in (403, 429):
                    raise RuntimeError(
                        f"Naver search blocked the crawler with HTTP {response.status_code}; "
                        "stop and retry later."
                    )
                response.raise_for_status()
                break
            except RuntimeError:
                raise
            except requests.RequestException:
                if attempt >= self.retries:
                    raise
                time.sleep(1.5 * (attempt + 1))

        return parse_search_results(
            response.text,
            target_date=target_date,
            max_results=max_results,
            reference_date=date.today(),
        )
