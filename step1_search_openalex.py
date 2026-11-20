"""

Step 1: 用 OpenAlex API 搜尋目標金屬相關電化學文獻，存成 CSV

"""

import time

import requests

import pandas as pd

from tqdm import tqdm

from config import (

    MAILTO, OPENALEX_API_KEY, SEARCH_QUERIES, MAX_RECORDS_PER_QUERY,

    FROM_YEAR, REQUEST_TIMEOUT, STEP1_OUTPUT

)

BASE_URL = "https://api.openalex.org/works"


def invert_abstract(inv_index: dict) -> str:

    if not inv_index:

        return ""

    word_positions = []

    for word, positions in inv_index.items():

        for pos in positions:

            word_positions.append((pos, word))

    word_positions.sort(key=lambda x: x[0])

    return " ".join(w for _, w in word_positions)


def request_with_retry(url, params, max_retries=5):

    for attempt in range(max_retries):

        try:

            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)  # ✅ 改用 config

            if resp.status_code == 429:

                retry_after = resp.headers.get("Retry-After")
                try:
                    msg = resp.json().get("message", "")
                except ValueError:
                    msg = resp.text[:200]

                retry_sec = int(retry_after) if retry_after and retry_after.isdigit() else None
                if "insufficient budget" in msg.lower() or (retry_sec is not None and retry_sec > 300):
                    hours = f"{retry_sec / 3600:.1f} 小時" if retry_sec else "UTC 午夜"
                    raise RuntimeError(
                        f"OpenAlex 每日配額已耗盡（約 {hours} 後重置）。"
                        f"請確認 config.py 的 OPENALEX_API_KEY 已正確填入。"
                    )

                wait = retry_sec if retry_sec else (30 + attempt * 30)

                print(f"  429 rate limit, waiting {wait}s ...")

                time.sleep(wait)

                continue

            if 500 <= resp.status_code < 600:

                wait = 10 + attempt * 10

                print(f"  Server error {resp.status_code}, waiting {wait}s ...")

                time.sleep(wait)

                continue

            resp.raise_for_status()

            return resp

        except requests.exceptions.RequestException as e:

            wait = 10 + attempt * 10

            print(f"  Request failed: {e}, waiting {wait}s ...")

            time.sleep(wait)

    raise RuntimeError("Max retries exceeded for OpenAlex request")


def fetch_single_query(search_query: str, max_records: int) -> list[dict]:

    records = []

    cursor = "*"

    pbar = tqdm(total=max_records, desc=f"  {search_query[:50]}", leave=False)

    while len(records) < max_records:

        params = {

            "search": search_query,

            "filter": f"from_publication_date:{FROM_YEAR}-01-01,type:article",  # ✅ 改用 config

            "per-page": 100,

            "cursor": cursor,

            "mailto": MAILTO,

            "select": (

                "id,doi,title,publication_year,primary_location,"

                "best_oa_location,open_access,"

                "abstract_inverted_index,cited_by_count,authorships"

            ),

        }

        if OPENALEX_API_KEY:

            params["api_key"] = OPENALEX_API_KEY

        resp = request_with_retry(BASE_URL, params)

        data = resp.json()

        results = data.get("results", [])

        if not results:

            break

        for item in results:

            primary = item.get("primary_location") or {}

            source = primary.get("source") or {}

            best_oa = item.get("best_oa_location") or {}

            oa = item.get("open_access") or {}

            authorships = item.get("authorships") or []

            authors = []

            for a in authorships:

                name = (a.get("author") or {}).get("display_name")

                if name:

                    authors.append(name)

            records.append({

                "query": search_query,

                "openalex_id": item.get("id"),

                "doi": item.get("doi"),

                "title": item.get("title"),

                "year": item.get("publication_year"),

                "journal": source.get("display_name"),

                "is_oa": oa.get("is_oa"),

                "oa_status": oa.get("oa_status"),

                "oa_url": oa.get("oa_url"),

                "best_pdf_url": best_oa.get("pdf_url"),

                "primary_pdf_url": primary.get("pdf_url"),

                "cited_by_count": item.get("cited_by_count"),

                "authors": "; ".join(authors[:5]),

                "abstract": invert_abstract(

                    item.get("abstract_inverted_index")

                ),

            })

            if len(records) >= max_records:

                break

        pbar.update(len(results))

        cursor = data.get("meta", {}).get("next_cursor")

        if not cursor:

            break

        time.sleep(0.5)

    pbar.close()

    return records


def run_search():

    all_records = []

    for q in SEARCH_QUERIES:

        print(f"Searching: {q}")

        recs = fetch_single_query(q, MAX_RECORDS_PER_QUERY)

        print(f"  Found {len(recs)} records")

        all_records.extend(recs)

    df = pd.DataFrame(all_records)

    before = len(df)

    df = df.drop_duplicates(subset=["openalex_id"])

    print(f"\nTotal: {before} -> deduplicated: {len(df)}")

    df = df.dropna(subset=["doi"])

    print(f"With DOI: {len(df)}")

    df.to_csv(STEP1_OUTPUT, index=False, encoding="utf-8-sig")  # ✅ 改用 config

    print(f"✅ Saved to {STEP1_OUTPUT}")

    return df

if __name__ == "__main__":

    run_search()
