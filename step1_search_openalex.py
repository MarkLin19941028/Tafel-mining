"""

Step 1: 用 OpenAlex API 搜尋目標金屬相關電化學文獻，存成 CSV

"""

import os
import time

import requests

import pandas as pd

from tqdm import tqdm

from config import (
    ENABLE_SEMANTIC_SCHOLAR,
    FROM_YEAR,
    MAILTO,
    MAX_RECORDS_PER_QUERY,
    OPENALEX_API_KEY,
    REQUEST_TIMEOUT,
    SEARCH_QUERIES,
    STEP1_OUTPUT,
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

                "s2_paper_id": None,

                "source_api": "openalex",

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


def _normalize_doi_key(doi) -> str:
    if pd.isna(doi):
        return ""
    return (
        str(doi)
        .replace("https://doi.org/", "")
        .replace("http://doi.org/", "")
        .strip()
        .lower()
    )


def _merge_sources(oa_df: pd.DataFrame, s2_df: pd.DataFrame) -> pd.DataFrame:
    """依標準化 DOI 合併 OpenAlex 與 Semantic Scholar 結果，互補摘要與 PDF URL。"""
    if oa_df.empty and s2_df.empty:
        return pd.DataFrame()
    if s2_df.empty:
        return oa_df
    if oa_df.empty:
        return s2_df

    oa_df = oa_df.copy()
    s2_df = s2_df.copy()
    oa_df["_doi_key"] = oa_df["doi"].apply(_normalize_doi_key)
    s2_df["_doi_key"] = s2_df["doi"].apply(_normalize_doi_key)

    oa_df = oa_df[oa_df["_doi_key"] != ""].drop_duplicates(subset=["_doi_key"], keep="first")
    s2_df = s2_df[s2_df["_doi_key"] != ""].drop_duplicates(subset=["_doi_key"], keep="first")

    s2_by_doi = s2_df.set_index("_doi_key").to_dict(orient="index")
    oa_keys = set(oa_df["_doi_key"])

    # 若兩邊皆有同一 DOI，補齊缺失欄位並標記 source_api
    for idx, row in oa_df.iterrows():
        k = row["_doi_key"]
        if k in s2_by_doi:
            s2_row = s2_by_doi[k]
            oa_df.at[idx, "source_api"] = "openalex+semantic_scholar"
            oa_df.at[idx, "s2_paper_id"] = s2_row.get("s2_paper_id")
            if (pd.isna(row.get("best_pdf_url")) or not str(row.get("best_pdf_url")).startswith("http")) and s2_row.get("best_pdf_url"):
                oa_df.at[idx, "best_pdf_url"] = s2_row.get("best_pdf_url")
            if (pd.isna(row.get("abstract")) or not str(row.get("abstract")).strip()) and s2_row.get("abstract"):
                oa_df.at[idx, "abstract"] = s2_row.get("abstract")

    # 加入僅存在於 Semantic Scholar 的新論文
    s2_only = s2_df[~s2_df["_doi_key"].isin(oa_keys)].copy()
    merged = pd.concat([oa_df, s2_only], ignore_index=True)
    merged = merged.drop(columns=["_doi_key"], errors="ignore")
    print(
        f"\n[Step 1 Merge] OpenAlex: {len(oa_df)} | Semantic Scholar 新增: {len(s2_only)} "
        f"| 跨來源重疊: {len(s2_df) - len(s2_only)} | 合計: {len(merged)}"
    )
    return merged


def _merge_with_existing_step1(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """增量補抓模式：依標準化 DOI 合併既有 Step 1 CSV 與新檢索結果，保留既有資料並補入新論文。"""
    if existing_df.empty:
        return new_df
    if new_df.empty:
        return existing_df

    ex = existing_df.copy()
    nw = new_df.copy()
    ex["_doi_key"] = ex["doi"].apply(_normalize_doi_key)
    nw["_doi_key"] = nw["doi"].apply(_normalize_doi_key)

    ex = ex[ex["_doi_key"] != ""].drop_duplicates(subset=["_doi_key"], keep="first")
    nw = nw[nw["_doi_key"] != ""].drop_duplicates(subset=["_doi_key"], keep="first")

    ex_keys = set(ex["_doi_key"])
    nw_by_doi = nw.set_index("_doi_key").to_dict(orient="index")

    for idx, row in ex.iterrows():
        k = row["_doi_key"]
        if k in nw_by_doi:
            nrow = nw_by_doi[k]
            if (pd.isna(row.get("best_pdf_url")) or not str(row.get("best_pdf_url")).startswith("http")) and nrow.get("best_pdf_url"):
                ex.at[idx, "best_pdf_url"] = nrow.get("best_pdf_url")
            if (pd.isna(row.get("abstract")) or not str(row.get("abstract")).strip()) and nrow.get("abstract"):
                ex.at[idx, "abstract"] = nrow.get("abstract")

    added = nw[~nw["_doi_key"].isin(ex_keys)].copy()
    merged = pd.concat([ex, added], ignore_index=True).drop(columns=["_doi_key"], errors="ignore")
    print(
        f"[Step 1 Append] 既有: {len(ex)} | 本次搜尋唯一 DOI: {len(nw)} "
        f"| 新增論文: {len(added)} | 合併後合計: {len(merged)}"
    )
    return merged


def run_search(append: bool | None = None):
    if append is None:
        append = os.environ.get("TAFEL_APPEND") == "1"

    all_records = []

    for q in SEARCH_QUERIES:

        print(f"Searching: {q}")

        recs = fetch_single_query(q, MAX_RECORDS_PER_QUERY)

        print(f"  Found {len(recs)} records")

        all_records.extend(recs)

    oa_df = pd.DataFrame(all_records)
    if not oa_df.empty:
        before = len(oa_df)
        oa_df = oa_df.drop_duplicates(subset=["openalex_id"])
        print(f"\n[OpenAlex] Total: {before} -> deduplicated: {len(oa_df)}")
        oa_df = oa_df.dropna(subset=["doi"])
        print(f"[OpenAlex] With DOI: {len(oa_df)}")

    s2_df = pd.DataFrame()
    if ENABLE_SEMANTIC_SCHOLAR:
        from semantic_scholar_fetcher import search_semantic_scholar

        s2_recs = search_semantic_scholar(SEARCH_QUERIES)
        if s2_recs:
            s2_df = pd.DataFrame(s2_recs).dropna(subset=["doi"])

    df = _merge_sources(oa_df, s2_df)

    if append and os.path.exists(STEP1_OUTPUT):
        try:
            existing_df = pd.read_csv(STEP1_OUTPUT)
            df = _merge_with_existing_step1(existing_df, df)
        except Exception as e:
            print(f"  ⚠️ 無法讀取既有 Step 1 檔案進行合併 ({e})，改以本次結果輸出。")

    df.to_csv(STEP1_OUTPUT, index=False, encoding="utf-8-sig")

    print(f"✅ Saved to {STEP1_OUTPUT} ({len(df)} papers)")

    return df

if __name__ == "__main__":

    run_search()
