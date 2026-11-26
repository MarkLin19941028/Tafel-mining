"""

Step 1: 用 OpenAlex API 搜尋目標金屬相關電化學文獻，存成 CSV

"""

import math
import os
import time

import requests

import pandas as pd

from tqdm import tqdm

from config import (
    ENABLE_EUROPEPMC_XML,
    ENABLE_OPENALEX,
    ENABLE_SEMANTIC_SCHOLAR,
    EUROPEPMC_MAX_PAPERS,
    FROM_YEAR,
    MAILTO,
    MAX_RECORDS_PER_QUERY,
    OPENALEX_API_KEY,
    REQUEST_TIMEOUT,
    REQUIRE_TARGET_IN_TITLE_ABSTRACT,
    SEARCH_QUERIES,
    STEP1_CURSOR_STATE_PATH,
    STEP1_EARLY_STOP_EMPTY_PAGES,
    STEP1_INFLIGHT_DEDUP,
    STEP1_MAX_SCAN_MULTIPLIER,
    STEP1_OUTPUT,
    build_target_regex,
)
from record_keys import (
    is_title_abstract_relevant,
    make_record_key,
    normalize_doi_key,
    normalize_pmcid,
    save_csv_atomic,
)
from step1_cursor_state import Step1CursorStore, make_cursor_key

BASE_URL = "https://api.openalex.org/works"
_SESSION = requests.Session()


class OpenAlexQuotaExhaustedError(RuntimeError):
    """OpenAlex 每日配額耗盡時拋出，供上層安全保存游標並降級續跑其餘資料源。"""


class OpenAlexInvalidCursorError(RuntimeError):
    """OpenAlex 拒收既有跨次游標（HTTP 400）時拋出，供查詢函式自動退回從頭掃描。"""


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

            resp = _SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)  # ✅ 複用連線池

            if resp.status_code == 429:

                retry_after = resp.headers.get("Retry-After")
                try:
                    msg = resp.json().get("message", "")
                except ValueError:
                    msg = resp.text[:200]

                retry_sec = int(retry_after) if retry_after and retry_after.isdigit() else None
                if "insufficient budget" in msg.lower() or (retry_sec is not None and retry_sec > 300):
                    hours = f"{retry_sec / 3600:.1f} 小時" if retry_sec else "UTC 午夜"
                    raise OpenAlexQuotaExhaustedError(
                        f"OpenAlex 每日配額已耗盡（約 {hours} 後重置）。"
                        f"請確認專案根目錄 .env 的 OPENALEX_API_KEY 已正確填入。"
                    )

                wait = retry_sec if retry_sec else (30 + attempt * 30)

                print(f"  429 rate limit, waiting {wait}s ...")

                time.sleep(wait)

                continue

            if resp.status_code == 400:
                if params.get("cursor") and params.get("cursor") != "*":
                    raise OpenAlexInvalidCursorError(
                        f"OpenAlex rejected cursor (HTTP 400): {resp.text[:200]}"
                    )
                raise RuntimeError(f"OpenAlex HTTP 400: {resp.text[:200]}")

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


def _is_missing_http_url(val) -> bool:
    if val is None or pd.isna(val):
        return True
    return not str(val).strip().startswith("http")


def _is_missing_text(val) -> bool:
    if val is None or pd.isna(val):
        return True
    return not str(val).strip()


def _record_missing_fields(best_pdf_url, abstract, pmcid) -> set[str]:
    missing: set[str] = set()
    if _is_missing_http_url(best_pdf_url):
        missing.add("best_pdf_url")
    if _is_missing_text(abstract):
        missing.add("abstract")
    if not normalize_pmcid(pmcid):
        missing.add("pmcid")
    return missing


def _load_existing_step1_index(
    step1_path: str,
) -> tuple[pd.DataFrame | None, set[str], dict[str, set[str]]]:
    """
    載入既有的 Step 1 CSV，建立：
    - known_keys: 所有既有論文的統一去重鍵（DOI / PMCID，不論下載成功或失敗）
    - needs_backfill: 缺少 best_pdf_url / abstract / pmcid 的既有鍵對應缺漏欄位集合
    """
    if not os.path.exists(step1_path):
        return None, set(), {}

    existing_df = pd.read_csv(step1_path)
    known_keys: set[str] = set()
    needs_backfill: dict[str, set[str]] = {}

    if existing_df.empty:
        return existing_df, known_keys, needs_backfill

    dois = existing_df["doi"] if "doi" in existing_df.columns else [None] * len(existing_df)
    pmcids = existing_df["pmcid"] if "pmcid" in existing_df.columns else [None] * len(existing_df)
    pdf_urls = existing_df["best_pdf_url"] if "best_pdf_url" in existing_df.columns else [None] * len(existing_df)
    abstracts = existing_df["abstract"] if "abstract" in existing_df.columns else [None] * len(existing_df)

    for d, p, u, a in zip(dois, pmcids, pdf_urls, abstracts):
        key = make_record_key(d, p)
        if not key:
            continue
        known_keys.add(key)
        missing = _record_missing_fields(u, a, p)
        if missing:
            needs_backfill[key] = missing

    return existing_df, known_keys, needs_backfill


def _build_openalex_record(item: dict, search_query: str) -> dict:
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

    return {
        "query": search_query,
        "openalex_id": item.get("id"),
        "s2_paper_id": None,
        "pmcid": None,
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
        "abstract": invert_abstract(item.get("abstract_inverted_index")),
    }


def fetch_single_query(
    search_query: str,
    max_records: int,
    known_keys: set[str] | None = None,
    needs_backfill: dict[str, set[str]] | None = None,
    max_scan_multiplier: int | None = None,
    early_stop_empty_pages: int | None = None,
    target_pat=None,
    start_cursor: str = "*",
    cursor_out: dict | None = None,
) -> list[dict]:
    if max_records <= 0:
        return []

    scan_mult = max(1, int(STEP1_MAX_SCAN_MULTIPLIER if max_scan_multiplier is None else max_scan_multiplier))
    early_stop = max(0, int(STEP1_EARLY_STOP_EMPTY_PAGES if early_stop_empty_pages is None else early_stop_empty_pages))
    inflight = known_keys is not None

    # OpenAlex 官方支援 per-page 最高 200；即時去重模式下一次抓 200 筆可將翻頁與延遲減半
    per_page = 200 if inflight else min(200, max(100, max_records))
    max_pages = max(1, math.ceil(max_records / per_page) * scan_mult)

    records: list[dict] = []
    new_count = 0
    skipped_dup = 0
    skipped_irrel = 0
    backfill_count = 0
    pages_scanned = 0
    empty_pages = 0
    cursor = start_cursor or "*"
    next_cursor_to_save: str | None = cursor if cursor != "*" else None
    exhausted = False

    pbar = tqdm(total=max_records, desc=f"  {search_query[:50]}", leave=False)

    try:
        while new_count < max_records:
            params = {
                "search": search_query,
                "filter": f"from_publication_date:{FROM_YEAR}-01-01,type:article",
                "per-page": per_page,
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

            try:
                resp = request_with_retry(BASE_URL, params)
            except OpenAlexInvalidCursorError:
                if cursor != "*" and pages_scanned == 0:
                    print("  ⚠️  OpenAlex 既有游標已失效，自動重設為從頭掃描 (*)")
                    cursor = "*"
                    next_cursor_to_save = None
                    continue
                raise

            data = resp.json()
            results = data.get("results", [])

            if not results:
                exhausted = True
                next_cursor_to_save = None
                break

            page_new = 0
            for item in results:
                key = make_record_key(item.get("doi"))
                if not key:
                    continue

                if inflight and key in known_keys:
                    skipped_dup += 1
                    missing = needs_backfill.get(key) if needs_backfill is not None else None
                    if missing:
                        best_oa = item.get("best_oa_location") or {}
                        best_pdf_url = best_oa.get("pdf_url")
                        can_fill_pdf = ("best_pdf_url" in missing) and (not _is_missing_http_url(best_pdf_url))
                        can_fill_abs = ("abstract" in missing) and bool(item.get("abstract_inverted_index"))
                        if can_fill_pdf or can_fill_abs:
                            records.append(_build_openalex_record(item, search_query))
                            backfill_count += 1
                            if can_fill_pdf:
                                missing.discard("best_pdf_url")
                            if can_fill_abs:
                                missing.discard("abstract")
                            if not missing:
                                needs_backfill.pop(key, None)
                    continue

                rec = _build_openalex_record(item, search_query)

                # 若同時具備標題與摘要但兩者皆未提及目標金屬，直接略過以免吃掉 MAX_RECORDS_PER_QUERY 配額
                if not is_title_abstract_relevant(
                    rec.get("title"),
                    rec.get("abstract"),
                    target_pat,
                    require_both_present=True,
                ):
                    skipped_irrel += 1
                    continue

                records.append(rec)
                new_count += 1
                page_new += 1
                pbar.update(1)

                if inflight:
                    known_keys.add(key)
                    if needs_backfill is not None:
                        missing_new = _record_missing_fields(rec["best_pdf_url"], rec["abstract"], rec["pmcid"])
                        if missing_new:
                            needs_backfill[key] = missing_new

                if new_count >= max_records:
                    break

            pages_scanned += 1
            empty_pages = (empty_pages + 1) if page_new == 0 else 0

            next_cursor = data.get("meta", {}).get("next_cursor")
            if not next_cursor:
                exhausted = True
                next_cursor_to_save = None
            else:
                next_cursor_to_save = next_cursor

            if new_count >= max_records:
                break

            if inflight:
                if early_stop > 0 and empty_pages >= early_stop:
                    break
                if pages_scanned >= max_pages:
                    break

            if not next_cursor:
                break

            cursor = next_cursor
            time.sleep(0.5)
    finally:
        pbar.close()
        if cursor_out is not None:
            cursor_out["next_cursor"] = None if exhausted else next_cursor_to_save
            cursor_out["exhausted"] = exhausted

    if inflight:
        bf_msg = f", backfill {backfill_count}" if backfill_count else ""
        irrel_msg = f", skipped {skipped_irrel} irrel" if skipped_irrel else ""
        print(f"  Found {new_count} new (skipped {skipped_dup} dup{irrel_msg}{bf_msg}, {pages_scanned} pages)")
    else:
        irrel_msg = f" (skipped {skipped_irrel} irrel)" if skipped_irrel else ""
        print(f"  Found {new_count} records{irrel_msg}")

    return records


def _attach_record_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "doi" not in out.columns:
        out["doi"] = None
    if "pmcid" not in out.columns:
        out["pmcid"] = None
    out["_rec_key"] = [
        make_record_key(d, p)
        for d, p in zip(out["doi"], out["pmcid"])
    ]
    out = out[out["_rec_key"] != ""].drop_duplicates(subset=["_rec_key"], keep="first")
    return out


def _append_source_tag(existing_tag, new_tag: str) -> str:
    parts = [p.strip() for p in str(existing_tag or "").split("+") if p.strip()]
    if new_tag not in parts:
        parts.append(new_tag)
    return "+".join(parts)


def _merge_source_into(
    base: pd.DataFrame,
    incoming: pd.DataFrame,
    tag: str,
    id_col: str | None = None,
    fill_pdf_url: bool = False,
) -> tuple[pd.DataFrame, int, int]:
    """將單一來源 DataFrame 依 _rec_key 併入 base，回傳 (merged, added_count, overlap_count)。"""
    if incoming.empty:
        return base, 0, 0
    if base.empty:
        return incoming.copy(), len(incoming), 0

    inc_by_key = incoming.set_index("_rec_key").to_dict(orient="index")
    base_keys = set(base["_rec_key"])
    if id_col and id_col not in base.columns:
        base[id_col] = None

    overlap = 0
    for idx, row in base.iterrows():
        k = row["_rec_key"]
        if k not in inc_by_key:
            continue
        overlap += 1
        irow = inc_by_key[k]
        base.at[idx, "source_api"] = _append_source_tag(row.get("source_api"), tag)
        if id_col:
            existing_id = normalize_pmcid(row.get(id_col)) if id_col == "pmcid" else row.get(id_col)
            incoming_id = normalize_pmcid(irow.get(id_col)) if id_col == "pmcid" else irow.get(id_col)
            if (not existing_id or pd.isna(existing_id)) and incoming_id:
                base.at[idx, id_col] = incoming_id
        if fill_pdf_url:
            if (pd.isna(row.get("best_pdf_url")) or not str(row.get("best_pdf_url")).startswith("http")) and irow.get("best_pdf_url"):
                base.at[idx, "best_pdf_url"] = irow.get("best_pdf_url")
        if (pd.isna(row.get("abstract")) or not str(row.get("abstract")).strip()) and irow.get("abstract"):
            base.at[idx, "abstract"] = irow.get("abstract")

    new_only = incoming[~incoming["_rec_key"].isin(base_keys)].copy()
    return pd.concat([base, new_only], ignore_index=True), len(new_only), overlap


def _merge_sources(
    oa_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    epmc_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """依統一鍵（DOI / PMCID）合併 OpenAlex、Semantic Scholar 與 Europe PMC 結果，互補摘要、PDF URL 與 PMCID。"""
    epmc_df = pd.DataFrame() if epmc_df is None else epmc_df

    oa_clean = _attach_record_key(oa_df) if not oa_df.empty else pd.DataFrame()
    s2_clean = _attach_record_key(s2_df) if not s2_df.empty else pd.DataFrame()
    epmc_clean = _attach_record_key(epmc_df) if not epmc_df.empty else pd.DataFrame()

    if oa_clean.empty and s2_clean.empty and epmc_clean.empty:
        return pd.DataFrame()

    base = oa_clean.copy()
    base, s2_added, s2_overlap = _merge_source_into(
        base, s2_clean, tag="semantic_scholar", id_col="s2_paper_id", fill_pdf_url=True
    )
    base, epmc_added, epmc_overlap = _merge_source_into(
        base, epmc_clean, tag="europepmc", id_col="pmcid", fill_pdf_url=False
    )

    merged = base.drop(columns=["_rec_key"], errors="ignore")
    print(
        f"\n[Step 1 Merge] OpenAlex: {len(oa_clean)} | "
        f"S2 新增: {s2_added} (重疊 {s2_overlap}) | "
        f"Europe PMC 新增: {epmc_added} (重疊 {epmc_overlap}) | "
        f"合計: {len(merged)}"
    )
    return merged


def _merge_with_existing_step1(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """增量補抓模式：依統一鍵（DOI / PMCID）合併既有 Step 1 CSV 與新檢索結果，保留既有資料並補入新論文。"""
    if existing_df.empty:
        return new_df
    if new_df.empty:
        return existing_df

    ex = _attach_record_key(existing_df)
    nw = _attach_record_key(new_df)

    ex_keys = set(ex["_rec_key"])
    nw_by_key = nw.set_index("_rec_key").to_dict(orient="index")

    for idx, row in ex.iterrows():
        k = row["_rec_key"]
        if k in nw_by_key:
            nrow = nw_by_key[k]
            if (pd.isna(row.get("best_pdf_url")) or not str(row.get("best_pdf_url")).startswith("http")) and nrow.get("best_pdf_url"):
                ex.at[idx, "best_pdf_url"] = nrow.get("best_pdf_url")
            if (pd.isna(row.get("abstract")) or not str(row.get("abstract")).strip()) and nrow.get("abstract"):
                ex.at[idx, "abstract"] = nrow.get("abstract")
            if not normalize_pmcid(row.get("pmcid")) and normalize_pmcid(nrow.get("pmcid")):
                ex.at[idx, "pmcid"] = normalize_pmcid(nrow.get("pmcid"))

    added = nw[~nw["_rec_key"].isin(ex_keys)].copy()
    merged = pd.concat([ex, added], ignore_index=True).drop(columns=["_rec_key"], errors="ignore")
    print(
        f"[Step 1 Append] 既有: {len(ex)} | 本次搜尋唯一論文: {len(nw)} "
        f"| 新增論文: {len(added)} | 合併後合計: {len(merged)}"
    )
    return merged


def _mark_batch_oa_quota_exhausted() -> None:
    """若由 run_all 批次模式啟動，寫入配額耗盡標記檔，讓後續金屬自動跳過 OpenAlex。"""
    marker_path = os.environ.get("TAFEL_OA_QUOTA_MARKER", "").strip()
    if not marker_path:
        return
    try:
        with open(marker_path, "w", encoding="utf-8") as f:
            f.write("exhausted\n")
    except OSError:
        pass


def run_search(append: bool | None = None, skip_openalex: bool | None = None):
    if append is None:
        append = os.environ.get("TAFEL_APPEND") == "1"
    if skip_openalex is None:
        skip_openalex = os.environ.get("TAFEL_SKIP_OPENALEX") == "1"

    existing_df: pd.DataFrame | None = None
    known_keys: set[str] | None = set() if STEP1_INFLIGHT_DEDUP else None
    needs_backfill: dict[str, set[str]] | None = {} if STEP1_INFLIGHT_DEDUP else None
    cursor_store = Step1CursorStore(STEP1_CURSOR_STATE_PATH)

    if append and os.path.exists(STEP1_OUTPUT):
        try:
            existing_df, ex_keys, ex_backfill = _load_existing_step1_index(STEP1_OUTPUT)
            if STEP1_INFLIGHT_DEDUP and known_keys is not None and needs_backfill is not None:
                known_keys.update(ex_keys)
                needs_backfill.update(ex_backfill)
                print(
                    f"[Step 1 In-Flight Dedup] 已載入既有索引：{len(ex_keys)} 篇論文"
                    f"（待回補缺漏欄位：{len(ex_backfill)} 篇；"
                    f"護欄：max_scan={STEP1_MAX_SCAN_MULTIPLIER}x, early_stop={STEP1_EARLY_STOP_EMPTY_PAGES} 空頁）"
                )
        except Exception as e:
            print(f"  ⚠️ 無法讀取既有 Step 1 檔案建立索引 ({e})，改以一般搜尋執行。")
            existing_df = None
    elif STEP1_INFLIGHT_DEDUP:
        print(
            f"[Step 1 In-Flight Dedup] 啟用本次跨查詢／跨來源即時去重"
            f"（護欄：max_scan={STEP1_MAX_SCAN_MULTIPLIER}x, early_stop={STEP1_EARLY_STOP_EMPTY_PAGES} 空頁）"
        )

    target_pat = build_target_regex() if REQUIRE_TARGET_IN_TITLE_ABSTRACT else None
    all_records = []
    interrupted_err: BaseException | None = None
    oa_enabled = ENABLE_OPENALEX and (not skip_openalex)

    if oa_enabled:
        skipped_exhausted = 0
        resumed_queries = 0
        try:
            for q in SEARCH_QUERIES:
                q_key = make_cursor_key("openalex", q, FROM_YEAR)
                start_cursor = "*"
                if append:
                    saved_cursor, is_exhausted = cursor_store.get(q_key)
                    if is_exhausted:
                        skipped_exhausted += 1
                        continue
                    if saved_cursor:
                        start_cursor = saved_cursor
                        resumed_queries += 1

                resume_note = " [續頁]" if start_cursor != "*" else ""
                print(f"Searching{resume_note}: {q}")
                cursor_box: dict = {}
                try:
                    recs = fetch_single_query(
                        q,
                        MAX_RECORDS_PER_QUERY,
                        known_keys=known_keys,
                        needs_backfill=needs_backfill,
                        target_pat=target_pat,
                        start_cursor=start_cursor,
                        cursor_out=cursor_box,
                    )
                    all_records.extend(recs)
                finally:
                    if cursor_box:
                        cursor_store.update(
                            q_key,
                            cursor_box.get("next_cursor"),
                            bool(cursor_box.get("exhausted", False)),
                        )
        except OpenAlexQuotaExhaustedError as e:
            _mark_batch_oa_quota_exhausted()
            print(
                f"\n⚠️  [Step 1] {e}\n"
                f"   已安全保存目前 OpenAlex 取得之 {len(all_records)} 筆記錄與游標進度，"
                f"將自動降級繼續執行 Semantic Scholar 與 Europe PMC..."
            )
        except (KeyboardInterrupt, RuntimeError) as e:
            interrupted_err = e
            print(f"\n⚠️  [Step 1] OpenAlex 檢索中途停止（{e}），將先安全保存目前已取得之 {len(all_records)} 筆記錄...")

        if skipped_exhausted or resumed_queries:
            print(
                f"[OpenAlex Cursor] 續頁查詢: {resumed_queries} 組 | "
                f"已見底略過: {skipped_exhausted} 組（0 API 請求）"
            )
    else:
        print("[OpenAlex] 已略過（ENABLE_OPENALEX=False 或 --skip-openalex）")

    oa_df = pd.DataFrame(all_records)
    if not oa_df.empty:
        before = len(oa_df)
        oa_df = oa_df.drop_duplicates(subset=["openalex_id"])
        print(f"\n[OpenAlex] Total: {before} -> deduplicated: {len(oa_df)}")
        oa_df = oa_df.dropna(subset=["doi"])
        print(f"[OpenAlex] With DOI: {len(oa_df)}")

    s2_df = pd.DataFrame()
    epmc_df = pd.DataFrame()

    if interrupted_err is None:
        if ENABLE_SEMANTIC_SCHOLAR:
            from semantic_scholar_fetcher import search_semantic_scholar

            s2_recs = search_semantic_scholar(
                SEARCH_QUERIES,
                known_keys=known_keys,
                needs_backfill=needs_backfill,
                append=append,
            )
            if s2_recs:
                s2_df = pd.DataFrame(s2_recs).dropna(subset=["doi"])

        if ENABLE_EUROPEPMC_XML and EUROPEPMC_MAX_PAPERS > 0:
            from europepmc_fetcher import build_europepmc_query, search_europepmc_metadata

            epmc_query = build_europepmc_query()
            epmc_key = make_cursor_key("europepmc", epmc_query, FROM_YEAR)
            epmc_start = "*"
            epmc_exhausted = False
            if append:
                saved_epmc, epmc_exhausted = cursor_store.get(epmc_key)
                if saved_epmc and not epmc_exhausted:
                    epmc_start = saved_epmc

            if append and epmc_exhausted:
                print("\n[Europe PMC] 既有查詢已掃描完畢（exhausted），略過重複請求（0 API 請求）。")
            else:
                epmc_cursor_box: dict = {}
                epmc_recs = search_europepmc_metadata(
                    known_keys=known_keys,
                    needs_backfill=needs_backfill,
                    start_cursor=epmc_start,
                    cursor_out=epmc_cursor_box,
                )
                if epmc_cursor_box:
                    cursor_store.update(
                        epmc_key,
                        epmc_cursor_box.get("next_cursor"),
                        bool(epmc_cursor_box.get("exhausted", False)),
                    )
                if epmc_recs:
                    epmc_df = pd.DataFrame(epmc_recs)

    df = _merge_sources(oa_df, s2_df, epmc_df)

    if (append or interrupted_err is not None) and os.path.exists(STEP1_OUTPUT):
        try:
            if existing_df is None:
                existing_df = pd.read_csv(STEP1_OUTPUT)
            df = _merge_with_existing_step1(existing_df, df)
        except Exception as e:
            print(f"  ⚠️ 無法讀取既有 Step 1 檔案進行合併 ({e})，改以本次結果輸出。")

    if not df.empty:
        save_csv_atomic(df, STEP1_OUTPUT)
        print(f"✅ Saved to {STEP1_OUTPUT} ({len(df)} papers)")

    if interrupted_err is not None:
        raise interrupted_err

    return df

if __name__ == "__main__":

    run_search()
