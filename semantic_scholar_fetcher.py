"""
Semantic Scholar Academic Graph API v1 擷取模組（防 429 強化版）
- 布林查詢合併 + /graph/v1/paper/search/bulk：將單一金屬 60~72 組細碎查詢濃縮為 5~7 組高密度布林查詢，API 請求次數減少 ~90%
- 全域單調時鐘節流器（Monotonic Pacemaker）+ 隨機微抖動（Jitter）+ AIMD 動態降速
- 本地磁碟快取（data/<METAL>/metadata/<metal>_s2_cache.json）：已成功查詢的組合不重複消耗配額
- Step 2：批次查詢（POST /paper/batch，最多 200 筆 DOI/次）補齊 openAccessPdf 連結
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

import config
from config import (
    FROM_YEAR,
    REQUEST_TIMEOUT,
    SEMANTIC_SCHOLAR_API_KEY,
    SEMANTIC_SCHOLAR_DELAY,
    SEMANTIC_SCHOLAR_MAX_PER_QUERY,
    SEMANTIC_SCHOLAR_MAX_QUERIES_NO_KEY,
)

S2_BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"

S2_SEARCH_FIELDS = (
    "paperId,externalIds,title,year,venue,isOpenAccess,"
    "openAccessPdf,citationCount,authors,abstract"
)
S2_BATCH_FIELDS = "paperId,externalIds,isOpenAccess,openAccessPdf"

# 全域持久化連線與單調時鐘節流狀態
_SESSION = requests.Session()
_LAST_REQ_TS: float = 0.0
_CURRENT_DELAY: float = max(1.1, float(SEMANTIC_SCHOLAR_DELAY))
_CONSECUTIVE_OK: int = 0
_API_KEY_REJECTED: bool = False


def _get_cache_path() -> Path:
    cache_path = getattr(config, "S2_CACHE_PATH", None)
    if cache_path:
        return Path(cache_path)
    return Path(config.METADATA_DIR) / f"{config.METAL_SYMBOL.lower()}_s2_cache.json"


def _load_s2_cache() -> dict[str, Any]:
    path = _get_cache_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("queries", {})
                data.setdefault("pdf_batch", {})
                return data
        except Exception:
            pass
    return {"queries": {}, "pdf_batch": {}}


def _save_s2_cache(cache: dict[str, Any]) -> None:
    path = _get_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _build_headers() -> dict[str, str]:
    headers = {
        "User-Agent": "TafelMiningBot/2.0 (academic electrochemistry research)",
        "Accept": "application/json",
    }
    if SEMANTIC_SCHOLAR_API_KEY and not _API_KEY_REJECTED:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    return headers


def _wait_for_rate_limit() -> None:
    """全域單調時鐘節流器：確保任意兩次 HTTP 請求之間至少間隔 _CURRENT_DELAY + jitter 秒。"""
    global _LAST_REQ_TS
    now = time.monotonic()
    if _LAST_REQ_TS > 0.0:
        elapsed = now - _LAST_REQ_TS
        target_wait = _CURRENT_DELAY + random.uniform(0.08, 0.25)
        if elapsed < target_wait:
            time.sleep(target_wait - elapsed)
    _LAST_REQ_TS = time.monotonic()


def _normalize_query_for_s2(query: str) -> str:
    """一般關鍵字搜尋（/paper/search）不需多層引號，去除引號並正規化空白。"""
    cleaned = query.replace('"', " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _format_doi(raw_doi: str | None) -> str | None:
    if not raw_doi:
        return None
    d = str(raw_doi).strip()
    if not d:
        return None
    if d.lower().startswith("http://doi.org/"):
        d = "https://doi.org/" + d[len("http://doi.org/"):]
    elif not d.lower().startswith("https://doi.org/"):
        d = f"https://doi.org/{d}"
    return d


def _s2_request_with_retry(
    method: str,
    url: str,
    *,
    params: dict | None = None,
    json_body: dict | None = None,
    max_retries: int = 4,
) -> requests.Response | None:
    """
    具備全域節流、AIMD 動態調速、429 指數退避與 403 API Key 自動降級的 HTTP 請求函式。
    """
    global _LAST_REQ_TS, _CURRENT_DELAY, _CONSECUTIVE_OK, _API_KEY_REJECTED

    for attempt in range(max_retries):
        _wait_for_rate_limit()
        headers = _build_headers()

        try:
            if method == "GET":
                resp = _SESSION.get(
                    url, params=params, headers=headers, timeout=REQUEST_TIMEOUT
                )
            else:
                resp = _SESSION.post(
                    url,
                    params=params,
                    json=json_body,
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                )
            _LAST_REQ_TS = time.monotonic()

            if resp.status_code == 200:
                _CONSECUTIVE_OK += 1
                base_delay = max(1.0, float(SEMANTIC_SCHOLAR_DELAY))
                if _CONSECUTIVE_OK >= 4 and _CURRENT_DELAY > base_delay:
                    _CURRENT_DELAY = max(base_delay, round(_CURRENT_DELAY * 0.85, 2))
                return resp

            if resp.status_code == 403 and SEMANTIC_SCHOLAR_API_KEY and not _API_KEY_REJECTED:
                print("  ⚠️  [S2] API Key 回傳 403（可能尚未生效或無該端點權限），自動切換為公開模式重試...")
                _API_KEY_REJECTED = True
                _CURRENT_DELAY = max(_CURRENT_DELAY, 2.5)
                continue

            if resp.status_code == 429:
                _CONSECUTIVE_OK = 0
                _CURRENT_DELAY = min(8.0, round(max(_CURRENT_DELAY * 1.5, 2.2), 2))
                retry_after = resp.headers.get("Retry-After")
                base_wait = (
                    int(retry_after)
                    if retry_after and retry_after.isdigit()
                    else min(60, 4 * (2 ** attempt))
                )
                wait = round(base_wait + random.uniform(0.3, 1.2), 1)
                print(
                    f"  [S2] 429 rate limit, waiting {wait}s "
                    f"(attempt {attempt + 1}/{max_retries}, adaptive delay -> {_CURRENT_DELAY}s) ..."
                )
                time.sleep(wait)
                _LAST_REQ_TS = time.monotonic()
                continue

            if 500 <= resp.status_code < 600:
                _CONSECUTIVE_OK = 0
                wait = 5 + attempt * 5
                print(f"  [S2] Server error {resp.status_code}, waiting {wait}s ...")
                time.sleep(wait)
                _LAST_REQ_TS = time.monotonic()
                continue

            if resp.status_code in (400, 403, 404):
                print(f"  [S2] HTTP {resp.status_code}: {resp.text[:160]}")
                return None

            resp.raise_for_status()
            return resp

        except requests.exceptions.RequestException as e:
            _CONSECUTIVE_OK = 0
            wait = 5 + attempt * 5
            print(f"  [S2] Request error: {e}, waiting {wait}s ...")
            time.sleep(wait)
            _LAST_REQ_TS = time.monotonic()

    return None


def build_grouped_s2_queries(modes: list[str] | tuple[str, ...] | str | None = None) -> list[str]:
    """
    依選取的搜尋模式（corrosion, cmp_etch, inhibitor, catalysis）組裝高容量布林查詢。
    因為 /paper/search/bulk 單次 HTTP 請求即可回傳最多 1,000 篇論文，
    濃縮為少數布林請求可將 API 呼叫次數減少 >90%，徹底避免連續請求觸發 429。
    """
    target = config.TARGET_METAL
    sym = config.METAL_SYMBOL
    compounds = list(getattr(config, "COMPOUNDS", []))
    if modes is not None:
        active_modes = config.parse_search_modes(modes)
    elif hasattr(config, "SEARCH_MODES"):
        active_modes = list(config.SEARCH_MODES)
    else:
        active_modes = config.parse_search_modes(getattr(config, "SEARCH_MODE", "corrosion"))

    metal_clause = f'("{target}" | "{sym}")' if target.lower() != sym.lower() else f'"{target}"'

    grouped: list[str] = []

    # 1. 一般金屬腐蝕極化與化合物／合金腐蝕極化
    if "corrosion" in active_modes:
        grouped.append(
            f'{metal_clause} + ("potentiodynamic polarization" | "corrosion current density" '
            f'| "polarization resistance" | "electrochemical corrosion" | "Tafel extrapolation" '
            f'| "Tafel slope" | "Tafel plot" | "Ecorr" | "icorr") '
            f'+ ("corrosion" | "polarization" | "anodic" | "cathodic")'
        )
        if compounds:
            comp_or = " | ".join(f'"{c}"' for c in compounds)
            grouped.append(
                f'({comp_or}) + ("potentiodynamic polarization" | "corrosion current density" | "Tafel slope" | "Ecorr")'
            )

    # 2. 半導體 CMP / 濕蝕刻 / 研磨液 / 氧化劑與電偶腐蝕環境
    if "cmp_etch" in active_modes:
        grouped.append(
            f'{metal_clause} + ("CMP" | "chemical mechanical" | "slurry" | "wet etching" '
            f'| "H2O2" | "hydrogen peroxide" | "KIO4" | "periodate" | "NaClO" | "hypochlorite" '
            f'| "Fe(NO3)3" | "KIO3" | "persulfate" | "SC-1" | "NH4OH" | "glycine" | "BTA" | "galvanic corrosion") '
            f'+ ("potentiodynamic" | "Tafel" | "Ecorr" | "icorr" | "corrosion current" | "polarization")'
        )

    # 3. 多樣化緩蝕劑（跨唑類、雜環、胺基酸、含硫氮、界面活性劑、離子液體與無機鈍化劑）
    if "inhibitor" in active_modes:
        grouped.append(
            f'{metal_clause} + ("corrosion inhibitor" | "corrosion inhibition" | "inhibition efficiency" '
            f'| "organic inhibitor" | "mixed-type inhibitor" | "benzotriazole" | "tolyltriazole" | "triazole" '
            f'| "tetrazole" | "imidazole" | "thiazole" | "pyrazole" | "Schiff base" | "amino acid" | "cysteine" '
            f'| "thiourea" | "surfactant" | "ionic liquid" | "molybdate" | "silicate") '
            f'+ ("potentiodynamic" | "Tafel" | "Ecorr" | "icorr" | "corrosion current" | "polarization")'
        )

    # 4. 電催化（HER / OER / ORR / CO2RR / 交換電流密度）
    if "catalysis" in active_modes:
        grouped.append(
            f'{metal_clause} + ("Tafel slope" | "Tafel plot" | "exchange current density" | "linear sweep voltammetry") '
            f'+ ("hydrogen evolution" | "oxygen evolution" | "oxygen reduction" | "CO2 reduction" | "electrocatalyst" | "catalyst")'
        )
        if compounds and "corrosion" not in active_modes:
            comp_or = " | ".join(f'"{c}"' for c in compounds)
            grouped.append(f'({comp_or}) + ("Tafel slope" | "electrocatalyst")')

    return grouped


def _select_s2_queries(queries: list[str], use_bulk: bool = True) -> list[str]:
    """
    選擇要送往 Semantic Scholar 的查詢清單：
    - 若啟用 use_bulk 且傳入的是完整展開查詢清單（> 8 組），自動使用 build_grouped_s2_queries() 濃縮為 5~7 組布林查詢。
    - 若呼叫端只傳入少量測試查詢（<= 8 組），則直接正規化後執行。
    """
    if use_bulk and len(queries) > 8:
        grouped = build_grouped_s2_queries()
        print(
            f"  [S2] 已將 {len(queries)} 組展開關鍵字濃縮為 {len(grouped)} 組高密度布林查詢（減少 ~{100 - int(len(grouped)/len(queries)*100)}% API 請求）"
        )
        return grouped

    seen: set[str] = set()
    normalized: list[str] = []
    for q in queries:
        nq = q.strip() if use_bulk else _normalize_query_for_s2(q)
        key = nq.lower()
        if nq and key not in seen:
            seen.add(key)
            normalized.append(nq)

    if SEMANTIC_SCHOLAR_API_KEY or len(normalized) <= SEMANTIC_SCHOLAR_MAX_QUERIES_NO_KEY:
        return normalized

    priority_terms = (
        "cmp", "chemical mechanical", "slurry", "wet etch", "h2o2",
        "periodate", "kio4", "naclo", "fe(no3)3", "sc-1", "inhibitor",
        "inhibition", "triazole", "imidazole", "potentiodynamic",
        "corrosion current", "tafel", "ecorr",
    )
    high = [q for q in normalized if any(t in q.lower() for t in priority_terms)]
    rest = [q for q in normalized if q not in high]
    return (high + rest)[:SEMANTIC_SCHOLAR_MAX_QUERIES_NO_KEY]


def _parse_s2_items(items: list[dict], query_str: str, max_per_query: int) -> list[dict]:
    recs: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        ext_ids = item.get("externalIds") or {}
        doi = _format_doi(ext_ids.get("DOI"))
        if not doi:
            continue

        oa_pdf_obj = item.get("openAccessPdf") or {}
        oa_pdf_url = oa_pdf_obj.get("url")
        authors = [
            a.get("name") for a in (item.get("authors") or []) if isinstance(a, dict) and a.get("name")
        ]

        recs.append({
            "query": query_str,
            "openalex_id": None,
            "s2_paper_id": item.get("paperId"),
            "source_api": "semantic_scholar",
            "doi": doi,
            "title": item.get("title"),
            "year": item.get("year"),
            "journal": item.get("venue"),
            "is_oa": bool(item.get("isOpenAccess")),
            "oa_status": oa_pdf_obj.get("status") if oa_pdf_obj else None,
            "oa_url": oa_pdf_url,
            "best_pdf_url": oa_pdf_url,
            "primary_pdf_url": None,
            "cited_by_count": item.get("citationCount"),
            "authors": "; ".join(authors[:5]),
            "abstract": item.get("abstract") or "",
        })
        if len(recs) >= max_per_query:
            break
    return recs


def _fetch_query_bulk(query_str: str, max_per_query: int, from_year: int) -> list[dict] | None:
    """
    使用 /graph/v1/paper/search/bulk 單次取回最多 1,000 筆結果（依引用數排序），
    只需 1 次 HTTP 請求即可填滿 max_per_query 筆具 DOI 的論文。
    回傳 list[dict]；若請求失敗則回傳 None。
    """
    params = {
        "query": query_str,
        "year": f"{from_year}-",
        "fields": S2_SEARCH_FIELDS,
        "sort": "citationCount:desc",
    }
    resp = _s2_request_with_retry("GET", S2_BULK_URL, params=params)
    if resp is None:
        return None
    payload = resp.json()
    items = payload.get("data") or []
    return _parse_s2_items(items, query_str, max_per_query)


def _fetch_query_standard(query_str: str, max_per_query: int, from_year: int) -> list[dict] | None:
    """標準 /graph/v1/paper/search 檢索（支援單元測試或作為 bulk 的備援）。"""
    clean_q = _normalize_query_for_s2(re.sub(r"[+|()]", " ", query_str))
    offset = 0
    query_recs: list[dict] = []

    while len(query_recs) < max_per_query:
        limit = min(100, max_per_query - len(query_recs))
        params = {
            "query": clean_q,
            "offset": offset,
            "limit": limit,
            "year": f"{from_year}-",
            "fields": S2_SEARCH_FIELDS,
        }
        resp = _s2_request_with_retry("GET", S2_SEARCH_URL, params=params)
        if resp is None:
            return None if not query_recs else query_recs

        payload = resp.json()
        items = payload.get("data") or []
        if not items:
            break

        parsed = _parse_s2_items(items, query_str, max_per_query - len(query_recs))
        query_recs.extend(parsed)

        next_offset = payload.get("next")
        if next_offset is None or len(items) < limit:
            break
        offset = next_offset

    return query_recs


def search_semantic_scholar(
    queries: list[str],
    max_per_query: int | None = None,
    from_year: int | None = None,
    use_cache: bool | None = None,
) -> list[dict]:
    """
    使用 Semantic Scholar Academic Graph API 檢索論文，
    回傳與 step1_search_openalex 相同 schema 的 list[dict]。
    """
    max_per_query = max_per_query or SEMANTIC_SCHOLAR_MAX_PER_QUERY
    from_year = from_year or FROM_YEAR
    use_bulk = bool(getattr(config, "SEMANTIC_SCHOLAR_USE_BULK", True)) and len(queries) > 1
    if use_cache is None:
        # 當執行完整 pipeline (len(queries) > 1) 且未設定強制刷快取時啟用本地快取
        use_cache = (len(queries) > 1) and (os.environ.get("TAFEL_FORCE_S2_REFRESH") != "1")

    selected_queries = _select_s2_queries(queries, use_bulk=use_bulk)

    key_status = (
        "已設定 API Key"
        if (SEMANTIC_SCHOLAR_API_KEY and not _API_KEY_REJECTED)
        else "無 API Key（公開額度模式）"
    )
    endpoint_mode = "Bulk Boolean (/search/bulk)" if use_bulk else "Standard (/search)"
    print(f"\n[Semantic Scholar] 開始搜尋（{len(selected_queries)} 組查詢，{endpoint_mode}，{key_status}）")

    cache = _load_s2_cache() if use_cache else {"queries": {}, "pdf_batch": {}}
    query_cache: dict[str, list[dict]] = cache.get("queries", {})
    cache_hits = 0

    all_records: list[dict] = []
    consecutive_failures = 0

    for q in tqdm(selected_queries, desc="Semantic Scholar search"):
        cache_key = f"{q}||y>={from_year}||m={max_per_query}"
        if use_cache and cache_key in query_cache:
            cached_recs = query_cache[cache_key]
            all_records.extend(cached_recs)
            cache_hits += 1
            continue

        if use_bulk:
            recs = _fetch_query_bulk(q, max_per_query, from_year)
            # 若 bulk 意外無回應，降級嘗試一次標準 search
            if recs is None:
                recs = _fetch_query_standard(q, max_per_query, from_year)
        else:
            recs = _fetch_query_standard(q, max_per_query, from_year)

        if recs is None:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                print(
                    "  ⚠️  [S2] 連續 3 組查詢失敗（可能已達速率上限），提前結束 S2 檢索並保留已取得結果。"
                )
                break
        else:
            consecutive_failures = 0
            all_records.extend(recs)
            if use_cache:
                query_cache[cache_key] = recs
                cache["queries"] = query_cache
                _save_s2_cache(cache)

    # 內部先依 DOI 去重回報統計
    unique_dois = {r["doi"].lower() for r in all_records if r.get("doi")}
    cache_msg = f"（快取命中 {cache_hits}/{len(selected_queries)} 組）" if use_cache and cache_hits else ""
    print(
        f"[Semantic Scholar] 檢索完成{cache_msg}，共取得 {len(all_records)} 筆記錄（{len(unique_dois)} 篇唯一 DOI）"
    )
    return all_records


def batch_lookup_pdf_urls(
    dois: list[str],
    batch_size: int = 200,
    use_cache: bool | None = None,
) -> dict[str, dict]:
    """
    使用 Semantic Scholar POST /paper/batch 一次查詢最多 200 筆 DOI 的 openAccessPdf URL。
    回傳 {normalized_doi_lower: {"s2_pdf_url": ..., "is_oa_s2": ..., "s2_paper_id": ...}}。
    """
    if use_cache is None:
        use_cache = (len(dois) > 5) and (os.environ.get("TAFEL_FORCE_S2_REFRESH") != "1")

    cache = _load_s2_cache() if use_cache else {"queries": {}, "pdf_batch": {}}
    batch_cache: dict[str, dict] = cache.get("pdf_batch", {})

    results: dict[str, dict] = {}
    clean_pairs: list[tuple[str, str]] = []
    seen: set[str] = set()

    for raw in dois:
        if not raw:
            continue
        d = str(raw).replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
        if not d:
            continue
        key = d.lower()
        if key in seen:
            continue
        seen.add(key)
        if use_cache and key in batch_cache:
            if batch_cache[key].get("s2_pdf_url"):
                results[key] = batch_cache[key]
            continue
        clean_pairs.append((key, f"DOI:{d}"))

    if not clean_pairs:
        if results:
            print(f"[Semantic Scholar] 從本地快取載入 {len(results)} 筆 OA PDF URL（0 API 請求）")
        return results

    chunks = [
        clean_pairs[i : i + batch_size]
        for i in range(0, len(clean_pairs), batch_size)
    ]
    print(f"[Semantic Scholar] 批次查詢 {len(clean_pairs)} 筆缺漏 PDF 的 DOI（共 {len(chunks)} 批）")

    for chunk in tqdm(chunks, desc="Semantic Scholar PDF batch"):
        ids = [s2_id for _, s2_id in chunk]
        resp = _s2_request_with_retry(
            "POST",
            S2_BATCH_URL,
            params={"fields": S2_BATCH_FIELDS},
            json_body={"ids": ids},
        )
        if resp is None:
            print("  ⚠️  [S2] Batch 查詢失敗，略過本批並繼續。")
            continue

        items = resp.json()
        if not isinstance(items, list):
            continue

        for (doi_key, _), item in zip(chunk, items):
            if not item or not isinstance(item, dict):
                if use_cache:
                    batch_cache[doi_key] = {}
                continue
            oa_pdf = item.get("openAccessPdf") or {}
            pdf_url = oa_pdf.get("url")
            if pdf_url and str(pdf_url).startswith("http"):
                info = {
                    "s2_pdf_url": pdf_url,
                    "is_oa_s2": bool(item.get("isOpenAccess")),
                    "s2_paper_id": item.get("paperId"),
                }
                results[doi_key] = info
                if use_cache:
                    batch_cache[doi_key] = info
            elif use_cache:
                batch_cache[doi_key] = {}

        if use_cache:
            cache["pdf_batch"] = batch_cache
            _save_s2_cache(cache)

    print(f"[Semantic Scholar] 批次補齊 {len(results)} 筆 OA PDF URL")
    return results
