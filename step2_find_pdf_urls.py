"""
Step 2: 尋找合法 OA PDF URL（Semantic Scholar Batch + Unpaywall）
- 支援相關性預篩（跳過 Step 3 會濾除的無關論文）
- 先以極快的 Semantic Scholar Batch API（200 DOI/次）補齊 OA PDF
- 預設跳過 OpenAlex / S2 已有 PDF URL 的論文，大幅減少冗餘 Unpaywall 查詢
- 支援連線池複用與中途原子存檔（Checkpoint），中斷後可透過 --append 無痛續跑
"""

import os
import time

import pandas as pd
import requests
from tqdm import tqdm

from config import (
    DOWNLOAD_STATE_PATH,
    ENABLE_SEMANTIC_SCHOLAR,
    MAILTO,
    PDF_DIR,
    REQUEST_TIMEOUT,
    REQUIRE_TARGET_IN_TITLE_ABSTRACT,
    STEP1_OUTPUT,
    STEP2_CHECKPOINT_EVERY,
    STEP2_OUTPUT,
    STEP2_SKIP_UNPAYWALL_IF_HAS_PDF,
    STEP2_UNPAYWALL_DELAY,
    TEXT_DIR,
    build_target_regex,
)
from record_keys import (
    is_title_abstract_relevant,
    normalize_doi_key,
    normalize_pmcid,
    save_csv_atomic,
)

UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
_SESSION = requests.Session()
_UNPAYWALL_COLS = (
    "unpaywall_pdf_url",
    "unpaywall_landing_url",
    "license",
    "is_oa_unpaywall",
)


def _has_http_url(val) -> bool:
    return pd.notna(val) and str(val).strip().startswith("http")


def find_pdf_url(doi: str) -> dict:
    doi_clean = str(doi).replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    url = f"{UNPAYWALL_BASE}/{doi_clean}"
    params = {"email": MAILTO}

    try:
        resp = _SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 404:
            return {"unpaywall_pdf_url": None, "license": None}
        if resp.status_code == 429:
            time.sleep(30)
            resp = _SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"unpaywall_pdf_url": None, "license": None, "error": str(e)}

    best_oa = data.get("best_oa_location") or {}
    return {
        "unpaywall_pdf_url": best_oa.get("url_for_pdf"),
        "unpaywall_landing_url": best_oa.get("url_for_landing_page"),
        "license": best_oa.get("license"),
        "is_oa_unpaywall": data.get("is_oa"),
    }


def _bootstrap_state_from_prev_step2(
    prev_df: pd.DataFrame,
    state_path: str = DOWNLOAD_STATE_PATH,
    pdf_dir: str = PDF_DIR,
    text_dir: str = TEXT_DIR,
) -> None:
    """
    在 --append 覆寫舊 Step 2 CSV 之前，若尚未建立下載狀態表，
    先以「舊 Step 2 CSV」執行離線初始化，避免本次新增論文在 Step 3 被誤標為 failed_legacy。
    """
    if os.path.exists(state_path):
        return
    from step3_download_pdfs import load_or_bootstrap_download_state

    load_or_bootstrap_download_state(state_path, prev_df, pdf_dir, text_dir)


def _pick_best_pdf(row):
    for col in ("unpaywall_pdf_url", "s2_pdf_url", "best_pdf_url", "primary_pdf_url"):
        val = row.get(col)
        if _has_http_url(val):
            return val
    return None


def _compute_relevance_mask(df: pd.DataFrame) -> pd.Series:
    if not REQUIRE_TARGET_IN_TITLE_ABSTRACT or df.empty:
        return pd.Series(True, index=df.index)
    pat = build_target_regex()
    titles = df.get("title", pd.Series("", index=df.index))
    abstracts = df.get("abstract", pd.Series("", index=df.index))
    return pd.Series(
        [is_title_abstract_relevant(t, a, pat, require_both_present=False) for t, a in zip(titles, abstracts)],
        index=df.index,
    )


def run_find_pdfs(append: bool | None = None):
    if append is None:
        append = os.environ.get("TAFEL_APPEND") == "1"

    df = pd.read_csv(STEP1_OUTPUT)
    print(f"Loaded {len(df)} papers from {STEP1_OUTPUT}")

    prev_unpaywall: dict[str, dict] = {}
    prev_s2_pdf: dict[str, str] = {}
    prev_seen_dois: set[str] = set()
    if append and os.path.exists(STEP2_OUTPUT):
        try:
            prev_df = pd.read_csv(STEP2_OUTPUT)
            # 先以舊 Step 2 清單鎖定既有下載狀態，避免覆寫後 Step 3 把本次新增論文誤標為 failed_legacy
            _bootstrap_state_from_prev_step2(prev_df)
            has_checked_col = "step2_checked" in prev_df.columns
            for _, prow in prev_df.iterrows():
                dk = normalize_doi_key(prow.get("doi"))
                if not dk:
                    continue
                if _has_http_url(prow.get("s2_pdf_url")):
                    prev_s2_pdf[dk] = str(prow.get("s2_pdf_url"))
                if "is_oa_unpaywall" in prow and (
                    pd.notna(prow.get("is_oa_unpaywall")) or pd.notna(prow.get("unpaywall_pdf_url"))
                ):
                    prev_unpaywall[dk] = {
                        "unpaywall_pdf_url": prow.get("unpaywall_pdf_url") if pd.notna(prow.get("unpaywall_pdf_url")) else None,
                        "unpaywall_landing_url": prow.get("unpaywall_landing_url") if pd.notna(prow.get("unpaywall_landing_url")) else None,
                        "license": prow.get("license") if pd.notna(prow.get("license")) else None,
                        "is_oa_unpaywall": prow.get("is_oa_unpaywall") if pd.notna(prow.get("is_oa_unpaywall")) else None,
                    }
                # 僅將「確實已處理完畢」的 DOI 加入 prev_seen_dois；舊版 CSV 無此欄時相容視為全部已查
                if (not has_checked_col) or bool(prow.get("step2_checked")):
                    prev_seen_dois.add(dk)
            if prev_seen_dois:
                print(
                    f"[Step 2 Append] 從既有 Step 2 檔案載入 {len(prev_seen_dois)} 筆已查過的 DOI，"
                    "僅對新論文查詢 Semantic Scholar / Unpaywall"
                )
        except Exception as e:
            print(f"  ⚠️ 無法讀取既有 Step 2 檔案快取 ({e})，將重新查詢全部 DOI。")

    # 避免若 STEP1_OUTPUT 意外帶有舊 Step 2 欄位造成重複欄名
    for col in (*_UNPAYWALL_COLS, "s2_pdf_url", "final_pdf_url", "step2_checked"):
        if col in df.columns:
            df = df.drop(columns=[col], errors="ignore")

    df = df.reset_index(drop=True)
    for col in _UNPAYWALL_COLS:
        df[col] = None
    df["step2_checked"] = False

    # 預先還原舊 Unpaywall 與 S2 快取
    if prev_unpaywall:
        for idx, doi_val in df["doi"].items():
            dk = normalize_doi_key(doi_val)
            if dk in prev_unpaywall:
                info = prev_unpaywall[dk]
                for col in _UNPAYWALL_COLS:
                    df.at[idx, col] = info.get(col)

    df["s2_pdf_url"] = (
        df["doi"].apply(lambda d: prev_s2_pdf.get(normalize_doi_key(d)))
        if prev_s2_pdf
        else None
    )

    relevant_mask = _compute_relevance_mask(df)
    skipped_irrel = int((~relevant_mask).sum())
    if skipped_irrel:
        print(
            f"[Step 2] 相關性預篩：保留 {int(relevant_mask.sum())} / {len(df)} 篇進入外部查詢"
            f"（略過 {skipped_irrel} 篇標題與摘要未提目標金屬者）"
        )

    # 1. 優先執行極快的 Semantic Scholar Batch API（200 DOI/次），先補齊無 PDF URL 者
    if ENABLE_SEMANTIC_SCHOLAR and not df.empty:
        def _needs_s2_lookup(idx_row) -> bool:
            idx, row = idx_row
            if not bool(relevant_mask.loc[idx]):
                return False
            if bool(normalize_pmcid(row.get("pmcid"))):
                return False
            dk = normalize_doi_key(row.get("doi"))
            if not dk or dk in prev_seen_dois:
                return False
            for col in ("unpaywall_pdf_url", "s2_pdf_url", "best_pdf_url", "primary_pdf_url"):
                if _has_http_url(row.get(col)):
                    return False
            return True

        missing_mask = pd.Series(
            [_needs_s2_lookup((i, r)) for i, r in df.iterrows()],
            index=df.index,
        )
        missing_dois = df.loc[missing_mask, "doi"].dropna().astype(str).tolist()
        if missing_dois:
            from semantic_scholar_fetcher import batch_lookup_pdf_urls

            s2_map = batch_lookup_pdf_urls(missing_dois)
            if s2_map:
                def _lookup_s2_pdf(doi_val):
                    dk = normalize_doi_key(doi_val)
                    if not dk:
                        return None
                    info = s2_map.get(dk)
                    return info["s2_pdf_url"] if info else None

                df.loc[missing_mask, "s2_pdf_url"] = df.loc[missing_mask, "doi"].apply(_lookup_s2_pdf)

    # 2. Unpaywall 查詢：跳過已有 PMCID、無關論文、舊已查 DOI，以及（預設）已有 PDF URL 者
    skipped_pmcid = 0
    skipped_has_pdf = 0
    queried_new = 0
    unpaywall_indices: list[int] = []

    for idx, row in df.iterrows():
        doi = row.get("doi")
        if pd.isna(doi):
            df.at[idx, "step2_checked"] = True
            continue
        dk = normalize_doi_key(doi)
        if not dk or dk in prev_unpaywall or dk in prev_seen_dois:
            df.at[idx, "step2_checked"] = True
            continue
        if not bool(relevant_mask.loc[idx]):
            df.at[idx, "step2_checked"] = True
            continue
        if bool(normalize_pmcid(row.get("pmcid"))):
            skipped_pmcid += 1
            df.at[idx, "step2_checked"] = True
            continue
        if STEP2_SKIP_UNPAYWALL_IF_HAS_PDF and any(
            _has_http_url(row.get(c)) for c in ("s2_pdf_url", "best_pdf_url", "primary_pdf_url")
        ):
            skipped_has_pdf += 1
            df.at[idx, "step2_checked"] = True
            continue
        unpaywall_indices.append(idx)

    if skipped_pmcid or skipped_has_pdf:
        print(
            f"[Step 2] Unpaywall 智慧跳過 — 已有 PDF URL: {skipped_has_pdf} 篇 | "
            f"已有 Europe PMC PMCID: {skipped_pmcid} 篇 | 實際待查: {len(unpaywall_indices)} 篇"
        )

    checkpoint_every = max(0, int(STEP2_CHECKPOINT_EVERY))
    try:
        for idx in tqdm(unpaywall_indices, total=len(unpaywall_indices), desc="Unpaywall lookup"):
            doi = df.at[idx, "doi"]
            info = find_pdf_url(doi)
            for col in _UNPAYWALL_COLS:
                if col in info:
                    df.at[idx, col] = info.get(col)
            df.at[idx, "step2_checked"] = True
            queried_new += 1

            if checkpoint_every > 0 and queried_new % checkpoint_every == 0:
                df["final_pdf_url"] = df.apply(_pick_best_pdf, axis=1)
                save_csv_atomic(df, STEP2_OUTPUT)

            if STEP2_UNPAYWALL_DELAY > 0:
                time.sleep(STEP2_UNPAYWALL_DELAY)
    finally:
        df["final_pdf_url"] = df.apply(_pick_best_pdf, axis=1)
        save_csv_atomic(df, STEP2_OUTPUT)

    n_pdf = int(df["final_pdf_url"].notna().sum())
    n_pmcid = (
        int(df["pmcid"].apply(lambda v: bool(normalize_pmcid(v))).sum())
        if "pmcid" in df.columns
        else 0
    )
    print(
        f"Papers with PDF URL: {n_pdf} / {len(df)} | With Europe PMC PMCID: {n_pmcid} "
        f"| Unpaywall queried: {queried_new}"
    )
    print(f"✅ Saved to {STEP2_OUTPUT}")
    return df


if __name__ == "__main__":
    run_find_pdfs()
