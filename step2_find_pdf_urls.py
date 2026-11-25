"""

Step 2: 用 Unpaywall API 根據 DOI 找合法 OA PDF URL

"""

import os
import time

import requests

import pandas as pd

from tqdm import tqdm

from config import (
    DOWNLOAD_STATE_PATH,
    ENABLE_SEMANTIC_SCHOLAR,
    MAILTO,
    PDF_DIR,
    REQUEST_TIMEOUT,
    STEP1_OUTPUT,
    STEP2_OUTPUT,
    TEXT_DIR,
)
from record_keys import normalize_doi_key, normalize_pmcid

UNPAYWALL_BASE = "https://api.unpaywall.org/v2"


def find_pdf_url(doi: str) -> dict:

    doi_clean = doi.replace("https://doi.org/", "").strip()

    url = f"{UNPAYWALL_BASE}/{doi_clean}"

    params = {"email": MAILTO}

    try:

        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)  # ✅

        if resp.status_code == 404:

            return {"unpaywall_pdf_url": None, "license": None}

        if resp.status_code == 429:

            time.sleep(30)

            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)  # ✅

        resp.raise_for_status()

        data = resp.json()

    except Exception as e:

        return {"unpaywall_pdf_url": None, "license": None, "error": str(e)}

    best_oa = data.get("best_oa_location") or {}

    pdf_url = best_oa.get("url_for_pdf")

    landing = best_oa.get("url_for_landing_page")

    license_info = best_oa.get("license")

    return {

        "unpaywall_pdf_url": pdf_url,

        "unpaywall_landing_url": landing,

        "license": license_info,

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


def run_find_pdfs(append: bool | None = None):
    if append is None:
        append = os.environ.get("TAFEL_APPEND") == "1"

    df = pd.read_csv(STEP1_OUTPUT)  # ✅

    print(f"Loaded {len(df)} papers from {STEP1_OUTPUT}")

    prev_unpaywall: dict[str, dict] = {}
    prev_s2_pdf: dict[str, str] = {}
    prev_seen_dois: set[str] = set()
    if append and os.path.exists(STEP2_OUTPUT):
        try:
            prev_df = pd.read_csv(STEP2_OUTPUT)
            # 先以舊 Step 2 清單鎖定既有下載狀態，避免覆寫後 Step 3 把本次新增論文誤標為 failed_legacy
            _bootstrap_state_from_prev_step2(prev_df)
            for _, prow in prev_df.iterrows():
                dk = normalize_doi_key(prow.get("doi"))
                if not dk:
                    continue
                prev_seen_dois.add(dk)
                if "is_oa_unpaywall" in prow and (pd.notna(prow.get("is_oa_unpaywall")) or pd.notna(prow.get("unpaywall_pdf_url"))):
                    prev_unpaywall[dk] = {
                        "unpaywall_pdf_url": prow.get("unpaywall_pdf_url") if pd.notna(prow.get("unpaywall_pdf_url")) else None,
                        "unpaywall_landing_url": prow.get("unpaywall_landing_url") if pd.notna(prow.get("unpaywall_landing_url")) else None,
                        "license": prow.get("license") if pd.notna(prow.get("license")) else None,
                        "is_oa_unpaywall": prow.get("is_oa_unpaywall") if pd.notna(prow.get("is_oa_unpaywall")) else None,
                    }
                if pd.notna(prow.get("s2_pdf_url")) and str(prow.get("s2_pdf_url")).startswith("http"):
                    prev_s2_pdf[dk] = str(prow.get("s2_pdf_url"))
            if prev_seen_dois:
                print(
                    f"[Step 2 Append] 從既有 Step 2 檔案載入 {len(prev_seen_dois)} 筆已查過的 DOI，"
                    "僅對新論文查詢 Unpaywall / Semantic Scholar"
                )
        except Exception as e:
            print(f"  ⚠️ 無法讀取既有 Step 2 檔案快取 ({e})，將重新查詢全部 DOI。")

    results = []
    skipped_pmcid = 0
    queried_new = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Unpaywall lookup"):

        doi = row.get("doi")

        if pd.isna(doi):

            results.append({})

            continue

        dk = normalize_doi_key(doi)
        if dk in prev_unpaywall:
            results.append(prev_unpaywall[dk])
            continue
        if dk in prev_seen_dois:
            results.append({})
            continue
        if bool(normalize_pmcid(row.get("pmcid"))):
            # 已有 Europe PMC PMCID，Step 3 可直接抓取開放全文 XML，免查 Unpaywall
            skipped_pmcid += 1
            results.append({})
            continue

        info = find_pdf_url(doi)
        queried_new += 1

        results.append(info)

        time.sleep(0.5)

    if skipped_pmcid:
        print(f"[Step 2] 已有 Europe PMC PMCID 直接略過 Unpaywall 查詢：{skipped_pmcid} 篇（新查 Unpaywall：{queried_new} 篇）")

    pdf_df = pd.DataFrame(results)

    # 避免若 STEP1_OUTPUT 意外帶有舊 Step 2 欄位造成重複欄名
    drop_existing = [c for c in pdf_df.columns if c in df.columns]
    if drop_existing:
        df = df.drop(columns=drop_existing, errors="ignore")

    df = pd.concat([df.reset_index(drop=True), pdf_df], axis=1)

    # 針對 Unpaywall + OpenAlex 仍無 PDF URL（且無 PMCID、非舊已查 DOI）的論文，使用 Semantic Scholar Batch API 補查
    df["s2_pdf_url"] = df["doi"].apply(lambda d: prev_s2_pdf.get(normalize_doi_key(d))) if prev_s2_pdf else None
    if ENABLE_SEMANTIC_SCHOLAR:
        def _needs_s2_lookup(row) -> bool:
            if bool(normalize_pmcid(row.get("pmcid"))):
                return False
            dk = normalize_doi_key(row.get("doi"))
            if not dk or dk in prev_seen_dois:
                return False
            for col in ["unpaywall_pdf_url", "s2_pdf_url", "best_pdf_url", "primary_pdf_url"]:
                val = row.get(col)
                if pd.notna(val) and str(val).startswith("http"):
                    return False
            return True

        missing_mask = df.apply(_needs_s2_lookup, axis=1)
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

    def pick_best_pdf(row):

        for col in ["unpaywall_pdf_url", "s2_pdf_url", "best_pdf_url", "primary_pdf_url"]:

            val = row.get(col)

            if pd.notna(val) and str(val).startswith("http"):

                return val

        return None

    df["final_pdf_url"] = df.apply(pick_best_pdf, axis=1)

    n_pdf = df["final_pdf_url"].notna().sum()
    n_pmcid = df["pmcid"].apply(lambda v: bool(normalize_pmcid(v))).sum() if "pmcid" in df.columns else 0

    print(f"Papers with PDF URL: {n_pdf} / {len(df)} | With Europe PMC PMCID: {n_pmcid}")

    df.to_csv(STEP2_OUTPUT, index=False, encoding="utf-8-sig")  # ✅

    print(f"✅ Saved to {STEP2_OUTPUT}")

    return df

if __name__ == "__main__":

    run_find_pdfs()
