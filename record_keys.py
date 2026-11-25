"""
共用的論文識別鍵與檔名正規化工具（Step 1 / Step 2 / Step 3 共用）。
確保 OpenAlex、Semantic Scholar 與 Europe PMC 在各步驟使用一致的去重鍵與檔名規則。
"""

import re
import pandas as pd


def clean_str(val) -> str:
    """將可能為 NaN / None 的值轉為乾淨字串。"""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    return "" if s.lower() == "nan" else s


def normalize_doi_key(doi) -> str:
    """正規化 DOI（去除 https://doi.org/ 前綴並轉小寫）。"""
    s = clean_str(doi)
    if not s:
        return ""
    return (
        s.replace("https://doi.org/", "")
        .replace("http://doi.org/", "")
        .strip()
        .lower()
    )


def normalize_pmcid(pmcid) -> str:
    """正規化 PMCID 字串。"""
    return clean_str(pmcid)


def make_record_key(doi, pmcid=None) -> str:
    """統一去重鍵：優先標準化 DOI，無 DOI 時退回 pmcid:<lower_pmcid>。"""
    dk = normalize_doi_key(doi)
    if dk:
        return dk
    pk = normalize_pmcid(pmcid).lower()
    return f"pmcid:{pk}" if pk else ""


def sanitize_base_name(identifier: str) -> str:
    """與 Step 3 / Step 4 / Europe PMC 一致的檔名主體規則（不含副檔名）。"""
    name = (
        str(identifier)
        .replace("https://doi.org/", "")
        .replace("http://doi.org/", "")
        .strip()
    )
    name = re.sub(r"[^\w\-.]", "_", name)
    return name[:120]
