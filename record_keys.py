"""
共用的論文識別鍵與檔名正規化工具（Step 1 / Step 2 / Step 3 共用）。
確保 OpenAlex、Semantic Scholar 與 Europe PMC 在各步驟使用一致的去重鍵與檔名規則。
"""

import os
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


def save_csv_atomic(df: pd.DataFrame, output_path: str) -> None:
    """原子寫入 CSV（先寫入 .tmp 再 os.replace），避免中斷造成檔案損毀。"""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = output_path + ".tmp"
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    os.replace(tmp_path, output_path)


def is_title_abstract_relevant(
    title,
    abstract,
    target_pat,
    *,
    require_both_present: bool = False,
) -> bool:
    """
    檢查論文標題／摘要是否命中目標金屬 regex：
    - target_pat 為 None 時一律回傳 True。
    - 標題與摘要皆空時回傳 True（不誤殺缺漏 metadata 的論文）。
    - require_both_present=True（Step 1 檢索期）時，僅在標題與摘要皆非空才進行過濾，
      避免 OpenAlex 缺摘要論文在 S2 / Europe PMC 回補前被過早剔除。
    """
    if target_pat is None:
        return True
    t = clean_str(title)
    a = clean_str(abstract)
    if not t and not a:
        return True
    if require_both_present and (not t or not a):
        return True
    return bool(target_pat.search(f"{t} {a}"))
