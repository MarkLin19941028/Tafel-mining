"""
Europe PMC 開放全文 XML 擷取與 JATS 解析模組
- 零 403 封鎖、免 API Key
- 直接將 JATS XML 解析成標準 .txt 與結構化 _tables.json（與 Step 4 / Step 5 完全相容）
"""

import json
import math
import os
import re
import time
import xml.etree.ElementTree as ET
import requests
from tqdm import tqdm

from config import (
    EUROPEPMC_DELAY,
    EUROPEPMC_MAX_PAPERS,
    FROM_YEAR,
    METAL_SYMBOL,
    REQUEST_TIMEOUT,
    REQUIRE_TARGET_IN_TITLE_ABSTRACT,
    SEARCH_MODES,
    STEP1_EARLY_STOP_EMPTY_PAGES,
    STEP1_MAX_SCAN_MULTIPLIER,
    TARGET_METAL,
    TEXT_DIR,
    build_target_regex,
    parse_search_modes,
)
from record_keys import make_record_key

EPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_XML_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

_EPMC_MODE_CLAUSES: dict[str, str] = {
    "corrosion": (
        '("potentiodynamic polarization" OR "corrosion current density" '
        'OR "Tafel extrapolation" OR ("Ecorr" AND "icorr"))'
    ),
    "cmp_etch": (
        '(("chemical mechanical" OR "CMP" OR "wet etching" OR "slurry" OR "H2O2") '
        'AND ("potentiodynamic" OR "Tafel" OR "corrosion current" OR "Ecorr"))'
    ),
    "inhibitor": (
        '(("corrosion inhibitor" OR "corrosion inhibition" OR "inhibition efficiency" OR "organic inhibitor") '
        'AND ("Tafel" OR "potentiodynamic" OR "polarization" OR "icorr" OR "Ecorr"))'
    ),
    "solvent": (
        '(("non-aqueous" OR "nonaqueous" OR "organic solvent" OR "organic media" '
        'OR "anhydrous" OR "semi-aqueous" OR "mixed solvent" OR "water-organic" '
        'OR "aqueous-organic" OR "trace water" OR "protic solvent" OR "aprotic solvent" '
        'OR "co-solvent" OR "ethylene glycol" OR "deep eutectic solvent") '
        'AND ("potentiodynamic" OR "Tafel" OR "corrosion current" OR "Ecorr" OR "icorr" OR "polarization"))'
    ),
    "catalysis": (
        '(("Tafel slope" OR "exchange current density" OR "linear sweep voltammetry") '
        'AND ("hydrogen evolution" OR "oxygen evolution" OR "oxygen reduction" OR "CO2 reduction" OR "electrocatalyst"))'
    ),
}


def build_europepmc_query(
    modes: list[str] | tuple[str, ...] | str | None = None,
    target_metal: str | None = None,
    metal_symbol: str | None = None,
    from_year: int | None = None,
) -> str:
    """依選取的搜尋模式組合組裝 Europe PMC 布林查詢字串。"""
    active_modes = SEARCH_MODES if modes is None else parse_search_modes(modes)
    t_metal = target_metal or TARGET_METAL
    m_sym = metal_symbol or METAL_SYMBOL
    f_year = FROM_YEAR if from_year is None else from_year

    clauses = [_EPMC_MODE_CLAUSES[m] for m in active_modes]
    topic_clause = clauses[0] if len(clauses) == 1 else f"({' OR '.join(clauses)})"
    return (
        f'("{t_metal}" OR "{m_sym}") '
        f"AND {topic_clause} "
        f"AND OPEN_ACCESS:y AND PUB_YEAR:[{f_year} TO 2100]"
    )


def sanitize_base_name(identifier: str) -> str:
    """與 step3 / step4 相同的檔名規則（不含副檔名），確保同一 DOI 自動去重。"""
    name = str(identifier).replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    name = re.sub(r"[^\w\-.]", "_", name)
    return name[:120]


def _elem_text(elem: ET.Element) -> str:
    """將 XML 節點內的所有文字（含上下標）展開為乾淨字串。"""
    txt = "".join(elem.itertext())
    return re.sub(r"\s+", " ", txt).strip()


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def parse_jats_xml(xml_text: str) -> tuple[str, list[list[list[str]]]]:
    """
    解析 Europe PMC JATS XML：
    - 回傳 (plain_text, tables)
    - plain_text 保留標題、摘要與正文段落（排除 <ref-list> 文末參考文獻）
    - tables 為 list[table]，每張 table 為 list[row]，每列為 list[cell_str]（展開 colspan）
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return "", []

    # 1. 擷取純文字（排除 ref-list 與 table-wrap 避免表格雜亂重複進正文）
    text_parts: list[str] = []

    def walk_text(node: ET.Element) -> None:
        tag = _local_tag(node.tag)
        if tag in ("ref-list", "back", "table-wrap", "fig"):
            return
        if tag in ("article-title", "title", "p"):
            t = _elem_text(node)
            if t:
                text_parts.append(t)
            return
        for child in node:
            walk_text(child)

    walk_text(root)
    plain_text = "\n\n".join(text_parts)

    # 2. 擷取結構化表格
    tables: list[list[list[str]]] = []
    for elem in root.iter():
        if _local_tag(elem.tag) != "table":
            continue
        rows: list[list[str]] = []
        for tr in elem.iter():
            if _local_tag(tr.tag) != "tr":
                continue
            row_cells: list[str] = []
            for cell in tr:
                ctag = _local_tag(cell.tag)
                if ctag not in ("th", "td"):
                    continue
                val = _elem_text(cell)
                try:
                    colspan = int(cell.attrib.get("colspan", "1"))
                except ValueError:
                    colspan = 1
                colspan = max(1, min(colspan, 12))
                row_cells.append(val)
                if colspan > 1:
                    row_cells.extend([""] * (colspan - 1))
            if any(c for c in row_cells):
                rows.append(row_cells)
        if len(rows) >= 2:
            tables.append(rows)

    return plain_text, tables


def lookup_pmcid_by_doi(doi: str) -> str | None:
    """依 DOI 查詢 Europe PMC 是否有開放全文（OPEN_ACCESS:y），有則回傳 PMCID。"""
    doi_clean = str(doi).replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    if not doi_clean:
        return None
    try:
        resp = requests.get(
            EPMC_SEARCH_URL,
            params={
                "query": f'DOI:"{doi_clean}" AND OPEN_ACCESS:y',
                "format": "json",
                "resultType": "lite",
                "pageSize": 1,
            },
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        results = resp.json().get("resultList", {}).get("result", [])
        if results:
            return results[0].get("pmcid")
    except Exception:
        return None
    return None


def fetch_fulltext_xml(pmcid: str) -> str | None:
    """下載指定 PMCID 的完整 JATS XML。"""
    if not pmcid:
        return None
    try:
        resp = requests.get(
            EPMC_XML_URL.format(pmcid=pmcid),
            timeout=REQUEST_TIMEOUT * 2,
        )
        if resp.status_code == 200 and "<article" in resp.text[:500]:
            return resp.text
    except Exception:
        return None
    return None


def save_xml_to_texts(
    xml_text: str,
    base_name: str,
    text_dir: str = TEXT_DIR,
) -> tuple[bool, bool, int]:
    """
    將解析後的文字與表格寫入 text_dir：
    - 若 .txt 不存在：寫入 .txt（wrote_txt=True）
    - 若 .txt 已存在：保留原 PDF .txt（wrote_txt=False），但若 XML 有結構化表格則更新 _tables.json（upgraded_tables=True）
    回傳 (wrote_txt, upgraded_tables, num_tables)
    """
    os.makedirs(text_dir, exist_ok=True)
    plain_text, tables = parse_jats_xml(xml_text)
    if not plain_text and not tables:
        return False, False, 0

    text_path = os.path.join(text_dir, base_name + ".txt")
    table_path = os.path.join(text_dir, base_name + "_tables.json")

    wrote_txt = False
    if not os.path.exists(text_path) and plain_text:
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(plain_text)
        wrote_txt = True

    upgraded_tables = False
    if tables or not os.path.exists(table_path):
        with open(table_path, "w", encoding="utf-8") as f:
            json.dump(tables, f, ensure_ascii=False, indent=2)
        upgraded_tables = bool(tables)

    return wrote_txt, upgraded_tables, len(tables)


def try_fallback_from_europepmc(
    doi: str,
    base_name: str,
    text_dir: str = TEXT_DIR,
) -> tuple[bool, int]:
    """Step 3 下載 PDF 失敗時的備援：透過 DOI 嘗試抓取 Europe PMC 開放全文 XML。"""
    pmcid = lookup_pmcid_by_doi(doi)
    if not pmcid:
        return False, 0
    xml_text = fetch_fulltext_xml(pmcid)
    if not xml_text:
        return False, 0
    wrote_txt, upgraded_tables, n_tables = save_xml_to_texts(xml_text, base_name, text_dir)
    return (wrote_txt or upgraded_tables), n_tables


def search_europepmc_metadata(
    max_papers: int | None = None,
    modes: list[str] | tuple[str, ...] | str | None = None,
    known_keys: set[str] | None = None,
    needs_backfill: dict[str, set[str]] | None = None,
    max_scan_multiplier: int | None = None,
    early_stop_empty_pages: int | None = None,
) -> list[dict]:
    """
    查詢 Europe PMC 開放取用文獻 metadata（供 Step 1 與 OpenAlex / Semantic Scholar 統一去重）。
    回傳與 Step 1 相容的 dict 清單（含 pmcid 與 doi）。
    支援傳入 known_keys / needs_backfill 進行搜尋過程即時去重與 PMCID 回補。
    """
    limit = EUROPEPMC_MAX_PAPERS if max_papers is None else max_papers
    if limit <= 0:
        return []

    scan_mult = max(1, int(STEP1_MAX_SCAN_MULTIPLIER if max_scan_multiplier is None else max_scan_multiplier))
    early_stop = max(0, int(STEP1_EARLY_STOP_EMPTY_PAGES if early_stop_empty_pages is None else early_stop_empty_pages))
    inflight = known_keys is not None
    max_pages = max(1, math.ceil(limit / 100) * scan_mult)

    active_modes = SEARCH_MODES if modes is None else parse_search_modes(modes)
    query = build_europepmc_query(modes=active_modes)
    print(f"\n[Europe PMC] Searching open-access papers (modes={','.join(active_modes)}, limit={limit})...")
    print(f"  Query: {query}")

    target_pat = build_target_regex() if REQUIRE_TARGET_IN_TITLE_ABSTRACT else None

    records: list[dict] = []
    seen_pmcids: set[str] = set()
    new_count = 0
    skipped_dup = 0
    backfill_count = 0
    pages_scanned = 0
    empty_pages = 0
    cursor_mark = "*"

    while new_count < limit:
        page_size = 100 if inflight else min(100, limit - new_count)
        try:
            resp = requests.get(
                EPMC_SEARCH_URL,
                params={
                    "query": query,
                    "format": "json",
                    "resultType": "core",
                    "pageSize": page_size,
                    "cursorMark": cursor_mark,
                },
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  ⚠️ Europe PMC search error: {e}")
            break

        results = data.get("resultList", {}).get("result", [])
        if not results:
            break

        page_new = 0
        for item in results:
            pmcid = item.get("pmcid")
            if not pmcid or pmcid in seen_pmcids:
                continue
            title = item.get("title") or ""
            abstract = item.get("abstractText") or ""
            if target_pat is not None:
                if not target_pat.search(f"{title} {abstract}"):
                    continue
            seen_pmcids.add(pmcid)
            raw_doi = (item.get("doi") or "").strip()
            doi_url = (
                raw_doi
                if (not raw_doi or raw_doi.startswith("http"))
                else f"https://doi.org/{raw_doi}"
            ) or None
            pub_year = item.get("pubYear")
            year_val = int(pub_year) if str(pub_year).isdigit() else None
            rec = {
                "query": query,
                "openalex_id": None,
                "s2_paper_id": None,
                "pmcid": pmcid,
                "source_api": "europepmc",
                "doi": doi_url,
                "title": title,
                "year": year_val,
                "journal": item.get("journalTitle"),
                "is_oa": True,
                "oa_status": "europepmc",
                "oa_url": f"https://europepmc.org/article/PMC/{pmcid}",
                "best_pdf_url": None,
                "primary_pdf_url": None,
                "cited_by_count": item.get("citedByCount"),
                "authors": item.get("authorString"),
                "abstract": abstract,
            }

            key = make_record_key(doi_url, pmcid)
            if inflight and key and key in known_keys:
                skipped_dup += 1
                missing = needs_backfill.get(key) if needs_backfill is not None else None
                if missing:
                    can_fill_pmcid = "pmcid" in missing
                    can_fill_abs = ("abstract" in missing) and bool(abstract.strip())
                    if can_fill_pmcid or can_fill_abs:
                        records.append(rec)
                        backfill_count += 1
                        if can_fill_pmcid:
                            missing.discard("pmcid")
                        if can_fill_abs:
                            missing.discard("abstract")
                        if not missing:
                            needs_backfill.pop(key, None)
                continue

            records.append(rec)
            new_count += 1
            page_new += 1
            if inflight and key:
                known_keys.add(key)

            if new_count >= limit:
                break

        pages_scanned += 1
        empty_pages = (empty_pages + 1) if page_new == 0 else 0

        if new_count >= limit:
            break
        if inflight:
            if early_stop > 0 and empty_pages >= early_stop:
                break
            if pages_scanned >= max_pages:
                break

        next_cursor = data.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor_mark:
            break
        cursor_mark = next_cursor
        time.sleep(EUROPEPMC_DELAY)

    if inflight:
        bf_msg = f", backfill {backfill_count}" if backfill_count else ""
        print(
            f"  Found {new_count} new open-access candidates with PMCID "
            f"(skipped {skipped_dup} dup{bf_msg}, {pages_scanned} pages)"
        )
    else:
        print(f"  Found {len(records)} open-access candidates with PMCID")
    return records


def run_europepmc_search(
    max_papers: int | None = None,
    text_dir: str = TEXT_DIR,
) -> dict:
    """
    主動向 Europe PMC 搜尋目標金屬的開放腐蝕全文 XML，
    解析並寫入 data/<METAL>/texts/（含去重與 XML 表格升級）。
    """
    candidates = search_europepmc_metadata(max_papers=max_papers)
    if not candidates:
        return {"new_txt": 0, "upgraded_tables": 0, "skipped": 0, "failed": 0}

    new_txt = 0
    upgraded_tables = 0
    skipped = 0
    failed = 0
    total_tables = 0

    for item in tqdm(candidates, desc="Fetching Europe PMC XML"):
        pmcid = item["pmcid"]
        base_name = sanitize_base_name(item["doi"] or pmcid)
        text_path = os.path.join(text_dir, base_name + ".txt")
        epmc_marker = os.path.join(text_dir, base_name + ".epmc_done")

        if os.path.exists(epmc_marker) and os.path.exists(text_path):
            skipped += 1
            continue

        xml_text = fetch_fulltext_xml(pmcid)
        if not xml_text:
            failed += 1
            time.sleep(EUROPEPMC_DELAY)
            continue

        w_txt, u_tbl, n_tbl = save_xml_to_texts(xml_text, base_name, text_dir)
        total_tables += n_tbl
        if w_txt:
            new_txt += 1
        elif u_tbl:
            upgraded_tables += 1
        else:
            skipped += 1

        # 標記已完成 XML 抽取，避免下次重跑重複下載同一篇 XML
        try:
            with open(epmc_marker, "w", encoding="utf-8") as f:
                f.write(pmcid)
        except OSError:
            pass

        time.sleep(EUROPEPMC_DELAY)

    print(
        f"  ✅ Europe PMC 完成 — 新增全文: {new_txt}, 升級 XML 表格: {upgraded_tables}, "
        f"跳過: {skipped}, 失敗: {failed}（共擷取 {total_tables} 張結構化表格）"
    )
    return {
        "new_txt": new_txt,
        "upgraded_tables": upgraded_tables,
        "skipped": skipped,
        "failed": failed,
        "total_tables": total_tables,
    }


if __name__ == "__main__":
    run_europepmc_search()
