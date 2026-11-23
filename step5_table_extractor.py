"""
Step 5 — Column-mapped structured table parser.
"""

import json
import re
from collections import Counter

import numpy as np

from config import (
    METAL_SYMBOL,
    MIN_MATERIAL_MENTIONS,
    TAFEL_MAX,
    TAFEL_MIN,
)
from step5_constants import (
    convert_log_icorr_to_uA_cm2,
    convert_to_she,
    normalize_ecorr_to_v,
    normalize_icorr_to_uA_cm2,
    normalize_text,
    parse_scientific,
)
from step5_text_extractor import (
    assign_material,
    format_counts,
    generate_solution_id,
    select_environment_for_record,
    select_ref_electrode,
)

# ====================================================================
# 10. TABLE EXTRACTION (Column-Mapped Structured Parser)
# ====================================================================

_COL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("sample", re.compile(r"\b(?:specimen|sample|electrode|alloy|material|coating|inhibitor|catalyst|system|condition|substrate)\b", re.I)),
    ("Ecorr", re.compile(r"(?:e\s*_?\s*corr|corrosion\s+potential|\bocp\b)", re.I)),
    ("log_icorr", re.compile(r"\b(?:log|lg)\s*\(?\s*[ij]\s*_?\s*corr", re.I)),
    ("icorr", re.compile(r"(?:[ij]\s*_?\s*corr|corrosion\s+current)", re.I)),
    ("beta_a", re.compile(r"(?:β\s*_?\s*a|\bb\s*_?\s*a\b|anodic\s+(?:tafel|slope))", re.I)),
    ("beta_c", re.compile(r"(?:β\s*_?\s*c|\bb\s*_?\s*c\b|cathodic\s+(?:tafel|slope))", re.I)),
    ("tafel", re.compile(r"(?:tafel\s+slope|\btafel\b)", re.I)),
    ("Rp", re.compile(r"(?:\br\s*_?\s*p\b|polarization\s+resistance)", re.I)),
    ("CR", re.compile(r"(?:corrosion\s+rate|\bcr\b|\bv\s*corr\b)", re.I)),
]

_ECHEM_COL_ROLES = {"Ecorr", "log_icorr", "icorr", "beta_a", "beta_c", "tafel", "Rp", "CR"}
_UNIT_IN_PARENS_RE = re.compile(r"[\(\[]([^\)\]]+)[\)\]]")
_TABLE_CAPTION_RE = re.compile(
    r"(?:^|\n)\s*Table\s+(\d+)\s*[\.:\-–—]?\s*([^\n]{5,240}(?:\n[^\n]{1,160})?)",
    re.I,
)
_TABLE_FOOTNOTE_RE = re.compile(
    r"(?:^|\n)\s*(?:Note\s*:|Notes\s*:|\*+\s*|[a-c]\s+(?=[A-Z0-9])|[†‡]\s*)([^\n]{5,200})",
    re.I,
)


def extract_table_captions(body_text: str) -> dict[int, str]:
    """從內文擷取 Table N 的標題與鄰近註腳說明，回傳 {table_idx_0based: caption_and_notes}。"""
    captions: dict[int, str] = {}
    if not body_text:
        return captions
    matches = list(_TABLE_CAPTION_RE.finditer(body_text))
    for i, m in enumerate(matches):
        try:
            t_num = int(m.group(1))
        except ValueError:
            continue
        idx = t_num - 1
        if idx < 0 or idx in captions:
            continue
        cap = re.sub(r"\s+", " ", m.group(2)).strip()
        # 往後搜尋至下一個 Table 標題前（最多 1500 字元）擷取可能的表格註腳
        search_end = min(len(body_text), m.end() + 1500)
        if i + 1 < len(matches):
            search_end = min(search_end, matches[i + 1].start())
        window = body_text[m.end():search_end]
        footnotes = [
            re.sub(r"\s+", " ", fm.group(1)).strip()
            for fm in _TABLE_FOOTNOTE_RE.finditer(window)
        ][:3]
        captions[idx] = " | ".join([cap] + footnotes) if footnotes else cap
    return captions


def _classify_table_header(cell: str) -> tuple[str | None, str]:
    c = normalize_text(str(cell or "")).replace("\n", " ").strip()
    if not c:
        return None, ""
    parens = [m.group(1).strip() for m in _UNIT_IN_PARENS_RE.finditer(c) if m.group(1).strip()]
    # 若表頭有多組括號（如 "log(icorr) (A/cm2)"），略過指標名稱本身，優先取最後一組單位括號
    unit = ""
    for p in reversed(parens):
        if not re.fullmatch(r"[ij]\s*_?\s*corr|e\s*_?\s*corr", p, re.I):
            unit = p
            break
    for role, pat in _COL_PATTERNS:
        if pat.search(c):
            return role, unit
    return None, unit


def parse_structured_table(
    table: list[list[str]],
    source_id: str,
    table_idx: int,
    paper_counts: Counter | None = None,
    paper_env: dict | None = None,
) -> list[dict]:
    """以表頭欄位對齊的方式解析電化學表格，精準配對每一列的樣品與數值。"""
    if not table or len(table) < 2:
        return []
    max_cols = max(len(r) for r in table)
    if max_cols < 2:
        return []

    # 在前兩列尋找表頭列
    header_row_idx = None
    col_map: dict[int, tuple[str, str]] = {}
    for r_idx in range(min(2, len(table))):
        cand_map: dict[int, tuple[str, str]] = {}
        for c_idx, cell in enumerate(table[r_idx]):
            role, unit = _classify_table_header(cell)
            if role and role not in {v[0] for v in cand_map.values()}:
                cand_map[c_idx] = (role, unit)
        if any(r in _ECHEM_COL_ROLES for r, _ in cand_map.values()):
            header_row_idx = r_idx
            col_map = cand_map
            break

    if header_row_idx is None:
        return []

    # 若下一列是純單位列（例如 "(mV)", "(mA/cm2)"），合併單位並跳過該列
    data_start = header_row_idx + 1
    if data_start < len(table):
        next_row = table[data_start]
        non_empty = [normalize_text(str(c or "")).strip() for c in next_row if str(c or "").strip()]
        if non_empty and all(parse_scientific(x) is None and len(x) <= 20 for x in non_empty):
            for c_idx, (role, unit) in list(col_map.items()):
                if not unit and c_idx < len(next_row):
                    u_txt = normalize_text(str(next_row[c_idx] or "")).strip("()[] ")
                    if u_txt:
                        col_map[c_idx] = (role, u_txt)
            data_start += 1

    # 找 sample 欄位（若表頭沒標出，預設用第 0 欄只要它不是電化學數值欄）
    sample_col = next((i for i, (r, _) in col_map.items() if r == "sample"), None)
    if sample_col is None and 0 not in col_map:
        sample_col = 0

    paper_env = paper_env or {}
    paper_els = paper_env.get("electrolytes", [])
    paper_ph_counts = paper_env.get("ph_counts", Counter())
    paper_temp_counts = paper_env.get("temp_counts", Counter())
    ref_counts = paper_env.get("ref_counts", Counter())
    refs = paper_env.get("refs", [])
    table_captions = paper_env.get("table_captions") or {}
    table_caption = table_captions.get(table_idx, "")
    if not table_caption and len(table_captions) == 1:
        # 若整篇論文只有單一編號表格（例如跨頁拆表或無編號附表），沿用該唯一表題
        table_caption = next(iter(table_captions.values()))
    header_text = " | ".join(normalize_text(str(c or "")) for c in table[header_row_idx])
    table_ref = select_ref_electrode([header_text, table_caption], ref_counts)
    primary_ref = table_ref or (refs[0] if refs else None)
    ordered_refs = (
        [primary_ref] + [r for r in refs if r != primary_ref]
        if primary_ref
        else list(refs)
    )
    paper_reactions = paper_env.get("reactions", [])
    if "corrosion" not in paper_reactions and any(
        r in {"Ecorr", "log_icorr", "icorr", "Rp", "CR"} for r, _ in col_map.values()
    ):
        paper_reactions = list(paper_reactions) + ["corrosion"]

    materials_sorted, materials_counts_str = format_counts(
        paper_counts or Counter(), MIN_MATERIAL_MENTIONS
    )
    paper_primary = paper_counts.most_common(1)[0][0] if paper_counts else None

    records = []
    for row_offset, row in enumerate(table[data_start:], start=1):
        norm_row = [normalize_text(str(c or "")).replace("\n", " ").strip() for c in row]
        row_text = " | ".join(norm_row)
        sample_name = norm_row[sample_col] if (sample_col is not None and sample_col < len(norm_row)) else None
        if sample_name == "":
            sample_name = None

        rec_notes = []
        ecorr_v_orig = ecorr_v_she = ecorr_raw = ecorr_unit = None
        icorr_uA = icorr_raw = icorr_unit = log_icorr = None
        beta_a = beta_c = tafel_gen = None
        cr_val = cr_unit = rp_val = rp_unit = None

        for c_idx, (role, unit) in col_map.items():
            if role == "sample" or c_idx >= len(norm_row):
                continue
            cell_str = norm_row[c_idx]
            val = parse_scientific(cell_str)
            if val is None:
                continue

            if role == "Ecorr":
                v, warn = normalize_ecorr_to_v(val, unit)
                ecorr_v_orig = v
                ecorr_raw = cell_str
                ecorr_unit = unit or "mV (inferred)"
                if warn:
                    rec_notes.append(warn)
                if v is not None and primary_ref:
                    ecorr_v_she = convert_to_she(
                        v, primary_ref,
                        paper_ph_counts.most_common(1)[0][0] if paper_ph_counts else None,
                    )
            elif role == "log_icorr":
                converted = convert_log_icorr_to_uA_cm2(val, unit)
                if converted is not None:
                    icorr_uA = converted
                    icorr_raw = cell_str
                    icorr_unit = f"log({unit or 'A/cm2'})"
                    log_icorr = np.log10(icorr_uA * 1e-6)
                    rec_notes.append("icorr converted from log(icorr)")
            elif role == "icorr":
                converted = normalize_icorr_to_uA_cm2(val, unit or "uA/cm2")
                if converted is not None:
                    icorr_uA = converted
                    icorr_raw = cell_str
                    icorr_unit = unit or "uA/cm2"
                    log_icorr = np.log10(icorr_uA * 1e-6)
                else:
                    rec_notes.append(f"rejected non-positive icorr={cell_str}")
            elif role == "beta_a":
                v = abs(val)
                if unit and "v" in unit.lower() and "mv" not in unit.lower():
                    v *= 1000.0
                if unit and "dec" not in unit.lower() and "mv" in unit.lower():
                    rec_notes.append(f"βa unit '({unit})' interpreted as mV/dec")
                if TAFEL_MIN <= v <= 1000:
                    beta_a = v
                    if v > TAFEL_MAX:
                        rec_notes.append(f"βa={v:.1f} exceeds TAFEL_MAX={TAFEL_MAX}")
            elif role == "beta_c":
                v = abs(val)
                if unit and "v" in unit.lower() and "mv" not in unit.lower():
                    v *= 1000.0
                if unit and "dec" not in unit.lower() and "mv" in unit.lower():
                    rec_notes.append(f"βc unit '({unit})' interpreted as mV/dec")
                if TAFEL_MIN <= v <= 1000:
                    beta_c = v
                    if v > TAFEL_MAX:
                        rec_notes.append(f"βc={v:.1f} exceeds TAFEL_MAX={TAFEL_MAX}")
            elif role == "tafel":
                v = abs(val)
                if unit and "v" in unit.lower() and "mv" not in unit.lower():
                    v *= 1000.0
                if TAFEL_MIN <= v <= 1000:
                    tafel_gen = str(round(v, 2))
            elif role == "CR":
                cr_val = val
                cr_unit = unit or "mm/yr"
            elif role == "Rp":
                rp_val = val
                rp_unit = unit or "ohm·cm2"

        if not any(x is not None for x in (ecorr_v_orig, icorr_uA, beta_a, beta_c, tafel_gen, cr_val, rp_val)):
            continue

        ctxs = [sample_name or "", row_text, table_caption]
        mat_id, mat_method = assign_material([sample_name or "", row_text], paper_counts or Counter())
        rec_els, rec_ph, rec_temp = select_environment_for_record(
            ctxs, paper_els, paper_ph_counts, paper_temp_counts
        )
        # 若環境已解析出該列/該表專屬 pH 且參考電極為 RHE，重新換算 Ecorr_V_SHE
        if ecorr_v_orig is not None and primary_ref:
            ecorr_v_she = convert_to_she(ecorr_v_orig, primary_ref, rec_ph)
        sol_id = generate_solution_id(rec_els, rec_ph, rec_temp) if (rec_els or rec_ph is not None) else None

        rec = {
            "source_id": f"{source_id}_table{table_idx + 1}_row{row_offset}",
            "sample_name": sample_name,
            "materials_raw": sample_name or "",
            "materials_normalised": materials_sorted,
            "material_mention_counts": materials_counts_str,
            "paper_primary_material": paper_primary,
            "target_mentions": (paper_counts or Counter()).get(METAL_SYMBOL, 0),
            "material_id_assigned": mat_id,
            "material_assignment_method": mat_method,
            "is_target_metal": (mat_id == METAL_SYMBOL),
            "metal_form": paper_env.get("metal_form"),
            "surface_preparation": paper_env.get("surface_preparation"),
            "record_electrolytes": rec_els,
            "electrolytes_raw": "; ".join(f"{e['conc']} {e['unit']} {e['species']}" for e in rec_els),
            "record_ph": rec_ph,
            "ph_values": str(rec_ph) if rec_ph is not None else None,
            "record_temp": rec_temp,
            "temperature_C_values": str(rec_temp) if rec_temp is not None else None,
            "solution_id_assigned": sol_id,
            "reference_electrodes": "; ".join(ordered_refs) if ordered_refs else None,
            "deaerated": paper_env.get("deaerated", False),
            "aerated": paper_env.get("aerated", False),
            "o2_saturated": paper_env.get("o2_saturated", False),
            "measurement_methods": paper_env.get("measurement_methods"),
            "scan_rates_raw": paper_env.get("scan_rates_raw"),
            "reactions": "; ".join(paper_reactions) if paper_reactions else None,
            "immersion_times_raw": paper_env.get("immersion_times_raw"),
            "extraction_method": "table_column_mapped",
            "context": f"Table {table_idx + 1}, Row {row_offset}: {row_text}",
            "Ecorr_V_original": ecorr_v_orig,
            "Ecorr_unit_original": ecorr_unit,
            "Ecorr_raw": ecorr_raw,
            "Ecorr_V_SHE": ecorr_v_she,
            "icorr_uA_cm2": icorr_uA,
            "icorr_raw": icorr_raw,
            "icorr_unit_original": icorr_unit,
            "log_icorr": log_icorr,
            "beta_a_mV_dec": beta_a,
            "beta_c_mV_dec": beta_c,
            "tafel_slopes_general_mV_dec": tafel_gen,
            "CR_reported_value": cr_val,
            "CR_reported_unit": cr_unit,
            "Rp_value": rp_val,
            "Rp_unit": rp_unit,
            "extra_quality_notes": "; ".join(rec_notes) if rec_notes else "",
        }
        records.append(rec)

    return records


def extract_from_tables(
    tables_json_path: str,
    source_id: str,
    paper_counts: Counter | None = None,
    paper_env: dict | None = None,
) -> list[dict]:
    records = []
    try:
        with open(tables_json_path, "r", encoding="utf-8") as f:
            tables = json.load(f)
    except Exception:
        return records

    for table_idx, table in enumerate(tables):
        parsed = parse_structured_table(
            table, source_id, table_idx,
            paper_counts=paper_counts,
            paper_env=paper_env,
        )
        records.extend(parsed)

    return records
