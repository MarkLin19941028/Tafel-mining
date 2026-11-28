"""
Step 6b: 非破壞性電化學腐蝕資料庫清洗、異常標記與權威基準驗證管線

功能：
1. 以「純新增標記欄位」方式處理 data/merged/electrochemical_database.csv，不刪除任何原始資料列、
   不改寫原始 Ecorr_V_SHE / icorr_uA_cm2 等欄位。
2. 執行四大品質與分類規則：
   - 物理邊界（Hard Bounds）：Ecorr ∈ [-2.0 (-2.2 for Mg), +1.5] V vs SHE，icorr ∈ [1e-4, 1e4] µA/cm²
   - 跨領域數據污染：僅比對論文 Title + Abstract（絕不誤用含陰極反應的 reactions 欄），排除電催化/電池污染
   - 表頭 −Ecorr 負號遺失還原：直接稽核對應 *_tables.json 表頭，於 Ecorr_V_SHE_cleaned 提供可驗證的還原值
   - 材料與環境精細分類：區分 pure_metal / nitride_barrier / carbon_mild_steel / stainless_steel / alloy /
     solder_alloy / coating_composite / oxide_compound 與 general_neutral / acidic / alkaline / cmp_oxidizing_etch 等
3. 依據 reference_benchmarks.json 產出清洗前後統計與四級權威基準吻合度評估報告（cleaning_validation_report.md）。

常用指令：
  .venv/bin/python step6b_clean_and_validate.py
  .venv/bin/python step6b_clean_and_validate.py --no-inplace --report-path data/merged/cleaning_validation_report_baseline.md
"""

from __future__ import annotations

import argparse
from datetime import datetime
from enum import IntEnum
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

import config
from step5_constants import NEGATED_ECORR_HEADER_NOTE, convert_to_she, get_ecorr_hard_bounds, normalize_text
from step5_table_extractor import classify_table_header, header_is_negated_ecorr

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARKS_PATH = PROJECT_ROOT / "reference_benchmarks.json"

ADDITIVE_CLEANING_COLUMNS: tuple[str, ...] = (
    "quality_flag",
    "exclusion_reason",
    "material_category",
    "environment_type",
    "Ecorr_V_SHE_cleaned",
    "correction_note",
)

NOBLE_METALS = frozenset({"Pt", "Ru", "Ir", "Pd", "Au", "Rh"})
NITRIDE_BARRIERS = frozenset({"TiN", "TaN", "MoN", "WN", "TiAlN"})
BENCHMARK_PURE_CATEGORIES = frozenset({"pure_metal", "nitride_barrier", "carbon_mild_steel"})

_TABLE_ROW_RE = re.compile(r"_table(\d+)_row\d+$")

_STAINLESS_RE = re.compile(
    r"\b(?:stainless|ss\s*3\d{2}|304l?|316l?|321|430|2205|2507|duplex|superduplex|austenitic|passivat(?:ed|ion))\b",
    re.I,
)
_CARBON_STEEL_RE = re.compile(
    r"\b(?:carbon\s+steel|mild\s+steel|low[- ]carbon\s+steel|c[- ]steel|q235|x\d{2}|"
    r"api\s*5\s*l|16mn|s400|armco|pure\s+iron|pure\s+fe|n80|p110|l80|j55|aisi\s*10\d{2})\b",
    re.I,
)
_SOLDER_RE = re.compile(
    r"\b(?:solder|sac\s*\d+|sn[- ]?3(?:\.0)?ag|sn[- ]?ag|sn[- ]?cu|sn[- ]?bi|sn[- ]?zn|sn[- ]?pb)\b",
    re.I,
)
_OXIDE_COMPOUND_RE = re.compile(
    r"\b(?:mno2|mn2o3|mn3o4|pbo2|ruo2|tio2|cu2o|cuo|fe2o3|fe3o4|manganese\s+(?:di)?oxide|"
    r"lead\s+(?:di)?oxide|ruthenium\s+(?:di)?oxide|birnessite)\b",
    re.I,
)
_COATING_SAMPLE_RE = re.compile(
    r"(?:@|\b(?:coating|coated|composite|nanocomposite|hybrid|doped|graphene|rgo|pvb|"
    r"epoxy|polymer|silane|mof|ldh|chitosan|polyaniline|pani|ppy|phosphatiz(?:ing|ed)|"
    r"anodiz(?:ing|ation|ed))\b|\b(?:film|layer)\s+on\b)",
    re.I,
)
_COATING_PAPER_RE = re.compile(
    r"\b(?:phosphatiz(?:ing|ed)|phosphate\s+coating|conversion\s+coating|anodiz(?:ing|ation|ed))\b",
    re.I,
)
_ALLOY_SAMPLE_RE = re.compile(
    r"\b(?:alloy|alloys|brass|bronze|inconel|monel|hastelloy|zircaloy|high[- ]?mn|"
    r"hadfield|twip|high[- ]?entropy|medium[- ]?entropy|hea|femn|fe[- ]?mn|ti[- ]?6al[- ]?4v|"
    r"aa\s*\d{4}|az\d{2}|mg[- ]?ca)\b",
    re.I,
)
_PURE_METAL_EXPLICIT_RE = re.compile(
    r"\bpure\s+(?:iron|fe|copper|cu|titanium|ti|aluminum|aluminium|al|cobalt|co|"
    r"tungsten|w|molybdenum|mo|nickel|ni|magnesium|mg|manganese|mn|zinc|zn|"
    r"chromium|cr|tantalum|ta|ruthenium|ru|silver|ag|tin|sn|platinum|pt|palladium|pd|gold|au|iridium|ir|lead|pb)\b",
    re.I,
)
_ANODIZING_RE = re.compile(r"\banodiz(?:ing|ation|ed)\b", re.I)

_CMP_ENV_RE = re.compile(
    r"\b(?:cmp|chemical\s+mechanical|slurry|h2o2|hydrogen\s+peroxide|kio4|kio3|"
    r"periodate|persulfate|kmno4|permanganate|naclo|hypochlorite|wet\s+etching|etchant|"
    r"sc-?1|fe\(no3\)3|ferric\s+nitrate)\b",
    re.I,
)
_ORGANIC_SOLVENT_RE = re.compile(
    r"\b(?:non[- ]?aqueous|nonaqueous|organic\s+solvent|organic\s+media|methanol|ethanol|"
    r"ethylene\s+glycol|dmso|dmf|acetonitrile|nmp|deep\s+eutectic|ionic\s+liquid)\b",
    re.I,
)
_ACID_SOLVENT_RE = re.compile(
    r"\b(?:hcl|h2so4|hno3|hclo4|hf|h3po4|hydrochloric|sulfuric|nitric|citric|oxalic|acetic|acidic)\b",
    re.I,
)
_BASE_SOLVENT_RE = re.compile(
    r"\b(?:naoh|koh|tmah|nh4oh|ammonia|alkaline|caoh2|ca\(oh\)2|sc-?1)\b",
    re.I,
)
_NEUTRAL_AQUEOUS_RE = re.compile(
    r"\b(?:nacl|seawater|sea\s+water|saline|pbs|ringer|hank|na2so4|neutral|tap\s+water|simulated\s+body)\b",
    re.I,
)
_ENERGY_DOMAIN_STRICT_RE = re.compile(
    r"\b(?:electrocatalyst|electrocatalysis|water\s+splitting|supercapacitor|pseudocapacitor|"
    r"zn[- ]?ion\s+battery|zinc[- ]?air|zn[- ]?air|al[- ]?air|lead[- ]?acid|li[- ]?ion\s+battery|"
    r"solar\s+cell|perovskite)\b",
    re.I,
)
_CATALYSIS_REACTION_TITLE_RE = re.compile(
    r"\b(?:her|oer|orr|hydrogen\s+evolution|oxygen\s+evolution|oxygen\s+reduction|fuel\s+cell)\b",
    re.I,
)
_SUBSTRATE_OTHER_RE = re.compile(
    r"\b(?:carbon\s+steel|mild\s+steel|stainless\s+steel|q235|x70|copper\s+alloy|"
    r"aluminum\s+alloy|aluminium\s+alloy|magnesium\s+alloy|nickel\s+(?:coating|corrosion|substrate))\b",
    re.I,
)


def _compile_keyword_regex(keywords: list[str]) -> re.Pattern | None:
    cleaned = [k.strip() for k in keywords if k and k.strip()]
    if not cleaned:
        return None
    parts = [r"(?<![A-Za-z0-9])" + re.escape(k) + r"(?![A-Za-z0-9])" for k in cleaned]
    return re.compile("|".join(parts), re.I)


def load_reference_benchmarks(path: Path | None = None) -> dict[str, Any]:
    """載入權威電化學基準設定檔（reference_benchmarks.json）。"""
    target = Path(path) if path is not None else DEFAULT_BENCHMARKS_PATH
    if not target.is_file():
        raise FileNotFoundError(f"找不到基準設定檔：{target}")
    return json.loads(target.read_text(encoding="utf-8"))


def _base_source_id(source_ref: Any) -> str:
    return _TABLE_ROW_RE.sub("", str(source_ref or "").strip())


def _normalize_doi_to_source_id(doi: str) -> str:
    clean = doi.strip()
    clean = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", clean, flags=re.I)
    return re.sub(r"[^a-zA-Z0-9._-]", "_", clean)


def build_paper_metadata_index(data_dir: Path) -> dict[str, dict[str, str]]:
    """建立 source_id（小寫）→ {title, abstract} 對照索引（含 DOI 與 PMCID）。"""
    index: dict[str, dict[str, str]] = {}
    if not data_dir.is_dir():
        return index
    for meta_csv in sorted(data_dir.glob("*/metadata/*_tafel_metadata.csv")):
        try:
            mdf = pd.read_csv(meta_csv)
        except Exception:
            continue
        for _, r in mdf.iterrows():
            title = _clean_str(r.get("title"))
            abstract = _clean_str(r.get("abstract"))
            entry = {"title": title, "abstract": abstract}
            doi = _clean_str(r.get("doi"))
            if doi:
                sid = _normalize_doi_to_source_id(doi).lower()
                if sid and (sid not in index or not index[sid]["title"]):
                    index[sid] = entry
            pmcid = _clean_str(r.get("pmcid"))
            if pmcid:
                pid = pmcid.lower()
                if pid not in index or not index[pid]["title"]:
                    index[pid] = entry
    return index


def build_table_json_index(data_dir: Path) -> dict[str, Path]:
    """掃描 data/*/texts/*_tables.json，建立 base_source_id（小寫）→ Path 索引。"""
    index: dict[str, Path] = {}
    if not data_dir.is_dir():
        return index
    for p in sorted(data_dir.glob("*/texts/*_tables.json")):
        key = p.name[: -len("_tables.json")].lower()
        index.setdefault(key, p)
    return index


def _inspect_table_ecorr_negation(
    table_path: Path,
    table_idx_0based: int,
    cache: dict[tuple[str, int], bool],
) -> bool:
    """檢查指定 *_tables.json 中對應表格的 Ecorr 表頭是否為負號前綴（−Ecorr）。"""
    ckey = (str(table_path), table_idx_0based)
    if ckey in cache:
        return cache[ckey]
    is_neg = False
    try:
        tables = json.loads(table_path.read_text(encoding="utf-8"))
        if isinstance(tables, list) and 0 <= table_idx_0based < len(tables):
            tbl = tables[table_idx_0based]
            if isinstance(tbl, list) and len(tbl) >= 2:
                for r_idx in range(min(2, len(tbl))):
                    row = tbl[r_idx]
                    ecorr_cols = [
                        (c_idx, cell)
                        for c_idx, cell in enumerate(row)
                        if classify_table_header(cell)[0] == "Ecorr"
                    ]
                    if ecorr_cols:
                        is_neg = any(header_is_negated_ecorr(cell) for _, cell in ecorr_cols)
                        break
    except Exception:
        is_neg = False
    cache[ckey] = is_neg
    return is_neg


def classify_material_category(
    material_id: Any,
    material_class: Any,
    sample_name: Any,
    title: str = "",
    abstract: str = "",
) -> str:
    """綜合樣品名與來源論文標題/摘要，區分純金屬、阻障層氮化物、碳鋼、不鏽鋼、焊料、合金與塗層氧化物。"""
    mid = _clean_str(material_id)
    mcls = _clean_str(material_class)
    sname = _clean_str(sample_name)
    paper_text = f"{title} {abstract}".strip()

    # 1. 明確化合物 / 塗層樣品訊號優先
    if mcls == "coating_composite" or (sname and _COATING_SAMPLE_RE.search(sname)):
        return "coating_composite"
    if _OXIDE_COMPOUND_RE.search(mid) or (sname and _OXIDE_COMPOUND_RE.search(sname)):
        return "oxide_compound"
    if mid == "Fe":
        if sname and _STAINLESS_RE.search(sname):
            return "stainless_steel"
        if sname and _CARBON_STEEL_RE.search(sname):
            return "carbon_mild_steel"
    if mid == "Sn" and sname and _SOLDER_RE.search(sname):
        return "solder_alloy"
    if mcls == "alloy" or (sname and _ALLOY_SAMPLE_RE.search(sname)):
        return "alloy"

    # 2. 樣品名未明確宣告純金屬時，結合論文標題/摘要判讀實際研究基材
    explicit_pure = bool(sname and _PURE_METAL_EXPLICIT_RE.search(sname))
    if not explicit_pure:
        if mid in NITRIDE_BARRIERS:
            return "nitride_barrier"
        if mid == "Fe":
            if _STAINLESS_RE.search(paper_text):
                return "stainless_steel"
            if _CARBON_STEEL_RE.search(paper_text):
                return "carbon_mild_steel"
        if mid == "Sn" and _SOLDER_RE.search(paper_text):
            return "solder_alloy"
        if mid == "Mn":
            if _OXIDE_COMPOUND_RE.search(paper_text):
                return "oxide_compound"
            if _COATING_PAPER_RE.search(paper_text) or "manganese phosphat" in paper_text.lower():
                return "coating_composite"
            if re.search(
                r"\b(?:high[- ]?mn|hadfield|twip|austenitic|high[- ]?entropy|medium[- ]?entropy|"
                r"hea|fe[- ]?mn|femn\d*|steel|alloy|scaffold|mg[- ]?ca)\b",
                paper_text,
                re.I,
            ):
                return "alloy"
        if mid == "Pb" and _OXIDE_COMPOUND_RE.search(paper_text):
            return "oxide_compound"
        if mid == "Ru" and re.search(r"\bruo2\b|ruthenium\s+(?:di)?oxide", paper_text, re.I):
            if not re.search(r"\b(?:metallic|pure)\s+ru|\bru\s+(?:liner|film|wafer|cmp|coupon)\b", paper_text, re.I):
                return "oxide_compound"
        if mid == "Al" and _ANODIZING_RE.search(paper_text):
            if not re.search(r"\b(?:bare|uncoated|untreated|substrate)\b", sname, re.I):
                return "coating_composite"
        if _COATING_PAPER_RE.search(paper_text) and mid not in ("Fe", "TiN", "TaN", "MoN", "WN"):
            return "coating_composite"

    if mid in NITRIDE_BARRIERS:
        return "nitride_barrier"
    if mcls == "compound":
        return "oxide_compound"
    return "pure_metal"


def _clean_str(val: Any) -> str:
    """安全轉換欄位值為去除空白的字串；NaN / None / 'nan' 一律轉成空字串。"""
    if val is None or pd.isna(val):
        return ""
    s = str(val).strip()
    return "" if s.lower() == "nan" else s


def classify_environment_type(
    material_id: Any,
    title: str,
    abstract: str,
    solution_description: Any,
    ph_val: Any,
    sol_row: dict[str, Any] | None = None,
) -> str:
    """區分一般中性腐蝕、酸蝕、鹼蝕、強氧化 CMP 蝕刻、有機溶劑與電催化/能源體系。"""
    mid = str(material_id or "").strip()
    ta = f"{title} {abstract}".strip()
    sol_desc = _clean_str(solution_description)
    sol_name = _clean_str((sol_row or {}).get("solution_name"))
    sol_text = f"{sol_desc} {sol_name}".strip()
    sol_cat = _clean_str((sol_row or {}).get("solution_category")).lower()
    ox_type = _clean_str((sol_row or {}).get("oxidizer_type"))
    acid_type = _clean_str((sol_row or {}).get("acid_type"))
    base_type = _clean_str((sol_row or {}).get("base_type"))
    org_type = _clean_str((sol_row or {}).get("organic_solvent_type"))

    # 電催化 / 能源體系
    if mid in NOBLE_METALS and (
        _ENERGY_DOMAIN_STRICT_RE.search(ta) or _CATALYSIS_REACTION_TITLE_RE.search(ta)
    ):
        return "electrocatalysis_energy"
    if _ENERGY_DOMAIN_STRICT_RE.search(ta) or _CATALYSIS_REACTION_TITLE_RE.search(title):
        return "electrocatalysis_energy"

    # 強氧化 CMP / 濕蝕刻漿料（HNO3 本身多屬酸蝕，除非標題或溶液含 CMP/H2O2 等強氧化蝕刻訊號）
    has_non_nitric_ox = bool(ox_type and ox_type.upper() != "HNO3")
    if has_non_nitric_ox or _CMP_ENV_RE.search(f"{title} {sol_text}"):
        return "cmp_oxidizing_etch"
    if mid == "Ru" and _CMP_ENV_RE.search(ta):
        return "cmp_oxidizing_etch"

    # 有機 / 非水溶劑
    if org_type or _ORGANIC_SOLVENT_RE.search(sol_text):
        return "organic_solvent"

    # 解析有效 pH
    ph_f: float | None = None
    for cand in (ph_val, (sol_row or {}).get("pH")):
        if cand is not None and not pd.isna(cand):
            try:
                v = float(cand)
                if 0.0 <= v <= 14.0:
                    ph_f = v
                    break
            except (TypeError, ValueError):
                pass

    if ph_f is not None:
        if ph_f < 5.0:
            return "acidic_corrosion"
        if ph_f > 9.0:
            return "alkaline_corrosion"
        return "general_neutral"

    if acid_type or ox_type.upper() == "HNO3" or sol_cat in ("acidic", "strongly_acidic", "weakly_acidic") or _ACID_SOLVENT_RE.search(sol_text):
        return "acidic_corrosion"
    if base_type or sol_cat in ("alkaline", "strongly_alkaline", "weakly_alkaline") or _BASE_SOLVENT_RE.search(sol_text):
        return "alkaline_corrosion"
    if sol_cat == "neutral" or _NEUTRAL_AQUEOUS_RE.search(f"{sol_text} {title} {abstract}"):
        return "general_neutral"
    return "unknown"


def _detect_substrate_misattribution(
    material_id: str,
    sample_name: str,
    title: str,
) -> bool:
    """偵測明顯將緩蝕劑／配體／「lead-free」等詞誤歸屬為目標金屬的紀錄。"""
    if not title:
        return False
    t_low = title.lower()
    s_low = sample_name.lower()

    if material_id == "Pb":
        if "lead-free" in t_low or "lead free" in t_low:
            return True
        if not re.search(r"\b(?:pb|lead)\b(?!\s*-?\s*free)", title, re.I):
            return True
    if material_id == "Sn":
        if _SUBSTRATE_OTHER_RE.search(title) and not re.search(
            r"\b(?:sn|tin|tinplate|solder|sac\d+|sn[- ]?(?:ag|cu|bi|zn|pb))\b",
            title,
            re.I,
        ):
            if not re.search(r"\b(?:pure\s+sn|pure\s+tin)\b", s_low):
                return True
    if material_id in ("Ru", "Pt", "Pd", "Au", "Ir", "Rh"):
        # 若標題明確寫出其他基材（如 mild steel / carbon steel / nickel）的緩蝕劑或錯合物，而非該貴金屬本身為電極/阻障層
        if _SUBSTRATE_OTHER_RE.search(title) and re.search(
            r"\b(?:complex|inhibitor|derivative|surfactant|for\s+(?:mild|carbon|stainless)\s+steel|for\s+nickel)\b",
            title,
            re.I,
        ):
            if not re.search(
                rf"\b(?:{material_id.lower()}|ruthenium|platinum|palladium|gold|iridium|rhodium)\b\s*(?:liner|barrier|cmp|wafer|film|electrode|foil)",
                t_low,
            ):
                return True
    return False


def clean_and_validate_dataframe(
    echem_df: pd.DataFrame,
    benchmarks: dict[str, Any],
    *,
    sol_df: pd.DataFrame | None = None,
    paper_meta_index: dict[str, dict[str, str]] | None = None,
    table_json_index: dict[str, Path] | None = None,
) -> pd.DataFrame:
    """
    對電化學主表進行非破壞性清洗與標記：保留所有原始欄位與資料列，新增 6 個品管與分類欄位。
    """
    df = echem_df.copy()
    if df.empty:
        for col in ADDITIVE_CLEANING_COLUMNS:
            df[col] = pd.Series(dtype="object" if col != "Ecorr_V_SHE_cleaned" else "float64")
        return df

    meta = benchmarks.get("metadata", {})
    global_bounds = meta.get("global_hard_bounds", {})
    g_ecorr_lo, g_ecorr_hi = global_bounds.get("ecorr_v_she", [-2.0, 1.5])
    g_icorr_lo, g_icorr_hi = global_bounds.get("icorr_uA_cm2", [1e-4, 1e4])
    global_excl_kws = list(meta.get("global_exclude_keywords", []))
    noble_excl_re = _compile_keyword_regex(global_excl_kws + ["battery", "batteries"])
    mat_benchmarks: dict[str, dict[str, Any]] = benchmarks.get("materials", {})

    # 預編譯各材料排除關鍵字正則
    mat_excl_res: dict[str, re.Pattern] = {}
    for m_key, m_cfg in mat_benchmarks.items():
        kws = list(m_cfg.get("exclude_keywords", []))
        # Fe 的 stainless/passivated 交由 material_category='stainless_steel' 分類處理，不直接 EXCLUDE
        if m_key == "Fe":
            continue
        pat = _compile_keyword_regex(kws)
        if pat is not None:
            mat_excl_res[m_key] = pat

    sol_lookup: dict[str, dict[str, Any]] = {}
    if sol_df is not None and not sol_df.empty and "solution_id" in sol_df.columns:
        for _, srow in sol_df.drop_duplicates(subset=["solution_id"]).iterrows():
            sol_lookup[str(srow["solution_id"])] = srow.to_dict()

    meta_idx = paper_meta_index or {}
    tbl_idx = table_json_index or {}
    tbl_header_cache: dict[tuple[str, int], bool] = {}

    q_flags: list[str] = []
    excl_reasons: list[str] = []
    mat_cats: list[str] = []
    env_types: list[str] = []
    ecorr_cleaned_list: list[float | None] = []
    corr_notes: list[str] = []

    for _, row in df.iterrows():
        mid = _clean_str(row.get("material_id"))
        mcls = _clean_str(row.get("material_class"))
        sname = _clean_str(row.get("sample_name"))

        src_ref = _clean_str(row.get("source_reference"))
        base_src = _base_source_id(src_ref)
        paper_info = meta_idx.get(base_src.lower(), {})
        title = paper_info.get("title", "")
        abstract = paper_info.get("abstract", "")
        title_abstract = f"{title} {abstract}".strip()

        sol_id = _clean_str(row.get("solution_id"))
        sol_row = sol_lookup.get(sol_id)

        # 1. 表頭 −Ecorr 負號遺失還原（僅對 table_column_mapped + Ecorr_V_original > 0 且表頭明確含 −Ecorr 者）
        ecorr_she_raw = row.get("Ecorr_V_SHE")
        ecorr_orig = row.get("Ecorr_V_original")
        ecorr_cleaned = float(ecorr_she_raw) if pd.notna(ecorr_she_raw) else np.nan
        note_parts: list[str] = []

        qnotes = _clean_str(row.get("quality_notes"))
        if (
            _clean_str(row.get("data_extraction_method")) == "table_column_mapped"
            and pd.notna(ecorr_orig)
            and float(ecorr_orig) > 0
            and NEGATED_ECORR_HEADER_NOTE not in qnotes
        ):
            m_tbl = _TABLE_ROW_RE.search(src_ref)
            tbl_path = tbl_idx.get(base_src.lower())
            if m_tbl and tbl_path is not None:
                tbl_0idx = int(m_tbl.group(1)) - 1
                if _inspect_table_ecorr_negation(tbl_path, tbl_0idx, tbl_header_cache):
                    ref_elec = row.get("reference_electrode")
                    ph_v = row.get("pH")
                    if pd.isna(ph_v) and sol_row:
                        ph_v = sol_row.get("pH")
                    ph_arg = float(ph_v) if pd.notna(ph_v) else None
                    if pd.notna(ref_elec) and str(ref_elec).strip():
                        recomputed = convert_to_she(-float(ecorr_orig), str(ref_elec), ph_arg)
                        if recomputed is not None:
                            ecorr_cleaned = round(float(recomputed), 4)
                            note_parts.append("negated_from_minus_ecorr_header")

        # 2. 材料與環境精細分類
        m_cat = classify_material_category(mid, mcls, sname, title=title, abstract=abstract)
        env_type = classify_environment_type(
            mid,
            title,
            abstract,
            row.get("solution_description"),
            row.get("pH"),
            sol_row=sol_row,
        )

        # 3. 硬邊界與領域污染排查（決定 quality_flag / exclusion_reason）
        reasons: list[str] = []

        mat_cfg = mat_benchmarks.get(mid, {})
        default_bounds = list(get_ecorr_hard_bounds(mid)) if "ecorr_v_she" not in global_bounds else [g_ecorr_lo, g_ecorr_hi]
        e_lo, e_hi = mat_cfg.get("ecorr_hard_bounds", default_bounds)
        if not pd.isna(ecorr_cleaned):
            if not (float(e_lo) <= float(ecorr_cleaned) <= float(e_hi)):
                reasons.append("ecorr_out_of_hard_bounds")

        icorr_val = row.get("icorr_uA_cm2")
        if pd.notna(icorr_val):
            ic_f = float(icorr_val)
            if not (float(g_icorr_lo) <= ic_f <= float(g_icorr_hi)):
                reasons.append("icorr_out_of_hard_bounds")

        # 跨領域污染（僅檢查 Title + Abstract，絕不誤用 reactions 欄）
        if title_abstract:
            if mid in NOBLE_METALS and noble_excl_re is not None and noble_excl_re.search(title_abstract):
                reasons.append("domain_pollution_noble_metal")
            elif mid in mat_excl_res and mat_excl_res[mid].search(title_abstract):
                if mid == "Al" and _ANODIZING_RE.search(title_abstract):
                    note_parts.append("anodizing_surface_treatment")
                elif mid == "Mn" and re.search(r"high[- ]?mn\s*steel", title_abstract, re.I) and not _OXIDE_COMPOUND_RE.search(title_abstract):
                    note_parts.append("high_mn_steel_alloy")
                else:
                    reasons.append("domain_pollution_material_keyword")

        if _detect_substrate_misattribution(mid, sname, title):
            reasons.append("substrate_misattributed")

        # 決策 quality_flag
        if reasons:
            q_flag = "EXCLUDE"
        elif (
            m_cat == "oxide_compound"
            or (mid == "Mn" and m_cat in ("alloy", "coating_composite"))
            or ("anodizing_surface_treatment" in note_parts)
        ):
            q_flag = "WARN"
        else:
            q_flag = "PASS"

        q_flags.append(q_flag)
        excl_reasons.append("; ".join(dict.fromkeys(reasons)))
        mat_cats.append(m_cat)
        env_types.append(env_type)
        ecorr_cleaned_list.append(None if pd.isna(ecorr_cleaned) else round(float(ecorr_cleaned), 4))
        corr_notes.append("; ".join(dict.fromkeys(note_parts)))

    df["quality_flag"] = q_flags
    df["exclusion_reason"] = excl_reasons
    df["material_category"] = mat_cats
    df["environment_type"] = env_types
    df["Ecorr_V_SHE_cleaned"] = ecorr_cleaned_list
    df["correction_note"] = corr_notes
    return df


class BenchmarkGrade(IntEnum):
    """基準吻合度嚴重度等級（數值越大代表偏離越嚴重）。"""

    MATCH = 0
    CLOSE = 1
    DEVIATE = 2


_GRADE_LABELS: dict[BenchmarkGrade, str] = {
    BenchmarkGrade.MATCH: "✅ 吻合",
    BenchmarkGrade.CLOSE: "🟡 接近",
    BenchmarkGrade.DEVIATE: "⚠️ 偏離",
}


def evaluate_benchmark_alignment(
    ecorr_med: float | None,
    icorr_med: float | None,
    n_valid: int,
    ecorr_range: list[float] | tuple[float, float] | None,
    icorr_range: list[float] | tuple[float, float] | None,
    *,
    n_ecorr: int | None = None,
    n_icorr: int | None = None,
) -> str:
    """
    四級權威基準吻合度評估：
    - 對應指標有效筆數 < 5 不納入定級；兩指標皆 < 5 → ℹ️ 樣本不足
    - 納入定級之指標皆落於區間內 → ✅ 吻合
    - 誤差在 ±0.15 V 及 ±0.5 個數量級（0.32× ~ 3.16×）內 → 🟡 接近
    - 超出容差 → ⚠️ 偏離
    """
    eff_n_e = n_valid if n_ecorr is None else n_ecorr
    eff_n_i = n_valid if n_icorr is None else n_icorr
    if (eff_n_e < 5 and eff_n_i < 5) or (ecorr_med is None and icorr_med is None):
        return "ℹ️ 樣本不足"
    if not ecorr_range and not icorr_range:
        return "—"

    statuses: list[BenchmarkGrade] = []
    if ecorr_med is not None and ecorr_range and eff_n_e >= 5:
        elo, ehi = float(ecorr_range[0]), float(ecorr_range[1])
        if elo <= ecorr_med <= ehi:
            statuses.append(BenchmarkGrade.MATCH)
        elif (elo - 0.15) <= ecorr_med <= (ehi + 0.15):
            statuses.append(BenchmarkGrade.CLOSE)
        else:
            statuses.append(BenchmarkGrade.DEVIATE)

    if icorr_med is not None and icorr_med > 0 and icorr_range and eff_n_i >= 5:
        ilo, ihi = float(icorr_range[0]), float(icorr_range[1])
        tol_factor = 10.0 ** 0.5  # ~3.162
        if ilo <= icorr_med <= ihi:
            statuses.append(BenchmarkGrade.MATCH)
        elif (ilo / tol_factor) <= icorr_med <= (ihi * tol_factor):
            statuses.append(BenchmarkGrade.CLOSE)
        else:
            statuses.append(BenchmarkGrade.DEVIATE)

    if not statuses:
        return "ℹ️ 樣本不足"
    return _GRADE_LABELS[max(statuses)]


def _fmt_stat(series: pd.Series, fmt: str) -> tuple[str, float | None, int]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "—", None, 0
    med = float(s.median())
    return f"{med:{fmt}} [{s.min():{fmt}}, {s.max():{fmt}}] (n={len(s)})", med, len(s)


def write_cleaning_validation_report(
    cleaned_df: pd.DataFrame,
    benchmarks: dict[str, Any],
    report_path: Path,
) -> Path:
    """產出清洗前後數值分佈、分層稽核與權威基準吻合度對比報告。"""
    mat_benchmarks: dict[str, dict[str, Any]] = benchmarks.get("materials", {})
    total = len(cleaned_df)
    q_counts = cleaned_df["quality_flag"].value_counts().to_dict() if total else {}
    n_pass = int(q_counts.get("PASS", 0))
    n_warn = int(q_counts.get("WARN", 0))
    n_excl = int(q_counts.get("EXCLUDE", 0))
    n_neg_step5 = int(
        cleaned_df["quality_notes"].fillna("").str.contains(NEGATED_ECORR_HEADER_NOTE).sum()
    ) if (total and "quality_notes" in cleaned_df.columns) else 0
    n_neg_step6b = int(
        cleaned_df["correction_note"].fillna("").str.contains("negated_from_minus_ecorr_header").sum()
    ) if total else 0
    n_neg_total = n_neg_step5 + n_neg_step6b

    lines: list[str] = [
        "# 電化學腐蝕參數資料庫清洗與權威基準驗證報告",
        "",
        f"> 產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"> 權威基準版本：`v{benchmarks.get('metadata', {}).get('version', '1.0')}`（條件：`{benchmarks.get('metadata', {}).get('reference_condition', '')}`）",
        "",
        "## 1. 非破壞性清洗總覽",
        "",
        "| 項目 | 筆數 | 佔比 |",
        "|---|---:|---:|",
        f"| 原始總紀錄數（完整保留） | {total} | 100.0% |",
        f"| `PASS`（高可信度腐蝕數據） | {n_pass} | {n_pass / total * 100:.1f}% |" if total else "| `PASS` | 0 | 0.0% |",
        f"| `WARN`（合法非標準態／塗層／高錳鋼等） | {n_warn} | {n_warn / total * 100:.1f}% |" if total else "| `WARN` | 0 | 0.0% |",
        f"| `EXCLUDE`（硬邊界異常／跨領域污染／誤歸屬） | {n_excl} | {n_excl / total * 100:.1f}% |" if total else "| `EXCLUDE` | 0 | 0.0% |",
        f"| 表頭 `−Ecorr` 負號還原合計（Step 5 擷取期 {n_neg_step5} + Step 6b 補救 {n_neg_step6b}） | {n_neg_total} | {n_neg_total / total * 100:.1f}% |" if total else "| `−Ecorr` 還原 | 0 | 0.0% |",
        "",
    ]

    if n_excl > 0:
        reason_counts: dict[str, int] = {}
        for rstr in cleaned_df.loc[cleaned_df["quality_flag"] == "EXCLUDE", "exclusion_reason"].fillna(""):
            for part in str(rstr).split(";"):
                p = part.strip()
                if p:
                    reason_counts[p] = reason_counts.get(p, 0) + 1
        lines.extend(["### 排除原因統計 (`exclusion_reason`)", "", "| 排除原因碼 | 筆數 |", "|---|---:|"])
        for reason, cnt in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{reason}` | {cnt} |")
        lines.append("")

    # 2. 各金屬清洗前後對比與基準評估
    lines.extend([
        "## 2. 各金屬清洗前後分佈與權威基準對比",
        "",
        "> 說明：**清洗前**使用原始 `Ecorr_V_SHE`；**清洗後 (All PASS)** 包含所有通過品管之腐蝕條件（含酸/鹼/CMP）；"
        "**基準條件子集**則篩選 `PASS` + `material_category ∈ {pure_metal, nitride_barrier, carbon_mild_steel}` + `environment_type == general_neutral` "
        "以對齊 25°C 中性含氧水溶液基準（評等基準：落於區間內=`✅ 吻合`；誤差 `≤ ±0.15 V` 且 `≤ ±0.5` decade=`🟡 接近`；`n < 5`=`ℹ️ 樣本不足`）。",
        "",
        "| 金屬 | 總筆數 | PASS | 基準子集 n | 清洗前 Ecorr [min, max] | PASS 全體 Ecorr | 基準子集 Ecorr | 權威 Ecorr (V SHE) | 清洗前 icorr (µA/cm²) | PASS 全體 icorr | 基準子集 icorr | 權威 icorr (µA/cm²) | 基準子集吻合度 |",
        "|---|---:|---:|---:|---|---|---|---|---|---|---|---|---|",
    ])

    for metal, grp in sorted(cleaned_df.groupby("material_id"), key=lambda kv: (-len(kv[1]), str(kv[0]))):
        m_str = str(metal)
        pass_grp = grp[grp["quality_flag"] == "PASS"]
        bench_grp = pass_grp[
            pass_grp["material_category"].isin(BENCHMARK_PURE_CATEGORIES)
            & (pass_grp["environment_type"] == "general_neutral")
        ]

        e_before_txt, _, _ = _fmt_stat(grp["Ecorr_V_SHE"], ".3f")
        e_pass_txt, _, _ = _fmt_stat(pass_grp["Ecorr_V_SHE_cleaned"], ".3f")
        e_bench_txt, e_bench_med, n_e_bench = _fmt_stat(bench_grp["Ecorr_V_SHE_cleaned"], ".3f")

        i_before_txt, _, _ = _fmt_stat(grp["icorr_uA_cm2"], ".3g")
        i_pass_txt, _, _ = _fmt_stat(pass_grp["icorr_uA_cm2"], ".3g")
        i_bench_txt, i_bench_med, n_i_bench = _fmt_stat(bench_grp["icorr_uA_cm2"], ".3g")

        cfg = mat_benchmarks.get(m_str, {})
        e_rng = cfg.get("ecorr_range")
        i_rng = cfg.get("icorr_range")
        e_rng_txt = f"[{e_rng[0]:+.2f}, {e_rng[1]:+.2f}]" if e_rng else "—"
        i_rng_txt = f"[{i_rng[0]:.4g}, {i_rng[1]:.4g}]" if i_rng else "—"

        grade = evaluate_benchmark_alignment(
            e_bench_med,
            i_bench_med,
            max(n_e_bench, n_i_bench),
            e_rng,
            i_rng,
            n_ecorr=n_e_bench,
            n_icorr=n_i_bench,
        )
        lines.append(
            f"| {m_str} | {len(grp)} | {len(pass_grp)} | {len(bench_grp)} | "
            f"{e_before_txt} | {e_pass_txt} | {e_bench_txt} | {e_rng_txt} | "
            f"{i_before_txt} | {i_pass_txt} | {i_bench_txt} | {i_rng_txt} | {grade} |"
        )
    lines.append("")

    # 3. 重點分層稽核（Fe, Ru, Mn, Sn, Pb）
    lines.extend([
        "## 3. 重點金屬材料／環境分層對比",
        "",
        "### 3.1 Fe 分層（純鐵／碳鋼 vs. 不鏽鋼）",
        "",
        "| `material_category` | 清洗後 PASS 筆數 | Ecorr_cleaned (V SHE) | icorr (µA/cm²) | 權威參考區間 |",
        "|---|---:|---|---|---|",
    ])
    fe_pass = cleaned_df[(cleaned_df["material_id"] == "Fe") & (cleaned_df["quality_flag"] == "PASS")]
    for cat in ("carbon_mild_steel", "stainless_steel", "pure_metal", "alloy", "coating_composite"):
        sub = fe_pass[fe_pass["material_category"] == cat]
        if sub.empty:
            continue
        et, _, _ = _fmt_stat(sub["Ecorr_V_SHE_cleaned"], ".3f")
        it, _, _ = _fmt_stat(sub["icorr_uA_cm2"], ".3g")
        ref_txt = (
            "Ecorr [-0.46, -0.36], icorr [10, 50]"
            if cat in ("carbon_mild_steel", "pure_metal")
            else ("Ecorr [+0.10, +0.25], icorr [0.1, 2]" if cat == "stainless_steel" else "—")
        )
        lines.append(f"| `{cat}` | {len(sub)} | {et} | {it} | {ref_txt} |")
    lines.append("")

    lines.extend([
        "### 3.2 Ru 環境分層（一般腐蝕 vs. CMP 強氧化蝕刻 vs. 已排除電催化）",
        "",
        "| `environment_type` / 狀態 | 筆數 | Ecorr_cleaned (V SHE) | icorr (µA/cm²) |",
        "|---|---:|---|---|",
    ])
    ru_all = cleaned_df[cleaned_df["material_id"] == "Ru"]
    for label, sub in [
        ("PASS — `general_neutral`", ru_all[(ru_all["quality_flag"] == "PASS") & (ru_all["environment_type"] == "general_neutral")]),
        ("PASS — `acidic_corrosion` / `alkaline_corrosion`", ru_all[(ru_all["quality_flag"] == "PASS") & (ru_all["environment_type"].isin(["acidic_corrosion", "alkaline_corrosion"]))]),
        ("PASS — `cmp_oxidizing_etch`", ru_all[(ru_all["quality_flag"] == "PASS") & (ru_all["environment_type"] == "cmp_oxidizing_etch")]),
        ("EXCLUDE（電催化／誤歸屬等）", ru_all[ru_all["quality_flag"] == "EXCLUDE"]),
    ]:
        if sub.empty:
            continue
        et, _, _ = _fmt_stat(sub["Ecorr_V_SHE_cleaned"], ".3f")
        it, _, _ = _fmt_stat(sub["icorr_uA_cm2"], ".3g")
        lines.append(f"| {label} | {len(sub)} | {et} | {it} |")
    lines.append("")

    lines.extend([
        "### 3.3 Mn / Sn / Pb 材料分類與排除明細",
        "",
        "| 金屬 | `material_category` | `quality_flag` | 筆數 | Ecorr_cleaned (V SHE) | icorr (µA/cm²) |",
        "|---|---|---|---:|---|---|",
    ])
    for m_chk in ("Mn", "Sn", "Pb"):
        sub_m = cleaned_df[cleaned_df["material_id"] == m_chk]
        if sub_m.empty:
            continue
        for (cat, qf), g in sub_m.groupby(["material_category", "quality_flag"]):
            et, _, _ = _fmt_stat(g["Ecorr_V_SHE_cleaned"], ".3f")
            it, _, _ = _fmt_stat(g["icorr_uA_cm2"], ".3g")
            lines.append(f"| {m_chk} | `{cat}` | `{qf}` | {len(g)} | {et} | {it} |")
    lines.append("")

    # 4. 關鍵物理異常溯源清單
    lines.extend(["## 4. 物理硬邊界異常標記明細（節錄）", ""])
    oob_e = cleaned_df[cleaned_df["exclusion_reason"].fillna("").str.contains("ecorr_out_of_hard_bounds")]
    oob_i = cleaned_df[cleaned_df["exclusion_reason"].fillna("").str.contains("icorr_out_of_hard_bounds")]
    lines.append(f"- 超出 Ecorr 硬邊界紀錄：**{len(oob_e)}** 筆；超出 icorr 硬邊界紀錄：**{len(oob_i)}** 筆。")
    lines.append("")
    sample_oob = pd.concat([oob_e.head(20), oob_i.sort_values("icorr_uA_cm2", ascending=False).head(15)]).drop_duplicates(subset=["record_id"])
    if not sample_oob.empty:
        lines.extend([
            "| record_id | 金屬 | 樣品 | Ecorr_V_SHE | icorr_uA_cm2 | `exclusion_reason` | 來源 |",
            "|---|---|---|---:|---:|---|---|",
        ])
        for _, r in sample_oob.iterrows():
            lines.append(
                f"| {r.get('record_id', '')} | {r.get('material_id', '')} | {r.get('sample_name', '')} | "
                f"{r.get('Ecorr_V_SHE', '')} | {r.get('icorr_uA_cm2', '')} | "
                f"`{r.get('exclusion_reason', '')}` | {r.get('source_reference', '')} |"
            )
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_clean_and_validate(
    *,
    settings: config.PipelineSettings | None = None,
    benchmarks_path: Path | None = None,
    input_csv: Path | None = None,
    output_csv: Path | None = None,
    report_path: Path | None = None,
    inplace: bool = True,
) -> tuple[pd.DataFrame, Path]:
    """執行 Step 6b 非破壞性清洗與權威基準驗證，回傳 (cleaned_df, report_path)。"""
    resolved_settings = settings or config.load_settings()
    sim_dir = Path(resolved_settings.simulator_data_dir)
    data_dir = Path(resolved_settings.data_root)

    echem_path = Path(input_csv) if input_csv is not None else (sim_dir / "electrochemical_database.csv")
    if not echem_path.exists():
        raise FileNotFoundError(f"找不到主表 {echem_path}，請先執行 Step 6 (step6_merge.py)。")

    sol_path = sim_dir / "solution_properties.csv"
    sol_df = pd.read_csv(sol_path) if sol_path.exists() else pd.DataFrame()
    echem_df = pd.read_csv(echem_path)

    benchmarks = load_reference_benchmarks(benchmarks_path)
    paper_meta_index = build_paper_metadata_index(data_dir)
    table_json_index = build_table_json_index(data_dir)

    print(
        f"[Step 6b] 載入 {len(echem_df)} 筆紀錄、"
        f"{len(paper_meta_index)} 篇文獻中繼資料與 {len(table_json_index)} 份表格索引..."
    )

    cleaned_df = clean_and_validate_dataframe(
        echem_df,
        benchmarks,
        sol_df=sol_df,
        paper_meta_index=paper_meta_index,
        table_json_index=table_json_index,
    )

    if output_csv is not None:
        target_csv = Path(output_csv)
    elif inplace:
        target_csv = echem_path
    else:
        target_csv = sim_dir / "electrochemical_database_cleaned.csv"

    target_csv.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(target_csv, index=False, encoding="utf-8-sig")

    target_report = (
        Path(report_path)
        if report_path is not None
        else (sim_dir / "cleaning_validation_report.md")
    )
    write_cleaning_validation_report(cleaned_df, benchmarks, target_report)

    q_counts = cleaned_df["quality_flag"].value_counts().to_dict()
    print(f"  ✅ 清洗標記完成（保留全部 {len(cleaned_df)} 筆）：{q_counts}")
    print(f"  💾 輸出資料表：{target_csv}")
    print(f"  📊 驗證報告：{target_report}")
    return cleaned_df, target_report


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 6b: 非破壞性電化學資料庫清洗、異常標記與權威基準驗證"
    )
    parser.add_argument("--benchmarks", type=Path, default=None, help="reference_benchmarks.json 路徑")
    parser.add_argument("--input-csv", type=Path, default=None, help="輸入 electrochemical_database.csv 路徑")
    parser.add_argument("--output-csv", type=Path, default=None, help="輸出 CSV 路徑（預設覆寫輸入表並保留所有原欄位與原資料列）")
    parser.add_argument("--report-path", type=Path, default=None, help="驗證報告輸出路徑")
    parser.add_argument(
        "--no-inplace",
        action="store_true",
        help="不覆寫原 electrochemical_database.csv，另存為 electrochemical_database_cleaned.csv",
    )
    return parser


def main() -> None:
    parser = build_cli_parser()
    args = parser.parse_args()
    run_clean_and_validate(
        benchmarks_path=args.benchmarks,
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        report_path=args.report_path,
        inplace=not args.no_inplace,
    )


if __name__ == "__main__":
    main()
