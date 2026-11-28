"""
====================================================================
  Tafel Mining Project — 執行情境與衍生設定引擎 (config.py)
  日常調整參數請編輯 `user_settings.py`（API 金鑰請設於 `.env`）
  匯入本模組不應產生建立資料夾、載入 .env 或印出橫幅等副作用。
====================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import json as _json
import os
from pathlib import Path as _Path
import re as _re
from typing import Any, Mapping, Sequence

import user_settings as _us

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
METALS_DB_PATH = _Path(PROJECT_ROOT) / "metals_db.json"

VALID_SEARCH_MODES: tuple[str, ...] = (
    "corrosion",
    "cmp_etch",
    "inhibitor",
    "solvent",
    "catalysis",
)

_OPENALEX_COMPOUND_BATCH_SIZE: int = 6
_OPENALEX_CUSTOM_QUERY_BATCH_SIZE: int = 3

# 在不分大小寫的外部搜尋引擎中容易與英文常見字混淆的元素／化合物符號（如 TiN vs tin）
_AMBIGUOUS_SYMBOL_CONTEXT: dict[str, tuple[str, ...]] = {
    "TiN": ("nitride", "coating", "thin film", "barrier", "PVD", "CVD", "ALD", "CMP", "sputtering"),
}
_QUERY_TOKEN_RE = _re.compile(r'"[^"]+"|\S+')


def build_metal_search_clause(
    target_metal: str,
    metal_symbol: str,
    *,
    syntax: str = "boolean",
) -> str:
    """
    建立目標金屬的布林搜尋子句：
    - syntax='boolean'：OpenAlex / Europe PMC 語法（AND / OR）
    - syntax='s2'：Semantic Scholar /search/bulk 語法（+ / |）
    對於易與英文單字混淆的符號（如 TiN vs tin），強制搭配材料領域限定詞。
    """
    and_op, or_op = (" + ", " | ") if syntax == "s2" else (" AND ", " OR ")
    if target_metal.lower() == metal_symbol.lower():
        return f'"{target_metal}"'

    ctx_terms = _AMBIGUOUS_SYMBOL_CONTEXT.get(metal_symbol)
    if ctx_terms:
        ctx_or = or_op.join(f'"{t}"' for t in ctx_terms)
        return f'("{target_metal}"{or_op}("{metal_symbol}"{and_op}({ctx_or})))'
    return f'("{target_metal}"{or_op}"{metal_symbol}")'


def _to_openalex_and_clause(raw_query: str) -> str:
    """將空白分隔的展開查詢（如 '"ruthenium" "CMP" "Tafel"'）轉為 OpenAlex 明確 AND 語法。"""
    tokens = _QUERY_TOKEN_RE.findall(raw_query.strip())
    if not tokens:
        return ""
    normalized: list[str] = []
    for tok in tokens:
        upper = tok.upper()
        if upper in ("AND", "OR", "NOT"):
            continue
        if tok.startswith('"') and tok.endswith('"'):
            normalized.append(tok)
        else:
            normalized.append(f'"{tok}"')
    return " AND ".join(normalized)


def _filter_compounds_for_search(
    compounds: Sequence[str],
    target_metal: str,
    metal_symbol: str,
) -> list[str]:
    """移除與 target_metal 或 metal_symbol 完全重複的裸詞（避免 TiN 等易混淆符號繞過限定詞）。"""
    skip = {target_metal.lower(), metal_symbol.lower()}
    return [c for c in compounds if c and c.strip().lower() not in skip]


def _load_dotenv(env_path: _Path) -> None:
    """從專案根目錄的 .env 載入環境變數（不覆蓋已設定的系統環境變數）。"""
    if not env_path.is_file():
        return
    with env_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if key:
                os.environ.setdefault(key, value)


def parse_search_modes(raw: str | Sequence[str] | None) -> list[str]:
    """
    將搜尋模式設定或命令列參數正規化為去重後的原子模式清單。
    合法原子模式：corrosion, cmp_etch, inhibitor, solvent, catalysis（另支援 'all' 展開全部 5 種）。
    """
    if raw is None:
        tokens: list[str] = []
    elif isinstance(raw, str):
        tokens = [t for t in _re.split(r"[\s,+]+", raw.strip()) if t]
    elif isinstance(raw, (list, tuple)):
        tokens = []
        for item in raw:
            tokens.extend(t for t in _re.split(r"[\s,+]+", str(item).strip()) if t)
    else:
        raise ValueError(f"不支援的搜尋模式型別：{type(raw).__name__}")

    expanded: list[str] = []
    for tok in tokens:
        low = tok.lower()
        if low == "all":
            expanded.extend(VALID_SEARCH_MODES)
        elif low == "corrosion_cmp":
            # 相容舊版複合模式名稱
            expanded.extend(["corrosion", "cmp_etch"])
        elif low in VALID_SEARCH_MODES:
            expanded.append(low)
        else:
            valid_str = ", ".join((*VALID_SEARCH_MODES, "all"))
            raise ValueError(f"不支援的搜尋模式 '{tok}'，可用選項：{valid_str}")

    deduped = list(dict.fromkeys(expanded))
    if not deduped:
        valid_str = ", ".join(VALID_SEARCH_MODES)
        raise ValueError(f"至少需指定一種搜尋模式（可用選項：{valid_str}）")
    return deduped


def _load_metals_db(
    db_path: _Path,
) -> tuple[dict[str, dict], dict[str, tuple[float, float]]]:
    """從 metals_db.json 載入各金屬搜尋預設與 Ecorr 合理電位區間。"""
    with db_path.open("r", encoding="utf-8") as f:
        raw = _json.load(f)

    default_range = raw.get("default_ecorr_valid_range_v_she", [-2.0, 2.0])
    presets: dict[str, dict] = {}
    ecorr_ranges: dict[str, tuple[float, float]] = {}

    for sym, entry in raw["metals"].items():
        if "target_metal" not in entry or "compounds" not in entry:
            raise ValueError(
                f"metals_db.json 中的金屬 '{sym}' 缺少 target_metal 或 compounds"
            )
        presets[sym] = {
            "target_metal": entry["target_metal"],
            "compounds": list(entry["compounds"]),
            "cmp_queries": list(entry.get("cmp_queries", [])),
            "inhibitor_queries": list(entry.get("inhibitor_queries", [])),
        }
        if "ecorr_valid_range_v_she" in entry:
            lo, hi = entry["ecorr_valid_range_v_she"]
            ecorr_ranges[sym] = (float(lo), float(hi))

    ecorr_ranges["default"] = (float(default_range[0]), float(default_range[1]))
    return presets, ecorr_ranges


METAL_PRESETS, ECORR_VALID_RANGE_V_SHE = _load_metals_db(METALS_DB_PATH)


def resolve_metal_symbol(raw: str) -> str:
    """大小寫不敏感地解析金屬符號（例如 'tin' -> 'TiN'）。"""
    lookup = {k.lower(): k for k in METAL_PRESETS}
    key = lookup.get(raw.strip().lower())
    if not key:
        valid = ", ".join(METAL_PRESETS.keys())
        raise ValueError(f"不支援的金屬 '{raw}'，可用清單：{valid}")
    return key


def _chunk_list(items: Sequence[str], size: int) -> list[list[str]]:
    item_list = list(items)
    return [item_list[i : i + size] for i in range(0, len(item_list), size)]


def build_expanded_openalex_queries(
    modes: str | Sequence[str],
    *,
    target_metal: str,
    metal_symbol: str,
    compounds: Sequence[str] = (),
    cmp_queries: Sequence[str] = (),
    inhibitor_queries: Sequence[str] = (),
) -> list[str]:
    """依選取的搜尋模式展開完整 OpenAlex 關鍵字查詢清單（並去重）。"""
    active_modes = parse_search_modes(modes)
    mode_map: dict[str, list[str]] = {
        "corrosion": [
            f'"{target_metal}" "potentiodynamic polarization"',
            f'"{target_metal}" "corrosion current density"',
            f'"{target_metal}" "Tafel extrapolation" corrosion',
            f'"{target_metal}" "Tafel plot" corrosion',
            f'"{target_metal}" "Tafel slope" corrosion',
            f'"{target_metal}" "electrochemical corrosion"',
            f'"{target_metal}" "polarization curve" corrosion',
            f'"{target_metal}" "polarization resistance" corrosion',
            f'"{target_metal}" "anodic polarization" corrosion',
            f'"{target_metal}" "cathodic polarization" corrosion',
            f'"{target_metal}" "Ecorr" "icorr"',
            f'"{metal_symbol}" "potentiodynamic polarization" corrosion',
            f'"{metal_symbol}" "corrosion current density" "Tafel"',
            *[f'"{comp}" "potentiodynamic polarization"' for comp in compounds],
            *[f'"{comp}" "corrosion current density"' for comp in compounds],
        ],
        "cmp_etch": [
            f'"{target_metal}" "chemical mechanical" "potentiodynamic"',
            f'"{target_metal}" "CMP" "Tafel"',
            f'"{target_metal}" "CMP" "corrosion current"',
            f'"{target_metal}" "slurry" "potentiodynamic polarization"',
            f'"{target_metal}" "wet etching" "potentiodynamic"',
            f'"{target_metal}" "H2O2" "potentiodynamic polarization"',
            f'"{target_metal}" "hydrogen peroxide" "Tafel" corrosion',
            f'"{target_metal}" "galvanic corrosion" "CMP"',
            f'"{metal_symbol}" "CMP" "potentiodynamic polarization"',
            f'"{metal_symbol}" "wet etching" "Tafel" corrosion',
            *cmp_queries,
        ],
        "inhibitor": [
            f'"{target_metal}" "corrosion inhibitor" "potentiodynamic polarization"',
            f'"{target_metal}" "corrosion inhibition" "Tafel"',
            f'"{target_metal}" "inhibition efficiency" "corrosion current"',
            f'"{target_metal}" "inhibition efficiency" "Ecorr" "icorr"',
            f'"{target_metal}" "organic inhibitor" "potentiodynamic"',
            f'"{target_metal}" "mixed-type inhibitor" "Tafel"',
            f'"{metal_symbol}" "corrosion inhibitor" "potentiodynamic"',
            f'"{metal_symbol}" "inhibition efficiency" "Tafel"',
            f'"{target_metal}" "triazole" "potentiodynamic polarization"',
            f'"{target_metal}" "imidazole" "potentiodynamic polarization"',
            f'"{target_metal}" "tetrazole" "corrosion" "polarization"',
            f'"{target_metal}" "thiazole" "corrosion" "Tafel"',
            f'"{target_metal}" "Schiff base" "potentiodynamic polarization"',
            f'"{target_metal}" "amino acid" "corrosion inhibitor" "polarization"',
            f'"{target_metal}" "cysteine" "corrosion" "potentiodynamic"',
            f'"{target_metal}" "thiourea" "corrosion" "polarization"',
            f'"{target_metal}" "surfactant" "corrosion inhibitor" "Tafel"',
            f'"{target_metal}" "ionic liquid" "corrosion inhibitor" "polarization"',
            f'"{target_metal}" "CMP" "corrosion inhibitor" "potentiodynamic"',
            f'"{target_metal}" "slurry" "inhibitor" "potentiodynamic"',
            *inhibitor_queries,
        ],
        "solvent": [
            f'"{target_metal}" "non-aqueous" "potentiodynamic polarization"',
            f'"{target_metal}" "nonaqueous" "corrosion" "Tafel"',
            f'"{target_metal}" "organic solvent" "potentiodynamic polarization"',
            f'"{target_metal}" "organic media" "corrosion" "polarization"',
            f'"{target_metal}" "anhydrous" "corrosion" "potentiodynamic"',
            f'"{target_metal}" "semi-aqueous" "corrosion" "polarization"',
            f'"{target_metal}" "mixed solvent" "potentiodynamic polarization"',
            f'"{target_metal}" "water-organic" "corrosion" "polarization"',
            f'"{target_metal}" "trace water" "corrosion" "polarization"',
            f'"{target_metal}" "protic solvent" "corrosion" "polarization"',
            f'"{target_metal}" "aprotic solvent" "corrosion" "polarization"',
            f'"{metal_symbol}" "non-aqueous" "potentiodynamic polarization"',
            f'"{metal_symbol}" "organic solvent" "corrosion" "Tafel"',
            f'"{target_metal}" "aqueous-organic" "corrosion" "polarization"',
            f'"{target_metal}" "co-solvent" "corrosion" "potentiodynamic"',
            f'"{target_metal}" "methanol" "potentiodynamic polarization"',
            f'"{target_metal}" "ethanol" "potentiodynamic polarization"',
            f'"{target_metal}" "ethylene glycol" "potentiodynamic polarization"',
            f'"{target_metal}" "DMSO" "corrosion" "polarization"',
            f'"{target_metal}" "DMF" "corrosion" "polarization"',
            f'"{target_metal}" "acetonitrile" "corrosion" "polarization"',
            f'"{target_metal}" "NMP" "corrosion" "polarization"',
            f'"{target_metal}" "deep eutectic solvent" "potentiodynamic polarization"',
            f'"{target_metal}" "ionic liquid" "potentiodynamic polarization" corrosion',
        ],
        "catalysis": [
            f'"{target_metal}" "Tafel slope"',
            f'"{target_metal}" "Tafel plot"',
            f'"{target_metal}" "polarization curve"',
            f'"{target_metal}" "exchange current density"',
            f'"{target_metal}" "linear sweep voltammetry" "Tafel"',
            f'"{target_metal}" "cyclic voltammetry" corrosion',
            f'"{target_metal}" "hydrogen evolution" "Tafel"',
            f'"{target_metal}" "oxygen evolution" "Tafel"',
            f'"{target_metal}" "oxygen reduction" "Tafel"',
            f'"{target_metal}" "CO2 reduction" "Tafel"',
            f'"{metal_symbol}" "Tafel slope" electrochemistry',
            f'"{metal_symbol}" "Tafel slope" electrocatalyst',
            f'"{metal_symbol}" "Tafel slope" catalyst',
            *[f'"{comp}" "Tafel slope"' for comp in compounds],
            *[f'"{comp}" "Tafel"' for comp in compounds],
        ],
    }
    return list(dict.fromkeys(q for m in active_modes for q in mode_map[m]))


def build_grouped_openalex_queries(
    modes: str | Sequence[str],
    *,
    target_metal: str,
    metal_symbol: str,
    compounds: Sequence[str] = (),
    cmp_queries: Sequence[str] = (),
    inhibitor_queries: Sequence[str] = (),
) -> list[str]:
    """
    依選取的搜尋模式組裝高密度 OpenAlex 布林查詢（AND / OR），
    將原本每金屬 ~110-126 組展開查詢濃縮為 ~12-18 組，大幅節省每日配額，
    並確保：
    - 通用極化/Tafel 詞強制搭配 "corrosion"（避免 Ru/Mo 被大量 HER/OER 電催化論文稀釋）
    - 易混淆符號（如 TiN vs tin）加上領域限定詞
    - cmp_queries / inhibitor_queries 轉為明確 AND 語法（避免 OpenAlex 解析為 0 筆）
    """
    active_modes = parse_search_modes(modes)
    metal_clause = build_metal_search_clause(target_metal, metal_symbol, syntax="boolean")
    filtered_compounds = _filter_compounds_for_search(compounds, target_metal, metal_symbol)
    queries: list[str] = []

    if "corrosion" in active_modes:
        queries.append(
            f'{metal_clause} AND ("potentiodynamic polarization" OR "corrosion current density" '
            f'OR "electrochemical corrosion" OR ("Ecorr" AND "icorr"))'
        )
        queries.append(
            f'{metal_clause} AND "corrosion" AND ("Tafel extrapolation" OR "Tafel plot" '
            f'OR "Tafel slope" OR "polarization curve" OR "polarization resistance" '
            f'OR "anodic polarization" OR "cathodic polarization")'
        )
        for batch in _chunk_list(filtered_compounds, _OPENALEX_COMPOUND_BATCH_SIZE):
            comp_or = " OR ".join(f'"{c}"' for c in batch)
            queries.append(
                f'({comp_or}) AND ("potentiodynamic polarization" OR "corrosion current density" '
                f'OR "electrochemical corrosion")'
            )

    if "cmp_etch" in active_modes:
        queries.append(
            f'{metal_clause} AND ("chemical mechanical" OR "CMP" OR "slurry" OR "wet etching" '
            f'OR "H2O2" OR "hydrogen peroxide" OR "galvanic corrosion") '
            f'AND ("potentiodynamic" OR "potentiodynamic polarization" OR "Tafel" OR "corrosion current")'
        )
        and_cmp = [c for c in (_to_openalex_and_clause(q) for q in cmp_queries) if c]
        for batch in _chunk_list(and_cmp, _OPENALEX_CUSTOM_QUERY_BATCH_SIZE):
            queries.append(" OR ".join(f"({q})" for q in batch))

    if "inhibitor" in active_modes:
        queries.append(
            f'{metal_clause} AND ("corrosion inhibitor" OR "corrosion inhibition" '
            f'OR "inhibition efficiency" OR "organic inhibitor" OR "mixed-type inhibitor") '
            f'AND ("potentiodynamic" OR "potentiodynamic polarization" OR "Tafel" '
            f'OR "corrosion current" OR ("Ecorr" AND "icorr"))'
        )
        queries.append(
            f'{metal_clause} AND ("triazole" OR "imidazole" OR "tetrazole" OR "thiazole" '
            f'OR "Schiff base" OR "amino acid" OR "cysteine" OR "thiourea" '
            f'OR "surfactant" OR "ionic liquid") '
            f'AND ("corrosion" OR "corrosion inhibitor") AND ("potentiodynamic" OR "polarization" OR "Tafel")'
        )
        and_inh = [c for c in (_to_openalex_and_clause(q) for q in inhibitor_queries) if c]
        for batch in _chunk_list(and_inh, _OPENALEX_CUSTOM_QUERY_BATCH_SIZE):
            queries.append(" OR ".join(f"({q})" for q in batch))

    if "solvent" in active_modes:
        queries.append(
            f'{metal_clause} AND ("non-aqueous" OR "nonaqueous" OR "organic solvent" OR "organic media" '
            f'OR "anhydrous" OR "semi-aqueous" OR "mixed solvent" OR "water-organic" '
            f'OR "aqueous-organic" OR "trace water" OR "protic solvent" OR "aprotic solvent" OR "co-solvent") '
            f'AND ("potentiodynamic" OR "potentiodynamic polarization" OR "Tafel" OR "polarization")'
        )
        queries.append(
            f'{metal_clause} AND ("methanol" OR "ethanol" OR "ethylene glycol" OR "DMSO" '
            f'OR "DMF" OR "acetonitrile" OR "NMP" OR "deep eutectic solvent" OR "ionic liquid") '
            f'AND "corrosion" AND ("potentiodynamic polarization" OR "polarization")'
        )

    if "catalysis" in active_modes:
        queries.append(
            f'{metal_clause} AND ("Tafel slope" OR "Tafel plot" OR "exchange current density" '
            f'OR "linear sweep voltammetry" OR "polarization curve") '
            f'AND ("hydrogen evolution" OR "oxygen evolution" OR "oxygen reduction" '
            f'OR "CO2 reduction" OR "electrocatalyst" OR "catalyst")'
        )
        for batch in _chunk_list(filtered_compounds, _OPENALEX_COMPOUND_BATCH_SIZE):
            comp_or = " OR ".join(f'"{c}"' for c in batch)
            queries.append(f'({comp_or}) AND ("Tafel slope" OR "Tafel")')

    return list(dict.fromkeys(queries))


def build_search_queries(
    modes: str | Sequence[str],
    *,
    target_metal: str,
    metal_symbol: str,
    compounds: Sequence[str] = (),
    cmp_queries: Sequence[str] = (),
    inhibitor_queries: Sequence[str] = (),
    grouped: bool = True,
) -> list[str]:
    """依指定的搜尋模式清單與金屬參數組裝並去重 OpenAlex 查詢清單。"""
    builder = build_grouped_openalex_queries if grouped else build_expanded_openalex_queries
    return builder(
        modes,
        target_metal=target_metal,
        metal_symbol=metal_symbol,
        compounds=compounds,
        cmp_queries=cmp_queries,
        inhibitor_queries=inhibitor_queries,
    )


def build_target_regex(
    target_metal: str,
    symbol: str,
    compounds: Sequence[str] = (),
) -> _re.Pattern:
    """
    建立「目標金屬」的 regex（Step 3 篩選與 Step 5 共用）。
    - 英文全名 / 多字詞化合物：不分大小寫 + 單字邊界
    - 元素符號 / 化學式：區分大小寫，前面不可接字母、後面不可接小寫字母
    """
    parts = [
        r"(?i:\b" + _re.escape(target_metal) + r"\b)",
        r"(?<![A-Za-z])" + _re.escape(symbol) + r"(?![a-z])",
    ]
    for comp in compounds:
        if " " in comp or comp.islower():
            parts.append(r"(?i:\b" + _re.escape(comp) + r"\b)")
        else:
            parts.append(r"(?<![A-Za-z])" + _re.escape(comp) + r"(?![a-z])")
    return _re.compile("|".join(parts))


@dataclass(frozen=True)
class PipelineSettings:
    """跨金屬共用預設、資料源開關、物理驗證門檻與合併輸出目錄。"""

    project_root: _Path
    data_root: _Path
    simulator_data_dir: _Path
    mp_cache_path: _Path
    mailto: str
    skip_existing_step1: bool
    skip_existing_step2: bool
    step1_inflight_dedup: bool
    step1_max_scan_multiplier: int
    step1_early_stop_empty_pages: int
    max_records_per_query: int
    step2_skip_unpaywall_if_has_pdf: bool
    step2_unpaywall_delay: float
    step2_checkpoint_every: int
    download_delay: float
    step3_max_workers: int
    step3_domain_403_threshold: int
    step3_max_pdf_mb: int
    request_timeout: int
    from_year: int
    default_search_modes: tuple[str, ...]
    enable_openalex: bool
    openalex_use_grouped_queries: bool
    enable_semantic_scholar: bool
    semantic_scholar_use_bulk: bool
    semantic_scholar_max_per_query: int
    semantic_scholar_max_queries_no_key: int
    semantic_scholar_delay: float
    enable_europepmc_xml: bool
    europepmc_max_papers: int
    europepmc_delay: float
    enable_mp_api: bool
    metals_to_run: tuple[str, ...]
    default_metal: str
    tafel_min: float
    tafel_max: float
    context_window: int
    filter_outliers: bool
    icorr_valid_range_ua_cm2: tuple[float, float]
    keep_only_target_metal: bool
    require_target_in_title_abstract: bool
    min_material_mentions: int
    openalex_api_key: str
    semantic_scholar_api_key: str
    mp_api_key: str
    metal_presets: dict[str, dict]
    ecorr_valid_range_v_she: dict[str, tuple[float, float]]

    def ensure_merged_dir(self) -> None:
        os.makedirs(self.simulator_data_dir, exist_ok=True)


@dataclass(frozen=True)
class MetalRunContext:
    """單一金屬在指定搜尋模式下的不可變執行情境與產物路徑。"""

    settings: PipelineSettings
    metal_symbol: str
    target_metal: str
    compounds: tuple[str, ...]
    cmp_queries: tuple[str, ...]
    inhibitor_queries: tuple[str, ...]
    search_modes: tuple[str, ...]
    search_queries: tuple[str, ...]
    enable_openalex: bool
    target_regex: _re.Pattern
    data_dir: str
    metadata_dir: str
    pdf_dir: str
    text_dir: str
    extracted_dir: str
    step1_output: str
    s2_cache_path: str
    step1_cursor_state_path: str
    step2_output: str
    download_state_path: str
    step5_output: str
    step5_material_output: str
    step5_solution_output: str
    step5_echem_output: str
    step5_raw_output: str
    step5_outliers_output: str

    def ensure_dirs(self) -> None:
        for d in (self.metadata_dir, self.pdf_dir, self.text_dir, self.extracted_dir):
            os.makedirs(d, exist_ok=True)

    def format_summary(self) -> str:
        s = self.settings
        s2_status = (
            "ON (API Key)"
            if (s.enable_semantic_scholar and s.semantic_scholar_api_key)
            else ("ON (Public)" if s.enable_semantic_scholar else "OFF")
        )
        mp_status = (
            "ON (API Key)"
            if (s.enable_mp_api and s.mp_api_key)
            else ("ON (Offline)" if s.enable_mp_api else "OFF")
        )
        oa_mode = "Grouped" if s.openalex_use_grouped_queries else "Expanded"
        oa_status = (
            f"ON ({oa_mode}, {'API Key' if s.openalex_api_key else 'Public'})"
            if self.enable_openalex
            else "OFF"
        )
        lines = [
            "=" * 58,
            f"  Target Metal : {self.target_metal} ({self.metal_symbol})",
            (
                f"  Search Modes : {', '.join(self.search_modes)}  |  "
                f"Europe PMC: {'ON (' + str(s.europepmc_max_papers) + ')' if s.enable_europepmc_xml else 'OFF'}"
            ),
            f"  OpenAlex     : {oa_status}",
            f"  S2 / MP API  : Semantic Scholar {s2_status}  |  Materials Project {mp_status}",
            (
                f"  Compounds    : {len(self.compounds)} types  |  "
                f"CMP Qs: {len(self.cmp_queries)}  |  Inhibitor Qs: {len(self.inhibitor_queries)}"
            ),
            f"  Queries      : {len(self.search_queries)} combinations",
            f"  Max/query    : {s.max_records_per_query} records",
            f"  From year    : {s.from_year}",
            f"  Data folder  : {self.data_dir}",
            f"  Step1 output : {self.step1_output}",
            f"  Step2 output : {self.step2_output}",
            f"  Step5 output : {self.step5_echem_output}",
            "=" * 58,
        ]
        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.format_summary())


_SETTINGS_FIELD_NAMES = frozenset(f.name for f in fields(PipelineSettings))


def _resolve_simulator_dir(
    project_root: _Path,
    data_root: _Path,
    explicit_sim_dir: str | _Path | None,
    has_explicit_data_root: bool,
) -> _Path:
    if explicit_sim_dir is not None:
        raw = str(explicit_sim_dir).strip()
        expanded = _Path(os.path.expanduser(raw))
        return expanded if expanded.is_absolute() else (project_root / expanded).resolve()
    if has_explicit_data_root:
        return data_root / "merged"
    if _us.SIMULATOR_DATA_DIR:
        raw = str(_us.SIMULATOR_DATA_DIR).strip()
        expanded = _Path(os.path.expanduser(raw))
        return expanded if expanded.is_absolute() else (project_root / expanded).resolve()
    return data_root / "merged"


def load_settings(
    *,
    data_root: str | _Path | None = None,
    simulator_data_dir: str | _Path | None = None,
    env: Mapping[str, str] | None = None,
    **overrides: Any,
) -> PipelineSettings:
    """從 user_settings.py 與環境變數建立不可變 PipelineSettings（無磁碟或列印副作用）。"""
    unknown = set(overrides) - _SETTINGS_FIELD_NAMES
    if unknown:
        raise TypeError(f"Unknown PipelineSettings override(s): {', '.join(sorted(unknown))}")

    project_root = _Path(overrides.pop("project_root", PROJECT_ROOT)).resolve()
    if env is None:
        _load_dotenv(project_root / ".env")
        env_map: Mapping[str, str] = os.environ
    else:
        env_map = env

    has_explicit_data_root = data_root is not None
    resolved_data_root = (
        _Path(data_root).resolve() if data_root is not None else (project_root / "data")
    )
    resolved_sim_dir = _resolve_simulator_dir(
        project_root,
        resolved_data_root,
        simulator_data_dir,
        has_explicit_data_root=has_explicit_data_root,
    )
    resolved_mp_cache = _Path(overrides.pop("mp_cache_path", resolved_sim_dir / "mp_cache.json"))

    raw_modes = overrides.pop("default_search_modes", _us.SEARCH_MODES)
    default_search_modes = tuple(parse_search_modes(raw_modes))

    raw_metals = overrides.pop("metals_to_run", _us.METALS_TO_RUN)
    metals_to_run = tuple(raw_metals)

    raw_icorr = overrides.pop("icorr_valid_range_ua_cm2", _us.ICORR_VALID_RANGE_UA_CM2)
    icorr_range = (float(raw_icorr[0]), float(raw_icorr[1]))

    values: dict[str, Any] = {
        "project_root": project_root,
        "data_root": resolved_data_root,
        "simulator_data_dir": resolved_sim_dir,
        "mp_cache_path": resolved_mp_cache,
        "mailto": _us.MAILTO,
        "skip_existing_step1": _us.SKIP_EXISTING_STEP1,
        "skip_existing_step2": _us.SKIP_EXISTING_STEP2,
        "step1_inflight_dedup": _us.STEP1_INFLIGHT_DEDUP,
        "step1_max_scan_multiplier": _us.STEP1_MAX_SCAN_MULTIPLIER,
        "step1_early_stop_empty_pages": _us.STEP1_EARLY_STOP_EMPTY_PAGES,
        "max_records_per_query": _us.MAX_RECORDS_PER_QUERY,
        "step2_skip_unpaywall_if_has_pdf": getattr(_us, "STEP2_SKIP_UNPAYWALL_IF_HAS_PDF", True),
        "step2_unpaywall_delay": getattr(_us, "STEP2_UNPAYWALL_DELAY", 0.2),
        "step2_checkpoint_every": getattr(_us, "STEP2_CHECKPOINT_EVERY", 200),
        "download_delay": _us.DOWNLOAD_DELAY,
        "step3_max_workers": getattr(_us, "STEP3_MAX_WORKERS", 4),
        "step3_domain_403_threshold": getattr(_us, "STEP3_DOMAIN_403_THRESHOLD", 15),
        "step3_max_pdf_mb": getattr(_us, "STEP3_MAX_PDF_MB", 50),
        "request_timeout": _us.REQUEST_TIMEOUT,
        "from_year": _us.FROM_YEAR,
        "default_search_modes": default_search_modes,
        "enable_openalex": getattr(_us, "ENABLE_OPENALEX", True),
        "openalex_use_grouped_queries": getattr(_us, "OPENALEX_USE_GROUPED_QUERIES", True),
        "enable_semantic_scholar": _us.ENABLE_SEMANTIC_SCHOLAR,
        "semantic_scholar_use_bulk": _us.SEMANTIC_SCHOLAR_USE_BULK,
        "semantic_scholar_max_per_query": _us.SEMANTIC_SCHOLAR_MAX_PER_QUERY,
        "semantic_scholar_max_queries_no_key": _us.SEMANTIC_SCHOLAR_MAX_QUERIES_NO_KEY,
        "semantic_scholar_delay": _us.SEMANTIC_SCHOLAR_DELAY,
        "enable_europepmc_xml": _us.ENABLE_EUROPEPMC_XML,
        "europepmc_max_papers": _us.EUROPEPMC_MAX_PAPERS,
        "europepmc_delay": _us.EUROPEPMC_DELAY,
        "enable_mp_api": _us.ENABLE_MP_API,
        "metals_to_run": metals_to_run,
        "default_metal": _us.DEFAULT_METAL,
        "tafel_min": _us.TAFEL_MIN,
        "tafel_max": _us.TAFEL_MAX,
        "context_window": _us.CONTEXT_WINDOW,
        "filter_outliers": _us.FILTER_OUTLIERS,
        "icorr_valid_range_ua_cm2": icorr_range,
        "keep_only_target_metal": _us.KEEP_ONLY_TARGET_METAL,
        "require_target_in_title_abstract": _us.REQUIRE_TARGET_IN_TITLE_ABSTRACT,
        "min_material_mentions": _us.MIN_MATERIAL_MENTIONS,
        "openalex_api_key": env_map.get("OPENALEX_API_KEY", ""),
        "semantic_scholar_api_key": env_map.get("SEMANTIC_SCHOLAR_API_KEY", ""),
        "mp_api_key": env_map.get("MP_API_KEY", ""),
        "metal_presets": METAL_PRESETS,
        "ecorr_valid_range_v_she": ECORR_VALID_RANGE_V_SHE,
    }
    values.update(overrides)
    return PipelineSettings(**values)


def build_metal_context(
    metal: str | None = None,
    *,
    search_modes: str | Sequence[str] | None = None,
    settings: PipelineSettings | None = None,
    search_queries: Sequence[str] | None = None,
    skip_openalex: bool = False,
    grouped_queries: bool | None = None,
) -> MetalRunContext:
    """建立單一金屬的執行情境（純記憶體計算，不建立資料夾也不印出訊息）。"""
    resolved_settings = settings or load_settings()
    metal_symbol = resolve_metal_symbol(metal or resolved_settings.default_metal)
    preset = resolved_settings.metal_presets[metal_symbol]

    target_metal = preset["target_metal"]
    compounds = tuple(preset["compounds"])
    cmp_queries = tuple(preset.get("cmp_queries", ()))
    inhibitor_queries = tuple(preset.get("inhibitor_queries", ()))

    modes = tuple(
        resolved_settings.default_search_modes
        if search_modes is None
        else parse_search_modes(search_modes)
    )
    use_grouped = (
        resolved_settings.openalex_use_grouped_queries
        if grouped_queries is None
        else grouped_queries
    )
    if search_queries is not None:
        queries = tuple(search_queries)
    else:
        queries = tuple(
            build_search_queries(
                modes,
                target_metal=target_metal,
                metal_symbol=metal_symbol,
                compounds=compounds,
                cmp_queries=cmp_queries,
                inhibitor_queries=inhibitor_queries,
                grouped=use_grouped,
            )
        )

    data_dir = str(resolved_settings.data_root / metal_symbol)
    metadata_dir = os.path.join(data_dir, "metadata")
    pdf_dir = os.path.join(data_dir, "pdfs")
    text_dir = os.path.join(data_dir, "texts")
    extracted_dir = os.path.join(data_dir, "extracted")
    prefix = metal_symbol.lower()

    return MetalRunContext(
        settings=resolved_settings,
        metal_symbol=metal_symbol,
        target_metal=target_metal,
        compounds=compounds,
        cmp_queries=cmp_queries,
        inhibitor_queries=inhibitor_queries,
        search_modes=modes,
        search_queries=queries,
        enable_openalex=resolved_settings.enable_openalex and (not skip_openalex),
        target_regex=build_target_regex(target_metal, metal_symbol, compounds),
        data_dir=data_dir,
        metadata_dir=metadata_dir,
        pdf_dir=pdf_dir,
        text_dir=text_dir,
        extracted_dir=extracted_dir,
        step1_output=os.path.join(metadata_dir, f"{prefix}_tafel_metadata.csv"),
        s2_cache_path=os.path.join(metadata_dir, f"{prefix}_s2_cache.json"),
        step1_cursor_state_path=os.path.join(metadata_dir, f"{prefix}_step1_cursor_state.json"),
        step2_output=os.path.join(metadata_dir, f"{prefix}_tafel_with_pdf.csv"),
        download_state_path=os.path.join(metadata_dir, f"{prefix}_download_state.json"),
        step5_output=os.path.join(metadata_dir, f"{prefix}_echem_extracted.csv"),
        step5_material_output=os.path.join(extracted_dir, f"{prefix}_material_properties.csv"),
        step5_solution_output=os.path.join(extracted_dir, f"{prefix}_solution_properties.csv"),
        step5_echem_output=os.path.join(extracted_dir, f"{prefix}_electrochemical_database.csv"),
        step5_raw_output=os.path.join(extracted_dir, f"{prefix}_raw_records.csv"),
        step5_outliers_output=os.path.join(extracted_dir, f"{prefix}_filtered_outliers.csv"),
    )


def default_metal_context(*, env: Mapping[str, str] | None = None) -> MetalRunContext:
    """供獨立執行 step1~step5 腳本時，依執行期環境變數建立預設 MetalRunContext。"""
    settings = load_settings(env=env)
    env_map = os.environ if env is None else env
    raw_metal = env_map.get("TAFEL_METAL") or settings.default_metal
    raw_modes = env_map.get("TAFEL_SEARCH_MODES") or env_map.get("TAFEL_SEARCH_MODE")
    skip_oa = env_map.get("TAFEL_SKIP_OPENALEX") == "1"
    return build_metal_context(
        raw_metal,
        search_modes=raw_modes if raw_modes else None,
        settings=settings,
        skip_openalex=skip_oa,
    )
