"""
====================================================================
  Tafel Mining Project — 內部設定載入與衍生引擎 (config.py)
  ⚠️ 日常調整參數請編輯 `user_settings.py`（API 金鑰請設於 `.env`）
====================================================================
"""

import json as _json
import os
import re as _re
from pathlib import Path as _Path

import user_settings as _us

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


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


_load_dotenv(_Path(PROJECT_ROOT) / ".env")

# ============================================================
#  1. 匯入使用者設定與環境變數 API 金鑰
# ============================================================

MAILTO = _us.MAILTO
SKIP_EXISTING_STEP1 = _us.SKIP_EXISTING_STEP1
SKIP_EXISTING_STEP2 = _us.SKIP_EXISTING_STEP2
MAX_RECORDS_PER_QUERY = _us.MAX_RECORDS_PER_QUERY
DOWNLOAD_DELAY = _us.DOWNLOAD_DELAY
REQUEST_TIMEOUT = _us.REQUEST_TIMEOUT
FROM_YEAR = _us.FROM_YEAR

ENABLE_SEMANTIC_SCHOLAR = _us.ENABLE_SEMANTIC_SCHOLAR
SEMANTIC_SCHOLAR_USE_BULK = _us.SEMANTIC_SCHOLAR_USE_BULK
SEMANTIC_SCHOLAR_MAX_PER_QUERY = _us.SEMANTIC_SCHOLAR_MAX_PER_QUERY
SEMANTIC_SCHOLAR_MAX_QUERIES_NO_KEY = _us.SEMANTIC_SCHOLAR_MAX_QUERIES_NO_KEY
SEMANTIC_SCHOLAR_DELAY = _us.SEMANTIC_SCHOLAR_DELAY

ENABLE_EUROPEPMC_XML = _us.ENABLE_EUROPEPMC_XML
EUROPEPMC_MAX_PAPERS = _us.EUROPEPMC_MAX_PAPERS
EUROPEPMC_DELAY = _us.EUROPEPMC_DELAY

ENABLE_MP_API = _us.ENABLE_MP_API

METALS_TO_RUN: list[str] = list(_us.METALS_TO_RUN)
DEFAULT_METAL: str = _us.DEFAULT_METAL

TAFEL_MIN = _us.TAFEL_MIN
TAFEL_MAX = _us.TAFEL_MAX
CONTEXT_WINDOW = _us.CONTEXT_WINDOW
FILTER_OUTLIERS = _us.FILTER_OUTLIERS
ICORR_VALID_RANGE_UA_CM2: tuple[float, float] = tuple(_us.ICORR_VALID_RANGE_UA_CM2)
KEEP_ONLY_TARGET_METAL = _us.KEEP_ONLY_TARGET_METAL
REQUIRE_TARGET_IN_TITLE_ABSTRACT = _us.REQUIRE_TARGET_IN_TITLE_ABSTRACT
MIN_MATERIAL_MENTIONS = _us.MIN_MATERIAL_MENTIONS

OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
MP_API_KEY = os.environ.get("MP_API_KEY", "")

# ============================================================
#  2. 模組化搜尋模式解析（支援單選、多選清單、空白/逗號字串與 all）
# ============================================================

VALID_SEARCH_MODES: tuple[str, ...] = (
    "corrosion",
    "cmp_etch",
    "inhibitor",
    "catalysis",
)


def parse_search_modes(raw: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """
    將搜尋模式設定或命令列參數正規化為去重後的原子模式清單。
    合法原子模式：corrosion, cmp_etch, inhibitor, catalysis（另支援 'all' 展開全部 4 種）。
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


_ENV_SEARCH_MODES = os.environ.get("TAFEL_SEARCH_MODES") or os.environ.get("TAFEL_SEARCH_MODE")
SEARCH_MODES: list[str] = parse_search_modes(
    _ENV_SEARCH_MODES if _ENV_SEARCH_MODES else _us.SEARCH_MODES
)
SEARCH_MODE: str = "+".join(SEARCH_MODES)

# ============================================================
#  3. 載入金屬資料庫 (metals_db.json)
# ============================================================

METALS_DB_PATH = _Path(PROJECT_ROOT) / "metals_db.json"


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


METAL_SYMBOL = resolve_metal_symbol(os.environ.get("TAFEL_METAL", DEFAULT_METAL))
_ACTIVE_PRESET = METAL_PRESETS[METAL_SYMBOL]
TARGET_METAL = _ACTIVE_PRESET["target_metal"]
COMPOUNDS = list(_ACTIVE_PRESET["compounds"])
CMP_QUERIES = list(_ACTIVE_PRESET.get("cmp_queries", []))
INHIBITOR_QUERIES = list(_ACTIVE_PRESET.get("inhibitor_queries", []))

# ============================================================
#  4. 依選取搜尋模式自動生成關鍵字清單
# ============================================================

_CORROSION_QUERIES = [
    # --- 金屬腐蝕 / 極化曲線核心 ---
    f'"{TARGET_METAL}" "potentiodynamic polarization"',
    f'"{TARGET_METAL}" "corrosion current density"',
    f'"{TARGET_METAL}" "Tafel extrapolation" corrosion',
    f'"{TARGET_METAL}" "Tafel plot" corrosion',
    f'"{TARGET_METAL}" "Tafel slope" corrosion',
    f'"{TARGET_METAL}" "electrochemical corrosion"',
    f'"{TARGET_METAL}" "polarization curve" corrosion',
    f'"{TARGET_METAL}" "polarization resistance" corrosion',
    f'"{TARGET_METAL}" "anodic polarization" corrosion',
    f'"{TARGET_METAL}" "cathodic polarization" corrosion',
    f'"{TARGET_METAL}" "Ecorr" "icorr"',
    f'"{METAL_SYMBOL}" "potentiodynamic polarization" corrosion',
    f'"{METAL_SYMBOL}" "corrosion current density" "Tafel"',

    # --- 化合物 / 合金腐蝕（自動展開）---
    *[f'"{comp}" "potentiodynamic polarization"' for comp in COMPOUNDS],
    *[f'"{comp}" "corrosion current density"' for comp in COMPOUNDS],
]

_CMP_ETCH_QUERIES = [
    # --- 半導體 CMP / 濕蝕刻 / 氧化劑環境通用查詢 ---
    f'"{TARGET_METAL}" "chemical mechanical" "potentiodynamic"',
    f'"{TARGET_METAL}" "CMP" "Tafel"',
    f'"{TARGET_METAL}" "CMP" "corrosion current"',
    f'"{TARGET_METAL}" "slurry" "potentiodynamic polarization"',
    f'"{TARGET_METAL}" "wet etching" "potentiodynamic"',
    f'"{TARGET_METAL}" "H2O2" "potentiodynamic polarization"',
    f'"{TARGET_METAL}" "hydrogen peroxide" "Tafel" corrosion',
    f'"{TARGET_METAL}" "galvanic corrosion" "CMP"',
    f'"{METAL_SYMBOL}" "CMP" "potentiodynamic polarization"',
    f'"{METAL_SYMBOL}" "wet etching" "Tafel" corrosion',
    # --- 該金屬專屬 CMP / 濕蝕刻 / 氧化劑查詢 ---
    *CMP_QUERIES,
]

_INHIBITOR_QUERIES = [
    # --- 通用緩蝕劑 × 定量極化核心 ---
    f'"{TARGET_METAL}" "corrosion inhibitor" "potentiodynamic polarization"',
    f'"{TARGET_METAL}" "corrosion inhibition" "Tafel"',
    f'"{TARGET_METAL}" "inhibition efficiency" "corrosion current"',
    f'"{TARGET_METAL}" "inhibition efficiency" "Ecorr" "icorr"',
    f'"{TARGET_METAL}" "organic inhibitor" "potentiodynamic"',
    f'"{TARGET_METAL}" "mixed-type inhibitor" "Tafel"',
    f'"{METAL_SYMBOL}" "corrosion inhibitor" "potentiodynamic"',
    f'"{METAL_SYMBOL}" "inhibition efficiency" "Tafel"',

    # --- 跨化學家族代表詞（確保搜集不同種類的 inhibitor）---
    f'"{TARGET_METAL}" "triazole" "potentiodynamic polarization"',
    f'"{TARGET_METAL}" "imidazole" "potentiodynamic polarization"',
    f'"{TARGET_METAL}" "tetrazole" "corrosion" "polarization"',
    f'"{TARGET_METAL}" "thiazole" "corrosion" "Tafel"',
    f'"{TARGET_METAL}" "Schiff base" "potentiodynamic polarization"',
    f'"{TARGET_METAL}" "amino acid" "corrosion inhibitor" "polarization"',
    f'"{TARGET_METAL}" "cysteine" "corrosion" "potentiodynamic"',
    f'"{TARGET_METAL}" "thiourea" "corrosion" "polarization"',
    f'"{TARGET_METAL}" "surfactant" "corrosion inhibitor" "Tafel"',
    f'"{TARGET_METAL}" "ionic liquid" "corrosion inhibitor" "polarization"',
    f'"{TARGET_METAL}" "CMP" "corrosion inhibitor" "potentiodynamic"',
    f'"{TARGET_METAL}" "slurry" "inhibitor" "potentiodynamic"',

    # --- 該金屬專屬緩蝕劑查詢（來自 metals_db.json）---
    *INHIBITOR_QUERIES,
]

_CATALYSIS_QUERIES = [
    f'"{TARGET_METAL}" "Tafel slope"',
    f'"{TARGET_METAL}" "Tafel plot"',
    f'"{TARGET_METAL}" "polarization curve"',
    f'"{TARGET_METAL}" "exchange current density"',
    f'"{TARGET_METAL}" "linear sweep voltammetry" "Tafel"',
    f'"{TARGET_METAL}" "cyclic voltammetry" corrosion',
    f'"{TARGET_METAL}" "hydrogen evolution" "Tafel"',
    f'"{TARGET_METAL}" "oxygen evolution" "Tafel"',
    f'"{TARGET_METAL}" "oxygen reduction" "Tafel"',
    f'"{TARGET_METAL}" "CO2 reduction" "Tafel"',
    f'"{METAL_SYMBOL}" "Tafel slope" electrochemistry',
    f'"{METAL_SYMBOL}" "Tafel slope" electrocatalyst',
    f'"{METAL_SYMBOL}" "Tafel slope" catalyst',
    *[f'"{comp}" "Tafel slope"' for comp in COMPOUNDS],
    *[f'"{comp}" "Tafel"' for comp in COMPOUNDS],
]

_MODE_QUERY_MAP: dict[str, list[str]] = {
    "corrosion": _CORROSION_QUERIES,
    "cmp_etch": _CMP_ETCH_QUERIES,
    "inhibitor": _INHIBITOR_QUERIES,
    "catalysis": _CATALYSIS_QUERIES,
}


def build_search_queries(modes: list[str] | tuple[str, ...] | str | None = None) -> list[str]:
    """依指定的搜尋模式清單組裝並去重 OpenAlex 查詢清單。"""
    active_modes = SEARCH_MODES if modes is None else parse_search_modes(modes)
    return list(dict.fromkeys(
        q for m in active_modes for q in _MODE_QUERY_MAP[m]
    ))


SEARCH_QUERIES: list[str] = build_search_queries(SEARCH_MODES)

# ============================================================
#  5. 資料夾與輸出檔案路徑（根據金屬自動歸類）
# ============================================================

if _us.SIMULATOR_DATA_DIR:
    _custom_sim = os.path.expanduser(str(_us.SIMULATOR_DATA_DIR).strip())
    SIMULATOR_DATA_DIR = (
        _custom_sim
        if os.path.isabs(_custom_sim)
        else os.path.abspath(os.path.join(PROJECT_ROOT, _custom_sim))
    )
else:
    SIMULATOR_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "merged")

os.makedirs(SIMULATOR_DATA_DIR, exist_ok=True)
MP_CACHE_PATH = os.path.join(SIMULATOR_DATA_DIR, "mp_cache.json")

DATA_DIR = os.path.join(PROJECT_ROOT, "data", METAL_SYMBOL)
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
TEXT_DIR = os.path.join(DATA_DIR, "texts")
EXTRACTED_DIR = os.path.join(DATA_DIR, "extracted")

# 若不是純讀取設定的靜默模式，自動建立當前金屬所需資料夾
if os.environ.get("TAFEL_QUIET_CONFIG") != "1":
    for _d in [METADATA_DIR, PDF_DIR, TEXT_DIR, EXTRACTED_DIR]:
        os.makedirs(_d, exist_ok=True)

STEP1_OUTPUT = os.path.join(
    METADATA_DIR,
    f"{METAL_SYMBOL.lower()}_tafel_metadata.csv",
)
S2_CACHE_PATH = os.path.join(
    METADATA_DIR,
    f"{METAL_SYMBOL.lower()}_s2_cache.json",
)
STEP2_OUTPUT = os.path.join(
    METADATA_DIR,
    f"{METAL_SYMBOL.lower()}_tafel_with_pdf.csv",
)
STEP5_OUTPUT = os.path.join(
    METADATA_DIR,
    f"{METAL_SYMBOL.lower()}_echem_extracted.csv",
)
STEP5_MATERIAL_OUTPUT = os.path.join(
    EXTRACTED_DIR,
    f"{METAL_SYMBOL.lower()}_material_properties.csv",
)
STEP5_SOLUTION_OUTPUT = os.path.join(
    EXTRACTED_DIR,
    f"{METAL_SYMBOL.lower()}_solution_properties.csv",
)
STEP5_ECHEM_OUTPUT = os.path.join(
    EXTRACTED_DIR,
    f"{METAL_SYMBOL.lower()}_electrochemical_database.csv",
)
STEP5_RAW_OUTPUT = os.path.join(
    EXTRACTED_DIR,
    f"{METAL_SYMBOL.lower()}_raw_records.csv",
)
STEP5_OUTLIERS_OUTPUT = os.path.join(
    EXTRACTED_DIR,
    f"{METAL_SYMBOL.lower()}_filtered_outliers.csv",
)

# ============================================================
#  6. 材料 regex pattern 產生器
# ============================================================

MATERIAL_PATTERNS = [
    *COMPOUNDS,
    f"{TARGET_METAL} oxide",
    f"{TARGET_METAL} hydroxide",
    f"{TARGET_METAL} foam",
    f"{TARGET_METAL} foil",
    f"{TARGET_METAL} electrode",
    f"{TARGET_METAL} nanowire",
    f"{TARGET_METAL} nanoparticle",
    f"{TARGET_METAL} thin film",
    f"{METAL_SYMBOL} electrode",
    f"{METAL_SYMBOL} foam",
    f"{METAL_SYMBOL} foil",
    f"{METAL_SYMBOL}-based",
]


def build_target_regex(target_metal=None, symbol=None, compounds=None):
    """
    建立「目標金屬」的 regex（Step 3 篩選與 Step 5 共用）。
    - 英文全名 / 多字詞化合物：不分大小寫 + 單字邊界
    - 元素符號 / 化學式：區分大小寫，前面不可接字母、後面不可接小寫字母
    """
    target_metal = target_metal or TARGET_METAL
    symbol = symbol or METAL_SYMBOL
    compounds = COMPOUNDS if compounds is None else compounds

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


# ============================================================
#  7. 印出目前設定（執行任何步驟時自動顯示）
# ============================================================


def print_config():
    print("=" * 58)
    print(f"  Target Metal : {TARGET_METAL} ({METAL_SYMBOL})")
    print(
        f"  Search Modes : {', '.join(SEARCH_MODES)}  |  "
        f"Europe PMC: {'ON (' + str(EUROPEPMC_MAX_PAPERS) + ')' if ENABLE_EUROPEPMC_XML else 'OFF'}"
    )
    s2_status = (
        "ON (API Key)"
        if (ENABLE_SEMANTIC_SCHOLAR and SEMANTIC_SCHOLAR_API_KEY)
        else ("ON (Public)" if ENABLE_SEMANTIC_SCHOLAR else "OFF")
    )
    mp_status = (
        "ON (API Key)"
        if (ENABLE_MP_API and MP_API_KEY)
        else ("ON (Offline)" if ENABLE_MP_API else "OFF")
    )
    print(f"  OpenAlex Key : {'設定完成' if OPENALEX_API_KEY else '未設定（共用 IP 免費額度）'}")
    print(f"  S2 / MP API  : Semantic Scholar {s2_status}  |  Materials Project {mp_status}")
    print(
        f"  Compounds    : {len(COMPOUNDS)} types  |  "
        f"CMP Qs: {len(CMP_QUERIES)}  |  Inhibitor Qs: {len(INHIBITOR_QUERIES)}"
    )
    print(f"  Queries      : {len(SEARCH_QUERIES)} combinations")
    print(f"  Max/query    : {MAX_RECORDS_PER_QUERY} records")
    print(f"  From year    : {FROM_YEAR}")
    print(f"  Data folder  : {DATA_DIR}")
    print(f"  Step1 output : {STEP1_OUTPUT}")
    print(f"  Step2 output : {STEP2_OUTPUT}")
    print(f"  Step5 output : {STEP5_ECHEM_OUTPUT}")
    print("=" * 58)


if os.environ.get("TAFEL_QUIET_CONFIG") != "1":
    print_config()
