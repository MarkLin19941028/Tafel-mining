"""
========================================
  Tafel Mining Project — 中央設定檔
  所有可自定義的參數都在這裡修改
========================================
"""

import json as _json
import os
import re as _re
from pathlib import Path as _Path


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


_load_dotenv(_Path(__file__).resolve().parent / ".env")

# ============================================================
#  1. 基本設定
# ============================================================

# ✏️ 你的 email（OpenAlex / Unpaywall 用）
MAILTO = "cy90207@gmail.com"

# ✏️ OpenAlex API Key（請設於 .env 或環境變數 OPENALEX_API_KEY）
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "")

# ✏️ 斷點續跑開關：True = 若輸出檔已存在則自動略過（可用 run_all.py --force 強制重跑）
SKIP_EXISTING_STEP1 = True
SKIP_EXISTING_STEP2 = True

# ✏️ 每組關鍵字最多抓幾篇（測試用 50，正式用 300）
MAX_RECORDS_PER_QUERY = 200

# ✏️ PDF 下載時每筆間隔秒數（避免被封鎖）
DOWNLOAD_DELAY = 2

# ✏️ API 請求 timeout（秒）
REQUEST_TIMEOUT = 30

# ✏️ 搜尋文獻的起始年份
FROM_YEAR = 2010

# ✏️ Step 1 搜尋模式：
#    "corrosion"     = 僅一般腐蝕極化曲線
#    "cmp_etch"      = 僅 CMP / 濕蝕刻 / 氧化劑環境（適合拉高 Ru/Co/W/TiN 數據量）
#    "corrosion_cmp" = 一般腐蝕 + CMP / 濕蝕刻 / 氧化劑環境（預設推薦）
#    "all"           = 腐蝕 + CMP/濕蝕刻 + 電催化
SEARCH_MODE = "corrosion_cmp"

# ✏️ Semantic Scholar API 設定（請設於 .env 或環境變數 SEMANTIC_SCHOLAR_API_KEY；留空亦可運作）
ENABLE_SEMANTIC_SCHOLAR = True
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
SEMANTIC_SCHOLAR_USE_BULK = True          # True = 將 60+ 條查詢濃縮為 3 組高容量布林查詢並走 /paper/search/bulk（減少 95% 請求與 429）
SEMANTIC_SCHOLAR_MAX_PER_QUERY = 200      # 每組 bulk 查詢最多保留幾篇具 DOI 論文（/search/bulk 單次最多可回傳 1000 篇）
SEMANTIC_SCHOLAR_MAX_QUERIES_NO_KEY = 15  # 未展開布林合併且無 Key 時的查詢組數上限
SEMANTIC_SCHOLAR_DELAY = 3.0              # 每次 HTTP 請求間的最小保證間隔秒數（遇 429 會自動 AIMD 動態拉長）

# ✏️ Europe PMC 開放全文 XML 設定（免 API Key、無 403 封鎖，且可直接取得結構化表格）
ENABLE_EUROPEPMC_XML = True
EUROPEPMC_MAX_PAPERS = 200    # 每個金屬主動從 Europe PMC 抓取的開放腐蝕論文上限
EUROPEPMC_DELAY = 0.5         # Europe PMC API 請求間隔秒數

# ✏️ Materials Project API 設定（Step 7 物理與 Pourbaix 熱力學特徵強化用；請設於 .env 或環境變數 MP_API_KEY）
ENABLE_MP_API = True
MP_API_KEY = os.environ.get("MP_API_KEY", "")

# ============================================================
#  2. 多金屬預設表與批次執行清單（金屬資料庫獨立於 metals_db.json）
# ============================================================

METALS_DB_PATH = _Path(__file__).resolve().parent / "metals_db.json"


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
        }
        if "ecorr_valid_range_v_she" in entry:
            lo, hi = entry["ecorr_valid_range_v_she"]
            ecorr_ranges[sym] = (float(lo), float(hi))

    ecorr_ranges["default"] = (float(default_range[0]), float(default_range[1]))
    return presets, ecorr_ranges


METAL_PRESETS, ECORR_VALID_RANGE_V_SHE = _load_metals_db(METALS_DB_PATH)

# ✏️ 執行 `python run_all.py` 時要批次跑的金屬清單（可註解掉不想跑的金屬）
#    也可透過命令列臨時覆蓋，例如：python run_all.py Cu Co W
METALS_TO_RUN: list[str] = [
    "Cu",
    "Co",
    "W",
    "Ru",
    "Ti",
    "TiN",
    "TaN",
    "Al",
    "Ta",
    "Mn",
    "Mo",
]

# ✏️ 單獨執行 step1~step5 時（未設 TAFEL_METAL 環境變數）的預設金屬
DEFAULT_METAL = "Cu"


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

# ============================================================
#  3. 搜尋關鍵字（根據上方金屬設定自動生成）
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

if SEARCH_MODE == "corrosion":
    SEARCH_QUERIES = list(dict.fromkeys(_CORROSION_QUERIES))
elif SEARCH_MODE == "cmp_etch":
    SEARCH_QUERIES = list(dict.fromkeys(_CMP_ETCH_QUERIES))
elif SEARCH_MODE == "corrosion_cmp":
    SEARCH_QUERIES = list(dict.fromkeys(_CORROSION_QUERIES + _CMP_ETCH_QUERIES))
else:
    SEARCH_QUERIES = list(dict.fromkeys(_CORROSION_QUERIES + _CMP_ETCH_QUERIES + _CATALYSIS_QUERIES))

# ============================================================
#  4. 資料夾路徑（根據金屬自動歸類）
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Step 6 合併輸出目錄（所有金屬合併後的資料表存放於此）
# ✏️ 若要接 corrosion_simulator，改成該專案的絕對路徑即可
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

# ============================================================
#  5. 輸出檔案路徑（所有步驟的產出位置）
# ============================================================

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
#  6. Step 5 資料過濾與物理合理區間設定
# ============================================================

TAFEL_MIN = 20    # mV/dec
TAFEL_MAX = 500   # mV/dec
CONTEXT_WINDOW = 150

# ✏️ True = 自動過濾超出物理合理區間的離群值（被濾除的筆數會另存於 *_filtered_outliers.csv）
FILTER_OUTLIERS = True

# 各金屬 Ecorr (V vs SHE) 合理腐蝕電位區間已由上方 metals_db.json 載入為 ECORR_VALID_RANGE_V_SHE

# ✏️ icorr (µA/cm²) 合理區間
ICORR_VALID_RANGE_UA_CM2: tuple[float, float] = (1e-4, 1e6)

# ============================================================
#  7. 材料 regex pattern 與 Step 3 / Step 5 材料判定設定
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

# ✏️ True = 最終資料表只保留目標金屬（及其化合物）的記錄與溶液
KEEP_ONLY_TARGET_METAL = False

# ✏️ True = Step 3 只下載「標題或摘要有提到目標金屬」的論文
REQUIRE_TARGET_IN_TITLE_ABSTRACT = True

# ✏️ 一篇論文中出現少於 N 次的金屬，視為順帶提及（不列入該論文的材料清單）
MIN_MATERIAL_MENTIONS = 3


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
#  8. 印出目前設定（執行任何步驟時自動顯示）
# ============================================================


def print_config():
    print("=" * 55)
    print(f"  Target Metal : {TARGET_METAL} ({METAL_SYMBOL})")
    print(f"  Search Mode  : {SEARCH_MODE}  |  Europe PMC: {'ON (' + str(EUROPEPMC_MAX_PAPERS) + ')' if ENABLE_EUROPEPMC_XML else 'OFF'}")
    s2_status = "ON (API Key)" if (ENABLE_SEMANTIC_SCHOLAR and SEMANTIC_SCHOLAR_API_KEY) else ("ON (Public)" if ENABLE_SEMANTIC_SCHOLAR else "OFF")
    mp_status = "ON (API Key)" if (ENABLE_MP_API and MP_API_KEY) else ("ON (Offline)" if ENABLE_MP_API else "OFF")
    print(f"  OpenAlex Key : {'設定完成' if OPENALEX_API_KEY else '未設定（共用 IP 免費額度）'}")
    print(f"  S2 / MP API  : Semantic Scholar {s2_status}  |  Materials Project {mp_status}")
    print(f"  Compounds    : {len(COMPOUNDS)} types  |  CMP queries: {len(CMP_QUERIES)}")
    print(f"  Queries      : {len(SEARCH_QUERIES)} combinations")
    print(f"  Max/query    : {MAX_RECORDS_PER_QUERY} records")
    print(f"  From year    : {FROM_YEAR}")
    print(f"  Data folder  : {DATA_DIR}")
    print(f"  Step1 output : {STEP1_OUTPUT}")
    print(f"  Step2 output : {STEP2_OUTPUT}")
    print(f"  Step5 output : {STEP5_ECHEM_OUTPUT}")
    print("=" * 55)


if os.environ.get("TAFEL_QUIET_CONFIG") != "1":
    print_config()
