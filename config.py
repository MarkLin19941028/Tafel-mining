"""
========================================
  Tafel Mining Project — 中央設定檔
  所有可自定義的參數都在這裡修改
========================================
"""

import os
import re as _re

# ============================================================
#  1. 基本設定
# ============================================================

# ✏️ 你的 email（OpenAlex / Unpaywall 用）
MAILTO = "cy90207@gmail.com"

# ✏️ OpenAlex API Key（可貼在引號內，或設為環境變數 OPENALEX_API_KEY）
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY") or "IvtlLFfENlf2upJBwE3p6b"

# ✏️ 斷點續跑開關：True = 若輸出檔已存在則自動略過（可用 run_all.py --force 強制重跑）
SKIP_EXISTING_STEP1 = True
SKIP_EXISTING_STEP2 = True

# ✏️ 每組關鍵字最多抓幾篇（測試用 50，正式用 300）
MAX_RECORDS_PER_QUERY = 2

# ✏️ PDF 下載時每筆間隔秒數（避免被封鎖）
DOWNLOAD_DELAY = 2

# ✏️ API 請求 timeout（秒）
REQUEST_TIMEOUT = 30

# ✏️ 搜尋文獻的起始年份
FROM_YEAR = 2010

# ✏️ Step 1 搜尋模式："corrosion"（聚焦腐蝕極化曲線，減少電催化雜訊）或 "all"（腐蝕 + 電催化）
SEARCH_MODE = "corrosion"

# ✏️ Europe PMC 開放全文 XML 設定（免 API Key、無 403 封鎖，且可直接取得結構化表格）
ENABLE_EUROPEPMC_XML = True
EUROPEPMC_MAX_PAPERS = 100    # 每個金屬主動從 Europe PMC 抓取的開放腐蝕論文上限
EUROPEPMC_DELAY = 0.5         # Europe PMC API 請求間隔秒數

# ============================================================
#  2. 多金屬預設表與批次執行清單（✏️ 改這裡就能增減金屬）
# ============================================================

METAL_PRESETS: dict[str, dict] = {
    "Cu": {
        "target_metal": "copper",
        "compounds": [
            "Cu2O", "CuO", "Cu(OH)2",
            "CuS", "Cu2S", "CuSe", "Cu3P",
            "Cu-Ni", "Cu-Zn", "Cu-Al", "Cu-Ti",
            "copper foam", "copper foil",
            "copper nanowire", "copper nanoparticle",
            "copper thin film", "copper alloy",
        ],
    },
    "Co": {
        "target_metal": "cobalt",
        "compounds": [
            "Co3O4", "CoO", "CoOOH", "Co(OH)2",
            "CoP", "Co2P", "CoS2", "CoSe2", "Co9S8",
            "CoN", "Co3C", "CoFe", "CoNi", "CoMo",
            "CoFe2O4", "ZIF-67", "Co-MOF",
            "cobalt phosphide", "cobalt sulfide",
            "cobalt oxide", "cobalt hydroxide",
        ],
    },
    "W": {
        "target_metal": "tungsten",
        "compounds": [
            "WC", "W2C", "WS2", "WSe2",
            "WO3", "WO2", "WP", "WN", "W2N",
            "NiW", "CoW", "MoW",
            "tungsten carbide", "tungsten disulfide",
            "tungsten oxide", "tungsten phosphide",
            "tungsten nitride", "tungsten thin film",
        ],
    },
    "Ru": {
        "target_metal": "ruthenium",
        "compounds": [
            "RuO2", "RuP", "RuP2", "RuS2", "RuSe2",
            "RuIr", "RuCo", "RuNi",
            "Ru/C", "SrRuO3",
            "Ru single atom", "Ru-SA",
            "ruthenium oxide", "ruthenium dioxide",
            "ruthenium nanoparticle",
        ],
    },
    "Ti": {
        "target_metal": "titanium",
        "compounds": [
            "TiO2", "Ti2O3", "TiO",
            "TiN", "TiC", "TiB2",
            "TiS2", "TiSe2",
            "Ti-6Al-4V", "Ti-Ni", "Ti-Cu", "Ti-Ta",
            "titanium dioxide", "titanium oxide",
            "titanium nitride", "titanium carbide",
            "titanium foil", "titanium alloy",
            "titanium implant", "titanium substrate",
            "anodized titanium", "titanium nanotube",
        ],
    },
    "TiN": {
        "target_metal": "titanium nitride",
        "compounds": [
            "TiN", "Ti2N",
            "TiN thin film", "TiN coating",
            "TiN nanoparticle", "TiN nanotube",
            "TiN electrode", "TiN substrate",
            "TiAlN", "TiCN", "TiSiN",
            "TiN/Si", "TiN/Ti", "TiN/Cu",
            "titanium nitride film",
            "titanium nitride coating",
            "titanium nitride electrode",
            "reactive sputtering TiN",
            "PVD TiN", "CVD TiN", "ALD TiN",
        ],
    },
    "TaN": {
        "target_metal": "tantalum nitride",
        "compounds": [
            "TaN", "Ta3N5", "Ta2N",
            "TaN thin film", "TaN coating",
            "TaN electrode", "TaN barrier",
            "TaN nanoparticle",
            "TaN/Si", "TaN/Cu", "TaN/Ti",
            "tantalum nitride film",
            "tantalum nitride coating",
            "tantalum nitride electrode",
            "tantalum nitride barrier",
            "reactive sputtering TaN",
            "PVD TaN", "CVD TaN", "ALD TaN",
            "TaTiN", "TaSiN", "TaAlN",
        ],
    },
    "Al": {
        "target_metal": "aluminum",
        "compounds": [
            "Al2O3", "AlOOH", "Al(OH)3",
            "AlN", "AlP",
            "Al-Cu", "Al-Mg", "Al-Zn", "Al-Si", "Al-Ti",
            "AA2024", "AA7075", "AA6061", "AA5052",
            "aluminum alloy", "aluminium alloy",
            "aluminum oxide", "aluminium oxide",
            "aluminum foil", "aluminium foil",
            "anodized aluminum", "anodized aluminium",
            "aluminum coating", "aluminium coating",
        ],
    },
    "Ta": {
        "target_metal": "tantalum",
        "compounds": [
            "Ta2O5", "TaO2", "TaON",
            "TaN", "Ta3N5", "Ta2N",
            "TaC", "TaB2", "TaS2", "TaSe2",
            "Ta-Ti", "Ta-W", "Ta-Cu",
            "tantalum oxide", "tantalum pentoxide",
            "tantalum nitride", "tantalum carbide",
            "tantalum foil", "tantalum thin film",
            "tantalum electrode", "tantalum capacitor",
            "anodized tantalum",
        ],
    },
    "Mn": {
        "target_metal": "manganese",
        "compounds": [
            "MnO2", "Mn2O3", "Mn3O4", "MnO", "MnOOH",
            "MnS", "MnSe", "MnP",
            "MnFe2O4", "MnCo2O4", "MnNiO",
            "NiMn", "CoMn", "FeMn",
            "manganese oxide", "manganese dioxide",
            "manganese phosphide", "manganese sulfide",
            "MnO2 nanosheet", "MnO2 nanowire",
            "birnessite", "spinel manganese",
            "manganese steel",
        ],
    },
    "Mo": {
        "target_metal": "molybdenum",
        "compounds": [
            "MoS2", "MoSe2", "MoTe2",
            "MoO2", "MoO3",
            "Mo2C", "MoC", "MoN", "MoP", "Mo2N",
            "NiMo", "CoMo", "FeMo",
            "1T-MoS2", "2H-MoS2",
            "molybdenum disulfide", "molybdenum carbide",
            "molybdenum phosphide", "molybdenum oxide",
            "molybdenum nitride",
            "MoS2 nanosheet", "MoS2 edge",
        ],
    },
}

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

SEARCH_QUERIES = (
    _CORROSION_QUERIES
    if SEARCH_MODE == "corrosion"
    else list(dict.fromkeys(_CORROSION_QUERIES + _CATALYSIS_QUERIES))
)

# ============================================================
#  4. 資料夾路徑（根據金屬自動歸類）
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Step 6 合併輸出目錄（所有金屬合併後的資料表存放於此）
# ✏️ 若要接 corrosion_simulator，改成該專案的絕對路徑即可
SIMULATOR_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "merged")
os.makedirs(SIMULATOR_DATA_DIR, exist_ok=True)

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

# ✏️ 各金屬 Ecorr (V vs SHE) 合理腐蝕電位區間（超出者視為異質基材主導或誤抓數值）
ECORR_VALID_RANGE_V_SHE: dict[str, tuple[float, float]] = {
    "Cu": (-0.6, 0.8),
    "Co": (-1.0, 0.6),
    "W": (-1.0, 0.8),
    "Ru": (-0.5, 1.4),
    "Ti": (-1.2, 0.8),
    "TiN": (-0.8, 1.0),
    "TaN": (-0.8, 1.0),
    "Al": (-1.8, 0.2),
    "Ta": (-1.2, 0.8),
    "Mn": (-1.5, 0.6),
    "Mo": (-1.0, 0.8),
    "default": (-2.0, 2.0),
}

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
KEEP_ONLY_TARGET_METAL = True

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
    print(f"  OpenAlex Key : {'設定完成' if OPENALEX_API_KEY else '未設定（共用 IP 免費額度）'}")
    print(f"  Compounds    : {len(COMPOUNDS)} types")
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
