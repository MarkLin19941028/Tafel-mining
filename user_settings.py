"""
====================================================================
  Tafel Mining — 使用者自定義設定檔 (user_settings.py)
  所有日常可調整的參數集中於此；API 金鑰請設定於專案根目錄的 .env
====================================================================
"""

# ============================================================
#  1. 基本連線與流程控制
# ============================================================

# ✏️ 聯絡 Email（供 OpenAlex / Unpaywall Polite Pool 識別，建議填寫真實信箱）
MAILTO: str = "cy90207@gmail.com"

# ✏️ 斷點續跑開關：True = 若輸出檔已存在則自動略過
#    （可用 `run_all.py --force` 強制覆寫重跑，或 `--append` 增量合併新論文）
SKIP_EXISTING_STEP1: bool = True
SKIP_EXISTING_STEP2: bool = True

# ✏️ Step 1 搜尋過程中即時比對去重（跳過既有 metadata 與本次已見論文，兼顧效能與增量效益）
STEP1_INFLIGHT_DEDUP: bool = True
STEP1_MAX_SCAN_MULTIPLIER: int = 3       # 單組查詢最大翻頁倍率（避免無限翻頁消耗配額）
STEP1_EARLY_STOP_EMPTY_PAGES: int = 2    # 連續幾頁無新論文即提前結束該組查詢

# ✏️ OpenAlex 每組關鍵字最多抓取篇數
#    測試建議 50，正式建議 200~300（稀缺材料可調至 500）
MAX_RECORDS_PER_QUERY: int = 300

# ✏️ Step 2 Unpaywall 最佳化設定
#    True = 若 OpenAlex / Semantic Scholar 已有 PDF URL 則跳過 Unpaywall（實測 99.8% 重複，可加速 20~50 倍）
STEP2_SKIP_UNPAYWALL_IF_HAS_PDF: bool = True
STEP2_UNPAYWALL_DELAY: float = 0.2       # Unpaywall 查詢間隔秒數（Polite Pool 額度充足）
STEP2_CHECKPOINT_EVERY: int = 200        # Step 2 每查詢幾筆自動原子存檔一次（防中斷重跑）

# ✏️ Step 3 下載 PDF 時同一網域（domain）的最小間隔秒數（避免觸發出版社防爬蟲封鎖）
DOWNLOAD_DELAY: float = 2.0
STEP3_MAX_WORKERS: int = 4               # Step 3 跨網域並行下載執行緒數（同網域仍嚴格遵守 DOWNLOAD_DELAY）
STEP3_DOMAIN_403_THRESHOLD: int = 15     # 同批次單一網域連續 403 達此門檻即熔斷跳過，改走其他候選或 Europe PMC
STEP3_MAX_PDF_MB: int = 50               # 單一 PDF 檔案大小上限 (MB)，避免卡死於異常大檔

# ✏️ HTTP API 請求逾時秒數
REQUEST_TIMEOUT: int = 30

# ✏️ 搜尋文獻的起始發表年份（建議 2000~2005；2000 年前 OA 比例低且多為掃描檔）
FROM_YEAR: int = 2005


# ============================================================
#  2. 搜尋模式組合（可自由多選搭配）
# ============================================================

# ✏️ 預設啟用的搜尋模式清單（亦可於執行時用 `run_all.py --search-mode ...` 臨時覆蓋）：
#    - "corrosion" : 一般金屬與化合物腐蝕極化曲線（Tafel / Ecorr / icorr）
#    - "cmp_etch"  : 半導體 CMP / 研磨液 / 濕蝕刻 / 氧化劑環境
#    - "inhibitor" : 多樣化腐蝕與 CMP 緩蝕劑（跨化學家族與專屬緩蝕劑）
#    - "solvent"   : 非水 / 有機 / 半水 / 混合溶劑與共晶溶劑環境
#    - "catalysis" : 電催化（HER / OER / ORR / CO2RR Tafel 斜率；注意主表僅收錄腐蝕類）
SEARCH_MODES: list[str] = [
    "corrosion",
    "cmp_etch",
    "inhibitor",
    "solvent",
    # "catalysis",  # 電催化（HER/OER/ORR/CO2RR）在 Step 5 主表會被剔除；有需要可用 --search-mode catalysis 臨時開啟
]


# ============================================================
#  3. 外部資料源開關與速率設定
# ============================================================

# ✏️ Semantic Scholar API（金鑰請設於 .env 的 SEMANTIC_SCHOLAR_API_KEY；留空亦可運作）
ENABLE_SEMANTIC_SCHOLAR: bool = True
SEMANTIC_SCHOLAR_USE_BULK: bool = True          # 建議維持 True：依選取模式濃縮為少數高密度布林查詢走 /search/bulk
SEMANTIC_SCHOLAR_MAX_PER_QUERY: int = 1000      # 建議 800~1000（/search/bulk 單次上限為 1,000）
SEMANTIC_SCHOLAR_MAX_QUERIES_NO_KEY: int = 15   # 僅在 USE_BULK=False 且無 Key 時生效的查詢組數上限
SEMANTIC_SCHOLAR_DELAY: float = 1.2             # 請求間最小間隔秒數（已設 API Key 適用 1.2s；遇 429 會自動 AIMD 動態拉長）

# ✏️ Europe PMC 開放全文 XML（免 API Key、零 403 封鎖，可直接取得原生結構化表格）
ENABLE_EUROPEPMC_XML: bool = True
EUROPEPMC_MAX_PAPERS: int = 800                 # 建議 500~1000
EUROPEPMC_DELAY: float = 0.5                    # Europe PMC API 請求間隔秒數

# ✏️ Materials Project API（Step 7 物理與 Pourbaix 熱力學強化；金鑰請設於 .env 的 MP_API_KEY）
ENABLE_MP_API: bool = True


# ============================================================
#  4. 批次執行金屬清單與輸出目錄
# ============================================================

# ✏️ 執行 `python run_all.py` 時預設批次跑的金屬清單（可註解掉暫時不跑的材料）
#    各材料的搜尋詞與電位區間定義於 metals_db.json；物性常數定義於 materials_physics.json
METALS_TO_RUN: list[str] = [
    "Cu",
    "Co",
    "W",
    "Ru",
    "Ti",
    "TiN",
    "TaN",
    "WN",
    "MoN",
    "Al",
    "Ta",
    "Mn",
    "Mo",
]

# ✏️ 單獨執行 step1 ~ step5 腳本（未指定 TAFEL_METAL 環境變數）時的預設材料
DEFAULT_METAL: str = "Cu"

# ✏️ Step 6 / Step 7 跨金屬合併輸出目錄：
#    設為 None 時預設輸出至本專案的 `data/merged`；若要串接外部模擬器可填絕對路徑字串
SIMULATOR_DATA_DIR: str | None = None


# ============================================================
#  5. Step 3 & Step 5 資料篩選與物理合理區間
# ============================================================

# ✏️ Tafel 斜率合理擷取範圍 (mV/dec)
TAFEL_MIN: float = 20
TAFEL_MAX: float = 500

# ✏️ Step 5 內文正則擷取時的前後文視窗字元數
CONTEXT_WINDOW: int = 150

# ✏️ True = 自動濾除超出物理合理區間的離群值（被濾除筆數會另存於 *_filtered_outliers.csv）
FILTER_OUTLIERS: bool = True

# ✏️ icorr (µA/cm²) 合理物理區間（各金屬 Ecorr 合理區間請見 metals_db.json）
ICORR_VALID_RANGE_UA_CM2: tuple[float, float] = (1e-4, 1e6)

# ✏️ True = 最終資料表僅保留目標金屬本身的記錄（排除同篇論文對照組中的其他金屬）
KEEP_ONLY_TARGET_METAL: bool = False

# ✏️ True = Step 3 僅下載「標題或摘要有提到目標金屬／化合物」的論文（強烈建議維持 True）
REQUIRE_TARGET_IN_TITLE_ABSTRACT: bool = True

# ✏️ 一篇論文內提及次數少於此門檻的材料，視為順帶提及（不作為主要候選材料）
MIN_MATERIAL_MENTIONS: int = 3
