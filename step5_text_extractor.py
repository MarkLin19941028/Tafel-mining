"""
Step 5 — Regex patterns, environment/material/solution helpers, and text extraction.
"""

import hashlib
import re
from collections import Counter, OrderedDict

import numpy as np

from config import (
    CONTEXT_WINDOW,
    METAL_SYMBOL,
    MIN_MATERIAL_MENTIONS,
    TAFEL_MAX,
    TAFEL_MIN,
    TARGET_METAL,
    build_target_regex,
)
from step5_constants import (
    KNOWN_MATERIALS,
    convert_log_icorr_to_uA_cm2,
    convert_to_she,
    normalize_ecorr_to_v,
    normalize_icorr_to_uA_cm2,
    normalize_text,
    parse_scientific,
)

# ====================================================================

# 5. REGEX PATTERNS

# ====================================================================

 

_NUM = r"[\-−–]?\s*\d+(?:\.\d+)?(?:\s*(?:[×xX*·⋅]\s*10\s*\^?\s*[\-−–]?\s*\d+|[eE][\-−–+]?\d+))?"

_TAFEL_UNIT = r"(?:mV|V)\s*(?:[/·⋅\s]\s*dec(?:ade)?(?:\s*[\-−–]\s*1)?|dec(?:ade)?\s*[\-−–]\s*1)"

TAFEL_PATTERN = re.compile(
    r"(?:Tafel\s+slope[s]?|(?:anodic|cathodic)\s+slope|\bb\s*[ac]?\s*[=≈:]|\bβ\s*[ac]?\s*[=≈:])"
    r"[^\.\n]{0,120}?"
    r"(?P<value>" + _NUM + r")"
    r"\s*"
    r"(?P<unit>" + _TAFEL_UNIT + r")",
    re.IGNORECASE,
)

TAFEL_ANODIC_PATTERN = re.compile(
    r"(?:anodic\s+(?:Tafel\s+)?slope|\bb\s*_?\s*a\b\s*(?:[=≈:]|of|was|is)?|\bβ\s*_?\s*a\b\s*(?:[=≈:]|of|was|is)?)"
    r"[^\.\n]{0,100}?"
    r"(?P<value>" + _NUM + r")"
    r"\s*"
    r"(?P<unit>" + _TAFEL_UNIT + r"|mV\b)",
    re.IGNORECASE,
)

TAFEL_CATHODIC_PATTERN = re.compile(
    r"(?:cathodic\s+(?:Tafel\s+)?slope|\bb\s*_?\s*c\b\s*(?:[=≈:]|of|was|is)?|\bβ\s*_?\s*c\b\s*(?:[=≈:]|of|was|is)?)"
    r"[^\.\n]{0,100}?"
    r"(?P<value>" + _NUM + r")"
    r"\s*"
    r"(?P<unit>" + _TAFEL_UNIT + r"|mV\b)",
    re.IGNORECASE,
)

# 依長度由長到短排列，避免 NaCl 截斷 NaClO4、HCl 截斷 HClO4
_ELECTROLYTE_SPECIES = sorted([
    # --- 酸 / 鹼 / 氧化劑 / 緩衝液 / 有機溶劑 / 螯合劑 ---
    "tetramethylammonium hydroxide", "ammonium hydroxide", "hydrogen peroxide",
    "hydrofluoric acid", "phosphate buffer", "borate buffer", "ethylene glycol",
    "citric acid", "oxalic acid", "acetic acid", "malic acid", "tartaric acid",
    "isopropanol", "(NH4)2S2O8", "Fe(NO3)3", "K3Fe(CN)6", "K4Fe(CN)6",
    "K2S2O8", "Na2S2O8", "Na2B4O7", "Na2HPO4", "KH2PO4", "K2HPO4", "NaH2PO4",
    "NaHCO3", "KHCO3", "NaClO4", "Na2SO4", "Na2CO3", "K2SO4", "K2CO3",
    "H2SO4", "H₂SO₄", "HClO4", "H3PO4", "NH4OH", "NH₄OH", "CuSO4", "CuCl2",
    "FeCl3", "KIO4", "KIO3", "NaClO", "H2O2", "H₂O₂", "HNO3", "NaOH", "NaCl",
    "KNO3", "NaNO3", "LiOH", "TMAH", "EDTA", "DMSO", "glycine",
    "KOH", "HCl", "KCl", "PBS", "IPA", "NMP", "HF", "EG",
    # --- 多樣化腐蝕與 CMP 緩蝕劑（唑類、雜環、含硫氮、胺基酸、界面活性劑、高分子、無機鹽）---
    "5-mercapto-1-phenyl-1H-tetrazole", "5-mercapto-1-phenyltetrazole",
    "sodium dodecylbenzenesulfonate", "cetyltrimethylammonium bromide",
    "3-amino-1,2,4-triazole", "2-mercaptobenzothiazole", "2-mercaptobenzimidazole",
    "hexamethylenetetramine", "sodium dodecyl sulfate", "polyvinylpyrrolidone",
    "polyethylene glycol", "polyethyleneimine", "8-hydroxyquinoline",
    "thiosemicarbazide", "5-aminotetrazole", "monoethanolamine",
    "potassium oleate", "polyacrylic acid", "1,3,4-thiadiazole",
    "sodium molybdate", "sodium silicate", "sodium tungstate", "sodium benzoate",
    "1,2,4-triazole", "aminotetrazole", "benzotriazole", "tolyltriazole",
    "benzimidazole", "sodium oleate", "sodium nitrite", "glutamic acid",
    "aspartic acid", "L-tryptophan", "Triton X-100", "L-methionine",
    "L-histidine", "L-cysteine", "L-arginine", "tryptophan", "methionine",
    "urotropine", "L-alanine", "histidine", "imidazole", "Ce(NO3)3",
    "L-lysine", "cysteine", "thiourea", "pyrazole", "thiazole", "pyridine",
    "Tween-80", "Tween 80", "arginine", "K2Cr2O7", "Na2MoO4", "Na2SiO3",
    "Na2WO4", "adenine", "guanine", "alanine", "purine", "lysine",
    "CeCl3", "NaNO2", "5-AT", "8-HQ", "CTAB", "DTAB", "HMTA", "SDBS",
    "BTA", "TTA", "TAZ", "ATA", "MBI", "MBT", "MPT", "BIA", "MEA",
    "SDS", "PEG", "PEI", "PAA", "PVP", "TU",
], key=len, reverse=True)

_SPECIES_REGEX = "|".join(
    re.escape(s).replace(r"\ ", r"\s+") for s in _ELECTROLYTE_SPECIES
)

# 單位中 M / mM / µM 區分大小寫，避免 10 m (公尺/分鐘) 被誤判成 10 M
ELECTROLYTE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9.])(?P<conc>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>(?:m|µ|μ|u)?M|(?i:m|µ|μ|u)?mol\s*(?:L|l|dm)\s*[\-−–]\s*[13]"
    r"|(?i:m|µ|μ|u)?mol\s*/\s*(?:L|l|dm3)|N|(?i:wt\.?\s*%)|(?i:vol\.?\s*%)|%"
    r"|(?i:ppm)|(?i:m?g\s*/\s*(?:L|l|dm3))|(?i:m?g\s+(?:L|l|dm)\s*[\-−–]\s*[13]))"
    r"[\s\n]+(?:of[\s\n]+)?"
    r"(?P<species>(?i:" + _SPECIES_REGEX + r"))(?![A-Za-z0-9₀-₉])"
)

# pH 後面必須有分隔符號或空白，且排除引用編號列表 (如 pH13,14,26 或 pH 13, 14)
PH_PATTERN = re.compile(
    r"\bpH(?:\s*(?:=|of|≈|~|was|is|:)\s*|\s+)"
    r"(?P<value>\d+(?:\.\d+)?)"
    r"(?!\s*,\s*\d+)"
)

# 溫度：區分大小寫的獨立 K，避免 5 kV / 100 kHz / 1 kΩ 被誤判；支援 22 ± 2 °C 取主值 22
TEMP_PATTERN = re.compile(
    r"(?<![±\d.])(?P<value>\d+(?:\.\d+)?)"
    r"(?:\s*[±]\s*\d+(?:\.\d+)?)?\s*"
    r"(?P<unit>°\s*C|℃|(?i:deg(?:ree)?s?\s*C(?:elsius)?)|(?<![A-Za-z])K(?![A-Za-zΩΩ\x00-\x08·])|(?i:Kelvin))"
)

ROOM_TEMP_PATTERN = re.compile(
    r"\b(?:room\s+temperature|ambient\s+temperature|ambient\s+conditions)\b",
    re.IGNORECASE,
)


def normalise_ref_electrode(raw: str) -> str:
    """將不同寫法的參考電極統一標準名稱。"""
    s = re.sub(r"\s+", " ", normalize_text(raw or "")).strip()
    low = s.lower()
    if "ag" in low and "agcl" in low:
        return "Ag/AgCl"
    if "sce" in low or "calomel" in low:
        return "SCE"
    if "hg/hgo" in low.replace(" ", ""):
        return "Hg/HgO"
    if "hg2so4" in low.replace(" ", "") or "mercury" in low and "sulfate" in low:
        return "Hg/Hg2SO4"
    if "mse" in low:
        return "MSE"
    if "rhe" in low or "reversible hydrogen" in low:
        return "RHE"
    if "she" in low or "nhe" in low or "standard hydrogen" in low or "normal hydrogen" in low:
        return "SHE"
    return s


def extract_ref_electrodes(text: str) -> Counter:
    """依出現次數統計參考電極（去除參考文獻，確保順序確定性）。"""
    counts: Counter = Counter()
    for m in REF_ELECTRODE_PATTERN.finditer(text):
        norm = normalise_ref_electrode(m.group("ref"))
        if norm:
            counts[norm] += 1
    return counts


def select_ref_electrode(contexts: list[str], paper_ref_counts: Counter) -> str | None:
    """優先取數值上下文中的參考電極，否則取全文出現最多次者（平手依字母序確保 100% 確定性）。"""
    ctx_text = " ".join(c for c in contexts if c)
    if ctx_text:
        ctx_refs = extract_ref_electrodes(ctx_text)
        if ctx_refs:
            return sorted(ctx_refs.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    if paper_ref_counts:
        return sorted(paper_ref_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return None

REF_ELECTRODE_PATTERN = re.compile(
    r"(?P<ref>"
    r"Ag\s*[/|]\s*AgCl\s*\([^)]*\)|Ag\s*[/|]\s*AgCl"
    r"|\bSCE\b|saturated\s+calomel"
    r"|Hg\s*[/|]\s*HgO"
    r"|Hg\s*[/|]\s*Hg[2₂]SO[4₄]"
    r"|\bRHE\b|reversible\s+hydrogen"
    r"|\bSHE\b|\bNHE\b|standard\s+hydrogen|normal\s+hydrogen"
    r"|\bMSE\b|mercury\s*[\-/]?\s*sulfate"
    r")",
    re.IGNORECASE,
)

 

ECORR_PATTERN = re.compile(

    r"(?:E\s*corr|corrosion\s+potential|open[\s\-]*circuit\s+potential|OCP)"

    r"[^\.\n]{0,80}?"

    r"(?P<value>" + _NUM + r")"

    r"\s*"

    r"(?P<unit>mV|V)",

    re.IGNORECASE

)

 

ICORR_PATTERN = re.compile(
    r"(?:\b[ij]\s*_?\s*corr\b|corrosion\s+current(?:\s+density)?)"
    r"[^\.\n]{0,90}?"
    r"(?P<value>" + _NUM + r")"
    r"\s*"
    r"(?P<unit>(?:n|µ|μ|u|m)?A\s*[/·⋅]?\s*(?:cm|mm|m)\s*[\-−–]?\s*2)",
    re.IGNORECASE,
)

RP_PATTERN = re.compile(
    r"(?:\bR\s*_?\s*p\b|polarization\s+resistance)"
    r"[^\.\n]{0,80}?"
    r"(?P<value>" + _NUM + r")"
    r"\s*"
    r"(?P<unit>(?:k|M)?(?:Ω|ohm)\s*[·⋅/]?\s*(?:cm|mm)\s*(?:2|[\-−–]\s*2))",
    re.IGNORECASE,
)

CR_PATTERN = re.compile(
    r"(?:corrosion\s+rate|\bCR\b|dissolution\s+rate|etch\s+rate)"
    r"[^\.\n]{0,80}?"
    r"(?P<value>" + _NUM + r")"
    r"\s*"
    r"(?P<unit>nm\s*/?\s*min|mm\s*/?\s*(?:yr|year|y|a)|(?:µ|μ|u)m\s*/?\s*(?:hr|h|hour)|mpy|mg\s*/?\s*cm2\s*/?\s*(?:hr|h))",
    re.IGNORECASE,
)

SCAN_RATE_PATTERN = re.compile(

    r"(?:scan\s*rate|sweep\s*rate)"

    r"[^\.\n]{0,60}?"

    r"(?P<value>\d+(?:\.\d+)?)"

    r"\s*"

    r"(?P<unit>mV\s*(?:s[\-−–]1|/\s*s)|V\s*(?:s[\-−–]1|/\s*s))",

    re.IGNORECASE

)

 

IMMERSION_TIME_PATTERN = re.compile(

    r"(?:immers(?:ion|ed)|expos(?:ure|ed)|soak(?:ed|ing)?|cleaning\s+time"

    r"|dip(?:ped|ping)?)"

    r"[^\.\n]{0,60}?"

    r"(?P<value>\d+(?:\.\d+)?)"

    r"\s*"

    r"(?P<unit>s(?:ec(?:ond)?s?)?|min(?:ute)?s?|h(?:(?:ou)?rs?)?|days?)",

    re.IGNORECASE

)

 

# --------------------------------------------------------------------
# 材料辨識（修正版）
#   舊版用 re.IGNORECASE 且沒有單字邊界，導致 "time"→Ti、"et al."→Al、
#   "data"→Ta、"corrosion"→Co 等大量誤判。新版：
#   - 元素符號 / 化學式：區分大小寫，前面不可接字母、後面不可接小寫字母
#   - 英文全名：不分大小寫 + \b 單字邊界
# --------------------------------------------------------------------

from materials_loader import build_text_recognition_tables

_BASE_SYMBOLS, _NAME_TO_SYMBOL, _MATERIAL_ALIASES = build_text_recognition_tables()
# 目標金屬一定要能被辨識（例如未來切換成其他金屬時）
_NAME_TO_SYMBOL.setdefault(TARGET_METAL.lower(), METAL_SYMBOL)

_ALL_SYMBOLS = sorted(
    set(_BASE_SYMBOLS) | set(KNOWN_MATERIALS.keys()) | {METAL_SYMBOL, "WN", "MoN"},
    key=len, reverse=True,          # 長的先比對：TiN / MoN 先於 Ti / Mo
)

SYMBOL_PATTERN = re.compile(
    r"(?<![A-Za-z])(?P<sym>" + "|".join(map(re.escape, _ALL_SYMBOLS)) + r")(?![a-z])"
)
NAME_PATTERN = re.compile(
    r"\b(?P<name>" + "|".join(
        map(re.escape, sorted(_NAME_TO_SYMBOL, key=len, reverse=True))
    ) + r")\b",
    re.IGNORECASE,
)
TARGET_PATTERN = build_target_regex()

# 參考電極 / 對電極造成的雜訊：Ag/AgCl、Pt counter electrode、Pt wire ...
_AUX_ELECTRODE_AFTER = re.compile(
    r"^\s*(?:counter|auxiliary|reference|wire|mesh|gauze|plate\s+counter)",
    re.IGNORECASE,
)


def _is_noise(sym: str, text: str, start: int, end: int) -> bool:
    after = text[end:end + 30]
    before = text[max(0, start - 3):start]
    if sym == "W":
        # 作者縮寫：Luc, W.、C. W. Lee
        if re.match(r"\s*\.(?![a-z0-9])", after):
            return True
        # 功率單位：100 W、50W；熱傳導 W/mK
        if re.search(r"\d\s*$", text[max(0, start - 4):start]) or after.startswith("/"):
            return True
    if sym == "Ag" and (after.startswith("/AgCl") or after.startswith("Cl")
                        or before.endswith("Ag/")):
        return True                      # Ag/AgCl 參考電極
    if sym in ("Ag", "Pt", "Au") and _AUX_ELECTRODE_AFTER.match(after):
        return True                      # Pt counter electrode、Pt wire ...
    return False


def find_materials(text: str) -> list[tuple[str, str]]:
    """回傳 [(原始字串, 標準化符號), ...]，依出現順序。"""
    found = []
    for m in NAME_PATTERN.finditer(text):
        raw = m.group("name")
        found.append((m.start(), -len(raw), 0, raw, _NAME_TO_SYMBOL[raw.lower()]))
    for m in SYMBOL_PATTERN.finditer(text):
        sym = m.group("sym")
        if not _is_noise(sym, text, m.start(), m.end()):
            found.append((m.start(), -len(sym), 0, sym, sym))
    # 目標金屬的化合物（config.COMPOUNDS，例如 copper foam、Cu3P、W2N thin film）
    for m in TARGET_PATTERN.finditer(text):
        raw = m.group(0)
        found.append((m.start(), -len(raw), 1, raw, METAL_SYMBOL))
    # 同一位置可能被多個 pattern 命中 → 優先取匹配較長且為精確材料符號/名稱者（例如 TiN 優先於 Ti）
    seen, out = set(), []
    for pos, _neg_len, _prio, raw, sym in sorted(found):
        if pos in seen:
            continue
        seen.add(pos)
        out.append((raw, sym))
    return out


def count_materials(text: str) -> Counter:
    return Counter(sym for _, sym in find_materials(text))


def assign_material(contexts: list[str], paper_counts: Counter) -> tuple[str, str]:
    """
    決定一筆數據屬於哪個材料，回傳 (material_id, method)。
      1. 數值前後文（context）出現的金屬；若含目標金屬則優先
      2. 否則用整篇論文出現最多次的金屬
      3. 目標金屬次數與第一名差距 < 20% 時，選目標金屬
    """
    ctx_counts = Counter()
    for ctx in contexts:
        if ctx:
            ctx_counts.update(count_materials(ctx))

    if ctx_counts:
        if METAL_SYMBOL in ctx_counts:
            return METAL_SYMBOL, "context"
        # 只採用在全文中也有一定份量的金屬，避免引用文獻順帶提及
        for sym, _ in ctx_counts.most_common():
            if paper_counts.get(sym, 0) >= MIN_MATERIAL_MENTIONS:
                return sym, "context"

    if paper_counts:
        top_sym, top_n = paper_counts.most_common(1)[0]
        target_n = paper_counts.get(METAL_SYMBOL, 0)
        if top_sym == METAL_SYMBOL:
            return METAL_SYMBOL, "paper_majority"
        if target_n >= 0.8 * top_n:
            return METAL_SYMBOL, "target_default"
        return top_sym, "paper_majority"

    return METAL_SYMBOL, "target_default"


_REF_HEADING = re.compile(
    r"\n\s*(?:References?|REFERENCES?|Bibliography|BIBLIOGRAPHY|Literature\s+Cited)\s*\n"
)


def strip_references(text: str) -> str:
    """去掉文末參考文獻段落（作者縮寫、引用標題會造成材料誤判）。"""
    hits = list(_REF_HEADING.finditer(text))
    if hits and hits[-1].start() > len(text) * 0.4:
        return text[:hits[-1].start()]
    return text


def format_counts(counts: Counter, min_n: int = 1) -> tuple[str, str]:
    """回傳 (依次數排序的材料清單, 'Cu:93; Pt:4' 形式的次數字串)"""
    items = [(s, n) for s, n in counts.most_common() if n >= min_n]
    return ("; ".join(s for s, _ in items),
            "; ".join(f"{s}:{n}" for s, n in items))

METAL_FORM_PATTERN = re.compile(

    r"(?P<form>"

    r"thin\s+film|bulk|foil|electrodeposit(?:ed)?"

    r"|sputter(?:ed)?|PVD|CVD|ALD|evaporat(?:ed|ion)"

    r"|electroplat(?:ed|ing)|plated|rolled"

    r"|nanoparticle|nanowire|foam|mesh|rod|disk|coupon"

    r")",

    re.IGNORECASE

)

 

SURFACE_PREP_PATTERN = re.compile(

    r"(?P<prep>"

    r"as[\-\s]?deposited|as[\-\s]?received"

    r"|polish(?:ed|ing)|CMP|mechanical(?:ly)?\s+polish"

    r"|etch(?:ed)?|acid[\-\s]?clean|pre[\-\s]?clean"

    r"|anneal(?:ed|ing)|heat[\-\s]?treat"

    r"|degreased|ultrasonicated|sonicated"

    r")",

    re.IGNORECASE

)

 

DEAERATION_PATTERN = re.compile(

    r"(?P<deaerated>"

    r"deaerat(?:ed|ion)|N2[\-\s]?(?:purge|saturat|bubbl)"

    r"|Ar[\-\s]?(?:purge|saturat|bubbl)|nitrogen[\-\s]?(?:purge|saturat)"

    r"|argon[\-\s]?(?:purge|saturat)|oxygen[\-\s]?free"

    r"|O2[\-\s]?(?:saturat|purge|bubbl)|aerated"

    r")",

    re.IGNORECASE

)

 

REACTION_PATTERNS = {

    "HER": re.compile(r"hydrogen\s+evolution|HER|\bH2\b\s+evolution", re.I),

    "OER": re.compile(r"oxygen\s+evolution|OER|\bO2\b\s+evolution", re.I),

    "ORR": re.compile(r"oxygen\s+reduction|ORR", re.I),

    "corrosion": re.compile(r"corrosion|anti[\-\s]?corros|corroded", re.I),

    "CO2RR": re.compile(r"CO2\s+reduction|CO₂\s+reduction|CO2RR", re.I),

}

 

MEASUREMENT_METHOD_PATTERN = re.compile(

    r"(?P<method>"

    r"potentiodynamic|potentiostatic|galvanostatic"

    r"|Tafel\s+extrapolation|linear\s+polarization|LPR"

    r"|EIS|impedance|Nyquist"

    r"|cyclic\s+voltamm(?:etry|ogram)|CV"

    r"|chronoamperometry|chronopotentiometry"

    r"|weight\s+loss|mass\s+loss|gravimetric"

    r"|profilometry|XRF|ICP[\-\s]?(?:OES|MS|AES)"

    r"|AFM|SEM|TEM|XPS|ellipsometry"

    r")",

    re.IGNORECASE

)

 

# ====================================================================

# 6. MATERIAL IDENTIFICATION / NORMALISATION

# ====================================================================

 

# _MATERIAL_ALIASES 已由上方 materials_loader.build_text_recognition_tables() 載入

 

def normalise_material(raw: str) -> str | None:

    if not raw:

        return None

    cleaned = raw.strip().lower()

    for word in ["thin film", "foil", "foam", "electrode", "bulk", "pure",

                 "nanoparticle", "nanowire", "np", "nw", "substrate",

                 "coating", "layer", "barrier", "liner", "metallic"]:

        cleaned = cleaned.replace(word, "").strip()

    cleaned = cleaned.strip()

    if cleaned in _MATERIAL_ALIASES:

        return _MATERIAL_ALIASES[cleaned]

    if cleaned.capitalize() in _MATERIAL_ALIASES:

        return _MATERIAL_ALIASES[cleaned.capitalize()]

    for key in KNOWN_MATERIALS:

        if key.lower() == cleaned:

            return key

    return cleaned if cleaned else None

 

# ====================================================================

# 7. SOLUTION CLASSIFICATION HELPERS

# ====================================================================

 

SPECIES_CATEGORIES: dict[str, dict] = {
    "H2O2": {"role": "oxidizer", "field": "oxidizer_type"},
    "H₂O₂": {"role": "oxidizer", "field": "oxidizer_type"},
    "hydrogen peroxide": {"role": "oxidizer", "field": "oxidizer_type"},
    "HNO3": {"role": "oxidizer", "field": "oxidizer_type"},
    "O3": {"role": "oxidizer", "field": "oxidizer_type"},
    "FeCl3": {"role": "oxidizer", "field": "oxidizer_type"},
    "Fe(NO3)3": {"role": "oxidizer", "field": "oxidizer_type"},
    "KIO4": {"role": "oxidizer", "field": "oxidizer_type"},
    "KIO3": {"role": "oxidizer", "field": "oxidizer_type"},
    "NaClO": {"role": "oxidizer", "field": "oxidizer_type"},
    "K2S2O8": {"role": "oxidizer", "field": "oxidizer_type"},
    "Na2S2O8": {"role": "oxidizer", "field": "oxidizer_type"},
    "(NH4)2S2O8": {"role": "oxidizer", "field": "oxidizer_type"},
    "HCl": {"role": "acid", "field": "acid_type"},
    "H2SO4": {"role": "acid", "field": "acid_type"},
    "H₂SO₄": {"role": "acid", "field": "acid_type"},
    "HClO4": {"role": "acid", "field": "acid_type"},
    "H3PO4": {"role": "acid", "field": "acid_type"},
    "HF": {"role": "acid", "field": "acid_type"},
    "hydrofluoric acid": {"role": "acid", "field": "acid_type"},
    "citric acid": {"role": "chelating_agent", "field": "chelating_agent_type"},
    "oxalic acid": {"role": "chelating_agent", "field": "chelating_agent_type"},
    "malic acid": {"role": "chelating_agent", "field": "chelating_agent_type"},
    "tartaric acid": {"role": "chelating_agent", "field": "chelating_agent_type"},
    "acetic acid": {"role": "acid", "field": "acid_type"},
    "KOH": {"role": "base", "field": "base_type"},
    "NaOH": {"role": "base", "field": "base_type"},
    "NH4OH": {"role": "base", "field": "base_type"},
    "NH₄OH": {"role": "base", "field": "base_type"},
    "ammonium hydroxide": {"role": "base", "field": "base_type"},
    "TMAH": {"role": "base", "field": "base_type"},
    "tetramethylammonium hydroxide": {"role": "base", "field": "base_type"},
    "LiOH": {"role": "base", "field": "base_type"},
    "NaCl": {"role": "electrolyte_additive", "field": "electrolyte_additive"},
    "KCl": {"role": "electrolyte_additive", "field": "electrolyte_additive"},
    "Na2SO4": {"role": "electrolyte_additive", "field": "electrolyte_additive"},
    "K2SO4": {"role": "electrolyte_additive", "field": "electrolyte_additive"},
    "NaHCO3": {"role": "electrolyte_additive", "field": "electrolyte_additive"},
    "KHCO3": {"role": "electrolyte_additive", "field": "electrolyte_additive"},
    "Na2CO3": {"role": "electrolyte_additive", "field": "electrolyte_additive"},
    "K2CO3": {"role": "electrolyte_additive", "field": "electrolyte_additive"},
    "NaClO4": {"role": "electrolyte_additive", "field": "electrolyte_additive"},
    "KNO3": {"role": "electrolyte_additive", "field": "electrolyte_additive"},
    "NaNO3": {"role": "electrolyte_additive", "field": "electrolyte_additive"},
    "Na2HPO4": {"role": "electrolyte_additive", "field": "electrolyte_additive"},
    "KH2PO4": {"role": "electrolyte_additive", "field": "electrolyte_additive"},
    "K2HPO4": {"role": "electrolyte_additive", "field": "electrolyte_additive"},
    "NaH2PO4": {"role": "electrolyte_additive", "field": "electrolyte_additive"},
    "Na2B4O7": {"role": "electrolyte_additive", "field": "electrolyte_additive"},
    "K3Fe(CN)6": {"role": "electrolyte_additive", "field": "electrolyte_additive"},
    "K4Fe(CN)6": {"role": "electrolyte_additive", "field": "electrolyte_additive"},
    "CuSO4": {"role": "electrolyte_additive", "field": "electrolyte_additive"},
    "CuCl2": {"role": "electrolyte_additive", "field": "electrolyte_additive"},
    "phosphate buffer": {"role": "electrolyte_additive", "field": "electrolyte_additive"},
    "PBS": {"role": "electrolyte_additive", "field": "electrolyte_additive"},
    "borate buffer": {"role": "electrolyte_additive", "field": "electrolyte_additive"},
    "EDTA": {"role": "chelating_agent", "field": "chelating_agent_type"},
    "glycine": {"role": "chelating_agent", "field": "chelating_agent_type"},
    "BTA": {"role": "inhibitor", "field": "inhibitor_type"},
    "benzotriazole": {"role": "inhibitor", "field": "inhibitor_type"},
    "TTA": {"role": "inhibitor", "field": "inhibitor_type"},
    "tolyltriazole": {"role": "inhibitor", "field": "inhibitor_type"},
    "TAZ": {"role": "inhibitor", "field": "inhibitor_type"},
    "1,2,4-triazole": {"role": "inhibitor", "field": "inhibitor_type"},
    "ATA": {"role": "inhibitor", "field": "inhibitor_type"},
    "3-amino-1,2,4-triazole": {"role": "inhibitor", "field": "inhibitor_type"},
    "5-AT": {"role": "inhibitor", "field": "inhibitor_type"},
    "5-aminotetrazole": {"role": "inhibitor", "field": "inhibitor_type"},
    "aminotetrazole": {"role": "inhibitor", "field": "inhibitor_type"},
    "MBI": {"role": "inhibitor", "field": "inhibitor_type"},
    "2-mercaptobenzimidazole": {"role": "inhibitor", "field": "inhibitor_type"},
    "MBT": {"role": "inhibitor", "field": "inhibitor_type"},
    "2-mercaptobenzothiazole": {"role": "inhibitor", "field": "inhibitor_type"},
    "MPT": {"role": "inhibitor", "field": "inhibitor_type"},
    "5-mercapto-1-phenyl-1H-tetrazole": {"role": "inhibitor", "field": "inhibitor_type"},
    "5-mercapto-1-phenyltetrazole": {"role": "inhibitor", "field": "inhibitor_type"},
    "BIA": {"role": "inhibitor", "field": "inhibitor_type"},
    "benzimidazole": {"role": "inhibitor", "field": "inhibitor_type"},
    "imidazole": {"role": "inhibitor", "field": "inhibitor_type"},
    "pyrazole": {"role": "inhibitor", "field": "inhibitor_type"},
    "thiazole": {"role": "inhibitor", "field": "inhibitor_type"},
    "1,3,4-thiadiazole": {"role": "inhibitor", "field": "inhibitor_type"},
    "adenine": {"role": "inhibitor", "field": "inhibitor_type"},
    "guanine": {"role": "inhibitor", "field": "inhibitor_type"},
    "purine": {"role": "inhibitor", "field": "inhibitor_type"},
    "8-HQ": {"role": "inhibitor", "field": "inhibitor_type"},
    "8-hydroxyquinoline": {"role": "inhibitor", "field": "inhibitor_type"},
    "pyridine": {"role": "inhibitor", "field": "inhibitor_type"},
    "thiourea": {"role": "inhibitor", "field": "inhibitor_type"},
    "TU": {"role": "inhibitor", "field": "inhibitor_type"},
    "thiosemicarbazide": {"role": "inhibitor", "field": "inhibitor_type"},
    "urotropine": {"role": "inhibitor", "field": "inhibitor_type"},
    "hexamethylenetetramine": {"role": "inhibitor", "field": "inhibitor_type"},
    "HMTA": {"role": "inhibitor", "field": "inhibitor_type"},
    "MEA": {"role": "inhibitor", "field": "inhibitor_type"},
    "monoethanolamine": {"role": "inhibitor", "field": "inhibitor_type"},
    "cysteine": {"role": "inhibitor", "field": "inhibitor_type"},
    "L-cysteine": {"role": "inhibitor", "field": "inhibitor_type"},
    "methionine": {"role": "inhibitor", "field": "inhibitor_type"},
    "L-methionine": {"role": "inhibitor", "field": "inhibitor_type"},
    "histidine": {"role": "inhibitor", "field": "inhibitor_type"},
    "L-histidine": {"role": "inhibitor", "field": "inhibitor_type"},
    "tryptophan": {"role": "inhibitor", "field": "inhibitor_type"},
    "L-tryptophan": {"role": "inhibitor", "field": "inhibitor_type"},
    "glutamic acid": {"role": "inhibitor", "field": "inhibitor_type"},
    "aspartic acid": {"role": "inhibitor", "field": "inhibitor_type"},
    "alanine": {"role": "inhibitor", "field": "inhibitor_type"},
    "L-alanine": {"role": "inhibitor", "field": "inhibitor_type"},
    "arginine": {"role": "inhibitor", "field": "inhibitor_type"},
    "L-arginine": {"role": "inhibitor", "field": "inhibitor_type"},
    "lysine": {"role": "inhibitor", "field": "inhibitor_type"},
    "L-lysine": {"role": "inhibitor", "field": "inhibitor_type"},
    "SDS": {"role": "inhibitor", "field": "inhibitor_type"},
    "sodium dodecyl sulfate": {"role": "inhibitor", "field": "inhibitor_type"},
    "SDBS": {"role": "inhibitor", "field": "inhibitor_type"},
    "sodium dodecylbenzenesulfonate": {"role": "inhibitor", "field": "inhibitor_type"},
    "CTAB": {"role": "inhibitor", "field": "inhibitor_type"},
    "cetyltrimethylammonium bromide": {"role": "inhibitor", "field": "inhibitor_type"},
    "DTAB": {"role": "inhibitor", "field": "inhibitor_type"},
    "Triton X-100": {"role": "inhibitor", "field": "inhibitor_type"},
    "Tween-80": {"role": "inhibitor", "field": "inhibitor_type"},
    "Tween 80": {"role": "inhibitor", "field": "inhibitor_type"},
    "PEG": {"role": "inhibitor", "field": "inhibitor_type"},
    "polyethylene glycol": {"role": "inhibitor", "field": "inhibitor_type"},
    "PEI": {"role": "inhibitor", "field": "inhibitor_type"},
    "polyethyleneimine": {"role": "inhibitor", "field": "inhibitor_type"},
    "PAA": {"role": "inhibitor", "field": "inhibitor_type"},
    "polyacrylic acid": {"role": "inhibitor", "field": "inhibitor_type"},
    "PVP": {"role": "inhibitor", "field": "inhibitor_type"},
    "polyvinylpyrrolidone": {"role": "inhibitor", "field": "inhibitor_type"},
    "potassium oleate": {"role": "inhibitor", "field": "inhibitor_type"},
    "sodium oleate": {"role": "inhibitor", "field": "inhibitor_type"},
    "sodium benzoate": {"role": "inhibitor", "field": "inhibitor_type"},
    "Na2MoO4": {"role": "inhibitor", "field": "inhibitor_type"},
    "sodium molybdate": {"role": "inhibitor", "field": "inhibitor_type"},
    "Na2SiO3": {"role": "inhibitor", "field": "inhibitor_type"},
    "sodium silicate": {"role": "inhibitor", "field": "inhibitor_type"},
    "Na2WO4": {"role": "inhibitor", "field": "inhibitor_type"},
    "sodium tungstate": {"role": "inhibitor", "field": "inhibitor_type"},
    "CeCl3": {"role": "inhibitor", "field": "inhibitor_type"},
    "Ce(NO3)3": {"role": "inhibitor", "field": "inhibitor_type"},
    "K2Cr2O7": {"role": "inhibitor", "field": "inhibitor_type"},
    "NaNO2": {"role": "inhibitor", "field": "inhibitor_type"},
    "sodium nitrite": {"role": "inhibitor", "field": "inhibitor_type"},
    "IPA": {"role": "organic_solvent", "field": "organic_solvent_type"},
    "isopropanol": {"role": "organic_solvent", "field": "organic_solvent_type"},
    "DMSO": {"role": "organic_solvent", "field": "organic_solvent_type"},
    "NMP": {"role": "organic_solvent", "field": "organic_solvent_type"},
    "EG": {"role": "organic_solvent", "field": "organic_solvent_type"},
    "ethylene glycol": {"role": "organic_solvent", "field": "organic_solvent_type"},
}

_SPECIES_CANONICAL: dict[str, str] = {
    "h₂so₄": "H2SO4", "h2so4": "H2SO4",
    "h₂o₂": "H2O2", "h2o2": "H2O2", "hydrogen peroxide": "H2O2",
    "nh₄oh": "NH4OH", "nh4oh": "NH4OH", "ammonium hydroxide": "NH4OH",
    "hydrofluoric acid": "HF", "hf": "HF",
    "tetramethylammonium hydroxide": "TMAH", "tmah": "TMAH",
    "phosphate buffer": "PBS", "pbs": "PBS",
    "ethylene glycol": "EG", "eg": "EG",
    "isopropanol": "IPA", "ipa": "IPA",
    "benzotriazole": "BTA", "bta": "BTA",
    "tolyltriazole": "TTA", "tta": "TTA",
    "1,2,4-triazole": "TAZ", "taz": "TAZ",
    "3-amino-1,2,4-triazole": "ATA", "ata": "ATA",
    "5-aminotetrazole": "5-AT", "5-at": "5-AT",
    "2-mercaptobenzimidazole": "MBI", "mbi": "MBI",
    "2-mercaptobenzothiazole": "MBT", "mbt": "MBT",
    "5-mercapto-1-phenyl-1h-tetrazole": "MPT",
    "5-mercapto-1-phenyltetrazole": "MPT", "mpt": "MPT",
    "benzimidazole": "BIA", "bia": "BIA",
    "8-hydroxyquinoline": "8-HQ", "8-hq": "8-HQ",
    "thiourea": "thiourea", "tu": "thiourea",
    "hexamethylenetetramine": "HMTA", "urotropine": "HMTA", "hmta": "HMTA",
    "monoethanolamine": "MEA", "mea": "MEA",
    "l-cysteine": "cysteine", "cysteine": "cysteine",
    "l-methionine": "methionine", "methionine": "methionine",
    "l-histidine": "histidine", "histidine": "histidine",
    "l-tryptophan": "tryptophan", "tryptophan": "tryptophan",
    "l-alanine": "alanine", "alanine": "alanine",
    "l-arginine": "arginine", "arginine": "arginine",
    "l-lysine": "lysine", "lysine": "lysine",
    "sodium dodecyl sulfate": "SDS", "sds": "SDS",
    "sodium dodecylbenzenesulfonate": "SDBS", "sdbs": "SDBS",
    "cetyltrimethylammonium bromide": "CTAB", "ctab": "CTAB",
    "tween 80": "Tween-80", "tween-80": "Tween-80",
    "polyethylene glycol": "PEG", "peg": "PEG",
    "polyethyleneimine": "PEI", "pei": "PEI",
    "polyacrylic acid": "PAA", "paa": "PAA",
    "polyvinylpyrrolidone": "PVP", "pvp": "PVP",
    "sodium molybdate": "Na2MoO4", "na2moo4": "Na2MoO4",
    "sodium silicate": "Na2SiO3", "na2sio3": "Na2SiO3",
    "sodium tungstate": "Na2WO4", "na2wo4": "Na2WO4",
    "sodium nitrite": "NaNO2", "nano2": "NaNO2",
}
_SPECIES_CAT_LOWER = {k.lower(): (k, v) for k, v in SPECIES_CATEGORIES.items()}

_REF_ELEC_BEFORE_RE = re.compile(
    r"(?:Ag\s*[/|]\s*AgCl|SCE\b|calomel|Hg\s*[/|]\s*HgO|Hg\s*[/|]\s*Hg[2₂]SO[4₄])"
    r"[\s(\[\-–,:/inof]{0,20}$",
    re.IGNORECASE,
)
_REF_ELEC_AFTER_RE = re.compile(
    r"^\s*[\)\]]?\s*(?:Ag\s*[/|]\s*AgCl|reference\s+electrode|as\s+(?:a\s+|the\s+)?reference)",
    re.IGNORECASE,
)
_PREP_WASH_NOISE_RE = re.compile(
    r"(?:wash(?:ed|ing)?|rins(?:ed|ing)?|clean(?:ed|ing)?|etch(?:ed|ing)?|digest(?:ed|ing)?|dissolv(?:ed|ing)?)\s+(?:with|in)\s*$"
    r"|ICP[\-\s]?(?:OES|MS|AES)",
    re.IGNORECASE,
)
_TEMP_NOISE_RE = re.compile(
    r"anneal|calcin|heat(?:ed|ing)?|dried|drying|ramp|furnace|oven|sinter|autoclave"
    r"|decompos|pyrolys|sputter|evaporat|melt|boil|thermograv|TGA|DSC|cur(?:ed|ing)",
    re.IGNORECASE,
)
_PH_THEORY_NOISE_RE = re.compile(
    r"computational\s+hydrogen|\bCHE\b|zero\s+potential|Pourbaix|standard\s+hydrogen",
    re.IGNORECASE,
)
_PH_RANGE_AFTER_RE = re.compile(r"^\s*(?:[–\-~]|to)\s*14\b", re.IGNORECASE)

_STRONG_ACIDS = {"HCl", "H2SO4", "HClO4", "HNO3", "HF"}
_STRONG_BASES = {"KOH", "NaOH", "LiOH", "TMAH"}


def normalise_species(raw: str) -> str:
    s = re.sub(r"\s+", " ", raw.strip())
    low = s.lower()
    if low in _SPECIES_CANONICAL:
        return _SPECIES_CANONICAL[low]
    if low in _SPECIES_CAT_LOWER:
        return _SPECIES_CAT_LOWER[low][0]
    return s


def lookup_species_category(species: str) -> dict | None:
    if not species:
        return None
    if species in SPECIES_CATEGORIES:
        return SPECIES_CATEGORIES[species]
    hit = _SPECIES_CAT_LOWER.get(species.lower())
    return hit[1] if hit else None


def extract_electrolytes(text: str) -> list[dict]:
    """擷取電解質並過濾參考電極填充液與清洗/消化液，依出現頻率排序並去重。"""
    counts: Counter = Counter()
    for m in ELECTROLYTE_PATTERN.finditer(text):
        before = text[max(0, m.start() - 45):m.start()]
        after = text[m.end():m.end() + 45]
        if _REF_ELEC_BEFORE_RE.search(before) or _REF_ELEC_AFTER_RE.search(after):
            continue
        if _PREP_WASH_NOISE_RE.search(before):
            continue
        conc = m.group("conc").strip()
        try:
            if float(conc) <= 0:
                continue
        except ValueError:
            pass
        unit = re.sub(r"\s+", " ", m.group("unit").strip())
        species = normalise_species(m.group("species"))
        counts[(conc, unit, species)] += 1
    return [
        {"conc": conc, "unit": unit, "species": species, "count": cnt}
        for (conc, unit, species), cnt in counts.most_common()
    ]


def extract_phs(text: str) -> Counter:
    """擷取實驗 pH 並排除理論定義（CHE / pH 0–14）與引用編號。"""
    counts: Counter = Counter()
    for m in PH_PATTERN.finditer(text):
        try:
            v = float(m.group("value"))
        except ValueError:
            continue
        if not (0.0 <= v <= 14.0):
            continue
        before = text[max(0, m.start() - 50):m.start()]
        after = text[m.end():m.end() + 35]
        if _PH_RANGE_AFTER_RE.match(after):
            continue
        if v == 0.0 and _PH_THEORY_NOISE_RE.search(before + " " + after):
            continue
        counts[round(v, 2)] += 1
    return counts


def extract_temperatures(text: str) -> Counter:
    """擷取溶液實驗溫度（°C），排除 kHz/kV/退火烘乾溫度，並辨識 room temperature。"""
    counts: Counter = Counter()
    for m in TEMP_PATTERN.finditer(text):
        try:
            v = float(m.group("value"))
        except ValueError:
            continue
        u = m.group("unit").strip()
        if ("K" in u or "kelvin" in u.lower()) and "C" not in u and "°" not in u and "℃" not in u:
            v -= 273.15
        if not (0.0 <= v <= 95.0):
            continue
        before = text[max(0, m.start() - 50):m.start()]
        after = text[m.end():m.end() + 35]
        if _TEMP_NOISE_RE.search(before + " " + after):
            continue
        counts[round(v, 2)] += 1
    if not counts and ROOM_TEMP_PATTERN.search(text):
        counts[25.0] = 1
    return counts


def _ph_consistent(ph: float, electrolytes: list[dict]) -> bool:
    species_set = {e["species"] for e in electrolytes}
    has_base = bool(species_set & _STRONG_BASES)
    has_acid = bool(species_set & _STRONG_ACIDS)
    if has_base and not has_acid and ph < 7.0:
        return False
    if has_acid and not has_base and ph > 7.0:
        return False
    return True


def select_environment_for_record(
    contexts: list[str],
    paper_electrolytes: list[dict],
    paper_ph_counts: Counter,
    paper_temp_counts: Counter,
) -> tuple[list[dict], float | None, float | None]:
    """優先從該筆數值的上下文擷取電解質/pH/溫度，找不到時回退到論文主要環境。"""
    ctx_text = " \n ".join(c for c in contexts if c)
    ctx_els = extract_electrolytes(ctx_text) if ctx_text else []
    ctx_phs = extract_phs(ctx_text) if ctx_text else Counter()
    ctx_temps = extract_temperatures(ctx_text) if ctx_text else Counter()

    if ctx_els:
        # 若該列上下文僅標註緩蝕劑/添加劑（例如表格列 "5 mM TTA"），保留論文主環境的基底酸/鹼/電解質鹽
        ctx_roles = {(lookup_species_category(e["species"]) or {}).get("role") for e in ctx_els}
        has_base_medium = bool(ctx_roles & {"acid", "base", "electrolyte_additive"})
        if not has_base_medium and paper_electrolytes:
            ctx_species = {e["species"] for e in ctx_els}
            base_media = [
                e for e in paper_electrolytes
                if e["species"] not in ctx_species
                and (lookup_species_category(e["species"]) or {}).get("role") in {"acid", "base", "electrolyte_additive"}
            ]
            # 取出現頻率最高的基底介質補齊
            chosen_els = ctx_els + base_media[:1]
        else:
            chosen_els = ctx_els
    else:
        chosen_els = paper_electrolytes

    # pH：優先選與電解質酸鹼不矛盾、且出現次數最多的值
    chosen_ph = None
    for ph_counts in (ctx_phs, paper_ph_counts):
        if not ph_counts:
            continue
        consistent = [(p, n) for p, n in ph_counts.most_common() if _ph_consistent(p, chosen_els)]
        if consistent:
            chosen_ph = consistent[0][0]
            break
    if chosen_ph is None:
        if ctx_phs:
            chosen_ph = ctx_phs.most_common(1)[0][0]
        elif paper_ph_counts and not chosen_els:
            chosen_ph = paper_ph_counts.most_common(1)[0][0]

    # 溫度：優先上下文，其次論文中出現最多次（平手時選最接近室溫 25°C）
    chosen_temp = None
    for t_counts in (ctx_temps, paper_temp_counts):
        if t_counts:
            chosen_temp = sorted(
                t_counts.items(),
                key=lambda kv: (-kv[1], abs(kv[0] - 25.0))
            )[0][0]
            break

    return chosen_els, chosen_ph, chosen_temp

def classify_solution_category(ph: float | None,

                               components: list[dict]) -> str:

    if ph is not None:

        if ph < 4:

            return "strongly_acidic"

        elif ph < 7:

            return "acidic"

        elif 7 <= ph <= 7.5:

            return "neutral"

        elif ph < 10:

            return "alkaline"

        else:

            return "strongly_alkaline"

    for comp in components:

        species = comp.get("species", "").lower()

        if species in ["hcl", "h2so4", "hf", "hno3", "hclo4", "h3po4"]:

            return "acidic"

        if species in ["koh", "naoh", "nh4oh", "tmah", "lioh"]:

            return "alkaline"

    return "unknown"

 

def canonicalize_electrolyte(comp: dict) -> tuple[str, str, str]:
    """將電解質濃度與單位正規化為標準形式（如 500 mM -> 0.5 M），用於 solution_id 去重。"""
    raw_conc = str(comp.get("conc", "?")).strip()
    raw_unit = normalize_text(str(comp.get("unit", ""))).strip()
    species = str(comp.get("species", "?")).strip()

    u_low = re.sub(r"\s+", " ", raw_unit).lower().replace("−", "-").replace("–", "-")
    u_compact = u_low.replace(" ", "")

    try:
        val = float(raw_conc)
    except ValueError:
        return raw_conc, raw_unit, species

    if u_compact in {"mm", "mmol/l", "mmoll-1", "mmoldm-3", "mmol/dm3"}:
        val *= 1e-3
        canon_unit = "M"
    elif u_compact in {"um", "µm", "μm", "umol/l", "µmol/l", "μmol/l"}:
        val *= 1e-6
        canon_unit = "M"
    elif u_compact in {"m", "mol/l", "moll-1", "moldm-3", "mol/dm3"}:
        canon_unit = "M"
    elif u_compact == "n":
        canon_unit = "N"
    elif u_compact in {"wt%", "wt.%", "w/w%", "mass%", "weight%"}:
        canon_unit = "wt%"
    elif u_compact in {"vol%", "vol.%", "v/v%"}:
        canon_unit = "vol%"
    elif u_compact in {"g/l", "gl-1", "gdm-3", "g/dm3"}:
        canon_unit = "g/L"
    elif u_compact in {"mg/l", "mgl-1", "ppm"}:
        canon_unit = "mg/L"
    else:
        canon_unit = raw_unit

    canon_conc = f"{val:.6g}"
    return canon_conc, canon_unit, species


def generate_solution_id(components: list[dict], ph: float | None,
                         temp: float | None) -> str:
    canon_comps = sorted({canonicalize_electrolyte(c) for c in components},
                         key=lambda x: (x[2], x[1], x[0]))
    parts = [f"{conc} {unit} {species}".strip() for conc, unit, species in canon_comps]

    if ph is not None:
        parts.append(f"pH{round(float(ph), 2):.6g}")

    if temp is not None:
        parts.append(f"{round(float(temp), 2):.6g}C")

    fingerprint = "|".join(parts) if parts else "unknown"
    short_hash = hashlib.md5(fingerprint.encode()).hexdigest()[:8]
    return f"SOL_{short_hash}"

 

# ====================================================================

# 8. HELPER: get context around a regex match

# ====================================================================

 

def get_context(text: str, match_obj, window: int = None) -> str:  # ✅ 用 config

    if window is None:

        window = CONTEXT_WINDOW

    start = max(0, match_obj.start() - window)

    end = min(len(text), match_obj.end() + window)

    return text[start:end].replace("\n", " ").strip()

 

def detect_reactions(text: str) -> list[str]:

    found = []

    for reaction, pattern in REACTION_PATTERNS.items():

        if pattern.search(text):

            found.append(reaction)

    return found

 

# ====================================================================

# 9. MAIN EXTRACTION FUNCTION

# ====================================================================

 

def extract_from_text(text: str, source_id: str,
                      paper_counts: Counter | None = None) -> list[dict]:

    text = normalize_text(text)
    records = []

    # --- 全文級別資訊（去除文末參考文獻，避免引用論文汙染實驗環境）---
    body_text = strip_references(text) if len(text) >= 600 else text

    electrolytes = extract_electrolytes(body_text)
    ph_counts = extract_phs(body_text)
    temp_counts = extract_temperatures(body_text)
    phs = [p for p, _ in ph_counts.most_common()]
    temps = [t for t, _ in temp_counts.most_common()]

    ref_counts = extract_ref_electrodes(body_text)
    refs = [r for r, _ in sorted(ref_counts.items(), key=lambda kv: (-kv[1], kv[0]))]

 

    ecorr_matches = list(ECORR_PATTERN.finditer(text))

    icorr_matches = list(ICORR_PATTERN.finditer(text))

    tafel_a_matches = list(TAFEL_ANODIC_PATTERN.finditer(text))

    tafel_c_matches = list(TAFEL_CATHODIC_PATTERN.finditer(text))

    tafel_all_matches = list(TAFEL_PATTERN.finditer(text))

    scan_rate_matches = list(SCAN_RATE_PATTERN.finditer(text))

    cr_matches = list(CR_PATTERN.finditer(text))

    rp_matches = list(RP_PATTERN.finditer(text))

    immersion_matches = list(IMMERSION_TIME_PATTERN.finditer(text))

    method_matches = list(MEASUREMENT_METHOD_PATTERN.finditer(text))

 

    material_hits = find_materials(text)

    form_matches = list(METAL_FORM_PATTERN.finditer(text))

    prep_matches = list(SURFACE_PREP_PATTERN.finditer(text))

    deaer_matches = list(DEAERATION_PATTERN.finditer(text))

 

    reactions = detect_reactions(text)

 

    unique_materials_raw = list(OrderedDict.fromkeys(raw for raw, _ in material_hits))[:30]

    if paper_counts is None:
        paper_counts = Counter(sym for _, sym in material_hits)

    materials_sorted, materials_counts_str = format_counts(
        paper_counts, MIN_MATERIAL_MENTIONS
    )
    paper_primary = paper_counts.most_common(1)[0][0] if paper_counts else None

    unique_forms = list({m.group("form").strip() for m in form_matches})

    unique_preps = list({m.group("prep").strip() for m in prep_matches})

    unique_methods = list({m.group("method").strip() for m in method_matches})

 

    is_deaerated = any(

        "deaerat" in m.group("deaerated").lower()

        or "nitrogen" in m.group("deaerated").lower()

        or "argon" in m.group("deaerated").lower()

        or "oxygen-free" in m.group("deaerated").lower()

        or "n2" in m.group("deaerated").lower()

        or "ar" in m.group("deaerated").lower()

        for m in deaer_matches

    ) if deaer_matches else False

 

    is_aerated = any(

        "aerated" in m.group("deaerated").lower()

        and "deaerat" not in m.group("deaerated").lower()

        for m in deaer_matches

    ) if deaer_matches else False

 

    is_o2_saturated = any(

        "o2" in m.group("deaerated").lower()

        and "saturat" in m.group("deaerated").lower()

        for m in deaer_matches

    ) if deaer_matches else False

 

    base_record = {

        "source_id": source_id,

        "materials_raw": "; ".join(unique_materials_raw),

        "materials_normalised": materials_sorted,
        "material_mention_counts": materials_counts_str,
        "paper_primary_material": paper_primary,
        "target_mentions": paper_counts.get(METAL_SYMBOL, 0),

        "metal_form": "; ".join(unique_forms) if unique_forms else None,

        "surface_preparation": "; ".join(unique_preps) if unique_preps else None,

        "electrolytes_raw": "; ".join(
            f"{e['conc']} {e['unit']} {e['species']}" for e in electrolytes
        ),
        "ph_values": "; ".join(str(p) for p in phs) if phs else None,
        "temperature_C_values": "; ".join(str(t) for t in temps) if temps else None,
        "reference_electrodes": "; ".join(refs) if refs else None,

        "deaerated": is_deaerated,

        "aerated": is_aerated,

        "o2_saturated": is_o2_saturated,

        "measurement_methods": "; ".join(unique_methods) if unique_methods else None,

        "scan_rates_raw": "; ".join(

            f"{m.group('value')} {m.group('unit')}" for m in scan_rate_matches

        ) if scan_rate_matches else None,

        "reactions": "; ".join(reactions) if reactions else None,

        "immersion_times_raw": "; ".join(

            f"{m.group('value')} {m.group('unit')}" for m in immersion_matches

        ) if immersion_matches else None,

        "extraction_method": "regex",

    }

 

    # --- Ecorr ---

    ecorr_values = []

    for m in ecorr_matches:

        raw_v = parse_scientific(m.group("value"))

        unit = m.group("unit")

        if raw_v is not None:

            v, warn = normalize_ecorr_to_v(raw_v, unit)

            ecorr_values.append({

                "raw": m.group("value"),

                "unit": unit,

                "value_V": v,
                "warn": warn,

                "context": get_context(text, m, 100),
                "context_wide": get_context(text, m),

            })

 

    # --- icorr ---

    icorr_values = []

    for m in icorr_matches:

        raw_v = parse_scientific(m.group("value"))

        unit = m.group("unit")

        if raw_v is not None:

            v = normalize_icorr_to_uA_cm2(raw_v, unit)

            if v is None:
                continue

            icorr_values.append({

                "raw": m.group("value"),

                "unit": unit,

                "value_uA_cm2": v,

                "context": get_context(text, m, 100),
                "context_wide": get_context(text, m),

            })

    # 支援內文中的 log(icorr) / lg(icorr) 表達式（例如 log(icorr) = -5.2 A/cm2）
    for m in re.finditer(
        r"\b(?:log|lg)\s*\(?\s*[ij]\s*_?\s*corr\s*\)?\s*(?:=|of|is|was|:)\s*"
        r"(?P<value>[\-−–]?\d+(?:\.\d+)?)\s*(?P<unit>(?:[µμunm]?A\s*(?:/|·|\s)\s*(?:cm|mm|m)\s*(?:\^?[-−]?2|²))?)",
        text,
        re.I,
    ):
        raw_v = parse_scientific(m.group("value"))
        if raw_v is None:
            continue
        unit = (m.group("unit") or "").strip()
        v = convert_log_icorr_to_uA_cm2(raw_v, unit)
        if v is not None:
            icorr_values.append({
                "raw": m.group("value"),
                "unit": f"log({unit or 'A/cm2'})",
                "value_uA_cm2": v,
                "context": get_context(text, m, 100),
                "context_wide": get_context(text, m),
            })

 

    # --- Tafel slopes ---

    beta_a_values = []

    for m in tafel_a_matches:

        raw_v = parse_scientific(m.group("value"))

        if raw_v is not None:

            if "V" in m.group("unit") and "mV" not in m.group("unit"):

                raw_v *= 1000

            beta_a_values.append(abs(raw_v))

 

    beta_c_values = []

    for m in tafel_c_matches:

        raw_v = parse_scientific(m.group("value"))

        if raw_v is not None:

            if "V" in m.group("unit") and "mV" not in m.group("unit"):

                raw_v *= 1000

            beta_c_values.append(abs(raw_v))

 

    cr_values = []

    for m in cr_matches:

        raw_v = parse_scientific(m.group("value"))

        if raw_v is not None:

            cr_values.append({

                "raw": m.group("value"),

                "unit": m.group("unit"),

                "value": raw_v,

            })

 

    rp_values = []

    for m in rp_matches:

        raw_v = parse_scientific(m.group("value"))

        if raw_v is not None:

            rp_values.append({

                "raw": m.group("value"),

                "unit": m.group("unit"),

                "value": raw_v,

            })

 

    tafel_a_ctx = [get_context(text, m) for m in tafel_a_matches]
    tafel_c_ctx = [get_context(text, m) for m in tafel_c_matches]
    tafel_all_ctx = [get_context(text, m) for m in tafel_all_matches]

    # --- 建立 records ---

    n_records = max(1, len(ecorr_values), len(icorr_values))

 

    for i in range(n_records):

        rec = dict(base_record)

        # ---- 依數值上下文判定材料 ----
        ctxs = []
        if i < len(ecorr_values):
            ctxs.append(ecorr_values[i]["context_wide"])
        if i < len(icorr_values):
            ctxs.append(icorr_values[i]["context_wide"])
        for lst in (tafel_a_ctx, tafel_c_ctx):
            if lst:
                ctxs.append(lst[i] if i < len(lst) else lst[0])
        if not ctxs and tafel_all_ctx:
            ctxs.append(tafel_all_ctx[0])
        if len(text) < 600:          # 表格列：整列本身就是上下文
            ctxs.append(text)
        mat_id, mat_method = assign_material(ctxs, paper_counts)
        rec["material_id_assigned"] = mat_id
        rec["material_assignment_method"] = mat_method
        rec["is_target_metal"] = (mat_id == METAL_SYMBOL)

        # ---- 依數值上下文配對實驗環境（電解質 / pH / 溫度）----
        rec_els, rec_ph, rec_temp = select_environment_for_record(
            ctxs, electrolytes, ph_counts, temp_counts
        )
        rec["record_electrolytes"] = rec_els
        rec["electrolytes_raw"] = "; ".join(
            f"{e['conc']} {e['unit']} {e['species']}" for e in rec_els
        )
        rec["record_ph"] = rec_ph
        rec["record_temp"] = rec_temp
        if rec_els or rec_ph is not None:
            rec["solution_id_assigned"] = generate_solution_id(rec_els, rec_ph, rec_temp)
        else:
            rec["solution_id_assigned"] = None

 

        if i < len(ecorr_values):

            ev = ecorr_values[i]

            rec["Ecorr_V_original"] = ev["value_V"]

            rec["Ecorr_unit_original"] = ev["unit"]

            rec["Ecorr_raw"] = ev["raw"]

            if refs:

                rec["Ecorr_V_SHE"] = convert_to_she(

                    ev["value_V"], refs[0],

                    phs[0] if phs else None

                )

            else:

                rec["Ecorr_V_SHE"] = None

        else:

            rec["Ecorr_V_original"] = None

            rec["Ecorr_unit_original"] = None

            rec["Ecorr_raw"] = None

            rec["Ecorr_V_SHE"] = None

 

        if i < len(icorr_values):

            iv = icorr_values[i]

            rec["icorr_uA_cm2"] = iv["value_uA_cm2"]

            rec["icorr_raw"] = iv["raw"]

            rec["icorr_unit_original"] = iv["unit"]

            if iv["value_uA_cm2"] is not None:

                rec["log_icorr"] = np.log10(

                    iv["value_uA_cm2"] * 1e-6

                ) if iv["value_uA_cm2"] > 0 else None

            else:

                rec["log_icorr"] = None

        else:

            rec["icorr_uA_cm2"] = None

            rec["icorr_raw"] = None

            rec["icorr_unit_original"] = None

            rec["log_icorr"] = None

 

        rec["beta_a_mV_dec"] = beta_a_values[i] if i < len(beta_a_values) else (

            beta_a_values[0] if beta_a_values else None

        )

        rec["beta_c_mV_dec"] = beta_c_values[i] if i < len(beta_c_values) else (

            beta_c_values[0] if beta_c_values else None

        )

 

        if rec["beta_a_mV_dec"] is None and rec["beta_c_mV_dec"] is None:

            general_tafel = []

            for m in tafel_all_matches:

                raw_v = parse_scientific(m.group("value"))

                if raw_v is not None:

                    if "V" in m.group("unit") and "mV" not in m.group("unit"):

                        raw_v *= 1000

                    if TAFEL_MIN <= abs(raw_v) <= TAFEL_MAX:            # ✅ 用 config

                        general_tafel.append(abs(raw_v))

            if general_tafel:

                rec["tafel_slopes_general_mV_dec"] = "; ".join(

                    str(v) for v in general_tafel

                )

            else:

                rec["tafel_slopes_general_mV_dec"] = None

        else:

            rec["tafel_slopes_general_mV_dec"] = None

 

        if i < len(cr_values):

            rec["CR_reported_value"] = cr_values[i]["value"]

            rec["CR_reported_unit"] = cr_values[i]["unit"]

        else:

            rec["CR_reported_value"] = cr_values[0]["value"] if cr_values else None

            rec["CR_reported_unit"] = cr_values[0]["unit"] if cr_values else None

 

        if i < len(rp_values):

            rec["Rp_value"] = rp_values[i]["value"]

            rec["Rp_unit"] = rp_values[i]["unit"]

        else:

            rec["Rp_value"] = rp_values[0]["value"] if rp_values else None

            rec["Rp_unit"] = rp_values[0]["unit"] if rp_values else None

 

        has_ecorr = rec["Ecorr_V_original"] is not None

        has_icorr = rec["icorr_uA_cm2"] is not None

        has_tafel = (rec["beta_a_mV_dec"] is not None

                     or rec["beta_c_mV_dec"] is not None

                     or rec["tafel_slopes_general_mV_dec"] is not None)

        has_cr = rec["CR_reported_value"] is not None

 

        if has_ecorr or has_icorr or has_tafel or has_cr:

            records.append(rec)

 

    return records
