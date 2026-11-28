"""
Step 5 — Regex patterns, environment/material/solution helpers, and text extraction.
"""

import hashlib
import re
from collections import Counter, OrderedDict
from functools import lru_cache

import numpy as np

import config
from step5_constants import (
    KNOWN_MATERIALS,
    convert_log_icorr_to_uA_cm2,
    normalize_ecorr_to_v,
    normalize_icorr_to_uA_cm2,
    normalize_text,
    parse_scientific,
)
from step5_schema import (
    PaperEnvironment,
    RawEchemRecord,
    load_chemical_species_registry,
    normalize_tafel_slope,
    resolve_ecorr_she,
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

# 化學物種統一由 chemical_species.json 衍生（依長度由長到短排列，避免 NaCl 截斷 NaClO4）
_SPECIES_REGISTRY = load_chemical_species_registry()
_ELECTROLYTE_SPECIES = list(_SPECIES_REGISTRY.electrolyte_species)
_SPECIES_REGEX = _SPECIES_REGISTRY.electrolyte_species_regex
SPECIES_CATEGORIES = _SPECIES_REGISTRY.species_categories

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
    compact = re.sub(r"[\s/\\|\-–—ǁ‖:]+", "", low)
    if ("ag" in low and "agcl" in low) or ("silver" in low and "chloride" in low):
        return "Ag/AgCl"
    if (
        "sce" in low
        or "calomel" in low
        or "kalmel" in low
        or "hghg2cl2" in compact
        or "hghgcl" in compact
    ):
        return "SCE"
    if "hghgo" in compact or ("mercury" in low and "oxide" in low):
        return "Hg/HgO"
    if (
        "hg2so4" in compact
        or "hgso4" in compact
        or ("mercur" in low and "sulfate" in low)
        or "smse" in low
        or "mms" in low
        or "sse" in low
        or ("saturated" in low and "sulfate" in low)
    ):
        return "Hg/Hg2SO4"
    if "mse" in low:
        return "MSE"
    if "cuso4" in compact or ("copper" in low and "sulfate" in low) or "cse" in low:
        return "Cu/CuSO4"
    if "rhe" in low or "reversible hydrogen" in low:
        return "RHE"
    if "she" in low or "nhe" in low or "standard hydrogen" in low or "normal hydrogen" in low:
        return "SHE"
    return s


def extract_ref_electrodes(text: str) -> Counter:
    """依出現次數統計參考電極（合併 PDF 斷行連字，確保順序確定性）。"""
    counts: Counter = Counter()
    if not text:
        return counts
    cleaned = re.sub(r"-\s*\n\s*", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    for m in REF_ELECTRODE_PATTERN.finditer(cleaned):
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
    # Ag/AgCl variants: separators (-, /, |, ǁ, ‖, :), reversed AgCl/Ag, full English names
    r"Ag\s*[/|\\\-–—ǁ‖:]\s*AgCl(?:\s*\([^)]*\))?"
    r"|AgCl\s*[/|\\\-–—ǁ‖:]\s*Ag(?:\s*\([^)]*\))?"
    r"|silver[\s/\-–—]+silver[\s\-]+chloride(?:\s+(?:reference\s+)?electrode)?"
    r"|silver\s+chloride\s+(?:reference\s+)?electrode"
    # SCE / Calomel variants: glued units (VSCE, mVSCE), calomel without 'saturated', Hg/Hg2Cl2
    r"|(?:(?<=[\s(/\[,;:])|(?<=mV)|(?<=V)|(?<=vs\.)|(?<=vs)|^)SCE\b"
    r"|saturated\s+(?:calomel|kalmel)(?:\s+(?:reference\s+)?electrode)?"
    r"|(?:calomel|kalmel)\s+(?:reference\s+)?electrode"
    r"|Hg\s*[/|\\\-–—ǁ‖:]\s*Hg[2₂]\s*Cl[2₂]"
    # Hg/HgO variants
    r"|Hg\s*[/|\\\-–—ǁ‖:]\s*HgO|mercury[\s/\-–—]+mercur(?:ic|ous)\s+oxide"
    # Hg/Hg2SO4 / MSE / SMSE / MMS / context-guarded SSE
    r"|Hg\s*[/|\\\-–—ǁ‖:]\s*Hg[2₂]\s*SO[4₄]"
    r"|mercur(?:y|ous)[\s/\-–—]+(?:mercurous\s+)?sulfate"
    r"|saturated\s+(?:mercurous\s+)?sulfate\s+(?:reference\s+)?electrode"
    r"|\b(?:MSE|SMSE|MMS)\b"
    r"|(?:(?<=vs\. )|(?<=vs )|(?<=reference electrode \())SSE\b"
    r"|\bSSE\b(?=\s+(?:reference|electrode))"
    # Cu/CuSO4 / context-guarded CSE
    r"|Cu\s*[/|\\\-–—ǁ‖:]\s*CuSO[4₄]"
    r"|copper[\s/\-–—]+(?:copper\s+)?sulfate\s+(?:reference\s+)?electrode"
    r"|(?:(?<=vs\. )|(?<=vs )|(?<=reference electrode \())CSE\b"
    r"|\bCSE\b(?=\s+(?:reference|electrode))"
    # RHE / SHE / NHE (including glued units like VRHE, VSHE, mVSHE)
    r"|(?:(?<=[\s(/\[,;:])|(?<=mV)|(?<=V)|(?<=vs\.)|(?<=vs)|^)(?:RHE|SHE|NHE)\b"
    r"|(?:reversible|standard|normal)\s+hydrogen(?:\s+(?:reference\s+)?electrode)?"
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

_BASE_SYMBOLS, _BASE_NAME_TO_SYMBOL, _MATERIAL_ALIASES = build_text_recognition_tables()


def _resolve_ctx(ctx: config.MetalRunContext | None) -> config.MetalRunContext:
    return ctx if ctx is not None else config.default_metal_context()


def _name_to_symbol_for(target_metal: str, metal_symbol: str) -> dict[str, str]:
    mapping = dict(_BASE_NAME_TO_SYMBOL)
    mapping.setdefault(target_metal.lower(), metal_symbol)
    return mapping


def _known_symbols_for(metal_symbol: str) -> list[str]:
    return sorted(
        set(_BASE_SYMBOLS) | set(KNOWN_MATERIALS.keys()) | {metal_symbol, "WN", "MoN"},
        key=len,
        reverse=True,
    )


@lru_cache(maxsize=32)
def _compile_material_patterns(
    target_metal: str,
    metal_symbol: str,
) -> tuple[dict[str, str], re.Pattern, re.Pattern]:
    name_to_symbol = _name_to_symbol_for(target_metal, metal_symbol)
    all_symbols = _known_symbols_for(metal_symbol)
    symbol_pat = re.compile(
        r"(?<![A-Za-z])(?P<sym>" + "|".join(map(re.escape, all_symbols)) + r")(?![a-z])"
    )
    name_pat = re.compile(
        r"\b(?P<name>" + "|".join(
            map(re.escape, sorted(name_to_symbol, key=len, reverse=True))
        ) + r")\b",
        re.IGNORECASE | re.ASCII,
    )
    return name_to_symbol, symbol_pat, name_pat


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


def find_materials(
    text: str,
    *,
    ctx: config.MetalRunContext | None = None,
) -> list[tuple[str, str]]:
    """回傳 [(原始字串, 標準化符號), ...]，依出現順序。"""
    resolved_ctx = _resolve_ctx(ctx)
    name_to_symbol, symbol_pat, name_pat = _compile_material_patterns(
        resolved_ctx.target_metal,
        resolved_ctx.metal_symbol,
    )
    found = []
    for m in name_pat.finditer(text):
        raw = m.group("name")
        sym = name_to_symbol.get(raw.lower())
        if sym:
            found.append((m.start(), -len(raw), 0, raw, sym))
    for m in symbol_pat.finditer(text):
        sym = m.group("sym")
        if not _is_noise(sym, text, m.start(), m.end()):
            found.append((m.start(), -len(sym), 0, sym, sym))
    # 目標金屬的化合物（ctx.compounds，例如 copper foam、Cu3P、W2N thin film）
    for m in resolved_ctx.target_regex.finditer(text):
        raw = m.group(0)
        found.append((m.start(), -len(raw), 1, raw, resolved_ctx.metal_symbol))
    # 同一位置可能被多個 pattern 命中 → 優先取匹配較長且為精確材料符號/名稱者（例如 TiN 優先於 Ti）
    seen, out = set(), []
    for pos, _neg_len, _prio, raw, sym in sorted(found):
        if pos in seen:
            continue
        seen.add(pos)
        out.append((raw, sym))
    return out


def count_materials(
    text: str,
    *,
    ctx: config.MetalRunContext | None = None,
) -> Counter:
    return Counter(sym for _, sym in find_materials(text, ctx=ctx))


def assign_material(
    contexts: list[str],
    paper_counts: Counter,
    *,
    ctx: config.MetalRunContext | None = None,
) -> tuple[str, str]:
    """
    決定一筆數據屬於哪個材料，回傳 (material_id, method)。
      1. 數值前後文（context）出現的金屬；若含目標金屬則優先
      2. 否則用整篇論文出現最多次的金屬
      3. 目標金屬次數與第一名差距 < 20% 時，選目標金屬
    """
    resolved_ctx = _resolve_ctx(ctx)
    metal_symbol = resolved_ctx.metal_symbol
    min_mentions = resolved_ctx.settings.min_material_mentions

    ctx_counts = Counter()
    for context_str in contexts:
        if context_str:
            ctx_counts.update(count_materials(context_str, ctx=resolved_ctx))

    if ctx_counts:
        if metal_symbol in ctx_counts:
            return metal_symbol, "context"
        # 只採用在全文中也有一定份量的金屬，避免引用文獻順帶提及
        for sym, _ in ctx_counts.most_common():
            if paper_counts.get(sym, 0) >= min_mentions:
                return sym, "context"

    if paper_counts:
        top_sym, top_n = paper_counts.most_common(1)[0]
        target_n = paper_counts.get(metal_symbol, 0)
        if top_sym == metal_symbol:
            return metal_symbol, "paper_majority"
        if target_n >= 0.8 * top_n:
            return metal_symbol, "target_default"
        return top_sym, "paper_majority"

    return metal_symbol, "target_default"


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
# 7. SOLUTION CLASSIFICATION HELPERS (backed by chemical_species.json)
# ====================================================================

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
    return _SPECIES_REGISTRY.normalise(raw)


def lookup_species_category(species: str) -> dict | None:
    return _SPECIES_REGISTRY.lookup_category(species)


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

 

def get_context(
    text: str,
    match_obj,
    window: int = None,
    *,
    ctx: config.MetalRunContext | None = None,
) -> str:
    if window is None:
        window = _resolve_ctx(ctx).settings.context_window
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

 

def extract_paper_environment(
    norm_text: str,
    body_text: str | None = None,
    *,
    table_captions: dict[int, str] | None = None,
) -> PaperEnvironment:
    """從單篇論文內文一次性擷取共用實驗環境（供內文與表格擷取共享）。"""
    body = (
        body_text
        if body_text is not None
        else (strip_references(norm_text) if len(norm_text) >= 600 else norm_text)
    )
    electrolytes = extract_electrolytes(body)
    ph_counts = extract_phs(body)
    temp_counts = extract_temperatures(body)
    ref_counts = extract_ref_electrodes(body)
    refs = [r for r, _ in sorted(ref_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    reactions = detect_reactions(norm_text)

    form_matches = list(METAL_FORM_PATTERN.finditer(norm_text))
    prep_matches = list(SURFACE_PREP_PATTERN.finditer(norm_text))
    method_matches = list(MEASUREMENT_METHOD_PATTERN.finditer(norm_text))
    scan_rate_matches = list(SCAN_RATE_PATTERN.finditer(norm_text))
    immersion_matches = list(IMMERSION_TIME_PATTERN.finditer(norm_text))
    deaer_matches = list(DEAERATION_PATTERN.finditer(norm_text))

    unique_forms = list(OrderedDict.fromkeys(m.group("form").strip() for m in form_matches))
    unique_preps = list(OrderedDict.fromkeys(m.group("prep").strip() for m in prep_matches))
    unique_methods = list(OrderedDict.fromkeys(m.group("method").strip() for m in method_matches))

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

    return PaperEnvironment(
        electrolytes=electrolytes,
        ph_counts=ph_counts,
        temp_counts=temp_counts,
        ref_counts=ref_counts,
        refs=refs,
        reactions=reactions,
        table_captions=dict(table_captions or {}),
        metal_form="; ".join(unique_forms) if unique_forms else None,
        surface_preparation="; ".join(unique_preps) if unique_preps else None,
        measurement_methods="; ".join(unique_methods) if unique_methods else None,
        scan_rates_raw="; ".join(
            f"{m.group('value')} {m.group('unit')}" for m in scan_rate_matches
        ) if scan_rate_matches else None,
        immersion_times_raw="; ".join(
            f"{m.group('value')} {m.group('unit')}" for m in immersion_matches
        ) if immersion_matches else None,
        deaerated=is_deaerated,
        aerated=is_aerated,
        o2_saturated=is_o2_saturated,
    )


def extract_from_text(
    text: str,
    source_id: str,
    paper_counts: Counter | None = None,
    *,
    ctx: config.MetalRunContext | None = None,
    paper_environment: PaperEnvironment | dict | None = None,
) -> list[dict]:
    resolved_ctx = _resolve_ctx(ctx)
    settings = resolved_ctx.settings
    metal_symbol = resolved_ctx.metal_symbol

    text = normalize_text(text)
    records = []

    env = (
        PaperEnvironment.from_mapping(paper_environment)
        if paper_environment is not None
        else extract_paper_environment(text)
    )
    electrolytes = env.electrolytes
    ph_counts = env.ph_counts
    temp_counts = env.temp_counts
    phs = [p for p, _ in ph_counts.most_common()]
    temps = [t for t, _ in temp_counts.most_common()]
    ref_counts = env.ref_counts
    refs = env.refs

    ecorr_matches = list(ECORR_PATTERN.finditer(text))
    icorr_matches = list(ICORR_PATTERN.finditer(text))
    tafel_a_matches = list(TAFEL_ANODIC_PATTERN.finditer(text))
    tafel_c_matches = list(TAFEL_CATHODIC_PATTERN.finditer(text))
    tafel_all_matches = list(TAFEL_PATTERN.finditer(text))
    cr_matches = list(CR_PATTERN.finditer(text))
    rp_matches = list(RP_PATTERN.finditer(text))

    material_hits = find_materials(text, ctx=resolved_ctx)
    unique_materials_raw = list(OrderedDict.fromkeys(raw for raw, _ in material_hits))[:30]

    if paper_counts is None:
        paper_counts = Counter(sym for _, sym in material_hits)

    materials_sorted, materials_counts_str = format_counts(
        paper_counts, settings.min_material_mentions
    )
    paper_primary = paper_counts.most_common(1)[0][0] if paper_counts else None

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
                "context_wide": get_context(text, m, ctx=resolved_ctx),
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
                "context_wide": get_context(text, m, ctx=resolved_ctx),
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
                "context_wide": get_context(text, m, ctx=resolved_ctx),
            })

    # --- Tafel slopes（與表格共用 normalize_tafel_slope 上下限與備註規則）---
    beta_a_values: list[tuple[float, list[str]]] = []
    for m in tafel_a_matches:
        raw_v = parse_scientific(m.group("value"))
        norm_v, notes = normalize_tafel_slope(
            raw_v,
            m.group("unit"),
            tafel_min=settings.tafel_min,
            tafel_max=settings.tafel_max,
            label="βa",
            allow_above_max=True,
        )
        if norm_v is not None:
            beta_a_values.append((norm_v, notes))

    beta_c_values: list[tuple[float, list[str]]] = []
    for m in tafel_c_matches:
        raw_v = parse_scientific(m.group("value"))
        norm_v, notes = normalize_tafel_slope(
            raw_v,
            m.group("unit"),
            tafel_min=settings.tafel_min,
            tafel_max=settings.tafel_max,
            label="βc",
            allow_above_max=True,
        )
        if norm_v is not None:
            beta_c_values.append((norm_v, notes))

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

    tafel_a_ctx = [get_context(text, m, ctx=resolved_ctx) for m in tafel_a_matches]
    tafel_c_ctx = [get_context(text, m, ctx=resolved_ctx) for m in tafel_c_matches]
    tafel_all_ctx = [get_context(text, m, ctx=resolved_ctx) for m in tafel_all_matches]

    # --- 建立 records ---
    n_records = max(1, len(ecorr_values), len(icorr_values))

    for i in range(n_records):
        rec_notes: list[str] = []

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

        mat_id, mat_method = assign_material(ctxs, paper_counts, ctx=resolved_ctx)

        # ---- 依數值上下文配對實驗環境（電解質 / pH / 溫度 / 參考電極）----
        rec_els, rec_ph, rec_temp = select_environment_for_record(
            ctxs, electrolytes, ph_counts, temp_counts
        )
        sol_id = generate_solution_id(rec_els, rec_ph, rec_temp) if (rec_els or rec_ph is not None) else None

        rec_ref = select_ref_electrode(ctxs, ref_counts)
        ordered_refs = (
            [rec_ref] + [r for r in refs if r != rec_ref]
            if rec_ref
            else list(refs)
        )

        ecorr_v_orig = ecorr_unit = ecorr_raw = ecorr_v_she = None
        if i < len(ecorr_values):
            ev = ecorr_values[i]
            ecorr_v_orig = ev["value_V"]
            ecorr_unit = ev["unit"]
            ecorr_raw = ev["raw"]
            if ev.get("warn"):
                rec_notes.append(ev["warn"])
            ecorr_v_she = resolve_ecorr_she(ecorr_v_orig, rec_ref, rec_ph)

        icorr_uA = icorr_raw = icorr_unit = log_icorr = None
        if i < len(icorr_values):
            iv = icorr_values[i]
            icorr_uA = iv["value_uA_cm2"]
            icorr_raw = iv["raw"]
            icorr_unit = iv["unit"]
            if icorr_uA is not None and icorr_uA > 0:
                log_icorr = np.log10(icorr_uA * 1e-6)

        beta_a_val: float | None = None
        if beta_a_values:
            ba_entry = beta_a_values[i] if i < len(beta_a_values) else beta_a_values[0]
            beta_a_val = ba_entry[0]
            rec_notes.extend(ba_entry[1])

        beta_c_val: float | None = None
        if beta_c_values:
            bc_entry = beta_c_values[i] if i < len(beta_c_values) else beta_c_values[0]
            beta_c_val = bc_entry[0]
            rec_notes.extend(bc_entry[1])

        tafel_general_str: str | None = None
        if beta_a_val is None and beta_c_val is None:
            general_tafel: list[float] = []
            for m in tafel_all_matches:
                raw_v = parse_scientific(m.group("value"))
                norm_v, _ = normalize_tafel_slope(
                    raw_v,
                    m.group("unit"),
                    tafel_min=settings.tafel_min,
                    tafel_max=settings.tafel_max,
                    label="Tafel",
                    allow_above_max=False,
                )
                if norm_v is not None:
                    general_tafel.append(norm_v)
            if general_tafel:
                tafel_general_str = "; ".join(str(v) for v in general_tafel)

        if i < len(cr_values):
            cr_val = cr_values[i]["value"]
            cr_unit = cr_values[i]["unit"]
        elif cr_values:
            cr_val = cr_values[0]["value"]
            cr_unit = cr_values[0]["unit"]
        else:
            cr_val = cr_unit = None

        if i < len(rp_values):
            rp_val = rp_values[i]["value"]
            rp_unit = rp_values[i]["unit"]
        elif rp_values:
            rp_val = rp_values[0]["value"]
            rp_unit = rp_values[0]["unit"]
        else:
            rp_val = rp_unit = None

        has_ecorr = ecorr_v_orig is not None
        has_icorr = icorr_uA is not None
        has_tafel = (
            beta_a_val is not None
            or beta_c_val is not None
            or tafel_general_str is not None
        )
        has_cr = cr_val is not None

        if has_ecorr or has_icorr or has_tafel or has_cr:
            record = RawEchemRecord(
                source_id=source_id,
                materials_raw="; ".join(unique_materials_raw),
                materials_normalised=materials_sorted,
                material_mention_counts=materials_counts_str,
                paper_primary_material=paper_primary,
                target_mentions=paper_counts.get(metal_symbol, 0),
                material_id_assigned=mat_id,
                material_assignment_method=mat_method,
                is_target_metal=(mat_id == metal_symbol),
                metal_form=env.metal_form,
                surface_preparation=env.surface_preparation,
                record_electrolytes=rec_els,
                electrolytes_raw="; ".join(
                    f"{e['conc']} {e['unit']} {e['species']}" for e in rec_els
                ),
                record_ph=rec_ph,
                ph_values="; ".join(str(p) for p in phs) if phs else None,
                record_temp=rec_temp,
                temperature_C_values="; ".join(str(t) for t in temps) if temps else None,
                solution_id_assigned=sol_id,
                reference_electrodes="; ".join(ordered_refs) if ordered_refs else None,
                deaerated=env.deaerated,
                aerated=env.aerated,
                o2_saturated=env.o2_saturated,
                measurement_methods=env.measurement_methods,
                scan_rates_raw=env.scan_rates_raw,
                reactions="; ".join(env.reactions) if env.reactions else None,
                immersion_times_raw=env.immersion_times_raw,
                extraction_method="regex",
                context=ctxs[0] if ctxs else "",
                Ecorr_V_original=ecorr_v_orig,
                Ecorr_unit_original=ecorr_unit,
                Ecorr_raw=ecorr_raw,
                Ecorr_V_SHE=ecorr_v_she,
                icorr_uA_cm2=icorr_uA,
                icorr_raw=icorr_raw,
                icorr_unit_original=icorr_unit,
                log_icorr=log_icorr,
                beta_a_mV_dec=beta_a_val,
                beta_c_mV_dec=beta_c_val,
                tafel_slopes_general_mV_dec=tafel_general_str,
                CR_reported_value=cr_val,
                CR_reported_unit=cr_unit,
                Rp_value=rp_val,
                Rp_unit=rp_unit,
                extra_quality_notes="; ".join(rec_notes) if rec_notes else "",
            )
            records.append(record.to_dict())

    return records
