"""
Step 5 — Material constants, reference-electrode conversions, and unit normalizers.
"""

import re
import unicodedata

# ====================================================================

# 1. KNOWN MATERIAL PROPERTIES (lookup table)

# ====================================================================

 

from materials_loader import build_known_materials

KNOWN_MATERIALS: dict[str, dict] = build_known_materials()

 

# ====================================================================

# 2. REFERENCE ELECTRODE CONVERSION

# ====================================================================

 

NERNST_SLOPE: float = 0.05916  # 2.303 RT / F at 298.15 K (V/pH)
PURE_MATERIAL_CLASSES: tuple[str, ...] = ("pure", "compound")
TABLE_EXTRACTION_METHODS: tuple[str, ...] = ("table_regex", "table_column_mapped")

REF_ELECTRODE_TO_SHE: dict[str, float] = {

    "SHE": 0.0,

    "standard hydrogen": 0.0,

    "NHE": 0.0,

    "SCE": 0.241,

    "saturated calomel": 0.241,

    "Hg/Hg2Cl2": 0.241,

    "Ag/AgCl": 0.197,

    "Ag/AgCl (3M KCl)": 0.210,

    "Ag/AgCl (3M)": 0.210,

    "Ag/AgCl (sat)": 0.197,

    "Ag/AgCl (sat. KCl)": 0.197,

    "Hg/HgO": 0.098,

    "Hg/Hg2SO4": 0.640,

    "MSE": 0.640,

    "SMSE": 0.640,

    "MMS": 0.640,

    "mercury sulfate": 0.640,

    "mercury-sulfate": 0.640,

    "Cu/CuSO4": 0.314,

    "CSE": 0.314,

    "RHE": None,

    "reversible hydrogen": None,

}

 

def convert_to_she(value_v: float, ref_electrode: str,

                   ph: float | None = None) -> float | None:

    if value_v is None:

        return None

    ref_clean = ref_electrode.strip()

    if ref_clean in REF_ELECTRODE_TO_SHE:

        offset = REF_ELECTRODE_TO_SHE[ref_clean]

        if offset is not None:

            return value_v + offset

        if ph is not None:

            return value_v - NERNST_SLOPE * ph

        return None

    ref_lower = ref_clean.lower()

    for key, offset in REF_ELECTRODE_TO_SHE.items():

        if key.lower() in ref_lower or ref_lower in key.lower():

            if offset is not None:

                return value_v + offset

            if ("rhe" in ref_lower or "reversible hydrogen" in ref_lower) and ph is not None:

                return value_v - NERNST_SLOPE * ph

            return None

    return None

 

# ====================================================================

# 3. UNIT CONVERSION HELPERS

# ====================================================================

 

def normalize_text(text: str) -> str:
    """將 PDF 數學斜體（𝑖corr、𝜇A、𝛽a）、上標數字（⁻²、⁻¹）與連字正規化。"""
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text)


def parse_scientific(raw: str) -> float | None:
    if raw is None:
        return None
    s = normalize_text(str(raw)).strip()
    if not s or s in {"-", "—", "–", "N/A", "n/a", "ND", "nd"}:
        return None
    # 去掉 ± 誤差後綴（例如 "181.87 ± 2.1" -> "181.87"）
    s = re.split(r"\s*±\s*", s)[0].strip()
    # 去掉括號備註（例如 "39.04 (8)" -> "39.04"）
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()
    # 統一負號並移除負號後的空白（例如 "− 652.46" -> "-652.46"）
    s = s.replace("−", "-").replace("–", "-").replace("—", "-").replace(",", "")
    s = re.sub(r"^-\s+", "-", s)

    try:
        return float(s)
    except ValueError:
        pass

    m = re.match(
        r"([\-]?\d+(?:\.\d+)?)\s*[×xX*·⋅]\s*10\s*\^?\s*([\-]?\s*\d+)",
        s,
    )
    if m:
        exp = re.sub(r"\s+", "", m.group(2))
        try:
            return float(m.group(1)) * (10 ** float(exp))
        except (ValueError, OverflowError):
            return None

    return None


_UNIT_SCALE_RE = re.compile(r"(?:[×xX*·⋅]\s*)?10\s*\^?\s*([+\-]?\s*\d+)")
_PURE_AMP_RE = re.compile(r"(?:^|[^a-z])a(?:/|cm|mm|m|$)")


def normalize_icorr_to_uA_cm2(value: float, unit: str) -> float | None:
    if value is None or value <= 0:
        return None
    u = normalize_text(unit or "").lower()
    u = u.replace("−", "-").replace("–", "-").replace("—", "-")

    # 解析表頭科學記號倍率（例如 "×10-6 A/cm2"、"10^-5 mA cm-2"）
    scale_factor = 1.0
    scale_m = _UNIT_SCALE_RE.search(u)
    if scale_m:
        exp_str = re.sub(r"\s+", "", scale_m.group(1))
        try:
            scale_factor = 10.0 ** float(exp_str)
        except (ValueError, OverflowError):
            return None
        u = u[:scale_m.start()] + " " + u[scale_m.end():]

    u = re.sub(r"[\s·⋅]", "", u)

    # 面積換算：目標為 cm^2
    area_factor = 1.0
    if "mm-2" in u or "/mm2" in u or "mm2" in u:
        area_factor = 100.0       # 1 A/mm2 = 100 A/cm2
    elif "m-2" in u or "/m2" in u:
        if "cm" not in u and "mm" not in u:
            area_factor = 1e-4    # 1 A/m2 = 1e-4 A/cm2

    # 電流換算：目標為 µA
    if "na" in u:
        curr_factor = 1e-3
    elif "µa" in u or "μa" in u or "ua" in u or "microa" in u:
        curr_factor = 1.0
    elif "ma" in u:
        curr_factor = 1000.0
    elif _PURE_AMP_RE.search(u):
        curr_factor = 1e6
    else:
        curr_factor = 1.0

    return value * scale_factor * curr_factor * area_factor


def convert_log_icorr_to_uA_cm2(log_value: float, unit: str = "") -> float | None:
    """將 log10(icorr) 還原為 µA/cm²。未標單位且數值 < -1 時預設原單位為 A/cm²。"""
    if log_value is None or not (-20.0 <= log_value <= 10.0):
        return None
    raw_val = 10.0 ** float(log_value)
    effective_unit = unit.strip() if unit and unit.strip() else ("A/cm2" if log_value < -1.0 else "uA/cm2")
    return normalize_icorr_to_uA_cm2(raw_val, effective_unit)


def normalize_ecorr_to_v(value: float, unit: str) -> tuple[float | None, str | None]:
    """
    將 Ecorr 轉成 V，並對常見表格單位誤植/省略做容錯。
    回傳 (value_in_V, warning_note_or_None)
    """
    if value is None:
        return None, None
    u = normalize_text(unit or "").lower().strip()
    if "mv" in u:
        return value / 1000.0, None
    if re.search(r"\bv\b", u) and "mv" not in u:
        # 若單位寫 V 但數值絕對值 > 10，極可能是 mV 筆誤
        if abs(value) > 10:
            return value / 1000.0, f"Ecorr={value} with unit '{unit}' auto-converted as mV"
        return value, None
    # 單位缺失或明顯誤植（例如論文表格寫成 Ecorr (mA/cm2) 但數值是 -652.46）
    if abs(value) > 50:
        note = f"Ecorr unit '{unit or 'missing'}' inferred as mV"
        return value / 1000.0, note
    return value, (f"Ecorr unit '{unit}' non-standard, assumed V" if u else None)

# ====================================================================

# 4. CORROSION RATE CALCULATOR (Faraday's Law)

# ====================================================================

 

def calc_corrosion_rate(icorr_uA_cm2: float, material_id: str) -> dict:

    props = KNOWN_MATERIALS.get(material_id)

    if props is None or icorr_uA_cm2 is None:

        return {"CR_mm_yr": None, "CR_nm_min": None, "CR_um_hr": None}

    ew = props["equivalent_weight_g_eq"]

    rho = props["density_g_cm3"]

    K = 3.27e-3

    cr_mm_yr = K * icorr_uA_cm2 * ew / rho

    cr_nm_min = cr_mm_yr * 1e6 / (365.25 * 24 * 60)

    cr_um_hr = cr_mm_yr * 1e3 / (365.25 * 24)

    return {

        "CR_mm_yr": round(cr_mm_yr, 6),

        "CR_nm_min": round(cr_nm_min, 4),

        "CR_um_hr": round(cr_um_hr, 6),

    }
