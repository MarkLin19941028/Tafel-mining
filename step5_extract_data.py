"""

Step 5 (Revised): Extract electrochemical data from raw text

and populate three standardised CSV tables:

  1. {metal}_material_properties.csv

  2. {metal}_solution_properties.csv

  3. {metal}_electrochemical_database.csv

"""

 

import os

import re

import json

import hashlib
import unicodedata

from collections import OrderedDict, Counter

 

import numpy as np

import pandas as pd

from tqdm import tqdm

 

from config import (                                        # ✅ 更新 import

    TEXT_DIR, EXTRACTED_DIR, METAL_SYMBOL,

    STEP5_MATERIAL_OUTPUT, STEP5_SOLUTION_OUTPUT,

    STEP5_ECHEM_OUTPUT, STEP5_RAW_OUTPUT,

    TAFEL_MIN, TAFEL_MAX, CONTEXT_WINDOW,
    TARGET_METAL, COMPOUNDS, KEEP_ONLY_TARGET_METAL, MIN_MATERIAL_MENTIONS,
    FILTER_OUTLIERS, ECORR_VALID_RANGE_V_SHE, ICORR_VALID_RANGE_UA_CM2,
    STEP5_OUTLIERS_OUTPUT, build_target_regex,

)

 

# ====================================================================

# 1. KNOWN MATERIAL PROPERTIES (lookup table)

# ====================================================================

 

KNOWN_MATERIALS: dict[str, dict] = {

    "Cu": {

        "material_name": "Copper",

        "symbol": "Cu",

        "category": "metal",

        "density_g_cm3": 8.96,

        "molar_mass_g_mol": 63.55,

        "valence": 2,

        "equivalent_weight_g_eq": 31.77,

        "standard_potential_V_SHE": 0.342,

        "melting_point_C": 1085,

        "electronegativity_pauling": 1.90,

        "work_function_eV": 4.65,

        "oxide_type": "CuO, Cu2O",

        "oxide_stability": "low",

        "passivation_tendency": 0.4,

        "pourbaix_stable_pH_min": 7.0,

        "pourbaix_stable_pH_max": 12.0,

        "crystal_structure": "FCC",

        "thermal_conductivity_W_mK": 401,

        "electrical_resistivity_ohm_cm": 1.68e-6,

        "common_corrosion_products": "CuCl2, Cu(OH)2, CuO",

    },

    "Co": {

        "material_name": "Cobalt",

        "symbol": "Co",

        "category": "metal",

        "density_g_cm3": 8.90,

        "molar_mass_g_mol": 58.93,

        "valence": 2,

        "equivalent_weight_g_eq": 29.47,

        "standard_potential_V_SHE": -0.280,

        "melting_point_C": 1495,

        "electronegativity_pauling": 1.88,

        "work_function_eV": 5.00,

        "oxide_type": "CoO, Co3O4",

        "oxide_stability": "medium",

        "passivation_tendency": 0.5,

        "pourbaix_stable_pH_min": 9.0,

        "pourbaix_stable_pH_max": 13.0,

        "crystal_structure": "HCP",

        "thermal_conductivity_W_mK": 100,

        "electrical_resistivity_ohm_cm": 6.24e-6,

        "common_corrosion_products": "CoCl2, Co(OH)2",

    },

    "W": {

        "material_name": "Tungsten",

        "symbol": "W",

        "category": "metal",

        "density_g_cm3": 19.25,

        "molar_mass_g_mol": 183.84,

        "valence": 6,

        "equivalent_weight_g_eq": 30.64,

        "standard_potential_V_SHE": -0.090,

        "melting_point_C": 3422,

        "electronegativity_pauling": 2.36,

        "work_function_eV": 4.55,

        "oxide_type": "WO3",

        "oxide_stability": "high",

        "passivation_tendency": 0.7,

        "pourbaix_stable_pH_min": 2.0,

        "pourbaix_stable_pH_max": 4.0,

        "crystal_structure": "BCC",

        "thermal_conductivity_W_mK": 173,

        "electrical_resistivity_ohm_cm": 5.28e-6,

        "common_corrosion_products": "WO3, WO4^2-",

    },

    "Ru": {

        "material_name": "Ruthenium",

        "symbol": "Ru",

        "category": "metal",

        "density_g_cm3": 12.37,

        "molar_mass_g_mol": 101.07,

        "valence": 3,

        "equivalent_weight_g_eq": 33.69,

        "standard_potential_V_SHE": 0.455,

        "melting_point_C": 2334,

        "electronegativity_pauling": 2.20,

        "work_function_eV": 4.71,

        "oxide_type": "RuO2",

        "oxide_stability": "high",

        "passivation_tendency": 0.8,

        "pourbaix_stable_pH_min": 0.0,

        "pourbaix_stable_pH_max": 14.0,

        "crystal_structure": "HCP",

        "thermal_conductivity_W_mK": 117,

        "electrical_resistivity_ohm_cm": 7.10e-6,

        "common_corrosion_products": "RuO4",

    },

    "Ti": {

        "material_name": "Titanium",

        "symbol": "Ti",

        "category": "metal",

        "density_g_cm3": 4.51,

        "molar_mass_g_mol": 47.87,

        "valence": 4,

        "equivalent_weight_g_eq": 11.97,

        "standard_potential_V_SHE": -1.630,

        "melting_point_C": 1668,

        "electronegativity_pauling": 1.54,

        "work_function_eV": 4.33,

        "oxide_type": "TiO2",

        "oxide_stability": "high",

        "passivation_tendency": 0.9,

        "pourbaix_stable_pH_min": 4.0,

        "pourbaix_stable_pH_max": 12.0,

        "crystal_structure": "HCP",

        "thermal_conductivity_W_mK": 21.9,

        "electrical_resistivity_ohm_cm": 4.20e-5,

        "common_corrosion_products": "TiO2, Ti(OH)4",

    },

    "TiN": {

        "material_name": "Titanium Nitride",

        "symbol": "TiN",

        "category": "barrier",

        "density_g_cm3": 5.22,

        "molar_mass_g_mol": 61.87,

        "valence": 3,

        "equivalent_weight_g_eq": 20.62,

        "standard_potential_V_SHE": -0.750,

        "melting_point_C": 2930,

        "electronegativity_pauling": 1.54,

        "work_function_eV": 4.20,

        "oxide_type": "TiO2",

        "oxide_stability": "high",

        "passivation_tendency": 0.85,

        "pourbaix_stable_pH_min": 3.0,

        "pourbaix_stable_pH_max": 11.0,

        "crystal_structure": "NaCl-type",

        "thermal_conductivity_W_mK": 19.2,

        "electrical_resistivity_ohm_cm": 2.50e-5,

        "common_corrosion_products": "TiO2",

    },

    "TaN": {

        "material_name": "Tantalum Nitride",

        "symbol": "TaN",

        "category": "barrier",

        "density_g_cm3": 14.30,

        "molar_mass_g_mol": 194.95,

        "valence": 5,

        "equivalent_weight_g_eq": 38.99,

        "standard_potential_V_SHE": -0.600,

        "melting_point_C": 3090,

        "electronegativity_pauling": 1.50,

        "work_function_eV": 4.25,

        "oxide_type": "Ta2O5",

        "oxide_stability": "high",

        "passivation_tendency": 0.9,

        "pourbaix_stable_pH_min": 2.0,

        "pourbaix_stable_pH_max": 12.0,

        "crystal_structure": "NaCl-type",

        "thermal_conductivity_W_mK": 8.0,

        "electrical_resistivity_ohm_cm": 1.35e-4,

        "common_corrosion_products": "Ta2O5",

    },

    "Al": {

        "material_name": "Aluminum",

        "symbol": "Al",

        "category": "metal",

        "density_g_cm3": 2.70,

        "molar_mass_g_mol": 26.98,

        "valence": 3,

        "equivalent_weight_g_eq": 8.99,

        "standard_potential_V_SHE": -1.660,

        "melting_point_C": 660,

        "electronegativity_pauling": 1.61,

        "work_function_eV": 4.28,

        "oxide_type": "Al2O3",

        "oxide_stability": "high",

        "passivation_tendency": 0.95,

        "pourbaix_stable_pH_min": 4.0,

        "pourbaix_stable_pH_max": 9.0,

        "crystal_structure": "FCC",

        "thermal_conductivity_W_mK": 237,

        "electrical_resistivity_ohm_cm": 2.65e-6,

        "common_corrosion_products": "Al(OH)3, Al2O3",

    },

    "Ta": {

        "material_name": "Tantalum",

        "symbol": "Ta",

        "category": "barrier",

        "density_g_cm3": 16.69,

        "molar_mass_g_mol": 180.95,

        "valence": 5,

        "equivalent_weight_g_eq": 36.19,

        "standard_potential_V_SHE": -0.600,

        "melting_point_C": 3017,

        "electronegativity_pauling": 1.50,

        "work_function_eV": 4.25,

        "oxide_type": "Ta2O5",

        "oxide_stability": "high",

        "passivation_tendency": 0.9,

        "pourbaix_stable_pH_min": 2.0,

        "pourbaix_stable_pH_max": 12.0,

        "crystal_structure": "BCC",

        "thermal_conductivity_W_mK": 57.5,

        "electrical_resistivity_ohm_cm": 1.31e-5,

        "common_corrosion_products": "Ta2O5",

    },

    "Mn": {

        "material_name": "Manganese",

        "symbol": "Mn",

        "category": "metal",

        "density_g_cm3": 7.21,

        "molar_mass_g_mol": 54.94,

        "valence": 2,

        "equivalent_weight_g_eq": 27.47,

        "standard_potential_V_SHE": -1.185,

        "melting_point_C": 1246,

        "electronegativity_pauling": 1.55,

        "work_function_eV": 4.10,

        "oxide_type": "MnO2, Mn3O4",

        "oxide_stability": "medium",

        "passivation_tendency": 0.5,

        "pourbaix_stable_pH_min": 7.0,

        "pourbaix_stable_pH_max": 14.0,

        "crystal_structure": "BCC",

        "thermal_conductivity_W_mK": 7.81,

        "electrical_resistivity_ohm_cm": 1.44e-4,

        "common_corrosion_products": "MnO2, Mn(OH)2",

    },

    "Mo": {

        "material_name": "Molybdenum",

        "symbol": "Mo",

        "category": "metal",

        "density_g_cm3": 10.28,

        "molar_mass_g_mol": 95.95,

        "valence": 6,

        "equivalent_weight_g_eq": 15.99,

        "standard_potential_V_SHE": -0.200,

        "melting_point_C": 2623,

        "electronegativity_pauling": 2.16,

        "work_function_eV": 4.36,

        "oxide_type": "MoO3, MoO2",

        "oxide_stability": "medium",

        "passivation_tendency": 0.6,

        "pourbaix_stable_pH_min": 2.0,

        "pourbaix_stable_pH_max": 6.0,

        "crystal_structure": "BCC",

        "thermal_conductivity_W_mK": 138,

        "electrical_resistivity_ohm_cm": 5.17e-6,

        "common_corrosion_products": "MoO3, MoO4^2-, H2MoO4",

    },

}

 

# ====================================================================

# 2. REFERENCE ELECTRODE CONVERSION

# ====================================================================

 

REF_ELECTRODE_TO_SHE: dict[str, float] = {

    "SHE": 0.0,

    "standard hydrogen": 0.0,

    "NHE": 0.0,

    "SCE": 0.241,

    "saturated calomel": 0.241,

    "Ag/AgCl": 0.197,

    "Ag/AgCl (3M KCl)": 0.210,

    "Ag/AgCl (3M)": 0.210,

    "Ag/AgCl (sat)": 0.197,

    "Ag/AgCl (sat. KCl)": 0.197,

    "Hg/HgO": 0.098,

    "Hg/Hg2SO4": 0.640,

    "MSE": 0.640,

    "mercury sulfate": 0.640,

    "mercury-sulfate": 0.640,

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

            return value_v - 0.059 * ph

        return None

    ref_lower = ref_clean.lower()

    for key, offset in REF_ELECTRODE_TO_SHE.items():

        if key.lower() in ref_lower or ref_lower in key.lower():

            if offset is not None:

                return value_v + offset

            if "rhe" in ref_lower and ph is not None:

                return value_v - 0.059 * ph

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
        exp = m.group(2).replace(" ", "")
        return float(m.group(1)) * (10 ** float(exp))

    return None


def normalize_icorr_to_uA_cm2(value: float, unit: str) -> float | None:
    if value is None:
        return None
    u = normalize_text(unit or "").lower()
    u = re.sub(r"[\s·⋅]", "", u).replace("−", "-").replace("–", "-")

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
    elif "a" in u:
        curr_factor = 1e6
    else:
        curr_factor = 1.0

    return value * curr_factor * area_factor


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
    "tetramethylammonium hydroxide", "ammonium hydroxide", "hydrogen peroxide",
    "hydrofluoric acid", "phosphate buffer", "borate buffer", "ethylene glycol",
    "benzotriazole", "citric acid", "oxalic acid", "acetic acid", "isopropanol",
    "K3Fe(CN)6", "K4Fe(CN)6", "Na2B4O7", "Na2HPO4", "KH2PO4", "K2HPO4", "NaH2PO4",
    "NaHCO3", "KHCO3", "NaClO4", "Na2SO4", "Na2CO3", "K2SO4", "K2CO3",
    "H2SO4", "H₂SO₄", "HClO4", "H3PO4", "NH4OH", "NH₄OH", "CuSO4", "CuCl2",
    "FeCl3", "H2O2", "H₂O₂", "HNO3", "NaOH", "NaCl", "KNO3", "NaNO3",
    "LiOH", "TMAH", "EDTA", "DMSO", "glycine",
    "KOH", "HCl", "KCl", "PBS", "BTA", "MBI", "IPA", "NMP", "HF", "EG",
], key=len, reverse=True)

_SPECIES_REGEX = "|".join(
    re.escape(s).replace(r"\ ", r"\s+") for s in _ELECTROLYTE_SPECIES
)

# 單位中 M / mM 區分大小寫，避免 10 m (公尺/分鐘) 被誤判成 10 M
ELECTROLYTE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9.])(?P<conc>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>mM|M|mol\s*(?:L|l)[\-−–]1|mol\s*/\s*(?:L|l)|(?i:wt\s*%)|%|(?i:ppm)|g\s*/\s*L)"
    r"[\s\n]+"
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

_BASE_SYMBOLS = ["Cu", "Co", "W", "Ru", "Ti", "Ta", "Al", "Mn", "Mo",
                 "Ni", "Fe", "Zn", "Mg", "Sn", "Pb", "Cr", "Pt", "Au", "Ag", "Pd", "Ir"]

_NAME_TO_SYMBOL = {
    "titanium nitride": "TiN", "tantalum nitride": "TaN",
    "tungsten nitride": "WN",
    "copper": "Cu", "cobalt": "Co", "tungsten": "W", "wolfram": "W",
    "ruthenium": "Ru", "titanium": "Ti", "tantalum": "Ta",
    "aluminum": "Al", "aluminium": "Al", "manganese": "Mn",
    "molybdenum": "Mo", "nickel": "Ni", "iron": "Fe", "zinc": "Zn",
    "platinum": "Pt", "gold": "Au", "silver": "Ag", "palladium": "Pd",
    "iridium": "Ir", "magnesium": "Mg", "az31": "Mg", "az91": "Mg",
    "stainless steel": "Fe", "carbon steel": "Fe", "mild steel": "Fe",
    "steel": "Fe", "316l": "Fe", "q235": "Fe", "chromium": "Cr",
    "lead": "Pb",
}
# 目標金屬一定要能被辨識（例如未來切換成其他金屬時）
_NAME_TO_SYMBOL.setdefault(TARGET_METAL.lower(), METAL_SYMBOL)

_ALL_SYMBOLS = sorted(
    set(_BASE_SYMBOLS) | set(KNOWN_MATERIALS.keys()) | {METAL_SYMBOL, "WN"},
    key=len, reverse=True,          # 長的先比對：TiN 先於 Ti
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
        found.append((m.start(), m.group("name"),
                      _NAME_TO_SYMBOL[m.group("name").lower()]))
    for m in SYMBOL_PATTERN.finditer(text):
        sym = m.group("sym")
        if not _is_noise(sym, text, m.start(), m.end()):
            found.append((m.start(), sym, sym))
    # 目標金屬的化合物（config.COMPOUNDS，例如 copper foam、Cu3P）
    for m in TARGET_PATTERN.finditer(text):
        found.append((m.start(), m.group(0), METAL_SYMBOL))
    # 同一位置可能被多個 pattern 命中 → 去重
    seen, out = set(), []
    for pos, raw, sym in sorted(found):
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

 

_MATERIAL_ALIASES: dict[str, str] = {

    "copper": "Cu", "cu": "Cu",

    "cobalt": "Co", "co": "Co",

    "tungsten": "W", "wolfram": "W", "w": "W",

    "ruthenium": "Ru", "ru": "Ru",

    "titanium": "Ti", "ti": "Ti",

    "tantalum": "Ta", "ta": "Ta",

    "aluminum": "Al", "aluminium": "Al", "al": "Al",

    "manganese": "Mn", "mn": "Mn",

    "nickel": "Ni", "ni": "Ni",

    "iron": "Fe", "fe": "Fe",

    "zinc": "Zn", "zn": "Zn",

    "molybdenum": "Mo", "mo": "Mo",

    "titanium nitride": "TiN", "tantalum nitride": "TaN",

    "cu2o": "Cu", "cuo": "Cu", "cu(oh)2": "Cu",

    "coo": "Co", "co3o4": "Co",

    "wo3": "W",

    "ruo2": "Ru",

    "tio2": "Ti",

    "al2o3": "Al",

    "ta2o5": "Ta",

    "moo3": "Mo", "moo2": "Mo",

}

 

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
    "HCl": {"role": "acid", "field": "acid_type"},
    "H2SO4": {"role": "acid", "field": "acid_type"},
    "H₂SO₄": {"role": "acid", "field": "acid_type"},
    "HClO4": {"role": "acid", "field": "acid_type"},
    "H3PO4": {"role": "acid", "field": "acid_type"},
    "HF": {"role": "acid", "field": "acid_type"},
    "hydrofluoric acid": {"role": "acid", "field": "acid_type"},
    "citric acid": {"role": "chelating_agent", "field": "chelating_agent_type"},
    "oxalic acid": {"role": "chelating_agent", "field": "chelating_agent_type"},
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
    "MBI": {"role": "inhibitor", "field": "inhibitor_type"},
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

    chosen_els = ctx_els if ctx_els else paper_electrolytes

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

 

def generate_solution_id(components: list[dict], ph: float | None,

                         temp: float | None) -> str:

    parts = []

    for comp in sorted(components, key=lambda c: c.get("species", "")):

        parts.append(f"{comp.get('conc', '?')}{comp.get('unit', '')} "

                     f"{comp.get('species', '?')}")

    if ph is not None:

        parts.append(f"pH{ph}")

    if temp is not None:

        parts.append(f"{temp}C")

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

            icorr_values.append({

                "raw": m.group("value"),

                "unit": unit,

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

 

# ====================================================================
# 10. TABLE EXTRACTION (Column-Mapped Structured Parser)
# ====================================================================

_COL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("sample", re.compile(r"\b(?:specimen|sample|electrode|alloy|material|coating|inhibitor|catalyst|system|condition|substrate)\b", re.I)),
    ("Ecorr", re.compile(r"(?:e\s*_?\s*corr|corrosion\s+potential|\bocp\b)", re.I)),
    ("icorr", re.compile(r"(?:[ij]\s*_?\s*corr|corrosion\s+current)", re.I)),
    ("beta_a", re.compile(r"(?:β\s*_?\s*a|\bb\s*_?\s*a\b|anodic\s+(?:tafel|slope))", re.I)),
    ("beta_c", re.compile(r"(?:β\s*_?\s*c|\bb\s*_?\s*c\b|cathodic\s+(?:tafel|slope))", re.I)),
    ("tafel", re.compile(r"(?:tafel\s+slope|\btafel\b)", re.I)),
    ("Rp", re.compile(r"(?:\br\s*_?\s*p\b|polarization\s+resistance)", re.I)),
    ("CR", re.compile(r"(?:corrosion\s+rate|\bcr\b|\bv\s*corr\b)", re.I)),
]

_ECHEM_COL_ROLES = {"Ecorr", "icorr", "beta_a", "beta_c", "tafel", "Rp", "CR"}
_UNIT_IN_PARENS_RE = re.compile(r"[\(\[]([^\)\]]+)[\)\]]")


def _classify_table_header(cell: str) -> tuple[str | None, str]:
    c = normalize_text(str(cell or "")).replace("\n", " ").strip()
    if not c:
        return None, ""
    unit_m = _UNIT_IN_PARENS_RE.search(c)
    unit = unit_m.group(1).strip() if unit_m else ""
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
    primary_ref = select_ref_electrode([], ref_counts) or (refs[0] if refs else None)
    paper_reactions = paper_env.get("reactions", [])
    if "corrosion" not in paper_reactions and any(
        r in {"Ecorr", "icorr", "Rp", "CR"} for r, _ in col_map.values()
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
            elif role == "icorr":
                icorr_uA = normalize_icorr_to_uA_cm2(val, unit or "uA/cm2")
                icorr_raw = cell_str
                icorr_unit = unit or "uA/cm2"
                if icorr_uA is not None and icorr_uA > 0:
                    log_icorr = np.log10(icorr_uA * 1e-6)
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

        ctxs = [sample_name or "", row_text]
        mat_id, mat_method = assign_material(ctxs, paper_counts or Counter())
        rec_els, rec_ph, rec_temp = select_environment_for_record(
            ctxs, paper_els, paper_ph_counts, paper_temp_counts
        )
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
            "reference_electrodes": "; ".join(refs) if refs else None,
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

# ====================================================================

# 11. BUILD THREE OUTPUT TABLES

# ====================================================================

 

def build_material_properties(raw_records: list[dict]) -> pd.DataFrame:

    detected_materials = set()

    for rec in raw_records:
        # 只列入「實際被指派到電化學數據」的材料（目標金屬另外保證存在）
        mid = rec.get("material_id_assigned")
        if mid in KNOWN_MATERIALS:
            detected_materials.add(mid)

    rows = []

    for mat_id in sorted(detected_materials):

        props = KNOWN_MATERIALS[mat_id]

        row = {"material_id": mat_id}

        row.update(props)

        row["notes"] = ""

        rows.append(row)

 

    # 確保目標金屬一定存在                                     # ✅ 用 config

    if METAL_SYMBOL not in detected_materials and METAL_SYMBOL in KNOWN_MATERIALS:

        props = KNOWN_MATERIALS[METAL_SYMBOL]

        row = {"material_id": METAL_SYMBOL}

        row.update(props)

        row["notes"] = "target material (not detected in papers)"

        rows.append(row)

 

    df = pd.DataFrame(rows)

    return df

 

def build_solution_properties(raw_records: list[dict]) -> pd.DataFrame:
    solutions: dict[str, dict] = {}
    sol_sources: dict[str, list[str]] = {}
    sol_materials: dict[str, list[str]] = {}

    for rec in raw_records:
        if KEEP_ONLY_TARGET_METAL and not rec.get("is_target_metal", False):
            continue

        components = rec.get("record_electrolytes")
        if components is None:
            # 舊資料格式相容
            electrolytes_raw = rec.get("electrolytes_raw", "")
            components = extract_electrolytes(electrolytes_raw) if electrolytes_raw else []

        ph = rec.get("record_ph")
        if ph is None and rec.get("ph_values"):
            try:
                ph = float(str(rec["ph_values"]).split(";")[0].strip())
            except ValueError:
                pass

        temp = rec.get("record_temp")
        if temp is None and rec.get("temperature_C_values"):
            try:
                temp = float(str(rec["temperature_C_values"]).split(";")[0].strip())
            except ValueError:
                pass

        if not components and ph is None:
            continue

        sol_id = rec.get("solution_id_assigned") or generate_solution_id(components, ph, temp)
        src_id = rec.get("source_id", "")
        mat_id = rec.get("material_id_assigned")

        if sol_id not in solutions:
            sol_name = "; ".join(
                f"{c['conc']} {c['unit']} {c['species']}" for c in components
            ) if components else (f"pH {ph} solution" if ph is not None else "unknown")

            sol_rec = {
                "solution_id": sol_id,
                "solution_name": sol_name,
                "solution_category": classify_solution_category(ph, components),
                "materials": "",
                "is_target_metal": False,
                "base_solvent": "DI water",
                "pH": ph,
                "temperature_C": temp,
                "oxidizer_type": None,
                "oxidizer_concentration": None,
                "oxidizer_concentration_unit": None,
                "acid_type": None,
                "acid_concentration": None,
                "acid_concentration_unit": None,
                "base_type": None,
                "base_concentration": None,
                "base_concentration_unit": None,
                "HF_concentration_wt_pct": None,
                "chelating_agent_type": None,
                "chelating_agent_concentration": None,
                "chelating_agent_concentration_unit": None,
                "inhibitor_type": None,
                "inhibitor_concentration": None,
                "inhibitor_concentration_unit": None,
                "organic_solvent_type": None,
                "organic_solvent_concentration": None,
                "electrolyte_additive": None,
                "electrolyte_additive_concentration": None,
                "electrolyte_additive_concentration_unit": None,
                "dissolved_oxygen_ppm": None,
                "conductivity_mS_cm": None,
                "ionic_strength_mol_L": None,
                "ORP_mV": None,
                "deaerated": rec.get("deaerated", False),
                "source": "",
                "notes": "",
            }

            # components 已依頻率由高到低排序 → 只填入每個角色最主要的第一個物種
            for comp in components:
                species = comp["species"]
                cat_info = lookup_species_category(species)
                if not cat_info:
                    continue
                role = cat_info["role"]
                if role == "oxidizer" and sol_rec["oxidizer_type"] is None:
                    sol_rec["oxidizer_type"] = species
                    sol_rec["oxidizer_concentration"] = comp["conc"]
                    sol_rec["oxidizer_concentration_unit"] = comp["unit"]
                elif role == "acid" and sol_rec["acid_type"] is None:
                    sol_rec["acid_type"] = species
                    sol_rec["acid_concentration"] = comp["conc"]
                    sol_rec["acid_concentration_unit"] = comp["unit"]
                    if species.lower() in ["hf", "hydrofluoric acid"]:
                        try:
                            sol_rec["HF_concentration_wt_pct"] = float(comp["conc"])
                        except ValueError:
                            pass
                elif role == "base" and sol_rec["base_type"] is None:
                    sol_rec["base_type"] = species
                    sol_rec["base_concentration"] = comp["conc"]
                    sol_rec["base_concentration_unit"] = comp["unit"]
                elif role == "chelating_agent" and sol_rec["chelating_agent_type"] is None:
                    sol_rec["chelating_agent_type"] = species
                    sol_rec["chelating_agent_concentration"] = comp["conc"]
                    sol_rec["chelating_agent_concentration_unit"] = comp["unit"]
                elif role == "inhibitor" and sol_rec["inhibitor_type"] is None:
                    sol_rec["inhibitor_type"] = species
                    sol_rec["inhibitor_concentration"] = comp["conc"]
                    sol_rec["inhibitor_concentration_unit"] = comp["unit"]
                elif role == "organic_solvent" and sol_rec["organic_solvent_type"] is None:
                    sol_rec["organic_solvent_type"] = species
                    sol_rec["organic_solvent_concentration"] = comp["conc"]
                elif role == "electrolyte_additive" and sol_rec["electrolyte_additive"] is None:
                    sol_rec["electrolyte_additive"] = species
                    sol_rec["electrolyte_additive_concentration"] = comp["conc"]
                    sol_rec["electrolyte_additive_concentration_unit"] = comp["unit"]

            solutions[sol_id] = sol_rec
            sol_sources[sol_id] = []
            sol_materials[sol_id] = []

        if src_id and src_id not in sol_sources[sol_id]:
            sol_sources[sol_id].append(src_id)
        if mat_id and mat_id not in sol_materials[sol_id]:
            sol_materials[sol_id].append(mat_id)
        if mat_id == METAL_SYMBOL:
            solutions[sol_id]["is_target_metal"] = True

    for sol_id, sol_rec in solutions.items():
        sol_rec["source"] = "; ".join(sol_sources[sol_id])
        sol_rec["materials"] = "; ".join(sol_materials[sol_id])

    df = pd.DataFrame(list(solutions.values()))
    return df

def build_electrochemical_database(

    raw_records: list[dict],

    solution_df: pd.DataFrame

) -> pd.DataFrame:

    rows = []

    record_counter = 0

 

    sol_lookup: dict[str, str] = {}

    if not solution_df.empty and "source" in solution_df.columns:

        for _, sol_row in solution_df.iterrows():

            sources = str(sol_row.get("source", "")).split(";")

            for src in sources:

                src = src.strip()

                if src:

                    sol_lookup[src] = sol_row["solution_id"]

 

    for rec in raw_records:

        record_counter += 1

        record_id = f"REC_{record_counter:04d}"

 

        material_id = rec.get("material_id_assigned")
        if not material_id:                      # 舊資料相容
            mat_norm = rec.get("materials_normalised") or ""
            material_id = next((m.strip() for m in mat_norm.split(";")
                                if m.strip() in KNOWN_MATERIALS), None)

        source_id = rec.get("source_id", "")

        solution_id = rec.get("solution_id_assigned") or sol_lookup.get(source_id)

        ph = rec.get("record_ph")
        if ph is None and rec.get("ph_values"):
            try:
                ph = float(str(rec["ph_values"]).split(";")[0].strip())
            except ValueError:
                pass

        temp = rec.get("record_temp")
        if temp is None and rec.get("temperature_C_values"):
            try:
                temp = float(str(rec["temperature_C_values"]).split(";")[0].strip())
            except ValueError:
                pass

        ref_elec = None

        ref_str = rec.get("reference_electrodes")

        if ref_str:

            ref_elec = ref_str.split(";")[0].strip()

 

        ecorr_v_she = rec.get("Ecorr_V_SHE")

        ecorr_v_orig = rec.get("Ecorr_V_original")

        icorr = rec.get("icorr_uA_cm2")

        log_icorr = rec.get("log_icorr")

        beta_a = rec.get("beta_a_mV_dec")

        beta_c = rec.get("beta_c_mV_dec")

        tafel_general = rec.get("tafel_slopes_general_mV_dec")

 

        cr_calc = None

        if icorr is not None and material_id is not None:

            cr_result = calc_corrosion_rate(icorr, material_id)

            cr_calc = cr_result.get("CR_nm_min")

 

        cr_reported_value = rec.get("CR_reported_value")

        cr_reported_unit = rec.get("CR_reported_unit")

        rp_value = rec.get("Rp_value")

        rp_unit = rec.get("Rp_unit")

 

        deaerated = rec.get("deaerated", False)

        aerated = rec.get("aerated", False)

        o2_sat = rec.get("o2_saturated", False)

        aeration = "deaerated" if deaerated else (

            "O2-saturated" if o2_sat else (

                "aerated" if aerated else None

            )

        )

 

        db_row = {

            "record_id": record_id,

            "material_id": material_id,

            "solution_id": solution_id,

            "metal": material_id,
            "is_target_metal": material_id == METAL_SYMBOL,
            "material_assignment_method": rec.get("material_assignment_method"),
            "paper_primary_material": rec.get("paper_primary_material"),
            "material_mention_counts": rec.get("material_mention_counts"),

            "sample_name": rec.get("sample_name"),
            "metal_form": rec.get("metal_form"),

            "surface_preparation": rec.get("surface_preparation"),

            "solution_description": rec.get("electrolytes_raw"),

            "pH": ph,

            "temperature_C": temp if temp is not None else 25,

            "reference_electrode": ref_elec,

            "Ecorr_V_SHE": round(ecorr_v_she, 4) if ecorr_v_she is not None else None,

            "Ecorr_V_original": round(ecorr_v_orig, 4) if ecorr_v_orig is not None else None,

            "icorr_uA_cm2": round(icorr, 4) if icorr is not None else None,

            "log_icorr": round(log_icorr, 3) if log_icorr is not None else None,

            "beta_a_mV_dec": round(beta_a, 1) if beta_a is not None else None,

            "beta_c_mV_dec": round(beta_c, 1) if beta_c is not None else None,

            "tafel_slopes_general": tafel_general,

            "Rp_ohm_cm2": rp_value,

            "Rp_unit": rp_unit,

            "CR_reported_value": cr_reported_value,

            "CR_reported_unit": cr_reported_unit,

            "CR_calculated_nm_min": cr_calc,

            "measurement_method": rec.get("measurement_methods"),

            "scan_rate": rec.get("scan_rates_raw"),

            "immersion_time": rec.get("immersion_times_raw"),

            "aeration_condition": aeration,

            "reaction_type": _classify_reaction_type(rec),
            "reactions": rec.get("reactions"),

            "source_type": "journal",

            "source_reference": source_id,

            "data_extraction_method": rec.get("extraction_method", "regex"),

            "data_quality_score": _estimate_quality(rec),

            "quality_notes": _quality_notes(rec),

            "notes": "",

        }

 

        rows.append(db_row)

 

    df = pd.DataFrame(rows)

    if KEEP_ONLY_TARGET_METAL and not df.empty:
        before = len(df)
        df = df[df["is_target_metal"]].reset_index(drop=True)
        print(f"  KEEP_ONLY_TARGET_METAL: {before} -> {len(df)} records")

    return df

 

def _estimate_quality(rec: dict) -> float:

    score = 0.3

    if rec.get("Ecorr_V_original") is not None:

        score += 0.15

    if rec.get("icorr_uA_cm2") is not None:

        score += 0.15

    if rec.get("beta_a_mV_dec") is not None:

        score += 0.1

    if rec.get("beta_c_mV_dec") is not None:

        score += 0.1

    if rec.get("ph_values"):

        score += 0.05

    if rec.get("reference_electrodes"):

        score += 0.05

    if rec.get("temperature_C_values"):

        score += 0.05

    if rec.get("measurement_methods"):

        score += 0.05

    if rec.get("extraction_method") == "table_regex":

        score += 0.1

    return min(round(score, 2), 1.0)

 

def check_outlier_reason(row: dict) -> str | None:
    """檢查單筆電化學記錄是否超出物理合理區間，回傳原因字串或 None。"""
    reasons = []
    mat = row.get("material_id") or "default"
    ecorr_she = row.get("Ecorr_V_SHE")
    if ecorr_she is not None and not pd.isna(ecorr_she):
        lo, hi = ECORR_VALID_RANGE_V_SHE.get(mat, ECORR_VALID_RANGE_V_SHE["default"])
        if not (lo <= float(ecorr_she) <= hi):
            reasons.append(f"Ecorr_V_SHE={ecorr_she} out of [{lo}, {hi}] V for {mat}")

    icorr = row.get("icorr_uA_cm2")
    if icorr is not None and not pd.isna(icorr):
        ilo, ihi = ICORR_VALID_RANGE_UA_CM2
        if not (ilo <= float(icorr) <= ihi):
            reasons.append(f"icorr_uA_cm2={icorr} out of [{ilo}, {ihi}] uA/cm2")

    return "; ".join(reasons) if reasons else None


def filter_echem_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """依 config.FILTER_OUTLIERS 分離正常資料與離群值。"""
    if df.empty or not FILTER_OUTLIERS:
        return df, pd.DataFrame(columns=list(df.columns) + ["outlier_reason"])

    reasons = [check_outlier_reason(r) for _, r in df.iterrows()]
    is_out = [r is not None for r in reasons]
    outliers = df[is_out].copy()
    outliers["outlier_reason"] = [r for r in reasons if r is not None]
    clean = df[[not x for x in is_out]].reset_index(drop=True)
    return clean, outliers


def _classify_reaction_type(rec: dict) -> str:
    has_corr_param = any(
        rec.get(k) is not None
        for k in ("Ecorr_V_original", "icorr_uA_cm2", "Rp_value", "CR_reported_value")
    ) or (rec.get("beta_a_mV_dec") is not None and rec.get("beta_c_mV_dec") is not None)
    rxn = str(rec.get("reactions") or "")
    has_cat = any(k in rxn for k in ("HER", "OER", "ORR", "CO2RR"))
    if has_corr_param:
        return "corrosion"
    if has_cat and rec.get("tafel_slopes_general_mV_dec") is not None:
        return "electrocatalysis"
    if "corrosion" in rxn:
        return "corrosion"
    return "unknown"


def _quality_notes(rec: dict) -> str:

    notes = []

    if rec.get("extraction_method") == "table_regex":

        notes.append("extracted from table")

    else:

        notes.append("extracted from text")

    if rec.get("Ecorr_V_SHE") is None and rec.get("Ecorr_V_original") is not None:

        notes.append("Ecorr not converted to SHE (unknown reference)")

    if not rec.get("reference_electrodes"):

        notes.append("reference electrode not identified")

    if not rec.get("ph_values"):

        notes.append("pH not found")

    if not rec.get("temperature_C_values"):

        notes.append("temperature not found, assumed 25°C")

    extra = rec.get("extra_quality_notes")
    if extra:
        notes.append(extra)
    return "; ".join(notes)

 

# ====================================================================

# 12. MAIN RUN FUNCTION

# ====================================================================

 

def run_extract():

    text_files = [f for f in os.listdir(TEXT_DIR) if f.endswith(".txt")]

    print(f"Found {len(text_files)} text files in {TEXT_DIR}")

 

    all_raw_records = []

 

    for txt_file in tqdm(text_files, desc="Extracting electrochemical data"):

        source_id = txt_file.replace(".txt", "")

        txt_path = os.path.join(TEXT_DIR, txt_file)

 

        with open(txt_path, "r", encoding="utf-8") as f:

            text = f.read()

 

        norm_text = normalize_text(text)
        body_text = strip_references(norm_text)
        paper_counts = count_materials(body_text)
        text_records = extract_from_text(norm_text, source_id, paper_counts=paper_counts)

        all_raw_records.extend(text_records)

        table_path = os.path.join(TEXT_DIR, source_id + "_tables.json")

        if os.path.exists(table_path):
            paper_env = {
                "electrolytes": extract_electrolytes(body_text),
                "ph_counts": extract_phs(body_text),
                "temp_counts": extract_temperatures(body_text),
                "ref_counts": extract_ref_electrodes(body_text),
                "refs": [r for r, _ in sorted(extract_ref_electrodes(body_text).items(), key=lambda kv: (-kv[1], kv[0]))],
                "reactions": detect_reactions(norm_text),
                "metal_form": text_records[0].get("metal_form") if text_records else None,
                "surface_preparation": text_records[0].get("surface_preparation") if text_records else None,
                "measurement_methods": text_records[0].get("measurement_methods") if text_records else None,
                "scan_rates_raw": text_records[0].get("scan_rates_raw") if text_records else None,
                "immersion_times_raw": text_records[0].get("immersion_times_raw") if text_records else None,
                "deaerated": text_records[0].get("deaerated", False) if text_records else False,
                "aerated": text_records[0].get("aerated", False) if text_records else False,
                "o2_saturated": text_records[0].get("o2_saturated", False) if text_records else False,
            }
            table_records = extract_from_tables(
                table_path, source_id, paper_counts=paper_counts, paper_env=paper_env
            )
            all_raw_records.extend(table_records)

    print(f"\nTotal raw records extracted: {len(all_raw_records)}")

 

    if not all_raw_records:

        print("No records found. Exiting.")

        return

 

    # ---- Build three tables ----

 

    # 1. material_properties

    print(f"\n--- Building {METAL_SYMBOL} material_properties ---")        # ✅

    mat_df = build_material_properties(all_raw_records)

    mat_df.to_csv(STEP5_MATERIAL_OUTPUT, index=False, encoding="utf-8-sig")  # ✅

    print(f"  Materials found: {len(mat_df)}")

    print(f"  ✅ Saved to {STEP5_MATERIAL_OUTPUT}")

    if not mat_df.empty:

        print(f"  Materials: {', '.join(mat_df['material_id'].tolist())}")

 

    # 2. solution_properties

    print(f"\n--- Building {METAL_SYMBOL} solution_properties ---")        # ✅

    sol_df = build_solution_properties(all_raw_records)

    sol_df.to_csv(STEP5_SOLUTION_OUTPUT, index=False, encoding="utf-8-sig")  # ✅

    print(f"  Unique solutions: {len(sol_df)}")

    print(f"  ✅ Saved to {STEP5_SOLUTION_OUTPUT}")

    if not sol_df.empty and "solution_category" in sol_df.columns:

        print("  Categories:")

        print(sol_df["solution_category"].value_counts().to_string(

            header=False

        ))

 

    # 3. electrochemical_database

    print(f"\n--- Building {METAL_SYMBOL} electrochemical_database ---")   # ✅

    echem_df = build_electrochemical_database(all_raw_records, sol_df)
    echem_df, outliers_df = filter_echem_outliers(echem_df)
    outliers_df.to_csv(STEP5_OUTLIERS_OUTPUT, index=False, encoding="utf-8-sig")
    if len(outliers_df) > 0:
        print(f"  🧹 Filtered {len(outliers_df)} outlier records → {STEP5_OUTLIERS_OUTPUT}")
        # 同步移除只屬於被過濾離群值的孤立溶液
        if not sol_df.empty and not echem_df.empty and "solution_id" in echem_df.columns:
            valid_sids = set(echem_df["solution_id"].dropna().unique())
            sol_df = sol_df[sol_df["solution_id"].isin(valid_sids)].reset_index(drop=True)
            sol_df.to_csv(STEP5_SOLUTION_OUTPUT, index=False, encoding="utf-8-sig")
    echem_df.to_csv(STEP5_ECHEM_OUTPUT, index=False, encoding="utf-8-sig")  # ✅

    print(f"  Total records: {len(echem_df)}")

    print(f"  ✅ Saved to {STEP5_ECHEM_OUTPUT}")

 

    if not echem_df.empty:

        print(f"\n  Records with Ecorr: "

              f"{echem_df['Ecorr_V_SHE'].notna().sum()}")

        print(f"  Records with icorr: "

              f"{echem_df['icorr_uA_cm2'].notna().sum()}")

        print(f"  Records with βa: "

              f"{echem_df['beta_a_mV_dec'].notna().sum()}")

        print(f"  Records with βc: "

              f"{echem_df['beta_c_mV_dec'].notna().sum()}")

        print(f"  Records with CR calc: "

              f"{echem_df['CR_calculated_nm_min'].notna().sum()}")

        print(f"  Records with material_id: "

              f"{echem_df['material_id'].notna().sum()}")

        print(f"  Records with solution_id: "

              f"{echem_df['solution_id'].notna().sum()}")

 

        if echem_df["material_id"].notna().any():

            print("\n  Material distribution:")

            print(echem_df["material_id"].value_counts().to_string(

                header=False

            ))

 

        print("\n  Data quality score statistics:")

        print(echem_df["data_quality_score"].describe().to_string())

 

        if echem_df["Ecorr_V_SHE"].notna().any():

            print("\n  Ecorr (V vs SHE) statistics:")

            print(echem_df["Ecorr_V_SHE"].describe().to_string())

 

        if echem_df["icorr_uA_cm2"].notna().any():

            print("\n  icorr (µA/cm²) statistics:")

            print(echem_df["icorr_uA_cm2"].describe().to_string())

 

    # ---- Raw records for debug ----

    raw_df = pd.DataFrame(all_raw_records)

    raw_df.to_csv(STEP5_RAW_OUTPUT, index=False, encoding="utf-8-sig")     # ✅

    print(f"\n  Raw records saved to {STEP5_RAW_OUTPUT}")

 

    print("\n=== Extraction complete ===")

 

    return mat_df, sol_df, echem_df

 

if __name__ == "__main__":

    run_extract()