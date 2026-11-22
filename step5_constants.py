"""
Step 5 — Material constants, reference-electrode conversions, and unit normalizers.
"""

import re
import unicodedata

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
        exp = re.sub(r"\s+", "", m.group(2))
        try:
            return float(m.group(1)) * (10 ** float(exp))
        except (ValueError, OverflowError):
            return None

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
