"""
Step 5 — Output table builders (material, solution, electrochemical) and quality/outlier helpers.
"""

import pandas as pd

from config import (
    ECORR_VALID_RANGE_V_SHE,
    FILTER_OUTLIERS,
    ICORR_VALID_RANGE_UA_CM2,
    KEEP_ONLY_TARGET_METAL,
    METAL_SYMBOL,
)
from step5_constants import (
    KNOWN_MATERIALS,
    calc_corrosion_rate,
)
from step5_text_extractor import (
    classify_solution_category,
    extract_electrolytes,
    generate_solution_id,
    lookup_species_category,
)

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
