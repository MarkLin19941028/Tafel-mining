"""
Step 5 (Revised): Extract electrochemical data from raw text
and populate three standardised CSV tables:
  1. {metal}_material_properties.csv
  2. {metal}_solution_properties.csv
  3. {metal}_electrochemical_database.csv
"""

import os

import pandas as pd
from tqdm import tqdm

from config import (
    METAL_SYMBOL,
    STEP5_ECHEM_OUTPUT,
    STEP5_MATERIAL_OUTPUT,
    STEP5_OUTLIERS_OUTPUT,
    STEP5_RAW_OUTPUT,
    STEP5_SOLUTION_OUTPUT,
    TEXT_DIR,
)
from step5_constants import (
    KNOWN_MATERIALS,
    REF_ELECTRODE_TO_SHE,
    calc_corrosion_rate,
    convert_to_she,
    normalize_ecorr_to_v,
    normalize_icorr_to_uA_cm2,
    normalize_text,
    parse_scientific,
)
from step5_text_extractor import (
    assign_material,
    classify_solution_category,
    count_materials,
    detect_reactions,
    extract_electrolytes,
    extract_from_text,
    extract_phs,
    extract_ref_electrodes,
    extract_temperatures,
    find_materials,
    format_counts,
    generate_solution_id,
    get_context,
    lookup_species_category,
    normalise_material,
    normalise_ref_electrode,
    normalise_species,
    select_environment_for_record,
    select_ref_electrode,
    strip_references,
)
from step5_table_extractor import (
    extract_from_tables,
    parse_structured_table,
)
from step5_table_builders import (
    build_electrochemical_database,
    build_material_properties,
    build_solution_properties,
    check_outlier_reason,
    filter_echem_outliers,
)

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
