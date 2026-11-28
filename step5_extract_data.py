"""
Step 5 (Revised): Extract electrochemical data from raw text
and populate three standardised CSV tables:
  1. {metal}_material_properties.csv
  2. {metal}_solution_properties.csv
  3. {metal}_electrochemical_database.csv
"""

from collections import Counter
from dataclasses import replace as dataclass_replace
import json
import os

import pandas as pd
from tqdm import tqdm

import config
from step5_constants import normalize_text
from step5_table_builders import (
    build_electrochemical_database,
    build_material_properties,
    build_solution_properties,
    filter_echem_outliers,
    is_eligible_echem_record,
)
from step5_table_extractor import (
    extract_from_tables,
    extract_table_captions,
)
from step5_text_extractor import (
    count_materials,
    extract_from_text,
    extract_paper_environment,
    extract_ref_electrodes,
    strip_references,
)

# ====================================================================

# 12. MAIN RUN FUNCTION

# ====================================================================

 

def run_extract(ctx: config.MetalRunContext | None = None):
    ctx = ctx or config.default_metal_context()
    ctx.ensure_dirs()
    settings = ctx.settings
    metal_symbol = ctx.metal_symbol
    text_dir = ctx.text_dir

    text_files = [f for f in os.listdir(text_dir) if f.endswith(".txt")]

    print(f"Found {len(text_files)} text files in {text_dir}")

 

    all_raw_records = []

 

    for txt_file in tqdm(text_files, desc="Extracting electrochemical data"):

        source_id = txt_file.replace(".txt", "")

        txt_path = os.path.join(text_dir, txt_file)

 

        with open(txt_path, "r", encoding="utf-8") as f:

            text = f.read()

 

        norm_text = normalize_text(text)
        body_text = strip_references(norm_text)
        paper_counts = count_materials(body_text, ctx=ctx)
        paper_environment = extract_paper_environment(
            norm_text,
            body_text,
            table_captions=extract_table_captions(body_text),
        )
        table_path = os.path.join(text_dir, source_id + "_tables.json")
        if os.path.exists(table_path):
            try:
                with open(table_path, "r", encoding="utf-8") as tf:
                    raw_tables = json.load(tf)
                if isinstance(raw_tables, list):
                    header_chunks: list[str] = []
                    for tbl in raw_tables:
                        if isinstance(tbl, list):
                            for hrow in tbl[:3]:
                                if isinstance(hrow, list):
                                    header_chunks.append(" ".join(str(c or "") for c in hrow))
                    if header_chunks:
                        tbl_refs = extract_ref_electrodes(" ".join(header_chunks))
                        if tbl_refs:
                            merged_ref_counts = Counter(paper_environment.ref_counts)
                            merged_ref_counts.update(tbl_refs)
                            merged_refs = [
                                r
                                for r, _ in sorted(
                                    merged_ref_counts.items(),
                                    key=lambda kv: (-kv[1], kv[0]),
                                )
                            ]
                            paper_environment = dataclass_replace(
                                paper_environment,
                                ref_counts=merged_ref_counts,
                                refs=merged_refs,
                            )
            except (OSError, json.JSONDecodeError):
                pass
        text_records = extract_from_text(
            norm_text,
            source_id,
            paper_counts=paper_counts,
            ctx=ctx,
            paper_environment=paper_environment,
        )

        table_records = []

        if os.path.exists(table_path):
            table_records = extract_from_tables(
                table_path,
                source_id,
                paper_counts=paper_counts,
                paper_environment=paper_environment,
                ctx=ctx,
            )

        # 論文級表格優先：若該論文已有至少 1 筆合格（且符合目標金屬篩選）電化學表格記錄，則捨棄內文 regex 記錄避免重複
        has_valid_table = any(
            is_eligible_echem_record(r)
            and (not settings.keep_only_target_metal or r.get("is_target_metal", False))
            for r in table_records
        )
        if has_valid_table:
            all_raw_records.extend(table_records)
        else:
            all_raw_records.extend(text_records)

    print(f"\nTotal raw records extracted: {len(all_raw_records)}")

 

    if not all_raw_records:

        print("No records found. Exiting.")

        return

 

    # ---- Build three tables ----

 

    # 1. material_properties

    print(f"\n--- Building {metal_symbol} material_properties ---")        # ✅

    mat_df = build_material_properties(all_raw_records, ctx=ctx)

    mat_df.to_csv(ctx.step5_material_output, index=False, encoding="utf-8-sig")  # ✅

    print(f"  Materials found: {len(mat_df)}")

    print(f"  ✅ Saved to {ctx.step5_material_output}")

    if not mat_df.empty:

        print(f"  Materials: {', '.join(mat_df['material_id'].tolist())}")

 

    # 2. solution_properties

    print(f"\n--- Building {metal_symbol} solution_properties ---")        # ✅

    sol_df = build_solution_properties(all_raw_records, ctx=ctx)

    sol_df.to_csv(ctx.step5_solution_output, index=False, encoding="utf-8-sig")  # ✅

    print(f"  Unique solutions: {len(sol_df)}")

    print(f"  ✅ Saved to {ctx.step5_solution_output}")

    if not sol_df.empty and "solution_category" in sol_df.columns:

        print("  Categories:")

        print(sol_df["solution_category"].value_counts().to_string(

            header=False

        ))

 

    # 3. electrochemical_database

    print(f"\n--- Building {metal_symbol} electrochemical_database ---")   # ✅

    echem_df = build_electrochemical_database(all_raw_records, sol_df, ctx=ctx)
    echem_df, outliers_df = filter_echem_outliers(echem_df, settings=settings)
    outliers_df.to_csv(ctx.step5_outliers_output, index=False, encoding="utf-8-sig")
    if len(outliers_df) > 0:
        print(f"  🧹 Filtered {len(outliers_df)} outlier records → {ctx.step5_outliers_output}")
    # 同步移除未在最終有效電化學主表中被引用的孤立溶液
    if not sol_df.empty and "solution_id" in sol_df.columns:
        valid_sids = (
            set(echem_df["solution_id"].dropna().unique())
            if (not echem_df.empty and "solution_id" in echem_df.columns)
            else set()
        )
        sol_df = sol_df[sol_df["solution_id"].isin(valid_sids)].reset_index(drop=True)
        sol_df.to_csv(ctx.step5_solution_output, index=False, encoding="utf-8-sig")
    echem_df.to_csv(ctx.step5_echem_output, index=False, encoding="utf-8-sig")  # ✅

    print(f"  Total records: {len(echem_df)}")

    print(f"  ✅ Saved to {ctx.step5_echem_output}")

 

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

    raw_df.to_csv(ctx.step5_raw_output, index=False, encoding="utf-8-sig")     # ✅

    print(f"\n  Raw records saved to {ctx.step5_raw_output}")

 

    print("\n=== Extraction complete ===")

 

    return mat_df, sol_df, echem_df

 

if __name__ == "__main__":
    _cli_ctx = config.default_metal_context()
    _cli_ctx.print_summary()
    run_extract(ctx=_cli_ctx)
