"""
Step 6: 合併所有金屬的 extracted 資料，輸出到 corrosion_simulator/data/
"""

import os
import re
import sys
import pandas as pd
from pathlib import Path
import ast
import json

# 優先從 config 讀取 SIMULATOR_DATA_DIR，讓路徑集中管理
try:
    from config import SIMULATOR_DATA_DIR as _CFG_SIM_DIR
    SIMULATOR_DATA_DIR = Path(_CFG_SIM_DIR)
except ImportError:
    # fallback：與 Tafel mining 同層的 corrosion_simulator/data/
    SIMULATOR_DATA_DIR = Path(__file__).parent.parent / "corrosion_simulator" / "data"

from step5_constants import KNOWN_MATERIALS
from step5_table_builders import (
    build_electrochemical_database,
    build_solution_properties,
    check_outlier_reason,
    classify_material_class,
    filter_echem_outliers,
)
from step5_text_extractor import (
    extract_electrolytes,
    generate_solution_id,
    lookup_species_category,
)
import step5_table_builders

MINING_DATA_DIR = Path(__file__).parent / "data"

# ============================================================
#  修復 1：SIMULATOR_DATA_DIR 路徑驗證
# ============================================================
# 若目錄的父層不存在，代表 corrosion_simulator 專案尚未建立，
# 給出明確的錯誤訊息而不是靜默建立錯誤路徑。
if not SIMULATOR_DATA_DIR.parent.exists():
    print(f"⚠️  WARNING: corrosion_simulator directory not found at:")
    print(f"   {SIMULATOR_DATA_DIR.parent}")
    print(f"   Will create output at: {SIMULATOR_DATA_DIR}")
    print(f"   If this is wrong, edit SIMULATOR_DATA_DIR in config.py")

os.makedirs(SIMULATOR_DATA_DIR, exist_ok=True)

# ============================================================
#  修復 3：自動偵測時檢查三個 CSV 是否齊全
# ============================================================
REQUIRED_SUFFIXES = [
    "material_properties.csv",
    "solution_properties.csv",
    "electrochemical_database.csv",
]

def _metal_is_complete(metal_dir: Path) -> bool:
    """只有三個 CSV 都存在才算完整，避免引入殘缺資料。"""
    extracted = metal_dir / "extracted"
    if not extracted.is_dir():
        return False
    prefix = metal_dir.name.lower()
    missing = [
        suffix for suffix in REQUIRED_SUFFIXES
        if not (extracted / f"{prefix}_{suffix}").exists()
    ]
    if missing:
        print(f"  ⚠️  {metal_dir.name}: incomplete step5 output, skipping "
              f"(missing: {missing})")
        return False
    return True

METALS_TO_MERGE = sorted([
    d.name for d in MINING_DATA_DIR.iterdir()
    if d.is_dir() and _metal_is_complete(d)
])

# ============================================================
#  合併邏輯
# ============================================================

def merge_csvs(filename: str, id_col: str = None) -> pd.DataFrame:
    """
    合併所有金屬的同名 CSV。

    Args:
        filename: 檔名後綴，例如 "material_properties.csv"
        id_col:   若指定，則於跨金屬層面依該欄位做全域去重
    """
    frames = []

    for metal in METALS_TO_MERGE:
        extracted_dir = MINING_DATA_DIR / metal / "extracted"
        prefix = metal.lower()
        target_file = extracted_dir / f"{prefix}_{filename}"

        if target_file.exists():
            df = pd.read_csv(target_file)
            df["_source_metal"] = metal
            frames.append(df)
            print(f"  {metal}: {len(df)} rows from {target_file.name}")
        else:
            print(f"  {metal}: {filename} not found, skipping")

    if not frames:
        print(f"  ⚠️ No data found for {filename}")
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)

    # 全域去重：跨金屬層面
    if id_col and id_col in merged.columns:
        before = len(merged)
        merged = merged.drop_duplicates(subset=[id_col], keep="first")
        dupes = before - len(merged)
        if dupes > 0:
            print(f"  Removed {dupes} cross-metal duplicates by {id_col}")

    return merged


def _merge_semicolon_values(series: pd.Series) -> str:
    seen: list[str] = []
    for val in series.dropna():
        for part in str(val).split(";"):
            p = part.strip()
            if p and p not in seen:
                seen.append(p)
    return "; ".join(seen)


def deduplicate_solutions_global(sol_df: pd.DataFrame) -> pd.DataFrame:
    """跨金屬依 solution_id 全域去重，並合併各金屬的 source / materials / _source_metal 清單。"""
    if sol_df.empty or "solution_id" not in sol_df.columns:
        return sol_df
    before = len(sol_df)
    merge_cols = {"source", "materials", "_source_metal"}
    rows = []
    for _, grp in sol_df.groupby("solution_id", sort=False):
        if len(grp) == 1:
            rows.append(grp.iloc[0].to_dict())
            continue
        merged_row = {}
        for col in sol_df.columns:
            if col in merge_cols:
                merged_row[col] = _merge_semicolon_values(grp[col])
            elif col == "is_target_metal":
                merged_row[col] = bool(grp[col].fillna(False).any())
            else:
                non_null = grp[col].dropna()
                merged_row[col] = non_null.iloc[0] if len(non_null) > 0 else None
        rows.append(merged_row)
    deduped = pd.DataFrame(rows, columns=sol_df.columns)
    removed = before - len(deduped)
    if removed > 0:
        print(f"  Merged {removed} duplicate solution_id rows globally")
    return deduped


_TABLE_ROW_SUFFIX_RE = re.compile(r"_table\d+_row\d+$")


def deduplicate_echem_cross_metal(echem_df: pd.DataFrame) -> pd.DataFrame:
    """跨金屬與同論文去重，優先保留表格擷取與品質分數較高的記錄。"""
    if echem_df.empty:
        return echem_df
    df = echem_df.copy()
    before = len(df)

    method_rank = {"table_column_mapped": 0, "table_regex": 1, "regex": 2}
    df["_method_rank"] = (
        df["data_extraction_method"].map(method_rank).fillna(3)
        if "data_extraction_method" in df.columns
        else 3
    )
    df["_qscore"] = (
        pd.to_numeric(df["data_quality_score"], errors="coerce").fillna(0.0)
        if "data_quality_score" in df.columns
        else 0.0
    )
    df = df.sort_values(by=["_method_rank", "_qscore"], ascending=[True, False], kind="mergesort")

    def _round_key(val) -> str:
        if pd.isna(val):
            return ""
        try:
            return f"{round(float(val), 4):.4f}"
        except (TypeError, ValueError):
            return str(val).strip()

    def _build_key(row) -> tuple:
        src = _TABLE_ROW_SUFFIX_RE.sub("", str(row.get("source_reference", "") or "").strip())
        mat = str(row.get("material_id", "") or "").strip()
        sol = str(row.get("solution_id", "") or "").strip()
        if sol.lower() == "nan":
            sol = ""
        ph_val = row.get("pH")
        ph_key = f"{round(float(ph_val), 2):.2f}" if pd.notna(ph_val) else ""
        sname = re.sub(r"\s+", " ", str(row.get("sample_name", "") or "").strip().lower())
        if sname == "nan":
            sname = ""
        e_val = row.get("Ecorr_V_SHE")
        if pd.isna(e_val):
            e_val = row.get("Ecorr_V_original")
        e_key = _round_key(e_val)
        i_key = _round_key(row.get("icorr_uA_cm2"))
        return (src, mat, sol, ph_key, sname, e_key, i_key)

    df["_dedup_key"] = df.apply(_build_key, axis=1)
    df = df.drop_duplicates(subset=["_dedup_key"], keep="first")
    df = df.drop(columns=["_method_rank", "_qscore", "_dedup_key"]).reset_index(drop=True)

    removed = before - len(df)
    if removed > 0:
        print(f"  Removed {removed} duplicate echem records across/within metals")
    return df


def write_quality_report(
    mat_df: pd.DataFrame,
    sol_df: pd.DataFrame,
    echem_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """產出 data/merged/quality_report.md，彙整欄位覆蓋率、數值分佈與過濾離群值摘要。"""
    from datetime import datetime

    lines = [
        "# Tafel Mining 資料品質報告",
        "",
        f"> 產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 總覽",
        "",
        "| 項目 | 數量 |",
        "|---|---|",
        f"| 合併金屬數 | {len(METALS_TO_MERGE)} ({', '.join(METALS_TO_MERGE)}) |",
        f"| 材料表 (`material_properties.csv`) | {len(mat_df)} |",
        f"| 溶液表 (`solution_properties.csv`) | {len(sol_df)} |",
        f"| 電化學主表 (`electrochemical_database.csv`) | {len(echem_df)} |",
        "",
    ]

    if not echem_df.empty:
        lines.extend([
            "## 2. 關鍵欄位覆蓋率",
            "",
            "| 欄位 | 有效筆數 | 覆蓋率 |",
            "|---|---:|---:|",
        ])
        for col in [
            "Ecorr_V_SHE", "icorr_uA_cm2", "beta_a_mV_dec", "beta_c_mV_dec",
            "tafel_slopes_general", "CR_reported_value", "Rp_ohm_cm2",
            "solution_id", "reference_electrode", "sample_name",
        ]:
            if col in echem_df.columns:
                n = int(echem_df[col].notna().sum())
                pct = n / len(echem_df) * 100
                lines.append(f"| `{col}` | {n} / {len(echem_df)} | {pct:.1f}% |")
        lines.append("")

        if "reaction_type" in echem_df.columns:
            lines.extend(["## 3. 反應類型與擷取來源", ""])
            rxn = echem_df["reaction_type"].value_counts(dropna=False).to_dict()
            ext = echem_df["data_extraction_method"].value_counts(dropna=False).to_dict() if "data_extraction_method" in echem_df.columns else {}
            mcls = echem_df["material_class"].value_counts(dropna=False).to_dict() if "material_class" in echem_df.columns else {}
            ref = echem_df["reference_electrode"].value_counts(dropna=True).head(8).to_dict() if "reference_electrode" in echem_df.columns else {}
            lines.append(f"- **反應類型 (`reaction_type`)**：`{rxn}`")
            if mcls:
                lines.append(f"- **材料類別 (`material_class`)**：`{mcls}`")
            lines.append(f"- **擷取方式 (`data_extraction_method`)**：`{ext}`")
            lines.append(f"- **主要參考電極**：`{ref}`")
            lines.append("")

        lines.extend([
            "## 4. 各金屬電化學數值分佈",
            "",
            "| 金屬 | 總筆數 | Ecorr (V vs SHE) 中位數 [min, max] | icorr (µA/cm²) 中位數 [min, max] | βa / βc 有效筆數 |",
            "|---|---:|---|---|---:|",
        ])
        group_col = "material_id" if "material_id" in echem_df.columns else "_source_metal"
        for metal, grp in echem_df.groupby(group_col):
            e_s = grp["Ecorr_V_SHE"].dropna() if "Ecorr_V_SHE" in grp.columns else pd.Series(dtype=float)
            i_s = grp["icorr_uA_cm2"].dropna() if "icorr_uA_cm2" in grp.columns else pd.Series(dtype=float)
            ba_n = int(grp["beta_a_mV_dec"].notna().sum()) if "beta_a_mV_dec" in grp.columns else 0
            bc_n = int(grp["beta_c_mV_dec"].notna().sum()) if "beta_c_mV_dec" in grp.columns else 0
            e_txt = f"{e_s.median():.3f} [{e_s.min():.3f}, {e_s.max():.3f}] (n={len(e_s)})" if len(e_s) else "—"
            i_txt = f"{i_s.median():.3g} [{i_s.min():.3g}, {i_s.max():.3g}] (n={len(i_s)})" if len(i_s) else "—"
            lines.append(f"| {metal} | {len(grp)} | {e_txt} | {i_txt} | {ba_n} / {bc_n} |")
        lines.append("")

    # 彙整各金屬被過濾的離群值清單
    outlier_frames = []
    for metal in METALS_TO_MERGE:
        of = MINING_DATA_DIR / metal / "extracted" / f"{metal.lower()}_filtered_outliers.csv"
        if of.exists():
            try:
                odf = pd.read_csv(of)
                if not odf.empty:
                    odf["_source_metal"] = metal
                    outlier_frames.append(odf)
            except Exception:
                pass

    lines.extend(["## 5. 自動過濾離群值摘要", ""])
    if outlier_frames:
        all_out = pd.concat(outlier_frames, ignore_index=True)
        lines.append(f"共自動濾除 **{len(all_out)}** 筆超出物理合理區間的記錄：")
        lines.append("")
        lines.append("| 金屬 | 樣品 (`sample_name`) | Ecorr_V_SHE | icorr_uA_cm2 | 過濾原因 | 來源 |")
        lines.append("|---|---|---:|---:|---|---|")
        for _, r in all_out.head(30).iterrows():
            lines.append(
                f"| {r.get('material_id', '')} | {r.get('sample_name', '')} | "
                f"{r.get('Ecorr_V_SHE', '')} | {r.get('icorr_uA_cm2', '')} | "
                f"{r.get('outlier_reason', '')} | {r.get('source_reference', '')} |"
            )
    else:
        lines.append("無離群值被過濾。")
    lines.append("")

    report_path = output_dir / "quality_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def build_full_material_properties(echem_df: pd.DataFrame) -> pd.DataFrame:
    """從 KNOWN_MATERIALS 建立完整 11 種專案目標材料的物性表，並標記 _source_metal。"""
    source_map: dict[str, str] = {}
    if not echem_df.empty and {"material_id", "_source_metal"}.issubset(echem_df.columns):
        for mat_id, grp in echem_df.groupby("material_id"):
            src = _merge_semicolon_values(grp["_source_metal"])
            if src:
                source_map[str(mat_id)] = src

    rows = []
    for mat_id in sorted(KNOWN_MATERIALS.keys()):
        props = KNOWN_MATERIALS[mat_id]
        row = {"material_id": mat_id}
        row.update(props)
        row["notes"] = "" if mat_id in source_map else "target material (from materials_physics.json)"
        row["_source_metal"] = source_map.get(mat_id, mat_id)
        rows.append(row)
    return pd.DataFrame(rows)


def harvest_cross_metal_and_raw_records(
    echem_primary: pd.DataFrame,
    sol_primary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    從各金屬的 *_raw_records.csv 與 data/merged/raw/ 歷史資料表回收所有屬於
    KNOWN_MATERIALS（11 種專案目標材料）的合格腐蝕記錄與溶液，並以嚴格品管規則過濾與去重。
    """
    orig_keep = step5_table_builders.KEEP_ONLY_TARGET_METAL
    step5_table_builders.KEEP_ONLY_TARGET_METAL = False
    try:
        all_recs: list[dict] = []
        for metal in METALS_TO_MERGE:
            raw_path = MINING_DATA_DIR / metal / "extracted" / f"{metal.lower()}_raw_records.csv"
            if not raw_path.exists():
                continue
            df_r = pd.read_csv(raw_path)
            for _, row in df_r.iterrows():
                rec = row.to_dict()
                for k, v in list(rec.items()):
                    if pd.isna(v):
                        rec[k] = None
                if isinstance(rec.get("record_electrolytes"), str):
                    try:
                        rec["record_electrolytes"] = ast.literal_eval(rec["record_electrolytes"])
                    except Exception:
                        rec["record_electrolytes"] = None
                if rec.get("material_id_assigned") in KNOWN_MATERIALS:
                    rec["is_target_metal"] = True
                    rec["_source_metal"] = metal
                    all_recs.append(rec)

        sol_from_raw_rec = build_solution_properties(all_recs) if all_recs else pd.DataFrame()
        if not sol_from_raw_rec.empty:
            sol_from_raw_rec["is_target_metal"] = True
            sol_from_raw_rec["_source_metal"] = "cross_metal"
        echem_from_raw_rec = (
            build_electrochemical_database(all_recs, sol_from_raw_rec)
            if all_recs
            else pd.DataFrame()
        )
        if not echem_from_raw_rec.empty:
            echem_from_raw_rec["is_target_metal"] = True
            src_to_m = {r["source_id"]: r["_source_metal"] for r in all_recs}
            echem_from_raw_rec["_source_metal"] = echem_from_raw_rec["source_reference"].map(src_to_m)
            echem_from_raw_rec, _ = filter_echem_outliers(echem_from_raw_rec)
    finally:
        step5_table_builders.KEEP_ONLY_TARGET_METAL = orig_keep

    # 整合 data/merged/raw/ 下的歷史結構化表格與溶液（若存在）
    raw_dir = SIMULATOR_DATA_DIR / "raw"
    raw_ml_path = raw_dir / "ml_training_dataset.csv"
    raw_sol_path = raw_dir / "solution_properties.csv"

    old_to_new_sid: dict[str, str] = {}
    sol_raw_df = pd.DataFrame()
    if raw_sol_path.exists():
        sol_raw = pd.read_csv(raw_sol_path)
        sol_raw_upgraded = []
        for _, r in sol_raw.iterrows():
            old_sid = str(r.get("solution_id", "") or "")
            name = str(r.get("solution_name", "") or "")
            comps = (
                extract_electrolytes(name)
                if name and not name.startswith("pH ") and name != "unknown"
                else []
            )
            ph = r.get("pH") if pd.notna(r.get("pH")) else None
            temp = r.get("temperature_C") if pd.notna(r.get("temperature_C")) else None
            new_sid = generate_solution_id(comps, ph, temp)
            if old_sid:
                old_to_new_sid[old_sid] = new_sid
            r_dict = r.to_dict()
            r_dict["solution_id"] = new_sid
            r_dict["is_target_metal"] = True
            r_dict["components_json"] = json.dumps(
                [
                    {
                        "species": c.get("species"),
                        "conc": c.get("conc"),
                        "unit": c.get("unit"),
                        "role": (lookup_species_category(c.get("species", "")) or {}).get("role"),
                    }
                    for c in comps
                ],
                ensure_ascii=False,
            )
            sol_raw_upgraded.append(r_dict)
        if sol_raw_upgraded:
            sol_raw_df = pd.DataFrame(sol_raw_upgraded)

    valid_raw_tables = pd.DataFrame()
    if raw_ml_path.exists():
        df_raw = pd.read_csv(raw_ml_path)

        def _recompute_sid(row):
            desc = str(row.get("solution_description", "") or "")
            comps = extract_electrolytes(desc) if desc and desc != "nan" else []
            ph = row.get("pH") if pd.notna(row.get("pH")) else None
            temp = row.get("temperature_C") if pd.notna(row.get("temperature_C")) else None
            if comps or ph is not None:
                return generate_solution_id(comps, ph, temp)
            old_sid = str(row.get("solution_id", "") or "")
            return old_to_new_sid.get(old_sid, old_sid) if old_sid and old_sid != "nan" else None

        df_raw["solution_id"] = [_recompute_sid(r) for _, r in df_raw.iterrows()]
        df_raw["material_class"] = [
            classify_material_class(m, s)
            for m, s in zip(df_raw["material_id"], df_raw["sample_name"])
        ]
        df_raw["_outlier"] = [check_outlier_reason(r) for _, r in df_raw.iterrows()]
        df_raw["_has_corr"] = (
            (df_raw["Ecorr_V_original"].notna() | df_raw["icorr_uA_cm2"].notna())
            & ~(df_raw["icorr_uA_cm2"].notna() & (df_raw["icorr_uA_cm2"] <= 0))
            & ~(
                df_raw["Ecorr_V_SHE"].isna()
                & df_raw["icorr_uA_cm2"].isna()
                & (df_raw["Ecorr_V_original"].abs() > 2.0)
            )
        )
        valid_raw_tables = df_raw[
            df_raw["material_id"].isin(KNOWN_MATERIALS)
            & (df_raw["reaction_type"] == "corrosion")
            & df_raw["_has_corr"]
            & df_raw["_outlier"].isna()
            & (df_raw["data_extraction_method"] == "table_column_mapped")
        ].copy()
        if not valid_raw_tables.empty:
            valid_raw_tables["is_target_metal"] = True

    # 若論文在 valid_raw_tables 已有結構化表格，移除 echem_from_raw_rec 內同一論文的內文 regex 記錄
    if not valid_raw_tables.empty and not echem_from_raw_rec.empty:
        raw_table_papers = {
            _TABLE_ROW_SUFFIX_RE.sub("", str(s or "").strip())
            for s in valid_raw_tables["source_reference"]
        }
        clean_srcs = echem_from_raw_rec["source_reference"].map(
            lambda s: _TABLE_ROW_SUFFIX_RE.sub("", str(s or "").strip())
        )
        echem_from_raw_rec = echem_from_raw_rec[
            ~(
                (echem_from_raw_rec["data_extraction_method"] == "regex")
                & clean_srcs.isin(raw_table_papers)
            )
        ].reset_index(drop=True)

    echem_cols = list(echem_primary.columns)
    echem_frames = [echem_primary[echem_cols]]
    if not echem_from_raw_rec.empty:
        for c in echem_cols:
            if c not in echem_from_raw_rec.columns:
                echem_from_raw_rec[c] = None
        echem_frames.append(echem_from_raw_rec[echem_cols])
    if not valid_raw_tables.empty:
        for c in echem_cols:
            if c not in valid_raw_tables.columns:
                valid_raw_tables[c] = None
        echem_frames.append(valid_raw_tables[echem_cols])

    combined_echem = pd.concat(echem_frames, ignore_index=True)
    combined_echem = deduplicate_echem_cross_metal(combined_echem)

    # 次級去重：同一論文、材料、樣品、pH、溫度、Ecorr、icorr 相同但因新舊表題環境解析導致 solution_id 不同者，優先保留新版記錄
    def _secondary_key(row) -> tuple:
        src = _TABLE_ROW_SUFFIX_RE.sub("", str(row.get("source_reference", "") or "").strip())
        mat = str(row.get("material_id", "") or "").strip()
        sname = re.sub(r"\s+", " ", str(row.get("sample_name", "") or "").strip().lower())
        if sname == "nan":
            sname = ""
        p_val = row.get("pH")
        p_key = f"{round(float(p_val), 2):.2f}" if pd.notna(p_val) else ""
        t_val = row.get("temperature_C")
        t_key = f"{round(float(t_val), 1):.1f}" if pd.notna(t_val) else ""
        e_val = row.get("Ecorr_V_SHE")
        if pd.isna(e_val):
            e_val = row.get("Ecorr_V_original")
        e_key = f"{round(float(e_val), 4):.4f}" if pd.notna(e_val) else ""
        i_val = row.get("icorr_uA_cm2")
        i_key = f"{round(float(i_val), 4):.4f}" if pd.notna(i_val) else ""
        return (src, mat, sname, p_key, t_key, e_key, i_key)

    combined_echem["_skey"] = combined_echem.apply(_secondary_key, axis=1)
    combined_echem = (
        combined_echem.drop_duplicates(subset=["_skey"], keep="first")
        .drop(columns=["_skey"])
        .reset_index(drop=True)
    )

    sol_cols = list(sol_primary.columns)
    sol_frames = [sol_primary[sol_cols]]
    for extra_sol in (sol_from_raw_rec, sol_raw_df):
        if not extra_sol.empty:
            for c in sol_cols:
                if c not in extra_sol.columns:
                    extra_sol[c] = None
            sol_frames.append(extra_sol[sol_cols])

    combined_sol = pd.concat(sol_frames, ignore_index=True)
    combined_sol = deduplicate_solutions_global(combined_sol)
    valid_sids = set(combined_echem["solution_id"].dropna().unique())
    combined_sol = combined_sol[combined_sol["solution_id"].isin(valid_sids)].reset_index(drop=True)

    return combined_echem, combined_sol


def run_merge():
    if not METALS_TO_MERGE:
        print("❌ No complete metal datasets found in data/. "
              "Run step1-5 for at least one metal first.")
        sys.exit(1)

    print(f"Found complete metals: {METALS_TO_MERGE}")
    print(f"Output dir: {SIMULATOR_DATA_DIR}\n")

    # ---- 1. solution_properties & electrochemical_database (含跨金屬與 raw 整合) ----
    print("--- Merging solution_properties & electrochemical_database ---")
    sol_primary = merge_csvs("solution_properties.csv")
    echem_primary = merge_csvs("electrochemical_database.csv")

    echem_df, sol_df = harvest_cross_metal_and_raw_records(echem_primary, sol_primary)

    # record_id 跨金屬重新編號（原各自從 REC_00001 開始）
    echem_df["record_id"] = [f"REC_{i+1:05d}" for i in range(len(echem_df))]

    sol_path = SIMULATOR_DATA_DIR / "solution_properties.csv"
    sol_df.to_csv(sol_path, index=False, encoding="utf-8-sig")
    print(f"  ✅ {len(sol_df)} solutions → {sol_path}\n")

    echem_path = SIMULATOR_DATA_DIR / "electrochemical_database.csv"
    echem_df.to_csv(echem_path, index=False, encoding="utf-8-sig")
    print(f"  ✅ {len(echem_df)} records → {echem_path}\n")

    # ---- 2. material_properties (完整 11 種專案目標材料) ----
    print("--- Building complete material_properties ---")
    mat_df = build_full_material_properties(echem_df)
    mat_path = SIMULATOR_DATA_DIR / "material_properties.csv"
    mat_df.to_csv(mat_path, index=False, encoding="utf-8-sig")
    print(f"  ✅ {len(mat_df)} materials → {mat_path}\n")

    # ---- 統計 ----
    print("=" * 55)
    print("  MERGE SUMMARY")
    print("=" * 55)
    print(f"  Metals merged:    {len(METALS_TO_MERGE)}  ({', '.join(METALS_TO_MERGE)})")
    print(f"  Materials:        {len(mat_df)}")
    print(f"  Solutions:        {len(sol_df)}")
    print(f"  Echem records:    {len(echem_df)}")

    if not echem_df.empty and "_source_metal" in echem_df.columns:
        print(f"\n  Records per metal:")
        print(echem_df["_source_metal"].value_counts().to_string())

    if not echem_df.empty:
        for col in ["Ecorr_V_SHE", "icorr_uA_cm2", "beta_a_mV_dec", "beta_c_mV_dec"]:
            if col in echem_df.columns:
                n = echem_df[col].notna().sum()
                pct = n / len(echem_df) * 100
                print(f"  {col}: {n} non-null ({pct:.1f}%)")

    print("=" * 55)
    report_path = write_quality_report(mat_df, sol_df, echem_df, SIMULATOR_DATA_DIR)
    print(f"  📊 Quality report → {report_path}")
    return mat_df, sol_df, echem_df


if __name__ == "__main__":
    run_merge()
