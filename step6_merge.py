"""
Step 6: 合併所有金屬的 extracted 資料，輸出到 corrosion_simulator/data/
"""

import os
import sys
import pandas as pd
from pathlib import Path

# 優先從 config 讀取 SIMULATOR_DATA_DIR，讓路徑集中管理
try:
    from config import SIMULATOR_DATA_DIR as _CFG_SIM_DIR
    SIMULATOR_DATA_DIR = Path(_CFG_SIM_DIR)
except ImportError:
    # fallback：與 Tafel mining 同層的 corrosion_simulator/data/
    SIMULATOR_DATA_DIR = Path(__file__).parent.parent / "corrosion_simulator" / "data"

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

def merge_csvs(filename: str, id_col: str = None,
               dedup_scope: str = "global") -> pd.DataFrame:
    """
    合併所有金屬的同名 CSV。

    Args:
        filename:    檔名後綴，例如 "material_properties.csv"
        id_col:      用於去重的欄位名稱
        dedup_scope: "global"（跨金屬去重）或 "per_metal"（每金屬內部去重）
                     - material_properties: global（同材料只留一筆）
                     - solution_properties: per_metal（同溶液跨金屬可保留）
                     - electrochemical_database: 不傳 id_col，不去重
    """
    frames = []

    for metal in METALS_TO_MERGE:
        extracted_dir = MINING_DATA_DIR / metal / "extracted"
        prefix = metal.lower()
        target_file = extracted_dir / f"{prefix}_{filename}"

        if target_file.exists():
            df = pd.read_csv(target_file)
            df["_source_metal"] = metal

            # 修復 2：per_metal 模式在合併前就去重，保留每個金屬自己的記錄
            if id_col and dedup_scope == "per_metal" and id_col in df.columns:
                before = len(df)
                df = df.drop_duplicates(subset=[id_col], keep="first")
                if len(df) < before:
                    print(f"  {metal}: removed {before - len(df)} "
                          f"intra-metal dupes by {id_col}")

            frames.append(df)
            print(f"  {metal}: {len(df)} rows from {target_file.name}")
        else:
            print(f"  {metal}: {filename} not found, skipping")

    if not frames:
        print(f"  ⚠️ No data found for {filename}")
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)

    # global 去重：跨金屬層面
    if id_col and dedup_scope == "global" and id_col in merged.columns:
        before = len(merged)
        merged = merged.drop_duplicates(subset=[id_col], keep="first")
        dupes = before - len(merged)
        if dupes > 0:
            print(f"  Removed {dupes} cross-metal duplicates by {id_col}")

    return merged


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
            ref = echem_df["reference_electrode"].value_counts(dropna=True).head(8).to_dict() if "reference_electrode" in echem_df.columns else {}
            lines.append(f"- **反應類型 (`reaction_type`)**：`{rxn}`")
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


def run_merge():
    if not METALS_TO_MERGE:
        print("❌ No complete metal datasets found in data/. "
              "Run step1-5 for at least one metal first.")
        sys.exit(1)

    print(f"Found complete metals: {METALS_TO_MERGE}")
    print(f"Output dir: {SIMULATOR_DATA_DIR}\n")

    # ---- 1. material_properties ----
    print("--- Merging material_properties ---")
    mat_df = merge_csvs("material_properties.csv",
                        id_col="material_id", dedup_scope="global")
    mat_path = SIMULATOR_DATA_DIR / "material_properties.csv"
    mat_df.to_csv(mat_path, index=False, encoding="utf-8-sig")
    print(f"  ✅ {len(mat_df)} materials → {mat_path}\n")

    # ---- 2. solution_properties ----
    # 修復 2：同一溶液在不同金屬的測量獨立保留（per_metal），
    # 跨金屬的完全相同 solution_id 才去重（實務上幾乎不會發生）
    print("--- Merging solution_properties ---")
    sol_df = merge_csvs("solution_properties.csv",
                        id_col="solution_id", dedup_scope="per_metal")
    sol_path = SIMULATOR_DATA_DIR / "solution_properties.csv"
    sol_df.to_csv(sol_path, index=False, encoding="utf-8-sig")
    print(f"  ✅ {len(sol_df)} solutions → {sol_path}\n")

    # ---- 3. electrochemical_database ----
    print("--- Merging electrochemical_database ---")
    echem_df = merge_csvs("electrochemical_database.csv")  # 不去重，直接合併

    # record_id 跨金屬重新編號（原各自從 REC_00001 開始）
    echem_df["record_id"] = [f"REC_{i+1:05d}" for i in range(len(echem_df))]

    echem_path = SIMULATOR_DATA_DIR / "electrochemical_database.csv"
    echem_df.to_csv(echem_path, index=False, encoding="utf-8-sig")
    print(f"  ✅ {len(echem_df)} records → {echem_path}\n")

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
