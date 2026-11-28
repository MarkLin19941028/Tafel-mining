"""
Unit tests for reference_benchmarks.json, Step 5 parser/outlier fixes,
Step 6b non-destructive cleaning & benchmark validation, and Step 7 integration.
"""

from collections import Counter
import json
import os
from pathlib import Path
import tempfile
import unittest

import pandas as pd

import config
from step5_table_builders import check_outlier_reason
from step5_table_extractor import parse_structured_table
import step6b_clean_and_validate as s6b
import step7_enrich_data as s7


class TestReferenceBenchmarksConfig(unittest.TestCase):
    def test_reference_benchmarks_has_43_materials_and_valid_schema(self):
        cfg = s6b.load_reference_benchmarks()
        meta = cfg["metadata"]
        self.assertEqual(meta["global_hard_bounds"]["ecorr_v_she"], [-2.0, 1.5])
        self.assertEqual(meta["global_hard_bounds"]["icorr_uA_cm2"], [0.0001, 10000.0])
        mats = cfg["materials"]
        self.assertEqual(len(mats), 43)
        self.assertEqual(mats["Mg"]["ecorr_hard_bounds"], [-2.2, 1.5])
        for sym, entry in mats.items():
            for req in ("category", "aliases", "ecorr_range", "icorr_range", "exclude_keywords", "notes"):
                self.assertIn(req, entry, f"{sym} missing {req}")
            self.assertEqual(len(entry["ecorr_range"]), 2)
            self.assertEqual(len(entry["icorr_range"]), 2)


class TestStep5ParserAndOutlierFixes(unittest.TestCase):
    def test_negated_ecorr_header_in_table_extractor(self):
        table = [
            ["system", "C (M)", "–Ecorr (mV/SCE)", "icorr (μA cm–2)", "βa (mV dec–1)"],
            ["blank", "1", "456.3 ± 6", "1104.1 ± 4.9", "112.8"],
        ]
        recs = parse_structured_table(
            table,
            "10.1021_acsomega.3c08282",
            0,
            paper_counts=Counter({"Fe": 10}),
            paper_env={"ref_counts": Counter({"SCE": 3}), "refs": ["SCE"]},
        )
        self.assertEqual(len(recs), 1)
        self.assertAlmostEqual(recs[0]["Ecorr_V_original"], -0.4563, places=4)
        self.assertAlmostEqual(recs[0]["Ecorr_V_SHE"], -0.2153, places=4)
        self.assertIn("Ecorr negated from minus header", recs[0]["extra_quality_notes"])

    def test_icorr_scientific_scale_outside_parens_and_dot_unit(self):
        table = [
            ["Sample", "Ecorr (mV vs SCE)", "jcorr 10−6 [A/cm2]"],
            ["SM-Control", "-185.31", "0.943"],
        ]
        recs = parse_structured_table(
            table,
            "10.3390_ma14154147",
            0,
            paper_counts=Counter({"Co": 10}),
            paper_env={"ref_counts": Counter({"SCE": 3}), "refs": ["SCE"]},
        )
        self.assertEqual(len(recs), 1)
        self.assertAlmostEqual(recs[0]["icorr_uA_cm2"], 0.943, places=4)

        table2 = [
            ["Sample", "-E corr (V)", "i x10-7 corr (A.cm-2)"],
            ["Sample-1", "0.450", "10.0"],
        ]
        recs2 = parse_structured_table(
            table2,
            "10.56053_10.4.1967",
            0,
            paper_counts=Counter({"Fe": 10}),
            paper_env={"ref_counts": Counter({"SCE": 3}), "refs": ["SCE"]},
        )
        self.assertEqual(len(recs2), 1)
        self.assertAlmostEqual(recs2[0]["Ecorr_V_original"], -0.450, places=4)
        self.assertAlmostEqual(recs2[0]["icorr_uA_cm2"], 1.0, places=4)

    def test_two_row_header_unit_right_alignment(self):
        table = [
            ["Alloy", "Ecorr", "icorr", "ipass", "Rp"],
            ["(mV vs. SCE)", "(µA/cm2)", "(µA/cm2)", "(kΩ·cm2)"],
            ["Ti–30Ta", "−38 ± 4", "0.58 ± 0.21", "0.57 ± 0.18", "495 ± 12"],
        ]
        recs = parse_structured_table(
            table,
            "10.3390_ma15020476",
            0,
            paper_counts=Counter({"Ti": 10}),
            paper_env={"ref_counts": Counter({"SCE": 3}), "refs": ["SCE"]},
        )
        self.assertEqual(len(recs), 1)
        self.assertAlmostEqual(recs[0]["Ecorr_V_original"], -0.038, places=4)
        self.assertAlmostEqual(recs[0]["Ecorr_V_SHE"], 0.203, places=4)
        self.assertAlmostEqual(recs[0]["icorr_uA_cm2"], 0.58, places=4)

    def test_rowspan_short_row_padding_prevents_column_shift(self):
        table = [
            ["Inhibitor Name", "Conc. of Inhibitor (M)", "Ecorr (mV)", "Icorr (mAcm−2)", "βa (mV/Decade)", "βc (mV/Decade)"],
            ["Without inhibitor", "0.00", "−487.3", "2.02", "208.1", "202.4"],
            ["IIb", "1 × 10−4", "−499.0", "0.0646", "118.0", "−137.4"],
            ["5 × 10−4", "−485.5", "0.0345", "114.8", "−161.6"],
        ]
        recs = parse_structured_table(
            table,
            "10.3390_ma15031146",
            0,
            paper_counts=Counter({"Fe": 10}),
            paper_env={"ref_counts": Counter({"SCE": 3}), "refs": ["SCE"]},
        )
        self.assertEqual(len(recs), 3)
        self.assertIn("IIb", recs[2]["sample_name"])
        self.assertAlmostEqual(recs[2]["Ecorr_V_original"], -0.4855, places=4)
        self.assertAlmostEqual(recs[2]["icorr_uA_cm2"], 34.5, places=3)
        self.assertAlmostEqual(recs[2]["beta_a_mV_dec"], 114.8, places=2)

    def test_outlier_global_hard_bounds_apply_to_alloys_and_coatings(self):
        self.assertIsNotNone(check_outlier_reason({
            "material_id": "Fe",
            "material_class": "alloy",
            "Ecorr_V_SHE": 630900.241,
            "icorr_uA_cm2": 10.0,
        }))
        self.assertIsNotNone(check_outlier_reason({
            "material_id": "Ni",
            "material_class": "coating_composite",
            "Ecorr_V_SHE": 6.436,
            "icorr_uA_cm2": 1.0,
        }))
        self.assertIsNotNone(check_outlier_reason({
            "material_id": "Ti",
            "material_class": "alloy",
            "Ecorr_V_SHE": -37.759,
            "icorr_uA_cm2": 0.58,
        }))
        # Mg alloy at -2.1 V is inside Mg's relaxed [-2.2, 1.5] V bound
        self.assertIsNone(check_outlier_reason({
            "material_id": "Mg",
            "material_class": "alloy",
            "Ecorr_V_SHE": -2.1,
            "icorr_uA_cm2": 20.0,
        }))


class TestStep6bCleanAndValidate(unittest.TestCase):
    def setUp(self):
        self.benchmarks = s6b.load_reference_benchmarks()

    def test_hard_bounds_and_mg_exception(self):
        df = pd.DataFrame([
            {"record_id": "R1", "material_id": "Mg", "material_class": "pure", "sample_name": "Blank", "Ecorr_V_SHE": -2.10, "Ecorr_V_original": -2.34, "icorr_uA_cm2": 20.0, "reactions": "corrosion", "source_reference": "10.1000_mg1", "data_extraction_method": "regex", "quality_notes": ""},
            {"record_id": "R2", "material_id": "Mg", "material_class": "pure", "sample_name": "Blank", "Ecorr_V_SHE": -2.30, "Ecorr_V_original": -2.54, "icorr_uA_cm2": 20.0, "reactions": "corrosion", "source_reference": "10.1000_mg2", "data_extraction_method": "regex", "quality_notes": ""},
            {"record_id": "R3", "material_id": "Fe", "material_class": "alloy", "sample_name": "SS 316L", "Ecorr_V_SHE": 630900.241, "Ecorr_V_original": 630900.0, "icorr_uA_cm2": None, "reactions": "corrosion", "source_reference": "10.1000_fe1", "data_extraction_method": "table_column_mapped", "quality_notes": ""},
            {"record_id": "R4", "material_id": "Cu", "material_class": "pure", "sample_name": "Blank", "Ecorr_V_SHE": 0.05, "Ecorr_V_original": -0.19, "icorr_uA_cm2": 1.5e5, "reactions": "corrosion", "source_reference": "10.1000_cu1", "data_extraction_method": "table_column_mapped", "quality_notes": ""},
        ])
        out = s6b.clean_and_validate_dataframe(df, self.benchmarks)
        self.assertEqual(len(out), 4)  # Non-destructive: all rows kept
        self.assertEqual(out.loc[0, "quality_flag"], "PASS")
        self.assertEqual(out.loc[1, "quality_flag"], "EXCLUDE")
        self.assertIn("ecorr_out_of_hard_bounds", out.loc[1, "exclusion_reason"])
        self.assertEqual(out.loc[2, "quality_flag"], "EXCLUDE")
        self.assertIn("ecorr_out_of_hard_bounds", out.loc[2, "exclusion_reason"])
        self.assertEqual(out.loc[3, "quality_flag"], "EXCLUDE")
        self.assertIn("icorr_out_of_hard_bounds", out.loc[3, "exclusion_reason"])

    def test_noble_metal_domain_pollution_uses_title_abstract_only(self):
        df = pd.DataFrame([
            # Clean corrosion Ru CMP paper whose reactions column mentions HER/ORR -> MUST PASS
            {"record_id": "R1", "material_id": "Ru", "material_class": "pure", "sample_name": "Ru wafer", "solution_description": "0.1 M H2O2", "pH": 9.0, "Ecorr_V_SHE": 0.54, "Ecorr_V_original": 0.30, "icorr_uA_cm2": 0.59, "reactions": "HER; ORR; corrosion", "source_reference": "10.1000_ru_cmp_table1_row1", "data_extraction_method": "table_column_mapped", "quality_notes": ""},
            # Electrocatalysis Ru paper in Title/Abstract -> MUST EXCLUDE
            {"record_id": "R2", "material_id": "Ru", "material_class": "pure", "sample_name": "Ru/C", "solution_description": "1 M KOH", "pH": 14.0, "Ecorr_V_SHE": 0.20, "Ecorr_V_original": 0.0, "icorr_uA_cm2": 200.0, "reactions": "HER; OER; corrosion", "source_reference": "10.1000_ru_cat_table1_row1", "data_extraction_method": "table_column_mapped", "quality_notes": ""},
        ])
        meta_idx = {
            "10.1000_ru_cmp": {"title": "Chemical Mechanical Polishing of Ruthenium Barrier Films in H2O2 Slurry", "abstract": "Corrosion and dissolution rate of Ru liner."},
            "10.1000_ru_cat": {"title": "Ru Electrocatalyst for Alkaline Hydrogen Evolution Reaction (HER)", "abstract": "Water splitting and OER/HER performance."},
        }
        out = s6b.clean_and_validate_dataframe(df, self.benchmarks, paper_meta_index=meta_idx)
        self.assertEqual(out.loc[0, "quality_flag"], "PASS")
        self.assertEqual(out.loc[0, "environment_type"], "cmp_oxidizing_etch")
        self.assertEqual(out.loc[1, "quality_flag"], "EXCLUDE")
        self.assertIn("domain_pollution_noble_metal", out.loc[1, "exclusion_reason"])
        self.assertEqual(out.loc[1, "environment_type"], "electrocatalysis_energy")

    def test_fe_mn_sn_pb_classification_and_header_recovery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tbl_path = Path(tmpdir) / "10.1000_fe_neg_tables.json"
            tbl_path.write_text(
                json.dumps([
                    [
                        ["Sample", "–Ecorr (mV vs SCE)", "icorr (uA/cm2)"],
                        ["Blank", "450", "25.0"],
                    ]
                ]),
                encoding="utf-8",
            )
            df = pd.DataFrame([
                {"record_id": "F1", "material_id": "Fe", "material_class": "pure", "sample_name": "Blank", "solution_description": "3.5% NaCl", "pH": 7.0, "reference_electrode": "SCE", "Ecorr_V_SHE": 0.691, "Ecorr_V_original": 0.450, "icorr_uA_cm2": 25.0, "reactions": "corrosion", "source_reference": "10.1000_fe_neg_table1_row1", "data_extraction_method": "table_column_mapped", "quality_notes": ""},
                {"record_id": "F2", "material_id": "Fe", "material_class": "pure", "sample_name": "Blank", "solution_description": "3.5% NaCl", "pH": 7.0, "reference_electrode": "SCE", "Ecorr_V_SHE": 0.15, "Ecorr_V_original": -0.091, "icorr_uA_cm2": 0.5, "reactions": "corrosion", "source_reference": "10.1000_fe_ss_table1_row1", "data_extraction_method": "table_column_mapped", "quality_notes": ""},
                {"record_id": "M1", "material_id": "Mn", "material_class": "pure", "sample_name": "Fe-Mn-Al-C", "solution_description": "3.5% NaCl", "pH": 7.0, "reference_electrode": "SCE", "Ecorr_V_SHE": -0.35, "Ecorr_V_original": -0.591, "icorr_uA_cm2": 15.0, "reactions": "corrosion", "source_reference": "10.1000_mn_steel_table1_row1", "data_extraction_method": "table_column_mapped", "quality_notes": ""},
                {"record_id": "S1", "material_id": "Sn", "material_class": "pure", "sample_name": "SnS inhibitor", "solution_description": "1 M HCl", "pH": 0.0, "reference_electrode": "SCE", "Ecorr_V_SHE": 0.05, "Ecorr_V_original": -0.191, "icorr_uA_cm2": 10.0, "reactions": "corrosion", "source_reference": "10.1000_sn_cs_table1_row1", "data_extraction_method": "table_column_mapped", "quality_notes": ""},
                {"record_id": "P1", "material_id": "Pb", "material_class": "pure", "sample_name": "PbO2", "solution_description": "1 M H2SO4", "pH": 0.5, "reference_electrode": "Ag/AgCl", "Ecorr_V_SHE": 0.20, "Ecorr_V_original": 0.003, "icorr_uA_cm2": 5.0, "reactions": "corrosion", "source_reference": "10.1000_pb_bat", "data_extraction_method": "regex", "quality_notes": ""},
            ])
            meta_idx = {
                "10.1000_fe_neg": {"title": "Corrosion Inhibition of Carbon Steel in 3.5% NaCl", "abstract": "Mild steel Q235."},
                "10.1000_fe_ss": {"title": "Passive Film Stability of 316L Stainless Steel in Seawater", "abstract": "316L SS."},
                "10.1000_mn_steel": {"title": "Corrosion of High-Mn Austenitic Steel in NaCl", "abstract": "High-Mn steel."},
                "10.1000_sn_cs": {"title": "Organotin Derivative as Inhibitor for Carbon Steel in HCl", "abstract": "Mild steel."},
                "10.1000_pb_bat": {"title": "Stability of PbO2 Anodes for Lead-Acid Battery", "abstract": "PbO2 electrode."},
            }
            out = s6b.clean_and_validate_dataframe(
                df,
                self.benchmarks,
                paper_meta_index=meta_idx,
                table_json_index={"10.1000_fe_neg": tbl_path},
            )
            # F1: header -Ecorr recovered into Ecorr_V_SHE_cleaned, original Ecorr_V_SHE unchanged
            self.assertAlmostEqual(out.loc[0, "Ecorr_V_SHE"], 0.691, places=4)
            self.assertAlmostEqual(out.loc[0, "Ecorr_V_SHE_cleaned"], -0.209, places=4)
            self.assertEqual(out.loc[0, "correction_note"], "negated_from_minus_ecorr_header")
            self.assertEqual(out.loc[0, "material_category"], "carbon_mild_steel")
            self.assertEqual(out.loc[0, "quality_flag"], "PASS")
            # F2: stainless steel classified separately, PASS
            self.assertEqual(out.loc[1, "material_category"], "stainless_steel")
            self.assertEqual(out.loc[1, "quality_flag"], "PASS")
            # M1: high-Mn steel -> alloy + WARN
            self.assertEqual(out.loc[2, "material_category"], "alloy")
            self.assertEqual(out.loc[2, "quality_flag"], "WARN")
            # S1: Sn inhibitor on carbon steel -> substrate_misattributed -> EXCLUDE
            self.assertEqual(out.loc[3, "quality_flag"], "EXCLUDE")
            self.assertIn("substrate_misattributed", out.loc[3, "exclusion_reason"])
            # P1: PbO2 lead-acid battery -> EXCLUDE
            self.assertEqual(out.loc[4, "quality_flag"], "EXCLUDE")

    def test_benchmark_four_tier_grading_and_step7_integration(self):
        self.assertEqual(s6b.evaluate_benchmark_alignment(0.0, 5.0, 4, [-0.12, 0.10], [1.0, 15.0]), "ℹ️ 樣本不足")
        self.assertEqual(s6b.evaluate_benchmark_alignment(0.0, 5.0, 10, [-0.12, 0.10], [1.0, 15.0]), "✅ 吻合")
        self.assertEqual(s6b.evaluate_benchmark_alignment(0.20, 5.0, 10, [-0.12, 0.10], [1.0, 15.0]), "🟡 接近")
        self.assertEqual(s6b.evaluate_benchmark_alignment(0.60, 5.0, 10, [-0.12, 0.10], [1.0, 15.0]), "⚠️ 偏離")

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = config.load_settings(data_root=tmpdir, enable_mp_api=False)
            sim_dir = settings.simulator_data_dir
            sim_dir.mkdir(parents=True, exist_ok=True)
            # Minimal material & solution & echem CSVs
            pd.DataFrame([
                {"material_id": "Cu", "material_name": "Copper", "symbol": "Cu", "category": "metal", "density_g_cm3": 8.96, "molar_mass_g_mol": 63.55, "valence": 2, "equivalent_weight_g_eq": 31.77, "standard_potential_V_SHE": 0.34}
            ]).to_csv(sim_dir / "material_properties.csv", index=False)
            pd.DataFrame([
                {"solution_id": "SOL_1", "solution_name": "0.5 M NaCl", "solution_category": "neutral", "pH": 7.0, "temperature_C": 25.0}
            ]).to_csv(sim_dir / "solution_properties.csv", index=False)
            echem = pd.DataFrame([
                {"record_id": "R1", "material_id": "Cu", "solution_id": "SOL_1", "metal": "Cu", "material_class": "pure", "sample_name": "Cu", "solution_description": "0.5 M NaCl", "pH": 7.0, "temperature_C": 25.0, "reference_electrode": "SCE", "Ecorr_V_SHE": 0.441, "Ecorr_V_original": 0.200, "icorr_uA_cm2": 5.0, "reaction_type": "corrosion", "reactions": "corrosion", "source_reference": "10.1000_cu_table1_row1", "data_extraction_method": "table_column_mapped", "quality_notes": ""},
                {"record_id": "R2", "material_id": "Cu", "solution_id": "SOL_1", "metal": "Cu", "material_class": "pure", "sample_name": "Cu bad", "solution_description": "0.5 M NaCl", "pH": 7.0, "temperature_C": 25.0, "reference_electrode": "SCE", "Ecorr_V_SHE": 5.0, "Ecorr_V_original": 4.759, "icorr_uA_cm2": 1e6, "reaction_type": "corrosion", "reactions": "corrosion", "source_reference": "10.1000_cu_bad", "data_extraction_method": "regex", "quality_notes": ""},
            ])
            echem.to_csv(sim_dir / "electrochemical_database.csv", index=False)
            # Table with negated -Ecorr for R1
            cu_txt_dir = Path(tmpdir) / "Cu" / "texts"
            cu_txt_dir.mkdir(parents=True, exist_ok=True)
            (cu_txt_dir / "10.1000_cu_tables.json").write_text(
                json.dumps([[["Sample", "−Ecorr (V vs SCE)", "icorr (uA/cm2)"], ["Cu", "0.200", "5.0"]]]),
                encoding="utf-8",
            )

            cleaned_df, report_path = s6b.run_clean_and_validate(settings=settings)
            self.assertEqual(len(cleaned_df), 2)
            self.assertTrue(report_path.exists())
            self.assertEqual(list(cleaned_df["quality_flag"]), ["PASS", "EXCLUDE"])
            self.assertAlmostEqual(cleaned_df.loc[0, "Ecorr_V_SHE_cleaned"], 0.041, places=4)

            # Default Step 7 keeps only PASS and aligns Ecorr_V_SHE to Ecorr_V_SHE_cleaned
            ml_df = s7.run_enrich(offline=True, settings=settings)
            self.assertEqual(len(ml_df), 1)
            self.assertAlmostEqual(ml_df.loc[0, "Ecorr_V_SHE"], 0.041, places=4)
            self.assertAlmostEqual(ml_df.loc[0, "Ecorr_V_SHE_raw"], 0.441, places=4)

            # quality_filter="all" keeps both rows
            ml_all = s7.run_enrich(offline=True, quality_filter="all", settings=settings)
            self.assertEqual(len(ml_all), 2)

    def test_step7_filters_out_of_bounds_after_rhe_backfill(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = config.load_settings(data_root=tmpdir, enable_mp_api=False)
            sim_dir = settings.simulator_data_dir
            sim_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([
                {"material_id": "Cu", "material_name": "Copper", "symbol": "Cu", "category": "metal", "density_g_cm3": 8.96, "molar_mass_g_mol": 63.55, "valence": 2, "equivalent_weight_g_eq": 31.77, "standard_potential_V_SHE": 0.34}
            ]).to_csv(sim_dir / "material_properties.csv", index=False)
            pd.DataFrame([
                {"solution_id": "SOL_A", "solution_name": "0.5 M H2SO4", "solution_category": "strongly_acidic", "acid_type": "H2SO4", "acid_concentration": 0.5, "acid_concentration_unit": "M", "pH": None}
            ]).to_csv(sim_dir / "solution_properties.csv", index=False)
            echem = pd.DataFrame([
                {"record_id": "R_OK", "material_id": "Cu", "solution_id": "SOL_A", "metal": "Cu", "material_class": "pure", "sample_name": "Cu", "solution_description": "0.5 M H2SO4", "pH": None, "temperature_C": 25.0, "reference_electrode": "RHE", "Ecorr_V_SHE": None, "Ecorr_V_original": 0.10, "icorr_uA_cm2": 5.0, "reaction_type": "corrosion", "reactions": "corrosion", "source_reference": "10.1000_rhe_ok", "data_extraction_method": "regex", "quality_notes": ""},
                {"record_id": "R_OOB", "material_id": "Cu", "solution_id": "SOL_A", "metal": "Cu", "material_class": "pure", "sample_name": "Cu", "solution_description": "0.5 M H2SO4", "pH": None, "temperature_C": 25.0, "reference_electrode": "RHE", "Ecorr_V_SHE": None, "Ecorr_V_original": 2.00, "icorr_uA_cm2": None, "reaction_type": "corrosion", "reactions": "corrosion", "source_reference": "10.1000_rhe_oob", "data_extraction_method": "regex", "quality_notes": ""},
            ])
            echem.to_csv(sim_dir / "electrochemical_database.csv", index=False)
            s6b.run_clean_and_validate(settings=settings)

            ml_pass = s7.run_enrich(offline=True, settings=settings)
            self.assertEqual(list(ml_pass["record_id"]), ["R_OK"])
            ml_all = s7.run_enrich(offline=True, quality_filter="all", settings=settings)
            self.assertEqual(list(ml_all["record_id"]), ["R_OK", "R_OOB"])

    def test_code_review_regressions(self):
        # 1. Unlisted pure/compound material at +1.8 V must still be rejected by global +1.5 V hard bound
        self.assertIsNotNone(check_outlier_reason({
            "material_id": "UnknownPureMetal",
            "material_class": "pure",
            "Ecorr_V_SHE": 1.8,
            "icorr_uA_cm2": 1.0,
        }))

        # 2. Fe with 'passivated' keyword classifies into stainless_steel instead of pure_metal
        self.assertEqual(
            s6b.classify_material_category(
                "Fe",
                "pure",
                "Blank",
                title="Corrosion of passivated iron in neutral chloride",
                abstract="Passivated steel film study.",
            ),
            "stainless_steel",
        )

        # 3. Four-tier benchmark grading ignores single metric whose n < 5
        #    E.g., Ru has Ecorr=0.45 (n=10, in [0.30, 0.60]) and icorr=25.4 (n=1 < 5, out of [0.01, 1.0])
        self.assertEqual(
            s6b.evaluate_benchmark_alignment(
                0.45,
                25.4,
                10,
                [0.30, 0.60],
                [0.01, 1.0],
                n_ecorr=10,
                n_icorr=1,
            ),
            "✅ 吻合",
        )


if __name__ == "__main__":
    unittest.main()
