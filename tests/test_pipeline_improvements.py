"""
Unit tests for Step 5 & Step 6 data filtering, normalization, classification, and deduplication.
"""

import json
import os
import unittest
from collections import Counter

os.environ["TAFEL_QUIET_CONFIG"] = "1"

import pandas as pd

from step5_constants import (
    convert_log_icorr_to_uA_cm2,
    normalize_icorr_to_uA_cm2,
)
from step5_table_builders import (
    _estimate_quality,
    build_solution_properties,
    check_outlier_reason,
    classify_material_class,
    is_eligible_echem_record,
)
from step5_table_extractor import (
    extract_table_captions,
    parse_structured_table,
)
from step5_text_extractor import (
    canonicalize_electrolyte,
    generate_solution_id,
)
from step6_merge import (
    deduplicate_echem_cross_metal,
    deduplicate_solutions_global,
)


class TestUnitAndIcorrParsing(unittest.TestCase):
    def test_header_scale_multiplier(self):
        # 1371 in (×10-6 A/cm2) => 1371 µA/cm² (not 1.371e12)
        val = normalize_icorr_to_uA_cm2(1371.0, "×10-6 A/cm2")
        self.assertAlmostEqual(val, 1371.0, places=4)

        val2 = normalize_icorr_to_uA_cm2(2.5, "10^-5 A cm-2")
        self.assertAlmostEqual(val2, 25.0, places=4)

        val3 = normalize_icorr_to_uA_cm2(3.0, "× 10^-3 mA/cm2")
        self.assertAlmostEqual(val3, 3.0, places=4)

    def test_reject_non_positive_icorr(self):
        self.assertIsNone(normalize_icorr_to_uA_cm2(-4058.0, "uA/cm2"))
        self.assertIsNone(normalize_icorr_to_uA_cm2(0.0, "mA/cm2"))

    def test_log_icorr_conversion(self):
        # log10(icorr [A/cm2]) = -5 => 10^-5 A/cm2 = 10 µA/cm2
        val = convert_log_icorr_to_uA_cm2(-5.0, "A/cm2")
        self.assertAlmostEqual(val, 10.0, places=5)

        # Missing unit with negative log value defaults to A/cm2
        val_default = convert_log_icorr_to_uA_cm2(-6.0, "")
        self.assertAlmostEqual(val_default, 1.0, places=5)

    def test_table_log_icorr_and_negative_icorr(self):
        table = [
            ["Sample", "Ecorr (mV)", "log(icorr) (A/cm2)"],
            ["Blank", "-250", "-5.0"],
        ]
        recs = parse_structured_table(
            table,
            "10.1000_test",
            0,
            paper_counts=Counter({"Cu": 10}),
            paper_env={"ref_counts": Counter({"SCE": 3}), "refs": ["SCE"]},
        )
        self.assertEqual(len(recs), 1)
        self.assertAlmostEqual(recs[0]["icorr_uA_cm2"], 10.0, places=4)

        neg_table = [
            ["Sample", "Ecorr (mV)", "icorr (uA/cm2)"],
            ["Blank", "-200", "-45.0"],
        ]
        recs_neg = parse_structured_table(
            neg_table,
            "10.1000_test_neg",
            0,
            paper_counts=Counter({"Cu": 10}),
            paper_env={"ref_counts": Counter({"SCE": 3}), "refs": ["SCE"]},
        )
        self.assertEqual(len(recs_neg), 1)
        self.assertIsNone(recs_neg[0]["icorr_uA_cm2"])


class TestMaterialClassAndEligibility(unittest.TestCase):
    def test_material_class_conditions_vs_alloys_coatings(self):
        self.assertEqual(classify_material_class("Cu", "Blank"), "pure")
        self.assertEqual(classify_material_class("Cu", "0 mM BTA"), "pure")
        self.assertEqual(classify_material_class("Cu", "pH 4"), "pure")
        self.assertEqual(classify_material_class("Cu", "As-received"), "pure")
        self.assertEqual(classify_material_class("Mo", "Mo0.05"), "pure")
        self.assertEqual(classify_material_class("TiN", "Blank"), "compound")

        self.assertEqual(classify_material_class("Co", "Zn-Co"), "alloy")
        self.assertEqual(classify_material_class("Ti", "Ti-6Al-4V"), "alloy")
        self.assertEqual(classify_material_class("Co", "CoCrMo"), "alloy")

        self.assertEqual(classify_material_class("Co", "Steel + Ni@Co-BC@SA"), "coating_composite")
        self.assertEqual(classify_material_class("Co", "Uncoated"), "pure")
        self.assertEqual(classify_material_class("Cu", "Epoxy coating"), "coating_composite")

    def test_eligibility_filter(self):
        # Electrocatalysis rejected
        self.assertFalse(is_eligible_echem_record({
            "reactions": "HER",
            "tafel_slopes_general_mV_dec": "45",
        }))
        # Corrosion with only Rp rejected (no Ecorr or icorr)
        self.assertFalse(is_eligible_echem_record({
            "reactions": "corrosion",
            "Rp_value": 1200.0,
        }))
        # Corrosion with Ecorr_V_original (even without SHE) accepted
        self.assertTrue(is_eligible_echem_record({
            "reactions": "corrosion",
            "Ecorr_V_original": -0.25,
            "Ecorr_V_SHE": None,
        }))
        # Corrosion with icorr accepted
        self.assertTrue(is_eligible_echem_record({
            "reactions": "corrosion",
            "icorr_uA_cm2": 5.2,
        }))

    def test_outlier_skips_pure_ecorr_for_alloy_and_coating(self):
        # Co valid Ecorr is [-1.0, 0.6]; -1.356 fails for pure Co, passes for alloy/coating
        self.assertIsNotNone(check_outlier_reason({
            "material_id": "Co",
            "material_class": "pure",
            "Ecorr_V_SHE": -1.356,
            "icorr_uA_cm2": 1.0,
        }))
        self.assertIsNone(check_outlier_reason({
            "material_id": "Co",
            "material_class": "alloy",
            "Ecorr_V_SHE": -1.356,
            "icorr_uA_cm2": 1.0,
        }))
        # Extreme icorr still fails even for alloy
        self.assertIsNotNone(check_outlier_reason({
            "material_id": "Co",
            "material_class": "alloy",
            "Ecorr_V_SHE": -1.356,
            "icorr_uA_cm2": 1e9,
        }))

    def test_quality_score_table_column_mapped_bonus(self):
        rec_regex = {"Ecorr_V_original": -0.2, "icorr_uA_cm2": 5.0, "extraction_method": "regex"}
        rec_table = {"Ecorr_V_original": -0.2, "icorr_uA_cm2": 5.0, "extraction_method": "table_column_mapped"}
        self.assertAlmostEqual(_estimate_quality(rec_table) - _estimate_quality(rec_regex), 0.1, places=2)


class TestSolutionStandardizationAndCaptions(unittest.TestCase):
    def test_solution_id_canonicalization_and_temp_distinction(self):
        c1 = [{"conc": "500", "unit": "mM", "species": "H2SO4"}]
        c2 = [{"conc": "0.50", "unit": "mol/L", "species": "H2SO4"}]
        c_norm = [{"conc": "0.5", "unit": "N", "species": "H2SO4"}]
        self.assertEqual(canonicalize_electrolyte(c1[0]), canonicalize_electrolyte(c2[0]))
        self.assertEqual(generate_solution_id(c1, 1.0, None), generate_solution_id(c2, 1.00, None))
        # Normality (N) must not be conflated with Molarity (M)
        self.assertNotEqual(generate_solution_id(c1, 1.0, None), generate_solution_id(c_norm, 1.0, None))
        # Missing temp (None) must stay distinct from explicit 25.0°C
        self.assertNotEqual(generate_solution_id(c1, 1.0, None), generate_solution_id(c1, 1.0, 25.0))

    def test_components_json_in_solution_properties(self):
        recs = [{
            "is_target_metal": True,
            "material_id_assigned": "Cu",
            "source_id": "10.1000_cmp",
            "record_electrolytes": [
                {"conc": "0.5", "unit": "M", "species": "H2SO4"},
                {"conc": "1", "unit": "wt%", "species": "H2O2"},
                {"conc": "10", "unit": "mM", "species": "BTA"},
            ],
            "record_ph": 2.0,
            "record_temp": None,
        }]
        sol_df = build_solution_properties(recs)
        self.assertEqual(len(sol_df), 1)
        self.assertTrue(pd.isna(sol_df.loc[0, "temperature_C"]))
        comps = json.loads(sol_df.loc[0, "components_json"])
        self.assertEqual(len(comps), 3)

    def test_table_caption_and_footnote_environment_extraction(self):
        body = (
            "Experimental section: baseline tests in 0.5 M H2SO4.\n"
            "Table 2. Polarization parameters of Cu in 1 wt% H2O2 at pH 9.\n"
            "Sample | Ecorr | icorr\n"
            "Note: Measured vs RHE at 40 °C.\n"
            "Other text.\n"
        )
        captions = extract_table_captions(body)
        self.assertIn(1, captions)
        self.assertIn("H2O2", captions[1])
        self.assertIn("RHE", captions[1])

    def test_table_primary_ref_written_first_in_reference_electrodes(self):
        table = [
            ["Sample", "Ecorr (V vs RHE)", "icorr (uA/cm2)"],
            ["Blank", "0.25", "5.0"],
        ]
        recs = parse_structured_table(
            table,
            "10.1000_rhe_table",
            0,
            paper_counts=Counter({"Cu": 10}),
            paper_env={"ref_counts": Counter({"SCE": 5, "RHE": 1}), "refs": ["SCE", "RHE"]},
        )
        self.assertEqual(len(recs), 1)
        self.assertTrue(recs[0]["reference_electrodes"].startswith("RHE"))


class TestStep6Deduplication(unittest.TestCase):
    def test_global_solution_dedup_merges_provenance(self):
        df = pd.DataFrame([
            {"solution_id": "SOL_abc12345", "pH": 2.0, "source": "doi1", "materials": "Cu", "_source_metal": "Cu", "is_target_metal": True},
            {"solution_id": "SOL_abc12345", "pH": 2.0, "source": "doi2", "materials": "Co", "_source_metal": "Co", "is_target_metal": True},
        ])
        out = deduplicate_solutions_global(df)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.loc[0, "source"], "doi1; doi2")
        self.assertEqual(out.loc[0, "materials"], "Cu; Co")
        self.assertEqual(out.loc[0, "_source_metal"], "Cu; Co")

    def test_cross_metal_echem_dedup_prefers_table_and_quality(self):
        df = pd.DataFrame([
            {
                "source_reference": "10.1000_dup",
                "material_id": "Cu",
                "solution_id": "SOL_11111111",
                "pH": 2.0,
                "sample_name": "Blank",
                "Ecorr_V_SHE": 0.1234,
                "Ecorr_V_original": -0.1176,
                "icorr_uA_cm2": 5.0,
                "data_extraction_method": "regex",
                "data_quality_score": 0.6,
            },
            {
                "source_reference": "10.1000_dup_table1_row1",
                "material_id": "Cu",
                "solution_id": "SOL_11111111",
                "pH": 2.0,
                "sample_name": "Blank",
                "Ecorr_V_SHE": 0.1234,
                "Ecorr_V_original": -0.1176,
                "icorr_uA_cm2": 5.0,
                "data_extraction_method": "table_column_mapped",
                "data_quality_score": 0.85,
            },
            {
                "source_reference": "10.1000_dup_table2_row1",
                "material_id": "Cu",
                "solution_id": "SOL_22222222",
                "pH": 7.0,
                "sample_name": "Blank",
                "Ecorr_V_SHE": 0.1234,
                "Ecorr_V_original": -0.1176,
                "icorr_uA_cm2": 5.0,
                "data_extraction_method": "table_column_mapped",
                "data_quality_score": 0.85,
            },
        ])
        out = deduplicate_echem_cross_metal(df)
        # Same solution deduplicated (keeping table), different solution preserved
        self.assertEqual(len(out), 2)
        self.assertTrue((out["data_extraction_method"] == "table_column_mapped").all())


if __name__ == "__main__":
    unittest.main()
