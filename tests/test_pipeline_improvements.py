"""
Unit tests for Step 5 & Step 6 data filtering, normalization, classification, and deduplication.
"""

import json
import os
import tempfile
import unittest
from collections import Counter

import pandas as pd

import config


def _make_test_ctx(
    tmpdir: str,
    metal: str = "Cu",
    *,
    search_modes: tuple[str, ...] | list[str] = ("corrosion",),
    search_queries: list[str] | tuple[str, ...] | None = None,
    skip_openalex: bool = False,
    **settings_overrides,
) -> config.MetalRunContext:
    settings = config.load_settings(data_root=tmpdir, **settings_overrides)
    return config.build_metal_context(
        metal,
        search_modes=list(search_modes),
        settings=settings,
        search_queries=search_queries,
        skip_openalex=skip_openalex,
    )

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


class TestNitrideMetalsAndInhibitors(unittest.TestCase):
    def test_databases_valid_with_wn_and_mon(self):
        from materials_loader import validate_databases
        from step5_constants import KNOWN_MATERIALS
        from step7_enrich_data import COMPOUND_OVERRIDES, resolve_base_element

        settings = config.load_settings()
        self.assertIn("WN", settings.metals_to_run)
        self.assertIn("MoN", settings.metals_to_run)
        errors, warnings = validate_databases(list(settings.metals_to_run))
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

        for sym, base in (("WN", "W"), ("MoN", "Mo")):
            self.assertIn(sym, KNOWN_MATERIALS)
            self.assertIn(sym, COMPOUND_OVERRIDES)
            self.assertEqual(resolve_base_element(sym), base)
            self.assertEqual(classify_material_class(sym, "Blank"), "compound")

    def test_nitride_material_recognition_priority(self):
        from step5_text_extractor import find_materials

        hits = [sym for _, sym in find_materials("Corrosion of PVD WN, TiN, TaN, and MoN thin films in H2O2")]
        for expected in ("WN", "TiN", "TaN", "MoN"):
            self.assertIn(expected, hits)

    def test_diverse_inhibitor_extraction_and_zero_conc_skip(self):
        from step5_text_extractor import extract_electrolytes, lookup_species_category

        txt = (
            "Polarization tests were performed in 0.5 M HCl containing "
            "0 mM BTA, 100 µM 2-mercaptobenzothiazole, 200 mg/L sodium dodecyl sulfate, "
            "and 5 mM of 1,2,4-triazole."
        )
        els = extract_electrolytes(txt)
        by_species = {e["species"]: e for e in els}
        # 0 mM BTA must be skipped so Blank isn't treated as active BTA
        self.assertNotIn("BTA", by_species)
        self.assertIn("HCl", by_species)
        for inh in ("MBT", "SDS", "TAZ"):
            self.assertIn(inh, by_species)
            self.assertEqual(lookup_species_category(inh)["role"], "inhibitor")

    def test_two_column_inhibitor_conc_table_and_base_medium_retention(self):
        table = [
            ["Inhibitor", "Conc. (mM)", "Ecorr (mV vs SCE)", "icorr (uA/cm2)"],
            ["Blank", "—", "-210", "45.0"],
            ["tolyltriazole", "1.0", "-185", "8.2"],
            ["", "5.0", "-170", "2.1"],
        ]
        paper_env = {
            "electrolytes": [{"conc": "0.5", "unit": "M", "species": "HCl", "count": 4}],
            "ph_counts": Counter({0.3: 2}),
            "temp_counts": Counter({25.0: 2}),
            "ref_counts": Counter({"SCE": 3}),
            "refs": ["SCE"],
        }
        recs = parse_structured_table(
            table,
            "10.1000_inh_table",
            0,
            paper_counts=Counter({"Cu": 10}),
            paper_env=paper_env,
        )
        self.assertEqual(len(recs), 3)
        for r in recs:
            r["is_target_metal"] = True
        sol_df = build_solution_properties(recs).set_index("solution_id")
        # Blank, 1.0 mM TTA, and carried-forward 5.0 mM TTA must yield 3 distinct solutions
        self.assertEqual(len(sol_df), 3)
        self.assertEqual(recs[1]["sample_name"], "tolyltriazole (1.0 mM)")
        self.assertEqual(recs[2]["sample_name"], "tolyltriazole (5.0 mM)")

        sol_1 = sol_df.loc[recs[1]["solution_id_assigned"]]
        sol_5 = sol_df.loc[recs[2]["solution_id_assigned"]]
        self.assertEqual(sol_1["inhibitor_type"], "TTA")
        self.assertEqual(sol_1["acid_type"], "HCl")
        self.assertEqual(sol_5["inhibitor_type"], "TTA")
        self.assertEqual(float(sol_5["inhibitor_concentration"]), 5.0)

    def test_s2_grouped_inhibitor_query_and_step1_append(self):
        import semantic_scholar_fetcher as s2f
        from step1_search_openalex import _merge_with_existing_step1

        grouped = s2f.build_grouped_s2_queries()
        self.assertTrue(any("corrosion inhibitor" in q and "triazole" in q for q in grouped))

        ex = pd.DataFrame([
            {"doi": "https://doi.org/10.1000/a", "title": "Old A", "best_pdf_url": None, "abstract": ""},
        ])
        nw = pd.DataFrame([
            {"doi": "10.1000/A", "title": "Old A", "best_pdf_url": "https://example.org/a.pdf", "abstract": "Filled"},
            {"doi": "https://doi.org/10.1000/b", "title": "New B", "best_pdf_url": None, "abstract": "New abs"},
        ])
        merged = _merge_with_existing_step1(ex, nw)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged.loc[0, "best_pdf_url"], "https://example.org/a.pdf")
        self.assertEqual(merged.loc[0, "abstract"], "Filled")

    def test_user_settings_and_modular_search_modes(self):
        import europepmc_fetcher as epmc
        import run_all
        import semantic_scholar_fetcher as s2f
        import user_settings

        settings = config.load_settings()
        cu_ctx = config.build_metal_context("Cu", settings=settings)

        # user_settings exports default 4 corrosion-focused modes (catalysis excluded by default)
        self.assertEqual(user_settings.SEARCH_MODES, ["corrosion", "cmp_etch", "inhibitor", "solvent"])
        self.assertEqual(settings.mailto, user_settings.MAILTO)

        # parse_search_modes handles lists, space/comma strings, 'all', and rejects unknown modes
        self.assertEqual(config.parse_search_modes(["cmp_etch", "inhibitor"]), ["cmp_etch", "inhibitor"])
        self.assertEqual(config.parse_search_modes("cmp_etch, inhibitor"), ["cmp_etch", "inhibitor"])
        self.assertEqual(
            config.parse_search_modes("all"),
            ["corrosion", "cmp_etch", "inhibitor", "solvent", "catalysis"],
        )
        self.assertEqual(config.parse_search_modes("solvent"), ["solvent"])
        with self.assertRaises(ValueError):
            config.parse_search_modes(["cmp_etch", "unknown_mode"])

        # Each engine strictly reflects only the selected modes
        oa_cmp_inh = config.build_search_queries(
            ["cmp_etch", "inhibitor"],
            target_metal=cu_ctx.target_metal,
            metal_symbol=cu_ctx.metal_symbol,
            compounds=cu_ctx.compounds,
            cmp_queries=cu_ctx.cmp_queries,
            inhibitor_queries=cu_ctx.inhibitor_queries,
            grouped=settings.openalex_use_grouped_queries,
        )
        self.assertTrue(any('"CMP"' in q for q in oa_cmp_inh))
        self.assertTrue(any('"corrosion inhibitor"' in q for q in oa_cmp_inh))
        self.assertFalse(any('"hydrogen evolution"' in q for q in oa_cmp_inh))
        self.assertFalse(any('"non-aqueous"' in q for q in oa_cmp_inh))

        oa_solvent = config.build_search_queries(
            ["solvent"],
            target_metal=cu_ctx.target_metal,
            metal_symbol=cu_ctx.metal_symbol,
            compounds=cu_ctx.compounds,
            cmp_queries=cu_ctx.cmp_queries,
            inhibitor_queries=cu_ctx.inhibitor_queries,
            grouped=settings.openalex_use_grouped_queries,
        )
        for kw in (
            "non-aqueous",
            "nonaqueous",
            "organic solvent",
            "organic media",
            "anhydrous",
            "semi-aqueous",
            "mixed solvent",
            "water-organic",
            "trace water",
            "protic solvent",
            "aprotic solvent",
        ):
            self.assertTrue(any(f'"{kw}"' in q for q in oa_solvent), f"Missing solvent keyword: {kw}")
        self.assertFalse(any('"CMP"' in q for q in oa_solvent))

        s2_cat = s2f.build_grouped_s2_queries(["catalysis"], ctx=cu_ctx)
        self.assertTrue(any("hydrogen evolution" in q for q in s2_cat))
        self.assertFalse(any("corrosion inhibitor" in q for q in s2_cat))

        s2_solvent = s2f.build_grouped_s2_queries(["solvent"], ctx=cu_ctx)
        self.assertEqual(len(s2_solvent), 1)
        self.assertIn('"non-aqueous"', s2_solvent[0])
        self.assertIn('"aprotic solvent"', s2_solvent[0])

        epmc_cmp_inh = epmc.build_europepmc_query(["cmp_etch", "inhibitor"], ctx=cu_ctx)
        self.assertIn('"chemical mechanical"', epmc_cmp_inh)
        self.assertIn('"corrosion inhibitor"', epmc_cmp_inh)
        self.assertNotIn('"Tafel extrapolation"', epmc_cmp_inh)

        epmc_solvent = epmc.build_europepmc_query(["solvent"], ctx=cu_ctx)
        self.assertIn('"non-aqueous"', epmc_solvent)
        self.assertIn('"semi-aqueous"', epmc_solvent)
        self.assertNotIn('"chemical mechanical"', epmc_solvent)

        # CLI parser supports both `Cu Co --search-mode cmp_etch inhibitor --append`
        # and `--search-mode cmp_etch inhibitor Cu Co --append`
        parser = run_all.build_cli_parser()
        a1 = parser.parse_args(["Cu", "Co", "--search-mode", "cmp_etch", "inhibitor", "--append"])
        m1, modes1 = run_all.resolve_cli_metals_and_modes(a1.metals, a1.search_mode, config)
        self.assertEqual(m1, ["Cu", "Co"])
        self.assertEqual(modes1, ["cmp_etch", "inhibitor"])
        self.assertTrue(a1.append)

        a2 = parser.parse_args(["--search-mode", "cmp_etch", "inhibitor", "Cu", "Co", "--append"])
        m2, modes2 = run_all.resolve_cli_metals_and_modes(a2.metals, a2.search_mode, config)
        self.assertEqual(m2, ["Cu", "Co"])
        self.assertEqual(modes2, ["cmp_etch", "inhibitor"])

        a_sol = parser.parse_args(["Cu", "--search-mode", "solvent", "--append"])
        m_sol, modes_sol = run_all.resolve_cli_metals_and_modes(a_sol.metals, a_sol.search_mode, config)
        self.assertEqual(m_sol, ["Cu"])
        self.assertEqual(modes_sol, ["solvent"])

        a3 = parser.parse_args(["Cu", "--from-step", "2", "--to-step", "3", "--retry-failed"])
        self.assertEqual(a3.from_step, 2)
        self.assertEqual(a3.to_step, 3)
        self.assertTrue(a3.retry_failed)


class TestUnifiedDedupAndDownloadState(unittest.TestCase):
    def test_step1_three_source_merge_and_append_with_pmcid(self):
        from step1_search_openalex import _merge_sources, _merge_with_existing_step1

        oa = pd.DataFrame([
            {
                "doi": "https://doi.org/10.1000/shared",
                "pmcid": None,
                "source_api": "openalex",
                "title": "Shared Cu paper",
                "best_pdf_url": None,
                "abstract": "",
            },
        ])
        s2 = pd.DataFrame([
            {
                "doi": "10.1000/SHARED",
                "s2_paper_id": "s2_1",
                "pmcid": None,
                "source_api": "semantic_scholar",
                "title": "Shared Cu paper",
                "best_pdf_url": "https://example.org/shared.pdf",
                "abstract": "From S2",
            },
        ])
        epmc = pd.DataFrame([
            {
                "doi": "https://doi.org/10.1000/shared",
                "pmcid": "PMC12345",
                "source_api": "europepmc",
                "title": "Shared Cu paper",
                "best_pdf_url": None,
                "abstract": "From EPMC",
            },
            {
                "doi": None,
                "pmcid": "PMC99999",
                "source_api": "europepmc",
                "title": "PMCID only Cu paper",
                "best_pdf_url": None,
                "abstract": "Copper polarization",
            },
        ])

        merged = _merge_sources(oa, s2, epmc)
        self.assertEqual(len(merged), 2)
        shared_row = merged.iloc[0]
        self.assertEqual(shared_row["pmcid"], "PMC12345")
        self.assertEqual(shared_row["best_pdf_url"], "https://example.org/shared.pdf")
        self.assertIn("openalex", shared_row["source_api"])
        self.assertIn("semantic_scholar", shared_row["source_api"])
        self.assertIn("europepmc", shared_row["source_api"])
        self.assertEqual(merged.iloc[1]["pmcid"], "PMC99999")

        # Append mode should deduplicate PMCID-only records and backfill pmcid on existing DOI
        ex = pd.DataFrame([
            {"doi": "https://doi.org/10.1000/shared", "pmcid": None, "best_pdf_url": None, "abstract": ""},
            {"doi": None, "pmcid": "PMC99999", "best_pdf_url": None, "abstract": "Old"},
        ])
        appended = _merge_with_existing_step1(ex, merged)
        self.assertEqual(len(appended), 2)
        self.assertEqual(appended.iloc[0]["pmcid"], "PMC12345")

    def test_step3_offline_bootstrap_and_retry_failed_queue(self):
        import tempfile
        import step3_download_pdfs as s3

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_dir = os.path.join(tmpdir, "pdfs")
            text_dir = os.path.join(tmpdir, "texts")
            state_path = os.path.join(tmpdir, "cu_download_state.json")
            os.makedirs(pdf_dir)
            os.makedirs(text_dir)

            # Simulate an already-downloaded PDF for doi_done
            done_base = s3.sanitize_base_name("10.1000/done")
            with open(os.path.join(pdf_dir, done_base + ".pdf"), "wb") as f:
                f.write(b"%PDF-1.4 dummy")

            df_old = pd.DataFrame([
                {"doi": "https://doi.org/10.1000/done", "final_pdf_url": "https://ex.org/done.pdf", "pmcid": None},
                {"doi": "https://doi.org/10.1000/fail", "final_pdf_url": "https://ex.org/fail1.pdf", "pmcid": None},
            ])
            state = s3.load_or_bootstrap_download_state(state_path, df_old, pdf_dir, text_dir)
            self.assertEqual(state["10.1000/done"]["status"], "done_pdf")
            self.assertEqual(state["10.1000/fail"]["status"], "failed_legacy")

            # After --append adds a brand-new paper and a new PMCID paper, only new ones enter queue
            df_new = pd.DataFrame([
                {"doi": "https://doi.org/10.1000/done", "final_pdf_url": "https://ex.org/done.pdf", "pmcid": None},
                {"doi": "https://doi.org/10.1000/fail", "final_pdf_url": "https://ex.org/fail1.pdf", "pmcid": None},
                {"doi": "https://doi.org/10.1000/brandnew", "final_pdf_url": "https://ex.org/new.pdf", "pmcid": None},
                {"doi": None, "final_pdf_url": None, "pmcid": "PMC77777"},
            ])
            pending, stats = s3.select_pending_rows(df_new, state, pdf_dir, text_dir, retry_failed=False)
            self.assertEqual([p["key"] for p in pending], ["10.1000/brandnew", "pmcid:pmc77777"])
            self.assertEqual(stats["skipped_done"], 1)
            self.assertEqual(stats["skipped_failed"], 1)

            # If old failed paper gets a new URL, it is automatically queued even without --retry-failed
            df_new_url = pd.DataFrame([
                {"doi": "https://doi.org/10.1000/fail", "final_pdf_url": "https://ex.org/fail2_new.pdf", "pmcid": None},
            ])
            pending_u, stats_u = s3.select_pending_rows(df_new_url, state, pdf_dir, text_dir, retry_failed=False)
            self.assertEqual([p["key"] for p in pending_u], ["10.1000/fail"])
            self.assertEqual(stats_u["retry_new_source"], 1)

            # With --retry-failed, old failed paper is queued even with the same URL
            pending_r, stats_r = s3.select_pending_rows(df_old, state, pdf_dir, text_dir, retry_failed=True)
            self.assertEqual([p["key"] for p in pending_r], ["10.1000/fail"])
            self.assertEqual(stats_r["retry_forced"], 1)

    def test_step3_fresh_metal_bootstrap_keeps_all_pending(self):
        import tempfile
        import step3_download_pdfs as s3

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_dir = os.path.join(tmpdir, "pdfs")
            text_dir = os.path.join(tmpdir, "texts")
            state_path = os.path.join(tmpdir, "al_download_state.json")
            os.makedirs(pdf_dir)
            os.makedirs(text_dir)

            df_fresh = pd.DataFrame([
                {"doi": "https://doi.org/10.1000/al1", "final_pdf_url": "https://ex.org/al1.pdf", "pmcid": None},
            ])
            state = s3.load_or_bootstrap_download_state(state_path, df_fresh, pdf_dir, text_dir)
            self.assertNotIn("10.1000/al1", state)
            pending, _ = s3.select_pending_rows(df_fresh, state, pdf_dir, text_dir, retry_failed=False)
            self.assertEqual([p["key"] for p in pending], ["10.1000/al1"])

    def test_first_append_on_legacy_metal_does_not_mark_new_papers_failed(self):
        import tempfile
        import step2_find_pdf_urls as s2
        import step3_download_pdfs as s3

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_dir = os.path.join(tmpdir, "pdfs")
            text_dir = os.path.join(tmpdir, "texts")
            state_path = os.path.join(tmpdir, "cu_download_state.json")
            os.makedirs(pdf_dir)
            os.makedirs(text_dir)

            # Existing metal has 1 downloaded PDF and 1 old failed DOI, and NO download_state.json yet
            done_base = s3.sanitize_base_name("10.1000/old_done")
            with open(os.path.join(pdf_dir, done_base + ".pdf"), "wb") as f:
                f.write(b"%PDF-1.4 dummy")

            prev_step2_df = pd.DataFrame([
                {"doi": "https://doi.org/10.1000/old_done", "final_pdf_url": "https://ex.org/done.pdf", "pmcid": None},
                {"doi": "https://doi.org/10.1000/old_fail", "final_pdf_url": "https://ex.org/fail.pdf", "pmcid": None},
            ])

            # Step 2 --append bootstraps state from OLD Step 2 CSV BEFORE writing merged CSV
            s2._bootstrap_state_from_prev_step2(
                prev_step2_df, state_path=state_path, pdf_dir=pdf_dir, text_dir=text_dir
            )

            # Then Step 3 runs on the merged CSV (which now includes a newly appended PDF paper)
            merged_step2_df = pd.DataFrame([
                {"doi": "https://doi.org/10.1000/old_done", "final_pdf_url": "https://ex.org/done.pdf", "pmcid": None},
                {"doi": "https://doi.org/10.1000/old_fail", "final_pdf_url": "https://ex.org/fail.pdf", "pmcid": None},
                {"doi": "https://doi.org/10.1000/newly_appended", "final_pdf_url": "https://ex.org/new.pdf", "pmcid": None},
            ])
            state = s3.load_or_bootstrap_download_state(state_path, merged_step2_df, pdf_dir, text_dir)
            pending, stats = s3.select_pending_rows(merged_step2_df, state, pdf_dir, text_dir, retry_failed=False)

            self.assertEqual(state["10.1000/old_fail"]["status"], s3.STATUS_FAILED_LEGACY)
            self.assertNotIn("10.1000/newly_appended", state)
            self.assertEqual([p["key"] for p in pending], ["10.1000/newly_appended"])
            self.assertEqual(stats["skipped_done"], 1)
            self.assertEqual(stats["skipped_failed"], 1)


class TestStep1InflightDedup(unittest.TestCase):
    def test_openalex_inflight_dedup_backfill_and_early_stop(self):
        from unittest.mock import MagicMock, patch
        import step1_search_openalex as s1

        def _oa_item(oid, doi, pdf_url=None, abstract_words=None):
            inv = {w: [i] for i, w in enumerate(abstract_words)} if abstract_words else None
            return {
                "id": oid,
                "doi": doi,
                "title": f"Title {oid}",
                "publication_year": 2024,
                "primary_location": {"source": {"display_name": "J. Electrochem."}, "pdf_url": None},
                "best_oa_location": {"pdf_url": pdf_url},
                "open_access": {"is_oa": bool(pdf_url), "oa_status": "gold" if pdf_url else "closed", "oa_url": pdf_url},
                "abstract_inverted_index": inv,
                "cited_by_count": 10,
                "authorships": [{"author": {"display_name": "Alice"}}],
            }

        # Page 1: 1 no-DOI item (ignored), 1 existing complete DOI (skipped), 1 existing incomplete DOI (backfilled)
        # Page 2: 2 brand-new DOIs -> satisfies max_records=2
        pages = [
            {
                "results": [
                    _oa_item("W0", None),
                    _oa_item("W1", "https://doi.org/10.1000/old_full", "https://ex.org/old.pdf", ["Old"]),
                    _oa_item("W2", "https://doi.org/10.1000/old_need_pdf", "https://ex.org/filled.pdf", ["Filled"]),
                ],
                "meta": {"next_cursor": "c2"},
            },
            {
                "results": [
                    _oa_item("W3", "https://doi.org/10.1000/new1", "https://ex.org/n1.pdf", ["New1"]),
                    _oa_item("W4", "https://doi.org/10.1000/new2", None, None),
                ],
                "meta": {"next_cursor": "c3"},
            },
        ]
        mock_resps = []
        for p in pages:
            r = MagicMock()
            r.json.return_value = p
            mock_resps.append(r)

        known = {"10.1000/old_full", "10.1000/old_need_pdf"}
        needs_backfill = {"10.1000/old_need_pdf": {"best_pdf_url", "abstract"}}

        with patch.object(s1, "request_with_retry", side_effect=mock_resps) as mock_req, \
             patch.object(s1.time, "sleep"):
            recs = s1.fetch_single_query(
                '"copper" "Tafel"',
                max_records=2,
                known_keys=known,
                needs_backfill=needs_backfill,
                max_scan_multiplier=3,
                early_stop_empty_pages=2,
            )

        self.assertEqual(mock_req.call_count, 2)
        dois = [r["doi"] for r in recs]
        # Backfilled old DOI + 2 new DOIs
        self.assertEqual(dois, [
            "https://doi.org/10.1000/old_need_pdf",
            "https://doi.org/10.1000/new1",
            "https://doi.org/10.1000/new2",
        ])
        # New keys added to known_keys; backfilled key cleared from needs_backfill
        self.assertIn("10.1000/new1", known)
        self.assertIn("10.1000/new2", known)
        self.assertNotIn("10.1000/old_need_pdf", needs_backfill)
        # New paper missing pdf/abstract is registered for downstream S2/EPMC backfill
        self.assertIn("best_pdf_url", needs_backfill["10.1000/new2"])
        self.assertIn("pmcid", needs_backfill["10.1000/new2"])

        # Test early-stop after 2 consecutive all-duplicate pages (does not scan page 3)
        dup_pages = [
            {"results": [_oa_item("W1", "https://doi.org/10.1000/old_full")], "meta": {"next_cursor": f"c{i+2}"}}
            for i in range(5)
        ]
        dup_resps = []
        for p in dup_pages:
            r = MagicMock()
            r.json.return_value = p
            dup_resps.append(r)

        with patch.object(s1, "request_with_retry", side_effect=dup_resps) as mock_req2, \
             patch.object(s1.time, "sleep"):
            recs2 = s1.fetch_single_query(
                '"copper" "Tafel"',
                max_records=2,
                known_keys=set(known),
                needs_backfill={},
                max_scan_multiplier=5,
                early_stop_empty_pages=2,
            )
        self.assertEqual(recs2, [])
        self.assertEqual(mock_req2.call_count, 2)

        # Test max_scan_multiplier page cap: 1 new paper per page, max_records=5, multiplier=2 -> stops at 2 pages
        cap_pages = [
            {"results": [_oa_item(f"WC{i}", f"https://doi.org/10.1000/cap_{i}")], "meta": {"next_cursor": f"cc{i+2}"}}
            for i in range(5)
        ]
        cap_resps = []
        for p in cap_pages:
            r = MagicMock()
            r.json.return_value = p
            cap_resps.append(r)

        with patch.object(s1, "request_with_retry", side_effect=cap_resps) as mock_req3, \
             patch.object(s1.time, "sleep"):
            recs3 = s1.fetch_single_query(
                '"copper" "Tafel"',
                max_records=5,
                known_keys=set(),
                needs_backfill={},
                max_scan_multiplier=2,
                early_stop_empty_pages=0,
            )
        self.assertEqual(len(recs3), 2)
        self.assertEqual(mock_req3.call_count, 2)

    def test_s2_bulk_inflight_dedup_zero_extra_requests_and_pure_cache(self):
        from unittest.mock import MagicMock, patch
        import semantic_scholar_fetcher as s2f

        def _s2_item(pid, doi, pdf_url=None, abstract="Abs"):
            return {
                "paperId": pid,
                "externalIds": {"DOI": doi},
                "title": f"S2 {pid}",
                "year": 2024,
                "venue": "Corr. Sci.",
                "isOpenAccess": bool(pdf_url),
                "openAccessPdf": {"url": pdf_url, "status": "GOLD"} if pdf_url else None,
                "citationCount": 20,
                "authors": [{"name": "Bob"}],
                "abstract": abstract,
            }

        payload = {
            "data": [
                _s2_item("p1", "10.1000/already_known", "https://ex.org/p1.pdf", "Filled abs"),
                _s2_item("p2", "10.1000/already_full", "https://ex.org/p2.pdf", "Full"),
                _s2_item("p3", "10.1000/s2_new1", "https://ex.org/p3.pdf", "New 1"),
                _s2_item("p4", "10.1000/s2_new2", None, "New 2"),
            ],
            "token": "next_page_token",
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload

        known = {"10.1000/already_known", "10.1000/already_full"}
        needs_backfill = {"10.1000/already_known": {"best_pdf_url"}}

        with patch.object(s2f, "_s2_request_with_retry", return_value=mock_resp) as mock_req:
            selected, raw_all, n_new, n_skip = s2f._fetch_query_bulk(
                '"copper" + "Tafel"',
                max_per_query=2,
                from_year=2005,
                known_keys=known,
                needs_backfill=needs_backfill,
                max_bulk_pages=2,
                return_raw=True,
            )

        # Satisfied 2 new papers within page 1 -> 0 extra HTTP requests!
        self.assertEqual(mock_req.call_count, 1)
        self.assertEqual(n_new, 2)
        self.assertEqual(n_skip, 2)
        # Raw cache preserves all 4 unfiltered items
        self.assertEqual(len(raw_all), 4)
        # Selected contains 2 new + 1 backfill for already_known
        sel_dois = [r["doi"] for r in selected]
        self.assertEqual(sel_dois, [
            "https://doi.org/10.1000/s2_new1",
            "https://doi.org/10.1000/s2_new2",
            "https://doi.org/10.1000/already_known",
        ])
        self.assertNotIn("10.1000/already_known", needs_backfill)

        # End-to-end cache reuse across different known_keys via search_semantic_scholar
        with tempfile.TemporaryDirectory() as td1:
            ctx1 = _make_test_ctx(td1, "Cu")
            with patch.object(s2f, "_select_s2_queries", return_value=['"copper" + "Tafel"']), \
                 patch.object(s2f, "_s2_request_with_retry", return_value=mock_resp) as mock_req2:
                # First call populates raw cache with all 4 items, returns first 1 new (p3)
                run1 = s2f.search_semantic_scholar(
                    ["q1", "q2"],
                    max_per_query=1,
                    from_year=2005,
                    use_cache=True,
                    known_keys={"10.1000/already_known", "10.1000/already_full"},
                    needs_backfill={},
                    ctx=ctx1,
                )
                self.assertEqual(mock_req2.call_count, 1)
                self.assertEqual([r["doi"] for r in run1], ["https://doi.org/10.1000/s2_new1"])

                # Second call reuses cache (0 extra API calls) with p3 now known -> yields p4!
                run2 = s2f.search_semantic_scholar(
                    ["q1", "q2"],
                    max_per_query=1,
                    from_year=2005,
                    use_cache=True,
                    known_keys={"10.1000/already_known", "10.1000/already_full", "10.1000/s2_new1"},
                    needs_backfill={},
                    ctx=ctx1,
                )
                self.assertEqual(mock_req2.call_count, 1)
                self.assertEqual([r["doi"] for r in run2], ["https://doi.org/10.1000/s2_new2"])

        # Degraded fallback (bulk returns None, standard succeeds) must NOT pollute bulk_raw cache
        with tempfile.TemporaryDirectory() as td2:
            ctx2 = _make_test_ctx(td2, "Cu")
            with patch.object(s2f, "_select_s2_queries", return_value=['"copper" + "Tafel"']), \
                 patch.object(s2f, "_fetch_query_bulk", return_value=None), \
                 patch.object(s2f, "_fetch_query_standard", return_value=([{"doi": "https://doi.org/10.1000/s2_new1"}], [{"doi": "https://doi.org/10.1000/s2_new1"}], 1, 0)):
                run_deg = s2f.search_semantic_scholar(
                    ["q1", "q2"],
                    max_per_query=1,
                    from_year=2005,
                    use_cache=True,
                    known_keys=set(),
                    ctx=ctx2,
                )
                self.assertEqual(len(run_deg), 1)
                self.assertEqual(s2f._load_s2_cache(ctx=ctx2)["queries"], {})

    def test_europepmc_inflight_dedup_preserves_pmcid_backfill(self):
        from unittest.mock import MagicMock, patch
        import europepmc_fetcher as epmc

        page1 = {
            "resultList": {
                "result": [
                    # Already known AND already has pmcid -> skip
                    {"pmcid": "PMC100", "doi": "10.1000/has_pmc", "title": "Copper corrosion 1", "abstractText": "Cu Tafel", "pubYear": "2024"},
                    # Already known DOI but missing pmcid -> keep as backfill (does not consume new limit)
                    {"pmcid": "PMC200", "doi": "10.1000/need_pmc", "title": "Copper corrosion 2", "abstractText": "Cu Tafel", "pubYear": "2024"},
                    # Brand-new paper -> counts toward limit=1
                    {"pmcid": "PMC300", "doi": "10.1000/epmc_new", "title": "Copper corrosion 3", "abstractText": "Cu Tafel", "pubYear": "2024"},
                ]
            },
            "nextCursorMark": "cursor2",
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = page1
        mock_resp.raise_for_status.return_value = None

        known = {"10.1000/has_pmc", "10.1000/need_pmc"}
        needs_backfill = {"10.1000/need_pmc": {"pmcid"}}

        with patch.object(epmc.requests, "get", return_value=mock_resp) as mock_get, \
             patch.object(epmc.time, "sleep"):
            recs = epmc.search_europepmc_metadata(
                max_papers=1,
                modes=["corrosion"],
                known_keys=known,
                needs_backfill=needs_backfill,
                max_scan_multiplier=3,
                early_stop_empty_pages=2,
            )

        self.assertEqual(mock_get.call_count, 1)
        pmcids = [r["pmcid"] for r in recs]
        self.assertEqual(pmcids, ["PMC200", "PMC300"])
        self.assertNotIn("10.1000/need_pmc", needs_backfill)
        self.assertIn("10.1000/epmc_new", known)

        # Consecutive empty-page early stop (stops after 2 all-duplicate pages)
        dup_resps = []
        for i in range(4):
            r = MagicMock()
            r.json.return_value = {
                "resultList": {
                    "result": [
                        {"pmcid": f"PMC90{i}", "doi": "10.1000/has_pmc", "title": "Copper corrosion", "abstractText": "Cu Tafel", "pubYear": "2024"}
                    ]
                },
                "nextCursorMark": f"cursor_{i+2}",
            }
            r.raise_for_status.return_value = None
            dup_resps.append(r)

        with patch.object(epmc.requests, "get", side_effect=dup_resps) as mock_get2, \
             patch.object(epmc.time, "sleep"):
            recs_empty = epmc.search_europepmc_metadata(
                max_papers=5,
                modes=["corrosion"],
                known_keys=set(known),
                needs_backfill={},
                max_scan_multiplier=5,
                early_stop_empty_pages=2,
            )
        self.assertEqual(recs_empty, [])
        self.assertEqual(mock_get2.call_count, 2)


class TestStep1To3Optimizations(unittest.TestCase):
    def test_step1_relevance_prefilter_and_interrupt_save(self):
        import tempfile
        from unittest.mock import MagicMock, patch
        import config
        import step1_search_openalex as s1

        def _oa_item(oid, doi, title, abstract_words=None):
            inv = {w: [i] for i, w in enumerate(abstract_words)} if abstract_words else None
            return {
                "id": oid,
                "doi": doi,
                "title": title,
                "publication_year": 2024,
                "primary_location": {"source": {"display_name": "J. Electrochem."}, "pdf_url": None},
                "best_oa_location": {"pdf_url": "https://ex.org/p.pdf"},
                "open_access": {"is_oa": True, "oa_status": "gold", "oa_url": "https://ex.org/p.pdf"},
                "abstract_inverted_index": inv,
                "cited_by_count": 5,
                "authorships": [{"author": {"display_name": "Alice"}}],
            }

        page = {
            "results": [
                # Has both title and abstract, neither mentions Cu/copper -> filtered out
                _oa_item("W1", "https://doi.org/10.1000/irrel", "Steel corrosion study", ["Iron", "acid"]),
                # Mentions copper in abstract -> kept
                _oa_item("W2", "https://doi.org/10.1000/rel", "Corrosion study", ["Copper", "Tafel"]),
                # Missing abstract -> preserved (not falsely dropped)
                _oa_item("W3", "https://doi.org/10.1000/noabs", "Polarization study", None),
            ],
            "meta": {"next_cursor": None},
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = page
        pat = config.build_target_regex("copper", "Cu", ["CuO"])

        with patch.object(s1, "request_with_retry", return_value=mock_resp), \
             patch.object(s1.time, "sleep"):
            recs = s1.fetch_single_query(
                '"copper" "Tafel"',
                max_records=5,
                known_keys=set(),
                needs_backfill={},
                target_pat=pat,
            )
        self.assertEqual([r["doi"] for r in recs], [
            "https://doi.org/10.1000/rel",
            "https://doi.org/10.1000/noabs",
        ])

        # Verify partial results are saved atomically when RuntimeError occurs mid-run
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _make_test_ctx(
                tmpdir,
                "Cu",
                search_queries=["q1", "q2"],
                enable_semantic_scholar=False,
                enable_europepmc_xml=False,
            )
            calls = [0]

            def _fake_fetch(*args, **kwargs):
                calls[0] += 1
                if calls[0] == 1:
                    return [{
                        "query": "q1",
                        "openalex_id": "W2",
                        "s2_paper_id": None,
                        "pmcid": None,
                        "source_api": "openalex",
                        "doi": "https://doi.org/10.1000/rel",
                        "title": "Copper study",
                        "year": 2024,
                        "journal": "J",
                        "is_oa": True,
                        "oa_status": "gold",
                        "oa_url": None,
                        "best_pdf_url": "https://ex.org/p.pdf",
                        "primary_pdf_url": None,
                        "cited_by_count": 5,
                        "authors": "Alice",
                        "abstract": "Copper Tafel",
                    }]
                raise RuntimeError("OpenAlex quota exceeded")

            with patch.object(s1, "fetch_single_query", side_effect=_fake_fetch):
                with self.assertRaises(RuntimeError):
                    s1.run_search(append=False, ctx=ctx)
            self.assertTrue(os.path.exists(ctx.step1_output))
            saved = pd.read_csv(ctx.step1_output)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved.loc[0, "doi"], "https://doi.org/10.1000/rel")

    def test_europepmc_retry_on_429_and_503(self):
        from unittest.mock import MagicMock, patch
        import europepmc_fetcher as epmc

        r429 = MagicMock(status_code=429)
        r503 = MagicMock(status_code=503)
        r200 = MagicMock(status_code=200)
        with patch.object(epmc.requests, "get", side_effect=[r429, r503, r200]) as mock_get, \
             patch.object(epmc.time, "sleep"):
            resp = epmc._epmc_get_with_retry("https://example.org/epmc", max_retries=3)
        self.assertIs(resp, r200)
        self.assertEqual(mock_get.call_count, 3)

    def test_step2_skips_irrelevant_and_existing_pdf_and_runs_s2_first(self):
        from unittest.mock import patch
        import step2_find_pdf_urls as s2

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _make_test_ctx(
                tmpdir,
                "Cu",
                step2_skip_unpaywall_if_has_pdf=True,
                step2_unpaywall_delay=0.0,
                require_target_in_title_abstract=True,
                enable_semantic_scholar=True,
            )
            ctx.ensure_dirs()
            pd.DataFrame([
                # Irrelevant -> skipped for both S2 and Unpaywall
                {"doi": "https://doi.org/10.1000/irrel", "pmcid": None, "title": "Steel corrosion", "abstract": "Iron only", "best_pdf_url": None, "primary_pdf_url": None},
                # Relevant + already has best_pdf_url -> skipped for both S2 and Unpaywall
                {"doi": "https://doi.org/10.1000/has_oa", "pmcid": None, "title": "Copper corrosion", "abstract": "Cu Tafel", "best_pdf_url": "https://oa.org/1.pdf", "primary_pdf_url": None},
                # Relevant + missing PDF -> S2 Batch finds it -> skipped for Unpaywall!
                {"doi": "https://doi.org/10.1000/s2_hit", "pmcid": None, "title": "Copper CMP", "abstract": "Cu slurry", "best_pdf_url": None, "primary_pdf_url": None},
                # Relevant + missing PDF -> S2 misses -> queried on Unpaywall
                {"doi": "https://doi.org/10.1000/unp_hit", "pmcid": None, "title": "CuO etching", "abstract": "Copper oxide", "best_pdf_url": None, "primary_pdf_url": None},
            ]).to_csv(ctx.step1_output, index=False)

            unp_calls = []

            def _fake_unp(doi, **_kwargs):
                unp_calls.append(doi)
                return {
                    "unpaywall_pdf_url": "https://unp.org/found.pdf",
                    "unpaywall_landing_url": "https://unp.org/landing",
                    "license": "cc-by",
                    "is_oa_unpaywall": True,
                }

            def _fake_s2_batch(dois, **_kwargs):
                return {"10.1000/s2_hit": {"s2_pdf_url": "https://s2.org/hit.pdf", "is_oa_s2": True, "s2_paper_id": "p1"}}

            with patch.object(s2, "find_pdf_url", side_effect=_fake_unp), \
                 patch("semantic_scholar_fetcher.batch_lookup_pdf_urls", side_effect=_fake_s2_batch) as mock_s2:
                out_df = s2.run_find_pdfs(append=False, ctx=ctx)

            # S2 Batch only received the 2 relevant missing-PDF DOIs
            self.assertEqual(mock_s2.call_count, 1)
            self.assertEqual(set(mock_s2.call_args[0][0]), {
                "https://doi.org/10.1000/s2_hit",
                "https://doi.org/10.1000/unp_hit",
            })
            # Unpaywall only queried the 1 DOI that S2 missed
            self.assertEqual(unp_calls, ["https://doi.org/10.1000/unp_hit"])
            self.assertEqual(out_df.loc[1, "final_pdf_url"], "https://oa.org/1.pdf")
            self.assertEqual(out_df.loc[2, "final_pdf_url"], "https://s2.org/hit.pdf")
            self.assertEqual(out_df.loc[3, "final_pdf_url"], "https://unp.org/found.pdf")

    def test_step3_circuit_breaker_and_pdf_streaming_guards(self):
        import tempfile
        from unittest.mock import MagicMock
        import step3_download_pdfs as s3

        # Circuit breaker trips at threshold and resets on success before threshold
        cb = s3.DomainCircuitBreaker(threshold=3)
        self.assertFalse(cb.record_403("mdpi.com"))
        self.assertFalse(cb.record_403("mdpi.com"))
        cb.record_success("mdpi.com")
        self.assertFalse(cb.record_403("mdpi.com"))
        self.assertFalse(cb.record_403("mdpi.com"))
        self.assertTrue(cb.record_403("mdpi.com"))
        self.assertTrue(cb.is_open("mdpi.com"))
        self.assertEqual(cb.tripped_domains(), ["mdpi.com"])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test.pdf")

            # 1. HTML Content-Type rejected immediately without writing file
            sess_html = MagicMock()
            resp_html = MagicMock(status_code=200, headers={"Content-Type": "text/html; charset=utf-8"})
            resp_html.__enter__.return_value = resp_html
            sess_html.get.return_value = resp_html
            ok, reason = s3.download_pdf("https://ex.org/login", save_path, session=sess_html)
            self.assertFalse(ok)
            self.assertIn("HTML", reason)
            self.assertFalse(os.path.exists(save_path))

            # 2. Non-%PDF first chunk rejected immediately
            sess_bad = MagicMock()
            resp_bad = MagicMock(status_code=200, headers={"Content-Type": "application/octet-stream"})
            resp_bad.iter_content.return_value = [b"<html>blocked</html>", b"more data"]
            resp_bad.__enter__.return_value = resp_bad
            sess_bad.get.return_value = resp_bad
            ok2, reason2 = s3.download_pdf("https://ex.org/fake.pdf", save_path, session=sess_bad)
            self.assertFalse(ok2)
            self.assertIn("HTML", reason2)
            self.assertFalse(os.path.exists(save_path))

            # 3. Oversized Content-Length rejected immediately
            sess_big = MagicMock()
            resp_big = MagicMock(
                status_code=200,
                headers={"Content-Type": "application/pdf", "Content-Length": str(10 * 1024 * 1024)},
            )
            resp_big.__enter__.return_value = resp_big
            sess_big.get.return_value = resp_big
            ok3, reason3 = s3.download_pdf("https://ex.org/huge.pdf", save_path, max_pdf_mb=2, session=sess_big)
            self.assertFalse(ok3)
            self.assertIn("超過上限", reason3)

            # 4. Valid PDF succeeds
            sess_ok = MagicMock()
            resp_ok = MagicMock(status_code=200, headers={"Content-Type": "application/pdf"})
            resp_ok.iter_content.return_value = [b"%PDF-1.7\n" + b"x" * 12000]
            resp_ok.__enter__.return_value = resp_ok
            sess_ok.get.return_value = resp_ok
            ok4, _ = s3.download_pdf("https://ex.org/good.pdf", save_path, max_pdf_mb=2, session=sess_ok)
            self.assertTrue(ok4)
            self.assertTrue(os.path.exists(save_path))

    def test_step2_checkpoint_interrupt_and_append_resume(self):
        from unittest.mock import patch
        import step2_find_pdf_urls as s2

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _make_test_ctx(
                tmpdir,
                "Cu",
                enable_semantic_scholar=False,
                step2_checkpoint_every=1,
                step2_unpaywall_delay=0.0,
            )
            ctx.ensure_dirs()
            pd.DataFrame([
                {"doi": "https://doi.org/10.1000/p1", "pmcid": None, "title": "Copper 1", "abstract": "Cu", "best_pdf_url": None, "primary_pdf_url": None},
                {"doi": "https://doi.org/10.1000/p2", "pmcid": None, "title": "Copper 2", "abstract": "Cu", "best_pdf_url": None, "primary_pdf_url": None},
                {"doi": "https://doi.org/10.1000/p3", "pmcid": None, "title": "Copper 3", "abstract": "Cu", "best_pdf_url": None, "primary_pdf_url": None},
            ]).to_csv(ctx.step1_output, index=False)

            calls_run1 = []

            def _unp_interrupt_on_second(doi, **_kwargs):
                calls_run1.append(doi)
                if doi.endswith("/p2"):
                    raise KeyboardInterrupt("simulated Ctrl+C")
                return {
                    "unpaywall_pdf_url": f"https://unp.org/{doi.split('/')[-1]}.pdf",
                    "unpaywall_landing_url": None,
                    "license": "cc-by",
                    "is_oa_unpaywall": True,
                }

            with patch.object(s2, "find_pdf_url", side_effect=_unp_interrupt_on_second):
                with self.assertRaises(KeyboardInterrupt):
                    s2.run_find_pdfs(append=False, ctx=ctx)

            self.assertTrue(os.path.exists(ctx.step2_output))
            mid_df = pd.read_csv(ctx.step2_output)
            self.assertTrue(bool(mid_df.loc[0, "step2_checked"]))
            self.assertFalse(bool(mid_df.loc[1, "step2_checked"]))
            self.assertFalse(bool(mid_df.loc[2, "step2_checked"]))

            # Resuming with append=True must skip p1 and query p2 & p3
            calls_run2 = []

            def _unp_ok(doi, **_kwargs):
                calls_run2.append(doi)
                return {
                    "unpaywall_pdf_url": f"https://unp.org/{doi.split('/')[-1]}.pdf",
                    "unpaywall_landing_url": None,
                    "license": "cc-by",
                    "is_oa_unpaywall": True,
                }

            with patch.object(s2, "find_pdf_url", side_effect=_unp_ok):
                final_df = s2.run_find_pdfs(append=True, ctx=ctx)

            self.assertEqual(calls_run2, [
                "https://doi.org/10.1000/p2",
                "https://doi.org/10.1000/p3",
            ])
            self.assertTrue(final_df["step2_checked"].all())
            self.assertEqual(final_df["final_pdf_url"].notna().sum(), 3)

    def test_step3_concurrent_breaker_triggers_epmc_fallback(self):
        from unittest.mock import patch
        import step3_download_pdfs as s3

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _make_test_ctx(
                tmpdir,
                "Cu",
                download_delay=0.0,
                europepmc_delay=0.0,
                step3_max_workers=2,
                step3_domain_403_threshold=1,
                enable_europepmc_xml=True,
            )
            ctx.ensure_dirs()

            pd.DataFrame([
                {"doi": "https://doi.org/10.1000/b1", "pmcid": None, "title": "Copper 1", "abstract": "Cu", "final_pdf_url": "https://blocked.example.com/1.pdf"},
                {"doi": "https://doi.org/10.1000/b2", "pmcid": None, "title": "Copper 2", "abstract": "Cu", "final_pdf_url": "https://blocked.example.com/2.pdf"},
                {"doi": "https://doi.org/10.1000/b3", "pmcid": None, "title": "Copper 3", "abstract": "Cu", "final_pdf_url": "https://blocked.example.com/3.pdf"},
            ]).to_csv(ctx.step2_output, index=False)

            pdf_attempts = []

            def _fake_download_pdf(url, save_path, **kwargs):
                pdf_attempts.append(url)
                return False, "403 Forbidden（出版社設有防爬蟲/訂閱限制）"

            def _fake_epmc_fallback(doi, base_name, tdir, **kwargs):
                if doi.endswith("/b3"):
                    with open(os.path.join(tdir, base_name + ".txt"), "w", encoding="utf-8") as f:
                        f.write("Copper Tafel data from EPMC fallback")
                    return True, 1
                return False, 0

            with patch.object(s3, "download_pdf", side_effect=_fake_download_pdf), \
                 patch("europepmc_fetcher.try_fallback_from_europepmc", side_effect=_fake_epmc_fallback):
                s3.run_download(retry_failed=False, ctx=ctx)

            with open(ctx.download_state_path, "r", encoding="utf-8") as f:
                saved_state = json.load(f)
            # b3 was rescued via Europe PMC fallback even after domain 403 breaker tripped
            self.assertEqual(saved_state["10.1000/b3"]["status"], s3.STATUS_DONE_EPMC)
            # At least 1 paper hit 403 and triggered breaker before all 3 could be attempted sequentially
            self.assertLessEqual(len(pdf_attempts), 3)

    def test_s2_append_param_advances_cached_bulk_token(self):
        from unittest.mock import MagicMock, patch
        import semantic_scholar_fetcher as s2f

        def _s2_item(pid, doi):
            return {
                "paperId": pid,
                "externalIds": {"DOI": doi},
                "title": f"S2 {pid}",
                "year": 2024,
                "venue": "Corr. Sci.",
                "isOpenAccess": False,
                "openAccessPdf": None,
                "citationCount": 10,
                "authors": [{"name": "Bob"}],
                "abstract": "Cu corrosion",
            }

        page1_resp = MagicMock()
        page1_resp.json.return_value = {"data": [_s2_item("p1", "10.1000/p1")], "token": "tok_page2"}
        page2_resp = MagicMock()
        page2_resp.json.return_value = {"data": [_s2_item("p2", "10.1000/p2")], "token": None}

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _make_test_ctx(tmpdir, "Cu")
            with patch.object(s2f, "_select_s2_queries", return_value=['"copper" + "Tafel"']), \
                 patch.object(s2f, "_s2_request_with_retry", side_effect=[page1_resp, page2_resp]) as mock_req:
                # First run caches page1 + tok_page2
                r1 = s2f.search_semantic_scholar(
                    ["q1", "q2"],
                    max_per_query=1,
                    from_year=2005,
                    use_cache=True,
                    known_keys=set(),
                    append=False,
                    ctx=ctx,
                )
                self.assertEqual([r["doi"] for r in r1], ["https://doi.org/10.1000/p1"])
                # Second run with append=True (no TAFEL_APPEND env var) advances via saved token to fetch p2!
                r2 = s2f.search_semantic_scholar(
                    ["q1", "q2"],
                    max_per_query=1,
                    from_year=2005,
                    use_cache=True,
                    known_keys={"10.1000/p1"},
                    append=True,
                    ctx=ctx,
                )
                self.assertEqual([r["doi"] for r in r2], ["https://doi.org/10.1000/p2"])
                self.assertEqual(mock_req.call_count, 2)


class TestStep1CursorGroupedAndQuotaDegradation(unittest.TestCase):
    def test_step1_cursor_store_and_openalex_epmc_resume_and_exhausted_skip(self):
        from unittest.mock import MagicMock, patch
        import europepmc_fetcher as epmc
        import step1_search_openalex as s1
        from step1_cursor_state import Step1CursorStore, make_cursor_key

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _make_test_ctx(
                tmpdir,
                "Cu",
                search_queries=["q_done", "q_resume"],
                max_records_per_query=10,
                enable_openalex=True,
                enable_semantic_scholar=False,
                enable_europepmc_xml=False,
            )
            ctx.ensure_dirs()
            from_year = ctx.settings.from_year
            store = Step1CursorStore(ctx.step1_cursor_state_path)

            k_done = make_cursor_key("openalex", "q_done", from_year)
            k_resume = make_cursor_key("openalex", "q_resume", from_year)
            store.update(k_done, None, exhausted=True)
            store.update(k_resume, "cursor_p2", exhausted=False)

            reloaded = Step1CursorStore(ctx.step1_cursor_state_path)
            self.assertEqual(reloaded.get(k_done), (None, True))
            self.assertEqual(reloaded.get(k_resume), ("cursor_p2", False))

            seen_cursors: list[str] = []

            def _oa_item(oid, doi):
                return {
                    "id": oid,
                    "doi": doi,
                    "title": f"Copper corrosion {oid}",
                    "publication_year": 2024,
                    "primary_location": {"source": {"display_name": "Corr. Sci."}, "pdf_url": None},
                    "best_oa_location": {"pdf_url": "https://ex.org/p.pdf"},
                    "open_access": {"is_oa": True, "oa_status": "gold", "oa_url": "https://ex.org/p.pdf"},
                    "abstract_inverted_index": {"Copper": [0], "Tafel": [1]},
                    "cited_by_count": 5,
                    "authorships": [{"author": {"display_name": "Alice"}}],
                }

            def _fake_oa_req(url, params, **_kwargs):
                seen_cursors.append(params["cursor"])
                r = MagicMock()
                r.json.return_value = {
                    "results": [_oa_item("W20", "https://doi.org/10.1000/p20")],
                    "meta": {"next_cursor": None},
                }
                return r

            with patch.object(s1, "request_with_retry", side_effect=_fake_oa_req), \
                 patch.object(s1.time, "sleep"):
                df = s1.run_search(append=True, ctx=ctx)

            # q_done was exhausted -> 0 API calls; q_resume started at cursor_p2 -> now exhausted
            self.assertEqual(seen_cursors, ["cursor_p2"])
            self.assertEqual(len(df), 1)
            after = Step1CursorStore(ctx.step1_cursor_state_path)
            self.assertEqual(after.get(k_resume), (None, True))

            # OpenAlex HTTP 400 invalid cursor automatically falls back to "*"
            oa_fallback_cursors: list[str] = []

            def _oa_400_then_ok(url, params, **_kwargs):
                oa_fallback_cursors.append(params["cursor"])
                if params["cursor"] == "expired_cur":
                    raise s1.OpenAlexInvalidCursorError("expired")
                r = MagicMock()
                r.json.return_value = {
                    "results": [_oa_item("W21", "https://doi.org/10.1000/p21")],
                    "meta": {"next_cursor": "fresh_c2"},
                }
                return r

            cbox: dict = {}
            with patch.object(s1, "request_with_retry", side_effect=_oa_400_then_ok), \
                 patch.object(s1.time, "sleep"):
                recs = s1.fetch_single_query(
                    '"copper" "Tafel"',
                    max_records=1,
                    known_keys=set(),
                    needs_backfill={},
                    start_cursor="expired_cur",
                    cursor_out=cbox,
                    ctx=ctx,
                )
            self.assertEqual(oa_fallback_cursors, ["expired_cur", "*"])
            self.assertEqual(len(recs), 1)
            self.assertEqual(cbox, {"next_cursor": "fresh_c2", "exhausted": False})

            # Europe PMC HTTP 400 invalid cursor falls back to "*" and reports exhausted when nextCursorMark repeats
            epmc_cursors: list[str] = []

            def _epmc_400_then_ok(url, params=None, timeout=30):
                epmc_cursors.append(params["cursorMark"])
                r = MagicMock()
                if params["cursorMark"] == "bad_epmc_cursor":
                    r.status_code = 400
                    return r
                r.status_code = 200
                r.raise_for_status.return_value = None
                r.json.return_value = {
                    "resultList": {
                        "result": [
                            {"pmcid": "PMC777", "doi": "10.1000/epmc777", "title": "Copper corrosion", "abstractText": "Cu Tafel", "pubYear": "2024"}
                        ]
                    },
                    "nextCursorMark": params["cursorMark"],
                }
                return r

            epmc_box: dict = {}
            with patch.object(epmc.requests, "get", side_effect=_epmc_400_then_ok), \
                 patch.object(epmc.time, "sleep"):
                epmc_recs = epmc.search_europepmc_metadata(
                    max_papers=5,
                    modes=["corrosion"],
                    known_keys=set(),
                    needs_backfill={},
                    start_cursor="bad_epmc_cursor",
                    cursor_out=epmc_box,
                    ctx=ctx,
                )
            self.assertEqual(epmc_cursors, ["bad_epmc_cursor", "*"])
            self.assertEqual(len(epmc_recs), 1)
            self.assertEqual(epmc_box, {"next_cursor": None, "exhausted": True})

    def test_openalex_grouped_queries_reduction_and_s2_detection(self):
        import semantic_scholar_fetcher as s2f

        cu_ctx = config.build_metal_context("Cu")
        grouped_all = config.build_search_queries(
            ["corrosion", "cmp_etch", "inhibitor", "solvent"],
            target_metal=cu_ctx.target_metal,
            metal_symbol=cu_ctx.metal_symbol,
            compounds=cu_ctx.compounds,
            cmp_queries=cu_ctx.cmp_queries,
            inhibitor_queries=cu_ctx.inhibitor_queries,
            grouped=True,
        )
        expanded_all = config.build_search_queries(
            ["corrosion", "cmp_etch", "inhibitor", "solvent"],
            target_metal=cu_ctx.target_metal,
            metal_symbol=cu_ctx.metal_symbol,
            compounds=cu_ctx.compounds,
            cmp_queries=cu_ctx.cmp_queries,
            inhibitor_queries=cu_ctx.inhibitor_queries,
            grouped=False,
        )
        self.assertLess(len(grouped_all), 25)
        self.assertGreater(len(expanded_all), 80)
        self.assertTrue(any(" AND " in q and " OR " in q for q in grouped_all))

        # S2 must recognize short grouped OpenAlex queries (e.g. solvent = 2 queries) and build S2 +/-/| queries
        oa_solvent = config.build_search_queries(
            ["solvent"],
            target_metal=cu_ctx.target_metal,
            metal_symbol=cu_ctx.metal_symbol,
            compounds=cu_ctx.compounds,
            cmp_queries=cu_ctx.cmp_queries,
            inhibitor_queries=cu_ctx.inhibitor_queries,
            grouped=True,
        )
        self.assertLessEqual(len(oa_solvent), 4)
        self.assertTrue(s2f._is_openalex_grouped_queries(oa_solvent))
        s2_selected = s2f._select_s2_queries(oa_solvent, use_bulk=True, ctx=cu_ctx)
        self.assertTrue(all(" + " in q for q in s2_selected))
        self.assertFalse(any(" AND " in q for q in s2_selected))

    def test_grouped_query_disambiguation_and_syntax_fixes(self):
        import europepmc_fetcher as epmc
        import semantic_scholar_fetcher as s2f

        ru_ctx = config.build_metal_context("Ru")
        tin_ctx = config.build_metal_context("TiN")

        # 1. OpenAlex corrosion grouped queries must anchor Tafel slope/plot with AND "corrosion"
        ru_oa_corr = config.build_grouped_openalex_queries(
            ["corrosion"],
            target_metal=ru_ctx.target_metal,
            metal_symbol=ru_ctx.metal_symbol,
            compounds=ru_ctx.compounds,
        )
        tafel_qs = [q for q in ru_oa_corr if '"Tafel slope"' in q]
        self.assertEqual(len(tafel_qs), 1)
        self.assertIn('AND "corrosion" AND', tafel_qs[0])

        # 2. OpenAlex custom CMP/inhibitor queries must use explicit AND (no bare space-separated phrases inside OR)
        ru_oa_cmp = config.build_grouped_openalex_queries(
            ["cmp_etch", "inhibitor"],
            target_metal=ru_ctx.target_metal,
            metal_symbol=ru_ctx.metal_symbol,
            cmp_queries=ru_ctx.cmp_queries,
            inhibitor_queries=ru_ctx.inhibitor_queries,
        )
        self.assertTrue(any('"ruthenium" AND "CMP" AND "potentiodynamic polarization"' in q for q in ru_oa_cmp))
        self.assertFalse(any('"ruthenium" "CMP"' in q for q in ru_oa_cmp))

        # 3. TiN must be disambiguated from 'tin' (Sn) across OpenAlex, S2, and Europe PMC,
        #    and bare "TiN" must not leak into compound batch queries
        tin_oa_corr = config.build_grouped_openalex_queries(
            ["corrosion"],
            target_metal=tin_ctx.target_metal,
            metal_symbol=tin_ctx.metal_symbol,
            compounds=tin_ctx.compounds,
        )
        self.assertIn('("TiN" AND ("nitride"', tin_oa_corr[0])
        compound_oa_qs = [q for q in tin_oa_corr if '"TiAlN"' in q or '"Ti2N"' in q]
        self.assertTrue(compound_oa_qs)
        self.assertFalse(any('("TiN" OR' in q or 'OR "TiN"' in q for q in compound_oa_qs))

        tin_s2_corr = s2f.build_grouped_s2_queries(["corrosion"], ctx=tin_ctx)
        self.assertIn('("TiN" + ("nitride"', tin_s2_corr[0])
        self.assertNotIn('"Tafel slope"', tin_s2_corr[1])
        self.assertFalse(tin_s2_corr[1].startswith('("TiN" |'))

        # 4. Europe PMC query includes TiN disambiguation, compounds, and relaxed FROM_YEAR
        tin_epmc = epmc.build_europepmc_query(["corrosion"], ctx=tin_ctx)
        self.assertIn('("TiN" AND ("nitride"', tin_epmc)
        self.assertIn('"TiAlN"', tin_epmc)
        self.assertIn(f"PUB_YEAR:[{tin_ctx.settings.from_year} TO 2100]", tin_epmc)
        self.assertEqual(tin_ctx.settings.from_year, 2000)

    def test_openalex_quota_exhausted_graceful_degradation_and_skip_flag(self):
        from unittest.mock import MagicMock, patch
        import europepmc_fetcher as epmc
        import run_all
        import semantic_scholar_fetcher as s2f
        import step1_search_openalex as s1
        from step1_cursor_state import Step1CursorStore, make_cursor_key

        # 1. request_with_retry raises OpenAlexQuotaExhaustedError on 429 insufficient budget
        r429 = MagicMock(status_code=429, headers={"Retry-After": "36000"})
        r429.json.return_value = {"message": "Insufficient budget for daily quota"}
        with patch.object(s1._SESSION, "get", return_value=r429):
            with self.assertRaises(s1.OpenAlexQuotaExhaustedError):
                s1.request_with_retry(s1.BASE_URL, {"search": "test"})

        # 2. run_search gracefully degrades on OpenAlexQuotaExhaustedError:
        #    saves partial OA records + resume cursor, updates oa_quota_state, and still runs S2 + EPMC
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _make_test_ctx(
                tmpdir,
                "Cu",
                search_queries=["q1"],
                max_records_per_query=10,
                enable_openalex=True,
                enable_semantic_scholar=True,
                enable_europepmc_xml=True,
            )
            ctx.ensure_dirs()
            quota_state = {"exhausted": False}

            def _fake_oa_fetch(q, max_records, **kwargs):
                cbox = kwargs.get("cursor_out")
                if cbox is not None:
                    cbox["next_cursor"] = "saved_mid_cursor"
                    cbox["exhausted"] = False
                raise s1.OpenAlexQuotaExhaustedError("OpenAlex 每日配額已耗盡")

            fake_s2_rec = {
                "query": "s2_q",
                "openalex_id": None,
                "s2_paper_id": "S2_1",
                "pmcid": None,
                "source_api": "semantic_scholar",
                "doi": "https://doi.org/10.1000/s2_only",
                "title": "Copper Tafel S2",
                "year": 2024,
                "journal": "J",
                "is_oa": True,
                "oa_status": "gold",
                "oa_url": None,
                "best_pdf_url": "https://ex.org/s2.pdf",
                "primary_pdf_url": None,
                "cited_by_count": 3,
                "authors": "Bob",
                "abstract": "Copper polarization",
            }
            fake_epmc_rec = {
                **fake_s2_rec,
                "s2_paper_id": None,
                "pmcid": "PMC888",
                "source_api": "europepmc",
                "doi": "https://doi.org/10.1000/epmc_only",
            }

            with patch.object(s1, "fetch_single_query", side_effect=_fake_oa_fetch), \
                 patch.object(s2f, "search_semantic_scholar", return_value=[fake_s2_rec]) as mock_s2, \
                 patch.object(epmc, "search_europepmc_metadata", return_value=[fake_epmc_rec]) as mock_epmc:
                df = s1.run_search(append=True, ctx=ctx, oa_quota_state=quota_state)

            self.assertEqual(mock_s2.call_count, 1)
            self.assertEqual(mock_epmc.call_count, 1)
            self.assertTrue(quota_state["exhausted"])
            self.assertTrue(os.path.exists(ctx.step1_output))
            self.assertEqual(len(df), 2)
            # Mid-query cursor was preserved
            saved_store = Step1CursorStore(ctx.step1_cursor_state_path)
            self.assertEqual(
                saved_store.get(make_cursor_key("openalex", "q1", ctx.settings.from_year)),
                ("saved_mid_cursor", False),
            )

            # 3. skip_openalex=True skips OA completely and still runs S2
            ctx_no_epmc = _make_test_ctx(
                tmpdir,
                "Cu",
                search_queries=["q1"],
                enable_openalex=True,
                enable_semantic_scholar=True,
                enable_europepmc_xml=False,
            )
            with patch.object(s1, "fetch_single_query") as mock_oa_skip, \
                 patch.object(s2f, "search_semantic_scholar", return_value=[fake_s2_rec]):
                s1.run_search(append=True, skip_openalex=True, ctx=ctx_no_epmc)
            self.assertEqual(mock_oa_skip.call_count, 0)

        # 4. CLI --skip-openalex flag
        parser = run_all.build_cli_parser()
        parsed = parser.parse_args(["WN", "--append", "--skip-openalex"])
        self.assertTrue(parsed.skip_openalex)


class TestRunContextIsolationAndBatchExecution(unittest.TestCase):
    def test_config_import_has_no_side_effects(self):
        import subprocess
        import sys

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        clean_env = os.environ.copy()
        for key in ("OPENALEX_API_KEY", "SEMANTIC_SCHOLAR_API_KEY", "MP_API_KEY"):
            clean_env.pop(key, None)
        clean_env["TAFEL_METAL"] = "INVALID_METAL_SHOULD_BE_IGNORED"

        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os, config, step6_merge, step7_enrich_data, run_all; "
                    "assert not os.environ.get('OPENALEX_API_KEY'); "
                    "assert not os.environ.get('SEMANTIC_SCHOLAR_API_KEY'); "
                    "assert not os.environ.get('MP_API_KEY'); "
                    "import auto_run"
                ),
            ],
            cwd=repo_root,
            env=clean_env,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(proc.stdout.strip(), "")

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _make_test_ctx(tmpdir, "Co")
            self.assertFalse(os.path.exists(ctx.data_dir))
            ctx.ensure_dirs()
            self.assertTrue(os.path.isdir(ctx.metadata_dir))
            self.assertTrue(os.path.isdir(ctx.extracted_dir))

    def test_independent_metal_contexts_do_not_leak(self):
        from step5_text_extractor import assign_material, find_materials

        with tempfile.TemporaryDirectory() as tmpdir:
            cu_ctx = _make_test_ctx(tmpdir, "Cu")
            co_ctx = _make_test_ctx(tmpdir, "Co")

            self.assertNotEqual(cu_ctx.step1_output, co_ctx.step1_output)
            self.assertNotEqual(cu_ctx.s2_cache_path, co_ctx.s2_cache_path)
            self.assertNotEqual(cu_ctx.step5_echem_output, co_ctx.step5_echem_output)
            self.assertTrue(cu_ctx.target_regex.search("Cu3P electrode"))
            self.assertFalse(co_ctx.target_regex.search("Cu3P electrode"))

            cu_hits = [sym for _, sym in find_materials("Potentiodynamic polarization of Cu3P electrode", ctx=cu_ctx)]
            co_hits = [sym for _, sym in find_materials("Potentiodynamic polarization of cobalt foil", ctx=co_ctx)]
            self.assertIn("Cu", cu_hits)
            self.assertIn("Co", co_hits)

            mat_co, _ = assign_material(["cobalt foil in acid"], Counter({"Co": 5}), ctx=co_ctx)
            mat_cu, _ = assign_material(["Cu3P electrode in acid"], Counter({"Cu": 5}), ctx=cu_ctx)
            self.assertEqual(mat_co, "Co")
            self.assertEqual(mat_cu, "Cu")

    def test_run_batch_error_isolation_and_in_memory_oa_quota(self):
        from unittest.mock import patch
        import run_all
        import step1_search_openalex as s1

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = config.load_settings(
                data_root=tmpdir,
                skip_existing_step1=False,
                enable_openalex=True,
            )

            seen_calls: list[tuple[str, bool, bool]] = []

            def _fake_search(*, append=False, skip_openalex=False, ctx=None, oa_quota_state=None):
                seen_calls.append((ctx.metal_symbol, skip_openalex, ctx.enable_openalex))
                if ctx.metal_symbol == "Cu":
                    oa_quota_state["exhausted"] = True
                return pd.DataFrame()

            with patch.object(s1, "run_search", side_effect=_fake_search):
                rc = run_all.run_batch(
                    ["Cu", "Co"],
                    from_step=1,
                    to_step=1,
                    no_merge=True,
                    no_enrich=True,
                    settings=settings,
                )
            self.assertEqual(rc, 0)
            self.assertEqual(
                seen_calls,
                [
                    ("Cu", False, True),
                    ("Co", True, False),
                ],
            )

            # Error in first metal must not abort the second metal
            executed: list[str] = []

            def _failing_then_ok(*, append=False, skip_openalex=False, ctx=None, oa_quota_state=None):
                executed.append(ctx.metal_symbol)
                if ctx.metal_symbol == "Cu":
                    raise RuntimeError("boom")
                return pd.DataFrame()

            with patch.object(s1, "run_search", side_effect=_failing_then_ok):
                rc_err = run_all.run_batch(
                    ["Cu", "Co"],
                    from_step=1,
                    to_step=1,
                    no_merge=True,
                    no_enrich=True,
                    settings=settings,
                )
            self.assertEqual(rc_err, 1)
            self.assertEqual(executed, ["Cu", "Co"])

            # Invalid metal symbol inside build_metal_context must also be isolated
            executed_after_invalid: list[str] = []

            def _record_ok(*, append=False, skip_openalex=False, ctx=None, oa_quota_state=None):
                executed_after_invalid.append(ctx.metal_symbol)
                return pd.DataFrame()

            with patch.object(s1, "run_search", side_effect=_record_ok):
                rc_invalid = run_all.run_batch(
                    ["INVALID_METAL", "Co"],
                    from_step=1,
                    to_step=1,
                    no_merge=True,
                    no_enrich=True,
                    settings=settings,
                )
            self.assertEqual(rc_invalid, 1)
            self.assertEqual(executed_after_invalid, ["Co"])

    def test_step6_discover_complete_metals_and_no_global_mutation(self):
        from pathlib import Path
        import step6_merge

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cu_ext = root / "Cu" / "extracted"
            co_ext = root / "Co" / "extracted"
            cu_ext.mkdir(parents=True)
            co_ext.mkdir(parents=True)

            for suffix in step6_merge.REQUIRED_SUFFIXES:
                (cu_ext / f"cu_{suffix}").write_text("col\n", encoding="utf-8")
            (co_ext / "co_material_properties.csv").write_text("col\n", encoding="utf-8")

            self.assertEqual(step6_merge.discover_complete_metals(root), ["Cu"])

        recs = [
            {
                "is_target_metal": False,
                "material_id_assigned": "Co",
                "source_id": "10.1000_cross",
                "record_electrolytes": [{"conc": "0.5", "unit": "M", "species": "H2SO4"}],
                "record_ph": 1.0,
                "record_temp": 25.0,
            }
        ]
        sol_df = build_solution_properties(recs, keep_only_target=False, target_symbol="")
        self.assertEqual(len(sol_df), 1)
        self.assertEqual(sol_df.loc[0, "materials"], "Co")


class TestStep5SchemaAndParity(unittest.TestCase):
    def test_text_extraction_context_she_and_tafel_bounds(self):
        from step5_constants import convert_to_she, normalize_text
        from step5_text_extractor import (
            count_materials,
            extract_from_text,
            extract_paper_environment,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _make_test_ctx(tmpdir, "Cu")
            # Paper-wide majority is SCE and pH 7.0 in methods, while the measurement
            # sentence >350 chars away uses RHE at pH 3.0.
            filler = " " + ("Electrochemical impedance and morphology characterization were performed. " * 6)
            paper_text = (
                "Electrochemical tests generally used a SCE reference electrode at pH 7.0. "
                "Control runs were also measured vs SCE at pH 7.0 in 0.5 M NaCl."
                + filler
                + "For pure copper in 0.5 M H2SO4 at pH 3.0, Ecorr = -0.10 V vs. RHE, "
                "icorr = 15 uA/cm2, ba = 600 mV/dec, bc = 120 mV/dec."
            )
            norm_text = normalize_text(paper_text)
            paper_counts = count_materials(norm_text, ctx=ctx)
            paper_environment = extract_paper_environment(norm_text, norm_text)
            recs = extract_from_text(
                norm_text,
                "10.1000_parity",
                paper_counts=paper_counts,
                ctx=ctx,
                paper_environment=paper_environment,
            )

            self.assertGreaterEqual(len(recs), 1)
            rhe_rec = next(r for r in recs if r.get("Ecorr_V_original") == -0.10)
            expected_she = convert_to_she(-0.10, "RHE", 3.0)
            self.assertAlmostEqual(rhe_rec["Ecorr_V_SHE"], expected_she, places=4)
            self.assertEqual(rhe_rec["beta_a_mV_dec"], 600.0)
            self.assertEqual(rhe_rec["beta_c_mV_dec"], 120.0)
            self.assertIn("exceeds TAFEL_MAX", rhe_rec["extra_quality_notes"])

            # Separate paper with out-of-bounds Tafel slopes (< TAFEL_MIN and > 1000)
            # Also verify plain-dict compatibility on extract_from_text(paper_environment=...)
            oob_text = normalize_text(
                "For pure copper in 0.5 M H2SO4 at pH 3.0, Ecorr = -0.20 V vs. SCE, "
                "icorr = 20 uA/cm2, ba = 5 mV/dec, bc = 2500 mV/dec."
            )
            oob_counts = count_materials(oob_text, ctx=ctx)
            oob_recs = extract_from_text(
                oob_text,
                "10.1000_oob",
                paper_counts=oob_counts,
                ctx=ctx,
                paper_environment={"refs": ["SCE"], "ref_counts": {"SCE": 1}},
            )
            self.assertGreaterEqual(len(oob_recs), 1)
            oob_rec = oob_recs[0]
            self.assertIsNone(oob_rec["beta_a_mV_dec"])
            self.assertIsNone(oob_rec["beta_c_mV_dec"])

    def test_paper_environment_survives_empty_text_records(self):
        import json
        from step5_constants import convert_to_she, normalize_text
        from step5_table_extractor import extract_from_tables, extract_table_captions
        from step5_text_extractor import (
            count_materials,
            extract_from_text,
            extract_paper_environment,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _make_test_ctx(tmpdir, "Cu")
            body = (
                "A polished copper thin film electrode was evaluated by potentiodynamic "
                "polarization in deaerated 0.5 M H2SO4 (pH 1.0) at 25 °C using a SCE "
                "reference electrode. Table 1 summarizes the electrochemical parameters "
                "for Cu in 0.5 M H2SO4."
            )
            norm_text = normalize_text(body)
            paper_counts = count_materials(norm_text, ctx=ctx)
            paper_environment = extract_paper_environment(
                norm_text,
                norm_text,
                table_captions=extract_table_captions(norm_text),
            )
            text_records = extract_from_text(
                norm_text,
                "10.1000_table_only",
                paper_counts=paper_counts,
                ctx=ctx,
                paper_environment=paper_environment,
            )
            self.assertEqual(text_records, [])

            table_path = os.path.join(tmpdir, "10.1000_table_only_tables.json")
            tables_payload = [
                [
                    [
                        "Sample",
                        "Ecorr (mV vs SCE)",
                        "icorr (uA/cm2)",
                        "ba (mV/dec)",
                        "bc (mV/dec)",
                    ],
                    ["Bare Cu", "-200", "25", "80", "110"],
                ]
            ]
            with open(table_path, "w", encoding="utf-8") as f:
                json.dump(tables_payload, f)

            table_records = extract_from_tables(
                table_path,
                "10.1000_table_only",
                paper_counts=paper_counts,
                paper_environment=paper_environment,
                ctx=ctx,
            )
            self.assertEqual(len(table_records), 1)
            rec = table_records[0]
            self.assertIn("thin film", rec["metal_form"] or "")
            self.assertIn("polished", rec["surface_preparation"] or "")
            self.assertTrue(rec["deaerated"])
            self.assertIn("potentiodynamic", rec["measurement_methods"] or "")
            self.assertAlmostEqual(
                rec["Ecorr_V_SHE"],
                convert_to_she(-0.200, "SCE", 1.0),
                places=4,
            )

    def test_chemical_species_registry_single_source_derivation(self):
        import json
        from step5_schema import load_chemical_species_registry

        default_reg = load_chemical_species_registry()
        self.assertEqual(default_reg.normalise("benzotriazole"), "BTA")
        self.assertEqual(default_reg.lookup_category("BTA")["role"], "inhibitor")
        self.assertIn("O3", default_reg.species_categories)
        self.assertNotIn("O3", default_reg.electrolyte_species)

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = os.path.join(tmpdir, "custom_species.json")
            custom_data = {
                "role_fields": {
                    "acid": "acids",
                    "inhibitor": "inhibitors",
                },
                "species": [
                    {"canonical": "HCl", "role": "acid", "aliases": ["hydrochloric acid"]},
                    {
                        "canonical": "MyNewInh",
                        "role": "inhibitor",
                        "aliases": ["my new inhibitor"],
                    },
                ],
            }
            with open(custom_path, "w", encoding="utf-8") as f:
                json.dump(custom_data, f)

            reg = load_chemical_species_registry(custom_path)
            self.assertEqual(reg.normalise("my new inhibitor"), "MyNewInh")
            self.assertEqual(
                reg.lookup_category("MyNewInh"),
                {"role": "inhibitor", "field": "inhibitors"},
            )
            self.assertIn("MyNewInh", reg.electrolyte_species)
            self.assertIn("my new inhibitor", reg.electrolyte_species)
            import re as _re
            el_m = _re.search(reg.electrolyte_species_regex, "0.5 M my new inhibitor", _re.I)
            self.assertIsNotNone(el_m)
            self.assertEqual(el_m.group(0).lower(), "my new inhibitor")
            m = reg.inhibitor_name_pattern.search("Cu in HCl with 1 mM my new inhibitor")
            self.assertIsNotNone(m)
            self.assertEqual(m.group("inh").lower(), "my new inhibitor")

            # Custom JSON with zero inhibitors must not match empty strings
            no_inh_path = os.path.join(tmpdir, "no_inhibitors.json")
            with open(no_inh_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "role_fields": {"acid": "acids"},
                        "species": [{"canonical": "HCl", "role": "acid"}],
                    },
                    f,
                )
            no_inh_reg = load_chemical_species_registry(no_inh_path)
            self.assertIsNone(no_inh_reg.inhibitor_name_pattern.search("Cu in 1 M HCl"))

    def test_step5_run_extract_end_to_end_csv_schema(self):
        import json
        import step5_extract_data as s5

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _make_test_ctx(tmpdir, "Cu")
            ctx.ensure_dirs()

            txt_path = os.path.join(ctx.text_dir, "10.1000_e2e.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(
                    "Polished copper thin film was tested by potentiodynamic polarization "
                    "in deaerated 0.5 M H2SO4 (pH 1.0) at 25 °C vs. SCE. "
                    "Table 1 shows the Tafel polarization parameters for Cu."
                )

            tables_path = os.path.join(ctx.text_dir, "10.1000_e2e_tables.json")
            with open(tables_path, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        [
                            [
                                "Sample",
                                "Ecorr (mV vs SCE)",
                                "icorr (uA/cm2)",
                                "ba (mV/dec)",
                                "bc (mV/dec)",
                            ],
                            ["Pure Cu", "-180", "18.5", "75", "115"],
                        ]
                    ],
                    f,
                )

            s5.run_extract(ctx=ctx)

            for out_path in (
                ctx.step5_material_output,
                ctx.step5_solution_output,
                ctx.step5_echem_output,
                ctx.step5_raw_output,
                ctx.step5_outliers_output,
            ):
                self.assertTrue(os.path.exists(out_path), f"Missing output: {out_path}")

            mat_df = pd.read_csv(ctx.step5_material_output)
            sol_df = pd.read_csv(ctx.step5_solution_output)
            echem_df = pd.read_csv(ctx.step5_echem_output)
            raw_df = pd.read_csv(ctx.step5_raw_output)

            self.assertEqual(len(echem_df), 1)
            self.assertEqual(len(raw_df), 1)
            self.assertGreaterEqual(len(mat_df), 1)
            self.assertGreaterEqual(len(sol_df), 1)

            for col in (
                "record_id",
                "material_id",
                "solution_id",
                "Ecorr_V_SHE",
                "icorr_uA_cm2",
                "beta_a_mV_dec",
                "beta_c_mV_dec",
                "pH",
                "temperature_C",
                "aeration_condition",
                "data_quality_score",
            ):
                self.assertIn(col, echem_df.columns)

            for col in ("solution_id", "solution_name", "solution_category", "components_json"):
                self.assertIn(col, sol_df.columns)

            for col in (
                "material_id",
                "material_name",
                "symbol",
                "category",
                "density_g_cm3",
                "standard_potential_V_SHE",
            ):
                self.assertIn(col, mat_df.columns)

            for col in (
                "source_id",
                "material_id_assigned",
                "solution_id_assigned",
                "Ecorr_V_SHE",
                "icorr_uA_cm2",
                "beta_a_mV_dec",
                "beta_c_mV_dec",
                "extraction_method",
            ):
                self.assertIn(col, raw_df.columns)

            self.assertEqual(echem_df.loc[0, "material_id"], "Cu")
            self.assertEqual(echem_df.loc[0, "aeration_condition"], "deaerated")
            self.assertEqual(echem_df.loc[0, "beta_a_mV_dec"], 75.0)
            self.assertEqual(echem_df.loc[0, "beta_c_mV_dec"], 115.0)


if __name__ == "__main__":
    unittest.main()
