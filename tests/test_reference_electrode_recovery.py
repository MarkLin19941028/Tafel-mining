"""Unit tests for Tier 1 reference electrode recovery and Tier 2a JATS <fig> caption retention."""

import unittest

from europepmc_fetcher import parse_jats_xml
from step5_constants import convert_to_she
from step5_text_extractor import extract_ref_electrodes, normalise_ref_electrode


class TestReferenceElectrodeRecovery(unittest.TestCase):
    def test_extract_ref_electrodes_variants(self):
        cases = [
            ("A silver/silver chloride reference electrode was used.", "Ag/AgCl"),
            ("Measurements used a silver-silver chloride electrode.", "Ag/AgCl"),
            ("The reference electrode was a silver chloride electrode.", "Ag/AgCl"),
            ("The reference electrode was Ag-AgCl.", "Ag/AgCl"),
            ("AgCl/Ag, 3 M KCl served as the reference electrode.", "Ag/AgCl"),
            ("The reference electrode was AgǁAgCl and the counter was graphite.", "Ag/AgCl"),
            ("PDF line break: Ag/\nAgCl (sat. KCl) reference.", "Ag/AgCl"),
            ("PDF hyphen break: silver-\nsilver chloride electrode.", "Ag/AgCl"),
            ("A calomel reference electrode was used for polarization.", "SCE"),
            ("Saturated Kalmel electrode (reference electrode).", "SCE"),
            ("Potential was measured against Hg/Hg2Cl2.", "SCE"),
            ("Ecorr (mVSCE) = -250 and E_pit (VSCE) = 0.12.", "SCE"),
            ("The reference electrode was saturated mercurous sulfate (SSE).", "Hg/Hg2SO4"),
            ("Potential measured vs. SSE at 25 °C.", "Hg/Hg2SO4"),
            ("Using an SMSE reference electrode in sulfuric acid.", "Hg/Hg2SO4"),
            ("A copper/copper sulfate reference electrode was used.", "Cu/CuSO4"),
            ("Potential measured vs. CSE in soil electrolyte.", "Cu/CuSO4"),
            ("Measured against Cu/CuSO4.", "Cu/CuSO4"),
            ("Potential reported in VRHE and mVSHE.", "RHE"),
        ]
        for text, expected in cases:
            with self.subTest(text=text, expected=expected):
                counts = extract_ref_electrodes(text)
                self.assertIn(expected, counts, f"Expected {expected!r} in {dict(counts)}")

    def test_extract_ref_electrodes_guardrails_no_false_positives(self):
        """Bare SSE (solid-state electrolyte) and CSE (non-electrochemical) must not false-match."""
        battery_text = (
            "All-solid-state lithium batteries using a sulfide solid-state electrolyte (SSE) "
            "were tested in coin cells. The CSE module showed high efficiency."
        )
        self.assertEqual(dict(extract_ref_electrodes(battery_text)), {})

    def test_convert_to_she_new_electrodes(self):
        self.assertAlmostEqual(convert_to_she(0.0, "Cu/CuSO4"), 0.314, places=6)
        self.assertAlmostEqual(convert_to_she(0.0, "CSE"), 0.314, places=6)
        self.assertAlmostEqual(convert_to_she(0.0, "Hg/Hg2Cl2"), 0.241, places=6)
        self.assertAlmostEqual(convert_to_she(0.0, "SMSE"), 0.640, places=6)
        self.assertAlmostEqual(convert_to_she(0.0, "MMS"), 0.640, places=6)
        self.assertEqual(normalise_ref_electrode("Cu/CuSO4"), "Cu/CuSO4")
        self.assertEqual(normalise_ref_electrode("Hg/Hg2Cl2"), "SCE")

    def test_parse_jats_xml_retains_fig_caption(self):
        xml = """
        <article>
          <body>
            <sec>
              <p>Potentiodynamic polarization tests were performed in 3.5 wt% NaCl.</p>
              <fig id="f1">
                <label>Fig. 3.</label>
                <caption><p>Potentiodynamic polarization curves of Ti in NaCl (vs. SCE).</p></caption>
                <graphic xlink:href="f1.jpg" xmlns:xlink="http://www.w3.org/1999/xlink"/>
              </fig>
            </sec>
          </body>
        </article>
        """
        plain_text, _ = parse_jats_xml(xml)
        self.assertIn("Fig. 3.", plain_text)
        self.assertIn("vs. SCE", plain_text)
        self.assertIn("SCE", extract_ref_electrodes(plain_text))


if __name__ == "__main__":
    unittest.main()
