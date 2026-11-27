"""Unit tests for step4_extract_text.py."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import pymupdf as fitz

import step4_extract_text


def _create_sample_pdf(pdf_path: Path, text: str = "Sample Tafel text") -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    # Draw a simple 2x2 grid so pdfplumber has lines if needed
    page.draw_rect(fitz.Rect(72, 100, 272, 160))
    doc.save(str(pdf_path))
    doc.close()


class TestStep4ExtractText(unittest.TestCase):
    def test_extract_tables_with_timeout_real_worker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "sample.pdf"
            _create_sample_pdf(pdf_path)

            tables, timed_out = step4_extract_text.extract_tables_with_timeout(
                str(pdf_path),
                timeout_sec=15,
            )
            self.assertFalse(timed_out)
            self.assertIsInstance(tables, list)

    def test_extract_tables_with_timeout_handles_timeout_and_cleans_tempfile(self):
        created_tmp_paths: list[str] = []
        orig_ntf = tempfile.NamedTemporaryFile

        def tracking_ntf(*args, **kwargs):
            handle = orig_ntf(*args, **kwargs)
            created_tmp_paths.append(handle.name)
            return handle

        with (
            mock.patch.object(step4_extract_text.tempfile, "NamedTemporaryFile", side_effect=tracking_ntf),
            mock.patch.object(
                step4_extract_text.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired(cmd=["python"], timeout=2),
            ),
        ):
            tables, timed_out = step4_extract_text.extract_tables_with_timeout(
                "dummy.pdf",
                timeout_sec=2,
            )

        self.assertEqual(tables, [])
        self.assertTrue(timed_out)
        self.assertEqual(len(created_tmp_paths), 1)
        self.assertFalse(Path(created_tmp_paths[0]).exists())

    def test_run_extract_writes_empty_tables_on_timeout_and_skips_on_rerun(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_dir = Path(tmpdir) / "pdfs"
            text_dir = Path(tmpdir) / "texts"
            pdf_dir.mkdir()
            text_dir.mkdir()

            pdf_path = pdf_dir / "paper.pdf"
            _create_sample_pdf(pdf_path, "Ecorr = -0.25 V")

            with mock.patch.object(
                step4_extract_text,
                "extract_tables_with_timeout",
                return_value=([], True),
            ) as mock_extract:
                step4_extract_text.run_extract(
                    pdf_dir=str(pdf_dir),
                    text_dir=str(text_dir),
                    table_timeout_sec=5,
                )
                self.assertEqual(mock_extract.call_count, 1)

            txt_file = text_dir / "paper.txt"
            tbl_file = text_dir / "paper_tables.json"
            self.assertTrue(txt_file.exists())
            self.assertIn("Ecorr = -0.25 V", txt_file.read_text(encoding="utf-8"))
            self.assertTrue(tbl_file.exists())
            self.assertEqual(json.loads(tbl_file.read_text(encoding="utf-8")), [])

            with mock.patch.object(
                step4_extract_text,
                "extract_tables_with_timeout",
                side_effect=AssertionError("Should skip already extracted PDF"),
            ):
                step4_extract_text.run_extract(
                    pdf_dir=str(pdf_dir),
                    text_dir=str(text_dir),
                    table_timeout_sec=5,
                )


if __name__ == "__main__":
    unittest.main()
