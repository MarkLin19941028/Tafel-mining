"""Unit tests for auto_run.py."""

import csv
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import auto_run


SAMPLE_REPORT = """# Tafel Mining 資料品質報告

> 產生時間：2026-11-27 02:18:15

## 1. 總覽

| 項目 | 數量 |
|---|---|
| 電化學主表 (`electrochemical_database.csv`) | 6925 |

## 4. 各金屬電化學數值分佈

| 金屬 | 總筆數 | Ecorr (V vs SHE) 中位數 [min, max] | icorr (µA/cm²) 中位數 [min, max] | βa / βc 有效筆數 |
|---|---:|---|---|---:|
| Ag | 58 | 0.242 [-0.799, 1.497] (n=45) | 1.58 [0.0055, 1e+03] (n=30) | 9 / 9 |
| Al | 341 | -0.380 [-1.559, 0.226] (n=257) | 3.55 [0.0001, 3.2e+03] (n=232) | 53 / 55 |
| Co | 424 | -0.021 [-1.175, 0.626] (n=331) | 2.47 [0.0001, 9.5e+05] (n=268) | 106 / 82 |
| Cu | 1,910 | 0.094 [-1.153, 0.789] (n=1372) | 6.16 [0.0001, 1e+06] (n=1441) | 707 / 641 |
| Mn | 21 | -0.025 [-1.083, 0.598] (n=19) | 14.6 [0.011, 79.4] (n=12) | 5 / 5 |
| Mo | 363 | -0.032 [-0.964, 0.748] (n=264) | 1.35 [0.0001, 1.38e+04] (n=182) | 48 / 46 |
| Ru | 114 | 0.355 [-0.499, 1.400] (n=66) | 135 [0.579, 3.47e+03] (n=21) | 2 / 2 |
| Ta | 161 | 0.013 [-37.759, 0.797] (n=123) | 0.231 [0.0017, 9e+03] (n=104) | 23 / 24 |
| TaN | 6 | 0.052 [0.027, 0.077] (n=2) | 3.36 [2.89, 5.66] (n=4) | 0 / 0 |
| Ti | 1355 | -0.017 [-37.759, 0.799] (n=1059) | 0.814 [0.0001, 3.73e+05] (n=956) | 279 / 294 |
| TiN | 119 | 0.140 [-0.696, 1.000] (n=85) | 4.88 [0.0001, 9.98e+03] (n=76) | 7 / 4 |
| W | 450 | -0.090 [-0.903, 0.780] (n=315) | 2.6 [0.0001, 3e+03] (n=335) | 89 / 90 |
| WN | 3 | -0.474 [-0.493, -0.455] (n=2) | 4.47 [0.776, 7.28] (n=3) | 0 / 0 |

## 5. 自動過濾離群值摘要

| 金屬 | 總筆數 | 備註 |
|---|---:|---|
| Cu | 99999 | 不應被第 4 節解析器讀到 |
"""


def _build_report_text(counts: dict[str, int]) -> str:
    rows = "\n".join(
        f"| {metal} | {cnt} | 0.0 [0.0, 0.0] (n={cnt}) | 1.0 [0.1, 2.0] (n={cnt}) | 1 / 1 |"
        for metal, cnt in counts.items()
    )
    return (
        "# Tafel Mining 資料品質報告\n\n"
        "## 4. 各金屬電化學數值分佈\n\n"
        "| 金屬 | 總筆數 | Ecorr | icorr | βa / βc |\n"
        "|---|---:|---|---|---:|\n"
        f"{rows}\n"
    )


class TestAutoRun(unittest.TestCase):
    def test_parse_quality_report_counts_and_missing_metal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "quality_report.md"
            report_path.write_text(SAMPLE_REPORT, encoding="utf-8")

            parsed = auto_run.parse_quality_report_counts(report_path)
            self.assertEqual(parsed["Cu"], 1910)
            self.assertEqual(parsed["Ti"], 1355)
            self.assertEqual(parsed["Ag"], 58)
            self.assertNotIn("MoN", parsed)

            tracked = auto_run.extract_tracked_counts(parsed, auto_run.DEFAULT_METALS)
            self.assertEqual(list(tracked.keys()), auto_run.DEFAULT_METALS)
            self.assertEqual(tracked["MoN"], 0)
            self.assertEqual(tracked["Cu"], 1910)

    def test_backup_merged_data_and_collision_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            merged_dir = root / "data" / "merged"
            old_data_dir = root / "old data"
            merged_dir.mkdir(parents=True)
            (merged_dir / "quality_report.md").write_text("report v1", encoding="utf-8")
            (merged_dir / "nested").mkdir()
            (merged_dir / "nested" / "extra.csv").write_text("a,b\n1,2\n", encoding="utf-8")

            fixed_time = datetime(2026, 11, 27, 2, 45)
            dest1 = auto_run.backup_merged_data(merged_dir, old_data_dir, now=fixed_time)
            self.assertEqual(dest1, old_data_dir / "20261127-0245")
            self.assertEqual((dest1 / "quality_report.md").read_text(encoding="utf-8"), "report v1")
            self.assertEqual((dest1 / "nested" / "extra.csv").read_text(encoding="utf-8"), "a,b\n1,2\n")

            (merged_dir / "quality_report.md").write_text("report v2", encoding="utf-8")
            dest2 = auto_run.backup_merged_data(merged_dir, old_data_dir, now=fixed_time)
            self.assertEqual(dest2, old_data_dir / "20261127-0245_2")
            self.assertEqual((dest2 / "quality_report.md").read_text(encoding="utf-8"), "report v2")

    def test_filter_metals_below_threshold_boundaries(self):
        metals = ["Cu", "Co", "W", "MoN"]
        counts = {"Cu": 5000, "Co": 5001, "W": 4999, "MoN": 0}
        remaining, excluded = auto_run.filter_metals_below_threshold(metals, counts, threshold=5000)
        self.assertEqual(remaining, ["W", "MoN"])
        self.assertEqual(excluded, ["Cu", "Co"])

    def test_run_workflow_two_rounds_excludes_threshold_metals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            merged_dir = root / "data" / "merged"
            old_data_dir = root / "old data"
            merged_dir.mkdir(parents=True)
            report_path = merged_dir / "quality_report.md"
            history_csv = old_data_dir / "auto_run_history.csv"

            # 初始狀態
            report_path.write_text(
                _build_report_text({"Cu": 1910, "Ti": 1355, "W": 450}),
                encoding="utf-8",
            )
            (merged_dir / "electrochemical_database.csv").write_text("id\n1\n", encoding="utf-8")

            invoked_metals: list[list[str]] = []

            def fake_run_pipeline(metals, search_modes, run_all_path, cwd, dry_run=False):
                invoked_metals.append(list(metals))
                self.assertEqual(search_modes, auto_run.DEFAULT_SEARCH_MODES)
                if len(invoked_metals) == 1:
                    # Round 1 後：Cu=5000, Ti=5123 達標，W=4999 未達標，MoN 仍未出現
                    report_path.write_text(
                        _build_report_text({"Cu": 5000, "Ti": 5123, "W": 4999}),
                        encoding="utf-8",
                    )
                else:
                    # Round 2 後：W 提升至 5200，MoN 新增 12 筆
                    report_path.write_text(
                        _build_report_text({"Cu": 5000, "Ti": 5123, "W": 5200, "MoN": 12}),
                        encoding="utf-8",
                    )
                return 0

            with mock.patch("auto_run.run_pipeline_command", side_effect=fake_run_pipeline):
                rc = auto_run.run_workflow(
                    rounds=2,
                    threshold=5000,
                    merged_dir=merged_dir,
                    old_data_dir=old_data_dir,
                    quality_report_path=report_path,
                    run_all_path=root / "run_all.py",
                    history_csv=history_csv,
                    cwd=root,
                )

            self.assertEqual(rc, 0)
            self.assertEqual(len(invoked_metals), 2)
            self.assertEqual(invoked_metals[0], auto_run.DEFAULT_METALS)
            expected_round2 = [m for m in auto_run.DEFAULT_METALS if m not in {"Cu", "Ti"}]
            self.assertEqual(invoked_metals[1], expected_round2)

            backup_dirs = [p for p in old_data_dir.iterdir() if p.is_dir()]
            self.assertEqual(len(backup_dirs), 2)
            for bdir in backup_dirs:
                self.assertTrue((bdir / "quality_report.md").exists())
                self.assertTrue((bdir / "electrochemical_database.csv").exists())

            with history_csv.open("r", encoding="utf-8", newline="") as fp:
                rows = list(csv.DictReader(fp))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["round"], "1")
            self.assertEqual(rows[0]["Cu"], "5000")
            self.assertEqual(rows[0]["Ti"], "5123")
            self.assertEqual(rows[0]["W"], "4999")
            self.assertEqual(rows[0]["MoN"], "0")
            self.assertEqual(rows[0]["excluded_metals"], "Cu,Ti")
            self.assertEqual(rows[1]["round"], "2")
            self.assertEqual(rows[1]["W"], "5200")
            self.assertEqual(rows[1]["MoN"], "12")
            self.assertEqual(rows[1]["excluded_metals"], "Cu,W,Ti")

    def test_run_workflow_stops_early_after_final_backup_when_all_metals_reach_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            merged_dir = root / "data" / "merged"
            old_data_dir = root / "old data"
            merged_dir.mkdir(parents=True)
            report_path = merged_dir / "quality_report.md"
            history_csv = old_data_dir / "auto_run_history.csv"

            report_path.write_text(_build_report_text({"Cu": 100, "Co": 200}), encoding="utf-8")

            invoked_calls = 0

            def fake_run_pipeline(metals, search_modes, run_all_path, cwd, dry_run=False):
                nonlocal invoked_calls
                invoked_calls += 1
                report_path.write_text(
                    _build_report_text({"Cu": 5000, "Co": 6500}),
                    encoding="utf-8",
                )
                return 0

            with mock.patch("auto_run.run_pipeline_command", side_effect=fake_run_pipeline):
                rc = auto_run.run_workflow(
                    rounds=5,
                    threshold=5000,
                    metals=["Cu", "Co"],
                    merged_dir=merged_dir,
                    old_data_dir=old_data_dir,
                    quality_report_path=report_path,
                    run_all_path=root / "run_all.py",
                    history_csv=history_csv,
                    cwd=root,
                )

            self.assertEqual(rc, 0)
            self.assertEqual(invoked_calls, 1)
            backup_dirs = [p for p in old_data_dir.iterdir() if p.is_dir()]
            self.assertEqual(len(backup_dirs), 2)
            with history_csv.open("r", encoding="utf-8", newline="") as fp:
                rows = list(csv.DictReader(fp))
            self.assertEqual(len(rows), 1)

    def test_case_insensitive_metal_matching(self):
        counts = {"Cu": 3200, "TiN": 500}
        tracked = auto_run.extract_tracked_counts(counts, ["cu", "tin", "mon"])
        self.assertEqual(tracked, {"cu": 3200, "tin": 500, "mon": 0})
        remaining, excluded = auto_run.filter_metals_below_threshold(
            ["cu", "tin", "mon"],
            counts,
            threshold=3000,
        )
        self.assertEqual(remaining, ["tin", "mon"])
        self.assertEqual(excluded, ["cu"])

    def test_parse_quality_report_raises_when_section4_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "quality_report.md"
            report_path.write_text("# Tafel Mining 資料品質報告\n\n## 1. 總覽\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                auto_run.parse_quality_report_counts(report_path)

    def test_record_counts_aligns_csv_columns_when_metals_change(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_csv = Path(tmpdir) / "auto_run_history.csv"
            auto_run.record_counts(
                round_idx=1,
                tracked_counts={"Cu": 1000, "Co": 500},
                prev_counts=None,
                threshold=3000,
                history_csv=history_csv,
            )
            auto_run.record_counts(
                round_idx=2,
                tracked_counts={"Co": 600, "Cu": 3100, "Ti": 400},
                prev_counts={"Cu": 1000, "Co": 500},
                threshold=3000,
                history_csv=history_csv,
            )
            with history_csv.open("r", encoding="utf-8", newline="") as fp:
                rows = list(csv.DictReader(fp))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["Cu"], "1000")
            self.assertEqual(rows[0]["Co"], "500")
            self.assertEqual(rows[0]["Ti"], "")
            self.assertEqual(rows[1]["Cu"], "3100")
            self.assertEqual(rows[1]["Co"], "600")
            self.assertEqual(rows[1]["Ti"], "400")

    def test_run_workflow_stops_immediately_on_sigint_exit_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            merged_dir = root / "data" / "merged"
            old_data_dir = root / "old data"
            merged_dir.mkdir(parents=True)
            report_path = merged_dir / "quality_report.md"
            report_path.write_text(_build_report_text({"Cu": 100}), encoding="utf-8")

            with mock.patch("auto_run.run_pipeline_command", return_value=130) as mocked_run:
                rc = auto_run.run_workflow(
                    rounds=5,
                    threshold=3000,
                    metals=["Cu"],
                    merged_dir=merged_dir,
                    old_data_dir=old_data_dir,
                    quality_report_path=report_path,
                    run_all_path=root / "run_all.py",
                    history_csv=old_data_dir / "auto_run_history.csv",
                    cwd=root,
                )
            self.assertEqual(rc, 130)
            self.assertEqual(mocked_run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
