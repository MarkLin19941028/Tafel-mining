#!/usr/bin/env python3
"""
自動批次執行腳本（備份 data/merged -> 執行 run_all.py -> 解析並記錄 quality_report.md 統計 -> 達標金屬下輪剔除）。

預設流程：
1) 於 old data/ 建立 yyyymmdd-hhmm 資料夾並複製 data/merged/ 所有檔案
2) 執行 python run_all.py Cu Co W Ru Ti TiN TaN WN MoN Al Ta Mn Mo --search-mode solvent corrosion cmp_etch inhibitor --append
3) 檢查 data/merged/quality_report.md「4. 各金屬電化學數值分佈」並記錄各目標金屬累計總筆數
4) 重複執行 1)
5) 重複執行 2)，並將上一輪已達門檻（預設 >= DEFAULT_THRESHOLD，可於下方自定義）的金屬從指令中剔除
6) 重複執行 3)
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import re
import shutil
import subprocess
import sys

import config

BASE_DIR = Path(__file__).resolve().parent
_SETTINGS = config.load_settings()

# === 可自定義設定區（預設金屬、檢索模式與合併目錄統一源自 user_settings.py / PipelineSettings） ===
DEFAULT_METALS = list(_SETTINGS.metals_to_run)
DEFAULT_SEARCH_MODES = list(_SETTINGS.default_search_modes)
DEFAULT_THRESHOLD = 1000  # 單一金屬累計筆數達標門檻（達此值將於下一輪從 run_all.py 指令中剔除）；建議 3000~5000（快速建模建議 3000，完整訓練集建議 5000，深度擴充可用 8000~10000）
DEFAULT_ROUNDS = 50  # 自動執行總輪數（每輪依序執行：備份 -> run_all.py -> 檢查與記錄報告）；建議先設 2~3（單輪耗時較長，建議 2 輪後再視成長幅度與硬碟空間調整）

MERGED_DIR = _SETTINGS.simulator_data_dir  # step6/step7 合併輸出與每輪備份的來源資料夾
OLD_DATA_DIR = BASE_DIR / "old data"  # 歷史備份資料夾（會在底下建立 yyyymmdd-hhmm 子資料夾）；建議維持 BASE_DIR / "old data"（若本機空間不足可改為外接硬碟路徑）
QUALITY_REPORT_PATH = MERGED_DIR / "quality_report.md"  # step6 產生的品質報告路徑（用來解析第 4 節各金屬筆數）
RUN_ALL_PATH = BASE_DIR / "run_all.py"  # 主管線執行腳本路徑；建議維持 BASE_DIR / "run_all.py"
DEFAULT_HISTORY_CSV = OLD_DATA_DIR / "auto_run_history.csv"  # 每輪累計筆數的 CSV 紀錄檔路徑；建議維持 OLD_DATA_DIR / "auto_run_history.csv"

_SECTION4_RE = re.compile(r"^##\s*4\.\s*各金屬電化學數值分佈\b")
_NEXT_SECTION_RE = re.compile(r"^##\s+")
_SEP_CELL_RE = re.compile(r"^[:\-\s]+$")


def _resolve_unique_backup_dir(old_data_dir: Path, base_name: str) -> Path:
    """回傳不與現有資料夾衝突的備份目的路徑（若同名已存在則依序加 _2, _3...）。"""
    candidate = old_data_dir / base_name
    if not candidate.exists():
        return candidate
    suffix = 2
    while True:
        candidate = old_data_dir / f"{base_name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def backup_merged_data(
    merged_dir: Path,
    old_data_dir: Path,
    now: datetime | None = None,
    dry_run: bool = False,
) -> Path:
    """在 old data/ 下建立 yyyymmdd-hhmm 資料夾並複製 data/merged/ 底下所有資料。"""
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M")
    dest_dir = _resolve_unique_backup_dir(old_data_dir, timestamp)

    if dry_run:
        print(f"[Dry-Run] 備份：{merged_dir} -> {dest_dir}", flush=True)
        return dest_dir

    old_data_dir.mkdir(parents=True, exist_ok=True)
    if merged_dir.exists():
        shutil.copytree(merged_dir, dest_dir)
        print(f"📦 已備份 {merged_dir} -> {dest_dir}", flush=True)
    else:
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"⚠️  來源資料夾不存在（已建立空備份資料夾）：{merged_dir} -> {dest_dir}",
            flush=True,
        )
    return dest_dir


def parse_quality_report_counts(report_path: Path) -> dict[str, int]:
    """解析 quality_report.md 第 4 節「各金屬電化學數值分佈」的各金屬總筆數。"""
    if not report_path.exists():
        raise FileNotFoundError(f"找不到品質報告：{report_path}")

    lines = report_path.read_text(encoding="utf-8").splitlines()
    found_section4 = False
    in_section4 = False
    counts: dict[str, int] = {}

    for raw_line in lines:
        line = raw_line.strip()
        if not in_section4:
            if _SECTION4_RE.match(line):
                found_section4 = True
                in_section4 = True
            continue

        if _NEXT_SECTION_RE.match(line):
            break

        if not line.startswith("|"):
            continue

        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 2:
            continue

        metal, raw_count = cells[0], cells[1]
        if not metal or metal == "金屬":
            continue
        if _SEP_CELL_RE.match(metal) or _SEP_CELL_RE.match(raw_count):
            continue

        normalized_count = raw_count.replace(",", "").strip()
        try:
            counts[metal] = int(normalized_count)
        except ValueError:
            continue

    if not found_section4:
        raise ValueError(f"品質報告中找不到「## 4. 各金屬電化學數值分佈」區段：{report_path}")

    return counts


def extract_tracked_counts(
    all_counts: dict[str, int],
    tracked_metals: list[str],
) -> dict[str, int]:
    """依目標金屬順序取出累計筆數（大小寫不相依），報告中尚未出現的金屬預設為 0。"""
    counts_ci = {k.lower(): v for k, v in all_counts.items()}
    return {metal: counts_ci.get(metal.lower(), 0) for metal in tracked_metals}


def filter_metals_below_threshold(
    metals: list[str],
    counts: dict[str, int],
    threshold: int,
) -> tuple[list[str], list[str]]:
    """將未達門檻 (< threshold) 與已達/超過門檻 (>= threshold) 的金屬分開並保留原順序（大小寫不相依）。"""
    counts_ci = {k.lower(): v for k, v in counts.items()}
    remaining: list[str] = []
    excluded: list[str] = []
    for metal in metals:
        if counts_ci.get(metal.lower(), 0) >= threshold:
            excluded.append(metal)
        else:
            remaining.append(metal)
    return remaining, excluded


def _append_history_csv(
    history_csv: Path,
    round_idx: int,
    threshold: int,
    total_tracked: int,
    excluded: list[str],
    tracked_counts: dict[str, int],
) -> None:
    """將本輪紀錄安全追加至 CSV；若目標金屬欄位變更則自動擴充並對齊既有標頭。"""
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    metals_order = list(tracked_counts.keys())
    base_cols = ["timestamp", "round", "threshold", "total_tracked", "excluded_metals"]
    target_header = [*base_cols, *metals_order]
    new_row: dict[str, str | int] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "round": round_idx,
        "threshold": threshold,
        "total_tracked": total_tracked,
        "excluded_metals": ",".join(excluded),
        **tracked_counts,
    }

    if not history_csv.exists() or history_csv.stat().st_size == 0:
        with history_csv.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=target_header)
            writer.writeheader()
            writer.writerow(new_row)
        return

    with history_csv.open("r", encoding="utf-8", newline="") as fp:
        existing_header = next(csv.reader(fp), [])

    missing_cols = [col for col in target_header if col not in existing_header]
    if not missing_cols:
        with history_csv.open("a", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=existing_header, restval="", extrasaction="ignore")
            writer.writerow(new_row)
        return

    union_header = [*existing_header, *missing_cols]
    with history_csv.open("r", encoding="utf-8", newline="") as fp:
        existing_rows = list(csv.DictReader(fp))
    with history_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=union_header, restval="")
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerow(new_row)


def record_counts(
    round_idx: int,
    tracked_counts: dict[str, int],
    prev_counts: dict[str, int] | None,
    threshold: int,
    history_csv: Path,
    dry_run: bool = False,
) -> None:
    """於終端列印各目標金屬統計表，並將紀錄追加寫入 CSV 歷史檔。"""
    total_tracked = sum(tracked_counts.values())
    _, excluded = filter_metals_below_threshold(
        list(tracked_counts.keys()),
        tracked_counts,
        threshold,
    )

    print(f"\n📊 [Round {round_idx}] 各目標金屬累計筆數（門檻 >= {threshold}）：", flush=True)
    print(f"  {'金屬':<6}  {'總筆數':>8}  {'本輪變化':>10}  {'狀態'}", flush=True)
    print("  " + "-" * 48, flush=True)
    for metal, count in tracked_counts.items():
        if prev_counts is None:
            delta_str = "—"
        else:
            delta = count - prev_counts.get(metal, 0)
            delta_str = f"{delta:+d}"
        status = f">= {threshold} (下輪剔除)" if count >= threshold else "繼續"
        print(f"  {metal:<6}  {count:>8}  {delta_str:>10}  {status}", flush=True)
    print("  " + "-" * 48, flush=True)
    print(f"  目標金屬合計：{total_tracked} 筆", flush=True)

    if dry_run:
        print(f"[Dry-Run] 略過寫入紀錄檔：{history_csv}", flush=True)
        return

    _append_history_csv(
        history_csv=history_csv,
        round_idx=round_idx,
        threshold=threshold,
        total_tracked=total_tracked,
        excluded=excluded,
        tracked_counts=tracked_counts,
    )
    print(f"📝 已追加紀錄至 {history_csv}", flush=True)


def _format_cmd_for_log(cmd: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in cmd)


def run_pipeline_command(
    metals: list[str],
    search_modes: list[str],
    run_all_path: Path,
    cwd: Path,
    dry_run: bool = False,
) -> int:
    """呼叫 run_all.py 執行指定金屬與搜尋模式的增量管線。"""
    cmd = [
        sys.executable,
        str(run_all_path),
        *metals,
        "--search-mode",
        *search_modes,
        "--append",
    ]
    prefix = "[Dry-Run] 執行指令：" if dry_run else "🚀 執行指令："
    print(f"{prefix}{_format_cmd_for_log(cmd)}", flush=True)
    if dry_run:
        return 0

    proc = subprocess.run(cmd, cwd=str(cwd))
    return proc.returncode


def run_workflow(
    rounds: int = DEFAULT_ROUNDS,
    threshold: int = DEFAULT_THRESHOLD,
    metals: list[str] | None = None,
    search_modes: list[str] | None = None,
    merged_dir: Path = MERGED_DIR,
    old_data_dir: Path = OLD_DATA_DIR,
    quality_report_path: Path = QUALITY_REPORT_PATH,
    run_all_path: Path = RUN_ALL_PATH,
    history_csv: Path = DEFAULT_HISTORY_CSV,
    cwd: Path = BASE_DIR,
    stop_on_error: bool = False,
    dry_run: bool = False,
) -> int:
    """依序執行多輪「備份 -> run_all.py -> 檢查統計與過濾門檻」。"""
    if rounds < 1:
        raise ValueError(f"rounds 必須大於等於 1，目前為 {rounds}")

    target_metals = list(dict.fromkeys(metals if metals is not None else DEFAULT_METALS))
    target_modes = list(search_modes if search_modes is not None else DEFAULT_SEARCH_MODES)

    last_counts: dict[str, int] | None = None

    for round_idx in range(1, rounds + 1):
        print("\n" + "=" * 65, flush=True)
        print(f"  AUTO RUN — Round {round_idx} / {rounds}", flush=True)
        print("=" * 65, flush=True)

        # 步驟 1 / 4：備份 data/merged/ 到 old data/yyyymmdd-hhmm
        backup_merged_data(merged_dir, old_data_dir, dry_run=dry_run)

        # 步驟 2 / 5：決定本輪要執行的金屬清單並執行 run_all.py
        if round_idx == 1:
            active_metals = list(target_metals)
            excluded_metals: list[str] = []
        else:
            active_metals, excluded_metals = filter_metals_below_threshold(
                target_metals,
                last_counts or {},
                threshold,
            )
            if excluded_metals:
                print(
                    f"✂️  依上一輪紀錄已達/超過 {threshold} 筆，剔除金屬：{', '.join(excluded_metals)}",
                    flush=True,
                )
            else:
                print(f"ℹ️  上一輪尚無金屬達到 {threshold} 筆，維持全部目標金屬。", flush=True)

        if not active_metals:
            print(
                f"✅ 所有目標金屬皆已達到或超過 {threshold} 筆，已完成最後一次備份並提前結束後續輪次。",
                flush=True,
            )
            break

        report_mtime_before = (
            quality_report_path.stat().st_mtime if quality_report_path.exists() else None
        )
        rc = run_pipeline_command(
            active_metals,
            target_modes,
            run_all_path=run_all_path,
            cwd=cwd,
            dry_run=dry_run,
        )
        if rc in (130, -2):
            print("\n🛑 偵測到使用者中斷（SIGINT），提前中止自動執行流程。", flush=True)
            return 130
        if rc != 0:
            print(
                f"⚠️  run_all.py 回傳非零結束碼：{rc}（將繼續檢查品質報告並記錄筆數）",
                flush=True,
            )
            if stop_on_error:
                return rc
        if (
            not dry_run
            and quality_report_path.exists()
            and report_mtime_before is not None
            and quality_report_path.stat().st_mtime == report_mtime_before
        ):
            print(
                "⚠️  本輪執行後 quality_report.md 修改時間未更新（可能 Step 6 未成功產出新報告）。",
                flush=True,
            )

        # 步驟 3 / 6：檢查 quality_report.md 第 4 節並記錄各金屬累計筆數
        try:
            all_counts = parse_quality_report_counts(quality_report_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"❌ 讀取或解析品質報告失敗：{exc}", flush=True)
            if stop_on_error or last_counts is None:
                return 1
            print("⚠️  保留上一輪各金屬筆數紀錄以繼續執行下一輪。", flush=True)
            continue

        tracked_counts = extract_tracked_counts(all_counts, target_metals)
        record_counts(
            round_idx=round_idx,
            tracked_counts=tracked_counts,
            prev_counts=last_counts,
            threshold=threshold,
            history_csv=history_csv,
            dry_run=dry_run,
        )
        last_counts = tracked_counts

    return 0


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="自動備份 data/merged、執行 run_all.py 並依 quality_report.md 門檻過濾金屬"
    )
    parser.add_argument(
        "--rounds",  # 總執行輪數；建議 2~3（與 DEFAULT_ROUNDS 相同）
        type=int,
        default=DEFAULT_ROUNDS,
        help=f"執行輪數（預設 {DEFAULT_ROUNDS}）",
    )
    parser.add_argument(
        "--threshold",  # 達標剔除門檻；建議 3000~5000（深度擴充可設 8000~10000）
        type=int,
        default=DEFAULT_THRESHOLD,
        help=f"累計總筆數達此門檻即於下一輪從指令中剔除（預設 {DEFAULT_THRESHOLD}）",
    )
    parser.add_argument(
        "--metals",  # 要執行與追蹤的目標金屬清單；建議維持預設 13 種或依需求指定子集
        nargs="+",
        default=list(DEFAULT_METALS),
        help="要執行與追蹤的目標金屬清單",
    )
    parser.add_argument(
        "--search-mode",  # 傳給 run_all.py 的檢索模式；建議保留預設 4 種（solvent corrosion cmp_etch inhibitor）
        nargs="+",
        default=list(DEFAULT_SEARCH_MODES),
        help="傳給 run_all.py 的 --search-mode 參數",
    )
    parser.add_argument(
        "--history-csv",  # 累計筆數 CSV 紀錄檔路徑；建議維持預設 old data/auto_run_history.csv
        type=Path,
        default=DEFAULT_HISTORY_CSV,
        help=f"統計紀錄輸出 CSV 路徑（預設 {DEFAULT_HISTORY_CSV}）",
    )
    parser.add_argument(
        "--stop-on-error",  # 是否在 run_all.py 任一金屬失敗時立即中止；建議預設關閉（False），讓其餘金屬與 Step 6 合併繼續完成
        action="store_true",
        help="若 run_all.py 回傳非零結束碼則立即中止",
    )
    parser.add_argument(
        "--dry-run",  # 演練預覽模式（不備份、不執行 run_all.py、不寫入 CSV）；建議正式執行前先加一次確認設定
        action="store_true",
        help="僅顯示預計執行的備份、指令與報告解析結果，不實際複製檔案、執行管線或寫入 CSV",
    )
    return parser


def main() -> int:
    parser = build_cli_parser()
    args = parser.parse_args()
    try:
        return run_workflow(
            rounds=args.rounds,
            threshold=args.threshold,
            metals=args.metals,
            search_modes=args.search_mode,
            history_csv=args.history_csv,
            stop_on_error=args.stop_on_error,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        print("\n🛑 收到中斷訊號，已安全結束 auto_run.py。", flush=True)
        return 130


if __name__ == "__main__":
    sys.exit(main())
