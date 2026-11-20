"""
一鍵批次執行完整 pipeline（支援多金屬自動切換、斷點續跑、容錯隔離與 Step 6 合併）

常用指令：
  .venv/bin/python run_all.py                  # 依 config.METALS_TO_RUN 跑全部金屬 + Step 6 合併
  .venv/bin/python run_all.py Cu Co W          # 臨時只跑指定金屬 + Step 6 合併
  .venv/bin/python run_all.py --from-step 5    # 只對所有金屬重跑 Step 5 + Step 6（不連網）
  .venv/bin/python run_all.py Cu --force       # 強制重跑 Cu（忽略 Step 1/2 略過設定）
  .venv/bin/python run_all.py --list           # 查看可用金屬清單與目前 METALS_TO_RUN
"""

import argparse
import os
import subprocess
import sys
import time


def run_single_metal(from_step: int = 1, force: bool = False) -> None:
    """在獨立子進程中執行單一金屬的 Step from_step ~ Step 5。"""
    import step1_search_openalex
    import step2_find_pdf_urls
    import step3_download_pdfs
    import step4_extract_text
    import step5_extract_data
    from config import (
        METAL_SYMBOL,
        SKIP_EXISTING_STEP1,
        SKIP_EXISTING_STEP2,
        STEP1_OUTPUT,
        STEP2_OUTPUT,
        STEP5_ECHEM_OUTPUT,
        TEXT_DIR,
    )

    if from_step <= 1:
        print("=" * 60)
        print(f"[{METAL_SYMBOL}] Step 1: Search OpenAlex")
        print("=" * 60)
        if not force and SKIP_EXISTING_STEP1 and os.path.exists(STEP1_OUTPUT):
            print(f"⏭️  已存在 Step 1 輸出，略過：{STEP1_OUTPUT}")
        else:
            step1_search_openalex.run_search()

    if from_step <= 2:
        print("\n" + "=" * 60)
        print(f"[{METAL_SYMBOL}] Step 2: Find PDF URLs via Unpaywall")
        print("=" * 60)
        if not force and SKIP_EXISTING_STEP2 and os.path.exists(STEP2_OUTPUT):
            print(f"⏭️  已存在 Step 2 輸出，略過：{STEP2_OUTPUT}")
        else:
            step2_find_pdf_urls.run_find_pdfs()

    if from_step <= 3:
        print("\n" + "=" * 60)
        print(f"[{METAL_SYMBOL}] Step 3: Download OA PDFs")
        print("=" * 60)
        step3_download_pdfs.run_download()

    if from_step <= 4:
        print("\n" + "=" * 60)
        print(f"[{METAL_SYMBOL}] Step 4: Extract text & tables from PDFs")
        print("=" * 60)
        step4_extract_text.run_extract()

    if from_step <= 5:
        print("\n" + "=" * 60)
        print(f"[{METAL_SYMBOL}] Step 5: Extract electrochemical data")
        print("=" * 60)
        txt_files = [f for f in os.listdir(TEXT_DIR) if f.endswith(".txt")] if os.path.isdir(TEXT_DIR) else []
        if not txt_files:
            print(f"⚠️  [{METAL_SYMBOL}] {TEXT_DIR} 中沒有任何文字檔，略過 Step 5。")
            return
        result = step5_extract_data.run_extract()
        mat_df, sol_df, echem_df = result if result else ([], [], [])
        print(
            f"✅ [{METAL_SYMBOL}] Step 5 完成 — "
            f"Materials: {len(mat_df)} | Solutions: {len(sol_df)} | Echem: {len(echem_df)}"
        )
        print(f"   Output: {STEP5_ECHEM_OUTPUT}")


def run_batch(
    metals: list[str],
    from_step: int = 1,
    force: bool = False,
    no_merge: bool = False,
) -> int:
    """依序以獨立子進程執行多個金屬，單一金屬失敗不中斷其餘金屬，最後執行 Step 6 合併。"""
    script_path = os.path.abspath(__file__)
    step6_path = os.path.join(os.path.dirname(script_path), "step6_merge.py")

    print("=" * 65)
    print(f"  TAFEL MINING BATCH PIPELINE")
    print(f"  Metals ({len(metals)}): {', '.join(metals)}")
    print(f"  From step : {from_step}  |  Force rerun: {force}  |  Merge: {not no_merge}")
    print("=" * 65)

    summary: list[tuple[str, str, float]] = []

    for idx, metal in enumerate(metals, start=1):
        print("\n" + "#" * 65)
        print(f"  [{idx}/{len(metals)}] Starting metal: {metal}")
        print("#" * 65)
        t0 = time.time()

        env = os.environ.copy()
        env["TAFEL_METAL"] = metal
        env.pop("TAFEL_QUIET_CONFIG", None)

        cmd = [
            sys.executable,
            script_path,
            "--single-metal",
            metal,
            "--from-step",
            str(from_step),
        ]
        if force:
            cmd.append("--force")

        proc = subprocess.run(cmd, env=env)
        elapsed = time.time() - t0

        if proc.returncode == 0:
            summary.append((metal, "✅ OK", elapsed))
        else:
            print(f"\n❌ [{metal}] 執行失敗 (exit code {proc.returncode})，記錄錯誤並繼續下一個金屬...")
            summary.append((metal, f"❌ FAILED (code {proc.returncode})", elapsed))

    # Step 6：以獨立子進程執行，確保重新掃描所有剛完成的金屬資料夾
    merge_ok = True
    if not no_merge:
        print("\n" + "=" * 65)
        print("  Step 6: Merge all completed metals")
        print("=" * 65)
        env = os.environ.copy()
        env["TAFEL_QUIET_CONFIG"] = "1"
        merge_proc = subprocess.run([sys.executable, step6_path], env=env)
        merge_ok = (merge_proc.returncode == 0)

    # 印出總結報表
    print("\n" + "=" * 65)
    print("  BATCH SUMMARY")
    print("=" * 65)
    ok_count = sum(1 for _, st, _ in summary if st.startswith("✅"))
    for metal, status, elapsed in summary:
        print(f"  {metal:<6s}  {status:<24s}  ({elapsed:.1f}s)")
    print("-" * 65)
    print(f"  Succeeded: {ok_count} / {len(summary)}")
    if not no_merge:
        print(f"  Step 6 Merge: {'✅ OK' if merge_ok else '❌ FAILED'}")
    print("=" * 65)

    return 0 if (ok_count == len(summary) and merge_ok) else 1


def main():
    parser = argparse.ArgumentParser(
        description="Tafel Mining 多金屬批次執行器（Step 1~5 + Step 6 合併）"
    )
    parser.add_argument(
        "metals",
        nargs="*",
        help="要執行的金屬符號（省略則使用 config.METALS_TO_RUN），例如：Cu Co W",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="從第幾個步驟開始執行（預設 1；例如 --from-step 5 只重跑 Step 5 + Step 6）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="強制重跑 Step 1 與 Step 2（忽略 config.SKIP_EXISTING_STEP1/2）",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="跑完後不執行 Step 6 合併",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出 config.py 中所有可用金屬預設與目前 METALS_TO_RUN",
    )
    parser.add_argument(
        "--single-metal",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    # 內部子進程模式：直接執行單一金屬
    if args.single_metal:
        os.environ["TAFEL_METAL"] = args.single_metal
        run_single_metal(from_step=args.from_step, force=args.force)
        return

    # 靜默讀取 config（避免協調器印出重複的預設金屬 banner 或建多餘資料夾）
    os.environ["TAFEL_QUIET_CONFIG"] = "1"
    import config

    if args.list:
        print("Available METAL_PRESETS:")
        for sym, info in config.METAL_PRESETS.items():
            mark = "✓" if sym in config.METALS_TO_RUN else " "
            print(f"  [{mark}] {sym:<4s}  {info['target_metal']:<18s}  ({len(info['compounds'])} compounds)")
        print(f"\nCurrent METALS_TO_RUN: {config.METALS_TO_RUN}")
        return

    raw_metals = args.metals if args.metals else config.METALS_TO_RUN
    if not raw_metals:
        print("❌ 沒有指定任何金屬，請在 config.py 設定 METALS_TO_RUN 或於命令列傳入金屬符號。")
        sys.exit(1)

    try:
        resolved_metals = [config.resolve_metal_symbol(m) for m in raw_metals]
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # 去重但保留順序
    metals_unique = list(dict.fromkeys(resolved_metals))
    exit_code = run_batch(
        metals=metals_unique,
        from_step=args.from_step,
        force=args.force,
        no_merge=args.no_merge,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
