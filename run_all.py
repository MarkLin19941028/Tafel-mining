"""
一鍵批次執行完整 pipeline（支援多金屬自動切換、斷點續跑、容錯隔離、Step 6 合併與 Step 7 物理特徵強化）

常用指令：
  .venv/bin/python run_all.py                                             # 依 user_settings.METALS_TO_RUN 跑全部金屬 + Step 6 + Step 7
  .venv/bin/python run_all.py Cu Co W                                     # 臨時只跑指定金屬 + Step 6 + Step 7
  .venv/bin/python run_all.py Cu Co --search-mode cmp_etch inhibitor --append  # 多選搜尋模式並增量合併
  .venv/bin/python run_all.py Cu --from-step 2 --to-step 3                # 只跑指定區間步驟（Step 2 ~ Step 3）
  .venv/bin/python run_all.py Cu --from-step 3 --to-step 3 --retry-failed # 重試先前下載失敗的論文
  .venv/bin/python run_all.py --from-step 5                               # 只對所有金屬重跑 Step 5 + Step 6 + Step 7
  .venv/bin/python run_all.py --from-step 7                               # 只重跑 Step 7（Materials Project + Pourbaix 特徵強化）
  .venv/bin/python run_all.py Cu --force                                  # 強制重跑 Cu（忽略 Step 1/2 略過設定）
  .venv/bin/python run_all.py --list                                      # 查看可用金屬清單與目前 METALS_TO_RUN
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time


def run_single_metal(
    from_step: int = 1,
    to_step: int = 7,
    force: bool = False,
    append: bool = False,
    retry_failed: bool = False,
    skip_openalex: bool = False,
) -> None:
    """在獨立子進程中執行單一金屬的 Step from_step ~ min(to_step, 5)。"""
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

    if from_step <= 1 <= to_step:
        print("=" * 60)
        print(f"[{METAL_SYMBOL}] Step 1: Search OpenAlex / Semantic Scholar / Europe PMC")
        print("=" * 60)
        if not force and not append and SKIP_EXISTING_STEP1 and os.path.exists(STEP1_OUTPUT):
            print(f"⏭️  已存在 Step 1 輸出，略過：{STEP1_OUTPUT}")
        else:
            step1_search_openalex.run_search(append=append, skip_openalex=skip_openalex)

    if from_step <= 2 <= to_step:
        print("\n" + "=" * 60)
        print(f"[{METAL_SYMBOL}] Step 2: Find PDF URLs via Unpaywall")
        print("=" * 60)
        if not force and not append and SKIP_EXISTING_STEP2 and os.path.exists(STEP2_OUTPUT):
            print(f"⏭️  已存在 Step 2 輸出，略過：{STEP2_OUTPUT}")
        else:
            step2_find_pdf_urls.run_find_pdfs(append=append)

    if from_step <= 3 <= to_step:
        print("\n" + "=" * 60)
        print(f"[{METAL_SYMBOL}] Step 3: Download OA PDFs & Europe PMC XML")
        print("=" * 60)
        step3_download_pdfs.run_download(retry_failed=retry_failed)

    if from_step <= 4 <= to_step:
        print("\n" + "=" * 60)
        print(f"[{METAL_SYMBOL}] Step 4: Extract text & tables from PDFs")
        print("=" * 60)
        step4_extract_text.run_extract()

    if from_step <= 5 <= to_step:
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
    to_step: int = 7,
    force: bool = False,
    append: bool = False,
    retry_failed: bool = False,
    search_modes: list[str] | None = None,
    no_merge: bool = False,
    no_enrich: bool = False,
    skip_openalex: bool = False,
) -> int:
    """依序以獨立子進程執行多個金屬，單一金屬失敗不中斷其餘金屬，最後依區間執行 Step 6 與 Step 7。"""
    script_path = os.path.abspath(__file__)
    step6_path = os.path.join(os.path.dirname(script_path), "step6_merge.py")
    step7_path = os.path.join(os.path.dirname(script_path), "step7_enrich_data.py")

    run_merge = (not no_merge) and (from_step <= 6 <= to_step)
    run_enrich = (not no_enrich) and (from_step <= 7 <= to_step)

    print("=" * 65)
    print(f"  TAFEL MINING BATCH PIPELINE")
    print(f"  Metals ({len(metals)}): {', '.join(metals)}")
    print(
        f"  Steps     : {from_step} -> {to_step}  |  Force: {force}  |  Append: {append}  |  "
        f"RetryFailed: {retry_failed}  |  SkipOA: {skip_openalex}"
    )
    print(f"  Post-steps: Merge: {run_merge}  |  Enrich: {run_enrich}")
    if search_modes:
        print(f"  Search modes override: {', '.join(search_modes)}")
    print("=" * 65)

    summary: list[tuple[str, str, float]] = []

    if from_step <= 5:
        oa_quota_marker = os.path.join(
            tempfile.gettempdir(),
            f"tafel_oa_quota_{os.getpid()}_{int(time.time())}.flag",
        )
        oa_quota_exhausted_in_batch = False
        try:
            for idx, metal in enumerate(metals, start=1):
                print("\n" + "#" * 65)
                print(f"  [{idx}/{len(metals)}] Starting metal: {metal}")
                print("#" * 65)
                t0 = time.time()

                effective_skip_oa = skip_openalex or oa_quota_exhausted_in_batch
                if oa_quota_exhausted_in_batch and not skip_openalex:
                    print("  ⚠️  [Batch] OpenAlex 每日配額於前一個金屬已耗盡，本金屬自動略過 OpenAlex 改跑 S2 + Europe PMC。")

                env = os.environ.copy()
                env["TAFEL_METAL"] = metal
                env["TAFEL_OA_QUOTA_MARKER"] = oa_quota_marker
                if search_modes:
                    env["TAFEL_SEARCH_MODES"] = ",".join(search_modes)
                if append:
                    env["TAFEL_APPEND"] = "1"
                if retry_failed:
                    env["TAFEL_RETRY_FAILED"] = "1"
                if effective_skip_oa:
                    env["TAFEL_SKIP_OPENALEX"] = "1"
                env.pop("TAFEL_QUIET_CONFIG", None)

                cmd = [
                    sys.executable,
                    script_path,
                    "--single-metal",
                    metal,
                    "--from-step",
                    str(from_step),
                    "--to-step",
                    str(to_step),
                ]
                if search_modes:
                    cmd.extend(["--search-mode", *search_modes])
                if force:
                    cmd.append("--force")
                if append:
                    cmd.append("--append")
                if retry_failed:
                    cmd.append("--retry-failed")
                if effective_skip_oa:
                    cmd.append("--skip-openalex")

                proc = subprocess.run(cmd, env=env)
                elapsed = time.time() - t0

                if not oa_quota_exhausted_in_batch and os.path.exists(oa_quota_marker):
                    oa_quota_exhausted_in_batch = True

                if proc.returncode == 0:
                    summary.append((metal, "✅ OK", elapsed))
                else:
                    print(f"\n❌ [{metal}] 執行失敗 (exit code {proc.returncode})，記錄錯誤並繼續下一個金屬...")
                    summary.append((metal, f"❌ FAILED (code {proc.returncode})", elapsed))
        finally:
            if os.path.exists(oa_quota_marker):
                try:
                    os.remove(oa_quota_marker)
                except OSError:
                    pass

    # Step 6：以獨立子進程執行，確保重新掃描所有剛完成的金屬資料夾
    merge_ok = True
    if run_merge:
        print("\n" + "=" * 65)
        print("  Step 6: Merge all completed metals")
        print("=" * 65)
        env = os.environ.copy()
        env["TAFEL_QUIET_CONFIG"] = "1"
        merge_proc = subprocess.run([sys.executable, step6_path], env=env)
        merge_ok = (merge_proc.returncode == 0)

    # Step 7：物理與 Pourbaix 熱力學特徵強化（Materials Project API + 內建熱力學表）
    enrich_ok = True
    if run_enrich and merge_ok:
        print("\n" + "=" * 65)
        print("  Step 7: Enrich physical & Pourbaix thermodynamic features")
        print("=" * 65)
        env = os.environ.copy()
        env["TAFEL_QUIET_CONFIG"] = "1"
        enrich_proc = subprocess.run([sys.executable, step7_path], env=env)
        enrich_ok = (enrich_proc.returncode == 0)

    # 印出總結報表
    print("\n" + "=" * 65)
    print("  BATCH SUMMARY")
    print("=" * 65)
    ok_count = sum(1 for _, st, _ in summary if st.startswith("✅"))
    if summary:
        for metal, status, elapsed in summary:
            print(f"  {metal:<6s}  {status:<24s}  ({elapsed:.1f}s)")
        print("-" * 65)
        print(f"  Succeeded: {ok_count} / {len(summary)}")
    if run_merge:
        print(f"  Step 6 Merge : {'✅ OK' if merge_ok else '❌ FAILED'}")
    if run_enrich:
        print(f"  Step 7 Enrich: {'✅ OK' if enrich_ok else '❌ FAILED'}")
    print("=" * 65)

    return 0 if (ok_count == len(summary) and merge_ok and enrich_ok) else 1


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tafel Mining 多金屬批次執行器（Step 1~5 + Step 6 合併 + Step 7 強化）"
    )
    parser.add_argument(
        "metals",
        nargs="*",
        help="要執行的金屬符號（省略則使用 user_settings.METALS_TO_RUN），例如：Cu Co W",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="從第幾個步驟開始執行（預設 1；例如 --from-step 5 跑 Step 5~7，--from-step 7 只跑物理強化）",
    )
    parser.add_argument(
        "--to-step",
        type=int,
        default=7,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="執行到第幾個步驟停止（預設 7；搭配 --from-step 可只跑局部區間，例如 --from-step 2 --to-step 3）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="強制重跑 Step 1 與 Step 2（忽略 user_settings.SKIP_EXISTING_STEP1/2 並覆寫）",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="增量補抓模式：Step 1 統一搜尋 OpenAlex/S2/Europe PMC 並依 DOI/PMCID 去重合併，Step 2/3 只處理新論文",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Step 3 重試先前下載失敗的論文（包含 failed 與離線初始化標記的 failed_legacy）",
    )
    parser.add_argument(
        "--skip-openalex",
        action="store_true",
        help="Step 1 略過 OpenAlex 檢索，僅執行 Semantic Scholar 與 Europe PMC",
    )
    parser.add_argument(
        "--search-mode",
        nargs="+",
        metavar="MODE",
        default=None,
        help=(
            "臨時指定一或多種搜尋模式（可多選組合：corrosion, cmp_etch, inhibitor, solvent, catalysis, all），"
            "例如：--search-mode cmp_etch inhibitor"
        ),
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="跑完後不執行 Step 6 合併",
    )
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help="跑完後不執行 Step 7 物理與熱力學特徵強化",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出 metals_db.json 中所有可用金屬預設與目前 METALS_TO_RUN",
    )
    parser.add_argument(
        "--check-db",
        action="store_true",
        help="檢查 metals_db.json 與 materials_physics.json 的完整性與一致性",
    )
    parser.add_argument(
        "--single-metal",
        help=argparse.SUPPRESS,
    )
    return parser


def resolve_cli_metals_and_modes(
    raw_metals: list[str] | None,
    raw_modes: list[str] | None,
    config_module,
) -> tuple[list[str], list[str] | None]:
    """
    解析命令列傳入的 metals 與 --search-mode（nargs="+"）：
    支援 `Cu Co --search-mode cmp_etch inhibitor` 與 `--search-mode cmp_etch inhibitor Cu Co` 兩種順序。
    """
    metals_out = list(raw_metals or [])
    if not raw_modes:
        return metals_out, None

    valid_mode_tokens = {*config_module.VALID_SEARCH_MODES, "all", "corrosion_cmp"}
    metal_lookup = {k.lower() for k in config_module.METAL_PRESETS}

    # 展開可能含逗號的 token，但若尾端接著金屬符號（當使用者把 --search-mode 寫在金屬前面時）自動歸還給 metals
    flat_tokens: list[str] = []
    for item in raw_modes:
        flat_tokens.extend(t for t in str(item).strip().split(",") if t)

    split_idx = len(flat_tokens)
    for i, tok in enumerate(flat_tokens):
        if tok.lower() not in valid_mode_tokens:
            split_idx = i
            break

    leading_modes = flat_tokens[:split_idx]
    trailing = flat_tokens[split_idx:]
    if leading_modes and trailing and all(t.lower() in metal_lookup for t in trailing):
        metals_out.extend(trailing)
        resolved_modes = config_module.parse_search_modes(leading_modes)
    else:
        resolved_modes = config_module.parse_search_modes(flat_tokens)

    return metals_out, resolved_modes


def main():
    parser = build_cli_parser()
    args = parser.parse_args()

    if args.from_step > args.to_step:
        print(f"❌ --from-step ({args.from_step}) 不可大於 --to-step ({args.to_step})。")
        sys.exit(1)

    # 內部子進程模式：直接執行單一金屬
    if args.single_metal:
        os.environ["TAFEL_METAL"] = args.single_metal
        if args.search_mode:
            os.environ["TAFEL_SEARCH_MODES"] = ",".join(args.search_mode)
        if args.append:
            os.environ["TAFEL_APPEND"] = "1"
        if args.retry_failed:
            os.environ["TAFEL_RETRY_FAILED"] = "1"
        if args.skip_openalex:
            os.environ["TAFEL_SKIP_OPENALEX"] = "1"
        run_single_metal(
            from_step=args.from_step,
            to_step=args.to_step,
            force=args.force,
            append=args.append,
            retry_failed=args.retry_failed,
            skip_openalex=args.skip_openalex,
        )
        return

    # 靜默讀取 config（避免協調器印出重複的預設金屬 banner 或建多餘資料夾）
    os.environ["TAFEL_QUIET_CONFIG"] = "1"
    import config

    if args.check_db:
        from materials_loader import validate_databases

        errors, warnings = validate_databases(config.METALS_TO_RUN)
        if errors:
            print(f"❌ 發現 {len(errors)} 項錯誤：")
            for err in errors:
                print(f"  [ERROR] {err}")
        if warnings:
            print(f"⚠️  發現 {len(warnings)} 項提醒：")
            for warn in warnings:
                print(f"  [WARN]  {warn}")
        if not errors and not warnings:
            print("✅ metals_db.json 與 materials_physics.json 檢查全數通過！")
        elif not errors:
            print("✅ 無阻擋性錯誤（Step 1~4 可正常執行；上述提醒僅影響 Step 5~7 物性特徵）。")
        sys.exit(1 if errors else 0)

    if args.list:
        print("Available METAL_PRESETS:")
        for sym, info in config.METAL_PRESETS.items():
            mark = "✓" if sym in config.METALS_TO_RUN else " "
            print(f"  [{mark}] {sym:<4s}  {info['target_metal']:<18s}  ({len(info['compounds'])} compounds)")
        print(f"\nCurrent METALS_TO_RUN : {config.METALS_TO_RUN}")
        print(f"Default SEARCH_MODES  : {config.SEARCH_MODES} (available: {list(config.VALID_SEARCH_MODES)})")
        return

    try:
        cli_metals, resolved_modes = resolve_cli_metals_and_modes(args.metals, args.search_mode, config)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    raw_metals = cli_metals if cli_metals else config.METALS_TO_RUN
    if not raw_metals:
        print("❌ 沒有指定任何金屬，請在 user_settings.py 設定 METALS_TO_RUN 或於命令列傳入金屬符號。")
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
        to_step=args.to_step,
        force=args.force,
        append=args.append,
        retry_failed=args.retry_failed,
        search_modes=resolved_modes,
        no_merge=args.no_merge,
        no_enrich=args.no_enrich,
        skip_openalex=args.skip_openalex,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
