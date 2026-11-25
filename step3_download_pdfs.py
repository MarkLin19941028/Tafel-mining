"""
Step 3: 下載 OA PDF 與 Europe PMC 開放全文 XML
- 統一讀取 Step 2 清單（整合 OpenAlex / Semantic Scholar / Europe PMC）
- 持久化下載狀態表（<metal>_download_state.json），支援零網路離線初始化與斷點續跑
- 進入下載前即過濾已完成與先前失敗的論文，進度條只處理真正待下載的新論文
- 按網域（domain）獨立節流 + 低併發執行緒池 + 單網域連續 403 熔斷器
- PDF 串流即時驗證（Content-Type、首塊 %PDF、大檔上限），避免浪費頻寬下載無效網頁
- 支援 --retry-failed 重試先前失敗的論文
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
import threading
import time
from urllib.parse import urlparse

import pandas as pd
import requests
from tqdm import tqdm

from config import (
    DOWNLOAD_DELAY,
    DOWNLOAD_STATE_PATH,
    ENABLE_EUROPEPMC_XML,
    EUROPEPMC_DELAY,
    EUROPEPMC_MAX_PAPERS,
    METAL_SYMBOL,
    PDF_DIR,
    REQUEST_TIMEOUT,
    REQUIRE_TARGET_IN_TITLE_ABSTRACT,
    STEP2_OUTPUT,
    STEP3_DOMAIN_403_THRESHOLD,
    STEP3_MAX_PDF_MB,
    STEP3_MAX_WORKERS,
    TEXT_DIR,
    build_target_regex,
)
from record_keys import (
    clean_str,
    make_record_key,
    normalize_doi_key,
    sanitize_base_name,
)

# 模擬主流桌面瀏覽器標頭，避免直接被 CDN / WAF 當作爬蟲阻擋
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,zh-TW;q=0.8,zh;q=0.7",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "cross-site",
}

PDF_URL_COLUMNS = (
    "final_pdf_url",
    "unpaywall_pdf_url",
    "s2_pdf_url",
    "best_pdf_url",
    "primary_pdf_url",
)

STATUS_DONE_PDF = "done_pdf"
STATUS_DONE_EPMC = "done_epmc"
STATUS_DONE_TXT = "done_txt"
STATUS_FAILED = "failed"
STATUS_FAILED_LEGACY = "failed_legacy"

DONE_STATUSES = frozenset({STATUS_DONE_PDF, STATUS_DONE_EPMC, STATUS_DONE_TXT})
FAILED_STATUSES = frozenset({STATUS_FAILED, STATUS_FAILED_LEGACY})

_THREAD_LOCAL = threading.local()


def _get_thread_session() -> requests.Session:
    sess = getattr(_THREAD_LOCAL, "session", None)
    if sess is None:
        sess = requests.Session()
        _THREAD_LOCAL.session = sess
    return sess


def _extract_host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


class DomainRateLimiter:
    """執行緒安全的按網域速率節流器：同網域嚴格間隔 delay 秒，不同網域互不阻塞。"""

    def __init__(self, delay: float):
        self.delay = max(0.0, float(delay))
        self._lock = threading.Lock()
        self._next_allowed: dict[str, float] = {}

    def wait(self, domain: str) -> None:
        if self.delay <= 0.0:
            return
        key = domain or "_default"
        with self._lock:
            now = time.monotonic()
            allowed_at = self._next_allowed.get(key, now)
            sleep_for = max(0.0, allowed_at - now)
            self._next_allowed[key] = max(now, allowed_at) + self.delay
        if sleep_for > 0.0:
            time.sleep(sleep_for)


class DomainCircuitBreaker:
    """執行緒安全的單網域連續 403 熔斷器：連續 403 達門檻後，本輪跳過該網域 PDF 請求。"""

    def __init__(self, threshold: int):
        self.threshold = max(0, int(threshold))
        self._lock = threading.Lock()
        self._consec_403: dict[str, int] = {}
        self._tripped: set[str] = set()

    def is_open(self, domain: str) -> bool:
        if self.threshold <= 0 or not domain:
            return False
        with self._lock:
            return domain in self._tripped

    def record_403(self, domain: str) -> bool:
        """記錄一次 403；若剛觸發熔斷則回傳 True。"""
        if self.threshold <= 0 or not domain:
            return False
        with self._lock:
            cnt = self._consec_403.get(domain, 0) + 1
            self._consec_403[domain] = cnt
            if cnt >= self.threshold and domain not in self._tripped:
                self._tripped.add(domain)
                return True
            return False

    def record_success(self, domain: str) -> None:
        if not domain:
            return
        with self._lock:
            self._consec_403[domain] = 0

    def tripped_domains(self) -> list[str]:
        with self._lock:
            return sorted(self._tripped)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _make_state_entry(
    status: str,
    reason: str,
    urls: list[str],
    pmcid: str | None = None,
    updated_at: str | None = None,
) -> dict:
    return {
        "status": status,
        "reason": reason,
        "urls": urls,
        "pmcid": pmcid or None,
        "updated_at": updated_at or _now_iso(),
    }


def _write_epmc_marker(marker_path: str, value: str) -> None:
    try:
        with open(marker_path, "w", encoding="utf-8") as f:
            f.write(value)
    except OSError:
        pass


def sanitize_filename(doi: str) -> str:
    return sanitize_base_name(doi) + ".pdf"


def extract_candidate_urls(row) -> list[str]:
    urls: list[str] = []
    for col in PDF_URL_COLUMNS:
        val = row.get(col) if hasattr(row, "get") else None
        s = clean_str(val)
        if s.startswith("http") and s not in urls:
            urls.append(s)
    return urls


def save_download_state(state_path: str, state: dict[str, dict]) -> None:
    """原子寫入下載狀態檔，避免中途 Ctrl+C 造成 JSON 損毀。"""
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    tmp_path = state_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, state_path)


def load_or_bootstrap_download_state(
    state_path: str,
    df_candidates: pd.DataFrame,
    pdf_dir: str,
    text_dir: str,
) -> dict[str, dict]:
    """
    載入既有下載狀態表；若檔案尚未建立，執行純本機離線初始化（零網路請求）：
    - 本地已有 .pdf / .txt / .epmc_done 者標記為 done_*
    - 若該金屬資料夾先前已執行過 Step 3（本地已有至少 1 份下載檔案），
      既有清單中仍無本地檔且有 PDF URL 的舊論文標記為 failed_legacy，避免每次重複重試
    - 全新尚未下載過的金屬不會標記 failed_legacy，所有論文維持待下載狀態
    """
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                return loaded
        except Exception as e:
            print(f"  ⚠️ 無法讀取下載狀態表 ({e})，將重新以本地檔案初始化。")

    existing_pdfs = (
        {f for f in os.listdir(pdf_dir) if f.endswith(".pdf")}
        if os.path.isdir(pdf_dir)
        else set()
    )
    existing_txts = (
        {f for f in os.listdir(text_dir) if f.endswith(".txt")}
        if os.path.isdir(text_dir)
        else set()
    )
    existing_epmc = (
        {f for f in os.listdir(text_dir) if f.endswith(".epmc_done")}
        if os.path.isdir(text_dir)
        else set()
    )
    has_existing_downloads = bool(existing_pdfs or existing_txts)

    state: dict[str, dict] = {}
    n_done = 0
    n_legacy_failed = 0
    ts = _now_iso()

    for _, row in df_candidates.iterrows():
        doi = clean_str(row.get("doi"))
        pmcid = clean_str(row.get("pmcid"))
        key = make_record_key(doi, pmcid)
        if not key or key in state:
            continue

        base_name = sanitize_base_name(doi or pmcid)
        pdf_file = base_name + ".pdf"
        txt_file = base_name + ".txt"
        epmc_file = base_name + ".epmc_done"
        urls = extract_candidate_urls(row)

        if epmc_file in existing_epmc and txt_file in existing_txts:
            state[key] = _make_state_entry(STATUS_DONE_EPMC, "已存在 Europe PMC XML 全文", urls, pmcid, ts)
            n_done += 1
        elif pdf_file in existing_pdfs:
            state[key] = _make_state_entry(STATUS_DONE_PDF, "已存在本地 PDF", urls, pmcid, ts)
            n_done += 1
        elif txt_file in existing_txts:
            state[key] = _make_state_entry(STATUS_DONE_TXT, "已存在本地文字檔", urls, pmcid, ts)
            n_done += 1
        elif has_existing_downloads and urls and not pmcid:
            state[key] = _make_state_entry(
                STATUS_FAILED_LEGACY, "先前批次未取得 PDF（離線初始化標記）", urls, None, ts
            )
            n_legacy_failed += 1

    save_download_state(state_path, state)
    print(
        f"[Step 3 State] 離線初始化完成：已完成 {n_done} 篇，"
        f"舊批次未取得標記 {n_legacy_failed} 篇（零網路請求）"
    )
    return state


def select_pending_rows(
    df_candidates: pd.DataFrame,
    state: dict[str, dict],
    pdf_dir: str,
    text_dir: str,
    retry_failed: bool = False,
) -> tuple[list[dict], dict[str, int]]:
    """
    在進入下載迴圈前，依本地檔案與狀態表過濾出真正需要下載的論文清單。
    """
    pending: list[dict] = []
    stats = {
        "total": len(df_candidates),
        "skipped_done": 0,
        "skipped_failed": 0,
        "retry_new_source": 0,
        "retry_forced": 0,
    }
    seen_keys: set[str] = set()
    ts = _now_iso()

    for _, row in df_candidates.iterrows():
        doi = clean_str(row.get("doi"))
        pmcid = clean_str(row.get("pmcid"))
        key = make_record_key(doi, pmcid)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)

        base_name = sanitize_base_name(doi or pmcid)
        save_path = os.path.join(pdf_dir, base_name + ".pdf")
        txt_path = os.path.join(text_dir, base_name + ".txt")
        epmc_marker = os.path.join(text_dir, base_name + ".epmc_done")
        candidate_urls = extract_candidate_urls(row)

        # 1. 本地已有 PDF 或純文字檔 -> 同步狀態並略過
        if os.path.exists(save_path) or os.path.exists(txt_path):
            if key not in state or state[key].get("status") not in DONE_STATUSES:
                done_status = (
                    STATUS_DONE_EPMC
                    if os.path.exists(epmc_marker)
                    else (STATUS_DONE_PDF if os.path.exists(save_path) else STATUS_DONE_TXT)
                )
                state[key] = _make_state_entry(done_status, "本地檔案已存在", candidate_urls, pmcid, ts)
            stats["skipped_done"] += 1
            continue

        # 2. 檢查是否為先前下載失敗的論文
        entry = state.get(key)
        if entry and entry.get("status") in FAILED_STATUSES:
            tried_urls = set(entry.get("urls") or [])
            tried_pmcid = clean_str(entry.get("pmcid"))
            has_new_url = any(u not in tried_urls for u in candidate_urls)
            has_new_pmcid = bool(pmcid) and pmcid != tried_pmcid

            if retry_failed:
                stats["retry_forced"] += 1
            elif has_new_url or has_new_pmcid:
                stats["retry_new_source"] += 1
            else:
                stats["skipped_failed"] += 1
                continue

        pending.append({
            "key": key,
            "doi": doi,
            "pmcid": pmcid,
            "base_name": base_name,
            "save_path": save_path,
            "txt_path": txt_path,
            "epmc_marker": epmc_marker,
            "candidate_urls": candidate_urls,
        })

    return pending, stats


def download_pdf(
    url: str,
    save_path: str,
    *,
    max_pdf_mb: int | None = None,
    session: requests.Session | None = None,
) -> tuple[bool, str]:
    """
    下載 PDF 檔案並於串流階段即時驗證：
    - 拒絕明確為 text/html 的回應
    - 讀取首塊即檢查 %PDF 檔頭
    - 限制單檔上限（STEP3_MAX_PDF_MB）與最小有效大小（10 KB）
    回傳 (成功與否, 狀態/失敗原因訊息)
    """
    if not url or not str(url).startswith("http"):
        return False, "無效的 URL"

    size_limit_mb = STEP3_MAX_PDF_MB if max_pdf_mb is None else max_pdf_mb
    max_bytes = max(1, int(size_limit_mb)) * 1024 * 1024
    client = session if session is not None else _get_thread_session()

    try:
        with client.get(
            url,
            headers=DEFAULT_HEADERS,
            timeout=REQUEST_TIMEOUT * 2,
            stream=True,
            allow_redirects=True,
        ) as resp:
            if resp.status_code == 403:
                return False, "403 Forbidden（出版社設有防爬蟲/訂閱限制）"
            if resp.status_code == 404:
                return False, "404 Not Found（PDF 連結已失效）"
            if resp.status_code != 200:
                return False, f"HTTP {resp.status_code}"

            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "text/html" in ctype and "pdf" not in ctype and "octet-stream" not in ctype:
                return False, "檔案內容為 HTML 網頁而非 PDF（出版社要求登入或跳轉）"

            content_len = resp.headers.get("Content-Length")
            if content_len and content_len.isdigit() and int(content_len) > max_bytes:
                return False, f"檔案超過上限 ({size_limit_mb} MB)"

            written = 0
            checked_header = False
            abort_reason: str | None = None
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=16384):
                    if not chunk:
                        continue
                    if not checked_header:
                        checked_header = True
                        if not chunk.startswith(b"%PDF"):
                            abort_reason = "檔案內容為 HTML 網頁而非 PDF（出版社要求登入或跳轉）"
                            break
                    written += len(chunk)
                    if written > max_bytes:
                        abort_reason = f"檔案超過上限 ({size_limit_mb} MB)"
                        break
                    f.write(chunk)

        if abort_reason is not None:
            if os.path.exists(save_path):
                os.remove(save_path)
            return False, abort_reason

        if written < 10_000:
            if os.path.exists(save_path):
                os.remove(save_path)
            return False, f"檔案過小 ({written} bytes)，非有效 PDF"

        return True, "成功"

    except requests.exceptions.Timeout:
        if os.path.exists(save_path):
            os.remove(save_path)
        return False, "連線超時 (Timeout)"
    except requests.exceptions.RequestException as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        return False, f"網路錯誤 ({type(e).__name__})"
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        return False, f"其他錯誤 ({e})"


def run_download(retry_failed: bool | None = None):
    if retry_failed is None:
        retry_failed = os.environ.get("TAFEL_RETRY_FAILED") == "1"

    if ENABLE_EUROPEPMC_XML:
        from europepmc_fetcher import (
            fetch_fulltext_xml,
            run_europepmc_search,
            save_xml_to_texts,
            try_fallback_from_europepmc,
        )

    if not os.path.exists(STEP2_OUTPUT):
        print(f"⚠️  找不到 Step 2 的輸出檔案：{STEP2_OUTPUT}")
        if ENABLE_EUROPEPMC_XML and EUROPEPMC_MAX_PAPERS > 0:
            run_europepmc_search()
        return

    df = pd.read_csv(STEP2_OUTPUT)
    print(f"Loaded {len(df)} papers from {STEP2_OUTPUT}")

    # 篩出「有 PDF 連結」或「有 Europe PMC PMCID」的候選論文
    has_pdf_mask = df.get("final_pdf_url", pd.Series(dtype=object)).notna()
    if "pmcid" in df.columns:
        has_pmcid_mask = df["pmcid"].apply(lambda v: bool(clean_str(v)))
    else:
        has_pmcid_mask = pd.Series(False, index=df.index)

    df_candidates = df[has_pdf_mask | has_pmcid_mask].copy()
    print(
        f"Downloadable candidates: {len(df_candidates)} "
        f"(PDF URL: {int(has_pdf_mask.sum())}, Europe PMC PMCID: {int(has_pmcid_mask.sum())})"
    )

    # 相關性篩選：標題或摘要必須提到目標金屬（或其化合物）
    if REQUIRE_TARGET_IN_TITLE_ABSTRACT and len(df_candidates):
        pat = build_target_regex()
        ta = (
            df_candidates.get("title", pd.Series("", index=df_candidates.index)).fillna("").astype(str)
            + " "
            + df_candidates.get("abstract", pd.Series("", index=df_candidates.index)).fillna("").astype(str)
        )
        mask = ta.apply(lambda s: bool(pat.search(s)))
        print(
            f"Relevance filter ({METAL_SYMBOL} in title/abstract): "
            f"kept {int(mask.sum())} / {len(df_candidates)}"
        )
        df_candidates = df_candidates[mask]

    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(TEXT_DIR, exist_ok=True)

    state = load_or_bootstrap_download_state(
        DOWNLOAD_STATE_PATH, df_candidates, PDF_DIR, TEXT_DIR
    )
    pending_rows, qstats = select_pending_rows(
        df_candidates, state, PDF_DIR, TEXT_DIR, retry_failed=retry_failed
    )

    extra_note = ""
    if qstats["retry_forced"]:
        extra_note += f" | --retry-failed 重試: {qstats['retry_forced']}"
    if qstats["retry_new_source"]:
        extra_note += f" | 補到新來源重試: {qstats['retry_new_source']}"
    workers = max(1, int(STEP3_MAX_WORKERS))
    print(
        f"[Step 3 Queue] 總候選: {qstats['total']} | 已下載略過: {qstats['skipped_done']} "
        f"| 先前失敗略過: {qstats['skipped_failed']} | 本次待下載: {len(pending_rows)}"
        f" (workers={workers}, domain_delay={DOWNLOAD_DELAY}s, 403_breaker={STEP3_DOMAIN_403_THRESHOLD}){extra_note}"
    )

    if not pending_rows:
        save_download_state(DOWNLOAD_STATE_PATH, state)
        print("✨ 沒有新的待下載論文（若需重試先前失敗的論文，請加上 --retry-failed）。")
        return

    pdf_limiter = DomainRateLimiter(DOWNLOAD_DELAY)
    epmc_limiter = DomainRateLimiter(EUROPEPMC_DELAY)
    breaker = DomainCircuitBreaker(STEP3_DOMAIN_403_THRESHOLD)
    state_lock = threading.Lock()

    success = 0
    epmc_direct = 0
    epmc_rescued = 0
    fail = 0
    fail_reasons: dict[str, int] = {}

    def _process_one(item: dict) -> tuple[str, dict, str, str | None]:
        key = item["key"]
        doi = item["doi"]
        pmcid = item["pmcid"]
        base_name = item["base_name"]
        save_path = item["save_path"]
        epmc_marker = item["epmc_marker"]
        candidate_urls = item["candidate_urls"]

        with state_lock:
            prev_entry = state.get(key) or {}
            prev_urls = list(prev_entry.get("urls") or [])
        merged_urls = list(dict.fromkeys([*prev_urls, *candidate_urls]))

        # 1. 若已有 Europe PMC PMCID，優先直接抓取開放全文 JATS XML（零 403、含結構化表格）
        if pmcid and ENABLE_EUROPEPMC_XML:
            epmc_limiter.wait("europepmc.org")
            xml_text = fetch_fulltext_xml(pmcid)
            if xml_text:
                w_txt, u_tbl, _ = save_xml_to_texts(xml_text, base_name, TEXT_DIR)
                if w_txt or u_tbl:
                    _write_epmc_marker(epmc_marker, pmcid)
                    return (
                        key,
                        _make_state_entry(STATUS_DONE_EPMC, "Europe PMC XML 成功", merged_urls, pmcid),
                        "epmc_direct",
                        None,
                    )

        # 2. 依序嘗試候選 PDF URL（按網域節流 + 連續 403 熔斷）
        downloaded = False
        last_reason = "無有效 PDF 連結" if not candidate_urls else "下載失敗"
        for url in candidate_urls:
            host = _extract_host(url)
            if breaker.is_open(host):
                last_reason = f"網域已觸發 403 熔斷略過 ({host})"
                continue

            pdf_limiter.wait(host)
            ok, reason = download_pdf(url, save_path)
            if ok:
                breaker.record_success(host)
                downloaded = True
                break
            if reason.startswith("403"):
                if breaker.record_403(host):
                    print(
                        f"\n  ⚡ [Step 3 Breaker] 網域 {host} 連續 {STEP3_DOMAIN_403_THRESHOLD} 次 403，"
                        "本輪自動跳過該網域後續 PDF 連結。"
                    )
            last_reason = reason

        if downloaded:
            return (
                key,
                _make_state_entry(STATUS_DONE_PDF, "PDF 下載成功", merged_urls, pmcid),
                "pdf",
                None,
            )

        # 3. Europe PMC DOI fallback 補救
        if ENABLE_EUROPEPMC_XML and doi and not pmcid:
            epmc_limiter.wait("europepmc.org")
            rescued, _ = try_fallback_from_europepmc(doi, base_name, TEXT_DIR)
            if rescued:
                _write_epmc_marker(epmc_marker, "fallback")
                return (
                    key,
                    _make_state_entry(STATUS_DONE_EPMC, "Europe PMC XML 補救成功", merged_urls, None),
                    "epmc_rescued",
                    None,
                )
            return (
                key,
                _make_state_entry(STATUS_FAILED, last_reason, merged_urls, None),
                "fail",
                last_reason,
            )

        reason_final = (
            "Europe PMC XML 下載失敗且無有效 PDF 連結"
            if (pmcid and not candidate_urls)
            else last_reason
        )
        return (
            key,
            _make_state_entry(STATUS_FAILED, reason_final, merged_urls, pmcid),
            "fail",
            reason_final,
        )

    def _record_result(res: tuple[str, dict, str, str | None], completed_idx: int) -> None:
        nonlocal success, epmc_direct, epmc_rescued, fail
        key, entry, outcome, reason = res
        with state_lock:
            state[key] = entry
            if outcome == "pdf":
                success += 1
            elif outcome == "epmc_direct":
                epmc_direct += 1
            elif outcome == "epmc_rescued":
                epmc_rescued += 1
            else:
                fail += 1
                if reason:
                    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            if completed_idx % 20 == 0:
                save_download_state(DOWNLOAD_STATE_PATH, state)

    try:
        if workers == 1:
            for idx, item in enumerate(
                tqdm(pending_rows, total=len(pending_rows), desc="Downloading papers"),
                start=1,
            ):
                _record_result(_process_one(item), idx)
        else:
            pool = ThreadPoolExecutor(max_workers=workers)
            interrupted = False
            try:
                futures = [pool.submit(_process_one, item) for item in pending_rows]
                for idx, fut in enumerate(
                    tqdm(as_completed(futures), total=len(futures), desc="Downloading papers"),
                    start=1,
                ):
                    _record_result(fut.result(), idx)
            except KeyboardInterrupt:
                interrupted = True
                print("\n⚠️  [Step 3] 收到中斷訊號，取消尚未執行的下載任務並立即保存狀態...")
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            finally:
                if not interrupted:
                    pool.shutdown(wait=True)
    finally:
        with state_lock:
            save_download_state(DOWNLOAD_STATE_PATH, state)

    print(
        f"\n下載統計：✅ PDF 成功: {success}, 🧬 Europe PMC XML 直接: {epmc_direct}, "
        f"🧬 Europe PMC 補救: {epmc_rescued}, ⏩ 已下載略過: {qstats['skipped_done']}, "
        f"⏭️ 先前失敗略過: {qstats['skipped_failed']}, ❌ 本次失敗: {fail}"
    )
    tripped = breaker.tripped_domains()
    if tripped:
        print(f"⚡ 本輪觸發 403 熔斷網域（{len(tripped)} 個）：{', '.join(tripped)}")
    if fail_reasons:
        print("\n失敗原因分佈：")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            print(f"  - {reason}: {count} 篇")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: 下載 OA PDF 與 Europe PMC 開放全文 XML")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="重試先前下載失敗（failed / failed_legacy）的論文",
    )
    cli_args = parser.parse_args()
    run_download(retry_failed=cli_args.retry_failed)
