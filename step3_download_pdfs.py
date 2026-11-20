"""
Step 3: 下載 OA PDF（強化版：詳細錯誤提示、防爬蟲 Headers、備援 URL 下載）
"""

import os
import re
import time
import requests
import pandas as pd
from tqdm import tqdm

from config import (
    DOWNLOAD_DELAY,
    ENABLE_EUROPEPMC_XML,
    EUROPEPMC_MAX_PAPERS,
    MAILTO,
    METAL_SYMBOL,
    PDF_DIR,
    REQUEST_TIMEOUT,
    REQUIRE_TARGET_IN_TITLE_ABSTRACT,
    STEP2_OUTPUT,
    TEXT_DIR,
    build_target_regex,
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


def sanitize_filename(doi: str) -> str:
    name = str(doi).replace("https://doi.org/", "")
    name = re.sub(r'[^\w\-.]', '_', name)
    return name[:120] + ".pdf"


def download_pdf(url: str, save_path: str) -> tuple[bool, str]:
    """
    下載 PDF 檔案並驗證檔頭是否為有效 PDF。
    回傳 (成功與否, 狀態/失敗原因訊息)
    """
    if not url or not str(url).startswith("http"):
        return False, "無效的 URL"

    try:
        resp = requests.get(
            url,
            headers=DEFAULT_HEADERS,
            timeout=REQUEST_TIMEOUT * 2,
            stream=True,
            allow_redirects=True
        )

        if resp.status_code == 403:
            return False, "403 Forbidden（出版社設有防爬蟲/訂閱限制）"
        elif resp.status_code == 404:
            return False, "404 Not Found（PDF 連結已失效）"
        elif resp.status_code != 200:
            return False, f"HTTP {resp.status_code}"

        # 分塊寫入檔案
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=16384):
                if chunk:
                    f.write(chunk)

        # 檢查檔案大小
        file_size = os.path.getsize(save_path)
        if file_size < 10_000:
            os.remove(save_path)
            return False, f"檔案過小 ({file_size} bytes)，非有效 PDF"

        # 檢查前 5 個 bytes 是否為 %PDF
        with open(save_path, "rb") as f:
            first_bytes = f.read(5)
        if not first_bytes.startswith(b"%PDF"):
            os.remove(save_path)
            return False, "檔案內容為 HTML 網頁而非 PDF（出版社要求登入或跳轉）"

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


def run_download():
    if ENABLE_EUROPEPMC_XML:
        from europepmc_fetcher import run_europepmc_search, try_fallback_from_europepmc

    if not os.path.exists(STEP2_OUTPUT):
        print(f"⚠️  找不到 Step 2 的輸出檔案：{STEP2_OUTPUT}")
        if ENABLE_EUROPEPMC_XML and EUROPEPMC_MAX_PAPERS > 0:
            run_europepmc_search()
        return

    df = pd.read_csv(STEP2_OUTPUT)
    print(f"Loaded {len(df)} papers from {STEP2_OUTPUT}")

    # 找出所有可能有 PDF 連結的候選欄位
    has_pdf_mask = df.get("final_pdf_url", pd.Series(dtype=object)).notna()
    df_with_pdf = df[has_pdf_mask].copy()
    print(f"Papers with PDF URL: {len(df_with_pdf)}")

    # 相關性篩選：標題或摘要必須提到目標金屬（或其化合物）
    if REQUIRE_TARGET_IN_TITLE_ABSTRACT and len(df_with_pdf):
        pat = build_target_regex()
        ta = (df_with_pdf.get("title", pd.Series("", index=df_with_pdf.index)).fillna("").astype(str)
              + " "
              + df_with_pdf.get("abstract", pd.Series("", index=df_with_pdf.index)).fillna("").astype(str))
        mask = ta.apply(lambda s: bool(pat.search(s)))
        print(f"Relevance filter ({METAL_SYMBOL} in title/abstract): "
              f"kept {mask.sum()} / {len(df_with_pdf)}")
        df_with_pdf = df_with_pdf[mask]

    os.makedirs(PDF_DIR, exist_ok=True)

    success = 0
    skip = 0
    fail = 0
    epmc_rescued = 0
    fail_reasons = {}

    for _, row in tqdm(
        df_with_pdf.iterrows(),
        total=len(df_with_pdf),
        desc="Downloading PDFs"
    ):
        doi = row["doi"]
        filename = sanitize_filename(doi)
        base_name = filename[:-4] if filename.endswith(".pdf") else filename
        save_path = os.path.join(PDF_DIR, filename)
        txt_path = os.path.join(TEXT_DIR, base_name + ".txt")

        if os.path.exists(save_path) or os.path.exists(txt_path):
            skip += 1
            continue

        # 優先嘗試 final_pdf_url，若失敗可嘗試備援 best_pdf_url
        candidate_urls = []
        for col in ["final_pdf_url", "best_pdf_url", "unpaywall_pdf_url", "primary_pdf_url"]:
            val = row.get(col)
            if pd.notna(val) and str(val).startswith("http") and val not in candidate_urls:
                candidate_urls.append(val)

        downloaded = False
        last_reason = "無有效連結"

        for url in candidate_urls:
            ok, reason = download_pdf(url, save_path)
            if ok:
                downloaded = True
                break
            else:
                last_reason = reason

        if downloaded:
            success += 1
        elif ENABLE_EUROPEPMC_XML:
            rescued, _ = try_fallback_from_europepmc(doi, base_name, TEXT_DIR)
            if rescued:
                epmc_rescued += 1
            else:
                fail += 1
                fail_reasons[last_reason] = fail_reasons.get(last_reason, 0) + 1
        else:
            fail += 1
            fail_reasons[last_reason] = fail_reasons.get(last_reason, 0) + 1

        time.sleep(DOWNLOAD_DELAY)

    print(
        f"\n下載統計：✅ PDF 成功: {success}, 🧬 Europe PMC XML 補救: {epmc_rescued}, "
        f"⏩ 已存在跳過: {skip}, ❌ 失敗: {fail}"
    )
    if fail_reasons:
        print("\n失敗原因分佈：")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            print(f"  - {reason}: {count} 篇")

    # 主動從 Europe PMC 補抓開放腐蝕全文 XML
    if ENABLE_EUROPEPMC_XML and EUROPEPMC_MAX_PAPERS > 0:
        run_europepmc_search()


if __name__ == "__main__":
    run_download()
