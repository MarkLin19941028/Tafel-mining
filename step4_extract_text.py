"""
Step 4: 從 PDF 抽取純文字與表格
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import pdfplumber
import pymupdf as fitz  # PyMuPDF
from tqdm import tqdm

TABLE_EXTRACT_TIMEOUT_SEC = float(os.environ.get("TAFEL_TABLE_EXTRACT_TIMEOUT", "60"))


def extract_text_pymupdf(pdf_path: str) -> str:
    """使用 PyMuPDF 從 PDF 抽取純文字。"""
    try:
        with fitz.open(pdf_path) as doc:
            pages_text = [page.get_text() for page in doc]
        return "\n".join(pages_text)
    except Exception as e:
        return f"[ERROR] {e}"


def _extract_tables_pdfplumber_direct(pdf_path: str) -> list[list[list[str]]]:
    """直接在當前進程使用 pdfplumber 抽取 PDF 表格。"""
    tables: list[list[list[str]]] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                if page_tables:
                    for t in page_tables:
                        cleaned = [
                            [str(cell).strip() if cell else "" for cell in row]
                            for row in t
                        ]
                        tables.append(cleaned)
    except Exception:
        pass
    return tables


def extract_tables_with_timeout(
    pdf_path: str,
    timeout_sec: float | None = TABLE_EXTRACT_TIMEOUT_SEC,
) -> tuple[list[list[list[str]]], bool]:
    """以獨立子進程執行 pdfplumber 表格抽取，超時即中止並回傳 ([], True)。"""
    if timeout_sec is None or timeout_sec <= 0:
        return _extract_tables_pdfplumber_direct(pdf_path), False

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    child_env = {**os.environ, "TAFEL_QUIET_CONFIG": "1"}
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--_table-worker",
        pdf_path,
        tmp_path,
    ]
    try:
        proc = subprocess.run(
            cmd,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if proc.returncode != 0:
            return [], False
        with open(tmp_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (data if isinstance(data, list) else []), False
    except subprocess.TimeoutExpired:
        return [], True
    except (OSError, json.JSONDecodeError, ValueError):
        return [], False
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def extract_tables_pdfplumber(
    pdf_path: str,
    timeout_sec: float | None = TABLE_EXTRACT_TIMEOUT_SEC,
) -> list[list[list[str]]]:
    """從 PDF 抽取表格（預設啟用子進程逾時保護）。"""
    tables, _ = extract_tables_with_timeout(pdf_path, timeout_sec=timeout_sec)
    return tables


def run_extract(
    pdf_dir: str | None = None,
    text_dir: str | None = None,
    table_timeout_sec: float | None = None,
) -> None:
    """批次從 PDF 目錄抽取純文字與表格至文字目錄。"""
    if pdf_dir is None or text_dir is None:
        from config import PDF_DIR as _cfg_pdf_dir, TEXT_DIR as _cfg_text_dir

        pdf_dir = pdf_dir or _cfg_pdf_dir
        text_dir = text_dir or _cfg_text_dir

    effective_timeout = TABLE_EXTRACT_TIMEOUT_SEC if table_timeout_sec is None else table_timeout_sec
    os.makedirs(text_dir, exist_ok=True)

    pdf_files = sorted(f for f in os.listdir(pdf_dir) if f.endswith(".pdf"))
    print(f"Found {len(pdf_files)} PDFs in {pdf_dir}")

    new_text = 0
    new_table = 0
    timed_out_count = 0
    skip = 0

    for filename in tqdm(pdf_files, desc="Extracting text"):
        pdf_path = os.path.join(pdf_dir, filename)
        base_name = filename[:-4]

        text_path = os.path.join(text_dir, base_name + ".txt")
        table_path = os.path.join(text_dir, base_name + "_tables.json")

        if os.path.exists(text_path) and os.path.exists(table_path):
            skip += 1
            continue

        if not os.path.exists(text_path):
            text = extract_text_pymupdf(pdf_path)
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)
            new_text += 1

        if not os.path.exists(table_path):
            tables, timed_out = extract_tables_with_timeout(pdf_path, timeout_sec=effective_timeout)
            if timed_out:
                tqdm.write(
                    f"⚠️  Table extraction timed out ({effective_timeout:.0f}s), skipping tables: {filename}"
                )
                timed_out_count += 1
            with open(table_path, "w", encoding="utf-8") as f:
                json.dump(tables, f, ensure_ascii=False, indent=2)
            new_table += 1

    print(
        f"\n✅ New text: {new_text}, New tables: {new_table}, "
        f"Timed out: {timed_out_count}, Skipped: {skip}"
    )


def _run_table_worker(pdf_path: str, out_json_path: str) -> int:
    tables = _extract_tables_pdfplumber_direct(pdf_path)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(tables, f, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--_table-worker":
        sys.exit(_run_table_worker(sys.argv[2], sys.argv[3]))
    run_extract()
