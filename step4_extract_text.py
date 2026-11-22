"""

Step 4: 從 PDF 抽取純文字與表格

"""

import os

import json

import pymupdf as fitz  # PyMuPDF

import pdfplumber

from tqdm import tqdm

 

from config import PDF_DIR, TEXT_DIR

 

def extract_text_pymupdf(pdf_path: str) -> str:

    try:

        doc = fitz.open(pdf_path)

        pages_text = []

        for page in doc:

            pages_text.append(page.get_text())

        doc.close()

        return "\n".join(pages_text)

    except Exception as e:

        return f"[ERROR] {e}"

 

def extract_tables_pdfplumber(pdf_path: str) -> list[list[list[str]]]:

    tables = []

    try:

        with pdfplumber.open(pdf_path) as pdf:

            for page in pdf.pages:

                page_tables = page.extract_tables()

                if page_tables:

                    for t in page_tables:

                        cleaned = []

                        for row in t:

                            cleaned.append([

                                str(cell).strip() if cell else ""

                                for cell in row

                            ])

                        tables.append(cleaned)

    except Exception:

        pass

    return tables

 

def run_extract():

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

    print(f"Found {len(pdf_files)} PDFs in {PDF_DIR}")

 

    new_text = 0

    new_table = 0

    skip = 0

 

    for filename in tqdm(pdf_files, desc="Extracting text"):

        pdf_path = os.path.join(PDF_DIR, filename)

        base_name = filename.replace(".pdf", "")

 

        text_path = os.path.join(TEXT_DIR, base_name + ".txt")

        table_path = os.path.join(TEXT_DIR, base_name + "_tables.json")

 

        if os.path.exists(text_path) and os.path.exists(table_path):

            skip += 1

            continue

 

        if not os.path.exists(text_path):

            text = extract_text_pymupdf(pdf_path)

            with open(text_path, "w", encoding="utf-8") as f:

                f.write(text)

            new_text += 1

 

        if not os.path.exists(table_path):

            tables = extract_tables_pdfplumber(pdf_path)

            with open(table_path, "w", encoding="utf-8") as f:

                json.dump(tables, f, ensure_ascii=False, indent=2)

            new_table += 1

 

    print(f"\n✅ New text: {new_text}, New tables: {new_table}, Skipped: {skip}")

 

if __name__ == "__main__":

    run_extract()