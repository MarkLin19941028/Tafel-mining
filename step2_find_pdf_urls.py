"""

Step 2: 用 Unpaywall API 根據 DOI 找合法 OA PDF URL

"""

import time

import requests

import pandas as pd

from tqdm import tqdm

from config import MAILTO, REQUEST_TIMEOUT, STEP1_OUTPUT, STEP2_OUTPUT

UNPAYWALL_BASE = "https://api.unpaywall.org/v2"


def find_pdf_url(doi: str) -> dict:

    doi_clean = doi.replace("https://doi.org/", "").strip()

    url = f"{UNPAYWALL_BASE}/{doi_clean}"

    params = {"email": MAILTO}

    try:

        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)  # ✅

        if resp.status_code == 404:

            return {"unpaywall_pdf_url": None, "license": None}

        if resp.status_code == 429:

            time.sleep(30)

            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)  # ✅

        resp.raise_for_status()

        data = resp.json()

    except Exception as e:

        return {"unpaywall_pdf_url": None, "license": None, "error": str(e)}

    best_oa = data.get("best_oa_location") or {}

    pdf_url = best_oa.get("url_for_pdf")

    landing = best_oa.get("url_for_landing_page")

    license_info = best_oa.get("license")

    return {

        "unpaywall_pdf_url": pdf_url,

        "unpaywall_landing_url": landing,

        "license": license_info,

        "is_oa_unpaywall": data.get("is_oa"),

    }


def run_find_pdfs():

    df = pd.read_csv(STEP1_OUTPUT)  # ✅

    print(f"Loaded {len(df)} papers from {STEP1_OUTPUT}")

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Unpaywall lookup"):

        doi = row.get("doi")

        if pd.isna(doi):

            results.append({})

            continue

        info = find_pdf_url(doi)

        results.append(info)

        time.sleep(0.5)

    pdf_df = pd.DataFrame(results)

    df = pd.concat([df.reset_index(drop=True), pdf_df], axis=1)

    def pick_best_pdf(row):

        for col in ["unpaywall_pdf_url", "best_pdf_url", "primary_pdf_url"]:

            val = row.get(col)

            if pd.notna(val) and str(val).startswith("http"):

                return val

        return None

    df["final_pdf_url"] = df.apply(pick_best_pdf, axis=1)

    n_pdf = df["final_pdf_url"].notna().sum()

    print(f"Papers with PDF URL: {n_pdf} / {len(df)}")

    df.to_csv(STEP2_OUTPUT, index=False, encoding="utf-8-sig")  # ✅

    print(f"✅ Saved to {STEP2_OUTPUT}")

    return df

if __name__ == "__main__":

    run_find_pdfs()
