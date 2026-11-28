"""
Step 7: 物理與電化學熱力學特徵強化（Materials Project API + 內建 Pourbaix 引擎）

功能：
1. 整併 data/merged/ 下的 electrochemical_database.csv、solution_properties.csv、material_properties.csv
2. 推算有效 pH（優先使用量測值，缺值時依酸鹼/鹽類組成與濃度推算）
3. 補齊內建元素/材料物理常數（功函數、標準電極電位 E°、電負度、熔點、莫耳體積）
4. 串接 Materials Project API（自動偵測 mp-api/pymatgen；若環境未安裝則自動使用原生 REST API v2，
   搭配 data/merged/mp_cache.json 本地快取），補齊：
   - DFT 加權功函數 (mp_work_function_eV) 與表面能 (mp_surface_energy_J_m2)
   - DFT 生成能、能隙、MP Material ID
   - 該 pH 與 Ecorr 下的 Pourbaix 熱力學特徵（mp_pourbaix_E_onset_V_SHE、mp_stable_phase_at_Ecorr、
     mp_corrosion_regime、mp_delta_G_pbx_eV_atom）
5. 產出：
   - data/merged/ml_training_dataset.csv
   - data/merged/galvanic_risk_preview.csv

常用指令：
  .venv/bin/python step7_enrich_data.py
  .venv/bin/python step7_enrich_data.py --refresh-cache
  .venv/bin/python step7_enrich_data.py --offline
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

import config
from step5_constants import (
    NERNST_SLOPE,
    PURE_MATERIAL_CLASSES,
    convert_to_she,
)

MP_REST_BASE = "https://api.materialsproject.org"
DG_H2O_EV = -2.4583  # ΔGf°(H2O, l) in eV at 298.15 K

# ============================================================
#  1. 內建元素與電化學熱力學常數表（CRC Handbook / Michaelson / Pourbaix Atlas）
# ============================================================

from materials_loader import build_compound_overrides, build_element_physics

ELEMENT_PHYSICS: dict[str, dict[str, Any]] = build_element_physics()
COMPOUND_OVERRIDES: dict[str, dict[str, Any]] = build_compound_overrides()


# ============================================================
#  2. 基礎元素與有效 pH 推算
# ============================================================

def resolve_base_element(material_id: Any, metal: Any = None) -> str:
    """從 material_id 或 metal 解析出元素符號（例如 TiN -> Ti, Cu2O -> Cu）。"""
    for raw in (material_id, metal):
        if pd.isna(raw) or not str(raw).strip():
            continue
        s = str(raw).strip()
        if s in COMPOUND_OVERRIDES:
            return COMPOUND_OVERRIDES[s]["base_element"]
        if s in ELEMENT_PHYSICS:
            return s
        # 擷取開頭的元素符號（例如 Cu2O -> Cu, Co3O4 -> Co, Ti-6Al-4V -> Ti）
        m = re.match(r"^([A-Z][a-z]?)", s)
        if m and m.group(1) in ELEMENT_PHYSICS:
            return m.group(1)
    return ""


def _parse_conc_molar(val: Any, unit: Any) -> float | None:
    if pd.isna(val):
        return None
    try:
        c = float(val)
    except (TypeError, ValueError):
        return None
    if c <= 0:
        return None
    u = str(unit).strip().lower() if pd.notna(unit) else "m"
    if u in ("mm", "mmol/l", "mmol l-1"):
        return c * 1e-3
    if u in ("um", "µm", "μm"):
        return c * 1e-6
    if u in ("m", "mol/l", "mol l-1", "n"):
        return c
    if u in ("wt%", "wt %", "%"):
        # 粗估：1 wt% ~ 0.15 M（僅作為無量測 pH 時的近似）
        return c * 0.15
    return c


def infer_effective_ph(row: pd.Series) -> tuple[float, str]:
    """
    解析有效 pH：
    1. electrochemical_database.pH（measured_echem）
    2. solution_properties.pH（measured_solution）
    3. 依酸/鹼類型與濃度推算（inferred_acid / inferred_base）
    4. 依 solution_category 或中性鹽推算（inferred_neutral / inferred_category）
    """
    for col, tag in (("pH", "measured_echem"), ("pH_sol", "measured_solution")):
        val = row.get(col)
        if pd.notna(val):
            try:
                ph = float(val)
                if 0.0 <= ph <= 14.0:
                    return round(ph, 2), tag
            except (TypeError, ValueError):
                pass

    acid = str(row.get("acid_type", "") or "").strip().lower()
    base = str(row.get("base_type", "") or "").strip().lower()
    desc = str(row.get("solution_description", "") or row.get("solution_name", "") or "").lower()

    # 酸類推算
    if acid or any(k in desc for k in ("hcl", "h2so4", "hno3", "hclo4", "hf", "h3po4", "citric", "oxalic")):
        c = _parse_conc_molar(row.get("acid_concentration"), row.get("acid_concentration_unit"))
        if c is None:
            m = re.search(r"(\d+(?:\.\d+)?)\s*(m?m)\s*(?:hcl|h2so4|hno3|hf)", desc)
            if m:
                c = float(m.group(1)) * (1e-3 if m.group(2) == "mm" else 1.0)
        c = c if (c and c > 0) else 0.1
        if any(k in (acid + " " + desc) for k in ("h2so4", "sulfuric")):
            ph = max(0.0, -math.log10(min(2.0, c * 1.2)))
            return round(ph, 2), "inferred_strong_acid"
        if any(k in (acid + " " + desc) for k in ("hcl", "hno3", "hclo4", "hydrochloric", "nitric")):
            ph = max(0.0, -math.log10(min(2.0, c)))
            return round(ph, 2), "inferred_strong_acid"
        if any(k in (acid + " " + desc) for k in ("citric", "oxalic", "acetic", "hf", "h3po4")):
            # 弱酸 pKa ~ 3.1，pH ≈ 0.5*(pKa - log10(c))
            ph = max(1.0, min(6.0, 0.5 * (3.1 - math.log10(max(1e-4, c)))))
            return round(ph, 2), "inferred_weak_acid"

    # 鹼類推算
    if base or any(k in desc for k in ("koh", "naoh", "tmah", "nh4oh", "ammonia")):
        c = _parse_conc_molar(row.get("base_concentration"), row.get("base_concentration_unit"))
        if c is None:
            m = re.search(r"(\d+(?:\.\d+)?)\s*(m?m)\s*(?:koh|naoh|tmah|nh4oh)", desc)
            if m:
                c = float(m.group(1)) * (1e-3 if m.group(2) == "mm" else 1.0)
        c = c if (c and c > 0) else 0.1
        if any(k in (base + " " + desc) for k in ("koh", "naoh", "tmah")):
            ph = min(14.0, 14.0 + math.log10(min(2.0, c)))
            return round(ph, 2), "inferred_strong_base"
        if any(k in (base + " " + desc) for k in ("nh4oh", "ammonia", "sc-1", "apm")):
            ph = min(12.5, 11.6 + 0.5 * math.log10(max(1e-4, c)))
            return round(ph, 2), "inferred_weak_base"

    cat = str(row.get("solution_category", "") or "").strip().lower()
    if cat == "acidic":
        return 2.0, "inferred_category_acidic"
    if cat == "alkaline":
        return 11.0, "inferred_category_alkaline"

    return 7.0, "inferred_neutral"


# ============================================================
#  3. Materials Project API 雙後端（mp-api/pymatgen 或原生 REST v2）+ 快取
# ============================================================

def _load_mp_cache(cache_path: Path) -> dict[str, Any]:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_mp_cache(cache_path: Path, cache: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _mp_rest_get(
    endpoint: str,
    params: dict[str, Any],
    api_key: str,
    *,
    timeout: int | None = None,
) -> list[dict]:
    req_timeout = timeout if timeout is not None else config.load_settings().request_timeout
    url = f"{MP_REST_BASE}{endpoint}"
    headers = {
        "X-API-KEY": api_key,
        "Accept": "application/json",
        "User-Agent": "TafelMiningStep7/1.0",
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=req_timeout)
        if resp.status_code != 200:
            return []
        payload = resp.json()
        data = payload.get("data")
        return data if isinstance(data, list) else []
    except Exception:
        return []


def fetch_mp_properties_for_material(
    key_formula: str,
    base_element: str,
    api_key: str,
    cache: dict[str, Any],
    refresh: bool = False,
    *,
    timeout: int | None = None,
) -> dict[str, Any]:
    """
    從 Materials Project 取得：
    - 最穩定晶相 summary（material_id, formation_energy_per_atom, band_gap, density）
    - 元素表面性質（weighted_work_function, weighted_surface_energy）
    - 對應主氧化物的 DFT 生成能（用於計算 DFT 校正 Pourbaix 氧化起始電位）
    """
    cache_key = key_formula or base_element
    if not cache_key:
        return {}
    if not refresh and cache_key in cache:
        return cache[cache_key]
    if not api_key:
        return {}

    result: dict[str, Any] = {
        "query_formula": cache_key,
        "base_element": base_element,
        "mp_material_id": None,
        "mp_formation_energy_eV_atom": None,
        "mp_band_gap_eV": None,
        "mp_density_g_cm3": None,
        "mp_work_function_eV": None,
        "mp_surface_energy_J_m2": None,
        "mp_oxide_formula": None,
        "mp_oxide_material_id": None,
        "mp_oxide_formation_energy_eV_atom": None,
        "mp_oxide_E0_V_SHE": None,
        "mp_backend": "rest_v2",
    }

    # 1. Summary（金屬或化合物如 TiN / TaN）
    summary_docs = _mp_rest_get(
        "/materials/summary/",
        {
            "formula": cache_key,
            "energy_above_hull_max": 0.05,
            "_fields": "material_id,formula_pretty,energy_above_hull,formation_energy_per_atom,band_gap,density",
            "_limit": 10,
        },
        api_key,
        timeout=timeout,
    )
    if summary_docs:
        summary_docs.sort(key=lambda d: d.get("energy_above_hull", 999.0))
        best = summary_docs[0]
        result["mp_material_id"] = best.get("material_id")
        result["mp_formation_energy_eV_atom"] = best.get("formation_energy_per_atom")
        result["mp_band_gap_eV"] = best.get("band_gap")
        result["mp_density_g_cm3"] = best.get("density")

    # 2. Surface properties（取 DFT 加權功函數與表面能；化合物回退查基底金屬）
    elem_mp_id = result["mp_material_id"] if cache_key == base_element else None
    if not elem_mp_id and base_element:
        elem_docs = _mp_rest_get(
            "/materials/summary/",
            {
                "formula": base_element,
                "energy_above_hull_max": 0.05,
                "_fields": "material_id,energy_above_hull",
                "_limit": 5,
            },
            api_key,
            timeout=timeout,
        )
        if elem_docs:
            elem_docs.sort(key=lambda d: d.get("energy_above_hull", 999.0))
            elem_mp_id = elem_docs[0].get("material_id")

    if elem_mp_id:
        surf_docs = _mp_rest_get(
            "/materials/surface_properties/",
            {
                "material_ids": elem_mp_id,
                "_fields": "material_id,weighted_work_function,weighted_surface_energy",
                "_limit": 5,
            },
            api_key,
            timeout=timeout,
        )
        if surf_docs:
            sdoc = surf_docs[0]
            if sdoc.get("weighted_work_function") is not None:
                result["mp_work_function_eV"] = round(float(sdoc["weighted_work_function"]), 4)
            if sdoc.get("weighted_surface_energy") is not None:
                result["mp_surface_energy_J_m2"] = round(float(sdoc["weighted_surface_energy"]), 4)

    # 3. 對應主氧化物 DFT 生成能 -> 計算 DFT 氧化平衡電位 E0(M / MaOb)
    ep = ELEMENT_PHYSICS.get(base_element)
    if ep and ep.get("oxide_formula"):
        ox_formula = ep["oxide_formula"]
        a_m, b_o = ep["oxide_stoich"]
        ox_docs = _mp_rest_get(
            "/materials/summary/",
            {
                "formula": ox_formula,
                "energy_above_hull_max": 0.05,
                "_fields": "material_id,formula_pretty,energy_above_hull,formation_energy_per_atom",
                "_limit": 10,
            },
            api_key,
            timeout=timeout,
        )
        if ox_docs:
            ox_docs.sort(key=lambda d: d.get("energy_above_hull", 999.0))
            ox = ox_docs[0]
            fe_atom = ox.get("formation_energy_per_atom")
            result["mp_oxide_formula"] = ox_formula
            result["mp_oxide_material_id"] = ox.get("material_id")
            if fe_atom is not None:
                fe_atom = float(fe_atom)
                result["mp_oxide_formation_energy_eV_atom"] = round(fe_atom, 4)
                # ΔGf°(MaOb, eV/f.u.) = (a + b) * formation_energy_per_atom
                # a M + b H2O <-> MaOb + 2b H+ + 2b e-
                # E° (V vs SHE) = (ΔGf°(MaOb) - b * ΔGf°(H2O)) / (2b)
                dg_ox_fu = (a_m + b_o) * fe_atom
                e0_oxide = (dg_ox_fu - b_o * DG_H2O_EV) / (2.0 * b_o)
                result["mp_oxide_E0_V_SHE"] = round(e0_oxide, 4)

    # 只在成功從 MP 取得至少一項有效資料時才寫入快取，避免暫時性斷網污染快取
    if result["mp_material_id"] or result["mp_work_function_eV"] or result["mp_oxide_E0_V_SHE"]:
        cache[cache_key] = result
    return result


def _try_build_pymatgen_pourbaix(base_element: str, api_key: str):
    """若環境已安裝 mp-api 與 pymatgen，嘗試建立完整 DFT PourbaixDiagram。"""
    try:
        from mp_api.client import MPRester  # type: ignore
        from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram  # type: ignore
    except ImportError:
        return None
    if not api_key or not base_element:
        return None
    try:
        with MPRester(api_key) as mpr:
            entries = mpr.get_pourbaix_entries([base_element])
        if entries:
            return PourbaixDiagram(entries, filter_solids=True)
    except Exception:
        return None
    return None


# ============================================================
#  4. Pourbaix 熱力學狀態計算（Layer 1 解析模型 + Layer 2 MP DFT 強化）
# ============================================================

def evaluate_pourbaix_analytical(
    material_key: str,
    base_element: str,
    ph: float,
    ecorr: float | None,
) -> dict[str, Any]:
    """
    Layer 1：基於標準電極電位 E° 與氧化物穩定 pH 區間的 Nernst / Pourbaix 解析模型。
    """
    ep = ELEMENT_PHYSICS.get(base_element)
    if not ep:
        return {
            "thermo_E_onset_V_SHE": None,
            "thermo_corrosion_regime": None,
            "thermo_stable_phase": None,
            "thermo_source": "unknown",
        }

    ovr = COMPOUND_OVERRIDES.get(material_key, {})
    e0 = ovr.get("E0_V", ep["E0_V"])
    valence = ep["valence"]
    pmin, pmax = ovr.get("passiv_pH", ep["passiv_pH"])
    e_trans = ep["transpassive_E"]
    oxide_phase = ovr.get("oxide_phase", ep["oxide_phase"])
    trans_phase = ep.get("transpassive_phase", ep["ion_alk"])

    # 活度 10^-6 M 條件下的金屬離子平衡電位：E_ion = E0 + (0.05916/n)*log10(1e-6)
    e_ion = e0 - (6.0 * NERNST_SLOPE / max(1, valence))

    # 氧化物形成平衡電位（在鈍化區下緣與離子線銜接，斜率 -0.05916 V/pH）
    e_ox_at_ph = e_ion - NERNST_SLOPE * max(0.0, ph - pmin)
    e_onset = e_ox_at_ph if ph >= pmin else e_ion

    regime = None
    stable_phase = None
    if ecorr is not None and not pd.isna(ecorr):
        e_val = float(ecorr)
        if e_val < e_onset:
            regime = "immunity"
            stable_phase = f"{material_key or base_element}(s)"
        elif e_val > e_trans - NERNST_SLOPE * max(0.0, ph - 7.0) * 0.5:
            regime = "transpassive"
            stable_phase = trans_phase
        elif pmin <= ph <= pmax:
            regime = "passivation"
            stable_phase = oxide_phase
        elif ph < pmin:
            regime = "corrosion"
            stable_phase = ep["ion_acid"]
        else:
            regime = "corrosion"
            stable_phase = ep["ion_alk"]

    return {
        "thermo_E_onset_V_SHE": round(e_onset, 4),
        "thermo_corrosion_regime": regime,
        "thermo_stable_phase": stable_phase,
        "thermo_source": "analytical_layer1",
    }


def evaluate_pourbaix_mp(
    material_key: str,
    base_element: str,
    ph: float,
    ecorr: float | None,
    mp_info: dict[str, Any],
    pbx_diagram: Any = None,
) -> dict[str, Any]:
    """
    Layer 2：Materials Project DFT Pourbaix 特徵計算。
    - 若有 pymatgen PourbaixDiagram，直接查詢精確穩定相與 ΔG_pbx。
    - 否則使用 MP REST v2 取得的 DFT 氧化物生成能（mp_oxide_E0_V_SHE）計算
      DFT 氧化平衡電位 E_ox(pH) = E0_ox - 0.05916 * pH 與相對熱力學驅動力 ΔG_pbx (eV/atom)。
    """
    out = {
        "mp_pourbaix_E_onset_V_SHE": None,
        "mp_stable_phase_at_Ecorr": None,
        "mp_corrosion_regime": None,
        "mp_delta_G_pbx_eV_atom": None,
    }

    # 路徑 A：pymatgen PourbaixDiagram（若已安裝）
    if pbx_diagram is not None and ecorr is not None and not pd.isna(ecorr):
        try:
            e_val = float(ecorr)
            stable = pbx_diagram.get_stable_entry(ph, e_val)
            phase_name = stable.name
            # 掃描該 pH 下金屬純相失去穩定性的電位作為 E_onset
            e_onset = None
            for e_test in np.linspace(-2.5, 2.0, 181):
                st = pbx_diagram.get_stable_entry(ph, float(e_test))
                if st.name.replace("(s)", "").strip() != base_element:
                    e_onset = round(float(e_test), 4)
                    break
            dg = None
            for entry in pbx_diagram.all_entries:
                if entry.name.replace("(s)", "").strip() == base_element:
                    dg = round(float(pbx_diagram.get_decomposition_energy(entry, ph, e_val)), 4)
                    break
            if phase_name.replace("(s)", "").strip() == base_element:
                regime = "immunity"
            elif "[ion]" in phase_name or "+" in phase_name or "-" in phase_name:
                regime = "corrosion"
            else:
                regime = "passivation"
            out.update({
                "mp_pourbaix_E_onset_V_SHE": e_onset,
                "mp_stable_phase_at_Ecorr": phase_name,
                "mp_corrosion_regime": regime,
                "mp_delta_G_pbx_eV_atom": dg,
            })
            return out
        except Exception:
            pass

    # 路徑 B：MP REST v2 DFT 氧化物熱力學模型（零額外套件相依，Python 3.14 原生支援）
    ep = ELEMENT_PHYSICS.get(base_element)
    e0_ox = mp_info.get("mp_oxide_E0_V_SHE")
    if not ep or e0_ox is None:
        return out

    ovr = COMPOUND_OVERRIDES.get(material_key, {})
    pmin, pmax = ovr.get("passiv_pH", ep["passiv_pH"])
    e_trans = ep["transpassive_E"]
    valence = ep["valence"]
    a_m, b_o = ep["oxide_stoich"]

    # 離子溶解線與 DFT 氧化物線取較低者作為氧化起始電位
    e_ion = ovr.get("E0_V", ep["E0_V"]) - (6.0 * NERNST_SLOPE / max(1, valence))
    e_ox_ph = float(e0_ox) - NERNST_SLOPE * ph
    e_onset_dft = min(e_ion, e_ox_ph) if pmin <= ph <= pmax else (e_ion if ph < pmin else e_ox_ph)
    out["mp_pourbaix_E_onset_V_SHE"] = round(e_onset_dft, 4)

    if ecorr is not None and not pd.isna(ecorr):
        e_val = float(ecorr)
        # 計算固態金屬相相對於氧化態的熱力學不穩定度 ΔG_pbx (eV / metal atom)
        # 越正代表金屬越不穩定（腐蝕/氧化驅動力越大）；<= 0 代表處於熱力學免疫區
        n_e_per_metal = (2.0 * b_o) / max(1, a_m)
        dg_pbx = max(0.0, n_e_per_metal * (e_val - e_onset_dft))
        out["mp_delta_G_pbx_eV_atom"] = round(dg_pbx, 4)

        if e_val < e_onset_dft:
            out["mp_corrosion_regime"] = "immunity"
            out["mp_stable_phase_at_Ecorr"] = f"{material_key or base_element}(s)"
        elif e_val > e_trans - NERNST_SLOPE * max(0.0, ph - 7.0) * 0.5:
            out["mp_corrosion_regime"] = "transpassive"
            out["mp_stable_phase_at_Ecorr"] = ep.get("transpassive_phase", ep["ion_alk"])
        elif pmin <= ph <= pmax:
            out["mp_corrosion_regime"] = "passivation"
            out["mp_stable_phase_at_Ecorr"] = f"{mp_info.get('mp_oxide_formula') or ep['oxide_formula']}(s)"
        elif ph < pmin:
            out["mp_corrosion_regime"] = "corrosion"
            out["mp_stable_phase_at_Ecorr"] = ep["ion_acid"]
        else:
            out["mp_corrosion_regime"] = "corrosion"
            out["mp_stable_phase_at_Ecorr"] = ep["ion_alk"]

    return out


# ============================================================
#  5. 電偶腐蝕風險預覽表
# ============================================================

def build_galvanic_risk_preview(df: pd.DataFrame) -> pd.DataFrame:
    """依各金屬中位數 Ecorr 與功函數差，建立兩兩配對的電偶腐蝕風險表。"""
    if df.empty or "Ecorr_V_SHE" not in df.columns:
        return pd.DataFrame()

    valid = df.dropna(subset=["Ecorr_V_SHE"]).copy()
    if valid.empty:
        return pd.DataFrame()

    group_col = "material_id" if "material_id" in valid.columns else "base_element"
    stats = (
        valid.groupby(group_col)
        .agg(
            ecorr_median=("Ecorr_V_SHE", "median"),
            wf=("elem_work_function_eV", "first"),
            mp_wf=("mp_work_function_eV", "first"),
            n=("Ecorr_V_SHE", "count"),
        )
        .reset_index()
    )
    # 只保留樣本數 >= 3 的金屬以減少雜訊
    stats = stats[stats["n"] >= 3].sort_values("ecorr_median", ascending=False).reset_index(drop=True)
    if len(stats) < 2:
        return pd.DataFrame()

    rows = []
    for i in range(len(stats)):
        for j in range(i + 1, len(stats)):
            cat = stats.iloc[i]
            ano = stats.iloc[j]
            delta_e = float(cat["ecorr_median"]) - float(ano["ecorr_median"])
            cat_wf = cat["mp_wf"] if pd.notna(cat["mp_wf"]) else cat["wf"]
            ano_wf = ano["mp_wf"] if pd.notna(ano["mp_wf"]) else ano["wf"]
            delta_wf = (
                round(float(cat_wf) - float(ano_wf), 4)
                if (pd.notna(cat_wf) and pd.notna(ano_wf))
                else None
            )
            if delta_e >= 0.30:
                risk = "High"
            elif delta_e >= 0.15:
                risk = "Moderate"
            else:
                risk = "Low"
            rows.append({
                "cathode_protected_metal": cat[group_col],
                "anode_corroded_metal": ano[group_col],
                "cathode_Ecorr_median_V_SHE": round(float(cat["ecorr_median"]), 4),
                "anode_Ecorr_median_V_SHE": round(float(ano["ecorr_median"]), 4),
                "delta_Ecorr_V": round(delta_e, 4),
                "delta_work_function_eV": delta_wf,
                "galvanic_risk_level": risk,
                "cathode_sample_count": int(cat["n"]),
                "anode_sample_count": int(ano["n"]),
            })

    return pd.DataFrame(rows).sort_values("delta_Ecorr_V", ascending=False)


# ============================================================
#  6. 主執行流程
# ============================================================

def run_enrich(
    refresh_cache: bool = False,
    offline: bool = False,
    include_alloys: bool = False,
    *,
    settings: config.PipelineSettings | None = None,
) -> pd.DataFrame:
    resolved_settings = settings or config.load_settings()
    merged_dir = Path(resolved_settings.simulator_data_dir)
    echem_path = merged_dir / "electrochemical_database.csv"
    sol_path = merged_dir / "solution_properties.csv"
    mat_path = merged_dir / "material_properties.csv"

    if not echem_path.exists():
        raise FileNotFoundError(f"找不到 {echem_path}，請先執行 Step 6 (step6_merge.py)。")

    echem_df = pd.read_csv(echem_path)
    sol_df = pd.read_csv(sol_path) if sol_path.exists() else pd.DataFrame()
    mat_df = pd.read_csv(mat_path) if mat_path.exists() else pd.DataFrame()

    print(f"[Step 7] 載入合併資料：Echem={len(echem_df)}, Solutions={len(sol_df)}, Materials={len(mat_df)}")

    df = echem_df.copy()

    # 預設只針對純金屬與目標化合物做純元素 DFT/Pourbaix 強化與電偶模擬
    if not include_alloys and "material_class" in df.columns:
        before = len(df)
        df = df[df["material_class"].isin(PURE_MATERIAL_CLASSES)].reset_index(drop=True)
        excluded = before - len(df)
        if excluded > 0:
            print(f"[Step 7] 排除 {excluded} 筆 alloy/coating_composite 記錄（保留 {len(df)} 筆 pure/compound）")

    # 1. 左連接 solution_properties（避免覆蓋 echem 原有欄位）
    if not sol_df.empty and "solution_id" in df.columns and "solution_id" in sol_df.columns:
        sol_dedup = sol_df.drop_duplicates(subset=["solution_id"], keep="first").copy()
        sol_rename = {}
        if "pH" in sol_dedup.columns:
            sol_rename["pH"] = "pH_sol"
        sol_dedup = sol_dedup.rename(columns=sol_rename)
        drop_overlap = [
            c for c in sol_dedup.columns
            if c in df.columns and c != "solution_id"
        ]
        sol_dedup = sol_dedup.drop(columns=drop_overlap, errors="ignore")
        df = df.merge(sol_dedup, on="solution_id", how="left")

    # 2. 左連接 material_properties
    if not mat_df.empty and "material_id" in df.columns and "material_id" in mat_df.columns:
        mat_dedup = mat_df.drop_duplicates(subset=["material_id"], keep="first").copy()
        drop_overlap = [
            c for c in mat_dedup.columns
            if c in df.columns and c != "material_id"
        ]
        mat_dedup = mat_dedup.drop(columns=drop_overlap, errors="ignore")
        df = df.merge(mat_dedup, on="material_id", how="left")

    # 3. 解析 base_element 與有效 pH
    df["base_element"] = df.apply(
        lambda r: resolve_base_element(r.get("material_id"), r.get("metal")),
        axis=1,
    )
    ph_pairs = df.apply(infer_effective_ph, axis=1)
    df["pH_effective"] = [p[0] for p in ph_pairs]
    df["pH_source"] = [p[1] for p in ph_pairs]

    # 若為 RHE 且 Step 5 缺實測 pH 導致無法換算，利用 Step 7 推算的 pH_effective 補算 Ecorr_V_SHE
    if {"Ecorr_V_SHE", "Ecorr_V_original", "reference_electrode"}.issubset(df.columns):
        ref_low = df["reference_electrode"].fillna("").astype(str).str.lower()
        rhe_mask = (
            df["Ecorr_V_SHE"].isna()
            & df["Ecorr_V_original"].notna()
            & df["pH_effective"].notna()
            & (ref_low.str.contains("rhe") | ref_low.str.contains("reversible hydrogen"))
        )
        if rhe_mask.any():
            converted = [
                convert_to_she(float(r["Ecorr_V_original"]), str(r["reference_electrode"]), float(r["pH_effective"]))
                for _, r in df.loc[rhe_mask].iterrows()
            ]
            df.loc[rhe_mask, "Ecorr_V_SHE"] = [
                round(v, 4) if v is not None else None for v in converted
            ]
            filled = sum(v is not None for v in converted)
            print(f"[Step 7] 以 pH_effective 補算 {filled} 筆 RHE 記錄的 Ecorr_V_SHE")

    # 下游 ML/模擬缺溫度時預設回退 25°C
    if "temperature_C" in df.columns:
        df["temperature_C"] = pd.to_numeric(df["temperature_C"], errors="coerce").fillna(25.0)

    # 4. 查詢 Materials Project API（使用快取避免重複請求）
    mp_api_key = resolved_settings.mp_api_key
    use_mp = resolved_settings.enable_mp_api and (not offline) and bool(mp_api_key)
    cache_path = Path(resolved_settings.mp_cache_path)
    mp_cache = _load_mp_cache(cache_path)

    unique_pairs: set[tuple[str, str]] = set()
    for _, r in df[["material_id", "base_element"]].drop_duplicates().iterrows():
        mk = str(r["material_id"]).strip() if pd.notna(r["material_id"]) else ""
        be = str(r["base_element"]).strip() if pd.notna(r["base_element"]) else ""
        if be:
            query_key = mk if mk in COMPOUND_OVERRIDES else be
            unique_pairs.add((query_key, be))

    mp_lookup: dict[str, dict[str, Any]] = {}
    pbx_diagrams: dict[str, Any] = {}

    if use_mp:
        print(f"[Step 7] 串接 Materials Project API（共 {len(unique_pairs)} 種唯一材料，快取：{cache_path.name}）")
        for query_key, be in sorted(unique_pairs):
            info = fetch_mp_properties_for_material(
                query_key,
                be,
                mp_api_key,
                mp_cache,
                refresh=refresh_cache,
                timeout=resolved_settings.request_timeout,
            )
            mp_lookup[query_key] = info
            if be not in pbx_diagrams:
                pbx_diagrams[be] = _try_build_pymatgen_pourbaix(be, mp_api_key)
        _save_mp_cache(cache_path, mp_cache)
    else:
        print("[Step 7] 離線模式或未設定 MP_API_KEY：使用快取資料（若有）與內建熱力學表")
        for query_key, _ in unique_pairs:
            if query_key in mp_cache:
                mp_lookup[query_key] = mp_cache[query_key]

    # 5. 逐筆填入物理常數、MP DFT 性質與 Pourbaix 熱力學特徵
    enriched_cols: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        mk = str(row.get("material_id", "") or "").strip()
        be = str(row.get("base_element", "") or "").strip()
        ph = float(row["pH_effective"])
        ecorr = row.get("Ecorr_V_SHE")
        ecorr_f = float(ecorr) if pd.notna(ecorr) else None

        ep = ELEMENT_PHYSICS.get(be, {})
        ovr = COMPOUND_OVERRIDES.get(mk, {})
        query_key = mk if mk in COMPOUND_OVERRIDES else be
        mp_info = mp_lookup.get(query_key, {})

        # Layer 1 解析 Pourbaix
        l1 = evaluate_pourbaix_analytical(mk or be, be, ph, ecorr_f)
        # Layer 2 MP DFT Pourbaix
        l2 = evaluate_pourbaix_mp(
            mk or be, be, ph, ecorr_f, mp_info, pbx_diagram=pbx_diagrams.get(be)
        )

        # 過電位驅動力 ΔE = Ecorr - E_onset（優先採 MP Pourbaix 起始電位，無則回退 Layer 1）
        e_onset_ref = (
            l2["mp_pourbaix_E_onset_V_SHE"]
            if l2["mp_pourbaix_E_onset_V_SHE"] is not None
            else l1["thermo_E_onset_V_SHE"]
        )
        delta_e = (
            round(ecorr_f - float(e_onset_ref), 4)
            if (ecorr_f is not None and e_onset_ref is not None)
            else None
        )

        enriched_cols.append({
            "elem_atomic_number": ep.get("Z"),
            "elem_atomic_mass": ep.get("mass"),
            "elem_pauling_en": ep.get("en"),
            "elem_work_function_eV": ovr.get("wf_eV", ep.get("wf_eV")),
            "elem_standard_potential_V_SHE": ovr.get("E0_V", ep.get("E0_V")),
            "elem_melting_point_K": ep.get("Tm_K"),
            "elem_molar_volume_cm3": ep.get("Vm_cm3"),
            "mp_material_id": mp_info.get("mp_material_id"),
            "mp_work_function_eV": mp_info.get("mp_work_function_eV"),
            "mp_surface_energy_J_m2": mp_info.get("mp_surface_energy_J_m2"),
            "mp_formation_energy_eV_atom": mp_info.get("mp_formation_energy_eV_atom"),
            "mp_band_gap_eV": mp_info.get("mp_band_gap_eV"),
            "thermo_E_onset_V_SHE": l1["thermo_E_onset_V_SHE"],
            "thermo_corrosion_regime": l1["thermo_corrosion_regime"],
            "thermo_stable_phase": l1["thermo_stable_phase"],
            "thermo_source": (
                "mp_pourbaix" if l2["mp_pourbaix_E_onset_V_SHE"] is not None else l1["thermo_source"]
            ),
            "mp_pourbaix_E_onset_V_SHE": l2["mp_pourbaix_E_onset_V_SHE"],
            "mp_stable_phase_at_Ecorr": l2["mp_stable_phase_at_Ecorr"],
            "mp_corrosion_regime": l2["mp_corrosion_regime"],
            "mp_delta_G_pbx_eV_atom": l2["mp_delta_G_pbx_eV_atom"],
            "delta_E_overpotential_V": delta_e,
        })

    enrich_df = pd.DataFrame(enriched_cols)
    # 清除暫存 pH_sol 欄位
    df = df.drop(columns=["pH_sol"], errors="ignore")
    final_df = pd.concat([df.reset_index(drop=True), enrich_df], axis=1)

    out_ml = merged_dir / "ml_training_dataset.csv"
    final_df.to_csv(out_ml, index=False, encoding="utf-8-sig")

    galv_df = build_galvanic_risk_preview(final_df)
    out_galv = merged_dir / "galvanic_risk_preview.csv"
    galv_df.to_csv(out_galv, index=False, encoding="utf-8-sig")

    # 印出強化覆蓋率摘要
    print("\n" + "=" * 60)
    print("  STEP 7 ENRICHMENT SUMMARY")
    print("=" * 60)
    print(f"  Output ML dataset : {out_ml} ({len(final_df)} rows, {len(final_df.columns)} cols)")
    print(f"  Galvanic preview  : {out_galv} ({len(galv_df)} pairs)")
    for col in [
        "elem_work_function_eV",
        "elem_standard_potential_V_SHE",
        "mp_work_function_eV",
        "thermo_E_onset_V_SHE",
        "mp_pourbaix_E_onset_V_SHE",
        "mp_stable_phase_at_Ecorr",
        "delta_E_overpotential_V",
    ]:
        if col in final_df.columns and len(final_df):
            n = int(final_df[col].notna().sum())
            print(f"  {col:<30s}: {n:4d} / {len(final_df)} ({n / len(final_df) * 100:5.1f}%)")
    print("=" * 60)
    return final_df


def main():
    parser = argparse.ArgumentParser(
        description="Step 7: Materials Project API 與電化學熱力學（Pourbaix）特徵強化"
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="忽略本地 mp_cache.json，重新向 Materials Project API 查詢",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="離線模式：不連線 Materials Project API，僅使用本地快取與內建熱力學表",
    )
    parser.add_argument(
        "--include-alloys",
        action="store_true",
        help="包含 alloy / coating_composite 記錄（預設只針對 pure / compound 做純元素特徵強化）",
    )
    args = parser.parse_args()
    run_enrich(
        refresh_cache=args.refresh_cache,
        offline=args.offline,
        include_alloys=args.include_alloys,
    )


if __name__ == "__main__":
    main()
