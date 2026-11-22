"""
Unified loader and validator for metals_db.json and materials_physics.json.

Provides backward-compatible views for:
  - step5_constants.KNOWN_MATERIALS
  - step5_text_extractor (_BASE_SYMBOLS, _NAME_TO_SYMBOL, _MATERIAL_ALIASES)
  - step7_enrich_data (ELEMENT_PHYSICS, COMPOUND_OVERRIDES)
  - run_all.py --check-db (validate_databases)
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
METALS_DB_PATH = PROJECT_ROOT / "metals_db.json"
PHYSICS_DB_PATH = PROJECT_ROOT / "materials_physics.json"


@lru_cache(maxsize=1)
def load_physics_db(db_path: Path = PHYSICS_DB_PATH) -> dict[str, dict[str, Any]]:
    with db_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_known_materials(db_path: Path = PHYSICS_DB_PATH) -> dict[str, dict[str, Any]]:
    """組裝 Step 5 使用的 KNOWN_MATERIALS（含化合物對 base_element 的自動 fallback）。"""
    raw = load_physics_db(db_path)
    elements = raw.get("elements", {})
    compounds = raw.get("compounds", {})
    result: dict[str, dict[str, Any]] = {}

    def _to_step5_entry(sym: str, entry: dict[str, Any], base: dict[str, Any] | None = None) -> dict[str, Any]:
        merged = {**(base or {}), **entry}
        passiv_ph = merged.get("passivation_pH")
        ph_min = passiv_ph[0] if isinstance(passiv_ph, list) and len(passiv_ph) == 2 else None
        ph_max = passiv_ph[1] if isinstance(passiv_ph, list) and len(passiv_ph) == 2 else None
        tm_k = merged.get("melting_point_K")
        tm_c = round(tm_k - 273.15, 2) if tm_k is not None else None

        return {
            "material_name": entry.get("material_name", sym),
            "symbol": sym,
            "category": entry.get("category", merged.get("category")),
            "density_g_cm3": entry.get("density_g_cm3"),
            "molar_mass_g_mol": merged.get("molar_mass_g_mol"),
            "valence": merged.get("valence"),
            "equivalent_weight_g_eq": merged.get("equivalent_weight_g_eq"),
            "standard_potential_V_SHE": merged.get("standard_potential_V_SHE"),
            "melting_point_C": tm_c,
            "electronegativity_pauling": entry.get("electronegativity_pauling")
            if "electronegativity_pauling" in entry
            else merged.get("electronegativity_pauling"),
            "work_function_eV": merged.get("work_function_eV"),
            "oxide_type": entry.get("oxide_type", merged.get("oxide_type")),
            "oxide_stability": entry.get("oxide_stability", merged.get("oxide_stability")),
            "passivation_tendency": entry.get("passivation_tendency", merged.get("passivation_tendency")),
            "pourbaix_stable_pH_min": ph_min,
            "pourbaix_stable_pH_max": ph_max,
            "crystal_structure": entry.get("crystal_structure", merged.get("crystal_structure")),
            "thermal_conductivity_W_mK": entry.get("thermal_conductivity_W_mK"),
            "electrical_resistivity_ohm_cm": entry.get("electrical_resistivity_ohm_cm"),
            "common_corrosion_products": entry.get("common_corrosion_products"),
        }

    for sym, entry in elements.items():
        if entry.get("density_g_cm3") is not None:
            result[sym] = _to_step5_entry(sym, entry)

    for sym, entry in compounds.items():
        if entry.get("density_g_cm3") is not None:
            base = elements.get(entry.get("base_element", ""))
            result[sym] = _to_step5_entry(sym, entry, base=base)

    return result


def build_element_physics(db_path: Path = PHYSICS_DB_PATH) -> dict[str, dict[str, Any]]:
    """組裝 Step 7 使用的 ELEMENT_PHYSICS（僅含具備熱力學／物性參數的純元素）。"""
    raw = load_physics_db(db_path)
    result: dict[str, dict[str, Any]] = {}

    for sym, entry in raw.get("elements", {}).items():
        if "atomic_number" not in entry:
            continue
        item: dict[str, Any] = {
            "Z": entry["atomic_number"],
            "mass": entry["molar_mass_g_mol"],
            "en": entry["electronegativity_pauling"],
            "wf_eV": entry["work_function_eV"],
            "E0_V": entry["standard_potential_V_SHE"],
            "Tm_K": entry["melting_point_K"],
            "Vm_cm3": entry["molar_volume_cm3_mol"],
            "valence": entry["valence"],
            "oxide_formula": entry["oxide_formula"],
            "oxide_stoich": tuple(entry["oxide_stoichiometry"]),
            "passiv_pH": tuple(entry["passivation_pH"]),
            "transpassive_E": entry["transpassive_potential_V_SHE"],
            "ion_acid": entry["ion_acid"],
            "ion_alk": entry["ion_alkaline"],
            "oxide_phase": entry["oxide_phase"],
        }
        if "transpassive_phase" in entry:
            item["transpassive_phase"] = entry["transpassive_phase"]
        result[sym] = item

    return result


def build_compound_overrides(db_path: Path = PHYSICS_DB_PATH) -> dict[str, dict[str, Any]]:
    """組裝 Step 7 使用的 COMPOUND_OVERRIDES。"""
    raw = load_physics_db(db_path)
    result: dict[str, dict[str, Any]] = {}

    for sym, entry in raw.get("compounds", {}).items():
        has_override = any(
            k in entry
            for k in ("work_function_eV", "standard_potential_V_SHE", "oxide_phase", "passivation_pH")
        )
        if not has_override or "base_element" not in entry:
            continue
        item: dict[str, Any] = {
            "base_element": entry["base_element"],
            "mp_formula": entry.get("mp_formula", sym),
        }
        if "work_function_eV" in entry:
            item["wf_eV"] = entry["work_function_eV"]
        if "standard_potential_V_SHE" in entry:
            item["E0_V"] = entry["standard_potential_V_SHE"]
        if "oxide_phase" in entry:
            item["oxide_phase"] = entry["oxide_phase"]
        if "passivation_pH" in entry:
            item["passiv_pH"] = tuple(entry["passivation_pH"])
        result[sym] = item

    return result


def build_text_recognition_tables(
    db_path: Path = PHYSICS_DB_PATH,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """組裝 Step 5 文本材料辨識用的 (base_symbols, name_to_symbol, material_aliases)。"""
    raw = load_physics_db(db_path)
    base_symbols: list[str] = []
    name_to_symbol: dict[str, str] = {}
    material_aliases: dict[str, str] = {}

    for sym, entry in raw.get("elements", {}).items():
        if entry.get("is_base_symbol"):
            base_symbols.append(sym)
        for name in entry.get("name_aliases", []):
            name_to_symbol[name.lower()] = sym
        if entry.get("aliases"):
            material_aliases[sym.lower()] = sym
            for alias in entry["aliases"]:
                material_aliases[alias.lower()] = sym

    for sym, entry in raw.get("compounds", {}).items():
        for name in entry.get("name_aliases", []):
            name_to_symbol[name.lower()] = sym
        for alias in entry.get("aliases", []):
            material_aliases[alias.lower()] = sym

    return base_symbols, name_to_symbol, material_aliases


def validate_databases(
    metals_to_run: list[str] | None = None,
    metals_db_path: Path = METALS_DB_PATH,
    physics_db_path: Path = PHYSICS_DB_PATH,
) -> tuple[list[str], list[str]]:
    """
    驗證 metals_db.json 與 materials_physics.json 的完整性與一致性。
    回傳 (errors, warnings)：
      - errors: 阻擋執行的結構錯誤（exit code = 1）
      - warnings: 不阻擋 Step 1~4 但影響 Step 5~7 物性計算的缺漏提醒（exit code = 0）
    """
    errors: list[str] = []
    warnings: list[str] = []

    metals_raw: dict[str, Any] = {}
    try:
        with metals_db_path.open("r", encoding="utf-8") as f:
            metals_raw = json.load(f)
    except Exception as e:
        errors.append(f"無法讀取 {metals_db_path.name}: {e}")

    physics_raw: dict[str, Any] = {}
    try:
        with physics_db_path.open("r", encoding="utf-8") as f:
            physics_raw = json.load(f)
    except Exception as e:
        errors.append(f"無法讀取 {physics_db_path.name}: {e}")

    if errors:
        return errors, warnings

    metals = metals_raw.get("metals")
    if not isinstance(metals, dict) or not metals:
        errors.append("metals_db.json 缺少有效的 'metals' 字典")
        return errors, warnings

    elements = physics_raw.get("elements", {})
    compounds = physics_raw.get("compounds", {})

    # 1. 檢查 METALS_TO_RUN 中的每個金屬是否都存在於 metals_db.json
    if metals_to_run:
        lookup = {k.lower(): k for k in metals}
        for sym in metals_to_run:
            if sym.lower() not in lookup:
                errors.append(f"METALS_TO_RUN 中的 '{sym}' 未定義於 metals_db.json")

    # 2. 檢查每個 compound 的 base_element 是否存在於 elements
    for csym, centry in compounds.items():
        base = centry.get("base_element")
        if not base:
            errors.append(f"materials_physics.json 化合物 '{csym}' 缺少 'base_element'")
        elif base not in elements:
            errors.append(
                f"materials_physics.json 化合物 '{csym}' 的 base_element='{base}' 不存在於 elements"
            )

    # 3. 檢查 metals_db.json 每個條目的必填欄位與對應物性完整度
    for sym, entry in metals.items():
        if not entry.get("target_metal"):
            errors.append(f"metals_db.json 金屬 '{sym}' 缺少必填欄位 'target_metal'")
        if "compounds" not in entry or not isinstance(entry["compounds"], list):
            errors.append(f"metals_db.json 金屬 '{sym}' 缺少必填欄位 'compounds' (list)")

        if "ecorr_valid_range_v_she" not in entry:
            warnings.append(f"'{sym}' 未設定 ecorr_valid_range_v_she（將沿用 default 區間）")

        if sym in elements:
            el = elements[sym]
            if el.get("density_g_cm3") is None:
                warnings.append(f"'{sym}' 在 materials_physics.json 缺少 density_g_cm3（Step 5 無法估算 CR）")
            if "atomic_number" not in el:
                warnings.append(f"'{sym}' 在 materials_physics.json 缺少熱力學常數（Step 7 無法計算 Pourbaix）")
        elif sym in compounds:
            cp = compounds[sym]
            if cp.get("density_g_cm3") is None:
                warnings.append(f"'{sym}' 在 materials_physics.json 缺少 density_g_cm3（Step 5 無法估算 CR）")
            base = cp.get("base_element")
            if base in elements and "atomic_number" not in elements[base]:
                warnings.append(f"'{sym}' 的主元素 '{base}' 缺少熱力學常數（Step 7 無法計算 Pourbaix）")
        else:
            warnings.append(f"'{sym}' 尚未登錄於 materials_physics.json（僅能執行 Step 1~4，Step 5~7 將缺物性特徵）")

    return errors, warnings
