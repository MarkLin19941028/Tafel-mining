"""
Step 5 — Shared domain schema, Chemical Species Registry, PaperEnvironment,
RawEchemRecord, and shared electrochemical value normalizers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, fields
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any, Mapping

from step5_constants import convert_to_she

DEFAULT_SPECIES_REGISTRY_PATH = Path(__file__).resolve().parent / "chemical_species.json"
TAFEL_HARD_UPPER_MV_DEC: float = 1000.0


@dataclass(frozen=True)
class ChemicalSpeciesRegistry:
    """從單一 JSON 來源衍生的化學物種表（正則、別名正規化、角色與緩蝕劑查詢）。"""

    role_fields: dict[str, str]
    electrolyte_species: tuple[str, ...]
    species_categories: dict[str, dict[str, str]]
    canonical_map: dict[str, str]
    inhibitor_names: tuple[str, ...]
    electrolyte_species_regex: str
    inhibitor_name_pattern: re.Pattern
    _categories_lower: dict[str, tuple[str, dict[str, str]]] = field(repr=False)

    def normalise(self, raw: str) -> str:
        s = re.sub(r"\s+", " ", (raw or "").strip())
        low = s.lower()
        if low in self.canonical_map:
            return self.canonical_map[low]
        if low in self._categories_lower:
            return self._categories_lower[low][0]
        return s

    def lookup_category(self, species: str) -> dict[str, str] | None:
        if not species:
            return None
        if species in self.species_categories:
            return self.species_categories[species]
        hit = self._categories_lower.get(species.lower())
        return hit[1] if hit else None


def _build_alternation(names: tuple[str, ...]) -> str:
    return "|".join(re.escape(s).replace(r"\ ", r"\s+") for s in names)


@lru_cache(maxsize=4)
def _load_registry_cached(resolved_path: str) -> ChemicalSpeciesRegistry:
    raw = json.loads(Path(resolved_path).read_text(encoding="utf-8"))
    role_fields: dict[str, str] = dict(raw["role_fields"])

    electrolyte_names: set[str] = set()
    species_categories: dict[str, dict[str, str]] = {}
    canonical_map: dict[str, str] = {}
    inhibitor_names: set[str] = set()

    for entry in raw["species"]:
        canonical = str(entry["canonical"]).strip()
        role = str(entry["role"]).strip()
        if role not in role_fields:
            raise ValueError(f"Unknown chemical species role '{role}' for '{canonical}'")
        cat_info = {"role": role, "field": role_fields[role]}
        match_in_regex = bool(entry.get("match_in_electrolyte_regex", True))
        all_names = [canonical, *[str(a).strip() for a in entry.get("aliases", ()) if str(a).strip()]]

        for name in all_names:
            species_categories[name] = cat_info
            canonical_map[name.lower()] = canonical
            if match_in_regex:
                electrolyte_names.add(name)
            if role == "inhibitor":
                inhibitor_names.add(name)

    sorted_electrolytes = tuple(sorted(electrolyte_names, key=len, reverse=True))
    sorted_inhibitors = tuple(sorted(inhibitor_names, key=len, reverse=True))
    electrolyte_regex = _build_alternation(sorted_electrolytes) if sorted_electrolytes else r"(?!)"
    inhibitor_pattern = (
        re.compile(
            r"(?<![A-Za-z0-9])(?P<inh>" + _build_alternation(sorted_inhibitors) + r")(?![A-Za-z0-9])",
            re.I,
        )
        if sorted_inhibitors
        else re.compile(r"(?!)")
    )
    categories_lower = {k.lower(): (k, v) for k, v in species_categories.items()}

    return ChemicalSpeciesRegistry(
        role_fields=role_fields,
        electrolyte_species=sorted_electrolytes,
        species_categories=species_categories,
        canonical_map=canonical_map,
        inhibitor_names=sorted_inhibitors,
        electrolyte_species_regex=electrolyte_regex,
        inhibitor_name_pattern=inhibitor_pattern,
        _categories_lower=categories_lower,
    )


def load_chemical_species_registry(
    path: str | Path | None = None,
) -> ChemicalSpeciesRegistry:
    """載入並快取化學物種登錄表（預設讀取專案根目錄 chemical_species.json）。"""
    resolved = Path(path).resolve() if path is not None else DEFAULT_SPECIES_REGISTRY_PATH
    return _load_registry_cached(str(resolved))


@dataclass(frozen=True)
class PaperEnvironment:
    """單篇論文共用的實驗環境中繼資料（供內文與表格擷取共享，不再依賴 text_records[0]）。"""

    electrolytes: list[dict] = field(default_factory=list)
    ph_counts: Counter = field(default_factory=Counter)
    temp_counts: Counter = field(default_factory=Counter)
    ref_counts: Counter = field(default_factory=Counter)
    refs: list[str] = field(default_factory=list)
    reactions: list[str] = field(default_factory=list)
    table_captions: dict[int, str] = field(default_factory=dict)
    metal_form: str | None = None
    surface_preparation: str | None = None
    measurement_methods: str | None = None
    scan_rates_raw: str | None = None
    immersion_times_raw: str | None = None
    deaerated: bool = False
    aerated: bool = False
    o2_saturated: bool = False

    @classmethod
    def from_mapping(
        cls,
        raw: PaperEnvironment | Mapping[str, Any] | None,
    ) -> PaperEnvironment:
        if isinstance(raw, cls):
            return raw
        if not raw:
            return cls()
        return cls(
            electrolytes=list(raw.get("electrolytes") or []),
            ph_counts=Counter(raw.get("ph_counts") or {}),
            temp_counts=Counter(raw.get("temp_counts") or {}),
            ref_counts=Counter(raw.get("ref_counts") or {}),
            refs=list(raw.get("refs") or []),
            reactions=list(raw.get("reactions") or []),
            table_captions=dict(raw.get("table_captions") or {}),
            metal_form=raw.get("metal_form"),
            surface_preparation=raw.get("surface_preparation"),
            measurement_methods=raw.get("measurement_methods"),
            scan_rates_raw=raw.get("scan_rates_raw"),
            immersion_times_raw=raw.get("immersion_times_raw"),
            deaerated=bool(raw.get("deaerated", False)),
            aerated=bool(raw.get("aerated", False)),
            o2_saturated=bool(raw.get("o2_saturated", False)),
        )


@dataclass
class RawEchemRecord:
    """內文與表格擷取共用的原始電化學記錄結構（to_dict() 與現有 CSV schema 完全一致）。"""

    source_id: str
    sample_name: str | None = None
    materials_raw: str = ""
    materials_normalised: str = ""
    material_mention_counts: str = ""
    paper_primary_material: str | None = None
    target_mentions: int = 0
    material_id_assigned: str | None = None
    material_assignment_method: str | None = None
    is_target_metal: bool = False
    metal_form: str | None = None
    surface_preparation: str | None = None
    record_electrolytes: list[dict] = field(default_factory=list)
    electrolytes_raw: str = ""
    record_ph: float | None = None
    ph_values: str | None = None
    record_temp: float | None = None
    temperature_C_values: str | None = None
    solution_id_assigned: str | None = None
    reference_electrodes: str | None = None
    deaerated: bool = False
    aerated: bool = False
    o2_saturated: bool = False
    measurement_methods: str | None = None
    scan_rates_raw: str | None = None
    reactions: str | None = None
    immersion_times_raw: str | None = None
    extraction_method: str = "regex"
    context: str = ""
    Ecorr_V_original: float | None = None
    Ecorr_unit_original: str | None = None
    Ecorr_raw: str | None = None
    Ecorr_V_SHE: float | None = None
    icorr_uA_cm2: float | None = None
    icorr_raw: str | None = None
    icorr_unit_original: str | None = None
    log_icorr: float | None = None
    beta_a_mV_dec: float | None = None
    beta_c_mV_dec: float | None = None
    tafel_slopes_general_mV_dec: str | None = None
    CR_reported_value: float | None = None
    CR_reported_unit: str | None = None
    Rp_value: float | None = None
    Rp_unit: str | None = None
    extra_quality_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def normalize_tafel_slope(
    value: float | None,
    unit: str,
    *,
    tafel_min: float,
    tafel_max: float,
    label: str = "β",
    allow_above_max: bool = True,
    note_mv_without_dec: bool = False,
) -> tuple[float | None, list[str]]:
    """
    將 Tafel 斜率統一換算為正數 mV/dec 並套用合理上下限檢查（內文與表格共用）。
    回傳 (normalized_mV_dec_or_None, quality_notes)。
    """
    if value is None:
        return None, []
    notes: list[str] = []
    v = abs(float(value))
    u = (unit or "").strip()
    u_low = u.lower()
    if u and "v" in u_low and "mv" not in u_low:
        v *= 1000.0
    if note_mv_without_dec and u and "dec" not in u_low and "mv" in u_low:
        notes.append(f"{label} unit '({u})' interpreted as mV/dec")

    upper_bound = TAFEL_HARD_UPPER_MV_DEC if allow_above_max else float(tafel_max)
    if not (tafel_min <= v <= upper_bound):
        return None, notes
    if v > tafel_max:
        notes.append(f"{label}={v:.1f} exceeds TAFEL_MAX={tafel_max}")
    return v, notes


def resolve_ecorr_she(
    ecorr_v: float | None,
    ref_electrode: str | None,
    ph: float | None,
) -> float | None:
    """將 Ecorr (V) 依參考電極與該筆 pH 換算至 SHE（內文與表格共用）。"""
    if ecorr_v is None or not ref_electrode:
        return None
    return convert_to_she(ecorr_v, ref_electrode, ph)
