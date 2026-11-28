# Tafel Mining

An automated literature discovery, retrieval, electrochemical data extraction, and thermodynamic enrichment pipeline for Tafel polarization parameters across metals and metal nitrides.

## Language

**Pipeline Settings**:
Cross-metal defaults, data source switches, physical validity bounds, and merged output locations shared across a batch of metals.
_Avoid_: global config, env vars

**Metal Run Context**:
The immutable specification for running discovery, retrieval, and extraction for a single target metal and chosen search modes, including its resolved artifact locations.
_Avoid_: active metal, config globals

**Paper Environment**:
The paper-level experimental conditions (electrolytes, pH, temperature, reference electrodes, metal form, surface preparation, aeration, measurement methods, and table captions) shared by text and table extraction for a single article.
_Avoid_: `paper_env`, `text_records[0]` metadata

**Chemical Species Registry**:
The single canonical catalog of electrolyte, oxidizer, acid, base, chelating agent, inhibitor, and organic solvent species with their aliases and chemical roles.
_Avoid_: species lists, chemical dicts
