# AIS-KB — Production-Grade AI Safety Mechanistic Interpretability Knowledge Base

Built from reference analysis of `adeft-master`, `eGOT-master`, `eidos-master`, `emmaa-master`, `gilda-master`, `indra-master`, `indra_cogex-main`, `indra_db-master`, `indra_world-master`, `Mechanist-main`.

## Production Features

- **Ontology**: ~250 concept AI Safety & Mechanistic Interpretability ontology with 7 branches (neural_component, model_intervention, adversarial_stimulus, model_behavior, mechanistic_assay, model_family, benchmark_dataset, evaluation_metric).
- **Grounder**: Full Gilda scoring (s_status, s_case, s_dash) + Adeft-style acronym disambiguation (`disambiguator.py`).
- **DB**: DuckDB production store with indexes, delta assembly, contradiction audit, belief scoring with DAG topological sort and refutation-aware discounting.
- **Exports**: GraphML (Cytoscape), JSONL, SQLite-compatible SQL.
- **CLI**: `ais-kb` command (`cli.py`).
- **API**: FastAPI server (`service/api.py`) exposing `/ingest`, `/query`, `/phenomenon-card`, `/circuits`, `/compare-models`, `/contradictions`.

## Quick Start

```bash
pip install -e .
ais-kb ingest --paper-id arXiv:2406.XXXXX
ais-kb query --min-belief 0.8
```

## Architecture

- `ontology/` — Ontology YAML, `graph.py`, `grounder.py`, `disambiguator.py`
- `core/` — Schema models (`AISEntity`, `MechanisticClaim`, `ScenarioContext`)
- `assembly/` — `preassembler.py`, `refinement_index.py`
- `belief/` — `scorer.py` (noisy-OR + DAG sorting + contradiction discount)
- `extraction/` — `llm_schema.py`, `prompts.py`, `processor.py`, `arxiv_loader.py`, `llm_client.py`
- `analysis/` — `circuits.py`, `comparison.py`, `phenomenon_card.py`
- `storage/` — `db.py` (DuckDB production store)
- `service/` — `api.py`
- `cli.py` — CLI entry point
