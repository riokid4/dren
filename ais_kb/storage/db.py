import json
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    import duckdb
except ImportError:
    duckdb = None

from ais_kb.core.schema import MechanisticClaim
from ais_kb.ontology.graph import AISOntologyGraph
from ais_kb.assembly.preassembler import ClaimAssembler
from ais_kb.belief.scorer import EvidenceScorer

class ProductionKnowledgeStore:
    def __init__(self, db_path: str | Path = ":memory:", ontology: Optional[AISOntologyGraph] = None):
        self.db_path = str(db_path)
        self.ontology = ontology or AISOntologyGraph()
        if duckdb is None:
            raise ImportError("duckdb is required for production store")
        self.conn = duckdb.connect(str(db_path))
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS seq_raw_id START 1;
        CREATE TABLE IF NOT EXISTS raw_claims (raw_id INTEGER PRIMARY KEY DEFAULT nextval('seq_raw_id'), mk_hash INTEGER NOT NULL, paper_id TEXT NOT NULL, claim_json TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_raw_mk ON raw_claims(mk_hash);
        CREATE INDEX IF NOT EXISTS idx_raw_paper ON raw_claims(paper_id);
        CREATE TABLE IF NOT EXISTS assembled_claims (mk_hash INTEGER PRIMARY KEY, source_id TEXT NOT NULL, target_id TEXT NOT NULL, effective_outcome INTEGER NOT NULL, model_family TEXT, param_scale TEXT, threat_model TEXT, benchmark TEXT, belief_score REAL NOT NULL, replication_count INTEGER NOT NULL, claim_json TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_assembled_source ON assembled_claims(source_id);
        CREATE INDEX IF NOT EXISTS idx_assembled_target ON assembled_claims(target_id);
        CREATE INDEX IF NOT EXISTS idx_assembled_model ON assembled_claims(model_family);
        CREATE INDEX IF NOT EXISTS idx_assembled_belief ON assembled_claims(belief_score);
        CREATE TABLE IF NOT EXISTS raw_to_assembled (raw_id INTEGER NOT NULL, mk_hash INTEGER NOT NULL, PRIMARY KEY (raw_id, mk_hash));
        CREATE INDEX IF NOT EXISTS idx_link_raw ON raw_to_assembled(raw_id);
        CREATE INDEX IF NOT EXISTS idx_link_mk ON raw_to_assembled(mk_hash);
        CREATE TABLE IF NOT EXISTS claim_links (source_mk_hash INTEGER NOT NULL, target_mk_hash INTEGER NOT NULL, link_type TEXT NOT NULL, delta_scenario_json TEXT, PRIMARY KEY (source_mk_hash, target_mk_hash, link_type));
        CREATE INDEX IF NOT EXISTS idx_links_source ON claim_links(source_mk_hash);
        CREATE INDEX IF NOT EXISTS idx_links_target ON claim_links(target_mk_hash);
        CREATE INDEX IF NOT EXISTS idx_links_type ON claim_links(link_type);
        """)
        self.conn.commit()

    def ingest_claims(self, new_claims: List[MechanisticClaim]) -> Dict[str, Any]:
        assembler = ClaimAssembler(self.ontology)
        rows = self.conn.execute("SELECT claim_json FROM assembled_claims").fetchall()
        for row in rows:
            claim = MechanisticClaim.model_validate_json(row[0])
            assembler.claims[claim.get_hash()] = claim
        for claim in new_claims:
            h = claim.get_hash()
            paper_id = claim.evidence[0].paper_id if claim.evidence else "unknown"
            self.conn.execute("INSERT OR IGNORE INTO raw_claims (mk_hash, paper_id, claim_json) VALUES (?, ?, ?)", (h, paper_id, claim.model_dump_json()))
            self.conn.execute("INSERT OR IGNORE INTO raw_to_assembled (raw_id, mk_hash) VALUES (?, ?)", (self.conn.execute("SELECT last_insert_rowid()").fetchone()[0], h))
        assembler.combine_replications(new_claims)
        dag_children = assembler.build_refinement_dag()
        audits = assembler.audit_contradictions_and_tweaks()
        scorer = EvidenceScorer()
        scores = scorer.score_all_claims(assembler.claims, dag_children)
        self.conn.execute("DELETE FROM claim_links")
        for h, claim in assembler.claims.items():
            ctx = claim.scenario_context
            self.conn.execute("INSERT OR REPLACE INTO assembled_claims (mk_hash, source_id, target_id, effective_outcome, model_family, param_scale, threat_model, benchmark, belief_score, replication_count, claim_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (h, claim.source_entity.canonical_id, claim.target_entity.canonical_id, claim.effective_outcome(), ctx.model_family, ctx.param_scale, ctx.threat_model, ctx.benchmark, claim.belief_score or 0.0, len(claim.evidence), claim.model_dump_json()))
            for parent_hash in claim.supports:
                self.conn.execute("INSERT OR IGNORE INTO claim_links (source_mk_hash, target_mk_hash, link_type, delta_scenario_json) VALUES (?, ?, 'supports', NULL)", (h, parent_hash))
            for rec in audits:
                self.conn.execute("INSERT OR IGNORE INTO claim_links (source_mk_hash, target_mk_hash, link_type, delta_scenario_json) VALUES (?, ?, ?, ?)", (rec["claim_1_hash"], rec["claim_2_hash"], rec["type"], json.dumps(rec["delta_scenario"])))
        self.conn.commit()
        return {"total_assembled_claims": len(assembler.claims), "refinement_parents": len(dag_children), "conflicts_and_tweaks": len(audits), "scores": scores}

    def get_claim(self, mk_hash: int) -> Optional[MechanisticClaim]:
        row = self.conn.execute("SELECT claim_json FROM assembled_claims WHERE mk_hash = ?", (mk_hash,)).fetchone()
        return MechanisticClaim.model_validate_json(row[0]) if row else None

    def query_claims(self, source_id: Optional[str] = None, target_id: Optional[str] = None, model_family: Optional[str] = None, min_belief: float = 0.0) -> List[MechanisticClaim]:
        query = "SELECT claim_json FROM assembled_claims WHERE belief_score >= ?"
        params: List[Any] = [min_belief]
        if source_id:
            query += " AND source_id = ?"; params.append(source_id)
        if target_id:
            query += " AND target_id = ?"; params.append(target_id)
        if model_family:
            query += " AND model_family = ?"; params.append(model_family)
        query += " ORDER BY belief_score DESC"
        return [MechanisticClaim.model_validate_json(r[0]) for r in self.conn.execute(query, params).fetchall()]

    def get_contradictions_and_tweaks(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT source_mk_hash, target_mk_hash, link_type, delta_scenario_json FROM claim_links WHERE link_type IN ('direct_contradiction', 'scenario_tweak_failure')").fetchall()
        return [{"claim_1": self.get_claim(r[0]), "claim_2": self.get_claim(r[1]), "type": r[2], "delta_scenario": json.loads(r[3]) if r[3] else {}} for r in rows]

    def export_graphml(self, path: str | Path):
        nodes = self.conn.execute("SELECT mk_hash, source_id, target_id FROM assembled_claims").fetchall()
        edges = self.conn.execute("SELECT source_mk_hash, target_mk_hash, link_type FROM claim_links").fetchall()
        graphml = '<?xml version="1.0" encoding="UTF-8"?><graphml xmlns="http://graphml.graphdrawing.org/xmlns"><graph edgedefault="directed">'
        for n in nodes:
            graphml += f'<node id="{n[0]}"/><data key="source">{n[1]}</data><data key="target">{n[2]}</data></node>'
        for e in edges:
            graphml += f'<edge source="{e[0]}" target="{e[1]}" label="{e[2]}"/>'
        graphml += '</graph></graphml>'
        Path(path).write_text(graphml, encoding="utf-8")
        return path

    def export_jsonl(self, path: str | Path):
        lines = []
        for row in self.conn.execute("SELECT claim_json FROM assembled_claims").fetchall():
            lines.append(row[0])
        Path(path).write_text("\n".join(lines), encoding="utf-8")
        return path
