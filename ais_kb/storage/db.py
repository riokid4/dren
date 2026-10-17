import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any
from ais_kb.core.schema import MechanisticClaim
from ais_kb.ontology.graph import AISOntologyGraph
from ais_kb.assembly.preassembler import ClaimAssembler
from ais_kb.belief.scorer import EvidenceScorer


class KnowledgeBaseStore:
    """Zero-config SQLite 4-table hash database for incremental storage and querying
    of assembled AI Safety claims, hierarchies, and scenario-tweak contradictions."""

    def __init__(self, db_path: str | Path = ":memory:", ontology: Optional[AISOntologyGraph] = None):
        self.db_path = str(db_path)
        self.ontology = ontology or AISOntologyGraph()
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS raw_claims (
                raw_id INTEGER PRIMARY KEY AUTOINCREMENT,
                mk_hash INTEGER NOT NULL,
                paper_id TEXT NOT NULL,
                claim_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS assembled_claims (
                mk_hash INTEGER PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                effective_outcome INTEGER NOT NULL,
                model_family TEXT,
                param_scale TEXT,
                threat_model TEXT,
                benchmark TEXT,
                belief_score REAL NOT NULL,
                replication_count INTEGER NOT NULL,
                claim_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS raw_to_assembled (
                raw_id INTEGER NOT NULL,
                mk_hash INTEGER NOT NULL,
                PRIMARY KEY (raw_id, mk_hash)
            );

            CREATE TABLE IF NOT EXISTS claim_links (
                source_mk_hash INTEGER NOT NULL,
                target_mk_hash INTEGER NOT NULL,
                link_type TEXT NOT NULL,
                delta_scenario_json TEXT,
                PRIMARY KEY (source_mk_hash, target_mk_hash, link_type)
            );
            """
        )
        self.conn.commit()

    def ingest_claims(self, new_claims: List[MechanisticClaim]) -> Dict[str, Any]:
        """Stores raw claims, loads existing assembled claims, runs full preassembly
        (deduplication + hierarchy + contradiction/tweak audit + belief scoring),
        and persists the updated Knowledge Base."""
        cur = self.conn.cursor()

        # 1. Load existing assembled claims from DB so incremental ingestion merges cleanly
        assembler = ClaimAssembler(self.ontology)
        for row in cur.execute("SELECT claim_json FROM assembled_claims"):
            existing_claim = MechanisticClaim.model_validate_json(row["claim_json"])
            assembler.claims[existing_claim.get_hash()] = existing_claim

        # 2. Insert new raw claims into `raw_claims` and `raw_to_assembled`
        for claim in new_claims:
            h = claim.get_hash()
            paper_id = claim.evidence[0].paper_id if claim.evidence else "unknown"
            cur.execute(
                "INSERT INTO raw_claims (mk_hash, paper_id, claim_json) VALUES (?, ?, ?)",
                (h, paper_id, claim.model_dump_json()),
            )
            raw_id = cur.lastrowid
            cur.execute(
                "INSERT OR IGNORE INTO raw_to_assembled (raw_id, mk_hash) VALUES (?, ?)",
                (raw_id, h),
            )

        # 3. Combine replications, build refinement DAG, audit tweaks, and compute belief
        assembler.combine_replications(new_claims)
        dag_children = assembler.build_refinement_dag()
        audits = assembler.audit_contradictions_and_tweaks()
        scorer = EvidenceScorer()
        scores = scorer.score_all_claims(assembler.claims, dag_children)

        # 4. Refresh `assembled_claims` and `claim_links`
        cur.execute("DELETE FROM claim_links")
        for h, claim in assembler.claims.items():
            ctx = claim.scenario_context
            cur.execute(
                """
                INSERT OR REPLACE INTO assembled_claims (
                    mk_hash, source_id, target_id, effective_outcome,
                    model_family, param_scale, threat_model, benchmark,
                    belief_score, replication_count, claim_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    h,
                    claim.source_entity.canonical_id,
                    claim.target_entity.canonical_id,
                    claim.effective_outcome(),
                    ctx.model_family,
                    ctx.param_scale,
                    ctx.threat_model,
                    ctx.benchmark,
                    claim.belief_score or 0.0,
                    len(claim.evidence),
                    claim.model_dump_json(),
                ),
            )

            # Store hierarchy support links (`child` supports `parent`)
            for parent_hash in claim.supports:
                cur.execute(
                    """
                    INSERT OR IGNORE INTO claim_links
                    (source_mk_hash, target_mk_hash, link_type, delta_scenario_json)
                    VALUES (?, ?, 'supports', NULL)
                    """,
                    (h, parent_hash),
                )

        # Store contradiction & scenario-tweak failure links
        for rec in audits:
            cur.execute(
                """
                INSERT OR IGNORE INTO claim_links
                (source_mk_hash, target_mk_hash, link_type, delta_scenario_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    rec["claim_1_hash"],
                    rec["claim_2_hash"],
                    rec["type"],
                    json.dumps(rec["delta_scenario"]),
                ),
            )

        self.conn.commit()
        return {
            "total_assembled_claims": len(assembler.claims),
            "refinement_parents": len(dag_children),
            "conflicts_and_tweaks": len(audits),
            "scores": scores,
        }

    def get_claim(self, mk_hash: int) -> Optional[MechanisticClaim]:
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT claim_json FROM assembled_claims WHERE mk_hash = ?", (mk_hash,)
        ).fetchone()
        return MechanisticClaim.model_validate_json(row["claim_json"]) if row else None

    def query_claims(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        model_family: Optional[str] = None,
        min_belief: float = 0.0,
    ) -> List[MechanisticClaim]:
        query = "SELECT claim_json FROM assembled_claims WHERE belief_score >= ?"
        params: List[Any] = [min_belief]
        if source_id:
            query += " AND source_id = ?"
            params.append(source_id)
        if target_id:
            query += " AND target_id = ?"
            params.append(target_id)
        if model_family:
            query += " AND model_family = ?"
            params.append(model_family)
        query += " ORDER BY belief_score DESC"

        cur = self.conn.cursor()
        return [
            MechanisticClaim.model_validate_json(row["claim_json"])
            for row in cur.execute(query, params)
        ]

    def get_contradictions_and_tweaks(self) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        rows = cur.execute(
            """
            SELECT source_mk_hash, target_mk_hash, link_type, delta_scenario_json
            FROM claim_links
            WHERE link_type IN ('direct_contradiction', 'scenario_tweak_failure')
            """
        ).fetchall()
        return [
            {
                "claim_1": self.get_claim(r["source_mk_hash"]),
                "claim_2": self.get_claim(r["target_mk_hash"]),
                "type": r["link_type"],
                "delta_scenario": json.loads(r["delta_scenario_json"]) if r["delta_scenario_json"] else {},
            }
            for r in rows
        ]
