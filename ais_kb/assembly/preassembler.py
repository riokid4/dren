from typing import List, Dict, Any
from collections import defaultdict
from ais_kb.core.schema import MechanisticClaim
from ais_kb.ontology.graph import AISOntologyGraph
from ais_kb.assembly.refinement_index import RefinementIndex


class ClaimAssembler:
    """Port of INDRA Preassembler: handles replication deduplication, sub-quadratic
    hierarchical refinement DAG construction, and scenario-tweak contradiction auditing."""

    def __init__(self, ontology: AISOntologyGraph):
        self.ontology = ontology
        self.claims: Dict[int, MechanisticClaim] = {}

    def combine_replications(self, new_claims: List[MechanisticClaim]) -> Dict[int, MechanisticClaim]:
        """Merges identical claims across papers and deduplicates their evidence records."""
        for claim in new_claims:
            h = claim.get_hash()
            if h in self.claims:
                existing = self.claims[h]
                seen_ev = {ev.matches_key() for ev in existing.evidence}
                for ev in claim.evidence:
                    if ev.matches_key() not in seen_ev:
                        existing.evidence.append(ev)
                        seen_ev.add(ev.matches_key())
            else:
                self.claims[h] = claim.model_copy(deep=True)
        return self.claims

    def build_refinement_dag(self) -> Dict[int, List[int]]:
        """Uses RefinementIndex set intersection to link specific claims (`child`)
        to general claims (`parent`) in sub-quadratic time.
        Returns `dag_children`: mapping of `parent_hash -> list[child_hash]`."""
        ref_idx = RefinementIndex(self.ontology)
        for h, claim in self.claims.items():
            claim.supports.clear()
            claim.supported_by.clear()
            ref_idx.add_claim(
                h,
                claim.source_entity.canonical_id,
                claim.target_entity.canonical_id,
            )

        dag_children: Dict[int, List[int]] = defaultdict(list)

        for child_hash, child_claim in self.claims.items():
            candidate_parent_hashes = ref_idx.get_candidate_parents(
                child_hash,
                child_claim.source_entity.canonical_id,
                child_claim.target_entity.canonical_id,
            )
            for parent_hash in candidate_parent_hashes:
                parent_claim = self.claims[parent_hash]
                if child_claim.refinement_of(parent_claim, self.ontology):
                    child_claim.supports.append(parent_hash)
                    parent_claim.supported_by.append(child_hash)
                    dag_children[parent_hash].append(child_hash)

        return dict(dag_children)

    def audit_contradictions_and_tweaks(self) -> List[Dict[str, Any]]:
        """Finds pairs of claims that test the same or refining (source -> target)
        relationship but report conflicting outcomes (e.g. Holds vs. Fails/Opposite).
        Partitions conflicts into:
          - 'direct_contradiction' (Delta D is empty: same scenario)
          - 'scenario_tweak_failure' (Delta D is non-empty: robustness failure when D is tweaked)
        """
        ref_idx = RefinementIndex(self.ontology)
        for h, c in self.claims.items():
            ref_idx.add_claim(h, c.source_entity.canonical_id, c.target_entity.canonical_id)
            c.contradicted_by.clear()

        audit_records: List[Dict[str, Any]] = []
        seen_pairs = set()

        for h1, c1 in self.claims.items():
            candidates = ref_idx.get_candidate_parents(
                h1, c1.source_entity.canonical_id, c1.target_entity.canonical_id
            )
            same_concepts = (
                ref_idx.index.get(("subj", c1.source_entity.canonical_id), set())
                & ref_idx.index.get(("obj", c1.target_entity.canonical_id), set())
            ) - {h1}

            for h2 in candidates | same_concepts:
                pair_id = tuple(sorted((h1, h2)))
                if pair_id in seen_pairs:
                    continue

                c2 = self.claims[h2]
                if c1.effective_outcome() == c2.effective_outcome():
                    continue

                seen_pairs.add(pair_id)
                delta_d = c1.scenario_context.diff_scenario(c2.scenario_context)
                conflict_type = (
                    "direct_contradiction" if len(delta_d) == 0 else "scenario_tweak_failure"
                )

                record = {
                    "type": conflict_type,
                    "claim_1_hash": h1,
                    "claim_2_hash": h2,
                    "claim_1_outcome": c1.effective_outcome(),
                    "claim_2_outcome": c2.effective_outcome(),
                    "delta_scenario": delta_d,
                }
                audit_records.append(record)

                c1.contradicted_by.append({
                    "conflicting_hash": h2,
                    "type": conflict_type,
                    "delta_scenario": delta_d,
                })
                c2.contradicted_by.append({
                    "conflicting_hash": h1,
                    "type": conflict_type,
                    "delta_scenario": {k: (v[1], v[0]) for k, v in delta_d.items()},
                })

        return audit_records
