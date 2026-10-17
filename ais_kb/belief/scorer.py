from typing import List, Dict, Set
from ais_kb.core.schema import MechanisticClaim, ClaimEvidence


class EvidenceScorer:
    """Port of INDRA BeliefEngine + SimpleScorer + Compositional Grounding Factor.
    Propagates evidence transitively up the refinement DAG."""

    TIER_PARAMS = {
        "level_3_causal_intervention": {"e_rand": 0.10, "e_syst": 0.02},
        "level_2_attribution": {"e_rand": 0.25, "e_syst": 0.05},
        "level_0_behavioral": {"e_rand": 0.30, "e_syst": 0.05},
        "level_1_correlational": {"e_rand": 0.40, "e_syst": 0.10},
    }

    def score_evidence_list(self, evidence_list: List[ClaimEvidence]) -> float:
        if not evidence_list:
            return 0.0

        tier_counts: Dict[str, int] = {}
        for ev in evidence_list:
            tier = ev.assay_tier if ev.assay_tier in self.TIER_PARAMS else "level_1_correlational"
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        failure_prod = 1.0
        for tier, count in tier_counts.items():
            params = self.TIER_PARAMS[tier]
            e_rand = params["e_rand"]
            e_syst = params["e_syst"]
            tier_fail_prob = e_syst + (1.0 - e_syst) * (e_rand ** count)
            failure_prod *= tier_fail_prob

        return 1.0 - failure_prod

    def collect_transitive_evidence(
        self,
        claim_hash: int,
        claims_by_hash: Dict[int, MechanisticClaim],
        dag_children: Dict[int, List[int]],
        visited: Set[int] | None = None,
    ) -> List[ClaimEvidence]:
        """Recursively collects deduplicated evidence from `claim_hash` and all its
        more-specific descendants in `dag_children`."""
        if visited is None:
            visited = set()
        if claim_hash in visited or claim_hash not in claims_by_hash:
            return []
        visited.add(claim_hash)

        claim = claims_by_hash[claim_hash]
        ev_by_key: Dict[str, ClaimEvidence] = {
            ev.matches_key(): ev for ev in claim.evidence
        }

        for child_hash in dag_children.get(claim_hash, []):
            child_evs = self.collect_transitive_evidence(
                child_hash, claims_by_hash, dag_children, visited
            )
            for ev in child_evs:
                ev_by_key.setdefault(ev.matches_key(), ev)

        return list(ev_by_key.values())

    def score_all_claims(
        self,
        claims_by_hash: Dict[int, MechanisticClaim],
        dag_children: Dict[int, List[int]],
    ) -> Dict[int, float]:
        """Computes hierarchical belief scores for all claims in the knowledge base."""
        scores: Dict[int, float] = {}
        for h, claim in claims_by_hash.items():
            transitive_ev = self.collect_transitive_evidence(h, claims_by_hash, dag_children)
            base_belief = self.score_evidence_list(transitive_ev)
            mean_grounding = 0.5 * (
                claim.source_entity.grounding_score + claim.target_entity.grounding_score
            )
            final_belief = round(base_belief * mean_grounding, 4)
            claim.belief_score = final_belief
            scores[h] = final_belief
        return scores
