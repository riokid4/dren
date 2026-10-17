from typing import Set, Dict, Tuple
from collections import defaultdict
from ais_kb.ontology.graph import AISOntologyGraph


class RefinementIndex:
    """Sub-quadratic hierarchy candidate pruning via inverted index and ontology ancestors.
    Ported from `indra_world/assembly/refinement.py` (`get_relevants_for_stmt`)."""

    def __init__(self, ontology: AISOntologyGraph):
        self.ontology = ontology
        # Maps (role, canonical_id) -> set of claim hashes
        self.index: Dict[Tuple[str, str], Set[int]] = defaultdict(set)

    def add_claim(self, claim_hash: int, subj_id: str, obj_id: str):
        self.index[("subj", subj_id)].add(claim_hash)
        self.index[("obj", obj_id)].add(claim_hash)

    def get_candidate_parents(self, claim_hash: int, subj_id: str, obj_id: str) -> Set[int]:
        """Finds all claims whose subject is equal/ancestor of `subj_id` AND whose
        object is equal/ancestor of `obj_id`."""
        subj_ancestors = {subj_id} | self.ontology.get_ancestors(subj_id)
        obj_ancestors = {obj_id} | self.ontology.get_ancestors(obj_id)

        subj_candidates: Set[int] = set()
        for anc in subj_ancestors:
            subj_candidates |= self.index.get(("subj", anc), set())

        obj_candidates: Set[int] = set()
        for anc in obj_ancestors:
            obj_candidates |= self.index.get(("obj", anc), set())

        return (subj_candidates & obj_candidates) - {claim_hash}
