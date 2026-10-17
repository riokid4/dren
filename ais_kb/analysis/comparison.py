from typing import List, Dict, Any, Set, Tuple
from ais_kb.core.schema import MechanisticClaim
from ais_kb.ontology.graph import AISOntologyGraph


class CrossModelComparator:
    """Clean port of MIRA's `DeltaGraph`: compares assembled mechanistic claims
    between two model families (e.g., 'Llama-3' vs 'Gemma-2') to quantify
    Circuit Universality (conserved mechanisms vs model-specific divergences)."""

    def __init__(self, ontology: AISOntologyGraph):
        self.ontology = ontology

    @staticmethod
    def _mechanism_signature(claim: MechanisticClaim) -> Tuple[str, str, int]:
        return (
            claim.source_entity.canonical_id,
            claim.target_entity.canonical_id,
            claim.effective_outcome(),
        )

    def compare_model_families(
        self,
        claims: List[MechanisticClaim],
        model_a: str,
        model_b: str,
    ) -> Dict[str, Any]:
        claims_a = [c for c in claims if c.scenario_context.model_family == model_a and c.effective_outcome() != 0]
        claims_b = [c for c in claims if c.scenario_context.model_family == model_b and c.effective_outcome() != 0]

        sigs_a: Set[Tuple[str, str, int]] = {self._mechanism_signature(c) for c in claims_a}
        sigs_b: Set[Tuple[str, str, int]] = {self._mechanism_signature(c) for c in claims_b}

        exact_conserved = sigs_a & sigs_b
        only_in_a = sigs_a - sigs_b
        only_in_b = sigs_b - sigs_a

        # Check ontology-refined matches among non-identical signatures
        ontology_conserved = []
        for sa in only_in_a:
            for sb in only_in_b:
                if sa[2] == sb[2]:  # Same outcome polarity
                    if (
                        self.ontology.refinement_of(sa[0], sb[0])
                        and self.ontology.refinement_of(sa[1], sb[1])
                    ) or (
                        self.ontology.refinement_of(sb[0], sa[0])
                        and self.ontology.refinement_of(sb[1], sa[1])
                    ):
                        ontology_conserved.append((sa, sb))

        union_count = len(sigs_a | sigs_b)
        universality_score = (
            round((len(exact_conserved) + len(ontology_conserved)) / union_count, 4)
            if union_count > 0
            else 0.0
        )

        return {
            "model_a": model_a,
            "model_b": model_b,
            "universality_score": universality_score,
            "conserved_mechanisms": sorted(list(exact_conserved)),
            "ontology_refined_matches": ontology_conserved,
            "unique_to_model_a": sorted(list(only_in_a)),
            "unique_to_model_b": sorted(list(only_in_b)),
        }
