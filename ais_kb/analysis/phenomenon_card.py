from typing import Dict, Any, List, Optional
from ais_kb.storage.db import KnowledgeBaseStore


class PhenomenonCardGenerator:
    """Generates structured Phenomenon Cards summarizing:
    [Model/Mechanism X in Scenario D] -> [Behavior F], Replications, Hierarchy,
    and Scenario-Tweak Robustness Failures (Delta D)."""

    def __init__(self, store: KnowledgeBaseStore):
        self.store = store

    def generate_card(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        model_family: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        matching_claims = self.store.query_claims(
            source_id=source_id,
            target_id=target_id,
            model_family=model_family,
        )

        cards: List[Dict[str, Any]] = []
        for claim in matching_claims:
            if claim.effective_outcome() == 0:
                # Failed tweaks are attached under the primary claim's robustness_failures
                continue

            replications = [
                {
                    "paper_id": ev.paper_id,
                    "assay_method": ev.assay_method,
                    "assay_tier": ev.assay_tier,
                    "quote": ev.verbatim_quote,
                }
                for ev in claim.evidence
            ]

            child_mechanisms = []
            for child_hash in claim.supported_by:
                child = self.store.get_claim(child_hash)
                if child:
                    child_mechanisms.append({
                        "source": child.source_entity.canonical_id,
                        "layer_scope": child.source_entity.layer_scope,
                        "component_id": child.source_entity.component_id,
                        "target": child.target_entity.canonical_id,
                        "belief_score": child.belief_score,
                    })

            robustness_failures = []
            direct_contradictions = []
            for conflict in claim.contradicted_by:
                other = self.store.get_claim(conflict["conflicting_hash"])
                entry = {
                    "conflicting_hash": conflict["conflicting_hash"],
                    "delta_scenario": conflict["delta_scenario"],
                    "papers": [ev.paper_id for ev in other.evidence] if other else [],
                    "quotes": [ev.verbatim_quote for ev in other.evidence] if other else [],
                }
                if conflict["type"] == "scenario_tweak_failure":
                    robustness_failures.append(entry)
                else:
                    direct_contradictions.append(entry)

            cards.append({
                "claim_hash": claim.get_hash(),
                "phenomenon": {
                    "source_mechanism": claim.source_entity.canonical_id,
                    "source_coordinates": {
                        "layer": claim.source_entity.layer_scope,
                        "component": claim.source_entity.component_id,
                        "state": claim.source_entity.state_modifier,
                    },
                    "target_behavior": claim.target_entity.canonical_id,
                    "polarity": "+1 (Promotes)" if claim.polarity == 1 else "-1 (Suppresses)",
                    "baseline_scenario": claim.scenario_context.model_dump(),
                },
                "belief_score": claim.belief_score,
                "replication_count": len(replications),
                "replications": replications,
                "supporting_sub_mechanisms": child_mechanisms,
                "scenario_tweak_failures": robustness_failures,
                "direct_contradictions": direct_contradictions,
            })

        return cards
