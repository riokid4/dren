from typing import List, Optional
from ais_kb.ontology.graph import AISOntologyGraph
from ais_kb.ontology.grounder import OntologyGrounder
from ais_kb.core.schema import ScenarioContext, ClaimEvidence, MechanisticClaim
from ais_kb.extraction.llm_schema import PaperExtractionOutput


class ExtractionProcessor:
    """Converts raw LLM `PaperExtractionOutput` JSON objects into ontology-grounded
    `MechanisticClaim` objects ready for `ClaimAssembler`."""

    def __init__(
        self,
        ontology: Optional[AISOntologyGraph] = None,
        grounder: Optional[OntologyGrounder] = None,
    ):
        self.ontology = ontology or AISOntologyGraph()
        self.grounder = grounder or OntologyGrounder(self.ontology)

    def process_paper_output(self, output: PaperExtractionOutput) -> List[MechanisticClaim]:
        claims: List[MechanisticClaim] = []

        for ext in output.claims:
            # Skip unverified speculation/hypotheses
            if ext.is_hypothesized:
                continue

            src_entity = self.grounder.create_entity(
                raw_text=ext.source_entity.raw_text,
                layer_scope=ext.source_entity.layer_scope,
                component_id=ext.source_entity.component_id,
                state_modifier=ext.source_entity.state_modifier,
            )

            tgt_entity = self.grounder.create_entity(
                raw_text=ext.target_entity.raw_text,
                layer_scope=ext.target_entity.layer_scope,
                component_id=ext.target_entity.component_id,
                state_modifier=ext.target_entity.state_modifier,
            )

            scenario_ctx = ScenarioContext(
                model_family=ext.scenario.model_family,
                param_scale=ext.scenario.param_scale,
                threat_model=ext.scenario.threat_model,
                benchmark=ext.scenario.benchmark,
            )

            evidence = ClaimEvidence(
                paper_id=output.paper_id,
                verbatim_quote=ext.verbatim_quote,
                section=ext.section,
                assay_method=ext.assay_method,
                assay_tier=ext.assay_tier,
                is_negated=ext.is_negated,
            )

            claim = MechanisticClaim(
                source_entity=src_entity,
                target_entity=tgt_entity,
                polarity=ext.polarity,
                is_negated=ext.is_negated,
                scenario_context=scenario_ctx,
                evidence=[evidence],
            )
            claims.append(claim)

        return claims
