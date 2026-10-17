import unittest
from ais_kb.ontology.graph import AISOntologyGraph
from ais_kb.ontology.grounder import OntologyGrounder
from ais_kb.core.schema import ScenarioContext, ClaimEvidence, MechanisticClaim
from ais_kb.assembly.preassembler import ClaimAssembler
from ais_kb.belief.scorer import EvidenceScorer
from ais_kb.extraction.llm_schema import PaperExtractionOutput, ExtractedClaim, ExtractedEntity, ExtractedScenario
from ais_kb.extraction.processor import ExtractionProcessor
from ais_kb.storage.db import KnowledgeBaseStore
from ais_kb.analysis.circuits import CircuitGraphAnalyzer
from ais_kb.analysis.comparison import CrossModelComparator
from ais_kb.analysis.phenomenon_card import PhenomenonCardGenerator


class TestAISKnowledgeBase(unittest.TestCase):
    def test_end_to_end_assembly_and_analysis(self):
        ont = AISOntologyGraph()
        grounder = OntologyGrounder(ont)

        # Paper 1: Steering refusal direction in Llama-3 8B (single-turn) -> induces refusal
        c1 = MechanisticClaim(
            source_entity=grounder.create_entity("refusal_feature", layer_scope="L14", state_modifier="steered"),
            target_entity=grounder.create_entity("harm_refusal"),
            polarity=1,
            scenario_context=ScenarioContext(
                model_family="Llama-3", param_scale="8B", threat_model="black_box_single_turn", benchmark="HarmBench"
            ),
            evidence=[ClaimEvidence(
                paper_id="arXiv:2406.0001",
                verbatim_quote="Steering the L14 refusal feature in Llama-3-8B induces refusal on single-turn HarmBench.",
                assay_method="activation_steering",
                assay_tier="level_3_causal_intervention",
            )],
        )

        # Paper 2 (Replication of Paper 1): Exact same setup and outcome
        c2 = MechanisticClaim(
            source_entity=grounder.create_entity("harm_refusal_latent", layer_scope="L14", state_modifier="steered"),
            target_entity=grounder.create_entity("safety_refusal"),
            polarity=1,
            scenario_context=ScenarioContext(
                model_family="Llama-3", param_scale="8B", threat_model="black_box_single_turn", benchmark="HarmBench"
            ),
            evidence=[ClaimEvidence(
                paper_id="arXiv:2408.0002",
                verbatim_quote="We replicate that steering L14 refusal latents in Llama-3-8B reliably triggers safety refusal.",
                assay_method="activation_steering",
                assay_tier="level_3_causal_intervention",
            )],
        )

        # Paper 3 (General Parent Claim): General feature_neuron -> safety_behavior in Llama-3
        c_general = MechanisticClaim(
            source_entity=grounder.create_entity("feature"),
            target_entity=grounder.create_entity("safety_behavior"),
            polarity=1,
            scenario_context=ScenarioContext(model_family="Llama-3"),
            evidence=[ClaimEvidence(
                paper_id="arXiv:2401.0003",
                verbatim_quote="Internal features correlate with safety behaviors in Llama-3.",
                assay_method="linear_probe",
                assay_tier="level_1_correlational",
            )],
        )

        # Paper 4 (Robustness Failure when Scenario D is tweaked to multi_turn!):
        c_tweak_fail = MechanisticClaim(
            source_entity=grounder.create_entity("refusal_feature", layer_scope="L14", state_modifier="steered"),
            target_entity=grounder.create_entity("harm_refusal"),
            polarity=1,
            is_negated=True,
            scenario_context=ScenarioContext(
                model_family="Llama-3", param_scale="8B", threat_model="black_box_multi_turn", benchmark="HarmBench"
            ),
            evidence=[ClaimEvidence(
                paper_id="arXiv:2409.0004",
                verbatim_quote="Under multi-turn Crescendo prompts, L14 refusal steering fails to maintain refusal.",
                assay_method="activation_steering",
                assay_tier="level_3_causal_intervention",
                is_negated=True,
            )],
        )

        # Paper 5 (2nd hop for Multi-Hop Circuit): harm_refusal (-1 suppresses) -> jailbreak_compliance
        c_hop2 = MechanisticClaim(
            source_entity=grounder.create_entity("harm_refusal"),
            target_entity=grounder.create_entity("jailbreak_compliance"),
            polarity=-1,
            scenario_context=ScenarioContext(
                model_family="Llama-3", param_scale="8B", threat_model="black_box_single_turn", benchmark="HarmBench"
            ),
            evidence=[ClaimEvidence(
                paper_id="arXiv:2407.0005",
                verbatim_quote="Triggering harm refusal directly suppresses jailbreak compliance on HarmBench.",
                assay_method="causal_intervention",
                assay_tier="level_3_causal_intervention",
            )],
        )

        # Paper 6 (Gemma-2 conserved mechanism for Cross-Model Universality test):
        c_gemma = MechanisticClaim(
            source_entity=grounder.create_entity("refusal_feature", layer_scope="L16", state_modifier="steered"),
            target_entity=grounder.create_entity("harm_refusal"),
            polarity=1,
            scenario_context=ScenarioContext(
                model_family="Gemma-2", param_scale="9B", threat_model="black_box_single_turn", benchmark="HarmBench"
            ),
            evidence=[ClaimEvidence(
                paper_id="arXiv:2410.0006",
                verbatim_quote="Steering the refusal direction in Gemma-2-9B induces harm refusal.",
                assay_method="activation_steering",
                assay_tier="level_3_causal_intervention",
            )],
        )

        db = KnowledgeBaseStore(":memory:", ont)
        summary = db.ingest_claims([c1, c2, c_general, c_tweak_fail, c_hop2, c_gemma])
        self.assertEqual(summary["total_assembled_claims"], 5)

        # Test PhenomenonCardGenerator
        card_gen = PhenomenonCardGenerator(db)
        cards = card_gen.generate_card(
            source_id="refusal_direction", target_id="refusal", model_family="Llama-3"
        )
        self.assertEqual(len(cards), 1)
        card = cards[0]
        self.assertEqual(card["replication_count"], 2)
        self.assertEqual(len(card["scenario_tweak_failures"]), 1)
        self.assertIn("threat_model", card["scenario_tweak_failures"][0]["delta_scenario"])

        # Test Multi-Hop Signed Circuit Discovery: refusal_direction[L14] (+1) -> refusal (-1) -> jailbreak_compliance (net -1)
        all_claims = db.query_claims()
        circuit_analyzer = CircuitGraphAnalyzer(all_claims)
        paths = circuit_analyzer.find_multi_hop_circuits(
            source_node="refusal_direction[L14]",
            target_node="jailbreak_compliance",
            expected_polarity=-1,
        )
        self.assertGreaterEqual(len(paths), 1)
        self.assertEqual(paths[0].net_polarity, -1)

        # Test Cross-Model Universality Comparator (Llama-3 vs Gemma-2)
        comparator = CrossModelComparator(ont)
        comp = comparator.compare_model_families(all_claims, "Llama-3", "Gemma-2")
        self.assertGreater(comp["universality_score"], 0.0)
        self.assertIn(("refusal_direction", "refusal", 1), comp["conserved_mechanisms"])


if __name__ == "__main__":
    unittest.main()
