import math
from dataclasses import dataclass
from itertools import product
from typing import List, Optional
import networkx as nx
from ais_kb.core.schema import MechanisticClaim, AISEntity


@dataclass
class VerifiedCircuitPath:
    nodes: List[str]
    claims: List[MechanisticClaim]
    net_polarity: int
    joint_belief: float
    scenario_coherent: bool
    divergent_axes: List[str]


class CircuitGraphAnalyzer:
    """Clean port of EMMAA signed graph assembly and multi-hop path verification.
    Assembles `MechanisticClaim` objects into a signed belief-weighted MultiDiGraph
    and discovers multi-hop causal circuits."""

    def __init__(self, claims: List[MechanisticClaim], min_belief: float = 0.0):
        self.claims = [c for c in claims if (c.belief_score or 0.0) >= min_belief]
        self.graph = self._build_signed_graph()

    @staticmethod
    def node_label(entity: AISEntity) -> str:
        coords = ":".join(filter(None, [entity.layer_scope, entity.component_id]))
        if coords:
            return f"{entity.canonical_id}[{coords}]"
        return entity.canonical_id

    def _build_signed_graph(self) -> nx.MultiDiGraph:
        g = nx.MultiDiGraph()
        for claim in self.claims:
            outcome = claim.effective_outcome()
            if outcome == 0:
                # Skip negated/null claims when building active forward causal paths
                continue

            u = self.node_label(claim.source_entity)
            v = self.node_label(claim.target_entity)
            sign = 0 if outcome == 1 else 1
            belief = max(claim.belief_score or 0.5, 1e-6)
            weight = -math.log(belief) + 1e-4

            g.add_node(u, canonical_id=claim.source_entity.canonical_id)
            g.add_node(v, canonical_id=claim.target_entity.canonical_id)
            g.add_edge(
                u,
                v,
                key=claim.get_hash(),
                sign=sign,
                belief=belief,
                weight=weight,
                claim=claim,
            )

            # Bridge coordinate-specific nodes (e.g. refusal_direction[L14]) with their canonical ID
            for node_str, cid in ((u, claim.source_entity.canonical_id), (v, claim.target_entity.canonical_id)):
                if node_str != cid:
                    g.add_edge(node_str, cid, key=f"ont_{node_str}", sign=0, belief=1.0, weight=1e-4, claim=None)
                    g.add_edge(cid, node_str, key=f"ont_rev_{node_str}", sign=0, belief=1.0, weight=1e-4, claim=None)

        return g

    def find_multi_hop_circuits(
        self,
        source_node: str,
        target_node: str,
        expected_polarity: int = 1,
        max_hops: int = 4,
    ) -> List[VerifiedCircuitPath]:
        """Finds all simple paths up to `max_hops` whose net sign parity matches
        `expected_polarity` (+1 or -1), ranked by scenario coherence and joint belief."""
        if source_node not in self.graph or target_node not in self.graph:
            return []

        target_sign = 0 if expected_polarity == 1 else 1
        discovered: List[VerifiedCircuitPath] = []

        for node_path in nx.all_simple_paths(self.graph, source_node, target_node, cutoff=max_hops):
            hop_edges = [
                list(self.graph[u][v].values())
                for u, v in zip(node_path[:-1], node_path[1:])
            ]
            for edge_chain in product(*hop_edges):
                net_sign = sum(e["sign"] for e in edge_chain) % 2
                if net_sign != target_sign:
                    continue

                joint_belief = 1.0
                path_claims: List[MechanisticClaim] = []
                for e in edge_chain:
                    joint_belief *= e["belief"]
                    if e["claim"] is not None:
                        path_claims.append(e["claim"])

                if not path_claims:
                    continue

                # Check scenario coherence across hops
                coherent = True
                div_axes = set()
                for i in range(len(path_claims)):
                    for j in range(i + 1, len(path_claims)):
                        diff = path_claims[i].scenario_context.diff_scenario(
                            path_claims[j].scenario_context
                        )
                        if diff:
                            coherent = False
                            div_axes.update(diff.keys())

                discovered.append(
                    VerifiedCircuitPath(
                        nodes=node_path,
                        claims=path_claims,
                        net_polarity=1 if net_sign == 0 else -1,
                        joint_belief=round(joint_belief, 4),
                        scenario_coherent=coherent,
                        divergent_axes=sorted(div_axes),
                    )
                )

        discovered.sort(
            key=lambda c: (c.scenario_coherent, c.joint_belief, -len(c.nodes)),
            reverse=True,
        )
        return discovered
