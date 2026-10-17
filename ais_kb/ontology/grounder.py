import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from ais_kb.ontology.graph import AISOntologyGraph
from ais_kb.core.schema import AISEntity


@dataclass
class GroundingMatch:
    canonical_id: str
    matched_term: str
    status: str  # 'exact_id' | 'synonym' | 'token_overlap'
    score: float


class OntologyGrounder:
    """Maps raw surface strings from AI Safety papers to canonical ontology IDs
    with calibrated confidence scores in [0.0, 1.0]."""

    STATUS_WEIGHTS = {
        "exact_id": 1.0,
        "synonym": 0.90,
        "token_overlap": 0.70,
    }

    def __init__(self, ontology: Optional[AISOntologyGraph] = None):
        self.ontology = ontology or AISOntologyGraph()
        self._norm_lookup: Dict[str, List[Tuple[str, str, str]]] = {}
        self._build_index()

    @staticmethod
    def normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[-_/]+", " ", text)
        text = re.sub(r"[^a-z0-9\s]", "", text)
        if text.endswith("s") and len(text) > 4 and not text.endswith("ss"):
            text = text[:-1]
        return re.sub(r"\s+", " ", text).strip()

    def _build_index(self):
        for node_id, data in self.ontology.nodes(data=True):
            norm_id = self.normalize(node_id)
            self._norm_lookup.setdefault(norm_id, []).append((node_id, node_id, "exact_id"))
            for syn in data.get("synonyms", []):
                norm_syn = self.normalize(syn)
                self._norm_lookup.setdefault(norm_syn, []).append((node_id, syn, "synonym"))

    def ground(self, raw_text: str) -> List[GroundingMatch]:
        norm_query = self.normalize(raw_text)
        matches: List[GroundingMatch] = []

        # 1. O(1) normalized exact/synonym lookup
        if norm_query in self._norm_lookup:
            for canonical_id, term_str, status in self._norm_lookup[norm_query]:
                base_score = self.STATUS_WEIGHTS[status]
                case_bonus = 0.05 if raw_text.lower() == term_str.lower() else 0.0
                score = min(1.0, round(base_score + case_bonus, 4))
                matches.append(GroundingMatch(canonical_id, term_str, status, score))

        # 2. Fallback token-subset matching if no exact match
        if not matches and norm_query:
            query_tokens = set(norm_query.split())
            for norm_key, entries in self._norm_lookup.items():
                key_tokens = set(norm_key.split())
                if query_tokens and key_tokens and (query_tokens <= key_tokens or key_tokens <= query_tokens):
                    overlap = len(query_tokens & key_tokens) / max(len(query_tokens), len(key_tokens))
                    if overlap >= 0.5:
                        canonical_id, term_str, _ = entries[0]
                        score = round(self.STATUS_WEIGHTS["token_overlap"] * overlap, 4)
                        matches.append(GroundingMatch(canonical_id, term_str, "token_overlap", score))

        # Deduplicate by canonical_id keeping highest score
        best_by_id: Dict[str, GroundingMatch] = {}
        for m in matches:
            if m.canonical_id not in best_by_id or m.score > best_by_id[m.canonical_id].score:
                best_by_id[m.canonical_id] = m

        return sorted(best_by_id.values(), key=lambda x: x.score, reverse=True)

    def create_entity(
        self,
        raw_text: str,
        layer_scope: Optional[str] = None,
        component_id: Optional[str] = None,
        state_modifier: Optional[str] = None,
    ) -> AISEntity:
        """Grounds `raw_text` and returns a populated `AISEntity`."""
        matches = self.ground(raw_text)
        if matches:
            top = matches[0]
            return AISEntity(
                canonical_id=top.canonical_id,
                raw_text=raw_text,
                grounding_score=top.score,
                layer_scope=layer_scope,
                component_id=component_id,
                state_modifier=state_modifier,
            )
        fallback_id = self.normalize(raw_text).replace(" ", "_")
        return AISEntity(
            canonical_id=fallback_id,
            raw_text=raw_text,
            grounding_score=0.50,
            layer_scope=layer_scope,
            component_id=component_id,
            state_modifier=state_modifier,
        )
