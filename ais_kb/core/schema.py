from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple, Any
import hashlib


class AISEntity(BaseModel):
    canonical_id: str = Field(..., description="Grounded ontology ID")
    raw_text: str = Field(..., description="Surface text from paper")
    grounding_score: float = Field(1.0, ge=0.0, le=1.0)
    layer_scope: Optional[str] = Field(None, description="e.g., L12, early, mid, late")
    component_id: Optional[str] = Field(None, description="e.g., H7, SAE_402")
    state_modifier: Optional[str] = Field(None, description="e.g., ablated, steered")

    def matches_key(self) -> str:
        components = [
            self.canonical_id,
            self.layer_scope or "",
            self.component_id or "",
            self.state_modifier or "",
        ]
        return hashlib.md5("|".join(components).encode()).hexdigest()[:16]

    def refinement_of(self, other: "AISEntity", ontology) -> bool:
        """Returns True if `self` is equal to or a more specific refinement of `other`."""
        if not ontology.refinement_of(self.canonical_id, other.canonical_id):
            return False

        def coord_refines(spec: Optional[str], gen: Optional[str], is_layer: bool = False) -> bool:
            if gen is None or spec == gen:
                return True
            if spec is None:
                return False
            if is_layer and gen in {"early", "mid", "late"} and spec.upper().startswith("L"):
                return True
            return False

        return (
            coord_refines(self.layer_scope, other.layer_scope, is_layer=True)
            and coord_refines(self.component_id, other.component_id)
            and coord_refines(self.state_modifier, other.state_modifier)
        )


class ScenarioContext(BaseModel):
    model_family: Optional[str] = Field(None, description="e.g., Llama-3, Gemma-2")
    param_scale: Optional[str] = Field(None, description="e.g., 8B, 70B")
    threat_model: Optional[str] = Field(None, description="e.g., black_box_single_turn")
    benchmark: Optional[str] = Field(None, description="e.g., HarmBench, TruthfulQA")

    def matches_key(self) -> str:
        parts = [
            self.model_family or "*",
            self.param_scale or "*",
            self.threat_model or "*",
            self.benchmark or "*",
        ]
        return "|".join(parts)

    def refinement_of(self, other: "ScenarioContext") -> bool:
        """Returns True if `self` specifies an equal or more specific scenario than `other`."""
        for field in ("model_family", "param_scale", "threat_model", "benchmark"):
            gen_val = getattr(other, field)
            spec_val = getattr(self, field)
            if gen_val is not None and spec_val != gen_val:
                return False
        return True

    def diff_scenario(self, other: "ScenarioContext") -> Dict[str, Tuple[Optional[str], Optional[str]]]:
        """Returns scenario axes that differ between `self` and `other` (Delta D)."""
        diffs: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        for field in ("model_family", "param_scale", "threat_model", "benchmark"):
            s1 = getattr(self, field)
            s2 = getattr(other, field)
            if s1 is not None and s2 is not None and s1 != s2:
                diffs[field] = (s1, s2)
        return diffs


class ClaimEvidence(BaseModel):
    paper_id: str
    verbatim_quote: str
    section: Optional[str] = None
    assay_method: str
    assay_tier: str = Field(
        "level_1_correlational",
        description="level_0_behavioral | level_1_correlational | level_2_attribution | level_3_causal_intervention",
    )
    is_negated: bool = False

    def matches_key(self) -> str:
        return f"{self.paper_id}|{hashlib.md5(self.verbatim_quote.encode()).hexdigest()[:12]}"


class MechanisticClaim(BaseModel):
    source_entity: AISEntity
    target_entity: AISEntity
    polarity: int = Field(..., description="+1 (promotes/increases) or -1 (suppresses/decreases)")
    is_negated: bool = Field(False, description="True if the empirical test showed NO effect or failed")
    scenario_context: ScenarioContext = Field(default_factory=ScenarioContext)
    evidence: List[ClaimEvidence] = Field(default_factory=list)
    supports: List[int] = Field(default_factory=list, description="More general parent claim hashes")
    supported_by: List[int] = Field(default_factory=list, description="More specific child claim hashes")
    contradicted_by: List[Dict[str, Any]] = Field(default_factory=list)
    belief_score: Optional[float] = None

    def effective_outcome(self) -> int:
        """Returns +1 (promotes), -1 (suppresses), or 0 (negated / failed to hold)."""
        if self.is_negated or (self.evidence and all(ev.is_negated for ev in self.evidence)):
            return 0
        return self.polarity

    def matches_key(self) -> str:
        raw = "|".join([
            self.source_entity.matches_key(),
            self.target_entity.matches_key(),
            str(self.effective_outcome()),
            self.scenario_context.matches_key(),
        ])
        return hashlib.sha256(raw.encode()).hexdigest()

    def get_hash(self) -> int:
        """Deterministic 63-bit integer hash (mk_hash)."""
        return int(self.matches_key()[:16], 16) % (2**63)

    def refinement_of(self, other: "MechanisticClaim", ontology) -> bool:
        """Returns True if `self` is a strictly more specific refinement of `other`."""
        if self.get_hash() == other.get_hash():
            return False
        if self.effective_outcome() != other.effective_outcome():
            return False
        return (
            self.source_entity.refinement_of(other.source_entity, ontology)
            and self.target_entity.refinement_of(other.target_entity, ontology)
            and self.scenario_context.refinement_of(other.scenario_context)
        )
