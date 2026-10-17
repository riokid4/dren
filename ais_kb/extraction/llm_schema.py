from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    """Surface entity extracted by the LLM before ontology grounding."""
    raw_text: str = Field(
        ...,
        description=(
            "Core concept name stripped of specific layer/head numbers "
            "(e.g., 'refusal_direction', 'induction_head', 'dpo', 'sycophancy')."
        ),
    )
    layer_scope: Optional[str] = Field(
        None,
        description="Layer index or broad region if stated (e.g., 'L14', 'L10-L12', 'early', 'mid', 'late').",
    )
    component_id: Optional[str] = Field(
        None,
        description="Specific attention head, neuron, or SAE latent ID if stated (e.g., 'H7', 'SAE_402').",
    )
    state_modifier: Optional[str] = Field(
        None,
        description="State or intervention applied (e.g., 'ablated', 'steered', 'patched', 'quantized').",
    )


class ExtractedScenario(BaseModel):
    """Experimental boundary conditions (Scenario D) extracted from the paper."""
    model_family: Optional[str] = Field(
        None, description="Model family tested (e.g., 'Llama-3', 'Gemma-2', 'Pythia', 'GPT-2')."
    )
    param_scale: Optional[str] = Field(
        None, description="Model parameter size (e.g., '8B', '70B', '1B')."
    )
    threat_model: Optional[str] = Field(
        None,
        description="Evaluation regime (e.g., 'black_box_single_turn', 'black_box_multi_turn', 'white_box', 'encoded_cipher').",
    )
    benchmark: Optional[str] = Field(
        None, description="Benchmark or dataset name (e.g., 'HarmBench', 'TruthfulQA', 'IOI')."
    )


class ExtractedClaim(BaseModel):
    """A single empirical finding extracted from a paper."""
    verbatim_quote: str = Field(
        ..., description="Exact sentence(s) from the paper supporting this empirical finding."
    )
    section: Literal["abstract", "experiments", "results", "discussion"] = Field(
        "results", description="Paper section where the finding is reported."
    )
    source_entity: ExtractedEntity = Field(
        ..., description="The upstream component, intervention, or stimulus (X)."
    )
    target_entity: ExtractedEntity = Field(
        ..., description="The downstream behavior, internal feature, or failure mode (F)."
    )
    polarity: Literal[1, -1] = Field(
        ...,
        description="+1 if source promotes/increases target; -1 if source suppresses/decreases target.",
    )
    is_negated: bool = Field(
        False,
        description="True ONLY if the experiment tested this link and found it FAILED or had NO effect.",
    )
    is_hypothesized: bool = Field(
        False,
        description="True if this is an untested hypothesis or future work speculation, NOT an empirical result.",
    )
    assay_method: str = Field(
        ..., description="Method used by authors (e.g., 'activation_steering', 'zero_ablation', 'linear_probe')."
    )
    assay_tier: Literal[
        "level_3_causal_intervention",
        "level_2_attribution",
        "level_1_correlational",
        "level_0_behavioral",
    ] = Field(
        ...,
        description=(
            "level_3_causal_intervention (ablation/patching/steering), "
            "level_2_attribution (gradients/SHAP), "
            "level_1_correlational (probing/logit lens), "
            "level_0_behavioral (input-output benchmark only)."
        ),
    )
    scenario: ExtractedScenario = Field(default_factory=ExtractedScenario)


class PaperExtractionOutput(BaseModel):
    """Top-level structured output returned by the LLM Reader for a paper."""
    paper_id: str = Field(..., description="arXiv ID, DOI, or unique paper identifier.")
    claims: List[ExtractedClaim] = Field(default_factory=list)
