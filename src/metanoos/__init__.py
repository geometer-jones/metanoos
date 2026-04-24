"""Complex-valued gated linear attention models."""

from metanoos.ablations import AblationSpec, ablation_model_kwargs, available_ablations, get_ablation
from metanoos.complex_ops import (
    ComplexEmbedding,
    ComplexLinear,
    ComplexRMSNorm,
    ModReLU,
    phase_feature,
    positive_gate,
)
from metanoos.corpora import (
    BundledCorpus,
    available_corpora,
    corpus_path,
    get_corpus,
    read_corpus_readme,
    read_corpus_sources,
    read_corpus_text,
)
from metanoos.layers import ComposedStateBlock, ComposedStateMixing, TemporalTransport
from metanoos.model import ComposedStateLanguageModel, ComposedStateOutput, real_parameter_count
from metanoos.state import (
    AssociativeState,
    StateMemoryEstimate,
    compose,
    identity_like,
    identity_state,
    local_state,
    measure,
    parallel_prefix_scan,
    prefix_scan,
    state_memory_estimate,
)

__all__ = [
    "AblationSpec",
    "AssociativeState",
    "BundledCorpus",
    "ComplexEmbedding",
    "ComplexLinear",
    "ComplexRMSNorm",
    "ComposedStateBlock",
    "ComposedStateLanguageModel",
    "ComposedStateMixing",
    "ComposedStateOutput",
    "ModReLU",
    "StateMemoryEstimate",
    "TemporalTransport",
    "ablation_model_kwargs",
    "available_ablations",
    "available_corpora",
    "compose",
    "corpus_path",
    "get_ablation",
    "get_corpus",
    "identity_like",
    "identity_state",
    "local_state",
    "measure",
    "parallel_prefix_scan",
    "phase_feature",
    "positive_gate",
    "prefix_scan",
    "read_corpus_readme",
    "read_corpus_sources",
    "read_corpus_text",
    "real_parameter_count",
    "state_memory_estimate",
]
