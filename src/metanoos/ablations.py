"""Named ablation presets for the complex GLA testbed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class AblationSpec:
    """A partial model configuration for one immediate-neighbor comparison."""

    name: str
    description: str
    model_kwargs: Mapping[str, object]

    def kwargs(self) -> dict[str, object]:
        return dict(self.model_kwargs)


_ABLATIONS: dict[str, AblationSpec] = {
    "complex_rotary_gla": AblationSpec(
        name="complex_rotary_gla",
        description="Default complex GLA: complex q/k/v, real normalizer, per-head complex decay.",
        model_kwargs={
            "transport": "rotary_decay",
            "feature_mode": "complex",
            "readout": "born",
        },
    ),
    "complex_decay_gla": AblationSpec(
        name="complex_decay_gla",
        description="Ablates rotation by forcing content transport to real positive decay.",
        model_kwargs={
            "transport": "decay",
            "feature_mode": "complex",
            "readout": "born",
        },
    ),
    "complex_linear_attention": AblationSpec(
        name="complex_linear_attention",
        description="Ablates temporal decay, leaving causal complex linear attention.",
        model_kwargs={
            "transport": "none",
            "feature_mode": "complex",
            "readout": "born",
        },
    ),
    "complex_rope_decay_gla": AblationSpec(
        name="complex_rope_decay_gla",
        description="Fixed RoPE q/k phase with real positive content decay.",
        model_kwargs={
            "transport": "decay",
            "feature_mode": "complex",
            "position_encoding": "rope",
            "readout": "born",
        },
    ),
    "complex_rope_rotary_gla": AblationSpec(
        name="complex_rope_rotary_gla",
        description="Fixed RoPE q/k phase plus learned per-head complex decay.",
        model_kwargs={
            "transport": "rotary_decay",
            "feature_mode": "complex",
            "position_encoding": "rope",
            "readout": "born",
        },
    ),
    "magnitude_feature_gla": AblationSpec(
        name="magnitude_feature_gla",
        description="Ablates q/k phase by using positive magnitude features for q and k.",
        model_kwargs={
            "transport": "rotary_decay",
            "feature_mode": "magnitude",
            "readout": "born",
        },
    ),
    "real_readout_gla": AblationSpec(
        name="real_readout_gla",
        description="Ablates Born readout by using Re(<x, e_v>) logits.",
        model_kwargs={
            "transport": "rotary_decay",
            "feature_mode": "complex",
            "readout": "real",
        },
    ),
}


def available_ablations() -> tuple[AblationSpec, ...]:
    return tuple(_ABLATIONS[name] for name in sorted(_ABLATIONS))


def get_ablation(name: str) -> AblationSpec:
    try:
        return _ABLATIONS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_ABLATIONS))
        raise KeyError(f"Unknown ablation '{name}'. Available ablations: {available}.") from exc


def ablation_model_kwargs(name: str, **overrides: object) -> dict[str, object]:
    kwargs = get_ablation(name).kwargs()
    kwargs.update(overrides)
    return kwargs
