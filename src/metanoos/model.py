"""Language-model wrapper for composed-state sequence blocks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from metanoos.complex_ops import ComplexEmbedding, ComplexRMSNorm
from metanoos.layers import ComposedStateBlock
from metanoos.state import AssociativeState


@dataclass(frozen=True)
class ComposedStateOutput:
    logits: Tensor
    loss: Tensor | None = None
    hidden_states: Tensor | None = None


class ComposedStateLanguageModel(nn.Module):
    """Token model with complex hidden states and gauge-aware vocabulary readout."""

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int = 1,
        head_dim: int | None = None,
        mlp_ratio: int = 4,
        readout: str = "born",
        transport: bool | str = "rotary_decay",
        feature_mode: str = "complex",
    ) -> None:
        super().__init__()
        if readout not in {"born", "real"}:
            raise ValueError("readout must be 'born' or 'real'")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.readout = readout
        self.embed = ComplexEmbedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                ComposedStateBlock(
                    d_model,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    transport=transport,
                    feature_mode=feature_mode,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = ComplexRMSNorm(d_model)
        self.readout_carrier = ComplexEmbedding(vocab_size, d_model)
        self.logit_bias = nn.Parameter(torch.zeros(vocab_size))

    def amplitudes(self, x: Tensor) -> Tensor:
        return x @ self.readout_carrier.weight.conj_physical().transpose(0, 1)

    def logits_from_hidden(self, x: Tensor) -> Tensor:
        amplitudes = self.amplitudes(x)
        if self.readout == "born":
            logits = amplitudes.abs().square()
        else:
            logits = amplitudes.real
        return logits + self.logit_bias.to(device=x.device, dtype=logits.dtype)

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> ComposedStateOutput:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.logits_from_hidden(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), labels.reshape(-1))
        return ComposedStateOutput(logits=logits, loss=loss, hidden_states=x)

    @torch.no_grad()
    def step(
        self,
        input_ids: Tensor,
        memories: list[AssociativeState | None] | None = None,
    ) -> tuple[Tensor, list[AssociativeState]]:
        """One-token recurrent inference using the same composition law as scan."""

        if input_ids.ndim != 1:
            raise ValueError("step expects token ids with shape [batch]")
        if memories is None:
            memories = [None] * len(self.blocks)
        if len(memories) != len(self.blocks):
            raise ValueError("one memory entry is required per block")

        x = self.embed(input_ids)
        new_memories: list[AssociativeState] = []
        for block, memory in zip(self.blocks, memories):
            x, new_memory = block.step(x, memory)
            new_memories.append(new_memory)
        x = self.norm(x)
        return self.logits_from_hidden(x), new_memories
