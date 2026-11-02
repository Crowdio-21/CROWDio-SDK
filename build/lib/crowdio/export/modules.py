"""PyTorch wrappers that become one ONNX graph per shard.

Every shard consumes and produces named tensors (see the manifest tensor
contract). Encoder shards carry the attention mask through unchanged because
later shards need it.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .adapters import EncoderAdapter


class EmbedShard(nn.Module):
    def __init__(self, adapter: EncoderAdapter):
        super().__init__()
        self.adapter = adapter
        self.embeddings = adapter.embed_module()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.adapter.embed(self.embeddings, input_ids)
        return hidden, attention_mask.clone()


class EncoderRangeShard(nn.Module):
    def __init__(self, adapter: EncoderAdapter, start_layer: int, end_layer: int):
        super().__init__()
        self.adapter = adapter
        self.layers = nn.ModuleList(adapter.layers()[start_layer:end_layer + 1])

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        layer_mask = self.adapter.layer_mask(attention_mask, hidden_states.dtype)
        for layer in self.layers:
            hidden_states = self.adapter.run_layer(layer, hidden_states, layer_mask, attention_mask)
        return hidden_states, attention_mask.clone()


class HeadShard(nn.Module):
    def __init__(self, adapter: EncoderAdapter):
        super().__init__()
        self.adapter = adapter
        self.head = adapter.head_module()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.adapter.head(self.head, hidden_states)


class FullShard(nn.Module):
    def __init__(self, adapter: EncoderAdapter):
        super().__init__()
        self.embed = EmbedShard(adapter)
        self.encoder = EncoderRangeShard(adapter, 0, adapter.num_layers - 1)
        self.head = HeadShard(adapter)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden, mask = self.embed(input_ids, attention_mask)
        hidden, _ = self.encoder(hidden, mask)
        return self.head(hidden)
