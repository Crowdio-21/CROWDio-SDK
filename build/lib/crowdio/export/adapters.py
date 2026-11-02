"""Architecture adapters: how to embed, run one layer, and apply the head.

Adapters hide model-family differences so the shard wrappers and the
verifier are generic. Implemented: DistilBERT and BERT sequence classifiers.
The registry is the extension point for MobileBERT, ViT-B/16 and the
BERT-large / ViT-L stress tests.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn

FLOPS_METHOD = {
    "method": "analytic_matmul_v1",
    "unit": "FLOPs per sequence",
    "formula": (
        "per encoder layer: 2*S*sum(in*out of nn.Linear) + 4*S*S*hidden (QK^T and AV); "
        "head: 2*sum(in*out of nn.Linear) on the pooled token; embeddings: 0 (lookups)"
    ),
}


def _module_bytes(*modules: nn.Module) -> int:
    seen = set()
    total = 0
    for module in modules:
        for tensor in list(module.parameters()) + list(module.buffers()):
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            total += tensor.numel() * tensor.element_size()
    return total


def _linear_macs(module: nn.Module) -> int:
    return sum(m.in_features * m.out_features for m in module.modules() if isinstance(m, nn.Linear))


def force_eager_attention(model: nn.Module) -> None:
    """Use eager attention so layers accept an additive 4D mask when traced."""
    setter = getattr(model, "set_attn_implementation", None)
    if callable(setter):
        try:
            setter("eager")
            return
        except Exception:
            pass
    config = getattr(model, "config", None)
    if config is not None:
        config._attn_implementation = "eager"


class EncoderAdapter:
    """Base adapter for an encoder-only sequence classifier."""

    architecture = "encoder"

    def __init__(self, model: nn.Module):
        self.model = model.eval()
        force_eager_attention(self.model)
        self.config = model.config

    # -- structure ---------------------------------------------------------
    @property
    def num_layers(self) -> int:
        return len(self.layers())

    @property
    def hidden_size(self) -> int:
        raise NotImplementedError

    @property
    def num_labels(self) -> int:
        return int(getattr(self.config, "num_labels", 2))

    @property
    def vocab_size(self) -> int:
        return int(self.config.vocab_size)

    @property
    def pad_token_id(self) -> int:
        value = getattr(self.config, "pad_token_id", None)
        return int(value) if value is not None else 0

    def embed_module(self) -> nn.Module:
        raise NotImplementedError

    def layers(self) -> List[nn.Module]:
        raise NotImplementedError

    def head_module(self) -> nn.Module:
        raise NotImplementedError

    # -- computation -------------------------------------------------------
    def embed(self, module: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def head(self, module: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def layer_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Additive ``[batch, 1, 1, seq]`` mask for eager attention."""
        mask = attention_mask[:, None, None, :].to(dtype)
        return (1.0 - mask) * torch.finfo(dtype).min

    def run_layer(self, layer: nn.Module, hidden: torch.Tensor, layer_mask: torch.Tensor,
                  attention_mask: torch.Tensor) -> torch.Tensor:
        output = layer(hidden, attention_mask=layer_mask)
        return output[0] if isinstance(output, tuple) else output

    def reference_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    # -- sizes -------------------------------------------------------------
    def sizes(self, model_id: str, reference_seq_len: int):
        from .planner import ModelSizes

        seq = reference_seq_len
        layers = self.layers()
        return ModelSizes(
            model_id=model_id,
            num_layers=len(layers),
            hidden_size=self.hidden_size,
            num_labels=self.num_labels,
            embed_bytes=_module_bytes(self.embed_module()),
            layer_bytes=[_module_bytes(layer) for layer in layers],
            head_bytes=_module_bytes(self.head_module()),
            embed_flops=0,
            layer_flops=[2 * seq * _linear_macs(layer) + 4 * seq * seq * self.hidden_size for layer in layers],
            head_flops=2 * _linear_macs(self.head_module()),
        )

    def source_info(self) -> Dict[str, Any]:
        id2label = getattr(self.config, "id2label", None) or {}
        return {
            "name_or_path": getattr(self.config, "_name_or_path", ""),
            "architecture": self.architecture,
            "model_type": getattr(self.config, "model_type", ""),
            "task": "sequence_classification",
            "num_labels": self.num_labels,
            "id2label": {str(k): v for k, v in id2label.items()},
            "pad_token_id": self.pad_token_id,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": int(getattr(self.config, "max_position_embeddings", 512)),
        }


class _DistilBertHead(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.pre_classifier = model.pre_classifier
        self.classifier = model.classifier
        self.dropout = model.dropout


class DistilBertAdapter(EncoderAdapter):
    architecture = "distilbert"

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self._head = _DistilBertHead(model)
        params = inspect.signature(self.layers()[0].forward).parameters
        self._legacy_layer_api = "x" in params and "hidden_states" not in params

    @property
    def hidden_size(self) -> int:
        return int(self.config.dim)

    def embed_module(self) -> nn.Module:
        return self.model.distilbert.embeddings

    def layers(self) -> List[nn.Module]:
        return list(self.model.distilbert.transformer.layer)

    def head_module(self) -> nn.Module:
        return self._head

    def embed(self, module: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        return module(input_ids=input_ids)

    def head(self, module: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        pooled = torch.relu(module.pre_classifier(hidden[:, 0]))
        return module.classifier(module.dropout(pooled))

    def run_layer(self, layer, hidden, layer_mask, attention_mask):
        if self._legacy_layer_api:  # transformers 4.x TransformerBlock(x, attn_mask)
            output = layer(x=hidden, attn_mask=attention_mask, head_mask=None, output_attentions=False)
            return output[0] if isinstance(output, tuple) else output
        return super().run_layer(layer, hidden, layer_mask, attention_mask)


class _BertHead(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.pooler = model.bert.pooler
        self.dropout = model.dropout
        self.classifier = model.classifier


class BertAdapter(EncoderAdapter):
    """BERT family. Single-segment inputs: token_type_ids are zeros."""

    architecture = "bert"

    def __init__(self, model: nn.Module):
        super().__init__(model)
        if model.bert.pooler is None:
            raise ValueError("BERT export needs the pooler (add_pooling_layer=True)")
        self._head = _BertHead(model)

    @property
    def hidden_size(self) -> int:
        return int(self.config.hidden_size)

    def embed_module(self) -> nn.Module:
        return self.model.bert.embeddings

    def layers(self) -> List[nn.Module]:
        return list(self.model.bert.encoder.layer)

    def head_module(self) -> nn.Module:
        return self._head

    def embed(self, module: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        return module(input_ids=input_ids, token_type_ids=torch.zeros_like(input_ids))

    def head(self, module: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        return module.classifier(module.dropout(module.pooler(hidden)))


ADAPTERS: Dict[str, Type[EncoderAdapter]] = {
    "distilbert": DistilBertAdapter,
    "bert": BertAdapter,
}

PLANNED_ADAPTERS = ("mobilebert", "vit")


def get_adapter(model: nn.Module) -> EncoderAdapter:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    adapter_cls = ADAPTERS.get(model_type)
    if adapter_cls is None:
        planned = " (planned, not implemented yet)" if model_type in PLANNED_ADAPTERS else ""
        raise NotImplementedError(
            f"No export adapter for model_type '{model_type}'{planned}. "
            f"Supported: {', '.join(sorted(ADAPTERS))}"
        )
    return adapter_cls(model)


def load_model(name_or_path: str) -> Tuple[nn.Module, Optional[Any]]:
    """Load a Hugging Face sequence classifier (eager attention) and its tokenizer."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    try:
        model = AutoModelForSequenceClassification.from_pretrained(name_or_path, attn_implementation="eager")
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(name_or_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    except Exception:
        tokenizer = None
    return model.eval(), tokenizer
