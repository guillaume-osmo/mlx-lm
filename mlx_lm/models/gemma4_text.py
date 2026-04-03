# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: Union[int, List[int]]
    num_attention_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    num_kv_shared_layers: int
    vocab_size_per_layer_input: int
    hidden_size_per_layer_input: int
    sliding_window: int
    max_position_embeddings: int
    rope_parameters: Dict[str, Dict]
    layer_types: List[str]
    final_logit_softcapping: Optional[float]
    global_head_dim: Optional[int] = None
    num_global_key_value_heads: Optional[int] = None
    attention_k_eq_v: bool = False
    attention_bias: bool = False
    use_double_wide_mlp: bool = False


class RMSNoScale(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, None, self.eps)


class MLP(nn.Module):
    def __init__(
        self, config: ModelArgs, layer_idx: int, is_kv_shared_layer: bool = False
    ):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = (
            config.intermediate_size[layer_idx]
            if isinstance(config.intermediate_size, list)
            else config.intermediate_size
        )
        if config.use_double_wide_mlp and is_kv_shared_layer:
            intermediate_size *= 2
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int, is_kv_shared_layer: bool):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.is_kv_shared_layer = is_kv_shared_layer

        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = (
            config.head_dim
            if self.is_sliding or not config.global_head_dim
            else config.global_head_dim
        )
        self.n_kv_heads = (
            config.num_global_key_value_heads
            if (
                not self.is_sliding
                and config.attention_k_eq_v
                and config.num_global_key_value_heads is not None
            )
            else config.num_key_value_heads
        )
        self.scale = 1.0
        self.use_alternative_attention = config.attention_k_eq_v and not self.is_sliding

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.n_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = (
            None
            if self.use_alternative_attention
            else nn.Linear(
                self.hidden_size,
                self.n_kv_heads * self.head_dim,
                bias=config.attention_bias,
            )
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNoScale(eps=config.rms_norm_eps)

        rope_params = config.rope_parameters[self.layer_type]
        self.rope = initialize_rope(
            dims=self.head_dim,
            base=rope_params["rope_theta"],
            traditional=False,
            max_position_embeddings=config.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        batch, seq_len, _ = x.shape
        queries = self.q_proj(x).reshape(batch, seq_len, self.n_heads, self.head_dim)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0

        if self.is_kv_shared_layer and cache is not None:
            cache_state = cache.state
            keys, values = cache_state[:2]
        else:
            keys = self.k_proj(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
            keys = self.k_norm(keys).transpose(0, 2, 1, 3)
            if self.v_proj is None:
                values = keys
            else:
                values = self.v_proj(x).reshape(
                    batch, seq_len, self.n_kv_heads, self.head_dim
                )
                values = self.v_norm(values).transpose(0, 2, 1, 3)

            keys = self.rope(keys, offset=offset)

            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

        queries = self.rope(queries, offset=offset)
        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return self.o_proj(output)


@partial(mx.compile, shapeless=True)
def clip_residual(x, y):
    if x.dtype != mx.float16:
        return x + y
    bound = mx.finfo(mx.float16).max
    return mx.clip(x.astype(mx.float32) + y.astype(mx.float32), -bound, bound).astype(
        mx.float16
    )


@partial(mx.compile, shapeless=True)
def logit_softcap(softcap, x):
    out = mx.tanh(x / softcap)
    return out * softcap


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int, is_kv_shared_layer: bool):
        super().__init__()
        self.self_attn = Attention(config, layer_idx, is_kv_shared_layer)
        self.mlp = MLP(config, layer_idx, is_kv_shared_layer=is_kv_shared_layer)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.per_layer_input_gate = nn.Linear(
            config.hidden_size,
            config.hidden_size_per_layer_input,
            bias=False,
        )
        self.per_layer_projection = nn.Linear(
            config.hidden_size_per_layer_input,
            config.hidden_size,
            bias=False,
        )
        self.post_per_layer_input_norm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.layer_scalar = mx.ones((1,))

    def __call__(
        self,
        x: mx.array,
        per_layer_input: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        attn = self.self_attn(self.input_layernorm(x), mask, cache)
        h = clip_residual(x, self.post_attention_layernorm(attn))

        ff = self.mlp(self.pre_feedforward_layernorm(h))
        h = clip_residual(h, self.post_feedforward_layernorm(ff))

        layer_input = self.per_layer_input_gate(h)
        layer_input = nn.gelu_approx(layer_input) * per_layer_input
        layer_input = self.per_layer_projection(layer_input)
        layer_input = self.post_per_layer_input_norm(layer_input)
        h = clip_residual(h, layer_input)

        return h * self.layer_scalar


class Gemma4Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.vocab_size_per_layer_input = config.vocab_size_per_layer_input
        self.final_logit_softcapping = config.final_logit_softcapping
        self.first_kv_shared_layer_idx = (
            config.num_hidden_layers - config.num_kv_shared_layers
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_tokens_per_layer = nn.Embedding(
            config.vocab_size_per_layer_input,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
        )
        self.per_layer_model_projection = nn.Linear(
            config.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
        )
        self.per_layer_projection_norm = nn.RMSNorm(
            config.hidden_size_per_layer_input,
            eps=config.rms_norm_eps,
        )
        self.layers = [
            TransformerBlock(
                config,
                layer_idx,
                is_kv_shared_layer=layer_idx >= self.first_kv_shared_layer_idx,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        concrete_layers = config.layer_types[: self.first_kv_shared_layer_idx]
        self.first_sliding_idx = concrete_layers.index("sliding_attention")
        self.first_full_idx = concrete_layers.index("full_attention")
        shared_full_idx = len(concrete_layers) - 1 - concrete_layers[::-1].index(
            "full_attention"
        )
        shared_sliding_idx = len(concrete_layers) - 1 - concrete_layers[::-1].index(
            "sliding_attention"
        )

        self.layer_idx_to_cache_idx = []
        for i, layer_type in enumerate(config.layer_types):
            if i < self.first_kv_shared_layer_idx:
                self.layer_idx_to_cache_idx.append(i)
            elif layer_type == "full_attention":
                self.layer_idx_to_cache_idx.append(shared_full_idx)
            else:
                self.layer_idx_to_cache_idx.append(shared_sliding_idx)

    def get_per_layer_inputs(self, input_ids: mx.array) -> mx.array:
        per_layer_inputs_mask = input_ids < self.vocab_size_per_layer_input
        tokens = mx.where(per_layer_inputs_mask, input_ids, mx.zeros_like(input_ids))
        result = self.embed_tokens_per_layer(tokens) * (
            self.hidden_size_per_layer_input**0.5
        )
        return result.reshape(
            *input_ids.shape,
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self,
        inputs_embeds: mx.array,
        per_layer_inputs: mx.array,
    ) -> mx.array:
        per_layer_projection = self.per_layer_model_projection(inputs_embeds) * (
            self.hidden_size**-0.5
        )
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
        return (per_layer_projection + per_layer_inputs) * (2.0**-0.5)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs) * (self.hidden_size**0.5)

        if inputs is None:
            raise ValueError("Gemma 4 text inference requires token ids.")

        per_layer_inputs = self.get_per_layer_inputs(inputs)
        per_layer_inputs = self.project_per_layer_inputs(h, per_layer_inputs)

        if cache is None:
            cache = [None] * self.first_kv_shared_layer_idx

        global_mask = create_attention_mask(h, cache[self.first_full_idx])
        sliding_mask = create_attention_mask(
            h,
            cache[self.first_sliding_idx],
            window_size=self.config.sliding_window,
        )

        for i, layer in enumerate(self.layers):
            layer_type = self.config.layer_types[i]
            mask = global_mask if layer_type == "full_attention" else sliding_mask
            h = layer(
                h,
                per_layer_inputs[:, :, i, :],
                mask=mask,
                cache=cache[self.layer_idx_to_cache_idx[i]],
            )

        out = self.norm(h)
        out = self.embed_tokens.as_linear(out)
        if self.final_logit_softcapping is not None:
            out = logit_softcap(self.final_logit_softcapping, out)
        return out

    def make_cache(self):
        caches = []
        for layer_type in self.config.layer_types[: self.first_kv_shared_layer_idx]:
            if layer_type == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.config.sliding_window))
        return caches


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Gemma4Model(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.model(inputs, cache=cache, input_embeddings=input_embeddings)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return self.model.make_cache()
