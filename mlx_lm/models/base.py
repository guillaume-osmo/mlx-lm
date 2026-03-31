# Copyright © 2023-2024 Apple Inc.

import inspect
import os
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
from mlx.utils import tree_map


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def create_causal_mask(
    N: int,
    offset: int = 0,
    window_size: Optional[int] = None,
    right_padding: Optional[mx.array] = None,
    left_padding: Optional[mx.array] = None,
):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    if right_padding is not None:
        mask = mask & (rinds < mx.expand_dims((offset + N) - right_padding, (1, 2, 3)))
    if left_padding is not None:
        mask = mask & (mx.expand_dims(left_padding, (1, 2, 3)) <= rinds)
    return mask


def create_attention_mask(
    h, cache=None, window_size: Optional[int] = None, return_array: bool = False
):
    N = h.shape[1]
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(N, return_array=return_array, window_size=window_size)
    if N == 1:
        return None
    if return_array or (window_size and N > window_size):
        return create_causal_mask(N, window_size=window_size)
    return "causal"


def create_ssm_mask(h, cache=None):
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(h.shape[1])
    return None


def quantized_scaled_dot_product_attention(
    queries: mx.array,
    q_keys: tuple[mx.array, mx.array, mx.array],
    q_values: tuple[mx.array, mx.array, mx.array],
    scale: float,
    mask: Optional[mx.array],
    group_size: int = 64,
    bits: int = 8,
) -> mx.array:
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    queries *= scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    scores = mx.quantized_matmul(
        queries, *q_keys, transpose=True, group_size=group_size, bits=bits
    )
    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores += mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(
        scores, *q_values, transpose=False, group_size=group_size, bits=bits
    )

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))

    return out




def _resolve_turbo_sparse_v_tau(cache) -> Optional[float]:
    tau = getattr(cache, "sparse_v_tau", None)
    if tau is not None:
        return float(tau)
    tau_env = os.environ.get("MLX_TQ_SPARSE_V_TAU")
    if tau_env is None:
        return None
    try:
        return float(tau_env)
    except ValueError:
        return None


def _resolve_turbo_min_fused_tokens(cache) -> int:
    threshold = getattr(cache, "min_fused_tokens", None)
    if threshold is not None:
        try:
            return max(0, int(threshold))
        except (TypeError, ValueError):
            return 0
    raw = os.environ.get("MLX_TQ_MIN_FUSED_TOKENS")
    if raw is None:
        return 256
    try:
        return max(0, int(raw))
    except ValueError:
        return 256


def _apply_turbo_mask(scores, mask, cache, n_compressed: int = 0):
    if mask is None:
        return scores
    if isinstance(mask, str):
        qL, kL = scores.shape[-2:]
        offset = max(0, getattr(cache, "offset", kL) - qL)
        q_indices = mx.arange(offset, offset + qL)
        k_indices = mx.arange(kL)
        mask = q_indices[:, None] >= k_indices[None]
    if mask.dtype == mx.bool_:
        if n_compressed > 0 and mask.shape[-1] == scores.shape[-1] - n_compressed:
            prefix = mx.ones((*mask.shape[:-1], n_compressed), dtype=mx.bool_)
            mask = mx.concatenate([prefix, mask], axis=-1)
        scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
    else:
        if n_compressed > 0 and mask.shape[-1] == scores.shape[-1] - n_compressed:
            prefix = mx.zeros((*mask.shape[:-1], n_compressed), dtype=mask.dtype)
            mask = mx.concatenate([prefix, mask], axis=-1)
        scores += mask
    return scores


def _matmul_buffer_values(probs, values):
    n_q_heads = probs.shape[1]
    n_kv_heads = values.shape[1]
    if n_q_heads % n_kv_heads == 0 and n_q_heads != n_kv_heads:
        n_repeats = n_q_heads // n_kv_heads
        probs = mx.reshape(
            probs,
            (
                probs.shape[0],
                n_kv_heads,
                n_repeats,
                probs.shape[2],
                probs.shape[3],
            ),
        )
        out = probs @ mx.expand_dims(values, axis=2)
        return mx.reshape(
            out,
            (
                out.shape[0],
                n_q_heads,
                out.shape[3],
                out.shape[4],
            ),
        )
    return probs @ values


def _fast_scaled_dot_product_attention(queries, keys, values, *, scale, mask, sinks=None):
    kwargs = dict(scale=scale, mask=mask)
    if sinks is not None:
        try:
            return mx.fast.scaled_dot_product_attention(
                queries,
                keys,
                values,
                sinks=sinks,
                **kwargs,
            )
        except TypeError as e:
            raise TypeError(
                "This MLX runtime does not support attention sinks in "
                "mx.fast.scaled_dot_product_attention()."
            ) from e
    return mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        **kwargs,
    )

def scaled_dot_product_attention(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: Optional[mx.array],
    sinks: Optional[mx.array] = None,
) -> mx.array:
    sparse_v_tau = _resolve_turbo_sparse_v_tau(cache)

    # TurboQuant packed decode attention: QK + softmax + AV in one native op.
    if hasattr(cache, "fused_attention") and keys is None and values is None:
        if sinks is not None:
            raise ValueError("TurboQuant fused SDPA does not support attention sinks.")
        if mask is None and (sparse_v_tau is None or sparse_v_tau <= 0):
            out = cache.fused_attention(queries * scale)
            if out is not None:
                return out

    # TurboQuant fused path: score directly from packed keys to avoid full key dequantization.
    if hasattr(cache, "fused_scores") and (
        keys is None
        or values is None
        or getattr(cache, "buffer_tokens", 0) > 0
    ):
        if sinks is not None:
            raise ValueError("TurboQuant fused SDPA does not support attention sinks.")
        queries_scaled = queries * scale
        compressed_tokens = int(getattr(cache, "compressed_tokens", 0) or 0)
        if (
            compressed_tokens > 0
            and compressed_tokens < _resolve_turbo_min_fused_tokens(cache)
            and hasattr(cache, "_dequantize_keys")
            and hasattr(cache, "_dequantize_values")
        ):
            full_keys = cache._dequantize_keys(dtype=queries.dtype)
            full_values = cache._dequantize_values(dtype=values.dtype if values is not None else queries.dtype)
            return _fast_scaled_dot_product_attention(
                queries,
                full_keys,
                full_values,
                scale=scale,
                mask=mask,
                sinks=sinks,
            )
        scores = cache.fused_scores(queries_scaled)
        if scores is not None:
            n_compressed = scores.shape[-1]
            buffer_scores = None
            if keys is not None and values is not None and keys.shape[-2] > 0:
                n_q_heads = queries.shape[1]
                n_kv_heads = keys.shape[1]
                if n_q_heads % n_kv_heads == 0 and n_q_heads != n_kv_heads:
                    n_repeats = n_q_heads // n_kv_heads
                    buf_keys = mx.repeat(keys, n_repeats, axis=1)
                else:
                    buf_keys = keys
                buffer_scores = (queries @ mx.transpose(buf_keys, (0, 1, 3, 2))) * scale
                scores = mx.concatenate([scores, buffer_scores], axis=-1)

            scores = _apply_turbo_mask(scores, mask, cache, n_compressed if buffer_scores is not None else 0)
            probs = mx.softmax(scores, axis=-1, precise=True)
            if sparse_v_tau is not None and sparse_v_tau > 0:
                probs = mx.where(probs >= sparse_v_tau, probs, 0.0)
                probs = probs / mx.maximum(mx.sum(probs, axis=-1, keepdims=True), 1e-8)

            if buffer_scores is not None:
                comp_probs = probs[..., :n_compressed]
                buf_probs = probs[..., n_compressed:]
                out = None
                if hasattr(cache, "fused_av"):
                    out = cache.fused_av(comp_probs)
                if out is None and hasattr(cache, "_dequantize_values"):
                    comp_values = cache._dequantize_values(limit=n_compressed, dtype=values.dtype)
                    out = _matmul_buffer_values(comp_probs, comp_values)
                if out is None:
                    raise ValueError("TurboQuant fused AV path unavailable for compressed-buffer decode.")
                return out + _matmul_buffer_values(buf_probs, values)

            if values is None and hasattr(cache, "fused_av"):
                out = cache.fused_av(probs)
                if out is not None:
                    return out

            if values is None:
                raise ValueError("TurboQuant fused AV path unavailable for this decode step.")

            return _matmul_buffer_values(probs, values)

    if hasattr(cache, "bits"):
        if sinks is not None:
            raise ValueError("Quantized SDPA does not support attention sinks.")
        return quantized_scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            group_size=cache.group_size,
            bits=cache.bits,
        )

    return _fast_scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=scale,
        mask=mask,
        sinks=sinks,
    )
