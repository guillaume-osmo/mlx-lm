"""TriAttention: Trigonometric KV Cache Compression.

Ported from blaizzy/mlx-vlm PR #985, based on "TriAttention: Efficient Long
Reasoning with Trigonometric KV Compression" (Lin et al., 2026,
arXiv:2604.04921).

Scores key importance using trigonometric series derived from pre-RoPE Q/K
concentration, then prunes low-importance tokens from the KV cache.
Post-RoPE keys are scored directly — no inverse RoPE needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .cache import _BaseCache, KVCache, RotatingKVCache, CacheList

# ──────────────────────────── defaults ────────────────────────────

DEFAULT_BUDGET = 2048
DEFAULT_DIVIDE_LENGTH = 128
DEFAULT_PROTECT_RECENT = 128
DEFAULT_PROTECT_INITIAL = 4
_DEFAULT_OFFSETS = mx.array([2**i for i in range(17)], dtype=mx.float32)


# ──────────────────────────── data classes ────────────────────────


@dataclass
class RoPEConfig:
    """RoPE configuration extracted from a model's attention layer."""

    head_dim: int
    rotated_dims: int
    traditional: bool
    omega: mx.array  # [n_freqs] angular frequencies
    proportional: bool = False


@dataclass
class TriAttentionCalibData:
    """Per-layer, per-head calibration statistics."""

    q_center_real: Dict[int, mx.array]  # layer -> [n_q_heads, n_freqs]
    q_center_imag: Dict[int, mx.array]
    q_mean_norm: Dict[int, mx.array]
    n_layers: int
    n_q_heads: int
    n_kv_heads: int


# ──────────────────────────── RoPE extraction ────────────────────


def _find_layers(model: nn.Module) -> Optional[list]:
    """Find the transformer layer list."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    return None


def _find_attention(layer: nn.Module) -> Optional[nn.Module]:
    return getattr(layer, "self_attn", None) or getattr(
        layer, "attention", None
    )


def _get_head_dim(attn: nn.Module) -> Optional[int]:
    hd = getattr(attn, "head_dim", None)
    if hd is not None:
        return hd
    n_heads = getattr(attn, "n_heads", None) or getattr(
        attn, "num_heads", None
    )
    if n_heads and hasattr(attn, "q_proj") and hasattr(attn.q_proj, "weight"):
        return attn.q_proj.weight.shape[0] // n_heads
    return None


def _compute_omega_standard(dims: int, base: float, scale: float) -> mx.array:
    exponents = mx.arange(0, dims, 2, dtype=mx.float32) / dims
    return (1.0 / (base**exponents)) / scale


def extract_rope_config(model: nn.Module) -> Optional[RoPEConfig]:
    """Extract RoPE configuration from a language model."""
    layers = _find_layers(model)
    if not layers:
        return None

    target_layer = layers[0]
    for layer in layers:
        attn = _find_attention(layer)
        if attn is not None and not getattr(attn, "is_sliding", False):
            target_layer = layer
            break

    attn = _find_attention(target_layer)
    if attn is None:
        return None

    rope = getattr(attn, "rope", None)
    if rope is None:
        return None

    head_dim = _get_head_dim(attn)
    if head_dim is None:
        return None

    if isinstance(rope, nn.RoPE):
        dims = rope.dims
        omega = _compute_omega_standard(dims, rope.base, rope.scale)
        return RoPEConfig(
            head_dim=head_dim,
            rotated_dims=dims,
            traditional=rope.traditional,
            omega=omega,
        )

    if hasattr(rope, "_freqs") and hasattr(rope, "rotated_dims"):
        omega = 1.0 / rope._freqs
        return RoPEConfig(
            head_dim=head_dim,
            rotated_dims=rope.rotated_dims,
            traditional=rope.traditional,
            omega=omega,
            proportional=True,
        )

    return None


def extract_model_info(
    model: nn.Module,
) -> Optional[Tuple[int, int, int, int, RoPEConfig]]:
    """Extract (n_layers, n_q_heads, n_kv_heads, head_dim, rope_config)."""
    layers = _find_layers(model)
    if not layers:
        return None

    n_layers = len(layers)

    attn = None
    for layer in layers:
        candidate = _find_attention(layer)
        if candidate is not None and not getattr(
            candidate, "is_sliding", False
        ):
            attn = candidate
            break
    if attn is None:
        attn = _find_attention(layers[0])
    if attn is None:
        return None

    n_q_heads = getattr(attn, "n_heads", None) or getattr(
        attn, "num_heads", None
    )
    n_kv_heads = (
        getattr(attn, "n_kv_heads", None)
        or getattr(attn, "num_key_value_heads", None)
        or n_q_heads
    )
    head_dim = _get_head_dim(attn)

    if n_q_heads is None or head_dim is None:
        return None

    rope_config = extract_rope_config(model)
    if rope_config is None:
        return None

    return n_layers, n_q_heads, n_kv_heads, head_dim, rope_config


# ──────────────────────────── scoring ─────────────────────────────


def _decompose_complex(
    vectors: mx.array, config: RoPEConfig
) -> Tuple[mx.array, mx.array]:
    """Decompose vectors into (real, imag) per frequency band."""
    n_freqs = config.rotated_dims // 2

    if config.proportional:
        half = config.head_dim // 2
        rd_half = config.rotated_dims // 2
        portion = mx.concatenate(
            [vectors[..., :rd_half], vectors[..., half : half + rd_half]],
            axis=-1,
        )
        if config.traditional:
            real = portion[..., :n_freqs]
            imag = portion[..., n_freqs:]
        else:
            real = portion[..., 0::2]
            imag = portion[..., 1::2]
    else:
        if config.traditional:
            real = vectors[..., :n_freqs]
            imag = vectors[..., n_freqs : 2 * n_freqs]
        else:
            real = vectors[..., 0 : config.rotated_dims : 2]
            imag = vectors[..., 1 : config.rotated_dims : 2]

    return real, imag


def score_keys(
    cached_keys: mx.array,
    current_pos: int,
    calib: TriAttentionCalibData,
    layer_idx: int,
    rope_config: RoPEConfig,
    offsets: mx.array = _DEFAULT_OFFSETS,
) -> mx.array:
    """Score cached keys for importance using the trigonometric series.

    Args:
        cached_keys: [B, n_kv_heads, S, head_dim] post-RoPE keys
        current_pos: absolute position of the current token
        calib: calibration data with Q centers
        layer_idx: transformer layer index
        rope_config: RoPE configuration
        offsets: [n_offsets] future position offsets

    Returns:
        [B, n_kv_heads, S] importance score per key
    """
    B, H_kv, S, _ = cached_keys.shape

    k_real, k_imag = _decompose_complex(cached_keys, rope_config)
    k_mag = mx.sqrt(k_real * k_real + k_imag * k_imag + 1e-12)
    k_phase = mx.arctan2(k_imag, k_real)

    q_cr = calib.q_center_real[layer_idx]
    q_ci = calib.q_center_imag[layer_idx]
    q_mn = calib.q_mean_norm[layer_idx]

    q_center_mag = mx.sqrt(q_cr * q_cr + q_ci * q_ci + 1e-12)
    q_center_phase = mx.arctan2(q_ci, q_cr)

    G = calib.n_q_heads // calib.n_kv_heads
    n_freqs = rope_config.rotated_dims // 2
    q_center_mag = q_center_mag.reshape(H_kv, G, n_freqs)
    q_center_phase = q_center_phase.reshape(H_kv, G, n_freqs)
    q_mean_norm = q_mn.reshape(H_kv, G, n_freqs)

    omega = rope_config.omega

    phi = q_center_phase[None, :, None, :, :] - k_phase[:, :, :, None, :]
    amp = q_center_mag[None, :, None, :, :] * k_mag[:, :, :, None, :]

    a = amp * mx.cos(phi)
    b = amp * mx.sin(phi)

    t = (current_pos + offsets).astype(mx.float32)
    t_omega = t[:, None] * omega[None, :]
    cos_tw = mx.cos(t_omega)
    sin_tw = mx.sin(t_omega)

    flat_shape = (B * H_kv * S * G, n_freqs)
    s_trig_flat = a.reshape(flat_shape) @ cos_tw.T - b.reshape(flat_shape) @ sin_tw.T

    s_trig = mx.mean(s_trig_flat, axis=-1).reshape(B, H_kv, S, G)

    norm_weight = q_mean_norm - q_center_mag
    s_norm = mx.sum(
        norm_weight[None, :, None, :, :] * k_mag[:, :, :, None, :],
        axis=-1,
    )

    s = s_trig + s_norm

    if G > 1:
        mean_s = mx.mean(s, axis=2, keepdims=True)
        var_s = mx.mean((s - mean_s) ** 2, axis=2, keepdims=True)
        z = (s - mean_s) / mx.sqrt(var_s + 1e-8)
        scores = mx.max(z, axis=-1)
    else:
        scores = s.squeeze(-1)

    return scores


# ──────────────────────────── KV cache ────────────────────────────


class TriAttentionKVCache(_BaseCache):
    """KV cache with trigonometric-series-based token pruning.

    When the cache exceeds ``budget`` tokens, scores all keys via the
    TriAttention trigonometric series and retains only the top-scoring
    ones.  Compression triggers every ``divide_length`` generated tokens.
    """

    def __init__(
        self,
        budget: int = DEFAULT_BUDGET,
        calib: Optional[TriAttentionCalibData] = None,
        layer_idx: int = 0,
        rope_config: Optional[RoPEConfig] = None,
        divide_length: int = DEFAULT_DIVIDE_LENGTH,
        protect_recent: int = DEFAULT_PROTECT_RECENT,
        protect_initial: int = DEFAULT_PROTECT_INITIAL,
    ):
        self.budget = budget
        self.calib = calib
        self.layer_idx = layer_idx
        self.rope_config = rope_config
        self.divide_length = divide_length
        self.protect_recent = protect_recent
        self.protect_initial = protect_initial

        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset: int = 0
        self._tokens_since_compress: int = 0
        self._offsets = _DEFAULT_OFFSETS

    @classmethod
    def from_cache(
        cls,
        cache: Any,
        budget: int,
        calib: TriAttentionCalibData,
        layer_idx: int,
        rope_config: RoPEConfig,
        **kwargs,
    ) -> "TriAttentionKVCache":
        """Hot-swap from an existing KVCache."""
        inst = cls(
            budget=budget,
            calib=calib,
            layer_idx=layer_idx,
            rope_config=rope_config,
            **kwargs,
        )
        keys, values = cache.state
        if keys is not None:
            inst.keys = keys
            inst.values = values
            inst.offset = cache.offset
            inst._tokens_since_compress = cache.offset
        return inst

    @property
    def _physical_size(self) -> int:
        return self.keys.shape[2] if self.keys is not None else 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        n_new = keys.shape[2]

        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)

        self.offset += n_new
        self._tokens_since_compress += n_new

        if (
            self._physical_size > self.budget
            and self._tokens_since_compress >= self.divide_length
            and self.calib is not None
            and self.rope_config is not None
        ):
            self._compress()

        return self.keys, self.values

    def _compress(self):
        S = self._physical_size
        if S <= self.budget:
            return

        scores = score_keys(
            self.keys,
            self.offset,
            self.calib,
            self.layer_idx,
            self.rope_config,
            self._offsets,
        )

        avg_scores = mx.mean(scores, axis=1)

        if self.protect_initial > 0:
            avg_scores = mx.concatenate(
                [
                    mx.full(
                        (avg_scores.shape[0], self.protect_initial),
                        1e9,
                        dtype=avg_scores.dtype,
                    ),
                    avg_scores[:, self.protect_initial :],
                ],
                axis=1,
            )
        if self.protect_recent > 0 and S > self.protect_recent:
            avg_scores = mx.concatenate(
                [
                    avg_scores[:, : -self.protect_recent],
                    mx.full(
                        (avg_scores.shape[0], self.protect_recent),
                        1e9,
                        dtype=avg_scores.dtype,
                    ),
                ],
                axis=1,
            )

        keep_count = min(self.budget, S)
        keep_idx = mx.argpartition(-avg_scores[0], kth=keep_count - 1)[
            :keep_count
        ]
        keep_idx = mx.sort(keep_idx)

        self.keys = self.keys[:, :, keep_idx, :]
        self.values = self.values[:, :, keep_idx, :]
        self._tokens_since_compress = 0

        mx.eval(self.keys, self.values)

    @property
    def state(self) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        if self.keys is None:
            return None, None
        return self.keys, self.values

    @state.setter
    def state(self, v):
        if v is not None and len(v) == 2:
            self.keys, self.values = v

    @property
    def nbytes(self) -> int:
        total = 0
        if self.keys is not None:
            total += self.keys.nbytes
        if self.values is not None:
            total += self.values.nbytes
        return total

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self._physical_size, n)
        if n > 0 and self.keys is not None:
            self.keys = self.keys[:, :, n:, :]
            self.values = self.values[:, :, n:, :]
        self.offset -= n
        return n

    @property
    def meta_state(self):
        return tuple(
            map(str, (self.budget, self.offset, self._tokens_since_compress))
        )

    @meta_state.setter
    def meta_state(self, v):
        self.budget, self.offset, self._tokens_since_compress = map(int, v)


# ──────────────────────────── calibration I/O ─────────────────────


def save_calibration(calib: TriAttentionCalibData, path: str) -> None:
    """Save calibration data to safetensors."""
    import numpy as np

    tensors = {}
    for layer_idx in range(calib.n_layers):
        tensors[f"layer.{layer_idx}.q_center_real"] = mx.array(
            calib.q_center_real[layer_idx], dtype=mx.float32
        )
        tensors[f"layer.{layer_idx}.q_center_imag"] = mx.array(
            calib.q_center_imag[layer_idx], dtype=mx.float32
        )
        tensors[f"layer.{layer_idx}.q_mean_norm"] = mx.array(
            calib.q_mean_norm[layer_idx], dtype=mx.float32
        )

    metadata = {
        "n_layers": str(calib.n_layers),
        "n_q_heads": str(calib.n_q_heads),
        "n_kv_heads": str(calib.n_kv_heads),
    }
    mx.save_safetensors(path, tensors, metadata=metadata)


def load_calibration(path: str) -> TriAttentionCalibData:
    """Load calibration data from safetensors."""
    tensors, metadata = mx.load(path, return_metadata=True)
    if not isinstance(tensors, dict):
        raise ValueError(f"Expected dict from {path}")

    n_layers = int(metadata["n_layers"])
    n_q_heads = int(metadata["n_q_heads"])
    n_kv_heads = int(metadata["n_kv_heads"])

    q_center_real = {}
    q_center_imag = {}
    q_mean_norm = {}

    for i in range(n_layers):
        q_center_real[i] = tensors[f"layer.{i}.q_center_real"]
        q_center_imag[i] = tensors[f"layer.{i}.q_center_imag"]
        q_mean_norm[i] = tensors[f"layer.{i}.q_mean_norm"]

    return TriAttentionCalibData(
        q_center_real=q_center_real,
        q_center_imag=q_center_imag,
        q_mean_norm=q_mean_norm,
        n_layers=n_layers,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
    )


# ──────────────────────────── generation integration ──────────────


def maybe_apply_triattention(
    prompt_cache: List[Any],
    model: nn.Module,
    calib_path: str,
    budget: int = DEFAULT_BUDGET,
    divide_length: int = DEFAULT_DIVIDE_LENGTH,
    protect_recent: int = DEFAULT_PROTECT_RECENT,
    protect_initial: int = DEFAULT_PROTECT_INITIAL,
) -> None:
    """Convert standard KVCache entries to TriAttentionKVCache in-place."""
    calib = load_calibration(calib_path)
    rope_config = extract_rope_config(model)
    if rope_config is None:
        raise ValueError(
            "TriAttention: could not extract RoPE config from model. "
            "This model may use an unsupported RoPE variant."
        )

    def convert_entry(entry, layer_idx):
        if isinstance(entry, TriAttentionKVCache):
            return entry
        if isinstance(entry, RotatingKVCache):
            return entry
        if isinstance(entry, KVCache):
            if entry.offset == 0:
                return TriAttentionKVCache(
                    budget=budget,
                    calib=calib,
                    layer_idx=layer_idx,
                    rope_config=rope_config,
                    divide_length=divide_length,
                    protect_recent=protect_recent,
                    protect_initial=protect_initial,
                )
            return TriAttentionKVCache.from_cache(
                entry,
                budget=budget,
                calib=calib,
                layer_idx=layer_idx,
                rope_config=rope_config,
                divide_length=divide_length,
                protect_recent=protect_recent,
                protect_initial=protect_initial,
            )
        if isinstance(entry, CacheList):
            entry.caches = [
                convert_entry(sub, layer_idx) for sub in entry.caches
            ]
            return entry
        if isinstance(entry, list):
            for i, sub in enumerate(entry):
                entry[i] = convert_entry(sub, layer_idx)
            return entry
        return entry

    for layer_idx in range(len(prompt_cache)):
        prompt_cache[layer_idx] = convert_entry(
            prompt_cache[layer_idx], layer_idx
        )
