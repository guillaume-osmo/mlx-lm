"""
TurboQuant KV cache compression (experimental).

PolarQuant from "TurboQuant: Redefining AI Efficiency with Extreme
Compression" (Google, ICLR 2026, https://arxiv.org/abs/2504.19874).

Data-oblivious KV cache quantization at 2-4 bits per coordinate via
random orthogonal rotation followed by Lloyd-Max optimal scalar
quantization. No calibration data needed.
"""

import math
import os
from functools import lru_cache
from typing import Optional

import mlx.core as mx

from .cache import _BaseCache, create_attention_mask

# fmt: off
# Lloyd-Max optimal centroids and boundaries for N(0,1).
# Scaled by 1/sqrt(head_dim) at runtime.
_CENTROIDS = {
    1: [-0.7978845608, 0.7978845608],
    2: [-1.5104, -0.4528,  0.4528,  1.5104],
    3: [-2.1519, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1519],
    4: [-2.7331, -2.0698, -1.6189, -1.2570, -0.9431, -0.6573,
        -0.3884, -0.1285,  0.1285,  0.3884,  0.6573,  0.9431,
         1.2570,  1.6189,  2.0698,  2.7331],
}
_BOUNDARIES = {
    1: [-5.0, 0.0, 5.0],
    2: [-5.0, -0.9816, 0.0, 0.9816, 5.0],
    3: [-5.0, -1.7479, -1.0499, -0.5005, 0.0, 0.5005, 1.0499, 1.7479, 5.0],
    4: [-5.0, -2.4015, -1.8443, -1.4380, -1.1001, -0.8002,
        -0.5229, -0.2585,  0.0,    0.2585,  0.5229,  0.8002,
         1.1001,  1.4380,  1.8443,  2.4015, 5.0],
}
# fmt: on


def _rotation_matrix_dense(dim, seed=42):
    """Haar-distributed random orthogonal matrix via QR of Gaussian."""
    key = mx.random.key(seed)
    g = mx.random.normal(shape=(dim, dim), key=key)
    q, r = mx.linalg.qr(g, stream=mx.cpu)
    sign = mx.sign(mx.diag(r))
    sign = mx.where(sign == 0, 1, sign)
    return q * sign


def _rotation_blocks_rotor3(dim, seed=42):
    """Independent random 3x3 rotation blocks built from QR."""
    num_blocks = dim // 3
    if num_blocks == 0:
        return None

    key = mx.random.key(seed)
    blocks = []
    for _ in range(num_blocks):
        key, subkey = mx.random.split(key)
        g = mx.random.normal(shape=(3, 3), key=subkey)
        q, r = mx.linalg.qr(g, stream=mx.cpu)
        sign = mx.sign(mx.diag(r))
        sign = mx.where(sign == 0, 1, sign)
        blocks.append(q * sign)
    return mx.stack(blocks, axis=0)


def _rotation_blocks_rotorquant(dim, seed=42):
    """3D block rotations parameterized by normalized rotor/quaternion coeffs."""
    num_blocks = dim // 3
    if num_blocks == 0:
        return None

    key = mx.random.key(seed)
    rotor = mx.random.normal(shape=(num_blocks, 4), key=key)
    rotor = rotor / mx.maximum(mx.linalg.norm(rotor, axis=-1, keepdims=True), 1e-8)
    w = rotor[:, 0:1]
    x = rotor[:, 1:2]
    y = rotor[:, 2:3]
    z = rotor[:, 3:4]

    r0 = mx.concatenate(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
        ],
        axis=1,
    )
    r1 = mx.concatenate(
        [
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
        ],
        axis=1,
    )
    r2 = mx.concatenate(
        [
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=1,
    )
    return mx.stack([r0, r1, r2], axis=1)


def _apply_block_rotation(vectors, block_rotation):
    if block_rotation is None:
        return vectors
    num_blocks = block_rotation.shape[0]
    if num_blocks == 0:
        return vectors
    rotated_dim = num_blocks * 3
    prefix = vectors[..., :rotated_dim]
    reshaped = prefix.reshape(*vectors.shape[:-1], num_blocks, 3)
    rotated = mx.sum(mx.expand_dims(reshaped, axis=-1) * block_rotation, axis=-2)
    rotated = rotated.reshape(*vectors.shape[:-1], rotated_dim)
    if rotated_dim == vectors.shape[-1]:
        return rotated
    return mx.concatenate([rotated, vectors[..., rotated_dim:]], axis=-1)


def _rotation_matrix(dim, seed=42, mode="dense"):
    if mode == "dense":
        return _rotation_matrix_dense(dim, seed=seed)
    if mode == "rotor3":
        return _rotation_blocks_rotor3(dim, seed=seed)
    if mode == "rotorquant":
        return _rotation_blocks_rotorquant(dim, seed=seed)
    raise ValueError(f"Unsupported rotation_mode: {mode}")


@lru_cache(maxsize=32)
def _cached_rotation_pair(dim, seed=42, mode="dense"):
    rotation = _rotation_matrix(dim, seed=seed, mode=mode)
    if rotation is None:
        fallback = _rotation_matrix_dense(dim, seed=seed)
        return fallback, fallback.T
    if mode == "dense":
        return rotation, rotation.T
    return rotation, mx.swapaxes(rotation, -1, -2)


@lru_cache(maxsize=32)
def _cached_qjl_projection(dim, seed=123):
    key = mx.random.key(seed)
    projection = mx.random.normal(shape=(dim, dim), key=key).astype(mx.float32)
    return projection, projection.T


def _load_codebook(bits, dim):
    s = 1.0 / math.sqrt(dim)
    c = mx.array(_CENTROIDS[bits], dtype=mx.float32) * s
    b = mx.array(_BOUNDARIES[bits], dtype=mx.float32) * s
    return c, b


def _apply_rotation(vectors, rotation_t, mode="dense"):
    if rotation_t.ndim == 2:
        return vectors @ rotation_t
    return _apply_block_rotation(vectors, rotation_t)


def _apply_inverse_rotation(vectors, rotation, mode="dense"):
    if rotation.ndim == 2:
        return vectors @ rotation
    return _apply_block_rotation(vectors, rotation)


def _quantize_unit(vectors, rotation_t, boundaries, mode="dense"):
    rotated = _apply_rotation(vectors, rotation_t, mode=mode)
    inner = boundaries[1:-1]
    indices = mx.zeros(rotated.shape, dtype=mx.uint8)
    for b in range(inner.shape[0]):
        indices = indices + (rotated > inner[b]).astype(mx.uint8)
    return indices


def _dequantize_unit(indices, rotation, centroids, mode="dense"):
    rotated = centroids[indices]
    return _apply_inverse_rotation(rotated, rotation, mode=mode)


def _quantize(vectors, rotation_t, boundaries, mode="dense"):
    norms = mx.linalg.norm(vectors, axis=-1, keepdims=True)
    indices = _quantize_unit(
        vectors / mx.maximum(norms, 1e-8), rotation_t, boundaries, mode=mode
    )
    return indices, norms


def _dequantize(indices, norms, rotation, centroids, mode="dense"):
    return _dequantize_unit(indices, rotation, centroids, mode=mode) * norms


def _quantize_qjl_residual(unit_vectors, mse_vectors, projection_t):
    residual = unit_vectors - mse_vectors
    gamma = mx.linalg.norm(residual, axis=-1, keepdims=True)
    unit_residual = residual / mx.maximum(gamma, 1e-8)
    signs = (unit_residual @ projection_t >= 0).astype(mx.uint8)
    zero_mask = mx.broadcast_to(gamma <= 1e-8, signs.shape)
    signs = mx.where(zero_mask, mx.zeros_like(signs), signs)
    return signs, gamma


def _dequantize_qjl(signs, gamma, projection):
    signs_pm = mx.where(signs > 0, 1.0, -1.0).astype(mx.float32)
    scale = math.sqrt(math.pi / 2.0) / projection.shape[0]
    return scale * gamma.astype(mx.float32) * (signs_pm @ projection)


def _quantize_prod(
    vectors,
    rotation_t,
    rotation,
    boundaries,
    centroids,
    projection_t,
    mode="dense",
):
    norms = mx.linalg.norm(vectors, axis=-1, keepdims=True)
    unit_vectors = vectors / mx.maximum(norms, 1e-8)
    indices = _quantize_unit(unit_vectors, rotation_t, boundaries, mode=mode)
    mse_vectors = _dequantize_unit(indices, rotation, centroids, mode=mode)
    qjl_signs, gamma = _quantize_qjl_residual(unit_vectors, mse_vectors, projection_t)
    return indices, qjl_signs, gamma, norms


def _dequantize_prod(
    indices,
    norms,
    rotation,
    centroids,
    qjl_signs,
    gamma,
    projection,
    mode="dense",
):
    mse_vectors = _dequantize_unit(indices, rotation, centroids, mode=mode)
    correction = _dequantize_qjl(qjl_signs, gamma, projection)
    return (mse_vectors + correction) * norms.astype(mx.float32)


def _pack(indices, bits):
    """Pack b-bit indices into uint32."""
    shape = indices.shape
    dim = shape[-1]
    vpi = 32 // bits
    n_packed = (dim + vpi - 1) // vpi
    pad_size = n_packed * vpi - dim
    if pad_size > 0:
        indices = mx.concatenate(
            [indices, mx.zeros((*shape[:-1], pad_size), dtype=indices.dtype)],
            axis=-1,
        )
    reshaped = indices.reshape(*shape[:-1], n_packed, vpi).astype(mx.uint32)
    shifts = mx.arange(vpi, dtype=mx.uint32) * bits
    shifted = reshaped << shifts
    packed = shifted[..., 0]
    for i in range(1, vpi):
        packed = packed | shifted[..., i]
    return packed


def _unpack(packed, bits, dim):
    """Unpack uint32 back to b-bit indices."""
    shape = packed.shape
    vpi = 32 // bits
    mask = (1 << bits) - 1
    shifts = mx.arange(vpi, dtype=mx.uint32) * bits
    extracted = (packed[..., None] >> shifts) & mask
    return extracted.reshape(*shape[:-1], shape[-1] * vpi)[..., :dim].astype(mx.uint8)


class TurboQuantKVCache(_BaseCache):
    """KV cache compressed with PolarQuant (experimental).

    Data-oblivious compression: random orthogonal rotation maps KV vectors
    to coordinates with a known Gaussian distribution, then Lloyd-Max
    optimal scalar quantizers compress each coordinate independently.
    Bit-packed into uint32 for storage, dequantized on fetch.

    Args:
        bits (int): Bits per coordinate (2, 3, or 4). Default: ``4``.
    """

    step = 256

    def __init__(
        self,
        bits: int = 4,
        rotation_mode: str = "dense",
        estimator_mode: str = "mse",
        sparse_v_tau: Optional[float] = None,
    ):
        if bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4, got {bits}")
        if rotation_mode not in ("dense", "rotor3", "rotorquant"):
            raise ValueError(
                "rotation_mode must be 'dense', 'rotor3', or 'rotorquant', "
                f"got {rotation_mode}"
            )
        if estimator_mode not in ("mse", "prod"):
            raise ValueError(
                "estimator_mode must be 'mse' or 'prod', "
                f"got {estimator_mode}"
            )
        self.turbo_bits = bits
        self.rotation_mode = rotation_mode
        self.estimator_mode = estimator_mode
        self.sparse_v_tau = sparse_v_tau
        self.offset = 0
        self._head_dim = None
        self._k_indices = None
        self._k_norms = None
        self._k_qjl_indices = None
        self._k_qjl_gamma = None
        self._v_indices = None
        self._v_norms = None
        self._k_bits = None
        self._v_bits = None
        self._k_centroids = None
        self._k_boundaries = None
        self._v_centroids = None
        self._v_boundaries = None
        self._rotation = None
        self._rotation_t = None
        self._qjl_projection = None
        self._qjl_projection_t = None
        self._value_dtype = None
        self._fused_enabled = os.environ.get("MLX_TQ_FUSED", "0").lower() not in (
            "0",
            "false",
            "f",
            "no",
        )

    def _init_codebook(self, head_dim):
        self._head_dim = head_dim
        self._k_bits = self.turbo_bits if self.estimator_mode == "mse" else self.turbo_bits - 1
        self._v_bits = self.turbo_bits
        self._k_centroids, self._k_boundaries = _load_codebook(
            self._k_bits, head_dim
        )
        self._v_centroids, self._v_boundaries = _load_codebook(
            self._v_bits, head_dim
        )
        self._rotation, self._rotation_t = _cached_rotation_pair(
            head_dim, mode=self.rotation_mode
        )
        if self.estimator_mode == "prod":
            self._qjl_projection, self._qjl_projection_t = _cached_qjl_projection(
                head_dim
            )
        else:
            self._qjl_projection = None
            self._qjl_projection_t = None

    def _dequantize_keys(self, limit=None, *, include_qjl=True, dtype=None):
        if self._k_indices is None:
            return None
        limit = self.offset if limit is None else limit
        indices = _unpack(self._k_indices[..., :limit, :], self._k_bits, self._head_dim)
        norms = self._k_norms[..., :limit, :]
        if self.estimator_mode == "prod":
            qjl_signs = _unpack(
                self._k_qjl_indices[..., :limit, :], 1, self._head_dim
            )
            if include_qjl:
                keys = _dequantize_prod(
                    indices,
                    norms,
                    self._rotation,
                    self._k_centroids,
                    qjl_signs,
                    self._k_qjl_gamma[..., :limit, :],
                    self._qjl_projection,
                    mode=self.rotation_mode,
                )
            else:
                keys = _dequantize(
                    indices,
                    norms,
                    self._rotation,
                    self._k_centroids,
                    mode=self.rotation_mode,
                )
        else:
            keys = _dequantize(
                indices, norms, self._rotation, self._k_centroids, mode=self.rotation_mode
            )
        if dtype is not None:
            keys = keys.astype(dtype)
        return mx.contiguous(keys)

    def _dequantize_values(self, limit=None, *, dtype=None):
        if self._v_indices is None:
            return None
        limit = self.offset if limit is None else limit
        indices = _unpack(self._v_indices[..., :limit, :], self._v_bits, self._head_dim)
        values = _dequantize(
            indices,
            self._v_norms[..., :limit, :],
            self._rotation,
            self._v_centroids,
            mode=self.rotation_mode,
        )
        if dtype is not None:
            values = values.astype(dtype)
        return mx.contiguous(values)

    def update_and_fetch(self, keys, values):
        B, n_kv_heads, num_steps, head_dim = keys.shape
        prev = self.offset
        self._value_dtype = values.dtype

        if self._k_centroids is None:
            self._init_codebook(head_dim)

        if self.estimator_mode == "prod":
            k_idx, k_qjl, k_gamma, k_norms = _quantize_prod(
                keys,
                self._rotation_t,
                self._rotation,
                self._k_boundaries,
                self._k_centroids,
                self._qjl_projection_t,
                mode=self.rotation_mode,
            )
            pk_qjl = _pack(k_qjl, 1)
            k_gamma = k_gamma.astype(mx.float16)
        else:
            k_idx, k_norms = _quantize(
                keys, self._rotation_t, self._k_boundaries, mode=self.rotation_mode
            )
            pk_qjl = None
            k_gamma = None
        v_idx, v_norms = _quantize(
            values, self._rotation_t, self._v_boundaries, mode=self.rotation_mode
        )
        pk = _pack(k_idx, self._k_bits)
        pv = _pack(v_idx, self._v_bits)
        k_norms = k_norms.astype(mx.float16)
        v_norms = v_norms.astype(mx.float16)

        if self._k_indices is None or (prev + num_steps) > self._k_indices.shape[2]:
            self._expand(
                B,
                n_kv_heads,
                num_steps,
                pk.shape[-1],
                pv.shape[-1],
                0 if pk_qjl is None else pk_qjl.shape[-1],
            )

        self._k_indices[..., prev : prev + num_steps, :] = pk
        self._k_norms[..., prev : prev + num_steps, :] = k_norms
        if pk_qjl is not None:
            self._k_qjl_indices[..., prev : prev + num_steps, :] = pk_qjl
            self._k_qjl_gamma[..., prev : prev + num_steps, :] = k_gamma
        self._v_indices[..., prev : prev + num_steps, :] = pv
        self._v_norms[..., prev : prev + num_steps, :] = v_norms
        self.offset += num_steps

        can_use_fused_decode = self._fused_enabled and B == 1 and num_steps == 1
        can_use_fused_av = can_use_fused_decode and hasattr(
            mx.fast, "turboquant_av_packed_values_batched"
        ) and self.estimator_mode == "mse"
        can_use_fused_qk = can_use_fused_decode and (
            (
                self.estimator_mode == "mse"
                and hasattr(mx.fast, "turboquant_qk_packed_scores")
            )
            or (
                self.estimator_mode == "prod"
                and hasattr(mx.fast, "turboquant_qk_prod_scores_batched")
            )
        )

        all_v = None
        if not (can_use_fused_qk and can_use_fused_av):
            all_v = self._dequantize_values(dtype=values.dtype)
        all_k = None if can_use_fused_qk else self._dequantize_keys(dtype=keys.dtype)
        return all_k, all_v

    def fused_scores(self, queries_scaled):
        """Compute QK scores from packed keys using fused native MLX op.

        Args:
            queries_scaled: [B, n_q_heads, L, D], already scaled by 1/sqrt(D).

        Returns:
            scores [B, n_q_heads, L, T] or None if unsupported.
        """
        if (
            not self._fused_enabled
            or self._k_indices is None
            or self.offset <= 0
            or queries_scaled.ndim != 4
        ):
            return None

        B, n_q_heads, L, _ = queries_scaled.shape
        n_kv_heads = self._k_indices.shape[1]
        if B != self._k_indices.shape[0]:
            return None
        if n_q_heads % n_kv_heads != 0:
            return None

        n_repeats = n_q_heads // n_kv_heads
        q_rot = _apply_rotation(
            queries_scaled, self._rotation_t, mode=self.rotation_mode
        )

        if self.estimator_mode == "prod":
            if not hasattr(mx.fast, "turboquant_qk_prod_scores_batched"):
                return None
            return mx.fast.turboquant_qk_prod_scores_batched(
                mx.contiguous(q_rot.astype(mx.float32)),
                mx.contiguous(queries_scaled.astype(mx.float32)),
                mx.contiguous(self._k_indices[..., : self.offset, :]),
                mx.contiguous(self._k_norms[..., : self.offset, 0].astype(mx.float32)),
                self._k_centroids,
                self._k_bits,
                mx.contiguous(self._k_qjl_indices[..., : self.offset, :]),
                mx.contiguous(self._k_qjl_gamma[..., : self.offset, 0].astype(mx.float32)),
                self._qjl_projection,
                n_repeats,
            )

        if not hasattr(mx.fast, "turboquant_qk_packed_scores"):
            return None

        if hasattr(mx.fast, "turboquant_qk_packed_scores_batched"):
            # Materialize stable contiguous inputs for the native batched op.
            q_in = mx.contiguous(q_rot.astype(mx.float32))
            kp_in = mx.contiguous(self._k_indices[..., : self.offset, :])
            kn_in = mx.contiguous(self._k_norms[..., : self.offset, 0].astype(mx.float32))
            c_in = self._k_centroids
            return mx.fast.turboquant_qk_packed_scores_batched(
                q_in,
                kp_in,
                kn_in,
                c_in,
                self._k_bits,
                n_repeats,
            )

        per_batch = []
        for b in range(B):
            per_head = []
            for hq in range(n_q_heads):
                hkv = hq // n_repeats
                s = mx.fast.turboquant_qk_packed_scores(
                    q_rot[b, hq],  # [L, D]
                    self._k_indices[b, hkv, : self.offset, :],  # [T, W]
                    self._k_norms[b, hkv, : self.offset, 0],  # [T]
                    self._k_centroids,
                    self._k_bits,
                )  # [L, T]
                per_head.append(s)
            per_batch.append(mx.stack(per_head, axis=0))
        return mx.stack(per_batch, axis=0)  # [B, n_q_heads, L, T]

    def fused_attention(self, queries_scaled):
        """Compute packed decode attention in one native op when supported."""
        if (
            not self._fused_enabled
            or self.estimator_mode != "mse"
            or not hasattr(mx.fast, "turboquant_decode_attention_packed_batched")
            or self._k_indices is None
            or self._v_indices is None
            or self.offset <= 0
            or queries_scaled.ndim != 4
        ):
            return None

        B, n_q_heads, _, _ = queries_scaled.shape
        n_kv_heads = self._k_indices.shape[1]
        if (
            B != self._k_indices.shape[0]
            or B != self._v_indices.shape[0]
            or n_q_heads % n_kv_heads != 0
        ):
            return None

        n_repeats = n_q_heads // n_kv_heads
        q_rot = _apply_rotation(
            queries_scaled, self._rotation_t, mode=self.rotation_mode
        )
        out = mx.fast.turboquant_decode_attention_packed_batched(
            mx.contiguous(q_rot.astype(mx.float32)),
            mx.contiguous(self._k_indices[..., : self.offset, :]),
            mx.contiguous(self._k_norms[..., : self.offset, 0].astype(mx.float32)),
            mx.contiguous(self._v_indices[..., : self.offset, :]),
            mx.contiguous(self._v_norms[..., : self.offset, 0].astype(mx.float32)),
            self._k_centroids,
            self._k_bits,
            n_repeats,
            self._head_dim,
        )
        out = _apply_inverse_rotation(out, self._rotation, mode=self.rotation_mode)
        if self._value_dtype is not None:
            out = out.astype(self._value_dtype)
        return out

    def fused_av(self, probs):
        """Compute attention output from packed values using native MLX op.

        Args:
            probs: [B, n_q_heads, L, T] attention probabilities.

        Returns:
            out [B, n_q_heads, L, D] or None if unsupported.
        """
        if (
            not self._fused_enabled
            or self.estimator_mode != "mse"
            or not hasattr(mx.fast, "turboquant_av_packed_values_batched")
            or self._v_indices is None
            or self.offset <= 0
            or probs.ndim != 4
        ):
            return None

        B, n_q_heads, _, T = probs.shape
        n_kv_heads = self._v_indices.shape[1]
        if B != self._v_indices.shape[0] or T != self.offset:
            return None
        if n_q_heads % n_kv_heads != 0:
            return None

        n_repeats = n_q_heads // n_kv_heads
        p_in = mx.contiguous(probs.astype(mx.float32))
        vp_in = mx.contiguous(self._v_indices[..., : self.offset, :])
        vn_in = mx.contiguous(self._v_norms[..., : self.offset, 0].astype(mx.float32))
        c_in = self._v_centroids
        out = mx.fast.turboquant_av_packed_values_batched(
            p_in,
            vp_in,
            vn_in,
            c_in,
            self._v_bits,
            n_repeats,
            self._head_dim,
        )
        out = _apply_inverse_rotation(out, self._rotation, mode=self.rotation_mode)
        if self._value_dtype is not None:
            out = out.astype(self._value_dtype)
        return out

    def _expand(self, B, n_kv_heads, new_steps, k_packed_dim, v_packed_dim, qjl_packed_dim):
        alloc = ((self.step + new_steps - 1) // self.step) * self.step
        shape = (B, n_kv_heads, alloc)

        def _new():
            arrays = [
                mx.zeros((*shape, k_packed_dim), dtype=mx.uint32),
                mx.zeros((*shape, 1), dtype=mx.float16),
                mx.zeros((*shape, v_packed_dim), dtype=mx.uint32),
                mx.zeros((*shape, 1), dtype=mx.float16),
            ]
            if self.estimator_mode == "prod":
                arrays.extend(
                    [
                        mx.zeros((*shape, qjl_packed_dim), dtype=mx.uint32),
                        mx.zeros((*shape, 1), dtype=mx.float16),
                    ]
                )
            return arrays

        if self._k_indices is not None and self.offset > 0:
            old = [
                self._k_indices[..., :self.offset, :],
                self._k_norms[..., :self.offset, :],
                self._v_indices[..., :self.offset, :],
                self._v_norms[..., :self.offset, :],
            ]
            if self.estimator_mode == "prod":
                old.extend(
                    [
                        self._k_qjl_indices[..., :self.offset, :],
                        self._k_qjl_gamma[..., :self.offset, :],
                    ]
                )
            arrays = [mx.concatenate([o, n], axis=2) for o, n in zip(old, _new())]
        else:
            arrays = _new()

        self._k_indices = arrays[0]
        self._k_norms = arrays[1]
        self._v_indices = arrays[2]
        self._v_norms = arrays[3]
        if self.estimator_mode == "prod":
            self._k_qjl_indices = arrays[4]
            self._k_qjl_gamma = arrays[5]
        else:
            self._k_qjl_indices = None
            self._k_qjl_gamma = None

    def size(self):
        return self.offset

    @property
    def state(self):
        if self._k_indices is None:
            return []
        state = [
            self._k_indices[..., :self.offset, :],
            self._k_norms[..., :self.offset, :],
            self._v_indices[..., :self.offset, :],
            self._v_norms[..., :self.offset, :],
        ]
        if self.estimator_mode == "prod":
            state.extend(
                [
                    self._k_qjl_indices[..., :self.offset, :],
                    self._k_qjl_gamma[..., :self.offset, :],
                ]
            )
        return state

    @state.setter
    def state(self, v):
        if v is not None and v:
            self._k_indices, self._k_norms, self._v_indices, self._v_norms = v[:4]
            if len(v) >= 6:
                self._k_qjl_indices, self._k_qjl_gamma = v[4:6]
            else:
                self._k_qjl_indices = None
                self._k_qjl_gamma = None
            self.offset = self._k_indices.shape[2]

    @property
    def meta_state(self):
        head_dim = self._head_dim or 0
        sparse_tau = -1.0 if self.sparse_v_tau is None else float(self.sparse_v_tau)
        return tuple(
            map(
                str,
                (
                    self.offset,
                    self.turbo_bits,
                    head_dim,
                    self.rotation_mode,
                    sparse_tau,
                    self.estimator_mode,
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, v):
        self.offset, self.turbo_bits = int(v[0]), int(v[1])
        head_dim = int(v[2])
        self.rotation_mode = v[3] if len(v) >= 4 else "dense"
        if self.rotation_mode not in ("dense", "rotor3", "rotorquant"):
            self.rotation_mode = "dense"
        if len(v) >= 5:
            sparse_tau = float(v[4])
            self.sparse_v_tau = None if sparse_tau < 0 else sparse_tau
        else:
            self.sparse_v_tau = None
        self.estimator_mode = v[5] if len(v) >= 6 else "mse"
        if self.estimator_mode not in ("mse", "prod"):
            self.estimator_mode = "mse"
        self._fused_enabled = os.environ.get("MLX_TQ_FUSED", "0").lower() not in (
            "0",
            "false",
            "f",
            "no",
        )
        if head_dim > 0:
            self._init_codebook(head_dim)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self._k_indices is None

    @property
    def nbytes(self):
        if self._k_indices is None:
            return 0
        arrays = [self._k_indices, self._k_norms, self._v_indices, self._v_norms]
        if self._k_qjl_indices is not None:
            arrays.extend([self._k_qjl_indices, self._k_qjl_gamma])
        return sum(a[..., :self.offset, :].nbytes for a in arrays)
