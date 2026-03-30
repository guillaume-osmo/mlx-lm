"""RotorQuant KV cache compression (experimental).

This implements a RotorQuant-style cache path where:

1. vectors are normalized,
2. embedded into Cl(3, 0) multivectors,
3. transformed with a per-group rotor sandwich,
4. quantized on the vector grades only,
5. optionally corrected with a 1-bit QJL residual on the key path.

The ``prod`` estimator follows Scrya's public reference more closely:
keys use ``bits - 1`` RotorQuant MSE plus a 1-bit QJL residual correction,
while values stay on the plain RotorQuant MSE path.
"""

import math
from functools import lru_cache
from typing import Optional

import mlx.core as mx

from .cache import _BaseCache, create_attention_mask
from .turboquant import (
    _apply_qjl_projection,
    _cached_qjl_projection,
    _dequantize_qjl,
    _fused_input,
    _load_codebook,
    _metal_available,
    _normalize_turbo_bits,
    _pack,
    _quantize_qjl_residual_packed,
    _resolve_qjl_projection_mode,
    _unpack,
)

_MV_DIM = 8
_VECTOR_SLICE = slice(1, 4)


def _normalize_rotor_bits(bits):
    bits = _normalize_turbo_bits(bits)
    if not isinstance(bits, int):
        raise ValueError(
            f"RotorQuant currently supports integer bit-widths only, got {bits}"
        )
    return bits


def _rotor_group_count(head_dim):
    return (head_dim + 2) // 3


def _rotor_padded_dim(head_dim):
    return _rotor_group_count(head_dim) * 3


@lru_cache(maxsize=64)
def _cached_rotors(num_groups, seed=42):
    key = mx.random.key(seed)
    key_biv, key_angle = mx.random.split(key)
    bivectors = mx.random.normal(shape=(num_groups, 3), key=key_biv).astype(mx.float32)
    bivector_norm = mx.maximum(
        mx.linalg.norm(bivectors, axis=-1, keepdims=True).astype(mx.float32),
        1e-8,
    )
    bivectors = bivectors / bivector_norm
    angles = (
        mx.random.uniform(shape=(num_groups, 1), key=key_angle).astype(mx.float32)
        * (2.0 * math.pi)
    )
    half_angles = 0.5 * angles
    scalar = mx.cos(half_angles)
    bivector = mx.sin(half_angles) * bivectors
    rotors = mx.concatenate([scalar, bivector], axis=-1).astype(mx.float32)
    rotor_norm = mx.maximum(
        mx.sqrt(mx.sum(rotors * rotors, axis=-1, keepdims=True)).astype(mx.float32),
        1e-8,
    )
    rotors = rotors / rotor_norm
    reverse = rotors * mx.array([1.0, -1.0, -1.0, -1.0], dtype=mx.float32)
    return rotors, reverse


def _embed_vectors_as_multivectors(vectors, padded_dim):
    pad = padded_dim - vectors.shape[-1]
    if pad > 0:
        vectors = mx.concatenate(
            [
                vectors,
                mx.zeros((*vectors.shape[:-1], pad), dtype=vectors.dtype),
            ],
            axis=-1,
        )
    grouped = vectors.reshape(*vectors.shape[:-1], padded_dim // 3, 3)
    mv = mx.zeros((*grouped.shape[:-2], grouped.shape[-2], _MV_DIM), dtype=vectors.dtype)
    mv[..., _VECTOR_SLICE] = grouped
    return mv


def _extract_vectors_from_multivectors(mv, head_dim):
    vectors = mx.stack([mv[..., 1], mv[..., 2], mv[..., 3]], axis=-1)
    vectors = vectors.reshape(*mv.shape[:-2], -1)
    return vectors[..., :head_dim]


def _gp_rotor_left(rotors, x):
    s = rotors[..., 0]
    p12 = rotors[..., 1]
    p13 = rotors[..., 2]
    p23 = rotors[..., 3]

    x0 = x[..., 0]
    x1 = x[..., 1]
    x2 = x[..., 2]
    x3 = x[..., 3]
    x4 = x[..., 4]
    x5 = x[..., 5]
    x6 = x[..., 6]
    x7 = x[..., 7]

    r0 = s * x0 - p12 * x4 - p13 * x5 - p23 * x6
    r1 = s * x1 + p12 * x2 + p13 * x3 + p23 * x7
    r2 = s * x2 - p12 * x1 + p23 * x3 - p13 * x7
    r3 = s * x3 - p13 * x1 - p23 * x2 + p12 * x7
    r4 = s * x4 + p12 * x0
    r5 = s * x5 + p13 * x0
    r6 = s * x6 + p23 * x0
    r7 = s * x7 - p23 * x1 + p13 * x2 - p12 * x3
    return mx.stack([r0, r1, r2, r3, r4, r5, r6, r7], axis=-1)


def _gp_mv_right(x, rotors):
    s = rotors[..., 0]
    p12 = rotors[..., 1]
    p13 = rotors[..., 2]
    p23 = rotors[..., 3]

    x0 = x[..., 0]
    x1 = x[..., 1]
    x2 = x[..., 2]
    x3 = x[..., 3]
    x4 = x[..., 4]
    x5 = x[..., 5]
    x6 = x[..., 6]
    x7 = x[..., 7]

    r0 = x0 * s - x4 * p12 - x5 * p13 - x6 * p23
    r1 = x1 * s - x2 * p12 - x3 * p13 + x7 * p23
    r2 = x2 * s + x1 * p12 - x3 * p23 - x7 * p13
    r3 = x3 * s + x1 * p13 + x2 * p23 + x7 * p12
    r4 = x0 * p12 + x4 * s + x5 * p23 - x6 * p13
    r5 = x0 * p13 + x5 * s - x4 * p23 + x6 * p12
    r6 = x0 * p23 + x6 * s + x4 * p13 - x5 * p12
    r7 = x7 * s + x1 * p23 - x2 * p13 + x3 * p12
    return mx.stack([r0, r1, r2, r3, r4, r5, r6, r7], axis=-1)


def _rotor_sandwich(rotors, mv):
    return _gp_mv_right(_gp_rotor_left(rotors, mv), rotors * mx.array([1.0, -1.0, -1.0, -1.0], dtype=rotors.dtype))


def _inverse_rotor_sandwich(rotors, rotor_reverse, mv):
    return _gp_mv_right(_gp_rotor_left(rotor_reverse, mv), rotors)


def _quantize_boundaries(values, boundaries):
    inner = boundaries[1:-1]
    indices = mx.zeros(values.shape, dtype=mx.uint8)
    for i in range(inner.shape[0]):
        indices = indices + (values > inner[i]).astype(mx.uint8)
    return indices


@lru_cache(maxsize=None)
def _rotor_encode_kernel():
    if not _metal_available():
        return None

    source = r"""
        auto row = thread_position_in_grid.x;
        auto group = thread_position_in_grid.y;

        if (row >= input_shape[0] || group >= NumGroups) {
            return;
        }

        const auto rotor_ptr = rotors + group * 4;
        float s = rotor_ptr[0];
        float p12 = rotor_ptr[1];
        float p13 = rotor_ptr[2];
        float p23 = rotor_ptr[3];

        int d0 = int(group) * 3;
        float x_mv[8] = {0.0f};
        if (d0 < OrigDim) {
            x_mv[1] = input[row * OrigDim + d0];
        }
        if (d0 + 1 < OrigDim) {
            x_mv[2] = input[row * OrigDim + d0 + 1];
        }
        if (d0 + 2 < OrigDim) {
            x_mv[3] = input[row * OrigDim + d0 + 2];
        }

        float temp[8];
        float rotated[8];
        temp[0] = s*x_mv[0] - p12*x_mv[4] - p13*x_mv[5] - p23*x_mv[6];
        temp[1] = s*x_mv[1] + p12*x_mv[2] + p13*x_mv[3] + p23*x_mv[7];
        temp[2] = s*x_mv[2] - p12*x_mv[1] + p23*x_mv[3] - p13*x_mv[7];
        temp[3] = s*x_mv[3] - p13*x_mv[1] - p23*x_mv[2] + p12*x_mv[7];
        temp[4] = s*x_mv[4] + p12*x_mv[0];
        temp[5] = s*x_mv[5] + p13*x_mv[0];
        temp[6] = s*x_mv[6] + p23*x_mv[0];
        temp[7] = s*x_mv[7] - p23*x_mv[1] + p13*x_mv[2] - p12*x_mv[3];

        rotated[0] = temp[0]*s + temp[4]*p12 + temp[5]*p13 + temp[6]*p23;
        rotated[1] = temp[1]*s + temp[2]*p12 + temp[3]*p13 - temp[7]*p23;
        rotated[2] = temp[2]*s - temp[1]*p12 + temp[3]*p23 + temp[7]*p13;
        rotated[3] = temp[3]*s - temp[1]*p13 - temp[2]*p23 - temp[7]*p12;
        rotated[4] = -temp[0]*p12 + temp[4]*s - temp[5]*p23 + temp[6]*p13;
        rotated[5] = -temp[0]*p13 + temp[5]*s + temp[4]*p23 - temp[6]*p12;
        rotated[6] = -temp[0]*p23 + temp[6]*s - temp[4]*p13 + temp[5]*p12;
        rotated[7] = temp[7]*s - temp[1]*p23 + temp[2]*p13 - temp[3]*p12;

        float q_mv[8] = {0.0f};
        for (int c = 0; c < 3; ++c) {
            int out_idx = d0 + c;
            if (out_idx >= PaddedDim) {
                break;
            }
            float value = rotated[1 + c];
            uint code = 0u;
            for (int i = 0; i < InnerCount; ++i) {
                code += value > boundaries[i] ? 1u : 0u;
            }
            indices[row * PaddedDim + out_idx] = code;
            q_mv[1 + c] = centroids[code];
        }

        float temp2[8];
        float final_mv[8];
        temp2[0] = s*q_mv[0] + p12*q_mv[4] + p13*q_mv[5] + p23*q_mv[6];
        temp2[1] = s*q_mv[1] - p12*q_mv[2] - p13*q_mv[3] + p23*q_mv[7];
        temp2[2] = s*q_mv[2] + p12*q_mv[1] - p23*q_mv[3] + p13*q_mv[7];
        temp2[3] = s*q_mv[3] + p13*q_mv[1] + p23*q_mv[2] - p12*q_mv[7];
        temp2[4] = s*q_mv[4] - p12*q_mv[0];
        temp2[5] = s*q_mv[5] - p13*q_mv[0];
        temp2[6] = s*q_mv[6] - p23*q_mv[0];
        temp2[7] = s*q_mv[7] + p23*q_mv[1] - p13*q_mv[2] + p12*q_mv[3];

        final_mv[0] = temp2[0]*s - temp2[4]*p12 - temp2[5]*p13 - temp2[6]*p23;
        final_mv[1] = temp2[1]*s - temp2[2]*p12 - temp2[3]*p13 + temp2[7]*p23;
        final_mv[2] = temp2[2]*s + temp2[1]*p12 - temp2[3]*p23 - temp2[7]*p13;
        final_mv[3] = temp2[3]*s + temp2[1]*p13 + temp2[2]*p23 + temp2[7]*p12;
        final_mv[4] = temp2[0]*p12 + temp2[4]*s + temp2[5]*p23 - temp2[6]*p13;
        final_mv[5] = temp2[0]*p13 + temp2[5]*s - temp2[4]*p23 + temp2[6]*p12;
        final_mv[6] = temp2[0]*p23 + temp2[6]*s + temp2[4]*p13 - temp2[5]*p12;
        final_mv[7] = temp2[7]*s + temp2[1]*p23 - temp2[2]*p13 + temp2[3]*p12;

        if (d0 < OrigDim) {
            recon[row * OrigDim + d0] = final_mv[1];
        }
        if (d0 + 1 < OrigDim) {
            recon[row * OrigDim + d0 + 1] = final_mv[2];
        }
        if (d0 + 2 < OrigDim) {
            recon[row * OrigDim + d0 + 2] = final_mv[3];
        }
    """
    return mx.fast.metal_kernel(
        name="rotorquant_encode_vectorgrade",
        input_names=["input", "rotors", "boundaries", "centroids"],
        output_names=["indices", "recon"],
        source=source,
    )


def _encode_vector_grade(vectors, rotors, rotor_reverse, boundaries, centroids, head_dim):
    padded_dim = _rotor_padded_dim(head_dim)
    num_groups = _rotor_group_count(head_dim)
    flat = vectors.reshape((-1, head_dim)).astype(mx.float32)

    kernel = _rotor_encode_kernel()
    inner = boundaries[1:-1].astype(mx.float32)
    if kernel is not None:
        indices, recon = kernel(
            inputs=[flat, rotors.astype(mx.float32), inner, centroids.astype(mx.float32)],
            template=[
                ("OrigDim", head_dim),
                ("PaddedDim", padded_dim),
                ("NumGroups", num_groups),
                ("InnerCount", inner.shape[0]),
            ],
            grid=(flat.shape[0], num_groups, 1),
            threadgroup=(1, 1, 1),
            output_shapes=[(flat.shape[0], padded_dim), (flat.shape[0], head_dim)],
            output_dtypes=[mx.uint32, mx.float32],
        )
        indices = indices.reshape(*vectors.shape[:-1], padded_dim).astype(mx.uint8)
        recon = recon.reshape(vectors.shape)
        return indices, recon

    mv = _embed_vectors_as_multivectors(vectors.astype(mx.float32), padded_dim)
    mv_rot = _rotor_sandwich(rotors, mv)
    rotated_vectors = mv_rot[..., _VECTOR_SLICE].reshape(*vectors.shape[:-1], padded_dim)
    indices = _quantize_boundaries(rotated_vectors, boundaries)

    q_vectors = centroids[indices]
    q_mv = mx.zeros_like(mv_rot)
    q_mv[..., _VECTOR_SLICE] = q_vectors.reshape(*vectors.shape[:-1], num_groups, 3)
    mv_recon = _inverse_rotor_sandwich(rotors, rotor_reverse, q_mv)
    recon = _extract_vectors_from_multivectors(mv_recon, head_dim).astype(mx.float32)
    return indices, recon


def _decode_vector_grade(indices, norms, rotors, rotor_reverse, centroids, head_dim):
    padded_dim = _rotor_padded_dim(head_dim)
    num_groups = _rotor_group_count(head_dim)
    q_vectors = centroids[indices]
    q_mv = mx.zeros((*indices.shape[:-1], num_groups, _MV_DIM), dtype=mx.float32)
    q_mv[..., _VECTOR_SLICE] = q_vectors.reshape(*indices.shape[:-1], num_groups, 3)
    mv_recon = _inverse_rotor_sandwich(rotors, rotor_reverse, q_mv)
    vectors = _extract_vectors_from_multivectors(mv_recon, head_dim)
    return vectors * norms.astype(mx.float32)


def _rotate_vector_grade(vectors, rotors, padded_dim):
    mv = _embed_vectors_as_multivectors(vectors.astype(mx.float32), padded_dim)
    mv_rot = _rotor_sandwich(rotors, mv)
    return _extract_vectors_from_multivectors(mv_rot, padded_dim).astype(mx.float32)


class RotorQuantKVCache(_BaseCache):
    """KV cache compressed with a real RotorQuant-style codec.

    ``mse`` stores only the vector-grade RotorQuant reconstruction.
    ``prod`` uses RotorQuant MSE on keys at ``bits - 1`` plus a packed 1-bit
    QJL residual correction, while values stay on the plain RotorQuant MSE path.
    """

    step = 256

    def __init__(
        self,
        bits: float = 4,
        estimator_mode: str = "mse",
        sparse_v_tau: Optional[float] = None,
        qjl_projection_mode: str = "auto",
        decode_buffer: bool = False,
        seed: int = 42,
    ):
        if estimator_mode not in ("mse", "prod"):
            raise ValueError(
                "RotorQuantKVCache estimator_mode must be 'mse' or 'prod', "
                f"got {estimator_mode}"
            )
        self.rotor_bits = _normalize_rotor_bits(bits)
        self.turbo_bits = self.rotor_bits
        self.estimator_mode = estimator_mode
        self.sparse_v_tau = sparse_v_tau
        self.qjl_projection_mode = (
            "dense" if estimator_mode == "prod" and qjl_projection_mode == "auto" else qjl_projection_mode
        )
        self.decode_buffer = bool(decode_buffer)
        self.seed = int(seed)
        self.rotation_mode = "rotorquant"
        self._k_bits = self.rotor_bits if estimator_mode == "mse" else max(self.rotor_bits - 1, 1)
        self._v_bits = self.rotor_bits
        self.offset = 0
        self._head_dim = None
        self._padded_dim = None
        self._num_groups = None
        self._k_centroids = None
        self._k_boundaries = None
        self._v_centroids = None
        self._v_boundaries = None
        self._rotors = None
        self._rotor_reverse = None
        self._k_indices = None
        self._k_norms = None
        self._k_qjl_indices = None
        self._k_qjl_gamma = None
        self._v_indices = None
        self._v_norms = None
        self._qjl_projection = None
        self._qjl_projection_t = None
        self._qjl_projection_runtime_mode = None
        self._k_deq_buf = None
        self._v_deq_buf = None
        self._deq_offset = 0
        self._deq_alloc = 0

    def _ensure_runtime_attrs(self):
        if not hasattr(self, "rotor_bits"):
            self.rotor_bits = 4
        if not hasattr(self, "turbo_bits"):
            self.turbo_bits = self.rotor_bits
        if not hasattr(self, "seed"):
            self.seed = 42
        if not hasattr(self, "decode_buffer"):
            self.decode_buffer = False
        if not hasattr(self, "rotation_mode"):
            self.rotation_mode = "rotorquant"
        if not hasattr(self, "estimator_mode"):
            self.estimator_mode = "mse"
        if not hasattr(self, "qjl_projection_mode"):
            self.qjl_projection_mode = "auto"
        if not hasattr(self, "_k_bits"):
            self._k_bits = (
                self.rotor_bits if self.estimator_mode == "mse" else max(self.rotor_bits - 1, 1)
            )
        if not hasattr(self, "_v_bits"):
            self._v_bits = self.rotor_bits
        if not hasattr(self, "sparse_v_tau"):
            self.sparse_v_tau = None
        if not hasattr(self, "_head_dim"):
            self._head_dim = None
        if not hasattr(self, "_padded_dim"):
            self._padded_dim = None
        if not hasattr(self, "_num_groups"):
            self._num_groups = None
        if not hasattr(self, "_k_centroids"):
            self._k_centroids = None
        if not hasattr(self, "_k_boundaries"):
            self._k_boundaries = None
        if not hasattr(self, "_v_centroids"):
            self._v_centroids = None
        if not hasattr(self, "_v_boundaries"):
            self._v_boundaries = None
        if not hasattr(self, "_rotors"):
            self._rotors = None
        if not hasattr(self, "_rotor_reverse"):
            self._rotor_reverse = None
        if not hasattr(self, "_k_indices"):
            self._k_indices = None
        if not hasattr(self, "_k_norms"):
            self._k_norms = None
        if not hasattr(self, "_k_qjl_indices"):
            self._k_qjl_indices = None
        if not hasattr(self, "_k_qjl_gamma"):
            self._k_qjl_gamma = None
        if not hasattr(self, "_v_indices"):
            self._v_indices = None
        if not hasattr(self, "_v_norms"):
            self._v_norms = None
        if not hasattr(self, "_qjl_projection"):
            self._qjl_projection = None
        if not hasattr(self, "_qjl_projection_t"):
            self._qjl_projection_t = None
        if not hasattr(self, "_qjl_projection_runtime_mode"):
            self._qjl_projection_runtime_mode = None
        if not hasattr(self, "_k_deq_buf"):
            self._k_deq_buf = None
        if not hasattr(self, "_v_deq_buf"):
            self._v_deq_buf = None
        if not hasattr(self, "_deq_offset"):
            self._deq_offset = 0
        if not hasattr(self, "_deq_alloc"):
            self._deq_alloc = 0

    def _init_codebook(self, head_dim):
        self._head_dim = int(head_dim)
        self._padded_dim = _rotor_padded_dim(head_dim)
        self._num_groups = _rotor_group_count(head_dim)
        codebook_dim = max(self._num_groups * _MV_DIM, 64)
        self._k_centroids, self._k_boundaries = _load_codebook(
            self._k_bits,
            codebook_dim,
        )
        self._v_centroids, self._v_boundaries = _load_codebook(
            self._v_bits,
            codebook_dim,
        )
        self._rotors, self._rotor_reverse = _cached_rotors(self._num_groups, self.seed)
        if self.estimator_mode == "prod":
            resolved = _resolve_qjl_projection_mode(
                head_dim,
                self.qjl_projection_mode,
            )
            self._qjl_projection_runtime_mode = resolved
            self._qjl_projection, self._qjl_projection_t = _cached_qjl_projection(
                head_dim,
                seed=self.seed + 1,
                mode=resolved,
            )
        else:
            self._qjl_projection = None
            self._qjl_projection_t = None
            self._qjl_projection_runtime_mode = None

    def _invalidate_decode_buffer(self):
        self._k_deq_buf = None
        self._v_deq_buf = None
        self._deq_offset = 0
        self._deq_alloc = 0

    def _ensure_decode_buffer_capacity(self, batch_size, n_kv_heads, total, k_dtype, v_dtype):
        if (
            self._k_deq_buf is not None
            and self._v_deq_buf is not None
            and total <= self._deq_alloc
        ):
            return

        alloc = ((total + self.step - 1) // self.step) * self.step
        k_shape = (batch_size, n_kv_heads, alloc, self._head_dim)
        v_shape = (batch_size, n_kv_heads, alloc, self._head_dim)
        if self._k_deq_buf is None or self._v_deq_buf is None:
            self._k_deq_buf = mx.zeros(k_shape, dtype=k_dtype)
            self._v_deq_buf = mx.zeros(v_shape, dtype=v_dtype)
        else:
            k_ext = mx.zeros(
                (batch_size, n_kv_heads, alloc - self._deq_alloc, self._head_dim),
                dtype=k_dtype,
            )
            v_ext = mx.zeros(
                (batch_size, n_kv_heads, alloc - self._deq_alloc, self._head_dim),
                dtype=v_dtype,
            )
            self._k_deq_buf = mx.concatenate(
                [self._k_deq_buf[..., : self._deq_offset, :], k_ext], axis=2
            )
            self._v_deq_buf = mx.concatenate(
                [self._v_deq_buf[..., : self._deq_offset, :], v_ext], axis=2
            )
        self._deq_alloc = alloc

    def _append_decode_buffer(self, prev, k_unit_recon, k_norms, v_unit_recon, v_norms, k_dtype, v_dtype):
        total = prev + k_unit_recon.shape[2]
        self._ensure_decode_buffer_capacity(
            k_unit_recon.shape[0],
            k_unit_recon.shape[1],
            total,
            k_dtype,
            v_dtype,
        )
        new_k = (k_unit_recon * k_norms.astype(mx.float32)).astype(k_dtype)
        new_v = (v_unit_recon * v_norms.astype(mx.float32)).astype(v_dtype)
        self._k_deq_buf[..., prev:total, :] = new_k
        self._v_deq_buf[..., prev:total, :] = new_v
        self._deq_offset = total
        return self._k_deq_buf[..., :total, :], self._v_deq_buf[..., :total, :]

    def _append_decode_buffer_raw(self, prev, keys, values):
        if self._k_deq_buf is not None and self._deq_offset != prev:
            return None, None
        if self._v_deq_buf is not None and self._deq_offset != prev:
            return None, None

        total = prev + keys.shape[2]
        self._ensure_decode_buffer_capacity(
            keys.shape[0],
            keys.shape[1],
            total,
            keys.dtype,
            values.dtype,
        )
        self._k_deq_buf[..., prev:total, :] = keys
        self._v_deq_buf[..., prev:total, :] = values
        self._deq_offset = total
        return self._k_deq_buf[..., :total, :], self._v_deq_buf[..., :total, :]

    def _materialize_decode_buffer(self, k_dtype, v_dtype):
        all_k = self._dequantize_keys(dtype=k_dtype)
        all_v = self._dequantize_values(dtype=v_dtype)
        if all_k is None or all_v is None:
            return all_k, all_v
        batch_size, n_kv_heads, total, _ = all_k.shape
        self._ensure_decode_buffer_capacity(
            batch_size, n_kv_heads, total, k_dtype, v_dtype
        )
        self._k_deq_buf[..., :total, :] = all_k
        self._v_deq_buf[..., :total, :] = all_v
        self._deq_offset = total
        return self._k_deq_buf[..., :total, :], self._v_deq_buf[..., :total, :]

    def _expand(self, B, n_kv_heads, new_steps, k_packed_dim, v_packed_dim, qjl_packed_dim=0):
        alloc = ((self.step + new_steps - 1) // self.step) * self.step
        shape = (B, n_kv_heads, alloc)

        def _new():
            arrays = [
                mx.zeros((*shape, k_packed_dim), dtype=mx.uint32),
                mx.zeros((*shape, 1), dtype=mx.float32),
                mx.zeros((*shape, v_packed_dim), dtype=mx.uint32),
                mx.zeros((*shape, 1), dtype=mx.float32),
            ]
            if self.estimator_mode == "prod":
                arrays.extend(
                    [
                        mx.zeros((*shape, qjl_packed_dim), dtype=mx.uint32),
                        mx.zeros((*shape, 1), dtype=mx.float32),
                    ]
                )
            return arrays

        if self._k_indices is not None and self.offset > 0:
            old = [
                self._k_indices[..., : self.offset, :],
                self._k_norms[..., : self.offset, :],
                self._v_indices[..., : self.offset, :],
                self._v_norms[..., : self.offset, :],
            ]
            if self.estimator_mode == "prod":
                old.extend(
                    [
                        self._k_qjl_indices[..., : self.offset, :],
                        self._k_qjl_gamma[..., : self.offset, :],
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

    def _dequantize_keys(self, limit=None, *, include_qjl=True, dtype=None):
        self._ensure_runtime_attrs()
        if self._k_indices is None:
            return None
        limit = self.offset if limit is None else limit
        indices = _unpack(
            self._k_indices[..., :limit, :],
            self._k_bits,
            self._padded_dim,
        )
        key_mse = _decode_vector_grade(
            indices,
            self._k_norms[..., :limit, :],
            self._rotors,
            self._rotor_reverse,
            self._k_centroids,
            self._head_dim,
        )
        keys = key_mse
        if self.estimator_mode == "prod" and include_qjl:
            qjl_signs = _unpack(
                self._k_qjl_indices[..., :limit, :],
                1,
                self._head_dim,
            )
            unit_correction = _dequantize_qjl(
                qjl_signs,
                self._k_qjl_gamma[..., :limit, :],
                self._qjl_projection,
            )
            keys = key_mse + unit_correction * self._k_norms[..., :limit, :].astype(mx.float32)
        if dtype is not None:
            keys = keys.astype(dtype)
        return mx.contiguous(keys)

    def _dequantize_values(self, limit=None, *, dtype=None):
        self._ensure_runtime_attrs()
        if self._v_indices is None:
            return None
        limit = self.offset if limit is None else limit
        indices = _unpack(
            self._v_indices[..., :limit, :],
            self._v_bits,
            self._padded_dim,
        )
        values = _decode_vector_grade(
            indices,
            self._v_norms[..., :limit, :],
            self._rotors,
            self._rotor_reverse,
            self._v_centroids,
            self._head_dim,
        )
        if dtype is not None:
            values = values.astype(dtype)
        return mx.contiguous(values)

    @property
    def compressed_tokens(self):
        return self.offset

    @property
    def buffer_tokens(self):
        return 0

    def fused_scores(self, queries_scaled):
        self._ensure_runtime_attrs()
        if (
            self.estimator_mode != "prod"
            or self._k_indices is None
            or self.offset <= 0
            or queries_scaled.ndim != 4
        ):
            return None

        limit = self.offset
        keys = self._dequantize_keys(limit=limit, include_qjl=False, dtype=mx.float32)
        if keys is None:
            return None

        B, n_q_heads, _, _ = queries_scaled.shape
        n_kv_heads = keys.shape[1]
        if B != keys.shape[0] or n_q_heads % n_kv_heads != 0:
            return None

        if n_q_heads != n_kv_heads:
            n_repeats = n_q_heads // n_kv_heads
            k_norms = mx.repeat(self._k_norms[..., :limit, 0], n_repeats, axis=1)
            k_gamma = mx.repeat(self._k_qjl_gamma[..., :limit, 0], n_repeats, axis=1)
            qjl_packed = mx.repeat(
                self._k_qjl_indices[..., :limit, :],
                n_repeats,
                axis=1,
            )
        else:
            n_repeats = 1
            k_norms = self._k_norms[..., :limit, 0]
            k_gamma = self._k_qjl_gamma[..., :limit, 0]
            qjl_packed = self._k_qjl_indices[..., :limit, :]

        queries_scaled = queries_scaled.astype(mx.float32)
        if (
            queries_scaled.shape[2] == 1
            and hasattr(mx.fast, "turboquant_qk_packed_scores_batched")
        ):
            q_rot = _rotate_vector_grade(
                queries_scaled,
                self._rotors,
                self._padded_dim,
            )
            scores = mx.fast.turboquant_qk_packed_scores_batched(
                _fused_input(q_rot, mx.float32),
                _fused_input(self._k_indices[..., :limit, :]),
                _fused_input(self._k_norms[..., :limit, 0], mx.float32),
                self._k_centroids,
                self._k_bits,
                n_repeats,
            )
        else:
            if n_q_heads != n_kv_heads:
                keys = mx.repeat(keys, n_repeats, axis=1)
            scores = queries_scaled @ mx.swapaxes(keys, -1, -2)

        q_proj = _apply_qjl_projection(queries_scaled, self._qjl_projection_t)
        qjl_signs = _unpack(qjl_packed, 1, self._head_dim).astype(mx.float32)
        qjl_signs = qjl_signs * 2.0 - 1.0
        correction = mx.einsum("bhld,bhtd->bhlt", q_proj, qjl_signs)
        correction = correction * k_norms[:, :, None, :] * k_gamma[:, :, None, :]
        correction = correction * (math.sqrt(math.pi / 2.0) / self._head_dim)
        return mx.contiguous(scores + correction)

    def update_and_fetch(self, keys, values):
        self._ensure_runtime_attrs()
        B, n_kv_heads, num_steps, head_dim = keys.shape
        prev = self.offset

        if self._k_centroids is None:
            self._init_codebook(head_dim)

        k_float = keys.astype(mx.float32)
        v_float = values.astype(mx.float32)

        if self.estimator_mode == "prod":
            # Scrya's public RotorQuantProd runs on the raw key/value vectors:
            # Stage 1 stores a RotorQuant MSE reconstruction, and Stage 2
            # quantizes the raw residual in the original vector space.
            k_norms = mx.ones((*k_float.shape[:-1], 1), dtype=mx.float32)
            v_norms = mx.ones((*v_float.shape[:-1], 1), dtype=mx.float32)
            k_idx, k_unit_recon = _encode_vector_grade(
                k_float,
                self._rotors,
                self._rotor_reverse,
                self._k_boundaries,
                self._k_centroids,
                self._head_dim,
            )
            v_idx, v_unit_recon = _encode_vector_grade(
                v_float,
                self._rotors,
                self._rotor_reverse,
                self._v_boundaries,
                self._v_centroids,
                self._head_dim,
            )
            pk_qjl, k_gamma = _quantize_qjl_residual_packed(
                k_float,
                k_unit_recon,
                self._qjl_projection_t,
            )
        else:
            k_norms = mx.linalg.norm(k_float, axis=-1, keepdims=True)
            v_norms = mx.linalg.norm(v_float, axis=-1, keepdims=True)
            k_unit = k_float / mx.maximum(k_norms, 1e-8)
            v_unit = v_float / mx.maximum(v_norms, 1e-8)

            k_idx, k_unit_recon = _encode_vector_grade(
                k_unit,
                self._rotors,
                self._rotor_reverse,
                self._k_boundaries,
                self._k_centroids,
                self._head_dim,
            )
            v_idx, v_unit_recon = _encode_vector_grade(
                v_unit,
                self._rotors,
                self._rotor_reverse,
                self._v_boundaries,
                self._v_centroids,
                self._head_dim,
            )
            pk_qjl = None
            k_gamma = None
        pk = _pack(k_idx, self._k_bits)
        pv = _pack(v_idx, self._v_bits)

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
            self._k_qjl_gamma[..., prev : prev + num_steps, :] = k_gamma.astype(mx.float32)
        self._v_indices[..., prev : prev + num_steps, :] = pv
        self._v_norms[..., prev : prev + num_steps, :] = v_norms
        self.offset += num_steps

        if self.decode_buffer and self.estimator_mode == "prod":
            _, buffered_v = self._append_decode_buffer_raw(prev, keys, values)
            if buffered_v is not None:
                return None, buffered_v
            return None, self._dequantize_values(dtype=values.dtype)

        if self.decode_buffer:
            return self._append_decode_buffer(
                prev,
                k_unit_recon,
                k_norms,
                v_unit_recon,
                v_norms,
                keys.dtype,
                values.dtype,
            )

        all_v = self._dequantize_values(dtype=values.dtype)
        if self.estimator_mode == "prod":
            return None, all_v
        all_k = self._dequantize_keys(dtype=keys.dtype)
        return all_k, all_v

    def size(self):
        return self.offset

    @property
    def state(self):
        self._ensure_runtime_attrs()
        if self._k_indices is None:
            return []
        arrays = [
            self._k_indices[..., : self.offset, :],
            self._k_norms[..., : self.offset, :],
            self._v_indices[..., : self.offset, :],
            self._v_norms[..., : self.offset, :],
        ]
        if self.estimator_mode == "prod":
            arrays.extend(
                [
                    self._k_qjl_indices[..., : self.offset, :],
                    self._k_qjl_gamma[..., : self.offset, :],
                ]
            )
        return arrays

    @state.setter
    def state(self, v):
        self._ensure_runtime_attrs()
        self._invalidate_decode_buffer()
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
        self._ensure_runtime_attrs()
        head_dim = self._head_dim or 0
        sparse_tau = -1.0 if self.sparse_v_tau is None else float(self.sparse_v_tau)
        return tuple(
            map(
                str,
                (
                    self.offset,
                    self.rotor_bits,
                    head_dim,
                    self.seed,
                    self.estimator_mode,
                    self.qjl_projection_mode,
                    sparse_tau,
                    "1" if self.decode_buffer else "0",
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, v):
        self._ensure_runtime_attrs()
        self.offset = int(v[0])
        self.rotor_bits = _normalize_rotor_bits(v[1])
        self.turbo_bits = self.rotor_bits
        self._k_bits = self.rotor_bits
        self._v_bits = self.rotor_bits
        head_dim = int(v[2])
        self.seed = int(v[3]) if len(v) > 3 else 42
        self.estimator_mode = v[4] if len(v) > 4 else "mse"
        if self.estimator_mode not in ("mse", "prod"):
            self.estimator_mode = "mse"
        if self.estimator_mode == "prod":
            self._k_bits = max(self.rotor_bits - 1, 1)
        self.qjl_projection_mode = v[5] if len(v) > 5 else (
            "dense" if self.estimator_mode == "prod" else "auto"
        )
        sparse_tau = float(v[6]) if len(v) > 6 else -1.0
        self.sparse_v_tau = None if sparse_tau < 0 else sparse_tau
        self.decode_buffer = len(v) > 7 and v[7] in ("1", "true")
        self.rotation_mode = "rotorquant"
        self._invalidate_decode_buffer()
        if head_dim > 0:
            self._init_codebook(head_dim)

    def is_trimmable(self):
        return True

    def trim(self, n):
        self._ensure_runtime_attrs()
        self._invalidate_decode_buffer()
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        self._ensure_runtime_attrs()
        return self._k_indices is None

    @property
    def nbytes(self):
        self._ensure_runtime_attrs()
        if self._k_indices is None:
            return 0
        arrays = [self._k_indices, self._k_norms, self._v_indices, self._v_norms]
        if self.estimator_mode == "prod" and self._k_qjl_indices is not None:
            arrays.extend([self._k_qjl_indices, self._k_qjl_gamma])
        return sum(a[..., : self.offset, :].nbytes for a in arrays)
