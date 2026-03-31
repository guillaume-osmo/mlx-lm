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

import numpy as np

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

_FALSY_ENV_VALUES = {"0", "false", "f", "no"}


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
    if mode == "wht":
        if not hasattr(mx, "hadamard_transform") or not _supports_hadamard_dim(dim):
            raise ValueError(
                f"WHT rotation does not support head_dim={dim}; "
                "supported sizes are m*2^k where m in {1, 12, 20, 28}."
            )
        return _random_signs(dim, seed + dim * 131 + 17)
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
    if mode == "wht":
        return rotation, rotation
    return rotation, mx.swapaxes(rotation, -1, -2)


def _supports_hadamard_dim(dim):
    if dim < 1:
        return False
    while dim % 2 == 0:
        dim //= 2
    return dim in (1, 12, 20, 28)


@lru_cache(maxsize=64)
def _random_signs(dim, seed=123):
    rng = np.random.default_rng(seed)
    return mx.array(rng.choice([-1.0, 1.0], size=(dim,)).astype(np.float32))


def _resolve_qjl_projection_mode(dim, requested="auto"):
    mode = (requested or "auto").lower()
    if mode not in ("auto", "dense", "wht"):
        mode = "auto"
    if mode == "dense":
        return "dense"
    if mode == "wht":
        if not _supports_hadamard_dim(dim):
            raise ValueError(
                f"QJL WHT projection does not support head_dim={dim}; "
                "supported sizes are m*2^k where m in {1, 12, 20, 28}."
            )
        return "wht"
    if hasattr(mx, "hadamard_transform") and _supports_hadamard_dim(dim):
        return "wht"
    return "dense"


@lru_cache(maxsize=64)
def _cached_qjl_projection(dim, seed=123, mode="auto"):
    resolved = _resolve_qjl_projection_mode(dim, mode)
    if resolved == "wht":
        signs = _random_signs(dim, seed + dim * 2971 + 17)
        return signs, signs
    key = mx.random.key(seed)
    projection = mx.random.normal(shape=(dim, dim), key=key).astype(mx.float32)
    return projection, projection.T


def _load_codebook(bits, dim):
    s = 1.0 / math.sqrt(dim)
    c = mx.array(_CENTROIDS[bits], dtype=mx.float32) * s
    b = mx.array(_BOUNDARIES[bits], dtype=mx.float32) * s
    return c, b


def _normalize_turbo_bits(bits):
    bits = float(bits)
    if bits < 2 or bits > 4:
        raise ValueError(f"bits must be between 2 and 4, got {bits}")
    rounded = round(bits)
    if math.isclose(bits, rounded, abs_tol=1e-6):
        return int(rounded)
    return bits


def _is_fractional_bits(bits):
    return not isinstance(bits, int)


def _normalize_override_bits(bits, name):
    if bits is None:
        return None
    bits = _normalize_turbo_bits(bits)
    if _is_fractional_bits(bits):
        raise ValueError(f"{name} must be an integer bit-width, got {bits}")
    return int(bits)


def _env_fused_enabled():
    value = os.environ.get("MLX_TQ_FUSED")
    if value is None:
        return None
    return value.lower() not in _FALSY_ENV_VALUES


def _prefer_fused_default(
    bits,
    key_bits,
    value_bits,
    rotation_mode,
    estimator_mode,
    qjl_residual,
):
    if rotation_mode not in ("dense", "wht") or estimator_mode != "prod" or qjl_residual:
        return False
    effective_key_bits = key_bits if key_bits is not None else int(bits)
    effective_value_bits = value_bits if value_bits is not None else int(bits)
    return effective_key_bits == 3 and effective_value_bits == 4


def _select_fractional_indices(keys, values, avg_bits):
    lower_bits = math.floor(avg_bits)
    upper_bits = math.ceil(avg_bits)
    if lower_bits == upper_bits:
        raise ValueError("fractional split selection requires a non-integer bit-width")

    dim = keys.shape[-1]
    if dim <= 1:
        raise ValueError("fractional TurboQuant requires head_dim > 1")

    high_count = int(round((avg_bits - lower_bits) * dim / (upper_bits - lower_bits)))
    high_count = max(1, min(dim - 1, high_count))

    scores = mx.mean(mx.abs(keys).astype(mx.float32), axis=(0, 1, 2))
    scores = scores + mx.mean(mx.abs(values).astype(mx.float32), axis=(0, 1, 2))
    mx.eval(scores)
    order = np.argsort(np.asarray(scores))
    high_idx = np.sort(order[-high_count:].astype(np.int32))
    low_mask = np.ones(dim, dtype=bool)
    low_mask[high_idx] = False
    low_idx = np.nonzero(low_mask)[0].astype(np.int32)
    restore_order = np.argsort(np.concatenate([low_idx, high_idx])).astype(np.int32)

    return (
        mx.array(low_idx, dtype=mx.int32),
        mx.array(high_idx, dtype=mx.int32),
        mx.array(restore_order, dtype=mx.int32),
    )


def _merge_split_tensors(low_tensor, high_tensor, restore_order):
    merged = mx.concatenate([low_tensor, high_tensor], axis=-1)
    return mx.take(merged, restore_order, axis=-1)


def _apply_rotation(vectors, rotation_t, mode="dense"):
    if rotation_t.ndim == 1:
        return mx.hadamard_transform(vectors.astype(mx.float32) * rotation_t.astype(mx.float32))
    if rotation_t.ndim == 2:
        return vectors @ rotation_t
    return _apply_block_rotation(vectors, rotation_t)


def _apply_inverse_rotation(vectors, rotation, mode="dense"):
    if rotation.ndim == 1:
        return mx.hadamard_transform(vectors.astype(mx.float32)) * rotation.astype(mx.float32)
    if rotation.ndim == 2:
        return vectors @ rotation
    return _apply_block_rotation(vectors, rotation)


def _quantize_unit(vectors, rotation_t, boundaries, mode="dense"):
    rotated = _apply_rotation(vectors, rotation_t, mode=mode)
    inner = boundaries[1:-1]
    kernel = _quantize_boundary_kernel()
    if kernel is not None and inner.size > 0:
        shape = rotated.shape
        flat = rotated.reshape((-1, shape[-1])).astype(mx.float32)
        indices = kernel(
            inputs=[flat, inner.astype(mx.float32)],
            template=[
                ("Length", shape[-1]),
                ("InnerCount", inner.shape[0]),
            ],
            grid=(shape[-1], flat.shape[0], 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(flat.shape[0], shape[-1])],
            output_dtypes=[mx.uint32],
        )[0]
        return indices.reshape(shape).astype(mx.uint8)

    indices = mx.zeros(rotated.shape, dtype=mx.uint8)
    for b in range(inner.shape[0]):
        indices = indices + (rotated > inner[b]).astype(mx.uint8)
    return indices


def _dequantize_unit(indices, rotation, centroids, mode="dense"):
    rotated = centroids[indices]
    return _apply_inverse_rotation(rotated, rotation, mode=mode)


def _quantize(vectors, rotation_t, boundaries, mode="dense"):
    vectors_f = vectors.astype(mx.float32)
    norms = mx.linalg.norm(vectors_f, axis=-1, keepdims=True)
    indices = _quantize_unit(
        vectors_f / mx.maximum(norms, 1e-8), rotation_t, boundaries, mode=mode
    )
    return indices, norms


def _dequantize(indices, norms, rotation, centroids, mode="dense"):
    return _dequantize_unit(indices, rotation, centroids, mode=mode) * norms


def _apply_qjl_projection(vectors, projection_t):
    if projection_t.ndim == 1:
        signs = projection_t.astype(mx.float32)
        return mx.hadamard_transform(vectors.astype(mx.float32) * signs)
    return vectors @ projection_t


def _apply_inverse_qjl_projection(projected, projection):
    if projection.ndim == 1:
        signs = projection.astype(mx.float32)
        return mx.hadamard_transform(projected.astype(mx.float32)) * signs
    return projected @ projection


def _quantize_qjl_residual(unit_vectors, mse_vectors, projection_t):
    residual = unit_vectors - mse_vectors
    gamma = mx.linalg.norm(residual, axis=-1, keepdims=True)
    unit_residual = residual / mx.maximum(gamma, 1e-8)
    signs = (_apply_qjl_projection(unit_residual, projection_t) >= 0).astype(mx.uint8)
    zero_mask = mx.broadcast_to(gamma <= 1e-8, signs.shape)
    signs = mx.where(zero_mask, mx.zeros_like(signs), signs)
    return signs, gamma


def _quantize_qjl_residual_packed(unit_vectors, mse_vectors, projection_t):
    residual = unit_vectors - mse_vectors
    gamma = mx.linalg.norm(residual, axis=-1, keepdims=True)
    unit_residual = residual / mx.maximum(gamma, 1e-8)
    projected = _apply_qjl_projection(unit_residual, projection_t)

    kernel = _pack_signbit_kernel()
    if kernel is not None:
        shape = projected.shape
        dim = shape[-1]
        packed_width = (dim + 31) // 32
        flat_projected = projected.reshape((-1, dim)).astype(mx.float32)
        flat_gamma = gamma.reshape((-1,)).astype(mx.float32)
        packed = kernel(
            inputs=[flat_projected, flat_gamma],
            template=[
                ("Length", dim),
                ("PackedWidth", packed_width),
            ],
            grid=(packed_width, flat_projected.shape[0], 1),
            threadgroup=(min(32, packed_width), 1, 1),
            output_shapes=[(flat_projected.shape[0], packed_width)],
            output_dtypes=[mx.uint32],
        )[0]
        return packed.reshape(*shape[:-1], packed_width), gamma

    signs = (projected >= 0).astype(mx.uint8)
    zero_mask = mx.broadcast_to(gamma <= 1e-8, signs.shape)
    signs = mx.where(zero_mask, mx.zeros_like(signs), signs)
    return _pack(signs, 1), gamma


def _dequantize_qjl(signs, gamma, projection):
    signs_pm = mx.where(signs > 0, 1.0, -1.0).astype(mx.float32)
    scale = math.sqrt(math.pi / 2.0) / projection.shape[0]
    return scale * gamma.astype(mx.float32) * _apply_inverse_qjl_projection(
        signs_pm, projection
    )


def _quantize_prod(
    vectors,
    rotation_t,
    rotation,
    boundaries,
    centroids,
    projection_t,
    mode="dense",
    pack_qjl=False,
):
    vectors_f = vectors.astype(mx.float32)
    norms = mx.linalg.norm(vectors_f, axis=-1, keepdims=True)
    unit_vectors = vectors_f / mx.maximum(norms, 1e-8)
    indices = _quantize_unit(unit_vectors, rotation_t, boundaries, mode=mode)
    mse_vectors = _dequantize_unit(indices, rotation, centroids, mode=mode)
    if pack_qjl:
        qjl_signs, gamma = _quantize_qjl_residual_packed(
            unit_vectors, mse_vectors, projection_t
        )
    else:
        qjl_signs, gamma = _quantize_qjl_residual(
            unit_vectors, mse_vectors, projection_t
        )
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


def _metal_available():
    return (
        mx.default_device() == mx.gpu
        and mx.metal.is_available()
        and hasattr(mx, "fast")
        and hasattr(mx.fast, "metal_kernel")
    )


@lru_cache(maxsize=None)
def _pack_legacy_layout_kernel():
    if not _metal_available():
        return None

    source = r"""
        auto word = thread_position_in_grid.x;
        auto row = thread_position_in_grid.y;

        if (row >= values_shape[0] || word >= PackedWidth) {
            return;
        }

        auto values_ptr = values + row * Length;
        uint packed_word = 0u;
        int base = int(word) * ValsPerWord;
        for (int i = 0; i < ValsPerWord; ++i) {
            int idx = base + i;
            if (idx >= Length) {
                break;
            }
            uint value = values_ptr[idx] & Mask;
            packed_word |= value << (i * Bits);
        }

        out[row * PackedWidth + word] = packed_word;
    """
    return mx.fast.metal_kernel(
        name="turboquant_pack_legacy_layout",
        input_names=["values"],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=None)
def _unpack_legacy_layout_kernel():
    if not _metal_available():
        return None

    source = r"""
        auto idx = thread_position_in_grid.x;
        auto row = thread_position_in_grid.y;

        if (row >= packed_shape[0] || idx >= Length) {
            return;
        }

        int word = int(idx) / ValsPerWord;
        int offset = (int(idx) % ValsPerWord) * Bits;
        uint packed_word = packed[row * PackedWidth + word];
        out[row * Length + idx] = (packed_word >> offset) & Mask;
    """
    return mx.fast.metal_kernel(
        name="turboquant_unpack_legacy_layout",
        input_names=["packed"],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=None)
def _quantize_boundary_kernel():
    if not _metal_available():
        return None

    source = r"""
        auto idx = thread_position_in_grid.x;
        auto row = thread_position_in_grid.y;

        if (row >= rotated_shape[0] || idx >= Length) {
            return;
        }

        float value = rotated[row * Length + idx];
        uint code = 0u;
        for (int i = 0; i < InnerCount; ++i) {
            code += value > boundaries[i] ? 1u : 0u;
        }
        out[row * Length + idx] = code;
    """
    return mx.fast.metal_kernel(
        name="turboquant_quantize_boundaries",
        input_names=["rotated", "boundaries"],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=None)
def _pack_signbit_kernel():
    if not _metal_available():
        return None

    source = r"""
        auto word = thread_position_in_grid.x;
        auto row = thread_position_in_grid.y;

        if (row >= projected_shape[0] || word >= PackedWidth) {
            return;
        }

        float gamma_value = gamma[row];
        uint packed_word = 0u;
        int base = int(word) * 32;
        bool keep = gamma_value > 1e-8f;

        for (int i = 0; i < 32; ++i) {
            int idx = base + i;
            if (idx >= Length) {
                break;
            }
            uint sign_bit = 0u;
            if (keep && projected[row * Length + idx] >= 0.0f) {
                sign_bit = 1u;
            }
            packed_word |= sign_bit << i;
        }

        out[row * PackedWidth + word] = packed_word;
    """
    return mx.fast.metal_kernel(
        name="turboquant_pack_signbit",
        input_names=["projected", "gamma"],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=None)
def _qjl_score_kernel():
    if not _metal_available():
        return None

    source = r"""
        auto lane = thread_position_in_grid.x;
        auto repeat_idx = thread_position_in_grid.y;
        auto n = thread_position_in_grid.z;

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto repeat_count = q_proj_shape[2];
        if (repeat_idx >= repeat_count) {
            return;
        }

        auto b = n / (kv_heads * token_count);
        auto rem = n % (kv_heads * token_count);
        auto h = rem / token_count;
        auto t = rem % token_count;

        auto q_ptr = q_proj + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;
        auto packed_ptr = qjl_packed + ((b * kv_heads + h) * token_count + t) * PackedWidth;

        float acc = 0.0f;
        for (int d = lane; d < Dim; d += 32) {
            int word_idx = d / 32;
            int offset = d % 32;
            uint bit = (packed_ptr[word_idx] >> offset) & 1u;
            float sign = bit ? 1.0f : -1.0f;
            acc += static_cast<float>(q_ptr[d]) * sign;
        }

        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            auto idx = (b * kv_heads + h) * token_count + t;
            out[((b * kv_heads + h) * repeat_count + repeat_idx) * token_count + t] =
                acc
                * static_cast<float>(norms[idx])
                * static_cast<float>(residual_norms[idx])
                * scale[0];
        }
    """
    return mx.fast.metal_kernel(
        name="turboquant_qjl_score",
        input_names=["q_proj", "norms", "residual_norms", "qjl_packed", "scale"],
        output_names=["out"],
        source=source,
    )


def _pack(indices, bits):
    """Pack b-bit indices into uint32."""
    shape = indices.shape
    dim = shape[-1]
    vpi = 32 // bits
    n_packed = (dim + vpi - 1) // vpi
    flat = indices.reshape((-1, dim)).astype(mx.uint32)
    mask = (1 << bits) - 1

    kernel = _pack_legacy_layout_kernel()
    if kernel is not None:
        packed = kernel(
            inputs=[flat],
            template=[
                ("Bits", bits),
                ("ValsPerWord", vpi),
                ("Length", dim),
                ("PackedWidth", n_packed),
                ("Mask", mask),
            ],
            grid=(n_packed, flat.shape[0], 1),
            threadgroup=(min(32, n_packed), 1, 1),
            output_shapes=[(flat.shape[0], n_packed)],
            output_dtypes=[mx.uint32],
        )[0]
        return packed.reshape(*shape[:-1], n_packed)

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
    flat = packed.reshape((-1, shape[-1])).astype(mx.uint32)
    mask = (1 << bits) - 1

    kernel = _unpack_legacy_layout_kernel()
    if kernel is not None:
        unpacked = kernel(
            inputs=[flat],
            template=[
                ("Bits", bits),
                ("ValsPerWord", vpi),
                ("Length", dim),
                ("PackedWidth", shape[-1]),
                ("Mask", mask),
            ],
            grid=(dim, flat.shape[0], 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(flat.shape[0], dim)],
            output_dtypes=[mx.uint32],
        )[0]
        return unpacked.reshape(*shape[:-1], dim).astype(mx.uint8)

    shifts = mx.arange(vpi, dtype=mx.uint32) * bits
    extracted = (packed[..., None] >> shifts) & mask
    return extracted.reshape(*shape[:-1], shape[-1] * vpi)[..., :dim].astype(mx.uint8)


def _fused_input(array, dtype=None):
    if dtype is not None and array.dtype != dtype:
        array = array.astype(dtype)
    return mx.contiguous(array)


def _metal_qjl_score(q_proj, norms, residual_norms, qjl_packed, scale):
    if (
        not _metal_available()
        or q_proj.ndim != 4
        or norms.ndim != 3
        or residual_norms.ndim != 3
        or qjl_packed.ndim != 4
        or q_proj.shape[0] != norms.shape[0]
        or q_proj.shape[1] != norms.shape[1]
        or norms.shape != residual_norms.shape
        or qjl_packed.shape[:3] != norms.shape
        or qjl_packed.shape[-1] != (q_proj.shape[-1] + 31) // 32
        or norms.shape[2] == 0
    ):
        return None

    kernel = _qjl_score_kernel()
    if kernel is None:
        return None

    B, H, R, D = q_proj.shape
    T = norms.shape[2]
    return kernel(
        inputs=[
            _fused_input(q_proj, mx.float32),
            _fused_input(norms, mx.float32),
            _fused_input(residual_norms, mx.float32),
            _fused_input(qjl_packed),
            _fused_input(scale, mx.float32),
        ],
        template=[
            ("Dim", D),
            ("PackedWidth", qjl_packed.shape[-1]),
        ],
        grid=(32, R, B * H * T),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H, R, T)],
        output_dtypes=[mx.float32],
    )[0]


class TurboQuantKVCache(_BaseCache):
    """KV cache compressed with PolarQuant (experimental).

    Data-oblivious compression: random orthogonal rotation maps KV vectors
    to coordinates with a known Gaussian distribution, then Lloyd-Max
    optimal scalar quantizers compress each coordinate independently.
    Bit-packed into uint32 for storage, dequantized on fetch.

    Args:
        bits (float): Bits per coordinate (2-4). Integer bit-widths use the
            existing packed path, while fractional values like ``2.5`` and
            ``3.5`` split the head dimension across neighboring integer
            bit-widths. Default: ``4``.
    """

    step = 256

    def __init__(
        self,
        bits: float = 4,
        key_bits: Optional[int] = None,
        value_bits: Optional[int] = None,
        rotation_mode: str = "dense",
        estimator_mode: str = "mse",
        qjl_residual: bool = True,
        sparse_v_tau: Optional[float] = None,
        sparse_v_mode: Optional[str] = None,
        sparse_v_percentile: Optional[float] = None,
        sparse_v_early_multiplier: float = 1.25,
        sparse_v_late_multiplier: float = 0.75,
        qjl_projection_mode: str = "auto",
        decode_buffer: bool = False,
        buffer_size: int = 0,
        flush_batch_size: int = 0,
        max_cache_tokens: int = 0,
    ):
        bits = _normalize_turbo_bits(bits)
        key_bits = _normalize_override_bits(key_bits, "key_bits")
        value_bits = _normalize_override_bits(value_bits, "value_bits")
        if (key_bits is not None or value_bits is not None) and _is_fractional_bits(bits):
            raise ValueError(
                "Explicit key_bits/value_bits overrides do not yet support "
                f"fractional turbo bits, got bits={bits}"
            )
        if rotation_mode not in ("dense", "wht", "rotor3", "rotorquant"):
            raise ValueError(
                "rotation_mode must be 'dense', 'wht', 'rotor3', or 'rotorquant', "
                f"got {rotation_mode}"
            )
        if estimator_mode not in ("mse", "prod"):
            raise ValueError(
                "estimator_mode must be 'mse' or 'prod', "
                f"got {estimator_mode}"
            )
        if qjl_projection_mode not in ("auto", "dense", "wht"):
            raise ValueError(
                "qjl_projection_mode must be 'auto', 'dense', or 'wht', "
                f"got {qjl_projection_mode}"
            )
        self.turbo_bits = bits
        self.key_bits_override = key_bits
        self.value_bits_override = value_bits
        self._fractional_split = (
            _is_fractional_bits(bits)
            and key_bits is None
            and value_bits is None
        )
        self.rotation_mode = rotation_mode
        self.estimator_mode = estimator_mode
        self.qjl_residual = bool(qjl_residual)
        self.sparse_v_tau = sparse_v_tau
        self.sparse_v_mode = sparse_v_mode
        self.sparse_v_percentile = sparse_v_percentile
        self.sparse_v_early_multiplier = float(sparse_v_early_multiplier)
        self.sparse_v_late_multiplier = float(sparse_v_late_multiplier)
        self.qjl_projection_mode = qjl_projection_mode
        self.decode_buffer = bool(decode_buffer)
        self.buffer_size = max(0, int(buffer_size))
        self.flush_batch_size = max(0, int(flush_batch_size or buffer_size))
        self.max_cache_tokens = max(0, int(max_cache_tokens))
        self._qjl_projection_runtime_mode = None
        self.offset = 0
        self._packed_offset = 0
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
        self._split_low_bits = None
        self._split_high_bits = None
        self._split_low_idx = None
        self._split_high_idx = None
        self._split_restore_order = None
        self._split_low_cache = None
        self._split_high_cache = None
        self._pending_split_state = None
        self._k_deq_buf = None
        self._v_deq_buf = None
        self._deq_offset = 0
        self._deq_alloc = 0
        self._buffer_keys = None
        self._buffer_values = None
        fused_env = _env_fused_enabled()
        self._fused_enabled = (
            _prefer_fused_default(
                bits,
                key_bits,
                value_bits,
                rotation_mode,
                estimator_mode,
                self.qjl_residual,
            )
            if fused_env is None
            else fused_env
        )

    def _ensure_runtime_attrs(self):
        if not hasattr(self, "_fractional_split"):
            self._fractional_split = False
        if not hasattr(self, "key_bits_override"):
            self.key_bits_override = None
        if not hasattr(self, "value_bits_override"):
            self.value_bits_override = None
        if not hasattr(self, "qjl_residual"):
            self.qjl_residual = True
        if not hasattr(self, "_split_low_bits"):
            self._split_low_bits = None
        if not hasattr(self, "_split_high_bits"):
            self._split_high_bits = None
        if not hasattr(self, "_split_low_idx"):
            self._split_low_idx = None
        if not hasattr(self, "_split_high_idx"):
            self._split_high_idx = None
        if not hasattr(self, "_split_restore_order"):
            self._split_restore_order = None
        if not hasattr(self, "_split_low_cache"):
            self._split_low_cache = None
        if not hasattr(self, "_split_high_cache"):
            self._split_high_cache = None
        if not hasattr(self, "_pending_split_state"):
            self._pending_split_state = None
        if not hasattr(self, "decode_buffer"):
            self.decode_buffer = False
        if not hasattr(self, "_k_deq_buf"):
            self._k_deq_buf = None
        if not hasattr(self, "_v_deq_buf"):
            self._v_deq_buf = None
        if not hasattr(self, "_deq_offset"):
            self._deq_offset = 0
        if not hasattr(self, "_deq_alloc"):
            self._deq_alloc = 0
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
        if not hasattr(self, "_head_dim"):
            self._head_dim = None
        if not hasattr(self, "_value_dtype"):
            self._value_dtype = None
        if not hasattr(self, "buffer_size"):
            self.buffer_size = 0
        if not hasattr(self, "flush_batch_size"):
            self.flush_batch_size = 0
        if not hasattr(self, "max_cache_tokens"):
            self.max_cache_tokens = 0
        if not hasattr(self, "_packed_offset"):
            self._packed_offset = self.offset if self._k_indices is not None else 0
        if not hasattr(self, "_buffer_keys"):
            self._buffer_keys = None
        if not hasattr(self, "_buffer_values"):
            self._buffer_values = None
        if not hasattr(self, "_fused_enabled"):
            fused_env = _env_fused_enabled()
            self._fused_enabled = (
                _prefer_fused_default(
                    self.turbo_bits,
                    self.key_bits_override,
                    self.value_bits_override,
                    self.rotation_mode,
                    self.estimator_mode,
                    self.qjl_residual,
                )
                if fused_env is None
                else fused_env
            )
        if not hasattr(self, "qjl_projection_mode"):
            self.qjl_projection_mode = "auto"
        if not hasattr(self, "_qjl_projection_runtime_mode"):
            self._qjl_projection_runtime_mode = None

    def _init_codebook(self, head_dim):
        self._head_dim = head_dim
        if self.key_bits_override is not None:
            self._k_bits = self.key_bits_override
        elif self.estimator_mode == "prod" and self.qjl_residual:
            self._k_bits = int(self.turbo_bits) - 1
        else:
            self._k_bits = int(self.turbo_bits)
        self._v_bits = (
            self.value_bits_override
            if self.value_bits_override is not None
            else int(self.turbo_bits)
        )
        self._k_centroids, self._k_boundaries = _load_codebook(
            self._k_bits, head_dim
        )
        self._v_centroids, self._v_boundaries = _load_codebook(
            self._v_bits, head_dim
        )
        self._rotation, self._rotation_t = _cached_rotation_pair(
            head_dim, mode=self.rotation_mode
        )
        if self.estimator_mode == "prod" and self.qjl_residual:
            resolved_qjl_mode = _resolve_qjl_projection_mode(
                head_dim, self.qjl_projection_mode
            )
            self._qjl_projection_runtime_mode = resolved_qjl_mode
            self._qjl_projection, self._qjl_projection_t = _cached_qjl_projection(
                head_dim, mode=resolved_qjl_mode
            )
        else:
            self._qjl_projection = None
            self._qjl_projection_t = None
            self._qjl_projection_runtime_mode = None

    def _init_fractional_split(self, keys, values):
        self._head_dim = keys.shape[-1]
        self._split_low_bits = math.floor(self.turbo_bits)
        self._split_high_bits = math.ceil(self.turbo_bits)
        (
            self._split_low_idx,
            self._split_high_idx,
            self._split_restore_order,
        ) = _select_fractional_indices(keys, values, self.turbo_bits)
        self._split_low_cache = TurboQuantKVCache(
            bits=self._split_low_bits,
            rotation_mode=self.rotation_mode,
            estimator_mode=self.estimator_mode,
            qjl_residual=self.qjl_residual,
            sparse_v_tau=self.sparse_v_tau,
            qjl_projection_mode=self.qjl_projection_mode,
            decode_buffer=self.decode_buffer,
            buffer_size=self.buffer_size,
            flush_batch_size=self.flush_batch_size,
            max_cache_tokens=self.max_cache_tokens,
        )
        self._split_high_cache = TurboQuantKVCache(
            bits=self._split_high_bits,
            rotation_mode=self.rotation_mode,
            estimator_mode=self.estimator_mode,
            qjl_residual=self.qjl_residual,
            sparse_v_tau=self.sparse_v_tau,
            qjl_projection_mode=self.qjl_projection_mode,
            decode_buffer=self.decode_buffer,
            buffer_size=self.buffer_size,
            flush_batch_size=self.flush_batch_size,
            max_cache_tokens=self.max_cache_tokens,
        )
        # Fractional mode uses the existing integer cache logic as a safe
        # fallback and always materializes dequantized tensors for attention.
        self._split_low_cache._fused_enabled = False
        self._split_high_cache._fused_enabled = False

    def _invalidate_decode_buffer(self):
        self._k_deq_buf = None
        self._v_deq_buf = None
        self._deq_offset = 0
        self._deq_alloc = 0

    @property
    def compressed_tokens(self):
        self._ensure_runtime_attrs()
        if self._fractional_split:
            if self._split_low_cache is None:
                return 0
            return self._split_low_cache.compressed_tokens
        return self._packed_offset

    @property
    def buffer_tokens(self):
        self._ensure_runtime_attrs()
        if self._fractional_split:
            if self._split_low_cache is None:
                return 0
            return self._split_low_cache.buffer_tokens
        if self._buffer_keys is None:
            return 0
        return self._buffer_keys.shape[2]

    def _sync_total_offset(self):
        self.offset = self.compressed_tokens + self.buffer_tokens

    def _supports_recent_buffer(self):
        if self.buffer_size <= 0 or self.decode_buffer or self._fractional_split:
            return False
        if not self._fused_enabled or not hasattr(mx.fast, "turboquant_av_packed_values_batched"):
            return False
        if self.estimator_mode == "mse" or not self.qjl_residual:
            return hasattr(mx.fast, "turboquant_qk_packed_scores")
        if hasattr(mx.fast, "turboquant_qk_prod_scores_batched"):
            return True
        return (
            os.environ.get("MLX_TQ_USE_METAL_QJL_SCORE", "0").lower()
            not in ("0", "false", "f", "no")
            and hasattr(mx.fast, "turboquant_qk_packed_scores_batched")
        )

    def _append_recent_buffer(self, keys, values):
        if self._buffer_keys is None:
            self._buffer_keys = mx.contiguous(keys)
            self._buffer_values = mx.contiguous(values)
        else:
            self._buffer_keys = mx.concatenate([self._buffer_keys, keys], axis=2)
            self._buffer_values = mx.concatenate([self._buffer_values, values], axis=2)
        self._sync_total_offset()

    def _compress_store(self, keys, values):
        B, n_kv_heads, num_steps, head_dim = keys.shape
        prev = self._packed_offset
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
                pack_qjl=True,
            )
            pk_qjl = k_qjl
            k_gamma = k_gamma.astype(mx.float32)
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
        k_norms = k_norms.astype(mx.float32)
        v_norms = v_norms.astype(mx.float32)

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
        self._packed_offset += num_steps
        self._sync_total_offset()

    def _eval_recent_storage(self):
        arrays = []
        if self._packed_offset > 0 and self._k_indices is not None:
            arrays.extend(
                [
                    self._k_indices[..., : self._packed_offset, :],
                    self._k_norms[..., : self._packed_offset, :],
                    self._v_indices[..., : self._packed_offset, :],
                    self._v_norms[..., : self._packed_offset, :],
                ]
            )
            if self.estimator_mode == "prod" and self._k_qjl_indices is not None:
                arrays.extend(
                    [
                        self._k_qjl_indices[..., : self._packed_offset, :],
                        self._k_qjl_gamma[..., : self._packed_offset, :],
                    ]
                )
        if arrays:
            mx.eval(*arrays)

    def _flush_recent_buffer(self, keep_recent=None):
        self._ensure_runtime_attrs()
        if self._buffer_keys is None or self._buffer_values is None:
            return
        keep_recent = self.buffer_size if keep_recent is None else max(0, int(keep_recent))
        n_flush = self.buffer_tokens - keep_recent
        if n_flush <= 0:
            return

        chunk_size = self.flush_batch_size or self.buffer_size or n_flush
        if chunk_size <= 0:
            chunk_size = n_flush

        keys_to_flush = self._buffer_keys[:, :, :n_flush, :]
        values_to_flush = self._buffer_values[:, :, :n_flush, :]
        self._buffer_keys = self._buffer_keys[:, :, n_flush:, :]
        self._buffer_values = self._buffer_values[:, :, n_flush:, :]
        mx.eval(keys_to_flush, values_to_flush, self._buffer_keys, self._buffer_values)

        for start in range(0, n_flush, chunk_size):
            end = min(start + chunk_size, n_flush)
            self._compress_store(
                keys_to_flush[:, :, start:end, :],
                values_to_flush[:, :, start:end, :],
            )
            self._eval_recent_storage()
        self._sync_total_offset()
        self._enforce_max_cache_tokens()

    def _evict_oldest(self, n):
        self._ensure_runtime_attrs()
        n = min(self.offset, max(0, int(n)))
        if n <= 0:
            return 0
        self._invalidate_decode_buffer()
        if self._fractional_split:
            if self._split_low_cache is None or self._split_high_cache is None:
                return 0
            self._split_low_cache._evict_oldest(n)
            self._split_high_cache._evict_oldest(n)
            self.offset = self._split_low_cache.offset
            return n

        drop_packed = min(self._packed_offset, n)
        if drop_packed > 0:
            keep_packed = self._packed_offset - drop_packed

            def _slice_or_none(arr):
                if arr is None:
                    return None
                if keep_packed <= 0:
                    return None
                return mx.contiguous(arr[..., drop_packed : self._packed_offset, :])

            self._k_indices = _slice_or_none(self._k_indices)
            self._k_norms = _slice_or_none(self._k_norms)
            self._v_indices = _slice_or_none(self._v_indices)
            self._v_norms = _slice_or_none(self._v_norms)
            self._k_qjl_indices = _slice_or_none(self._k_qjl_indices)
            self._k_qjl_gamma = _slice_or_none(self._k_qjl_gamma)
            self._packed_offset = keep_packed

        drop_buffer = n - drop_packed
        if drop_buffer > 0 and self._buffer_keys is not None:
            keep_buffer = self.buffer_tokens - drop_buffer
            if keep_buffer > 0:
                self._buffer_keys = mx.contiguous(
                    self._buffer_keys[..., drop_buffer:, :]
                )
                self._buffer_values = mx.contiguous(
                    self._buffer_values[..., drop_buffer:, :]
                )
            else:
                self._buffer_keys = None
                self._buffer_values = None
        self._sync_total_offset()
        return n

    def _enforce_max_cache_tokens(self):
        self._ensure_runtime_attrs()
        if self.max_cache_tokens > 0 and self.offset > self.max_cache_tokens:
            self._evict_oldest(self.offset - self.max_cache_tokens)

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

    def _append_decode_buffer_raw(self, prev, keys, values):
        if self._k_deq_buf is not None and self._deq_offset != prev:
            return None, None
        if self._v_deq_buf is not None and self._deq_offset != prev:
            return None, None

        total = prev + keys.shape[2]
        batch_size = keys.shape[0]
        n_kv_heads = keys.shape[1]
        self._ensure_decode_buffer_capacity(
            batch_size, n_kv_heads, total, keys.dtype, values.dtype
        )
        self._k_deq_buf[..., prev:total, :] = keys
        self._v_deq_buf[..., prev:total, :] = values
        self._deq_offset = total
        return self._k_deq_buf[..., :total, :], self._v_deq_buf[..., :total, :]

    def _append_decode_buffer(
        self,
        prev,
        num_steps,
        k_dtype,
        v_dtype,
        k_idx,
        k_norms,
        v_idx,
        v_norms,
        pk_qjl,
        k_gamma,
    ):
        if (
            self._k_deq_buf is None
            or self._v_deq_buf is None
            or self._deq_offset != prev
        ):
            return None, None

        total = prev + num_steps
        batch_size = k_idx.shape[0]
        n_kv_heads = k_idx.shape[1]
        self._ensure_decode_buffer_capacity(
            batch_size, n_kv_heads, total, k_dtype, v_dtype
        )

        if self.estimator_mode == "prod" and self.qjl_residual:
            qjl_signs = _unpack(pk_qjl, 1, self._head_dim)
            new_k = _dequantize_prod(
                k_idx,
                k_norms,
                self._rotation,
                self._k_centroids,
                qjl_signs,
                k_gamma,
                self._qjl_projection,
                mode=self.rotation_mode,
            )
        else:
            new_k = _dequantize(
                k_idx, k_norms, self._rotation, self._k_centroids, mode=self.rotation_mode
            )
        new_v = _dequantize(
            v_idx, v_norms, self._rotation, self._v_centroids, mode=self.rotation_mode
        )

        if new_k.dtype != k_dtype:
            new_k = new_k.astype(k_dtype)
        if new_v.dtype != v_dtype:
            new_v = new_v.astype(v_dtype)

        self._k_deq_buf[..., prev:total, :] = new_k
        self._v_deq_buf[..., prev:total, :] = new_v
        self._deq_offset = total
        return self._k_deq_buf[..., :total, :], self._v_deq_buf[..., :total, :]

    def _dequantize_keys(self, limit=None, *, include_qjl=True, dtype=None):
        self._ensure_runtime_attrs()
        if self._fractional_split:
            if self._split_low_cache is None or self._split_high_cache is None:
                return None
            low = self._split_low_cache._dequantize_keys(
                limit=limit,
                include_qjl=include_qjl,
                dtype=dtype,
            )
            high = self._split_high_cache._dequantize_keys(
                limit=limit,
                include_qjl=include_qjl,
                dtype=dtype,
            )
            keys = _merge_split_tensors(low, high, self._split_restore_order)
            if dtype is not None:
                keys = keys.astype(dtype)
            return mx.contiguous(keys)
        limit = self.offset if limit is None else limit
        packed_limit = min(limit, self._packed_offset)
        keys = None
        if self._k_indices is not None and packed_limit > 0:
            indices = _unpack(
                self._k_indices[..., :packed_limit, :], self._k_bits, self._head_dim
            )
            norms = self._k_norms[..., :packed_limit, :]
            if (
                self.estimator_mode == "prod"
                and self.qjl_residual
                and self._k_qjl_indices is not None
                and self._k_qjl_gamma is not None
                and self._qjl_projection is not None
            ):
                qjl_signs = _unpack(
                    self._k_qjl_indices[..., :packed_limit, :], 1, self._head_dim
                )
                if include_qjl:
                    keys = _dequantize_prod(
                        indices,
                        norms,
                        self._rotation,
                        self._k_centroids,
                        qjl_signs,
                        self._k_qjl_gamma[..., :packed_limit, :],
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
                    indices,
                    norms,
                    self._rotation,
                    self._k_centroids,
                    mode=self.rotation_mode,
                )
        buf_limit = max(0, limit - packed_limit)
        if buf_limit > 0 and self._buffer_keys is not None:
            buf = self._buffer_keys[..., :buf_limit, :]
            keys = buf if keys is None else mx.concatenate([keys, buf], axis=2)
        if keys is None:
            return None
        if dtype is not None:
            keys = keys.astype(dtype)
        return mx.contiguous(keys)

    def _dequantize_values(self, limit=None, *, dtype=None):
        self._ensure_runtime_attrs()
        if self._fractional_split:
            if self._split_low_cache is None or self._split_high_cache is None:
                return None
            low = self._split_low_cache._dequantize_values(limit=limit, dtype=dtype)
            high = self._split_high_cache._dequantize_values(limit=limit, dtype=dtype)
            values = _merge_split_tensors(low, high, self._split_restore_order)
            if dtype is not None:
                values = values.astype(dtype)
            return mx.contiguous(values)
        limit = self.offset if limit is None else limit
        packed_limit = min(limit, self._packed_offset)
        values = None
        if self._v_indices is not None and packed_limit > 0:
            indices = _unpack(
                self._v_indices[..., :packed_limit, :], self._v_bits, self._head_dim
            )
            values = _dequantize(
                indices,
                self._v_norms[..., :packed_limit, :],
                self._rotation,
                self._v_centroids,
                mode=self.rotation_mode,
            )
        buf_limit = max(0, limit - packed_limit)
        if buf_limit > 0 and self._buffer_values is not None:
            buf = self._buffer_values[..., :buf_limit, :]
            values = buf if values is None else mx.concatenate([values, buf], axis=2)
        if values is None:
            return None
        if dtype is not None:
            values = values.astype(dtype)
        return mx.contiguous(values)

    def update_and_fetch(self, keys, values):
        self._ensure_runtime_attrs()
        if self._fractional_split:
            self._value_dtype = values.dtype
            if self._split_low_cache is None or self._split_high_cache is None:
                self._init_fractional_split(keys, values)

            low_keys = mx.take(keys, self._split_low_idx, axis=-1)
            high_keys = mx.take(keys, self._split_high_idx, axis=-1)
            low_values = mx.take(values, self._split_low_idx, axis=-1)
            high_values = mx.take(values, self._split_high_idx, axis=-1)

            dk_low, dv_low = self._split_low_cache.update_and_fetch(low_keys, low_values)
            dk_high, dv_high = self._split_high_cache.update_and_fetch(
                high_keys, high_values
            )
            self.offset = self._split_low_cache.offset

            dk = _merge_split_tensors(dk_low, dk_high, self._split_restore_order)
            dv = _merge_split_tensors(dv_low, dv_high, self._split_restore_order)
            if keys.dtype != dk.dtype:
                dk = dk.astype(keys.dtype)
            if values.dtype != dv.dtype:
                dv = dv.astype(values.dtype)
            return mx.contiguous(dk), mx.contiguous(dv)

        if self._supports_recent_buffer():
            _, _, num_steps, head_dim = keys.shape
            self._value_dtype = values.dtype
            if self._k_centroids is None:
                self._init_codebook(head_dim)
            self._append_recent_buffer(keys, values)
            if num_steps == 1:
                flush_threshold = self.buffer_size + (
                    self.flush_batch_size or self.buffer_size
                )
                if flush_threshold <= 0 or self.buffer_tokens > flush_threshold:
                    self._flush_recent_buffer(self.buffer_size)
            self._enforce_max_cache_tokens()
            return self._buffer_keys, self._buffer_values

        B, n_kv_heads, num_steps, head_dim = keys.shape
        prev = self._packed_offset
        self._value_dtype = values.dtype

        if self._k_centroids is None:
            self._init_codebook(head_dim)

        if self.estimator_mode == "prod" and self.qjl_residual:
            k_idx, k_qjl, k_gamma, k_norms = _quantize_prod(
                keys,
                self._rotation_t,
                self._rotation,
                self._k_boundaries,
                self._k_centroids,
                self._qjl_projection_t,
                mode=self.rotation_mode,
                pack_qjl=True,
            )
            pk_qjl = k_qjl
            # These tensors are tiny relative to the packed cache, so keep them
            # in float32 and avoid float16->float32 casts on every decode step.
            k_gamma = k_gamma.astype(mx.float32)
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
        k_norms = k_norms.astype(mx.float32)
        v_norms = v_norms.astype(mx.float32)

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
        self._packed_offset += num_steps
        self._sync_total_offset()
        self._enforce_max_cache_tokens()
        prev = max(0, self._packed_offset - num_steps)

        if self.decode_buffer:
            buffered_k, buffered_v = self._append_decode_buffer_raw(prev, keys, values)
            if buffered_k is not None and buffered_v is not None:
                return buffered_k, buffered_v
            if num_steps <= 4:
                buffered_k, buffered_v = self._append_decode_buffer(
                    prev,
                    num_steps,
                    keys.dtype,
                    values.dtype,
                    k_idx,
                    k_norms,
                    v_idx,
                    v_norms,
                    pk_qjl,
                    k_gamma,
                )
                if buffered_k is not None and buffered_v is not None:
                    return buffered_k, buffered_v
            return self._materialize_decode_buffer(keys.dtype, values.dtype)

        can_use_fused_decode = self._fused_enabled and B == 1 and num_steps == 1
        can_use_native_decode = can_use_fused_decode and (
            (
                (self.estimator_mode == "mse" or not self.qjl_residual)
                and self._k_bits == self._v_bits
                and hasattr(mx.fast, "turboquant_decode_attention_packed_batched")
            )
            or (
                self.estimator_mode == "prod"
                and self.qjl_residual
                and hasattr(mx.fast, "turboquant_decode_attention_prod_batched")
            )
        )
        can_use_fused_av = can_use_fused_decode and hasattr(
            mx.fast, "turboquant_av_packed_values_batched"
        )
        can_use_fused_qk = can_use_fused_decode and (
            (
                (self.estimator_mode == "mse" or not self.qjl_residual)
                and hasattr(mx.fast, "turboquant_qk_packed_scores")
            )
            or (
                self.estimator_mode == "prod"
                and self.qjl_residual
                and hasattr(mx.fast, "turboquant_qk_prod_scores_batched")
            )
        )

        all_v = None
        if not (can_use_native_decode or (can_use_fused_qk and can_use_fused_av)):
            all_v = self._dequantize_values(dtype=values.dtype)
        all_k = None if (can_use_native_decode or can_use_fused_qk) else self._dequantize_keys(
            dtype=keys.dtype
        )
        return all_k, all_v

    def fused_scores(self, queries_scaled):
        """Compute QK scores from packed keys using fused native MLX op.

        Args:
            queries_scaled: [B, n_q_heads, L, D], already scaled by 1/sqrt(D).

        Returns:
            scores [B, n_q_heads, L, T] or None if unsupported.
        """
        self._ensure_runtime_attrs()
        packed_tokens = self.compressed_tokens
        if self._fractional_split:
            return None
        if (
            not self._fused_enabled
            or self._k_indices is None
            or packed_tokens <= 0
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
        q_rot_in = _fused_input(q_rot, mx.float32)

        if self.estimator_mode == "prod" and self.qjl_residual:
            fast_scores = self._fused_prod_scores_metal(
                queries_scaled,
                q_rot_in,
                n_repeats,
            )
            if fast_scores is not None:
                return fast_scores
            if not hasattr(mx.fast, "turboquant_qk_prod_scores_batched"):
                return None
            return mx.fast.turboquant_qk_prod_scores_batched(
                q_rot_in,
                _fused_input(queries_scaled, mx.float32),
                _fused_input(self._k_indices[..., : packed_tokens, :]),
                _fused_input(self._k_norms[..., : packed_tokens, 0], mx.float32),
                self._k_centroids,
                self._k_bits,
                _fused_input(self._k_qjl_indices[..., : packed_tokens, :]),
                _fused_input(self._k_qjl_gamma[..., : packed_tokens, 0], mx.float32),
                self._qjl_projection,
                n_repeats,
            )

        if not hasattr(mx.fast, "turboquant_qk_packed_scores"):
            return None

        if hasattr(mx.fast, "turboquant_qk_packed_scores_batched"):
            # Materialize stable contiguous inputs for the native batched op.
            q_in = q_rot_in
            kp_in = _fused_input(self._k_indices[..., : packed_tokens, :])
            kn_in = _fused_input(self._k_norms[..., : packed_tokens, 0], mx.float32)
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
                    self._k_indices[b, hkv, : packed_tokens, :],  # [T, W]
                    self._k_norms[b, hkv, : packed_tokens, 0],  # [T]
                    self._k_centroids,
                    self._k_bits,
                )  # [L, T]
                per_head.append(s)
            per_batch.append(mx.stack(per_head, axis=0))
        return mx.stack(per_batch, axis=0)  # [B, n_q_heads, L, T]

    def fused_attention(self, queries_scaled):
        """Compute packed decode attention in one native op when supported."""
        self._ensure_runtime_attrs()
        packed_tokens = self.compressed_tokens
        if self._fractional_split:
            return None
        if (
            not self._fused_enabled
            or self._k_indices is None
            or self._v_indices is None
            or packed_tokens <= 0
            or queries_scaled.ndim != 4
            or self.buffer_tokens > 0
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
        q_rot_in = _fused_input(q_rot, mx.float32)
        used_model_basis_native = False
        supports_model_basis_native = getattr(self._rotation, "ndim", 2) != 1
        if self.estimator_mode == "prod" and self.qjl_residual:
            fast_scores = self._fused_prod_scores_metal(
                queries_scaled,
                q_rot_in,
                n_repeats,
            )
            if fast_scores is not None:
                probs = mx.softmax(fast_scores, axis=-1, precise=True)
                out = self.fused_av(probs)
                if out is not None:
                    return out
            if supports_model_basis_native and hasattr(
                mx.fast, "turboquant_decode_attention_prod_model_batched"
            ):
                out = mx.fast.turboquant_decode_attention_prod_model_batched(
                    q_rot_in,
                    _fused_input(queries_scaled, mx.float32),
                    _fused_input(self._k_indices[..., : packed_tokens, :]),
                    _fused_input(self._k_norms[..., : packed_tokens, 0], mx.float32),
                    self._k_centroids,
                    self._k_bits,
                    _fused_input(self._k_qjl_indices[..., : packed_tokens, :]),
                    _fused_input(self._k_qjl_gamma[..., : packed_tokens, 0], mx.float32),
                    self._qjl_projection,
                    _fused_input(self._v_indices[..., : packed_tokens, :]),
                    _fused_input(self._v_norms[..., : packed_tokens, 0], mx.float32),
                    self._v_centroids,
                    self._v_bits,
                    n_repeats,
                    self._head_dim,
                    self._rotation,
                )
                used_model_basis_native = True
            elif not hasattr(mx.fast, "turboquant_decode_attention_prod_batched"):
                return None
            else:
                out = mx.fast.turboquant_decode_attention_prod_batched(
                    q_rot_in,
                    _fused_input(queries_scaled, mx.float32),
                    _fused_input(self._k_indices[..., : packed_tokens, :]),
                    _fused_input(self._k_norms[..., : packed_tokens, 0], mx.float32),
                    self._k_centroids,
                    self._k_bits,
                    _fused_input(self._k_qjl_indices[..., : packed_tokens, :]),
                    _fused_input(self._k_qjl_gamma[..., : packed_tokens, 0], mx.float32),
                    self._qjl_projection,
                    _fused_input(self._v_indices[..., : packed_tokens, :]),
                    _fused_input(self._v_norms[..., : packed_tokens, 0], mx.float32),
                    self._v_centroids,
                    self._v_bits,
                    n_repeats,
                    self._head_dim,
                )
        else:
            if self._k_bits != self._v_bits:
                return None
            if supports_model_basis_native and hasattr(
                mx.fast, "turboquant_decode_attention_packed_model_batched"
            ):
                out = mx.fast.turboquant_decode_attention_packed_model_batched(
                    q_rot_in,
                    _fused_input(self._k_indices[..., : packed_tokens, :]),
                    _fused_input(self._k_norms[..., : packed_tokens, 0], mx.float32),
                    _fused_input(self._v_indices[..., : packed_tokens, :]),
                    _fused_input(self._v_norms[..., : packed_tokens, 0], mx.float32),
                    self._k_centroids,
                    self._k_bits,
                    n_repeats,
                    self._head_dim,
                    self._rotation,
                )
                used_model_basis_native = True
            elif not hasattr(mx.fast, "turboquant_decode_attention_packed_batched"):
                return None
            else:
                out = mx.fast.turboquant_decode_attention_packed_batched(
                    q_rot_in,
                    _fused_input(self._k_indices[..., : packed_tokens, :]),
                    _fused_input(self._k_norms[..., : packed_tokens, 0], mx.float32),
                    _fused_input(self._v_indices[..., : packed_tokens, :]),
                    _fused_input(self._v_norms[..., : packed_tokens, 0], mx.float32),
                    self._k_centroids,
                    self._k_bits,
                    n_repeats,
                    self._head_dim,
                )
        if not used_model_basis_native:
            out = _apply_inverse_rotation(out, self._rotation, mode=self.rotation_mode)
        if self._value_dtype is not None:
            out = out.astype(self._value_dtype)
        return out

    def _fused_prod_scores_metal(self, queries_scaled, q_rot_in, n_repeats):
        packed_tokens = self.compressed_tokens
        if (
            self.estimator_mode != "prod"
            or not self.qjl_residual
            or os.environ.get("MLX_TQ_USE_METAL_QJL_SCORE", "0").lower()
            in ("0", "false", "f", "no")
            or not hasattr(mx.fast, "turboquant_qk_packed_scores_batched")
            or queries_scaled.ndim != 4
            or queries_scaled.shape[2] != 1
            or self._qjl_projection_t is None
            or self._k_qjl_indices is None
            or self._k_qjl_gamma is None
            or packed_tokens <= 0
        ):
            return None

        B, Hq, _, D = queries_scaled.shape
        Hkv = self._k_indices.shape[1]
        if Hq != Hkv * n_repeats:
            return None

        mse_scores = mx.fast.turboquant_qk_packed_scores_batched(
            q_rot_in,
            _fused_input(self._k_indices[..., : packed_tokens, :]),
            _fused_input(self._k_norms[..., : packed_tokens, 0], mx.float32),
            self._k_centroids,
            self._k_bits,
            n_repeats,
        )

        q_model = _fused_input(queries_scaled, mx.float32)
        q_proj = _apply_qjl_projection(q_model, self._qjl_projection_t)
        q_proj = _fused_input(mx.reshape(q_proj, (B, Hkv, n_repeats, D)), mx.float32)
        qjl_scale = mx.array(
            [math.sqrt(math.pi / 2.0) / D],
            dtype=mx.float32,
        )
        corr_scores = _metal_qjl_score(
            q_proj,
            self._k_norms[..., : packed_tokens, 0],
            self._k_qjl_gamma[..., : packed_tokens, 0],
            self._k_qjl_indices[..., : packed_tokens, :],
            qjl_scale,
        )
        if corr_scores is None:
            return None
        corr_scores = mx.reshape(corr_scores, (B, Hq, 1, packed_tokens))
        return mse_scores + corr_scores

    def fused_av(self, probs):
        """Compute attention output from packed values using native MLX op.

        Args:
            probs: [B, n_q_heads, L, T] attention probabilities.

        Returns:
            out [B, n_q_heads, L, D] or None if unsupported.
        """
        self._ensure_runtime_attrs()
        packed_tokens = self.compressed_tokens
        if self._fractional_split:
            return None
        if (
            not self._fused_enabled
            or not hasattr(mx.fast, "turboquant_av_packed_values_batched")
            or self._v_indices is None
            or packed_tokens <= 0
            or probs.ndim != 4
        ):
            return None

        B, n_q_heads, _, T = probs.shape
        n_kv_heads = self._v_indices.shape[1]
        if B != self._v_indices.shape[0] or T != packed_tokens:
            return None
        if n_q_heads % n_kv_heads != 0:
            return None

        n_repeats = n_q_heads // n_kv_heads
        p_in = _fused_input(probs, mx.float32)
        vp_in = _fused_input(self._v_indices[..., : packed_tokens, :])
        vn_in = _fused_input(self._v_norms[..., : packed_tokens, 0], mx.float32)
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
        packed_tokens = self.compressed_tokens

        def _new():
            arrays = [
                mx.zeros((*shape, k_packed_dim), dtype=mx.uint32),
                mx.zeros((*shape, 1), dtype=mx.float32),
                mx.zeros((*shape, v_packed_dim), dtype=mx.uint32),
                mx.zeros((*shape, 1), dtype=mx.float32),
            ]
            if self.estimator_mode == "prod" and self.qjl_residual:
                arrays.extend(
                    [
                        mx.zeros((*shape, qjl_packed_dim), dtype=mx.uint32),
                        mx.zeros((*shape, 1), dtype=mx.float32),
                    ]
                )
            return arrays

        if self._k_indices is not None and packed_tokens > 0:
            old = [
                self._k_indices[..., :packed_tokens, :],
                self._k_norms[..., :packed_tokens, :],
                self._v_indices[..., :packed_tokens, :],
                self._v_norms[..., :packed_tokens, :],
            ]
            if self.estimator_mode == "prod" and self.qjl_residual:
                old.extend(
                    [
                        self._k_qjl_indices[..., :packed_tokens, :],
                        self._k_qjl_gamma[..., :packed_tokens, :],
                    ]
                )
            arrays = [mx.concatenate([o, n], axis=2) for o, n in zip(old, _new())]
        else:
            arrays = _new()

        self._k_indices = arrays[0]
        self._k_norms = arrays[1]
        self._v_indices = arrays[2]
        self._v_norms = arrays[3]
        if self.estimator_mode == "prod" and self.qjl_residual:
            self._k_qjl_indices = arrays[4]
            self._k_qjl_gamma = arrays[5]
        else:
            self._k_qjl_indices = None
            self._k_qjl_gamma = None

    def size(self):
        return self.offset

    @property
    def state(self):
        self._ensure_runtime_attrs()
        if self._fractional_split:
            if self._split_low_cache is None or self._split_high_cache is None:
                return []
            return [
                self._split_low_idx,
                self._split_high_idx,
                self._split_restore_order,
                self._split_low_cache.state,
                self._split_high_cache.state,
            ]
        if self.buffer_tokens > 0:
            self._flush_recent_buffer(0)
        if self._k_indices is None or self._packed_offset <= 0:
            return []
        state = [
            self._k_indices[..., :self._packed_offset, :],
            self._k_norms[..., :self._packed_offset, :],
            self._v_indices[..., :self._packed_offset, :],
            self._v_norms[..., :self._packed_offset, :],
        ]
        if self.estimator_mode == "prod" and self.qjl_residual:
            state.extend(
                [
                    self._k_qjl_indices[..., :self._packed_offset, :],
                    self._k_qjl_gamma[..., :self._packed_offset, :],
                ]
            )
        return state

    @state.setter
    def state(self, v):
        self._ensure_runtime_attrs()
        self._invalidate_decode_buffer()
        if v is not None and v:
            if len(v) == 5 and isinstance(v[3], (list, tuple)) and isinstance(v[4], (list, tuple)):
                self._pending_split_state = v
                self._split_low_idx, self._split_high_idx, self._split_restore_order = v[:3]
                self._k_indices = None
                self._k_norms = None
                self._k_qjl_indices = None
                self._k_qjl_gamma = None
                self._v_indices = None
                self._v_norms = None
                self._packed_offset = 0
                self._buffer_keys = None
                self._buffer_values = None
                self.offset = 0
                return
            self._k_indices, self._k_norms, self._v_indices, self._v_norms = v[:4]
            if len(v) >= 6:
                self._k_qjl_indices, self._k_qjl_gamma = v[4:6]
            else:
                self._k_qjl_indices = None
                self._k_qjl_gamma = None
            self._packed_offset = self._k_indices.shape[2]
            self._buffer_keys = None
            self._buffer_values = None
            self.offset = self._packed_offset

    @property
    def meta_state(self):
        self._ensure_runtime_attrs()
        head_dim = self._head_dim or 0
        sparse_tau = -1.0 if self.sparse_v_tau is None else float(self.sparse_v_tau)
        decode_buffer = "1" if self.decode_buffer else "0"
        if self._fractional_split:
            low_meta = ()
            high_meta = ()
            if self._split_low_cache is not None:
                low_meta = tuple(map(str, self._split_low_cache.meta_state))
            if self._split_high_cache is not None:
                high_meta = tuple(map(str, self._split_high_cache.meta_state))
            return (
                str(self.offset),
                str(float(self.turbo_bits)),
                str(head_dim),
                self.rotation_mode,
                str(sparse_tau),
                self.estimator_mode,
                self._qjl_projection_runtime_mode or self.qjl_projection_mode,
                decode_buffer,
                str(self.buffer_size),
                str(self.flush_batch_size),
                "-" if self.key_bits_override is None else str(self.key_bits_override),
                "-" if self.value_bits_override is None else str(self.value_bits_override),
                "1" if self.qjl_residual else "0",
                str(self.max_cache_tokens),
                "split",
                low_meta,
                high_meta,
            )
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
                    self._qjl_projection_runtime_mode or self.qjl_projection_mode,
                    decode_buffer,
                    self.buffer_size,
                    self.flush_batch_size,
                    "-" if self.key_bits_override is None else self.key_bits_override,
                    "-" if self.value_bits_override is None else self.value_bits_override,
                    "1" if self.qjl_residual else "0",
                    self.max_cache_tokens,
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, v):
        self._ensure_runtime_attrs()
        self.offset = int(v[0])
        self.turbo_bits = _normalize_turbo_bits(v[1])
        head_dim = int(v[2])
        self.rotation_mode = v[3] if len(v) >= 4 else "dense"
        if self.rotation_mode not in ("dense", "wht", "rotor3", "rotorquant"):
            self.rotation_mode = "dense"
        if len(v) >= 5:
            sparse_tau = float(v[4])
            self.sparse_v_tau = None if sparse_tau < 0 else sparse_tau
        else:
            self.sparse_v_tau = None
        self.estimator_mode = v[5] if len(v) >= 6 else "mse"
        if self.estimator_mode not in ("mse", "prod"):
            self.estimator_mode = "mse"
        next_idx = 6
        if len(v) > next_idx and v[next_idx] in ("auto", "dense", "wht"):
            self.qjl_projection_mode = v[next_idx]
            next_idx += 1
        else:
            self.qjl_projection_mode = "auto"
        if len(v) > next_idx and v[next_idx] in ("0", "1", "false", "true"):
            self.decode_buffer = v[next_idx] in ("1", "true")
            next_idx += 1
        else:
            self.decode_buffer = False
        if len(v) > next_idx and v[next_idx] != "split":
            self.buffer_size = int(v[next_idx])
            next_idx += 1
        else:
            self.buffer_size = 0
        if len(v) > next_idx and v[next_idx] != "split":
            self.flush_batch_size = int(v[next_idx])
            next_idx += 1
        else:
            self.flush_batch_size = self.buffer_size
        if len(v) > next_idx and v[next_idx] != "split":
            self.key_bits_override = (
                None if v[next_idx] in ("-", "", "none", "None") else int(v[next_idx])
            )
            next_idx += 1
        else:
            self.key_bits_override = None
        if len(v) > next_idx and v[next_idx] != "split":
            self.value_bits_override = (
                None if v[next_idx] in ("-", "", "none", "None") else int(v[next_idx])
            )
            next_idx += 1
        else:
            self.value_bits_override = None
        if len(v) > next_idx and v[next_idx] != "split":
            self.qjl_residual = v[next_idx] in ("1", "true", "True")
            next_idx += 1
        else:
            self.qjl_residual = True
        if len(v) > next_idx and v[next_idx] != "split":
            self.max_cache_tokens = int(v[next_idx])
            next_idx += 1
        else:
            self.max_cache_tokens = 0
        self._qjl_projection_runtime_mode = None
        self._fractional_split = len(v) > next_idx and v[next_idx] == "split"
        self._fused_enabled = os.environ.get("MLX_TQ_FUSED", "0").lower() not in (
            "0",
            "false",
            "f",
            "no",
        )
        self._invalidate_decode_buffer()
        self._buffer_keys = None
        self._buffer_values = None
        self._packed_offset = self.offset
        if self._fractional_split:
            self._split_low_bits = math.floor(float(self.turbo_bits))
            self._split_high_bits = math.ceil(float(self.turbo_bits))
            if self._pending_split_state is not None:
                (
                    self._split_low_idx,
                    self._split_high_idx,
                    self._split_restore_order,
                    low_state,
                    high_state,
                ) = self._pending_split_state
                low_meta = v[next_idx + 1] if len(v) > next_idx + 1 else ()
                high_meta = v[next_idx + 2] if len(v) > next_idx + 2 else ()
                self._split_low_cache = TurboQuantKVCache.from_state(low_state, low_meta)
                self._split_high_cache = TurboQuantKVCache.from_state(
                    high_state, high_meta
                )
                self._split_low_cache._fused_enabled = False
                self._split_high_cache._fused_enabled = False
                self.offset = self._split_low_cache.offset
                self._packed_offset = self._split_low_cache.compressed_tokens
                self._pending_split_state = None
            self._head_dim = head_dim if head_dim > 0 else None
            return
        if head_dim > 0:
            self._init_codebook(head_dim)

    def is_trimmable(self):
        return True

    def trim(self, n):
        self._ensure_runtime_attrs()
        self._invalidate_decode_buffer()
        if self._fractional_split:
            if self._split_low_cache is None or self._split_high_cache is None:
                return 0
            n = min(self.offset, n)
            self._split_low_cache.trim(n)
            self._split_high_cache.trim(n)
            self.offset = self._split_low_cache.offset
            return n
        n = min(self.offset, n)
        trim_buf = min(self.buffer_tokens, n)
        if trim_buf > 0:
            keep = self.buffer_tokens - trim_buf
            if keep > 0:
                self._buffer_keys = self._buffer_keys[:, :, :keep, :]
                self._buffer_values = self._buffer_values[:, :, :keep, :]
            else:
                self._buffer_keys = None
                self._buffer_values = None
        remaining = n - trim_buf
        if remaining > 0:
            self._packed_offset = max(0, self._packed_offset - remaining)
        self._sync_total_offset()
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        self._ensure_runtime_attrs()
        if self._fractional_split:
            return self._split_low_cache is None or self._split_low_cache.empty()
        return self._packed_offset <= 0 and self.buffer_tokens <= 0

    @property
    def nbytes(self):
        self._ensure_runtime_attrs()
        if self._fractional_split:
            if self._split_low_cache is None or self._split_high_cache is None:
                return 0
            index_arrays = [
                self._split_low_idx,
                self._split_high_idx,
                self._split_restore_order,
            ]
            index_bytes = sum(a.nbytes for a in index_arrays if a is not None)
            return index_bytes + self._split_low_cache.nbytes + self._split_high_cache.nbytes
        total = 0
        if self._k_indices is not None and self._packed_offset > 0:
            arrays = [self._k_indices, self._k_norms, self._v_indices, self._v_norms]
            if self._k_qjl_indices is not None:
                arrays.extend([self._k_qjl_indices, self._k_qjl_gamma])
            total += sum(a[..., :self._packed_offset, :].nbytes for a in arrays)
        if self._buffer_keys is not None:
            total += self._buffer_keys.nbytes
        if self._buffer_values is not None:
            total += self._buffer_values.nbytes
        return total
