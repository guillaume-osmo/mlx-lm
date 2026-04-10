"""
Data-driven codebook calibration for TurboQuant KV cache compression.

Replaces the default Gaussian-optimal Lloyd-Max codebooks with codebooks
derived from real KV-cache data via uniform quantile estimation.  The
calibrated centroids and boundaries have the same shapes as the hardcoded
ones, so fused Metal kernels need no changes.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from .cache import KVCache, make_prompt_cache


# ---------------------------------------------------------------------------
# Core calibration algorithm
# ---------------------------------------------------------------------------

def calibrate_codebook(
    samples_flat: np.ndarray,
    bits: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute codebook centroids and boundaries via uniform quantile estimation.

    Args:
        samples_flat: 1-D float64/float32 array of rotated, unit-normalised
            KV-cache coordinates collected from a calibration pass.
        bits: Quantisation bit-width (1-4).

    Returns:
        (centroids, boundaries) as float32 numpy arrays.
        centroids has shape ``[2**bits]``, boundaries ``[2**bits + 1]``.
    """
    num_entries = 1 << bits
    flat = samples_flat.astype(np.float64).ravel()
    if flat.size < num_entries:
        raise ValueError(
            f"Need at least {num_entries} samples for {bits}-bit calibration, "
            f"got {flat.size}"
        )

    # Lloyd-Max iteration on the empirical distribution.
    #
    # 1. Initialise centroids via equal-probability quantile bin means.
    # 2. Iterate: boundaries = midpoints, centroids = conditional means.
    #
    # This converges to the MSE-optimal scalar quantizer for the actual
    # data distribution (matching the hardcoded Gaussian Lloyd-Max tables
    # when the data is truly N(0,1), and adapting to non-Gaussian shapes
    # for real KV-cache coordinates).

    # --- Initialisation: equal-probability quantile bins ---
    bin_edges_q = np.linspace(0.0, 1.0, num_entries + 1)
    bin_edges = np.quantile(flat, bin_edges_q)

    centroids = np.empty(num_entries, dtype=np.float64)
    for i in range(num_entries):
        lo, hi = bin_edges[i], bin_edges[i + 1] if i < num_entries - 1 else np.inf
        mask = (flat >= lo) & (flat < hi) if i < num_entries - 1 else (flat >= lo)
        centroids[i] = flat[mask].mean() if mask.any() else lo
    centroids.sort()

    # --- Lloyd-Max iteration ---
    max_iters = 50
    for _ in range(max_iters):
        # Boundaries: midpoints between adjacent centroids
        boundaries = np.empty(num_entries + 1, dtype=np.float64)
        boundaries[0] = -np.inf
        boundaries[-1] = np.inf
        for i in range(1, num_entries):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0

        # Centroids: conditional mean in each bin
        new_centroids = np.empty_like(centroids)
        for i in range(num_entries):
            mask = (flat >= boundaries[i]) & (flat < boundaries[i + 1])
            new_centroids[i] = flat[mask].mean() if mask.any() else centroids[i]
        new_centroids.sort()

        if np.max(np.abs(new_centroids - centroids)) < 1e-7:
            centroids = new_centroids
            break
        centroids = new_centroids

    centroids = centroids.astype(np.float32)

    # Final boundaries with ±5.0 endpoints (matching Gaussian codebook convention).
    boundaries_out = np.empty(num_entries + 1, dtype=np.float32)
    boundaries_out[0] = -5.0
    boundaries_out[-1] = 5.0
    for i in range(1, num_entries):
        boundaries_out[i] = (float(centroids[i - 1]) + float(centroids[i])) / 2.0

    return centroids, boundaries_out


# ---------------------------------------------------------------------------
# Sample collection via standard FP16 forward pass
# ---------------------------------------------------------------------------

def collect_rotated_samples(
    model: nn.Module,
    tokenizer,
    tokens: mx.array,
    rotation_mode: str = "dense",
    max_samples: int = 500_000,
    prefill_step_size: int = 512,
) -> np.ndarray:
    """Run a forward pass and collect rotated, unit-normalised KV coordinates.

    This creates a standard (uncompressed) KV cache, runs the model forward on
    *tokens*, then extracts K/V from every layer, applies the same rotation and
    normalisation that ``_quantize`` uses, and returns a flat sample array.

    Args:
        model: Loaded MLX language model.
        tokenizer: Corresponding tokenizer (unused directly but kept for API
            symmetry with ``run_calibration``).
        tokens: 1-D int32 token array to use as calibration data.
        rotation_mode: Rotation type (``dense``, ``wht``, ``rotor3``,
            ``rotorquant``).  Must match the mode used at inference time.
        max_samples: Cap on the total number of scalar coordinate samples to
            keep.  If the forward pass produces more, a uniform random subset
            is returned.
        prefill_step_size: Number of tokens to process per forward chunk.

    Returns:
        1-D float32 numpy array of rotated, unit-normalised coordinates.
    """
    from .turboquant import _apply_rotation, _cached_rotation_pair

    # 1. Build a standard FP16 cache (no TurboQuant).
    cache = make_prompt_cache(model)

    # 2. Prefill in chunks.
    tokens_1d = tokens.reshape(-1)
    T = tokens_1d.shape[0]
    for start in range(0, T, prefill_step_size):
        end = min(start + prefill_step_size, T)
        chunk = tokens_1d[start:end][None]  # [1, chunk_len]
        model(chunk, cache=cache)
        mx.eval([c.state for c in cache if hasattr(c, "state")])

    # 3. Extract K/V, apply rotation, collect coordinates.
    all_samples: List[np.ndarray] = []
    total_collected = 0

    for layer_cache in cache:
        if not isinstance(layer_cache, KVCache):
            continue
        for tensor in (layer_cache.keys, layer_cache.values):
            if tensor is None:
                continue
            # tensor: [B, n_heads, seq_len, head_dim]
            t = tensor[..., : layer_cache.offset, :]
            if t.size == 0:
                continue

            head_dim = t.shape[-1]
            _, rotation_t = _cached_rotation_pair(head_dim, mode=rotation_mode)

            # Unit-normalise (matching _quantize at turboquant.py:351-357)
            t_f = t.astype(mx.float32)
            norms = mx.linalg.norm(t_f, axis=-1, keepdims=True)
            unit = t_f / mx.maximum(norms, 1e-8)

            # Apply rotation
            rotated = _apply_rotation(unit, rotation_t, mode=rotation_mode)
            mx.eval(rotated)

            flat = np.asarray(rotated).ravel()
            all_samples.append(flat)
            total_collected += flat.size

    if not all_samples:
        raise RuntimeError("No KV-cache samples collected — is the model loaded?")

    combined = np.concatenate(all_samples)

    # Reservoir-subsample if over budget.
    if combined.size > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(combined.size, size=max_samples, replace=False)
        combined = combined[idx]

    return combined.astype(np.float32)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_calibration(
    model: nn.Module,
    tokenizer,
    bits_list: List[int],
    tokens: mx.array,
    rotation_mode: str = "dense",
    max_samples: int = 500_000,
    prefill_step_size: int = 512,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Run end-to-end calibration and return codebooks for each bit-width.

    Args:
        model: Loaded MLX language model.
        tokenizer: Corresponding tokenizer.
        bits_list: List of integer bit-widths to calibrate (e.g. ``[3, 4]``).
        tokens: 1-D int32 token array for calibration.
        rotation_mode: Rotation type to use.
        max_samples: Maximum coordinate samples to collect.
        prefill_step_size: Prefill chunk size.

    Returns:
        Dict mapping each bit-width to ``{"centroids": np.array,
        "boundaries": np.array}``.
    """
    samples = collect_rotated_samples(
        model,
        tokenizer,
        tokens,
        rotation_mode=rotation_mode,
        max_samples=max_samples,
        prefill_step_size=prefill_step_size,
    )

    result: Dict[int, Dict[str, np.ndarray]] = {}
    for bits in bits_list:
        c, b = calibrate_codebook(samples, bits)
        result[bits] = {"centroids": c, "boundaries": b}

    return result


# ---------------------------------------------------------------------------
# Persistence (safetensors)
# ---------------------------------------------------------------------------

def save_codebook(codebook_dict: Dict[int, Dict[str, np.ndarray]], path: str) -> None:
    """Save calibrated codebooks to a ``.safetensors`` file.

    Args:
        codebook_dict: Output of ``run_calibration``.
        path: Destination file path (should end in ``.safetensors``).
    """
    tensors = {}
    bits_present = []
    for bits, cb in codebook_dict.items():
        tensors[f"centroids_{bits}"] = mx.array(cb["centroids"], dtype=mx.float32)
        tensors[f"boundaries_{bits}"] = mx.array(cb["boundaries"], dtype=mx.float32)
        bits_present.append(str(bits))

    metadata = {
        "format": "turboquant_codebook_v1",
        "bits": ",".join(sorted(bits_present)),
    }
    mx.save_safetensors(path, tensors, metadata=metadata)


def load_codebook(path: str) -> Dict[int, Dict[str, np.ndarray]]:
    """Load calibrated codebooks from a ``.safetensors`` file.

    Args:
        path: Source file path.

    Returns:
        Dict mapping bit-width to ``{"centroids": np.array,
        "boundaries": np.array}``.
    """
    tensors, metadata = mx.load(path, return_metadata=True)
    if not isinstance(tensors, dict):
        raise ValueError(f"Expected dict from {path}, got {type(tensors)}")

    bits_str = metadata.get("bits", "")
    if not bits_str:
        # Infer from tensor keys.
        bits_set = set()
        for key in tensors:
            if key.startswith("centroids_"):
                bits_set.add(int(key.split("_", 1)[1]))
        bits_list = sorted(bits_set)
    else:
        bits_list = [int(b) for b in bits_str.split(",")]

    result: Dict[int, Dict[str, np.ndarray]] = {}
    for bits in bits_list:
        c_key = f"centroids_{bits}"
        b_key = f"boundaries_{bits}"
        if c_key not in tensors or b_key not in tensors:
            raise ValueError(
                f"Missing {c_key} or {b_key} in codebook file {path}"
            )
        c = np.array(tensors[c_key])
        b = np.array(tensors[b_key])
        expected_entries = 1 << bits
        if c.shape != (expected_entries,):
            raise ValueError(
                f"centroids_{bits} has shape {c.shape}, expected ({expected_entries},)"
            )
        if b.shape != (expected_entries + 1,):
            raise ValueError(
                f"boundaries_{bits} has shape {b.shape}, expected ({expected_entries + 1},)"
            )
        result[bits] = {"centroids": c, "boundaries": b}

    return result
