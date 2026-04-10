"""TriAttention Calibration: compute Q-center statistics for KV cache compression.

Usage::

    python -m mlx_lm.triattention_calibrate \\
        --model mlx-community/Qwen2.5-7B-Instruct-4bit \\
        --output triattention_calib.safetensors \\
        --max-tokens 4096

Based on blaizzy/mlx-vlm PR #985 and arXiv:2604.04921.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .models.triattention import (
    RoPEConfig,
    TriAttentionCalibData,
    _decompose_complex,
    _find_attention,
    _find_layers,
    extract_model_info,
    save_calibration,
)

DEFAULT_CALIBRATION_TEXT = (
    "Mathematics is the study of numbers, shapes, and patterns. "
    "The Pythagorean theorem states that in a right triangle, "
    "the square of the hypotenuse equals the sum of the squares "
    "of the other two sides. In calculus, the derivative measures "
    "the rate of change of a function. Integration computes the "
    "area under a curve. Linear algebra deals with vectors and "
    "matrices. Probability theory quantifies uncertainty.\n\n"
    "Computer science encompasses algorithms, data structures, "
    "and computational theory. Machine learning enables computers "
    "to learn from data without explicit programming. Neural networks "
    "are inspired by biological neurons and form the basis of deep "
    "learning. Transformers use self-attention mechanisms to process "
    "sequences in parallel, enabling breakthroughs in natural language "
    "processing and computer vision.\n\n"
    "Physics explores the fundamental laws governing the universe. "
    "Newton's laws of motion describe how objects move under forces. "
    "Einstein's theory of relativity revolutionized our understanding "
    "of space and time. Quantum mechanics describes behavior at atomic "
    "scales, where particles exhibit wave-particle duality. "
    "The standard model of particle physics classifies fundamental "
    "particles and their interactions.\n\n"
    "Biology studies living organisms and their interactions. DNA "
    "carries genetic information encoded in sequences of nucleotides. "
    "Evolution through natural selection drives species adaptation. "
    "Ecology examines relationships between organisms and their "
    "environment. The human brain contains approximately 86 billion "
    "neurons connected by trillions of synapses."
)


class _CaptureWrapper:
    """Wrapper that captures pre-RoPE Q vectors."""

    def __init__(self, wrapped: nn.Module, capture_list: List[mx.array]):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_capture_list", capture_list)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_wrapped"), name)

    def __call__(self, x, mask=None, cache=None, **kwargs):
        wrapped = object.__getattribute__(self, "_wrapped")
        capture_list = object.__getattribute__(self, "_capture_list")

        B, L, _ = x.shape
        n_heads = getattr(wrapped, "n_heads", None) or getattr(
            wrapped, "num_heads", None
        )
        if n_heads is not None:
            q = wrapped.q_proj(x).reshape(B, L, n_heads, -1)
            if hasattr(wrapped, "q_norm"):
                q = wrapped.q_norm(q)
            capture_list.append(mx.stop_gradient(q))

        return wrapped(x, mask=mask, cache=cache, **kwargs)


def _install_capture_hooks(
    model: nn.Module,
    captures: Dict[int, List[mx.array]],
    skip_sliding: bool = True,
) -> List[Any]:
    """Install capture wrappers on attention layers."""
    layers = _find_layers(model)
    if layers is None:
        raise ValueError("Cannot find transformer layers in model")

    hooks = []
    for layer_idx, layer in enumerate(layers):
        attr_name = None
        attn = None
        for name in ("self_attn", "attention"):
            if hasattr(layer, name):
                attr_name = name
                attn = getattr(layer, name)
                break
        if attn is None:
            continue

        if skip_sliding and getattr(attn, "is_sliding", False):
            continue

        captures[layer_idx] = []
        wrapper = _CaptureWrapper(attn, captures[layer_idx])
        setattr(layer, attr_name, wrapper)
        hooks.append((layer, attr_name, attn))

    return hooks


def _remove_hooks(hooks: List[Any]) -> None:
    for layer, attr_name, original_attn in hooks:
        setattr(layer, attr_name, original_attn)


def compute_statistics(
    captures: Dict[int, List[mx.array]],
    rope_config: RoPEConfig,
    n_q_heads: int,
    n_kv_heads: int,
    n_layers: int,
) -> TriAttentionCalibData:
    """Compute frequency-domain statistics from captured Q vectors."""
    n_freqs = rope_config.rotated_dims // 2

    q_center_real = {}
    q_center_imag = {}
    q_mean_norm_dict = {}

    for layer_idx in range(n_layers):
        if layer_idx not in captures or not captures.get(layer_idx):
            q_center_real[layer_idx] = mx.zeros((n_q_heads, n_freqs))
            q_center_imag[layer_idx] = mx.zeros((n_q_heads, n_freqs))
            q_mean_norm_dict[layer_idx] = mx.zeros((n_q_heads, n_freqs))

    for layer_idx in sorted(captures.keys()):
        layer_qs = captures[layer_idx]
        if not layer_qs:
            q_center_real[layer_idx] = mx.zeros((n_q_heads, n_freqs))
            q_center_imag[layer_idx] = mx.zeros((n_q_heads, n_freqs))
            q_mean_norm_dict[layer_idx] = mx.zeros((n_q_heads, n_freqs))
            continue

        all_q = mx.concatenate(layer_qs, axis=1)
        mx.eval(all_q)

        center_real_list = []
        center_imag_list = []
        mean_norm_list = []

        for head_idx in range(n_q_heads):
            q_head = all_q[0, :, head_idx, :]

            real, imag = _decompose_complex(q_head, rope_config)

            cr = mx.mean(real, axis=0)
            ci = mx.mean(imag, axis=0)

            mag = mx.sqrt(real * real + imag * imag + 1e-12)
            mn = mx.mean(mag, axis=0)

            center_real_list.append(cr)
            center_imag_list.append(ci)
            mean_norm_list.append(mn)

        q_center_real[layer_idx] = mx.stack(center_real_list)
        q_center_imag[layer_idx] = mx.stack(center_imag_list)
        q_mean_norm_dict[layer_idx] = mx.stack(mean_norm_list)

        mx.eval(
            q_center_real[layer_idx],
            q_center_imag[layer_idx],
            q_mean_norm_dict[layer_idx],
        )

    return TriAttentionCalibData(
        q_center_real=q_center_real,
        q_center_imag=q_center_imag,
        q_mean_norm=q_mean_norm_dict,
        n_layers=n_layers,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
    )


def calibrate(
    model_path: str,
    output_path: str = "triattention_calib.safetensors",
    calibration_text: Optional[str] = None,
    max_tokens: int = 4096,
) -> None:
    """Run calibration and save statistics."""
    from mlx_lm import load
    from .models.cache import make_prompt_cache

    print(f"Loading model: {model_path}")
    model, tokenizer = load(
        model_path, tokenizer_config={"trust_remote_code": True}
    )

    info = extract_model_info(model)
    if info is None:
        raise ValueError(
            f"Cannot extract model info from {model_path}. "
            "Unsupported architecture."
        )
    n_layers, n_q_heads, n_kv_heads, head_dim, rope_config = info
    print(
        f"Model: {n_layers} layers, {n_q_heads} Q heads, "
        f"{n_kv_heads} KV heads, head_dim={head_dim}, "
        f"rotated_dims={rope_config.rotated_dims}"
    )

    text = calibration_text or DEFAULT_CALIBRATION_TEXT
    tokens = tokenizer.encode(text)
    if isinstance(tokens, list):
        tokens = tokens[:max_tokens]
    else:
        tokens = tokens[:max_tokens].tolist()

    input_ids = mx.array([tokens])
    print(f"Calibration tokens: {len(tokens)}")

    captures: Dict[int, List[mx.array]] = {}
    hooks = _install_capture_hooks(model, captures)
    print(f"Installed hooks on {len(hooks)} attention layers")

    print("Running forward pass...")
    cache = make_prompt_cache(model)

    try:
        model(input_ids, cache=cache)
        mx.eval()
    finally:
        _remove_hooks(hooks)

    captured_layers = [k for k, v in captures.items() if v]
    print(
        f"Captured Q vectors from {len(captured_layers)} layers "
        f"(skipped {n_layers - len(captured_layers)} sliding layers)"
    )

    print("Computing frequency-domain statistics...")
    calib = compute_statistics(
        captures, rope_config, n_q_heads, n_kv_heads, n_layers
    )

    if captured_layers:
        diag_layer = captured_layers[0]
        diag_mag = mx.sqrt(
            calib.q_center_real[diag_layer] ** 2
            + calib.q_center_imag[diag_layer] ** 2
            + 1e-12
        )
        mean_mrl = mx.mean(
            diag_mag / (calib.q_mean_norm[diag_layer] + 1e-6)
        ).item()
        print(f"Layer {diag_layer} mean MRL (Q concentration): {mean_mrl:.4f}")

    save_calibration(calib, output_path)
    print(f"Saved calibration to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="TriAttention calibration: compute Q-center statistics"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model path or HF repo."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="triattention_calib.safetensors",
        help="Output path for calibration file.",
    )
    parser.add_argument(
        "--calibration-text", type=str, default=None, help="Custom text."
    )
    parser.add_argument(
        "--calibration-file",
        type=str,
        default=None,
        help="Path to a text file for calibration data.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096, help="Max tokens to process."
    )

    args = parser.parse_args()

    text = args.calibration_text
    if args.calibration_file:
        with open(args.calibration_file, "r") as f:
            text = f.read()

    calibrate(
        model_path=args.model,
        output_path=args.output,
        calibration_text=text,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
