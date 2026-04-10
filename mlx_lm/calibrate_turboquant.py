"""
Standalone TurboQuant codebook calibration tool.

Usage::

    python -m mlx_lm.calibrate_turboquant \\
        --model mlx-community/Qwen2.5-7B-4bit \\
        --turbo-kv-bits 4 \\
        --turbo-rotation-mode dense \\
        --calibration-tokens 4096 \\
        --output codebook.safetensors

This runs a forward pass on calibration data, collects rotated KV-cache
coordinate samples, and computes quantile-estimated codebooks that can be
loaded at inference time via ``--turbo-codebook-path``.
"""

import argparse
import time

import mlx.core as mx

from mlx_lm import load
from mlx_lm.models.turboquant_calibrate import (
    run_calibration,
    save_codebook,
)


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate TurboQuant codebooks from real model data."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path or Hugging Face repo.",
    )
    parser.add_argument(
        "--turbo-kv-bits",
        type=float,
        default=4,
        help="TurboQuant bit-width (2-4). Default: 4.",
    )
    parser.add_argument(
        "--turbo-key-bits",
        type=int,
        default=None,
        help="Optional key bit-width override.",
    )
    parser.add_argument(
        "--turbo-value-bits",
        type=int,
        default=None,
        help="Optional value bit-width override.",
    )
    parser.add_argument(
        "--turbo-rotation-mode",
        type=str,
        default="dense",
        choices=["dense", "wht", "rotor3", "rotorquant"],
        help="Rotation mode. Default: dense.",
    )
    parser.add_argument(
        "--calibration-tokens",
        type=int,
        default=4096,
        help="Number of tokens to use for calibration. Default: 4096.",
    )
    parser.add_argument(
        "--calibration-text",
        type=str,
        default=None,
        help="Text to use for calibration. If not provided, uses a generic "
        "prompt repeated to fill --calibration-tokens.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500_000,
        help="Maximum coordinate samples to collect. Default: 500000.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output .safetensors file path.",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = load(
        args.model,
        tokenizer_config={"trust_remote_code": True},
    )

    # Build calibration tokens
    if args.calibration_text is not None:
        tokens = tokenizer.encode(args.calibration_text)
    else:
        # Generic calibration prompt
        cal_text = (
            "The quick brown fox jumps over the lazy dog. "
            "In a hole in the ground there lived a hobbit. "
            "It was a bright cold day in April, and the clocks were striking thirteen. "
            "Call me Ishmael. "
            "It is a truth universally acknowledged, that a single man in "
            "possession of a good fortune, must be in want of a wife. "
        )
        tokens = tokenizer.encode(cal_text)
        # Repeat to fill requested length
        while len(tokens) < args.calibration_tokens:
            tokens = tokens + tokens
    tokens = tokens[: args.calibration_tokens]

    # Determine bit widths to calibrate
    bits_set = {int(args.turbo_kv_bits)}
    if args.turbo_key_bits:
        bits_set.add(int(args.turbo_key_bits))
    if args.turbo_value_bits:
        bits_set.add(int(args.turbo_value_bits))

    print(
        f"Calibrating codebooks for bit-widths {sorted(bits_set)} "
        f"using {len(tokens)} tokens..."
    )
    t0 = time.perf_counter()
    codebook = run_calibration(
        model,
        tokenizer,
        bits_list=sorted(bits_set),
        tokens=mx.array(tokens),
        rotation_mode=args.turbo_rotation_mode,
        max_samples=args.max_samples,
    )
    elapsed = time.perf_counter() - t0
    print(f"Calibration completed in {elapsed:.1f}s")

    for bits, cb in sorted(codebook.items()):
        print(f"  {bits}-bit: centroids={cb['centroids'].tolist()}")

    save_codebook(codebook, args.output)
    print(f"Saved calibrated codebooks to {args.output}")


if __name__ == "__main__":
    main()
