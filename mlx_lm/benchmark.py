# Copyright © 2025 Apple Inc.

import argparse

import mlx.core as mx

from mlx_lm import batch_generate, load, stream_generate
from mlx_lm.generate import DEFAULT_MODEL
from mlx_lm.utils import pipeline_load


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="LLM benchmarking script")
    parser.add_argument(
        "--model",
        type=str,
        help=(
            "The path to the local model directory or Hugging Face repo. "
            f"If no model is specified, then {DEFAULT_MODEL} is used."
        ),
        default=None,
    )
    parser.add_argument(
        "--prompt-tokens",
        "-p",
        default=512,
        help="Length of prompt",
        type=int,
    )
    parser.add_argument(
        "--generation-tokens",
        "-g",
        default=1024,
        help="Length of completion",
        type=int,
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        default=1,
        help="Batch size",
        type=int,
    )
    parser.add_argument(
        "--num-trials",
        "-n",
        default=5,
        help="Number of timing trials",
        type=int,
    )
    parser.add_argument(
        "--turbo-kv-bits",
        type=int,
        default=None,
        help="[Experimental] Enable TurboQuant KV cache compression (2-4 bits).",
    )
    parser.add_argument(
        "--turbo-rotation-mode",
        type=str,
        choices=["dense", "rotor3", "rotorquant"],
        default="dense",
        help="[Experimental] TurboQuant rotation mode.",
    )
    parser.add_argument(
        "--turbo-estimator-mode",
        type=str,
        choices=["mse", "prod"],
        default="mse",
        help="[Experimental] TurboQuant estimator mode.",
    )
    parser.add_argument(
        "--turbo-sparse-v-tau",
        type=float,
        default=None,
        help="[Experimental] Optional sparse-V threshold in TurboQuant fused decode.",
    )
    return parser


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    mx.random.seed(0)

    group = mx.distributed.init()
    rank = group.rank()

    def rprint(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)

    model_path = args.model or DEFAULT_MODEL

    if group.size() > 1:
        model, tokenizer, config = pipeline_load(args.model, return_config=True)
    else:
        model, tokenizer, config = load(
            args.model, return_config=True, tokenizer_config={"trust_remote_code": True}
        )

    # Empty to avoid early stopping
    tokenizer._eos_token_ids = {}

    prompt_tokens = args.prompt_tokens
    generation_tokens = args.generation_tokens
    batch_size = args.batch_size
    vocab_size = config.get("vocab_size") or config["text_config"]["vocab_size"]
    prompts = mx.random.randint(0, vocab_size, (batch_size, prompt_tokens)).tolist()
    prompt = prompts[0]

    def single_bench():
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=generation_tokens,
            turbo_kv_bits=args.turbo_kv_bits,
            turbo_rotation_mode=args.turbo_rotation_mode,
            turbo_estimator_mode=args.turbo_estimator_mode,
            turbo_sparse_v_tau=args.turbo_sparse_v_tau,
        ):
            pass
        return response

    def batch_bench():
        return batch_generate(
            model,
            tokenizer,
            prompts,
            max_tokens=generation_tokens,
        ).stats

    if batch_size > 1 and args.turbo_kv_bits is not None:
        rprint(
            "[WARNING] TurboQuant benchmark flags currently apply only for --batch-size 1; "
            "running regular batch path for this run."
        )

    if batch_size == 1:
        _bench = single_bench
    else:
        _bench = batch_bench

    rprint("Running warmup..")
    _bench()

    report_keys = ["prompt_tps", "generation_tps", "peak_memory"]
    rprint(f"Timing with {prompt_tokens=}, {generation_tokens=}, {batch_size=}.")
    responses = []
    for i in range(args.num_trials):
        response = _bench()
        responses.append(response)
        results = [(k, getattr(response, k)) for k in report_keys]
        results = [f"{k}={v:.3f}" for k, v in results]
        rprint(f"Trial {i+1}:  " + ", ".join(results))

    def avg(k):
        vals = (getattr(response, k) for response in responses)
        return sum(vals) / args.num_trials

    results = [(k, avg(k)) for k in report_keys]
    results = [f"{k}={v:.3f}" for k, v in results]
    rprint(f"Averages: " + ", ".join(results))


if __name__ == "__main__":
    main()
