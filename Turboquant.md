# TurboQuant Model Guide

This file is the short practical companion to the main [README](README.md).
It only answers one question:

**If you want the best speed/accuracy/memory compromise on this machine, which model/profile should you start with?**

Machine used for all rows below:
- Apple M3 Max
- 128 GB unified memory
- Python 3.12

This guide assumes the forked MLX runtime from `guillaume-osmo/mlx` on branch
`codex/turboquant-prod-qk`, not the stock `ml-explore/mlx` package.

Recommended runtime flags for the exact profiles below:

```bash
export MLX_TQ_FUSED=1
export MLX_TQ_QK_CENTROID_LUT=1
```

`Exact match` means greedy generation produced the same generated suffix as native on the listed workload.

## Best Model By Family

| Family | Recommended model | Workload | Recommended exact profile | Native gen tok/s | Compressed gen tok/s | Cache MB (native -> compressed) | Exact match | Recommendation |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| Qwen (compact) | `mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit` | `2048` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, edge-4 FP16 layers | `48.95` | `50.17` | `126.0 -> 55.52` | `16/16` | Best small Qwen-family speed/accuracy compromise in the fused/LUT rerun |
| Qwen (mid-size) | `mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit` | `2048` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, no edge layers | `24.85` | `24.89` | `432.0 -> 107.78` | `16/16` | Best exact 14B tradeoff measured so far |
| Qwen (long context) | `mlx-community/Qwen3.5-35B-A3B-4bit` | `16384` prompt / `50` decode | `prod`, `K=3`, `V=4`, QJL **off**, dense rotation, fused on, FP16 late layers `[23,27,31,35,39]` | `34.56` | `35.52` | `356.41 -> 231.52` | `50/50` | Best long-context result; exact and slightly faster than native |
| Llama | `mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit` | `2048` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, edge-4 FP16 layers | `47.49` | `47.82` | `288.0 -> 118.84` | `16/16` | Best Llama-family speed/accuracy compromise in the fused/LUT rerun |
| Mistral | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | `2048` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, edge-4 FP16 layers | `47.04` | `46.77` | `288.0 -> 118.84` | `16/16` | Best Mistral-family exact compromise |
| Gemma | `mlx-community/gemma-4-e2b-it-4bit` | `2048` prompt / `16` decode | `mse`, `3.5-bit`, half split, edge-2 FP16 layers | `55.54` | `89.73` | `19.50 -> 12.40` | `16/16` | Strongest Gemma-family exact result so far on this machine |
| SmolLM | `Irfanuruchi/SmolLM2-1.7B-Instruct-MLX-4bit` | `2048` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, fused on | `111.56` | `109.01` | `432.0 -> 130.18` | `16/16` | Best SmolLM-family memory/throughput compromise in the refreshed fused/LUT run |

## Full Exact Benchmark Table

| Model | Workload | Best exact profile so far | Native gen tok/s | Compressed gen tok/s | Cache MB (native -> compressed) | Exact match | Takeaway |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | `4096` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, edge-4 FP16 layers | `45.67` | `44.67` | `238.0 -> 106.89` | `16/16` | Best exact speed/accuracy point in the fused/LUT rerun |
| `mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit` | `2048` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, edge-4 FP16 layers | `48.95` | `50.17` | `126.0 -> 55.52` | `16/16` | Best compact Qwen-family tradeoff in the refreshed fused/LUT run |
| `mlx-community/Qwen2.5-32B-Instruct-4bit` | `4096` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, no edge layers | `11.20` | `10.50` | `1088.0 -> 275.13` | `16/16` | Best exact compressed profile in the refreshed fused/LUT run |
| `mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit` | `2048` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, no edge layers | `24.85` | `24.89` | `432.0 -> 107.78` | `16/16` | Best exact 14B tradeoff measured so far |
| `mlx-community/Qwen3.5-35B-A3B-4bit` | `16384` prompt / `50` decode | `prod`, `K=3`, `V=4`, QJL **off**, dense rotation, fused on, FP16 late layers `[23,27,31,35,39]` | `34.56` | `35.52` | `356.41 -> 231.52` | `50/50` | Exact long-context winner on this machine |
| `mlx-community/Qwen3.5-35B-A3B-4bit` | `32768` prompt / `50` decode | `prod`, `K=3`, `V=4`, QJL **off**, dense rotation, fused on, FP16 late layers `[23,27,31,35,39]` | `8.76` | `9.12` | `676.41 -> 429.02` | `50/50` | Exact and slightly faster than native at 32K |
| `mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit` | `2048` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, edge-4 FP16 layers | `47.49` | `47.82` | `288.0 -> 118.84` | `16/16` | Best Llama-family speed/accuracy compromise in the refreshed fused/LUT run |
| `mlx-community/Meta-Llama-3.1-8B-Instruct-8bit` | `2048` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, edge-4 FP16 layers | `27.95` | `26.91` | `288.0 -> 118.84` | `16/16` | Best exact compressed profile in the refreshed fused/LUT run |
| `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | `2048` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, edge-4 FP16 layers | `47.04` | `46.77` | `288.0 -> 118.84` | `16/16` | Best Mistral-family exact compromise |
| `mlx-community/gemma-4-e2b-it-4bit` | `2048` prompt / `16` decode | `mse`, `3.5-bit`, half split, edge-2 FP16 layers | `55.54` | `89.73` | `19.50 -> 12.40` | `16/16` | Strongest Gemma-family exact result so far on this machine |
| `unsloth/gemma-4-E4B-it-UD-MLX-4bit` | `2048` prompt / `16` decode | `mse`, `4-bit` | `33.93` | `9.95` | `56.00 -> 35.14` | `16/16` | Exact and healthy, but clearly not a speed win on this short harness |
| `mlx-community/gemma-4-e4b-it-6bit` | `2048` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, edge-2 FP16 layers | `9.37` | `35.24` | `56.00 -> 34.57` | `16/16` | Exact and surprisingly fast on this short harness; treat as provisional until a longer rerun confirms it |
| `Irfanuruchi/SmolLM2-1.7B-Instruct-MLX-4bit` | `2048` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, fused on | `111.56` | `109.01` | `432.0 -> 130.18` | `16/16` | Best SmolLM-family memory/throughput compromise in the refreshed fused/LUT run |
| `mlx-community/SmolLM3-3B-4bit` | `2048` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off** | `89.85` | `74.64` | `162.0 -> 42.18` | `16/16` | Native stays faster, but the exact cache reduction is strong |

## Gemma 4 Sanity Check

To make sure the Gemma 4 benchmark rows are not just "fast nonsense", we also
ran a plain generation prompt on the supported Gemma 4 checkpoints.

Prompt:

```text
In two short sentences, explain why exact-match validation matters for TurboQuant.
```

Outputs:

```text
mlx-community/gemma-4-e2b-it-4bit
Exact-match validation ensures that the input data precisely matches the expected format, which is crucial for the high-precision calculations TurboQuant relies upon. This prevents errors in complex algorithms by guaranteeing the integrity of the data before processing.
```

```text
unsloth/gemma-4-E4B-it-UD-MLX-4bit
Exact-match validation ensures that the data ingested by TurboQuant precisely aligns with expected formats and identifiers. This strictness is crucial for maintaining data integrity and enabling accurate, reliable quantitative analysis.
```

```text
mlx-community/gemma-4-e4b-it-6bit
Exact-match validation ensures that the data ingested into TurboQuant precisely aligns with expected formats and identifiers. This accuracy is crucial for reliable quantitative analysis, preventing misinterpretations or errors in financial modeling.
```

This is worth calling out because the short benchmark continuation for
`gemma-4-e2b-it-4bit` was unusually easy. The sanity-check prompt confirms that
the Gemma 4 adapters in this branch produce normal text, not empty output.

## What The Numbers Mean

- `mse, 4-bit`:
  the safest compressed baseline for many dense models
- `prod, K=3, V=4, QJL off`:
  the strongest exact asymmetric profile in this branch for several families
- `edge-4 FP16 layers`:
  keep a small set of sensitive layers in higher precision
- `late layers [23,27,31,35,39]`:
  the current Qwen3.5 long-context winner

## Rule Of Thumb

1. Start with native.
2. Try `mse, 4-bit`.
3. If you need more speed/quality, try `prod, K=3, V=4, QJL off`.
4. Only then tune protected FP16 layers.

## How To Reproduce

Run the full validated benchmark sweep with the faster fused runtime enabled:

```bash
MLX_TQ_FUSED=1 \
MLX_TQ_QK_CENTROID_LUT=1 \
/Users/tgg/turboquant-m3max/.venv-py3.12-local/bin/python \
  /Users/tgg/Downloads/turboquant_m3max_bundle/turboquantbenchmarch.py \
  --models qwen25_7b qwen25_32b qwen35_long_exact \
           deepseek_r1_qwen_7b deepseek_r1_qwen_14b deepseek_r1_llama_8b \
           llama31_8b mistral_7b_v03 gemma4_e2b smollm2_17b smollm3_3b \
  --trials 1
```

List the available reproducible benchmark cases:

```bash
/Users/tgg/turboquant-m3max/.venv-py3.12-local/bin/python \
  /Users/tgg/Downloads/turboquant_m3max_bundle/turboquantbenchmarch.py --list
```

Print the exact generation command for one winning preset:

```bash
MLX_TQ_FUSED=1 \
MLX_TQ_QK_CENTROID_LUT=1 \
python3 /Users/tgg/Downloads/turboquant_m3max_bundle/turboquantdemo.py \
  --preset qwen35_long_exact --print-only
```

Run a single validated winner preset end-to-end:

```bash
MLX_TQ_FUSED=1 \
MLX_TQ_QK_CENTROID_LUT=1 \
python3 /Users/tgg/Downloads/turboquant_m3max_bundle/turboquantdemo.py \
  --preset deepseek_qwen_7b_best
```

Available demo presets include:

- `qwen35_long_exact`
- `deepseek_qwen_7b_best`
- `deepseek_qwen_14b_best`
- `deepseek_llama_8b_best`
- `mistral_7b_best`
- `gemma4_e2b_best`
- `smollm2_17b_best`
- `smollm3_3b_best`

## Important Caveat

These rows are **measured exact-match results on this machine**, not universal paper claims.
TurboQuant reduces KV cache memory reliably.
Speed still depends strongly on the model family and on the extra preprocessing cost:

- rotation
- quantization
- packing
- optional QJL correction
- decode-buffer maintenance
