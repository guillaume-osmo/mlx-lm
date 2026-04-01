# TurboQuant Model Guide

This file is the short practical companion to the main [README](README.md).
It only answers one question:

**If you want the best speed/accuracy/memory compromise on this machine, which model/profile should you start with?**

Machine used for all rows below:
- Apple M3 Max
- 128 GB unified memory
- Python 3.12

`Exact match` means greedy generation produced the same generated suffix as native on the listed workload.

## Best Model By Family

| Family | Recommended model | Workload | Recommended exact profile | Native gen tok/s | Compressed gen tok/s | Cache MB (native -> compressed) | Exact match | Recommendation |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| Qwen (compact) | `mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit` | `2048` prompt / `16` decode | `mse`, `4-bit` | `52.44` | `49.10` | `126.0 -> 36.83` | `16/16` | Best small Qwen-family starting point |
| Qwen (mid-size) | `mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit` | `2048` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, no edge layers | `24.70` | `22.79` | `432.0 -> 107.78` | `16/16` | Best exact 14B tradeoff measured so far |
| Qwen (long context) | `mlx-community/Qwen3.5-35B-A3B-4bit` | `16384` prompt / `50` decode | `prod`, `K=3`, `V=4`, QJL **off**, dense rotation, fused on, FP16 late layers `[23,27,31,35,39]` | `34.56` | `35.52` | `356.41 -> 231.52` | `50/50` | Best long-context result; exact and slightly faster than native |
| Llama | `mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit` | `2048` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, edge-4 FP16 layers | `46.96` | `44.15` | `288.0 -> 118.84` | `16/16` | Best Llama-family speed/accuracy compromise |
| Mistral | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | `2048` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, edge-4 FP16 layers | `48.75` | `46.59` | `288.0 -> 118.84` | `16/16` | Best Mistral-family exact compromise |
| Gemma | `mlx-community/gemma-3-text-12b-it-4bit` | `2048` prompt / `16` decode | `mse`, `4-bit` | `26.73` | `26.64` | `464.0 -> 367.09` | `16/16` | Exact, but the gain is small; native is still reasonable |
| SmolLM | `Irfanuruchi/SmolLM2-1.7B-Instruct-MLX-4bit` | `2048` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, fused on | `107.38` | `119.74` | `432.0 -> 157.62` | `16/16` | Strongest speed+memory win in the small-model family |

## Full Exact Benchmark Table

| Model | Workload | Best exact profile so far | Native gen tok/s | Compressed gen tok/s | Cache MB (native -> compressed) | Exact match | Takeaway |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | `4096` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off** | `43.4` | `39.8` | `238.0 -> 54.4` | `16/16` | Strong memory win, moderate speed cost |
| `mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit` | `2048` prompt / `16` decode | `mse`, `4-bit` | `52.44` | `49.10` | `126.0 -> 36.83` | `16/16` | Best compact Qwen-family tradeoff |
| `mlx-community/Qwen2.5-32B-Instruct-4bit` | `4096` prompt / `16` decode | `mse`, `4-bit` | `9.8` | `10.1` | `1088.0 -> 273.0` | `16/16` | Best current dense 32B profile |
| `mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit` | `2048` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, no edge layers | `24.70` | `22.79` | `432.0 -> 107.78` | `16/16` | Best exact 14B tradeoff measured so far |
| `mlx-community/Qwen3.5-35B-A3B-4bit` | `16384` prompt / `50` decode | `prod`, `K=3`, `V=4`, QJL **off**, dense rotation, fused on, FP16 late layers `[23,27,31,35,39]` | `34.56` | `35.52` | `356.41 -> 231.52` | `50/50` | Exact long-context winner on this machine |
| `mlx-community/Qwen3.5-35B-A3B-4bit` | `32768` prompt / `50` decode | `prod`, `K=3`, `V=4`, QJL **off**, dense rotation, fused on, FP16 late layers `[23,27,31,35,39]` | `8.76` | `9.12` | `676.41 -> 429.02` | `50/50` | Exact and slightly faster than native at 32K |
| `mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit` | `2048` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, edge-4 FP16 layers | `46.96` | `44.15` | `288.0 -> 118.84` | `16/16` | Best Llama-family speed/accuracy compromise |
| `mlx-community/Meta-Llama-3.1-8B-Instruct-8bit` | `2048` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, fused on | `26.01` | `24.57` | `288.0 -> 76.55` | `16/16` | Strong cache reduction, older non-DeepSeek Llama baseline |
| `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | `2048` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, edge-4 FP16 layers | `48.75` | `46.59` | `288.0 -> 118.84` | `16/16` | Best Mistral-family exact compromise |
| `mlx-community/gemma-3-text-12b-it-4bit` | `2048` prompt / `16` decode | `mse`, `4-bit` | `26.73` | `26.64` | `464.0 -> 367.09` | `16/16` | Exact, but only a modest memory reduction |
| `Irfanuruchi/SmolLM2-1.7B-Instruct-MLX-4bit` | `2048` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, fused on | `107.38` | `119.74` | `432.0 -> 157.62` | `16/16` | Strongest speed+memory win among the small models |
| `mlx-community/SmolLM3-3B-4bit` | `2048` prompt / `16` decode | `mse`, `4-bit` | `93.45` | `81.07` | `162.0 -> 45.39` | `16/16` | Native stays faster, but memory reduction is strong |

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

Run the full validated benchmark sweep with:

```bash
/Users/tgg/turboquant-m3max/.venv-py3.12-local/bin/python \
  /Users/tgg/Downloads/turboquant_m3max_bundle/turboquantbenchmarch.py \
  --models qwen35_long_exact deepseek_r1_qwen_7b deepseek_r1_qwen_14b \
           deepseek_r1_llama_8b mistral_7b_v03 gemma3_text_12b smollm2_17b smollm3_3b \
  --trials 1
```

List the available reproducible benchmark cases:

```bash
/Users/tgg/turboquant-m3max/.venv-py3.12-local/bin/python \
  /Users/tgg/Downloads/turboquant_m3max_bundle/turboquantbenchmarch.py --list
```

Print the exact generation command for one winning preset:

```bash
python3 /Users/tgg/Downloads/turboquant_m3max_bundle/turboquantdemo.py \
  --preset qwen35_long_exact --print-only
```

Run a single validated winner preset end-to-end:

```bash
python3 /Users/tgg/Downloads/turboquant_m3max_bundle/turboquantdemo.py \
  --preset deepseek_qwen_7b_best
```

Available demo presets include:

- `qwen35_long_exact`
- `deepseek_qwen_7b_best`
- `deepseek_qwen_14b_best`
- `deepseek_llama_8b_best`
- `mistral_7b_best`
- `gemma_12b_best`
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
