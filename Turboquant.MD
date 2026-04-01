# TurboQuant Model Guide

This file is the short practical companion to the main [README](/Users/tgg/turboquant-m3max/mlx-lm/README.md).
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
| SmolLM | `mlx-community/SmolLM3-3B-4bit` | `2048` prompt / `16` decode | `mse`, `4-bit` | `93.45` | `81.07` | `162.0 -> 45.39` | `16/16` | Native is faster, but memory reduction is large and exact |

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

## Important Caveat

These rows are **measured exact-match results on this machine**, not universal paper claims.
TurboQuant reduces KV cache memory reliably.
Speed still depends strongly on the model family and on the extra preprocessing cost:

- rotation
- quantization
- packing
- optional QJL correction
- decode-buffer maintenance
