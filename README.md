## MLX LM 

MLX LM is a Python package for generating text and fine-tuning large language
models on Apple silicon with MLX.

Some key features include:

* Integration with the Hugging Face Hub to easily use thousands of LLMs with a
  single command. 
* Support for quantizing and uploading models to the Hugging Face Hub.
* [Low-rank and full model
  fine-tuning](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
  with support for quantized models.
* Distributed inference and fine-tuning with `mx.distributed`

The easiest way to get started is to install the `mlx-lm` package:

**With `pip`**:

```sh
pip install mlx-lm
```

**With `conda`**:

```sh
conda install -c conda-forge mlx-lm
```

### Quick Start

To generate text with an LLM use:

```bash
mlx_lm.generate --prompt "How tall is Mt Everest?"
```

To chat with an LLM use:

```bash
mlx_lm.chat
```

This will give you a chat REPL that you can use to interact with the LLM. The
chat context is preserved during the lifetime of the REPL.

Commands in `mlx-lm` typically take command line options which let you specify
the model, sampling parameters, and more. Use `-h` to see a list of available
options for a command, e.g.:

```bash
mlx_lm.generate -h
```

The default model for generation and chat is
`mlx-community/Llama-3.2-3B-Instruct-4bit`.  You can specify any MLX-compatible
model with the `--model` flag. Thousands are available in the
[MLX Community](https://huggingface.co/mlx-community) Hugging Face
organization.

### Python API

You can use `mlx-lm` as a module:

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

prompt = "Write a story about Einstein"

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True
)

text = generate(model, tokenizer, prompt=prompt, verbose=True)
```

To see a description of all the arguments you can do:

```
>>> help(generate)
```

Check out the [generation
example](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/examples/generate_response.py)
to see how to use the API in more detail. Check out the [batch generation
example](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/examples/batch_generate_response.py)
to see how to efficiently generate continuations for a batch of prompts.

The `mlx-lm` package also comes with functionality to quantize and optionally
upload models to the Hugging Face Hub.

You can convert models using the Python API:

```python
from mlx_lm import convert

repo = "mistralai/Mistral-7B-Instruct-v0.3"
upload_repo = "mlx-community/My-Mistral-7B-Instruct-v0.3-4bit"

convert(repo, quantize=True, upload_repo=upload_repo)
```

This will generate a 4-bit quantized Mistral 7B and upload it to the repo
`mlx-community/My-Mistral-7B-Instruct-v0.3-4bit`. It will also save the
converted model in the path `mlx_model` by default.

To see a description of all the arguments you can do:

```
>>> help(convert)
```

#### Streaming

For streaming generation, use the `stream_generate` function. This yields
a generation response object.

For example,

```python
from mlx_lm import load, stream_generate

repo = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
model, tokenizer = load(repo)

prompt = "Write a story about Einstein"

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True
)

for response in stream_generate(model, tokenizer, prompt, max_tokens=512):
    print(response.text, end="", flush=True)
print()
```

#### Sampling

The `generate` and `stream_generate` functions accept `sampler` and
`logits_processors` keyword arguments. A sampler is any callable which accepts
a possibly batched logits array and returns an array of sampled tokens.  The
`logits_processors` must be a list of callables which take the token history
and current logits as input and return the processed logits. The logits
processors are applied in order.

Some standard sampling functions and logits processors are provided in
`mlx_lm.sample_utils`.

### Command Line

You can also use `mlx-lm` from the command line with:

```
mlx_lm.generate --model mistralai/Mistral-7B-Instruct-v0.3 --prompt "hello"
```

This will download a Mistral 7B model from the Hugging Face Hub and generate
text using the given prompt.

For a full list of options run:

```
mlx_lm.generate --help
```

To quantize a model from the command line run:

```
mlx_lm.convert --hf-path mistralai/Mistral-7B-Instruct-v0.3 -q
```

For more options run:

```
mlx_lm.convert --help
```

You can upload new models to Hugging Face by specifying `--upload-repo` to
`convert`. For example, to upload a quantized Mistral-7B model to the
[MLX Hugging Face community](https://huggingface.co/mlx-community) you can do:

```
mlx_lm.convert \
    --hf-path mistralai/Mistral-7B-Instruct-v0.3 \
    -q \
    --upload-repo mlx-community/my-4bit-mistral
```

Models can also be converted and quantized directly in the
[mlx-my-repo](https://huggingface.co/spaces/mlx-community/mlx-my-repo) Hugging
Face Space.

### Long Prompts and Generations 

`mlx-lm` has some tools to scale efficiently to long prompts and generations:

- A rotating fixed-size key-value cache.
- Prompt caching

To use the rotating key-value cache pass the argument `--max-kv-size n` where
`n` can be any integer. Smaller values like `512` will use very little RAM but
result in worse quality. Larger values like `4096` or higher will use more RAM
but have better quality.

Caching prompts can substantially speedup reusing the same long context with
different queries. To cache a prompt use `mlx_lm.cache_prompt`. For example:

```bash
cat prompt.txt | mlx_lm.cache_prompt \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --prompt - \
  --prompt-cache-file mistral_prompt.safetensors
``` 

Then use the cached prompt with `mlx_lm.generate`:

```
mlx_lm.generate \
    --prompt-cache-file mistral_prompt.safetensors \
    --prompt "\nSummarize the above text."
```

The cached prompt is treated as a prefix to the supplied prompt. Also notice
when using a cached prompt, the model to use is read from the cache and need
not be supplied explicitly.

Prompt caching can also be used in the Python API in order to avoid
recomputing the prompt. This is useful in multi-turn dialogues or across
requests that use the same context. See the
[example](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/examples/chat.py)
for more usage details.

### Experimental TurboQuant Tuning

This branch adds experimental TurboQuant KV-cache compression to
`mlx_lm.generate` and `mlx_lm.benchmark`.

The short version is:

- **cache memory savings are real and repeatable**
- **speed is model-dependent**
- **there is no single best profile for every model family**

Paper results should be treated as a direction, not as a drop-in expectation
for MLX. The compressed path adds real math before the fast attention kernel
can run:

- rotate queries / keys into the compression basis
- normalize and quantize keys / values
- bit-pack compressed tensors
- optionally apply the 1-bit QJL residual correction
- maintain the recent decode buffer and merge it with compressed history

That extra preprocessing cost is why compressed KV can be **much smaller in
memory without automatically being faster than native**. On some models the
cost is well amortized and throughput stays near native or improves. On other
models the main win is cache size, not raw decode speed.

One important implementation detail in this branch: the asymmetric exact winner
`prod, K=3, V=4, QJL off` now auto-enables the safe fused path by default.
That fused path is **not** the one-pass single-bit-width decode kernel, because
that primitive assumes the same bit-width for keys and values. For `K=3/V=4`
we instead route through the correct split path:

- fused packed-key scores
- softmax
- fused packed-value aggregation

That fix matters because it removes a real formal bug: forcing `K=3/V=4`
through a shared-bit decode primitive is mathematically wrong.

#### Exact-Match Results On This Machine

Measured on:

- Apple M3 Max
- 128 GB unified memory

`Exact match` below means: **greedy generation produced the exact same token
suffix as native on the listed workload and prompt**. This is a measured
property of these runs, not a blanket guarantee for every prompt.

| Model | Workload | Best exact profile so far | Native gen tok/s | Compressed gen tok/s | Cache MB (native -> compressed) | Exact match | Takeaway |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | `4096` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off** | `43.4` | `39.8` | `238.0 -> 54.4` | `16/16` | strong memory win, moderate speed cost |
| `mlx-community/Qwen2.5-32B-Instruct-4bit` | `4096` prompt / `16` decode | `mse`, `4-bit` | `9.8` | `10.1` | `1088.0 -> 273.0` | `16/16` | best current dense 32B profile |
| `mlx-community/Qwen3.5-35B-A3B-4bit` | `16384` prompt / `50` decode | `prod`, `K=3`, `V=4`, QJL **off**, fused on | `34.56` | `34.46` | `356.41 -> 131.61` | `50/50` | near-native speed, large cache win |
| `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | `2048` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, fused on | `46.54` | `42.87` | `288.0 -> 90.65` | `16/16` | very good compromise |
| `mlx-community/Meta-Llama-3.1-8B-Instruct-8bit` | `2048` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, fused on | `27.69` | `27.81` | `288.0 -> 90.65` | `16/16` | cache win at speed parity |
| `Irfanuruchi/SmolLM2-1.7B-Instruct-MLX-4bit` | `2048` prompt / `16` decode | `prod`, `K=3`, `V=4`, QJL **off**, fused on | `107.38` | `119.74` | `432.0 -> 157.62` | `16/16` | best speed+memory result so far |

These runs are why this branch now recommends a **model-tuned** workflow
instead of a single "paper" profile.

#### Two Practical Profiles

If you want a clean starting point, use one of these two profiles first:

**1. Simple dense baseline for models that prefer `mse`**

```bash
--turbo-kv-bits 4 \
--turbo-estimator-mode mse \
--turbo-fp16-layers 2 \
--turbo-decode-buffer
```

Best current example:

- `mlx-community/Qwen2.5-32B-Instruct-4bit`

**2. Best exact profile so far for several 7B/8B-style families**

```bash
--turbo-kv-bits 4 \
--turbo-key-bits 3 \
--turbo-value-bits 4 \
--turbo-estimator-mode prod \
--turbo-disable-qjl \
--turbo-fp16-layers 2 \
--turbo-decode-buffer
```

In current builds, this profile auto-enables the fused packed path when the
runtime supports it. Use `MLX_TQ_FUSED=0` only if you want to force the older
reference path for debugging.

Best current examples:

- `mlx-community/Qwen3.5-35B-A3B-4bit` at long context
- `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
- `mlx-community/Meta-Llama-3.1-8B-Instruct-8bit`
- `Irfanuruchi/SmolLM2-1.7B-Instruct-MLX-4bit`

For `Qwen2.5-7B-Instruct-4bit`, the same `prod no-QJL k3/v4` family is also
currently the best exact TurboQuant profile, but with a larger throughput hit
than on Mistral / Llama / SmolLM2.

#### QJL Is Not A Universal Default

QJL is a tuning knob, not a rule.

Current local evidence:

- `Qwen2.5-7B-Instruct-4bit`: best exact `prod` profile is **without** QJL
- `Qwen2.5-32B-Instruct-4bit`: best exact profile is still plain `4-bit mse`
- `Qwen3.5-35B-A3B-4bit`: `prod + QJL` can still help at short context, but at
  `16K` and `32K` the best exact profile is currently `prod` **without** QJL

If you care about exactness, benchmark `prod` both **with** and **without**
QJL on the target model before deciding.

#### Upstream Inputs We Actually Used

This branch is informed by several public experiments, but not all of their
claims transferred directly:

| Source | What mattered here | What did **not** transfer automatically |
| --- | --- | --- |
| `yzamari/mlx-fork` | native MLX fused TurboQuant runtime ideas, including the `mx.fast.turboquant_attention` family and the surrounding Metal integration | public benchmark ratios in README/docs were **not** used as truth without local reproduction |
| `yzamari/mlx-turboquant` / `turboQuantPlayground` | recent-buffer decode, batch-flush / prefill-bypass ideas, MLX long-context experimentation | the public `yza_fused` profile was too lossy on our Qwen3.5 long-context exact-match runs |
| `tonbistudio/turboquant-pytorch` | strong separation between attention-score validation and **real text generation** validation; useful ablations around QJL and long context | PyTorch / CUDA speed claims do not say much by themselves about MLX runtime behavior |
| `DeadByDawn101/turboquant-mlx` | a useful reminder that an implementation can have correct math on paper while still missing the live QJL wiring or compressed-domain attention kernel | README-level “implementation status” honesty is not a performance result |
| `scrya-com/rotorquant` and `kpalastro/mlx_rotorquant` | rotor / fewer-parameter rotation design space, fused-kernel motivation, and why microkernel speed is worth chasing | RotorQuant has **not** yet replaced the exact TurboQuant winners on this branch |

The practical rule we use in this repo is:

- copy upstream **runtime ideas** aggressively
- trust only **local exact-match generation** for final profile selection
- assume paper or README speedups are hypotheses until reproduced on the target model

Recent upstream notes that matter for this branch:

- the latest `yzamari/mlx-fork` work on `turboquant_attention` added broader
  4-bit support; this branch already carries the equivalent fused-runtime path
  in local `mlx`
- the latest `yzamari/mlx-turboquant` / `turboQuantPlayground` updates are
  mainly benchmark/docs updates, not proof that one public profile is exact on
  every model family
- the latest `tonbistudio/turboquant-pytorch` correction explicitly walked back
  earlier invalid generation claims, which is one reason this README only shows
  **real generation exact-match** tables
- the latest `DeadByDawn101/turboquant-mlx` status note is aligned with what we
  saw locally: a mathematically plausible design can still miss live QJL wiring
  or compressed-domain attention support

#### Practical Decision Rule

1. Benchmark native first.
2. Benchmark `4-bit mse`.
3. If needed, try FP16 edge layers.
4. Then try asymmetric `K/V` bits.
5. Only after that, benchmark `prod`.
6. If you use `prod`, test both **QJL on** and **QJL off**.

Two important caveats:

1. `--turbo-max-kv-size` is still a research knob. Naive fixed-size eviction
   reduced memory further, but clearly hurt exactness in the current runs.
2. Fractional `2.5` / `3.5`-bit modes exist in this branch, but they are not
   the current throughput winners.

### Supported Models

`mlx-lm` supports thousands of LLMs available on the Hugging Face Hub. If the
model you want to run is not supported, file an
[issue](https://github.com/ml-explore/mlx-lm/issues/new) or better yet, submit
a pull request. Many supported models are available in various quantization
formats in the [MLX Community](https://huggingface.co/mlx-community) Hugging
Face organization.

For some models the tokenizer may require you to enable the `trust_remote_code`
option. You can do this by passing `--trust-remote-code` in the command line.
If you don't specify the flag explicitly, you will be prompted to trust remote
code in the terminal when running the model. 

Tokenizer options can also be set in the Python API. For example:

```python
model, tokenizer = load(
    "qwen/Qwen-7B",
    tokenizer_config={"eos_token": "<|endoftext|>", "trust_remote_code": True},
)
```

### Large Models

> [!NOTE]
    This requires macOS 15.0 or higher to work.

Models which are large relative to the total RAM available on the machine can
be slow. `mlx-lm` will attempt to make them faster by wiring the memory
occupied by the model and cache. This requires macOS 15 or higher to
work.

If you see the following warning message:

> [WARNING] Generating with a model that requires ...

then the model will likely be slow on the given machine. If the model fits in
RAM then it can often be sped up by increasing the system wired memory limit.
To increase the limit, set the following `sysctl`:

```bash
sudo sysctl iogpu.wired_limit_mb=N
```

The value `N` should be larger than the size of the model in megabytes but
smaller than the memory size of the machine.
