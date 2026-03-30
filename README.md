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

This branch includes experimental TurboQuant KV-cache compression features in
`mlx_lm.generate` / `mlx_lm.benchmark`, including:

- separate key/value bit-widths
- optional 1-bit QJL residual correction for `prod` mode
- FP16 "edge layers" to protect the first/last layers
- an incremental decode buffer for faster compressed decode
- optional oldest-token eviction for fixed-size compressed caches

The main practical takeaway from the current Apple Silicon runs is simple:
there is **no single best compressed-KV profile across models**. The best
speed/accuracy trade-off is model-dependent, so the recommended way to use
TurboQuant in this branch is to start with a per-model profile and then tune
around it.

This also means that paper-level results should be treated as a direction, not
as a drop-in expectation for every model on MLX. In practice, **memory
reduction transfers much more reliably than speedups**. Speed can improve on
some model families, but the compressed path still pays a real preprocessing
cost for rotation, quantization, packing, optional QJL correction, and decode
buffer maintenance. If that overhead is not amortized well by the target
architecture and workload, the run will become smaller in memory but not
necessarily faster in tokens/sec.

Measured on this exact machine:

- Apple M3 Max
- 16 CPU cores
- 128 GB unified memory

Current local reference points on this machine (greedy decode, exact-match
check against native on the generated suffix):

| Model | Workload | Native gen tok/s | Recommended compressed profile | Compressed gen tok/s | Cache MB | Notes |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | `4096` prompt / `16` decode | `41.2` | `--turbo-kv-bits 4 --turbo-key-bits 3 --turbo-value-bits 4 --turbo-estimator-mode prod --turbo-disable-qjl --turbo-fp16-layers 4 --turbo-decode-buffer` | `40.1` | `54.4` | exact, `-2.7%` decode speed vs native, `-77.1%` cache; QJL was not consistently helpful on this 7B family |
| `mlx-community/Qwen2.5-32B-Instruct-4bit` | `4096` prompt / `16` decode | `9.9` | `--turbo-kv-bits 4 --turbo-estimator-mode mse --turbo-fp16-layers 2 --turbo-decode-buffer` | `8.8` | `273.0` | exact, `-11.2%` decode speed vs native, `-74.9%` cache; the simple 4-bit MSE profile was the most robust winner |
| `mlx-community/Qwen3.5-35B-A3B-4bit` | `8192` prompt / `16` decode | `18.0` | `--turbo-kv-bits 4 --turbo-estimator-mode prod --turbo-fp16-layers 2 --turbo-qjl-projection-mode wht --turbo-decode-buffer` | `18.4` | `86.8` | exact, `+1.8%` decode speed vs native, `-55.8%` cache; this hybrid model benefited from the QJL-backed `prod` path |

These numbers are the main reason this branch recommends a model-tuned
workflow instead of a single “paper” profile:

- on some families, compressed KV is almost free in throughput
- on some families, compressed KV is mainly a memory win
- and on some families, the best path changes depending on whether QJL helps

The underlying reason is that this MLX implementation still pays a real
preprocessing bill before the fast attention path can benefit:

- rotate / normalize keys and values
- quantize and pack indices
- optionally compute and apply the QJL residual correction
- maintain the incremental decode buffer

That preprocessing cost is exactly why this branch can be **much smaller in
cache memory without automatically matching native speed**. If the model
architecture amortizes the compressed path well, you can win on both. If it
does not, you still get the memory reduction, but the tokens/sec gain may stay
flat or even regress slightly.

Two extra notes are worth keeping in mind:

1. `--turbo-max-kv-size` is currently an aggressive research knob, not a safe
   default. Naive fixed-size eviction reduced cache size further but clearly
   hurt exactness in the current runs.
2. Fractional `2.5` / `3.5`-bit TurboQuant modes are implemented and exact in
   this branch, but they currently use a safe split-cache fallback that is not
   yet a speed winner. If you care primarily about throughput, start with the
   integer-bit profiles above.

#### What the Main Options Mean

If you just want the best practical profile, these are the important knobs:

- `--turbo-kv-bits N`
  Turn on TurboQuant compression at `N` bits. In the current MLX path, `4`
  bits is the safest starting point for real workloads.
- `--turbo-key-bits K --turbo-value-bits V`
  Override key and value precision separately. This is useful when keys are
  more sensitive than values and can improve the speed/accuracy compromise on
  some models.
- `--turbo-estimator-mode mse`
  The simplest and most robust option. Start here first.
- `--turbo-estimator-mode prod`
  A more aggressive path that can help on some models, especially when paired
  with QJL, but should always be benchmarked on the target model family.
- `--turbo-disable-qjl`
  Turns off the 1-bit QJL residual correction in `prod` mode. This can help or
  hurt depending on the model. It is not a universally good or bad toggle.
- `--turbo-fp16-layers N`
  Keep the first `N` and last `N` layers in their default FP16 cache form.
  This is one of the most useful stability knobs in practice.
- `--turbo-decode-buffer`
  Keeps a live dequantized decode buffer to reduce the decode penalty of
  compressed KV. This is usually the right default when throughput matters.
- `--turbo-max-kv-size`
  Enforces a hard compressed-cache limit by evicting the oldest tokens. This
  saves more memory, but should be treated as a separate experiment because it
  can easily hurt accuracy.

#### Practical Decision Rule

If your goal is a strong speed/accuracy compromise on Apple Silicon, the most
reliable workflow today is:

1. Benchmark native first.
2. Benchmark `4-bit mse`.
3. Try `--turbo-fp16-layers`.
4. Try asymmetric `K/V` bits.
5. Only then test `prod` and `QJL`.

In other words: **do not assume a paper profile transfers directly**. Benchmark
and tune on the exact model family you care about.

Example benchmark command:

```bash
mlx_lm.benchmark \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --prompt-tokens 8192 \
  --generation-tokens 16 \
  --turbo-kv-bits 4 \
  --turbo-estimator-mode prod \
  --turbo-fp16-layers 2 \
  --turbo-qjl-projection-mode wht \
  --turbo-decode-buffer
```

If you are unsure where to start:

- start with plain `4-bit mse`
- try asymmetric `K/V` bits next
- enable `prod` + QJL only if it helps on your target model
- treat fixed-size eviction as a separate experiment, not as the baseline
- benchmark on the exact model family you care about before generalizing

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
