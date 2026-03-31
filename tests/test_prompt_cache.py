# Copyright © 2024 Apple Inc.

import copy
import os
import tempfile
import unittest
from types import SimpleNamespace

import mlx.core as mx

from mlx_lm.generate import generate_step, maybe_quantize_kv_cache
from mlx_lm.models.base import (
    _apply_turbo_sparse_v,
    _compute_turbo_sparse_v_mask,
    create_attention_mask,
    create_causal_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    CacheList,
    ChunkedKVCache,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
    load_prompt_cache,
    make_prompt_cache,
    save_prompt_cache,
    trim_prompt_cache,
)
from mlx_lm.models.rotorquant import RotorQuantKVCache
from mlx_lm.models.turboquant import (
    TurboQuantKVCache,
    _cached_qjl_projection,
    _metal_available,
    _metal_qjl_score,
    _pack,
    _quantize_qjl_residual,
    _quantize_qjl_residual_packed,
    _unpack,
)
from mlx_lm.utils import load

HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"


class DummyCacheModel:
    def __init__(self, caches):
        self.layers = [object()] * len(caches)
        self._caches = caches

    def make_cache(self):
        return copy.deepcopy(self._caches)


class TestPromptCache(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir_fid = tempfile.TemporaryDirectory()
        cls.test_dir = cls.test_dir_fid.name
        cls.model, cls.tokenizer = load(HF_MODEL_PATH)

    @classmethod
    def tearDownClass(cls):
        cls.test_dir_fid.cleanup()

    def test_save_load(self):
        cache = [KVCache() for _ in range(4)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 4))
            c.update_and_fetch(x, x)
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")
        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        self.assertTrue(len(cache), len(loaded_cache))
        for c, lc in zip(cache, loaded_cache):
            self.assertEqual(c.offset, lc.offset)
            self.assertTrue(mx.array_equal(c.state[0], lc.state[0]))
            self.assertTrue(mx.array_equal(c.state[1], lc.state[1]))

        # Test with metadata
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")
        metadata = {"a": "b", "c": "d"}
        save_prompt_cache(cache_file, cache, metadata)
        _, loaded_metadata = load_prompt_cache(cache_file, return_metadata=True)
        self.assertEqual(metadata, loaded_metadata)

    def test_save_load_rotating_cache(self):
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")

        # Test with rotating cache
        cache = [RotatingKVCache(max_size=8, keep=2) for _ in range(4)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 4))
            c.update_and_fetch(x, x)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        self.assertTrue(len(cache), len(loaded_cache))
        for c, lc in zip(cache, loaded_cache):
            self.assertEqual(c.offset, lc.offset)
            self.assertEqual(c.keep, lc.keep)
            self.assertEqual(c.max_size, lc.max_size)
            self.assertEqual(c.step, lc.step)
            self.assertTrue(mx.array_equal(c.state[0], lc.state[0]))
            self.assertTrue(mx.array_equal(c.state[1], lc.state[1]))

        # Do a couple single token updates to get a rotation
        for _ in range(2):
            for c in cache:
                x = mx.random.uniform(shape=(1, 8, 1, 4))
                c.update_and_fetch(x, x)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)

        for c, lc in zip(cache, loaded_cache):
            x = mx.random.uniform(shape=(1, 8, 1, 4))
            k, v = c.update_and_fetch(x, x)
            lk, lv = lc.update_and_fetch(x, x)
            self.assertEqual(c.offset, lc.offset)
            self.assertTrue(mx.array_equal(k, lk))
            self.assertTrue(mx.array_equal(v, lv))

    def test_save_load_mixed_cache(self):
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")

        cache = [
            ArraysCache(size=2),
            KVCache(),
            RotatingKVCache(8),
            ArraysCache(size=2),
            ChunkedKVCache(256),
        ]
        for c in cache:
            if isinstance(c, ArraysCache):
                c[0] = mx.random.uniform(shape=(4, 4, 4))
                c[1] = mx.random.uniform(shape=(4, 4, 4))
            else:
                x = mx.random.uniform(shape=(4, 4, 7, 4))
                y = mx.random.uniform(shape=(4, 4, 7, 4))
                c.update_and_fetch(x, y)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        for c, lc in zip(cache, loaded_cache):
            if isinstance(c, ArraysCache):
                self.assertTrue(mx.array_equal(c[0], lc[0]))
                self.assertTrue(mx.array_equal(c[1], lc[1]))
            else:
                x = mx.random.uniform(shape=(4, 4, 1, 4))
                y = mx.random.uniform(shape=(4, 4, 1, 4))
                k, v = c.update_and_fetch(x, y)
                lk, lv = lc.update_and_fetch(x, y)
                self.assertEqual(c.offset, lc.offset)
                self.assertTrue(mx.array_equal(k, lk))
                self.assertTrue(mx.array_equal(v, lv))

    def test_save_load_cache_list(self):
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")

        cache = [
            ArraysCache(size=2),
            KVCache(),
            RotatingKVCache(8),
            ArraysCache(size=2),
            ChunkedKVCache(256),
        ]
        for c in cache:
            if isinstance(c, ArraysCache):
                c[0] = mx.random.uniform(shape=(4, 4, 4))
                c[1] = mx.random.uniform(shape=(4, 4, 4))
            else:
                x = mx.random.uniform(shape=(4, 4, 7, 4))
                y = mx.random.uniform(shape=(4, 4, 7, 4))
                c.update_and_fetch(x, y)
        cache = [CacheList(*cache)]

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        for c, lc in zip(cache[0].caches, loaded_cache[0].caches):
            if isinstance(c, ArraysCache):
                self.assertTrue(mx.array_equal(c[0], lc[0]))
                self.assertTrue(mx.array_equal(c[1], lc[1]))
            else:
                x = mx.random.uniform(shape=(4, 4, 1, 4))
                y = mx.random.uniform(shape=(4, 4, 1, 4))
                k, v = c.update_and_fetch(x, y)
                lk, lv = lc.update_and_fetch(x, y)
                self.assertEqual(c.offset, lc.offset)
                self.assertTrue(mx.array_equal(k, lk))
                self.assertTrue(mx.array_equal(v, lv))

    def test_save_load_arrays_cache(self):
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")

        cache = [ArraysCache(size=2)]
        cache[0][0] = mx.zeros((1, 4, 4))
        cache[0][1] = mx.zeros((1, 4, 4))

        save_prompt_cache(cache_file, cache)
        loaded = load_prompt_cache(cache_file)

        # Try to make a mask
        mask = loaded[0].make_mask(4)

    def test_cache_with_generate(self):
        model, tokenizer = self.model, self.tokenizer
        prompt = tokenizer.encode("this is a prompt", return_tensors="mlx")[0]
        results = list(generate_step(prompt, model, max_tokens=4))
        toks, all_logits = zip(*results)

        prompt_cache = make_prompt_cache(model)
        i = 0
        for tok, logits in generate_step(
            prompt, model, prompt_cache=prompt_cache, max_tokens=2
        ):
            self.assertEqual(tok, toks[i])
            self.assertTrue(mx.allclose(logits, all_logits[i]))
            i += 1

        for tok, logits in generate_step(
            mx.array([toks[i]]), model, prompt_cache=prompt_cache, max_tokens=1
        ):
            i += 1
            self.assertEqual(tok, toks[i])
            self.assertTrue(mx.allclose(logits, all_logits[i]))

    def test_trim_cache(self):
        cache = [KVCache() for _ in range(2)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 4))
            c.update_and_fetch(x, x)

        # Trim
        num_trimmed = trim_prompt_cache(cache, 7)
        self.assertEqual(num_trimmed, 7)

        # Trim more tokens than remain
        num_trimmed = trim_prompt_cache(cache, 4)
        self.assertEqual(num_trimmed, 3)

        # Can't trim arrays cache
        cache = [ArraysCache(size=2) for _ in range(2)]
        for c in cache:
            c[0] = mx.zeros((5, 5))
            c[1] = mx.zeros((5, 5))
        num_trimmed = trim_prompt_cache(cache, 7)
        self.assertEqual(num_trimmed, 0)

        # All cache's have to be trimmable
        cache = [ArraysCache(size=2), KVCache()]
        cache[0][0] = mx.zeros((5, 5))
        cache[0][1] = mx.zeros((5, 5))
        x = mx.random.uniform(shape=(1, 8, 10, 4))
        cache[1].update_and_fetch(x, x)
        num_trimmed = trim_prompt_cache(cache, 1)
        self.assertEqual(num_trimmed, 0)

        cache = [RotatingKVCache(max_size=6) for _ in range(2)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 5, 4))
            c.update_and_fetch(x, x)

        num_trimmed = trim_prompt_cache(cache, 4)
        self.assertEqual(num_trimmed, 4)

        # Can't trim fixed-size KV cache after processing
        # more than max_kv_size tokens
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 4))
            c.update_and_fetch(x, x)

        num_trimmed = trim_prompt_cache(cache, 4)
        self.assertEqual(num_trimmed, 0)

        cache = [QuantizedKVCache() for _ in range(2)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 64))
            c.update_and_fetch(x, x)

        num_trimmed = trim_prompt_cache(cache, 7)
        self.assertEqual(num_trimmed, 7)

        # Trim more tokens than remain
        num_trimmed = trim_prompt_cache(cache, 4)
        self.assertEqual(num_trimmed, 3)

    def test_trim_cache_with_generate(self):
        model, tokenizer = self.model, self.tokenizer
        prompt = tokenizer.encode("this is a prompt", return_tensors="mlx")[0]

        prompt_cache = make_prompt_cache(model)

        # Generate one token so we process the full prompt
        last_tok, _ = next(generate_step(prompt, model, prompt_cache=prompt_cache))
        last_tok = mx.array([last_tok])

        # Generate two more tokens
        results = zip(
            range(2), generate_step(last_tok, model, prompt_cache=prompt_cache)
        )
        toks, all_logits = zip(*(r[1] for r in results))

        # To get back to the cache just after processing the prompt,
        # trim by 3 tokens
        trim_prompt_cache(prompt_cache, 3)

        # Generate the same thing again
        results = zip(
            range(2), generate_step(last_tok, model, prompt_cache=prompt_cache)
        )
        second_toks, second_all_logits = zip(*(r[1] for r in results))
        self.assertEqual(toks, second_toks)
        self.assertTrue(
            all(mx.allclose(l, l2) for l, l2 in zip(all_logits, second_all_logits))
        )

    def test_cache_copying(self):
        cache = [KVCache()]

        x = mx.random.uniform(shape=(1, 8, 10, 4))
        cache[0].update_and_fetch(x, x)

        y = mx.random.uniform(shape=(1, 8, 1, 4))
        cache[0].update_and_fetch(y, y)

        old_cache = copy.deepcopy(cache)

        trim_prompt_cache(cache, 1)

        self.assertTrue(old_cache[0].offset, 11)
        self.assertTrue(cache[0].offset, 10)

        z = mx.random.uniform(shape=(1, 8, 1, 4))
        cache[0].update_and_fetch(z, z)

        self.assertTrue(mx.allclose(old_cache[0].keys[..., 10:11, :], y))
        self.assertTrue(mx.allclose(cache[0].keys[..., 10:11, :], z))

    def test_save_load_quantized_cache(self):
        cache = [QuantizedKVCache(bits=4, group_size=32) for _ in range(4)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 32))
            c.update_and_fetch(x, x)
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")
        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        self.assertTrue(loaded_cache[0].bits == cache[0].bits)
        self.assertTrue(loaded_cache[0].group_size == cache[0].group_size)
        self.assertTrue(len(cache), len(loaded_cache))
        for c, lc in zip(cache, loaded_cache):
            self.assertEqual(c.offset, lc.offset)
            # Loop over quantized tuple
            for i in range(3):
                self.assertTrue(mx.array_equal(c.state[0][i], lc.state[0][i]))
                self.assertTrue(mx.array_equal(c.state[1][i], lc.state[1][i]))

        # Test with metadata
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")
        metadata = {"a": "b", "c": "d"}
        save_prompt_cache(cache_file, cache, metadata)
        _, loaded_metadata = load_prompt_cache(cache_file, return_metadata=True)
        self.assertEqual(metadata, loaded_metadata)

    def test_cache_to_quantized(self):
        model, tokenizer = self.model, self.tokenizer
        prompt = tokenizer.encode("this is a prompt", return_tensors="mlx")[0]
        results = zip(range(4), generate_step(prompt, model))
        toks, all_logits = zip(*(r[1] for r in results))

        prompt_cache = make_prompt_cache(model)
        i = 0
        for _, (tok, logits) in zip(
            range(2), generate_step(prompt, model, prompt_cache=prompt_cache)
        ):
            self.assertEqual(tok, toks[i])
            self.assertTrue(mx.allclose(logits, all_logits[i]))
            i += 1

        prompt_cache = [c.to_quantized(bits=8, group_size=32) for c in prompt_cache]

        for _, (tok, logits) in zip(
            range(1),
            generate_step(mx.array([toks[i]]), model, prompt_cache=prompt_cache),
        ):
            i += 1
            self.assertEqual(tok, toks[i])
            self.assertTrue(mx.allclose(logits, all_logits[i], rtol=4e-2))

    def test_cache_list(self):
        c = CacheList(KVCache(), KVCache())
        self.assertTrue(c.is_trimmable())
        k = mx.zeros((1, 2, 8, 8))
        v = mx.zeros((1, 2, 8, 8))
        c[0].update_and_fetch(k, v)
        c[1].update_and_fetch(k, v)
        m = c.trim(5)
        self.assertEqual(m, 5)

        c = CacheList(ArraysCache(size=2), KVCache())
        self.assertFalse(c.is_trimmable())

        c1 = CacheList(ArraysCache(size=1), KVCache())
        c1[0][0] = mx.random.normal(shape=(1, 2, 4, 4))
        c1[1].update_and_fetch(
            mx.random.normal(shape=(1, 2, 5, 4)), mx.random.normal(shape=(1, 2, 5, 4))
        )

        c2 = CacheList(ArraysCache(size=1), KVCache())
        c2[0][0] = mx.random.normal(shape=(1, 2, 4, 4))
        c2[1].update_and_fetch(
            mx.random.normal(shape=(1, 2, 7, 4)), mx.random.normal(shape=(1, 2, 7, 4))
        )

        merged_cache = CacheList.merge((c1, c2))
        c1_ex = merged_cache.extract(0)
        self.assertTrue(mx.array_equal(c1_ex[0][0], c1[0][0]))
        self.assertTrue(mx.array_equal(c1_ex[1].state[0], c1[1].state[0]))
        c2_ex = merged_cache.extract(1)
        self.assertTrue(mx.array_equal(c2_ex[0][0], c2[0][0]))
        self.assertTrue(mx.array_equal(c2_ex[1].state[0], c2[1].state[0]))

    def test_make_mask_with_cache(self):
        # For 1 time step with no cache, don't need a mask
        mask = create_attention_mask(mx.zeros((1, 1)), cache=None, return_array=False)
        self.assertEqual(mask, None)

        mask = create_attention_mask(mx.zeros((1, 1)), cache=None, return_array=True)
        self.assertEqual(mask, None)

        # Regular causal mask
        mask = create_attention_mask(mx.zeros((1, 4)), cache=None, return_array=False)
        self.assertEqual(mask, "causal")

        mask = create_attention_mask(mx.zeros((1, 4)), cache=None, return_array=True)
        self.assertTrue(mx.array_equal(mask, create_causal_mask(4)))

        # With a window size
        mask = create_attention_mask(
            mx.zeros((1, 4)), cache=None, window_size=4, return_array=False
        )
        self.assertEqual(mask, "causal")

        mask = create_attention_mask(
            mx.zeros((1, 4)), cache=None, window_size=3, return_array=False
        )
        self.assertTrue(mx.array_equal(mask, create_causal_mask(4, window_size=3)))

        # With a regular KV cache
        cache = KVCache()
        mask = create_attention_mask(mx.zeros((1, 4)), cache=cache, return_array=False)
        self.assertEqual(mask, "causal")

        mask = create_attention_mask(mx.zeros((1, 4)), cache=cache, return_array=True)
        self.assertTrue(mx.array_equal(mask, create_causal_mask(4)))

        k = v = mx.zeros((1, 2, 16, 8))
        cache.update_and_fetch(k, v)
        mask = create_attention_mask(mx.zeros((1, 4)), cache=cache, return_array=True)
        self.assertEqual(mask.shape, (4, 20))

    def test_rotating_cache_mask(self):
        cache = RotatingKVCache(max_size=8)

        mask = cache.make_mask(4, window_size=5)
        self.assertEqual(mask, "causal")
        mask = create_attention_mask(mx.zeros((1, 4, 32)), cache, window_size=5)
        self.assertEqual(mask, "causal")
        mask = create_attention_mask(
            mx.zeros((1, 4, 32)), cache, window_size=5, return_array=True
        )
        self.assertEqual(mask.dtype, mx.bool_)
        self.assertEqual(mask.shape, (4, 4))

        mask = cache.make_mask(6, window_size=5)
        self.assertEqual(mask.dtype, mx.bool_)
        self.assertEqual(mask.sum(axis=-1).max(), 5)
        cmask = create_attention_mask(mx.zeros((1, 6, 32)), cache, window_size=5)
        self.assertTrue(mx.array_equal(cmask, mask))

        mask = cache.make_mask(1, window_size=5)
        self.assertEqual(mask, None)
        mask = create_attention_mask(mx.zeros((1, 1, 32)), cache, window_size=5)
        self.assertEqual(mask, None)

        kv = mx.zeros((1, 1, 10, 32))
        cache.update_and_fetch(kv, kv)
        mask = cache.make_mask(3, window_size=5)
        self.assertEqual(mask.shape, (3, 10))
        self.assertTrue(mx.all(mask.sum(axis=-1) == 5))
        for i in range(3):
            s = 11 - 3 + i
            self.assertTrue(mx.all(mask[s - 5 : s]))
        cmask = create_attention_mask(mx.zeros((1, 3, 32)), cache, window_size=5)
        self.assertTrue(mx.array_equal(cmask, mask))

        mask = cache.make_mask(1)
        self.assertEqual(mask, None)
        mask = create_attention_mask(mx.zeros((1, 1, 32)), cache)
        self.assertEqual(mask, None)

        mask = cache.make_mask(1, window_size=5)
        self.assertEqual(mask.tolist(), [True] + [False] * 3 + [True] * 4)
        cmask = create_attention_mask(mx.zeros((1, 1, 32)), cache, window_size=5)
        self.assertTrue(mx.array_equal(cmask, mask))

        kv = mx.zeros((1, 1, 1, 32))
        cache.update_and_fetch(kv, kv)

        mask = cache.make_mask(1, window_size=5)
        self.assertEqual(mask.tolist(), [True] * 2 + [False] * 3 + [True] * 3)
        cmask = create_attention_mask(mx.zeros((1, 1, 32)), cache, window_size=5)
        self.assertTrue(mx.array_equal(cmask, mask))

    def test_batch_kv_cache(self):
        cache = BatchKVCache(left_padding=[2, 3, 4])
        k, v = mx.zeros((3, 1, 4, 8)), mx.zeros((3, 1, 4, 8))
        # Update works
        k, v = cache.update_and_fetch(k, v)
        self.assertEqual(k.shape, (3, 1, 4, 8))

        # State can be evaluated
        mx.eval(cache.state)

        # State can be set
        cache.state = cache.state

        # Test filtering
        cache.filter([0, 1])

        # In this case filtering left shifts the cache so it has zero padding
        self.assertEqual(cache.state[0].shape, (2, 1, 2, 8))

        mask = cache.make_mask(1)
        self.assertEqual(mask[0].squeeze().tolist(), [True, True, True])
        self.assertEqual(mask[1].squeeze().tolist(), [False, True, True])

        # Test extension
        cache_a = BatchKVCache(left_padding=[2, 1, 2])
        cache_b = BatchKVCache(left_padding=[3, 0])

        k = mx.zeros((3, 1, 8, 1))
        v = mx.zeros((3, 1, 8, 1))
        cache_a.update_and_fetch(k, v)

        k = mx.zeros((2, 1, 4, 1))
        v = mx.zeros((2, 1, 4, 1))
        cache_b.update_and_fetch(k, v)

        cache_a.extend(cache_b)
        self.assertEqual(cache_a.keys.shape[0], 5)
        self.assertEqual(cache_a.values.shape[0], 5)
        self.assertEqual(cache_a.offset.tolist(), [6, 7, 6, 1, 4])
        self.assertEqual(cache_a.left_padding.tolist(), [2, 1, 2, 7, 4])

    def test_batch_rotating_kv_cache(self):
        cache = BatchRotatingKVCache(max_size=4, left_padding=[2, 0])
        mask = cache.make_mask(4)
        self.assertFalse(mx.any(mask[0, 0, 0, :]))
        self.assertTrue(
            mx.array_equal(mask[1, 0, 0, :], mx.array([True, False, False, False]))
        )

        # Batch update works
        k, v = mx.zeros((2, 1, 4, 8)), mx.zeros((2, 1, 4, 8))
        k, v = cache.update_and_fetch(k, v)

        mask = cache.make_mask(4)
        k, v = mx.zeros((2, 1, 4, 8)), mx.zeros((2, 1, 4, 8))
        k, v = cache.update_and_fetch(k, v)
        self.assertEqual(mask.shape[-2:], (4, k.shape[2]))
        self.assertEqual(
            mask[0, 0, 0, :].tolist(), [False, True, True, True, False, False, False]
        )

        # Single query update works
        cache = BatchRotatingKVCache(max_size=4, left_padding=[2, 0])
        k, v = mx.zeros((2, 1, 4, 8)), mx.zeros((2, 1, 4, 8))
        k, v = cache.update_and_fetch(k, v)

        mask = cache.make_mask(1)
        k, v = mx.zeros((2, 1, 1, 8)), mx.zeros((2, 1, 1, 8))

        k, v = cache.update_and_fetch(k, v)
        self.assertEqual(mask.shape[-2:], (1, k.shape[2]))
        self.assertEqual(mask[0, 0, 0].tolist(), [True, False, True, True])
        self.assertEqual(mask[1, 0, 0].tolist(), [True, True, True, True])

        # Check filtering
        cache = BatchRotatingKVCache(max_size=4, left_padding=[2, 0, 3])
        k, v = mx.zeros((3, 1, 3, 8)), mx.zeros((3, 1, 3, 8))
        cache.update_and_fetch(k, v)
        cache.filter(mx.array([1]))
        self.assertEqual(cache.keys.shape, (1, 1, 3, 8))

        # Check extend
        cache = BatchRotatingKVCache(max_size=4, left_padding=[2, 1])
        other = BatchRotatingKVCache(max_size=4, left_padding=[2, 2])
        k, v = mx.zeros((2, 1, 5, 8)), mx.zeros((2, 1, 5, 8))
        cache.update_and_fetch(k, v)
        other.update_and_fetch(k, v)
        k, v = mx.zeros((2, 1, 1, 8)), mx.zeros((2, 1, 1, 8))
        cache.update_and_fetch(k, v)
        cache.extend(other)

        # Check mask when going from prompt -> extend -> prompt
        cache = BatchRotatingKVCache(max_size=8, left_padding=[4])
        k, v = mx.zeros((1, 1, 8, 8)), mx.zeros((1, 1, 8, 8))
        cache.update_and_fetch(k, v)

        mask = cache.make_mask(1)
        self.assertEqual(
            mask.squeeze().tolist(), [True, False, False, False, True, True, True, True]
        )

        k, v = mx.zeros((1, 1, 1, 8)), mx.zeros((1, 1, 1, 8))
        cache.update_and_fetch(k, v)

        mask = cache.make_mask(2)
        expected = mx.array(
            [
                [False, False, False, True, True, True, True, True, False],
                [False, False, False, True, True, True, True, True, True],
            ]
        )
        self.assertTrue(mx.array_equal(mask.squeeze(), expected))

    def test_save_load_batch_caches(self):
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")

        cache = [
            ArraysCache(size=2, left_padding=[1, 2]),
            BatchKVCache(left_padding=[1, 2]),
            BatchRotatingKVCache(max_size=10, left_padding=[1, 2]),
        ]
        for c in cache:
            if isinstance(c, ArraysCache):
                c[0] = mx.random.uniform(shape=(4, 4, 4))
                c[1] = mx.random.uniform(shape=(4, 4, 4))
            else:
                x = mx.random.uniform(shape=(4, 4, 7, 4))
                y = mx.random.uniform(shape=(4, 4, 7, 4))
                c.update_and_fetch(x, y)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        left_padding = mx.array([1, 2])
        for c, lc in zip(cache, loaded_cache):
            self.assertTrue(mx.array_equal(c.left_padding, left_padding))

    def test_rotating_cache_updates(self):
        cache = RotatingKVCache(max_size=8)
        k = v = mx.zeros((1, 1, 10, 1))
        cache.update_and_fetch(k, v)

        for _ in range(3):
            k = v = mx.zeros((1, 1, 1, 1))
            cache.update_and_fetch(k, v)

        k = v = mx.zeros((1, 1, 3, 1))
        k, v = cache.update_and_fetch(k, v)
        self.assertEqual(k.shape[2], 10)
        self.assertEqual(v.shape[2], 10)

    def test_merge_with_empty_caches(self):
        c1 = ArraysCache(2)
        c2 = ArraysCache(2)
        c2[0] = mx.zeros((1, 4))
        c2[1] = mx.zeros((1, 4))
        c_out = ArraysCache.merge((c1, c2))
        self.assertEqual(c_out[0].shape, (2, 4))
        self.assertEqual(c_out[1].shape, (2, 4))

        c1 = KVCache()
        c2 = KVCache()
        kv = mx.zeros((1, 4, 4, 4))
        c2.update_and_fetch(kv, kv)
        c_out = KVCache.merge((c1, c2))
        self.assertEqual(c_out.keys.shape, (2, 4, 4, 4))

        c1 = RotatingKVCache(max_size=4)
        c2 = RotatingKVCache(max_size=4)
        kv = mx.zeros((1, 4, 4, 4))
        c2.update_and_fetch(kv, kv)
        c_out = KVCache.merge((c1, c2))
        self.assertEqual(c_out.keys.shape, (2, 4, 4, 4))

    def test_window_mask_with_full_kv_cache(self):
        c = KVCache()
        kv = mx.zeros((1, 1, 32, 128))
        c.update_and_fetch(kv, kv)

        h = mx.zeros((1, 1, 1, 128))
        mask = create_attention_mask(h, c, window_size=4)
        expected = create_causal_mask(1, offset=32, window_size=4)
        self.assertTrue(mx.array_equal(mask, expected))

    def test_turboquant_cache_basic(self):
        """Test TurboQuantKVCache update_and_fetch, offset, shapes."""
        for bits in [2, 3, 4]:
            c = TurboQuantKVCache(bits=bits)
            self.assertTrue(c.empty())
            self.assertEqual(c.size(), 0)

            # Prefill
            k = mx.random.uniform(shape=(1, 8, 10, 64))
            v = mx.random.uniform(shape=(1, 8, 10, 64))
            dk, dv = c.update_and_fetch(k, v)
            mx.eval(dk, dv)

            self.assertEqual(c.size(), 10)
            self.assertFalse(c.empty())
            self.assertEqual(dk.shape, (1, 8, 10, 64))
            self.assertEqual(dv.shape, (1, 8, 10, 64))

            # Decode step
            k2 = mx.random.uniform(shape=(1, 8, 1, 64))
            v2 = mx.random.uniform(shape=(1, 8, 1, 64))
            dk2, dv2 = c.update_and_fetch(k2, v2)
            mx.eval(dk2, dv2)

            self.assertEqual(c.size(), 11)
            self.assertEqual(dk2.shape, (1, 8, 11, 64))

    def test_turboquant_cache_quality(self):
        """Test that TurboQuant dequantized output is close to input."""
        c = TurboQuantKVCache(bits=4)
        k = mx.random.normal(shape=(1, 8, 32, 128))
        v = mx.random.normal(shape=(1, 8, 32, 128))
        dk, dv = c.update_and_fetch(k, v)
        mx.eval(dk, dv, k, v)

        # Cosine similarity per vector should be high at 4-bit
        cos_k = mx.mean(
            mx.sum(k * dk, axis=-1)
            / (mx.linalg.norm(k, axis=-1) * mx.linalg.norm(dk, axis=-1) + 1e-8)
        )
        mx.eval(cos_k)
        self.assertGreater(float(cos_k), 0.95)

    def test_turboquant_fractional_cache_basic(self):
        """Test fractional-bit TurboQuant wiring and shapes."""
        c = TurboQuantKVCache(bits=3.5)
        self.assertTrue(c.empty())
        self.assertEqual(c.size(), 0)

        k = mx.random.uniform(shape=(1, 8, 10, 64))
        v = mx.random.uniform(shape=(1, 8, 10, 64))
        dk, dv = c.update_and_fetch(k, v)
        mx.eval(dk, dv)

        self.assertEqual(c.size(), 10)
        self.assertFalse(c.empty())
        self.assertEqual(dk.shape, (1, 8, 10, 64))
        self.assertEqual(dv.shape, (1, 8, 10, 64))
        self.assertTrue(c._fractional_split)
        self.assertEqual(c._split_low_bits, 3)
        self.assertEqual(c._split_high_bits, 4)
        self.assertIsNotNone(c._split_low_cache)
        self.assertIsNotNone(c._split_high_cache)

    def test_turboquant_fractional_quality_improves_over_lower_integer(self):
        """Test 3.5-bit reconstruction is better than 3-bit reconstruction."""
        mx.random.seed(0)
        k = mx.random.normal(shape=(1, 4, 32, 64))
        v = mx.random.normal(shape=(1, 4, 32, 64))

        c3 = TurboQuantKVCache(bits=3)
        c35 = TurboQuantKVCache(bits=3.5)
        dk3, dv3 = c3.update_and_fetch(k, v)
        dk35, dv35 = c35.update_and_fetch(k, v)
        err3 = mx.mean((k - dk3) ** 2) + mx.mean((v - dv3) ** 2)
        err35 = mx.mean((k - dk35) ** 2) + mx.mean((v - dv35) ** 2)
        mx.eval(err3, err35)

        self.assertLess(float(err35), float(err3))

    def test_turboquant_cache_trim(self):
        """Test TurboQuantKVCache trim."""
        c = TurboQuantKVCache(bits=3)
        k = mx.random.uniform(shape=(1, 4, 8, 64))
        c.update_and_fetch(k, k)
        self.assertEqual(c.size(), 8)
        self.assertTrue(c.is_trimmable())

        trimmed = c.trim(3)
        self.assertEqual(trimmed, 3)
        self.assertEqual(c.size(), 5)

    def test_turboquant_save_load(self):
        """Test TurboQuantKVCache save/load roundtrip."""
        cache_file = os.path.join(self.test_dir, "turbo_cache.safetensors")

        cache = [TurboQuantKVCache(bits=3) for _ in range(4)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 64))
            c.update_and_fetch(x, x)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        self.assertEqual(len(cache), len(loaded_cache))

        for c, lc in zip(cache, loaded_cache):
            self.assertEqual(c.offset, lc.offset)
            self.assertEqual(c.turbo_bits, lc.turbo_bits)
            self.assertEqual(getattr(lc, "rotation_mode", "dense"), "dense")

    def test_turboquant_to_turboquant(self):
        """Test KVCache.to_turboquant conversion."""
        kv_cache = KVCache()
        k = mx.random.normal(shape=(1, 8, 16, 128))
        v = mx.random.normal(shape=(1, 8, 16, 128))
        kv_cache.update_and_fetch(k, v)

        tq_cache = kv_cache.to_turboquant(bits=4)
        self.assertEqual(tq_cache.size(), 16)
        self.assertFalse(tq_cache.empty())

    def test_turboquant_to_turboquant_prod_wht(self):
        """Test KVCache.to_turboquant forwards the QJL projection mode."""
        kv_cache = KVCache()
        k = mx.random.normal(shape=(1, 8, 16, 128))
        v = mx.random.normal(shape=(1, 8, 16, 128))
        kv_cache.update_and_fetch(k, v)

        tq_cache = kv_cache.to_turboquant(
            bits=3,
            estimator_mode="prod",
            qjl_projection_mode="wht",
        )
        self.assertEqual(tq_cache.size(), 16)
        self.assertEqual(tq_cache.estimator_mode, "prod")
        self.assertEqual(tq_cache.qjl_projection_mode, "wht")
        self.assertEqual(tq_cache._qjl_projection_runtime_mode, "wht")

    def test_turboquant_to_turbo_quantized_alias(self):
        """Test upstream-style alias forwards the richer local TurboQuant args."""
        kv_cache = KVCache()
        k = mx.random.normal(shape=(1, 8, 16, 128))
        v = mx.random.normal(shape=(1, 8, 16, 128))
        kv_cache.update_and_fetch(k, v)

        tq_cache = kv_cache.to_turbo_quantized(
            bits=3,
            estimator_mode="prod",
            qjl_projection_mode="wht",
        )
        self.assertEqual(tq_cache.size(), 16)
        self.assertEqual(tq_cache.estimator_mode, "prod")
        self.assertEqual(tq_cache.qjl_projection_mode, "wht")

    def test_turboquant_to_turboquant_decode_buffer(self):
        """Test KVCache.to_turboquant forwards the decode-buffer mode."""
        kv_cache = KVCache()
        k = mx.random.normal(shape=(1, 8, 16, 128))
        v = mx.random.normal(shape=(1, 8, 16, 128))
        kv_cache.update_and_fetch(k, v)

        tq_cache = kv_cache.to_turboquant(bits=3, decode_buffer=True)
        self.assertEqual(tq_cache.size(), 16)
        self.assertTrue(tq_cache.decode_buffer)

    def test_turboquant_to_turboquant_recent_buffer(self):
        """Test KVCache.to_turboquant forwards recent-buffer settings."""
        kv_cache = KVCache()
        k = mx.random.normal(shape=(1, 8, 16, 128))
        v = mx.random.normal(shape=(1, 8, 16, 128))
        kv_cache.update_and_fetch(k, v)

        tq_cache = kv_cache.to_turboquant(
            bits=4,
            buffer_size=16,
            flush_batch_size=8,
        )
        self.assertEqual(tq_cache.buffer_size, 16)
        self.assertEqual(tq_cache.flush_batch_size, 8)

    def test_turboquant_to_turboquant_asymmetric_bits(self):
        """Test KVCache.to_turboquant forwards separate K/V bit-widths."""
        kv_cache = KVCache()
        k = mx.random.normal(shape=(1, 8, 16, 128))
        v = mx.random.normal(shape=(1, 8, 16, 128))
        kv_cache.update_and_fetch(k, v)

        tq_cache = kv_cache.to_turboquant(bits=4, key_bits=4, value_bits=3)
        self.assertEqual(tq_cache.key_bits_override, 4)
        self.assertEqual(tq_cache.value_bits_override, 3)
        self.assertEqual(tq_cache._k_bits, 4)
        self.assertEqual(tq_cache._v_bits, 3)

    def test_turboquant_to_turboquant_max_kv_size(self):
        """Test KVCache.to_turboquant forwards a max compressed KV window."""
        kv_cache = KVCache()
        k = mx.random.normal(shape=(1, 8, 16, 128))
        v = mx.random.normal(shape=(1, 8, 16, 128))
        kv_cache.update_and_fetch(k, v)

        tq_cache = kv_cache.to_turboquant(bits=4, max_cache_tokens=8)
        self.assertEqual(tq_cache.max_cache_tokens, 8)
        self.assertEqual(tq_cache.offset, 8)

    def test_rotorquant_to_rotorquant(self):
        """Test KVCache.to_rotorquant conversion."""
        kv_cache = KVCache()
        k = mx.random.normal(shape=(1, 8, 16, 128))
        v = mx.random.normal(shape=(1, 8, 16, 128))
        kv_cache.update_and_fetch(k, v)

        rq_cache = kv_cache.to_rotorquant(bits=4)
        self.assertEqual(rq_cache.size(), 16)
        self.assertFalse(rq_cache.empty())
        self.assertIsInstance(rq_cache, RotorQuantKVCache)
        self.assertEqual(rq_cache.rotation_mode, "rotorquant")

    def test_rotorquant_to_rotorquant_decode_buffer(self):
        """Test KVCache.to_rotorquant forwards the decode-buffer mode."""
        kv_cache = KVCache()
        k = mx.random.normal(shape=(1, 8, 16, 128))
        v = mx.random.normal(shape=(1, 8, 16, 128))
        kv_cache.update_and_fetch(k, v)

        rq_cache = kv_cache.to_rotorquant(bits=4, decode_buffer=True)
        self.assertEqual(rq_cache.size(), 16)
        self.assertTrue(rq_cache.decode_buffer)
        self.assertEqual(rq_cache.rotation_mode, "rotorquant")

    def test_rotorquant_prod_mode_stores_qjl_residual(self):
        """Test RotorQuant prod mode uses bits-1 on keys plus 1-bit QJL residual."""
        cache = RotorQuantKVCache(bits=4, estimator_mode="prod")
        k = mx.random.normal(shape=(1, 4, 12, 64))
        v = mx.random.normal(shape=(1, 4, 12, 64))

        dk, dv = cache.update_and_fetch(k, v)
        mx.eval(dv)

        self.assertIsNone(dk)
        self.assertEqual(cache.estimator_mode, "prod")
        self.assertEqual(cache._k_bits, 3)
        self.assertEqual(cache._v_bits, 4)
        self.assertIsNotNone(cache._k_qjl_indices)
        self.assertIsNotNone(cache._k_qjl_gamma)
        self.assertEqual(dv.shape, v.shape)

    def test_rotorquant_prod_fused_scores_match_dequantized_reference(self):
        """Test RotorQuant prod score path matches explicit dequantized attention."""
        cache = RotorQuantKVCache(bits=4, estimator_mode="prod")
        cache.min_fused_tokens = 0

        k = mx.random.normal(shape=(1, 1, 9, 64))
        v = mx.random.normal(shape=(1, 1, 9, 64))
        _, dv = cache.update_and_fetch(k, v)

        q = mx.random.normal(shape=(1, 1, 3, 64))
        scale = 1.0 / (64**0.5)
        ref_k = cache._dequantize_keys(dtype=k.dtype)
        ref_v = cache._dequantize_values(dtype=v.dtype)
        ref = scaled_dot_product_attention(q, ref_k, ref_v, None, scale, None)
        out = scaled_dot_product_attention(q, None, dv, cache, scale, None)
        mx.eval(ref, out)

        self.assertTrue(mx.allclose(out, ref, atol=1e-4, rtol=1e-4))

    def test_make_prompt_cache_turboquant_keeps_edge_layers_fp16(self):
        """Test TurboQuant cache routing keeps first/last layers uncompressed."""
        model = DummyCacheModel([KVCache() for _ in range(6)])

        prompt_cache = make_prompt_cache(
            model,
            turbo_kv_bits=3,
            turbo_fp16_layers=1,
        )

        self.assertIsInstance(prompt_cache[0], KVCache)
        self.assertIsInstance(prompt_cache[-1], KVCache)
        for c in prompt_cache[1:-1]:
            self.assertIsInstance(c, TurboQuantKVCache)
            self.assertEqual(c.turbo_bits, 3)

    def test_make_prompt_cache_turboquant_preserves_mixed_caches(self):
        """Test non-KV cache entries survive TurboQuant routing unchanged."""
        model = DummyCacheModel(
            [
                KVCache(),
                ArraysCache(size=2),
                KVCache(),
                ArraysCache(size=2),
                KVCache(),
            ]
        )

        prompt_cache = make_prompt_cache(
            model,
            turbo_kv_bits=3.5,
            turbo_fp16_layers=1,
        )

        self.assertIsInstance(prompt_cache[0], KVCache)
        self.assertIsInstance(prompt_cache[1], ArraysCache)
        self.assertIsInstance(prompt_cache[2], TurboQuantKVCache)
        self.assertEqual(prompt_cache[2].turbo_bits, 3.5)
        self.assertIsInstance(prompt_cache[3], ArraysCache)
        self.assertIsInstance(prompt_cache[4], KVCache)

    def test_make_prompt_cache_turboquant_forwards_decode_buffer(self):
        """Test TurboQuant cache routing forwards decode-buffer mode."""
        model = DummyCacheModel([KVCache() for _ in range(4)])

        prompt_cache = make_prompt_cache(
            model,
            turbo_kv_bits=3,
            turbo_fp16_layers=1,
            turbo_decode_buffer=True,
        )

        self.assertIsInstance(prompt_cache[0], KVCache)
        self.assertIsInstance(prompt_cache[-1], KVCache)
        self.assertTrue(prompt_cache[1].decode_buffer)
        self.assertTrue(prompt_cache[2].decode_buffer)

    def test_make_prompt_cache_rotorquant_keeps_edge_layers_fp16(self):
        """Test RotorQuant cache routing keeps first/last layers uncompressed."""
        model = DummyCacheModel([KVCache() for _ in range(6)])

        prompt_cache = make_prompt_cache(
            model,
            rotor_kv_bits=4,
            rotor_fp16_layers=1,
        )

        self.assertIsInstance(prompt_cache[0], KVCache)
        self.assertIsInstance(prompt_cache[-1], KVCache)
        for c in prompt_cache[1:-1]:
            self.assertIsInstance(c, RotorQuantKVCache)
            self.assertEqual(c.turbo_bits, 4)
            self.assertEqual(c.rotation_mode, "rotorquant")

    def test_make_prompt_cache_rotorquant_preserves_mixed_caches(self):
        """Test non-KV cache entries survive RotorQuant routing unchanged."""
        model = DummyCacheModel(
            [
                KVCache(),
                ArraysCache(size=2),
                KVCache(),
                ArraysCache(size=2),
                KVCache(),
            ]
        )

        prompt_cache = make_prompt_cache(
            model,
            rotor_kv_bits=4,
            rotor_fp16_layers=1,
        )

        self.assertIsInstance(prompt_cache[0], KVCache)
        self.assertIsInstance(prompt_cache[1], ArraysCache)
        self.assertIsInstance(prompt_cache[2], RotorQuantKVCache)
        self.assertEqual(prompt_cache[2].rotation_mode, "rotorquant")
        self.assertIsInstance(prompt_cache[3], ArraysCache)
        self.assertIsInstance(prompt_cache[4], KVCache)

    def test_make_prompt_cache_rotorquant_forwards_decode_buffer(self):
        """Test RotorQuant cache routing forwards decode-buffer mode."""
        model = DummyCacheModel([KVCache() for _ in range(4)])

        prompt_cache = make_prompt_cache(
            model,
            rotor_kv_bits=4,
            rotor_fp16_layers=1,
            rotor_decode_buffer=True,
        )

        self.assertIsInstance(prompt_cache[0], KVCache)
        self.assertIsInstance(prompt_cache[-1], KVCache)
        self.assertTrue(prompt_cache[1].decode_buffer)
        self.assertTrue(prompt_cache[2].decode_buffer)
        self.assertEqual(prompt_cache[1].rotation_mode, "rotorquant")

    def test_maybe_quantize_kv_cache_keeps_edge_layers_fp16(self):
        prompt_cache = [KVCache() for _ in range(5)]
        x = mx.random.uniform(shape=(1, 2, 4, 64))
        for cache in prompt_cache:
            cache.update_and_fetch(x, x)

        maybe_quantize_kv_cache(
            prompt_cache,
            quantized_kv_start=0,
            kv_group_size=64,
            kv_bits=4,
            quantized_kv_fp16_layers=1,
        )

        self.assertIsInstance(prompt_cache[0], KVCache)
        self.assertIsInstance(prompt_cache[-1], KVCache)
        for layer in prompt_cache[1:-1]:
            self.assertIsInstance(layer, QuantizedKVCache)

    def test_maybe_quantize_kv_cache_quantizes_all_layers_without_edges(self):
        prompt_cache = [KVCache() for _ in range(4)]
        x = mx.random.uniform(shape=(1, 2, 4, 64))
        for cache in prompt_cache:
            cache.update_and_fetch(x, x)

        maybe_quantize_kv_cache(
            prompt_cache,
            quantized_kv_start=0,
            kv_group_size=64,
            kv_bits=4,
            quantized_kv_fp16_layers=0,
        )

        for layer in prompt_cache:
            self.assertIsInstance(layer, QuantizedKVCache)

    def test_make_prompt_cache_rejects_multiple_compressed_backends(self):
        """Test cache creation rejects mixing TurboQuant and RotorQuant."""
        model = DummyCacheModel([KVCache() for _ in range(2)])

        with self.assertRaises(ValueError):
            make_prompt_cache(
                model,
                turbo_kv_bits=3,
                rotor_kv_bits=4,
            )

    def test_make_prompt_cache_turboquant_rejects_rotating_fallback(self):
        """Test TurboQuant creation is explicit when only rotating fallback exists."""

        class DummyNoCacheModel:
            def __init__(self):
                self.layers = [object()] * 2

        with self.assertRaises(ValueError):
            make_prompt_cache(
                DummyNoCacheModel(),
                max_kv_size=32,
                turbo_kv_bits=3,
            )

    def test_turboquant_pack_unpack_roundtrip(self):
        """Test low-bit pack/unpack preserves the legacy packed layout."""
        values = mx.random.randint(0, 8, shape=(2, 3, 5, 17), dtype=mx.uint32)
        for bits in (1, 2, 3, 4):
            max_level = 1 << bits
            clipped = (values % max_level).astype(mx.uint8)
            packed = _pack(clipped, bits)
            unpacked = _unpack(packed, bits, clipped.shape[-1])
            self.assertTrue(mx.array_equal(clipped, unpacked))

    def test_turboquant_qjl_packed_residual_matches_unpacked(self):
        """Test packed QJL signs preserve the same 1-bit residual payload."""
        unit_vectors = mx.random.normal(shape=(1, 2, 7, 64))
        mse_vectors = mx.random.normal(shape=(1, 2, 7, 64))
        projection, projection_t = _cached_qjl_projection(64, mode="wht")

        unpacked_signs, gamma = _quantize_qjl_residual(
            unit_vectors,
            mse_vectors,
            projection_t,
        )
        packed_signs, packed_gamma = _quantize_qjl_residual_packed(
            unit_vectors,
            mse_vectors,
            projection_t,
        )
        recovered_signs = _unpack(packed_signs, 1, unit_vectors.shape[-1])

        self.assertTrue(mx.array_equal(unpacked_signs, recovered_signs))
        self.assertTrue(mx.allclose(gamma, packed_gamma))

    def test_turboquant_metal_qjl_score_matches_reference(self):
        """Test packed QJL score kernel matches unpacked sign-dot reference."""
        if not _metal_available():
            self.skipTest("Metal QJL score kernel unavailable")

        q_proj = mx.random.normal(shape=(1, 2, 3, 64), dtype=mx.float32)
        sign_bits = mx.random.randint(0, 2, shape=(1, 2, 5, 64), dtype=mx.uint32)
        packed = _pack(sign_bits.astype(mx.uint8), 1)
        norms = mx.random.uniform(shape=(1, 2, 5), low=0.5, high=1.5)
        residual_norms = mx.random.uniform(shape=(1, 2, 5), low=0.05, high=0.5)
        scale = mx.array([0.123], dtype=mx.float32)

        fast = _metal_qjl_score(q_proj, norms, residual_norms, packed, scale)
        unpacked = _unpack(packed, 1, q_proj.shape[-1]).astype(mx.float32)
        signs = unpacked * 2.0 - 1.0
        ref = (
            mx.einsum("bhrd,bhtd->bhrt", q_proj, signs)
            * norms[:, :, None, :]
            * residual_norms[:, :, None, :]
            * scale[0]
        )
        mx.eval(fast, ref)

        self.assertTrue(mx.allclose(fast, ref, atol=1e-5, rtol=1e-5))

    def test_turboquant_prod_mode_uses_qjl_residual(self):
        """Test prod mode stores a separate 1-bit QJL residual for keys."""
        c = TurboQuantKVCache(bits=3, estimator_mode="prod")
        k = mx.random.normal(shape=(1, 4, 12, 64))
        v = mx.random.normal(shape=(1, 4, 12, 64))
        dk, dv = c.update_and_fetch(k, v)
        mx.eval(dk, dv)

        self.assertEqual(c.estimator_mode, "prod")
        self.assertIsNotNone(c._k_qjl_indices)
        self.assertIsNotNone(c._k_qjl_gamma)
        self.assertLess(c._k_indices.shape[-1], c._v_indices.shape[-1])
        self.assertEqual(dk.shape, k.shape)
        self.assertEqual(dv.shape, v.shape)

    def test_turboquant_prod_mode_without_qjl_falls_back_to_plain_key_quant(self):
        """Test prod-mode ablation can disable QJL and still quantize cleanly."""
        c = TurboQuantKVCache(bits=4, estimator_mode="prod", qjl_residual=False)
        k = mx.random.normal(shape=(1, 4, 12, 64))
        v = mx.random.normal(shape=(1, 4, 12, 64))
        dk, dv = c.update_and_fetch(k, v)
        mx.eval(dk, dv)

        self.assertEqual(c.estimator_mode, "prod")
        self.assertFalse(c.qjl_residual)
        self.assertEqual(c._k_bits, 4)
        self.assertIsNone(c._k_qjl_indices)
        self.assertIsNone(c._k_qjl_gamma)
        self.assertEqual(dk.shape, k.shape)
        self.assertEqual(dv.shape, v.shape)

    def test_turboquant_max_cache_tokens_evicts_oldest_tokens(self):
        """Test TurboQuant can evict the oldest compressed tokens to a fixed window."""
        c = TurboQuantKVCache(bits=4, max_cache_tokens=10)
        k0 = mx.random.normal(shape=(1, 2, 8, 32))
        v0 = mx.random.normal(shape=(1, 2, 8, 32))
        k1 = mx.random.normal(shape=(1, 2, 8, 32))
        v1 = mx.random.normal(shape=(1, 2, 8, 32))

        c.update_and_fetch(k0, v0)
        dk, dv = c.update_and_fetch(k1, v1)
        ref_k = mx.concatenate([k0, k1], axis=2)[..., -10:, :]
        ref_v = mx.concatenate([v0, v1], axis=2)[..., -10:, :]
        mx.eval(dk, dv, ref_k, ref_v)

        self.assertEqual(c.offset, 10)
        self.assertEqual(dk.shape[2], 10)
        self.assertEqual(dv.shape[2], 10)
        self.assertLess(float(mx.max(mx.abs(dk - ref_k))), 2.0)
        self.assertLess(float(mx.max(mx.abs(dv - ref_v))), 2.0)

    def test_turboquant_prod_correction_is_unbiased_in_expectation(self):
        """Test prod-mode QJL correction improves score reconstruction in expectation."""
        mx.random.seed(0)
        k = mx.random.normal(shape=(1, 1, 96, 128))
        v = mx.random.normal(shape=(1, 1, 96, 128))
        q = mx.random.normal(shape=(1, 1, 48, 128))

        c = TurboQuantKVCache(bits=3, estimator_mode="prod")
        c.update_and_fetch(k, v)
        dk_mse = c._dequantize_keys(include_qjl=False, dtype=k.dtype)

        prod_samples = []
        for seed in range(16):
            sample = TurboQuantKVCache(bits=3, estimator_mode="prod")
            sample._init_codebook(128)
            sample._qjl_projection, sample._qjl_projection_t = _cached_qjl_projection(
                128, seed=seed
            )
            dk_prod, _ = sample.update_and_fetch(k, v)
            prod_samples.append(dk_prod)
        dk_prod = mx.mean(mx.stack(prod_samples, axis=0), axis=0)

        scores_true = q @ mx.swapaxes(k, -1, -2)
        scores_mse = q @ mx.swapaxes(dk_mse, -1, -2)
        scores_prod = q @ mx.swapaxes(dk_prod, -1, -2)
        err_mse = mx.mean(mx.square(scores_true - scores_mse))
        err_prod = mx.mean(mx.square(scores_true - scores_prod))
        mx.eval(err_mse, err_prod)

        self.assertLess(float(err_prod), float(err_mse))

    def test_turboquant_decode_buffer_incremental_matches_raw_cache(self):
        """Test decode-buffer mode keeps a live raw KV mirror for fast decode."""
        mx.random.seed(0)
        k = mx.random.normal(shape=(1, 4, 8, 64))
        v = mx.random.normal(shape=(1, 4, 8, 64))
        k_next = mx.random.normal(shape=(1, 4, 1, 64))
        v_next = mx.random.normal(shape=(1, 4, 1, 64))

        buffered = TurboQuantKVCache(
            bits=3,
            estimator_mode="prod",
            qjl_projection_mode="wht",
            decode_buffer=True,
        )

        buffered.update_and_fetch(k, v)
        dk, dv = buffered.update_and_fetch(k_next, v_next)
        ref_k = mx.concatenate([k, k_next], axis=2)
        ref_v = mx.concatenate([v, v_next], axis=2)
        diff_k = mx.max(mx.abs(dk - ref_k))
        diff_v = mx.max(mx.abs(dv - ref_v))
        mx.eval(diff_k, diff_v, dk, dv)

        self.assertIsNotNone(buffered._k_deq_buf)
        self.assertIsNotNone(buffered._v_deq_buf)
        self.assertEqual(buffered._deq_offset, 9)
        self.assertLess(float(diff_k), 1e-5)
        self.assertLess(float(diff_v), 1e-5)

    def test_turboquant_decode_buffer_uses_raw_append_before_materialize(self):
        """Test decode-buffer mode reuses raw K/V tensors instead of re-dequantizing them."""
        mx.random.seed(0)
        k = mx.random.normal(shape=(1, 4, 8, 64))
        v = mx.random.normal(shape=(1, 4, 8, 64))
        k_next = mx.random.normal(shape=(1, 4, 1, 64))
        v_next = mx.random.normal(shape=(1, 4, 1, 64))

        cache = TurboQuantKVCache(
            bits=3,
            estimator_mode="prod",
            qjl_projection_mode="wht",
            decode_buffer=True,
        )

        def _unexpected_materialize(*_args, **_kwargs):
            raise AssertionError("decode buffer should append raw keys/values directly")

        cache._materialize_decode_buffer = _unexpected_materialize
        dk, dv = cache.update_and_fetch(k, v)
        dk2, dv2 = cache.update_and_fetch(k_next, v_next)
        mx.eval(dk, dv, dk2, dv2)

        self.assertEqual(dk.shape, k.shape)
        self.assertEqual(dv.shape, v.shape)
        self.assertEqual(dk2.shape[-2], 9)
        self.assertEqual(dv2.shape[-2], 9)
        self.assertEqual(cache._deq_offset, 9)

    def test_turboquant_recent_buffer_attention_matches_reference(self):
        """Test mixed compressed+recent-buffer attention matches materialized KV."""
        if not hasattr(mx.fast, "turboquant_qk_packed_scores"):
            self.skipTest("packed TurboQuant score kernel unavailable")
        if not hasattr(mx.fast, "turboquant_av_packed_values_batched"):
            self.skipTest("packed TurboQuant AV kernel unavailable")

        mx.random.seed(0)
        cache = TurboQuantKVCache(bits=4, buffer_size=4, flush_batch_size=4)
        cache._fused_enabled = True
        cache.min_fused_tokens = 0

        k0 = mx.random.normal(shape=(1, 4, 8, 64))
        v0 = mx.random.normal(shape=(1, 4, 8, 64))
        cache.update_and_fetch(k0, v0)

        self.assertEqual(cache.compressed_tokens, 0)
        self.assertEqual(cache.buffer_tokens, 8)

        k1 = mx.random.normal(shape=(1, 4, 1, 64))
        v1 = mx.random.normal(shape=(1, 4, 1, 64))
        buf_k, buf_v = cache.update_and_fetch(k1, v1)

        self.assertEqual(cache.compressed_tokens, 5)
        self.assertEqual(cache.buffer_tokens, 4)
        self.assertEqual(cache.size(), 9)
        self.assertEqual(buf_k.shape[-2], 4)
        self.assertEqual(buf_v.shape[-2], 4)

        q = mx.random.normal(shape=(1, 4, 1, 64))
        scale = 1.0 / (64**0.5)
        ref_k = cache._dequantize_keys(dtype=k1.dtype)
        ref_v = cache._dequantize_values(dtype=v1.dtype)
        ref = scaled_dot_product_attention(q, ref_k, ref_v, None, scale, None)
        out = scaled_dot_product_attention(q, buf_k, buf_v, cache, scale, None)
        self.assertTrue(mx.allclose(out, ref, atol=1e-4, rtol=1e-4))

    def test_turboquant_rotor3_mode(self):
        """Test TurboQuant rotor3 mode wiring and cache state."""
        c = TurboQuantKVCache(bits=4, rotation_mode="rotor3", sparse_v_tau=1e-4)
        k = mx.random.normal(shape=(1, 8, 12, 96))
        v = mx.random.normal(shape=(1, 8, 12, 96))
        dk, dv = c.update_and_fetch(k, v)
        mx.eval(dk, dv)

        self.assertEqual(c.size(), 12)
        self.assertEqual(c.rotation_mode, "rotor3")
        self.assertAlmostEqual(c.sparse_v_tau, 1e-4, places=8)

    def test_turbo_sparse_v_percentile_masks_bottom_half(self):
        probs = mx.array([[[[0.01, 0.02, 0.03, 0.04, 0.30, 0.60]]]], dtype=mx.float32)
        cache = SimpleNamespace(sparse_v_mode="percentile", sparse_v_percentile=50.0)
        mask = _compute_turbo_sparse_v_mask(probs, cache)
        mx.eval(mask)
        self.assertEqual(mask.reshape(-1).tolist(), [False, False, False, True, True, True])

    def test_turbo_sparse_v_adaptive_keeps_more_in_late_layers(self):
        probs = mx.array([[[[0.001, 0.002, 0.004, 0.008, 0.16, 0.825]]]], dtype=mx.float32)
        early = SimpleNamespace(
            sparse_v_mode="adaptive",
            sparse_v_percentile=50.0,
            sparse_v_early_multiplier=1.25,
            sparse_v_late_multiplier=0.75,
            layer_idx=0,
            num_layers=10,
        )
        late = SimpleNamespace(
            sparse_v_mode="adaptive",
            sparse_v_percentile=50.0,
            sparse_v_early_multiplier=1.25,
            sparse_v_late_multiplier=0.75,
            layer_idx=9,
            num_layers=10,
        )
        early_mask = _compute_turbo_sparse_v_mask(probs, early)
        late_mask = _compute_turbo_sparse_v_mask(probs, late)
        mx.eval(early_mask, late_mask)
        self.assertLessEqual(int(mx.sum(early_mask).item()), int(mx.sum(late_mask).item()))

    def test_turbo_sparse_v_renormalizes_masked_probs(self):
        probs = mx.array([[[[0.01, 0.02, 0.07, 0.90]]]], dtype=mx.float32)
        cache = SimpleNamespace(sparse_v_mode="percentile", sparse_v_percentile=50.0)
        sparse = _apply_turbo_sparse_v(probs, cache)
        mx.eval(sparse)
        self.assertTrue(mx.allclose(mx.sum(sparse, axis=-1), mx.ones((1, 1, 1)), atol=1e-6))
        self.assertEqual(float(sparse[..., 0].item()), 0.0)

    def test_turboquant_rotorquant_nondivisible_dim_stays_independent(self):
        """Test RotorQuant stays blockwise even when head dim is not divisible by 3."""
        c = TurboQuantKVCache(bits=4, rotation_mode="rotorquant")
        k = mx.random.normal(shape=(1, 4, 6, 64))
        v = mx.random.normal(shape=(1, 4, 6, 64))
        dk, dv = c.update_and_fetch(k, v)
        mx.eval(dk, dv)

        self.assertEqual(c.rotation_mode, "rotorquant")
        self.assertEqual(c._rotation.ndim, 3)
        self.assertEqual(c._rotation.shape[0], 64 // 3)

    def test_turboquant_prod_save_load(self):
        """Test prod-mode TurboQuant cache save/load roundtrip."""
        cache_file = os.path.join(self.test_dir, "turbo_cache_prod.safetensors")

        cache = [TurboQuantKVCache(bits=3, estimator_mode="prod") for _ in range(2)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 64))
            c.update_and_fetch(x, x)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)

        for c, lc in zip(cache, loaded_cache):
            self.assertEqual(c.offset, lc.offset)
            self.assertEqual(c.turbo_bits, lc.turbo_bits)
            self.assertEqual(lc.estimator_mode, "prod")
            self.assertIsNotNone(lc._k_qjl_indices)
            self.assertIsNotNone(lc._k_qjl_gamma)

    def test_turboquant_fractional_save_load(self):
        """Test fractional-bit TurboQuant cache save/load roundtrip."""
        cache_file = os.path.join(self.test_dir, "turbo_cache_frac.safetensors")

        cache = [TurboQuantKVCache(bits=3.5)]
        x = mx.random.uniform(shape=(1, 8, 10, 64))
        cache[0].update_and_fetch(x, x)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)

        self.assertEqual(len(loaded_cache), 1)
        loaded = loaded_cache[0]
        self.assertEqual(loaded.offset, 10)
        self.assertEqual(float(loaded.turbo_bits), 3.5)
        self.assertTrue(loaded._fractional_split)
        self.assertIsNotNone(loaded._split_low_cache)
        self.assertIsNotNone(loaded._split_high_cache)

    def test_turboquant_decode_buffer_save_load(self):
        """Test decode-buffer flag survives save/load roundtrip."""
        cache_file = os.path.join(self.test_dir, "turbo_cache_decode_buffer.safetensors")

        cache = [TurboQuantKVCache(bits=3, decode_buffer=True)]
        x = mx.random.normal(shape=(1, 8, 10, 64))
        cache[0].update_and_fetch(x, x)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        self.assertTrue(loaded_cache[0].decode_buffer)

    def test_turboquant_experimental_flags_save_load(self):
        """Test asymmetric bits, QJL ablation, and max window survive save/load."""
        cache_file = os.path.join(self.test_dir, "turbo_cache_experimental.safetensors")

        cache = [
            TurboQuantKVCache(
                bits=4,
                key_bits=4,
                value_bits=3,
                estimator_mode="prod",
                qjl_residual=False,
                max_cache_tokens=12,
            )
        ]
        x = mx.random.normal(shape=(1, 8, 10, 64))
        cache[0].update_and_fetch(x, x)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        loaded = loaded_cache[0]
        self.assertEqual(loaded.key_bits_override, 4)
        self.assertEqual(loaded.value_bits_override, 3)
        self.assertFalse(loaded.qjl_residual)
        self.assertEqual(loaded.max_cache_tokens, 12)

    def test_rotorquant_save_load(self):
        """Test RotorQuant cache save/load roundtrip."""
        cache_file = os.path.join(self.test_dir, "rotor_cache.safetensors")

        cache = [RotorQuantKVCache(bits=4, decode_buffer=True)]
        x = mx.random.normal(shape=(1, 8, 10, 64))
        cache[0].update_and_fetch(x, x)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)

        self.assertEqual(len(loaded_cache), 1)
        self.assertIsInstance(loaded_cache[0], RotorQuantKVCache)
        self.assertTrue(loaded_cache[0].decode_buffer)
        self.assertEqual(loaded_cache[0].rotation_mode, "rotorquant")

    def test_rotorquant_prod_save_load(self):
        """Test RotorQuant prod-mode cache save/load roundtrip."""
        cache_file = os.path.join(self.test_dir, "rotor_cache_prod.safetensors")

        cache = [RotorQuantKVCache(bits=4, estimator_mode="prod")]
        x = mx.random.normal(shape=(1, 8, 10, 64))
        cache[0].update_and_fetch(x, x)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)

        self.assertEqual(len(loaded_cache), 1)
        self.assertIsInstance(loaded_cache[0], RotorQuantKVCache)
        self.assertEqual(loaded_cache[0].estimator_mode, "prod")
        self.assertIsNotNone(loaded_cache[0]._k_qjl_indices)
        self.assertIsNotNone(loaded_cache[0]._k_qjl_gamma)

    def test_turboquant_fused_av_matches_dequantized_values(self):
        """Test fused AV restores values back to the model basis."""
        if not hasattr(mx.fast, "turboquant_av_packed_values_batched"):
            self.skipTest("Native TurboQuant AV op unavailable")

        prev = os.environ.get("MLX_TQ_FUSED")
        os.environ["MLX_TQ_FUSED"] = "0"
        try:
            cache = TurboQuantKVCache(bits=4, rotation_mode="rotorquant")
            k = mx.random.normal(shape=(1, 1, 5, 96))
            v = mx.random.normal(shape=(1, 1, 5, 96))
            _, dv = cache.update_and_fetch(k, v)
            mx.eval(dv)

            probs = mx.array([[[[0.05, 0.15, 0.3, 0.25, 0.25]]]], dtype=mx.float32)
            out_ref = probs @ dv

            cache._fused_enabled = True
            out_fused = cache.fused_av(probs)
            mx.eval(out_ref, out_fused)

            self.assertTrue(mx.allclose(out_fused, out_ref, rtol=2e-3, atol=2e-3))
        finally:
            if prev is None:
                os.environ.pop("MLX_TQ_FUSED", None)
            else:
                os.environ["MLX_TQ_FUSED"] = prev

    def test_turboquant_fused_attention_matches_reference(self):
        """Test fused packed decode attention matches dequantized SDPA."""
        if not hasattr(mx.fast, "turboquant_decode_attention_packed_batched"):
            self.skipTest("Native TurboQuant decode attention op unavailable")

        prev = os.environ.get("MLX_TQ_FUSED")
        os.environ["MLX_TQ_FUSED"] = "0"
        try:
            cache = TurboQuantKVCache(bits=4, rotation_mode="rotorquant")
            k = mx.random.normal(shape=(1, 1, 7, 64))
            v = mx.random.normal(shape=(1, 1, 7, 64))
            dk, dv = cache.update_and_fetch(k, v)

            q = mx.random.normal(shape=(1, 1, 1, 64))
            scores = q @ mx.swapaxes(dk, -1, -2)
            probs = mx.softmax(scores, axis=-1, precise=True)
            out_ref = probs @ dv

            cache._fused_enabled = True
            out_fused = cache.fused_attention(q)
            mx.eval(out_ref, out_fused)

            self.assertTrue(mx.allclose(out_fused, out_ref, rtol=3e-3, atol=3e-3))
        finally:
            if prev is None:
                os.environ.pop("MLX_TQ_FUSED", None)
            else:
                os.environ["MLX_TQ_FUSED"] = prev

    def test_turboquant_prod_fused_scores_match_reference(self):
        """Test prod-mode fused scores match dequantized key scores."""
        if not hasattr(mx.fast, "turboquant_qk_prod_scores_batched"):
            self.skipTest("Native TurboQuant prod QK op unavailable")

        prev = os.environ.get("MLX_TQ_FUSED")
        os.environ["MLX_TQ_FUSED"] = "1"
        try:
            cache = TurboQuantKVCache(bits=3, estimator_mode="prod")
            k = mx.random.normal(shape=(1, 1, 11, 64))
            v = mx.random.normal(shape=(1, 1, 11, 64))
            dk, _ = cache.update_and_fetch(k, v)

            q = mx.random.normal(shape=(1, 1, 2, 64))
            scores_ref = q @ mx.swapaxes(dk, -1, -2)
            scores_fused = cache.fused_scores(q)
            mx.eval(scores_ref, scores_fused)

            self.assertTrue(mx.allclose(scores_fused, scores_ref, rtol=4e-3, atol=4e-3))
        finally:
            if prev is None:
                os.environ.pop("MLX_TQ_FUSED", None)
            else:
                os.environ["MLX_TQ_FUSED"] = prev

    def test_turboquant_prod_fused_attention_matches_reference(self):
        """Test prod-mode fused decode attention matches dequantized SDPA."""
        if not hasattr(mx.fast, "turboquant_decode_attention_prod_batched"):
            self.skipTest("Native TurboQuant prod decode attention op unavailable")

        prev = os.environ.get("MLX_TQ_FUSED")
        os.environ["MLX_TQ_FUSED"] = "0"
        try:
            cache = TurboQuantKVCache(bits=3, estimator_mode="prod")
            k = mx.random.normal(shape=(1, 1, 9, 64))
            v = mx.random.normal(shape=(1, 1, 9, 64))
            dk, dv = cache.update_and_fetch(k, v)

            q = mx.random.normal(shape=(1, 1, 1, 64))
            scores = q @ mx.swapaxes(dk, -1, -2)
            probs = mx.softmax(scores, axis=-1, precise=True)
            out_ref = probs @ dv

            cache._fused_enabled = True
            out_fused = cache.fused_attention(q)
            mx.eval(out_ref, out_fused)

            self.assertTrue(mx.allclose(out_fused, out_ref, rtol=4e-3, atol=4e-3))
        finally:
            if prev is None:
                os.environ.pop("MLX_TQ_FUSED", None)
            else:
                os.environ["MLX_TQ_FUSED"] = prev

    def test_turboquant_prod_no_qjl_k3_v4_defaults_fused_on(self):
        """The exact k3/v4 winner should auto-enable the fused packed path."""
        prev = os.environ.pop("MLX_TQ_FUSED", None)
        try:
            cache = TurboQuantKVCache(
                bits=4,
                key_bits=3,
                value_bits=4,
                estimator_mode="prod",
                qjl_residual=False,
            )
            self.assertTrue(cache._fused_enabled)

            x = mx.random.normal(shape=(1, 2, 3, 128))
            cache.update_and_fetch(x, x)
            self.assertEqual(cache._k_bits, 3)
            self.assertEqual(cache._v_bits, 4)
        finally:
            if prev is None:
                os.environ.pop("MLX_TQ_FUSED", None)
            else:
                os.environ["MLX_TQ_FUSED"] = prev

    def test_turboquant_prod_no_qjl_env_can_disable_fused_default(self):
        prev = os.environ.get("MLX_TQ_FUSED")
        os.environ["MLX_TQ_FUSED"] = "0"
        try:
            cache = TurboQuantKVCache(
                bits=4,
                key_bits=3,
                value_bits=4,
                estimator_mode="prod",
                qjl_residual=False,
            )
            self.assertFalse(cache._fused_enabled)
        finally:
            if prev is None:
                os.environ.pop("MLX_TQ_FUSED", None)
            else:
                os.environ["MLX_TQ_FUSED"] = prev

    def test_turboquant_prod_no_qjl_k3_v4_skips_invalid_one_pass_decode_op(self):
        """k3/v4 must not route through the single-bit-width one-pass decode op."""
        prev = os.environ.pop("MLX_TQ_FUSED", None)
        try:
            cache = TurboQuantKVCache(
                bits=4,
                key_bits=3,
                value_bits=4,
                rotation_mode="dense",
                estimator_mode="prod",
                qjl_residual=False,
            )
            k = mx.random.normal(shape=(1, 2, 7, 128))
            v = mx.random.normal(shape=(1, 2, 7, 128))
            cache.update_and_fetch(k, v)

            q = mx.random.normal(shape=(1, 8, 1, 128))
            self.assertIsNone(cache.fused_attention(q))
        finally:
            if prev is None:
                os.environ.pop("MLX_TQ_FUSED", None)
            else:
                os.environ["MLX_TQ_FUSED"] = prev

    def test_turboquant_prod_no_qjl_k3_v4_fused_scores_av_matches_reference_gqa128(self):
        """Exact winner math should match the dequantized path for head_dim=128."""
        if not hasattr(mx.fast, "turboquant_qk_packed_scores_batched"):
            self.skipTest("Native TurboQuant packed score op unavailable")
        if not hasattr(mx.fast, "turboquant_av_packed_values_batched"):
            self.skipTest("Native TurboQuant packed AV op unavailable")

        prev = os.environ.pop("MLX_TQ_FUSED", None)
        try:
            cache = TurboQuantKVCache(
                bits=4,
                key_bits=3,
                value_bits=4,
                rotation_mode="dense",
                estimator_mode="prod",
                qjl_residual=False,
            )
            k = mx.random.normal(shape=(1, 2, 9, 128))
            v = mx.random.normal(shape=(1, 2, 9, 128))
            dk, dv = cache.update_and_fetch(k, v)

            q = mx.random.normal(shape=(1, 8, 1, 128))
            repeats = q.shape[1] // dk.shape[1]
            dk_rep = mx.repeat(dk, repeats=repeats, axis=1)
            dv_rep = mx.repeat(dv, repeats=repeats, axis=1)
            scores = q @ mx.swapaxes(dk_rep, -1, -2)
            probs = mx.softmax(scores, axis=-1, precise=True)
            out_ref = probs @ dv_rep

            scores_fused = cache.fused_scores(q)
            probs_fused = mx.softmax(scores_fused, axis=-1, precise=True)
            out_fused = cache.fused_av(probs_fused)
            mx.eval(out_ref, scores_fused, probs_fused, out_fused)

            self.assertTrue(mx.allclose(out_fused, out_ref, rtol=4e-3, atol=4e-3))
        finally:
            if prev is None:
                os.environ.pop("MLX_TQ_FUSED", None)
            else:
                os.environ["MLX_TQ_FUSED"] = prev

    def test_turboquant_prod_no_qjl_k3_v4_fused_scores_av_matches_reference_gqa256(self):
        """Exact winner math should match the dequantized path for head_dim=256."""
        if not hasattr(mx.fast, "turboquant_qk_packed_scores_batched"):
            self.skipTest("Native TurboQuant packed score op unavailable")
        if not hasattr(mx.fast, "turboquant_av_packed_values_batched"):
            self.skipTest("Native TurboQuant packed AV op unavailable")

        prev = os.environ.pop("MLX_TQ_FUSED", None)
        try:
            cache = TurboQuantKVCache(
                bits=4,
                key_bits=3,
                value_bits=4,
                rotation_mode="dense",
                estimator_mode="prod",
                qjl_residual=False,
            )
            k = mx.random.normal(shape=(1, 2, 7, 256))
            v = mx.random.normal(shape=(1, 2, 7, 256))
            dk, dv = cache.update_and_fetch(k, v)

            q = mx.random.normal(shape=(1, 16, 1, 256))
            repeats = q.shape[1] // dk.shape[1]
            dk_rep = mx.repeat(dk, repeats=repeats, axis=1)
            dv_rep = mx.repeat(dv, repeats=repeats, axis=1)
            scores = q @ mx.swapaxes(dk_rep, -1, -2)
            probs = mx.softmax(scores, axis=-1, precise=True)
            out_ref = probs @ dv_rep

            scores_fused = cache.fused_scores(q)
            probs_fused = mx.softmax(scores_fused, axis=-1, precise=True)
            out_fused = cache.fused_av(probs_fused)
            mx.eval(out_ref, scores_fused, probs_fused, out_fused)

            self.assertTrue(mx.allclose(out_fused, out_ref, rtol=5e-3, atol=5e-3))
        finally:
            if prev is None:
                os.environ.pop("MLX_TQ_FUSED", None)
            else:
                os.environ["MLX_TQ_FUSED"] = prev

    def test_turboquant_with_model(self):
        """Test TurboQuantKVCache with actual model generation."""
        num_layers = len(self.model.layers)
        args = self.model.args
        head_dim = getattr(
            args, "head_dim", args.hidden_size // args.num_attention_heads
        )

        # FP16 baseline
        fp16_cache = [KVCache() for _ in range(num_layers)]
        prompt = mx.array([[1, 2, 3, 4, 5]])
        logits_fp16 = self.model(prompt, cache=fp16_cache)
        mx.eval(logits_fp16)

        # TurboQuant
        tq_cache = [TurboQuantKVCache(bits=4) for _ in range(num_layers)]
        logits_tq = self.model(prompt, cache=tq_cache)
        mx.eval(logits_tq)

        self.assertEqual(logits_fp16.shape, logits_tq.shape)

        # Logit cosine similarity should be reasonable
        l1 = logits_fp16[0, -1].astype(mx.float32)
        l2 = logits_tq[0, -1].astype(mx.float32)
        cos = float(
            mx.sum(l1 * l2) / (mx.linalg.norm(l1) * mx.linalg.norm(l2) + 1e-8)
        )
        self.assertGreater(cos, 0.9)

    def test_turboquant_nbytes(self):
        """Test TurboQuantKVCache memory is less than FP16."""
        kv = KVCache()
        tq = TurboQuantKVCache(bits=3)

        k = mx.random.uniform(shape=(1, 8, 256, 128))
        v = mx.random.uniform(shape=(1, 8, 256, 128))

        kv.update_and_fetch(k, v)
        tq.update_and_fetch(k, v)
        mx.eval(kv.keys, tq._k_indices)

        self.assertLess(tq.nbytes, kv.nbytes)


if __name__ == "__main__":
    unittest.main()
