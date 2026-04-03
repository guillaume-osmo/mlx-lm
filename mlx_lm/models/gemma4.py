# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from . import gemma4_text
from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = gemma4_text.Model(
            gemma4_text.ModelArgs.from_dict(args.text_config)
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )

    def sanitize(self, weights):
        weights = tree_unflatten(list(weights.items()))
        for k in ["audio_tower", "vision_tower", "embed_audio", "embed_vision"]:
            weights.pop(k, None)
        lm_weights = dict(tree_flatten(weights["language_model"]))
        weights["language_model"] = tree_unflatten(list(lm_weights.items()))
        return dict(tree_flatten(weights))

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()
