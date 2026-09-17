# SPDX-License-Identifier: Apache-2.0
"""Model-free coverage for Cosmos2.5 VAE decoder compilation."""

import torch.nn as nn

from fastvideo.models.vaes.cosmos25wanvae import (
    Cosmos25Decoder3d,
    Cosmos25WanVAE,
    _is_cosmos25_vae_decoder,
)


def test_cosmos25_vae_registers_decoder_compile_condition() -> None:
    conditions = getattr(Cosmos25WanVAE, "_compile_conditions", None)

    assert conditions
    assert _is_cosmos25_vae_decoder in conditions


def test_cosmos25_vae_compile_condition_matches_only_top_level_decoder() -> None:
    decoder = object.__new__(Cosmos25Decoder3d)

    assert _is_cosmos25_vae_decoder("decoder", decoder) is True
    assert _is_cosmos25_vae_decoder("decoder.block", decoder) is False
    assert _is_cosmos25_vae_decoder("encoder", decoder) is False
    assert _is_cosmos25_vae_decoder("decoder", nn.Linear(1, 1)) is False
