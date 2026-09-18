# SPDX-License-Identifier: Apache-2.0
"""Model-free coverage for Cosmos2.5 VAE decoder compilation."""

from unittest.mock import patch

import pytest
import torch.nn as nn

from fastvideo.models.vaes.cosmos25wanvae import (
    Cosmos25AttentionBlock,
    Cosmos25Decoder3d,
    Cosmos25ResidualBlock,
    Cosmos25WanVAE,
    _is_cosmos25_vae_decoder,
    _is_cosmos25_vae_decoder_region,
)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase


def _empty_module(module_type):
    module = object.__new__(module_type)
    nn.Module.__init__(module)
    return module


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


def test_cosmos25_vae_registers_bounded_regional_compile_profile() -> None:
    profiles = getattr(Cosmos25WanVAE, "_compile_condition_profiles", None)

    assert profiles
    assert profiles["regional"] == [_is_cosmos25_vae_decoder_region]


def test_cosmos25_vae_regional_profile_matches_only_decoder_compute_blocks() -> None:
    residual = object.__new__(Cosmos25ResidualBlock)
    attention = object.__new__(Cosmos25AttentionBlock)

    assert _is_cosmos25_vae_decoder_region("decoder.middle.0", residual) is True
    assert _is_cosmos25_vae_decoder_region("decoder.middle.1", attention) is True
    assert _is_cosmos25_vae_decoder_region("decoder.upsamples.3", residual) is True
    assert _is_cosmos25_vae_decoder_region("encoder.middle.0", residual) is False
    assert _is_cosmos25_vae_decoder_region("decoder", residual) is False
    assert _is_cosmos25_vae_decoder_region("decoder.head.0", nn.Identity()) is False


def test_cosmos25_vae_regional_profile_compiles_regions_in_place() -> None:
    vae = _empty_module(Cosmos25WanVAE)
    decoder = _empty_module(Cosmos25Decoder3d)
    residual = _empty_module(Cosmos25ResidualBlock)
    attention = _empty_module(Cosmos25AttentionBlock)
    decoder.residual = residual
    decoder.attention = attention
    decoder.other = nn.Identity()
    vae.decoder = decoder

    with patch("fastvideo.pipelines.composed_pipeline_base.torch.compile", side_effect=lambda fn, **_kwargs: fn) as call:
        compiled_count = ComposedPipelineBase._compile_with_conditions(
            vae,
            {"backend": "inductor"},
            condition_profile="regional",
        )

    assert compiled_count == 2
    assert [entry.args[0].__self__ for entry in call.call_args_list] == [residual, attention]


def test_unknown_vae_compile_profile_fails_closed() -> None:
    vae = _empty_module(Cosmos25WanVAE)

    with pytest.raises(ValueError, match="does not declare VAE compile profile"):
        ComposedPipelineBase._compile_with_conditions(vae, {}, condition_profile="missing")
