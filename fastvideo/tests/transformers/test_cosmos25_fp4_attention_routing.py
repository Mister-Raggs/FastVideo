# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

from fastvideo.configs.models.dits.cosmos2_5 import Cosmos25ArchConfig
from fastvideo.models.dits import cosmos2_5
from fastvideo.platforms import AttentionBackendEnum


class _RecordingAttention(nn.Module):

    def __init__(self, *args, supported_attention_backends=None, **kwargs) -> None:
        super().__init__()
        self.supported_attention_backends = supported_attention_backends


def test_cosmos25_routes_fp4_to_self_attention_only(monkeypatch) -> None:
    monkeypatch.setattr(cosmos2_5, "DistributedAttention", _RecordingAttention)
    monkeypatch.setattr(cosmos2_5, "LocalAttention", _RecordingAttention)
    supported = Cosmos25ArchConfig()._supported_attention_backends

    block = cosmos2_5.Cosmos25TransformerBlock(
        num_attention_heads=1,
        attention_head_dim=128,
        cross_attention_dim=128,
        supported_attention_backends=supported,
    )

    self_backends = block.attn1.attn.supported_attention_backends
    cross_backends = block.attn2.attn.supported_attention_backends

    assert self_backends == supported
    assert AttentionBackendEnum.ATTN_QAT_INFER in self_backends
    assert cross_backends == tuple(backend for backend in supported
                                   if backend is not AttentionBackendEnum.ATTN_QAT_INFER)


def test_cosmos25_default_cross_attention_backends_remain_dense(monkeypatch) -> None:
    monkeypatch.setattr(cosmos2_5, "LocalAttention", _RecordingAttention)

    attention = cosmos2_5.Cosmos25CrossAttention(
        dim=128,
        cross_attention_dim=128,
        num_heads=1,
    )

    assert attention.attn.supported_attention_backends == (
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TORCH_SDPA,
    )
