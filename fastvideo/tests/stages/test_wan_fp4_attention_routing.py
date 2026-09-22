from types import SimpleNamespace

import torch.nn as nn

from fastvideo.models.wan import transformer as wan_transformer
from fastvideo.models.wan.config import WanVideoArchConfig
from fastvideo.pipelines.stages import denoising
from fastvideo.platforms import AttentionBackendEnum


class _RecordingAttention(nn.Module):

    def __init__(self, *args, supported_attention_backends=None, **kwargs) -> None:
        super().__init__()
        self.supported_attention_backends = supported_attention_backends


def test_wan_routes_fp4_to_self_attention_only(monkeypatch) -> None:
    monkeypatch.setattr(wan_transformer, "DistributedAttention", _RecordingAttention)
    monkeypatch.setattr(wan_transformer, "LocalAttention", _RecordingAttention)

    for added_kv_proj_dim in (None, 128):
        block = wan_transformer.WanTransformerBlock(
            dim=128,
            ffn_dim=256,
            num_heads=1,
            cross_attn_norm=True,
            added_kv_proj_dim=added_kv_proj_dim,
            supported_attention_backends=WanVideoArchConfig()._supported_attention_backends,
        )

        assert AttentionBackendEnum.ATTN_QAT_INFER in block.attn1.supported_attention_backends
        assert AttentionBackendEnum.ATTN_QAT_INFER not in block.attn2.attn.supported_attention_backends


def test_denoising_stage_preserves_fp4_backend_request(monkeypatch) -> None:
    captured = {}

    def _get_attn_backend(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(denoising, "get_attn_backend", _get_attn_backend)
    transformer = SimpleNamespace(
        hidden_size=128,
        num_attention_heads=1,
        config=SimpleNamespace(_resolved_attention_backend=AttentionBackendEnum.ATTN_QAT_INFER),
    )

    denoising.DenoisingStage(transformer, scheduler=object())

    assert AttentionBackendEnum.ATTN_QAT_INFER in captured["supported_attention_backends"]
    assert captured["requested"] is AttentionBackendEnum.ATTN_QAT_INFER
