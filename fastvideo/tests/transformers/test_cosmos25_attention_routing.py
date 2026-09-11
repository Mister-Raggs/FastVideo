from __future__ import annotations

import fastvideo.attention.layer as attention_layer
from fastvideo.models.dits.cosmos2_5 import Cosmos25TransformerBlock
from fastvideo.platforms import AttentionBackendEnum


class _AttentionImpl:

    def __init__(self, **_kwargs) -> None:
        pass


class _AttentionBackend:

    @staticmethod
    def get_impl_cls():
        return _AttentionImpl

    @staticmethod
    def get_name() -> str:
        return "TORCH_SDPA"


def test_cosmos25_fp4_request_only_reaches_self_attention(monkeypatch) -> None:
    resolutions: list[dict] = []

    def resolve_backend(*_args, **kwargs):
        resolutions.append(kwargs)
        return _AttentionBackend

    monkeypatch.setattr(attention_layer, "get_attn_backend", resolve_backend)

    requested_backends = (
        AttentionBackendEnum.ATTN_QAT_INFER,
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TORCH_SDPA,
    )
    Cosmos25TransformerBlock(
        num_attention_heads=2,
        attention_head_dim=4,
        cross_attention_dim=8,
        mlp_ratio=1.0,
        adaln_lora_dim=4,
        supported_attention_backends=requested_backends,
    )

    assert resolutions[0]["supported_attention_backends"] == requested_backends
    assert resolutions[0].get("default_backend") is None
    assert resolutions[1]["supported_attention_backends"] == (
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TORCH_SDPA,
    )
    assert resolutions[1]["default_backend"] == AttentionBackendEnum.TORCH_SDPA
