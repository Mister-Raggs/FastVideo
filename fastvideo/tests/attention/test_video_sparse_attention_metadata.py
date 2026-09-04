import math

import pytest
import torch

from fastvideo.attention.backends import video_sparse_attn as vsa_module
from fastvideo.attention.backends.video_sparse_attn import (
    VSA_TILE_SIZE,
    VideoSparseAttentionImpl,
    VideoSparseAttentionMetadataBuilder,
    _compute_cur_topk,
)


def _build_metadata(
    cache_tile_buf: bool,
    raw_latent_shape=(4, 4, 4),
    VSA_sparsity=0.5,
    tile_size=64,
    native_128=None,
):
    return VideoSparseAttentionMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=raw_latent_shape,
        patch_size=(1, 1, 1),
        VSA_sparsity=VSA_sparsity,
        device=torch.device("cpu"),
        cache_tile_buf=cache_tile_buf,
        tile_size=tile_size,
        native_128=native_128,
    )


def test_vsa_tile_does_not_cache_training_scratch_when_disabled():
    metadata = _build_metadata(cache_tile_buf=False)
    impl = object.__new__(VideoSparseAttentionImpl)
    x = torch.ones(1, 64, 2, 2)

    tiled = impl.tile(x, metadata)

    assert tiled.shape == x.shape
    assert metadata.tile_buf is None


def test_vsa_tile_caches_scratch_by_default():
    metadata = _build_metadata(cache_tile_buf=True)
    impl = object.__new__(VideoSparseAttentionImpl)
    x = torch.ones(1, 64, 2, 2)

    tiled = impl.tile(x, metadata)

    assert metadata.tile_buf is tiled


def test_vsa_tile_cached_and_uncached_produce_identical_values():
    # The cache flag must be a pure performance knob: enabling it (inference /
    # memory-rich training) and disabling it (#1423 OOM-safe training) must
    # yield bit-identical tilings. Use a padded shape (dit T=5 -> padded T=8)
    # so the zero-pad-position scatter is actually exercised.
    impl = object.__new__(VideoSparseAttentionImpl)
    md_cached = _build_metadata(cache_tile_buf=True, raw_latent_shape=(5, 4, 4))
    md_uncached = _build_metadata(cache_tile_buf=False, raw_latent_shape=(5, 4, 4))

    total_seq_length = md_cached.total_seq_length
    x = torch.randn(2, total_seq_length, 3, 4)

    cached = impl.tile(x, md_cached).clone()
    uncached = impl.tile(x, md_uncached)

    assert cached.shape == uncached.shape
    assert torch.equal(cached, uncached)
    # The cached path stashes the buffer; the uncached path does not.
    assert md_cached.tile_buf is not None
    assert md_uncached.tile_buf is None


def test_vsa_forward_cur_topk_uses_padded_kv_block_count(monkeypatch):
    impl = object.__new__(VideoSparseAttentionImpl)
    metadata = _build_metadata(cache_tile_buf=True, raw_latent_shape=(5, 32, 32), VSA_sparsity=0.75)
    block_elements = math.prod(VSA_TILE_SIZE)
    padded_seq_len = metadata.variable_block_sizes.numel() * block_elements
    expected_topk = math.ceil((1 - metadata.VSA_sparsity) * metadata.variable_block_sizes.numel())
    unpadded_topk = math.ceil((1 - metadata.VSA_sparsity) * (metadata.total_seq_length / block_elements))
    captured = {}

    def fake_video_sparse_attn(
        query,
        key,
        value,
        variable_block_sizes,
        q_variable_block_sizes,
        topk,
        block_size,
        compress_attn_weight,
        native_128,
    ):
        captured["topk"] = topk
        captured["block_size"] = block_size
        captured["native_128"] = native_128
        assert torch.equal(variable_block_sizes, metadata.variable_block_sizes)
        assert torch.equal(q_variable_block_sizes, metadata.variable_block_sizes)
        return query

    monkeypatch.setattr(vsa_module, "video_sparse_attn", fake_video_sparse_attn)

    query = torch.ones(1, padded_seq_len, 1, 1)
    output = impl.forward(query, query, query, query, metadata)

    assert unpadded_topk < expected_topk
    assert captured["topk"] == expected_topk
    assert captured["block_size"] == VSA_TILE_SIZE
    assert captured["native_128"] is None
    assert output.shape == query.shape


def test_vsa_128_geometry_preserves_standard_wan_padded_token_count():
    tile64 = _build_metadata(cache_tile_buf=True, raw_latent_shape=(21, 28, 52))
    tile128 = _build_metadata(
        cache_tile_buf=True,
        raw_latent_shape=(21, 28, 52),
        tile_size=128,
        native_128=True,
    )

    padded64 = tile64.variable_block_sizes.numel() * math.prod(tile64.tile_size)
    padded128 = tile128.variable_block_sizes.numel() * math.prod(tile128.tile_size)

    assert tile64.tile_size == (4, 4, 4)
    assert tile128.tile_size == (8, 4, 4)
    assert tile128.native_128 is True
    assert padded128 == padded64
    assert tile128.total_seq_length == tile64.total_seq_length


def test_vsa_128_forward_passes_explicit_native_route(monkeypatch):
    impl = object.__new__(VideoSparseAttentionImpl)
    metadata = _build_metadata(cache_tile_buf=True, raw_latent_shape=(8, 4, 4), tile_size=128, native_128=True)
    captured = {}

    def fake_video_sparse_attn(*args, **kwargs):
        captured.update(kwargs)
        return args[0]

    monkeypatch.setattr(vsa_module, "video_sparse_attn", fake_video_sparse_attn)
    query = torch.ones(1, 128, 1, 1)

    output = impl.forward(query, query, query, query, metadata)

    assert captured["block_size"] == (8, 4, 4)
    assert captured["native_128"] is True
    assert output.shape == query.shape


def test_vsa_metadata_rejects_unknown_tile_size():
    with torch.no_grad(), pytest.raises(ValueError, match="tile_size must be one of"):
        _build_metadata(cache_tile_buf=True, tile_size=256)


def test_vsa_native_128_rejects_default_tile_geometry():
    with pytest.raises(ValueError, match="native_128 requires tile_size=128"):
        _build_metadata(cache_tile_buf=True, native_128=True)


def test_vsa_cur_topk_clamps_to_valid_block_range():
    metadata = _build_metadata(cache_tile_buf=True, raw_latent_shape=(5, 32, 32), VSA_sparsity=1.0)
    num_kv_blocks = metadata.variable_block_sizes.numel()

    assert _compute_cur_topk(metadata) == 1

    metadata.VSA_sparsity = -0.01
    assert _compute_cur_topk(metadata) == num_kv_blocks
