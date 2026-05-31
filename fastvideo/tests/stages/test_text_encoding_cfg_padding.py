"""
Padding-length policy test for the Wan/Cosmos batched-CFG trim+zero-pad
recipe in ``TextEncodingStage.forward``.

The recipe (W3b) tokenizes at ``padding="max_length"`` so the T5 encoder
sees canonical input, then trims+zero-pads the pos/neg embeddings to a
common length. ``cfg_pad_embeds_to_max_length`` (W3g) controls that
common length:

  * True  (default) -> pad to the full tokenizer max_length, matching
    diffusers' canonical training distribution. Every cross-attention
    then attends over max_length (mostly zero) K/V positions.
  * False -> trim to ``max(real_pos_len, real_neg_len)`` so the DiT only
    processes real tokens. Cheaper per step; deviates from canonical.

These tests stub ``encode_text`` so we can assert the exact output seq
length + zero-padding without standing up a real T5 encoder.
"""
import types
from typing import Any

import torch

from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.text_encoding import TextEncodingStage

_MAX_LEN = 10
_POS_REAL = 4
_NEG_REAL = 7
_HIDDEN = 8


class _WanVideoConfig:
    """Stand-in whose class name lands in the recipe allowlist via MRO."""


def _make_fastvideo_args(pad_to_max: bool) -> Any:
    return types.SimpleNamespace(
        pipeline_config=types.SimpleNamespace(
            text_encoder_configs=[object()],
            dit_config=_WanVideoConfig(),
        ),
        cfg_pad_embeds_to_max_length=pad_to_max,
    )


def _embed_with_real_len(real_len: int) -> torch.Tensor:
    """(1, _MAX_LEN, _HIDDEN): real positions are 1.0, padded positions
    are a nonzero sentinel (9.0) so trimming/zeroing them is observable.
    """
    t = torch.full((1, _MAX_LEN, _HIDDEN), 9.0)
    t[:, :real_len] = 1.0
    return t


def _mask_with_real_len(real_len: int) -> torch.Tensor:
    m = torch.zeros(1, _MAX_LEN, dtype=torch.long)
    m[:, :real_len] = 1
    return m


def _make_stage() -> TextEncodingStage:
    """Bypass ``__init__`` and stub ``encode_text`` to return canonical
    max_length-padded pos/neg tensors keyed off the prompt text."""
    stage = TextEncodingStage.__new__(TextEncodingStage)
    torch.nn.Module.__init__(stage)
    stage.tokenizers = [object()]
    stage.text_encoders = [object()]
    stage._last_audio_embeds = None

    def _fake_encode_text(prompt_text, fastvideo_args, encoder_index, return_attention_mask=False, padding=None):
        real = _NEG_REAL if prompt_text == "neg" else _POS_REAL
        return [_embed_with_real_len(real)], [_mask_with_real_len(real)]

    stage.encode_text = _fake_encode_text  # type: ignore[method-assign]
    return stage


def _make_batch() -> ForwardBatch:
    return ForwardBatch(
        data_type="video",
        prompt="pos",
        negative_prompt="neg",
        prompt_embeds=[],
        prompt_attention_mask=[],
        negative_prompt_embeds=[],
        negative_attention_mask=[],
        do_classifier_free_guidance=True,
        guidance_scale=7.5,
    )


def _run(pad_to_max: bool) -> ForwardBatch:
    stage = _make_stage()
    return stage.forward(_make_batch(), _make_fastvideo_args(pad_to_max))


def test_pad_to_max_length_keeps_full_recipe_shape() -> None:
    """Default policy pads pos+neg to the full tokenizer max_length and
    zeros the padded positions (diffusers-canonical)."""
    batch = _run(pad_to_max=True)
    pe = batch.prompt_embeds[0]
    ne = batch.negative_prompt_embeds[0]
    assert pe.shape[1] == _MAX_LEN
    assert ne.shape[1] == _MAX_LEN
    # Real positions preserved, padded positions zeroed.
    assert torch.equal(pe[:, :_POS_REAL], torch.ones(1, _POS_REAL, _HIDDEN))
    assert torch.equal(pe[:, _POS_REAL:], torch.zeros(1, _MAX_LEN - _POS_REAL, _HIDDEN))
    assert torch.equal(ne[:, :_NEG_REAL], torch.ones(1, _NEG_REAL, _HIDDEN))
    assert torch.equal(ne[:, _NEG_REAL:], torch.zeros(1, _MAX_LEN - _NEG_REAL, _HIDDEN))


def test_trim_policy_pads_to_longest_real_prompt() -> None:
    """W3g policy trims pos+neg to ``max(real_pos, real_neg)`` while
    keeping the two shapes matched for the CFG cat, and preserving real
    tokens + zeroing the residual pad."""
    batch = _run(pad_to_max=False)
    pe = batch.prompt_embeds[0]
    ne = batch.negative_prompt_embeds[0]
    target = max(_POS_REAL, _NEG_REAL)
    assert pe.shape[1] == target
    assert ne.shape[1] == target
    # Pos still has its 4 real tokens, then zero pad up to the neg length.
    assert torch.equal(pe[:, :_POS_REAL], torch.ones(1, _POS_REAL, _HIDDEN))
    assert torch.equal(pe[:, _POS_REAL:], torch.zeros(1, target - _POS_REAL, _HIDDEN))
    # Neg's real length == target, so it is exactly its real tokens.
    assert torch.equal(ne, torch.ones(1, _NEG_REAL, _HIDDEN))


def test_masks_track_embed_padding_length() -> None:
    """Attention masks must be trimmed/padded to the same target length
    as the embeddings under both policies."""
    batch_max = _run(pad_to_max=True)
    batch_trim = _run(pad_to_max=False)
    assert batch_max.prompt_attention_mask[0].shape[1] == _MAX_LEN
    assert batch_max.negative_attention_mask[0].shape[1] == _MAX_LEN
    target = max(_POS_REAL, _NEG_REAL)
    assert batch_trim.prompt_attention_mask[0].shape[1] == target
    assert batch_trim.negative_attention_mask[0].shape[1] == target
    # Trimmed pos mask: 4 ones then zero pad to target.
    expected_pos_mask = torch.zeros(1, target, dtype=batch_trim.prompt_attention_mask[0].dtype)
    expected_pos_mask[:, :_POS_REAL] = 1
    assert torch.equal(batch_trim.prompt_attention_mask[0], expected_pos_mask)


def test_real_positions_identical_across_policies() -> None:
    """The two policies differ ONLY in trailing zero pad: the real-token
    embeddings are bit-identical, which is what makes the A/B a clean
    one-variable comparison (DiT cross-attn length)."""
    pe_max = _run(pad_to_max=True).prompt_embeds[0]
    pe_trim = _run(pad_to_max=False).prompt_embeds[0]
    ne_max = _run(pad_to_max=True).negative_prompt_embeds[0]
    ne_trim = _run(pad_to_max=False).negative_prompt_embeds[0]
    assert torch.equal(pe_max[:, :_POS_REAL], pe_trim[:, :_POS_REAL])
    assert torch.equal(ne_max[:, :_NEG_REAL], ne_trim[:, :_NEG_REAL])
