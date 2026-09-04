"""Model-free dispatch checks for the opt-in native Triton VSA-128 path."""

from __future__ import annotations

import torch

import fastvideo_kernel.block_sparse_attn_256 as vsa128


def _inputs(requires_grad: bool = False) -> tuple[torch.Tensor, ...]:
    q = torch.zeros(1, 2, 256, 8, requires_grad=requires_grad)
    k = torch.zeros_like(q, requires_grad=requires_grad)
    v = torch.zeros_like(q, requires_grad=requires_grad)
    block_map = torch.ones(1, 2, 2, 2, dtype=torch.bool)
    sizes = torch.full((2,), 128, dtype=torch.int32)
    return q, k, v, block_map, sizes


def test_native_128_flag_selects_native_op_during_inference(monkeypatch) -> None:
    native_result = (object(), object())
    monkeypatch.setenv("FASTVIDEO_VSA_TRITON", "1")
    monkeypatch.setenv("FASTVIDEO_VSA_TRITON_NATIVE_128", "1")
    monkeypatch.setattr(
        vsa128,
        "block_sparse_attn_triton_128_from_mask_inference",
        lambda *_args: native_result,
    )
    monkeypatch.setattr(
        vsa128,
        "_triton_via_route_a_128",
        lambda *_args: (_ for _ in ()).throw(AssertionError("route A should not run")),
    )

    with torch.inference_mode():
        result = vsa128.block_sparse_attn_128(*_inputs())

    assert result is native_result


def test_native_128_flag_preserves_route_a_when_gradients_are_needed(monkeypatch) -> None:
    route_a_result = (object(), object())
    monkeypatch.setenv("FASTVIDEO_VSA_TRITON", "1")
    monkeypatch.setenv("FASTVIDEO_VSA_TRITON_NATIVE_128", "1")
    monkeypatch.setattr(
        vsa128,
        "block_sparse_attn_triton_128_from_mask_inference",
        lambda *_args: (_ for _ in ()).throw(AssertionError("inference op should not run")),
    )
    monkeypatch.setattr(vsa128, "_triton_via_route_a_128", lambda *_args: route_a_result)

    result = vsa128.block_sparse_attn_128(*_inputs(requires_grad=True))

    assert result is route_a_result


def test_explicit_route_choice_overrides_worker_environment(monkeypatch) -> None:
    route_a_result = (object(), object())
    monkeypatch.setenv("FASTVIDEO_VSA_TRITON", "1")
    monkeypatch.setenv("FASTVIDEO_VSA_TRITON_NATIVE_128", "1")
    monkeypatch.setattr(
        vsa128,
        "block_sparse_attn_triton_128_from_mask_inference",
        lambda *_args: (_ for _ in ()).throw(AssertionError("native route should be explicitly disabled")),
    )
    monkeypatch.setattr(vsa128, "_triton_via_route_a_128", lambda *_args: route_a_result)

    with torch.inference_mode():
        result = vsa128.block_sparse_attn_128(*_inputs(), native_triton=False)

    assert result is route_a_result


def test_explicit_native_choice_overrides_disabled_worker_environment(monkeypatch) -> None:
    native_result = (object(), object())
    monkeypatch.setenv("FASTVIDEO_VSA_TRITON", "1")
    monkeypatch.setenv("FASTVIDEO_VSA_TRITON_NATIVE_128", "0")
    monkeypatch.setattr(
        vsa128,
        "block_sparse_attn_triton_128_from_mask_inference",
        lambda *_args: native_result,
    )
    monkeypatch.setattr(
        vsa128,
        "_triton_via_route_a_128",
        lambda *_args: (_ for _ in ()).throw(AssertionError("route A should be explicitly disabled")),
    )

    with torch.inference_mode():
        result = vsa128.block_sparse_attn_128(*_inputs(), native_triton=True)

    assert result is native_result
